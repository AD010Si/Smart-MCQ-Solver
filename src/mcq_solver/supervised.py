"""Supervised models: fine-tuned DistilBERT (multiple choice) and an Attention-BiLSTM from scratch."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .config import OPTION_LETTERS, SEED

LABEL_MAP = {c: i for i, c in enumerate(OPTION_LETTERS)}


# ---------------------------------------------------------------- DistilBERT
def encode_row(row, tokenizer, max_length=256):
    enc = tokenizer(
        [str(row["prompt"])] * 5, [str(row[c]) for c in OPTION_LETTERS],
        truncation=True, max_length=max_length, padding="max_length", return_tensors="pt",
    )
    return dict(enc.items())


class MCQDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer, max_length=256):
        self.df, self.tokenizer, self.max_length = frame.reset_index(drop=True), tokenizer, max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        item = encode_row(row, self.tokenizer, self.max_length)
        item["labels"] = torch.tensor(LABEL_MAP[row["answer"]], dtype=torch.long)
        return item


@dataclass
class MultipleChoiceCollator:
    def __call__(self, features):
        labels = torch.stack([f.pop("labels") for f in features])
        batch = {k: torch.stack([f[k] for f in features]) for k in features[0]}
        batch["labels"] = labels
        return batch


def train_distilbert(kb_df, eval_df, model_name="distilbert-base-uncased", output_dir="outputs/distilbert"):
    """Fine-tune on kb_df; returns (model, tokenizer). Note: eval_df is used for epoch selection."""
    from transformers import AutoModelForMultipleChoice, AutoTokenizer, Trainer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name)
    args = TrainingArguments(
        output_dir=output_dir, num_train_epochs=4, learning_rate=2e-5,
        per_device_train_batch_size=8, per_device_eval_batch_size=8,
        eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        report_to="none", fp16=torch.cuda.is_available(), seed=SEED,
    )
    Trainer(
        model=model, args=args, train_dataset=MCQDataset(kb_df, tok),
        eval_dataset=MCQDataset(eval_df, tok), data_collator=MultipleChoiceCollator(),
    ).train()
    return model, tok


def predict_distilbert(model, tokenizer, eval_df) -> List[str]:
    model.eval()
    device = next(model.parameters()).device
    preds = []
    with torch.no_grad():
        for _, row in eval_df.iterrows():
            batch = {k: v.unsqueeze(0).to(device) for k, v in encode_row(row, tokenizer).items()}
            logits = model(**batch).logits[0].cpu().numpy()
            preds.append(" ".join(OPTION_LETTERS[i] for i in np.argsort(-logits)[:3]))
    return preds


# ------------------------------------------------------------ Attention-BiLSTM
class OptionPairDataset(Dataset):
    def __init__(self, frame, tokenizer, max_length=128):
        self.df, self.tokenizer, self.max_length = frame.reset_index(drop=True), tokenizer, max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ids, attn = [], []
        for c in OPTION_LETTERS:
            enc = self.tokenizer(
                str(row["prompt"]) + " " + str(row[c]), truncation=True,
                max_length=self.max_length, padding="max_length", return_tensors="pt",
            )
            ids.append(enc["input_ids"].squeeze(0))
            attn.append(enc["attention_mask"].squeeze(0))
        return {
            "input_ids": torch.stack(ids), "attention_mask": torch.stack(attn),
            "labels": torch.tensor(LABEL_MAP[row["answer"]], dtype=torch.long),
        }


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive attention pooling over LSTM states."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h, mask):
        scores = self.v(torch.tanh(self.W(h))).squeeze(-1).masked_fill(mask == 0, -1e9)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), h).squeeze(1), weights


class AttentionBiLSTMScorer(nn.Module):
    """Embedding -> BiLSTM -> additive attention -> scalar score per (prompt, option)."""

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, pad_id=0, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        lstm_h = hidden_dim // 2
        self.lstm = nn.LSTM(embed_dim, lstm_h, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.attn = AdditiveAttention(lstm_h * 2)
        self.scorer = nn.Linear(lstm_h * 2, 1)

    def encode_option(self, input_ids, attention_mask):
        emb = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(emb, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=input_ids.size(1))
        ctx, _ = self.attn(h, attention_mask)
        return self.scorer(ctx).squeeze(-1)

    def forward(self, input_ids, attention_mask):
        B, n, L = input_ids.shape
        flat = self.encode_option(input_ids.view(B * n, L), attention_mask.view(B * n, L))
        return flat.view(B, n)


def train_bilstm(kb_df, eval_df, tokenizer, epochs=5, lr=1e-3, batch_size=16):
    """Train from scratch; returns (model, eval_loader, history)."""
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    train_loader = DataLoader(OptionPairDataset(kb_df, tokenizer), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(OptionPairDataset(eval_df, tokenizer), batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionBiLSTMScorer(tokenizer.vocab_size, pad_id=pad_id).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = {"train_loss": [], "eval_loss": []}

    def run(loader, train):
        model.train(train)
        total, n = 0.0, 0
        with torch.set_grad_enabled(train):
            for b in loader:
                labels = b["labels"].to(device)
                loss = F.cross_entropy(model(b["input_ids"].to(device), b["attention_mask"].to(device)), labels)
                if train:
                    opt.zero_grad(); loss.backward(); opt.step()
                total += loss.item() * labels.size(0); n += labels.size(0)
        return total / max(n, 1)

    for epoch in range(1, epochs + 1):
        history["train_loss"].append(run(train_loader, True))
        history["eval_loss"].append(run(eval_loader, False))
        print(f"Epoch {epoch}/{epochs}  train_loss={history['train_loss'][-1]:.4f}  eval_loss={history['eval_loss'][-1]:.4f}")
    return model, eval_loader, history


def predict_bilstm(model, eval_loader) -> List[str]:
    device = next(model.parameters()).device
    model.eval()
    preds = []
    with torch.no_grad():
        for b in eval_loader:
            probs = torch.softmax(model(b["input_ids"].to(device), b["attention_mask"].to(device)), dim=-1).cpu().numpy()
            preds += [" ".join(OPTION_LETTERS[i] for i in np.argsort(-p)[:3]) for p in probs]
    return preds
