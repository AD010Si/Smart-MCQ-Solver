"""Zero-shot LLM candidate ranking via the log-likelihood of each answer letter."""
from __future__ import annotations

import pandas as pd

from .config import OPTION_LETTERS

SYSTEM_PROMPT = (
    "You are an expert assistant answering multiple-choice questions. "
    "Use ONLY the provided context. Respond with a single letter."
)


class LLMLetterScorer:
    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct", max_prompt_tokens=1024):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_prompt_tokens = max_prompt_tokens
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="auto")
        self.llm.eval()

    def build_prompt(self, row: pd.Series, context: str) -> str:
        options = "\n".join(f"{c}: {row[c]}" for c in OPTION_LETTERS)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {row['prompt']}\n\nOptions:\n{options}\n\nAnswer:"},
        ]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def rank(self, row: pd.Series, context: str) -> str:
        torch = self.torch
        prefix = self.tok(
            self.build_prompt(row, context), return_tensors="pt",
            truncation=True, max_length=self.max_prompt_tokens,
        ).to(self.llm.device)
        n = prefix.input_ids.shape[1]
        scores = {}
        with torch.no_grad():
            for letter in OPTION_LETTERS:
                letter_ids = self.tok(letter, return_tensors="pt", add_special_tokens=False).input_ids.to(self.llm.device)
                full = torch.cat([prefix.input_ids, letter_ids], dim=1)
                logits = self.llm(input_ids=full, attention_mask=torch.ones_like(full)).logits[0]
                scores[letter] = sum(
                    torch.log_softmax(logits[n + i - 1], dim=-1)[t].item()
                    for i, t in enumerate(letter_ids[0])
                )
        return " ".join(sorted(scores, key=scores.get, reverse=True)[:3])
