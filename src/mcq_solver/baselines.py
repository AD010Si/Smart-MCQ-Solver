"""Baselines: hand-crafted features and embedding similarity."""
from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .config import OPTION_LETTERS


def extract_features(row: pd.Series) -> dict:
    prompt = set(str(row["prompt"]).lower().split())
    feats = {}
    for col in OPTION_LETTERS:
        text = str(row[col])
        words = text.lower().split()
        feats[col] = {
            "char_len": len(text),
            "word_count": len(words),
            "unique_words": len(set(words)) / max(len(words), 1),
            "overlap": len(prompt & set(words)),
        }
    return feats


def feature_rank(row: pd.Series) -> str:
    """score = char_len + 5 * overlap_with_prompt + 50 * unique_word_ratio."""
    feats = extract_features(row)
    scores = {
        c: f["char_len"] + f["overlap"] * 5 + f["unique_words"] * 50 for c, f in feats.items()
    }
    return " ".join(sorted(scores, key=scores.get, reverse=True)[:3])


def rank_by_embedding_similarity(prompt_embs: np.ndarray, option_embs: dict) -> List[str]:
    """Rank options by cosine similarity (inputs must be L2-normalised)."""
    preds = []
    for i in range(len(prompt_embs)):
        sims = {c: float(np.dot(prompt_embs[i], option_embs[c][i])) for c in OPTION_LETTERS}
        preds.append(" ".join(sorted(sims, key=sims.get, reverse=True)[:3]))
    return preds


def sbert_baseline(eval_df: pd.DataFrame, model_name: str = "all-MiniLM-L6-v2") -> List[str]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    enc = lambda s: model.encode(s.astype(str).tolist(), normalize_embeddings=True)
    return rank_by_embedding_similarity(
        enc(eval_df["prompt"]), {c: enc(eval_df[c]) for c in OPTION_LETTERS}
    )
