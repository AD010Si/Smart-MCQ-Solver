"""Ranking metrics. Predictions are strings such as 'C A E' (top-3, best first)."""
from __future__ import annotations

from typing import Dict, Sequence

from sklearn.metrics import accuracy_score, f1_score

from .config import OPTION_LETTERS


def map3(y_true: Sequence[str], predicted: Sequence[str]) -> float:
    """Mean Average Precision at 3: 1, 1/2, 1/3 for rank 1, 2, 3; else 0."""
    score = 0.0
    for true, preds in zip(y_true, predicted):
        for rank, pred in enumerate(str(preds).split()[:3], start=1):
            if pred == true:
                score += 1.0 / rank
                break
    return score / len(y_true)


def evaluate(y_true: Sequence[str], predicted: Sequence[str], model_name: str) -> Dict[str, float]:
    """Return MAP@3, top-1 accuracy and macro-F1 for one model."""
    top1 = [str(p).split()[0] if str(p).split() else "A" for p in predicted]
    return {
        "model": model_name,
        "MAP@3": map3(y_true, predicted),
        "Accuracy": accuracy_score(y_true, top1),
        "Macro-F1": f1_score(y_true, top1, labels=OPTION_LETTERS, average="macro", zero_division=0),
    }
