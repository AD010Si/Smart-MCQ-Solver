"""Data loading and the leakage-controlled grouped split."""
from __future__ import annotations

import os
from typing import Tuple

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from .config import DATA_DIR, OPTION_LETTERS, SEED


def load_competition_data(data_dir: str = DATA_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train.csv / test.csv (columns: id, prompt, A-E, [answer])."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    for path in (train_path, test_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found. Download the competition data and place it in "
                f"'{data_dir}/' (see data/README.md), or set MCQ_DATA_DIR."
            )
    return pd.read_csv(train_path), pd.read_csv(test_path)


def add_option_set(df: pd.DataFrame) -> pd.DataFrame:
    """Add an 'options_set' column: the full A-E option signature of each row."""
    out = df.copy()
    out["options_set"] = out[OPTION_LETTERS].astype(str).agg(" ".join, axis=1)
    return out


def grouped_split(
    train: pd.DataFrame, test_size: float = 0.20, seed: int = SEED
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split so that no identical A-E option set appears in both partitions.

    Near-duplicate questions share option sets; a plain random split would let
    retrieval or fine-tuned models memorise answers. Returns (kb_df, eval_df).
    """
    df = add_option_set(train)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    kb_idx, eval_idx = next(gss.split(df, groups=df["options_set"]))
    kb_df = df.iloc[kb_idx].reset_index(drop=True)
    eval_df = df.iloc[eval_idx].reset_index(drop=True)
    overlap = set(kb_df["options_set"]) & set(eval_df["options_set"])
    assert not overlap, f"Found {len(overlap)} overlapping option groups"
    return kb_df, eval_df
