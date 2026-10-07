import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_df():
    rng = np.random.default_rng(0)
    rows = []
    for i in range(200):
        g = i // 4  # 4 near-duplicate questions per option set
        rows.append({
            "id": i, "prompt": f"What is topic {g} about?",
            **{c: f"option {c} for group {g} " + "x" * int(rng.integers(1, 30)) for c in "ABCDE"},
            "answer": "ABCDE"[i % 5],
        })
    return pd.DataFrame(rows)
