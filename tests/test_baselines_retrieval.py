import pytest
from mcq_solver.baselines import feature_rank


def test_feature_rank_prefers_longest(toy_df):
    row = toy_df.iloc[0].copy()
    row["A"], row["B"], row["C"], row["D"], row["E"] = "a", "bb" * 50, "c" * 5, "d", "e"
    assert feature_rank(row).split()[0] == "B"
    assert len(feature_rank(row).split()) == 3


def test_hybrid_reranker_exact_signature_shortcut(toy_df):
    pytest.importorskip("rank_bm25")
    from mcq_solver.retrieval import HybridReranker
    rr = HybridReranker(toy_df)
    row = toy_df.iloc[0]
    assert rr.predict(row).split()[0] == rr.sig_to_answer["||".join(str(row[c]) for c in "ABCDE")]


def test_hybrid_reranker_returns_three_unique_letters(toy_df):
    pytest.importorskip("rank_bm25")
    from mcq_solver.retrieval import HybridReranker
    rr = HybridReranker(toy_df.iloc[:100])
    row = toy_df.iloc[150].copy()
    row["C"] = row["C"] + " changed so the signature is unseen"
    out = rr.predict(row).split()
    assert len(out) == 3 and len(set(out)) == 3
