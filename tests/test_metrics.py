import pytest
from mcq_solver.metrics import evaluate, map3


def test_map3_ranks():
    assert map3(["A"], ["A B C"]) == 1.0
    assert map3(["A"], ["B A C"]) == pytest.approx(0.5)
    assert map3(["A"], ["C D A"]) == pytest.approx(1 / 3)
    assert map3(["A"], ["B C D"]) == 0.0


def test_map3_mean_and_extra_predictions_ignored():
    assert map3(["A", "B"], ["A B C", "C D E A B"]) == pytest.approx(0.5)


def test_evaluate_keys():
    out = evaluate(["A", "B"], ["A C D", "B A C"], "m")
    assert out["MAP@3"] == 1.0 and out["Accuracy"] == 1.0 and out["model"] == "m"
