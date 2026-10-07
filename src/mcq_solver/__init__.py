"""Smart MCQ Solver: ranking answers to five-option multiple-choice questions."""

from .config import OPTION_LETTERS, SEED
from .data import grouped_split, load_competition_data
from .metrics import evaluate, map3

__all__ = ["OPTION_LETTERS", "SEED", "grouped_split", "load_competition_data", "evaluate", "map3"]
__version__ = "0.1.0"
