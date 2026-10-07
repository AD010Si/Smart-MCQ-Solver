"""Shared constants. The data directory can be overridden with MCQ_DATA_DIR."""
import os

OPTION_LETTERS = ["A", "B", "C", "D", "E"]
SEED = 42
DATA_DIR = os.environ.get("MCQ_DATA_DIR", "data")
