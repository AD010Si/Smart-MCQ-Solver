#!/usr/bin/env python
"""Run selected experiments on the grouped holdout and write results/results.csv.

Example:
    python scripts/run_experiments.py --models features hybrid minilm
    python scripts/run_experiments.py --models all          # needs a GPU for distilbert/qwen
"""
import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcq_solver import evaluate, grouped_split, load_competition_data  # noqa: E402
from mcq_solver.baselines import feature_rank, sbert_baseline  # noqa: E402

CHOICES = ["features", "minilm", "hybrid", "dense_rag", "distilbert", "bilstm", "qwen"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["features", "hybrid"], choices=CHOICES + ["all"])
    p.add_argument("--data-dir", default=None)
    p.add_argument("--out", default="results/results.csv")
    args = p.parse_args()
    models = CHOICES if "all" in args.models else args.models

    train, _ = load_competition_data(*([args.data_dir] if args.data_dir else []))
    kb_df, eval_df = grouped_split(train)
    y = eval_df["answer"].tolist()
    print(f"knowledge/train rows: {len(kb_df)} | eval rows: {len(eval_df)}")

    rows, dense = [], None
    for name in models:
        if name == "features":
            preds, label = [feature_rank(r) for _, r in eval_df.iterrows()], "Feature ranking"
        elif name == "minilm":
            preds, label = sbert_baseline(eval_df), "MiniLM similarity"
        elif name == "hybrid":
            from mcq_solver.retrieval import HybridReranker
            rr = HybridReranker(kb_df)
            preds, label = [rr.predict(r) for _, r in eval_df.iterrows()], "RAG: TF-IDF+BM25 reranking"
        elif name in ("dense_rag", "qwen"):
            from mcq_solver.retrieval import DenseRetriever
            dense = dense or DenseRetriever(kb_df)
            if name == "dense_rag":
                preds, label = [dense.rank_by_context_similarity(r) for _, r in eval_df.iterrows()], "RAG: dense retrieval + similarity"
            else:
                from mcq_solver.llm_scoring import LLMLetterScorer
                scorer = LLMLetterScorer()
                preds = []
                for _, r in eval_df.iterrows():
                    q = str(r["prompt"]) + " " + " ".join(str(r[c]) for c in "ABCDE")
                    preds.append(scorer.rank(r, " ".join(dense.retrieve(q, k=3))))
                label = "RAG + Qwen2.5-3B likelihood"
        elif name == "distilbert":
            from mcq_solver.supervised import predict_distilbert, train_distilbert
            model, tok = train_distilbert(kb_df, eval_df)
            preds, label = predict_distilbert(model, tok, eval_df), "Fine-tuned DistilBERT"
        elif name == "bilstm":
            from transformers import AutoTokenizer
            from mcq_solver.supervised import predict_bilstm, train_bilstm
            model, loader, _ = train_bilstm(kb_df, eval_df, AutoTokenizer.from_pretrained("distilbert-base-uncased"))
            preds, label = predict_bilstm(model, loader), "Attention-BiLSTM"
        res = evaluate(y, preds, label)
        print(f"{label:38s} MAP@3={res['MAP@3']:.4f}  Acc={res['Accuracy']:.4f}  F1={res['Macro-F1']:.4f}")
        rows.append(res)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).round(4).to_csv(args.out.replace(".csv", "_rerun.csv"), index=False)
    print(f"wrote {args.out.replace('.csv', '_rerun.csv')}")


if __name__ == "__main__":
    main()
