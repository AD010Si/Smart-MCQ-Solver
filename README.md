# Smart MCQ Solver — Retrieval, Transformers & LLM Ranking

A portfolio-ready version of my machine-learning competition project for ranking the three most likely answers to multiple-choice questions.

## Highlights

- Built and compared heuristic, embedding, supervised Transformer, BiLSTM, retrieval, reranking and LLM-based approaches.
- Used a grouped evaluation split to avoid complete A–E option-set overlap between the knowledge/training partition and evaluation partition.
- Implemented dense retrieval with Sentence Transformers + FAISS and sparse retrieval/reranking with TF-IDF/BM25.
- Fine-tuned DistilBERT for multiple-choice classification.
- Evaluated Qwen 2.5 3B candidate likelihood scoring.
- Best recorded grouped-holdout result: **0.7996 MAP@3**.

## Repository

```text
smart-mcq-solver/
├── README.md
├── requirements.txt
└── smart_mcq_solver.ipynb
```

## Notebook

Open `smart_mcq_solver_portfolio.ipynb` to see the complete documented workflow.

## Important

This is a research/portfolio project developed from a competition workflow. It is not presented as a production system. Expensive model cells may require a CUDA-enabled GPU.
