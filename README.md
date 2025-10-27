# Credit Risk GraphLite — Temporal Relational Features

This project simulates borrower–lender relationships over time and builds
graph-derived relational features (e.g., neighbor delinquency rate, degree, PageRank proxy)
to augment a standard credit default classifier.

No heavy GNN libraries — just NetworkX + scikit-learn.

## Quickstart
```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python simulate.py --n_borrowers 4000 --months 12
python build_features.py --edges data/edges.csv --events data/events.csv --out data/features.csv
python train.py --features data/features.csv --model artifacts/model.pkl
python score.py --features data/features.csv --model artifacts/model.pkl --out artifacts/scores.csv
```

## Files
- `simulate.py` — generate synthetic borrower events and monthly relations
- `build_features.py` — compute temporal graph stats per borrower-month
- `train.py` — train classifier
- `score.py` — produce scores
