import argparse, numpy as np, pandas as pd
from datetime import datetime
rng = np.random.default_rng(7)

def main(n_borrowers, months):
    borrowers = [f"B{i:05d}" for i in range(n_borrowers)]
    # monthly edges: co-merchant / co-applicant style links
    edges = []
    events = []
    for m in range(months):
        for _ in range(n_borrowers//2):
            a, b = rng.choice(borrowers, 2, replace=False)
            w = rng.uniform(0.1, 1.0)  # tie strength
            edges.append({"month": m, "src": a, "dst": b, "weight": w})
        # borrower events
        for b in borrowers:
            util = rng.beta(2,5)  # utilization proxy
            late = 1 if rng.uniform() < max(0.02, util*0.2) else 0
            events.append({"month": m, "borrower": b, "utilization": util, "late": late})
    pd.DataFrame(edges).to_csv("data/edges.csv", index=False)
    pd.DataFrame(events).to_csv("data/events.csv", index=False)
    print("Wrote data/edges.csv and data/events.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_borrowers", type=int, default=3000)
    ap.add_argument("--months", type=int, default=12)
    a = ap.parse_args()
    main(a.n_borrowers, a.months)
