import argparse, pandas as pd, pickle

def main(feat_path, model_path, out_path):
    df = pd.read_csv(feat_path)
    X = df[["deg","pagerank","nbr_late_rate","nbr_util_mean","utilization"]]
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    df["score"] = clf.predict_proba(X)[:,1]
    df.to_csv(out_path, index=False)
    print("Wrote", out_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="artifacts/scores.csv")
    a = ap.parse_args()
    main(a.features, a.model, a.out)
