import argparse, pandas as pd, pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import GradientBoostingClassifier

def main(feat_path, model_path):
    df = pd.read_csv(feat_path)
    X = df[["deg","pagerank","nbr_late_rate","nbr_util_mean","utilization"]]
    y = df["late"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(X_train, y_train)
    auc = roc_auc_score(y_test, clf.predict_proba(X_test)[:,1])
    print("AUC:", round(auc, 4))
    print(classification_report(y_test, clf.predict(X_test)))
    import os
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    print("Saved model to", model_path)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--model", default="artifacts/model.pkl")
    a = ap.parse_args()
    main(a.features, a.model)
