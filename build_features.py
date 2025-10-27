import argparse, pandas as pd, numpy as np, networkx as nx

def neighbor_rate(G, node, attr_name, default=0.0):
    nbrs = list(G.neighbors(node))
    if not nbrs:
        return default
    vals = [G.nodes[n].get(attr_name, 0.0) for n in nbrs]
    return float(np.mean(vals))

def main(edges_path, events_path, out_path):
    edges = pd.read_csv(edges_path)
    events = pd.read_csv(events_path)
    frames = []
    for m, dfm in edges.groupby("month"):
        # Build graph for this month
        G = nx.Graph()
        month_events = events.query("month == @m").copy()
        # attach node attrs
        for _, r in month_events.iterrows():
            G.add_node(r["borrower"], utilization=float(r["utilization"]), late=int(r["late"]))
        for _, r in dfm.iterrows():
            G.add_edge(r["src"], r["dst"], weight=float(r["weight"]))
        # features per borrower
        feats = []
        pr = nx.pagerank(G, alpha=0.85) if G.number_of_nodes() else {}
        for b in month_events["borrower"]:
            deg = G.degree(b)
            nbr_late = neighbor_rate(G, b, "late", 0.0)
            nbr_util = neighbor_rate(G, b, "utilization", 0.0)
            feats.append({
                "month": m, "borrower": b,
                "deg": deg,
                "pagerank": pr.get(b, 0.0),
                "nbr_late_rate": nbr_late,
                "nbr_util_mean": nbr_util,
                "utilization": G.nodes[b].get("utilization", 0.0),
                "late": G.nodes[b].get("late", 0)
            })
        frames.append(pd.DataFrame(feats))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_path, index=False)
    print("Wrote", out_path, "rows:", len(out))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--edges", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", default="data/features.csv")
    a = ap.parse_args()
    main(a.edges, a.events, a.out)
