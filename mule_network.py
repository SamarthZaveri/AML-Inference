"""
graph/mule_network.py
======================
1. Selects the top-10 mule accounts by mule_score.
2. Expands their sub-network using a heuristic BFS/weighted-DFS (3–5 hops).
3. Attaches KPI metrics per account for visualisation.

FIX NOTES (vs original):
  - graph["node_ids"] / graph["edge_index"] previously assumed graph.pt was a
    plain dict — it is a HeteroData object.  node_ids and the node_id_map are
    now read from graph_meta.pt (populated by create_graph.save_outputs).
    edge_index and edge_weights are read from graph_weights.pt.
  - graph_meta["node_id_map"] KeyError is resolved — create_graph now
    explicitly populates this key.
"""

from pathlib import Path
from collections import deque
import pickle
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("pipeline.mule_network")


def _load(path: Path):
    try:
        import torch
        return torch.load(path, map_location="cpu")
    except (ImportError, Exception):
        with open(path, "rb") as fh:
            return pickle.load(fh)


def extract_mule_network(
    predictions_csv: Path,
    graph_dir: Path,
    features_csv: Path,
    top_n: int = 10,
    hops: int = 3,
    fan_out: int = 15,
) -> dict:
    """
    Build the mule sub-network and attach KPI metrics.

    Parameters
    ----------
    predictions_csv : Output of rgcn_inference — ranked accounts.
    graph_dir       : Directory with graph.pt / graph_meta.pt / graph_weights.pt.
    features_csv    : Raw (joined) transactions CSV for KPI computation.
                      Must have columns: Account, Account.1, Amount Paid,
                      Amount Received, Is Laundering, Payment Currency.
    top_n           : Number of seed mule accounts (default 10).
    hops            : BFS expansion depth (3–5).
    fan_out         : Max neighbours expanded per node per hop.

    Returns
    -------
    dict with keys:
        seeds         — list of top-N mule account IDs
        nodes         — list of dicts  (id, mule_score, kpi, is_seed, hop)
        edges         — list of dicts  (src, dst, weight, txn_count)
        graph_meta    — raw graph_meta dict
    """
    hops = max(3, min(5, hops))

    # ── Load artefacts ────────────────────────────────────────────────────────
    preds      = pd.read_csv(predictions_csv)
    # FIX: graph.pt is HeteroData — don't index it as a dict.
    #      All flat data we need (node_ids, node_id_map, labels) lives in graph_meta.pt
    #      edge_index and edge_weights live in graph_weights.pt
    graph_meta    = _load(graph_dir / "graph_meta.pt")
    graph_weights = _load(graph_dir / "graph_weights.pt")
    features      = pd.read_csv(features_csv)

    # FIX: these keys are now guaranteed by create_graph.save_outputs
    node_ids    = graph_meta["node_ids"]          # list[str], index → account_id
    node_id_map = graph_meta["node_id_map"]       # dict[str, int], account_id → index
    N           = len(node_ids)

    scores_arr = preds.set_index("account_id")["mule_score"].to_dict()

    # Build adjacency from the flat edge lists stored in graph_weights
    edge_index   = graph_weights.get("edge_index",   [])   # list of (src_idx, dst_idx)
    edge_weights_list = graph_weights.get("edge_weights", [])

    # Pad edge weights with 1.0 if lengths differ
    if len(edge_weights_list) < len(edge_index):
        edge_weights_list = list(edge_weights_list) + [1.0] * (
            len(edge_index) - len(edge_weights_list)
        )

    adj_out: dict[int, list] = {}   # src → [(dst, weight)]
    adj_in:  dict[int, list] = {}   # dst → [(src, weight)]
    for (src, dst), w in zip(edge_index, edge_weights_list):
        adj_out.setdefault(src, []).append((dst, float(w)))
        adj_in.setdefault(dst, []).append((src, float(w)))

    # ── Select top-N seeds ────────────────────────────────────────────────────
    seeds_df = preds.nsmallest(top_n, "rank")
    seed_ids = seeds_df["account_id"].tolist()
    log.info("Top-%d mule seeds: %s …", top_n, seed_ids[:3])

    # ── Heuristic BFS expansion ───────────────────────────────────────────────
    def _score(node_idx: int) -> float:
        if node_idx >= N:
            return 0.0
        nid = node_ids[node_idx]
        return scores_arr.get(nid, 0.0)

    visited: dict[int, int] = {}   # node_idx → hop at which first reached
    queue: deque = deque()

    for sid in seed_ids:
        idx = node_id_map.get(str(sid))
        if idx is not None and idx not in visited:
            visited[idx] = 0
            queue.append((idx, 0))

    selected_edges: list[tuple] = []   # (src_idx, dst_idx, weight)

    while queue:
        node_idx, depth = queue.popleft()
        if depth >= hops:
            continue

        neighbours = adj_out.get(node_idx, [])
        ranked = sorted(
            neighbours,
            key=lambda x: x[1] * (_score(x[0]) + 1e-6),
            reverse=True,
        )[:fan_out]

        for nb_idx, w in ranked:
            selected_edges.append((node_idx, nb_idx, w))
            if nb_idx not in visited:
                visited[nb_idx] = depth + 1
                queue.append((nb_idx, depth + 1))

        in_neighbours = adj_in.get(node_idx, [])
        ranked_in = sorted(
            in_neighbours,
            key=lambda x: x[1] * (_score(x[0]) + 1e-6),
            reverse=True,
        )[:fan_out]

        for nb_idx, w in ranked_in:
            selected_edges.append((nb_idx, node_idx, w))
            if nb_idx not in visited:
                visited[nb_idx] = depth + 1
                queue.append((nb_idx, depth + 1))

    # ── Build KPI dict ────────────────────────────────────────────────────────
    kpi_map = _compute_kpis(features)

    # ── Serialise nodes ───────────────────────────────────────────────────────
    seed_set = set(str(s) for s in seed_ids)
    nodes = []
    for nidx, hop in visited.items():
        if nidx >= N:
            continue
        nid = node_ids[nidx]
        nodes.append({
            "id":         nid,
            "mule_score": float(scores_arr.get(nid, 0.0)),
            "is_seed":    str(nid) in seed_set,
            "hop":        hop,
            "kpi":        kpi_map.get(str(nid), {}),
        })

    # ── Serialise edges (deduplicate) ─────────────────────────────────────────
    edge_map: dict[tuple, dict] = {}
    for src_idx, dst_idx, w in selected_edges:
        if src_idx >= N or dst_idx >= N:
            continue
        key = (node_ids[src_idx], node_ids[dst_idx])
        if key not in edge_map:
            edge_map[key] = {"src": key[0], "dst": key[1], "weight": w, "txn_count": 1}
        else:
            edge_map[key]["weight"]    = max(edge_map[key]["weight"], w)
            edge_map[key]["txn_count"] += 1

    log.info(
        "Network: %d nodes, %d edges (hops=%d, seeds=%d)",
        len(nodes), len(edge_map), hops, len(seed_ids),
    )
    return {
        "seeds":      seed_ids,
        "nodes":      nodes,
        "edges":      list(edge_map.values()),
        "graph_meta": graph_meta,
    }


def _compute_kpis(df: pd.DataFrame) -> dict:
    """
    Per-account KPI metrics displayed in the visualisation tooltip.
    Returns dict: account_id (str) → kpi_dict
    """
    kpis = {}

    for acc, grp in df.groupby("Account"):
        kpis[str(acc)] = {
            "total_sent":       round(float(grp["Amount Paid"].sum()), 2),
            "txn_count_sent":   int(len(grp)),
            "avg_sent":         round(float(grp["Amount Paid"].mean()), 2),
            "unique_receivers": int(grp["Account.1"].nunique()),
            "laundering_flag":  int(grp["Is Laundering"].max()),
            "currencies_used":  int(grp["Payment Currency"].nunique()),
        }

    for acc, grp in df.groupby("Account.1"):
        entry = kpis.setdefault(str(acc), {})
        entry["total_received"]     = round(float(grp["Amount Received"].sum()), 2)
        entry["txn_count_received"] = int(len(grp))
        entry["avg_received"]       = round(float(grp["Amount Received"].mean()), 2)
        entry["unique_senders"]     = int(grp["Account"].nunique()),
        entry.setdefault("laundering_flag", int(grp["Is Laundering"].max()))
        entry["laundering_flag"] = max(
            entry["laundering_flag"], int(grp["Is Laundering"].max())
        )

    return kpis