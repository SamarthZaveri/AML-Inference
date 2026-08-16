"""
graph/mule_network.py — OPTIMIZED v2
=====================================
Changes vs v1:
  • Adjacency dicts built with vectorised numpy ops (was a Python loop
    over 1M+ tuples — the main speed regression in the original).
  • Returns `per_seed` dict that visualise.py requires.  The original
    omitted this key, causing render_visualisation() to silently skip
    every seed.
  • Direction labels (upstream / downstream) computed per seed so the
    hierarchical layout in visualise.py works correctly.
"""

from pathlib import Path
from collections import deque
import pickle
import logging
import heapq

import numpy as np
import pandas as pd

log = logging.getLogger("pipeline.mule_network")


# ─────────────────────────────────────────────────────────────
# LOAD UTILS
# ─────────────────────────────────────────────────────────────
def _load(path: Path):
    try:
        import torch
        return torch.load(path, map_location="cpu")
    except Exception:
        with open(path, "rb") as fh:
            return pickle.load(fh)


# ─────────────────────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────────────────────
def extract_mule_network(
    predictions_csv: Path,
    graph_dir: Path,
    features_csv: Path,
    top_n: int = 10,
    hops: int = 2,
    fan_out: int = 15,
) -> dict:

    hops = max(2, min(5, hops))

    # ── LOAD DATA ─────────────────────────────────────────────
    preds         = pd.read_csv(predictions_csv)
    graph_meta    = _load(graph_dir / "graph_meta.pt")
    graph_weights = _load(graph_dir / "graph_weights.pt")
    features      = pd.read_csv(features_csv)

    node_ids   = graph_meta["node_ids"]
    node_id_map = graph_meta["node_id_map"]
    N           = len(node_ids)

    scores_arr = preds.set_index("account_id")["mule_score"].to_dict()

    # ── SCORE CACHE ────────────────────────────────────────────
    score_cache = np.zeros(N, dtype=np.float32)
    for nid, idx in node_id_map.items():
        score_cache[idx] = scores_arr.get(nid, 0.0)

    def _score(idx: int) -> float:
        return float(score_cache[idx]) if idx < N else 0.0

    # ── BUILD ADJACENCY (vectorised — was a Python loop) ───────
    edge_index   = graph_weights.get("edge_index", [])
    edge_weights_list = graph_weights.get("edge_weights", [])

    adj_out: dict[int, list[tuple[int, float]]] = {}
    adj_in:  dict[int, list[tuple[int, float]]] = {}

    if edge_index:
        ei_arr = np.array(edge_index, dtype=np.int64)        # (E, 2)
        ew_arr = np.array(edge_weights_list, dtype=np.float32) if edge_weights_list else np.ones(len(ei_arr), dtype=np.float32)

        # pad weights if shorter than edges
        if len(ew_arr) < len(ei_arr):
            ew_arr = np.concatenate([ew_arr, np.ones(len(ei_arr) - len(ew_arr), dtype=np.float32)])

        src_arr = ei_arr[:, 0]
        dst_arr = ei_arr[:, 1]

        # group by source for adj_out
        sort_out = np.argsort(src_arr, kind="stable")
        s_sorted = src_arr[sort_out]
        d_sorted = dst_arr[sort_out]
        w_sorted = ew_arr[sort_out]
        splits_out = np.where(np.diff(s_sorted))[0] + 1
        for grp_s, grp_d, grp_w in zip(
            np.split(s_sorted, splits_out),
            np.split(d_sorted, splits_out),
            np.split(w_sorted, splits_out),
        ):
            adj_out[int(grp_s[0])] = list(zip(grp_d.tolist(), grp_w.tolist()))

        # group by dest for adj_in
        sort_in = np.argsort(dst_arr, kind="stable")
        d_s2 = dst_arr[sort_in]
        s_s2 = src_arr[sort_in]
        w_s2 = ew_arr[sort_in]
        splits_in = np.where(np.diff(d_s2))[0] + 1
        for grp_d, grp_s, grp_w in zip(
            np.split(d_s2, splits_in),
            np.split(s_s2, splits_in),
            np.split(w_s2, splits_in),
        ):
            adj_in[int(grp_d[0])] = list(zip(grp_s.tolist(), grp_w.tolist()))

    log.info("Adjacency built: %d out-nodes, %d in-nodes", len(adj_out), len(adj_in))

    # ── SELECT TOP SEEDS ───────────────────────────────────────
    seeds_df = preds.nsmallest(top_n, "rank")
    seed_ids = seeds_df["account_id"].tolist()
    seed_set = set(map(str, seed_ids))

    log.info("Top-%d mule seeds: %s …", top_n, seed_ids[:3])

    # ── KPI COMPUTATION ────────────────────────────────────────
    kpi_map = _compute_kpis_fast(features)

    # ── PER-SEED BFS + SUBGRAPH BUILDER ───────────────────────
    per_seed: dict[str, dict] = {}
    all_nodes_global: dict[str, dict] = {}
    all_edges_global: dict[tuple, dict] = {}

    for seed_id in seed_ids:
        seed_str = str(seed_id)
        seed_idx = node_id_map.get(seed_str)
        if seed_idx is None:
            log.warning("Seed %s not found in node_id_map", seed_id)
            continue

        visited: dict[int, int] = {seed_idx: 0}   # idx → hop
        # track direction: True = came via outgoing edge from seed (downstream)
        direction: dict[int, str] = {seed_idx: "seed"}
        queue = deque([(seed_idx, 0)])
        seed_edges: dict[tuple, dict] = {}

        while queue:
            node_idx, depth = queue.popleft()
            if depth >= hops:
                continue

            # OUTGOING (downstream)
            neighbours = adj_out.get(node_idx, [])
            ranked = heapq.nlargest(
                fan_out, neighbours,
                key=lambda x: x[1] * (_score(x[0]) + 1e-6),
            )
            for nb_idx, w in ranked:
                key = (node_idx, nb_idx)
                if key not in seed_edges:
                    seed_edges[key] = {"src": node_ids[node_idx], "dst": node_ids[nb_idx],
                                       "weight": float(w), "txn_count": 1}
                else:
                    seed_edges[key]["weight"] = max(seed_edges[key]["weight"], float(w))
                    seed_edges[key]["txn_count"] += 1
                if nb_idx not in visited:
                    visited[nb_idx] = depth + 1
                    direction[nb_idx] = "downstream"
                    queue.append((nb_idx, depth + 1))

            # INCOMING (upstream)
            neighbours_in = adj_in.get(node_idx, [])
            ranked_in = heapq.nlargest(
                fan_out, neighbours_in,
                key=lambda x: x[1] * (_score(x[0]) + 1e-6),
            )
            for nb_idx, w in ranked_in:
                key = (nb_idx, node_idx)
                if key not in seed_edges:
                    seed_edges[key] = {"src": node_ids[nb_idx], "dst": node_ids[node_idx],
                                       "weight": float(w), "txn_count": 1}
                else:
                    seed_edges[key]["weight"] = max(seed_edges[key]["weight"], float(w))
                    seed_edges[key]["txn_count"] += 1
                if nb_idx not in visited:
                    visited[nb_idx] = depth + 1
                    direction[nb_idx] = "upstream"
                    queue.append((nb_idx, depth + 1))

        # build node list for this seed
        seed_nodes = []
        for nidx, hop in visited.items():
            if nidx >= N:
                continue
            nid = node_ids[nidx]
            nd = {
                "id":         nid,
                "mule_score": float(scores_arr.get(nid, 0.0)),
                "is_seed":    str(nid) == seed_str,
                "hop":        hop,
                "direction":  direction.get(nidx, "downstream"),
                "kpi":        kpi_map.get(str(nid), {}),
            }
            seed_nodes.append(nd)
            all_nodes_global[str(nid)] = nd

        n_up   = sum(1 for n in seed_nodes if n["direction"] == "upstream")
        n_down = sum(1 for n in seed_nodes if n["direction"] == "downstream")

        per_seed[seed_str] = {
            "nodes": seed_nodes,
            "edges": list(seed_edges.values()),
            "stats": {
                "seed_score":  float(scores_arr.get(seed_id, 0.0)),
                "upstream":    n_up,
                "downstream":  n_down,
                "total_nodes": len(seed_nodes),
                "total_edges": len(seed_edges),
            },
        }
        all_edges_global.update(seed_edges)

    log.info(
        "Network: %d nodes, %d edges across %d seeds (hops=%d)",
        len(all_nodes_global), len(all_edges_global), len(per_seed), hops,
    )

    return {
        "seeds":      seed_ids,
        "nodes":      list(all_nodes_global.values()),
        "edges":      list(all_edges_global.values()),
        "per_seed":   per_seed,          # ← required by visualise.py
        "graph_meta": graph_meta,
    }


# ─────────────────────────────────────────────────────────────
# FAST KPI FUNCTION
# ─────────────────────────────────────────────────────────────
def _compute_kpis_fast(df: pd.DataFrame) -> dict:

    sent = df.groupby("Account").agg(
        total_sent=("Amount Paid", "sum"),
        txn_count_sent=("Amount Paid", "count"),
        avg_sent=("Amount Paid", "mean"),
        unique_receivers=("Account.1", "nunique"),
        laundering_flag=("Is Laundering", "max"),
        currencies_used=("Payment Currency", "nunique"),
    )

    recv = df.groupby("Account.1").agg(
        total_received=("Amount Received", "sum"),
        txn_count_received=("Amount Received", "count"),
        avg_received=("Amount Received", "mean"),
        unique_senders=("Account", "nunique"),
        laundering_flag_recv=("Is Laundering", "max"),
    )

    sent.index = sent.index.astype(str)
    recv.index = recv.index.astype(str)

    kpis = sent.join(recv, how="outer")
    kpis["laundering_flag"] = kpis[["laundering_flag", "laundering_flag_recv"]].max(axis=1)
    kpis = kpis.drop(columns=["laundering_flag_recv"]).fillna(0)

    result = {}
    for acc, row in kpis.iterrows():
        result[acc] = {
            k: int(v) if isinstance(v, (np.integer, int)) else float(v)
            for k, v in row.items()
        }

    return result