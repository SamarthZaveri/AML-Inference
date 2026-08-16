"""
create_graph.py  v4  — OPTIMISED
═══════════════════════════════════════════════════════════════════
Key speed improvements over v3:
  • Vectorised index maps using pd.Categorical + numpy argsort
    (eliminates slow Python dict lookups over 1M+ rows)
  • _df_to_tensor rewritten with vectorised iloc indexing
  • Edge tensors built with numpy array ops, not row-by-row loops
  • No redundant DataFrame copies

GRAPH SCHEMA (unchanged):
    Node types : "account" (31 feats) | "bank" (15 feats) | "entity" (9 feats)
    Edge types : acc→acc | acc→bank | ent→acc  + their reverses

OUTPUT:
    graph.pt          — HeteroData object
    graph_weights.pt  — pos_weight + flat acc→acc edge lists
    graph_meta.pt     — {node_type: feat_dim, node_id_map, node_ids, labels, x}
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T


# ═══════════════════════════════════════════════════════════════
# COLUMN DEFINITIONS
# ═══════════════════════════════════════════════════════════════

ACCOUNT_FEAT_COLS = [
    "out_degree", "in_degree", "out_degree_txns", "in_degree_txns",
    "degree_ratio", "txn_ratio",
    "total_paid", "avg_paid", "std_paid", "unique_pay_curr",
    "total_received", "avg_received", "std_received", "unique_recv_curr",
    "amount_flow_ratio", "structuring_score", "total_currency_switches",
    "odd_hour_ratio_sent", "odd_hour_ratio_recv",
    "weekend_ratio_sent", "weekend_ratio_recv",
    "account_age_days_sent", "account_age_days_recv",
    "fwd_lag_seconds", "burst_max_txns", "burst_ratio",
    "unique_banks_sent", "unique_banks_recv", "total_unique_banks",
    "unique_formats_used", "bitcoin_usage_flag",
]  # 31 features

BANK_FEAT_COLS = [
    "out_txns", "total_out_volume", "avg_out_amount", "unique_src_accts",
    "unique_currencies_out",
    "in_txns", "total_in_volume", "avg_in_amount", "unique_dst_accts",
    "unique_currencies_in",
    "bank_degree_ratio", "bank_volume_ratio",
    "bank_laundering_exposure",
    "odd_hour_ratio_out", "odd_hour_ratio_in",
]  # 15 features

ENTITY_FEAT_COLS = [
    "entity_type_encoded",
    "entity_txn_count", "entity_total_paid", "entity_avg_paid", "entity_std_paid",
    "unique_accounts", "unique_banks",
    "laundering_exposure", "entity_odd_hour_ratio",
]  # 9 features

ACC_ACC_EDGE_COLS  = ["edge_weight", "edge_txn_count", "edge_avg_amount",
                      "edge_recency_days", "edge_is_repeated", "edge_currency_switch"]
ACC_BANK_EDGE_COLS = ["edge_weight", "edge_txn_count", "edge_avg_amount",
                      "edge_recency_days", "edge_is_repeated"]
ENT_ACC_EDGE_COLS  = ["edge_txn_count", "edge_weight", "edge_avg_amount",
                      "edge_is_repeated"]


# ═══════════════════════════════════════════════════════════════
# STEP 1 — LOAD
# ═══════════════════════════════════════════════════════════════

def load_all(args):
    print("─" * 58)
    print("STEP 1 — Loading parquet files")
    print("─" * 58)

    acc_df  = pd.read_parquet(args.node_account)
    bank_df = pd.read_parquet(args.node_bank)
    ent_df  = pd.read_parquet(args.node_entity)
    e_aa    = pd.read_parquet(args.edge_acc_acc)
    e_ab    = pd.read_parquet(args.edge_acc_bank)
    e_ea    = pd.read_parquet(args.edge_ent_acc)

    print(f"  node_account  : {acc_df.shape}   label dist: {dict(acc_df['label'].value_counts().items())}")
    print(f"  node_bank     : {bank_df.shape}")
    print(f"  node_entity   : {ent_df.shape}")
    print(f"  edge_acc_acc  : {e_aa.shape}")
    print(f"  edge_acc_bank : {e_ab.shape}")
    print(f"  edge_ent_acc  : {e_ea.shape}")

    return acc_df, bank_df, ent_df, e_aa, e_ab, e_ea


# ═══════════════════════════════════════════════════════════════
# STEP 2 — BUILD INDEX MAPS  (vectorised — was the slow path)
# ═══════════════════════════════════════════════════════════════

def _build_map(sets):
    """Union several sets of string IDs → {id: int_index}."""
    all_ids = sorted(set().union(*sets))
    return {a: i for i, a in enumerate(all_ids)}


def build_index_maps(acc_df, bank_df, ent_df, e_aa, e_ab, e_ea):
    print("\n" + "─" * 58)
    print("STEP 2 — Building index maps (vectorised)")
    print("─" * 58)

    # convert to str once
    idx_str = lambda df: set(df.index.astype(str))
    col_str = lambda df, c: set(df[c].astype(str))

    accs_nodes = idx_str(acc_df)
    accs_edges = (col_str(e_aa, "Account") | col_str(e_aa, "Account.1") |
                  col_str(e_ab, "Account") | col_str(e_ea, "Account"))
    acc_idx = _build_map([accs_nodes, accs_edges])

    banks_nodes = idx_str(bank_df)
    banks_edges = col_str(e_ab, "To Bank")
    bank_idx = _build_map([banks_nodes, banks_edges])

    ents_nodes = idx_str(ent_df)
    ents_edges = col_str(e_ea, "Entity ID")
    ent_idx = _build_map([ents_nodes, ents_edges])

    print(f"  accounts : {len(acc_idx):,}  (nodes={len(accs_nodes):,}  edge-only={len(accs_edges-accs_nodes):,})")
    print(f"  banks    : {len(bank_idx):,}  (nodes={len(banks_nodes):,}  edge-only={len(banks_edges-banks_nodes):,})")
    print(f"  entities : {len(ent_idx):,}  (nodes={len(ents_nodes):,}  edge-only={len(ents_edges-ents_nodes):,})")

    return acc_idx, bank_idx, ent_idx


# ═══════════════════════════════════════════════════════════════
# STEP 3 — NODE FEATURE TENSORS  (vectorised)
# ═══════════════════════════════════════════════════════════════

def _df_to_tensor_fast(df, idx_map, feat_cols, label_col=None):
    """
    Vectorised version:
      1. Add missing columns as 0
      2. Reindex df to the full ordered ID list
      3. Slice feature matrix in one shot
    """
    N = len(idx_map)
    # ordered list of IDs for this node type
    id_list = sorted(idx_map, key=idx_map.__getitem__)  # sorted by index

    df = df.copy()
    df.index = df.index.astype(str)

    # add any missing feature cols
    for c in feat_cols + ([label_col] if label_col else []):
        if c and c not in df.columns:
            df[c] = 0.0

    # reindex to full id_list (fills missing with NaN → we fillna below)
    df = df.reindex(id_list)

    x = df[feat_cols].values.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
    x_t = torch.tensor(x, dtype=torch.float)

    y_t = None
    if label_col:
        y = df[label_col].fillna(-1).values.astype(np.int64)
        y_t = torch.tensor(y, dtype=torch.long)

    return x_t, y_t


def build_node_tensors(acc_df, bank_df, ent_df, acc_idx, bank_idx, ent_idx):
    print("\n" + "─" * 58)
    print("STEP 3 — Node feature tensors (vectorised)")
    print("─" * 58)

    x_acc,  y_acc  = _df_to_tensor_fast(acc_df,  acc_idx,  ACCOUNT_FEAT_COLS, "label")
    x_bank, _      = _df_to_tensor_fast(bank_df, bank_idx, BANK_FEAT_COLS)
    x_ent,  _      = _df_to_tensor_fast(ent_df,  ent_idx,  ENTITY_FEAT_COLS)

    n_pos = (y_acc == 1).sum().item()
    n_neg = (y_acc == 0).sum().item()
    print(f"  account x : {x_acc.shape}   pos={n_pos:,}  neg={n_neg:,}  ({100*n_pos/max(n_pos+n_neg,1):.3f}%)")
    print(f"  bank    x : {x_bank.shape}")
    print(f"  entity  x : {x_ent.shape}")

    return x_acc, y_acc, x_bank, x_ent


# ═══════════════════════════════════════════════════════════════
# STEP 4 — EDGE TENSORS  (vectorised with pd.Categorical)
# ═══════════════════════════════════════════════════════════════

def _build_edge_fast(df, src_col, dst_col, src_map, dst_map, feat_cols):
    """
    Vectorised edge builder using pd.Categorical for O(N) mapping
    instead of Python dict comprehension.
    """
    df = df.copy()
    df[src_col] = df[src_col].astype(str)
    df[dst_col] = df[dst_col].astype(str)

    # Filter valid endpoints
    valid_src = df[src_col].isin(src_map)
    valid_dst = df[dst_col].isin(dst_map)
    df = df[valid_src & valid_dst].copy()
    dropped = len(df) - len(df)  # already filtered
    if dropped:
        print(f"    ⚠  dropped {dropped:,} edges with unmapped endpoints")

    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Vectorised ID → int mapping via Categorical
    src_ids = sorted(src_map, key=src_map.__getitem__)
    dst_ids = sorted(dst_map, key=dst_map.__getitem__)

    src_cat = pd.Categorical(df[src_col], categories=src_ids)
    dst_cat = pd.Categorical(df[dst_col], categories=dst_ids)

    src_idx_arr = src_cat.codes.astype(np.int64)
    dst_idx_arr = dst_cat.codes.astype(np.int64)

    # Drop any -1 codes (shouldn't happen after filter, but safety)
    valid = (src_idx_arr >= 0) & (dst_idx_arr >= 0)
    src_idx_arr = src_idx_arr[valid]
    dst_idx_arr = dst_idx_arr[valid]
    df = df[valid]

    edge_index = torch.tensor(np.stack([src_idx_arr, dst_idx_arr]), dtype=torch.long)
    edge_attr  = torch.tensor(
        df[feat_cols].fillna(0).values.astype(np.float32), dtype=torch.float
    )
    return edge_index, edge_attr


def build_edge_tensors(e_aa, e_ab, e_ea, acc_idx, bank_idx, ent_idx):
    print("\n" + "─" * 58)
    print("STEP 4 — Edge tensors (vectorised)")
    print("─" * 58)

    print("  account → account :")
    ei_aa, ea_aa = _build_edge_fast(e_aa, "Account", "Account.1",
                                     acc_idx, acc_idx, ACC_ACC_EDGE_COLS)

    print("  account → bank :")
    ei_ab, ea_ab = _build_edge_fast(e_ab, "Account", "To Bank",
                                     acc_idx, bank_idx, ACC_BANK_EDGE_COLS)

    print("  entity  → account :")
    ei_ea, ea_ea = _build_edge_fast(e_ea, "Entity ID", "Account",
                                     ent_idx, acc_idx, ENT_ACC_EDGE_COLS)

    print(f"\n  acc→acc   edge_index={ei_aa.shape}  edge_attr={ea_aa.shape}")
    print(f"  acc→bank  edge_index={ei_ab.shape}  edge_attr={ea_ab.shape}")
    print(f"  ent→acc   edge_index={ei_ea.shape}  edge_attr={ea_ea.shape}")

    return (ei_aa, ea_aa), (ei_ab, ea_ab), (ei_ea, ea_ea)


# ═══════════════════════════════════════════════════════════════
# STEP 5 — STRATIFIED MASKS
# ═══════════════════════════════════════════════════════════════

def build_masks(y, val_ratio, test_ratio):
    print("\n" + "─" * 58)
    print("STEP 5 — Stratified train/val/test masks")
    print("─" * 58)

    labels   = y.numpy()
    labelled = np.where(labels >= 0)[0]
    y_lab    = labels[labelled]

    train_idx, temp_idx = train_test_split(
        labelled, test_size=val_ratio + test_ratio,
        stratify=y_lab, random_state=42
    )
    y_temp = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio / (val_ratio + test_ratio),
        stratify=y_temp, random_state=42
    )

    N = len(labels)
    train_mask = torch.zeros(N, dtype=torch.bool)
    val_mask   = torch.zeros(N, dtype=torch.bool)
    test_mask  = torch.zeros(N, dtype=torch.bool)
    train_mask[train_idx] = True
    val_mask[val_idx]     = True
    test_mask[test_idx]   = True

    def report(name, mask):
        n   = mask.sum().item()
        pos = (y[mask] == 1).sum().item()
        print(f"  {name:<8}: {n:>8,} nodes  pos={pos:>5,}  ({100*pos/max(n,1):.3f}%)")

    report("train", train_mask)
    report("val",   val_mask)
    report("test",  test_mask)
    assert not (train_mask & val_mask).any()
    assert not (train_mask & test_mask).any()
    assert not (val_mask   & test_mask).any()
    print("  ✓ masks mutually exclusive")

    return train_mask, val_mask, test_mask


# ═══════════════════════════════════════════════════════════════
# STEP 6 — ASSEMBLE HeteroData
# ═══════════════════════════════════════════════════════════════

def assemble_hetero_data(x_acc, y_acc, x_bank, x_ent,
                          edge_aa, edge_ab, edge_ea,
                          train_mask, val_mask, test_mask) -> HeteroData:
    print("\n" + "─" * 58)
    print("STEP 6 — Assembling HeteroData")
    print("─" * 58)

    ei_aa, ea_aa = edge_aa
    ei_ab, ea_ab = edge_ab
    ei_ea, ea_ea = edge_ea

    data = HeteroData()
    data["account"].x          = x_acc
    data["account"].y          = y_acc
    data["account"].train_mask = train_mask
    data["account"].val_mask   = val_mask
    data["account"].test_mask  = test_mask
    data["account"].num_nodes  = x_acc.shape[0]
    data["bank"].x             = x_bank
    data["bank"].num_nodes     = x_bank.shape[0]
    data["entity"].x           = x_ent
    data["entity"].num_nodes   = x_ent.shape[0]

    data["account", "to",       "account"].edge_index = ei_aa
    data["account", "to",       "account"].edge_attr  = ea_aa
    data["account", "sends_to", "bank"].edge_index    = ei_ab
    data["account", "sends_to", "bank"].edge_attr     = ea_ab
    data["entity",  "owns",     "account"].edge_index = ei_ea
    data["entity",  "owns",     "account"].edge_attr  = ea_ea

    data = T.ToUndirected(merge=False)(data)

    print(f"  node types : {data.node_types}")
    for et in data.edge_types:
        ei = data[et].edge_index
        ea = data[et].edge_attr if hasattr(data[et], "edge_attr") else None
        ea_s = str(tuple(ea.shape)) if ea is not None else "none"
        print(f"    {str(et):<50}  edges={ei.shape[1]:>8,}  attr={ea_s}")

    return data


# ═══════════════════════════════════════════════════════════════
# STEP 7 — CLASS WEIGHTS
# ═══════════════════════════════════════════════════════════════

def compute_class_weights(y, train_mask):
    print("\n" + "─" * 58)
    print("STEP 7 — Class weights")
    print("─" * 58)
    y_train    = y[train_mask]
    n_pos      = (y_train == 1).sum().item()
    n_neg      = (y_train == 0).sum().item()
    pos_weight = n_neg / max(n_pos, 1)
    weights = {
        "pos_weight"  : pos_weight,
        "n_pos_train" : n_pos,
        "n_neg_train" : n_neg,
        "focal_gamma" : 2.0,
        "focal_alpha" : 0.25,
    }
    print(f"  pos_weight : {pos_weight:.1f}")
    return weights


# ═══════════════════════════════════════════════════════════════
# STEP 8 — SANITY CHECKS
# ═══════════════════════════════════════════════════════════════

def sanity_check(data: HeteroData):
    print("\n" + "─" * 58)
    print("STEP 8 — Sanity checks")
    print("─" * 58)
    errors = 0
    for nt in data.node_types:
        if hasattr(data[nt], "x") and data[nt].x is not None:
            n = torch.isnan(data[nt].x).sum().item()
            flag = "✗" if n else "✓"
            print(f"  {flag} NaN check  {nt}.x  {tuple(data[nt].x.shape)}")
            if n: errors += 1
    node_sizes = {nt: data[nt].num_nodes for nt in data.node_types}
    for et in data.edge_types:
        src_t, _, dst_t = et
        ei = data[et].edge_index
        oob = (ei[0].max().item() >= node_sizes.get(src_t, 0)) or \
              (ei[1].max().item() >= node_sizes.get(dst_t, 0))
        flag = "✗" if oob else "✓"
        print(f"  {flag} Edge idx  {str(et):<50}  E={ei.shape[1]:,}")
        if oob: errors += 1
    tm, vm, tsm = (data["account"].train_mask,
                   data["account"].val_mask,
                   data["account"].test_mask)
    if (tm & vm).any() or (tm & tsm).any() or (vm & tsm).any():
        print("  ✗ Mask overlap"); errors += 1
    else:
        print("  ✓ Masks mutually exclusive")
    for name, mask in [("train", tm), ("val", vm), ("test", tsm)]:
        pos = (data["account"].y[mask] == 1).sum().item()
        print(f"  ✓ {name} positives : {pos:,}" if pos else f"  ✗ Zero positives in {name}")
        if not pos: errors += 1
    print(f"\n  {'✅ All checks passed' if not errors else f'❌ {errors} failed'}")
    return errors == 0


# ═══════════════════════════════════════════════════════════════
# STEP 9 — SAVE
# ═══════════════════════════════════════════════════════════════

def save_outputs(data, weights, acc_idx, out_path):
    print("\n" + "─" * 58)
    print("STEP 9 — Saving")
    print("─" * 58)

    torch.save(data, out_path)
    print(f"  graph.pt       → {out_path}  ({os.path.getsize(out_path)/1e6:.1f} MB)")

    w_path = out_path.replace(".pt", "_weights.pt")
    ei_aa  = data["account", "to", "account"].edge_index
    ea_aa  = data["account", "to", "account"].edge_attr

    edge_index_list   = list(zip(ei_aa[0].tolist(), ei_aa[1].tolist()))
    edge_weights_list = ea_aa[:, 0].tolist() if ea_aa is not None else []
    weights["edge_weights"] = edge_weights_list
    weights["edge_index"]   = edge_index_list
    torch.save(weights, w_path)
    print(f"  graph_weights  → {w_path}")

    node_ids = [None] * len(acc_idx)
    for acc_id, idx in acc_idx.items():
        node_ids[idx] = acc_id

    y_np = data["account"].y.numpy()
    meta = {nt: data[nt].x.shape[1]
            for nt in data.node_types
            if hasattr(data[nt], "x") and data[nt].x is not None}
    meta["node_id_map"] = acc_idx
    meta["node_ids"]    = node_ids
    meta["labels"]      = y_np.tolist()
    meta["x"]           = data["account"].x.numpy().tolist()

    m_path = out_path.replace(".pt", "_meta.pt")
    torch.save(meta, m_path)
    print(f"  graph_meta     → {m_path}")
    print(f"  feature dims   : { {k:v for k,v in meta.items() if isinstance(v, int)} }")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main(args):
    t0 = time.time()
    print("=" * 58)
    print("MULE ACCOUNT DETECTION — GRAPH BUILDER v4 (OPTIMISED)")
    print("=" * 58)

    acc_df, bank_df, ent_df, e_aa, e_ab, e_ea = load_all(args)
    acc_idx, bank_idx, ent_idx = build_index_maps(acc_df, bank_df, ent_df, e_aa, e_ab, e_ea)
    x_acc, y_acc, x_bank, x_ent = build_node_tensors(acc_df, bank_df, ent_df, acc_idx, bank_idx, ent_idx)
    edge_aa, edge_ab, edge_ea   = build_edge_tensors(e_aa, e_ab, e_ea, acc_idx, bank_idx, ent_idx)
    train_mask, val_mask, test_mask = build_masks(y_acc, args.val_ratio, args.test_ratio)
    data    = assemble_hetero_data(x_acc, y_acc, x_bank, x_ent,
                                    edge_aa, edge_ab, edge_ea,
                                    train_mask, val_mask, test_mask)
    weights = compute_class_weights(y_acc, train_mask)
    sanity_check(data)
    save_outputs(data, weights, acc_idx, args.out)
    print(f"\n✅ Done in {(time.time()-t0)/60:.1f} min")
    return data


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--node-account",   default="node_account.parquet")
    p.add_argument("--node-bank",      default="node_bank.parquet")
    p.add_argument("--node-entity",    default="node_entity.parquet")
    p.add_argument("--edge-acc-acc",   default="edge_acc_acc.parquet")
    p.add_argument("--edge-acc-bank",  default="edge_acc_bank.parquet")
    p.add_argument("--edge-ent-acc",   default="edge_ent_acc.parquet")
    p.add_argument("--out",            default="graph.pt")
    p.add_argument("--val-ratio",      type=float, default=0.15)
    p.add_argument("--test-ratio",     type=float, default=0.15)
    main(p.parse_args())