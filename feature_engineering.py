"""
feature_engineering.py  v3
═══════════════════════════════════════════════════════════════════
Extracts features for ALL three node types in the heterogeneous graph:

    account  — one row per unique Account ID         (~515k nodes)
    bank     — one row per unique Bank ID            (~72 nodes)
    entity   — one row per unique Entity ID          (~varies)

And two edge tables:
    account→account edges  (from Account / Account.1 pairs)
    account→bank    edges  (from Account / To Bank pairs)
    entity→account  edges  (from Entity ID / Account pairs)

OUTPUT FILES (written to output_dir, or CWD if not given):
    node_account.parquet   — account-level features + label
    node_bank.parquet      — bank-level features
    node_entity.parquet    — entity-level features
    edge_acc_acc.parquet   — account→account edge features
    edge_acc_bank.parquet  — account→bank    edge features
    edge_ent_acc.parquet   — entity→account  edge features  (ownership)
"""

import os
import pandas as pd
import numpy as np
import warnings
import time
from pathlib import Path

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
CHUNK_SIZE      = 500_000
ODD_HOUR_START  = 0
ODD_HOUR_END    = 6
STRUCTURING_THR = 10_000
STRUCTURING_PCT = 0.10

ENTITY_TYPE_MAP = {
    "sole proprietorship": 0,
    "partnership":         1,
    "corporation":         2,
    "country":             3,
}


# ─────────────────────────────────────────────────────────────
# STEP 1 — LOAD
# ─────────────────────────────────────────────────────────────

def load_data(filepath: str) -> pd.DataFrame:
    print("─" * 55)
    print("STEP 1 — Loading data")
    print("─" * 55)

    t0 = time.time()
    df = pd.read_csv(filepath, engine="pyarrow")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    print(f"  load time    : {time.time()-t0:.1f}s — {len(df):,} rows")

    df["currency_switched"] = (
        df["Payment Currency"] != df["Receiving Currency"]
    ).astype(np.int8)
    df["is_odd"]     = df["Timestamp"].dt.hour.between(
        ODD_HOUR_START, ODD_HOUR_END - 1
    ).astype(np.int8)
    df["is_weekend"] = df["Timestamp"].dt.dayofweek.isin([5, 6]).astype(np.int8)

    def encode_entity(name):
        if pd.isna(name):
            return 4
        n = str(name).lower()
        for k, v in ENTITY_TYPE_MAP.items():
            if k in n:
                return v
        return 4
    df["entity_type_encoded"] = df["Entity Name"].apply(encode_entity).astype(np.int8)

    print(f"\n  Loaded       : {len(df):,} rows")
    print(f"  Accounts     : {df['Account'].nunique():,}")
    print(f"  Banks        : {df['Bank ID'].nunique():,}")
    print(f"  Entities     : {df['Entity ID'].nunique():,}")
    print(f"  Laundering % : {df['Is Laundering'].mean()*100:.3f}%\n")
    return df


# ─────────────────────────────────────────────────────────────
# STEP 2 — ACCOUNT NODE FEATURES
# ─────────────────────────────────────────────────────────────

def compute_account_features(df: pd.DataFrame) -> pd.DataFrame:
    print("─" * 55)
    print("STEP 2 — Account node features")
    print("─" * 55)

    # ── Graph topology ────────────────────────────────────────
    out_deg  = df.groupby("Account")["Account.1"].nunique().rename("out_degree")
    in_deg   = df.groupby("Account.1")["Account"].nunique().rename("in_degree")
    out_txns = df.groupby("Account")["Account.1"].count().rename("out_degree_txns")
    in_txns  = df.groupby("Account.1")["Account"].count().rename("in_degree_txns")
    topo     = pd.concat([out_deg, in_deg, out_txns, in_txns], axis=1).fillna(0)
    topo["degree_ratio"] = topo["out_degree"]      / (topo["in_degree"]      + 1e-6)
    topo["txn_ratio"]    = topo["out_degree_txns"] / (topo["in_degree_txns"] + 1e-6)

    # ── Amount / flow ─────────────────────────────────────────
    sent = df.groupby("Account").agg(
        total_paid      = ("Amount Paid",      "sum"),
        avg_paid        = ("Amount Paid",      "mean"),
        std_paid        = ("Amount Paid",      "std"),
        unique_pay_curr = ("Payment Currency", "nunique"),
    ).fillna(0)
    recv = df.groupby("Account.1").agg(
        total_received   = ("Amount Received",    "sum"),
        avg_received     = ("Amount Received",    "mean"),
        std_received     = ("Amount Received",    "std"),
        unique_recv_curr = ("Receiving Currency", "nunique"),
    ).fillna(0)
    amt = pd.concat([sent, recv], axis=1).fillna(0)
    amt["amount_flow_ratio"] = amt["total_paid"] / (amt["total_received"] + 1e-6)

    lo = STRUCTURING_THR * (1 - STRUCTURING_PCT)
    amt = amt.join(
        df[df["Amount Paid"].between(lo, STRUCTURING_THR)]
          .groupby("Account")["Amount Paid"].count().rename("struct_sent"), how="left"
    ).join(
        df[df["Amount Received"].between(lo, STRUCTURING_THR)]
          .groupby("Account.1")["Amount Received"].count().rename("struct_recv"), how="left"
    ).fillna(0)
    amt["structuring_score"] = amt["struct_sent"] + amt["struct_recv"]
    amt.drop(columns=["struct_sent", "struct_recv"], inplace=True)

    amt = amt.join(
        df.groupby("Account")["currency_switched"].sum().rename("cc_sent"), how="left"
    ).join(
        df.groupby("Account.1")["currency_switched"].sum().rename("cc_recv"), how="left"
    ).fillna(0)
    amt["total_currency_switches"] = amt["cc_sent"] + amt["cc_recv"]
    amt.drop(columns=["cc_sent", "cc_recv"], inplace=True)

    # ── Temporal ──────────────────────────────────────────────
    def flag_ratio(grp, col, name):
        g = df.groupby(grp)[col]
        return (g.sum() / (g.count() + 1e-6)).rename(name)

    odd_sent  = flag_ratio("Account",   "is_odd",     "odd_hour_ratio_sent")
    odd_recv  = flag_ratio("Account.1", "is_odd",     "odd_hour_ratio_recv")
    wknd_sent = flag_ratio("Account",   "is_weekend", "weekend_ratio_sent")
    wknd_recv = flag_ratio("Account.1", "is_weekend", "weekend_ratio_recv")

    ts_s    = df.groupby("Account")["Timestamp"].agg(["min","max"])
    ts_r    = df.groupby("Account.1")["Timestamp"].agg(["min","max"])
    age_s   = ((ts_s["max"]-ts_s["min"]).dt.total_seconds()/86400).rename("account_age_days_sent")
    age_r   = ((ts_r["max"]-ts_r["min"]).dt.total_seconds()/86400).rename("account_age_days_recv")

    hour_bin    = df["Timestamp"].dt.floor("1h")
    burst_max   = (df.groupby(["Account", hour_bin]).size()
                     .groupby(level=0).max().rename("burst_max_txns"))
    total_sent  = df.groupby("Account")["Amount Paid"].count().rename("total_txns_sent")
    burst_ratio = (burst_max / (total_sent + 1e-6)).rename("burst_ratio")

    med_send = df.groupby("Account")["Timestamp"].median()
    med_recv = df.groupby("Account.1")["Timestamp"].median()
    lag_df   = pd.concat([med_send.rename("ms"), med_recv.rename("mr")], axis=1)
    fwd_lag  = ((lag_df["ms"] - lag_df["mr"]).dt.total_seconds()
                 .clip(lower=0).rename("fwd_lag_seconds").fillna(0))

    temp = pd.concat([
        odd_sent, odd_recv, wknd_sent, wknd_recv,
        age_s, age_r, fwd_lag, burst_max, burst_ratio
    ], axis=1).fillna(0)

    # ── Network diversity ─────────────────────────────────────
    banks_s  = df.groupby("Account")["To Bank"].nunique().rename("unique_banks_sent")
    banks_r  = df.groupby("Account.1")["From Bank"].nunique().rename("unique_banks_recv")
    fmt_div  = df.groupby("Account")["Payment Format"].nunique().rename("unique_formats_used")
    btc_flag = (
        df[df["Payment Format"] == "Bitcoin"]
          .groupby("Account")["Payment Format"].count()
          .gt(0).astype(int).rename("bitcoin_usage_flag")
    )
    net = pd.concat([banks_s, banks_r, fmt_div, btc_flag], axis=1).fillna(0)
    net["total_unique_banks"] = net["unique_banks_sent"] + net["unique_banks_recv"]

    # ── Label ─────────────────────────────────────────────────
    label = df.groupby("Account")["Is Laundering"].max().rename("label")

    # ── Assemble ──────────────────────────────────────────────
    acc = (topo.join(amt,   how="outer")
               .join(temp,  how="outer")
               .join(net,   how="outer")
               .join(label, how="left")
               .fillna(0))

    print(f"  shape        : {acc.shape}")
    print(f"  label dist   : {dict(acc['label'].value_counts().items())}")
    return acc


# ─────────────────────────────────────────────────────────────
# STEP 3 — BANK NODE FEATURES
# ─────────────────────────────────────────────────────────────

def compute_bank_features(df: pd.DataFrame) -> pd.DataFrame:
    print("─" * 55)
    print("STEP 3 — Bank node features")
    print("─" * 55)

    sent = df.groupby("From Bank").agg(
        out_txns         = ("Amount Paid",   "count"),
        total_out_volume = ("Amount Paid",   "sum"),
        avg_out_amount   = ("Amount Paid",   "mean"),
        unique_src_accts = ("Account",       "nunique"),
        unique_currencies_out = ("Payment Currency", "nunique"),
    )
    recv = df.groupby("To Bank").agg(
        in_txns          = ("Amount Received", "count"),
        total_in_volume  = ("Amount Received", "sum"),
        avg_in_amount    = ("Amount Received", "mean"),
        unique_dst_accts = ("Account.1",       "nunique"),
        unique_currencies_in = ("Receiving Currency", "nunique"),
    )

    bank = pd.concat([sent, recv], axis=1).fillna(0)
    bank["bank_degree_ratio"]  = bank["out_txns"]         / (bank["in_txns"]         + 1e-6)
    bank["bank_volume_ratio"]  = bank["total_out_volume"]  / (bank["total_in_volume"]  + 1e-6)

    launder_from = df.groupby("From Bank")["Is Laundering"].mean().rename("launder_exposure_out")
    launder_to   = df.groupby("To Bank")["Is Laundering"].mean().rename("launder_exposure_in")
    bank = bank.join(launder_from, how="left").join(launder_to, how="left").fillna(0)
    bank["bank_laundering_exposure"] = (bank["launder_exposure_out"] + bank["launder_exposure_in"]) / 2
    bank.drop(columns=["launder_exposure_out","launder_exposure_in"], inplace=True)

    odd_from = (df.groupby("From Bank")["is_odd"].sum() /
                (df.groupby("From Bank")["is_odd"].count() + 1e-6)).rename("odd_hour_ratio_out")
    odd_to   = (df.groupby("To Bank")["is_odd"].sum() /
                (df.groupby("To Bank")["is_odd"].count() + 1e-6)).rename("odd_hour_ratio_in")
    bank = bank.join(odd_from, how="left").join(odd_to, how="left").fillna(0)

    # FIX: Bank node features are keyed by From Bank / To Bank (bank name strings),
    # not by Bank ID. The original code joins bank_names by Bank ID but the index
    # is From Bank. We keep the bank name reference using the sending-side key.
    bank_names = df.drop_duplicates("From Bank").set_index("From Bank")["Bank Name"]
    bank = bank.join(bank_names.rename("bank_name"), how="left")

    print(f"  shape        : {bank.shape}")
    print(f"  banks found  : {len(bank):,}")
    return bank


# ─────────────────────────────────────────────────────────────
# STEP 4 — ENTITY NODE FEATURES
# ─────────────────────────────────────────────────────────────

def compute_entity_features(df: pd.DataFrame) -> pd.DataFrame:
    print("─" * 55)
    print("STEP 4 — Entity node features")
    print("─" * 55)

    entity = df.groupby("Entity ID").agg(
        entity_type_encoded  = ("entity_type_encoded",  "first"),
        entity_txn_count     = ("Amount Paid",           "count"),
        entity_total_paid    = ("Amount Paid",           "sum"),
        entity_avg_paid      = ("Amount Paid",           "mean"),
        entity_std_paid      = ("Amount Paid",           "std"),
        unique_accounts      = ("Account",               "nunique"),
        unique_banks         = ("From Bank",             "nunique"),
        laundering_exposure  = ("Is Laundering",         "mean"),
    ).fillna(0)

    odd = (df.groupby("Entity ID")["is_odd"].sum() /
           (df.groupby("Entity ID")["is_odd"].count() + 1e-6)).rename("entity_odd_hour_ratio")
    entity = entity.join(odd, how="left").fillna(0)

    entity_names = df.groupby("Entity ID")["Entity Name"].first()
    entity = entity.join(entity_names.rename("entity_name"), how="left")

    print(f"  shape        : {entity.shape}")
    print(f"  entities     : {len(entity):,}")
    print(f"  type dist    :\n{entity['entity_type_encoded'].value_counts().to_string()}")
    return entity


# ─────────────────────────────────────────────────────────────
# STEP 5 — ACCOUNT→ACCOUNT EDGE FEATURES
# ─────────────────────────────────────────────────────────────

def compute_acc_acc_edges(df: pd.DataFrame) -> pd.DataFrame:
    print("─" * 55)
    print("STEP 5 — Account→Account edge features")
    print("─" * 55)

    ref_time = df["Timestamp"].max()

    edges = df.groupby(["Account", "Account.1"]).agg(
        edge_weight              = ("Amount Paid",        "sum"),
        edge_txn_count           = ("Amount Paid",        "count"),
        edge_avg_amount          = ("Amount Paid",        "mean"),
        edge_last_ts             = ("Timestamp",          "max"),
        edge_currency_switch_sum = ("currency_switched",  "sum"),
        edge_laundering          = ("Is Laundering",      "max"),
    ).reset_index()

    edges["edge_recency_days"]    = (ref_time - edges["edge_last_ts"]).dt.total_seconds() / 86400
    edges["edge_is_repeated"]     = (edges["edge_txn_count"] > 1).astype(np.int8)
    edges["edge_currency_switch"] = edges["edge_currency_switch_sum"] / (edges["edge_txn_count"] + 1e-6)
    edges.drop(columns=["edge_last_ts", "edge_currency_switch_sum"], inplace=True)

    print(f"  shape        : {edges.shape}")
    return edges


# ─────────────────────────────────────────────────────────────
# STEP 6 — ACCOUNT→BANK EDGE FEATURES
# ─────────────────────────────────────────────────────────────

def compute_acc_bank_edges(df: pd.DataFrame) -> pd.DataFrame:
    print("─" * 55)
    print("STEP 6 — Account→Bank edge features")
    print("─" * 55)

    ref_time = df["Timestamp"].max()

    edges = df.groupby(["Account", "To Bank"]).agg(
        edge_weight     = ("Amount Paid",   "sum"),
        edge_txn_count  = ("Amount Paid",   "count"),
        edge_avg_amount = ("Amount Paid",   "mean"),
        edge_last_ts    = ("Timestamp",     "max"),
        edge_laundering = ("Is Laundering", "max"),
    ).reset_index()

    edges["edge_recency_days"] = (ref_time - edges["edge_last_ts"]).dt.total_seconds() / 86400
    edges["edge_is_repeated"]  = (edges["edge_txn_count"] > 1).astype(np.int8)
    edges.drop(columns=["edge_last_ts"], inplace=True)

    print(f"  shape        : {edges.shape}")
    return edges


# ─────────────────────────────────────────────────────────────
# STEP 7 — ENTITY→ACCOUNT EDGE FEATURES
# ─────────────────────────────────────────────────────────────

def compute_ent_acc_edges(df: pd.DataFrame) -> pd.DataFrame:
    print("─" * 55)
    print("STEP 7 — Entity→Account edge features")
    print("─" * 55)

    edges = df.groupby(["Entity ID", "Account"]).agg(
        edge_txn_count  = ("Amount Paid",   "count"),
        edge_weight     = ("Amount Paid",   "sum"),
        edge_avg_amount = ("Amount Paid",   "mean"),
        edge_laundering = ("Is Laundering", "max"),
    ).reset_index()

    edges["edge_is_repeated"] = (edges["edge_txn_count"] > 1).astype(np.int8)

    print(f"  shape        : {edges.shape}")
    return edges


# ─────────────────────────────────────────────────────────────
# STEP 8 — NORMALISE & EXPORT
# ─────────────────────────────────────────────────────────────

def normalise_and_export(acc, bank, entity,
                          e_acc_acc, e_acc_bank, e_ent_acc,
                          output_dir: Path):
    print("─" * 55)
    print("STEP 8 — Normalising and exporting")
    print("─" * 55)

    def log_minmax(df, log_cols, skip_cols):
        d = df.copy()
        for c in log_cols:
            if c in d.columns:
                d[c] = np.log1p(d[c])
        feat_cols  = [c for c in d.select_dtypes(include=[np.number]).columns
                      if c not in skip_cols]
        cmin, cmax = d[feat_cols].min(), d[feat_cols].max()
        d[feat_cols] = (d[feat_cols] - cmin) / (cmax - cmin + 1e-9)
        for c in feat_cols:
            d[c] = d[c].astype(np.float32)
        return d

    # ── Account nodes ────────────────────────────────────────
    acc_skip = {"label", "bitcoin_usage_flag"}
    acc_log  = ["total_paid", "total_received", "out_degree_txns",
                "in_degree_txns", "burst_max_txns", "structuring_score",
                "fwd_lag_seconds"]
    acc_out = log_minmax(acc, acc_log, acc_skip)
    acc_out["label"] = acc_out["label"].astype(np.int8)
    acc_path = output_dir / "node_account.parquet"
    acc_out.to_parquet(acc_path)
    _report(str(acc_path), acc_out)

    # ── Bank nodes ────────────────────────────────────────────
    bank_skip  = {"bank_name"}
    bank_log   = ["out_txns", "in_txns", "total_out_volume", "total_in_volume"]
    bank_feat  = bank.drop(columns=["bank_name"], errors="ignore")
    bank_out   = log_minmax(bank_feat, bank_log, bank_skip)
    bank_path  = output_dir / "node_bank.parquet"
    bank_out.to_parquet(bank_path)
    _report(str(bank_path), bank_out)

    # ── Entity nodes ──────────────────────────────────────────
    ent_skip  = {"entity_name", "entity_type_encoded"}
    ent_log   = ["entity_txn_count", "entity_total_paid"]
    ent_feat  = entity.drop(columns=["entity_name"], errors="ignore")
    ent_out   = log_minmax(ent_feat, ent_log, ent_skip)
    ent_out["entity_type_encoded"] = ent_out["entity_type_encoded"].astype(np.int8)
    ent_path  = output_dir / "node_entity.parquet"
    ent_out.to_parquet(ent_path)
    _report(str(ent_path), ent_out)

    # ── Edges ─────────────────────────────────────────────────
    def compress_edges(df, log_cols, id_cols):
        d = df.copy()
        for c in log_cols:
            if c in d.columns:
                d[c] = np.log1p(d[c]).astype(np.float32)
        for c in d.columns:
            if c in id_cols:
                continue
            if d[c].dtype == np.float64:
                d[c] = d[c].astype(np.float32)
            elif d[c].dtype == np.int64:
                d[c] = d[c].astype(np.int32)
        return d

    e_aa_out = compress_edges(e_acc_acc,  ["edge_weight","edge_avg_amount"], ["Account","Account.1"])
    e_aa_path = output_dir / "edge_acc_acc.parquet"
    e_aa_out.to_parquet(e_aa_path)
    _report(str(e_aa_path), e_aa_out)

    e_ab_out = compress_edges(e_acc_bank, ["edge_weight","edge_avg_amount"], ["Account","To Bank"])
    e_ab_path = output_dir / "edge_acc_bank.parquet"
    e_ab_out.to_parquet(e_ab_path)
    _report(str(e_ab_path), e_ab_out)

    e_ea_out = compress_edges(e_ent_acc,  ["edge_weight","edge_avg_amount"], ["Entity ID","Account"])
    e_ea_path = output_dir / "edge_ent_acc.parquet"
    e_ea_out.to_parquet(e_ea_path)
    _report(str(e_ea_path), e_ea_out)

    return acc_out, bank_out, ent_out, e_aa_out, e_ab_out, e_ea_out


def _report(name, df):
    mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"  → {name:<50} {str(df.shape):<18} {mb:.1f} MB")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def run_pipeline(filepath: str, output_dir: str = None):
    """
    Run full feature engineering pipeline.

    Parameters
    ----------
    filepath   : Path to the normalised/joined CSV.
    output_dir : Directory to write parquet files. Defaults to CWD.
    """
    t0 = time.time()
    out = Path(output_dir) if output_dir else Path.cwd()
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("MULE ACCOUNT DETECTION — FEATURE ENGINEERING v3")
    print("=" * 55)

    df       = load_data(filepath)
    acc      = compute_account_features(df)
    bank     = compute_bank_features(df)
    entity   = compute_entity_features(df)
    e_aa     = compute_acc_acc_edges(df)
    e_ab     = compute_acc_bank_edges(df)
    e_ea     = compute_ent_acc_edges(df)

    acc_out, bank_out, ent_out, e_aa_out, e_ab_out, e_ea_out = \
        normalise_and_export(acc, bank, entity, e_aa, e_ab, e_ea, out)

    print(f"\n✅ Done in {(time.time()-t0)/60:.1f} min")
    print(f"\nOutput files written to: {out}")

    return acc_out, bank_out, ent_out, e_aa_out, e_ab_out, e_ea_out


if __name__ == "__main__":
    filepath = r'/Users/samarth/Desktop/AML/Mule_account/data/joined_data.csv'
    run_pipeline(filepath)