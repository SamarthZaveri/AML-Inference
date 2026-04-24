"""
AML Detection Pipeline — Reusable Entry Point
==============================================
Usage:
    python pipeline.py --input transactions.csv [--output ./output] [--hops 3]

Steps:
    1. Schema validation / join-based normalisation
    2. Feature engineering
    3. Graph construction
    4. RGCN inference (mule-account scoring)
    5. Top-10 mule network extraction + visualisation
"""

import argparse
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pipeline")

# ── canonical column order ────────────────────────────────────────────────────
REQUIRED_COLUMNS = [
    "Timestamp", "From Bank", "Account", "To Bank", "Account.1",
    "Amount Received", "Receiving Currency", "Amount Paid",
    "Payment Currency", "Payment Format", "Is Laundering",
    "Bank Name", "Bank ID", "Account Number", "Entity ID", "Entity Name",
]


def validate_schema(csv_path: Path) -> bool:
    """Return True when the CSV already has the canonical schema."""
    import pandas as pd
    try:
        df = pd.read_csv(csv_path, nrows=0, sep=None, engine="python")
        return list(df.columns) == REQUIRED_COLUMNS
    except Exception as exc:
        log.warning("Schema read error: %s", exc)
        return False


def run_pipeline(csv_path: Path, output_dir: Path, hops: int = 3) -> dict:
    """
    Execute the full AML detection pipeline.

    Parameters
    ----------
    csv_path   : Path to the raw transaction CSV.
    output_dir : Directory where all artefacts are written.
    hops       : Network-expansion depth around top mule nodes (3–5).

    Returns
    -------
    dict with keys: normalised_csv, graph_dir,
                    predictions_csv, visualisation_dir
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    artefacts: dict = {}

    t0 = time.time()

    log.info("▶ Step 1 — Using input CSV directly (no join step)")

    if not validate_schema(csv_path):
        log.warning("⚠ Schema mismatch — proceeding anyway (no join step implemented)")

    normalised_csv = csv_path.resolve()   # 🔥 absolute path fix
    artefacts["normalised_csv"] = normalised_csv

    # ── Step 2 : Feature engineering ─────────────────────────────────────────
    # FIX: was "preprocessig" (typo) and wrong function name "run_feature_engineering".
    # feature_engineering.run_pipeline() writes parquet files to whatever the
    # current working directory is, so we temporarily chdir to output_dir.
    log.info("▶ Step 2 — Feature engineering")
    import os
    from feature_engineering import run_pipeline as run_feature_engineering
    _orig_cwd = Path.cwd()
    os.chdir(output_dir)
    try:
        run_feature_engineering(str(normalised_csv))
    finally:
        os.chdir(_orig_cwd)
    log.info("  ✓ Feature parquets written to %s", output_dir)
    artefacts["features_dir"] = output_dir

    # ── Step 3 : Graph construction ───────────────────────────────────────────
    # FIX: create_graph.py exposes main(args), not run_pipeline().
    # We build a namespace that matches its argparse spec.
    log.info("▶ Step 3 — Graph construction")
    import argparse as _ap
    from create_graph import main as run_create_graph

    graph_dir = output_dir / "graph"
    graph_dir.mkdir(exist_ok=True)

    graph_args = _ap.Namespace(
        node_account  = str(output_dir / "node_account.parquet"),
        node_bank     = str(output_dir / "node_bank.parquet"),
        node_entity   = str(output_dir / "node_entity.parquet"),
        edge_acc_acc  = str(output_dir / "edge_acc_acc.parquet"),
        edge_acc_bank = str(output_dir / "edge_acc_bank.parquet"),
        edge_ent_acc  = str(output_dir / "edge_ent_acc.parquet"),
        out           = str(graph_dir / "graph.pt"),
        val_ratio     = 0.15,
        test_ratio    = 0.15,
    )
    run_create_graph(graph_args)
    log.info("  ✓ Graph artefacts → %s", graph_dir)
    artefacts["graph_dir"] = graph_dir

    # ── Step 4 : RGCN inference ───────────────────────────────────────────────
    log.info("▶ Step 4 — RGCN inference")
    from rgcn_inference import run_inference
    predictions_csv = output_dir / "predictions.csv"
    run_inference(graph_dir, predictions_csv)
    log.info("  ✓ Predictions CSV → %s", predictions_csv)
    artefacts["predictions_csv"] = predictions_csv

    # ── Step 5 : Mule-network extraction + visualisation ─────────────────────
    # FIX: mule_network expects the raw transactions CSV (for KPI computation),
    # not a features CSV. Pass normalised_csv (the joined transactions file).
    log.info("▶ Step 5 — Mule-network extraction & visualisation (hops=%d)", hops)
    from mule_network import extract_mule_network
    from visualise import render_visualisation

    network_data = extract_mule_network(
        predictions_csv=predictions_csv,
        graph_dir=graph_dir,
        features_csv=normalised_csv,      # raw transactions CSV for KPI groupby
        top_n=10,
        hops=hops,
    )
    vis_dir = output_dir / "visualisation"
    vis_dir.mkdir(exist_ok=True)
    render_visualisation(network_data, vis_dir)
    log.info("  ✓ Visualisation subgraphs → %s", vis_dir)
    artefacts["visualisation_dir"] = vis_dir

    elapsed = time.time() - t0
    log.info("Pipeline complete in %.1fs  |  artefacts: %s", elapsed, output_dir)
    return artefacts


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AML Detection Pipeline")
    parser.add_argument("--input",  required=True,  help="Path to raw transaction CSV")
    parser.add_argument("--output", default="./output", help="Output directory (default: ./output)")
    parser.add_argument("--hops",   type=int, default=3,
                        help="Graph expansion hops around top-10 mule nodes (3–5, default: 3)")
    args = parser.parse_args()

    csv_path   = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    hops       = max(3, min(5, args.hops))   # clamp 3–5

    if not csv_path.exists():
        log.error("Input file not found: %s", csv_path)
        sys.exit(1)
    if csv_path.suffix.lower() != ".csv":
        log.error("Only .csv files are accepted. Got: %s", csv_path.suffix)
        sys.exit(1)

    run_pipeline(csv_path, output_dir, hops=hops)


if __name__ == "__main__":
    main()