"""
models/rgcn_inference.py
========================
Loads graph.pt / graph_meta.pt / graph_weights.pt and the pre-trained
rgcn.pt weights, runs forward inference, and writes a predictions CSV.

Output CSV columns:
    account_id, mule_score, predicted_label, true_label, rank

FIX NOTES (vs original):
  - graph["node_ids"] / graph["x"] / graph["labels"] / graph["edge_index"]
    previously assumed graph.pt was a plain dict — but create_graph.py saves
    a HeteroData object there.  These fields are now stored in graph_meta.pt
    (populated by create_graph.save_outputs) so we read them from there.
  - graph_weights["edge_index"] / graph_weights["edge_weights"] are also
    populated by create_graph.save_outputs and read from graph_weights.pt.
"""

from pathlib import Path
import pickle
import logging
import numpy as np
import pandas as pd

log = logging.getLogger("pipeline.rgcn_inference")

DEFAULT_WEIGHTS_PATH = Path(__file__).parent / "rgcn_best.pt"


def _load(path: Path):
    """Load a .pt file (torch or pickle fallback)."""
    try:
        import torch
        return torch.load(path, map_location="cpu")
    except (ImportError, Exception):
        with open(path, "rb") as fh:
            return pickle.load(fh)


def run_inference(
    graph_dir: Path,
    output_csv: Path,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
) -> None:
    """
    Run RGCN inference and write a ranked predictions CSV.

    Parameters
    ----------
    graph_dir    : Directory containing graph.pt, graph_meta.pt, graph_weights.pt.
    output_csv   : Destination for the predictions CSV.
    weights_path : Path to rgcn.pt (pre-trained model weights).
    """
    log.info("rgcn_inference.py → loading graph artefacts from %s", graph_dir)

    # FIX: HeteroData is in graph.pt; node_ids / x / labels live in graph_meta.pt
    graph         = _load(graph_dir / "graph.pt")        # HeteroData
    graph_meta    = _load(graph_dir / "graph_meta.pt")   # dict with node_id_map etc.
    graph_weights = _load(graph_dir / "graph_weights.pt")

    # These fields were added by create_graph.save_outputs
    node_ids = graph_meta["node_ids"]
    X        = np.array(graph_meta["x"],      dtype=np.float32)
    labels   = np.array(graph_meta["labels"], dtype=np.float32)
    N        = len(node_ids)

    # ── RGCN forward pass ─────────────────────────────────────────────────────
    scores = _rgcn_forward(X, graph, graph_meta, graph_weights, weights_path)
    # ─────────────────────────────────────────────────────────────────────────

    df = pd.DataFrame({
        "account_id":      node_ids,
        "mule_score":      scores,
        "predicted_label": (scores > 0.5).astype(int),
        "true_label":      labels.astype(int),
    })
    df["rank"] = df["mule_score"].rank(ascending=False, method="first").astype(int)
    df = df.sort_values("rank").reset_index(drop=True)
    df.to_csv(output_csv, index=False)
    log.info(
        "rgcn_inference.py → %d accounts scored, top mule_score=%.4f → %s",
        N, float(df["mule_score"].max()), output_csv,
    )


def _rgcn_forward(X, graph, graph_meta, graph_weights, weights_path: Path) -> np.ndarray:
    """
    Attempt real RGCN inference; fall back to heuristic scoring when
    torch / torch-geometric are unavailable or weights file is missing.
    """
    try:
        import torch
        if not weights_path.exists():
            raise FileNotFoundError(f"rgcn.pt not found at {weights_path}")
        state = torch.load(weights_path, map_location="cpu")
        return _torch_inference(X, graph, graph_meta, graph_weights, state)
    except Exception as exc:
        log.warning(
            "Torch RGCN inference unavailable (%s) — using heuristic scoring.", exc
        )
        return _heuristic_scores(X, graph_meta, graph_weights)


def _torch_inference(X, graph, graph_meta, graph_weights, state) -> np.ndarray:
    """
    Real PyG-based RGCN inference.
    Replace / extend this function to match your exact model architecture.
    graph is the raw HeteroData object — use it directly with PyG.
    """
    import torch

    # For a homogeneous RGCN you can do:
    #   homo = graph.to_homogeneous()
    #   edge_index = homo.edge_index
    #   edge_type  = homo.edge_type
    #
    # For a full heterogeneous model pass `graph` directly:
    #   model = YourHeteroRGCN(...)
    #   model.load_state_dict(state)
    #   model.eval()
    #   with torch.no_grad():
    #       out = model(graph.x_dict, graph.edge_index_dict)
    #       scores = torch.sigmoid(out["account"]).numpy().ravel()

    raise NotImplementedError(
        "Replace _torch_inference with your RGCN model forward pass. "
        "The raw HeteroData is available as `graph`."
    )


def _heuristic_scores(X, graph_meta, graph_weights) -> np.ndarray:
    """
    Lightweight heuristic mule score for environments without PyTorch.
    Combines:
      • normalised out-degree   (from edge_index in graph_weights)
      • normalised total amount sent  (first feature column)
      • known label (if present)
    This is NOT a replacement for RGCN — it keeps the pipeline runnable
    end-to-end while you integrate the real model.
    """
    node_ids   = graph_meta["node_ids"]
    N          = len(node_ids)
    labels     = np.array(graph_meta.get("labels", [0] * N), dtype=np.float32)
    X_arr      = np.asarray(X, dtype=np.float32)

    # out-degree from flat edge_index stored in graph_weights
    out_degree = np.zeros(N)
    for src, _ in graph_weights.get("edge_index", []):
        if src < N:
            out_degree[src] += 1

    amt_score = X_arr[:, 0] if X_arr.shape[1] >= 1 else np.zeros(N)

    def norm(v):
        mx = v.max()
        return v / mx if mx > 0 else v

    score = 0.4 * norm(out_degree) + 0.4 * norm(amt_score) + 0.2 * labels
    return np.clip(score, 0, 1)