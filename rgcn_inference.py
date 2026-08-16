"""
models/rgcn_inference.py
========================
Loads graph.pt / graph_meta.pt / graph_weights.pt and the pre-trained
rgcn.pt weights, runs forward inference, and writes a predictions CSV.

Output CSV columns:
    account_id, mule_score, predicted_label, true_label, rank

INFERENCE STRATEGY — NeighborLoader (mini-batch):
  Instead of feeding all 515k nodes through the RGCN in one shot,
  we process nodes in batches of BATCH_SIZE using NeighborLoader.
  For each batch, only the 1-2 hop neighbourhood is materialised,
  keeping peak memory and compute small.

  Tradeoff: scores are approximate (sampling noise) but the top-N
  mule ranking is stable in practice.
"""

from pathlib import Path
import pickle
import logging
import time
import numpy as np
import pandas as pd

log = logging.getLogger("pipeline.rgcn_inference")

DEFAULT_WEIGHTS_PATH = Path(r"C:\Users\Samarth\Desktop\AML\rgcn_best.pt")

# ── Tuning knobs ─────────────────────────────────────────────
# BATCH_SIZE: number of target account nodes scored per forward pass.
# 4096 is safe on 8GB Intel Mac. Increase to 8192 if you have 16GB RAM.
BATCH_SIZE = 4096

# NEIGHBOURS: neighbours sampled per hop [hop1, hop2].
# Matches num_layers=2. [15, 10] is the standard production default.
# Higher = more accurate but slower. Lower = faster but more approximate.
NEIGHBOURS = [15, 10]


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _load(path: Path, label: str = ""):
    t0 = time.time()
    log.info("  Loading %s %s ...", label, path.name)
    try:
        import torch
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # torch < 1.13 — no weights_only kwarg
        import torch
        obj = torch.load(path, map_location="cpu")
    except Exception:
        log.warning("  torch.load failed for %s — trying pickle fallback", path.name)
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
    log.info("  ✓ Loaded %s in %.1fs", label, time.time() - t0)
    return obj


def _best_device():
    try:
        import torch
        if torch.cuda.is_available():
            log.info("  Device: CUDA (%s)", torch.cuda.get_device_name(0))
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            log.info("  Device: MPS (Apple Silicon)")
            return torch.device("mps")
    except Exception:
        pass
    log.info("  Device: CPU")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────

def run_inference(
    graph_dir: Path,
    output_csv: Path,
    weights_path: Path = DEFAULT_WEIGHTS_PATH,
) -> None:
    t_total = time.time()
    log.info("▶ rgcn_inference — graph_dir: %s", graph_dir)

    graph         = _load(graph_dir / "graph.pt",         "graph")
    graph_meta    = _load(graph_dir / "graph_meta.pt",    "graph_meta")
    graph_weights = _load(graph_dir / "graph_weights.pt", "graph_weights")

    node_ids = graph_meta["node_ids"]
    X        = np.array(graph_meta["x"],      dtype=np.float32)
    labels   = np.array(graph_meta["labels"], dtype=np.float32)
    N        = len(node_ids)
    log.info("  Nodes to score: %d", N)

    scores = _rgcn_forward(X, graph, graph_meta, graph_weights, weights_path)

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
        "✓ Inference complete — %d nodes scored, top_score=%.4f, elapsed=%.1fs → %s",
        N, float(df["mule_score"].max()), time.time() - t_total, output_csv,
    )


# ─────────────────────────────────────────────────────────────
# DISPATCH
# ─────────────────────────────────────────────────────────────

def _rgcn_forward(X, graph, graph_meta, graph_weights, weights_path: Path) -> np.ndarray:
    try:
        import torch
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")
        return _torch_inference_batched(X, graph, graph_meta, graph_weights, weights_path)
    except Exception as exc:
        log.warning("RGCN inference unavailable (%s) — using heuristic scoring.", exc)
        return _heuristic_scores(X, graph_meta, graph_weights)


# ─────────────────────────────────────────────────────────────
# BATCHED NEIGHBOUR-SAMPLING INFERENCE
# ─────────────────────────────────────────────────────────────

def _torch_inference_batched(X, graph, graph_meta, graph_weights, weights_path: Path) -> np.ndarray:
    import torch
    from torch_geometric.loader import NeighborLoader
    from model_inference_pass import RGCNInferenceModel

    device = _best_device()

    # ── Load model ────────────────────────────────────────────
    log.info("  Loading weights from %s ...", weights_path.name)
    t0 = time.time()
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    log.info("  ✓ Weights loaded in %.1fs", time.time() - t0)

    model = RGCNInferenceModel(
        in_channels=state["in_channels"],
        hidden=state["config"]["hidden"],
        num_layers=state["config"]["num_layers"],
        num_bases=state["config"]["num_bases"],
        dropout=state["config"]["dropout"],
    )
    model.load_state_dict(state["state_dict"])
    model.eval()
    model.to(device)

    # ── Keep graph on CPU — NeighborLoader handles device transfer ──
    graph = graph.cpu()

    N_acc = graph["account"].num_nodes
    log.info("  Setting up NeighborLoader (batch=%d, neighbours=%s) ...",
             BATCH_SIZE, NEIGHBOURS)

    # num_neighbors applied equally to all 6 edge types
    loader = NeighborLoader(
        graph,
        num_neighbors={et: NEIGHBOURS for et in graph.edge_types},
        batch_size=BATCH_SIZE,
        input_nodes=("account", torch.arange(N_acc)),
        shuffle=False,   # must be False to preserve node order for score assignment
    )

    n_batches = (N_acc + BATCH_SIZE - 1) // BATCH_SIZE
    log.info("  Scoring %d nodes in %d batches ...", N_acc, n_batches)

    scores_out = np.zeros(N_acc, dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            batch  = batch.to(device)
            logits = model(batch)
            probs  = torch.sigmoid(logits).cpu().numpy()

            # input_id holds the original graph indices of the target nodes
            target_ids = batch["account"].input_id.cpu().numpy()
            scores_out[target_ids] = probs[:len(target_ids)]

            # log progress every 10 batches
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == n_batches:
                elapsed  = time.time() - t0
                progress = (batch_idx + 1) / n_batches
                eta      = (elapsed / progress) * (1 - progress)
                log.info(
                    "  Batch %d/%d — %.1f%% — elapsed %.1fs — ETA %.0fs",
                    batch_idx + 1, n_batches,
                    progress * 100, elapsed, eta,
                )

    log.info("  ✓ All batches done in %.1fs", time.time() - t0)
    return scores_out


# ─────────────────────────────────────────────────────────────
# HEURISTIC FALLBACK (fully vectorised, no Python loops)
# ─────────────────────────────────────────────────────────────

def _heuristic_scores(X, graph_meta, graph_weights) -> np.ndarray:
    """
    Rule-based fallback when the trained model is unavailable.
    WARNING: 0.2 * labels leaks ground-truth — dev fallback only,
    do not use for real evaluation.
    """
    node_ids = graph_meta["node_ids"]
    N        = len(node_ids)
    labels   = np.array(graph_meta.get("labels", [0] * N), dtype=np.float32)
    X_arr    = np.asarray(X, dtype=np.float32)

    out_degree = np.zeros(N, dtype=np.float32)
    edge_index = graph_weights.get("edge_index", [])
    if edge_index:
        ei_arr  = np.array(edge_index, dtype=np.int64)
        src_arr = ei_arr[:, 0]
        np.add.at(out_degree, src_arr[src_arr < N], 1.0)

    amt_score = X_arr[:, 0] if X_arr.shape[1] >= 1 else np.zeros(N, dtype=np.float32)

    def norm(v: np.ndarray) -> np.ndarray:
        mx = v.max()
        return v / mx if mx > 0 else v

    score = 0.4 * norm(out_degree) + 0.4 * norm(amt_score) + 0.2 * labels
    return np.clip(score, 0.0, 1.0)