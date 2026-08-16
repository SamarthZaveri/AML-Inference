"""
visualisation/visualise.py  v5  — FIXED PARALLEL RENDERING
===========================================================
Fixes:
  • macOS/Windows multiprocessing pickle crash
    ("Can't pickle local object ... _worker")
  • Uses top-level worker function for ProcessPoolExecutor
  • Keeps parallel rendering support
  • Compatible with mule_network.py v2 per_seed output

Features:
  • Per-seed subgraphs built from per_seed dict
  • Parallel PNG rendering
  • Hierarchical money-flow layout
  • Node colour/size encodes mule score and direction
  • Edge width encodes transaction weight
"""

from pathlib import Path
from typing import Any
import json
import logging
import math
import concurrent.futures

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

log = logging.getLogger("pipeline.visualise")

# ─────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────
C_SEED       = "#e63946"
C_UPSTREAM   = "#4895ef"
C_DOWNSTREAM = "#f4a261"
C_NEUTRAL    = "#90be6d"

C_EDGE_FWD   = "#f4a261"
C_EDGE_BWD   = "#4895ef"

BG           = "#12111a"
TEXT_C       = "#e8e8f0"


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _node_color(nd: dict) -> str:
    if nd["is_seed"]:
        return C_SEED

    d = nd.get("direction", "")

    if d == "upstream":
        return C_UPSTREAM

    if d == "downstream":
        return C_DOWNSTREAM

    return C_NEUTRAL


def _node_size(nd: dict) -> int:
    base = 1600 if nd["is_seed"] else 300
    return int(base + nd["mule_score"] * 600)


# ─────────────────────────────────────────────────────────────
# Hierarchical layout
# ─────────────────────────────────────────────────────────────
def _hierarchical_pos(G: nx.DiGraph, seed_id: str) -> dict:

    tiers = {
        "upstream": [],
        "seed": [],
        "downstream": [],
    }

    for n in G.nodes():

        d = G.nodes[n].get("direction", "downstream")

        if n == seed_id:
            tiers["seed"].append(n)

        elif d == "upstream":
            tiers["upstream"].append(n)

        else:
            tiers["downstream"].append(n)

    pos = {}

    y_map = {
        "upstream": 1.0,
        "seed": 0.0,
        "downstream": -1.0,
    }

    for tier_name, nodes in tiers.items():

        y = y_map[tier_name]
        n = len(nodes)

        for i, node in enumerate(sorted(nodes)):

            x = (i - (n - 1) / 2) * (1.2 if n > 1 else 0)

            pos[node] = (x, y)

    missing = [n for n in G.nodes() if n not in pos]

    if missing:
        sp = nx.spring_layout(G, seed=42)

        for n in missing:
            pos[n] = sp.get(n, (0, -2))

    return pos


# ─────────────────────────────────────────────────────────────
# Empty graph figure
# ─────────────────────────────────────────────────────────────
def _empty_fig(seed_id, png_path, json_path):

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG)

    ax.text(
        0.5,
        0.5,
        f"No subgraph data for {seed_id}",
        ha="center",
        va="center",
        color=TEXT_C,
        fontsize=14,
    )

    ax.axis("off")

    fig.savefig(
        png_path,
        dpi=80,
        bbox_inches="tight",
        facecolor=BG,
    )

    plt.close(fig)

    json_path.write_text(
        json.dumps({
            "seed_id": seed_id,
            "nodes": [],
            "edges": [],
        })
    )


# ─────────────────────────────────────────────────────────────
# Draw one graph
# ─────────────────────────────────────────────────────────────
def _draw_one(seed_id: str, seed_data: dict, output_dir: Path):

    nodes = seed_data["nodes"]
    edges = seed_data["edges"]
    stats = seed_data["stats"]

    png_path  = output_dir / f"subgraph_{seed_id}.png"
    json_path = output_dir / f"subgraph_{seed_id}.json"

    # Build graph
    G = nx.DiGraph()

    node_by_id = {str(n["id"]): n for n in nodes}

    for nd in nodes:

        G.add_node(
            str(nd["id"]),
            mule_score=nd["mule_score"],
            is_seed=nd["is_seed"],
            hop=nd.get("hop", 0),
            direction=nd.get("direction", "downstream"),
            kpi=nd.get("kpi", {}),
        )

    max_w = max((e["weight"] for e in edges), default=1.0) or 1.0

    for e in edges:

        s = str(e["src"])
        d = str(e["dst"])

        if G.has_node(s) and G.has_node(d):

            G.add_edge(
                s,
                d,
                weight=float(e["weight"]),
                txn_count=int(e["txn_count"]),
            )

    seed_str = str(seed_id)

    # Layout
    if len(G.nodes) == 0:
        _empty_fig(seed_id, png_path, json_path)
        return

    if len(G.nodes) <= 30:

        pos = _hierarchical_pos(G, seed_str)

    else:

        try:
            pos = nx.kamada_kawai_layout(G, weight="weight")

        except Exception:
            pos = nx.spring_layout(
                G,
                seed=42,
                k=1.8 / math.sqrt(len(G.nodes)),
            )

    # Visual attrs
    node_list = list(G.nodes())

    node_colors = [
        _node_color({
            "is_seed": n == seed_str,
            "mule_score": G.nodes[n].get("mule_score", 0),
            "direction": G.nodes[n].get("direction", "downstream"),
        })
        for n in node_list
    ]

    node_sizes = [
        _node_size({
            "is_seed": n == seed_str,
            "mule_score": G.nodes[n].get("mule_score", 0),
        })
        for n in node_list
    ]

    edge_list = list(G.edges(data=True))

    edge_widths = [
        max(0.5, (d.get("weight", 1.0) / max_w) * 6)
        for _, _, d in edge_list
    ]

    edge_colors = []

    for u, v, _ in edge_list:

        u_dir = G.nodes[u].get("direction", "downstream")

        edge_colors.append(
            C_EDGE_BWD if u_dir == "upstream" else C_EDGE_FWD
        )

    # Labels
    labels = {}

    for n in node_list:

        short = str(n)[-10:] if len(str(n)) > 10 else str(n)

        prefix = "★ " if n == seed_str else ""

        score = G.nodes[n].get("mule_score", 0)

        labels[n] = f"{prefix}{short}\n{score:.2f}"

    # Figure
    n_nodes = len(G.nodes)

    figw = max(10, min(22, 8 + n_nodes * 0.18))
    figh = figw * 0.65

    fig, ax = plt.subplots(
        figsize=(figw, figh),
        facecolor=BG,
    )

    ax.set_facecolor(BG)

    ax.set_title(
        (
            f"Money Trail — Seed: {seed_id}   "
            f"[↑{stats.get('upstream',0)} upstream  "
            f"↓{stats.get('downstream',0)} downstream]  "
            f"Score: {stats.get('seed_score',0):.3f}"
        ),
        color=TEXT_C,
        fontsize=11,
        pad=10,
        fontweight="bold",
    )

    # Draw edges
    if edge_list:

        for (u, v, d), ew, ec in zip(
            edge_list,
            edge_widths,
            edge_colors,
        ):

            nx.draw_networkx_edges(
                G,
                pos,
                edgelist=[(u, v)],
                width=ew,
                edge_color=ec,
                alpha=0.65,
                arrowsize=14,
                arrowstyle="-|>",
                connectionstyle="arc3,rad=0.06",
                ax=ax,
            )

    # Draw nodes
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=node_list,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=1.5,
        edgecolors=[
            C_SEED if n == seed_str else "#2a2a3a"
            for n in node_list
        ],
        ax=ax,
    )

    # Draw labels
    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=6.5,
        font_color=TEXT_C,
        font_family="monospace",
        ax=ax,
    )

    # Tier labels
    ax.text(
        0.01,
        0.98,
        "↑ UPSTREAM (Funding sources)",
        transform=ax.transAxes,
        color=C_UPSTREAM,
        fontsize=8,
        va="top",
        fontweight="bold",
    )

    ax.text(
        0.01,
        0.50,
        "⬤ SEED MULE ACCOUNT",
        transform=ax.transAxes,
        color=C_SEED,
        fontsize=8,
        va="center",
        fontweight="bold",
    )

    ax.text(
        0.01,
        0.02,
        "↓ DOWNSTREAM (Money recipients)",
        transform=ax.transAxes,
        color=C_DOWNSTREAM,
        fontsize=8,
        va="bottom",
        fontweight="bold",
    )

    # KPI box
    kpi = node_by_id.get(seed_str, {}).get("kpi", {})

    if kpi:

        lines = [f"Account: {seed_id}"]

        for k in [
            "total_sent",
            "total_received",
            "txn_count_sent",
            "unique_receivers",
            "unique_senders",
            "currencies_used",
        ]:

            if k in kpi:

                v = kpi[k]

                if isinstance(v, (int, float)):
                    lines.append(
                        f"{k.replace('_',' ').title()}: {v:,}"
                    )
                else:
                    lines.append(f"{k}: {v}")

        kpi_text = "\n".join(lines[:7])

        ax.text(
            0.99,
            0.98,
            kpi_text,
            transform=ax.transAxes,
            color=TEXT_C,
            fontsize=6.5,
            va="top",
            ha="right",
            fontfamily="monospace",
            bbox=dict(
                boxstyle="round,pad=0.4",
                fc="#1e1e2e",
                ec=C_SEED,
                alpha=0.85,
            ),
        )

    # Legend
    legend = [
        mpatches.Patch(color=C_SEED, label="Seed mule"),
        mpatches.Patch(color=C_UPSTREAM, label="Upstream funder"),
        mpatches.Patch(color=C_DOWNSTREAM, label="Downstream recipient"),
        mpatches.Patch(color=C_NEUTRAL, label="Low-risk node"),
        mpatches.Patch(color=C_EDGE_FWD, label="Money flow →"),
        mpatches.Patch(color=C_EDGE_BWD, label="Funding flow ←"),
    ]

    ax.legend(
        handles=legend,
        loc="lower right",
        framealpha=0.6,
        facecolor="#1e1e2e",
        edgecolor=C_SEED,
        labelcolor=TEXT_C,
        fontsize=7.5,
    )

    ax.axis("off")

    plt.tight_layout(pad=1.0)

    fig.savefig(
        png_path,
        dpi=100,
        bbox_inches="tight",
        facecolor=BG,
    )

    plt.close(fig)

    log.info("Subgraph PNG → %s", png_path)

    # JSON export
    subgraph_json = {
        "seed_id": seed_id,
        "nodes": [
            {
                "id": str(n),
                "mule_score": float(
                    G.nodes[n].get("mule_score", 0)
                ),
                "is_seed": G.nodes[n].get("is_seed", False),
                "hop": int(G.nodes[n].get("hop", 0)),
                "direction": G.nodes[n].get(
                    "direction",
                    "downstream",
                ),
                "kpi": G.nodes[n].get("kpi", {}),
            }
            for n in G.nodes()
        ],
        "edges": [
            {
                "src": u,
                "dst": v,
                "weight": float(d.get("weight", 0)),
                "txn_count": int(d.get("txn_count", 0)),
            }
            for u, v, d in G.edges(data=True)
        ],
        "stats": stats,
    }

    json_path.write_text(
        json.dumps(subgraph_json, indent=2),
        encoding="utf-8",
    )

    return {
        "png_path": str(png_path),
        "json_path": str(json_path),
        "stats": stats,
    }


# ─────────────────────────────────────────────────────────────
# TOP-LEVEL WORKER
# IMPORTANT:
# Must be top-level to be pickleable on macOS/Windows
# ─────────────────────────────────────────────────────────────
def _render_worker(args):

    seed_id, seed_data, output_dir = args

    if not seed_data:
        log.warning("No per_seed data for %s", seed_id)
        return seed_id, None

    try:

        r = _draw_one(
            str(seed_id),
            seed_data,
            Path(output_dir),
        )

        return seed_id, r

    except Exception as exc:

        log.error(
            "Error rendering seed %s: %s",
            seed_id,
            exc,
        )

        return seed_id, None


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────
def render_visualisation(
    network_data: dict,
    output_dir: Path,
) -> dict:

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seeds = network_data["seeds"]

    per_seed = network_data.get("per_seed", {})

    results = {}

    max_workers = min(4, len(seeds))

    worker_args = [
        (
            sid,
            per_seed.get(str(sid)),
            str(output_dir),
        )
        for sid in seeds
    ]

    # Parallel rendering
    if max_workers > 1:

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as pool:

            futures = {
                pool.submit(_render_worker, args): args[0]
                for args in worker_args
            }

            for fut in concurrent.futures.as_completed(
                futures
            ):

                sid, r = fut.result()

                if r:
                    results[sid] = r

    # Sequential fallback
    else:

        for args in worker_args:

            sid, r = _render_worker(args)

            if r:
                results[sid] = r

    # Index JSON
    index = {
        "seeds": [str(s) for s in seeds],
        "subgraphs": {
            str(sid): {
                "png": f"subgraph_{sid}.png",
                "json": f"subgraph_{sid}.json",
            }
            for sid in seeds
        },
    }

    (output_dir / "index.json").write_text(
        json.dumps(index, indent=2)
    )

    log.info(
        "render_visualisation → %d subgraph PNGs + JSONs written to %s",
        len(seeds),
        output_dir,
    )

    return results