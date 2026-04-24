"""
visualisation/visualise.py
===========================
Renders the mule sub-network as SEPARATE per-seed subgraphs using NetworkX
+ Matplotlib.  One subgraph plot per top-N seed mule node.

Each subgraph shows:
  • The seed node (★) at the centre — large red node.
  • All BFS-expanded neighbours coloured by mule_score and labelled by hop.
  • Edge thickness ∝ transaction weight.
  • A KPI annotation box for the seed node.

Output
------
  <output_dir>/subgraph_<seed_id>.png   — one PNG per seed
  <output_dir>/subgraph_<seed_id>.json  — subgraph data (nodes + edges) for
                                          frontend consumption

The function also returns a dict:
    {
      seed_id: {
        "figure": matplotlib.figure.Figure,   # embed directly in frontend
        "nx_graph": networkx.DiGraph,         # raw graph object
        "json_path": Path,                    # path to JSON file
        "png_path":  Path,                    # path to PNG file
      },
      ...
    }

Frontend integration
--------------------
Each Figure can be converted to a base64 PNG for embedding in React / HTML:

    import base64, io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode()
    # <img src="data:image/png;base64,{b64}" />

Or serve the PNG files directly as static assets.
"""

from pathlib import Path
from typing import Any
import json
import logging
import math

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for servers
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

log = logging.getLogger("pipeline.visualise")

# ── Colour palette ─────────────────────────────────────────────────────────────
COLOUR_SEED  = "#e63946"   # confirmed top-N seed   — red
COLOUR_HIGH  = "#f4a261"   # mule_score ≥ 0.6       — orange
COLOUR_MED   = "#e9c46a"   # mule_score ≥ 0.3       — yellow
COLOUR_LOW   = "#90be6d"   # mule_score < 0.3        — green
COLOUR_EDGE  = "#adb5bd"
BG_COLOUR    = "#1a1a2e"
TEXT_COLOUR  = "#ececec"


def _node_colour(node: dict) -> str:
    if node["is_seed"]:
        return COLOUR_SEED
    s = node["mule_score"]
    if s >= 0.6:
        return COLOUR_HIGH
    if s >= 0.3:
        return COLOUR_MED
    return COLOUR_LOW


def _node_size(node: dict, base_scale: float = 1.0) -> int:
    base = 800 if node["is_seed"] else 250
    return int((base + node["mule_score"] * 400) * base_scale)


def _build_subgraph(seed_id, all_nodes: list, all_edges: list) -> nx.DiGraph:
    """
    Build a NetworkX DiGraph containing the seed node and all nodes/edges
    that are reachable from it within the pre-expanded network_data.
    Only include edges where both endpoints exist in the subgraph node set.
    """
    # Collect all node IDs connected to this seed via any edge path
    seed_str   = str(seed_id)
    node_by_id = {str(n["id"]): n for n in all_nodes}

    # BFS over the edge list to find the connected component of this seed
    adj: dict[str, set] = {}
    for e in all_edges:
        s, d = str(e["src"]), str(e["dst"])
        adj.setdefault(s, set()).add(d)
        adj.setdefault(d, set()).add(s)   # undirected component discovery

    if seed_str not in adj and seed_str not in node_by_id:
        return nx.DiGraph()

    visited: set[str] = set()
    queue = [seed_str]
    while queue:
        cur = queue.pop()
        if cur in visited:
            continue
        visited.add(cur)
        for nb in adj.get(cur, []):
            if nb not in visited:
                queue.append(nb)

    G = nx.DiGraph()
    for nid in visited:
        if nid in node_by_id:
            n = node_by_id[nid]
            G.add_node(nid,
                       mule_score=n["mule_score"],
                       is_seed=n["is_seed"],
                       hop=n["hop"],
                       kpi=n["kpi"])

    for e in all_edges:
        s, d = str(e["src"]), str(e["dst"])
        if s in visited and d in visited and G.has_node(s) and G.has_node(d):
            G.add_edge(s, d,
                       weight=float(e["weight"]),
                       txn_count=int(e["txn_count"]))
    return G


def _kpi_text(seed_id, kpi: dict) -> str:
    """Format KPI dict into a multiline annotation string."""
    lines = [f"Account: {seed_id}"]
    display_keys = [
        "total_sent", "txn_count_sent", "avg_sent",
        "total_received", "txn_count_received",
        "unique_receivers", "unique_senders",
        "laundering_flag", "currencies_used",
    ]
    for k in display_keys:
        if k in kpi:
            label = k.replace("_", " ").title()
            lines.append(f"{label}: {kpi[k]}")
    return "\n".join(lines)


def _draw_subgraph(
    seed_id,
    G: nx.DiGraph,
    output_path: Path,
) -> plt.Figure:
    """
    Draw a single seed's subgraph and save to output_path.
    Returns the matplotlib Figure so callers can embed it directly.
    """
    if len(G.nodes) == 0:
        log.warning("Subgraph for seed %s is empty — skipping.", seed_id)
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=BG_COLOUR)
        ax.text(0.5, 0.5, f"No subgraph data for\n{seed_id}",
                ha="center", va="center", color=TEXT_COLOUR, fontsize=14)
        ax.axis("off")
        fig.savefig(output_path, dpi=120, bbox_inches="tight",
                    facecolor=BG_COLOUR)
        plt.close(fig)
        return fig

    seed_str = str(seed_id)

    # ── Layout ───────────────────────────────────────────────
    # Put seed at origin; use spring layout seeded for reproducibility
    if len(G.nodes) > 1:
        pos = nx.spring_layout(G, seed=42, k=2.0 / math.sqrt(len(G.nodes)))
    else:
        pos = {seed_str: (0.0, 0.0)}

    # Force seed to centre
    if seed_str in pos:
        cx, cy = pos[seed_str]
        pos = {n: (x - cx, y - cy) for n, (x, y) in pos.items()}

    # ── Visual attributes ────────────────────────────────────
    node_list   = list(G.nodes())
    node_data   = [G.nodes[n] for n in node_list]
    node_colors = [COLOUR_SEED if n == seed_str else
                   _node_colour({"is_seed": False,
                                  "mule_score": G.nodes[n].get("mule_score", 0)})
                   for n in node_list]
    node_sizes  = [_node_size({"is_seed": n == seed_str,
                                 "mule_score": G.nodes[n].get("mule_score", 0)},
                                base_scale=1.2)
                   for n in node_list]

    edges       = list(G.edges(data=True))
    max_w       = max((d.get("weight", 1.0) for _, _, d in edges), default=1.0) or 1.0
    edge_widths = [max(0.5, (d.get("weight", 1.0) / max_w) * 5) for _, _, d in edges]
    edge_list   = [(u, v) for u, v, _ in edges]

    # Node labels: seed gets ★, others get truncated ID
    labels = {}
    for n in node_list:
        if n == seed_str:
            labels[n] = f"★ {n}"
        else:
            labels[n] = str(n)[:10] + ("…" if len(str(n)) > 10 else "")

    # ── Figure ───────────────────────────────────────────────
    figsize_base = max(8, min(20, 6 + len(G.nodes) * 0.15))
    fig, ax = plt.subplots(figsize=(figsize_base, figsize_base * 0.75),
                           facecolor=BG_COLOUR)
    ax.set_facecolor(BG_COLOUR)
    ax.set_title(
        f"Mule Subgraph — Seed: {seed_id}   "
        f"({len(G.nodes)} nodes, {len(G.edges)} edges)",
        color=TEXT_COLOUR, fontsize=13, pad=12,
    )

    # Draw edges first (underneath nodes)
    nx.draw_networkx_edges(
        G, pos, edgelist=edge_list, width=edge_widths,
        edge_color=COLOUR_EDGE, alpha=0.7,
        arrowsize=12, arrowstyle="-|>",
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=node_list,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=2,
        edgecolors=[COLOUR_SEED if n == seed_str else "#555" for n in node_list],
        ax=ax,
    )

    # Draw labels
    nx.draw_networkx_labels(
        G, pos, labels=labels,
        font_size=7, font_color=TEXT_COLOUR,
        ax=ax,
    )

    # ── KPI annotation for seed node ─────────────────────────
    seed_kpi  = G.nodes[seed_str].get("kpi", {}) if seed_str in G.nodes else {}
    kpi_text  = _kpi_text(seed_id, seed_kpi)
    ax.annotate(
        kpi_text,
        xy=pos.get(seed_str, (0, 0)),
        xytext=(0.02, 0.98),
        textcoords="axes fraction",
        fontsize=7.5,
        color=TEXT_COLOUR,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", fc="#16213e", ec=COLOUR_SEED, alpha=0.85),
        arrowprops=dict(arrowstyle="->", color=COLOUR_SEED, lw=1.2),
    )

    # ── Legend ────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=COLOUR_SEED,  label="Seed mule"),
        mpatches.Patch(color=COLOUR_HIGH,  label="High risk (≥0.6)"),
        mpatches.Patch(color=COLOUR_MED,   label="Medium (≥0.3)"),
        mpatches.Patch(color=COLOUR_LOW,   label="Low (<0.3)"),
    ]
    ax.legend(
        handles=legend_patches,
        loc="lower right", framealpha=0.6,
        facecolor="#16213e", edgecolor=COLOUR_SEED,
        labelcolor=TEXT_COLOUR, fontsize=8,
    )

    ax.axis("off")
    plt.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight", facecolor=BG_COLOUR)
    log.info("  Subgraph PNG → %s", output_path)
    return fig


def render_visualisation(
    network_data: dict,
    output_dir: Path,
) -> dict[Any, dict]:
    """
    Render one NetworkX subgraph plot per seed node.

    Parameters
    ----------
    network_data : dict returned by graph.mule_network.extract_mule_network.
    output_dir   : Directory to write PNG and JSON files.

    Returns
    -------
    dict keyed by seed_id:
        {
          "figure":   matplotlib.figure.Figure,
          "nx_graph": networkx.DiGraph,
          "json_path": Path,
          "png_path":  Path,
        }
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seeds      = network_data["seeds"]
    all_nodes  = network_data["nodes"]
    all_edges  = network_data["edges"]

    results: dict = {}

    for seed_id in seeds:
        log.info("Building subgraph for seed: %s", seed_id)

        G = _build_subgraph(seed_id, all_nodes, all_edges)

        png_path  = output_dir / f"subgraph_{seed_id}.png"
        json_path = output_dir / f"subgraph_{seed_id}.json"

        fig = _draw_subgraph(seed_id, G, png_path)

        # ── JSON export (for frontend / API consumption) ──────────────────
        subgraph_json = {
            "seed_id": seed_id,
            "nodes": [
                {
                    "id":         n,
                    "mule_score": float(G.nodes[n].get("mule_score", 0)),
                    "is_seed":    G.nodes[n].get("is_seed", False),
                    "hop":        int(G.nodes[n].get("hop", -1)),
                    "kpi":        G.nodes[n].get("kpi", {}),
                }
                for n in G.nodes()
            ],
            "edges": [
                {
                    "src":       u,
                    "dst":       v,
                    "weight":    float(d.get("weight", 0)),
                    "txn_count": int(d.get("txn_count", 0)),
                }
                for u, v, d in G.edges(data=True)
            ],
            "stats": {
                "num_nodes": len(G.nodes),
                "num_edges": len(G.edges),
                "seed_score": float(
                    next((n["mule_score"] for n in all_nodes
                          if str(n["id"]) == str(seed_id)), 0.0)
                ),
            },
        }
        json_path.write_text(json.dumps(subgraph_json, indent=2), encoding="utf-8")

        results[seed_id] = {
            "figure":   fig,
            "nx_graph": G,
            "json_path": json_path,
            "png_path":  png_path,
        }

        plt.close(fig)

    # ── Summary index JSON ────────────────────────────────────────────────
    index = {
        "seeds": [str(s) for s in seeds],
        "subgraphs": {
            str(sid): {
                "png":  str(results[sid]["png_path"].name),
                "json": str(results[sid]["json_path"].name),
                "num_nodes": len(results[sid]["nx_graph"].nodes),
                "num_edges": len(results[sid]["nx_graph"].edges),
            }
            for sid in seeds
        },
    }
    (output_dir / "index.json").write_text(
        json.dumps(index, indent=2), encoding="utf-8"
    )
    log.info(
        "render_visualisation → %d subgraph PNGs + JSONs written to %s",
        len(seeds), output_dir,
    )
    return results