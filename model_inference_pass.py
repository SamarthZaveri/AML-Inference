import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import FastRGCNConv


# MUST match training
EDGE_TYPE_ORDER = [
    ("account", "to", "account"),
    ("account", "sends_to", "bank"),
    ("entity", "owns", "account"),
    ("account", "rev_to", "account"),
    ("bank", "rev_sends_to", "account"),
    ("account", "rev_owns", "entity"),
]

NODE_TYPES = ["account", "bank", "entity"]


class RGCNInferenceModel(nn.Module):
    def __init__(self, in_channels, hidden=128, num_layers=2, num_bases=30, dropout=0.3):
        super().__init__()

        self.node_types = NODE_TYPES
        self.type_to_id = {nt: i for i, nt in enumerate(NODE_TYPES)}
        self.hidden = hidden
        self.dropout = dropout

        # input projections
        self.input_proj = nn.ModuleDict({
            nt: nn.Sequential(
                nn.Linear(in_channels[nt], hidden),
                nn.GELU()
            )
            for nt in NODE_TYPES
        })

        # RGCN layers
        self.convs = nn.ModuleList([
            FastRGCNConv(hidden, hidden, num_relations=6, num_bases=num_bases)
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden) for _ in range(num_layers)
        ])

        # head
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def _to_homo(self, graph):
        device = next(self.parameters()).device

        offsets = {}
        cursor = 0

        for nt in NODE_TYPES:
            offsets[nt] = cursor
            cursor += graph[nt].num_nodes

        N = cursor

        max_feat = max(graph[nt].x.shape[1] for nt in NODE_TYPES)

        x = torch.zeros(N, max_feat, device=device)
        node_type = torch.zeros(N, dtype=torch.long, device=device)

        for nt in NODE_TYPES:
            off = offsets[nt]
            n = graph[nt].num_nodes
            f = graph[nt].x.shape[1]

            x[off:off+n, :f] = graph[nt].x
            node_type[off:off+n] = self.type_to_id[nt]

        edge_index_list = []
        edge_type_list = []

        for rel_id, et in enumerate(EDGE_TYPE_ORDER):
            if et not in graph.edge_types:
                continue

            ei = graph[et].edge_index

            src, _, dst = et

            shifted = torch.stack([
                ei[0] + offsets[src],
                ei[1] + offsets[dst]
            ])

            edge_index_list.append(shifted)
            edge_type_list.append(torch.full((ei.shape[1],), rel_id, device=device))

        edge_index = torch.cat(edge_index_list, dim=1)
        edge_type = torch.cat(edge_type_list)

        return x, edge_index, edge_type, offsets

    def forward(self, graph):
        x, edge_index, edge_type, offsets = self._to_homo(graph)

        h = torch.zeros(x.shape[0], self.hidden, device=x.device)

        for nt in NODE_TYPES:
            mask = (x[:, 0] == x[:, 0])  # dummy init

        # proper masking
        for nt in NODE_TYPES:
            tid = self.type_to_id[nt]
            start = offsets[nt]
            end = start + graph[nt].num_nodes

            h[start:end] = self.input_proj[nt](x[start:end, :graph[nt].x.shape[1]])

        for i, conv in enumerate(self.convs):
            h_new = conv(h, edge_index, edge_type)
            h = self.norms[i](h + h_new)
            h = F.dropout(h, p=self.dropout, training=False)

        acc_start = offsets["account"]
        acc_end = acc_start + graph["account"].num_nodes

        acc_h = h[acc_start:acc_end]

        logits = self.head(acc_h).squeeze(-1)

        return logits