# AML-Inference

Graph-based mule account detection pipeline for anti-money-laundering (AML) analysis.
Given a raw transaction CSV, this pipeline builds a heterogeneous graph across accounts,
banks, and entities, scores each account with a trained R-GCN model, and surfaces the
top suspected mule networks with interactive visualisations.


The pipeline runs in five stages:

1. **Schema validation** — checks the input CSV matches the expected transaction schema
2. **Feature engineering** — computes node and edge features for all three node types
3. **Graph construction** — assembles a PyTorch Geometric `HeteroData` object with
   train/val/test splits
4. **R-GCN inference** — scores every account node using a trained Relational GCN
5. **Mule network extraction** — pulls the top-10 highest-risk accounts, expands their
   local neighbourhood, and renders subgraph visualisations

## Setup

```bash
git clone https://github.com/SamarthZaveri/AML-Inference.git
cd AML-Inference
pip install -r requirements.txt
```

<img width="1919" height="910" alt="Screenshot 2026-08-17 012903" src="https://github.com/user-attachments/assets/ce938ee4-e4c8-46ee-bd16-6ecc7e25cbf5" />
<img width="1919" height="907" alt="Screenshot 2026-08-17 012822" src="https://github.com/user-attachments/assets/119c0b55-1548-4fe7-8cd8-a567b4f18ee9" />
<img width="1917" height="904" alt="Screenshot 2026-08-17 012743" src="https://github.com/user-attachments/assets/da87ee06-c77c-4324-a4df-2fbc68d0a65e" />
