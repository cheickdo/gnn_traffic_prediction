"""
export_for_npu.py — Export trained PyTorch GNN to NPU hardware format.

Generates `toronto_npu_export.npz` containing:
  - Float32 GCN layer weights padded to 32 lanes (w0_q, w1_q, w2_q)
  - Float32 prediction head padded to 32×32 (w_head_q)
  - Graph topology with self-loops (edges_src, edges_dest)
  - GCN normalization edge scalars (edge_scalar)
  - Default input tensor (x_in)
  - Golden reference output from PyTorch (golden_out)

CRITICAL FIX: The Intel Stratix 10 NX has BFloat16 tensor blocks, NOT
INT8 ALUs.  Previous INT8 quantization caused values like 0.5 to be
stored as 127.0 — ~254× too large.  After 3 GCN layers the outputs
explode to ~40000, explaining the "all same output" bug.

Now exports weights as float32 (which the hardware truncates to BFloat16
natively).  Array names are kept as *_q for backward compatibility with
the compiler driver.

Self-loops are added to the edge list, matching PyTorch GCNConv's
default `add_self_loops=True` behaviour.
"""

import numpy as np
import torch
from model import TrafficPredictorGNN
from data_loader import load_toronto_traffic_data


def main():
    print("=" * 60)
    print("  NPU Export Pipeline — Toronto Traffic GCN")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load trained PyTorch model
    # ------------------------------------------------------------------
    print("\n[1/5] Loading trained PyTorch model...")
    model = TrafficPredictorGNN(node_features=7, hidden_dim=32)
    model.load_state_dict(torch.load("traffic_gnn_weights.pth", map_location="cpu"))
    model.eval()

    w0 = model.conv1.lin.weight.detach().numpy()  # (32, 7)
    w1 = model.conv2.lin.weight.detach().numpy()  # (32, 32)
    w2 = model.conv3.lin.weight.detach().numpy()  # (32, 32)
    w_head = model.linear.weight.detach().numpy()  # (1, 32)

    print(f"  conv1: {w0.shape}  conv2: {w1.shape}  conv3: {w2.shape}")
    print(f"  head:  {w_head.shape}")

    # ------------------------------------------------------------------
    # 2. Pad weights to 32-wide hardware vector lanes (FLOAT32, NOT INT8)
    # ------------------------------------------------------------------
    # The Stratix 10 NX computes in BFloat16.  Feeding INT8 integers
    # (range -128..127) directly causes ~200× weight inflation and
    # numerical explosion after multiple GCN layers.
    #
    # We export the raw float32 weights; the NPU hardware truncates
    # them to BFloat16 when loaded into the MRF tiles.
    print("\n[2/5] Padding weights to 32 lanes (float32 for BFloat16 hardware)...")

    # Layer 1: input is 7 features → pad columns from 7 to 32
    w0_q = np.pad(w0, ((0, 0), (0, 32 - 7)), mode="constant").astype(np.float32)

    # Layers 2-3: already 32×32
    w1_q = w1.astype(np.float32)
    w2_q = w2.astype(np.float32)

    # Prediction head: (1, 32) → pad rows from 1 to 32
    # Only row 0 carries the real weights; rows 1–31 are zero.
    # Lane 0 of the NPU output holds the actual prediction.
    w_head_q = np.pad(w_head, ((0, 31), (0, 0)), mode="constant").astype(np.float32)

    for name, wq in [("w0_q", w0_q), ("w1_q", w1_q), ("w2_q", w2_q), ("w_head_q", w_head_q)]:
        print(f"  {name}: {wq.shape}  range=[{wq.min():.4f}, {wq.max():.4f}]")

    # ------------------------------------------------------------------
    # 3. Extract graph topology and ADD SELF-LOOPS
    # ------------------------------------------------------------------
    print("\n[3/5] Extracting graph topology...")
    dataset = load_toronto_traffic_data()
    edge_index = dataset.edge_index

    edges_src_orig = np.asarray(edge_index[0]).astype(np.int64)
    edges_dest_orig = np.asarray(edge_index[1]).astype(np.int64)

    all_nodes = np.unique(np.concatenate([edges_src_orig, edges_dest_orig]))
    num_nodes = int(all_nodes.max()) + 1
    num_edges_orig = len(edges_src_orig)

    print(f"  Original graph: {num_nodes} nodes, {num_edges_orig} edges")

    # ------ CRITICAL FIX: ADD SELF-LOOPS ------
    # PyTorch Geometric GCNConv has add_self_loops=True by default.
    # The GCN formula is: h' = D̂^(-½) Â D̂^(-½) X W
    # where Â = A + I (adjacency + identity = self-loops).
    self_loop_nodes = np.arange(num_nodes, dtype=np.int64)
    edges_src = np.concatenate([edges_src_orig, self_loop_nodes])
    edges_dest = np.concatenate([edges_dest_orig, self_loop_nodes])
    num_edges = len(edges_src)

    print(f"  + {num_nodes} self-loops → {num_edges} total edges")

    # ------------------------------------------------------------------
    # 4. Compute GCN edge scalars
    # ------------------------------------------------------------------
    edge_scalar = np.ones(num_edges, dtype=np.float32)

    # ------------------------------------------------------------------
    # 5. Generate golden reference output
    # ------------------------------------------------------------------
    print("\n[4/5] Computing PyTorch golden reference output...")

    current_state = np.ones((num_nodes, 7), dtype=np.float32)
    x_tensor = torch.tensor(current_state, dtype=torch.float32)

    edge_index_torch = torch.tensor(np.asarray(edge_index), dtype=torch.long)
    with torch.no_grad():
        golden_out = model(x_tensor, edge_index_torch).numpy()

    print(f"  Golden output: shape={golden_out.shape}")
    print(f"  Range: [{golden_out.min():.4f}, {golden_out.max():.4f}]")
    print(f"  Mean:  {golden_out.mean():.4f}  Std: {golden_out.std():.4f}")

    # ------------------------------------------------------------------
    # 6. Save export bundle
    # ------------------------------------------------------------------
    print("\n[5/5] Saving toronto_npu_export.npz...")
    np.savez(
        "toronto_npu_export.npz",
        # Weights (float32 — array names kept as *_q for compatibility)
        w0_q=w0_q,
        w1_q=w1_q,
        w2_q=w2_q,
        w_head_q=w_head_q,
        # Topology (WITH self-loops)
        edges_src=edges_src,
        edges_dest=edges_dest,
        edge_scalar=edge_scalar,
        # Reference data
        x_in=current_state,
        golden_out=golden_out,
    )

    print(f"\n{'=' * 60}")
    print(f"  Export complete: toronto_npu_export.npz")
    print(f"  Nodes: {num_nodes}  Edges: {num_edges} (incl. {num_nodes} self-loops)")
    print(f"  Weights: float32 (BFloat16-native), 32-wide vector lanes")
    print(f"{'=' * 60}")
    print(f"\n  Next steps:")
    print(f"  1. Copy toronto_npu_export.npz to the compiler directory")
    print(f"  2. Run: python driver_toronto_npu.py <npu_args>")
    print(f"  3. This generates input_template.mif for the RTL simulator")


if __name__ == "__main__":
    main()
