"""
export_for_npu.py — Export trained PyTorch GNN to NPU hardware format.

The Stratix 10 NX RTL uses INT8 weights in the MRF (matrix register file)
multiplied with BFloat16 activations in the VRF.  The compiler enforces
weight_data_type = np.int8.

The INT8 values are used AS-IS by the hardware — there is no scale factor.
INT8 value 4 means the hardware multiplies by 4.0.

QUANTIZATION STRATEGY:
  Per-row quantization with max_int=8 maps each weight matrix row's
  largest absolute value to ±8.  This:
    - Matches the working 26_gcn5_100_latency.py pattern (weights [0,4])
    - Keeps dot products bounded (32 lanes × max 8 = 256 per layer)
    - Prevents overflow after 3 GCN layers with normalization
    - Preserves relative weight structure within each row
"""

import numpy as np
import torch
from model import TrafficPredictorGNN
from data_loader import load_toronto_traffic_data


MAX_INT = 8  # Target max int8 value — keep small to prevent overflow


def quantize_per_row(w: np.ndarray, max_int: int = MAX_INT) -> np.ndarray:
    """Per-row symmetric INT8 quantization.

    Each row's largest absolute value maps to ±max_int.
    This preserves relative structure within each row while keeping
    values in a hardware-safe range.
    """
    w_q = np.zeros_like(w, dtype=np.int8)
    for r in range(w.shape[0]):
        row_max = np.max(np.abs(w[r]))
        if row_max > 0:
            scale = row_max / max_int
            w_q[r] = np.clip(np.round(w[r] / scale), -128, 127).astype(np.int8)
    return w_q


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
    # 2. Pad and quantize weights to INT8 (per-row, max_int={MAX_INT})
    # ------------------------------------------------------------------
    print(f"\n[2/5] Quantizing weights (per-row, max_int={MAX_INT})...")

    # Layer 1: (32, 7) → pad to (32, 32)
    w0_padded = np.pad(w0, ((0, 0), (0, 32 - 7)), mode="constant")
    w0_q = quantize_per_row(w0_padded)

    # Layers 2-3: already (32, 32)
    w1_q = quantize_per_row(w1)
    w2_q = quantize_per_row(w2)

    # Head: (1, 32) → pad to (32, 32)
    w_head_padded = np.pad(w_head, ((0, 31), (0, 0)), mode="constant")
    w_head_q = quantize_per_row(w_head_padded)

    for name, wq in [("w0_q", w0_q), ("w1_q", w1_q), ("w2_q", w2_q), ("w_head_q", w_head_q)]:
        nz = np.count_nonzero(wq)
        print(f"  {name}: {wq.shape}  range=[{wq.min()}, {wq.max()}]  "
              f"nonzero={nz}/{wq.size} ({100*nz/wq.size:.0f}%)")

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

    self_loop_nodes = np.arange(num_nodes, dtype=np.int64)
    edges_src = np.concatenate([edges_src_orig, self_loop_nodes])
    edges_dest = np.concatenate([edges_dest_orig, self_loop_nodes])
    num_edges = len(edges_src)

    print(f"  + {num_nodes} self-loops → {num_edges} total edges")

    # ------------------------------------------------------------------
    # 4. Edge scalars
    # ------------------------------------------------------------------
    edge_scalar = np.ones(num_edges, dtype=np.float32)

    # ------------------------------------------------------------------
    # 5. Golden reference output
    # ------------------------------------------------------------------
    print("\n[4/5] Computing PyTorch golden reference output...")
    current_state = np.ones((num_nodes, 7), dtype=np.float32)
    x_tensor = torch.tensor(current_state, dtype=torch.float32)
    edge_index_torch = torch.tensor(np.asarray(edge_index), dtype=torch.long)
    with torch.no_grad():
        golden_out = model(x_tensor, edge_index_torch).numpy()

    print(f"  Golden output: range=[{golden_out.min():.4f}, {golden_out.max():.4f}]  "
          f"mean={golden_out.mean():.4f}  std={golden_out.std():.4f}")

    # ------------------------------------------------------------------
    # 6. Save
    # ------------------------------------------------------------------
    print("\n[5/5] Saving toronto_npu_export.npz...")
    np.savez(
        "toronto_npu_export.npz",
        w0_q=w0_q,
        w1_q=w1_q,
        w2_q=w2_q,
        w_head_q=w_head_q,
        edges_src=edges_src,
        edges_dest=edges_dest,
        edge_scalar=edge_scalar,
        x_in=current_state,
        golden_out=golden_out,
    )

    print(f"\n{'=' * 60}")
    print(f"  Export complete: toronto_npu_export.npz")
    print(f"  Nodes: {num_nodes}  Edges: {num_edges}")
    print(f"  Weights: INT8 per-row quantized, max_int={MAX_INT}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
