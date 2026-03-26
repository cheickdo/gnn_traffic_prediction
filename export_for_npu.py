import numpy as np
import torch
from model import TrafficPredictorGNN
from data_loader import load_toronto_traffic_data

def quantize_int8(w):
    maxabs = np.max(np.abs(w))
    scale = 1.0 if maxabs == 0 else maxabs / 127.0
    return np.clip(np.round(w / scale), -128, 127).astype(np.int8)

print("Loading PyTorch Spatio-Temporal Model...")
model = TrafficPredictorGNN(node_features=7, hidden_dim=32)
model.load_state_dict(torch.load("traffic_gnn_weights.pth", map_location="cpu"))

w0 = model.conv1.lin.weight.detach().numpy()
w1 = model.conv2.lin.weight.detach().numpy()
w2 = model.conv3.lin.weight.detach().numpy()
w_head = model.linear.weight.detach().numpy()

print("Quantizing and Padding to 32-wide Vector Lanes...")
w0_q = quantize_int8(np.pad(w0, ((0, 0), (0, 32 - 7)), mode='constant'))
w1_q = quantize_int8(w1)
w2_q = quantize_int8(w2)
w_head_q = quantize_int8(np.pad(w_head, ((0, 31), (0, 0)), mode='constant'))

print("Extracting Graph Topology...")
dataset = load_toronto_traffic_data()
edge_index = dataset.edge_index
edges_src = edge_index[0].numpy()
edges_dest = edge_index[1].numpy()

# 1. Create a standard input state (Free Flowing Traffic)
num_nodes = len(dataset.features[0].edge_index[0]) # Get actual node count
current_state = np.ones((num_nodes, 7), dtype=np.float32)
x_tensor = torch.tensor(current_state)

# 2. Run PyTorch to get the Golden Reference
print("Calculating PyTorch Golden Output...")
with torch.no_grad():
    golden_out = model(x_tensor, edge_index).numpy()

# Pack everything into the bundle
np.savez(
    "toronto_npu_export.npz",
    edges_src=edges_src,
    edges_dest=edges_dest,
    w0_q=w0_q,
    w1_q=w1_q,
    w2_q=w2_q,
    w_head_q=w_head_q,
    x_in=current_state,       
    golden_out=golden_out     
)
print("✅ Exported toronto_npu_export.npz with Golden Reference!")
