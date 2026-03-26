import asyncio
import os
import struct
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import networkx as nx

from model import TrafficPredictorGNN
from data_loader import load_toronto_traffic_data

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model, edge_index, node_coords, kdtree, nx_graph, osm_graph = None, None, None, None, None, None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_SPEED_KMH = 40.0 
BASE_SPEED_MS = DEFAULT_SPEED_KMH * 1000.0 / 3600.0
OSM_BBOX = (-79.395, 43.643, -79.370, 43.658)

import struct

def hex_to_bfloat16_float(hex_str):
    """Converts a 4-character hex string (BFloat16) back to a standard Float32."""
    try:
        padded_hex = hex_str.strip() + "0000"
        return struct.unpack('>f', struct.pack('>I', int(padded_hex, 16)))[0]
    except Exception:
        return 0.0

def parse_raw_hex_output(filepath, num_nodes):
    """
    Parses the raw hex dump from the C++ simulator.
    Accounts for the 480-feature hardware padding (stride of 12).
    """
    predictions = []
    if not os.path.exists(filepath):
        print(f"ERROR: Simulator output not found at {filepath}")
        return np.ones((num_nodes, 1)) * 0.5 

    with open(filepath, 'r') as f:
        lines = f.readlines()
        
    # We stride by 12 because the hardware outputs 12 vectors per node.
    # We only want the 1st vector of each 12-vector block.
    stride = 12
    for i in range(0, min(len(lines), num_nodes * stride), stride):
        line = lines[i].strip()
        if len(line) >= 4:
            # Lane 0 is the right-most 4 characters of the 640-bit word
            lane0_hex = line[-4:] 
            val = hex_to_bfloat16_float(lane0_hex)
            predictions.append([val])
            
    # Safety pad in case the simulator crashed early
    while len(predictions) < num_nodes:
        predictions.append([0.0])
                
    return np.array(predictions, dtype=np.float32)

def float_to_bf16_hex(f: float):
    """Converts a standard 32-bit float to a 16-bit BFloat16 Hex string."""
    u32 = struct.unpack('>I', struct.pack('>f', f))[0]
    return f"{u32 >> 16:04x}"

def patch_npu_mif(current_state, edge_index, template_path="input_template.mif", output_path="input.mif"):
    """
    Surgically overwrites the X Tensor in the hardware FIFO queue with real-time API data.
    """
    num_nodes = current_state.shape[0]
    num_edges = edge_index.shape[1]
    
    # 1. REPLICATE NPU HARDWARE SORTING
    # The NPU scheduler sorts nodes by degree to optimize DSP utilization.
    adj_matrix = [[] for _ in range(num_nodes)]
    for i in range(num_edges):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        adj_matrix[u].append(v)
        
    sorted_nodes = sorted([{ "id": i, "deg": len(l) } for i, l in enumerate(adj_matrix)], key=lambda x: x["deg"])
    
    ordered_nodes = []
    count = 0
    while count < num_nodes:
        for i in range(3): # DOT_PER_DSP = 3
            if count >= num_nodes: break
            ordered_nodes.append(sorted_nodes[count]["id"])
            count += 1
            
    # 2. CALCULATE HARDWARE MEMORY ADDRESS
    # Offset = mfu0_add_ones(3) + mvu_zeros(1) + mvu_edges(num_edges) + mfu1_weights(num_edges)
    start_addr = 4 + (2 * num_edges)
    
    # 3. FORMAT THE BFLOAT16 VECTOR LANES
    node_hex_map = {}
    for n_id in ordered_nodes:
        features = current_state[n_id] # Your 7-channel Spatio-Temporal Data
        
        # Pad to the rigid 40-lane hardware width
        padded = np.zeros(40, dtype=np.float32)
        padded[:len(features)] = features
        
        # Pack the 640-bit word (Lane 39 at the MSB/Left, Lane 0 at the LSB/Right)
        hex_word = "".join([float_to_bf16_hex(padded[i]) for i in range(39, -1, -1)])
        node_hex_map[n_id] = hex_word

    # 4. SURGICAL FILE OVERWRITE
    with open(template_path, "r") as f:
        lines = f.readlines()
        
    with open(output_path, "w") as f:
        for line in lines:
            if ":" in line and line.strip().endswith(";"):
                addr_str = line.split(":")[0].strip()
                if addr_str.isdigit():
                    addr = int(addr_str)
                    
                    # If the address is inside our X Tensor block, overwrite it
                    if start_addr <= addr < start_addr + num_nodes:
                        node_idx = ordered_nodes[addr - start_addr]
                        hex_str = node_hex_map[node_idx]
                        f.write(f"{addr}: {hex_str};\n")
                        continue 
            f.write(line)
    
    print(f"Patched NPU memory from addr {start_addr} to {start_addr + num_nodes - 1}.")

# --- [Keep haversine_m, build_routing_graph exactly as before] ---
def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin(np.radians(lat2 - lat1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return float(2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))

def build_routing_graph(edges, coords):
    g = nx.Graph()
    for i in range(len(coords)): g.add_node(i)
    for k in range(edges.shape[1]):
        u, v = int(edges[0, k]), int(edges[1, k])
        if u != v: g.add_edge(u, v, distance=haversine_m(coords[u][0], coords[u][1], coords[v][0], coords[v][1]))
    return g

def apply_osm_ai_times(G: nx.MultiDiGraph, tmc_kdtree: KDTree, speed_ratios: np.ndarray):
    for u, v, k, d in G.edges(keys=True, data=True):
        lat = 0.5 * (float(G.nodes[u]["y"]) + float(G.nodes[v]["y"]))
        lng = 0.5 * (float(G.nodes[u]["x"]) + float(G.nodes[v]["x"]))
        _, ti = tmc_kdtree.query([lat, lng])
        
        # Use the predicted speed ratio (0.1 to 1.0)
        ratio = max(0.1, float(speed_ratios[int(ti)]))
        edge_speed_ms = BASE_SPEED_MS * ratio
        
        length_m = float(d.get("length", 1.0))
        d["std_time_sec"] = length_m / BASE_SPEED_MS
        d["ai_time_sec"] = length_m / edge_speed_ms

def get_osm_path_data(G, path, weight_key):
    coords, total_time = [], 0.0
    for u, v in zip(path[:-1], path[1:]):
        k = min(G[u][v].keys(), key=lambda kk: float(G[u][v][kk].get(weight_key, 1e30)))
        total_time += float(G[u][v][k].get(weight_key, 0.0))
        geom = G[u][v][k].get("geometry")
        if geom: coords.extend([{"lat": float(lat), "lng": float(lon)} for lon, lat in geom.coords])
        else: coords.extend([{"lat": float(G.nodes[u]["y"]), "lng": float(G.nodes[u]["x"])}, {"lat": float(G.nodes[v]["y"]), "lng": float(G.nodes[v]["x"])}])
    return coords, total_time

@app.on_event("startup")
async def load_ai_assets():
    global model, edge_index, node_coords, kdtree, nx_graph, osm_graph
    dataset = load_toronto_traffic_data()
    edge_index = next(iter(dataset)).edge_index.to(device)

    df = pd.read_csv("svc_raw_data_speed_2020_2024.csv")
    df = df[(df["latitude"] >= 43.643) & (df["latitude"] <= 43.658) & (df["longitude"] >= -79.395) & (df["longitude"] <= -79.370)]
    nodes_df = df.drop_duplicates(subset=["centreline_id"])[["centreline_id", "latitude", "longitude"]].sort_values("centreline_id").reset_index(drop=True)
    node_coords = nodes_df[["latitude", "longitude"]].values
    kdtree = KDTree(node_coords)

    nx_graph = build_routing_graph(edge_index.cpu().numpy(), node_coords)
    try:
        import osmnx as ox
        osm_graph = ox.graph_from_bbox(OSM_BBOX, network_type="drive", simplify=True)
    except: osm_graph = None

    model = TrafficPredictorGNN(node_features=7, hidden_dim=32)
    model.load_state_dict(torch.load("traffic_gnn_weights.pth", map_location=device))
    model.to(device).eval()

class TrafficPoint(BaseModel):
    lat: float; lng: float

class RouteRequest(BaseModel):
    custom_traffic: List[TrafficPoint]
    start_pt: Optional[TrafficPoint] = None
    end_pt: Optional[TrafficPoint] = None

'''
In-flight code
@app.post("/predict_route")
async def predict_route(req: RouteRequest):
    global edge_index, node_coords, kdtree, nx_graph, osm_graph
    
    if edge_index is None:
        return {"error": "Model not initialized"}

    num_nodes = node_coords.shape[0]
    current_state = np.ones((num_nodes, 7), dtype=np.float32) 

    bottleneck_indices = []
    if req.custom_traffic:
        for point in req.custom_traffic:
            _, node_idx = kdtree.query([point.lat, point.lng])
            node_idx = int(node_idx)
            current_state[node_idx, 0:4] = 0.1 
            bottleneck_indices.append(node_idx)

    # ---------------------------------------------------------
    # 1. PREPARE THE HANDSHAKE (Clear stale simulator data)
    # ---------------------------------------------------------
    # Update this path to exactly where your external simulator drops the output
    sim_out_path = "simulator/output.mif" 
    
    if os.path.exists(sim_out_path):
        os.remove(sim_out_path)
        print("Cleared old simulator output.")

    # ---------------------------------------------------------
    # 2. HOT PATCH THE NPU HARDWARE MEMORY
    # ---------------------------------------------------------
    patch_npu_mif(
        current_state, 
        edge_index.cpu().numpy(), 
        template_path="compiler/input_template.mif", 
        output_path="compiler/input.mif"
    )
    print("Patched input.mif. Waiting for external NPU simulation...")

    # ---------------------------------------------------------
    # 3. ASYNCHRONOUS WAIT LOOP (Listen for output.mif)
    # ---------------------------------------------------------
    timeout_seconds = 60.0
    elapsed = 0.0
    
    while not os.path.exists(sim_out_path):
        await asyncio.sleep(0.5) # Yields control back to the server so it doesn't freeze
        elapsed += 0.5
        if elapsed >= timeout_seconds:
            print("Simulation timeout!")
            return {"error": "Hardware simulation timed out after 60 seconds."}

    # Add a tiny 200ms buffer to ensure the external C++ process has fully finished writing the file
    await asyncio.sleep(0.2)
    print("Detected new output.mif! Decoding Silicon Output...")

    # ---------------------------------------------------------
    # 4. EXTRACT AND DECODE THE SILICON OUTPUT
    # ---------------------------------------------------------
    npu_preds_numpy = load_sim_output(sim_out_path, num_nodes=num_nodes)
    
    # Convert back to a PyTorch tensor so the routing logic works seamlessly
    pred_ratio = torch.tensor(npu_preds_numpy).to(device)

    # Re-pin the manual bottlenecks just to be safe
    if req.custom_traffic:
        for idx in bottleneck_indices:
            pred_ratio[idx, 0] = 0.1 

    all_ratios = pred_ratio.squeeze().cpu().numpy()
    travel_data = {"standard_time_min": 0, "ai_time_min": 0, "std_route": [], "ai_route": []}

    # ... [Keep the rest of your NetworkX shortest_path logic exactly the same] ...
'''

@app.post("/predict_route")
async def predict_full_map(req: RouteRequest):
    num_nodes = len(node_coords)
    
    # 1. Initialize a 7-channel tensor (Default: 1.0 Free Flow)
    current_state = np.ones((num_nodes, 7), dtype=np.float32)

    # 2. Inject Contextual Metadata (Simulating 5:00 PM Rush Hour on a Weekday)
    time_mins = 17 * 60 
    current_state[:, 4] = np.sin(2 * np.pi * time_mins / 1440.0)
    current_state[:, 5] = np.cos(2 * np.pi * time_mins / 1440.0)
    current_state[:, 6] = 0.0 # 0 = Weekday

    # 3. Inject Historical User Bottlenecks
    bottleneck_indices = []
    for point in req.custom_traffic:
        _, node_idx = kdtree.query([point.lat, point.lng])
        node_idx = int(node_idx)
        # Force the last 4 historical time steps to severe gridlock (0.1 Speed Ratio)
        current_state[node_idx, 0:4] = 0.1 
        bottleneck_indices.append(node_idx)

    '''
    sim_done_path = "simulator/sim_done"
    out_file_path = "simulator/out_file" 
    
    if os.path.exists(sim_done_path): os.remove(sim_done_path)
    if os.path.exists(out_file_path): os.remove(out_file_path)

    # ---------------------------------------------------------
    # 2. HOT PATCH THE NPU HARDWARE MEMORY
    # ---------------------------------------------------------
    patch_npu_mif(
        current_state, 
        edge_index.cpu().numpy(), 
        template_path="compiler/input_template.mif", 
        output_path="compiler/input.mif"
    )
    print("Patched input.mif. Waiting for external NPU simulation...")

    # ---------------------------------------------------------
    # 3. ASYNCHRONOUS WAIT LOOP (Listen for sim_done)
    # ---------------------------------------------------------
    timeout_seconds = 120.0
    elapsed = 0.0
    
    while not os.path.exists(sim_done_path):
        await asyncio.sleep(0.1) # Yields control so the web server doesn't freeze
        elapsed += 0.1
        if elapsed >= timeout_seconds:
            print("Simulation timeout!")
            return {"error": "Hardware simulation timed out after 60 seconds."}

    # Add a tiny 100ms buffer to ensure the OS has finished flushing out_file to disk
    await asyncio.sleep(0.1)
    print("Detected sim_done! Decoding Silicon Output...")

    # ---------------------------------------------------------
    # 4. EXTRACT AND DECODE THE SILICON OUTPUT
    # ---------------------------------------------------------
    npu_preds_numpy = parse_raw_hex_output(out_file_path, num_nodes=num_nodes)
    
    # Optional: If you haven't implemented Sigmoid in the C++ MFU yet, do it in Python here:
    # npu_preds_numpy = 1 / (1 + np.exp(-npu_preds_numpy))

    # Convert back to a PyTorch tensor so the routing logic works seamlessly
    pred_ratio = torch.tensor(npu_preds_numpy).to(device)

    # Re-pin the manual bottlenecks just to be safe
    if req.custom_traffic:
        for idx in bottleneck_indices:
            pred_ratio[idx, 0] = 0.1 

    all_ratios = pred_ratio.squeeze().cpu().numpy()
    travel_data = {"standard_time_min": 0, "ai_time_min": 0, "std_route": [], "ai_route": []}
    '''

    x = torch.tensor(current_state).to(device)
    #np.save("x_array_dump.npy", x.detach().cpu().numpy())

    print(x)

    with torch.no_grad():
        pred_ratio = model(x, edge_index)
        for idx in bottleneck_indices:
            pred_ratio[idx, 0] = 0.1 # Pin original jams

    all_ratios = pred_ratio.squeeze().cpu().numpy()
    travel_data = {"standard_time_min": 0, "ai_time_min": 0, "std_route": [], "ai_route": []}

    if req.start_pt and req.end_pt and osm_graph is not None:
        import osmnx as ox
        orig = int(ox.distance.nearest_nodes(osm_graph, req.start_pt.lng, req.start_pt.lat))
        dest = int(ox.distance.nearest_nodes(osm_graph, req.end_pt.lng, req.end_pt.lat))
        
        apply_osm_ai_times(osm_graph, kdtree, all_ratios)
        std_path = nx.shortest_path(osm_graph, orig, dest, weight="std_time_sec")
        ai_path = nx.shortest_path(osm_graph, orig, dest, weight="ai_time_sec")
        
        travel_data["std_route"], _ = get_osm_path_data(osm_graph, std_path, "std_time_sec")
        travel_data["ai_route"], ai_sec = get_osm_path_data(osm_graph, ai_path, "ai_time_sec")
        _, std_sec_in_traffic = get_osm_path_data(osm_graph, std_path, "ai_time_sec")
        
        travel_data["standard_time_min"] = int(std_sec_in_traffic // 60)
        travel_data["ai_time_min"] = int(ai_sec // 60)

    # Invert the ratio for Deck.gl (0.1 ratio = 0.9 Congestion Heat)
    results = [{"lat": float(node_coords[i][0]), "lng": float(node_coords[i][1]), "congestion": 1.0 - float(all_ratios[i])} for i in range(num_nodes)]
    return {"predictions": results, "travel": travel_data, "status": "success"}