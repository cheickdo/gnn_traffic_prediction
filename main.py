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
import subprocess

def transfer_mif_to_remote(local_path="compiler/input.mif"):
    """
    Pushes the newly generated input.mif file to the remote EDA server via SCP.
    Assumes SSH keys are configured for passwordless entry.
    """
    remote_user = "doumbiac"
    remote_host = "betzgrp-wintermute.eecg.utoronto.ca"
    remote_dir = "~/Documents/npu_gnn_bringup/rtl/mif_files/"
    remote_target = f"{remote_user}@{remote_host}:{remote_dir}"
    
    print(f"Transferring {local_path} to remote server...")
    try:
        # Use subprocess to execute the SCP command
        subprocess.run(["scp", local_path, remote_target], check=True)
        print("Transfer complete.")
    except subprocess.CalledProcessError as e:
        print(f"SCP transfer failed: {e}")

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
    Surgically overwrites the X Tensor in the hardware FIFO queue by hunting 
    for the exact hexadecimal signature of the dummy baseline data.
    """
    num_nodes = current_state.shape[0]
    num_edges = edge_index.shape[1]
    
    # 1. REPLICATE NPU HARDWARE SORTING
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
            
    # 2. FORMAT THE BFLOAT16 VECTOR LANES
    node_hex_map = {}
    for n_id in ordered_nodes:
        features = current_state[n_id] 
        padded = np.zeros(40, dtype=np.float32)
        padded[:len(features)] = features
        hex_word = "".join([float_to_bf16_hex(padded[i]) for i in range(39, -1, -1)])
        node_hex_map[n_id] = hex_word

    # 3. SURGICAL SIGNATURE MATCHING
    # The dummy feature payload is exactly 33 zeros followed by 7 ones (3f80)
    dummy_signature = ("0000" * 33) + ("3f80" * 7)

    with open(template_path, "r") as f:
        lines = f.readlines()
        
    with open(output_path, "w") as f:
        node_idx = 0
        for line in lines:
            # If we find the dummy payload, overwrite it with live API data
            if dummy_signature in line and node_idx < num_nodes:
                n_id = ordered_nodes[node_idx]
                hex_str = node_hex_map[n_id]
                
                # Keep the line number prefix (e.g., "0: ")
                addr_prefix = line.split(":")[0].strip()
                f.write(f"{addr_prefix}: {hex_str};\n")
                node_idx += 1
            else:
                f.write(line)
    
    print(f" Successfully patched {node_idx} hardware memory addresses via signature matching.")

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
    
    print("🔧 Shrinking backend to match 21-Node NPU Hardware...")
    
    # 1. Load the exact 21-node topology from the NPU export
    blob = np.load("toronto_npu_export.npz")
    src = blob["edges_src"].astype(np.int64)
    dst = blob["edges_dest"].astype(np.int64)
    edge_index = torch.tensor([src, dst], dtype=torch.long).to(device)
    
    num_npu_nodes = int(max(src.max(), dst.max()) + 1)
    
    # 2. Get real Toronto coordinates, but ONLY take the first 21 
    # to represent our toy hardware graph physically on the map.
    df = pd.read_csv("svc_raw_data_speed_2020_2024.csv")
    df = df[(df["latitude"] >= 43.643) & (df["latitude"] <= 43.658) & (df["longitude"] >= -79.395) & (df["longitude"] <= -79.370)]
    nodes_df = df.drop_duplicates(subset=["centreline_id"])[["centreline_id", "latitude", "longitude"]].sort_values("centreline_id").reset_index(drop=True)
    
    node_coords = nodes_df[["latitude", "longitude"]].values[:num_npu_nodes]
    kdtree = KDTree(node_coords)
    
    # 3. Build the routing graph using ONLY these 21 nodes
    nx_graph = build_routing_graph(edge_index.cpu().numpy(), node_coords)
    
    # Disable full-scale OSM routing since we are restricted to a 21-node micro-graph
    osm_graph = None 
    
    print(f"Micro-Graph Initialized: {num_npu_nodes} Nodes strictly bound to UI.")

class TrafficPoint(BaseModel):
    lat: float; lng: float

class RouteRequest(BaseModel):
    custom_traffic: List[TrafficPoint]
    start_pt: Optional[TrafficPoint] = None
    end_pt: Optional[TrafficPoint] = None

@app.get("/valid_nodes")
async def get_valid_nodes():
    """Exposes the exact NPU hardware node coordinates to the frontend for UI snapping."""
    if node_coords is None:
        return {"nodes": []}
    return {"nodes": [{"lat": float(c[0]), "lng": float(c[1])} for c in node_coords]}
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

    sim_done_path = "sim_done"
    out_file_path = "out_file" 
    
    if os.path.exists(sim_done_path): os.remove(sim_done_path)
    if os.path.exists(out_file_path): os.remove(out_file_path)

    # ---------------------------------------------------------
    # 2. HOT PATCH THE NPU HARDWARE MEMORY
    # ---------------------------------------------------------
    patch_npu_mif(
        current_state, 
        edge_index.cpu().numpy(), 
        template_path="input_template.mif", 
        output_path="input.mif"
    )
    print("Patched input.mif. Waiting for external NPU simulation...")
    
    # PUSH TO REMOTE SERVER
    transfer_mif_to_remote("input.mif")
    print("Waiting for remote NPU simulation...")

    # ---------------------------------------------------------
    # 3. ASYNCHRONOUS WAIT LOOP (Listen for sim_done)
    # ---------------------------------------------------------
    timeout_seconds = 120.0
    elapsed = 0.0
    
    remote_sim_done = "doumbiac@betzgrp-wintermute.eecg.utoronto.ca:~/Documents/npu_gnn_bringup/rtl/sim_done"
    remote_out_file = "doumbiac@betzgrp-wintermute.eecg.utoronto.ca:~/Documents/npu_gnn_bringup/rtl/out_file"
    local_out_file = "out_file"
    
    simulation_finished = False

    while elapsed < timeout_seconds:
        # Check if the sim_done file exists on the remote machine
        # 'ssh' returns exit code 0 if the file exists (using 'test -f')
        check_cmd = ["ssh", "doumbiac@betzgrp-wintermute.eecg.utoronto.ca", "test", "-f", "~/Documents/npu_gnn_bringup/rtl/sim_done"]
        result = subprocess.run(check_cmd)
        
        if result.returncode == 0:
            simulation_finished = True
            break
            
        await asyncio.sleep(0.5) # Yield control to the web server
        elapsed += 0.5

    if not simulation_finished:
        return {"error": "Hardware simulation timed out after 150 seconds."}

    # ---------------------------------------------------------
    # 4. FETCH AND DECODE THE SILICON OUTPUT
    # ---------------------------------------------------------
    print("Detected remote sim_done! Fetching Silicon Output...")
    try:
        # Pull the out_file back from the server
        subprocess.run(["scp", remote_out_file, local_out_file], check=True)
    except subprocess.CalledProcessError:
        return {"error": "Failed to retrieve the simulation output from the remote server."}

    npu_preds_numpy = parse_raw_hex_output(local_out_file, num_nodes=num_nodes)

    # Optional: If you haven't implemented Sigmoid in the C++ MFU yet, do it in Python here:
    npu_preds_numpy = 1 / (1 + np.exp(-npu_preds_numpy))

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
    '''
    # --- ROUTE USING THE 21-NODE HARDWARE GRAPH ---
    if req.start_pt and req.end_pt:
        # 1. Snap the user's map clicks to the nearest 21-node ID
        _, orig = kdtree.query([req.start_pt.lat, req.start_pt.lng])
        _, dest = kdtree.query([req.end_pt.lat, req.end_pt.lng])
        orig, dest = int(orig), int(dest)

        # 2. Apply the Silicon NPU predictions to the network edges
        for u, v, data in nx_graph.edges(data=True):
            dist = data['distance']
            data['std_time_sec'] = dist / BASE_SPEED_MS
            
            # Average the AI congestion ratio of the two connecting nodes
            edge_ratio = max(0.1, float((all_ratios[u] + all_ratios[v]) / 2.0))
            data['ai_time_sec'] = dist / (BASE_SPEED_MS * edge_ratio)

        # 3. Calculate Shortest Paths through the 21-node toy graph
        try:
            std_path = nx.shortest_path(nx_graph, orig, dest, weight="std_time_sec")
            ai_path = nx.shortest_path(nx_graph, orig, dest, weight="ai_time_sec")

            # 4. Extract Total Times and Coordinate Arrays for the Frontend map
            std_time_sec = sum(nx_graph[u][v]['std_time_sec'] for u, v in zip(std_path[:-1], std_path[1:]))
            ai_time_sec = sum(nx_graph[u][v]['ai_time_sec'] for u, v in zip(ai_path[:-1], ai_path[1:]))
            
            # Convert seconds to minutes (minimum 1 minute to avoid showing 0)
            travel_data["standard_time_min"] = max(1, int(std_time_sec // 60))
            travel_data["ai_time_min"] = max(1, int(ai_time_sec // 60))
            
            travel_data["std_route"] = [{"lat": float(node_coords[node][0]), "lng": float(node_coords[node][1])} for node in std_path]
            travel_data["ai_route"] = [{"lat": float(node_coords[node][0]), "lng": float(node_coords[node][1])} for node in ai_path]
            
        except nx.NetworkXNoPath:
            print("No valid path found between these nodes in the toy graph.")

    # Invert the ratio for Deck.gl (0.1 ratio = 0.9 Congestion Heat)
    results = [{"lat": float(node_coords[i][0]), "lng": float(node_coords[i][1]), "congestion": 1.0 - float(all_ratios[i])} for i in range(num_nodes)]
    return {"predictions": results, "travel": travel_data, "status": "success"}
    # Invert the ratio for Deck.gl (0.1 ratio = 0.9 Congestion Heat)
    results = [{"lat": float(node_coords[i][0]), "lng": float(node_coords[i][1]), "congestion": 1.0 - float(all_ratios[i])} for i in range(num_nodes)]
    return {"predictions": results, "travel": travel_data, "status": "success"}
