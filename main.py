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

    x = torch.tensor(current_state).to(device)

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