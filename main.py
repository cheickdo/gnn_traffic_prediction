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

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

model = None
edge_index = None
node_coords = None
kdtree = None
nx_graph = None
osm_graph = None  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Base city speed limit
DEFAULT_SPEED_KMH = 40.0 
BASE_SPEED_MS = DEFAULT_SPEED_KMH * 1000.0 / 3600.0

OSM_BBOX = (-79.410, 43.640, -79.355, 43.675)

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
        if u != v:
            d = haversine_m(coords[u][0], coords[u][1], coords[v][0], coords[v][1])
            g.add_edge(u, v, distance=d)
    return g

def apply_osm_ai_times(G: nx.MultiDiGraph, tmc_kdtree: KDTree, pred_norm: np.ndarray):
    for u, v, k, d in G.edges(keys=True, data=True):
        geom = d.get("geometry")
        if geom is not None:
            m = geom.interpolate(0.5, normalized=True)
            lat, lng = float(m.y), float(m.x)
        else:
            lat = 0.5 * (float(G.nodes[u]["y"]) + float(G.nodes[v]["y"]))
            lng = 0.5 * (float(G.nodes[u]["x"]) + float(G.nodes[v]["x"]))
            
        _, ti = tmc_kdtree.query([lat, lng])
        congestion_index = float(pred_norm[int(ti)])
        
        # ETA MATH: Max speed is 100%, Min speed is 10% (during total gridlock)
        speed_multiplier = max(0.1, 1.0 - congestion_index)
        edge_speed_ms = BASE_SPEED_MS * speed_multiplier
        
        length_m = float(d.get("length", 1.0))
        d["std_time_sec"] = length_m / BASE_SPEED_MS
        d["ai_time_sec"] = length_m / edge_speed_ms

def get_osm_path_data(G, path, weight_key):
    coords = []
    total_time = 0.0
    for u, v in zip(path[:-1], path[1:]):
        k = min(G[u][v].keys(), key=lambda kk: float(G[u][v][kk].get(weight_key, 1e30)))
        total_time += float(G[u][v][k].get(weight_key, 0.0))
        geom = G[u][v][k].get("geometry")
        if geom:
            coords.extend([{"lat": float(lat), "lng": float(lon)} for lon, lat in geom.coords])
        else:
            coords.extend([{"lat": float(G.nodes[u]["y"]), "lng": float(G.nodes[u]["x"])},
                           {"lat": float(G.nodes[v]["y"]), "lng": float(G.nodes[v]["x"])}])
    return coords, total_time

def load_osm_drive_network():
    global osm_graph
    try:
        import osmnx as ox
        cache = Path(__file__).resolve().parent / "cache" / "osmnx"
        cache.mkdir(parents=True, exist_ok=True)
        ox.settings.cache_folder = str(cache)
        ox.settings.data_folder = str(cache)
        osm_graph = ox.graph_from_bbox(OSM_BBOX, network_type="drive", simplify=True)
    except Exception:
        osm_graph = None

@app.on_event("startup")
async def load_ai_assets():
    global model, edge_index, node_coords, kdtree, nx_graph
    dataset = load_toronto_traffic_data()
    edge_index = next(iter(dataset)).edge_index.to(device)

    df = pd.read_csv("svc_raw_data_speed_2020_2024.csv")
    df = df[(df["latitude"] >= 43.640) & (df["latitude"] <= 43.675) & (df["longitude"] >= -79.410) & (df["longitude"] <= -79.355)]
    nodes_df = df.drop_duplicates(subset=["centreline_id"])[["centreline_id", "latitude", "longitude"]].sort_values("centreline_id").reset_index(drop=True)
    node_coords = nodes_df[["latitude", "longitude"]].values
    kdtree = KDTree(node_coords)

    nx_graph = build_routing_graph(edge_index.cpu().numpy(), node_coords)
    load_osm_drive_network()

    model = TrafficPredictorGNN(node_features=1, hidden_dim=32)
    model.load_state_dict(torch.load("traffic_gnn_weights.pth", map_location=device))
    model.to(device)
    model.eval()

class TrafficPoint(BaseModel):
    lat: float
    lng: float

class RouteRequest(BaseModel):
    custom_traffic: List[TrafficPoint]
    start_pt: Optional[TrafficPoint] = None
    end_pt: Optional[TrafficPoint] = None

@app.post("/predict_route")
async def predict_full_map(req: RouteRequest):
    num_nodes = len(node_coords)
    current_state = np.zeros((num_nodes, 1))

    bottleneck_indices = []
    for point in req.custom_traffic:
        _, node_idx = kdtree.query([point.lat, point.lng])
        node_idx = int(node_idx)
        current_state[node_idx, 0] = 1.0 # 100% Congested
        bottleneck_indices.append(node_idx)

    x = torch.tensor(current_state, dtype=torch.float32).to(device)

    # ONE-SHOT SPATIAL MATRIX MULTIPLICATION
    with torch.no_grad():
        pred_congestion = model(x, edge_index)
        # Pin original bottlenecks to 1.0
        for idx in bottleneck_indices:
            pred_congestion[idx, 0] = 1.0

    all_predictions = pred_congestion.squeeze().cpu().numpy()

    travel_data = {"standard_time_min": 0, "ai_time_min": 0, "std_route": [], "ai_route": []}
    route_message = ""

    if req.start_pt and req.end_pt:
        _, start_idx = kdtree.query([req.start_pt.lat, req.start_pt.lng])
        _, end_idx = kdtree.query([req.end_pt.lat, req.end_pt.lng])
        
        try:
            if osm_graph is not None:
                import osmnx as ox
                orig = int(ox.distance.nearest_nodes(osm_graph, req.start_pt.lng, req.start_pt.lat))
                dest = int(ox.distance.nearest_nodes(osm_graph, req.end_pt.lng, req.end_pt.lat))
                
                apply_osm_ai_times(osm_graph, kdtree, all_predictions)
                
                std_path = nx.shortest_path(osm_graph, orig, dest, weight="std_time_sec")
                ai_path = nx.shortest_path(osm_graph, orig, dest, weight="ai_time_sec")
                
                travel_data["std_route"], std_sec = get_osm_path_data(osm_graph, std_path, "std_time_sec")
                travel_data["ai_route"], ai_sec = get_osm_path_data(osm_graph, ai_path, "ai_time_sec")
                
                travel_data["standard_time_min"] = int(std_sec // 60)
                travel_data["ai_time_min"] = int(ai_sec // 60)
            else:
                # TMC Fallback Logic (simplified for brevity)
                for u, v in nx_graph.edges():
                    pred = 0.5 * (float(all_predictions[u]) + float(all_predictions[v]))
                    d = nx_graph[u][v]["distance"]
                    speed_multiplier = max(0.1, 1.0 - pred)
                    nx_graph[u][v]["std_time_sec"] = d / BASE_SPEED_MS
                    nx_graph[u][v]["ai_time_sec"] = d / (BASE_SPEED_MS * speed_multiplier)
                    
                std_path = nx.shortest_path(nx_graph, source=start_idx, target=end_idx, weight="std_time_sec")
                ai_path = nx.shortest_path(nx_graph, source=start_idx, target=end_idx, weight="ai_time_sec")
                
                travel_data["std_route"] = [{"lat": float(node_coords[n][0]), "lng": float(node_coords[n][1])} for n in std_path]
                travel_data["ai_route"] = [{"lat": float(node_coords[n][0]), "lng": float(node_coords[n][1])} for n in ai_path]
                
                travel_data["standard_time_min"] = int(sum(nx_graph[u][v]["std_time_sec"] for u, v in zip(std_path[:-1], std_path[1:])) // 60)
                travel_data["ai_time_min"] = int(sum(nx_graph[u][v]["ai_time_sec"] for u, v in zip(ai_path[:-1], ai_path[1:])) // 60)
                
        except nx.NetworkXNoPath:
            route_message = "No valid driving route found."

    results = [{"lat": float(node_coords[i][0]), "lng": float(node_coords[i][1]), "congestion": float(all_predictions[i])} for i in range(num_nodes)]

    return {"predictions": results, "travel": travel_data, "route_message": route_message, "status": "success"}