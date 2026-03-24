from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
import numpy as np
from scipy.spatial import KDTree
import pandas as pd

from model import TrafficPredictorGNN
from data_loader import load_toronto_traffic_data

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Variables
model = None
edge_index = None
edge_attr = None
node_coords = None
kdtree = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_VOLUME = 500.0 # Normalization constant

@app.on_event("startup")
async def load_ai_assets():
    global model, edge_index, edge_attr, node_coords, kdtree
    
    print("Loading Toronto graph data into memory...")
    dataset = load_toronto_traffic_data()
    
    # Extract static graph structure from the first time step
    first_step = next(iter(dataset))
    edge_index = first_step.edge_index.to(device)
    edge_attr = first_step.edge_attr.to(device) if first_step.edge_attr is not None else None
    
    # Extract raw coordinates for the frontend
    df1 = pd.read_csv("svc_raw_data_volume_2015_2019.csv")
    df2 = pd.read_csv("svc_raw_data_volume_2020_2024.csv")
    df = pd.concat([df1, df2], ignore_index=True)
    nodes_df = df.drop_duplicates(subset=['centreline_id'])[['centreline_id', 'latitude', 'longitude']]
    nodes_df = nodes_df.sort_values('centreline_id').reset_index(drop=True)

    node_coords = nodes_df[['latitude', 'longitude']].values
    kdtree = KDTree(node_coords)

    print("Loading trained GNN weights...")
    model = TrafficPredictorGNN(node_features=1, hidden_dim=32)
    # Use map_location=device to load GPU weights on CPU if necessary
    model.load_state_dict(torch.load("traffic_gnn_weights.pth", map_location=device))
    model.to(device)
    model.eval()
    print("Backend is ready for full-graph predictions!")

# Data Models
class TrafficPoint(BaseModel):
    lat: float
    lng: float
    volume: float

class MapRequest(BaseModel):
    custom_traffic: List[TrafficPoint]

@app.post("/predict_map")
async def predict_full_map(req: MapRequest):
    num_nodes = len(node_coords)
    current_state = np.full((num_nodes, 1), 20.0 / MAX_VOLUME)
    
    # 1. Track our bottlenecks so we can keep them active over time
    bottleneck_indices = []
    for point in req.custom_traffic:
        dist, node_idx = kdtree.query([point.lat, point.lng])
        current_state[node_idx, 0] = min(point.volume / MAX_VOLUME, 1.0)
        bottleneck_indices.append((node_idx, current_state[node_idx, 0]))
        
    x = torch.tensor(current_state, dtype=torch.float32).to(device)
    
    # 2. THE FIX: Auto-Regressive Loop (Simulate 1 hour / 4 time steps)
    future_steps = 4 
    
    with torch.no_grad():
        for step in range(future_steps):
            # Predict the next 15-minute interval
            x = model(x, edge_index, edge_attr)
            
            # Force the original bottlenecks to stay congested
            # Otherwise, the GNN naturally predicts the jam clears up
            for idx, initial_vol in bottleneck_indices:
                x[idx, 0] = initial_vol 
            
    # Extract the final state after 1 hour has passed
    all_predictions = x.squeeze().cpu().numpy()
    
    # 3. Format output for the frontend map rendering
    results = []
    for i in range(num_nodes):
        pred_vol = max(0, all_predictions[i] * MAX_VOLUME)
        results.append({
            "lat": node_coords[i][0],
            "lng": node_coords[i][1],
            "predicted_volume": int(pred_vol)
        })
        
    return {"predictions": results, "status": "success"}
