from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import random

app = FastAPI()

# Configure CORS (Cross-Origin Resource Sharing)
# This is required so your frontend on port 8080 can talk to this backend on port 8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For the fair demo, allowing all origins is fine
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the data structure we expect from the frontend
class Coordinate(BaseModel):
    lat: float
    lng: float

class RouteRequest(BaseModel):
    origin: Coordinate
    destination: Coordinate

@app.post("/predict")
async def predict_traffic(route: RouteRequest):
    # 1. Receive the data
    print(f"Routing from: [{route.origin.lat}, {route.origin.lng}]")
    print(f"Routing to: [{route.destination.lat}, {route.destination.lng}]")
    
    # 2. Node Mapping (Placeholder)
    # TODO: Use libraries like osmnx here to snap these raw GPS coordinates 
    # to the nearest intersection nodes on your road network graph.
    
    # 3. GNN Inference (Placeholder)
    # TODO: Feed the path sequence into your PyTorch Geometric STGCN model here.
    # For now, we will mock the AI's output so the frontend has something to display.
    mock_eta = random.randint(12, 55)
    traffic_states = ["Clear", "Moderate", "Heavy Congestion", "Standstill"]
    mock_state = random.choice(traffic_states)

    # 4. Return the prediction to the Leaflet frontend
    return {
        "eta_minutes": mock_eta,
        "traffic_state": mock_state,
        "status": "success"
    }
