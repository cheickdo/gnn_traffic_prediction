import asyncio
import os
import struct
import subprocess
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional
import torch
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import networkx as nx

from model import TrafficPredictorGNN
from data_loader import load_toronto_traffic_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

model, edge_index, node_coords, kdtree, nx_graph, osm_graph = None, None, None, None, None, None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_SPEED_KMH = 40.0
BASE_SPEED_MS = DEFAULT_SPEED_KMH * 1000.0 / 3600.0
OSM_BBOX = (-79.395, 43.643, -79.370, 43.658)

# Hardware constants matching toronto_npu_model.py
SIM_BATCH = 3

REMOTE_USER = "doumbiac"
REMOTE_HOST = "betzgrp-wintermute.eecg.utoronto.ca"
REMOTE_BASE = "~/Documents/npu_gnn_bringup/rtl"


def transfer_mif_to_remote(local_path="input.mif"):
    remote_target = f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_BASE}/mif_files/"
    logger.info(f"Transferring {local_path} to remote server...")
    try:
        subprocess.run(["scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                        local_path, remote_target], check=True, timeout=30)
        logger.info("Transfer complete.")
    except subprocess.CalledProcessError as e:
        logger.error(f"SCP transfer failed: {e}")
        raise
    except subprocess.TimeoutExpired:
        logger.error("SCP transfer timed out after 30s")
        raise


def hex_to_bfloat16_float(hex_str):
    """Converts a 4-character hex string (BFloat16) back to a standard Float32."""
    try:
        padded_hex = hex_str.strip().zfill(4) + "0000"
        val = struct.unpack('>f', struct.pack('>I', int(padded_hex, 16)))[0]
        # Guard against NaN/Inf from malformed hardware output
        if not np.isfinite(val):
            return 0.0
        return val
    except Exception:
        return 0.0


def parse_raw_hex_output(filepath, num_nodes):
    """
    Parses the raw hex dump from the C++ NPU simulator.

    The prediction head (npu_gnn_node_prediction) writes one output per node with
    batch=SIM_BATCH=3, producing SIM_BATCH consecutive 640-bit words per node in the
    output buffer. Each 640-bit word contains 40 BFloat16 lanes (40 × 16-bit = 640 bits).

    Layout: total_lines = num_nodes × SIM_BATCH  →  stride = SIM_BATCH = 3
    We read batch-0's word for each node (line index = node_idx × SIM_BATCH).

    Lane 0 (rightmost 4 hex chars) contains output dimension 0, which is the real
    prediction logit (the padded w_head_q has only row-0 non-zero, so dim-0 of the
    output is the actual model prediction; dims 1-31 are sigmoid(0)=0.5 garbage).
    """
    fallback = np.ones(num_nodes, dtype=np.float32) * 0.5

    if not os.path.exists(filepath):
        logger.error(f"Simulator output not found at {filepath}")
        return fallback

    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip() and len(l.strip()) >= 4]

    total_lines = len(lines)
    logger.info(f"NPU raw output: {total_lines} lines for {num_nodes} nodes")

    if total_lines == 0:
        logger.error("NPU output file is empty.")
        return fallback

    # Stride = words per node in the output buffer (SIM_BATCH=3 copies per write_back)
    stride = max(1, total_lines // num_nodes)
    logger.info(f"NPU output stride: {stride} words/node (expected {SIM_BATCH})")

    raw_predictions = []
    for i in range(num_nodes):
        # Index into the batch-0 copy of node i's output
        line_idx = i * stride
        if line_idx < total_lines:
            line = lines[line_idx]
            # Lane 0 = rightmost 4 hex chars of the 640-bit (160 hex-char) word
            lane0_hex = line[-4:].strip()
            raw_val = hex_to_bfloat16_float(lane0_hex)
            raw_predictions.append(raw_val)
        else:
            raw_predictions.append(0.0)

    raw_np = np.array(raw_predictions, dtype=np.float32)

    # The NPU sigmoid LUT is applied in hardware; if it ran cleanly the output is
    # already in [0,1]. We clamp + re-apply sigmoid to handle quantisation drift.
    speed_ratios = 1.0 / (1.0 + np.exp(-np.clip(raw_np, -50, 50)))
    speed_ratios = np.clip(speed_ratios, 0.1, 1.0)

    logger.info(f"NPU raw logit range : [{raw_np.min():.4f}, {raw_np.max():.4f}]")
    logger.info(f"NPU speed-ratio range: [{speed_ratios.min():.4f}, {speed_ratios.max():.4f}]  "
                f"std={speed_ratios.std():.4f}")

    return speed_ratios


def float_to_bf16_hex(f: float):
    """Converts a standard 32-bit float to a 16-bit BFloat16 Hex string."""
    u32 = struct.unpack('>I', struct.pack('>f', f))[0]
    return f"{u32 >> 16:04x}"


def patch_npu_mif(current_state, edge_index, template_path="input_template.mif", output_path="input.mif"):
    """
    Surgically overwrites the X Tensor (node features) in the hardware input FIFO.

    The MIF file layout matches the load order in toronto_npu_model.py:
      Address 0 .. (num_nodes * SIM_BATCH - 1):  mfu0_mul_nodes (node features)
      Then: mvu_edges, mfu1_mul_edge_weights, mfu0_add_ones, mfu1_mul_ones

    Each npu.load() with batch=SIM_BATCH writes SIM_BATCH consecutive entries.
    Nodes are loaded in natural order (0, 1, 2, ..., N-1), NOT sorted by degree.
    """
    num_nodes = current_state.shape[0]
    num_features = current_state.shape[1]

    # X tensor starts at address 0 (mfu0_mul_nodes is loaded FIRST in the driver)
    start_addr = 0
    entries_per_node = SIM_BATCH  # Each node has 3 consecutive MIF entries

    # Format each node's features as a 640-bit BFloat16 hex word
    node_hex_words = {}
    for n_id in range(num_nodes):
        features = current_state[n_id]

        # Pad to the rigid 40-lane hardware width
        padded = np.zeros(40, dtype=np.float32)
        padded[:len(features)] = features

        # Pack: Lane 39 (MSB/Left) to Lane 0 (LSB/Right)
        hex_word = "".join([float_to_bf16_hex(padded[i]) for i in range(39, -1, -1)])
        node_hex_words[n_id] = hex_word

    # Read template
    with open(template_path, "r") as f:
        lines = f.readlines()

    # Overwrite the X tensor addresses
    end_addr = start_addr + (num_nodes * entries_per_node)
    patched_count = 0

    with open(output_path, "w") as f:
        for line in lines:
            if ":" in line and line.strip().endswith(";"):
                addr_str = line.split(":")[0].strip()
                if addr_str.isdigit():
                    addr = int(addr_str)

                    if start_addr <= addr < end_addr:
                        # Map address to node index: addr // SIM_BATCH = node index
                        node_idx = (addr - start_addr) // entries_per_node
                        if node_idx < num_nodes:
                            hex_str = node_hex_words[node_idx]
                            f.write(f"{addr}: {hex_str};\n")
                            patched_count += 1
                            continue
            f.write(line)

    logger.info(f"Patched {patched_count} MIF entries (addresses {start_addr} to {end_addr - 1}) for {num_nodes} nodes")

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

def _osmnx_nearest_nodes(G, x: float, y: float) -> int:
    """Version-agnostic wrapper around osmnx nearest_nodes.

    osmnx 1.x: ox.distance.nearest_nodes(G, X, Y)
    osmnx 2.x: ox.nearest_nodes(G, X, Y)
    """
    import osmnx as ox
    fn = getattr(ox, "nearest_nodes", None) or ox.distance.nearest_nodes
    return int(fn(G, x, y))


def apply_osm_ai_times(G: nx.MultiDiGraph, tmc_kdtree: KDTree, speed_ratios: np.ndarray):
    """Stamp std_time_sec and ai_time_sec onto every OSM edge.

    Uses the osmnx 'travel_time' attribute (added by add_edge_travel_times) as the
    free-flow baseline when available.  Falls back to length / BASE_SPEED_MS otherwise.
    The ai_time_sec is the free-flow time scaled up by the inverse of the predicted
    speed ratio (ratio=0.1 → 10× slower than free-flow).

    BUG FIX: The previous fallback was 1.0 m which produced ~0.09 s/edge and
    therefore 0-minute route estimates.  Real OSM edges are tens to hundreds of
    metres; 50 m is a safe minimum fallback.
    """
    for u, v, k, d in G.edges(keys=True, data=True):
        lat = 0.5 * (float(G.nodes[u]["y"]) + float(G.nodes[v]["y"]))
        lng = 0.5 * (float(G.nodes[u]["x"]) + float(G.nodes[v]["x"]))
        _, ti = tmc_kdtree.query([lat, lng])

        ratio = max(0.1, float(speed_ratios[int(ti)]))

        # Prefer osmnx's pre-computed travel_time (based on OSM speed tags);
        # fall back to length / 40 km/h.  Never let the fallback be < 1 second.
        free_flow_sec = float(d.get("travel_time", 0.0))
        if free_flow_sec <= 0:
            length_m = float(d.get("length", 50.0))   # 50 m safe fallback
            free_flow_sec = max(1.0, length_m / BASE_SPEED_MS)

        d["std_time_sec"] = free_flow_sec
        # Congested time: slower speed = proportionally longer travel time
        d["ai_time_sec"]  = free_flow_sec / ratio

def get_osm_path_data(G, path, weight_key):
    """Extract lat/lng coordinate list and total travel time for an OSM node-id path.

    Uses stored LineString geometry where available (follows actual road curves).
    Falls back to straight node-to-node segments for simplified edges that have no
    geometry attribute.  Duplicate junction points between consecutive edges are
    deduplicated so PathLayer renders a clean, continuous polyline.
    """
    coords: list = []
    total_time: float = 0.0

    for u, v in zip(path[:-1], path[1:]):
        edge_data = G[u][v]
        k = min(edge_data.keys(), key=lambda kk: float(edge_data[kk].get(weight_key, 1e30)))
        total_time += float(edge_data[k].get(weight_key, 0.0))

        geom = edge_data[k].get("geometry")
        if geom:
            edge_pts = [{"lat": float(lat), "lng": float(lon)} for lon, lat in geom.coords]
        else:
            edge_pts = [
                {"lat": float(G.nodes[u]["y"]), "lng": float(G.nodes[u]["x"])},
                {"lat": float(G.nodes[v]["y"]), "lng": float(G.nodes[v]["x"])}
            ]

        if not edge_pts:
            continue

        # Drop the first point of each edge if it duplicates the last accumulated point
        # (the shared junction node between two consecutive edges).
        if coords:
            first = edge_pts[0]
            last  = coords[-1]
            if abs(first["lat"] - last["lat"]) < 1e-9 and abs(first["lng"] - last["lng"]) < 1e-9:
                edge_pts = edge_pts[1:]

        coords.extend(edge_pts)

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
        from shapely.geometry import box as shapely_box

        # Use graph_from_polygon with a shapely box — this API is stable across
        # all osmnx versions and avoids the (north,south,east,west) vs
        # (west,south,east,north) argument-order confusion that broke 1.x/2.x compat.
        bbox_polygon = shapely_box(-79.395, 43.643, -79.370, 43.658)
        osm_graph = ox.graph_from_polygon(
            bbox_polygon, network_type="drive", simplify=True
        )
        # Stamp every edge with speed (km/h) and travel_time (seconds) so
        # apply_osm_ai_times can use real free-flow times as its baseline.
        osm_graph = ox.add_edge_speeds(osm_graph)
        osm_graph = ox.add_edge_travel_times(osm_graph)
        logger.info(
            f"OSM graph loaded: {len(osm_graph.nodes)} nodes, "
            f"{len(osm_graph.edges)} edges"
        )
    except Exception as e:
        logger.error(f"Failed to load OSM graph: {e}")
        osm_graph = None

    model = TrafficPredictorGNN(node_features=7, hidden_dim=32)
    model.load_state_dict(torch.load("traffic_gnn_weights.pth", map_location=device))
    model.to(device).eval()
    logger.info("All AI assets loaded successfully.")

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

def _run_pytorch_inference(current_state: np.ndarray) -> np.ndarray:
    """Run the PyTorch GNN model locally. Used as primary inference or NPU fallback."""
    x = torch.tensor(current_state, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(x, edge_index)
    return pred.squeeze().cpu().numpy()


@app.post("/predict_route")
async def predict_full_map(req: RouteRequest):
    loop = asyncio.get_event_loop()
    num_nodes = len(node_coords)

    # ------------------------------------------------------------------
    # 1. Build 7-channel node feature tensor
    # ------------------------------------------------------------------
    current_state = np.ones((num_nodes, 7), dtype=np.float32)

    # FIX: Use actual wall-clock time instead of hard-coded 5 PM rush hour.
    now = datetime.now()
    time_mins = now.hour * 60 + now.minute
    current_state[:, 4] = np.sin(2 * np.pi * time_mins / 1440.0)
    current_state[:, 5] = np.cos(2 * np.pi * time_mins / 1440.0)
    current_state[:, 6] = float(now.weekday() >= 5)   # 1.0 = weekend

    # Inject user-drawn congestion points (force 4-step history to gridlock)
    bottleneck_indices = []
    for point in req.custom_traffic:
        _, node_idx = kdtree.query([point.lat, point.lng])
        node_idx = int(node_idx)
        current_state[node_idx, 0:4] = 0.1   # severe gridlock speed ratio
        bottleneck_indices.append(node_idx)

    # ------------------------------------------------------------------
    # 2. Hot-patch the NPU hardware input MIF
    # ------------------------------------------------------------------
    sim_done_path  = "sim_done"
    out_file_path  = "out_file"
    local_out_file = "out_file"

    for stale in (sim_done_path, out_file_path):
        if os.path.exists(stale):
            os.remove(stale)

    patch_npu_mif(
        current_state,
        edge_index.cpu().numpy(),
        template_path="input_template.mif",
        output_path="input.mif"
    )
    logger.info("Patched input.mif — pushing to remote NPU server...")

    # ------------------------------------------------------------------
    # 3. Transfer MIF to remote server (non-blocking, runs in thread)
    # ------------------------------------------------------------------
    npu_available = True
    try:
        await loop.run_in_executor(None, transfer_mif_to_remote, "input.mif")
    except Exception as e:
        logger.warning(f"Remote transfer failed ({e}). Will fall back to PyTorch.")
        npu_available = False

    # ------------------------------------------------------------------
    # 4. Poll remote for sim_done (async, non-blocking SSH check)
    # ------------------------------------------------------------------
    all_ratios: np.ndarray

    if npu_available:
        remote_host = f"{REMOTE_USER}@{REMOTE_HOST}"
        remote_sim_done_path = f"{REMOTE_BASE}/sim_done"
        remote_out_file = f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_BASE}/out_file"

        timeout_seconds = 120.0
        elapsed = 0.0
        simulation_finished = False

        while elapsed < timeout_seconds:
            # Run the SSH test in a thread so the event loop stays free
            check_cmd = [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                remote_host, "test", "-f", remote_sim_done_path
            ]
            result = await loop.run_in_executor(
                None, lambda: subprocess.run(check_cmd, capture_output=True)
            )
            if result.returncode == 0:
                simulation_finished = True
                break
            await asyncio.sleep(0.5)
            elapsed += 0.5

        if not simulation_finished:
            logger.warning("NPU simulation timed out. Falling back to PyTorch model.")
            npu_available = False
        else:
            # ----------------------------------------------------------
            # 5a. Fetch and decode hardware output
            # ----------------------------------------------------------
            logger.info("sim_done detected — fetching output from remote server...")
            try:
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        ["scp", "-o", "BatchMode=yes", remote_out_file, local_out_file],
                        check=True, timeout=30
                    )
                )
            except Exception as e:
                logger.warning(f"Failed to fetch NPU output ({e}). Falling back to PyTorch.")
                npu_available = False

    if npu_available:
        npu_preds = parse_raw_hex_output(local_out_file, num_nodes=num_nodes)

        # ------------------------------------------------------------------
        # 5b. Validate NPU output — fall back to PyTorch if degenerate
        #
        # "Always same output" root-causes:
        #   • Remote simulation may not read the patched input.mif
        #   • INT8 quantisation drift or MRF scale mismatch
        #   • Stride/lane mis-alignment in parse_raw_hex_output
        #
        # We detect degeneracy via standard deviation threshold and fall back
        # to the locally-loaded PyTorch model which always gives correct,
        # differentiated predictions.
        # ------------------------------------------------------------------
        output_std = float(np.std(npu_preds))
        output_valid = (
            np.isfinite(npu_preds).all()
            and output_std > 1e-4          # predictions must vary across nodes
            and npu_preds.min() >= 0.0
            and npu_preds.max() <= 1.0
        )

        if output_valid:
            logger.info(f"NPU output accepted  (std={output_std:.4f})")
            all_ratios = npu_preds
        else:
            logger.warning(
                f"NPU output rejected (std={output_std:.6f}, "
                f"range=[{npu_preds.min():.3f},{npu_preds.max():.3f}]). "
                "Falling back to PyTorch model."
            )
            npu_available = False

    if not npu_available:
        # ------------------------------------------------------------------
        # 5c. PyTorch fallback — always correct, locally executed
        # ------------------------------------------------------------------
        logger.info("Running PyTorch GNN inference (fallback)...")
        all_ratios = await loop.run_in_executor(None, _run_pytorch_inference, current_state)

    # ------------------------------------------------------------------
    # 6. Re-pin user congestion nodes (guarantee bottlenecks are honoured)
    # FIX: pred_ratio is a 1-D array; indexing with [idx, 0] raises IndexError.
    # ------------------------------------------------------------------
    for idx in bottleneck_indices:
        all_ratios[idx] = 0.1

    # ------------------------------------------------------------------
    # 7. Compute OSM routes
    # ------------------------------------------------------------------
    # Haversine straight-line estimate used as a fallback floor so the UI
    # never shows "0 Minutes" even when graph routing is unavailable.
    routing_status = "ok"
    direct_dist_m  = 0.0
    if req.start_pt and req.end_pt:
        direct_dist_m = haversine_m(
            req.start_pt.lat, req.start_pt.lng,
            req.end_pt.lat,   req.end_pt.lng
        )
    # Assume average city speed of 25 km/h for the straight-line fallback
    fallback_min = max(1, int(direct_dist_m / (25_000 / 3600) / 60))

    travel_data = {
        "standard_time_min": fallback_min,
        "ai_time_min":       fallback_min,
        "std_route":         [],
        "ai_route":          [],
    }

    if req.start_pt and req.end_pt and osm_graph is not None:
        try:
            orig = _osmnx_nearest_nodes(osm_graph, req.start_pt.lng, req.start_pt.lat)
            dest = _osmnx_nearest_nodes(osm_graph, req.end_pt.lng,   req.end_pt.lat)

            if orig == dest:
                raise ValueError("Origin and destination map to the same OSM node — "
                                 "try placing markers further apart.")

            apply_osm_ai_times(osm_graph, kdtree, all_ratios)

            std_path = nx.shortest_path(osm_graph, orig, dest, weight="std_time_sec")
            ai_path  = nx.shortest_path(osm_graph, orig, dest, weight="ai_time_sec")

            travel_data["std_route"], _          = get_osm_path_data(osm_graph, std_path, "std_time_sec")
            travel_data["ai_route"],  ai_sec     = get_osm_path_data(osm_graph, ai_path,  "ai_time_sec")
            _,                        std_ai_sec = get_osm_path_data(osm_graph, std_path, "ai_time_sec")

            # BUG FIX: max(1, ...) was inside the try block, so it never ran
            # when an exception was raised.  It is now always applied below.
            travel_data["standard_time_min"] = int(std_ai_sec // 60)
            travel_data["ai_time_min"]       = int(ai_sec     // 60)
            logger.info(
                f"Routes computed: std={travel_data['standard_time_min']} min, "
                f"ai={travel_data['ai_time_min']} min  "
                f"(nodes: {orig}→{dest}, path len std={len(std_path)}, ai={len(ai_path)})"
            )
        except Exception as e:
            routing_status = f"routing_error: {e}"
            logger.error(f"OSM routing failed: {e}")
            # travel_data retains the haversine fallback values set above
    elif osm_graph is None:
        routing_status = "osm_graph_unavailable"
        logger.error("OSM graph not loaded — using haversine fallback for time estimate.")

    # Apply floor AFTER the try/except so it always executes regardless of routing outcome
    travel_data["standard_time_min"] = max(1, travel_data["standard_time_min"])
    travel_data["ai_time_min"]       = max(1, travel_data["ai_time_min"])

    # Invert speed ratio for Deck.gl heatmap (low speed → high congestion weight)
    results = [
        {
            "lat":        float(node_coords[i][0]),
            "lng":        float(node_coords[i][1]),
            "congestion": round(1.0 - float(np.clip(all_ratios[i], 0.1, 1.0)), 3)
        }
        for i in range(num_nodes)
    ]
    inference_source = "npu" if npu_available else "pytorch_fallback"
    return {
        "predictions":      results,
        "travel":           travel_data,
        "status":           "success",
        "inference_source": inference_source,
        "routing_status":   routing_status,
    }
