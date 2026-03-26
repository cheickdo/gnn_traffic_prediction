"""
main.py — FastAPI backend for the Toronto NPU Traffic Prediction System.

End-to-end pipeline:
  1. Build 7-channel node feature tensor from user-drawn congestion
  2. Hot-patch NPU hardware input MIF with BFloat16 features
  3. Transfer MIF to remote Stratix 10 NX FPGA via SCP
  4. Poll for RTL simulation completion (sim_done sentinel)
  5. Fetch and decode hardware output (640-bit BFloat16 words)
  6. Apply predictions to OSM road network for route planning
  7. Return heatmap, routes, and pipeline telemetry to frontend
"""

import asyncio
import os
import struct
import subprocess
import time
import logging
from datetime import datetime
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import torch
import numpy as np
from scipy.spatial import KDTree
import pandas as pd
import networkx as nx

from model import TrafficPredictorGNN
from data_loader import load_toronto_traffic_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="NPU Traffic Prediction Engine", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error_response(detail: str, status_code: int = 200) -> JSONResponse:
    """Build a JSON error response with explicit CORS headers."""
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "detail": detail,
            "predictions": [],
            "travel": {
                "standard_time_min": 0,
                "ai_time_min": 0,
                "std_route": [],
                "ai_route": [],
            },
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
        },
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch ALL unhandled exceptions and return JSON instead of 500 HTML."""
    logger.exception(f"Unhandled exception on {request.url.path}: {exc}")
    return _error_response(str(exc))


from fastapi.exceptions import RequestValidationError


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return 200 JSON for pydantic validation errors so the frontend can parse them."""
    detail = "; ".join(
        f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()
    )
    logger.warning(f"Validation error on {request.url.path}: {detail}")
    return _error_response(f"Invalid request: {detail}")

# ──────────────────────────────────────────────────────────────────────────────
# Global state
# ──────────────────────────────────────────────────────────────────────────────
model, edge_index, node_coords, kdtree, nx_graph, osm_graph = (
    None, None, None, None, None, None,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_SPEED_KMH = 40.0
BASE_SPEED_MS = DEFAULT_SPEED_KMH * 1000.0 / 3600.0
OSM_BBOX = (-79.395, 43.643, -79.370, 43.658)

# NPU hardware constants (Intel Stratix 10 NX)
SIM_BATCH = 3
DOTW = 40      # 40 vector lanes × 16-bit BFloat16 = 640-bit words
NCORE = 3      # 3 DSP cores
NPU_WORD_BITS = 640

REMOTE_USER = "doumbiac"
REMOTE_HOST = "betzgrp-wintermute.eecg.utoronto.ca"
REMOTE_BASE = "~/Documents/npu_gnn_bringup/rtl"


# ──────────────────────────────────────────────────────────────────────────────
# BFloat16 Conversion Utilities
# ──────────────────────────────────────────────────────────────────────────────

def float_to_bf16_hex(f: float) -> str:
    """Convert IEEE-754 float32 to 4-char BFloat16 hex (top 16 bits)."""
    u32 = struct.unpack(">I", struct.pack(">f", f))[0]
    return f"{u32 >> 16:04x}"


def hex_to_bfloat16_float(hex_str: str) -> float:
    """Convert 4-char BFloat16 hex back to float32."""
    try:
        padded_hex = hex_str.strip().zfill(4) + "0000"
        val = struct.unpack(">f", struct.pack(">I", int(padded_hex, 16)))[0]
        return val if np.isfinite(val) else 0.0
    except Exception:
        return 0.0


# ──────────────────────────────────────────────────────────────────────────────
# NPU Hardware Output Parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_raw_hex_output(filepath: str, num_nodes: int) -> np.ndarray:
    """Parse out_file produced by npu_tb.sv → per-node speed ratios.

    ── Testbench output format ─────────────────────────────────────────
    The $fwrite sits inside both a lane loop (i=0..DOTW-1) and a core
    loop (j=0..NCORE-1), so every 640-bit output word is repeated
    DOTW × NCORE = 120 times as consecutive identical lines.

    ── 640-bit word layout (MSB-first hex) ─────────────────────────────
    chars 0-3   = lane 39 (MSB)    chars 156-159 = lane 0 (LSB)
    Lane 0 holds the real prediction (output dim 0 from the padded head).

    ── Sigmoid ─────────────────────────────────────────────────────────
    The RTL MFU applies sigmoid via hardware LUT → values in out_file
    are already in (0,1).  Applying sigmoid again compresses everything
    to ~0.5 ("always same output" bug).
    """
    fallback = np.ones(num_nodes, dtype=np.float32) * 0.5

    if not os.path.exists(filepath):
        logger.error(f"NPU out_file not found: {filepath}")
        return fallback

    with open(filepath, "r") as f:
        lines = [
            l.strip()
            for l in f
            if l.strip() and len(l.strip()) >= 4 and "x" not in l.lower()
        ]

    total_lines = len(lines)
    logger.info(f"NPU out_file: {total_lines} lines for {num_nodes} nodes")

    if total_lines == 0:
        logger.error("NPU out_file is empty or contains only unknown (x) bits.")
        return fallback

    stride = max(1, total_lines // num_nodes)
    logger.info(
        f"NPU stride: {stride} lines/node  "
        f"(expected DOTW×NCORE = {DOTW}×{NCORE} = {DOTW * NCORE})"
    )

    raw_predictions: list = []
    raw_hex_samples: list = []

    for i in range(num_nodes):
        line_idx = i * stride
        if line_idx < total_lines:
            line = lines[line_idx]
            lane0_hex = line[-4:]
            raw_val = hex_to_bfloat16_float(lane0_hex)
            raw_predictions.append(raw_val)
            if i < 5:
                raw_hex_samples.append(f"node{i}: 0x{lane0_hex} = {raw_val:.6f}")
        else:
            raw_predictions.append(0.5)

    raw_np = np.array(raw_predictions, dtype=np.float32)

    # Log sample decoded values for debugging
    for s in raw_hex_samples:
        logger.info(f"  BF16 sample: {s}")

    # ── Sigmoid detection ─────────────────────────────────────────────
    post_sigmoid_frac = float(np.mean((raw_np > 0.0) & (raw_np < 1.0)))

    if post_sigmoid_frac >= 0.9:
        speed_ratios = raw_np.copy()
        logger.info(
            f"NPU: hardware sigmoid detected ({post_sigmoid_frac*100:.0f}% in (0,1)) "
            "— using values directly"
        )
    else:
        speed_ratios = 1.0 / (1.0 + np.exp(-np.clip(raw_np, -50.0, 50.0)))
        logger.info(
            f"NPU: raw logits detected ({post_sigmoid_frac*100:.0f}% in (0,1)) "
            "— applying software sigmoid"
        )

    speed_ratios = np.clip(speed_ratios, 0.1, 1.0)
    logger.info(
        f"NPU speed-ratio: [{speed_ratios.min():.4f}, {speed_ratios.max():.4f}]  "
        f"std={speed_ratios.std():.4f}  mean={speed_ratios.mean():.4f}"
    )
    return speed_ratios


# ──────────────────────────────────────────────────────────────────────────────
# MIF Patching — Hot-swap node features in hardware input FIFO
# ──────────────────────────────────────────────────────────────────────────────

def patch_npu_mif(
    current_state: np.ndarray,
    edge_index_np: np.ndarray,
    template_path: str = "input_template.mif",
    output_path: str = "input.mif",
) -> int:
    """Overwrite X tensor (node features) in hardware MIF.

    MIF layout mirrors the load order in toronto_npu_model.py:
      Address 0..(num_nodes×SIM_BATCH-1): mfu0_mul_nodes (node features)
    Each node has SIM_BATCH=3 consecutive entries.
    Returns the number of patched entries.
    """
    num_nodes = current_state.shape[0]
    start_addr = 0
    entries_per_node = SIM_BATCH

    # Format each node's features as 640-bit BFloat16 hex
    node_hex_words: dict = {}
    for n_id in range(num_nodes):
        features = current_state[n_id]
        padded = np.zeros(DOTW, dtype=np.float32)
        padded[: len(features)] = features
        # Pack: lane 39 (MSB/left) → lane 0 (LSB/right)
        hex_word = "".join(float_to_bf16_hex(padded[i]) for i in range(DOTW - 1, -1, -1))
        node_hex_words[n_id] = hex_word

    with open(template_path, "r") as f:
        lines = f.readlines()

    end_addr = start_addr + num_nodes * entries_per_node
    patched = 0

    with open(output_path, "w") as f:
        for line in lines:
            if ":" in line and line.strip().endswith(";"):
                addr_str = line.split(":")[0].strip()
                if addr_str.isdigit():
                    addr = int(addr_str)
                    if start_addr <= addr < end_addr:
                        node_idx = (addr - start_addr) // entries_per_node
                        if node_idx < num_nodes:
                            f.write(f"{addr}: {node_hex_words[node_idx]};\n")
                            patched += 1
                            continue
            f.write(line)

    logger.info(
        f"MIF patched: {patched} entries (addr {start_addr}–{end_addr - 1}) "
        f"for {num_nodes} nodes"
    )
    return patched


# ──────────────────────────────────────────────────────────────────────────────
# Remote NPU Transfer
# ──────────────────────────────────────────────────────────────────────────────

def transfer_mif_to_remote(local_path: str = "input.mif") -> float:
    """SCP the patched MIF to the remote FPGA server.

    Also deletes stale sim_done and out_file on the remote FIRST so the
    testbench knows to start a fresh simulation.  Returns elapsed seconds.
    """
    remote_host = f"{REMOTE_USER}@{REMOTE_HOST}"
    remote_target = f"{remote_host}:{REMOTE_BASE}/mif_files/"
    t0 = time.perf_counter()

    # 1. Delete stale sentinel & output on the remote
    subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            remote_host,
            f"rm -f {REMOTE_BASE}/sim_done {REMOTE_BASE}/out_file",
        ],
        timeout=15,
    )

    # 2. Upload the new MIF
    subprocess.run(
        [
            "scp", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            local_path, remote_target,
        ],
        check=True,
        timeout=30,
    )
    elapsed = time.perf_counter() - t0
    logger.info(f"SCP transfer complete ({elapsed:.2f}s)")
    return elapsed


# ──────────────────────────────────────────────────────────────────────────────
# Geometry & Routing Helpers
# ──────────────────────────────────────────────────────────────────────────────

def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6_371_000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (
        np.sin(np.radians(lat2 - lat1) / 2) ** 2
        + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    )
    return float(2 * r * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def build_routing_graph(edges, coords):
    g = nx.Graph()
    for i in range(len(coords)):
        g.add_node(i)
    for k in range(edges.shape[1]):
        u, v = int(edges[0, k]), int(edges[1, k])
        if u != v:
            g.add_edge(
                u, v,
                distance=haversine_m(
                    coords[u][0], coords[u][1], coords[v][0], coords[v][1]
                ),
            )
    return g


def _osmnx_nearest_nodes(G, x: float, y: float) -> int:
    """Version-agnostic wrapper (osmnx 1.x vs 2.x)."""
    import osmnx as ox

    fn = getattr(ox, "nearest_nodes", None) or ox.distance.nearest_nodes
    return int(fn(G, x, y))


def apply_osm_ai_times(
    G: nx.MultiDiGraph, tmc_kdtree: KDTree, speed_ratios: np.ndarray
):
    """Stamp std_time_sec and ai_time_sec on every OSM edge."""
    for u, v, k, d in G.edges(keys=True, data=True):
        lat = 0.5 * (float(G.nodes[u]["y"]) + float(G.nodes[v]["y"]))
        lng = 0.5 * (float(G.nodes[u]["x"]) + float(G.nodes[v]["x"]))
        _, ti = tmc_kdtree.query([lat, lng])

        ratio = max(0.1, float(speed_ratios[int(ti)]))

        free_flow_sec = float(d.get("travel_time", 0.0))
        if free_flow_sec <= 0:
            length_m = float(d.get("length", 50.0))
            free_flow_sec = max(1.0, length_m / BASE_SPEED_MS)

        d["std_time_sec"] = free_flow_sec
        d["ai_time_sec"] = free_flow_sec / ratio


def get_osm_path_data(G, path, weight_key):
    """Extract coordinates and total time for an OSM node-id path."""
    coords: list = []
    total_time = 0.0

    for u, v in zip(path[:-1], path[1:]):
        edge_data = G[u][v]
        k = min(
            edge_data.keys(),
            key=lambda kk: float(edge_data[kk].get(weight_key, 1e30)),
        )
        total_time += float(edge_data[k].get(weight_key, 0.0))

        geom = edge_data[k].get("geometry")
        if geom:
            edge_pts = [
                {"lat": float(lat), "lng": float(lon)} for lon, lat in geom.coords
            ]
        else:
            edge_pts = [
                {"lat": float(G.nodes[u]["y"]), "lng": float(G.nodes[u]["x"])},
                {"lat": float(G.nodes[v]["y"]), "lng": float(G.nodes[v]["x"])},
            ]

        if not edge_pts:
            continue

        # Deduplicate shared junction nodes between consecutive edges
        if coords:
            first = edge_pts[0]
            last = coords[-1]
            if (
                abs(first["lat"] - last["lat"]) < 1e-9
                and abs(first["lng"] - last["lng"]) < 1e-9
            ):
                edge_pts = edge_pts[1:]

        coords.extend(edge_pts)

    return coords, total_time


# ──────────────────────────────────────────────────────────────────────────────
# PyTorch Inference (fallback)
# ──────────────────────────────────────────────────────────────────────────────

def _run_pytorch_inference(current_state: np.ndarray) -> np.ndarray:
    """Run the PyTorch GNN model locally."""
    x = torch.tensor(current_state, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(x, edge_index)
    return pred.squeeze().cpu().numpy()


# ──────────────────────────────────────────────────────────────────────────────
# Startup — Load All Assets
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_ai_assets():
    global model, edge_index, node_coords, kdtree, nx_graph, osm_graph

    logger.info("Loading AI assets...")
    t0 = time.perf_counter()

    dataset = load_toronto_traffic_data()
    edge_index = next(iter(dataset)).edge_index.to(device)

    df = pd.read_csv("svc_raw_data_speed_2020_2024.csv")
    df = df[
        (df["latitude"] >= 43.643)
        & (df["latitude"] <= 43.658)
        & (df["longitude"] >= -79.395)
        & (df["longitude"] <= -79.370)
    ]
    nodes_df = (
        df.drop_duplicates(subset=["centreline_id"])[
            ["centreline_id", "latitude", "longitude"]
        ]
        .sort_values("centreline_id")
        .reset_index(drop=True)
    )
    node_coords = nodes_df[["latitude", "longitude"]].values
    kdtree = KDTree(node_coords)
    nx_graph = build_routing_graph(edge_index.cpu().numpy(), node_coords)

    try:
        import osmnx as ox
        from shapely.geometry import box as shapely_box

        bbox_polygon = shapely_box(-79.395, 43.643, -79.370, 43.658)
        osm_graph = ox.graph_from_polygon(
            bbox_polygon, network_type="drive", simplify=True
        )
        osm_graph = ox.add_edge_speeds(osm_graph)
        osm_graph = ox.add_edge_travel_times(osm_graph)
        logger.info(
            f"OSM graph: {len(osm_graph.nodes)} nodes, {len(osm_graph.edges)} edges"
        )
    except Exception as e:
        logger.error(f"OSM graph load failed: {e}")
        osm_graph = None

    model = TrafficPredictorGNN(node_features=7, hidden_dim=32)
    model.load_state_dict(
        torch.load("traffic_gnn_weights.pth", map_location=device)
    )
    model.to(device).eval()

    elapsed = time.perf_counter() - t0
    logger.info(f"All assets loaded in {elapsed:.1f}s  ({len(node_coords)} nodes)")


# ──────────────────────────────────────────────────────────────────────────────
# Static File Serving
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")


# ──────────────────────────────────────────────────────────────────────────────
# Health / Info Endpoint
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "osm_graph_loaded": osm_graph is not None,
        "num_nodes": len(node_coords) if node_coords is not None else 0,
        "device": str(device),
        "npu_target": "Intel Stratix 10 NX",
        "npu_remote": f"{REMOTE_USER}@{REMOTE_HOST}",
        "bf16_word_width": NPU_WORD_BITS,
        "vector_lanes": DOTW,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────────────────────

class TrafficPoint(BaseModel):
    lat: float
    lng: float


class RouteRequest(BaseModel):
    custom_traffic: List[TrafficPoint]
    start_pt: Optional[TrafficPoint] = None
    end_pt: Optional[TrafficPoint] = None


# ──────────────────────────────────────────────────────────────────────────────
# Main Prediction Endpoint
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/predict_route")
async def predict_full_map(req: RouteRequest):
    loop = asyncio.get_event_loop()
    num_nodes = len(node_coords)
    pipeline_timing: Dict[str, float] = {}
    pipeline_t0 = time.perf_counter()

    if node_coords is None or model is None:
        return {"status": "error", "detail": "Server still loading — try again in a few seconds."}

    # ──────────────────────────────────────────────────────────────────
    # 1. Build 7-channel node feature tensor
    # ──────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    current_state = np.ones((num_nodes, 7), dtype=np.float32)

    now = datetime.now()
    time_mins = now.hour * 60 + now.minute
    current_state[:, 4] = np.sin(2 * np.pi * time_mins / 1440.0)
    current_state[:, 5] = np.cos(2 * np.pi * time_mins / 1440.0)
    current_state[:, 6] = float(now.weekday() >= 5)

    # Inject user-drawn congestion
    bottleneck_indices: list = []
    for point in req.custom_traffic:
        _, node_idx = kdtree.query([point.lat, point.lng])
        node_idx = int(node_idx)
        current_state[node_idx, 0:4] = 0.1
        bottleneck_indices.append(node_idx)

    pipeline_timing["feature_build_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ──────────────────────────────────────────────────────────────────
    # 2. Hot-patch NPU input MIF
    # ──────────────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    sim_done_path = "sim_done"
    out_file_path = "out_file"
    local_out_file = "out_file"

    for stale in (sim_done_path, out_file_path):
        if os.path.exists(stale):
            os.remove(stale)

    patch_npu_mif(
        current_state,
        edge_index.cpu().numpy(),
        template_path="input_template.mif",
        output_path="input.mif",
    )
    pipeline_timing["mif_patch_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    # ──────────────────────────────────────────────────────────────────
    # 3. Transfer MIF to remote FPGA (non-blocking)
    # ──────────────────────────────────────────────────────────────────
    npu_available = True
    inference_source = "npu"

    t0 = time.perf_counter()
    try:
        scp_elapsed = await loop.run_in_executor(
            None, transfer_mif_to_remote, "input.mif"
        )
        pipeline_timing["scp_transfer_ms"] = round(scp_elapsed * 1000, 2)
    except Exception as e:
        logger.warning(f"SCP failed ({e}) — falling back to PyTorch")
        pipeline_timing["scp_transfer_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        npu_available = False

    # ──────────────────────────────────────────────────────────────────
    # 4. Poll remote for simulation completion
    # ──────────────────────────────────────────────────────────────────
    all_ratios: np.ndarray

    if npu_available:
        remote_host = f"{REMOTE_USER}@{REMOTE_HOST}"
        remote_sim_done = f"{REMOTE_BASE}/sim_done"
        remote_out_file = f"{REMOTE_USER}@{REMOTE_HOST}:{REMOTE_BASE}/out_file"

        t0 = time.perf_counter()
        timeout_seconds = 120.0
        elapsed = 0.0
        simulation_finished = False

        while elapsed < timeout_seconds:
            check_cmd = [
                "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                remote_host, "test", "-f", remote_sim_done,
            ]
            result = await loop.run_in_executor(
                None, lambda: subprocess.run(check_cmd, capture_output=True)
            )
            if result.returncode == 0:
                simulation_finished = True
                break
            await asyncio.sleep(0.5)
            elapsed += 0.5

        pipeline_timing["simulation_wait_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )

        if not simulation_finished:
            logger.warning("NPU simulation timed out — falling back to PyTorch")
            npu_available = False
        else:
            # ──────────────────────────────────────────────────────────
            # 5a. Fetch hardware output
            # ──────────────────────────────────────────────────────────
            t0 = time.perf_counter()
            try:
                await loop.run_in_executor(
                    None,
                    lambda: subprocess.run(
                        [
                            "scp", "-o", "BatchMode=yes",
                            remote_out_file, local_out_file,
                        ],
                        check=True,
                        timeout=30,
                    ),
                )
                pipeline_timing["fetch_output_ms"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )
            except Exception as e:
                logger.warning(f"Failed to fetch NPU output ({e})")
                pipeline_timing["fetch_output_ms"] = round(
                    (time.perf_counter() - t0) * 1000, 2
                )
                npu_available = False

    # ──────────────────────────────────────────────────────────────────
    # 5b. Decode NPU output
    # ──────────────────────────────────────────────────────────────────
    if npu_available:
        t0 = time.perf_counter()
        npu_preds = parse_raw_hex_output(local_out_file, num_nodes=num_nodes)
        pipeline_timing["decode_bf16_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )

        if not np.isfinite(npu_preds).all():
            logger.warning("NPU output contains NaN/Inf — falling back to PyTorch")
            npu_available = False
        else:
            logger.info(
                f"NPU output accepted — "
                f"std={float(np.std(npu_preds)):.4f}  "
                f"mean={float(np.mean(npu_preds)):.4f}  "
                f"range=[{npu_preds.min():.4f}, {npu_preds.max():.4f}]"
            )
            all_ratios = npu_preds
            inference_source = "npu"

    # ──────────────────────────────────────────────────────────────────
    # 5c. PyTorch fallback (SSH/SCP failure, timeout, NaN/Inf only)
    # ──────────────────────────────────────────────────────────────────
    if not npu_available:
        t0 = time.perf_counter()
        logger.info("Running PyTorch GNN inference (fallback)...")
        all_ratios = await loop.run_in_executor(
            None, _run_pytorch_inference, current_state
        )
        pipeline_timing["pytorch_inference_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
        inference_source = "pytorch"

    # ──────────────────────────────────────────────────────────────────
    # 6. Re-pin user congestion nodes
    # ──────────────────────────────────────────────────────────────────
    for idx in bottleneck_indices:
        all_ratios[idx] = 0.1

    # ──────────────────────────────────────────────────────────────────
    # 7. Compute OSM routes
    # ──────────────────────────────────────────────────────────────────
    routing_status = "ok"
    direct_dist_m = 0.0
    if req.start_pt and req.end_pt:
        direct_dist_m = haversine_m(
            req.start_pt.lat, req.start_pt.lng,
            req.end_pt.lat, req.end_pt.lng,
        )
    fallback_min = max(1, int(direct_dist_m / (25_000 / 3600) / 60))

    travel_data = {
        "standard_time_min": fallback_min,
        "ai_time_min": fallback_min,
        "std_route": [],
        "ai_route": [],
    }

    if req.start_pt and req.end_pt and osm_graph is not None:
        t0 = time.perf_counter()
        try:
            orig = _osmnx_nearest_nodes(
                osm_graph, req.start_pt.lng, req.start_pt.lat
            )
            dest = _osmnx_nearest_nodes(
                osm_graph, req.end_pt.lng, req.end_pt.lat
            )

            if orig == dest:
                raise ValueError(
                    "Origin and destination map to the same OSM node"
                )

            apply_osm_ai_times(osm_graph, kdtree, all_ratios)

            std_path = nx.shortest_path(
                osm_graph, orig, dest, weight="std_time_sec"
            )
            ai_path = nx.shortest_path(
                osm_graph, orig, dest, weight="ai_time_sec"
            )

            travel_data["std_route"], _ = get_osm_path_data(
                osm_graph, std_path, "std_time_sec"
            )
            travel_data["ai_route"], ai_sec = get_osm_path_data(
                osm_graph, ai_path, "ai_time_sec"
            )
            _, std_ai_sec = get_osm_path_data(
                osm_graph, std_path, "ai_time_sec"
            )

            travel_data["standard_time_min"] = int(std_ai_sec // 60)
            travel_data["ai_time_min"] = int(ai_sec // 60)

            logger.info(
                f"Routes: std={travel_data['standard_time_min']}min  "
                f"ai={travel_data['ai_time_min']}min  "
                f"(path len std={len(std_path)}, ai={len(ai_path)})"
            )
        except Exception as e:
            routing_status = f"routing_error: {e}"
            logger.error(f"OSM routing failed: {e}")

        pipeline_timing["routing_ms"] = round(
            (time.perf_counter() - t0) * 1000, 2
        )
    elif osm_graph is None:
        routing_status = "osm_graph_unavailable"

    # Always enforce minimum 1 minute
    travel_data["standard_time_min"] = max(1, travel_data["standard_time_min"])
    travel_data["ai_time_min"] = max(1, travel_data["ai_time_min"])

    # ──────────────────────────────────────────────────────────────────
    # 8. Build response
    # ──────────────────────────────────────────────────────────────────
    pipeline_timing["total_ms"] = round(
        (time.perf_counter() - pipeline_t0) * 1000, 2
    )

    # Invert speed ratio for heatmap (low speed → high congestion)
    predictions = [
        {
            "lat": float(node_coords[i][0]),
            "lng": float(node_coords[i][1]),
            "congestion": round(
                1.0 - float(np.clip(all_ratios[i], 0.1, 1.0)), 3
            ),
        }
        for i in range(num_nodes)
    ]

    # Compute time savings percentage
    std_min = travel_data["standard_time_min"]
    ai_min = travel_data["ai_time_min"]
    savings_pct = (
        round((1.0 - ai_min / std_min) * 100, 1) if std_min > 0 else 0.0
    )

    return {
        "predictions": predictions,
        "travel": travel_data,
        "status": "success",
        "inference_source": inference_source,
        "routing_status": routing_status,
        "time_savings_pct": savings_pct,
        "pipeline": pipeline_timing,
        "hardware": {
            "target": "Intel Stratix 10 NX",
            "word_width": NPU_WORD_BITS,
            "precision": "BFloat16",
            "vector_lanes": DOTW,
            "cores": NCORE,
            "num_nodes": num_nodes,
            "num_congestion_points": len(bottleneck_indices),
        },
    }
