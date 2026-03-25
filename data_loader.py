import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

def create_edge_index_from_coords(coords_df, k=8):
    print("Building road network using K-Nearest Neighbors...")
    coords_array = coords_df[['latitude', 'longitude']].values
    num_nodes = len(coords_array)
    k_query = min(k + 1, num_nodes)

    tree = KDTree(coords_array)
    distances, indices = tree.query(coords_array, k=k_query)

    edge_index_list = []
    edge_weights_list = []

    for i in range(num_nodes):
        for j in range(1, k_query):
            neighbor_idx = int(indices[i, j])
            dist = float(distances[i, j])
            edge_index_list.append([i, neighbor_idx])
            edge_weights_list.append(1.0 / (dist + 1e-5))

    edge_index = np.array(edge_index_list, dtype=np.int64).T
    edge_weights = np.array(edge_weights_list, dtype=np.float64)

    # Undirected message passing
    rev = edge_index[[1, 0], :]
    edge_index = np.concatenate([edge_index, rev], axis=1)
    edge_weights = np.concatenate([edge_weights, edge_weights])

    return edge_index, edge_weights

def load_toronto_traffic_data():
    print("Loading Real Speed Data (This might take a moment)...")
    
    # NEW: Read the speed dataset
    df = pd.read_csv("svc_raw_data_speed_2020_2024.csv")
    
    # Downtown Bounding Box Filter
    LAT_MIN, LAT_MAX = 43.640, 43.675  
    LNG_MIN, LNG_MAX = -79.410, -79.355 
    
    df = df[(df['latitude'] >= LAT_MIN) & (df['latitude'] <= LAT_MAX) & 
            (df['longitude'] >= LNG_MIN) & (df['longitude'] <= LNG_MAX)].copy()
    
    print(f"Filtered for Downtown Core. Remaining records: {len(df)}")

    # Define the midpoint speeds for each bin (in km/h)
    speed_bins = {
        'vol_1_19kph': 10.0,
        'vol_20_25kph': 22.5,
        'vol_26_30kph': 28.0,
        'vol_31_35kph': 33.0,
        'vol_36_40kph': 38.0,
        'vol_41_45kph': 43.0,
        'vol_46_50kph': 48.0,
        'vol_51_55kph': 53.0,
        'vol_56_60kph': 58.0,
        'vol_61_65kph': 63.0,
        'vol_66_70kph': 68.0,
        'vol_71_75kph': 73.0,
        'vol_76_80kph': 78.0,
        'vol_81_160kph': 90.0 # Cap outliers to 90 for averages
    }

    # 1. Calculate Total Volume
    df['total_volume'] = df[list(speed_bins.keys())].sum(axis=1)
    
    # 2. Calculate Weighted Average Speed
    # (vol * speed) + (vol * speed) / total_vol
    weighted_sum = sum(df[col] * speed for col, speed in speed_bins.items())
    df['avg_speed_kmh'] = np.where(df['total_volume'] > 0, weighted_sum / df['total_volume'], 40.0)

    # 3. Calculate Congestion Index (0.0 = Free Flow, 1.0 = Gridlock)
    # Assuming baseline max speed is 40km/h
    df['congestion_index'] = 1.0 - (df['avg_speed_kmh'] / 40.0)
    df['congestion_index'] = df['congestion_index'].clip(lower=0.0, upper=1.0)

    # Extract unique nodes
    nodes_df = df.drop_duplicates(subset=['centreline_id'])[['centreline_id', 'latitude', 'longitude']]
    nodes_df = nodes_df.sort_values('centreline_id').reset_index(drop=True)
    node_ids = nodes_df['centreline_id'].tolist()
    print(f"Found {len(node_ids)} unique downtown speed intersections.")

    edge_index, edge_weights = create_edge_index_from_coords(nodes_df, k=8)

    # Pivot into Time-Series using the new Congestion Index
    print("Pivoting data into time-series format...")
    grouped = df.groupby(['time_start', 'centreline_id'])['congestion_index'].mean().reset_index()
    pivot_df = grouped.pivot(index='time_start', columns='centreline_id', values='congestion_index')
    pivot_df = pivot_df.reindex(columns=node_ids, fill_value=0.0)
    pivot_df = pivot_df.fillna(0.0) # 0.0 means free flowing traffic
    
    raw_congestion = pivot_df.values
    num_time_steps = raw_congestion.shape[0]
    num_nodes = raw_congestion.shape[1]
    
    features = raw_congestion.reshape((num_time_steps, num_nodes, 1))
    targets = np.zeros((num_time_steps, num_nodes))
    targets[:-1, :] = raw_congestion[1:, :] 
    
    features = features[:-1]
    targets = targets[:-1]

    dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_weights,
        features=features,
        targets=targets
    )
    
    return dataset

if __name__ == "__main__":
    dataset = load_toronto_traffic_data()
    print("Real Speed Data loaded and normalized successfully!")