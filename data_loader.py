import pandas as pd
import numpy as np
from scipy.spatial import KDTree
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

def create_directed_edge_index(coords_df, k=8):
    print("Building Asymmetric (Directed) road network...")
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

    # INDUSTRY UPGRADE 3: ASYMMETRIC ADJACENCY
    # We purposefully DO NOT add reverse edges. Traffic flows directionally.
    return edge_index, edge_weights

def load_toronto_traffic_data():
    print("Loading Spatio-Temporal Speed Data...")
    df = pd.read_csv("svc_raw_data_speed_2020_2024.csv")
    
    LAT_MIN, LAT_MAX = 43.643, 43.658  
    LNG_MIN, LNG_MAX = -79.395, -79.370 
    
    df = df[(df['latitude'] >= LAT_MIN) & (df['latitude'] <= LAT_MAX) & 
            (df['longitude'] >= LNG_MIN) & (df['longitude'] <= LNG_MAX)].copy()

    speed_bins = {
        'vol_1_19kph': 10.0, 'vol_20_25kph': 22.5, 'vol_26_30kph': 28.0,
        'vol_31_35kph': 33.0, 'vol_36_40kph': 38.0, 'vol_41_45kph': 43.0,
        'vol_46_50kph': 48.0, 'vol_51_55kph': 53.0, 'vol_56_60kph': 58.0,
        'vol_61_65kph': 63.0, 'vol_66_70kph': 68.0, 'vol_71_75kph': 73.0,
        'vol_76_80kph': 78.0, 'vol_81_160kph': 90.0
    }

    df['total_volume'] = df[list(speed_bins.keys())].sum(axis=1)
    weighted_sum = sum(df[col] * speed for col, speed in speed_bins.items())
    df['avg_speed_kmh'] = np.where(df['total_volume'] > 0, weighted_sum / df['total_volume'], 40.0)

    # INDUSTRY UPGRADE 4: TARGET NORMALIZATION (SPEED RATIO)
    # 1.0 = Free Flowing, 0.1 = Gridlock
    df['speed_ratio'] = df['avg_speed_kmh'] / 40.0
    df['speed_ratio'] = df['speed_ratio'].clip(lower=0.1, upper=1.0)

    # INDUSTRY UPGRADE 2: CONTEXTUAL METADATA
    df['time_start'] = pd.to_datetime(df['time_start'])
    df['hour'] = df['time_start'].dt.hour
    df['minute'] = df['time_start'].dt.minute
    time_in_mins = df['hour'] * 60 + df['minute']
    
    df['sin_time'] = np.sin(2 * np.pi * time_in_mins / 1440.0)
    df['cos_time'] = np.cos(2 * np.pi * time_in_mins / 1440.0)
    df['is_weekend'] = df['time_start'].dt.dayofweek.isin([5, 6]).astype(float)

    nodes_df = df.drop_duplicates(subset=['centreline_id']).sort_values('centreline_id').reset_index(drop=True)
    node_ids = nodes_df['centreline_id'].tolist()
    num_nodes = len(node_ids)

    edge_index, edge_weights = create_directed_edge_index(nodes_df, k=8)

    # Prepare Time-Series Data
    time_steps = sorted(df['time_start'].unique())
    num_steps = len(time_steps)
    
    speed_matrix = np.ones((num_steps, num_nodes))
    sin_matrix = np.zeros(num_steps)
    cos_matrix = np.zeros(num_steps)
    wknd_matrix = np.zeros(num_steps)

    grouped = df.groupby(['time_start', 'centreline_id']).first().reset_index()
    for t_idx, t_val in enumerate(time_steps):
        t_data = grouped[grouped['time_start'] == t_val]
        sin_matrix[t_idx] = t_data['sin_time'].iloc[0] if len(t_data) > 0 else 0
        cos_matrix[t_idx] = t_data['cos_time'].iloc[0] if len(t_data) > 0 else 0
        wknd_matrix[t_idx] = t_data['is_weekend'].iloc[0] if len(t_data) > 0 else 0
        
        for _, row in t_data.iterrows():
            n_idx = node_ids.index(row['centreline_id'])
            speed_matrix[t_idx, n_idx] = row['speed_ratio']

    # INDUSTRY UPGRADE 1: FEATURE CHANNEL STACKING (Sliding Window)
    WINDOW = 4 # 1 Hour of history (4 x 15min)
    X, Y = [], []
    
    for i in range(num_steps - WINDOW):
        # Channels 0-3: Historical Speeds
        hist = speed_matrix[i : i+WINDOW, :].T # Shape: (nodes, 4)
        
        # Channels 4-6: Metadata for the target time step
        meta = np.zeros((num_nodes, 3))
        meta[:, 0] = sin_matrix[i + WINDOW - 1]
        meta[:, 1] = cos_matrix[i + WINDOW - 1]
        meta[:, 2] = wknd_matrix[i + WINDOW - 1]
        
        # Stack into 7-Channel Input Tensor
        x_step = np.concatenate([hist, meta], axis=1) # Shape: (nodes, 7)
        y_step = speed_matrix[i + WINDOW, :] # Target state
        
        X.append(x_step)
        Y.append(y_step)

    dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_weights,
        features=np.array(X, dtype=np.float32),
        targets=np.array(Y, dtype=np.float32)
    )
    
    return dataset

if __name__ == "__main__":
    dataset = load_toronto_traffic_data()
    
    edge_index = dataset.edge_index 
    
    print(f"src_ids = {edge_index[0].tolist()}")
    print(f"dst_ids = {edge_index[1].tolist()}")