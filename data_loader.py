import pandas as pd
import numpy as np
from scipy.spatial import distance
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

def create_edge_index_from_coords(coords_df, distance_threshold=0.02):
    """
    Creates an adjacency matrix (edges) by connecting intersections 
    that are geographically close to each other.
    """
    print("Building road network graph from coordinates...")
    coords_array = coords_df[['latitude', 'longitude']].values
    
    # Calculate Euclidean distance between all pairs of intersections
    dist_matrix = distance.cdist(coords_array, coords_array, 'euclidean')
    
    edge_index_list = []
    edge_weights_list = []
    
    num_nodes = len(coords_array)
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:  # Don't connect a node to itself
                dist = dist_matrix[i, j]
                if dist < distance_threshold:
                    # Create a directional edge from node i to node j
                    edge_index_list.append([i, j])
                    # Weight is inversely proportional to distance (closer = stronger connection)
                    edge_weights_list.append(1.0 / (dist + 1e-5))
                    
    edge_index = np.array(edge_index_list).T # PyG expects shape (2, num_edges)
    edge_weights = np.array(edge_weights_list)
    
    return edge_index, edge_weights

def load_toronto_traffic_data():
    print("Loading raw CSV files (This might take a moment)...")
    
    # 1. Load the volume datasets
    file1 = "svc_raw_data_volume_2015_2019.csv"
    file2 = "svc_raw_data_volume_2020_2024.csv"
    
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    
    # Combine the datasets into one large DataFrame
    df = pd.concat([df1, df2], ignore_index=True)
    
    print(f"Loaded {len(df)} total traffic records.")

    # 2. Extract unique nodes (Intersections) and their coordinates
    nodes_df = df.drop_duplicates(subset=['centreline_id'])[['centreline_id', 'latitude', 'longitude']]
    nodes_df = nodes_df.sort_values('centreline_id').reset_index(drop=True)
    node_ids = nodes_df['centreline_id'].tolist()
    print(f"Found {len(node_ids)} unique intersections.")

    # 3. Build the Graph Edges based on proximity
    edge_index, edge_weights = create_edge_index_from_coords(nodes_df, distance_threshold=0.03)
    print(f"Created {edge_index.shape[1]} edges between intersections.")

    # 4. Process Temporal Features (The Traffic Volumes)
    print("Pivoting data into time-series format...")
    # Group by time step and intersection, summing the traffic volume
    grouped = df.groupby(['time_start', 'centreline_id'])['volume_15min'].sum().reset_index()
    
    # Pivot so Rows = Time Steps, Columns = Intersections
    pivot_df = grouped.pivot(index='time_start', columns='centreline_id', values='volume_15min')
    
    # Ensure all nodes exist in the pivot table, even if they had no traffic at a specific time
    pivot_df = pivot_df.reindex(columns=node_ids, fill_value=0)
    
    # Fill missing time intervals with 0 (No traffic recorded)
    pivot_df = pivot_df.fillna(0)
    
    print(f"Processed {len(pivot_df)} unique 15-minute time steps.")

    # 5. Format into NumPy arrays for the GNN
    # Features shape required: (num_time_steps, num_nodes, num_features)
    raw_volumes = pivot_df.values
    num_time_steps = raw_volumes.shape[0]
    num_nodes = raw_volumes.shape[1]
    num_features = 1 # We are only tracking 'volume_15min'
    
    features = raw_volumes.reshape((num_time_steps, num_nodes, num_features))
    
    # Target shape required: (num_time_steps, num_nodes)
    # The target is to predict the traffic volume of the *next* time step
    targets = np.zeros((num_time_steps, num_nodes))
    targets[:-1, :] = raw_volumes[1:, :] # Shift everything up by 1 time step
    
    # Note: The very last time step won't have a "next" target, so we drop it
    features = features[:-1]
    targets = targets[:-1]

    # Normalize the data (Neural networks train much better on scaled numbers 0 to 1)
    max_vol = np.max(features)
    if max_vol > 0:
        features = features / max_vol
        targets = targets / max_vol

    # 6. Wrap it in the PyG Temporal Iterator
    print("Packaging into StaticGraphTemporalSignal...")
    dataset = StaticGraphTemporalSignal(
        edge_index=edge_index,
        edge_weight=edge_weights,
        features=features,
        targets=targets
    )
    
    return dataset

# For testing this script directly:
if __name__ == "__main__":
    dataset = load_toronto_traffic_data()
    print("Data loaded successfully!")
