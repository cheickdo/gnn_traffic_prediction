import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN

class TrafficPredictorGNN(torch.nn.Module):
    def __init__(self, node_features: int, hidden_dim: int):
        super(TrafficPredictorGNN, self).__init__()
        
        # Spatio-Temporal Layer: TGCN expects (num_nodes, in_channels)
        # in_channels is now 1 (the current 15-min volume)
        self.recurrent_layer = TGCN(in_channels=node_features, out_channels=hidden_dim)
        
        # Linear Layer: Maps the 32 hidden dimensions down to 1 prediction
        self.linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight=None):
        """
        x: Node features. Shape: (num_nodes, 1)
        edge_index: Graph connectivity. Shape: (2, num_edges)
        """
        h = self.recurrent_layer(x, edge_index, edge_weight)
        h = F.relu(h)
        
        prediction = self.linear(h)
        return prediction
