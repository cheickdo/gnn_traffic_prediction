import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class TrafficPredictorGNN(torch.nn.Module):
    def __init__(self, node_features: int, hidden_dim: int):
        super(TrafficPredictorGNN, self).__init__()
        
        # The first layer now ingests 7 features (4 history + 3 metadata)
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight=None):
        h1 = F.elu(self.conv1(x, edge_index, edge_weight))
        h2 = F.elu(self.conv2(h1, edge_index, edge_weight)) + h1 
        h3 = F.elu(self.conv3(h2, edge_index, edge_weight)) + h2
        
        out = self.linear(h3)
        
        # Predicts the Speed Ratio (0.0 to 1.0)
        return torch.sigmoid(out)