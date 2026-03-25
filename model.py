import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

class TrafficPredictorGNN(torch.nn.Module):
    def __init__(self, node_features: int, hidden_dim: int):
        super(TrafficPredictorGNN, self).__init__()
        
        # 'max' aggregation forces gridlock to aggressively bleed into neighboring streets
        self.conv1 = SAGEConv(node_features, hidden_dim, aggr='max')
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr='max')
        self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr='max')
        
        self.linear = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_weight=None):
        h1 = F.elu(self.conv1(x, edge_index))
        h2 = F.elu(self.conv2(h1, edge_index)) + h1 
        h3 = F.elu(self.conv3(h2, edge_index)) + h2
        
        out = self.linear(h3)
        
        # NEW: Sigmoid mathematically bounds the output between 0.0 and 1.0.
        # 0.0 = Free Flowing Speed. 1.0 = Gridlock Penalty.
        return torch.sigmoid(out)