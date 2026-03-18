import torch
import time
from model import TrafficPredictorGNN
from data_loader import load_toronto_traffic_data

# --- 1. Hyperparameters ---
LEARNING_RATE = 0.01
EPOCHS = 50 
HIDDEN_DIM = 32
NODE_FEATURES = 1 

# SPEEDUP 1: Hardware Device Selection
# This automatically checks if you have an Apple Silicon GPU (MPS), 
# an Nvidia GPU (CUDA), or defaults to standard CPU.
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Hardware Accelerator enabled: {device}")

# --- 2. Initialize Model, Optimizer, and Loss Function ---
model = TrafficPredictorGNN(node_features=NODE_FEATURES, hidden_dim=HIDDEN_DIM)

# SPEEDUP 2: Move the model to the GPU
model = model.to(device)

# SPEEDUP 3: PyTorch 2.0 Compilation (Optional but highly recommended)
# This analyzes your GNN and fuses operations together to make them run faster.
# If you get an error with this line on your specific setup, just comment it out.
try:
    model = torch.compile(model)
    print("PyTorch model compilation successful.")
except Exception as e:
    print("Skipping torch.compile (requires PyTorch 2.0+).")

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss() 

def train_model(dataset):
    model.train()
    
    # We will track the time it takes to complete epochs
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        step = 0
        
        for time_step in dataset:
            optimizer.zero_grad()
            
            # SPEEDUP 4: Move the data to the GPU for this specific time step
            # Matrix math is much faster when the data and model share the same VRAM
            x = time_step.x.to(device)
            edge_index = time_step.edge_index.to(device)
            edge_attr = time_step.edge_attr.to(device) if time_step.edge_attr is not None else None
            y = time_step.y.to(device)
            
            # Forward pass
            y_pred = model(x, edge_index, edge_attr)
            
            # Calculate loss 
            loss = loss_fn(y_pred.squeeze(), y.squeeze())
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            step += 1
            
        avg_loss = total_loss / step
        
        # Print progress every 5 epochs to reduce console clutter (which also slows down training!)
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:02d} | Average MSE Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
            
    print("\nTraining Complete!")
    torch.save(model.state_dict(), "traffic_gnn_weights.pth")
    print("Saved model weights to 'traffic_gnn_weights.pth'")

if __name__ == "__main__":
    print("Loading Toronto Open Data...")
    toronto_dataset = load_toronto_traffic_data()
    
    print("Starting training...")
    train_model(toronto_dataset)
