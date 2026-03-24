import torch
import time
from model import TrafficPredictorGNN
from data_loader import load_toronto_traffic_data

# --- 1. Hyperparameters ---
LEARNING_RATE = 0.01
EPOCHS = 50 
HIDDEN_DIM = 32
NODE_FEATURES = 1 

# --- 2. Hardware Acceleration ---
# Automatically detects your NVIDIA GPU (CUDA)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Hardware Accelerator enabled: {device}")

# --- 3. Initialize Model, Optimizer, and Loss Function ---
model = TrafficPredictorGNN(node_features=NODE_FEATURES, hidden_dim=HIDDEN_DIM)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss() 

# --- 4. Training Loop ---
def train_model(dataset):
    model.train()
    start_time = time.time()
    
    for epoch in range(EPOCHS):
        total_loss = 0
        step = 0
        
        for time_step in dataset:
            optimizer.zero_grad()
            
            # Move data to the GPU for this specific time step
            x = time_step.x.to(device)
            edge_index = time_step.edge_index.to(device)
            edge_attr = time_step.edge_attr.to(device) if time_step.edge_attr is not None else None
            y = time_step.y.to(device)
            
            # Forward pass
            y_pred = model(x, edge_index, edge_attr)
            
            # Flatten the tensors for easier masking
            y_pred_flat = y_pred.squeeze()
            y_true_flat = y.squeeze()
            
            # Create a mask that is True only where actual traffic volume was recorded (> 0)
            mask = y_true_flat > 0
            
            # Only calculate loss and update weights if there is actual data in this time step
            if mask.sum() > 0: 
                loss = loss_fn(y_pred_flat[mask], y_true_flat[mask])
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                step += 1
            
        # Protect against dividing by zero if an epoch had no data at all
        avg_loss = total_loss / step if step > 0 else 0 
        
        # Print progress every 5 epochs
        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:02d} | Average Masked MSE Loss: {avg_loss:.4f} | Time: {elapsed:.1f}s")
            
    print("\nTraining Complete!")
    
    # Save the trained weights to a file so FastAPI can load them later
    torch.save(model.state_dict(), "traffic_gnn_weights.pth")
    print("Saved model weights to 'traffic_gnn_weights.pth'")

# --- 5. Execution ---
if __name__ == "__main__":
    print("Loading Toronto Open Data...")
    toronto_dataset = load_toronto_traffic_data()
    
    print("Starting training...")
    train_model(toronto_dataset)
