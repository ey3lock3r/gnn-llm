import torch
from aptp_gnn import APTPGNN_V7

def test_v7_convergence():
    print("Starting v7.0 Convergence Test...")
    d_model = 256
    depth = 4
    model = APTPGNN_V7(depth=depth, d_model=d_model)
    
    # Ensure no-grad setup
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"FAILED: Parameter {name} has requires_grad=True")
            return
            
    # Synthetic Task: X -> Target
    x = torch.randn(8, d_model)
    target = torch.randn(8, d_model)
    
    initial_loss = model.train_step(x, target, lr=1e-2)
    print(f"Initial Loss: {initial_loss.item():.6f}")
    
    for i in range(101):
        loss = model.train_step(x, target, lr=5e-2)
        if i % 20 == 0:
            print(f"Step {i}, Loss: {loss.item():.6f}")
            
    final_loss = loss
    if final_loss < initial_loss:
        print(f"SUCCESS: Loss decreased from {initial_loss.item():.4f} to {final_loss.item():.4f}")
    else:
        print(f"FAILURE: Loss did not decrease. {initial_loss.item():.4f} -> {final_loss.item():.4f}")

if __name__ == "__main__":
    test_v7_convergence()
