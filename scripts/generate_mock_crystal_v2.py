import torch
import os

def generate_multi_layer_crystal():
    print("Generating Phase XXXI Multi-Layer Brain Crystal...")
    
    vocab_size = 50257
    dim_l1 = 3000   # 感知层
    dim_l2 = 1024   # 逻辑层
    
    # L1: 基础感受器与词表关联
    W_L1 = torch.randn(vocab_size, dim_l1) * 0.01
    P_L1 = torch.eye(dim_l1) + torch.randn(dim_l1, dim_l1) * 0.001
    
    # L2: 高层逻辑耦合层
    W_L2 = torch.randn(dim_l1, dim_l2) * 0.02
    P_L2 = torch.eye(dim_l2) + torch.randn(dim_l2, dim_l2) * 0.005
    
    state = {
        'version': '31.0',
        'vocab_size': vocab_size,
        'represent_dim': dim_l1,
        'dim_l2': dim_l2,
        'W_receptors': W_L1,
        'P_topo': P_L1,
        'W_L2': W_L2,
        'P_L2': P_L2,
    }
    
    os.makedirs("tempdata", exist_ok=True)
    save_path = "tempdata/mother_language_state_v2.pt"
    torch.save(state, save_path)
    print(f"Success! Crystal saved to {save_path}")

if __name__ == "__main__":
    generate_multi_layer_crystal()
