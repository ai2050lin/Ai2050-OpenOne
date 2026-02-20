
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json

# --- Environment Definition (Defined here to avoid import issues) ---
class EmbodiedEnv:
    def __init__(self, dim=128, num_obstacles=5):
        self.dim = dim
        self.state = np.random.randn(dim) / np.sqrt(dim)
        self.goal = np.random.randn(dim) / np.sqrt(dim)
        self.obstacles = [np.random.randn(dim) / np.sqrt(dim) for _ in range(num_obstacles)]
        self.obstacle_radius = 0.5
        self.max_steps = 200
        self.current_step = 0
        
    def reset(self):
        self.state = np.random.randn(self.dim) / np.sqrt(self.dim)
        self.current_step = 0
        return self.state
    
    def step(self, action_vector):
        self.current_step += 1
        v = action_vector / (np.linalg.norm(action_vector) + 1e-6) * 0.1
        self.state = self.state + v
        move_cost = np.linalg.norm(v)
        obstacle_cost = 0
        for obs in self.obstacles:
            dist = np.linalg.norm(self.state - obs)
            if dist < self.obstacle_radius:
                obstacle_cost += (self.obstacle_radius - dist) * 10.0
                diff = (self.state - obs) / (dist + 1e-6)
                self.state += diff * (self.obstacle_radius - dist)
        dist_to_goal = np.linalg.norm(self.state - self.goal)
        reward = - (move_cost + obstacle_cost + 0.1 * dist_to_goal)
        done = (dist_to_goal < 0.1) or (self.current_step >= self.max_steps)
        return self.state, reward, done, {"dist": dist_to_goal}

# --- Configuration ---
DIM = 128
D_MODEL = 256
N_LAYERS = 4
LR = 1e-4
EPOCHS = 50
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata")
os.makedirs(RES_DIR, exist_ok=True)

# --- Model ---
class EmbodiedFiberNet(nn.Module):
    def __init__(self, dim, d_model):
        super().__init__()
        self.state_enc = nn.Linear(dim, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=512, batch_first=True, norm_first=True)
            for _ in range(N_LAYERS)
        ])
        self.action_head = nn.Linear(d_model, dim)
        
    def forward(self, state):
        h = self.state_enc(state).unsqueeze(1)
        for layer in self.layers:
            h = layer(h)
        latent = h.squeeze(1)
        action = self.action_head(latent)
        return action, latent

# --- Training ---
def train_embodied():
    print(f"[*] Starting Phase 6: Embodied Alignment (Self-Contained) on {DEV}")
    env = EmbodiedEnv(dim=DIM)
    model = EmbodiedFiberNet(DIM, D_MODEL).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    
    results = []
    
    for ep in range(1, EPOCHS+1):
        total_reward = 0
        state = env.reset()
        done = False
        step = 0
        
        while not done and step < 200:
            state_pt = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(DEV)
            action_pred, _ = model(state_pt)
            
            # Simple Geodesic Target
            target_direction = env.goal - state
            target_pt = torch.tensor(target_direction, dtype=torch.float32).unsqueeze(0).to(DEV)
            
            loss = F.mse_loss(action_pred, target_pt)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            action_np = action_pred.detach().cpu().numpy().squeeze()
            state, reward, done, info = env.step(action_np)
            total_reward += reward
            step += 1
            
        avg_dist = info["dist"]
        print(f"Ep {ep} | Reward: {total_reward:.2f} | Final Dist: {avg_dist:.4f}")
        
        results.append({"ep": ep, "reward": total_reward, "dist": avg_dist})
        
        if ep % 5 == 0:
            with open(os.path.join(RES_DIR, "phase6_embodied_results.json"), "w") as f:
                json.dump(results, f)
            torch.save(model.state_dict(), os.path.join(RES_DIR, f"fiber_embodied_ep{ep}.pth"))

    print("[+] Phase 6 Training Complete.")

if __name__ == "__main__":
    train_embodied()
