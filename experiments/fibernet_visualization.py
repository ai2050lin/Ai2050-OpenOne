
import copy
import os
import random
import sys

import matplotlib.pyplot as plt

# import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fibernet_v2 import DecoupledFiberNet

# --- Configuration ---
BATCH_SIZE = 32
EPOCHS_EN = 30
EPOCHS_FR = 30
LR = 0.01

# Vocabulary (Tiny SVO)
EN_SUBJ = ["I", "You", "We", "They"]
EN_VERB = ["love", "see", "know", "help"]
EN_OBJ  = ["him", "her", "it", "them"]

FR_SUBJ = ["Je", "Tu", "Nous", "Ils"]
FR_VERB = ["aime", "vois", "sais", "aide"]
FR_OBJ  = ["le", "la", "lui", "les"]

def build_vocab(words):
    token_to_id = {"<PAD>": 0}
    id_to_token = {0: "<PAD>"}
    for w in words:
        if w not in token_to_id:
            idx = len(token_to_id)
            token_to_id[w] = idx
            id_to_token[idx] = w
    return token_to_id, id_to_token

EN_VOCAB, _ = build_vocab(EN_SUBJ + EN_VERB + EN_OBJ)
FR_VOCAB, _ = build_vocab(FR_SUBJ + FR_VERB + FR_OBJ)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Data Generation ---
def generate_svo_data(subjs, verbs, objs, vocab, n_samples=500):
    data = []
    for _ in range(n_samples):
        s = random.choice(subjs)
        v = random.choice(verbs)
        o = random.choice(objs)
        ids = [vocab[s], vocab[v], vocab[o]]
        data.append(ids)
    return torch.tensor(data, dtype=torch.long)

data_en = generate_svo_data(EN_SUBJ, EN_VERB, EN_OBJ, EN_VOCAB, 500)
data_fr = generate_svo_data(FR_SUBJ, FR_VERB, FR_OBJ, FR_VOCAB, 500)

# --- Training Helper ---
def train_model(model, data, epochs, name, freeze_logic=False):
    print(f"Training {name}...")
    model.to(device)
    if freeze_logic:
        params = [p for n, p in model.named_parameters() if "content_embed" in n or "head" in n]
    else:
        params = model.parameters()
    
    optimizer = optim.Adam(params, lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        inputs = data[:, :-1]
        targets = data[:, 1:]
        logits = model(inputs)
        loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")
    return model

# --- Visualization Helper ---
# We need to hook into the model to get attention weights
# LogicDrivenAttention calculates 'attn' [batch, head, seq, seq]
activation = {}
def get_activation(name):
    def hook(model, input, output):
        # We need to capture the 'attn' variable inside forward
        # But standard hooks capture output.
        # LogicDrivenAttention returns `self.W_O(output)`
        # The 'attn' matrix isn't returned.
        # We might need to monkey-patch or use a custom forward for visualization.
        pass
    return hook

# Alternative: Extend the class to return attn
class VisualizableFiberNet(DecoupledFiberNet):
    def forward_with_attn(self, input_ids):
        # Re-implement forward to expose attn
        batch, seq = input_ids.shape
        device = input_ids.device
        positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        curr_logic = self.pos_embed(positions)
        curr_memory = self.content_embed(input_ids)
        
        attns = []
        
        for layer in self.layers:
            # Logic Evolve
            res_l = curr_logic
            curr_logic, _ = layer.logic_attn(curr_logic, curr_logic, curr_logic)
            curr_logic = layer.logic_norm1(res_l + curr_logic)
            res_l = curr_logic
            curr_logic = layer.logic_norm2(res_l + layer.logic_ffn(curr_logic))
            
            # Memory Evolve (Logic Driven)
            # Access internal attention
            # Re-compute logic-driven attention manually to get weights
            lda = layer.attn
            head_dim_logic = lda.d_logic // lda.nhead
            Q = lda.W_Q(curr_logic).reshape(batch, seq, lda.nhead, head_dim_logic).transpose(1, 2)
            K = lda.W_K(curr_logic).reshape(batch, seq, lda.nhead, head_dim_logic).transpose(1, 2)
            scores = (Q @ K.transpose(-2, -1)) / (head_dim_logic ** 0.5)
            attn_weights = torch.softmax(scores, dim=-1) # [batch, head, seq, seq]
            attns.append(attn_weights)
            
            # Apply
            head_dim_memory = lda.d_memory // lda.nhead
            V = lda.W_V(curr_memory).reshape(batch, seq, lda.nhead, head_dim_memory).transpose(1, 2)
            transported = attn_weights @ V
            transported = transported.transpose(1, 2).flatten(2)
            transported = lda.W_O(transported)
            
            res_m = curr_memory
            curr_memory = layer.mem_norm1(res_m + transported)
            res_m = curr_memory
            curr_memory = layer.mem_norm2(res_m + layer.mem_ffn(curr_memory))
            
        return curr_memory, attns

# --- Run ---

# 1. Train English
print("\n--- 1. English Pre-training ---")
model = VisualizableFiberNet(vocab_size=len(EN_VOCAB), d_model=32, n_layers=1, group_type='circle', max_len=3)
train_model(model, data_en, EPOCHS_EN, "FiberNet (EN)")

# Get Attention for "I love her"
idx_en = torch.tensor([[EN_VOCAB["I"], EN_VOCAB["love"], EN_VOCAB["her"]]], device=device)
_, attns_en = model.forward_with_attn(idx_en)
attn_en = attns_en[0][0, 0].detach().cpu().numpy() # Layer 0, Batch 0, Head 0

# 2. Transfer to French
print("\n--- 2. French Transfer (Frozen Logic) ---")
logic_state = copy.deepcopy(model.state_dict())
model_fr = VisualizableFiberNet(vocab_size=len(FR_VOCAB), d_model=32, n_layers=1, group_type='circle', max_len=3)

# Load Logic Weights
new_state = model_fr.state_dict()
for k, v in logic_state.items():
    if "content_embed" not in k and "head" not in k:
        new_state[k] = v
model_fr.load_state_dict(new_state)

train_model(model_fr, data_fr, EPOCHS_FR, "FiberNet (FR)", freeze_logic=True)

# Get Attention for "Je aime la"
idx_fr = torch.tensor([[FR_VOCAB["Je"], FR_VOCAB["aime"], FR_VOCAB["la"]]], device=device)
_, attns_fr = model_fr.forward_with_attn(idx_fr)
attn_fr = attns_fr[0][0, 0].detach().cpu().numpy()

# --- Plot Comparison ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# sns.heatmap(attn_en, ax=axes[0], annot=True, cmap="Blues", xticklabels=["I", "love", "her"], yticklabels=["I", "love", "her"])
im0 = axes[0].imshow(attn_en, cmap="Blues")
axes[0].set_xticks(range(3))
axes[0].set_xticklabels(["I", "love", "her"])
axes[0].set_yticks(range(3))
axes[0].set_yticklabels(["I", "love", "her"])
for i in range(3):
    for j in range(3):
        axes[0].text(j, i, f"{attn_en[i, j]:.2f}", ha="center", va="center", color="black")

# sns.heatmap(attn_fr, ax=axes[1], annot=True, cmap="Blues", xticklabels=["Je", "aime", "la"], yticklabels=["Je", "aime", "la"])
im1 = axes[1].imshow(attn_fr, cmap="Blues")
axes[1].set_xticks(range(3))
axes[1].set_xticklabels(["Je", "aime", "la"])
axes[1].set_yticks(range(3))
axes[1].set_yticklabels(["Je", "aime", "la"])
for i in range(3):
    for j in range(3):
        axes[1].text(j, i, f"{attn_fr[i, j]:.2f}", ha="center", va="center", color="black")
axes[1].set_title("French Logic Stream (Frozen)")

plt.suptitle("Logic Invariance: Same Structure, Different Language")
os.makedirs('tempdata/fibernet', exist_ok=True)
plt.savefig('tempdata/fibernet/structure_invariance.png')
print("Saved plot to tempdata/fibernet/structure_invariance.png")
