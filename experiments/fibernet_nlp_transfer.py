
import copy
import os
import random
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fibernet_v2 import DecoupledFiberNet

# --- Configuration ---
BATCH_SIZE = 32
EPOCHS_EN = 50
EPOCHS_FR = 50
LR = 0.01

# Vocabulary (Tiny SVO)
# English
EN_SUBJ = ["I", "You", "We", "They"]
EN_VERB = ["love", "see", "know", "help"]
EN_OBJ  = ["him", "her", "it", "them"]

# French (Simplified SVO)
FR_SUBJ = ["Je", "Tu", "Nous", "Ils"]
FR_VERB = ["aime", "vois", "sais", "aide"]
FR_OBJ  = ["le", "la", "lui", "les"]

# Mapping to IDs
def build_vocab(words):
    token_to_id = {"<PAD>": 0}
    id_to_token = {0: "<PAD>"}
    for w in words:
        if w not in token_to_id:
            idx = len(token_to_id)
            token_to_id[w] = idx
            id_to_token[idx] = w
    return token_to_id, id_to_token

EN_WORDS = EN_SUBJ + EN_VERB + EN_OBJ
EN_VOCAB, EN_ID2WORD = build_vocab(EN_WORDS)
FR_WORDS = FR_SUBJ + FR_VERB + FR_OBJ
FR_VOCAB, FR_ID2WORD = build_vocab(FR_WORDS)

VOCAB_SIZE_EN = len(EN_VOCAB)
VOCAB_SIZE_FR = len(FR_VOCAB)
SEQ_LEN = 3 # Subj, Verb, Obj

# --- Data Generation ---
def generate_svo_data(subjs, verbs, objs, vocab, n_samples=500):
    X = []
    Y = [] # Next token prediction task? Or Masked? 
    # Let's do simple Next Token Prediction for last position?
    # Or just "Classify the sentence validness"? 
    # No, let's do: Input: Subj, Verb -> Output: Obj (Causal)
    # Actually, standard LM training: 
    # Input: [S, V] -> Target: [V, O]
    # But for simplicity: Input [S, V, O] -> Reconstruction? 
    # Let's do: Input [S, V] -> Predict O. 
    # Simplest: Classification. Given S and V, predict compatible O?
    # But here all O are compatible with all V.
    # The task is structure learning: "After a Verb, comes an Object".
    # So we want the model to output a token from the "Object" category.
    
    # Standard Causal LM:
    # Seq: S -> V -> O
    # Loss calculated on V (from S) and O (from S, V).
    
    data = []
    for _ in range(n_samples):
        s = random.choice(subjs)
        v = random.choice(verbs)
        o = random.choice(objs)
        # IDs
        ids = [vocab[s], vocab[v], vocab[o]]
        data.append(ids)
    
    return torch.tensor(data, dtype=torch.long)

data_en = generate_svo_data(EN_SUBJ, EN_VERB, EN_OBJ, EN_VOCAB, 1000)
data_fr = generate_svo_data(FR_SUBJ, FR_VERB, FR_OBJ, FR_VOCAB, 1000)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Training Function ---
def train_model(model, data, epochs, name, freeze_logic=False):
    print(f"\n--- Training {name} (Freeze Logic: {freeze_logic}) ---")
    model.to(device)
    
    # Identify parameters to optimize
    if freeze_logic:
        # Freeze everything EXCEPT content_embeddings and head
        params = []
        frozen_names = []
        for n, p in model.named_parameters():
             # Logic Stream: pos_embed, layers.*.logic_*, layers.*.attn.W_Q/K
             # We ALSO freeze interaction physics: layers.*.attn.W_V/O and mem_norm/ffn
             # We ONLY train the "Dictionary" (Embeddings + Head)
            if "content_embed" in n or "head" in n:
                params.append(p)
                p.requires_grad = True
            else:
                p.requires_grad = False
                frozen_names.append(n)
        # print("Frozen:", len(frozen_names), "params")
    else:
        params = model.parameters()
        
    optimizer = optim.Adam(params, lr=LR)
    criterion = nn.CrossEntropyLoss()
    
    losses = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Causal LM Task
        inputs = data[:, :-1] # S, V
        targets = data[:, 1:] # V, O
        
        logits = model(inputs) # [Batch, 2, Vocab]
        
        # Reshape for loss
        logits = logits.reshape(-1, logits.size(-1))
        targets = targets.reshape(-1)
        
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"[{name}] Epoch {epoch}: Loss {loss.item():.4f}")
            
    return losses

# --- Experiment ---

# 1. Train FiberNet on English (Full)
print("Initializing FiberNet...")
model = DecoupledFiberNet(vocab_size=VOCAB_SIZE_EN, d_model=32, n_layers=2, group_type='circle', max_len=5)

print("\nStep 1: Pre-training on English...")
loss_en = train_model(model, data_en, EPOCHS_EN, "FiberNet (English)", freeze_logic=False)

# 2. Prepare for French (Transfer)
print("\nStep 2: Adapting to French...")

# Save Logic State
logic_state = copy.deepcopy(model.state_dict())

# Re-initialize Content Embeddings & Head for French Vocab
# We need to construct a new model to handle size change, but load old weights
model_fr = DecoupledFiberNet(vocab_size=VOCAB_SIZE_FR, d_model=32, n_layers=2, group_type='circle', max_len=5)

# Load Logic Weights from English Model
new_state = model_fr.state_dict()
for name, param in logic_state.items():
    if "content_embed" not in name and "head" not in name:
        # Transfer Logic & Physics weights
        new_state[name] = param
model_fr.load_state_dict(new_state)

# Train on French (Frozen Logic)
loss_fr_transfer = train_model(model_fr, data_fr, EPOCHS_FR, "FiberNet (French Transfer)", freeze_logic=True)

# 3. Baseline: Train French from Scratch
print("\nStep 3: Training French from Scratch (Baseline)...")
model_baseline = DecoupledFiberNet(vocab_size=VOCAB_SIZE_FR, d_model=32, n_layers=2, group_type='circle', max_len=5)
loss_fr_scratch = train_model(model_baseline, data_fr, EPOCHS_FR, "FiberNet (French Scratch)", freeze_logic=False)

# --- Plot ---
plt.figure(figsize=(10, 5))
plt.plot(loss_en, label='English Pre-train (Reference)', linestyle=':', alpha=0.5)
plt.plot(loss_fr_scratch, label='French (Scratch)', color='blue')
plt.plot(loss_fr_transfer, label='French (Transfer - Logic Frozen)', color='red', linewidth=2)
plt.title('Logic Decoupling: Zero-shot Syntax Transfer')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

os.makedirs('tempdata/fibernet', exist_ok=True)
plt.savefig('tempdata/fibernet/nlp_transfer.png')
print("Saved plot to tempdata/fibernet/nlp_transfer.png")
