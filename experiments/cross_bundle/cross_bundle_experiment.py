
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.spatial.transform import Rotation as R
from torch.utils.data import DataLoader, Dataset

from models.entanglement_model import DualFiberNet

# --- Configuration ---
NUM_OBJECTS = 3 # Cube, Sphere, Pyramid
OBJECT_NAMES = ["Cube", "Sphere", "Pyramid"]
AXES = ["X", "Y", "Z"]
ANGLES = [90]

# Vocab: PAD, START, STOP, ROTATE, X, Y, Z
VOCAB = ["<PAD>", "<START>", "<STOP>", "Rotate", "X", "Y", "Z"]
VOCAB_MAP = {w: i for i, w in enumerate(VOCAB)}
IDX_MAP = {i: w for i, w in enumerate(VOCAB)}

class CrossBundleDataset(Dataset):
    def __init__(self, num_samples=1000, split="train"):
        self.samples = []
        
        # Train: 
        # - Single rotations (Length 1)
        # - Same-axis double rotations (Length 2, e.g. X-X, Y-Y)
        # Test: 
        # - Mixed-axis double rotations (Length 2, e.g. X-Y, Z-X)
        # This tests Compositional Generalization:
        # Can the model combine "Rotate X" and "Rotate Y" syntax/semantics 
        # when it has only seen "Rotate X ... X" and "Rotate Y ... Y"?
        
        for _ in range(num_samples):
            obj_idx = np.random.randint(0, NUM_OBJECTS)
            
            # Determine sequence type based on split
            if split == "train":
                # 50% Length 1, 50% Length 2 (Same Axis)
                if np.random.rand() < 0.5:
                    steps = 1
                    axes_seq = [AXES[np.random.randint(0, 3)]]
                else:
                    steps = 2
                    # Same axis for both steps
                    axis = AXES[np.random.randint(0, 3)]
                    axes_seq = [axis, axis]
            else: # test
                steps = 2
                # Different axes
                a1 = AXES[np.random.randint(0, 3)]
                a2 = AXES[np.random.randint(0, 3)]
                while a1 == a2:
                    a2 = AXES[np.random.randint(0, 3)]
                axes_seq = [a1, a2]
            
            rot_matrices = [np.eye(3)] # Start with Identity
            text_tokens = [VOCAB_MAP["<START>"]]
            
            current_rot = R.from_matrix(np.eye(3))
            
            for axis in axes_seq:
                # Update Rotation
                # Rotate 90 deg around axis
                r_step = R.from_euler(axis.lower(), 90, degrees=True)
                current_rot = current_rot * r_step
                rot_matrices.append(current_rot.as_matrix())
                
                # Update Text
                text_tokens.append(VOCAB_MAP["Rotate"])
                text_tokens.append(VOCAB_MAP[axis])
            
            text_tokens.append(VOCAB_MAP["<STOP>"])
            
            self.samples.append({
                "obj_idx": obj_idx,
                "rot_matrices": np.array(rot_matrices, dtype=np.float32),
                "text_tokens": text_tokens,
                "seq_description": "-".join(axes_seq)
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return item

def collate_fn(batch):
    # dynamic padding
    max_visual_len = max([len(x['rot_matrices']) for x in batch])
    max_text_len = max([len(x['text_tokens']) for x in batch])
    
    b_size = len(batch)
    
    rot_mats = torch.zeros(b_size, max_visual_len, 3, 3)
    # objects
    obj_ids = torch.zeros(b_size, dtype=torch.long)
    
    input_ids = torch.zeros(b_size, max_text_len, dtype=torch.long)
    
    # Fill
    for i, x in enumerate(batch):
        v_len = len(x['rot_matrices'])
        rot_mats[i, :v_len] = torch.tensor(x['rot_matrices'])
        # Pad last matrix if needed? Or zero? Zero is bad for rotation. Identity padding.
        if v_len < max_visual_len:
             rot_mats[i, v_len:] = torch.eye(3) 
             
        obj_ids[i] = x['obj_idx']
        
        t_len = len(x['text_tokens'])
        input_ids[i, :t_len] = torch.tensor(x['text_tokens'])
        
    return rot_mats, obj_ids, input_ids

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Datasets
    train_dataset = CrossBundleDataset(num_samples=2000, split="train")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    test_dataset = CrossBundleDataset(num_samples=200, split="test") 
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    print(f"Train Set: {len(train_dataset)} samples (Single & Homogenous Pairs)")
    print(f"Test Set: {len(test_dataset)} samples (Heterogenous Pairs - Zero Shot)")
    
    # 2. Model
    model = DualFiberNet(NUM_OBJECTS, len(VOCAB), d_model=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion_ce = nn.CrossEntropyLoss(ignore_index=0) # PAD=0
    criterion_geo = nn.MSELoss()
    
    EPOCHS = 20
    
    print("\n--- Phase 1: Entanglement Training ---")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        geo_loss_total = 0
        lang_loss_total = 0
        
        for rot_mats, obj_ids, input_ids in train_loader:
            rot_mats, obj_ids, input_ids = rot_mats.to(device), obj_ids.to(device), input_ids.to(device)
            
            # Forward
            # Input to language model is input_ids[:, :-1]
            # Target is input_ids[:, 1:]
            lang_in = input_ids[:, :-1]
            lang_target = input_ids[:, 1:]
            
            l_logits, bridged_state, l_manifold = model((rot_mats, obj_ids), lang_in)
            
            # Language Loss
            l_loss = criterion_ce(l_logits.reshape(-1, len(VOCAB)), lang_target.reshape(-1))
            
            # Geometric Loss
            # Align bridged visual state with language manifold
            # We need to align the timesteps.
            # Visual: [I, R1] (len 2)
            # Language: [START, Rot, X] (len 3 in input, len 3 in output)
            # This alignment is tricky.
            # "Identity" visual matches "START" language?
            # "R1" visual matches "START Rot X"? Or just the end state?
            # Let's align the LAST state of visual with the LAST state of language for now.
            # Or pool them.
            
            v_final = bridged_state[:, -1, :] 
            l_final = l_manifold[:, -1, :]
            geo_loss = criterion_geo(v_final, l_final)
            
            loss = l_loss + 1.0 * geo_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            geo_loss_total += geo_loss.item()
            lang_loss_total += l_loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} (Lang: {lang_loss_total/len(train_loader):.4f}, Geo: {geo_loss_total/len(train_loader):.4f})")


    # 3. Zero-Shot Testing
    # We feed visual sequences of length 2 (which model never saw in training)
    # and ask it to generate text.
    print("\n--- Phase 2: Zero-Shot Compositional Generation ---")
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for i in range(10): # Test 10 samples qualitatively
            sample = test_dataset[i]
            rot_mats = torch.tensor(sample['rot_matrices']).unsqueeze(0).to(device)
            obj_id = torch.tensor([sample['obj_idx']]).to(device)
            
            # Ground Truth
            gt_text = "".join([IDX_MAP[t] + " " for t in sample['text_tokens'] if t > 2]).replace("<STOP>", "")
            
            # Generate
            generated_ids = model.generate_from_visual((rot_mats, obj_id), start_token_id=1, max_len=10) # 1=START
            gen_text = "".join([IDX_MAP[t] + " " for t in generated_ids if t > 2]).replace("<STOP>", "")
            
            print(f"Input: {sample['seq_description']}")
            print(f"  Truth: {gt_text}")
            print(f"  Pred : {gen_text}")
            
            if gen_text.strip() == gt_text.strip():
                correct += 1
            total += 1
            
    print(f"\nQualitative Accuracy: {correct}/{total}")
    
    # Quantitative
    correct = 0
    total = 0
    with torch.no_grad():
        for rot_mats, obj_ids, input_ids in test_loader:
             rot_mats, obj_ids, input_ids = rot_mats.to(device), obj_ids.to(device), input_ids.to(device)
             
             # Create list of batch items
             b_size = rot_mats.shape[0] # Fixed .shape() to .shape[]
             for b in range(b_size):
                 rm = rot_mats[b:b+1]
                 oid = obj_ids[b:b+1]
                 target_ids = input_ids[b].tolist()
                 # Filter PAD and specialized tokens for comparison string
                 target_text = "".join([IDX_MAP[t] + " " for t in target_ids if t > 2]).strip()
                 
                 gen_ids = model.generate_from_visual((rm, oid), start_token_id=1, max_len=10)
                 gen_text = "".join([IDX_MAP[t] + " " for t in gen_ids if t > 2]).strip()
                 
                 if gen_text == target_text:
                     correct += 1
                 total += 1
    
    print(f"Full Test Set Zero-Shot Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    train()
