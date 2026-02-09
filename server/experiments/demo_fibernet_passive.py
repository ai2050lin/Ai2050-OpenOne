
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fiber_net import FiberNet


def generate_passive_data(num_samples=1000):
    """
    Generates synthetic Passive -> Active pairs.
    Input: "Object WAS Verb-ed BY Subject"
    Structure Tokens: [S_OBJ, S_WAS, S_VERB_ED, S_BY, S_SUBJ]
    
    Target: "Subject Verb Object" (Active)
    Structure Tokens: [S_SUBJ, S_VERB, S_OBJ]
    """
    # Structure Vocab Mapping
    S_SUBJ, S_VERB, S_OBJ = 0, 1, 2
    S_WAS, S_VERB_ED, S_BY = 3, 4, 5
    
    # Content Vocab
    CAT, DOG, CHASE, EAT, MOUSE, BONE = 0, 1, 2, 3, 4, 5
    
    data = []
    
    for _ in range(num_samples):
        # Logic: "The MOUSE(4) WAS CHASED(2) BY the CAT(0)" -> "CAT(0) CHASE(2) MOUSE(4)"
        subj = np.random.choice([CAT, DOG])
        verb = np.random.choice([CHASE, EAT])
        obj = np.random.choice([MOUSE, BONE])
        
        # Input Structure: [OBJ, WAS, VERB_ED, BY, SUBJ]
        # Input Content:   [obj, (pad), verb, (pad), subj]
        # Note: 'WAS' and 'BY' are function words, they don't carry fiber content in this simplified model,
        # or we can view them as having 'null' content or specific content.
        # For simplicity v1: We map content ids to input positions directly.
        # Let's say WAS/BY have content ID 0 (or a special PAD id).
        # We'll use the verb content for 'VERB_ED'.
        
        input_structure = [S_OBJ, S_WAS, S_VERB_ED, S_BY, S_SUBJ]
        input_content = [obj, 0, verb, 0, subj] # 0 acts as filler for function words
        
        # Target Structure: [S_SUBJ, S_VERB, S_OBJ] (Active)
        # Target Content:   [subj, verb, obj]
        
        target_content = [subj, verb, obj]
        
        data.append((input_structure, input_content, target_content))
        
    return data

def train_fibernet(model, data, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on {len(data)} Passive->Active samples...")
    
    for epoch in range(epochs):
        total_loss = 0
        for str_in, con_in, con_out in data:
            str_tensor = torch.LongTensor([str_in]) 
            con_tensor = torch.LongTensor([con_in]) 
            target = torch.LongTensor([con_out])   
            
            optimizer.zero_grad()
            logits, _, _ = model(str_tensor, con_tensor) 
            
            # FiberNet v1 output length is same as input length because of RNN seq-to-seq logic in v1?
            # Wait, v1 ManifoldStream uses LSTM(batch_first=True) and returns output for *each* input step.
            # Input len = 5, Target len = 3.
            # We need to grab the LAST 3 outputs? Or does the Manifold learn to compress?
            # Issue: v1 Manifold is Seq2Seq (Many-to-Many) same length.
            # "Identity" worked because Input Len == Output Len.
            # specific fix for v1: We just evaluate the first 3 outputs, OR we PAD the target to 5.
            # Let's PAD the target to 5: [SUBJ, VERB, OBJ, PAD, PAD]
            # And mask the loss.
            
            # Padding target to match input length (5)
            # We want model to predict [SUBJ, VERB, OBJ] at first 3 positions.
            # (Or maybe at last 3? Let's say first 3 for immediate "Active Voice Translation")
            
            # Actually, standard RNN seq2seq usually generates step-by-step.
            # FiberNet v1 `forward` returns `m_states` of shape [batch, input_len, d].
            # So output will have 5 steps.
            # We train it to output [SUBJ, VERB, OBJ, PAD, PAD].
            
            PAD_CONTENT = 0 # Using 0 as PAD for simplicity
            
            target_padded = torch.cat([target, torch.LongTensor([[PAD_CONTENT, PAD_CONTENT]])], dim=1)
            
            # Calculate loss only on first 3 (Active sentence reconstruction)
            # logic: logits is [1, 5, vocab]
            # target_padded is [1, 5]
            
            loss = criterion(logits.view(-1, logits.size(-1)), target_padded.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data):.4f}")

def alice_test_passive(model):
    print("\n=== THE ALICE TEST (Passive -> Active) ===")
    
    # 1. Define new concepts
    ALICE = 6
    PIZZA = 7
    
    # 2. Inject
    with torch.no_grad():
        # CRITICAL FIX: Normalize injected vectors to match trained embedding statistics.
        # If random vectors have larger norm, they will dominate Attention (Softmax).
        trained_mean = model.fiber_stream.fiber_memory.weight.mean()
        trained_std = model.fiber_stream.fiber_memory.weight.std()
        
        alice_vec = torch.randn(model.d_fiber)
        alice_vec = (alice_vec - alice_vec.mean()) / alice_vec.std() * trained_std + trained_mean
        
        pizza_vec = torch.randn(model.d_fiber)
        pizza_vec = (pizza_vec - pizza_vec.mean()) / pizza_vec.std() * trained_std + trained_mean
        
        model.fiber_stream.fiber_memory.weight[ALICE] = alice_vec
        model.fiber_stream.fiber_memory.weight[PIZZA] = pizza_vec
    
    # 3. Test: "Pizza(7) WAS Chase(2)-ed BY Alice(6)"
    # We expect output: "Alice(6) Chase(2) Pizza(7)"
    
    S_SUBJ, S_VERB, S_OBJ = 0, 1, 2
    S_WAS, S_VERB_ED, S_BY = 3, 4, 5
    CHASE = 2
    
    # Content Input: [PIZZA, 0, CHASE, 0, ALICE]
    input_str = [S_OBJ, S_WAS, S_VERB_ED, S_BY, S_SUBJ]
    input_con = [PIZZA, 0, CHASE, 0, ALICE]
    
    print(f"Input: 'Pizza(7) WAS Chased(2) BY Alice(6)'")
    
    t_str = torch.LongTensor([input_str])
    t_con = torch.LongTensor([input_con])
    
    with torch.no_grad():
        logits, _, _ = model(t_str, t_con)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
    
    # We look at first 3 outputs
    pred_seq = preds[0][:3].tolist()
    print(f"Prediction: {pred_seq} (Full: {preds[0].tolist()})")
    
    # Check
    # Expected: [ALICE, CHASE, PIZZA] => [6, 2, 7]
    if pred_seq == [ALICE, CHASE, PIZZA]:
        print("\n[PASS] Passive-to-Active Alice Test Passed!")
        print("Structure learned: 'Obj ... By Subj' -> 'Subj Verb Obj'")
        print("Parallel Transport confirmed: Alice moved from Pos 4 to Pos 0, Pizza from Pos 0 to Pos 2.")
    else:
        print("\n[FAIL] Passive-to-Active Alice Test Failed.")
        print(f"Expected: {[ALICE, CHASE, PIZZA]}")
        print(f"Got:      {pred_seq}")

def main():
    # Vocab Config
    S_VOCAB = 10 
    C_VOCAB = 10 
    
    model = FiberNet(S_VOCAB, C_VOCAB, d_manifold=32, d_fiber=64)
    
    # 1. Train on Passive -> Active task
    data = generate_passive_data(2000)
    train_fibernet(model, data, epochs=30) # Need slightly more epochs for harder task
    
    # 2. Test One-Shot
    alice_test_passive(model)

if __name__ == "__main__":
    main()
