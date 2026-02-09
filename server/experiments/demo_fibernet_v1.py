
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fiber_net import FiberNet


def generate_synthetic_data(num_samples=1000):
    """
    Generates synthetic (Structure, Content) pairs.
    Structure: [SUBJ, VERB, OBJ] -> [OBJ, WAS, VERB-ed, BY, SUBJ]
    
    Vocabulary Mapping:
    Structure: {S_SUBJ:0, S_VERB:1, S_OBJ:2, S_PREP:3, S_AUX:4}
    Content: {CAT:0, DOG:1, CHASE:2, EAT:3, MOUSE:4, BONE:5}
    """
    # Simply mapping for readability
    S_SUBJ, S_VERB, S_OBJ, S_PREP, S_AUX = 0, 1, 2, 3, 4
    
    # 0-5 are content words
    CAT, DOG, CHASE, EAT, MOUSE, BONE = 0, 1, 2, 3, 4, 5
    
    data = []
    
    # Simple grammar: SUBJ VERB OBJ
    # Logic: If input is SUBJ VERB OBJ, output is passive voice mapping or just a logical copy
    # For this demo, let's learn Identity: Output should match Input Content but Transported
    
    for _ in range(num_samples):
        # Randomly pick content
        subj = np.random.choice([CAT, DOG])
        verb = np.random.choice([CHASE, EAT])
        obj = np.random.choice([MOUSE, BONE])
        
        # Input: "Subject Verb Object"
        input_structure = [S_SUBJ, S_VERB, S_OBJ]
        input_content = [subj, verb, obj]
        
        # Target: "Subject Verb Object" (Identity task for simplicity of v1)
        # The key is that the Manifold learns the SEQUENCE structure
        target_content = [subj, verb, obj]
        
        data.append((input_structure, input_content, target_content))
        
    return data

def train_fibernet(model, data, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print(f"Training on {len(data)} samples for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        for str_in, con_in, con_out in data:
            str_tensor = torch.LongTensor([str_in]) # [1, 3]
            con_tensor = torch.LongTensor([con_in]) # [1, 3]
            target = torch.LongTensor([con_out])    # [1, 3]
            
            optimizer.zero_grad()
            logits, _, _ = model(str_tensor, con_tensor) # logits: [1, 3, vocab]
            
            # Reshape for loss
            loss = criterion(logits.view(-1, logits.size(-1)), target.view(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(data):.4f}")

def alice_test(model):
    print("\n=== THE ALICE TEST (One-Shot Injection) ===")
    
    # 1. Define new concept "Alice" (ID: 6) and "Pizza" (ID: 7)
    # They are NOT in the training data (0-5)
    ALICE = 6
    PIZZA = 7
    
    print(f"Injecting new concepts: ALICE (id={ALICE}), PIZZA (id={PIZZA})")
    
    # 2. Inject into Fiber Memory (Manually assigning random vectors)
    # We do NOT propagate gradients or retrain the Manifold/Connection
    with torch.no_grad():
        # Just ensure the embedding layer is large enough or resize it
        # In this script, we initialized with size 10, so index 6, 7 are valid but untrained (random)
        # Let's explicitly set them to be sure they are distinctive 'concepts'
        model.fiber_stream.fiber_memory.weight[ALICE] = torch.randn(model.d_fiber)
        model.fiber_stream.fiber_memory.weight[PIZZA] = torch.randn(model.d_fiber)
        
    # 3. Test Inference
    # Input: "Alice Chase Pizza" (New Subject, Old Verb, New Object)
    # Structure is same: [SUBJ, VERB, OBJ]
    S_SUBJ, S_VERB, S_OBJ = 0, 1, 2
    CHASE = 2 # Known verb
    
    test_str = torch.LongTensor([[S_SUBJ, S_VERB, S_OBJ]])
    test_con = torch.LongTensor([[ALICE, CHASE, PIZZA]])
    
    print("Input: Alice(6) Chase(2) Pizza(7)")
    
    with torch.no_grad():
        logits, fiber_out, _ = model(test_str, test_con)
        probs = torch.softmax(logits, dim=-1)
        preds = torch.argmax(probs, dim=-1)
        
    print(f"Prediction: {preds.tolist()}")
    
    # 4. Check Parallel Transport
    # Did the model correctly identify that output position 0 is ALICE?
    # Since the manifold learned "Pos 0 -> Pos 0 Identity" mapping, it should transport the vector at pos 0 in fiber stream to output.
    
    success = (preds[0][0].item() == ALICE) and (preds[0][2].item() == PIZZA)
    
    if success:
        print("\n[PASS] Alice Test Passed!")
        print("The model correctly transported the UNSEEN concept 'Alice' based on syntactic rules learned from 'Cat/Dog'.")
        print("This confirms the Manifold (Logic) and Fiber (Knowledge) are decoupled.")
    else:
        print("\n[FAIL] Alice Test Failed.")
        print(f"Expected: [{ALICE}, {CHASE}, {PIZZA}]")
        print(f"Got:      {preds[0].tolist()}")

def main():
    # Config
    S_VOCAB = 10 # Structure vocab
    C_VOCAB = 10 # Content vocab (0-5 used, 6-9 reserved for new)
    
    model = FiberNet(S_VOCAB, C_VOCAB, d_manifold=16, d_fiber=32)
    
    # 1. Generate & Train
    data = generate_synthetic_data(200)
    train_fibernet(model, data, epochs=30)
    
    # 2. Run Test
    alice_test(model)

if __name__ == "__main__":
    main()
