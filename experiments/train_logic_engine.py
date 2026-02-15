
import os
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fibernet_v2 import DecoupledFiberNet
from scripts.generate_logic_data import LogicDatasetGenerator


def train_logic_engine():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Pure Logic Engine on {device}...")
    
    # 1. Config
    VOCAB_SIZE = 128
    D_MODEL = 64
    N_LAYERS = 2
    
    # 2. Model: PureLogicFiberNet
    # We use the standard DecoupledFiberNet, but...
    model = DecoupledFiberNet(vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS, group_type='circle').to(device)
    
    # CRITICAL: Freeze Memory Stream Embeddings!
    # We want logic to drive EVERYTHING. Memory contents are just random IDs.
    # The model must figure out "Token A followed by Token B means Token C" purely via structural mapping,
    # because we are NOT allowing it to adjust the internal vector of Token 12.
    # Wait... if Token 12's vector is random and frozen, how does Comp(Token 1, Token 2) -> Token 3 work?
    # Logic Stream must perform the calculation in its own space, and then "Select" the correct memory from the vocabulary?
    # Actually, for Z_12, we DO need to learn the output projection (Head) or the relation.
    # If we freeze embeddings, can we learn the task?
    # Yes, if Logic Stream can attend to the correct "Next Token" in the vocab.
    # But usually GPT predicts by dot product with embedding matrix.
    # If embedding matrix is frozen random, then Logic Stream + Memory Stream output must align with that random vector.
    # This forces the network to actually PERFORM the group operation in the vector space!
    # "If I rotate vector A by vector B, I must land on vector C".
    # Since A, B, C are fixed random vectors on Hypersphere, this is HARD.
    # This is exactly what we want: forcing it to discover the geometric isomorphism.
    
    model.content_embed.requires_grad_(False)
    # Note: LieGroupEmbedding parameters are 'phases' or 'weight'.
    # We freeze them.
    for param in model.content_embed.parameters():
        param.requires_grad = False
        
    print("Memory Embeddings Frozen. Model must perform geometric computation on fixed vectors.")
    
    # 3. Data
    generator = LogicDatasetGenerator(vocab_size=VOCAB_SIZE)
    
    # Task 1: Cyclic (Z_12)
    x_cyc, y_cyc = generator.generate_cyclic_data(n_samples=2000, modulus=12)
    
    # Task 2: Transitive (Linear Chain)
    x_trans, y_trans = generator.generate_transitive_data(n_samples=2000, n_nodes=20)
    
    # Task 3: Symmetric (Clusters)
    x_sym, y_sym = generator.generate_symmetric_data(n_samples=2000, n_clusters=10)
    
    # Combine? Or Train Sequentially?
    # Let's train on Cyclic first, as it's the hardest (Group Structure).
    tasks = [
        ("Cyclic (Z_12)", x_cyc, y_cyc),
        # ("Transitive", x_trans, y_trans), # Commented out for initial test
        # ("Symmetric", x_sym, y_sym)
    ]
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.005)
    criterion = nn.CrossEntropyLoss()
    
    for task_name, X, Y in tasks:
        print(f"\n--- Training on {task_name} ---")
        X, Y = X.to(device), Y.to(device)
        
        losses = []
        for epoch in range(100):
            optimizer.zero_grad()
            logits = model(X) # [B, Seq, Vocab]
            
            # Predict last token?
            # Model outputs [B, Seq, Vocab]. Target is [B].
            # We want to predict the "Next" token after the sequence.
            # Wait, FiberNet forward returns prediction for ALL positions (shifted).
            # generate_logic_data returns X=[a, b], Y=c.
            # So we feed [a, b], we want output at pos 1 (corresponding to input b) to predict c?
            # Standard GPT: Input [a, b]. Pos 0 predicts b. Pos 1 predicts c.
            # Our targets Y is just 'c'.
            # So we check logits[:,-1,:] vs Y.
            
            last_token_logits = logits[:, -1, :]
            loss = criterion(last_token_logits, Y)
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            
            if epoch % 10 == 0:
                acc = (last_token_logits.argmax(dim=-1) == Y).float().mean()
                print(f"Epoch {epoch}: Loss {loss.item():.4f}, Acc {acc.item():.4f}")
                
        # Plot
        plt.figure()
        plt.plot(losses)
        plt.title(f"Logic Engine Training: {task_name}")
        plt.savefig(f"tempdata/logic_engine_{task_name.split()[0]}.png")
        print(f"Saved plot for {task_name}")

if __name__ == "__main__":
    train_logic_engine()
