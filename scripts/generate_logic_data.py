
import random

import numpy as np
import torch


class LogicDatasetGenerator:
    """
    Generates pure abstract relational data to train the Logic Engine.
    No natural language, just symbols obeying specific mathematical rules.
    """
    def __init__(self, vocab_size=100, seq_len=10):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        
    def generate_cyclic_data(self, n_samples=1000, modulus=12):
        """
        Generates group operation data: a + b = c (mod n)
        Sequence: [a, '+', b, '=', ?] -> Target: c
        But we want "Logic Stream" style, so maybe just a causal sequence.
        Format: [a, b, c] where c = (a + b) % n.
        Let's map integers 0..n-1 to random tokens in vocab to avoid "number" bias.
        """
        data = []
        targets = []
        
        # Token mapping: 0 -> token_id_x
        # We reserve first 'modulus' tokens for these numbers for simplicity, 
        # OR we randomly map them each time to force abstract reasoning?
        # Let's use fixed mapping for now: Token i = Integer i.
        
        for _ in range(n_samples):
            a = random.randint(0, modulus - 1)
            b = random.randint(0, modulus - 1)
            c = (a + b) % modulus
            
            # Sequence: a, b
            # Target: c
            data.append([a, b])
            targets.append(c)
            
        return torch.tensor(data, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

    def generate_transitive_data(self, n_samples=1000, n_nodes=20):
        """
        Generates transitive chain data: A -> B -> C -> ?
        Task: Predict next node in a valid chain.
        We create a random directed graph (DAG) or just linear chains.
        Graph: 0 -> 1 -> 2 -> 3 ...
        Sequence: [node_i, node_i+1, ?] -> Target: node_i+2
        """
        data = []
        targets = []
        
        # Create a "World Model" (A long chain or tree)
        # Random linear chain: node_0 -> node_1 -> ... -> node_k
        chain = list(range(n_nodes))
        random.shuffle(chain)
        
        for _ in range(n_samples):
            # Pick a start point
            start = random.randint(0, n_nodes - 3)
            # Seq: node[i], node[i+1]
            seq = [chain[start], chain[start+1]]
            target = chain[start+2]
            
            data.append(seq)
            targets.append(target)
            
        return torch.tensor(data, dtype=torch.long), torch.tensor(targets, dtype=torch.long)
    
    def generate_symmetric_data(self, n_samples=1000, n_clusters=5):
        """
        Generates equivalence data: A ~ B.
        If we see A, we expect B (or any member of A's cluster).
        Cluster 1: {0, 1, 2}
        Cluster 2: {3, 4, 5}
        Seq: [0] -> Target: 1 or 2 (Randomly)
        """
        data = []
        targets = []
        
        vocab_per_cluster = self.vocab_size // n_clusters
        clusters = []
        for i in range(n_clusters):
            clusters.append(list(range(i*vocab_per_cluster, (i+1)*vocab_per_cluster)))
            
        for _ in range(n_samples):
            c_idx = random.randint(0, n_clusters-1)
            cluster = clusters[c_idx]
            
            item = random.choice(cluster)
            target = random.choice(cluster)
            # Don't predict self? Or do? Symmetry allows reflexive.
            # Let's avoid trivial self-prediction
            while target == item and len(cluster) > 1:
                target = random.choice(cluster)
                
            data.append([item])
            targets.append(target)
            
        return torch.tensor(data, dtype=torch.long), torch.tensor(targets, dtype=torch.long)

if __name__ == "__main__":
    # Test
    gen = LogicDatasetGenerator(vocab_size=100)
    x, y = gen.generate_cyclic_data(5)
    print("Cyclic:", x, "->", y)
    
    x, y = gen.generate_transitive_data(5)
    print("Transitive:", x, "->", y)
    
    x, y = gen.generate_symmetric_data(5)
    print("Symmetric:", x, "->", y)
