
import argparse
import os
import random
from typing import List, Tuple

import networkx as nx


def generate_cyclic_group(n_samples: int, p: int = 113) -> List[str]:
    """
    Generates addition modulo p: "x + y = z"
    represented as: "ZN_ADD ARG_x ARG_y EQRES RES_z"
    Using abstract tokens to avoid number bias.
    """
    data = []
    for _ in range(n_samples):
        a = random.randint(0, p - 1)
        b = random.randint(0, p - 1)
        c = (a + b) % p
        # Format: ZN_ADD SYM_{a} SYM_{b} EQ SYM_{c}
        line = f"ZN_ADD SYM_{a} SYM_{b} EQ SYM_{c}"
        data.append(line)
    return data

def generate_permutation_group(n_samples: int, n: int = 5) -> List[str]:
    """
    Generates composition of permutations S_n.
    "p1 o p2 = p3"
    Permutations are represented as tuples, e.g., (1,0,2).
    """
    import itertools
    base_perms = list(itertools.permutations(range(n)))
    # We might not want to use all n! permutations if n is large, but for n=5, 120 is small.
    # For S_5, |S_5| = 120. We can sample random pairs.
    
    data = []
    for _ in range(n_samples):
        # Pick two random permutations
        p1 = random.choice(base_perms)
        p2 = random.choice(base_perms)
        
        # Composition p3 = p1(p2(x)) -> p3[i] = p1[p2[i]]
        # Check composition order definition. Usually (p1 o p2)(i) = p1(p2(i))
        p3 = tuple(p1[p2[i]] for i in range(n))
        
        # Tokenize: PERM_{p1} COMPOSE PERM_{p2} EQ PERM_{p3}
        # To make it readable/tokenizable, we can stringify the tuple: 102
        s1 = "".join(map(str, p1))
        s2 = "".join(map(str, p2))
        s3 = "".join(map(str, p3))
        
        line = f"SN_COMPOSE PERM_{s1} PERM_{s2} EQ PERM_{s3}"
        data.append(line)
    return data

def generate_transitive_logic(n_samples: int, n_nodes: int = 20, max_depth: int = 3) -> List[str]:
    """
    Generates reduced transitive inference chains.
    A -> B, B -> C, therefore A -> C.
    
    We generate a random DAG, pick two connected nodes, and provide the path as premises.
    """
    data = []
    
    # Pool of node names
    nodes = [f"NODE_{i}" for i in range(n_nodes)]
    
    count = 0
    while count < n_samples:
        # Create a small random chain length 2 to max_depth
        length = random.randint(2, max_depth)
        
        # Pick distinct nodes
        chain_nodes = random.sample(nodes, length + 1)
        
        # Construct premises: A->B, B->C, ...
        premises = []
        for i in range(length):
            premises.append(f"{chain_nodes[i]} IMPLIES {chain_nodes[i+1]}")
            
        # Conclusion: Start -> End
        conclusion = f"{chain_nodes[0]} IMPLIES {chain_nodes[-1]}"
        
        # Shuffle premises? Standard logic usually allows any order, but for initial curriculum fixed order might be easier.
        # Let's simple concat.
        
        # Format: PREMISE [A->B] [B->C] CONCLUSION [A->C]
        line = "TRANS_LOGIC " + " AND ".join(premises) + " THEREFORE " + conclusion
        data.append(line)
        count += 1
        
    return data

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic logic corpus")
    parser.add_argument("--output_dir", type=str, default="data/logic_core", help="Output directory")
    parser.add_argument("--zn_samples", type=int, default=10000, help="Number of Z_n samples")
    parser.add_argument("--sn_samples", type=int, default=10000, help="Number of S_n samples")
    parser.add_argument("--trans_samples", type=int, default=10000, help="Number of Transitive samples")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Cyclic Group
    print(f"Generating Z_113 data ({args.zn_samples} samples)...")
    zn_data = generate_cyclic_group(args.zn_samples, p=113)
    
    # 2. Permutation Group (S_5)
    print(f"Generating S_5 data ({args.sn_samples} samples)...")
    sn_data = generate_permutation_group(args.sn_samples, n=5)
    
    # 3. Transitive Logic
    print(f"Generating Transitive Logic data ({args.trans_samples} samples)...")
    trans_data = generate_transitive_logic(args.trans_samples, n_nodes=50)
    
    # Combine and Shuffle
    all_data = zn_data + sn_data + trans_data
    random.shuffle(all_data)
    
    output_path = os.path.join(args.output_dir, "logic_corpus_v1.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in all_data:
            f.write(line + "\n")
            
    print(f"Saved {len(all_data)} samples to {output_path}")

if __name__ == "__main__":
    main()
