
import numpy as np
import torch
from torch.utils.data import Dataset


class GroupTheoryDataset(Dataset):
    """
    Synthetic dataset for group operations.
    Generates (a, b, result) tuples for a given finite group.
    
    Supported Groups:
    - 'Z_n': Cyclic group of order n (Modular Addition).
    - 'S_n': Symmetric group of degree n (Permutations) - TODO
    """
    def __init__(self, group_type='Z_n', order=113, num_samples=10000, seed=42):
        super().__init__()
        self.group_type = group_type
        self.order = order
        self.num_samples = num_samples
        self.seed = seed
        
        self.data = self._generate_data()
        
    def _generate_data(self):
        torch.manual_seed(self.seed)
        data = []
        
        if self.group_type == 'Z_n':
            # Generate random pairs (a, b)
            a = torch.randint(0, self.order, (self.num_samples,))
            b = torch.randint(0, self.order, (self.num_samples,))
            # Operation: (a + b) % n
            res = (a + b) % self.order
            
            # Form input sequence: [a, op_token, b, eq_token] -> target: [res]
            # For simplicity in this toy model, we just return (a, b, res)
            # The model will receive embeddings of a and b.
            # In a real sequence model, we would add special tokens.
            return torch.stack([a, b, res], dim=1)
            
        elif self.group_type == 'S_n':
            raise NotImplementedError("S_n not yet implemented.")
            
        return torch.tensor(data)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Return x (input) and y (target)
        # x: [a, b]
        # y: [res]
        row = self.data[idx]
        return row[:2], row[2]

# Test usage
if __name__ == "__main__":
    ds = GroupTheoryDataset(order=97, num_samples=5)
    print("Sample data (Z_97):")
    for i in range(len(ds)):
        x, y = ds[i]
        print(f"Input: {x.tolist()}, Target: {y.item()} | Check: {(x[0]+x[1])%97 == y}")
