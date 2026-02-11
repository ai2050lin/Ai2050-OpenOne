
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.fiber_net import ConnectionLayer, CurvatureMonitor, FiberBundle


class ToyFiberNet(nn.Module):
    """
    Simplified FiberNet for Toy Experiment (e.g., Modular Addition Z_n).
    
    Structure:
    - Manifold Stream: Learns the "operator" logic (e.g., +).
      - Input: [SEQ_START, OP_ADD, SEQ_END] 
      - But for this simple task, we can just use a constant apprisal vector for "+".
    - Fiber Stream: Learns the "operand" content (e.g., 5, 7).
    
    Forward Pass:
    1. Embed inputs a, b into Fiber Space F_a, F_b.
    2. Embed operator '+' into Manifold Space M_op.
    3. Connection: Transport F_a along M_op to interact with F_b?
       Actually, group addition a+b=c can be viewed as:
       Starting at Identity '0' (or 'a'), move by 'b' along the manifold geodesics defined by '+'.
       
       Let's model it as:
       - Initial State: Fiber(a) at Manifold(pos_1)
       - Action: Move by Fiber(b) direction
       - Connection A_mu: Should rotate Fiber(a) by angle(b) to get Fiber(a+b).
    
    Wait, the `ConnectionLayer` in `fiber_net.py` is attention-based:
    Output = Sum(Softmax(Q*K) * V)
    
    For Z_n addition (cyclic group), the "transport" is a rotation.
    If we use complex number representation (RoPE-like), a+b corresponds to multiplying exp(i*theta_a) * exp(i*theta_b).
    
    Let's see if FiberNet can learn to "select" the correct result from the vocabulary
    using the Connection mechanism without explicit complex multiplication hardcoding.
    
    We will strictly use the FiberBundle class structure.
    """
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # We need a 'structure' vocab (operators) and 'content' vocab (numbers)
        # For Z_n, content is 0..n-1. Structure could be just a dummy 'ADD' token.
        self.structure_vocab_size = 2 # 0: PAD, 1: ADD
        
        self.fiber_net = FiberBundle(
            structure_vocab=self.structure_vocab_size,
            content_vocab=self.vocab_size,
            d_manifold=d_model,
            d_fiber=d_model * 4, # Make fiber larger to hold semantic content
            max_len=3 # [a, b, result]
        )
        
    def forward(self, a_idx, b_idx):
        # Construct input sequences
        # Manifold: [ADD, ADD] (The operation is constant)
        batch_size = a_idx.size(0)
        structure_ids = torch.ones(batch_size, 2, dtype=torch.long, device=a_idx.device) # [ADD, ADD]
        
        # Fiber: [a, b]
        content_ids = torch.stack([a_idx, b_idx], dim=1)
        
        # Forward pass through FiberBundle
        # Returns: logits [batch, seq_len, vocab]
        logits, transported, manifold_states = self.fiber_net(structure_ids, content_ids)
        
        # We want to predict the *next* token, or the result.
        # Let's say we are predicting 'c' from 'a' and 'b'.
        # The ConnectionLayer mixes information.
        # The last output state should contain the result.
        
        # We'll take the global mean or just the last token's output as the prediction for 'c'
        # Actually standard autoregressive:
        # Input: a, b
        # Target: c
        # The model outputs a sequence. The last vector is used to predict 'c'.
        
        last_token_logits = logits[:, -1, :] # [batch, vocab]
        return last_token_logits

class ToyTransformer(nn.Module):
    """
    Standard small Transformer for baseline comparison.
    """
    def __init__(self, vocab_size, d_model=64, n_head=4, n_layer=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 3, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model*4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, a_idx, b_idx):
        x = torch.stack([a_idx, b_idx], dim=1) # [batch, 2]
        emb = self.embedding(x) + self.pos_embed[:, :2, :]
        out = self.transformer(emb)
        # Take mean or last? Let's take last.
        last = out[:, -1, :]
        return self.fc_out(last)
