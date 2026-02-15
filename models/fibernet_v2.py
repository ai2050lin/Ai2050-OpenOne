
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LieGroupEmbedding(nn.Module):
    """
    几何先验嵌入层 (Physics-Informed Geometric Embedding)
    Same as before: learns phases for circle group.
    """
    def __init__(self, num_embeddings, embedding_dim, group_type='circle'):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.group_type = group_type
        
        if group_type == 'circle':
            assert embedding_dim % 2 == 0
            self.phases = nn.Parameter(torch.rand(num_embeddings, embedding_dim // 2) * 2 * math.pi)
        else:
            self.weight = nn.Parameter(torch.randn(num_embeddings, embedding_dim))

    def forward(self, input):
        if self.group_type == 'circle':
            theta = F.embedding(input, self.phases)
            cos_term = torch.cos(theta)
            sin_term = torch.sin(theta)
            out = torch.stack([cos_term, sin_term], dim=-1) 
            out = out.flatten(start_dim=-2) 
            return out
        else:
            return F.embedding(input, self.weight)

class LogicDrivenAttention(nn.Module):
    """
    核心组件：逻辑驱动注意力
    Logic Stream 提供 Q, K (决定谁看谁)
    Memory Stream 提供 V (被搬运的内容)
    """
    def __init__(self, d_logic, d_memory, nhead=2):
        super().__init__()
        self.d_logic = d_logic
        self.d_memory = d_memory
        self.nhead = nhead
        
        # Structure projections (Logic)
        self.W_Q = nn.Linear(d_logic, d_logic)
        self.W_K = nn.Linear(d_logic, d_logic)
        
        # Content projections (Memory)
        self.W_V = nn.Linear(d_memory, d_memory)
        self.W_O = nn.Linear(d_memory, d_memory)
        
    def forward(self, x_logic, x_memory, mask=None):
        batch, seq, _ = x_logic.shape
        head_dim_logic = self.d_logic // self.nhead
        head_dim_memory = self.d_memory // self.nhead
        
        # 1. Compute Structure (Attention Matrix) from Logic
        Q = self.W_Q(x_logic).reshape(batch, seq, self.nhead, head_dim_logic).transpose(1, 2)
        K = self.W_K(x_logic).reshape(batch, seq, self.nhead, head_dim_logic).transpose(1, 2)
        
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(head_dim_logic)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = torch.softmax(scores, dim=-1) # [batch, head, seq, seq]
        
        # 2. Apply Structure to Content (Transport Memory)
        V = self.W_V(x_memory).reshape(batch, seq, self.nhead, head_dim_memory).transpose(1, 2)
        
        # Transport!
        # logic says "Pos 2 look at Pos 1" (Attn Matrix)
        # memory at Pos 2 becomes weighted sum of memory at Pos 1
        output = attn @ V # [batch, head, seq, dim_head_mem]
        
        output = output.transpose(1, 2).flatten(2)
        return self.W_O(output)

class FiberLayer(nn.Module):
    def __init__(self, d_logic, d_memory, nhead=2, dim_feedforward=128):
        super().__init__()
        self.attn = LogicDrivenAttention(d_logic, d_memory, nhead)
        
        # Logic update (Self-conatined? No, usually coupled)
        # To simplify, we let logic evolve independently via standard transformer layer elsewhere?
        # OR we package it here.
        # Let's package: Logic evolves by itself. Memory evolves by Logic.
        
        # Logic Self-Attn (Standard)
        self.logic_attn = nn.MultiheadAttention(d_logic, nhead, batch_first=True)
        self.logic_norm1 = nn.LayerNorm(d_logic)
        self.logic_ffn = nn.Sequential(
            nn.Linear(d_logic, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_logic)
        )
        self.logic_norm2 = nn.LayerNorm(d_logic)
        
        # Memory Cross-Attn (Logic-Driven)
        self.mem_norm1 = nn.LayerNorm(d_memory)
        self.mem_ffn = nn.Sequential(
            nn.Linear(d_memory, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, d_memory)
        )
        self.mem_norm2 = nn.LayerNorm(d_memory)
        
    def forward(self, x_logic, x_memory, src_mask=None):
        # 1. Evolve Logic (Standard Transformer Step)
        # Logic only cares about itself (and position)
        res_l = x_logic
        x_logic, _ = self.logic_attn(x_logic, x_logic, x_logic, attn_mask=src_mask)
        x_logic = self.logic_norm1(res_l + x_logic)
        
        res_l = x_logic
        x_logic = self.logic_ffn(x_logic)
        x_logic = self.logic_norm2(res_l + x_logic)
        
        # 2. Evolve Memory (Driven by Logic)
        # Note: We use the *updated* logic to drive memory, or previous?
        # Let's use updated, akin to "current layer's structure"
        res_m = x_memory
        # LogicDrivenAttention calculates Q, K from x_logic, and V from x_memory
        transported = self.attn(x_logic, x_memory) 
        x_memory = self.mem_norm1(res_m + transported)
        
        res_m = x_memory
        x_memory = self.mem_ffn(x_memory)
        x_memory = self.mem_norm2(res_m + x_memory)
        
        return x_logic, x_memory

class DecoupledFiberNet(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, group_type='circle', max_len=512):
        super().__init__()
        self.d_model = d_model
        
        # 1. Logic Stream
        self.d_logic = d_model // 2
        self.pos_embed = nn.Embedding(max_len, self.d_logic)
        
        # 2. Memory Stream
        self.d_memory = d_model
        self.content_embed = LieGroupEmbedding(vocab_size, self.d_memory, group_type=group_type)
        
        # 3. Layers
        self.layers = nn.ModuleList([
            FiberLayer(self.d_logic, self.d_memory, nhead=2, dim_feedforward=128)
            for _ in range(n_layers)
        ])
        
        self.head = nn.Linear(self.d_memory, vocab_size, bias=False)
        
    def forward(self, input_ids=None, memory_embeds=None):
        """
        Args:
            input_ids: [Batch, Seq] - Standard token inputs
            memory_embeds: [Batch, Seq, D_Memory] - Optional override for memory stream (e.g. from Vision)
        """
        device = input_ids.device if input_ids is not None else memory_embeds.device
        
        # Determine Batch and Seq
        if input_ids is not None:
            batch, seq = input_ids.shape
        else:
            batch, seq, _ = memory_embeds.shape
        
        # Init Logic (Pos only)
        positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        curr_logic = self.pos_embed(positions)
        
        # Init Memory
        if memory_embeds is not None:
            curr_memory = memory_embeds
        else:
            curr_memory = self.content_embed(input_ids)
        
        # Evolve
        for layer in self.layers:
            curr_logic, curr_memory = layer(curr_logic, curr_memory)
            
        return self.head(curr_memory)
