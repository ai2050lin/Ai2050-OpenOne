
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NFBTConnection(nn.Module):
    """
    显式联络与平行移动算子 (Explicit Connection & Parallel Transport)
    实现 \nabla。根据底流形坐标偏移 \Delta x 计算纤维变换 T_{i->j}。
    """
    def __init__(self, d_logic, d_memory):
        super().__init__()
        self.d_logic = d_logic
        self.d_memory = d_memory
        
        # 联络映射：从底流形位移映射到纤维变换群的李代数
        # 这里使用一个小型网络来模拟联络系数 \Gamma
        self.gamma_net = nn.Sequential(
            nn.Linear(d_logic, d_logic * 2),
            nn.ReLU(),
            nn.Linear(d_logic * 2, d_memory * d_memory)
        )
        
    def forward(self, delta_x, fiber_v):
        """
        Args:
            delta_x: 底流形位移 [Batch, Seq, d_logic]
            fiber_v: 纤维内容 [Batch, Seq, d_memory]
        Returns:
            transported_v: 平行移动后的纤维内容
        """
        batch, seq, _ = delta_x.shape
        
        # 计算变换矩阵 (平行移动算子 T)
        # 在真实的纤维丛中，这应该是 exp(\int \Gamma dx) 的离散近似
        transform_mat = self.gamma_net(delta_x).reshape(batch, seq, self.d_memory, self.d_memory)
        
        # 施加变换：T(v)
        # 这里使用 batch 矩阵乘法
        transported_v = torch.matmul(transform_mat, fiber_v.unsqueeze(-1)).squeeze(-1)
        
        return transported_v

class FiberBundleAttention(nn.Module):
    """
    几何化注意力机制：基于联络的截面平移
    它不仅是加权和，而是沿联络平移后的截面积分。
    """
    def __init__(self, d_logic, d_memory, nhead=2):
        super().__init__()
        self.d_logic = d_logic
        self.d_memory = d_memory
        self.nhead = nhead
        
        # 逻辑投影 (用于计算注意力权重，即路径贡献)
        self.W_Q = nn.Linear(d_logic, d_logic)
        self.W_K = nn.Linear(d_logic, d_logic)
        
        # 几何联络
        self.connection = NFBTConnection(d_logic, d_memory)
        self.out_proj = nn.Linear(d_memory, d_memory)
        
    def forward(self, x_logic, x_memory):
        batch, seq, _ = x_logic.shape
        
        # 1. 计算注意力权重 (Path Selection)
        Q = self.W_Q(x_logic)
        K = self.W_K(x_logic)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_logic)
        attn_weights = torch.softmax(attn_logits, dim=-1) # [B, Seq, Seq]
        
        # 2. 显式平行移动 (Parallel Transport)
        # 对于每一对 (i, j)，我们计算位移 delta_x = x_j - x_i
        # 然后计算 transported_v_{i->j}
        
        # 广播计算所有对之间的位移
        x_j = x_logic.unsqueeze(2) # [B, Seq, 1, D]
        x_i = x_logic.unsqueeze(1) # [B, 1, Seq, D]
        delta_x_all = x_j - x_i    # [B, Seq, Seq, D]
        
        # 获取纤维内容并在所有路径上准备平移
        v_i = x_memory.unsqueeze(1).expand(-1, seq, -1, -1) # [B, Seq, Seq, D_mem]
        
        # 展平以进行批处理联络运算
        delta_x_flat = delta_x_all.reshape(-1, 1, self.d_logic)
        v_i_flat = v_i.reshape(-1, 1, self.d_memory)
        
        # 执行平行移动
        transported_v_flat = self.connection(delta_x_flat, v_i_flat)
        transported_v_all = transported_v_flat.reshape(batch, seq, seq, self.d_memory)
        
        # 3. 联络结算 (Section Settlement)
        # \sigma(x_j) = \sum_i \alpha_{ij} * T_{i->j}(\sigma(x_i))
        # attn_weights shape: [B, Seq, Seq_i] -> 我们需要作用在 i 维上
        settled_v = torch.einsum('bji, bjid -> bjd', attn_weights, transported_v_all)
        
        return self.out_proj(settled_v)

class NFBTLayer(nn.Module):
    def __init__(self, d_logic, d_memory, nhead=2):
        super().__init__()
        self.bundle_attn = FiberBundleAttention(d_logic, d_memory, nhead)
        self.logic_update = nn.Sequential(
            nn.Linear(d_logic, d_logic * 2),
            nn.ReLU(),
            nn.Linear(d_logic * 2, d_logic)
        )
        self.norm_l = nn.LayerNorm(d_logic)
        self.norm_m = nn.LayerNorm(d_memory)
        
    def forward(self, x_logic, x_memory):
        # 1. 底流形演化 (Propagation on M)
        res_l = x_logic
        x_logic = self.norm_l(res_l + self.logic_update(x_logic))
        
        # 2. 纤维丛联络结算 (Transport on E)
        res_m = x_memory
        transported_m = self.bundle_attn(x_logic, x_memory)
        x_memory = self.norm_m(res_m + transported_m)
        
        return x_logic, x_memory

class FiberBundleNetwork(nn.Module):
    """
    真正的纤维丛网络实现 (Pure NFBT Architecture)
    """
    def __init__(self, vocab_size, d_logic=16, d_memory=64, n_layers=3):
        super().__init__()
        self.d_logic = d_logic
        self.d_memory = d_memory
        
        # 初始化：底流形映射坐标与初始纤维截面
        self.logic_init = nn.Embedding(vocab_size, d_logic)
        self.fiber_init = nn.Embedding(vocab_size, d_memory)
        
        self.layers = nn.ModuleList([
            NFBTLayer(d_logic, d_memory) for _ in range(n_layers)
        ])
        
        self.head = nn.Linear(d_memory, vocab_size)
        
    def forward(self, input_ids):
        # 1. 初始截面与坐标投影
        x_logic = self.logic_init(input_ids)
        x_memory = self.fiber_init(input_ids)
        
        # 2. 几何演化
        for layer in self.layers:
            x_logic, x_memory = layer(x_logic, x_memory)
            
        return self.head(x_memory)
