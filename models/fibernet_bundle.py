
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NFBTConnection(nn.Module):
    """
    显式联络与平行移动算子 (Explicit Connection & Parallel Transport)
    实现 \nabla。引入李群指数映射 (Lie Group Exponential Map) 确保正交性。
    """
    def __init__(self, d_logic, d_memory):
        super().__init__()
        self.d_logic = d_logic
        self.d_memory = d_memory
        
        # 联络映射：从底流形位移映射到李代数 so(n)
        # 反对称矩阵空间维度为 n(n-1)/2
        self.so_dim = d_memory * (d_memory - 1) // 2
        self.gamma_net = nn.Sequential(
            nn.Linear(d_logic, d_logic * 2),
            nn.Tanh(), # 保证李代数元素的稳定性
            nn.Linear(d_logic * 2, self.so_dim)
        )
        
    def _to_skew_symmetric(self, vec):
        """将向量映射为反对称矩阵 (Skew-Symmetric Matrix)"""
        batch_size = vec.shape[0]
        # 初始化全零矩阵
        skew = torch.zeros(batch_size, self.d_memory, self.d_memory, device=vec.device)
        
        # 填充上三角并镜像到下三角 (使用反对称约束)
        tri_indices = torch.triu_indices(self.d_memory, self.d_memory, offset=1)
        # 展开 tri_indices 以便批处理应用
        # 这里使用简单循环或广播技巧
        # 对于 AGI 项目，我们追求代码的可读性与正确性
        idx_row, idx_col = tri_indices
        skew[:, idx_row, idx_col] = vec
        skew[:, idx_col, idx_row] = -vec
        return skew

    def forward(self, delta_x, fiber_v):
        """
        Returns: transported_v, T_matrix
        """
        # 1. 计算位移诱导的李代数参数
        lie_params = self.gamma_net(delta_x.reshape(-1, self.d_logic))
        skew_mat = self._to_skew_symmetric(lie_params)
        
        # 2. 指数映射：T = exp(Omega) \in SO(n)
        # 保证变换矩阵的正交性，实现保度量平移
        T = torch.matrix_exp(skew_mat)
        
        # 3. 执行平移 (施加算子 T 到纤维向量 v)
        # fiber_v: [Total, 1, d_memory]
        # T: [Total, d_mem, d_mem]
        # out = T @ v.T
        transported_v = torch.matmul(T, fiber_v.transpose(-2, -1)).transpose(-2, -1)
        return transported_v, T

class FiberBundleAttention(nn.Module):
    """
    几何化注意力机制：引入李群轨道平移与轨迹分析
    """
    def __init__(self, d_logic, d_memory, nhead=2):
        super().__init__()
        self.d_logic = d_logic
        self.d_memory = d_memory
        
        self.W_Q = nn.Linear(d_logic, d_logic)
        self.W_K = nn.Linear(d_logic, d_logic)
        
        self.connection = NFBTConnection(d_logic, d_memory)
        self.out_proj = nn.Linear(d_memory, d_memory)
        
    def forward(self, x_logic, x_memory):
        batch, seq, _ = x_logic.shape
        
        # 1. Attention Weights (Path Probabilities)
        Q = self.W_Q(x_logic)
        K = self.W_K(x_logic)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_logic)
        attn_weights = torch.softmax(attn_logits, dim=-1) # [B, Seq, Seq]
        
        # 2. Parallel Transport via Exponential Map
        x_j = x_logic.unsqueeze(2) 
        x_i = x_logic.unsqueeze(1) 
        delta_x_all = x_j - x_i # [B, Seq, Seq, D_logic]
        
        v_i = x_memory.unsqueeze(1).expand(-1, seq, -1, -1)
        
        # 展平处理
        delta_x_flat = delta_x_all.reshape(-1, 1, self.d_logic)
        v_i_flat = v_i.reshape(-1, 1, self.d_memory)
        
        # 得到正交变换后的纤维内容
        transported_v_flat, T_all_flat = self.connection(delta_x_flat, v_i_flat)
        transported_v_all = transported_v_flat.reshape(batch, seq, seq, self.d_memory)
        
        # 3. 联络结算 (Section Settlement)
        # \sigma(x_j) = \sum_i \alpha_{ij} * T_{i \to j}(\sigma(x_i))
        settled_v = torch.einsum('bji, bjid -> bjd', attn_weights, transported_v_all)
        
        # 暴露 T_all 供曲率计算/测地线约束 (Shape: [B, Seq, Seq, D_m, D_m])
        T_all = T_all_flat.reshape(batch, seq, seq, self.d_memory, self.d_memory)
        
        return self.out_proj(settled_v), T_all

class NFBTLayer(nn.Module):
    def __init__(self, d_logic, d_memory, nhead=2):
        super().__init__()
        self.bundle_attn = FiberBundleAttention(d_logic, d_memory, nhead)
        self.logic_update = nn.Sequential(
            nn.Linear(d_logic, d_logic * 2),
            nn.Tanh(),
            nn.Linear(d_logic * 2, d_logic)
        )
        self.norm_l = nn.LayerNorm(d_logic)
        self.norm_m = nn.LayerNorm(d_memory)
        
    def forward(self, x_logic, x_memory):
        # 1. Base Manifold Propagation
        res_l = x_logic
        x_logic = self.norm_l(res_l + self.logic_update(x_logic))
        
        # 2. Connection-based Transport
        res_m = x_memory
        transported_m, T_matrices = self.bundle_attn(x_logic, x_memory)
        
        x_memory = self.norm_m(res_m + transported_m)
        
        return x_logic, x_memory, T_matrices

class FiberBundleNetwork(nn.Module):
    """
    具有李群约束的纤维丛网络 (Pure NFBT 2.0 Architecture)
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
            x_logic, x_memory, _ = layer(x_logic, x_memory)
            
        return self.head(x_memory)
