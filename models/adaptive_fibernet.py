"""
Adaptive FiberNet: 从零开始学习几何结构
==========================================

核心改进：
1. 可学习的几何嵌入（不再固定为李群）
2. 自动结构发现模块
3. 渐进式解耦机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LearnableGeometryEmbedding(nn.Module):
    """
    可学习的几何嵌入层
    - 初始化为标准嵌入
    - 逐步学习数据的隐含几何结构
    - 可选：约束为流形结构（如圆环、球面）
    """
    
    def __init__(self, vocab_size, d_model, manifold_type='learnable'):
        super().__init__()
        self.d_model = d_model
        self.manifold_type = manifold_type
        
        # 基础嵌入
        self.base_embedding = nn.Embedding(vocab_size, d_model)
        
        # 几何约束参数
        if manifold_type == 'circle':
            # S^1 圆环：每个 token 对应一个相位角
            self.theta = nn.Parameter(torch.randn(vocab_size, 1) * 2 * math.pi)
            self.radius = nn.Parameter(torch.ones(vocab_size, 1))
        elif manifold_type == 'torus':
            # T^2 环面：两个相位角
            self.theta = nn.Parameter(torch.randn(vocab_size, 2) * 2 * math.pi)
        elif manifold_type == 'sphere':
            # S^2 球面：球坐标
            self.theta = nn.Parameter(torch.randn(vocab_size, 2) * math.pi)
            self.phi = nn.Parameter(torch.randn(vocab_size, 1) * 2 * math.pi)
        else:
            # 纯可学习，无几何约束
            self.geo_embedding = nn.Embedding(vocab_size, d_model)
        
        # 混合权重（控制几何先验的强度）
        self.geo_weight = nn.Parameter(torch.tensor(0.0))  # 初始为0，逐步增加
        
    def forward(self, x):
        base = self.base_embedding(x)
        
        if self.manifold_type == 'learnable':
            geo = self.geo_embedding(x)
            return base + torch.sigmoid(self.geo_weight) * geo
        elif self.manifold_type == 'circle':
            # 圆环嵌入：[cos(θ), sin(θ), ...]
            circle = torch.cat([
                self.radius[x] * torch.cos(self.theta[x]),
                self.radius[x] * torch.sin(self.theta[x])
            ], dim=-1)
            # 扩展到 d_model 维度
            circle = F.pad(circle, (0, self.d_model - circle.size(-1)))
            return base + torch.sigmoid(self.geo_weight) * circle
        else:
            return base


class StructureDiscoveryModule(nn.Module):
    """
    自动结构发现模块
    - 分析激活空间的几何结构
    - 估计内在维度、曲率、群结构
    """
    
    def __init__(self, d_model, n_heads=4):
        super().__init__()
        self.d_model = d_model
        
        # 内在维度估计器
        self.dim_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()  # 输出 0-1 的维度比例
        )
        
        # 群结构检测器
        self.group_detector = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # 结构分类器：判断数据属于哪种几何结构
        self.structure_classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # [none, circular, toroidal, hierarchical]
        )
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        返回: 结构信息字典
        """
        # 估计内在维度比例
        dim_ratio = self.dim_estimator(x.mean(dim=1))  # (batch, 1)
        
        # 检测群结构（通过自注意力模式）
        attn_out, attn_weights = self.group_detector(x, x, x)
        
        # 分类几何结构
        structure_logits = self.structure_classifier(x.mean(dim=1))
        structure_type = structure_logits.argmax(dim=-1)
        
        return {
            'intrinsic_dim_ratio': dim_ratio,
            'attention_pattern': attn_weights,
            'structure_logits': structure_logits,
            'structure_type': structure_type
        }


class ProgressiveDisentangler(nn.Module):
    """
    渐进式解耦模块
    - 训练初期：类似 Transformer 的纠缠模式
    - 训练后期：逐步分离 Logic 和 Memory
    """
    
    def __init__(self, d_model, n_heads, max_epochs=1000):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.max_epochs = max_epochs
        
        # 当前训练进度（从外部设置）
        self.register_buffer('progress', torch.tensor(0.0))
        
        # Logic Stream（纯位置）
        self.logic_pos_embed = nn.Embedding(512, d_model)  # 最大序列长度
        self.logic_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.logic_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.logic_norm = nn.LayerNorm(d_model)
        
        # Memory Stream（内容）
        self.memory_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.memory_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.memory_norm = nn.LayerNorm(d_model)
        
        # 混合门控：控制纠缠 vs 解耦的比例
        self.gate = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, positions=None):
        """
        x: 内容嵌入
        positions: 位置索引
        """
        batch_size, seq_len, _ = x.shape
        
        if positions is None:
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Logic Stream: 仅处理位置
        logic_input = self.logic_pos_embed(positions)
        logic_out, logic_attn = self.logic_attn(logic_input, logic_input, logic_input)
        logic_out = self.logic_norm(logic_input + self.logic_ffn(logic_out))
        
        # Memory Stream: 内容 + 逻辑指导
        # 渐进式：初期使用自己的 Q,K，后期使用 Logic 的注意力
        memory_out, memory_attn = self.memory_attn(x, x, x)
        
        # 关键：使用 Logic 注意力指导 Memory
        # progress 接近 1 时，完全使用 logic_attn
        alpha = self.progress.item()
        
        # 混合注意力
        combined_attn = (1 - alpha) * memory_attn + alpha * logic_attn
        
        # 应用混合注意力到 Memory
        V = x
        combined_out = torch.bmm(combined_attn, V)
        combined_out = self.memory_norm(x + self.memory_ffn(combined_out))
        
        # 门控混合最终输出
        gate_weight = self.gate(x)
        output = gate_weight * logic_out + (1 - gate_weight) * combined_out
        
        return output, logic_attn, memory_attn, alpha
    
    def update_progress(self, epoch):
        """更新训练进度"""
        self.progress.fill_(min(epoch / self.max_epochs, 1.0))


class AdaptiveFiberNetBlock(nn.Module):
    """
    自适应 FiberNet 块
    - 集成结构发现和渐进解耦
    """
    
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model * 4
        
        self.d_model = d_model
        
        # 自注意力（标准 Transformer 风格）
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer Norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # 结构发现
        self.structure_discovery = StructureDiscoveryModule(d_model, n_heads)
        
        # 渐进解耦（仅在检测到结构后激活）
        self.disentangler = ProgressiveDisentangler(d_model, n_heads)
        
        # 解耦开关
        self.use_disentangle = False
        
    def forward(self, x, use_structure=True):
        # 标准自注意力
        attn_out, attn_weights = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        
        # FFN
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        if use_structure:
            # 结构发现
            struct_info = self.structure_discovery(x)
            
            # 如果检测到几何结构，启用解耦
            if struct_info['structure_type'].any() > 0:
                self.use_disentangle = True
            
            # 渐进解耦
            if self.use_disentangle:
                x, logic_attn, mem_attn, alpha = self.disentangler(x)
                struct_info['disentangle_alpha'] = alpha
                struct_info['logic_attn'] = logic_attn
                struct_info['mem_attn'] = mem_attn
            
            return x, struct_info
        
        return x, {}


class AdaptiveFiberNet(nn.Module):
    """
    完整的自适应 FiberNet
    - 可以从零开始学习
    - 自动发现并利用几何结构
    """
    
    def __init__(self, vocab_size, d_model=256, n_heads=8, n_layers=6, 
                 max_seq_len=512, dropout=0.1, manifold_type='learnable'):
        super().__init__()
        
        self.d_model = d_model
        
        # 可学习几何嵌入
        self.embedding = LearnableGeometryEmbedding(vocab_size, d_model, manifold_type)
        
        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        # 堆叠的自适应块
        self.layers = nn.ModuleList([
            AdaptiveFiberNetBlock(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        # 输出头
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        # 结构统计
        self.structure_stats = {
            'discovered_structures': [],
            'intrinsic_dims': [],
            'disentangle_progress': []
        }
        
    def forward(self, input_ids, return_structure=True):
        """
        input_ids: (batch, seq_len)
        """
        batch_size, seq_len = input_ids.shape
        
        # 嵌入
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :seq_len, :]
        
        # 逐层处理
        structure_info_all = []
        for layer in self.layers:
            x, struct_info = layer(x, use_structure=return_structure)
            if struct_info:
                structure_info_all.append(struct_info)
        
        # 输出
        x = self.output_norm(x)
        logits = self.output_proj(x)
        
        if return_structure and structure_info_all:
            return logits, structure_info_all
        return logits
    
    def update_disentangle_progress(self, epoch, max_epochs=1000):
        """更新所有层的解耦进度"""
        for layer in self.layers:
            layer.disentangler.update_progress(epoch)


# ============ 测试代码 ============

def test_adaptive_fibernet():
    """测试自适应 FiberNet"""
    
    # 配置
    vocab_size = 1000
    d_model = 128
    n_heads = 4
    n_layers = 2
    
    # 创建模型
    model = AdaptiveFiberNet(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        manifold_type='learnable'  # 从零学习
    )
    
    # 测试输入
    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # 前向传播
    logits, structure_info = model(input_ids)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {logits.shape}")
    print(f"\n结构信息层数: {len(structure_info)}")
    
    for i, info in enumerate(structure_info):
        print(f"\nLayer {i}:")
        print(f"  内在维度比例: {info['intrinsic_dim_ratio'].mean():.3f}")
        print(f"  结构类型: {info['structure_type']}")
        if 'disentangle_alpha' in info:
            print(f"  解耦程度: {info['disentangle_alpha']:.3f}")
    
    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


if __name__ == "__main__":
    test_adaptive_fibernet()
