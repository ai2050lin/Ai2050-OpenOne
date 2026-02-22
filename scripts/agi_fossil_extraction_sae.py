import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, d_model, d_sae, l1_coeff=1e-3):
        """
        初始化高维稀疏字典提纯器 (SAE)
        
        d_model: 化石模型密闭特征维度 (e.g. 768)
        d_sae: 十万维甚至百万维的正交特征大厅
        l1_coeff: L1 正则化系数, 逼迫特征稀疏(特异)
        """
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff
        
        # Encoder: 强行拉伸进入超高维
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        
        # Decoder: 解卷重建基底 (需要约束 Norm=1 以防作弊)
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))
        self.b_dec = nn.Parameter(torch.zeros(d_model))
        
        # 正交初始化
        nn.init.orthogonal_(self.W_enc)
        nn.init.orthogonal_(self.W_dec)

    def forward(self, x):
        """
        前向提纯传播
        x: [Batch, d_model] 拦截到的模型致密残差流或激活
        """
        # 1. 提取高维隔离特征 (只允许极少数点火)
        # 去掉 decoder bias 偏移，进行拉伸
        x_centered = x - self.b_dec
        hidden = F.relu(x_centered @ self.W_enc + self.b_enc)
        
        # 2. 尝试无损折叠回原空间
        x_reconstructed = hidden @ self.W_dec + self.b_dec
        
        # 3. 计算物理代价
        # 重建代价 (必须无损)
        l2_loss = (x_reconstructed - x).pow(2).sum(dim=-1).mean()
        # 稀疏/特异代价 (强迫在高维里正交互斥点火)
        l1_loss = self.l1_coeff * hidden.sum(dim=-1).mean()
        
        loss = l2_loss + l1_loss
        
        return loss, x_reconstructed, hidden, l2_loss, l1_loss
        
    @torch.no_grad()
    def normalize_decoder(self):
        """
        每次反向传播后调用: 强制解码器列向量在高维超球面上(Norm=1)
        约束模型真正拉开向量间的正交角度，而不是靠拉伸长度。
        """
        self.W_dec.data = F.normalize(self.W_dec.data, p=2, dim=1)

def build_fossil_probe_pipeline():
    """
    假想化的挂载管线流程示范
    """
    print("[AGI Foundation] 正在组装化石解剖探针...")
    
    # 模拟从 Llama 或 FiberNet 截获的 4096 维致密语义特征 (Batch 1000)
    d_model_fossil = 4096 
    mock_activations = torch.randn(1000, d_model_fossil) 
    
    # 构建十万维提纯大厅 (简化测试设为 32768)
    d_sae_expansion = 32768 
    sae = SparseAutoencoder(d_model_fossil, d_sae_expansion, l1_coeff=0.01)
    
    print(f"- 化石输入维数: {d_model_fossil}")
    print(f"- 正交超空维数: {d_sae_expansion} (8x 膨胀提纯)")
    
    optimizer = torch.optim.Adam(sae.parameters(), lr=1e-3)
    
    print("\n[AGI Foundation] 启动一阶 L1 解卷洗刷 (Mock Run)...")
    sae.train()
    for step in range(1, 11):
        optimizer.zero_grad()
        loss, recon, hidden, l2, l1 = sae(mock_activations)
        loss.backward()
        optimizer.step()
        
        # 几何外壳强制约束
        sae.normalize_decoder()
        
        if step % 2 == 0:
            active_features = (hidden > 0).float().mean().item()
            print(f"Step {step} | Loss: {loss.item():.2f} | 活跃维度占比: {active_features:.4%} (越低越特异正交)")
            
    print("\n提纯器部署完成。下一步: 将真实语料接通 TransformerLens 监听钩子 (Hooks)。")

if __name__ == "__main__":
    build_fossil_probe_pipeline()
    
