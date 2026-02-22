import torch
import torch.nn.functional as F
import ssl
import torchvision
from torchvision import transforms
import numpy as np
import os
import matplotlib.pyplot as plt

ssl._create_default_https_context = ssl._create_unverified_context

class AGIMotherEngineV1:
    """
    全息物理母体引擎 v1.0
    彻底抛弃反向传播与监督学习矩阵，基于四大微观数学法则构建：
    1. 孤立表征提取 (WTA 侧抑制与正交化生长)
    2. 知识图谱自组织 (LTP Hebbian 引力降维)
    3. 稳态修剪 (LTD 物理衰败与 GC 自洽稀疏)
    4. 能量回音流水 (脉冲势能工作流)
    """
    def __init__(self, input_dim=784, represent_dim=100, memory_dim=10000):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[AGI Mother] Booting on {self.device}")
        
        # 1. 第一阶感受簇矩阵 (从感官层 784 提取 represent_dim 维特异性概念)
        # 模拟：通过侧抑制压榨出的独立神经受体
        # 初始化为空白神经原
        self.W_receptors = torch.randn(represent_dim, input_dim, device=self.device) * 0.01
        
        # 2. 第二阶高层常识张量图谱 (联通 represent_dim 个概念，而不是直接10000)
        # 用稠密矩阵初始化，通过 LTD 自发退化为稀疏图结构来模拟突触引力场 (P_topo)
        self.P_topo = torch.zeros(represent_dim, represent_dim, device=self.device)
        
        # 学习率与物理衰减半衰期
        self.lr_receptor = 0.05
        self.lateral_inhibition_strength = 0.5 
        
        # 图谱生长系数
        self.lr_topo = 0.005
        self.decay_rate = 0.001       # 每次迭代的神经半衰期
        self.prune_threshold = 0.01   # 神经斩断死亡红线 

    def wash_receptors_with_sanger(self, batch_x):
        """
        大名鼎鼎的微观方程 01: Sanger 广义 Hebbian 学习法则 + 侧抑制
        用于在白板连接上，自动冲刷出互相正交的独立视网膜皮层 V1 特征
        """
        batch_size = batch_x.size(0)
        # 前向激活
        y = torch.mm(batch_x, self.W_receptors.t()) # [B, represent_dim]
        
        # Oja + Sanger 侧向抑制矩阵算符
        # 对每一个输出神经元产生侧向压制，强迫它们去认领不同的像素分布，而不是全部扎堆在同一个亮斑
        for i in range(batch_size):
            x_i = batch_x[i].unsqueeze(0) # [1, input_dim]
            y_i = y[i].unsqueeze(1) # [represent_dim, 1]
            
            # Hebbian 生长项: x * y
            # 基础 Oja 衰减项: y^2 * W
            # Sanger 侧抑制: 让排名在前的神经元抑制排名在后的神经元的学习
            
            # 为加速矩阵运算，使用近似局部侧向罚项
            # 对 y 的放电能量产生 WTA 竞争激活
            
            # 使用简单的 Hebb - Oja
            W_update = torch.mm(y_i, x_i) # [represent_dim, input_dim]
            
            # 添加下三角的侧抑制 (逼迫基底正交，这正是涌现的唯一源泉)
            y_tril = torch.tril(torch.mm(y_i, y_i.t())) # [represent_dim, represent_dim]
            inhibition = torch.mm(y_tril, self.W_receptors)
            
            self.W_receptors += self.lr_receptor * (W_update - inhibition) / batch_size

    def wash_topology_with_hebbian(self, batch_x):
        """
        大名鼎鼎的微观方程 02 & 03: Hebbian LTP 与 LTD 物理稳态修剪
        用于在巨大的潜空间中，将经常一起点亮的孤立特征进行“引力绑定”，从而长出“常识图谱”
        """
        # 1. 取得当前感官输入在第一阶感受器上的激活模式
        energy_spikes = F.relu(torch.mm(batch_x, self.W_receptors.t())) # [B, represent_dim]
        
        # 过滤掉微弱噪声，只承认真正的“放电”
        spike_mask = energy_spikes > 0.5
        energy_spikes = energy_spikes * spike_mask
        
        # 2. Hebbian LTP 原则: Fire together, wire together
        # 如果神经元 i 和 j 在此时同时高频放电，则它们之间的拓扑引力/边权重增加
        # 使用批量外积 sum_b (e_b x e_b^T)
        co_activation = torch.mm(energy_spikes.t(), energy_spikes) # [represent_dim, represent_dim]
        
        # 为了不让同一点的自重无限放大，把对角线清零
        # 这就迫使网络去学习“概念之间的关系”，而不是“自我陶醉”
        co_activation.fill_diagonal_(0)
        
        # 增加突触权重
        self.P_topo += self.lr_topo * co_activation
        
    def stasis_and_metabolism(self):
        """
        如果没有这一步，网络在一亿次冲刷后必定 OOM 内存爆炸死机 (或全等于1陷入癫痫) 
        这就是稳态可塑性与睡眠截断机制 (LTD & GC)
        """
        # 1. 生理衰减 (Exponential Decay)
        self.P_topo *= (1.0 - self.decay_rate)
        
        # 2. 垃圾回收 (Garbage Collection / Synaptic Pruning)
        # 将微弱到一定程度的连结直接斩断为 0，这正是大脑保持极度稀疏 (Biosparse) 的秘密！
        prune_mask = self.P_topo < self.prune_threshold
        self.P_topo[prune_mask] = 0.0

def plot_receptors(W, title="Receptors"):
    os.makedirs('tempdata', exist_ok=True)
    W = W.cpu().numpy()
    n_filters = W.shape[0]
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for i in range(100):
        if i < n_filters:
            ax = axes[i // 10, i % 10]
            # 归一化便于可视化
            w_img = W[i].reshape(28, 28)
            vmax = np.abs(w_img).max()
            ax.imshow(w_img, cmap='RdBu', vmin=-vmax, vmax=vmax)
            ax.axis('off')
    plt.suptitle(title)
    plt.savefig(f'tempdata/{title}.png')
    plt.close()
    print(f"[AGI Mother] Filter map saved to tempdata/{title}.png")

def plot_topology(P_topo, title="Topology"):
    os.makedirs('tempdata', exist_ok=True)
    P = P_topo.cpu().numpy()
    plt.figure(figsize=(8, 8))
    plt.imshow(P, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(f"{title} - NonZero: {np.count_nonzero(P)}")
    plt.savefig(f'tempdata/{title}.png')
    plt.close()
    print(f"[AGI Mother] Topology map saved to tempdata/{title}.png")

def main():
    print("==================================================")
    print(" Phase XXVIII: 物理生命体母差分引擎实装 - 第二&三阶引力图谱测试 ")
    print("==================================================")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 模拟真实世界无休止的高维光学冲击 (The physical reality bombardment)
    dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
    
    mother_engine = AGIMotherEngineV1(input_dim=784, represent_dim=100)
    
    plot_receptors(mother_engine.W_receptors, title="step_000_chaos_white_noise")
    plot_topology(mother_engine.P_topo, title="step_000_empty_topology")
    
    print("[AGI Mother] 开始模拟自然感官剥离冲刷与 Hebbian 图谱结晶...")
    steps = 0
    max_steps = 1500
    
    for epoch in range(1):
        for data, _ in loader:
            batch_x = data.view(data.size(0), -1).to(mother_engine.device) # [B, 784]
            
            # 第一阶：孤立晶体提取
            mother_engine.wash_receptors_with_sanger(batch_x)
            
            # 第二阶：Hebbian 图谱生长
            mother_engine.wash_topology_with_hebbian(batch_x)
            
            # 第三阶：稳态物理衰变
            mother_engine.stasis_and_metabolism()
            
            steps += 1
            if steps % 300 == 0:
                print(f" - Submerging step {steps}/{max_steps} ... (WTA Inhibition + Hebbian Binding + LTD GC)")
                plot_receptors(mother_engine.W_receptors, title=f"step_{steps}_crystallization")
                plot_topology(mother_engine.P_topo, title=f"step_{steps}_topology")
            if steps >= max_steps:
                break
        if steps >= max_steps:
            break
            
    print("[AGI Mother] 第二阶与第三阶测试完成！请检查 tempdata 文件夹下的 topology 图谱！")

if __name__ == "__main__":
    main()
