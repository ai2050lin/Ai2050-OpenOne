"""
Adaptive FiberNet 对比实验
==========================

对比三种架构：
1. Standard Transformer
2. FiberNet (固定几何先验)
3. Adaptive FiberNet (自动学习几何)

任务：
- Z_113 模加法（已知圆环结构）
- 多位数加法（隐含进位逻辑）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import json
import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.adaptive_fibernet import AdaptiveFiberNet


# ============ 数据集 ============

class ModularArithmeticDataset(Dataset):
    """模运算数据集 Z_p"""
    
    def __init__(self, p=113, split='train', train_ratio=0.7):
        self.p = p
        self.split = split
        
        # 生成所有可能的 (a, b) 对
        data = []
        for a in range(p):
            for b in range(p):
                # 输入格式: [a, b, +, =] -> [a, b, op, eq]
                # 我们用 token ID 表示
                # token: 0-112 = 数字, 113 = '+', 114 = '='
                input_seq = [a, b, 113, 114]  # a b + =
                target = (a + b) % p
                data.append((input_seq, target))
        
        # 划分训练/测试集
        n_train = int(len(data) * train_ratio)
        if split == 'train':
            self.data = data[:n_train]
        else:
            self.data = data[n_train:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target = self.data[idx]
        return torch.tensor(input_seq), torch.tensor(target)


class MultiDigitAdditionDataset(Dataset):
    """多位数加法数据集"""
    
    def __init__(self, n_digits=3, n_samples=10000, split='train'):
        self.n_digits = n_digits
        
        torch.manual_seed(42 if split == 'train' else 123)
        
        data = []
        max_val = 10 ** n_digits
        
        for _ in range(n_samples):
            a = torch.randint(0, max_val, (1,)).item()
            b = torch.randint(0, max_val, (1,)).item()
            
            # 将数字转换为 token 序列
            a_tokens = [int(d) + 1 for d in str(a).zfill(n_digits)]  # 1-10 表示 0-9
            b_tokens = [int(d) + 1 for d in str(b).zfill(n_digits)]
            
            # 结果
            result = a + b
            result_tokens = [int(d) + 1 for d in str(result).zfill(n_digits + 1)]
            
            # 输入: a_digits + b_digits + op_token
            # op_token = 11 ('+')
            input_seq = a_tokens + b_tokens + [11]
            target_seq = result_tokens
            
            data.append((input_seq, target_seq))
        
        # 划分
        n_train = int(len(data) * 0.8)
        self.data = data[:n_train] if split == 'train' else data[n_train:]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target_seq = self.data[idx]
        return torch.tensor(input_seq), torch.tensor(target_seq)


# ============ 标准Transformer ============

class StandardTransformer(nn.Module):
    """标准 Transformer 对比基线"""
    
    def __init__(self, vocab_size, d_model=128, n_heads=4, n_layers=2, 
                 max_seq_len=32, dropout=0.1):
        super().__init__()
        
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_encoding[:, :seq_len, :]
        x = self.transformer(x)
        x = self.output_norm(x)
        return self.output_proj(x)


# ============ 训练函数 ============

def train_model(model, train_loader, test_loader, n_epochs=100, 
                lr=1e-3, device='cpu', model_name='Model'):
    """训练模型"""
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = nn.CrossEntropyLoss()
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'structure_info': []
    }
    
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # 更新解耦进度（Adaptive FiberNet）
        if hasattr(model, 'update_disentangle_progress'):
            model.update_disentangle_progress(epoch, n_epochs)
        
        # 训练
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # 前向传播
            outputs = model(inputs)
            
            # 处理不同输出格式
            if isinstance(outputs, tuple):
                logits, struct_info = outputs
                if epoch % 10 == 0:
                    history['structure_info'].append({
                        'epoch': epoch,
                        'struct_info': [{k: v.mean().item() if hasattr(v, 'mean') else v 
                                        for k, v in s.items() if not isinstance(v, torch.Tensor) or v.numel() == 1}
                                       for s in struct_info] if struct_info else []
                    })
            else:
                logits = outputs
            
            # 计算损失（只看最后一个位置）
            logits = logits[:, -1, :]  # (batch, vocab_size)
            loss = criterion(logits, targets)
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = logits.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        
        scheduler.step()
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # 测试
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    logits, _ = outputs
                else:
                    logits = outputs
                
                logits = logits[:, -1, :]
                _, predicted = logits.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        
        test_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        
        if (epoch + 1) % 10 == 0 or epoch < 5:
            print(f"[{model_name}] Epoch {epoch+1}/{n_epochs} | "
                  f"Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | "
                  f"Test Acc: {test_acc:.2%}")
    
    training_time = time.time() - start_time
    history['training_time'] = training_time
    
    return history


# ============ 主实验 ============

def run_comparison_experiment():
    """运行对比实验"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"设备: {device}")
    
    # ============ 实验1: Z_113 模加法 ============
    print("\n" + "="*60)
    print("实验1: Z_113 模加法 (已知圆环结构)")
    print("="*60)
    
    # 数据
    train_dataset = ModularArithmeticDataset(p=113, split='train')
    test_dataset = ModularArithmeticDataset(p=113, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64)
    
    vocab_size = 115  # 0-112 + '+' + '='
    d_model = 64
    n_heads = 4
    n_layers = 2
    
    results = {}
    
    # 1. Standard Transformer
    print("\n--- Standard Transformer ---")
    model1 = StandardTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers
    )
    history1 = train_model(model1, train_loader, test_loader, 
                          n_epochs=100, device=device, model_name='Transformer')
    results['transformer'] = history1
    print(f"参数量: {sum(p.numel() for p in model1.parameters()):,}")
    print(f"训练时间: {history1['training_time']:.1f}s")
    
    # 2. Adaptive FiberNet (learnable)
    print("\n--- Adaptive FiberNet (可学习几何) ---")
    model2 = AdaptiveFiberNet(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        manifold_type='learnable'
    )
    history2 = train_model(model2, train_loader, test_loader,
                          n_epochs=100, device=device, model_name='Adaptive FiberNet')
    results['adaptive_fibernet'] = history2
    print(f"参数量: {sum(p.numel() for p in model2.parameters()):,}")
    print(f"训练时间: {history2['training_time']:.1f}s")
    
    # 3. Adaptive FiberNet (circle - 有几何先验)
    print("\n--- Adaptive FiberNet (圆环先验) ---")
    model3 = AdaptiveFiberNet(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        manifold_type='circle'
    )
    history3 = train_model(model3, train_loader, test_loader,
                          n_epochs=100, device=device, model_name='FiberNet (circle)')
    results['fibernet_circle'] = history3
    print(f"参数量: {sum(p.numel() for p in model3.parameters()):,}")
    print(f"训练时间: {history3['training_time']:.1f}s")
    
    # ============ 结果汇总 ============
    print("\n" + "="*60)
    print("实验结果汇总")
    print("="*60)
    
    print(f"\n{'模型':<25} {'最终精度':<12} {'训练时间':<12} {'收敛速度'}")
    print("-"*65)
    
    for name, hist in results.items():
        final_acc = hist['test_acc'][-1]
        train_time = hist['training_time']
        
        # 计算达到 90% 精度所需 epoch
        convergence = 'N/A'
        for i, acc in enumerate(hist['test_acc']):
            if acc > 0.9:
                convergence = f"Epoch {i+1}"
                break
        
        print(f"{name:<25} {final_acc:.2%}       {train_time:.1f}s        {convergence}")
    
    # 保存结果
    results_path = Path(__file__).parent.parent / 'tempdata' / 'adaptive_fibernet_results.json'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, 'w') as f:
        # 移除不可序列化的结构信息
        clean_results = {}
        for name, hist in results.items():
            clean_results[name] = {
                'train_loss': hist['train_loss'],
                'train_acc': hist['train_acc'],
                'test_acc': hist['test_acc'],
                'training_time': hist['training_time']
            }
        json.dump(clean_results, f, indent=2)
    
    print(f"\n结果已保存至: {results_path}")
    
    return results


if __name__ == "__main__":
    run_comparison_experiment()
