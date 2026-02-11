import json
import math
import os
import random
import time

import torch

# 配置路径
LOG_DIR = r"d:\develop\TransformerLens-main\experiments\toy_experiment"
LOG_FILE = os.path.join(LOG_DIR, "training_log.json")

def simulate_training():
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
        
    metrics = {
        "Transformer": [],
        "FiberNet": []
    }
    
    print(f"Starting AGI Training Dynamics Simulation...")
    print(f"Logging to: {LOG_FILE}")
    
    for epoch in range(200):
        # 模拟 Transformer: 慢速收敛，后期波动
        # 逻辑：初期增长慢，受注意力噪声影响，后期可能遇到瓶颈
        trans_acc = 100 * (1 - math.exp(-epoch / 50)) + random.uniform(-2, 2)
        trans_acc = max(0, min(95, trans_acc))
        
        # 模拟 FiberNet: 快速几何收敛，曲率驱动
        # 逻辑：一旦几何联络对齐，准确率呈 S 型快速爆发，且趋近 100%
        fiber_acc = 100 / (1 + math.exp(-(epoch - 30) / 10)) + random.uniform(-0.5, 0.5)
        fiber_acc = max(0, min(99.9, fiber_acc))
        
        # 计算模拟曲率 (Curvature / Logic Error)
        # FiberNet 的错误率低意味着曲率接近 0
        fiber_curvature = 1.0 - (fiber_acc / 100.0)
        
        metrics["Transformer"].append({
            "epoch": epoch,
            "accuracy": trans_acc,
            "loss": 1.0 / (trans_acc + 1)
        })
        
        metrics["FiberNet"].append({
            "epoch": epoch,
            "accuracy": fiber_acc,
            "curvature": fiber_curvature
        })
        
        # 写入文件供前端读取
        with open(LOG_FILE, 'w') as f:
            json.dump(metrics, f)
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Transformer Accuracy={trans_acc:.2f}%, FiberNet Accuracy={fiber_acc:.2f}%")
            
        time.sleep(1.0) # 模拟真实训练时间间隔

if __name__ == "__main__":
    try:
        simulate_training()
    except KeyboardInterrupt:
        print("\nTraining simulation stopped.")
