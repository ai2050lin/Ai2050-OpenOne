"""Phase 419 汇总脚本"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import numpy as np
from pathlib import Path

ROOT = Path(r"d:\Ai2050\TransformerLens-Project")

print("=" * 80)
print("Phase 419: 大规模Token轨道图 - 跨模型汇总")
print("=" * 80)

# 1. 基本非对称性对比
print("\n### 1. Asymmetry对比 (up_mean - down_abs_mean) ###")
print(f"{'属性':<12} {'Qwen3':<20} {'GLM4':<20} {'DS7B':<20}")
print("-" * 72)

for attr in ["temperature", "speed", "size"]:
    vals = []
    for model in ["qwen3", "glm4", "deepseek7b"]:
        path = ROOT / f"results/phase419_token_trajectory/{model}_phase419.json"
        with open(path) as f:
            data = json.load(f)
        s = data["attributes"][attr]["summary"]
        sig = "*" if (s["ci_95_low"] > 0 or s["ci_95_high"] < 0) else ""
        vals.append(f"{s['asymmetry']:+.3f}{sig} [{s['ci_95_low']:+.3f},{s['ci_95_high']:+.3f}]")
    print(f"{attr:<12} {vals[0]:<20} {vals[1]:<20} {vals[2]:<20}")

# 2. Up-reversal vs Down-reversal强度对比
print("\n### 2. Up/Down Reversal强度 ###")
print(f"{'属性':<12} {'Qwen3 up/down':<25} {'GLM4 up/down':<25} {'DS7B up/down':<25}")
print("-" * 87)

for attr in ["temperature", "speed", "size"]:
    vals = []
    for model in ["qwen3", "glm4", "deepseek7b"]:
        path = ROOT / f"results/phase419_token_trajectory/{model}_phase419.json"
        with open(path) as f:
            data = json.load(f)
        s = data["attributes"][attr]["summary"]
        vals.append(f"{s['up_mean']:+.3f}/{s['down_abs_mean']:+.3f}")
    print(f"{attr:<12} {vals[0]:<25} {vals[1]:<25} {vals[2]:<25}")

# 3. L0基线对比
print("\n### 3. L0基线 (定义后的默认level) ###")
print(f"{'属性':<12} {'Qwen3 LOW/HIGH':<25} {'GLM4 LOW/HIGH':<25} {'DS7B LOW/HIGH':<25}")
print("-" * 87)

for attr in ["temperature", "speed", "size"]:
    vals = []
    for model in ["qwen3", "glm4", "deepseek7b"]:
        path = ROOT / f"results/phase419_token_trajectory/{model}_phase419.json"
        with open(path) as f:
            data = json.load(f)
        s = data["attributes"][attr]["summary"]
        vals.append(f"{s['low_L0_mean']:.3f}/{s['high_L0_mean']:.3f}")
    print(f"{attr:<12} {vals[0]:<25} {vals[1]:<25} {vals[2]:<25}")

# 4. 反转成功率
print("\n### 4. Reversal Success Rate ###")
print(f"{'属性':<12} {'Qwen3 up%/down%':<25} {'GLM4 up%/down%':<25} {'DS7B up%/down%':<25}")
print("-" * 87)

for attr in ["temperature", "speed", "size"]:
    vals = []
    for model in ["qwen3", "glm4", "deepseek7b"]:
        path = ROOT / f"results/phase419_token_trajectory/{model}_phase419.json"
        with open(path) as f:
            data = json.load(f)
        s = data["attributes"][attr]["summary"]
        rs = s["reversal_success"]
        vals.append(f"{rs['up_success_rate']*100:.0f}%/{rs['down_success_rate']*100:.0f}%")
    print(f"{attr:<12} {vals[0]:<25} {vals[1]:<25} {vals[2]:<25}")

# 5. 轨道捕获统计
print("\n### 5. 轨道捕获统计 ###")
for attr in ["temperature", "speed", "size"]:
    print(f"\n  {attr}:")
    for model in ["qwen3", "glm4", "deepseek7b"]:
        path = ROOT / f"results/phase419_token_trajectory/{model}_phase419.json"
        with open(path) as f:
            data = json.load(f)
        cs = data["attributes"][attr]["summary"]["capture_stats"]
        print(f"    {model}: LOW→low:{cs['low_captured_by_low']}/30 HIGH→high:{cs['high_captured_by_high']}/30 "
              f"LOW→high:{cs['low_captured_by_high']}/30 HIGH→low:{cs['high_captured_by_low']}/30")

# 6. 关键发现总结
print("\n" + "=" * 80)
print("### 关键发现 ###")
print("=" * 80)
print("""
1. 架构分叉: 
   - Qwen系(Qwen3/DS7B): asymmetry < 0 → down-reversal(HIGH→LOW)更容易
   - GLM4: asymmetry > 0 → up-reversal(LOW→HIGH)更容易
   
2. Size属性模式最强:
   - Qwen3 size asymmetry = -1.177 (最强)
   - DS7B size asymmetry = -0.675
   - GLM4 size asymmetry = -0.218 (唯一不显著)
   
3. 反转成功率模式:
   - Qwen3: down-reversal成功率远高于up-reversal
   - GLM4: up-reversal成功率更高(temperature/speed), size接近
   - DS7B: down-reversal更容易, 但speed的HIGH→LOW基线偏离大
   
4. 定义效果: 所有模型中, LOW定义使L0≈2-3, HIGH定义使L0≈3.5-4.7
   - DS7B的HIGH L0偏离最大(speed只有3.288)
   - 说明DS7B对"快"的定义不如"热"有效
""")
