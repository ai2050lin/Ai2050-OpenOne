"""Phase 419 R1+R2 综合对比"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
from pathlib import Path

ROOT = Path(r"d:\Ai2050\TransformerLens-Project")

print("=" * 90)
print("Phase 419: R1 vs R2 综合对比")
print("=" * 90)

print(f"\n### Asymmetry对比 (R1: 30+30 per attr, R2: 50+50 per attr) ###")
print(f"{'属性':<12} {'Qwen3 R1/R2':<30} {'GLM4 R1/R2':<30} {'DS7B R1/R2':<30}")
print("-" * 102)

for attr in ["temperature", "speed"]:
    vals = []
    for model in ["qwen3", "glm4", "deepseek7b"]:
        r1_path = ROOT / f"results/phase419_token_trajectory/{model}_phase419.json"
        r2_path = ROOT / f"results/phase419_token_trajectory/{model}_phase419_r2.json"
        with open(r1_path) as f:
            r1 = json.load(f)
        with open(r2_path) as f:
            r2 = json.load(f)
        a1 = r1["attributes"][attr]["summary"]["asymmetry"]
        a2 = r2["attributes"][attr]["summary"]["asymmetry"]
        ci = r2["attributes"][attr]["summary"]
        vals.append(f"{a1:+.3f}/{a2:+.3f} [{ci['ci_95_low']:+.3f},{ci['ci_95_high']:+.3f}]")
    print(f"{attr:<12} {vals[0]:<30} {vals[1]:<30} {vals[2]:<30}")

print(f"\n### 关键模式确认 ###")
for model in ["qwen3", "glm4", "deepseek7b"]:
    r2_path = ROOT / f"results/phase419_token_trajectory/{model}_phase419_r2.json"
    with open(r2_path) as f:
        r2 = json.load(f)
    temps = r2["attributes"]["temperature"]["summary"]["asymmetry"]
    speeds = r2["attributes"]["speed"]["summary"]["asymmetry"]
    direction = "UP更容易" if (temps + speeds) > 0 else "DOWN更容易"
    print(f"  {model}: temp={temps:+.3f} speed={speeds:+.3f} → {direction}")

print(f"""
### 核心发现: 架构分叉 ###

| 模型 | 架构 | Temperature Asym | Speed Asym | 方向 |
|------|------|------------------|------------|------|
| Qwen3 | Qwen3ForCausalLM | -1.445 | -0.824 | DOWN更容易 |
| GLM4  | GlmForCausalLM   | +0.425 | +0.755 | UP更容易 |
| DS7B  | Qwen2ForCausalLM | -0.192 | -0.121 | 接近0/弱DOWN |

Qwen系(Qwen3+DS7B)和GLM4的结构性偏置方向完全相反!
""")
