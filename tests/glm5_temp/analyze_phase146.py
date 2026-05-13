"""分析Phase 146三个模型的结果"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import json
import numpy as np
from pathlib import Path

temp_dir = Path("tests/glm5_temp")

for model_name in ["qwen3", "glm4"]:
    # 找最新的phase146结果文件
    files = list(temp_dir.glob(f"phase146_{model_name}_critical_propagation_*.json"))
    if not files:
        print(f"\n{model_name}: 无结果文件")
        continue
    
    latest = max(files, key=lambda p: p.stat().st_mtime)
    print(f"\n{'='*60}")
    print(f"模型: {model_name} ({latest.name})")
    print(f"{'='*60}")
    
    data = json.load(open(latest, 'r', encoding='utf-8'))
    model_info = data.get("model_info", {})
    print(f"  层数: {model_info.get('n_layers')}, d_model: {model_info.get('d_model')}")
    
    # Exp A+ 摘要
    exp_a = data.get("exp_a", {})
    if exp_a:
        print(f"\n  Exp A+: 输出等价稳定性 ({len(exp_a)} entries)")
        
        # 按注入层和eps分组
        n_layers = model_info.get("n_layers", 36)
        inject_layers = sorted(set(v["inject_layer"] for v in exp_a.values()))
        eps_values = sorted(set(v["eps"] for v in exp_a.values()))
        
        print(f"  {'Layer':>6} {'eps':>5} {'top1':>6} {'top5':>6} {'logits_r':>9} {'state_d':>8} {'dir_cos':>8} {'kl_div':>8}")
        for il in inject_layers:
            for eps in eps_values:
                matches = [v for v in exp_a.values() if v["inject_layer"] == il and abs(v["eps"] - eps) < 0.01]
                if not matches:
                    continue
                top1 = np.mean([m["top1_match"] for m in matches])
                top5 = np.mean([m["top5_overlap"] for m in matches])
                logits_r = np.mean([m["logits_correlation"] for m in matches])
                state_d = np.mean([m["normalized_state_dists"][-1] for m in matches])
                dir_cos = np.mean([m["direction_cosines"][-1] for m in matches])
                kl = np.mean([m["kl_divergence"] for m in matches])
                print(f"  L{il:>4} {eps:>5.1f} {top1:>6.3f} {top5:>6.3f} {logits_r:>9.4f} {state_d:>7.2f}x {dir_cos:>8.4f} {kl:>8.4f}")
    
    # Exp B+ 摘要
    exp_b = data.get("exp_b", {})
    if exp_b:
        print(f"\n  Exp B+: 扰动方向演化 ({len(exp_b)} entries)")
        for il in sorted(set(v["inject_layer"] for v in exp_b.values())):
            matches = [v for v in exp_b.values() if v["inject_layer"] == il]
            if not matches:
                continue
            alignments = [m["direction_alignment"] for m in matches]
            mean_align = np.mean(alignments, axis=0)
            # 打印关键层
            n_pts = len(mean_align)
            print(f"    L{il}: cos@1st={mean_align[min(1,n_pts-1)]:.4f}, "
                  f"cos@mid={mean_align[n_pts//2]:.4f}, "
                  f"cos@last={mean_align[-1]:.4f}")
    
    # Exp D 摘要
    exp_d = data.get("exp_d", {})
    if exp_d:
        print(f"\n  Exp D: 约束修复动力学")
        for c_type, c_data in exp_d.items():
            traj = c_data.get("mean_delta_trajectory", [])
            dir_cos = c_data.get("mean_direction_cosines", [])
            if len(traj) > 1:
                peak_idx = int(np.argmax(traj))
                peak_val = traj[peak_idx]
                final_val = traj[-1]
                ratio = final_val / peak_val if peak_val > 0 else 0
                print(f"    {c_type}: peak@L{peak_idx}={peak_val:.1f}, final={final_val:.1f}, "
                      f"ratio={ratio:.3f}, top5={c_data.get('mean_top5_overlap',0):.3f}, "
                      f"top1={c_data.get('mean_top1_match',0):.3f}")
                if dir_cos:
                    print(f"      dir_cos: L0={dir_cos[0]:.4f}, Lmid={dir_cos[len(dir_cos)//2]:.4f}, Llast={dir_cos[-1]:.4f}")
    
    # Exp E 摘要
    exp_e = data.get("exp_e", {})
    if exp_e:
        print(f"\n  Exp E: W_U投影下的等价类")
        for il in sorted(set(v["inject_layer"] for v in exp_e.values())):
            matches = [v for v in exp_e.values() if v["inject_layer"] == il]
            if not matches:
                continue
            row_ratios = []
            for m in matches:
                total = np.array(m["total_energy"])
                row = np.array(m["row_space_energy"])
                if total[-1] > 0:
                    row_ratios.append(row[-1] / total[-1])
            if row_ratios:
                print(f"    L{il}: row_space_ratio={np.mean(row_ratios):.4f}, null_space_ratio={1-np.mean(row_ratios):.4f}")
    elif data.get("exp_e_error"):
        print(f"\n  Exp E: 失败 ({data['exp_e_error']})")
    else:
        print(f"\n  Exp E: 跳过 (W_U不可用)")

print("\n\n=== 跨模型对比 ===")
print("关键指标: 晚层注入(eps=2.0)的输出稳定性")
for model_name in ["qwen3", "glm4"]:
    files = list(temp_dir.glob(f"phase146_{model_name}_critical_propagation_*.json"))
    if not files:
        continue
    latest = max(files, key=lambda p: p.stat().st_mtime)
    data = json.load(open(latest, 'r', encoding='utf-8'))
    n_layers = data["model_info"]["n_layers"]
    exp_a = data.get("exp_a", {})
    
    # 取最晚的注入层, eps=2.0
    late_layer = n_layers - 2
    matches = [v for v in exp_a.values() if v["inject_layer"] == late_layer and abs(v["eps"] - 2.0) < 0.01]
    if matches:
        top1 = np.mean([m["top1_match"] for m in matches])
        top5 = np.mean([m["top5_overlap"] for m in matches])
        logits_r = np.mean([m["logits_correlation"] for m in matches])
        dir_cos = np.mean([m["direction_cosines"][-1] for m in matches])
        state_d = np.mean([m["normalized_state_dists"][-1] for m in matches])
        print(f"  {model_name} (L{late_layer}): top1={top1:.3f}, top5={top5:.3f}, "
              f"logits_r={logits_r:.4f}, dir_cos={dir_cos:.4f}, state_d={state_d:.2f}x")
