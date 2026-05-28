"""Phase 292 cross-model comparison"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

models = ['qwen3', 'glm4', 'deepseek7b']
data = {}
for m in models:
    try:
        data[m] = json.load(open(f'results/phase292_contract_atlas/{m}_atlas.json', encoding='utf-8'))
    except:
        data[m] = {}

print("=" * 80)
print("PHASE 292: NEGATION CONTRACT ATLAS — CROSS-MODEL COMPARISON")
print("=" * 80)

# === Exp A: Block Heatmap — best blocks ===
print("\n=== EXP A: BEST BLOCKS (α=1.0, attn-only) ===")
for m in models:
    d = data[m].get("exp_A_block_heatmap", {})
    if not d: continue
    # Find best block for each size
    for bs in [1, 2, 4]:
        best = max((v for k,v in d.items() if v.get("block_size")==bs), 
                   key=lambda x: x["mean_prog"], default=None)
        if best:
            print(f"  {m:>12} bs={bs} L{best['block_start']}-{best['block_end']}: "
                  f"PROG={best['mean_prog']:.3f} KR={best['mean_kr']:.2f}")

# === Exp A: Synergy ===
print("\n=== EXP A: SYNERGY (block4 - sum_of_single) ===")
for m in models:
    syn = data[m].get("exp_A_synergy", {})
    if not syn: continue
    best = max(syn.items(), key=lambda x: x[1]["synergy"])
    worst = min(syn.items(), key=lambda x: x[1]["synergy"])
    print(f"  {m:>12}: best={best[0]} syn={best[1]['synergy']:.3f}, "
          f"worst={worst[0]} syn={worst[1]['synergy']:.3f}")

# === Exp B: Position Sensitivity ===
print("\n=== EXP B: POSITION SENSITIVITY (attn, α=1.0) ===")
for m in models:
    d = data[m].get("exp_B_position", {})
    if not d: continue
    # Compare operator vs last at L0 block
    for bstart in [0]:
        op_key = f"L{bstart}_operator"
        last_key = f"L{bstart}_last"
        all_key = f"L{bstart}_all"
        op = d.get(op_key, {}).get("mean_prog", 0)
        last = d.get(last_key, {}).get("mean_prog", 0)
        all_ = d.get(all_key, {}).get("mean_prog", 0)
        sensitivity = op - last  # positive = operator more important
        print(f"  {m:>12} L{bstart}: operator={op:.3f} last={last:.3f} all={all_:.3f} "
              f"sensitivity={sensitivity:+.3f}")

# === Exp C: Component Sensitivity ===
print("\n=== EXP C: COMPONENT SENSITIVITY (α=1.0) ===")
for m in models:
    d = data[m].get("exp_C_component", {})
    if not d: continue
    for bstart in [0]:
        attn = d.get(f"L{bstart}_attn_only", {}).get("mean_prog", 0)
        mlp = d.get(f"L{bstart}_mlp_only", {}).get("mean_prog", 0)
        both = d.get(f"L{bstart}_both", {}).get("mean_prog", 0)
        resid = d.get(f"L{bstart}_resid_after", {}).get("mean_prog", 0)
        mlp_advantage = mlp - attn  # positive = MLP more important
        print(f"  {m:>12} L{bstart}: attn={attn:.3f} mlp={mlp:.3f} both={both:.3f} "
              f"resid={resid:.3f} mlp_adv={mlp_advantage:+.3f}")

# === Exp D: Subtype α curves at best block ===
print("\n=== EXP D: SUBTYPE α CURVES (L0 block) ===")
for m in models:
    d = data[m].get("exp_D_subtype_alpha", {})
    if not d: continue
    subtypes = sorted(set(v["subtype"] for v in d.values()))
    print(f"  {m:>12}:")
    for st in subtypes:
        alpha_progs = []
        for a in [0, 0.25, 0.5, 0.75, 1.0]:
            key = f"L0_{st}_a{a:.2f}"
            prog = d.get(key, {}).get("mean_prog", 0)
            alpha_progs.append(prog)
        # Compute α curve slope (prog@α=1 - prog@α=0)
        slope = alpha_progs[-1] - alpha_progs[0]
        # Linearity: check if α=0.5 ≈ midpoint
        midpoint = (alpha_progs[0] + alpha_progs[-1]) / 2
        actual_mid = alpha_progs[2]
        linearity = abs(actual_mid - midpoint)
        print(f"    {st:>20}: α0={alpha_progs[0]:.3f} α0.5={actual_mid:.3f} α1={alpha_progs[-1]:.3f} "
              f"slope={slope:.3f} lin_dev={linearity:.3f}")

print("\n" + "=" * 80)
