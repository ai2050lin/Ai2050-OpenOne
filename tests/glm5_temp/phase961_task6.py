#!/usr/bin/env python3
"""Phase 961 Task 6: Cross-model comparison — loads saved results and compares."""
import sys, json
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))
from phase961_mode_head_mechanism import RESULT_DIR, log, ensure_dir

ensure_dir(RESULT_DIR)

all_results = []
for m in ["qwen3", "glm4", "deepseek7b"]:
    p = RESULT_DIR / f"{m}_result.json"
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        all_results.append(data)
        log(f"Loaded {m}: tasks={list(data.keys())}")
    else:
        log(f"Missing: {p}")

log(f"\n{'='*80}")
log("CROSS-MODEL COMPARISON")
log(f"{'='*80}")

# 1. Attention Pattern Comparison
log("\n--- 1. Attention Pattern Comparison ---")
log(f"{'Model':<14} {'Head':<12} {'Content':>8} {'Punct':>8} {'Func':>8} {'Space':>8} {'Special':>8} {'Entropy':>8}")
for mr in all_results:
    t1 = mr.get("task1")
    if not t1: continue
    for head_key, mass in t1.get("attention_mass_by_category", {}).items():
        log(f"{mr['model']:<14} {head_key:<12} "
            f"{mass.get('content', 0):>8.3f} {mass.get('punct', 0):>8.3f} "
            f"{mass.get('function', 0):>8.3f} {mass.get('space', 0):>8.3f} "
            f"{mass.get('special', 0):>8.3f} {mass.get('mean_entropy', 0):>8.3f}")

# 2. Head Output vs Mode Direction
log("\n--- 2. Head Output vs Mode Direction ---")
log(f"{'Model':<14} {'Head':<12} {'Role':<18} {'||O_h||':>8} {'cos_en':>8} {'cos_cn':>8} {'cos_EOS':>8} {'cos_per':>8} {'Top3 tokens':>35}")
for mr in all_results:
    t23 = mr.get("task2_3")
    if not t23: continue
    for head_key, ha in t23.get("head_analysis", {}).items():
        top3 = ha.get("top_promoted_tokens_en", [])[:3]
        top3_str = " ".join(f"'{t['token']}'" for t in top3)
        wu = ha.get("wu_cosines", {})
        log(f"{mr['model']:<14} {head_key:<12} {ha['role']:<18} {ha['norm_Oh_en']:>8.3f} "
            f"{ha['cos_Oh_en_vs_dmode']:>8.4f} {ha['cos_Oh_cn_vs_dmode']:>8.4f} "
            f"{wu.get('EOS', 0):>8.4f} {wu.get('.', 0):>8.4f} {top3_str:>35}")

# 3. Boost Effects
log("\n--- 3. Boost Effects (logit analysis) ---")
log(f"{'Model':<14} {'Head':<12} {'a=1.5 dEOS':>12} {'a=2.0 dEOS':>12} {'a=3.0 dEOS':>12} {'a=2.0 argmax':>14}")
for mr in all_results:
    t4 = mr.get("task4")
    if not t4: continue
    ls = t4.get("logit_summary", {})
    for head_key, s in ls.items():
        d15 = s.get("delta_EOS_a1.5", {}).get("mean", 0)
        d20 = s.get("delta_EOS_a2.0", {}).get("mean", 0)
        d30 = s.get("delta_EOS_a3.0", {}).get("mean", 0)
        ac20 = s.get("argmax_changed_a2.0", {}).get("mean", s.get("argmax_chg_a2.0", {}).get("mean", 0))
        log(f"{mr['model']:<14} {head_key:<12} {d15:>12.4f} {d20:>12.4f} {d30:>12.4f} {ac20:>14.2f}")

# 4. Rollout summary (if available)
log("\n--- 4. Rollout Summary ---")
for mr in all_results:
    t4 = mr.get("task4")
    if not t4: continue
    rs = t4.get("rollout_summary", {})
    if rs:
        log(f"  {mr['model']} ({t4.get('primary_head', 'N/A')}):")
        for alpha_key, alpha_data in rs.items():
            log(f"    {alpha_key}: eos={alpha_data.get('eos_rate', 0):.2f}  "
                f"switch={alpha_data.get('lang_switch_rate', 0):.2f}  "
                f"tokens={alpha_data.get('mean_tokens', 0):.1f}")

# 5. Joint Intervention
log("\n--- 5. Joint Intervention ---")
log(f"{'Model':<14} {'Head':<12} {'Condition':<25} {'EOS':>6} {'Clean':>6} {'Switch':>7} {'Tokens':>7}")
for mr in all_results:
    t5 = mr.get("task5")
    if not t5: continue
    for cond, s in t5.get("summary", {}).items():
        log(f"{mr['model']:<14} {t5.get('primary_head', ''):<12} {cond:<25} "
            f"{s.get('eos_rate', 0):>6.2f} {s.get('strict_clean_rate', 0):>6.2f} "
            f"{s.get('lang_switch_rate', 0):>7.2f} {s.get('mean_tokens', 0):>7.1f}")

# Save comparison
comparison = {
    "models": [mr["model"] for mr in all_results],
    "attention_patterns": [],
    "head_output_mode": [],
    "boost_effects": [],
    "joint_intervention": [],
}
for mr in all_results:
    mn = mr["model"]
    t1 = mr.get("task1", {})
    for hk, m in t1.get("attention_mass_by_category", {}).items():
        comparison["attention_patterns"].append({"model": mn, "head": hk, **m})
    t23 = mr.get("task2_3", {})
    for hk, ha in t23.get("head_analysis", {}).items():
        comparison["head_output_mode"].append({"model": mn, "head": hk, **ha})
    t4 = mr.get("task4", {})
    for hk, s in t4.get("logit_summary", {}).items():
        comparison["boost_effects"].append({"model": mn, "head": hk, **s})
    t5 = mr.get("task5", {})
    for cond, s in t5.get("summary", {}).items():
        comparison["joint_intervention"].append({"model": mn, "condition": cond, **s})

save_path = RESULT_DIR / "task6_cross_model.json"
save_path.write_text(json.dumps(comparison, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
log(f"\nSaved: {save_path}")
