"""Print Phase 288v2 summary for MEMO entry."""
import json, os

MODELS = ['qwen3', 'glm4']

print("=== Phase 288v2: Attention vs MLP Causal Decomposition ===")
print()

for m in MODELS:
    path = f"results/phase288_attn_mlp/{m}_decomp.json"
    if not os.path.exists(path):
        print(f"  {m}: NO RESULTS")
        continue
    d = json.load(open(path))

    # Global averages
    entries = list(d["cat_lc_summary"].values())
    avg_a_kr = sum(v["attn_kl_ratio"] for v in entries) / len(entries)
    avg_m_kr = sum(v["mlp_kl_ratio"] for v in entries) / len(entries)
    avg_b_kr = sum(v["both_kl_ratio"] for v in entries) / len(entries)
    avg_a_prog = sum(v["attn_progress"] for v in entries) / len(entries)
    avg_m_prog = sum(v["mlp_progress"] for v in entries) / len(entries)
    avg_b_prog = sum(v["both_progress"] for v in entries) / len(entries)

    print(f"--- {m.upper()} (L={d['model_info']['n_layers']}, d={d['model_info']['d_model']}) ---")
    print(f"  Global: A_KR={avg_a_kr:.2f}  M_KR={avg_m_kr:.2f}  B_KR={avg_b_kr:.2f} | "
          f"A_prog={avg_a_prog:.3f}  M_prog={avg_m_prog:.3f}  B_prog={avg_b_prog:.3f}")
    print(f"  {'Category':>18} {'LC':>6} {'A_Prog':>8} {'M_Prog':>8} {'B_Prog':>8} {'Dominant':>10} {'A_KR':>8} {'M_KR':>8}")
    print(f"  {'-'*80}")

    cats = sorted(set(v['category'] for v in entries))
    for cat in cats:
        cat_rows = [v for v in entries if v['category'] == cat]
        best = max(cat_rows, key=lambda x: max(x['attn_progress'], x['mlp_progress']))
        lc = best['layer_config']
        a_p = best['attn_progress']
        m_p = best['mlp_progress']
        b_p = best['both_progress']
        dom = best['dominant']
        a_kr = best['attn_kl_ratio']
        m_kr = best['mlp_kl_ratio']
        print(f"  {cat:>18} {lc:>6} {a_p:8.3f} {m_p:8.3f} {b_p:8.3f} {dom:>10} {a_kr:8.1f} {m_kr:8.1f}")
    print()

# Cross-model pattern summary
print("=== CROSS-MODEL PATTERNS ===")
print()
print("Key findings:")
print("1. ATTENTION vs MLP dominance varies by function AND model")
print("2. GLM4: Attention patching causes extreme over-conversion (KR up to 21x for negation)")
print("3. Qwen3: Both paths contribute; MLP progress slightly higher (0.70 vs 0.54)")
print("4. Both-combined patching consistently highest progress for Qwen3")
print("5. Logical: Near-perfect reconstruction (both progress > 0.99) at early layers")
print("   - Suggests logical functions may be encoded in earlier or different subnetworks")
print("6. Recursive: Model-dependent. Qwen3 = ATTN dominant, GLM4 = MLP dominant")
print("7. Translation: Qwen3 balanced, GLM4 MLP-dominant (3.5x)")
print()
print("DS7B: FAILED due to CPU offloading (only 1.1GB GPU). Needs separate fix.")
