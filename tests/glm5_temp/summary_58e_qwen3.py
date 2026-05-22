import json, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

r = json.load(open('results/subspace_topology/exp4e_axis_direction_qwen3.json', encoding='utf-8'))

print("=== Phase 58e: Axis Direction Analysis (Qwen3) ===\n")

for pk, pd in r['results'].items():
    print(f"\n--- {pk}: {pd['w_a']}/{pd['w_b']} ({pd['relation']}) ---")
    for lk in sorted(pd["layers"].keys(), key=int):
        ld = pd["layers"][lk]
        delta_top = [t["token"].strip()[:12] for t in ld["delta_decoded_top5"][:3]]
        neg_top = [t["token"].strip()[:12] for t in ld["neg_delta_decoded_top5"][:3]]
        print(f"  L{lk}: axis_sep={ld['axis_separation']:.3f} sv={ld['shared_singular_value']:.3f}")
        print(f"    +delta({pd['w_a']}vs{pd['w_b']}): {delta_top}")
        print(f"    -delta({pd['w_b']}vs{pd['w_a']}): {neg_top}")
        if ld["shared_decoded_top5"]:
            shared_top = [t["token"].strip()[:12] for t in ld["shared_decoded_top5"][:3]]
            neg_shared_top = [t["token"].strip()[:12] for t in ld["neg_shared_decoded_top5"][:3]]
            print(f"    shared+: {shared_top}")
            print(f"    shared-: {neg_shared_top}")

print("\n\n=== KEY COMPARISON: Deep Layer (L35/L27) Delta Decoding ===\n")
# 提取最深层的结果
for pk, pd in r['results'].items():
    deepest = max(pd["layers"].keys(), key=int)
    ld = pd["layers"][deepest]
    delta_top = [t["token"].strip()[:15] for t in ld["delta_decoded_top5"][:5]]
    neg_top = [t["token"].strip()[:15] for t in ld["neg_delta_decoded_top5"][:5]]
    print(f"{pk:20s} ({pd['relation']:10s}) L{deepest}:")
    print(f"  +delta({pd['w_a']}vs{pd['w_b']}): {delta_top}")
    print(f"  -delta({pd['w_b']}vs{pd['w_a']}): {neg_top}")
    print(f"  axis_sep={ld['axis_separation']:.3f}")
