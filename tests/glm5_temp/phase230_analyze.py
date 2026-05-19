import json, sys
with open('tests/glm5_temp/phase230_qwen3_results.json','r',encoding='utf-8') as f:
    r = json.load(f)

print('=== Exp1: Per-adjective stability ===')
best_layer = None
best_val = 0
for lk, lv in r['exp1'].items():
    if lv['same_adj_mean_cos'] > best_val:
        best_val = lv['same_adj_mean_cos']
        best_layer = lk

print(f'Best layer: {best_layer} (same_cos={best_val:.4f})')
pa = r['exp1'][best_layer]['per_adj_stability']
for adj, d in sorted(pa.items(), key=lambda x: x[1]['mean_cos'], reverse=True):
    print(f'  {adj:12s}: cos={d["mean_cos"]:.4f} norm={d["delta_norm"]:.4f}')

print()
print('=== Layer-by-layer same vs cross ===')
for lk in sorted(r['exp1'].keys(), key=lambda x: int(x[1:])):
    lv = r['exp1'][lk]
    sc = lv['same_adj_mean_cos']
    cc = lv['cross_adj_mean_cos']
    if sc > 0:
        print(f'  {lk}: same={sc:.4f} cross={cc:.4f} sep={lv["separation_ratio"]:.2f}x')

print()
print('=== Exp2: Operation encoding ===')
for lk in sorted(r['exp2'].keys(), key=lambda x: int(x[1:])):
    lv = r['exp2'][lk]
    sc = lv['same_op_mean_cos']
    if sc > 0:
        print(f'  {lk}: same={sc:.4f} cross={lv["cross_op_mean_cos"]:.4f} sep={lv["separation_ratio"]:.2f}x')

print()
print('=== Exp4: Causal ===')
for lk in sorted(r['exp4'].keys(), key=lambda x: int(x[1:])):
    lv = r['exp4'][lk]
    for adj, d in lv.items():
        print(f'  {lk} {adj}: change={d["mean_prob_change"]:.8f}')
