# -*- coding: utf-8 -*-
import io, json
SD = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913'
def load(ph, sub):
    return json.load(io.open(SD + '\\phase%s\\%s\\result.json' % (ph, sub), encoding='utf-8'))

for ph, sub, keys in [
    ('3189', 'g5c2_residual_structure', ['top1', 'cross_med']),
    ('3190', 'g5c3_shared_direction_robustness', ['loo_max_delta', 'ks_seen_ustar']),
    ('3191', 'g5c4_ftr25_heldout_upgrade', ['min_med_abs_cos']),
    ('3192', 'g5c5_gap4_amplitude_attribution', ['amp_ratio', 'dir_ratio'])]:
    r = load(ph, sub)
    for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'):
        print(ph, m, {k: round(r['per_model'][m][k], 4) for k in keys})

reg = json.load(io.open(SD + r'\phase3202\g5f1_ftr04_recheck\atlas_registry_v1_25.json', encoding='utf-8'))
f25 = [f for f in reg['features'] if f['id'] == 'FTR-25'][0]
print('FTR-25 anchors:')
for a in f25['anchors']:
    print(' ', a['src'], a['role'], json.dumps(a['asserts'], ensure_ascii=False)[:400])
gl = json.load(io.open(SD + r'\phase3175\g5a12_residual_arms\gap_ledger_v1_5.json', encoding='utf-8'))
g4 = [g for g in gl['gaps'] if g['id'] == 'GAP-4'][0]
print('GAP-4 mechanism_note tail:', g4['mechanism_note'][-400:])
