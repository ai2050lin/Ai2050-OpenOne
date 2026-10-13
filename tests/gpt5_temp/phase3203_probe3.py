# -*- coding: utf-8 -*-
import io, json
SD = r'D:/AI2050/Ai2050-OpenOne/tests/glm5/result/rdc_query_construction_20260913'
def load(ph, sub):
    return json.load(io.open(SD + '\\phase%s\\%s\\result.json' % (ph, sub), encoding='utf-8'))

r = load('3169', 'g5a6_oov_panel')
print('3169 overall-ish:', {k: json.dumps(r[k])[:200] for k in ('panel', 'gate')})
print('3169 pm keys:', list(r['per_model']['qwen3-4b'].keys()))

r = load('3172', 'g5a9_port_calibration')
print('---3172 overall:', json.dumps(r['overall'], ensure_ascii=False)[:500])
print('3172 pooled:', json.dumps(r['pooled'], ensure_ascii=False)[:300])

r = load('3175', 'g5a12_residual_arms')
print('---3175 overall:', json.dumps(r['overall'], ensure_ascii=False)[:500])
print('3175 delta_gate:', json.dumps(r['delta_gate'], ensure_ascii=False)[:300])

for ph, sub in [('3188','g5c1_gap4_dv_attribution'), ('3189','g5c2_residual_structure'),
                ('3190','g5c3_shared_direction_robustness'), ('3191','g5c4_ftr25_heldout_upgrade'),
                ('3192','g5c5_gap4_amplitude_attribution')]:
    r = load(ph, sub)
    print('---%s keys:' % ph, list(r.keys()))
    for k in r:
        v = r[k]
        if isinstance(v, dict) and k in ('per_model', 'arm', 'overall', 'summary', 'gates'):
            print(' ', k, json.dumps(v, ensure_ascii=False)[:500])
