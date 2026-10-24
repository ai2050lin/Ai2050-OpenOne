# -*- coding: utf-8 -*-
"""3250 engineering-arm check: call atlas_fidelity coroutine directly (read-only)
and assert the v2 doc shape + incumbent numbers match the sealed result."""
import asyncio, json, sys

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne')
from server.ai_rnd_service import atlas_fidelity

doc = asyncio.get_event_loop().run_until_complete(atlas_fidelity('qwen3-4b'))
assert doc['schema'] == 'atlas_fidelity_v2'
phs = {p['phase']: p for p in doc['phases']}
assert set(phs) == {'3239', '3240', '3246', '3247', '3248', '3249', '3250'}, set(phs)
# sealed numbers cross-check: 3250 XC_p5_b8
p50 = phs['3250']
i = p50['recipes'].index('XC_p5_b8')
r = json.load(open(r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913\phase3250\g5h13_kernel_boundary\result.json', encoding='utf-8'))
agg = r['aggregate']['XC_p5_b8']
assert abs(p50['agg_fid_matrix'][i][p50 is not None and 4] - agg['fam_fid_mean']['SCLS']) < 1e-6
assert abs(p50['agg_r2'][i] - agg['r2_oov_mean']) < 1e-6
assert abs(p50['agg_kl'][i] - agg['kl_mean']) < 1e-6
# incumbent block
inc = {x['recipe']: x for x in doc['incumbent']}
assert set(inc) == {'XC_p5_b8', 'XC_p5_b4', 'XC_p3_b6', 'K_p3_T3'}
assert inc['XC_p5_b8']['tag'] == '现役综合最优'
assert abs(inc['XC_p5_b8']['fam_fid']['SCLS'] - agg['fam_fid_mean']['SCLS']) < 1e-6
# milestone lookups (frontend dependencies)
assert phs['3246']['recipes'].count('XC_b4') == 1
assert phs['3247']['recipes'].count('XC_poly_b4') == 1
assert phs['3248']['recipes'].count('XC_p3_b4') == 1
assert phs['3250']['recipes'].count('ORC_ls_cls') == 1
# closure present where the sealed npz carries it (3246 sealed npz has none)
for ph in ('3247', '3248', '3249', '3250'):
    assert 'closure' in phs[ph], ph
assert 'closure' not in phs['3246'] and 'closure' not in phs['3239']
print('ENDPOINT CHECK OK: 7 phases, incumbent 4 items, sealed numbers match, milestone lookups OK')
