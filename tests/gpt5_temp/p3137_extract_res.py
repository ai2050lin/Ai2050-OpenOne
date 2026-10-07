# -*- coding: utf-8 -*-
import json, io, hashlib
import numpy as np
RDIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
D37 = RDIR + r'\phase3137\omega_p135_modecoop_k0anat_coorddecomp_l35recheck'
out = []
raw = io.open(D37 + r'\result.json', 'rb').read()
out.append('res37 sha8 = %s' % hashlib.sha256(raw).hexdigest()[:8])
r37 = json.loads(raw.decode('utf-8'))
out.append('verdict = %s' % r37['verdict'])
out.append('runtime=%.1f smoke=%s seal=%s' % (r37['runtime_s'], r37['smoke'], r37['seal_sha8']))
pa = r37['part_a']
out.append('part_a keys = %s' % sorted(pa.keys()))
for k in sorted(pa.keys()):
    v = pa[k]
    if isinstance(v, (int, float, str, bool)):
        out.append('pa.%s = %r' % (k, v))
out.append('frozen keys = %s' % sorted(r37.get('frozen', {}).keys()) if 'frozen' in r37 else 'no frozen key')
pe = r37['part_e']
out.append('part_e keys = %s' % sorted(pe.keys()))
if 'trials' in pe:
    out.append('e_trials = %s' % json.dumps({k: {'chg': v['chg'], 'first': v['first']} for k, v in pe['trials'].items()}))
if 'mode_gap' in pe:
    out.append('mode_gap = %s' % json.dumps(pe['mode_gap']))
if 'gates' in pe:
    out.append('e_gates = %s' % json.dumps(pe['gates']))
if 'soft' in pe:
    out.append('e_soft = %s' % json.dumps(pe['soft']))
pf = r37['part_f']
out.append('part_f = %s' % json.dumps(pf, sort_keys=True)[:1200])
pg = r37['part_g']
out.append('part_g = %s' % json.dumps(pg, sort_keys=True)[:1200])
ph = r37['part_h']
out.append('part_h = %s' % json.dumps(ph, sort_keys=True)[:800])
z37 = np.load(D37 + r'\p135_readout.npz', allow_pickle=False)
out.append('npz keys n=%d: %s' % (len(z37.files), sorted(z37.files)))
for k in sorted(z37.files):
    a = z37[k]
    out.append('npz %s shape=%s dtype=%s sha8=%s' % (k, a.shape, a.dtype, hashlib.sha256(a.tobytes()).hexdigest()[:8]))
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3137_res.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('OK')
