# -*- coding: utf-8 -*-
import json
P = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase10\result_phase10.json'
R = json.load(open(P, encoding='utf-8'))
out = []
for k in ['profile_abs', 'profile_rel']:
    d = R[k]
    out.append('=== %s : keys=%s' % (k, list(d.keys())))
    for site in ['6', '7', '12', '22', '34']:
        if site in d:
            v = d[site]
            if isinstance(v, dict):
                out.append('  [%s] keys=%s' % (site, list(v.keys())))
                out.append('      ' + json.dumps({kk: (vv if not isinstance(vv, list) else vv) for kk, vv in v.items()}, ensure_ascii=False)[:1200])
            else:
                out.append('  [%s] %s' % (site, json.dumps(v, ensure_ascii=False)[:600]))
out.append('')
out.append('=== profile_R_ext ===')
out.append(json.dumps(R['profile_R_ext'], ensure_ascii=False, indent=1))
out.append('')
out.append('=== E5 ===')
out.append(json.dumps(R['E5'], ensure_ascii=False, indent=1))
out.append('')
out.append('=== x_star_rel (head) ===')
out.append(json.dumps({k: R['x_star_rel'][k] for k in list(R['x_star_rel'].keys())}, ensure_ascii=False, indent=1))
out.append('')
out.append('=== E3 other sites (12/20/34 rows alpha=1) ===')
for s in ['7', '12', '20', '34']:
    e = R['E3'][s]
    out.append('  [%s] overlap=%.6f rows=%s pc=%s' % (s, e['overlap'],
        [(r['alpha'], round(r['dDonor'], 6)) for r in e['rows']], [round(x, 4) for x in e['principal_cos']]))
open(r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase10\_probe_result_struct2.txt', 'w', encoding='utf-8').write('\n'.join(out))
print('done')
