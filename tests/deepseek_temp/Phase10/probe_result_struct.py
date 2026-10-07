# -*- coding: utf-8 -*-
import json, io
R = json.load(open(r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase10\result_phase10.json', encoding='utf-8'))
J = json.load(open(r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase10\judgement_phase10.json', encoding='utf-8'))
out = []
out.append('=== result top-level keys ===')
for k in R.keys():
    v = R[k]
    t = type(v).__name__
    if isinstance(v, dict):
        out.append('%-22s dict len=%d keys=%s' % (k, len(v), list(v.keys())[:14]))
    elif isinstance(v, list):
        out.append('%-22s list len=%d  head=%s' % (k, len(v), v[:4]))
    else:
        out.append('%-22s %s = %s' % (k, t, v))
out.append('')
out.append('=== judgement top-level keys ===')
for k in J.keys():
    v = J[k]
    if isinstance(v, dict):
        out.append('%-24s dict len=%d keys=%s' % (k, len(v), list(v.keys())[:20]))
    elif isinstance(v, list):
        out.append('%-24s list len=%d head=%s' % (k, len(v), v[:6]))
    else:
        out.append('%-24s %s = %s' % (k, type(v).__name__, v))
# 针对关键派生字段展开
for key in ['curve','curves','classes','E3','E4','E6','floors','decision','decisions','J_profile','xstar','spearman','conf',
            'own_basis','basis','F1','F3','F6','verdict','labels','classifier','x_star','y_sat']:
    if key in R:
        out.append('')
        out.append('--- R[%s] ---' % key)
        out.append(json.dumps(R[key], ensure_ascii=False, indent=1)[:3500])
open(r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase10\_probe_result_struct.txt','w',encoding='utf-8').write('\n'.join(out))
print('done', len(out))
