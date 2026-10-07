# -*- coding: utf-8 -*-
"""Phase 12 元数据探针：确认可继承产物的存在性与 sha256，检查 Phase 11 result 的键结构。"""
import os, io, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


CAND = [
    r'tests\deepseek_temp\Phase11\result_phase11.json',
    r'tests\deepseek_temp\Phase11\execution_phase11.json',
    r'tests\deepseek_temp\Phase10\result_phase10.json',
    r'tests\deepseek_temp\Phase10\execution_phase10.json',
    r'tests\deepseek_temp\Phase8\execution_phase8.json',
    r'tests\deepseek_temp\Phase9\result_phase9.json',
    r'tests\deepseek_temp\Phase9\execution_phase9.json',
    r'tests\deepseek_temp\Phase8\result_phase8.json',
    r'models\hf\qwen3-4b\config.json',
]
out = []
for rel in CAND:
    p = os.path.join(ROOT, rel)
    ex = os.path.exists(p)
    out.append('%-58s exists=%s %s' % (rel, ex, sha(p) if ex else ''))
print(chr(10).join(out))

R = json.load(io.open(os.path.join(ROOT, CAND[0]), encoding='utf-8'))
print()
print('--- result_phase11 top keys ---')
print(sorted(R.keys()))
print()
print('profile_abs["6"] keys:', sorted(R['profile_abs']['6'].keys()))
print('profile_abs["6"] J/x*/cls:', R['profile_abs']['6']['jump_ratio'], R['profile_abs']['6']['x_star'], R['profile_abs']['6']['cls'])
print('profile_abs["34"] J/x*/cls:', R['profile_abs']['34']['jump_ratio'], R['profile_abs']['34']['x_star'], R['profile_abs']['34']['cls'])
print('J_inject profile:')
S = [6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34]
for s in S:
    d = R['profile_abs'][str(s)]
    print('   L%-3d J=%8.4f x*=%s cls=%s' % (s, d['jump_ratio'], ('%.4f' % d['x_star']) if d['x_star'] is not None else 'n/a', d['cls']))
print()
print('E3_verdict.L:', R['E3_verdict'].get('L'))
print('E4:', R['E4'])
print('full_L6:', R['full_L6'])
print('dose_coord.rbar_ell:', R['dose_coord']['rbar_ell'])
print('dose_coord.r_R:', R['dose_coord']['r_R'])
print('dose_coord.mean_n6:', R['dose_coord']['mean_n6'])
print('E1 sites:', sorted(R['E1'].keys()))
print('E1["6"] alphas:', [r['alpha'] for r in R['E1']['6']])
print('verdict:', R['verdict'])
