# -*- coding: utf-8 -*-
"""Phase 3164 独立磁盘复核（独立进程 re-hash + seal 字节级重构 + 结构断言）。"""
import os, sys, json, io, hashlib
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(RDIR, 'phase3164')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3164_verify_out.txt')
R = []
NP = 0

def chk(name, ok, note=''):
    global NP
    NP += 1
    R.append('%s %s%s' % ('PASS' if ok else 'FAIL', name, (' | %s' % note) if note else ''))

def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]

def seal_recon(p):
    raw = json.load(io.open(p, encoding='utf-8'))
    seal = raw.pop('seal_sha8')
    blob = json.dumps(raw, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    return sha8(blob) == seal, seal

# 1. seal 字节级重构（json.dump Windows CRLF -> dumps LF 不匹配 3161 教训；
#    本套 seal = sha8(磁盘文件首写含 res_sha8 无 seal_sha8 的字节)。重构：读文件、pop seal、
#    按 sort_keys/indent=1 dumps -> LF；而磁盘 json.dump 是 indent=1 无 sort（3164 seal_result
#    用 sort_keys=True dumps 后写、再读回加 seal 再 dump）-> 直接对拍 seal 字段与
#    「re-dump(原字段除 seal) 的 LF blob」不成立时用「磁盘文本 pop seal 行」法。
def seal_recon_disk(p):
    raw = json.load(io.open(p, encoding='utf-8'))
    seal = raw.pop('seal_sha8')
    txt = io.open(p, encoding='utf-8', newline='').read()
    # 删除顶层 seal_sha8 字段（最后一个键，前一行尾逗号一并去掉），保持 CRLF 字节不变
    import re
    ms_ = list(re.finditer(r',\r?\n[ \t]*"seal_sha8": "[0-9a-f]{8}"', txt))
    if not ms_:
        return False, seal
    m = ms_[-1]  # 顶层 seal_sha8 是最后一次 dump 的最后一个字段；嵌套 models 也有同名键
    blob = (txt[:m.start()] + txt[m.end():]).encode('utf-8')
    return sha8(blob) == seal, seal

for sub, names in [
    (os.path.join(PDIR, 'g5a2_c_steer'), ['result_qwen3-14b.json', 'result_glm4.json', 'result_summary.json']),
    (os.path.join(RDIR, 'phase3164', 'g5a2b_position_shift_cross_model', 'qwen3-14b'), ['result.json']),
    (os.path.join(RDIR, 'phase3164', 'g5a2b_position_shift_cross_model', 'glm4'), ['result.json']),
    (os.path.join(RDIR, 'phase3164', 'g5a2b_position_shift_cross_model', 'summary'), ['result_summary.json']),
    (os.path.join(PDIR, 'g5a2c_massive_cross_model', 'qwen3-4b'), ['result.json']),
    (os.path.join(PDIR, 'g5a2c_massive_cross_model', 'qwen3-14b'), ['result.json']),
    (os.path.join(PDIR, 'g5a2c_massive_cross_model', 'glm4'), ['result.json']),
    (os.path.join(PDIR, 'g5a2c_massive_cross_model', 'summary'), ['result_summary.json']),
]:
    for nm in names:
        p = os.path.join(sub, nm)
        if not os.path.exists(p):
            chk('seal %s' % nm, False, 'missing %s' % p)
            continue
        ok, seal = seal_recon_disk(p)
        chk('seal reconstruct %s' % nm, ok, 'seal=%s' % seal)

# 2. verdict/sha 字段一致性 + 判决读数
import json as J
A14 = J.load(io.open(os.path.join(PDIR, 'g5a2_c_steer', 'result_qwen3-14b.json'), encoding='utf-8'))
AGL = J.load(io.open(os.path.join(PDIR, 'g5a2_c_steer', 'result_glm4.json'), encoding='utf-8'))
B14 = J.load(io.open(os.path.join(RDIR, 'phase3164', 'g5a2b_position_shift_cross_model', 'qwen3-14b', 'result.json'), encoding='utf-8'))
BGL = J.load(io.open(os.path.join(RDIR, 'phase3164', 'g5a2b_position_shift_cross_model', 'glm4', 'result.json'), encoding='utf-8'))
C4 = J.load(io.open(os.path.join(PDIR, 'g5a2c_massive_cross_model', 'qwen3-4b', 'result.json'), encoding='utf-8'))
C14 = J.load(io.open(os.path.join(PDIR, 'g5a2c_massive_cross_model', 'qwen3-14b', 'result.json'), encoding='utf-8'))
CGL = J.load(io.open(os.path.join(PDIR, 'g5a2c_massive_cross_model', 'glm4', 'result.json'), encoding='utf-8'))
chk('A 14b cls zero_like_q06', A14['cls'] == 'zero_like_q06', 'C=%s' % A14['C_steer_main']['value'])
chk('A glm4 cls zero_like_q06', AGL['cls'] == 'zero_like_q06', 'C=%s' % AGL['C_steer_main']['value'])
chk('A 14b identity==0', A14['floors']['F1_identity_maxd'] == 0.0)
chk('A glm4 identity==0', AGL['floors']['F1_identity_maxd'] == 0.0)
chk('A 14b cells 441', A14['cells']['total'] == 441, 'eligible=%s' % A14['cells']['eligible'])
chk('A glm4 cells 441', AGL['cells']['total'] == 441)
chk('B 14b supported', B14['cls'] == 'rope_relative_supported', 'KL_B=%s' % B14['kl_b_max'])
chk('B glm4 supported', BGL['cls'] == 'rope_relative_supported', 'KL_B=%s' % BGL['kl_b_max'])
chk('B 14b top1 all', B14['top1_b_ok'] == B14['top1_b_tot'])
chk('B glm4 top1 all', BGL['top1_b_ok'] == BGL['top1_b_tot'])
chk('C d1 match x3', all(r['d1_match'] for r in (C4, C14, CGL)),
    'd1=%s' % [r['d1_3157'] for r in (C4, C14, CGL)])
chk('C cls agree', len({C4['cls'], C14['cls'], CGL['cls']}) == 1, C4['cls'])

# 3. parts 结构（cells 441 无重不漏，per model）
for m in ('qwen3-14b', 'glm4'):
    pdir = os.path.join(PDIR, 'g5a2_c_steer', '_parts_%s' % m)
    keys = set()
    okk = True
    for k in range(4):
        pp = os.path.join(pdir, 'anchor%d.json' % k)
        if not os.path.exists(pp):
            okk = False
            break
        d = J.load(io.open(pp, encoding='utf-8'))
        keys |= set(d['rows'].keys())
    chk('parts %s 441 cells no-dup' % m, okk and len(keys) == 441, 'n=%s' % len(keys))
    zp = os.path.join(pdir, 'axis.npz')
    if os.path.exists(zp):
        z = np.load(zp)
        chk('axis %s v1 unit' % m, abs(float(np.linalg.norm(z['v1'])) - 1.0) < 1e-9)

# 4. 轴(b) collect.npz 在盘
for m in ('qwen3-14b', 'glm4'):
    p = os.path.join(RDIR, 'phase3164', 'g5a2b_position_shift_cross_model', m, 'collect.npz')
    ok = os.path.exists(p)
    if ok:
        z = np.load(p)
        ok = z['H'].shape[0] == 36 and int(B14['n_seq']) == 36
    chk('b collect.npz %s 36 seqs' % m, ok)

# 5. 五写在场
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
led = J.load(io.open(LEDGER, encoding='utf-8'))
items = led['measurements'] if isinstance(led, dict) and 'measurements' in led else led
chk('ledger n==316', len(items) == 316, 'n=%d' % len(items))
chk('ledger 3164 entry', any(int(it.get('phase', 0)) == 3164 for it in items))
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
mtxt = io.open(MEMO, encoding='utf-8').read()
chk('MEMO Phase 3164', '## Phase 3164' in mtxt)
chk('MEMO prereg 3165', '预注册 Phase 3165' in mtxt)
import time as _t
daily = os.path.join(ROOT, '.workbuddy', 'memory', _t.strftime('%Y-%m-%d') + '.md')
chk('daily 3164', os.path.exists(daily) and '3164' in io.open(daily, encoding='utf-8').read())
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
wtxt = io.open(WMEM, encoding='utf-8').read()
chk('MEMORY 3164', '3164 图谱缺口②跨模型同口径复测闭环' in wtxt)

open(OUTP, 'w', encoding='utf-8').write('\n'.join(R) + '\nTOTAL %d/%d PASS\n' % (
    sum(1 for r in R if r.startswith('PASS')), NP))
print('\n'.join(R))
print('TOTAL %d/%d PASS' % (sum(1 for r in R if r.startswith('PASS')), NP))
sys.exit(0 if all(r.startswith('PASS') for r in R) else 1)
