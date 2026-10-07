# -*- coding: utf-8 -*-
# p3154_verify.py: Phase 3154 独立磁盘复核（独立进程 re-hash + 标记校验）
import os, io, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3154', 'g1p4_mfd_multifactor_disentangle')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')
MEMORY = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')

PASS = []
FAIL = []

def chk(name, cond, detail=''):
    (PASS if cond else FAIL).append('%s %s' % (name, detail))

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

# 1. 产物磁盘哈希（与 digest/rev_note 记录对照）
EXP = {
    ('qwen3-4b', 'result.json'): '3bf5a776',
    ('qwen3-14b', 'result.json'): 'd008ff39',
    ('glm4', 'result.json'): 'fa5ab91a',
    ('summary', 'result_summary.json'): '9fede5d5',
    ('qwen3-4b', 'collect.npz'): 'bb99438f',
    ('qwen3-14b', 'collect.npz'): '6d142b9d',
    ('glm4', 'collect.npz'): 'cf2659f1',
}
for (mode, fn), want in EXP.items():
    p = os.path.join(RDIR, mode, fn)
    chk('exists %s/%s' % (mode, fn), os.path.exists(p))
    if os.path.exists(p):
        got = sha8(p)
        chk('sha8 %s/%s' % (mode, fn), got == want, 'got=%s want=%s' % (got, want))
for mode in ('qwen3-4b', 'qwen3-14b', 'glm4', 'summary'):
    chk('execution.json %s' % mode, os.path.exists(os.path.join(RDIR, mode, 'execution.json')))
    d = json.load(io.open(os.path.join(RDIR, mode, 'execution.json'), encoding='utf-8'))
    chk('design frozen before obs %s' % mode, d.get('frozen_before') == 'any model observation')

# 2. result 内嵌 res_sha8 与 verdict 尾部一致（seal 回填双写校验）
for mode, fn in [('qwen3-4b', 'result.json'), ('qwen3-14b', 'result.json'),
                 ('glm4', 'result.json'), ('summary', 'result_summary.json')]:
    r = json.load(io.open(os.path.join(RDIR, mode, fn), encoding='utf-8'))
    chk('verdict tail==res_sha8 %s' % mode,
        r['verdict'].endswith('sha8_' + r['res_sha8']),
        '%s vs %s' % (r['verdict'][-8:], r['res_sha8']))
    chk('seal field present %s' % mode, bool(r.get('seal_sha8')))

# 3. Ledger
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms = led['measurements']
chk('ledger n=305', len(ms) == 305, 'n=%d' % len(ms))
chk('last entry phase=3154', ms[-1].get('phase') == 3154)
chk('ledger 3154 not duplicated', sum(1 for m in ms if m.get('phase') == 3154) == 1)
chk('chain field', isinstance(led.get('ledger_sha256_8'), str) and len(led['ledger_sha256_8']) == 8)
chk('N-line entries intact (phases 16-21 present)',
    any(m.get('phase') == 16 for m in ms) and any(m.get('phase') == 21 for m in ms))

# 4. MEMO
txt = open(MEMO, 'rb').read().decode('utf-8')
chk('memo BOM head', txt.startswith('\ufeff## AGI'))
chk('memo 3154 section', '## Phase 3154: 多因素混杂分解（G1-P4）' in txt)
chk('memo 3155 prereg', '### 预注册 Phase 3155：G2-P1 多关系族与算子可分离性' in txt)
chk('memo old prereg intact (原 3154 顺延说明)', '顺延为 Phase 3155' in txt)
chk('memo 3153 section intact', '## Phase 3153: 失败模态解剖（G1-P3）' in txt)

# 5. Daily + MEMORY
chk('daily 3154', '## Phase 3154 (gpt5 线)' in io.open(DAILY, encoding='utf-8').read())
chk('memory G-line block', '## G 线（AGI_GPT5_MEMO）3154 状态' in io.open(MEMORY, encoding='utf-8').read())
chk('memory deepseek section intact', '## 5 下一步 / 死线' in io.open(MEMORY, encoding='utf-8').read())

# 6. SMOKE 产物在位
chk('smoke result', os.path.exists(os.path.join(RDIR, 'qwen3-4b', 'smoke', 'result.json')))
chk('smoke collect', os.path.exists(os.path.join(RDIR, 'qwen3-4b', 'smoke', 'collect_smoke.npz')))

rep = ['PASS=%d FAIL=%d' % (len(PASS), len(FAIL))] + ['FAIL: ' + f for f in FAIL] + PASS
io.open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3154_verify_out.txt'), 'w',
        encoding='utf-8').write('\n'.join(rep))
print('VERIFY DONE PASS=%d FAIL=%d' % (len(PASS), len(FAIL)))
