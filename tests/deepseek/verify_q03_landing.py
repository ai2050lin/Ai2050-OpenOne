# -*- coding: utf-8 -*-
"""Q03 落账独立复核（新进程）：memo Phase 37 + queue Q03 sealed + 受保护文件。"""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
QD = os.path.join(ROOT, 'research', 'deepseek', 'atlas', 'phase_queue_v1.json')
OUT = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'verify_q03_landing.txt')
rows = []
P = F = 0

def chk(name, ok, detail=''):
    global P, F
    if ok:
        P += 1
    else:
        F += 1
    rows.append('  [%s] %-52s %s' % ('PASS' if ok else 'FAIL', name, detail))

def sha8b(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

rows.append('Q03 落账独立复核（新进程，重新读盘）')
rows.append('=' * 74)

# ---- 1) memo ----
rows.append('1) 研究日志（AGI_DEEPSEEK_MEMO.md）')
raw = open(MEMO, 'rb').read()
t = raw.decode('utf-8-sig')
chk('memo 当前 sha8/大小', len(raw) == 700044 and sha8b(MEMO) == 'df987b1f',
    '%d B / %s' % (len(raw), sha8b(MEMO)))
chk('pre-Phase37 前缀完整 (692727 B = 5b3bdcec)',
    hashlib.sha256(raw[:692727]).hexdigest()[:8] == '5b3bdcec',
    hashlib.sha256(raw[:692727]).hexdigest()[:8])
chk('Phase 37 在位', '## Phase 37' in t, '')
i = t.find('## Phase 37')
tail = t[i:]
tl = tail.split('\n')
chk('Phase 37 区间为纯 CRLF（bare_lf=0）',
    tail.count('\n') - tail.count('\r\n') == 0,
    'bare_lf=%d' % (tail.count('\n') - tail.count('\r\n')))
chk('Phase 37 含 E_read 池化值 0.373350', '0.373350' in tail, '')
chk('Phase 37 含逐位复现 drift=0.00e+00', '0.00e+00' in tail, '')
chk('Phase 37 含 5% 门 0/3', '0/3' in tail, '')
chk('Phase 37 含 6.63× 比率', '6.63' in tail, '')
chk('Phase 37 记录并发写者 45 裸 LF', '45 个裸 LF' in tail, '')
chk('Phase 37 记录 design_sha 9a26f72f', '9a26f72f' in tail, '')
chk('Phase 36（他人）仍完整', '## Phase 36' in t and 'E4/E4b' in t, '')
chk('Phase 35（本线 R8）仍完整', 'A 闸门 seal 执行与关闭' in t, '')

# ---- 2) queue ----
rows.append('2) 议程队列（phase_queue_v1.json）')
qd = json.loads(open(QD, 'rb').read().decode('utf-8-sig'))
q03 = [x for x in qd['queue'] if x['id'] == 'Q03'][0]
chk('Q03.status == sealed', q03['status'] == 'sealed', q03['status'])
chk('Q03.sealed_at 存在', bool(q03.get('sealed_at')), str(q03.get('sealed_at')))
chk('Q03.seal_record 指向 q03_result.json',
    q03.get('seal_record', '').endswith('q03_result.json'), q03.get('seal_record', ''))
chk('sealed_items 含 Q03', 'Q03' in qd['sealed_items'], str(qd['sealed_items']))
chk('sealed_items 计数 = 6', len(qd['sealed_items']) == 6, str(len(qd['sealed_items'])))
pend = [x['id'] for x in qd['queue'] if x['status'] == 'pending']
chk('pending 计数 = 24', len(pend) == 24, str(len(pend)))
chk('Q04 仍 pending（下一项）', 'Q04' in pend, '')
chk('queue count = 30', qd.get('count') == 30, str(qd.get('count')))
chk('status_updated_by = 本 Phase 脚本',
    qd.get('status_updated_by', '').endswith('append_memo_q03_phase37.py'), qd.get('status_updated_by', ''))

# ---- 3) 受保护文件 ----
rows.append('3) 受保护（跨线/冻结）文件未变')
prot = [
    ('AGI_GPT5_MEMO.md', os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md'), '2a84776b'),
    ('atlas_ledger.json', os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'), 'bbda63df'),
    ('proposition_ledger.json',
     os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913\phase3103\omega_p101_formula_audit\proposition_ledger.json'), '9a3c6ff4'),
    ('metric_dict.json', os.path.join(ROOT, r'research\deepseek\atlas\metric_dict.json'), '03887e51'),
    ('RDC_TESTPLAN_v1.md(archive)', os.path.join(ROOT, r'tests\deepseek_temp\_archive_r6\gpt5_docs\RDC_TESTPLAN_v1.md'), '71b85673'),
    ('3152 script', os.path.join(ROOT, r'tests\glm5\phase3152_g1p2_tri_model_k1.py'), 'd795f41c'),
    ('3151 script', os.path.join(ROOT, r'tests\glm5\phase3151_g1p1_combo_additive_vs_interaction.py'), '292ed9f3'),
]
for lbl, p, e in prot:
    if not os.path.exists(p):
        chk('%s sha8' % lbl, False, 'MISSING')
        continue
    chk('%s sha8' % lbl, sha8b(p) == e, '%s == %s' % (sha8b(p), e))

# ---- 4) 产物 ----
rows.append('4) Q03 产物指纹')
for lbl in ['q03_execution.json', 'q03_result.json', 'q03_report.txt', 'verify_q03.txt']:
    p = os.path.join(ROOT, 'tests', 'deepseek', 'result', lbl)
    rows.append('     %-24s %8d B  %s' % (lbl, os.path.getsize(p), sha8b(p)))

rows.append('')
rows.append('=' * 74)
rows.append('汇总: PASS=%d  FAIL=%d  ->  %s' % (P, F, 'ALL_PASS' if F == 0 else 'HAS_FAILURE'))
txt = '\n'.join(rows)
open(OUT, 'w', encoding='utf-8').write(txt + '\n')
print(txt)
