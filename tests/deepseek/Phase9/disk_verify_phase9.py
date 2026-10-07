# -*- coding: utf-8 -*-
"""Phase 9 独立磁盘复核：产物存在性 / 哈希 / 备忘录基线对账 / Ledger / 判决一致性 / 文档锚点。"""
import os, io, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P9 = os.path.join(ROOT, 'tests', 'deepseek', 'Phase9')
P9T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase9')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
WLOG = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')
MEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
SKILL = os.path.join(os.path.expanduser('~'), '.workbuddy', 'skills', 'rdc-main-axis-probe', 'SKILL.md')

o = []
def w(s=''):
    o.append(str(s)); print(s)

def sha8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

ok = []
def chk(tag, cond, detail=''):
    ok.append(bool(cond))
    w('[%s] %-46s %s' % ('OK ' if cond else 'FAIL', tag, detail))

# ---------- [1] 脚本与产物 ----------
w('=== [1] Phase9 脚本 / 产物存在性 ===')
for d, tag in [(P9, 'deepseek'), (P9T, 'deepseek_temp')]:
    fs = sorted(os.listdir(d))
    w('    %s (%d): %s' % (tag, len(fs), ', '.join(fs)))
need_s = ['n2h1a2_threshold_gain.py', 'gen_execution_phase9.py', 'run_phase9.py',
          'closeout_phase9.py', 'do_append_phase9.py', 'closeout_docs.py', 'disk_verify_phase9.py']
chk('脚本 7 件齐备', all(os.path.isfile(os.path.join(P9, f)) for f in need_s))
need_t = ['N2h1a2_design_seal.json', 'N2h1a2_design_seal_amend1.json', 'execution_phase9.json',
          'n2h1a2_report_qwen3-4b.txt', 'result_phase9.json', 'judgement_phase9.json',
          'memo_append_phase9.md', 'verify_append_phase9.txt', 'closeout_phase9.txt',
          'closeout_docs.txt', 'atlas_ledger_backup_pre_phase9.json',
          'memo_baseline_preappend_phase9.json', '_formal_stdout.log', '_smoke_stdout.log']
chk('temp 产物 14 件齐备', all(os.path.isfile(os.path.join(P9T, f)) for f in need_t),
    '缺: %s' % [f for f in need_t if not os.path.isfile(os.path.join(P9T, f))])
chk('smoke/ 三件套', all(os.path.isfile(os.path.join(P9T, 'smoke', f)) for f in
                        ['result_phase9.json', 'n2h1a2_report_qwen3-4b.txt']))

# ---------- [2] sha 与冻结一致性 ----------
w('')
w('=== [2] 冻结与结果 sha ===')
R = json.load(io.open(os.path.join(P9T, 'result_phase9.json'), encoding='utf-8'))
J = json.load(io.open(os.path.join(P9T, 'judgement_phase9.json'), encoding='utf-8'))
execj = json.load(io.open(os.path.join(P9T, 'execution_phase9.json'), encoding='utf-8'))
chk('result.seal_sha8 == 实际 seal sha8', R['seal_sha8'] == sha8(os.path.join(P9T, 'N2h1a2_design_seal.json')),
    '%s vs %s' % (R['seal_sha8'], sha8(os.path.join(P9T, 'N2h1a2_design_seal.json'))))
chk('result.exec_sha8 == 实际 exec sha8', R['exec_sha8'] == sha8(os.path.join(P9T, 'execution_phase9.json')),
    '%s vs %s' % (R['exec_sha8'], sha8(os.path.join(P9T, 'execution_phase9.json'))))
chk('judgement.meta 三个 sha8 与 result 一致',
    J['meta']['seal_sha8'] == R['seal_sha8'] and J['meta']['exec_sha8'] == R['exec_sha8']
    and J['meta']['result_sha8'] == sha8(os.path.join(P9T, 'result_phase9.json')))
# F5 面板继承
E8 = json.load(io.open(os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8', 'execution_phase8.json'),
                       encoding='utf-8'))
chk('F5 面板逐元素继承 Phase 8',
    json.dumps(E8['discovery'], sort_keys=True) == json.dumps(execj['discovery'], sort_keys=True)
    and json.dumps(E8['confirmation'], sort_keys=True) == json.dumps(execj['confirmation'], sort_keys=True)
    and json.dumps(E8['pairs_all'], sort_keys=True) == json.dumps(execj['pairs_all'], sort_keys=True))
_p8exec_sha = sha8(os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase8', 'execution_phase8.json'))
chk('inherits_panel_sha256 前缀 == Phase8 exec sha8',
    execj['inherits_panel_sha256'][:8] == _p8exec_sha,
    '%s vs %s' % (execj['inherits_panel_sha256'][:8], _p8exec_sha))

# ---------- [3] 关键数字（比特级与统计级）----------
w('')
w('=== [3] 关键数字 ===')
chk('D1@alpha=1 与 Phase 8 diff6 臂逐位相同', abs(R['full'] - 10.574739583333335) < 1e-12,
    '%.15f' % R['full'])
amp = R['curves']['amp']
chk('amp 平坦：J < 2 且幂律 gamma < 0.5', amp['jump_ratio'] < 2.0 and amp['gamma'] < 0.5,
    'J=%.2f gamma=%.3f span %.4f->%.4f' % (amp['jump_ratio'], amp['gamma'], amp['y'][0], amp['y'][-1]))
d7 = R['D7_detail']
chk('D7 kill_frac < 0.10（上游不必要）', d7['kill_frac'] < 0.10, 'kill=%.4f' % d7['kill_frac'])
chk('D7 alpha=1 恒等偏差 == 0', abs(d7['identity_dev']) < 1e-12)
chk('F3 恒等自检两站点 == 0', R['floors']['F3_dev']['L6out'] < 1e-6 and R['floors']['F3_dev']['L5out'] < 1e-6)
chk('F1 形式失败已如实记录', R['floors']['F1_ok'] is False, 'D4_max=%.4f' % R['floors']['D4_max'])
chk('判决为 H0 三连（预注册树）', R['verdict']['D1'].startswith('H0') and R['verdict']['D2'].startswith('H0')
    and R['verdict']['amp'].startswith('H0'))
chk('确认集同判', R['confirmation']['same_verdict_D1'] and R['confirmation']['same_verdict_D2'],
    'full_ratio=%.3f' % R['confirmation']['full_ratio'])

# ---------- [4] Ledger ----------
w('')
w('=== [4] Ledger ===')
L = json.load(io.open(LEDGER, encoding='utf-8'))
chk('measurements n == 292', len(L['measurements']) == 292, 'n=%d' % len(L['measurements']))
chk('尾条为 Phase 9', L['measurements'][-1]['phase'] == 9,
    L['measurements'][-1]['verdict'][:70])
bk = os.path.join(P9T, 'atlas_ledger_backup_pre_phase9.json')
chk('Ledger 备份存在且可解析', os.path.isfile(bk) and bool(json.load(io.open(bk, encoding='utf-8'))),
    'backup sha8 %s' % sha8(bk))

# ---------- [5] 备忘录 ----------
w('')
w('=== [5] 备忘录与基线 ===')
mb = open(MEMO, 'rb').read()
T = mb.decode('utf-8-sig')
lines = T.splitlines()
B = json.load(io.open(os.path.join(INFRA, 'memo_baseline.json'), encoding='utf-8'))
chk('bytes/lines/sha256 与基线一致',
    B['bytes'] == len(mb) and B['lines'] == len(lines) and B['sha256'] == hashlib.sha256(mb).hexdigest(),
    '%d / %d / %s' % (len(mb), len(lines), hashlib.sha256(mb).hexdigest()[:8]))
chk('BOM 保留 / bare_lf == 0', mb[:3] == b'\xef\xbb\xbf' and (mb.count(b'\n') - mb.count(b'\r\n')) == 0,
    'crlf=%d' % mb.count(b'\r\n'))
ph = [i + 1 for i, l in enumerate(lines) if l.startswith('## Phase ')]
chk('Phase 节 9 个，Phase9 标题行存在', len(ph) == 9 and any(l.startswith('## Phase 9:') for l in lines),
    'lines=%s' % ph)
for k in ['10.574739583333335', 'J=1.42', 'J=5.41', '0.0578', '0.169', '4.224', 'e99ebd02', 'ece7ed8c']:
    chk('memo 含锚点 %s' % k, k in T, 'count=%d' % T.count(k))
chk('Phase 9 生效否证语句存在', 'Phase 8 §7' in T and '否证' in T)

# ---------- [6] 文档锚点 ----------
w('')
w('=== [6] wlog / MEMORY / 技能 ===')
WL = io.open(WLOG, encoding='utf-8', errors='replace').read()
chk('wlog 含 Phase 9 节', 'Phase 9 / N2h1-α-2' in WL, 'wlog bytes %d' % os.path.getsize(WLOG))
MM = io.open(MEM, encoding='utf-8', errors='replace').read()
for k in ['Phase 9', 'x\\* ≈ 0.59–0.63', '10.574739583333335', 'n=**292**', 'amp', '装置铁律（Phase 9 新增）']:
    chk('MEMORY 含 %s' % k, k in MM, 'count=%d' % MM.count(k))
SK = io.open(SKILL, encoding='utf-8', errors='replace').read() if os.path.isfile(SKILL) else ''
chk('技能已更新（含 Phase 9 关键词）',
    ('剂量' in SK or 'dose' in SK) and ('amp' in SK) and ('x*' in SK or '半饱和' in SK),
    'skill bytes %d' % len(SK))

w('')
w('=== 汇总: %d/%d 项通过 ===' % (sum(ok), len(ok)))
OUT = os.path.join(P9T, 'disk_verify_phase9.txt')
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(o))
print('DONE ->', OUT)
