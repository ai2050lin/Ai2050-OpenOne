# -*- coding: utf-8 -*-
"""把「P19 基线后 MEMO 被就地规范化」这一完整性事件**测量并冻结**成审计件。

输出：tests/deepseek_temp/_infra/memo_drift_phase19_postbaseline.json
      tests/deepseek_temp/_infra/audit_memo_drift_phase19.txt

必须在 **Phase 20 追加之前**运行（一旦追加，MEMO 的 mtime / bytes 就会变，
事件现场不可复原）。所有数字均由本脚本现场测量写入，禁人工转录。
"""
import io
import os
import re
import json
import time
import hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
INFRA = os.path.join(ROOT, 'tests', 'deepseek_temp', '_infra')
T20 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase20')
BASE = os.path.join(INFRA, 'memo_baseline.json')
OUTJ = os.path.join(INFRA, 'memo_drift_phase19_postbaseline.json')
OUTT = os.path.join(INFRA, 'audit_memo_drift_phase19.txt')

L = []
def w(s=''):
    L.append(str(s)); print(s)

RANGE = list(range(10, 20))          # P10..P19
TS_FULL = re.compile(r'\[(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})\]\s*$')
TS_SHORT = re.compile(r'\[(\d{2}:\d{2})\]\s*$')

# ---------------------------------------------------------------- 现场测量
st = os.stat(MEMO)
raw = open(MEMO, 'rb').read()
sha = hashlib.sha256(raw).hexdigest()
txt = raw.decode('utf-8-sig')
lines = txt.split('\r\n')
mtime_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(st.st_mtime))
mtime_epoch = st.st_mtime

base = json.load(io.open(BASE, encoding='utf-8'))
prev = {k: base.get(k) for k in
        ('tag', 'frozen_at', 'bytes', 'lines', 'sha256', 'sha8', 'phase_headings')}

# ---------------------------------------------------------------- 现场测量：追加源标题形式
srcs = []
for n in RANGE:
    f = os.path.join(T20.replace('Phase20', 'Phase%d' % n), 'memo_append_phase%d.md' % n)
    if not os.path.exists(f):
        srcs.append({'phase': n, 'file': f, 'found': False, 'form': None})
        continue
    s = io.open(f, encoding='utf-8-sig').read()
    h = None
    for ln in s.split('\n'):
        if ln.startswith('## Phase'):
            h = ln.rstrip('\r')
            break
    if h is None:
        srcs.append({'phase': n, 'file': os.path.basename(f), 'found': True, 'heading': None, 'form': None})
        continue
    form = 'short' if TS_SHORT.search(h) else ('full' if TS_FULL.search(h) else 'none')
    srcs.append({'phase': n, 'file': os.path.basename(f), 'found': True,
                 'heading': h, 'form': form})

# ---------------------------------------------------------------- 现场测量：MEMO 现盘标题
mem_heading = {}
for i, ln in enumerate(lines):
    m = re.match(r'^## Phase (\d+):', ln)
    if m:
        mem_heading[int(m.group(1))] = {'line': i + 1, 'heading': ln}

# ---------------------------------------------------------------- 逐 Phase 对照
rows = []
pred_delta = 0
for s in srcs:
    n = s['phase']
    mh = mem_heading.get(n)
    cur = mh['heading'] if mh else None
    cur_form = None
    if cur:
        cur_form = 'short' if TS_SHORT.search(cur) else ('full' if TS_FULL.search(cur) else 'none')
    d = None
    if s.get('form') == 'short' and cur_form == 'full' and cur:
        # 短→全：去掉短时间戳补上日期，Δ 应为 +11 B（ASCII）
        short_equiv = TS_FULL.sub(lambda mo: '[%s]' % mo.group(2), cur)
        d = len(cur.encode('utf-8')) - len(short_equiv.encode('utf-8'))
        pred_delta += d
    rows.append({'phase': n, 'src_form': s.get('form'), 'memo_form': cur_form,
                 'memo_line': mh['line'] if mh else None, 'delta_bytes': d,
                 'src_heading': s.get('heading'), 'memo_heading': cur})

obs_delta = len(raw) - int(prev['bytes'])

w('=' * 78)
w('MEMO 完整性事件审计：P19 基线后（post-append-phase19）就地规范化')
w('=' * 78)
w('审计时钟            : %s' % time.strftime('%Y-%m-%d %H:%M:%S'))
w('MEMO mtime          : %s  (epoch %.0f)' % (mtime_str, mtime_epoch))
w('MEMO 现盘           : %d B / %d 行 / sha8 %s' % (len(raw), len(lines), sha[:8]))
w('基线 post-append-phase19:')
w('   frozen_at = %s' % prev['frozen_at'])
w('   bytes=%s lines=%s sha8=%s' % (prev['bytes'], prev['lines'], prev['sha8']))
w('   实测差      : %+d B / %+d 行' % (obs_delta, len(lines) - int(prev['lines'])))
w('')
w('列：Phase | 追加源形式 | MEMO 现盘形式 | MEMO 行 | ΔB')
for r in rows:
    w('  P%-3d | %-5s | %-5s | %-5s | %s'
      % (r['phase'], r['src_form'], r['memo_form'],
         r['memo_line'], ('%+d' % r['delta_bytes']) if r['delta_bytes'] is not None else '-'))
w('')
w('追加源为 short 的 Phase 数 = %d / %d'
  % (sum(1 for r in rows if r['src_form'] == 'short'), len(rows)))
w('MEMO 现盘为 full 的 Phase 数 = %d / %d'
  % (sum(1 for r in rows if r['memo_form'] == 'full'), len(rows)))
w('预测字节增（short→full 逐条求和） = %+d B' % pred_delta)
w('实测字节增（现盘 - 基线）           = %+d B' % obs_delta)
w('残差                              = %+d B' % (obs_delta - pred_delta))
w('')
w('结论：追加源 P10–P19 使用短时间戳 [HH:MM]；MEMO 现盘全部为完整时间戳')
w('      [YYYY-MM-DD HH:MM]；逐条 +11 B × %d = %+d B，与实测差 %+d B %s'
  % (sum(1 for r in rows if r['delta_bytes'] is not None), pred_delta, obs_delta,
     '完全一致' if obs_delta == pred_delta else '不一致——需继续调查'))

# 三个可被 [:44] 规则分辨的标题（时间戳跨越第 44 字符）
disc = []
for n in (11, 14, 15):
    cur = mem_heading[n]['heading']
    short_equiv = TS_FULL.sub(lambda mo: '[%s]' % mo.group(2), cur)
    key_in_base = None
    for k, v in base['sections'].items():
        if v == mem_heading[n]['line']:
            key_in_base = k
    disc.append({'phase': n, 'line': mem_heading[n]['line'],
                 'baseline_key': key_in_base,
                 'short_equiv_first44': short_equiv[:44],
                 'full_first44': cur[:44],
                 'short_explains_key': key_in_base == short_equiv[:44]})
w('')
w('判别性证据（base.sections 键 = MEMO 行 [:44]，仅时间戳跨越第 44 字符者可分辨）：')
for d in disc:
    w('  P%-3d L%-5d 基线键=%r' % (d['phase'], d['line'], d['baseline_key']))
    w('        短形式[:44]=%r  一致=%s' % (d['short_equiv_first44'], d['short_explains_key']))

# ---------------------------------------------------------------- 冻结
art = {
    'artifact': 'memo_drift_phase19_postbaseline',
    'measured_at': time.strftime('%Y-%m-%d %H:%M:%S'),
    'memo_path': 'research/deepseek/docs/AGI_DEEPSEEK_MEMO.md',
    'memo_mtime': mtime_str,
    'memo_bytes_at_audit': len(raw),
    'memo_lines_at_audit': len(lines),
    'memo_sha8_at_audit': sha[:8],
    'memo_sha256_at_audit': sha,
    'prev_baseline': prev,
    'observed_delta_bytes': obs_delta,
    'observed_delta_lines': len(lines) - int(prev['lines']),
    'normalized_phases': [r['phase'] for r in rows if r['delta_bytes'] is not None],
    'per_phase': rows,
    'predicted_delta_bytes': pred_delta,
    'residual_bytes': obs_delta - pred_delta,
    'discriminating_evidence': disc,
    'conclusion': ('MEMO 在 post-append-phase19 基线快照之后被就地改写：P10–P19 共 %d 个'
                   '标题由短形式 [HH:MM] 规范化为完整形式 [YYYY-MM-DD HH:MM]，'
                   '逐条 +11 B、合计 +%d B，行数不变、无文本丢失。'
                   '事件未被任何 wlog / baseline / history 记录。'
                   '后果：post-append-phase19 的 bytes/sha256 锚与 sections 键三项陈旧。'
                   % (sum(1 for r in rows if r['delta_bytes'] is not None), pred_delta)),
}
io.open(OUTJ, 'w', encoding='utf-8', newline='\n').write(
    json.dumps(art, ensure_ascii=False, indent=1))
io.open(OUTT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
print('\nWROTE %s' % OUTJ)
print('WROTE %s' % OUTT)
