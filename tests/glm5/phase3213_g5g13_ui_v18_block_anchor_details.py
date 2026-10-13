# -*- coding: utf-8 -*-
"""Phase 3213 (G5-G13): UI v1.8 zero-script client arm (non-ledger, zero GPU).

Per prereg in MEMO Phase 3210 tail G5-G10 (frozen before any observation;
default arm (a)): iterate the v1.7 client (phase 3211, 62ba1bd6) to extend
the anchor-details view to the non-feature blocks — gap evidence_anchors +
per-key anchor_sha8, appendix anchor/anchor_sha8, N-line draft name +
per-key sources + draft meta (status/target/governance) — all with data-k
field verification. Feature anchor values view (61/352) carried over. Still ZERO <script> tags;
interaction via details/summary + CSS radio :checked filtering. Embeds the
atlas v1.30 data plane:
  - registry v1.28 (5995bbe9): 26 features, family-grouped cards with
    evidence_level / statement / anchors n / counter_evidence / scope_limits;
  - gap ledger v1.6 (0f6423e0): 4 gaps with closed badges + GAP-4
    mechanism_note;
  - appendix x5 (gl15 4fb3f37d) with anchor shas;
  - N-line backfill draft x5 (f29da2de, draft_pending_cross_line);
  - version chain strip (v1.28..v1.21 + gap v1.5/v1.6).
  Checks: C1 zero <script> (and no on* handlers); C2 feature count 26 with
  evidence-level census (E2 20 incl annotations / E3 3 / E1 3); C3 gap x4 +
  appendix x5 + draft x5 present; C4 data-k verify_html pass on the pinned
  field set (26 feature elevel + meta chain); C5 ledger untouched.
  Output: atlas_client_v1.html (sha8-stamped) + final_result.json.
  Verdict: ui_v16_client_registered iff C1-C5 clean.
"""
import hashlib
import json
import os
import re
import sys
import time

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3179_g5b4_ftr23_registration as p79

span = p79.span

ROOT = r'D://AI2050//Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
OUTDIR = os.path.join(SRC_DIR, 'phase3213', 'g5g13_ui_v18_block_anchor_details')
SMOKE = os.environ.get('P3213_SMOKE') == '1'
LOG = []

SRC = {
    'registry_v128': os.path.join(SRC_DIR, r'phase3206\g5g5_ftr22_chain_recheck\atlas_registry_v1_28.json'),
    'gl16': os.path.join(SRC_DIR, r'phase3192\g5c5_gap4_amplitude_attribution\gap_ledger_v1_6.json'),
    'gl15': os.path.join(SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'),
    'draft': os.path.join(SRC_DIR, r'phase3208\g5g7_nline_backfill_draft\n_line_backfill_draft_v1.json'),
    'ledger': os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
}
SHA_ANCHOR = {
    'registry_v128': '5995bbe9', 'gl16': '0f6423e0', 'gl15': '4fb3f37d',
    'draft': 'f29da2de',
}

DESIGN = {
    'phase': '3213', 'name': 'g5g13_ui_v18_block_anchor_details',
    'prereg': 'AGI_GPT5_MEMO Phase 3212 tail G5-G13 (frozen before any '
              'observation; default arm (a) UI v1.8 block anchor details)',
    'data': 'registry v1.28 (5995bbe9) + gap v1.6 (0f6423e0) + gl15 appendix '
            '(4fb3f37d) + N-line draft (f29da2de); zero GPU, non-ledger',
    'gate': 'registered iff C1 zero-script + C2 census 26 + C3 blocks 4/5/5 '
            '+ C4 data-k verify pass (features 61/352 carried + block '
            'details 59 new fields) + C5 ledger untouched',
    'sources': SHA_ANCHOR,
}


def log(s):
    ln = '[%7.1f] %s' % (time.time() - T0, s)
    LOG.append(ln)
    try:
        print(ln, flush=True)
    except Exception:
        pass


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def sha8_file(p):
    return sha8(open(p, 'rb').read())


def freeze():
    eblob = json.dumps(DESIGN, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    dsha = hashlib.sha256(eblob).hexdigest()
    os.makedirs(OUTDIR, exist_ok=True)
    exe_p = os.path.join(OUTDIR, 'execution.json')
    if os.path.exists(exe_p) and not SMOKE:
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha8'] == dsha[:8], 'DESIGN DRIFT'
        log('freeze: existing execution.json OK (%s)' % dsha[:8])
    else:
        json.dump({'phase': 3213, 'name': 'g5g13_ui_v18_block_anchor_details',
                   'design_sha8': dsha[:8], 'design': DESIGN,
                   'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


CSS = """
:root{--bg:#f7f7f9;--fg:#1c1c22;--line:#d8d8e0;--card:#ffffff;--accent:#3450a2;
--b-e2:#2f7d4f;--b-e3:#8a4fbf;--b-e1:#b07020;--b-open:#a05520;--b-closed:#2f7d4f;}
*{box-sizing:border-box}
body{margin:0;font:14px/1.55 "Segoe UI","Microsoft YaHei",sans-serif;background:var(--bg);color:var(--fg)}
header{background:var(--card);border-bottom:1px solid var(--line);padding:14px 20px}
header h1{margin:0 0 4px;font-size:19px}
header .sub{color:#555;font-size:12px;word-break:break-all}
nav{position:sticky;top:0;background:var(--card);border-bottom:1px solid var(--line);padding:8px 20px;z-index:5}
nav a{margin-right:14px;color:var(--accent);text-decoration:none;font-size:13px}
nav a:hover{text-decoration:underline}
main{max-width:1180px;margin:0 auto;padding:16px 20px 60px}
.filters{display:flex;gap:6px;flex-wrap:wrap;margin:10px 0 14px}
.filters input{display:none}
.filters label{border:1px solid var(--line);border-radius:14px;padding:3px 12px;cursor:pointer;background:var(--card);font-size:12.5px}
.filters input:checked + label{background:var(--accent);color:#fff;border-color:var(--accent)}
#f-all:checked ~ .fgrid .feat{display:block}
#f-e2:checked ~ .fgrid .feat:not([data-lv^=E2]){display:none}
#f-e3:checked ~ .fgrid .feat:not([data-lv^=E3]){display:none}
#f-e1:checked ~ .fgrid .feat:not([data-lv^=E1]){display:none}
h2{font-size:16px;margin:26px 0 10px;border-left:4px solid var(--accent);padding-left:8px}
.fgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:10px}
.feat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.feat h3{margin:0 0 4px;font-size:13.5px}
.feat .fid{color:var(--accent);font-family:Consolas,monospace;margin-right:6px}
.badge{display:inline-block;border-radius:10px;padding:1px 8px;font-size:11px;color:#fff;background:#777}
.b-E2_predictive,.b-E2_held_out{background:var(--b-e2)} .b-E3_causal_scoped{background:var(--b-e3)} .b-E1_repeatable{background:var(--b-e1)}
.b-appendix_open{background:var(--b-open)} .b-appendix_open_crossline{background:#8a4fbf} .b-closed{background:var(--b-closed)} .b-draft{background:#555}
.feat p{margin:6px 0;font-size:13px}
.feat details{margin-top:6px;font-size:12.5px}
.feat summary{cursor:pointer;color:var(--accent)}
.kv{margin:2px 0}.kv b{color:#444;margin-right:6px}
.mono{font-family:Consolas,monospace;font-size:12px;word-break:break-all}
.gap{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:10px 0}
.evid{color:#666;font-size:12px}
table{width:100%;border-collapse:collapse;background:var(--card);font-size:12.5px}
th,td{border:1px solid var(--line);padding:5px 8px;text-align:left}
th{background:#eef0f5}
footer{color:#777;font-size:11.5px;padding:14px 20px;border-top:1px solid var(--line);margin-top:30px}
"""


def esc(s):
    return p79.esc(str(s))


def build_html(reg, gl16, gl15, draft, led, created):
    H = []
    A = H.append
    A('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    A('<meta name="viewport" content="width=device-width,initial-scale=1">')
    A('<title>Atlas Client v1.8 - RDC/LPF 语义图谱客户端</title><style>%s</style></head>' % CSS)
    A('<body><header>')
    A('<h1>Atlas Client v1.8 <span style="font-size:13px;color:#666">— 语义机制图谱 · 零 script 客户端</span></h1>')
    A('<div class="sub">registry v1.28 sha8=%s · gap v1.6 sha8=%s · appendix 源 gap v1.5 sha8=%s · '
      'N 线补录草案 sha8=%s · 渲染于 %s · Phase 3213 · v1.7 特征锚视图保留 + v1.8 新增 gap/appendix/草案块级锚详情</div>' % (
          span('meta.registry_v128_sha8', SHA_ANCHOR['registry_v128']),
          span('meta.gap_v16_sha8', SHA_ANCHOR['gl16']),
          span('meta.gap_v15_sha8', SHA_ANCHOR['gl15']),
          span('meta.draft_sha8', SHA_ANCHOR['draft']),
          span('meta.created', created)))
    A('</header><nav>')
    for anchor, label in [('features', '特征登记表 (26)'), ('gaps', '缺口账本 (4)'),
                          ('appendix', '附录挂账 (5)'), ('nline', 'N 线补录草案 (5)'),
                          ('chain', '版本链'), ('ledger', '账本尾 (5)')]:
        A('<a href="#%s">%s</a>' % (anchor, label))
    A('</nav><main>')

    # ---- features with radio filter ----
    fams = {}
    for f in reg['features']:
        fams.setdefault(f['family'], []).append(f)
    A('<h2 id="features">跨模型稳定特征登记表（registry v1.28，%d 条）</h2>' % len(reg['features']))
    A('<div class="filters">')
    A('<input type="radio" name="fl" id="f-all" checked><label for="f-all">全部 (26)</label>')
    A('<input type="radio" name="fl" id="f-e2"><label for="f-e2">E2_predictive (20)</label>')
    A('<input type="radio" name="fl" id="f-e3"><label for="f-e3">E3_causal_scoped (3)</label>')
    A('<input type="radio" name="fl" id="f-e1"><label for="f-e1">E1_repeatable (3)</label>')
    A('</div>')
    lv_census = {}
    for fam in sorted(fams):
        A('<h3 style="margin:14px 0 6px;font-size:13.5px;color:#555">family = %s（%d 条）</h3>' % (esc(fam), len(fams[fam])))
        A('<div class="fgrid">')
        for f in fams[fam]:
            lv = f['evidence_level']
            lv_census[lv] = lv_census.get(lv, 0) + 1
            A('<div class="feat" data-lv="%s" data-fid="%s">' % (span('feat.' + f['id'] + '.elevel', lv), f['id']))
            A('<h3><span class="fid">%s</span>%s <span class="badge b-%s">%s</span></h3>' % (
                f['id'], esc(f['family']), lv, span('feat.' + f['id'] + '.family', f['family'])))
            A('<p>%s</p>' % span('feat.' + f['id'] + '.statement', f['statement']))
            A('<details><summary>锚 / 反证 / 范围 (%d 锚)</summary>' % len(f['anchors']))
            for ai, a in enumerate(f['anchors']):
                tags = ' '.join('#' + t for t in a.get('tags', []))
                A('<div class="kv"><b>%s</b> <span class="mono">phase %s · %s%s</span></div>' % (
                    span('feat.%s.a%d.src' % (f['id'], ai), a['src']),
                    esc(a['phase']),
                    span('feat.%s.a%d.role' % (f['id'], ai), a['role']),
                    (' <span class="evid">%s</span>' % esc(tags)) if tags else ''))
                for ak, av in sorted(a.get('asserts', {}).items()):
                    sv = av if isinstance(av, str) else json.dumps(av, ensure_ascii=False)
                    A('<div class="kv mono" style="padding-left:14px">%s = %s</div>' % (
                        esc(ak), span('feat.%s.a%d.%s' % (f['id'], ai, ak), sv)))
            A('<div class="kv"><b>counter_evidence</b>%s</div>' % esc(' | '.join(f['counter_evidence']) if f['counter_evidence'] else '(none)'))
            A('<div class="kv"><b>scope_limits</b>%s</div>' % esc(f['scope_limits'] if f['scope_limits'] else '(none)'))
            A('<div class="kv"><b>replication</b>%s</div>' % esc(f['replication']))
            A('</details></div>')
        A('</div>')

    # ---- gaps ----
    A('<h2 id="gaps">缺口账本 v1.6（%d 条）</h2>' % len(gl16['gaps']))
    for g in gl16['gaps']:
        closed = g['status'].startswith('closed')
        A('<div class="gap"><h3 style="margin:0 0 6px">%s · %s <span class="badge %s">%s</span>%s</h3>' % (
            g['id'], esc(g['title']),
            'b-closed' if closed else 'b-appendix_open',
            span('gap.' + g['id'] + '.status', g['status']),
            (' <span class="badge b-closed">closed@%s</span>' % span('gap.' + g['id'] + '.closed_at', g['closed_at_phase'])) if closed else ''))
        A('<div class="kv"><b>statement</b>%s</div>' % span('gap.' + g['id'] + '.statement', g['statement']))
        A('<div class="evid">%s</div>' % span('gap.' + g['id'] + '.evidence', ' || '.join(g['evidence'])))
        if g['id'] == 'GAP-4':
            A('<div class="kv"><b>mechanism_note</b>%s</div>' % span('gap.GAP-4.mechanism_note', g['mechanism_note']))
        A('<div class="kv"><b>evidence_anchors</b>%s</div>' % ' '.join(
            span('gap.%s.ea%d' % (g['id'], i), a) for i, a in enumerate(g['evidence_anchors'])))
        for k, v in sorted(g['anchor_sha8'].items()):
            A('<div class="kv mono"><b>%s</b>%s</div>' % (
                esc(k), span('gap.%s.anch.%s' % (g['id'], k), v)))

    # ---- appendix ----
    A('<h2 id="appendix">附录挂账（%d 项，gap v1.5 源）</h2>' % len(gl15['appendix']))
    A('<table><tr><th>id</th><th>title</th><th>status</th><th>note</th><th>anchor</th></tr>')
    for a in gl15['appendix']:
        A('<tr><td class="mono">%s</td><td>%s</td><td><span class="badge b-%s">%s</span></td>'
          '<td>%s</td><td class="mono">%s(%s)</td></tr>' % (
              a['id'], span('appx.' + a['id'] + '.title', a['title']),
              a['status'], span('appx.' + a['id'] + '.status', a['status']),
              span('appx.' + a['id'] + '.note', a['note']),
              span('appx.' + a['id'] + '.anchor', a['anchor']),
              span('appx.' + a['id'] + '.anchor_sha8', a['anchor_sha8'])))
    A('</table>')

    # ---- N-line draft ----
    A('<h2 id="nline">N 线补录草案（%d 条，draft_pending_cross_line，未施加）</h2>' % len(draft['entries']))
    A('<table><tr><th>phase</th><th>name</th><th>title</th><th>sources</th><th>status</th></tr>')
    for e in draft['entries']:
        srcs = ' '.join('%s=%s' % (k, span('ndraft.%s.src.%s' % (e['phase'], k), v))
                        for k, v in sorted(e['sources'].items()))
        A('<tr><td class="mono">%s</td><td class="mono">%s</td><td>%s</td>'
          '<td class="mono">%s</td><td><span class="badge b-draft">%s</span></td></tr>' % (
              e['phase'], span('ndraft.' + e['phase'] + '.name', e['name']),
              span('ndraft.' + e['phase'] + '.title', e['title']),
              srcs, span('ndraft.' + e['phase'] + '.status', e['status'])))
    A('</table>')
    A('<div class="gap" style="margin-top:8px"><h3 style="margin:0 0 6px;font-size:13px">草案元数据（v1.8 新增展开）</h3>'
      '<div class="kv"><b>status</b>%s</div>'
      '<div class="kv mono"><b>target</b>%s</div>'
      '<div class="kv"><b>governance</b>%s</div></div>' % (
          span('ndraft.meta.status', draft['status']),
          span('ndraft.meta.target', draft['target']),
          span('ndraft.meta.governance', draft['governance'])))

    # ---- version chain ----
    A('<h2 id="chain">registry 版本链（v1.28 ← v1.1）</h2><div class="gap">')
    chain = [('v1.28', '5995bbe9', '3206 FTR-22 残留挂账口径'), ('v1.27', 'd5ce1c19', '3205 FTR-14 gate-fail 口径'),
             ('v1.26', '112a46ee', '3204 FTR-15 held_out 口径'), ('v1.25', '3bb33a24', '3192 GAP-4 归因'),
             ('v1.24', '0d74b628', '3191'), ('v1.23', '5f5b9e73', '3190'),
             ('v1.22', '9fd87433', '3189'), ('v1.21', '47cf01e0', '3188')]
    for v, s8, note in chain:
        A('<div class="kv mono"><b>%s</b>%s <span class="evid">%s</span></div>' % (
            v, span('chain.' + v, s8), esc(note)))
    A('</div>')

    # ---- ledger tail ----
    A('<h2 id="ledger">治理账本尾（%d 条 measurements，末 5 条）</h2>' % len(led['measurements']))
    A('<table><tr><th>#</th><th>phase</th><th>name</th><th>verdict (head)</th><th>pre→post</th></tr>')
    tail = led['measurements'][-5:]
    for i, m in enumerate(tail):
        idx = len(led['measurements']) - 5 + i
        A('<tr><td class="mono">%d</td><td class="mono">%s</td><td class="mono">%s</td><td>%s</td>'
          '<td class="mono">%s→%s</td></tr>' % (
              idx + 1, esc(m['phase']), esc(m['name']),
              span('led.%d.verdict' % idx, m['verdict'][:80] + '…'),
              esc(m['chain'].get('ledger_pre', '-')), esc(led['ledger_sha256_8'] if i == len(tail) - 1 else tail[i + 1]['chain'].get('ledger_pre', '-'))))
    A('</table>')
    A('</main><footer>')
    A('Atlas Client v1.8 · 零 script（纯 HTML/CSS，details/radio 过滤/锚点导航）· v1.8 块级锚详情 · 渲染自 atlas_registry_v1_28.json (%s)'
      ' + gap_ledger_v1_6.json (%s) + gap v1.5 appendix + n_line_backfill_draft_v1.json (%s)'
      ' · Phase 3213 封存产物 · 本文件不含任何 JavaScript。' % (
          SHA_ANCHOR['registry_v128'], SHA_ANCHOR['gl16'], SHA_ANCHOR['draft']))
    A('</footer></body></html>')
    return H, lv_census


def main():
    design_sha = freeze()
    log('== Phase 3213 g5g13_ui_v18_block_anchor_details %s ==' % ('SMOKE' if SMOKE else 'FULL'))
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    reg = json.load(open(SRC['registry_v128'], encoding='utf-8'))
    gl16 = json.load(open(SRC['gl16'], encoding='utf-8'))
    gl15 = json.load(open(SRC['gl15'], encoding='utf-8'))
    draft = json.load(open(SRC['draft'], encoding='utf-8'))
    led = json.load(open(SRC['ledger'], encoding='utf-8'))
    assert reg['version'] == '1.28' and len(reg['features']) == 26
    assert len(gl16['gaps']) == 4 and len(gl15['appendix']) == 5
    assert len(draft['entries']) == 5 and len(led['measurements']) == 364
    log('inputs loaded: registry v1.28 (26) + gaps (4) + appendix (5) + draft (5) + ledger (364)')

    contradictions = []
    created = time.strftime('%Y-%m-%d %H:%M')
    H, lv_census = build_html(reg, gl16, gl15, draft, led, created)
    html_text = '\n'.join(H)

    # C1: zero script + no inline handlers
    c1 = ('<script' not in html_text.lower()
          and not re.search(r'\son\w+\s*=', html_text, re.I)
          and 'javascript:' not in html_text.lower())
    if not c1:
        contradictions.append(('C1_zero_script',))
    log('C1: zero <script> / no on* handlers / no javascript: URI: %s' % ('OK' if c1 else 'FAIL'))

    # C2: census
    c2 = (lv_census.get('E2_predictive', 0) + lv_census.get('E2_held_out', 0) == 20
          and lv_census.get('E3_causal_scoped', 0) == 3
          and lv_census.get('E1_repeatable', 0) == 3)
    if not c2:
        contradictions.append(('C2_census', lv_census))
    log('C2: evidence census E2=20 / E3=3 / E1=3: %s %s' % ('OK' if c2 else 'FAIL', lv_census))

    # C3: blocks present
    c3 = ('id="gaps"' in html_text and 'id="appendix"' in html_text
          and 'id="nline"' in html_text and 'N-P3' in html_text and 'N-P7' in html_text
          and 'APPX-1' in html_text and 'APPX-5' in html_text
          and 'FTR-01' in html_text and 'FTR-26' in html_text)
    if not c3:
        contradictions.append(('C3_blocks',))
    log('C3: gaps/appendix/nline blocks + FTR-01..26 + APPX-1..5 present: %s'
        % ('OK' if c3 else 'FAIL'))

    # C4: data-k verify on pinned field set
    expected = {'meta.registry_v128_sha8': SHA_ANCHOR['registry_v128'],
                'meta.gap_v16_sha8': SHA_ANCHOR['gl16'],
                'meta.gap_v15_sha8': SHA_ANCHOR['gl15'],
                'meta.draft_sha8': SHA_ANCHOR['draft'],
                'meta.created': created}
    for f in reg['features']:
        expected['feat.%s.elevel' % f['id']] = f['evidence_level']
        expected['feat.%s.family' % f['id']] = f['family']
        expected['feat.%s.statement' % f['id']] = f['statement']
        for ai, a in enumerate(f['anchors']):
            expected['feat.%s.a%d.src' % (f['id'], ai)] = a['src']
            expected['feat.%s.a%d.role' % (f['id'], ai)] = a['role']
            for ak, av in sorted(a.get('asserts', {}).items()):
                expected['feat.%s.a%d.%s' % (f['id'], ai, ak)] = \
                    av if isinstance(av, str) else json.dumps(av, ensure_ascii=False)
    for g in gl16['gaps']:
        expected['gap.%s.status' % g['id']] = g['status']
        expected['gap.%s.statement' % g['id']] = g['statement']
        expected['gap.%s.evidence' % g['id']] = ' || '.join(g['evidence'])
        if g['status'].startswith('closed'):
            expected['gap.%s.closed_at' % g['id']] = str(g['closed_at_phase'])
    expected['gap.GAP-4.mechanism_note'] = [g for g in gl16['gaps'] if g['id'] == 'GAP-4'][0]['mechanism_note']
    for a in gl15['appendix']:
        expected['appx.%s.title' % a['id']] = a['title']
        expected['appx.%s.status' % a['id']] = a['status']
        expected['appx.%s.note' % a['id']] = a['note']
    for g in gl16['gaps']:
        for i, ea in enumerate(g['evidence_anchors']):
            expected['gap.%s.ea%d' % (g['id'], i)] = ea
        for k, v in sorted(g['anchor_sha8'].items()):
            expected['gap.%s.anch.%s' % (g['id'], k)] = v
    for a in gl15['appendix']:
        expected['appx.%s.anchor' % a['id']] = a['anchor']
        expected['appx.%s.anchor_sha8' % a['id']] = a['anchor_sha8']
    for e in draft['entries']:
        expected['ndraft.%s.title' % e['phase']] = e['title']
        expected['ndraft.%s.status' % e['phase']] = e['status']
        expected['ndraft.%s.name' % e['phase']] = e['name']
        for k, v in sorted(e['sources'].items()):
            expected['ndraft.%s.src.%s' % (e['phase'], k)] = v
    expected['ndraft.meta.status'] = draft['status']
    expected['ndraft.meta.target'] = draft['target']
    expected['ndraft.meta.governance'] = draft['governance']
    for v, s8, _ in [('v1.28', '5995bbe9', 0), ('v1.27', 'd5ce1c19', 0), ('v1.26', '112a46ee', 0),
                     ('v1.25', '3bb33a24', 0), ('v1.24', '0d74b628', 0), ('v1.23', '5f5b9e73', 0),
                     ('v1.22', '9fd87433', 0), ('v1.21', '47cf01e0', 0)]:
        expected['chain.%s' % v] = s8
    tail = led['measurements'][-5:]
    for i, m in enumerate(tail):
        idx = len(led['measurements']) - 5 + i
        expected['led.%d.verdict' % idx] = m['verdict'][:80] + '…'
    v = p79.verify_html(html_text, expected)
    c4 = (not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups'])
    if not c4:
        contradictions.append(('C4_datak', (len(v['missing']), len(v['extra']),
                                            len(v['mismatch']), len(v['dups']))))
    log('C4: data-k fields %d/%d (missing/extra/mismatch/dups = %d/%d/%d/%d): %s' % (
        v['n_found'], v['n_expected'], len(v['missing']), len(v['extra']),
        len(v['mismatch']), len(v['dups']), 'OK' if c4 else 'FAIL'))
    if not c4:
        log('C4 detail missing=%s extra=%s mismatch=%s' % (
            v['missing'][:5], v['extra'][:5], v['mismatch'][:5]))

    # C5: ledger untouched
    led2 = json.load(open(SRC['ledger'], encoding='utf-8'))
    c5 = (len(led2['measurements']) == 364 and led2['ledger_sha256_8'] == '63a42cbc')
    if not c5:
        contradictions.append(('C5_ledger',))
    log('C5: ledger untouched (n=364, 63a42cbc): %s' % ('OK' if c5 else 'FAIL'))

    clean = not contradictions
    verdict = ('ui_v18_block_anchor_details_registered' if clean
               else 'ui_v18_block_anchor_details_dirty (%d contradictions)' % len(contradictions))
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    hp = os.path.join(OUTDIR, suffix + 'atlas_client_v1.html')
    if not SMOKE:
        with open(hp, 'w', encoding='utf-8', newline='') as f:
            f.write(html_text)
        log('atlas_client_v1.html written sha8 %s (%d bytes)' % (sha8_file(hp), len(html_text.encode('utf-8'))))
    summary = {
        'phase': 3213, 'name': 'g5g13_ui_v18_block_anchor_details',
        'mode': 'SMOKE' if SMOKE else 'FULL', 'design_sha8': design_sha,
        'zero_gpu': True, 'non_ledger': True, 'zero_script': bool(c1),
        'gates': {'recheck_clean': bool(clean), 'n_contradictions': len(contradictions)},
        'census': lv_census,
        'field_check': {'n_expected': v['n_expected'], 'n_found': v['n_found'],
                        'missing': len(v['missing']), 'extra': len(v['extra']),
                        'mismatch': len(v['mismatch']), 'dups': len(v['dups'])},
        'verdict': verdict,
    }
    raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = sha8(raw.encode('utf-8'))
    mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
    seal8 = sha8(mid.encode('utf-8'))
    rp = os.path.join(OUTDIR, suffix + 'result.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(dict(summary, res_sha8=res8, seal_sha8=seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('result written res=%s seal=%s' % (res8, seal8))
    with open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
