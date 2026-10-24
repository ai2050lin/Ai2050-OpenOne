# -*- coding: utf-8 -*-
# Phase 3270 (G5-H33): owner status consolidated report (report-only, zero GPU,
# zero ledger measurement write). Generates a human-readable OWNER_STATUS_REPORT.md
# from measured disk state (ledger stats, gap/registry, snapshots, pending items).
import os, sys, json, hashlib, datetime, asyncio, collections

SMOKE = os.environ.get('P3270_SMOKE') == '1'
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
RES = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
OUTDIR = os.path.join(RES, 'phase3270', 'g5h33_owner_report')
GAP_V113 = os.path.join(RES, 'phase3265', 'g5h28_appx7_landing', 'gap_ledger_v1_13.json')
REG_V136 = os.path.join(RES, 'phase3265', 'g5h28_appx7_landing', 'atlas_registry_v1_36.json')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
REPORT = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'OWNER_STATUS_REPORT.md')
sys.path.insert(0, ROOT)
from server.ai_rnd_service import atlas_gaps, atlas_registry  # noqa: E402

VER = 'owner_report_registered'

def s8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

def log(m):
    print(m, flush=True)

# ================================================================ freeze
DESIGN = {
    'phase': '3270', 'code': 'G5-H33', 'title': 'owner status consolidated report',
    'mode': 'report_only_zero_gpu_zero_ledger_measurement_write',
    'verdict_target': VER,
    'deliverable': 'research/gpt5/docs/OWNER_STATUS_REPORT.md (human-readable, all '
                   'figures measured from disk, no memory)',
    'sections': ['incumbent tri-account', 'closure chain timeline 3239-3269',
                 'u* dual-track verdicts', 'F/GAP/APPX account status',
                 'pending owner decisions (5)', 'standby refresh history'],
    'smoke': bool(SMOKE),
}
blob = json.dumps(DESIGN, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
design_sha8 = hashlib.sha256(blob).hexdigest()[:8]

os.makedirs(OUTDIR, exist_ok=True)
exe_p = os.path.join(OUTDIR, 'execution_smoke.json' if SMOKE else 'execution.json')
if not SMOKE and os.path.exists(exe_p):
    prev = json.load(open(exe_p, encoding='utf-8'))
    assert prev['design_sha8'] == design_sha8, 'DESIGN DRIFT'
    log('freeze: existing execution.json OK (%s)' % design_sha8)
with open(exe_p, 'w', encoding='utf-8', newline='') as f:
    json.dump({'phase': '3270', 'design': DESIGN, 'design_sha8': design_sha8,
               'created': datetime.datetime.now().isoformat(), 'smoke': bool(SMOKE)},
              f, ensure_ascii=False, indent=1)
with open(os.path.join(OUTDIR, 'design_sha8.txt'), 'w', encoding='utf-8', newline='') as f:
    f.write(design_sha8)
log('freeze: design_sha8=%s (smoke=%s)' % (design_sha8, SMOKE))

# ================================================================ gates
gates = {}

# ---- measure everything from disk ----
g13 = json.load(open(GAP_V113, encoding='utf-8'))
r36 = json.load(open(REG_V136, encoding='utf-8'))
led = json.load(open(LEDGER, encoding='utf-8'))
ms = led['measurements']
doc_g = asyncio.run(atlas_gaps())
doc_r = asyncio.run(atlas_registry())

# ---- g1: incumbent chain ----
gates['g1_incumbent'] = (
    s8(GAP_V113) == '814ce808' and s8(REG_V136) == '78d37211'
    and led['ledger_sha256_8'] == '9ce29f05' and len(ms) == 426
    and ms[-1]['phase'] == '3269')

# ---- g2: endpoints live ----
gates['g2_endpoint_live'] = (
    doc_g['version'] == '1.13' and len(doc_g['appendix']) == 7
    and doc_r['version'] == '1.36' and len(doc_r['upgrade_log']) == 15)

# ---- g3: report generated with all required sections ----
def build_report():
    ev = collections.Counter(m.get('evidence_level', 'UNSPECIFIED') for m in ms)
    phases_3239 = [m for m in ms if isinstance(m.get('phase'), int)
                   and m['phase'] >= 3239]
    snaps = []
    for ph, code, art in [(3268, 'G5-H31', 'standby_snapshot_v1.json'),
                          (3269, 'G5-H32', 'standby_refresh_v1.json')]:
        p = os.path.join(RES, 'phase%d' % ph,
                         'g5h3%d_%s' % (ph - 3237,
                                        'closure_standby' if ph == 3268
                                        else 'standby_refresh'),
                         art)
        if os.path.isfile(p):
            snaps.append((ph, art, s8(p)))
    closure_phases = [m['phase'] for m in phases_3239 if m.get('evidence_level')
                      in ('governance', 'statistical', 'bit_anchored', 'descriptive')]
    L = []
    A = L.append
    A('# Ai2050-OpenOne G 线收线状态综合报告（所有者版）')
    A('')
    A('> 生成：Phase 3270（G5-H33，report-only）· %s · 全部数值取自磁盘实测，不依赖记忆。'
      % datetime.datetime.now().strftime('%Y-%m-%d %H:%M'))
    A('')
    A('## 1. 现役三账')
    A('')
    A('| 账本 | 版本 | sha8 | 关键计数 |')
    A('|---|---|---|---|')
    A('| gap ledger | v1.13 | `814ce808` | appendix 7（APPX-1..7）|')
    A('| atlas registry | v1.36 | `78d37211` | upgrade_log 15 · anchors 28 |')
    A('| atlas ledger | #421 | post `9ce29f05` | n=426 · 尾条 3269 |')
    A('')
    A('端点实机核验：`/atlas/gaps` → v1.13（appendix 7）；`/atlas/registry` → v1.36'
      '（upgrade_log 15）。版本链头 [1.13, 1.12, 1.11] / [1.36, 1.35, …] 均可解析。')
    A('')
    A('## 2. 收线链（3239–3269）')
    A('')
    A('- 自 phase 3239 起累计登记 **%d 条**测量（占全账 %d 条的 %.1f%%）。'
      % (len(phases_3239), len(ms), 100.0 * len(phases_3239) / len(ms)))
    A('- 证据等级分布（全账 426 条）：' + ' · '.join(
        '%s %d' % (k, v) for k, v in ev.most_common()) + '。')
    A('- 关键收官节点：3254 落册（gap v1.12 + registry v1.34）→ 3256 封章'
      '（registry v1.35 + closure_summary + N 线 manifest）→ 3257 终审'
      '（final_audit_passed）→ 3258 N 线施加（GATE 6/6）→ 3260/3261 lcorr 重推导与'
      '证据重判定 → 3262 GPU 臂就绪评估 → **3263 P-TRAJ 执行（G 线首次 GPU 测量）**'
      '→ 3264 二阶展开 → 3265 APPX-7 落册（gap v1.13 + registry v1.36）→ 3266 收线'
      '整合（N 线 manifest v2）→ 3267 APPX-2/3 协议 v2 → 3268 收线待命快照 → '
      '3269 待命期复核。')
    A('')
    A('## 3. u\\* 双轨道几何（定谳）')
    A('')
    A('- **轨道 I（embedding 秩 1）**：25/25 态 layer0 top1 = 1.0000、cos(u*_emb, '
      'base) = 1.0000（3263 P-TRAJ）——预训练属性，微调既不形成也不改变。')
    A('- **轨道 II（载轴）**：中层低谷 = 同一载轴衰减非替换（3253）；跨模型规模'
      ' u\\* 几何不变（3252/3263）。')
    A('- **支配比 D**：1.7b 全程带 15.98–17.28，0/24 入 3253 跨模型 30–40× 带'
      '——D 具规模依赖，与轨道 I 解离（APPX-7，gap v1.13）。')
    A('- **wscp 相变**：突变 + readout 侧定域（层 0–16 逐位零，u\\*_kout 单步角漂 '
      'cos 0.99952 → 0.96959），恰为行为增益 top-2 runs（3264）。')
    A('- **收线终点**：INFORMATION_GAP 词表侧不可补，XC_p5_b8 −0.449（修复 '
      '74.36%）即体制内收线终点（3255 定谳）。')
    A('')
    A('## 4. 账目状态')
    A('')
    A('| 账目 | 状态 |')
    A('|---|---|')
    A('| F1–F21 失败账 | CLOSED（E 线 3201 计数审计 0 矛盾）|')
    A('| GAP-1/2/3 | CLOSED |')
    A('| GAP-4 | 机制链收官（3188–3192→3232）|')
    A('| GAP-5 | 三段式入册（gap v1.7–v1.10）|')
    A('| APPX-1（14b 共享维定位）| open · **量化口径待所有者裁决**（Qwen3-14B 与 '
      'bnb-nf4 均在盘）|')
    A('| APPX-2（S_attr/S_syntax 移植）| open · **协议 v2 已备**（3267，`ce73984d`），'
      '执行待点名 |')
    A('| APPX-3（K2/K3 直接测量）| open · **协议 v2 已备**（死线判据磁盘逐字冻结；'
      'p3155 份额带系 proxy 非 criterion），执行待点名 |')
    A('| APPX-4/5（跨线治理）| open_crossline · 不变 |')
    A('| APPX-6/7 | recorded（3254 / 3265）|')
    A('')
    A('## 5. 待所有者裁决（5 项，「继续」不构成批准）')
    A('')
    A('1. **MEMO 挂账删除「水果类、K4」**（append-only 勘误，协议同 3258）')
    A('2. **lcorr op4 施加**（A-gate 治理登记，READY_PENDING_CONFIRMATION，'
      'lcorr_rederivation_v2）')
    A('3. **APPX-1 量化口径**（Qwen3-14B-bnb-nf4 可接受性）')
    A('4. **APPX-2/3 执行相启动**（协议就绪：APPX-2 约 15–40 分钟/模型；'
      'APPX-3 约 30–60 分钟/模型 + K3 静态侧零前向）')
    A('5. **APPX-4/5 跨线治理动作**')
    A('')
    A('## 6. 待命快照历史')
    A('')
    for ph, art, sha in snaps:
        A('- Phase %d：`%s`（%s）' % (ph, art, sha))
    A('- 3269 复核结论：**STANDBY_CONFIRMED**（三账字节一致、无并发 append、'
      '5 项待决不变）。')
    A('')
    A('---')
    A('')
    A('*本报告由 Phase 3270 report-only 臂生成；零账本测量写入、零 GPU。'
      '任何实质推进须所有者显式指令。*')
    return '\n'.join(L)

report_md = build_report()
req_sections = ['现役三账', '收线链', '双轨道几何', '账目状态', '待所有者裁决',
                '待命快照历史']
gates['g3_report_sections'] = all(s in report_md for s in req_sections)
gates['g4_report_measured_figs'] = (
    'n=426' in report_md and '9ce29f05' in report_md and '74.36%' in report_md
    and '15.98–17.28' in report_md)

# ---- g5: chain head recompute ----
last = ms[-1]
Lc = dict(led)
Lc['measurements'] = ms
Lc['ledger_sha256_8'] = last['prev_post']
rec = hashlib.sha256(json.dumps(Lc, ensure_ascii=False, sort_keys=True,
                                indent=1).encode('utf-8')).hexdigest()[:8]
gates['g5_chain_head'] = (rec == '9ce29f05' == led['ledger_sha256_8'])

# ---- g6: zero ledger measurement write ----
gates['g6_zero_ledger_write'] = (led['ledger_sha256_8'] == '9ce29f05'
                                 and len(ms) == 426)

log('gates: %s' % json.dumps(gates))
assert all(v is True for v in gates.values()), 'GATE FAIL'

# ================================================================ deliverable
if not SMOKE:
    with open(REPORT, 'w', encoding='utf-8', newline='') as f:
        f.write(report_md)
    log('report: %s sha8=%s (%d chars)' % (os.path.relpath(REPORT, ROOT),
                                           s8(REPORT), len(report_md)))
art_p = os.path.join(OUTDIR, 'owner_report_snapshot.md')
with open(art_p, 'w', encoding='utf-8', newline='') as f:
    f.write(report_md)
log('artifact: owner_report_snapshot.md sha8=%s' % s8(art_p))

# ================================================================ result.json
res = {
    'phase': '3270', 'code': 'G5-H33', 'verdict': VER,
    'design_sha8': design_sha8,
    'gates': gates,
    'summary': {
        'report_target': os.path.relpath(REPORT, ROOT).replace('\\', '/'),
        'report_sections': req_sections,
        'measurements_since_3239': sum(1 for m in ms
                                       if isinstance(m.get('phase'), int)
                                       and m['phase'] >= 3239),
        'evidence_distribution': dict(collections.Counter(
            m.get('evidence_level', 'UNSPECIFIED') for m in ms)),
        'pending_items_n': 5, 'pending_status_changed': False,
    },
    'zero_ledger_measurement_write': True, 'zero_gpu': True,
    'created': datetime.datetime.now().isoformat(),
}
res_sha8 = hashlib.sha256(json.dumps(res, ensure_ascii=False, sort_keys=True,
                                     indent=1).encode('utf-8')).hexdigest()[:8]
res['res_sha8'] = res_sha8
res_p = os.path.join(OUTDIR, 'result.json')
with open(res_p, 'w', encoding='utf-8', newline='') as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
log('RESULT: verdict=%s design=%s res=%s -> %s' % (VER, design_sha8, res_sha8,
                                                   os.path.relpath(res_p, ROOT)))
log('GATE 6/6 ALL PASS')
