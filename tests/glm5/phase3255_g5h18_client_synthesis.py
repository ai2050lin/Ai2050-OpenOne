# -*- coding: utf-8 -*-
"""Phase 3255 G5-H18: registry v1.34 client-side upgrade (engineering arm) +
u* geometry trilogy synthesis (doc arm, zero GPU).

Prereg (frozen before any observation): MEMO Phase 3254 tail arms.
  (a) ENGINEERING ARM: /atlas/gaps endpoint + appendix field (gap v1.12
      auto-discovered -- 12-generation chain) + frontend GapLedgerPanel
      (gaps status row + APPX-1..6 appendix cards, APPX-6 highlighted)
      mounted in AtlasLens overview; stale v1.28 select title retired.
      Checks: endpoint coroutine check (v1.12 + appendix + sealed anchors)
      + live playwright UI check (panel renders endpoint content verbatim,
      zero console errors).
  (b) DOC ARM: u* geometry trilogy synthesis -> MEMO special section
      (3189/3197 readout structure -> 3251 lexical decoupling -> 3252
      embedding rank-1 + layer trace -> 3253 carrier continuity => two-track
      geometry + engineering implication for INFORMATION_GAP).
  (c) 5 appendix GPU items + N-line application stay deferred.

SMOKE: P3255_SMOKE=1 asserts engineering artifacts only (no MEMO write).
"""
import json
import os
import time
import hashlib
import asyncio

T0 = time.time()
NAME = 'g5h18_client_synthesis'
SMOKE = os.environ.get('P3255_SMOKE') == '1'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTDIR = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913'
                          r'\phase3255', NAME)
LOG = []

SVC = os.path.join(ROOT, 'server', 'ai_rnd_service.py')
PANEL = os.path.join(ROOT, r'frontend\src\components\app\rdc_fusion\GapLedgerPanel.jsx')
LENS = os.path.join(ROOT, r'frontend\src\components\app\rdc_fusion\AtlasLens.jsx')
CSS = os.path.join(ROOT, r'frontend\src\components\app\rdc_fusion\rdc_fusion.css')
EP_OUT = os.path.join(ROOT, r'tests\gpt5_temp\p3255_endpoint_out.txt')
UI_OUT = os.path.join(ROOT, r'tests\gpt5_temp\p3255_ui_out.txt')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')

DESIGN = {
    'phase': '3255', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3254 tail arms (a)+(b), frozen before any '
              'observation',
    'engineering_arm': '(a) /atlas/gaps + appendix field (gap v1.12 glob '
                       'auto-discovered, 12-generation chain); frontend '
                       'GapLedgerPanel (protocol-key driven: id/title/status/'
                       'note/anchor/anchor_sha8, zero experiment literals) '
                       'mounted in AtlasLens overview; checks = endpoint '
                       'coroutine assertion + live playwright UI check '
                       '(17 assertions, zero console errors)',
    'doc_arm': '(b) u* geometry trilogy synthesis -> MEMO special section: '
               '3189/3197 readout structure -> 3251 lexical decoupling -> '
               '3252 embedding rank-1 + layer trace -> 3253 carrier '
               'continuity; two-track geometry conclusion + engineering '
               'implication for the 3246 INFORMATION_GAP',
    'gpu_arm': '(c) 5 appendix GPU items + N-line application stay deferred',
    'sources': {'svc': None, 'panel': None, 'lens': None, 'css': None,
                'ep_out': None, 'ui_out': None},
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
        json.dump({'phase': 3255, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def check_engineering():
    # server patch: appendix in /atlas/gaps response
    svc = open(SVC, encoding='utf-8').read()
    assert '"appendix": data.get("appendix", [])' in svc
    assert 'v1.12' in svc.split('async def atlas_gaps')[1].split('@router')[0]
    # panel file: protocol keys + data-k
    panel = open(PANEL, encoding='utf-8').read()
    assert 'data-k="gap_ledger_panel"' in panel
    for k in ('/api/ai-rnd/atlas/gaps', 'anchor_sha8', 'status',
              'APPX-6', 'supersedes'):
        assert k in panel, k
    # mounted in AtlasLens + stale title retired
    lens = open(LENS, encoding='utf-8').read()
    assert "import GapLedgerPanel from './GapLedgerPanel.jsx'" in lens
    assert '<GapLedgerPanel />' in lens
    assert 'v1.28' not in lens
    assert '3254' in lens  # updated select title
    # css classes
    css = open(CSS, encoding='utf-8').read()
    assert '.glp-gaps' in css and '.glp-appx.glp-newest' in css
    log('engineering artifacts: svc patch + panel + mount + css OK')
    # endpoint check output
    ep = open(EP_OUT, encoding='utf-8').read()
    assert 'ENDPOINT CHECK OK' in ep and '9d0f1825' in ep
    log('endpoint check: %s' % ep.strip().splitlines()[-1])
    # live UI check output
    ui = open(UI_OUT, encoding='utf-8').read()
    assert 'UI CHECK: 17/17' in ui, ui.strip().splitlines()[-3:]
    assert 'FAIL' not in ui and 'console errors: none' in ui
    log('live UI check: 17/17 PASS, zero console errors')
    # live endpoint cross-check (in-process)
    sys_path = ROOT
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from server.ai_rnd_service import atlas_gaps
    doc = asyncio.get_event_loop().run_until_complete(atlas_gaps())
    assert doc['version'] == '1.12' and len(doc['appendix']) == 6
    assert [a['id'] for a in doc['appendix']] == ['APPX-%d' % i for i in range(1, 7)]
    return {'endpoint_version': doc['version'], 'appendix_n': len(doc['appendix']),
            'ui_check': '17/17', 'version_chain_n': len(doc['versions'])}


SYNTH_TITLE = '## u* 几何图谱综论（3189→3253 链收束；3255 (b) 臂，零 GPU 文档）'


def doc_arm():
    mb = open(MEMO, 'rb').read()
    if SYNTH_TITLE.encode('utf-8') in mb:
        log('doc arm: synthesis section already present (idempotent)')
        return 'already_present'
    lines = []
    A_ = lines.append
    A_('')
    A_(SYNTH_TITLE)
    A_('')
    A_('**链路**：3189 封存（readout OOV 残差场 top1=0.3065/0.2936/0.3008；类均值方向 v_c cos 0.72–0.85）→ 3190/3197 方向稳健性与**符号定谳**（anti-class-mean 对比轴）→ **3251 词汇解耦**（u* 与 10 类 first-token embedding 差分 |cos|≤0.144、随规模衰减 0.144→0.033→0.025、glm4 低于伪类 null；类结构由激活侧 v_c 承载）→ **3252 层轨迹**（embedding 层即秩 1 top1=1.0000×3；U 形：中层低谷 0.14–0.21 → readout 回升；近邻稳定 0.966/0.971/0.926）→ **3253 子空间机制**（σ₁(mid,out)=0.628/0.746/0.335——低谷=同一载轴衰减非替换；embedding 秩 1 方向与被输送方向解离 σ₁≤0.20）。')
    A_('')
    A_('**合成定谳：双轨道几何。**OOV 崩塌几何含两条从 embedding 输出同点出发但随即分离的轨道：')
    A_('')
    A_('- **轨道 I（embedding 秩 1 方向）**：embedding 几何的即时产物，中层前消散，**不达 readout**（mid-emb / emb-out σ₁ 全 ≤0.20）——它是 embedding 表几何的指纹，不是崩塌面板的载体；')
    A_('- **轨道 II（载轴 carrier）**：中层成形（低谷层即与 readout 深度亲和 σ₁ 0.34–0.75）、深层重聚焦、readout 处支配崩塌面板——**anti-class-mean 轴沿此轴经深度幅度调制重聚焦**。')
    A_('')
    A_('**词汇侧两轨皆不可及**（3251）：OOV 保真转正的信息缺口（3246 INFORMATION_GAP，剩余 −0.53）**无法从词表 embedding 侧补足**。协议内唯一杠杆 = 激活侧类均值 v_c 注入族；其行为效力边界由 D 线 E3 干预链（3225–3232→F20）钉定：**读出几何≠行为控制**。因此声明体制内 −0.449（XC_p5_b8，修复 74%）即信息论意义上可达的收线终点，除非引入偏移型新机制类别或越出泄漏声明界（3246 oracle 上界 +0.329）。')
    A_('')
    A_('**证据级**：全链 E2（cross_model + held-out：3 模型指纹一致 + 3246 oracle 预测）；u* 链各代均有 bitwise 锚（3253 起含跨 phase drift_3252 装置锚）。**限界**：last-token hidden states（pre-final-norm）；R=8 基；无跨层正交化。')
    A_('')
    body = (chr(10).join(lines) + chr(10)).encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    log('doc arm: synthesis section appended (%d bytes)' % len(body))
    return 'appended'


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    dsha = freeze()
    log('SMOKE=%s' % SMOKE)

    eng = check_engineering()
    doc_status = None
    if not SMOKE:
        doc_status = doc_arm()

    res = {'phase': 3255, 'name': NAME, 'mode': 'smoke' if SMOKE else 'full',
           'design_sha8': dsha,
           'engineering': eng,
           'doc_arm_status': doc_status,
           'verdict': 'SMOKE_OK' if SMOKE else 'client_synthesis_registered',
           'zero_gpu': True, 'src': 'p3255',
           'note': 'registry v1.34 client-side upgrade live-verified (17/17); '
                   'u* trilogy synthesis in MEMO; sealed files untouched '
                   '(registry v1.34 3deabf25 / gap v1.12 aa8cff21)'}
    blob = json.dumps(res, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    rsha = sha8(blob)
    res['res_sha8'] = rsha
    suffix = '_smoke' if SMOKE else ''
    json.dump(res, open(os.path.join(OUTDIR, 'result%s.json' % suffix), 'w', encoding='utf-8'),
              ensure_ascii=False, sort_keys=True, indent=1)
    log('VERDICT: %s res_sha8=%s' % (res['verdict'], rsha))
    open(os.path.join(OUTDIR, 'run_log%s.txt' % suffix), 'w', encoding='utf-8').write(
        '\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
