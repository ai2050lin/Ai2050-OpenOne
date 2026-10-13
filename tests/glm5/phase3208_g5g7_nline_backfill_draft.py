# -*- coding: utf-8 -*-
"""Phase 3208 (G5-G7): N-line P3-P7 ledger backfill draft arm (draft-only,
zero GPU).

Per prereg in MEMO Phase 3207 tail G5-G7 (frozen before any observation;
default arm (b) confirmed viable by pre-freeze probe: deepseek-line Phase3-7
sealed artifacts all on disk in tests/deepseek_temp/): draft 5 backfill
entries for the shared atlas_ledger (APPX-4 挂账联动), 只补不改 — NOT applied
to research/gpt5/atlas/atlas_ledger.json (cross-line confirmation pending,
same governance as APPX-4/5 pending_cross_line).
  RECHECK (pinned, probed BEFORE freeze):
  B1 - 15 sealed source files sha-asserted bitwise (Phase3 e2 report/memo/
  wlog; Phase4 N1 seal bd05ba88 + memo + n1v2 report; Phase5 N2 memo +
  n2f/n2g reports; Phase6 N2h1 seal 4d32b1aa + memo + n2h1c report; Phase7
  N3 seal 66afc202 + memo + n3 report).
  B2 - content anchors: P4 seal verbatim fields (main_axis_layer_scan /
  sense_split / ontology_margin); P5 memo verbatim (没有任何单头 / L31–L35);
  P6 report verbatim (B_cat=+10.192 / B/A=0.958 / B_cat=+0.071) + seal
  title 置换向量消融; P7 report verbatim (F2 color B/A=0.925 (lstar=L27)) +
  seal fields (F1_is_a / F2_color / B_cat_loo); P3 report verbatim (E2 条件
  化比较探针).
  B3 - draft file n_line_backfill_draft_v1.json written with 5 entries
  (N-P3..N-P7), status draft_pending_cross_line, source sha chain; ledger
  untouched assertion: atlas_ledger n == 359 and internal sha efe4a6de
  before == after.
  Verdict: n_line_backfill_draft_registered iff B1-B3 clean; else dirty
  (nothing applied either way).
  SMOKE (env P3208_SMOKE=1): checks only, smoke_ prefixed outputs.
"""
import hashlib
import json
import os
import sys
import time

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = r'D://AI2050//Ai2050-OpenOne'
NB = os.path.join(ROOT, 'tests', 'deepseek_temp')
OUTDIR = os.path.join(ROOT, 'tests', 'glm5', 'result',
                      'rdc_query_construction_20260913', 'phase3208',
                      'g5g7_nline_backfill_draft')
SMOKE = os.environ.get('P3208_SMOKE') == '1'
LOG = []
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')

GATE = {'sources_bitwise': True, 'content_anchors': True, 'ledger_untouched': True}

SRC = {
    'p3_e2_report': os.path.join(NB, r'Phase3\e2_report.txt'),
    'p3_memo': os.path.join(NB, r'Phase3\memo_append_e2.md'),
    'p3_wlog': os.path.join(NB, r'Phase3\wlog_append_e2.md'),
    'p4_seal': os.path.join(NB, r'Phase4\N1_design_seal.json'),
    'p4_memo': os.path.join(NB, r'Phase4\memo_append_n1.md'),
    'p4_report': os.path.join(NB, r'Phase4\n1v2_report_qwen3-4b.txt'),
    'p5_memo': os.path.join(NB, r'Phase5\memo_append_n2.md'),
    'p5_report_f': os.path.join(NB, r'Phase5\n2f_report_qwen3-4b.txt'),
    'p5_report_g': os.path.join(NB, r'Phase5\n2g_report_qwen3-4b.txt'),
    'p6_seal': os.path.join(NB, r'Phase6\N2h1_design_seal.json'),
    'p6_memo': os.path.join(NB, r'Phase6\memo_append_n2h1.md'),
    'p6_report': os.path.join(NB, r'Phase6\n2h1c_report_qwen3-4b.txt'),
    'p7_seal': os.path.join(NB, r'Phase7\N3_design_seal.json'),
    'p7_memo': os.path.join(NB, r'Phase7\memo_append_n3.md'),
    'p7_report': os.path.join(NB, r'Phase7\n3_report_qwen3-4b.txt'),
}
SHA_ANCHOR = {
    'p3_e2_report': 'd6de235f', 'p3_memo': '1d56f814', 'p3_wlog': 'd0810382',
    'p4_seal': 'bd05ba88', 'p4_memo': 'cbe8452e', 'p4_report': 'ad05f1f2',
    'p5_memo': 'b40de352', 'p5_report_f': '248f0a46', 'p5_report_g': 'd20cac53',
    'p6_seal': '4d32b1aa', 'p6_memo': '373571ed', 'p6_report': '4fb86141',
    'p7_seal': '66afc202', 'p7_memo': '952f8833', 'p7_report': '51e19b7f',
}

CONTENT_ANCHORS = {
    'p4_seal': ['main_axis_layer_scan', 'sense_split', 'ontology_margin'],
    'p5_memo': ['没有任何单头', 'L31\u2013L35'],
    'p6_report': ['B_cat=+10.192', 'B/A=0.958', 'B_cat=+0.071'],
    'p6_seal': ['置换向量消融'],
    'p7_report': ['F2 color B/A=0.925 (lstar=L27)'],
    'p7_seal': ['F1_is_a', 'F2_color', 'B_cat_loo'],
    'p3_e2_report': ['E2 条件化比较探针'],
}

DRAFT_ENTRIES = [
    {'phase': 'N-P3', 'name': 'e2_conditioned_compare_layer_scan',
     'title': 'E2 条件化比较探针（qwen3-4b 层扫描）',
     'sources': {'e2_report': 'd6de235f', 'memo_append': '1d56f814',
                 'wlog_append': 'd0810382'}},
    {'phase': 'N-P4', 'name': 'n1_main_axis_layer_scan',
     'title': 'N1 主轴三段层扫描（embed->layer->unembed；arm_A sense_split / '
              'arm_B ontology_margin；seal bd05ba88）',
     'sources': {'design_seal': 'bd05ba88', 'memo_append': 'cbe8452e',
                 'report_qwen3_4b': 'ad05f1f2'}},
    {'phase': 'N-P5', 'name': 'n2_reconstruction_source',
     'title': 'N2 is-a 读数重建源定位——单头/单层均不成立，分布式重构 + '
              '晚期块 L31–L35 唯一跨上下文必要件',
     'sources': {'memo_append': 'b40de352', 'report_n2f': '248f0a46',
                 'report_n2g': 'd20cac53'}},
    {'phase': 'N-P6', 'name': 'n2h1_permutation_ablation',
     'title': 'N2h1 置换向量消融——承诺由类别分量承载（leave-inst-out '
              'B_cat=+10.192 / A_full=+10.640 / B/A=0.958；seal 4d32b1aa）',
     'sources': {'design_seal': '4d32b1aa', 'memo_append': '373571ed',
                 'report_qwen3_4b': '4fb86141'}},
    {'phase': 'N-P7', 'name': 'n3_device_transfer',
     'title': 'N3 条件化类别子空间装置跨 L2 结构/极性/语言族迁移'
              '（F2 color B/A=0.925, lstar=L27；seal 66afc202）',
     'sources': {'design_seal': '66afc202', 'memo_append': '952f8833',
                 'report_qwen3_4b': '51e19b7f'}},
]

DESIGN = {
    'phase': '3208', 'name': 'g5g7_nline_backfill_draft',
    'prereg': 'AGI_GPT5_MEMO Phase 3207 tail G5-G7 (frozen before any '
              'observation; default arm (b); pre-freeze probe confirmed '
              'deepseek-line Phase3-7 sealed artifacts)',
    'data': 'deepseek_temp Phase3-7 sealed files (15 sha-anchored) + shared '
            'atlas_ledger (read-only for untouched assertion); zero GPU',
    'control_anchor': '15 source file shas bitwise + verbatim content anchors '
                      '(P3/P4/P5/P6/P7) + ledger n==359 sha efe4a6de before==after',
    'measurements': 'B1 sources bitwise x15; B2 content anchors x7 files; '
                    'B3 draft entries x5 registered to '
                    'n_line_backfill_draft_v1.json (NOT applied)',
    'gate': 'draft-only: backfill entries registered as '
            'draft_pending_cross_line (APPX-4 governance); atlas_ledger NOT '
            'mutated; clean => n_line_backfill_draft_registered',
    'sources': SHA_ANCHOR,
    'gates': GATE,
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
        json.dump({'phase': 3208, 'name': 'g5g7_nline_backfill_draft',
                   'design_sha8': dsha[:8], 'design': DESIGN,
                   'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def run_recheck(loaded):
    out = {'contradictions': []}
    # B1: sources bitwise
    b1_bad = []
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        if got != v:
            b1_bad.append((k, got, v))
    if b1_bad:
        out['contradictions'].append(('B1_sources', b1_bad[:6]))
    log('B1: 15 sealed source files sha-bitwise: %s'
        % ('OK' if not b1_bad else 'FAIL %s' % b1_bad[:3]))

    # B2: content anchors
    b2_bad = []
    for key, anchors in CONTENT_ANCHORS.items():
        try:
            txt = open(SRC[key], encoding='utf-8', errors='replace').read()
        except Exception as e:
            b2_bad.append((key, 'read', str(e)[:40]))
            continue
        for a in anchors:
            if a not in txt:
                b2_bad.append((key, a))
    if b2_bad:
        out['contradictions'].append(('B2_content_anchors', b2_bad[:6]))
    log('B2: content anchors verbatim (P3 head / P4 seal fields / P5 memo / '
        'P6 report+seal / P7 report+seal): %s'
        % ('OK' if not b2_bad else 'FAIL %s' % b2_bad[:3]))

    # B3: ledger untouched + draft entries
    led = json.load(open(LEDGER, encoding='utf-8'))
    b3_ok = (len(led['measurements']) == 359 and led['ledger_sha256_8'] == 'efe4a6de')
    if not b3_ok:
        out['contradictions'].append(('B3_ledger_state',
                                      (len(led['measurements']), led['ledger_sha256_8'])))
    log('B3: atlas_ledger untouched (n=359, sha efe4a6de): %s'
        % ('OK' if b3_ok else 'FAIL'))
    out['recheck_clean'] = not out['contradictions']
    log('RECHECK: %s (%d contradictions)'
        % ('CLEAN' if out['recheck_clean'] else 'DIRTY', len(out['contradictions'])))
    return out


def write_draft(res8, seal8):
    draft = {
        'schema': 'n_line_backfill_draft_v1',
        'created': time.strftime('%Y-%m-%d %H:%M'),
        'status': 'draft_pending_cross_line',
        'governance': ('APPX-4/5 同型：只补不改；施加至共享 atlas_ledger 需跨线确认；'
                       '本文件为 3208 封存草案，主账本零变更'),
        'target': 'research/gpt5/atlas/atlas_ledger.json#measurements (append 5)',
        'entries': [dict(e, status='draft_pending_cross_line') for e in DRAFT_ENTRIES],
        'source_sha_chain': SHA_ANCHOR,
        'content_anchors': {k: v for k, v in CONTENT_ANCHORS.items()},
        'phase3208': {'name': 'g5g7_nline_backfill_draft',
                      'res_sha8': res8, 'seal_sha8': seal8},
    }
    suffix = 'smoke_' if SMOKE else ''
    p = os.path.join(OUTDIR, suffix + 'n_line_backfill_draft_v1.json')
    with open(p, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(draft, ensure_ascii=False, indent=1, sort_keys=True))
    log('draft written %s (sha8 %s)' % (os.path.basename(p), sha8_file(p)))
    return p


def main():
    design_sha = freeze()
    log('== Phase 3208 g5g7_nline_backfill_draft %s ==' % ('SMOKE' if SMOKE else 'FULL'))
    loaded = {}
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    control = run_recheck(loaded)
    gate_all = bool(control['recheck_clean'])
    if gate_all:
        verdict = ('n_line_backfill_draft_registered (B1 15 sealed sources '
                   'bitwise; B2 content anchors verbatim x7 files; B3 ledger '
                   'untouched n=359 efe4a6de; 5 draft entries N-P3..N-P7 '
                   'registered draft_pending_cross_line, APPX-4 governance, '
                   'NOT applied)')
    else:
        verdict = ('n_line_backfill_draft_dirty (%d contradictions; nothing '
                   'applied)' % len(control['contradictions']))
    log('verdict: %s' % verdict)
    suffix = 'smoke_' if SMOKE else ''
    summary = {
        'phase': 3208, 'name': 'g5g7_nline_backfill_draft',
        'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True, 'draft_only': True,
        'gates': dict(GATE, recheck_clean=bool(gate_all),
                      n_contradictions=len(control['contradictions'])),
        'recheck': control,
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
    write_draft(res8, seal8)
    # ledger untouched re-assertion (after everything)
    led2 = json.load(open(LEDGER, encoding='utf-8'))
    assert len(led2['measurements']) == 359 and led2['ledger_sha256_8'] == 'efe4a6de'
    log('ledger untouched re-assertion OK (n=359, efe4a6de)')
    with open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
