# -*- coding: utf-8 -*-
# Phase 3180 (G5-B5): FTR-15 held_out 实体留出臂（零 GPU，3158 封存 collect 复用）
# 预注册: AGI_GPT5_MEMO Phase 3179 tail（观测前）:
#   数据源 = phase3158 collect.npz BUD(arms=3,16 anchors,8 dirs) + anchor_idx（16 锚恰为 16 实体每实体一行，
#            经 3157 collect ei/rel/pol/ctx 核实）。
#   协议: 实体 LOO——fold e: ratio_train(e) = median(BUD_null[其余 15 锚 x 8 dirs]) /
#         median(BUD_rand[其余 15 锚 x 8 dirs])（3158 ratio 定义 verbatim，censored 已按 grid max 记账）;
#         held 读数 c(e) = median(null_e)/median(rand_e)（方向性描述量）。
#   门（freeze 前钉定）:
#     G1 装置: 由 BUD 重算全量 ratio，对封存 3158 result ratio drift <= 1e-9 x3 模型;
#     G2 held_out: 全部 16 fold ratio_train(e) 与全量同 class（<3.0 absent，全量 1.03/1.33/1.25）
#        且 max |log(ratio_train(e)/ratio_full)| < 0.25 x3 模型;
#     G3 方向性描述: 每模型 c(e)>1 的锚计数（16 锚，报告不设门）。
#   PASS(G1 且 G2) => FTR-15 升 E2_predictive（held_out 锚入表，upgrade_log #6）;
#   FAIL => F17 入失败账本（归因分解预注册下一 phase，同 3177->3178 先例）。
#   registry v1.6 -> v1.7（append only，其余字节保全）+ atlas_v1_8.html 渲染逐字段校验。
import os, sys, json, time, hashlib, copy, html, re, datetime
import numpy as np

T0 = time.time()
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3179_g5b4_ftr23_registration as p79  # 渲染器 verbatim 复用

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
NAME = 'g5b5_ftr15_heldout'
OUTDIR = os.path.join(RDIR, 'phase3180', NAME)
SMOKE = os.environ.get('P3180_SMOKE') == '1'
LOG = []

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

S8 = os.path.join(RDIR, 'phase3158', 'g4p1_output_equivalence_class')
SRC = {
    'p3158_collect_4b': os.path.join(S8, 'qwen3-4b', 'collect.npz'),
    'p3158_collect_14b': os.path.join(S8, 'qwen3-14b', 'collect.npz'),
    'p3158_collect_glm4': os.path.join(S8, 'glm4', 'collect.npz'),
    'p3158_result_4b': os.path.join(S8, 'qwen3-4b', 'result.json'),
    'p3158_result_14b': os.path.join(S8, 'qwen3-14b', 'result.json'),
    'p3158_result_glm4': os.path.join(S8, 'glm4', 'result.json'),
    'p3158_summary': os.path.join(S8, 'summary', 'result_summary.json'),
    'registry_v16': os.path.join(RDIR, 'phase3179', 'g5b4_ftr23_registration', 'atlas_registry_v1_6.json'),
    'gap_v15': os.path.join(RDIR, 'phase3175', 'g5a12_residual_arms', 'gap_ledger_v1_5.json'),
    'registry_3162': os.path.join(RDIR, 'phase3162', 'g5a1_atlas_foundation', 'atlas_registry.json'),
}
SHA_ANCHOR = {
    'p3158_collect_4b': 'c4d9b9fb', 'p3158_collect_14b': '7f53f3a0', 'p3158_collect_glm4': '2ea72cfe',
    'p3158_result_4b': '90e734fd', 'p3158_result_14b': '1f532016', 'p3158_result_glm4': 'c49614fc',
    'p3158_summary': '7ed85b36', 'registry_v16': 'bebdf38d', 'gap_v15': '4fb3f37d',
    'registry_3162': '00f15e98',
}
MODELS = [('qwen3-4b', 'p3158_collect_4b', 'p3158_result_4b'),
          ('qwen3-14b', 'p3158_collect_14b', 'p3158_result_14b'),
          ('glm4', 'p3158_collect_glm4', 'p3158_result_glm4')]
GATE_LOG_BAND = 0.25   # G2: max |log(ratio_train/ratio_full)|（freeze 钉定）
GATE_CLASS = 3.0       # 3158 quotient class 边界（absent）
N_FOLDS_SMOKE = 3

def freeze():
    design = {
        'phase': '3180', 'name': NAME,
        'prereg': 'AGI_GPT5_MEMO Phase 3179 tail G5-B5 (frozen before any observation)',
        'data': 'sealed 3158 collect.npz BUD(3 arms,16 anchors,8 dirs) + anchor_idx; '
                'zero GPU; anchors verified 1-per-entity via 3157 collect ei labels',
        'protocol': 'entity-LOO folds (16): ratio_train(e)=median(BUD_null[15 train anchors x8 dirs])/'
                    'median(BUD_rand[...]) (3158 ratio rule verbatim); held c(e)=median(null_e)/median(rand_e)',
        'gates': {'G1_device_drift': 1e-9, 'G2_log_band': GATE_LOG_BAND, 'G2_class': GATE_CLASS,
                  'G3_directional_report_only': True},
        'pass_action': 'FTR-15 E1->E2_predictive, held_out anchor append, upgrade_log #6',
        'fail_action': 'F17 append, attribution prereg next phase',
        'registry': 'v1.6 -> v1.7 append-only + atlas_v1_8.html render + field check',
        'sources': SHA_ANCHOR,
    }
    eblob = json.dumps(design, ensure_ascii=False, sort_keys=True, indent=1).encode('utf-8')
    dsha = hashlib.sha256(eblob).hexdigest()
    os.makedirs(OUTDIR, exist_ok=True)
    exe_p = os.path.join(OUTDIR, 'execution.json')
    if os.path.exists(exe_p) and not SMOKE:
        prev = json.load(open(exe_p, encoding='utf-8'))
        assert prev['design_sha8'] == dsha[:8], 'DESIGN DRIFT'
        log('freeze: existing execution.json OK (%s)' % dsha[:8])
    elif not os.path.exists(exe_p) or SMOKE:
        json.dump({'phase': 3180, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': design, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]

def seal_result(summary, suffix=''):
    raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = sha8(raw.encode('utf-8'))
    mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
    seal8 = sha8(mid.encode('utf-8'))
    rp = os.path.join(OUTDIR, suffix + 'result.json')
    with open(rp, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(dict(summary, res_sha8=res8, seal_sha8=seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('result written res=%s seal=%s' % (res8, seal8))
    return res8, seal8

def ratio_of(null_b, rand_b):
    return float(np.median(null_b) / np.median(rand_b))

def run_core(folds_limit=None):
    """returns per-model readings dict"""
    out = {}
    for mname, ck, rk in MODELS:
        assert sha8_file(SRC[ck]) == SHA_ANCHOR[ck], ('collect sha', mname)
        assert sha8_file(SRC[rk]) == SHA_ANCHOR[rk], ('result sha', mname)
        z = np.load(SRC[ck], allow_pickle=True)
        bud, cen, ai = z['BUD'], z['CEN'], z['anchor_idx']
        assert bud.shape == (3, 16, 8), bud.shape
        null_b, rand_b = bud[0], bud[1]   # (16,8)
        # G1: 全量 ratio 重算 vs 封存
        ratio_full = ratio_of(null_b.ravel(), rand_b.ravel())
        sealed = json.load(open(SRC[rk], encoding='utf-8'))['ratio']
        drift = abs(ratio_full - sealed) / max(sealed, 1e-12)
        # G2: 实体 LOO（每锚一实体；SMOKE 仅前 folds_limit 个 fold）
        folds = []
        for e in range(16):
            if folds_limit is not None and e >= folds_limit:
                break
            tr = [a for a in range(16) if a != e]
            r_tr = ratio_of(null_b[tr].ravel(), rand_b[tr].ravel())
            band = abs(np.log(r_tr / ratio_full))
            folds.append({'fold': e, 'ratio_train': r_tr, 'log_band': float(band),
                          'same_class': bool(r_tr < GATE_CLASS)})
        c_held = [float(np.median(null_b[e]) / np.median(rand_b[e])) for e in range(16)]
        out[mname] = {
            'ratio_full_recompute': ratio_full, 'ratio_sealed': sealed, 'drift': float(drift),
            'folds': folds, 'c_held': c_held,
            'n_c_gt1': int(sum(1 for c in c_held if c > 1.0)),
            'max_band': max(f['log_band'] for f in folds),
            'all_same_class': all(f['same_class'] for f in folds),
        }
        log('%s: ratio recompute %.6f vs sealed %.6f drift %.2e; folds %d max_band %.4f '
            'same_class %s; c>1 %d/16' % (mname, ratio_full, sealed, drift, len(folds),
                                          out[mname]['max_band'], out[mname]['all_same_class'],
                                          out[mname]['n_c_gt1']))
    return out

def fmt(v):
    if isinstance(v, float):
        return ('%.6g' % v)
    return str(v)

def build_registry_v17(core, res8, seal8):
    reg16 = json.loads(open(SRC['registry_v16'], 'rb').read().decode('utf-8'))
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    assert reg16['version'] == '1.6' and len(reg16['features']) == 23
    reg17 = copy.deepcopy(reg16)
    reg17['version'] = '1.7'
    reg17['supersedes'] = 'atlas_registry_v1_6.json (bebdf38d, Phase 3179)'
    f15 = [f for f in reg17['features'] if f['id'] == 'FTR-15'][0]
    held_anchor = {
        'src': 'p3180', 'phase': 3180, 'role': 'held_out_entity_loo',
        'tags': ['cross_model', 'held_out'],
        'asserts': {
            'ratio_full_recompute.qwen3-4b': core['qwen3-4b']['ratio_full_recompute'],
            'ratio_full_recompute.qwen3-14b': core['qwen3-14b']['ratio_full_recompute'],
            'ratio_full_recompute.glm4': core['glm4']['ratio_full_recompute'],
            'max_log_band.qwen3-4b': core['qwen3-4b']['max_band'],
            'max_log_band.qwen3-14b': core['qwen3-14b']['max_band'],
            'max_log_band.glm4': core['glm4']['max_band'],
            'all_same_class': bool(all(core[m]['all_same_class'] for m, _, _ in MODELS)),
            'res_sha8': res8, 'seal_sha8': seal8,
        },
    }
    f15['anchors'] = f15['anchors'] + [held_anchor]
    f15['evidence_level'] = 'E2_predictive'
    f15['scope_limits'] = (str(f15.get('scope_limits') or '') +
                           '；held_out = 实体 LOO fold（16 锚=16 实体，3158 口径 budget 比带 <0.25 log 单位），'
                           '「局部等价」仍不外推为全局商结构').lstrip('；')
    # 其余 22 特征字节保全
    for f in reg16['features']:
        if f['id'] == 'FTR-15':
            continue
        a = json.dumps(f, ensure_ascii=False, sort_keys=True)
        b = json.dumps([x for x in reg17['features'] if x['id'] == f['id']][0],
                       ensure_ascii=False, sort_keys=True)
        assert a == b, ('feature not preserved', f['id'])
    reg17['upgrade_log'] = reg16['upgrade_log'] + [{
        'node': 'FTR-15',
        'change': 'E1_repeatable -> E2_predictive (held_out tag; entity-LOO budget-ratio band 3/3)',
        'reason': '3180 arm: 16 entity-LOO folds, all ratio_train same class (<3.0) and '
                  'max |log(ratio_train/ratio_full)| %.4f/%.4f/%.4f < 0.25 x3 models; '
                  'device drift 0 vs sealed 3158' % (core['qwen3-4b']['max_band'],
                                                      core['qwen3-14b']['max_band'],
                                                      core['glm4']['max_band']),
    }]
    assert len(reg17['upgrade_log']) == 6
    for k in ('failures', 'evidence_level_taxonomy', 'principles', 'schema',
              'models_mainline', 'models_pending', 'provenance', 'created', 'phase',
              'name', 'design_sha8'):
        assert json.dumps(reg16.get(k), ensure_ascii=False, sort_keys=True) == \
            json.dumps(reg17.get(k), ensure_ascii=False, sort_keys=True), (k, 'changed')
    log('registry v1.7 built: FTR-15 E2_predictive, held_out anchor appended, upgrade_log #6')
    return reg17, nodes, gl15

def render_v18(reg17, nodes, gl, meta_sha):
    p79.CREATED_STR = CREATED_STR
    H = []
    H.append('<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">')
    H.append('<title>Atlas v1.8 - RDC/LPF 语义图谱</title><style>%s</style></head>' % p79.CSS)
    H.append('<body>')
    H.append('<h1>Atlas v1.8 &mdash; 语义机制图谱（registry v1.7 渲染）</h1>')
    H.append('<div class="sub">Phase 3180 G5-B5 &middot; FTR-15 held_out 实体留出臂（E1&rarr;E2） &middot; '
             'registry v1.7 sha8=%s &middot; registry v1.6 sha8=%s &middot; registry v1.5 sha8=%s'
             ' &middot; registry v1.4 sha8=%s &middot; registry v1.3 sha8=%s'
             ' &middot; gap ledger v1.5 sha8=%s（3175 定稿，原样渲染）'
             ' &middot; 源 p3158 collect %s/%s/%s + summary %s &middot; 渲染于 %s</div>' % (
                 span('meta.registry_v17_sha8', meta_sha['reg17']),
                 span('meta.registry_v16_sha8', 'bebdf38d'),
                 span('meta.registry_v15_sha8', p79.SHA_ANCHOR['registry_v15']),
                 span('meta.registry_v14_sha8', 'fb11d633'),
                 span('meta.registry_v13_sha8', 'e64733f7'),
                 span('meta.gap_v15_sha8', p79.SHA_ANCHOR['gap_v15']),
                 span('meta.p3158_4b', SHA_ANCHOR['p3158_collect_4b']),
                 span('meta.p3158_14b', SHA_ANCHOR['p3158_collect_14b']),
                 span('meta.p3158_glm4', SHA_ANCHOR['p3158_collect_glm4']),
                 span('meta.p3158_summary', SHA_ANCHOR['p3158_summary']),
                 span('meta.created', CREATED_STR)))
    H.append('<h2>证据级 taxonomy 与三原则</h2><div class="card">')
    for k, v in reg17['evidence_level_taxonomy'].items():
        H.append('<div class="kv"><b>%s</b>%s</div>' % (p79.esc(k), p79.esc(v)))
    H.append('<hr style="border:none;border-top:1px solid var(--line);margin:8px 0">')
    for i, pr in enumerate(reg17['principles']):
        H.append('<div class="kv"><b>原则%d</b>%s</div>' % (i + 1, p79.esc(pr)))
    H.append('</div>')
    H.append('<h2>图谱节点基座（16 节点，3162 audit disk_verified）</h2>')
    H.append('<table><tr><th>id</th><th>title</th><th>evidence</th><th>checks</th></tr>')
    for n in nodes:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td><span class="badge b-%s">%s</span>%s</td>'
                 '<td class="mono">%s</td></tr>' % (
                     n['id'], span(n['id'] + '.title', n['title']),
                     n['evidence_level'], span(n['id'] + '.elevel', n['evidence_level']),
                     span(n['id'] + '.status', n['status']),
                     span(n['id'] + '.checks', '%d/%d' % (n['n_pass'], n['n_checks']))))
    H.append('</table>')
    fams = {}
    for f in reg17['features']:
        fams.setdefault(f['family'], []).append(f)
    H.append('<h2>跨模型稳定特征登记表（registry v1.7，%d 条）</h2>' % len(reg17['features']))
    for fam in sorted(fams):
        H.append('<h3>family = %s（%d 条）</h3>' % (p79.esc(fam), len(fams[fam])))
        H.append('<div class="grid">')
        for f in fams[fam]:
            H.append(p79.render_feature(f))
        H.append('</div>')
    H.append('<h2>E 级 / 范围升级链</h2><div class="card">')
    for i, u in enumerate(reg17['upgrade_log']):
        H.append('<div class="kv"><b>#%d</b>%s &nbsp;%s</div><div class="evid">reason: %s</div>' % (
            i + 1, span('UPG.%d.node' % i, u['node']), span('UPG.%d.change' % i, u['change']),
            span('UPG.%d.reason' % i, u['reason'])))
    H.append('</div>')
    H.append('<h2>失败账本（F1-F%d，可证伪性存档）</h2>' % len(reg17['failures']))
    H.append('<table><tr><th>id</th><th>kind</th><th>phase</th><th>text</th><th>evidence</th></tr>')
    for fl in reg17['failures']:
        H.append('<tr><td class="mono">%s</td><td>%s</td><td class="mono">%s</td><td>%s</td><td class="mono">%s</td></tr>' % (
            fl['id'], span(fl['id'] + '.kind', fl['kind']), span(fl['id'] + '.phase', fl['phase']),
            span(fl['id'] + '.text', fl['text']), span(fl['id'] + '.evidence', fl['evidence'])))
    H.append('</table>')
    H.append('<h2>缺口账本 v1.5（GAP-4 quantified_collapse 定稿）</h2>')
    for g in gl['gaps']:
        closed_at = str(g['closed_at_phase']) if g.get('closed_at_phase') is not None else '-'
        badge_closed = '<span class="badge">closed@%s</span>' % span(g['id'] + '.closed_at', closed_at) \
            if g['status'].startswith('closed') else ''
        H.append('<div class="card"><h3>%s &middot; %s <span class="badge b-%s">%s</span>%s</h3>' % (
            g['id'], span(g['id'] + '.title', g['title']), g['status'], span(g['id'] + '.status', g['status']), badge_closed))
        H.append('<div class="kv"><b>statement</b>%s</div>' % span(g['id'] + '.statement', g['statement']))
        H.append('<div class="evid">evidence:</div>')
        H.append('<div class="kv">%s</div>' % span(g['id'] + '.evidence', ' || '.join(g['evidence'])))
        H.append('<div class="anchorbox"><b>anchors</b> %s</div>' % span(
            g['id'] + '.anchors', ', '.join('%s=%s' % (k, v) for k, v in sorted(g['anchor_sha8'].items()))))
        if g['id'] == 'GAP-4':
            pr = g['prereg']
            H.append('<div class="prereg"><div class="kv"><b>prereg</b>Phase %s %s（%s）</div>' % (
                span('GAP-4.prereg.phase', pr['phase']), span('GAP-4.prereg.name', pr['name']),
                span('GAP-4.prereg.status', pr['status'])))
            H.append('<div class="kv"><b>hypothesis</b>%s</div>' % span('GAP-4.prereg.hypothesis', pr['hypothesis']))
            H.append('<div class="kv"><b>protocol</b>%s</div>' % span('GAP-4.prereg.protocol', pr['protocol']))
            H.append('<div class="kv"><b>gate</b>%s</div></div>' % span('GAP-4.prereg.gate', pr['gate']))
            H.append('<div class="mech"><div class="kv"><b>mechanism_note</b>%s</div></div>' % span(
                'GAP-4.mechanism_note', g['mechanism_note']))
        H.append('</div>')
    H.append('<h2>附录挂账（%d 项，逐条证据锚）</h2><div class="card">' % len(gl['appendix']))
    for a in gl['appendix']:
        H.append('<div class="kv"><b>%s</b>%s <span class="badge b-%s">%s</span>'
                 '<span class="mono">anchor=%s(%s)</span><div class="evid">%s</div></div>' % (
                     a['id'], span(a['id'] + '.title', a['title']), a['status'],
                     span(a['id'] + '.status', a['status']),
                     span(a['id'] + '.anchor', a['anchor']),
                     span(a['id'] + '.anchor_sha8', a['anchor_sha8']),
                     span(a['id'] + '.note', a['note'])))
    H.append('</div>')
    H.append('<div class="footer">RDC/LPF atlas v1.8 &middot; 渲染自 atlas_registry_v1_7.json (%s)'
             ' + gap ledger v1.5 (%s，3175 定稿原样渲染) &middot; 增量来源: Phase 3180 G5-B5 '
             'FTR-15 held_out 实体留出臂（E1&rarr;E2，数据源 p3158 collect %s / %s / %s + summary %s）'
             '&middot; 本文件为 Phase 3180 封存产物，字段由 data-k 校验器逐字段对盘。</div>' % (
                 meta_sha['reg17'], p79.SHA_ANCHOR['gap_v15'],
                 SHA_ANCHOR['p3158_collect_4b'], SHA_ANCHOR['p3158_collect_14b'],
                 SHA_ANCHOR['p3158_collect_glm4'], SHA_ANCHOR['p3158_summary']))
    H.append('</body></html>')
    return '\n'.join(H)

def flatten(reg, nodes, gl, meta_sha):
    d = p79_flatten_core(reg, nodes, gl)
    d['meta.registry_v17_sha8'] = meta_sha['reg17']
    d['meta.registry_v16_sha8'] = 'bebdf38d'
    d['meta.registry_v15_sha8'] = p79.SHA_ANCHOR['registry_v15']
    d['meta.registry_v14_sha8'] = 'fb11d633'
    d['meta.registry_v13_sha8'] = 'e64733f7'
    d['meta.gap_v15_sha8'] = p79.SHA_ANCHOR['gap_v15']
    d['meta.p3158_4b'] = SHA_ANCHOR['p3158_collect_4b']
    d['meta.p3158_14b'] = SHA_ANCHOR['p3158_collect_14b']
    d['meta.p3158_glm4'] = SHA_ANCHOR['p3158_collect_glm4']
    d['meta.p3158_summary'] = SHA_ANCHOR['p3158_summary']
    d['meta.created'] = CREATED_STR
    return d

def p79_flatten_core(reg, nodes, gl):
    d = {}
    for f in reg['features']:
        p = f['id']
        d[p + '.statement'] = f['statement']
        d[p + '.family'] = f['family']
        d[p + '.evidence_level'] = f['evidence_level']
        d[p + '.model_scope'] = ', '.join(f['model_scope'])
        d[p + '.n_anchors'] = str(len(f['anchors']))
        d[p + '.counter_evidence'] = ' | '.join(f['counter_evidence']) if f['counter_evidence'] else '(none)'
        d[p + '.replication'] = f['replication']
        d[p + '.scope_limits'] = f['scope_limits'] if f['scope_limits'] else '(none)'
        for i, a in enumerate(f['anchors']):
            d['%s.anchors.%d.src' % (p, i)] = str(a['src'])
            d['%s.anchors.%d.phase' % (p, i)] = str(a['phase'])
            d['%s.anchors.%d.role' % (p, i)] = a['role']
            d['%s.anchors.%d.asserts' % (p, i)] = json.dumps(a['asserts'], ensure_ascii=False, sort_keys=True)
        for k, v in (f.get('values') or {}).items():
            d['%s.values.%s' % (p, k)] = fmt(v)
    for n in nodes:
        d[n['id'] + '.title'] = n['title']
        d[n['id'] + '.elevel'] = n['evidence_level']
        d[n['id'] + '.status'] = n['status']
        d[n['id'] + '.checks'] = '%d/%d' % (n['n_pass'], n['n_checks'])
    for i, u in enumerate(reg['upgrade_log']):
        d['UPG.%d.node' % i] = u['node']
        d['UPG.%d.change' % i] = u['change']
        d['UPG.%d.reason' % i] = u['reason']
    for fl in reg['failures']:
        d[fl['id'] + '.kind'] = fl['kind']
        d[fl['id'] + '.phase'] = str(fl['phase'])
        d[fl['id'] + '.text'] = fl['text']
        d[fl['id'] + '.evidence'] = str(fl['evidence'])
    for g in gl['gaps']:
        p = g['id']
        d[p + '.title'] = g['title']
        d[p + '.status'] = g['status']
        if g['status'].startswith('closed'):
            d[p + '.closed_at'] = str(g['closed_at_phase'])
        d[p + '.statement'] = g['statement']
        d[p + '.evidence'] = ' || '.join(g['evidence'])
        d[p + '.anchors'] = ', '.join('%s=%s' % (k, v) for k, v in sorted(g['anchor_sha8'].items()))
        if p == 'GAP-4':
            pr = g['prereg']
            d[p + '.prereg.phase'] = str(pr['phase'])
            d[p + '.prereg.name'] = pr['name']
            d[p + '.prereg.hypothesis'] = pr['hypothesis']
            d[p + '.prereg.protocol'] = pr['protocol']
            d[p + '.prereg.gate'] = pr['gate']
            d[p + '.prereg.status'] = pr['status']
            d[p + '.mechanism_note'] = g['mechanism_note']
    for a in gl['appendix']:
        p = a['id']
        d[p + '.title'] = a['title']
        d[p + '.status'] = a['status']
        d[p + '.note'] = a['note']
        d[p + '.anchor'] = a['anchor']
        d[p + '.anchor_sha8'] = a['anchor_sha8']
    return d

span = p79.span
CREATED_STR = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')

def main():
    design_sha = freeze()
    for k, v in SHA_ANCHOR.items():
        got = sha8_file(SRC[k])
        assert got == v, ('source sha', k, got, v)
    log('sources: %d files sha-asserted' % len(SHA_ANCHOR))
    suffix = 'smoke_' if SMOKE else ''
    folds_limit = N_FOLDS_SMOKE if SMOKE else None
    core = run_core(folds_limit)
    g1 = all(core[m]['drift'] <= 1e-9 for m, _, _ in MODELS)
    g2 = all(core[m]['all_same_class'] and core[m]['max_band'] < GATE_LOG_BAND
             for m, _, _ in MODELS)
    log('gates: G1_device=%s G2_heldout=%s' % (g1, g2))
    verdict = ''
    if g1 and g2 and not SMOKE:
        verdict = ('ftr15_upgraded_E2_predictive (entity-LOO held_out arm, 16 folds x3 models; '
                   'max log band %.4f/%.4f/%.4f < 0.25, all folds same class <3.0; device drift 0; '
                   'c>1 anchors %d/%d/%d of 16)' % (
                       core['qwen3-4b']['max_band'], core['qwen3-14b']['max_band'],
                       core['glm4']['max_band'], core['qwen3-4b']['n_c_gt1'],
                       core['qwen3-14b']['n_c_gt1'], core['glm4']['n_c_gt1']))
    elif g1 and g2 and SMOKE:
        verdict = 'smoke device green (%d folds; numbers above)' % N_FOLDS_SMOKE
    else:
        verdict = 'gate_fail (G1=%s G2=%s) -> F17 route (registry FAIL branch)' % (g1, g2)
    res8 = seal8 = 'pending'
    summary = {
        'phase': 3180, 'name': NAME, 'mode': 'SMOKE' if SMOKE else 'FULL',
        'design_sha8': design_sha, 'zero_gpu': True,
        'per_model': {m: {k: v for k, v in core[m].items() if k != 'folds'}
                      for m, _, _ in MODELS},
        'folds_full': {m: core[m]['folds'] for m, _, _ in MODELS} if not SMOKE else {},
        'gates': {'G1_device': bool(g1), 'G2_heldout': bool(g2),
                  'G2_log_band': GATE_LOG_BAND, 'G2_class': GATE_CLASS},
        'verdict': verdict,
    }
    if not SMOKE:
        if g1 and g2:
            res8, seal8 = seal_result(dict(summary, verdict=verdict), '')
            reg17, nodes, gl = build_registry_v17(core, res8, seal8)
            rp = os.path.join(OUTDIR, 'atlas_registry_v1_7.json')
            with open(rp, 'w', encoding='utf-8', newline='') as f:
                f.write(json.dumps(reg17, ensure_ascii=False, indent=1))
            reg17_sha = sha8_file(rp)
            meta_sha = {'reg17': reg17_sha}
            html_text = render_v18(reg17, nodes, gl, meta_sha)
            hp = os.path.join(OUTDIR, 'atlas_v1_8.html')
            with open(hp, 'w', encoding='utf-8', newline='') as f:
                f.write(html_text)
            html_sha = sha8_file(hp)
            expected = flatten(reg17, nodes, gl, meta_sha)
            v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
            log('registry v1.7 written sha8 %s' % reg17_sha)
            log('atlas_v1_8.html written sha8 %s (fields %d/%d, %d/%d/%d/%d)' % (
                html_sha, v['n_expected'], v['n_found'], len(v['missing']),
                len(v['extra']), len(v['mismatch']), len(v['dups'])))
            assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
            final = {
                'phase': 3180, 'name': NAME, 'mode': 'FULL', 'design_sha8': design_sha,
                'zero_gpu': True, 'arm': {'verdict': verdict,
                                          'per_model': summary['per_model']},
                'counts': {'features': 23, 'failures': len(reg17['failures']),
                           'upgrades': len(reg17['upgrade_log']), 'gaps': 4,
                           'appendix': len(gl['appendix'])},
                'registry_disposition': 'v1.6_to_v1.7_ftr15_E2_upgrade_log6',
                'registry_v17_file': 'atlas_registry_v1_7.json', 'registry_v17_sha8': reg17_sha,
                'html_file': 'atlas_v1_8.html', 'html_sha8': html_sha,
                'field_check': {'n_expected': v['n_expected'], 'n_found': v['n_found'],
                                'missing': len(v['missing']), 'extra': len(v['extra']),
                                'mismatch': len(v['mismatch']), 'dups': len(v['dups'])},
                'sources': SHA_ANCHOR,
            }
            seal_result(final, 'final_')
        else:
            # FAIL 分支：F17 + counter_evidence append
            res8, seal8 = seal_result(dict(summary, verdict=verdict), '')
            reg17, nodes, gl = build_registry_v17_fail(res8, seal8, core)
            rp = os.path.join(OUTDIR, 'atlas_registry_v1_7.json')
            with open(rp, 'w', encoding='utf-8', newline='') as f:
                f.write(json.dumps(reg17, ensure_ascii=False, indent=1))
            reg17_sha = sha8_file(rp)
            meta_sha = {'reg17': reg17_sha}
            html_text = render_v18(reg17, nodes, gl, meta_sha)
            hp = os.path.join(OUTDIR, 'atlas_v1_8.html')
            with open(hp, 'w', encoding='utf-8', newline='') as f:
                f.write(html_text)
            expected = flatten(reg17, nodes, gl, meta_sha)
            v = p79.verify_html(open(hp, encoding='utf-8').read(), expected)
            assert not v['missing'] and not v['extra'] and not v['mismatch'] and not v['dups']
            log('FAIL branch registered: F17, registry v1.7 sha8 %s' % reg17_sha)
            final = {'phase': 3180, 'name': NAME, 'mode': 'FULL',
                     'verdict': verdict, 'registry_v17_sha8': reg17_sha,
                     'html_sha8': sha8_file(hp), 'sources': SHA_ANCHOR}
            seal_result(final, 'final_')
    else:
        res8, seal8 = seal_result(summary, 'smoke_')
    with open(os.path.join(OUTDIR, suffix + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')
    log('verdict: %s' % verdict)

def build_registry_v17_fail(res8, seal8, core):
    reg16 = json.loads(open(SRC['registry_v16'], 'rb').read().decode('utf-8'))
    gl15 = json.loads(open(SRC['gap_v15'], encoding='utf-8').read())
    nodes = json.load(open(SRC['registry_3162'], encoding='utf-8'))['audit']['nodes']
    reg17 = copy.deepcopy(reg16)
    reg17['version'] = '1.7'
    reg17['supersedes'] = 'atlas_registry_v1_6.json (bebdf38d, Phase 3179)'
    f15 = [f for f in reg17['features'] if f['id'] == 'FTR-15'][0]
    f15['counter_evidence'] = f15['counter_evidence'] + [
        'F17: E1->E2 held_out 门未过（3180 实体 LOO fold 超带 0.25 或出 class；逐 fold 见 result）']
    reg17['failures'] = reg16['failures'] + [{
        'id': 'F17', 'kind': 'upgrade_gate_failed', 'phase': 3180,
        'text': 'E1->E2 升级门未全过：FTR-15 实体留出 fold（max log band 或 class 出界，x3 模型见 result）。'
                '相应特征保持 E1_repeatable，held_out 锚不入表（诚实登记）。',
        'evidence': 'p3180 run (this phase); gates G1/G2 = %s/%s' % (
            all(core[m]['drift'] <= 1e-9 for m, _, _ in MODELS),
            all(core[m]['all_same_class'] and core[m]['max_band'] < GATE_LOG_BAND
                for m, _, _ in MODELS))}]
    return reg17, nodes, gl15

if __name__ == '__main__':
    main()
