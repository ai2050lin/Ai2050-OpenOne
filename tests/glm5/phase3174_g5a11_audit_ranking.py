# -*- coding: utf-8 -*-
# Phase 3174: G5-A11 atlas v1.3 integrity audit + gap ranking adjudication (zero GPU).
#
# Ten gates:
#   G1  source sha8 anchors (18 sealed artifacts, all measured 2026-10-09 probes)
#   G2  independent-process SHADOW re-render of Phase 3173 (patch L40 count==1,
#       registry/gap byte-identity + HTML line-diff timestamp whitelist + data-k counts)
#   G3  registry v1.2 structural invariants incl. v1.1 byte-preservation re-assert
#   G4  disk resolvability: full phase3* tree sha8 index; every feature anchor and
#       GAP-4 anchor_sha8 must resolve on disk
#   G5  independent numeric re-derivation of FTR-22 / FTR-21 from sealed sources
#       (exact float equality where deterministic)
#   G6  E-taxonomy audit + tag-completeness findings (recorded, NO retroactive
#       demotion) + FTR-03 prereg-premise correction (already E2_predictive)
#   G7  E1->E2 upgrade-path assessment (zero-GPU candidates via existing npz)
#   G8  pre-registration DRAFT artifact for Phase 3175 (structural-residual cross-arm)
#   G9  gap ranking adjudication (3 ranked items)
#   G10 ledger state check (n>=325, phase 3173 present)
#
# SMOKE: env P3174_SMOKE=1 -> subset (child render runs 3173 smoke mode; G7 content,
# G10 and html spot-checks skipped). FULL = everything.
#
# Immutability: this script writes ONLY into its own OUTDIR and the gpt5_temp shadow
# area. The sealed phase3173 directory is read-only here.
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
GPT = os.path.join(ROOT, 'tests', 'gpt5_temp')
OUTDIR = os.path.join(SRC_DIR, 'phase3174', 'g5a11_audit')
SMOKE = os.environ.get('P3174_SMOKE', '') == '1'

SEALDIR = os.path.join(SRC_DIR, 'phase3173', 'g5a10_atlas_v13')
P3170 = os.path.join(SRC_DIR, 'phase3170', 'g5a7_atlas_v11')
P73SRC = os.path.join(ROOT, 'tests', 'glm5', 'phase3173_g5a10_atlas_v13.py')
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')

LOG = []


def log(s):
    LOG.append('[3174] ' + s)
    print('[3174] ' + s, flush=True)


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def jload(p):
    return json.load(io.open(p, encoding='utf-8'))


# ------------------------------------------------------------------- SHA anchors
# All values measured on real disk 2026-10-09 (probes p3174_src.txt / p3174_probe.txt).
SHA_ANCHOR = {
    'reg12': '0e5abcaf', 'gl14': '2413a0cc', 'html13': 'ff875ca3',
    'exec73': '3b90c5e0', 'res73': 'f98ae4c4',
    'smoke_reg12': '0e5abcaf', 'smoke_gl14': '2413a0cc',
    'smoke_html13': '2e54d61c', 'smoke_res73': '3c4c604d',
    'reg11': '1fedbd80', 'gl13': '2436ec08', 'gl11': '57716a37',
    'r69': '5b51c2c1', 'r71': 'a43cf48c', 'r72': 'd8ddc481',
    'q03': '57827730', 'rev3151b': 'dd0cc176',
    'agate': '24c60160',
}
SHA_PATHS = {
    'reg12': os.path.join(SEALDIR, 'atlas_registry_v1_2.json'),
    'gl14': os.path.join(SEALDIR, 'gap_ledger_v1_4.json'),
    'html13': os.path.join(SEALDIR, 'atlas_v1_3.html'),
    'exec73': os.path.join(SEALDIR, 'execution.json'),
    'res73': os.path.join(SEALDIR, 'result.json'),
    'smoke_reg12': os.path.join(SEALDIR, 'smoke_atlas_registry_v1_2.json'),
    'smoke_gl14': os.path.join(SEALDIR, 'smoke_gap_ledger_v1_4.json'),
    'smoke_html13': os.path.join(SEALDIR, 'smoke_atlas_v1_3.html'),
    'smoke_res73': os.path.join(SEALDIR, 'smoke_result.json'),
    'reg11': os.path.join(P3170, 'atlas_registry_v1_1.json'),
    'gl13': os.path.join(P3170, 'gap_ledger_v1_3.json'),
    'gl11': os.path.join(P3170, 'gap_ledger_v1_1.json'),
    'r69': os.path.join(SRC_DIR, 'phase3169', 'g5a6_oov_panel', 'result.json'),
    'r71': os.path.join(SRC_DIR, 'phase3171', 'g5a8_collapse_mechanism', 'result.json'),
    'r72': os.path.join(SRC_DIR, 'phase3172', 'g5a9_port_calibration', 'result.json'),
    'q03': os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q03_result.json'),
    'rev3151b': os.path.join(SRC_DIR, 'phase3151', 'g1p1_combo_additive_vs_interaction',
                             'result_rev3151b.json'),
    'agate': os.path.join(ROOT, 'research', 'deepseek', 'atlas', 'a_gate_closure_v1.json'),
}

# Static expectations derived from sealed artifacts (observed in pre-freeze probes,
# 2026-10-09; tag-completeness findings are metadata notes, NOT evidence-level changes).
EXPECTED_E1 = ['FTR-04', 'FTR-06', 'FTR-08', 'FTR-14', 'FTR-15']
EXPECTED_E3 = ['FTR-10', 'FTR-11', 'FTR-16']
EXPECTED_TAXONOMY_NOTES = ['FTR-05', 'FTR-09', 'FTR-12', 'FTR-13', 'FTR-18', 'FTR-19']
HTML_FIELDS_FULL = 637

DESIGN = dict(
    phase=3174, name='g5a11_audit', prereg_id='G5-A11', line='G',
    mode='zero GPU audit; independent-process shadow re-render; no writes into sealed dirs',
    smoke='env P3174_SMOKE=1 runs a subset: child render executes Phase 3173 in smoke '
          'mode and is compared against sealed smoke artifacts; G7 content, G10 and '
          'html spot-checks are skipped. The smoke key is a static mode description.',
    gates={
        'G1': '18 source sha8 file anchors asserted (3173 full+smoke artifacts, 3170 '
              'registry/gap ledgers, 3169/3171/3172 results, q03, rev3151b, agate)',
        'G2': 'shadow re-render: patch 3173 source L40 (count==1) to a gpt5_temp shadow '
              'OUTDIR, subprocess-run, compare - registry and gap ledger byte-identical; '
              'html line-diff where EVERY differing line must contain a timestamp '
              '(YYYY-MM-DD HH:MM) and line counts must match; data-k key sequences equal',
        'G3': 'registry v1.2: version, 22 features FTR-01..22 sequential, v1.1 '
              'features[0:21] byte-preserved (canonical json per index), failures=14 '
              'with F13/F14 tail, upgrade_log=3, every feature >=2 anchors with >=2 '
              'distinct srcs',
        'G4': 'three-track anchor resolution: (i) content-sha anchors (asserts '
              'res_sha8/seal_sha8) resolved to their source result documents by sha '
              'match, then EVERY other assert key navigated inside the doc (dotted '
              'path, digit-leaf list index, node:Nxx via nodes[id]) and value-compared; '
              '(ii) no-sha anchors resolved to the canonical doc of their src; '
              '(iii) GAP-4 anchor_sha8 file shas resolved by disk index; '
              'derived keys k0_drift_* whitelisted here and recomputed in G5',
        'G5': 'FTR-22 values re-derived exactly from p3172/p3169/p3171 sealed results '
              '(curve, port_frac, per-model k8 band, E_newent pooled, k0 drift) and '
              'FTR-22/FTR-21 anchor asserts cross-checked; FTR-21 p3169 gate-ratio '
              'float present; q03 pooled_mean re-asserted',
        'G6': 'evidence levels subset {E1,E2,E3}, E1/E3 sets as expected, E3 has '
              'intervention tag, E2 tag-completeness findings == expected 6-id list, '
              'FTR-03 premise correction recorded (already E2 with cross_model+held_out)',
        'G7': 'E1->E2 assessment for the 5 real E1 features: FTR-04/08 zero-GPU via '
              '3169 npz, FTR-06 zero-GPU unembed-only rebuild, FTR-14/15 need GPU',
        'G8': 'prereg draft artifact for Phase 3175 (G5-A12 structural-residual '
              'cross-arm) written into OUTDIR, schema-checked, sha8 recorded',
        'G9': 'gap ranking adjudication: 3 ranked items rendered into result',
        'G10': 'ledger n>=325 with phase 3173 entry (atlas_v13_closed)',
    },
    expected={
        'features': 22, 'failures': 14, 'upgrades': 3, 'html_fields_full': 637,
        'e1_ids': EXPECTED_E1, 'e3_ids': EXPECTED_E3,
        'taxonomy_notes': EXPECTED_TAXONOMY_NOTES,
    },
    immutability='sealed 3173/3170/3169/3171/3172 dirs read-only; shadow artifacts under '
                 'tests/gpt5_temp/p3174_shadow*; audit products only in phase3174/g5a11_audit',
)


# ------------------------------------------------------------------- freeze
def freeze():
    os.makedirs(OUTDIR, exist_ok=True)
    exep = os.path.join(OUTDIR, 'execution.json')
    body = dict(DESIGN)
    raw = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)
    d8 = sha8(raw.encode('utf-8'))
    body['design_sha8'] = d8
    raw2 = json.dumps(body, ensure_ascii=False, indent=1, sort_keys=True)
    if os.path.exists(exep):
        prev = jload(exep)
        if prev.get('design_sha8') != d8:
            raise SystemExit('DRIFT: execution.json design_sha8 %s != current %s'
                             % (prev.get('design_sha8'), d8))
        log('freeze: existing execution.json OK (%s)' % d8)
    else:
        with io.open(exep, 'w', encoding='utf-8') as f:
            f.write(raw2)
        log('freeze: execution.json written design_sha8=%s' % d8)
    return d8


# ------------------------------------------------------------------- G2 shadow
TS_RE = re.compile(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}')
DK_RE = re.compile(r'data-k="([^"]+)"')


def run_shadow(child_smoke):
    shadow_dir = os.path.join(GPT, 'p3174_shadow')
    if os.path.isdir(shadow_dir):
        shutil.rmtree(shadow_dir)
    src = io.open(P73SRC, encoding='utf-8').read()
    old = "OUTDIR = os.path.join(SRC_DIR, 'phase3173', 'g5a10_atlas_v13')"
    new = "OUTDIR = r'%s'" % shadow_dir
    assert src.count(old) == 1, ('patch target count', src.count(old))
    psrc = os.path.join(GPT, 'p3174_shadow_3173.py')
    with io.open(psrc, 'w', encoding='utf-8') as f:
        f.write(src.replace(old, new))
    env = dict(os.environ)
    env.pop('P3173_SMOKE', None)
    if child_smoke:
        env['P3173_SMOKE'] = '1'
    r = subprocess.run([sys.executable, psrc], cwd=ROOT, env=env,
                       capture_output=True, encoding='utf-8', errors='replace')
    conp = os.path.join(GPT, 'p3174_shadow_console.txt')
    with io.open(conp, 'w', encoding='utf-8') as f:
        f.write(r.stdout + '\n--- stderr ---\n' + r.stderr)
    assert r.returncode == 0, ('shadow run failed', r.returncode, r.stderr[-800:])
    assert 'result written' in r.stdout, 'shadow run did not reach seal'
    suf = 'smoke_' if child_smoke else ''
    return shadow_dir, suf


def cmp_shadow(shadow_dir, suf):
    out = {}
    # registry + gap: byte identity
    for tag, fn in (('registry', 'atlas_registry_v1_2.json'), ('gap', 'gap_ledger_v1_4.json')):
        a = open(os.path.join(SEALDIR, suf + fn), 'rb').read()
        b = open(os.path.join(shadow_dir, suf + fn), 'rb').read()
        out[tag + '_identical'] = (a == b)
        assert a == b, ('shadow not byte-identical', tag, len(a), len(b))
    # html: line-level diff with timestamp whitelist
    ha = io.open(os.path.join(SEALDIR, suf + 'atlas_v1_3.html'), encoding='utf-8').read()
    hb = io.open(os.path.join(shadow_dir, suf + 'atlas_v1_3.html'), encoding='utf-8').read()
    la, lb = ha.splitlines(), hb.splitlines()
    assert len(la) == len(lb), ('html line count', len(la), len(lb))
    diff_idx = [i for i in range(len(la)) if la[i] != lb[i]]
    bad = [i for i in diff_idx if not (TS_RE.search(la[i]) and TS_RE.search(lb[i]))]
    assert not bad, ('non-timestamp html diff at lines', bad[:5],
                     [(la[i][:120], lb[i][:120]) for i in bad[:3]])
    ka, kb = DK_RE.findall(ha), DK_RE.findall(hb)
    assert ka == kb, 'data-k key sequences differ'
    out['html_lines'] = len(la)
    out['html_diff_lines'] = len(diff_idx)
    out['html_diff_all_timestamp'] = True
    out['datak_sealed'] = len(ka)
    out['datak_shadow'] = len(kb)
    out['keys_identical'] = True
    if not SMOKE:
        assert len(ka) == HTML_FIELDS_FULL, ('datak count', len(ka))
    return out


# ------------------------------------------------------------------- G4 anchor resolution
SRC_ROOTS = {
    'p3151': [os.path.join(SRC_DIR, 'phase3151')],
    'p3154': [os.path.join(SRC_DIR, 'phase3154')],
    'p3155': [os.path.join(SRC_DIR, 'phase3155')],
    'p3156': [os.path.join(SRC_DIR, 'phase3156')],
    'p3157': [os.path.join(SRC_DIR, 'phase3157')],
    'p3158': [os.path.join(SRC_DIR, 'phase3158')],
    'p3159': [os.path.join(SRC_DIR, 'phase3159')],
    'p3160': [os.path.join(SRC_DIR, 'phase3160')],
    'p3161': [os.path.join(SRC_DIR, 'phase3161')],
    'p3162': [os.path.join(SRC_DIR, 'phase3162')],
    'p3163': [os.path.join(SRC_DIR, 'phase3163')],
    'p3164a': [os.path.join(SRC_DIR, 'phase3164')],
    'p3164b': [os.path.join(SRC_DIR, 'phase3164')],
    'p3164c': [os.path.join(SRC_DIR, 'phase3164')],
    'p3165': [os.path.join(SRC_DIR, 'phase3165')],
    'p3166': [os.path.join(SRC_DIR, 'phase3166')],
    'p3169': [os.path.join(SRC_DIR, 'phase3169')],
    'p3171': [os.path.join(SRC_DIR, 'phase3171')],
    'p3172': [os.path.join(SRC_DIR, 'phase3172')],
    'q03': [os.path.join(ROOT, 'tests', 'deepseek', 'result')],
    'q05': [os.path.join(ROOT, 'tests', 'deepseek', 'result')],
    'q06': [os.path.join(ROOT, 'tests', 'deepseek', 'result')],
    'agate': [os.path.join(ROOT, 'research', 'deepseek', 'atlas')],
}
# derived assert keys computed at registry build time from TWO sources; recomputed in G5
DERIVED_KEYS = {'k0_drift_ratio', 'k0_drift_E_oov', 'k0_drift_E_seen'}
CANON_SPECIAL = {
    'p3162': os.path.join(SRC_DIR, 'phase3162', 'g5a1_atlas_foundation', 'result_audit.json'),
    'agate': SHA_PATHS['agate'],
    'q03': SHA_PATHS['q03'],
    'q05': os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q05_result.json'),
    'q06': os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q06_result.json'),
}
_CAND_CACHE = {}


def load_candidates(src):
    if src in _CAND_CACHE:
        return _CAND_CACHE[src]
    docs = []
    for root in SRC_ROOTS[src]:
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith('.json'):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(p) > 5 * 1024 * 1024:
                        continue
                    d = json.load(io.open(p, encoding='utf-8'))
                except Exception:
                    continue
                if isinstance(d, dict):
                    docs.append((os.path.relpath(p, ROOT), d))
    _CAND_CACHE[src] = docs
    return docs


def resolve_doc(src, res8, seal8):
    for path, d in load_candidates(src):
        if res8 and d.get('res_sha8') != res8:
            continue
        if seal8 and d.get('seal_sha8') != seal8:
            continue
        if not res8 and not seal8:
            continue
        return path, d
    return None, None


def nav(doc, key):
    parts = key.split('.')
    cur = doc
    for i, pt in enumerate(parts):
        if isinstance(cur, dict):
            if pt in cur:
                cur = cur[pt]
                continue
            if pt.startswith('node:') and isinstance(cur.get('nodes'), list):
                nid = pt.split(':', 1)[1]
                hit = [it for it in cur['nodes'] if isinstance(it, dict) and it.get('id') == nid]
                if hit:
                    cur = hit[0]
                    continue
            return False, None
        if isinstance(cur, list) and pt.isdigit() and int(pt) < len(cur):
            cur = cur[int(pt)]
            continue
        return False, None
    return True, cur


def val_eq(actual, expect):
    if isinstance(expect, bool) or isinstance(actual, bool):
        return actual == expect
    if isinstance(expect, float):
        return (isinstance(actual, (int, float))
                and abs(float(actual) - expect) <= 1e-9 * max(1.0, abs(expect)))
    return actual == expect


# ------------------------------------------------------------------- G4b file-sha index
def build_index():
    idx = {}
    n_files = 0
    for dirpath, dirnames, filenames in os.walk(SRC_DIR):
        rel = os.path.relpath(dirpath, SRC_DIR)
        top = rel.split(os.sep)[0]
        if not top.startswith('phase3'):
            continue
        for fn in filenames:
            p = os.path.join(dirpath, fn)
            idx.setdefault(sha8_file(p), []).append(os.path.relpath(p, SRC_DIR))
            n_files += 1
    for key in ('q03', 'agate'):
        idx.setdefault(sha8_file(SHA_PATHS[key]), []).append(SHA_PATHS[key])
    return idx, n_files


def main():
    log('== Phase 3174 g5a11_audit %s ==' % ('SMOKE' if SMOKE else 'FULL'))
    d8 = freeze()
    n_checks = 0

    # ---------------- G1 source anchors
    for k, v in SHA_ANCHOR.items():
        assert sha8_file(SHA_PATHS[k]) == v, ('G1 sha mismatch', k, v)
        n_checks += 1
    log('G1 %d source sha8 anchors OK' % len(SHA_ANCHOR))

    # load sealed sources
    reg12 = jload(SHA_PATHS['reg12'])
    reg11 = jload(SHA_PATHS['reg11'])
    gl14 = jload(SHA_PATHS['gl14'])
    r69 = jload(SHA_PATHS['r69'])
    r71 = jload(SHA_PATHS['r71'])
    r72 = jload(SHA_PATHS['r72'])
    q03 = jload(SHA_PATHS['q03'])
    feats = reg12['features']

    # content-level seals of measurement sources (3173 recipe verbatim)
    assert r69['res_sha8'] == '49430a39' and r69['seal_sha8'] == '018c6024'
    assert r71['res_sha8'] == '6a29201c' and r71['seal_sha8'] == 'cc7ccedd'
    assert r72['res_sha8'] == '463f42c8' and r72['seal_sha8'] == 'cdb85525'
    assert r71['verdict'] == ('g5a8_collapse_mechanism|full|3_models|'
                              'ratio_S_main=0.7511/0.7167/0.8205|mixed_across_models')
    assert r72['verdict'] == ('g5a9_port_calibration|full|3_models|ratio_k8=1.8002|'
                              'borderline_partial_recovery')
    assert q03['summary']['pooled_mean'] == 0.37335047125816345
    n_checks += 6

    # ---------------- G2 shadow re-render
    shadow_dir, suf = run_shadow(child_smoke=SMOKE)
    shadow = cmp_shadow(shadow_dir, suf)
    n_checks += 6
    log('G2 shadow re-render OK (child=%s): registry/gap byte-identical, html diff '
        'lines=%d all timestamp, datak %d==%d'
        % ('smoke' if SMOKE else 'full', shadow['html_diff_lines'],
           shadow['datak_sealed'], shadow['datak_shadow']))

    # ---------------- G3 registry invariants
    assert reg12['version'] == '1.2'
    assert len(feats) == 22
    assert [f['id'] for f in feats] == ['FTR-%02d' % i for i in range(1, 23)]
    for i in range(21):
        a = json.dumps(feats[i], ensure_ascii=False, sort_keys=True)
        b = json.dumps(reg11['features'][i], ensure_ascii=False, sort_keys=True)
        assert a == b, ('G3 v1.1 feature not preserved', feats[i]['id'])
    fails = reg12['failures']
    assert len(fails) == 14 and fails[12]['id'] == 'F13' and fails[13]['id'] == 'F14'
    assert len(reg12['upgrade_log']) == 3
    for f in feats:
        assert len(f['anchors']) >= 2, ('G3 anchors', f['id'])
        assert len(set(str(a.get('src')) for a in f['anchors'])) >= 2, ('G3 srcs', f['id'])
    n_checks += 22 + 22 + 22
    log('G3 registry v1.2 invariants OK (22 ftr, v1.1 preserved 21/21, failures 14, upgrades 3)')

    # ---------------- G4 anchor resolution (three tracks)
    resolved_cache = {}
    content_sha_anchors = 0
    nav_n = 0
    derived_n = 0
    resolved_paths = []
    for f in feats:
        for a in f['anchors']:
            src = a.get('src')
            asr = a.get('asserts', {})
            res8 = asr.get('res_sha8')
            seal8 = asr.get('seal_sha8')
            if res8 or seal8:
                path, doc = resolve_doc(src, res8, seal8)
                assert path is not None, ('G4 unresolved content sha', f['id'], src, res8, seal8)
                content_sha_anchors += 1
                resolved_cache.setdefault(src, (path, doc))
            elif src in resolved_cache:
                path, doc = resolved_cache[src]
            elif src in CANON_SPECIAL:
                path = CANON_SPECIAL[src]
                doc = jload(path)
            else:
                raise AssertionError(('G4 no-sha src without canonical doc', f['id'], src))
            resolved_paths.append({'ftr': f['id'], 'src': src, 'doc': path})
            for k, v in asr.items():
                if k in ('res_sha8', 'seal_sha8'):
                    continue
                if k in DERIVED_KEYS:
                    derived_n += 1
                    continue
                ok, actual = nav(doc, k)
                assert ok and val_eq(actual, v), ('G4 nav', f['id'], src, k, actual, v)
                nav_n += 1
    # GAP-4 anchor_sha8 = file shas -> disk index
    idx, n_files = build_index()
    g4anchors = gl14['gaps'][3]['anchor_sha8'] if gl14['gaps'][3]['id'] == 'GAP-4' else None
    assert g4anchors is not None, 'GAP-4 not at index 3'
    for src, h in sorted(g4anchors.items()):
        assert h in idx, ('G4 unresolved GAP-4 file sha', src, h)
    n_checks += content_sha_anchors + nav_n + len(g4anchors)
    log('G4 anchors: %d content-sha resolved, %d asserts navigated (0 fail), '
        '%d derived keys -> G5, GAP-4 %d file shas resolved (index %d files)'
        % (content_sha_anchors, nav_n, derived_n, len(g4anchors), n_files))

    # ---------------- G5 numeric re-derivation
    KS = ['0', '1', '2', '4', '8']
    rk = {k: r72['pooled'][k]['ratio'] for k in KS}
    assert rk['0'] == r69['gate']['ratio_B']
    assert rk['0'] == 2.5388114997805062
    assert rk['8'] == 1.800176217907229
    seq = [rk[k] for k in KS]
    assert all(seq[i] >= seq[i + 1] for i in range(4)), ('G5 monotone', seq)
    port = (rk['0'] - rk['8']) / rk['0']
    assert 0.28 <= port <= 0.30, ('G5 port_frac', port)
    MS = feats[21]['model_scope']
    assert MS == ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
    r8m = {m: r72['per_model'][m]['kcurves']['8']['ratio'] for m in MS}
    for m in MS:
        assert 1.5 <= r8m[m] < 2.0, ('G5 k8 band', m, r8m[m])
    en0 = sum(r72['per_model'][m]['kcurves']['0']['E_newent'] for m in MS) / 3.0
    en8 = sum(r72['per_model'][m]['kcurves']['8']['E_newent'] for m in MS) / 3.0
    assert 0.73 <= en0 <= 0.75 and 0.52 <= en8 <= 0.54, ('G5 E_newent', en0, en8)
    rs = {m: r71['per_model'][m]['slots']['k_main']['ratio_S'] for m in MS}
    assert rs['qwen3-4b'] < 0.8 and rs['qwen3-14b'] < 0.8 and rs['glm4-9b'] > 0.8
    assert r71['overall']['main_cls'] == 'mixed_across_models'
    f22 = feats[21]
    assert f22['id'] == 'FTR-22' and f22['evidence_level'] == 'E2_predictive'
    V = f22['values']
    for k in KS:
        assert V['ratio_k' + k] == rk[k], ('G5 V.ratio', k)
    assert V['port_removed_frac'] == port
    for m in MS:
        assert V['ratio_k8_' + m.replace('-', '_')] == r8m[m], ('G5 V.k8', m)
        assert V['p3171_ratio_S_' + m.replace('-', '_')] == rs[m], ('G5 V.rs', m)
    assert V['E_newent_pooled_mean3_k0'] == en0 and V['E_newent_pooled_mean3_k8'] == en8
    assert V['k0_drift_ratio_vs_3169'] == abs(rk['0'] - r69['gate']['ratio_B'])
    for a in f22['anchors']:
        asr = a['asserts']
        for k in KS:
            if 'pooled.%s.ratio' % k in asr:
                assert asr['pooled.%s.ratio' % k] == rk[k]
        if 'per_model.qwen3-4b.kcurves.8.ratio' in asr:
            for m in MS:
                assert asr['per_model.%s.kcurves.8.ratio' % m] == r8m[m]
        if 'per_model.qwen3-4b.slots.k_main.ratio_S' in asr:
            for m in MS:
                assert asr['per_model.%s.slots.k_main.ratio_S' % m] == rs[m]
    assert '1.8002' in f22['statement'] and '29.1' in f22['statement']
    f21 = feats[20]
    assert f21['id'] == 'FTR-21' and f21['evidence_level'] == 'E2_predictive'
    hit21 = 0
    for a in f21['anchors']:
        for v in a.get('asserts', {}).values():
            if isinstance(v, float) and abs(v - r69['gate']['ratio_B']) < 1e-9:
                hit21 += 1
    assert hit21 >= 1, ('G5 FTR-21 gate ratio not found in asserts', hit21)
    # FTR-22 p3169 derived assert keys (whitelisted in G4) recomputed here
    a69 = [a for a in f22['anchors'] if a.get('src') == 'p3169'][0]
    assert a69['asserts']['k0_drift_ratio'] == abs(rk['0'] - r69['gate']['ratio_B'])
    assert a69['asserts']['k0_drift_E_oov'] == abs(r72['pooled']['0']['E_oov']
                                                   - r69['gate']['pooled_E_oov_B'])
    assert a69['asserts']['k0_drift_E_seen'] == abs(r72['pooled']['0']['E_seen']
                                                    - r69['gate']['pooled_E_seen_B'])
    n_checks += 5 + 5 + 3 + 12 + 2 + 3 + 1 + 3
    log('G5 re-derivation OK: FTR-22 curve/port/k8/E_newent/p3171 exact; FTR-21 gate ratio present')

    # ---------------- G6 taxonomy audit
    levels = sorted(set(f['evidence_level'] for f in feats))
    assert levels == ['E1_repeatable', 'E2_predictive', 'E3_causal_scoped']
    e1 = [f['id'] for f in feats if f['evidence_level'] == 'E1_repeatable']
    e3 = [f['id'] for f in feats if f['evidence_level'] == 'E3_causal_scoped']
    assert e1 == EXPECTED_E1 and e3 == EXPECTED_E3, ('G6 level sets', e1, e3)
    for f in feats:
        if f['evidence_level'] == 'E3_causal_scoped':
            assert any('intervention' in a.get('tags', []) for a in f['anchors']), f['id']
    notes = []
    for f in feats:
        if f['evidence_level'] != 'E2_predictive':
            continue
        tags = set(t for a in f['anchors'] for t in a.get('tags', []))
        if not ('cross_model' in tags and 'held_out' in tags):
            notes.append(f['id'])
    assert sorted(notes) == EXPECTED_TAXONOMY_NOTES, ('G6 notes', sorted(notes))
    ftr03 = feats[2]
    assert ftr03['id'] == 'FTR-03'
    assert ftr03['evidence_level'] == 'E2_predictive'
    t3 = set(t for a in ftr03['anchors'] for t in a.get('tags', []))
    assert 'cross_model' in t3 and 'held_out' in t3
    premise_correction = {
        'claim_in_prereg_3173': 'FTR-03 E1->E2 upgrade path evaluation',
        'registry_v12_reality': ('FTR-03 already E2_predictive: anchors p3154 '
                                 '(held_out,cross_model) + p3166 (held_out,cross_model), '
                                 'LOEO AUC 0.856/0.863/0.781'),
        'action': ('prereg premise corrected, no registry change needed; E1->E2 '
                   'evaluation rerouted to the actual E1 set FTR-04/06/08/14/15'),
    }
    n_checks += 4 + 6 + 2
    log('G6 taxonomy OK: E1=%s E3=%s, %d tag notes, FTR-03 premise corrected (already E2)'
        % (','.join(e1), ','.join(e3), len(notes)))

    # ---------------- G7 E1->E2 assessment
    E1_ASSESS = [
        {'id': 'FTR-04', 'family': 'S', 'missing': 'held_out predictive anchor (cross_model already x2)',
         'zero_gpu_candidate': True,
         'path': 'held-out class-axis readout on 3169 collect npz seen-class rows '
                 '(LOEO / new-entity rows already on disk); protocol freeze then run',
         'basis': '3169 collect npz sealed 4b 36eb4ff0 / 14b 97f98575 / glm4 9cc3f8b4'},
        {'id': 'FTR-06', 'family': 'S', 'missing': 'cross_model rebuild of S_attr/S_syntax '
                                                   '(vocab bound to 4b tokenizer face)',
         'zero_gpu_candidate': True,
         'path': 'unembed-only rebuild per model with per-model single-token word lists '
                 '(3166 S_class recipe verbatim); no GPU forward needed',
         'basis': '3166 S_class_rebuild ok(n=10 words=100) x3 via W_U rows'},
        {'id': 'FTR-08', 'family': 'RxK', 'missing': 'held_out predictive anchor',
         'zero_gpu_candidate': True,
         'path': 'R_logic x K_entity angle on 3169 new-entity rows (E_newent arm) = '
                 'held-out entity generalization, zero GPU',
         'basis': '3169 npz E_newent rows on disk; 3166 census recipe'},
        {'id': 'FTR-14', 'family': 'context', 'missing': 'held_out (new relations/themes) '
                                                         'predictive anchor; p3162 audit tag is not predictive',
         'zero_gpu_candidate': False,
         'path': 'needs GPU new-relation generalization run (3157 protocol extension)',
         'basis': '-'},
        {'id': 'FTR-15', 'family': 'mech', 'missing': 'held_out predictive anchor beyond '
                                                      '3158 equivalence panel',
         'zero_gpu_candidate': False,
         'path': 'needs GPU new-input equivalence-class protocol',
         'basis': '-'},
    ]
    assert [a['id'] for a in E1_ASSESS] == EXPECTED_E1
    n_zg = sum(1 for a in E1_ASSESS if a['zero_gpu_candidate'])
    assert n_zg == 3
    n_checks += 2
    log('G7 E1->E2 assessment OK: 3 zero-GPU candidates (FTR-04/06/08), 2 need GPU (FTR-14/15)')

    # ---------------- G8 prereg draft artifact (Phase 3175)
    prereg = {
        'prereg_id': 'G5-A12',
        'target_phase': 3175,
        'status': 'draft_pending_freeze',
        'title': '谱外类结构残留定位——校准实体面板内/面板外交叉臂',
        'created_draft': '2026-10-09',
        'drafted_in': 'Phase 3174 G5-A11',
        'motivation': ('3172 定判 borderline_partial_recovery：端口校准 k=8 仅移除 29.1% 崩塌量，'
                       '残留 ~1.8x 过量误差为结构性成分（F13）。候选解释：H1 实体熟悉度'
                       '（3172 校准实体全部来自冻结谱外词表，对模型而言可能比面板测试实体更陌生，'
                       '恢复受限）；H2 类端口残差（one-hot 端口机制只能部分学习，与校准实体来源无关）。'),
        'hypotheses': {
            'H1_entity_familiarity': '校准实体来自面板词表内时恢复更深（ratio_in < ratio_out）',
            'H2_port_residual': '两臂恢复曲线形状一致，差异 |delta|<=0.15',
        },
        'arms': {
            'in_panel': '校准实体取自 3169 已采集面板词表（复用 collect npz 行，零新 forward）',
            'out_panel': '校准实体取自面板词表外（新词表，需 GPU forward 采集 H 行）',
            'design': '每谱外类 2k 个校准实体（面板内/外各半），k in {0,1,2,4,8}；'
                      '校准与测试实体不相交（3172 verbatim）',
        },
        'protocol': 'Phase 3172 verbatim: Q03 ridge、口径 B、E_oov/E_seen/E_newent 同口径；'
                    '三模型 qwen3-4b / qwen3-14b / glm4-9b；RandomState 种子在 3175 freeze 时冻结',
        'primary_gate': {
            'statistic': 'delta = pooled ratio_out(k=8) - pooled ratio_in(k=8)',
            'delta_le_0.15': 'port_residual_dominant（H2 支持，mechanism_note 定稿）',
            'delta_gt_0.15': 'entity_familiarity_component_confirmed（H1 支持）',
            'delta_lt_minus_0.15': 'anomaly_register（面板外反而恢复更好——登记异常并复核装置）',
        },
        'secondary': ['两臂 k 梯度形状（饱和点）对比', 'E_newent 监控（非类特异改善）',
                      'per-class 一致性（4 类同向）'],
        'immutable_predicates': [
            'k=0 臂必须复现 3169 封存 gate.ratio_B=2.5388114997805062（drift<1e-9）',
            'in-panel 臂复用 3169 collect npz 行（零新 forward）；out-panel 臂词表在 3175 freeze '
            '时冻结且先于任何 GPU forward（预注册先于观测纪律）',
        ],
        'tbd_frozen_at_execution': ['每类实体池大小', '面板外精确词表', 'RandomState 种子',
                                    'GPU 采集批处理'],
        'models': ['qwen3-4b', 'qwen3-14b', 'glm4-9b'],
        'gpu': True,
        'evidence_level_target': 'E2（若 delta 判定支持 H2 且与 3171/3172 三角一致，可评估 mechanism_note 定稿；'
                                 'E3 需端口级干预升级，另行预注册）',
    }
    suf4 = 'smoke_' if SMOKE else ''
    pr_path = os.path.join(OUTDIR, suf4 + 'prereg_3175_structural_residual_draft.json')
    with io.open(pr_path, 'w', encoding='utf-8') as f:
        json.dump(prereg, f, ensure_ascii=False, indent=1)
    pr_sha = sha8_file(pr_path)
    for k in ('prereg_id', 'target_phase', 'status', 'hypotheses', 'arms', 'protocol',
              'primary_gate', 'immutable_predicates', 'tbd_frozen_at_execution', 'models'):
        assert k in prereg, ('G8 schema', k)
    n_checks += 1
    log('G8 prereg 3175 draft written (%s, sha8 %s)' % (os.path.basename(pr_path), pr_sha))

    # ---------------- G9 gap ranking adjudication
    ranking = [
        {'rank': 1, 'item': 'structural_residual_localization',
         'vehicle': 'Phase 3175 G5-A12 GPU cross-arm (prereg draft artifact registered this phase)',
         'why': 'GAP-4 mechanism completion: separates entity-familiarity vs class-port '
                'residual (~1.8x); mechanism_note finalization path; E2 with E3 potential',
         'gpu': True},
        {'rank': 2, 'item': 'e1_e2_batch_upgrade',
         'vehicle': 'zero-GPU phase on 3169/3166 npz: FTR-04 held-out class-axis readout, '
                    'FTR-08 held-out entity probe, FTR-06 unembed-only rebuild',
         'why': 'cheapest evidence-level gains (3 features E1->E2 candidates); prereg '
                'premise FTR-03 corrected (already E2, no action)',
         'gpu': False},
        {'rank': 3, 'item': 'n_line_ledger_backfill',
         'vehicle': 'cross-line administrative patch (N 线 P3-P7 补 atlas_ledger)',
         'why': '挂账清偿，无新测量，低风险', 'gpu': False},
    ]
    assert [r['rank'] for r in ranking] == [1, 2, 3]
    n_checks += 1
    log('G9 ranking: 1 structural-residual GPU arm / 2 E1->E2 zero-GPU batch / 3 N-line ledger backfill')

    # ---------------- G10 ledger state + html spot checks (FULL only)
    if not SMOKE:
        led = jload(LEDGER)
        ms_ = led['measurements']
        e73 = [m for m in ms_ if m.get('phase') == 3173]
        assert len(ms_) >= 325 and len(e73) == 1
        assert 'atlas_v13_closed' in e73[0]['verdict']
        html_sealed = io.open(SHA_PATHS['html13'], encoding='utf-8').read()
        for i in range(1, 23):
            assert ('FTR-%02d' % i) in html_sealed, ('html missing', i)
        for g in ('GAP-1', 'GAP-2', 'GAP-3', 'GAP-4'):
            assert g in html_sealed
        for t in ('F13', 'F14', 'mechanism_note', 'quantified_collapse'):
            assert t in html_sealed
        n_checks += 1 + 22 + 4 + 4
        log('G10/G-html: ledger n=%d has 3173; sealed html spot checks OK' % len(ms_))

    # ---------------- result + seal
    verdict = ('g5a11_audit|shadow_reg_gl_identical|html_diff_lines_%d_all_timestamp|'
               'anchors_%d_sha_%d_nav_resolved|taxonomy_notes_6|'
               'ftr03_premise_corrected_already_E2|e1_zerogpu_candidates_3|'
               'prereg3175_%s|ranking_3_items|PASS') % (
        shadow['html_diff_lines'], content_sha_anchors, nav_n, pr_sha)
    summary = {
        'phase': 3174, 'name': 'g5a11_audit', 'smoke': SMOKE,
        'design_sha8': d8, 'verdict': verdict,
        'prereg_file': os.path.basename(pr_path), 'prereg_sha8': pr_sha,
        'shadow': shadow,
        'anchors': {'content_sha_anchors': content_sha_anchors, 'nav_asserts': nav_n,
                    'derived_keys_recomputed_in_g5': derived_n,
                    'gap4_file_shas': len(g4anchors), 'index_files': n_files,
                    'resolved_docs': resolved_paths},
        'taxonomy': {'e1_ids': e1, 'e3_ids': e3, 'tag_notes': sorted(notes),
                     'premise_correction': premise_correction},
        'e1_assessment': E1_ASSESS,
        'ranking': ranking,
        'html_spot': None if SMOKE else {'ftr_ids': 22, 'gap_ids': 4, 'extras': 4,
                                         'datak': HTML_FIELDS_FULL},
        'sources': dict(SHA_ANCHOR),
    }
    raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
    res8 = sha8(raw.encode('utf-8'))
    mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
    seal8 = sha8(mid.encode('utf-8'))
    rpath = os.path.join(OUTDIR, suf4 + 'result.json')
    with io.open(rpath, 'w', encoding='utf-8') as f:
        f.write(json.dumps(dict(summary, res_sha8=res8, seal_sha8=seal8),
                           ensure_ascii=False, indent=1, sort_keys=True))
    log('result written res=%s seal=%s' % (res8, seal8))
    log('verdict: %s' % verdict)
    log('checks total: %d' % n_checks)
    with io.open(os.path.join(OUTDIR, suf4 + 'run_log.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(LOG) + '\n')


if __name__ == '__main__':
    main()
