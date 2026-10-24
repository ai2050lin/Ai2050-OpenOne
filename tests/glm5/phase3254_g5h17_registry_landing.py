# -*- coding: utf-8 -*-
"""Phase 3254 G5-H17: registry/gap append LANDING (executes the 3253 previewed
diff) + post-landing byte audit.

Prereg (frozen before any observation): MEMO Phase 3253 tail arms.
  (a) LANDING ARM (zero GPU, doc): construct and land
        gap_ledger_v1_11.json (5934631b)  -> v1_12  : appendix.append(APPX-6
             verbatim per 6-key schema {id,title,anchor,status,note,
             anchor_sha8}; version 1.11->1.12; updated -> 2026-10-20;
             supersedes -> v1_11 pointer; gaps GAP-1..5 + APPX-1..5 bytes
             untouched)
        atlas_registry_v1_33.json (a30215fd) -> v1_34: provenance.anchor_files
             adds p3246/p3250/p3252/p3253 (result.json FILE sha8, the
             convention verified against a1_behavioral_band ac5d4d74);
             upgrade_log.append(1 entry); version 1.33->1.34; supersedes ->
             v1_33 pointer; features/failures/taxonomy/principles/census
             bytes untouched.
      APPEND-ONLY BYTE DISCIPLINE: both new files serialize with the exact
      writer of their predecessors (json.dumps ensure_ascii=False indent=1 --
      round-trip byte-exact on both sealed files); REVERT CHECK: reverting
      the planned edits on the new objects reproduces the sealed predecessor
      bytes bit-for-bit; pointer diff == the planned pointer set EXACTLY.
  (b) POST-LANDING AUDIT: reload both landed files from disk; pointer diff
      vs sealed predecessors == planned set; three-ledger consistency
      (gap v1.12 APPX-6 anchor_sha8 == 9d0f1825 as frozen in ledger #404;
      registry v1.34 anchor p3253 file-sha == sealed 3253 result file sha;
      sealed v1.11/v1.33 byte-unchanged after landing).
  (c) 5 appendix GPU items stay deferred (APPX-1..5 remain open/open_crossline).

SMOKE: P3254_SMOKE=1 builds both planned blobs, runs ALL diff/revert/audit
assertions, writes smoke artifacts ONLY (no v1_12/v1_34 files).
"""
import json
import os
import time
import hashlib
import copy

T0 = time.time()
NAME = 'g5h17_registry_landing'
SMOKE = os.environ.get('P3254_SMOKE') == '1'
SRC_DIR = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTDIR = os.path.join(SRC_DIR, 'phase3254', NAME)
LOG = []
TODAY = '2026-10-20'

SRC = {
    'gap_v111': os.path.join(SRC_DIR, r'phase3237\g5g31_registry_landing\gap_ledger_v1_11.json'),
    'registry_v133': os.path.join(SRC_DIR, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json'),
    'appx6_draft': os.path.join(SRC_DIR, r'phase3252\g5h15_ustar_layer_trace\gap_appendix_draft_v1_12.json'),
    'p3246_result': os.path.join(SRC_DIR, r'phase3246\g5h9_regime_boundary\result.json'),
    'p3250_result': os.path.join(SRC_DIR, r'phase3250\g5h13_kernel_boundary\result.json'),
    'p3252_result': os.path.join(SRC_DIR, r'phase3252\g5h15_ustar_layer_trace\result.json'),
    'p3253_result': os.path.join(SRC_DIR, r'phase3253\g5h16_ustar_subspace\result.json'),
    'p3253_preview': os.path.join(SRC_DIR, r'phase3253\g5h16_ustar_subspace\registry_append_preview_v1_34.json'),
}
SHA_ANCHOR = {
    'gap_v111': '5934631b', 'registry_v133': 'a30215fd', 'appx6_draft': 'c3de8895',
    'p3246_result': '41b42d9b', 'p3250_result': 'f989ff42',
    'p3252_result': 'd45c08e9', 'p3253_result': 'dc2ccd4a',
}
ANCHORS_NEW = ['p3246', 'p3250', 'p3252', 'p3253']

DESIGN = {
    'phase': '3254', 'name': NAME,
    'prereg': 'AGI_GPT5_MEMO Phase 3253 tail arms (a)+(b), frozen before any '
              'observation; executes the 3253 previewed diff '
              '(registry_append_preview_v1_34.json, append_trigger_MET)',
    'landing_gap': 'gap_ledger_v1_11 (5934631b) -> v1_12: appendix.append('
                   'APPX-6 verbatim, 6-key schema per gap ledger v1.11 entry '
                   'schema); version 1.11->1.12; updated -> ' + TODAY + '; '
                   'supersedes -> "gap_ledger_v1_11.json (5934631b, Phase '
                   '3237)"; gaps + APPX-1..5 bytes untouched',
    'landing_registry': 'atlas_registry_v1_33 (a30215fd) -> v1_34: '
                        'provenance.anchor_files adds p3246/p3250/p3252/p3253 '
                        '(result.json FILE sha8); upgrade_log.append(1); '
                        'version 1.33->1.34; supersedes -> "atlas_registry_'
                        'v1_33.json (a30215fd, Phase 3238)"; features/'
                        'failures/taxonomy/principles/census_3238 untouched',
    'byte_discipline': 'writer = json.dumps(ensure_ascii=False, indent=1) '
                       '(round-trip byte-exact on both sealed predecessors); '
                       'REVERT CHECK: reverting planned edits reproduces the '
                       'sealed predecessor bytes bitwise; pointer diff == '
                       'planned pointer set EXACTLY (no other change)',
    'audit_arm': '(b) post-landing audit: disk reload; pointer diff == '
                 'planned set; three-ledger consistency (APPX-6 anchor_sha8 '
                 '== 9d0f1825 per ledger #404 freeze; registry anchor '
                 'p3253 file-sha == sealed 3253 result file sha; sealed '
                 'v1.11/v1.33 byte-unchanged)',
    'gpu_arm': '(c) 5 appendix GPU items stay deferred',
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
        json.dump({'phase': 3254, 'name': NAME, 'design_sha8': dsha[:8],
                   'design': DESIGN, 'frozen_before': 'any observation',
                   'created': time.strftime('%Y-%m-%d %H:%M:%S')},
                  open(exe_p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        log('freeze: execution.json written (%s)' % dsha[:8])
    return dsha[:8]


def ptr_diff(a, b, path='$'):
    """Exact recursive pointer diff; returns list of changed pointers."""
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        out = []
        for k in a:
            if k not in b:
                out.append(path + '.' + k + ' (REMOVED)')
            else:
                out += ptr_diff(a[k], b[k], path + '.' + k)
        for k in b:
            if k not in a:
                out.append(path + '.' + k + ' (ADDED)')
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [path + ' (LEN %d->%d)' % (len(a), len(b))]
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += ptr_diff(x, y, path + '[%d]' % i)
        return out
    return [] if a == b else [path + ' (%r -> %r)' % (a, b)]


def build_gap_v112(gap111, draft):
    entry_full = draft['entry']
    assert entry_full['id'] == 'APPX-6'
    # land per 6-key schema, verbatim field values from the frozen draft
    schema_keys = ['id', 'title', 'anchor', 'status', 'note', 'anchor_sha8']
    assert sorted(schema_keys) == sorted(gap111['appendix'][0].keys())
    entry = {k: entry_full[k] for k in schema_keys}
    new = copy.deepcopy(gap111)
    new['appendix'].append(entry)
    new['version'] = '1.12'
    new['updated'] = TODAY
    new['supersedes'] = 'gap_ledger_v1_11.json (5934631b, Phase 3237)'
    planned = {'$.appendix (LEN 5->6)', '$.version', '$.updated', '$.supersedes'}
    return new, entry, planned


def build_registry_v134(reg133):
    new = copy.deepcopy(reg133)
    for key in ANCHORS_NEW:
        p = SRC['%s_result' % key]
        new['provenance']['anchor_files'][key] = {
            'path': os.path.relpath(p, ROOT), 'sha8': sha8_file(p)}
    new['upgrade_log'].append({
        'node': 'provenance.anchor_files',
        'change': 'append landing: 4 anchors added (p3246 regime boundary, '
                  'p3250 kernel boundary/incumbent XC_p5_b8, p3252 ustar '
                  'layer trace, p3253 ustar subspace mechanism); GAP ledger '
                  'companion append lands APPX-6 (fidelity-closure chain '
                  '3239-3250 + ustar lexical decoupling); features/failures '
                  'UNCHANGED (no E-level upgrade claimed by 3239-3253 -- all '
                  'characterization)',
        'reason': 'G5-H17 (Phase 3254): preregistered landing arm (MEMO 3253 '
                  'tail a); executes registry_append_preview_v1_34.json '
                  '(append_trigger_MET, 3253 arm (b)); append-only byte '
                  'discipline with revert check'})
    new['version'] = '1.34'
    new['supersedes'] = 'atlas_registry_v1_33.json (a30215fd, Phase 3238)'
    planned = set()
    for key in ANCHORS_NEW:
        planned.add('$.provenance.anchor_files.' + key + ' (ADDED)')
    planned.add('$.upgrade_log (LEN 12->13)')
    planned.add('$.version')
    planned.add('$.supersedes')
    return new, planned


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    dsha = freeze()
    log('SMOKE=%s' % SMOKE)

    for k, v in SHA_ANCHOR.items():
        actual = sha8_file(SRC[k])
        assert actual == v, ('src sha anchor', k, actual, v)
    log('src sha anchors OK (%d files, sealed v1.11/v1.33 included)' % len(SHA_ANCHOR))

    gap111_raw = open(SRC['gap_v111'], 'rb').read()
    reg133_raw = open(SRC['registry_v133'], 'rb').read()
    gap111 = json.loads(gap111_raw.decode('utf-8'))
    reg133 = json.loads(reg133_raw.decode('utf-8'))
    draft = json.load(open(SRC['appx6_draft'], encoding='utf-8'))
    preview = json.load(open(SRC['p3253_preview'], encoding='utf-8'))
    assert preview['trigger_status'] == 'append_trigger_MET'
    log('3253 preview loaded: append_trigger_MET confirmed (executing)')

    # ---- build planned versions
    gap112, appx6_entry, gap_planned = build_gap_v112(gap111, draft)
    reg134, reg_planned = build_registry_v134(reg133)
    log('built: gap v1.12 (appendix %d -> %d) + registry v1.34 '
        '(anchor_files %d -> %d, upgrade_log %d -> %d)'
        % (len(gap111['appendix']), len(gap112['appendix']),
           len(reg133['provenance']['anchor_files']),
           len(reg134['provenance']['anchor_files']),
           len(reg133['upgrade_log']), len(reg134['upgrade_log'])))

    # ---- byte discipline: revert check + pointer diff
    gap112_blob = json.dumps(gap112, ensure_ascii=False, indent=1).encode('utf-8')
    reg134_blob = json.dumps(reg134, ensure_ascii=False, indent=1).encode('utf-8')
    gap_revert = copy.deepcopy(gap112)
    gap_revert['appendix'] = gap_revert['appendix'][:-1]
    gap_revert['version'] = gap111['version']
    gap_revert['updated'] = gap111['updated']
    gap_revert['supersedes'] = gap111['supersedes']
    ok_gap = json.dumps(gap_revert, ensure_ascii=False, indent=1).encode('utf-8') == gap111_raw
    reg_revert = copy.deepcopy(reg134)
    for key in ANCHORS_NEW:
        del reg_revert['provenance']['anchor_files'][key]
    reg_revert['upgrade_log'] = reg_revert['upgrade_log'][:-1]
    reg_revert['version'] = reg133['version']
    reg_revert['supersedes'] = reg133['supersedes']
    ok_reg = json.dumps(reg_revert, ensure_ascii=False, indent=1).encode('utf-8') == reg133_raw
    assert ok_gap, 'gap revert check FAILED'
    assert ok_reg, 'registry revert check FAILED'
    log('REVERT CHECK: both reproduce sealed predecessor bytes bitwise')

    def ptr_key(s):
        # normalize: pointer only, strip value annotation + ADDED/REMOVED tag
        base = s.split(' (')[0]
        return base + ' (EVENT)' if '(ADDED)' in s or '(REMOVED)' in s or '(LEN' in s else base

    d_gap = set(ptr_key(s) for s in ptr_diff(gap111, gap112))
    d_reg = set(ptr_key(s) for s in ptr_diff(reg133, reg134))
    gap_planned_n = {'$.appendix (EVENT)', '$.version', '$.updated', '$.supersedes'}
    reg_planned_n = set('$.provenance.anchor_files.' + k + ' (EVENT)' for k in ANCHORS_NEW)
    reg_planned_n |= {'$.upgrade_log (EVENT)', '$.version', '$.supersedes'}
    assert d_gap == gap_planned_n, ('gap diff != planned', sorted(d_gap))
    assert d_reg == reg_planned_n, ('registry diff != planned', sorted(d_reg))
    log('POINTER DIFF == planned set exactly (gap %d ptrs, registry %d ptrs)'
        % (len(d_gap), len(d_reg)))

    # ---- land (formal only)
    gap112_path = os.path.join(OUTDIR, 'gap_ledger_v1_12.json')
    reg134_path = os.path.join(OUTDIR, 'atlas_registry_v1_34.json')
    if not SMOKE:
        with open(gap112_path, 'w', encoding='utf-8', newline='') as f:
            f.write(gap112_blob.decode('utf-8'))
        with open(reg134_path, 'w', encoding='utf-8', newline='') as f:
            f.write(reg134_blob.decode('utf-8'))
        log('LANDED: gap_ledger_v1_12.json (%s) + atlas_registry_v1_34.json (%s)'
            % (sha8(open(gap112_path, 'rb').read()),
               sha8(open(reg134_path, 'rb').read())))

    # ---- arm (b): post-landing audit
    audit = {}
    if not SMOKE:
        g12 = json.load(open(gap112_path, encoding='utf-8'))
        r34 = json.load(open(reg134_path, encoding='utf-8'))
        def ptr_key2(s):
            base = s.split(' (')[0]
            return base + ' (EVENT)' if '(ADDED)' in s or '(REMOVED)' in s or '(LEN' in s else base
        assert set(ptr_key2(s) for s in ptr_diff(gap111, g12)) == gap_planned_n
        assert set(ptr_key2(s) for s in ptr_diff(reg133, r34)) == reg_planned_n
        audit['pointer_diff_on_disk'] = 'planned set exactly (both files)'
        # three-ledger consistency
        c1 = g12['appendix'][-1]['anchor_sha8'] == '9d0f1825' \
            and g12['appendix'][-1]['id'] == 'APPX-6'
        c2 = r34['provenance']['anchor_files']['p3253']['sha8'] == sha8_file(SRC['p3253_result'])
        c3 = sha8_file(SRC['gap_v111']) == '5934631b' and sha8_file(SRC['registry_v133']) == 'a30215fd'
        c4 = g12['version'] == '1.12' and r34['version'] == '1.34'
        c5 = len(g12['appendix']) == 6 and all(
            a['id'] != 'APPX-6' for a in g12['appendix'][:5])
        ledger = json.load(open(os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
                                encoding='utf-8'))
        ph = {e.get('phase') for e in ledger['measurements']}
        c6 = {3246, 3250, 3252, 3253}.issubset(ph)
        audit['three_ledger'] = {
            'appx6_anchor_9d0f1825': c1,
            'reg_p3253_sha_matches_sealed': c2,
            'sealed_v111_v133_byte_unchanged': c3,
            'versions_bumped': c4,
            'appx1_5_untouched_before_appx6': c5,
            'ledger_phases_3246_3250_3252_3253_present': c6}
        assert all([c1, c2, c3, c4, c5, c6]), audit
        log('AUDIT: pointer diff + 6 consistency checks ALL OK')

    res = {'phase': 3254, 'name': NAME, 'mode': 'smoke' if SMOKE else 'full',
           'design_sha8': dsha,
           'gap_v112': {'appendix_n': len(gap112['appendix']),
                        'appx6': appx6_entry,
                        'planned_ptrs': sorted(gap_planned),
                        'file_sha8': None if SMOKE else sha8(open(gap112_path, 'rb').read())},
           'registry_v134': {'anchor_files_n': len(reg134['provenance']['anchor_files']),
                             'anchors_added': ANCHORS_NEW,
                             'planned_ptrs': sorted(reg_planned),
                             'file_sha8': None if SMOKE else sha8(open(reg134_path, 'rb').read())},
           'revert_check': {'gap': ok_gap, 'registry': ok_reg},
           'audit': audit,
           'verdict': 'SMOKE_OK' if SMOKE else 'registry_landed',
           'zero_gpu': True, 'src': 'p3254',
           'note': 'landing executes 3253 preview (append_trigger_MET); '
                   'append-only byte discipline with bitwise revert check; '
                   'sealed v1.11/v1.33 untouched'}
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
