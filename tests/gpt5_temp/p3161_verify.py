# -*- coding: utf-8 -*-
"""Independent on-disk verification for Phase 3161 closeout (separate process, re-hash)."""
import hashlib, json, io, os
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
PDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3161', 'g4p4_head_attribution')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3161_verify_out.txt')
R = []
ok = True


def chk(name, cond, extra=''):
    global ok
    ok = ok and bool(cond)
    R.append('%-34s %s %s' % (name, 'PASS' if cond else 'FAIL', extra))


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


MS = ('qwen3-4b', 'qwen3-14b', 'glm4')

# 1. disk sha baseline (matches closeout run)
exp = {'res_qwen3-4b': 'ba10c034', 'npz_qwen3-4b': '8f39d6cb',
       'res_qwen3-14b': '44bb3336', 'npz_qwen3-14b': 'd701aef4',
       'res_glm4': 'ac8b5f67', 'npz_glm4': '5b93533d',
       'res_summary': 'ec0e4488', 'res_smoke': 'b262f75f', 'npz_smoke': '34839b76'}
got = {}
for k, (m, kind) in {
    'res_qwen3-4b': ('qwen3-4b', 'result.json'), 'npz_qwen3-4b': ('qwen3-4b', 'collect.npz'),
    'res_qwen3-14b': ('qwen3-14b', 'result.json'), 'npz_qwen3-14b': ('qwen3-14b', 'collect.npz'),
    'res_glm4': ('glm4', 'result.json'), 'npz_glm4': ('glm4', 'collect.npz'),
    'res_summary': ('summary', 'result_summary.json'),
    'res_smoke': ('qwen3-4b', os.path.join('smoke', 'result.json')),
    'npz_smoke': ('qwen3-4b', os.path.join('smoke', 'collect.npz')),
}.items():
    got[k] = sha8_file(os.path.join(PDIR, m, kind))
chk('disk sha baseline 9/9', got == exp, str({k: (got[k], exp[k]) for k in got if got[k] != exp[k]}))

# 2. verdict/cls coherence across models
clss = []
for m in MS:
    r = json.load(io.open(os.path.join(PDIR, m, 'result.json'), encoding='utf-8'))
    clss.append(r['cls'])
    # seal byte-level reconstruction: file content minus seal_sha8 field, no-sort dumps
    raw = json.load(io.open(os.path.join(PDIR, m, 'result.json'), encoding='utf-8'))
    seal = raw.pop('seal_sha8')
    # seal was hashed over the on-disk file written by json.dump in Windows text mode
    # -> real newlines are CRLF; reproduce that byte-exactly
    blob = json.dumps(raw, ensure_ascii=False, indent=1).encode('utf-8').replace(b'\n', b'\r\n')
    chk('seal reconstruct %s' % m, hashlib.sha256(blob).hexdigest()[:8] == seal,
        'recomputed=%s seal=%s' % (hashlib.sha256(blob).hexdigest()[:8], seal))
chk('cls agreement 3/3', len(set(clss)) == 1, '/'.join(clss))

# 3. ledger entry
led = json.loads(io.open(LEDGER, encoding='utf-8').read())
n = len(led['measurements'])
e3161 = [m for m in led['measurements'] if m.get('phase') == 3161]
chk('ledger n=314 + 3161 entry', n == 314 and len(e3161) == 1, 'n=%d' % n)

# 4. MEMO section + prereg + BOM/EOL
mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8')
chk('memo BOM', mb[:3] == b'\xef\xbb\xbf')
chk('memo EOL uniform CRLF', mb.count(b'\n') == mb.count(b'\r\n'))
chk('memo 3161 section', '## Phase 3161: 消耗的头归因（G4-P4）' in mt)
chk('memo 3163 prereg', '预注册 Phase 3163：G4-P5 消耗冗余性判别' in mt)

# 5. daily + workspace MEMORY markers
daily = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
chk('daily 3161 marker', os.path.exists(daily) and '3161 消耗头归因闭环' in io.open(daily, encoding='utf-8').read())
mem = io.open(os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md'), encoding='utf-8').read()
chk('workspace MEMORY 3161', '3161 消耗头归因闭环（2026-10-09）' in mem)

# 6. array-level: per-anchor parts vs collect slices bitwise (4b), and 14b npz sanity
z = np.load(os.path.join(PDIR, 'qwen3-4b', 'collect.npz'))
pair = [('SHARE_NL', 'share_nl'), ('SHARE_LM', 'share_lm'),
        ('NONE_CURVE', 'none_curve'), ('ALL3_CURVE', 'all3_curve')]
for big, small in pair:
    allok = True
    for i in range(4):
        pv = np.load(os.path.join(PDIR, 'qwen3-4b', '_parts', 'collect_anchor%d.npz' % i))[small]
        cv = z[big][i]
        allok = allok and np.array_equal(pv.astype(np.float32), cv.astype(np.float32))
    chk('4b anchor parts == collect[%s] bitwise x4' % big, allok, z[big].shape)
z14 = np.load(os.path.join(PDIR, 'qwen3-14b', 'collect.npz'))
chk('14b npz l_mid=20', int(z14['l_mid']) == 20, 'R shape=%s' % (z14['R'].shape,))

with io.open(OUTP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(R) + '\nVERDICT: %s\n' % ('ALL PASS' if ok else 'HAS FAIL'))
print('VERIFY DONE: %s' % ('ALL PASS' if ok else 'HAS FAIL'))
