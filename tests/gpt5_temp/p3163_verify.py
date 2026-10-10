# -*- coding: utf-8 -*-
"""Independent on-disk verification for Phase 3163 closeout (separate process, re-hash).
Expected disk shas parsed from the closeout output file (no manual transcription)."""
import hashlib, json, io, os, re
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
PDIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                    'phase3163', 'g4p5_redundancy')
CO_OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3163_closeout_out.txt')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3163_verify_out.txt')
R = []
ok = True

def chk(name, cond, extra=''):
    global ok
    ok = ok and bool(cond)
    R.append('%-42s %s %s' % (name, 'PASS' if cond else 'FAIL', extra))

def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]

MS = ('qwen3-4b', 'qwen3-14b', 'glm4')

# 1. disk sha baseline (parsed from closeout output, independently re-hashed here)
co = io.open(CO_OUT, encoding='utf-8').read()
msh = re.search(r'disk sha: (\{.*?\})\n', co, re.S)
assert msh, 'closeout disk sha line not found'
exp = json.loads(msh.group(1))
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
chk('disk sha baseline 9/9 (independent re-hash)', got == exp,
    str({k: (got[k], exp[k]) for k in got if got[k] != exp[k]}))

# 2. seal byte-level reconstruction + clsA coherence
clssA = []
for m in MS:
    raw = json.load(io.open(os.path.join(PDIR, m, 'result.json'), encoding='utf-8'))
    clssA.append(raw['clsA'])
    seal = raw.pop('seal_sha8')
    blob = json.dumps(raw, ensure_ascii=False, indent=1).encode('utf-8').replace(b'\n', b'\r\n')
    chk('seal reconstruct %s' % m, hashlib.sha256(blob).hexdigest()[:8] == seal,
        'recomputed=%s seal=%s' % (hashlib.sha256(blob).hexdigest()[:8], seal))
raws = json.load(io.open(os.path.join(PDIR, 'summary', 'result_summary.json'), encoding='utf-8'))
seal_s = raws.pop('seal_sha8')
blob_s = json.dumps(raws, ensure_ascii=False, indent=1).encode('utf-8').replace(b'\n', b'\r\n')
chk('seal reconstruct summary', hashlib.sha256(blob_s).hexdigest()[:8] == seal_s,
    'recomputed=%s seal=%s' % (hashlib.sha256(blob_s).hexdigest()[:8], seal_s))
chk('clsA agreement 3/3', len(set(clssA)) == 1, '/'.join(clssA))

# 3. ledger entry
led = json.loads(io.open(LEDGER, encoding='utf-8').read())
n = len(led['measurements'])
e3163 = [m for m in led['measurements'] if m.get('phase') == 3163]
chk('ledger n=315 + 3163 entry', n == 315 and len(e3163) == 1, 'n=%d' % n)

# 4. MEMO section + prereg + BOM/EOL
mb = open(MEMO, 'rb').read()
mt = mb.decode('utf-8')
chk('memo BOM', mb[:3] == b'\xef\xbb\xbf')
chk('memo EOL uniform CRLF', mb.count(b'\n') == mb.count(b'\r\n'))
chk('memo 3163 section', '## Phase 3163: 消耗冗余性判别（G4-P5）' in mt)
chk('memo 3164 prereg (G5-A2)', '预注册 Phase 3164' in mt)

# 5. daily + workspace MEMORY markers
daily = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
chk('daily 3163 marker', os.path.exists(daily) and '3163 消耗冗余性判别闭环' in io.open(daily, encoding='utf-8').read())
mem = io.open(os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md'), encoding='utf-8').read()
chk('workspace MEMORY 3163', '3163 消耗冗余性判别闭环（2026-10-09）' in mem)

# 6. array-level device re-verification from sealed npz (all three models):
#    A identity window (row 1) slots L_mid..L_mid+3 bitwise constant;
#    C identity window (row 3) slots L_mid..NL-1 bitwise constant;
#    SHARE_A3 == CURVES[...,1,L_mid+3]; SHARE_NL == CURVES[..., :, NL]
for m in MS:
    z = np.load(os.path.join(PDIR, m, 'collect.npz'))
    CUR = z['CURVES'].astype(np.float32)  # (NA, ND, 4, NH)
    LM = int(z['l_mid'])
    NH = CUR.shape[3]
    NLc = NH - 1
    wa = CUR[..., 1, LM:LM + 4]
    okA = all(np.array_equal(wa[..., 0], wa[..., k]) for k in (1, 2, 3))
    wc = CUR[..., 3, LM:NLc]
    okC = all(np.array_equal(wc[..., 0], wc[..., k]) for k in range(1, wc.shape[-1]))
    chk('%s device: A-window bitwise + C-window bitwise' % m, okA and okC,
        'na=%d nd=%d' % (CUR.shape[0], CUR.shape[1]))
    chk('%s SHARE_A3==CURVES[A,L_mid+3]' % m,
        np.array_equal(z['SHARE_A3'].astype(np.float32), CUR[..., 1, LM + 3]))
    chk('%s SHARE_NL==CURVES[:, :, NL]' % m,
        np.array_equal(z['SHARE_NL'].astype(np.float32), CUR[..., :, NLc]))

# 7. per-anchor parts vs collect slices bitwise (14b/glm4, the per-anchor models)
for m in ('qwen3-14b', 'glm4'):
    z = np.load(os.path.join(PDIR, m, 'collect.npz'))
    pair = [('SHARE_NL', 'share_nl'), ('SHARE_LM', 'share_lm'), ('SHARE_A3', 'share_a3'),
            ('CURVES', 'curves')]
    allok = True
    for big, small in pair:
        for i in range(4):
            pv = np.load(os.path.join(PDIR, m, '_parts', 'collect_anchor%d.npz' % i))[small]
            cv = z[big][i]
            allok = allok and np.array_equal(pv.astype(np.float32), cv.astype(np.float32))
    chk('%s anchor parts == collect bitwise x4 x4' % m, allok)

with io.open(OUTP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(R) + '\nVERDICT: %s\n' % ('ALL PASS' if ok else 'HAS FAIL'))
print('VERIFY DONE: %s' % ('ALL PASS' if ok else 'HAS FAIL'))
