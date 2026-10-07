# -*- coding: utf-8 -*-
"""Patch4: two Part B bugs.
B1: forward_track2 called as (toks[c], base) -- swapped
    args; correct semantics: prompt=PID[j], gen=toks[c]
    so margin index t aligns with generated-token pos t
    and s0 replay becomes bit-exact vs ref18.
B2: pad_info accumulates across both dcodes but
    pad_sens re-indexes from 0 for A1 -> reads P
    entries. Filter by p['dir'] == dcode."""
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3122_omega_p120_write_content_readout_'
     r'sentence_causal_dist_recon.py')
src = io.open(P, encoding='utf-8').read()


def rep(tag, old, new, n=1):
    global src
    c = src.count(old)
    assert c == n, ('[%s] expect %d, got %d'
                    % (tag, n, c))
    src = src.replace(old, new)


# ---- B1: swapped forward args ----
rep('B1',
    """        for c in SCOND:
            md, mf = forward_track2(
                toks[c], base, None)
""",
    """        for c in SCOND:
            md, mf = forward_track2(
                prompt_ids, toks[c], None)
""")

# ---- B2a: first pad_sens loop ----
rep('B2a',
    """pad_sens = {}
for dcode in ('P', 'A1'):
    ann = annP_c if dcode == 'P' else annA_c
    keep = set()
    pad_j = 0
""",
    """pad_sens = {}
for dcode in ('P', 'A1'):
    ann = annP_c if dcode == 'P' else annA_c
    pads = [p for p in pad_info
            if p['dir'] == dcode]
    keep = set()
    pad_j = 0
""")

# ---- B2b: second pad_sens loop ----
rep('B2b',
    """pad_sens[dcode] = {
        'n_clean': len(e_c),
""",
    """pad_sens[dcode] = {
        'n_clean': len(e_c),
""", 1)  # placeholder no-op anchor (keeps structure)
# second loop: reset pad_j then index pads
rep('B2b2',
    """    e_c = []
    base_all = ref18[dcode].astype(np.float64)
    pad_j = 0
""",
    """    e_c = []
    base_all = ref18[dcode].astype(np.float64)
    pad_j = 0
    # (pads already filtered above for this dcode)
""")

rep('B2c1',
    """        if best is None:
            continue
        info = pad_info[pad_j]
""",
    """        if best is None:
            continue
        info = pads[pad_j]
""", 1)
rep('B2c2',
    """        (k1, k2) = best
        info = pad_info[pad_j]
""",
    """        (k1, k2) = best
        info = pads[pad_j]
""", 1)

# ---- post assertions ----
assert src.count('forward_track2(') == 2, 'calls'  # def + call
assert 'forward_track2(\n                prompt_ids, toks[c], None)' \
    in src, 'fixed call missing'
assert src.count('info = pads[pad_j]') == 2, 'pads idx'
assert 'info = pad_info[pad_j]' not in src, 'old idx'
assert src.count('pads = [p for p in pad_info') == 1, \
    'pads filter count'

py_compile.compile(P, doraise=True)
io.open(P, 'w', encoding='utf-8').write(src)
msg = 'patch4 OK: B1+B2 fixed; COMPILE_OK'
io.open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
        r'\p3122_patch4_out.txt', 'w',
        encoding='utf-8').write(msg + chr(10))
print(msg)
