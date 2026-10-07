# -*- coding: utf-8 -*-
import io
F = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase3136_omega_p134_conddose_crossmatrix_w8drop.py'
src = io.open(F, encoding='utf-8').read()

# --- 1: constants: add SWAP_L ---
old1 = """# dvec carrier layers (frozen 3135 fp32)
CAP_L_D = [29, 33, 38]
"""
new1 = """# dvec carrier layers (frozen 3135 fp32)
CAP_L_D = [29, 33, 38]
# swap4 layers for the reference field
# (3134/3135 frozen convention)
SWAP_L = [8, 9, 13, 29]
"""
assert src.count(old1) == 1, 'p1 %d' % src.count(old1)
src = src.replace(old1, new1)

# --- 2: seal constants ---
old2 = """        'CAP_L_D': CAP_L_D, 'L17': L17,
        'L35': L35, 'DOSE_COND': DOSE_COND,"""
new2 = """        'CAP_L_D': CAP_L_D, 'SWAP_L': SWAP_L,
        'L17': L17,
        'L35': L35, 'DOSE_COND': DOSE_COND,"""
assert src.count(old2) == 1, 'p2 %d' % src.count(old2)
src = src.replace(old2, new2)

# --- 3: drop frozen dvec_full load ---
old3 = """dvec_full = {
    int(l): z35['dvec_full_%d' % l]
    .astype(np.float32)
    for l in ([L17] + CAP_L_D)}
norms = {l: float(np.median(np.linalg.norm("""
new3 = """norms = {l: float(np.median(np.linalg.norm("""
assert src.count(old3) == 1, 'p3 %d' % src.count(old3)
src = src.replace(old3, new3)

# --- 4: insert swap capture after b1_base ck_save ---
old4 = """    ck_save('b1_base', {
        'base_fp16': {
            str(l): base_states[l]
            .astype(np.float16)
            for l in ALL_L}})

# --- session-internal generation base ---"""
new4 = """    ck_save('b1_base', {
        'base_fp16': {
            str(l): base_states[l]
            .astype(np.float16)
            for l in ALL_L}})
_S1K = CK['data'].get('b1_swap')
if _S1K is not None:
    swap_states = {
        int(k): v.astype(np.float32)
        for k, v in
        _S1K['swap_fp16'].items()}
    log('B1 swap RESUMED from ckpt')
else:
    swap_states = capture_states(
        rows_cap, ALL_L,
        swap_layers=SWAP_L)
    log('B1 swap4 capture done')
    ck_save('b1_swap', {
        'swap_fp16': {
            str(l): swap_states[l]
            .astype(np.float16)
            for l in ALL_L}})
dvec_full = {}
for l in ALL_L:
    _d = (swap_states[l]
          - base_states[l])
    dvec_full[l] = _d.astype(
        np.float32)
    if l in CAP_L_D:
        log('B1 local dvec L%02d '
            'med||d||=%.4f (3135 '
            'anchor %.4f)'
            % (l, float(np.median(
                np.linalg.norm(
                    dvec_full[l],
                    axis=1))),
               DVEC_MED_35[
                   CAP_L_D.index(l)
                   + 1]))
del swap_states
gc.collect()

# --- session-internal generation base ---"""
assert src.count(old4) == 1, 'p4 %d' % src.count(old4)
src = src.replace(old4, new4)

io.open(F, 'w', encoding='utf-8').write(src)
chk = io.open(F, encoding='utf-8').read()
for frag in ("SWAP_L = [8, 9, 13, 29]",
             "'SWAP_L': SWAP_L",
             "_S1K = CK['data'].get('b1_swap')",
             "B1 local dvec L%02d "):
    assert chk.count(frag) >= 1, frag
assert chk.count("z35['dvec_full_") == 0, 'old dvec_full load remains'
print('PATCH OK rev3136b (4 segments)')
