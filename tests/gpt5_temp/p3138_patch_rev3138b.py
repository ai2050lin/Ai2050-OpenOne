# -*- coding: utf-8 -*-
import io
F = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3138_omega_p136_statebank_'
     r'bicdecomp_probecontrast.py')
src = io.open(F, encoding='utf-8').read()

# --- 1: SEAL constants mark ---
old1 = ("        'TF_SEED': TF_SEED,\n"
        "        'RNG_SEED': RNG_SEED},")
new1 = ("        'TF_SEED': TF_SEED,\n"
        "        'RNG_SEED': RNG_SEED,\n"
        "        'BANK_FP16_SCALED':"
        " True},")
assert src.count(old1) == 1, 'p1 %d' \
    % src.count(old1)
src = src.replace(old1, new1)

# --- 2: bank save with per-layer scale ---
old2 = ("        st = capture_states(\n"
        "            rows_T[dc][t], ALL_L)\n"
        "        arr = np.stack(\n"
        "            [st[l] for l in ALL_L],\n"
        "            axis=1).astype(np.float16)\n"
        "        np.savez(fp,\n"
        "                 states=arr,\n"
        "                 lens=np.array(\n"
        "                     lens_T[dc][t],\n"
        "                     dtype=np.int32))")
new2 = ("        st = capture_states(\n"
        "            rows_T[dc][t], ALL_L)\n"
        "        arr = np.stack(\n"
        "            [st[l] for l in ALL_L],\n"
        "            axis=1)\n"
        "        amax = np.abs(arr).max(\n"
        "            axis=(0, 2))\n"
        "        scale = np.maximum(\n"
        "            1.0, amax / 6e4) \\\n"
        "            .astype(np.float32)\n"
        "        arr16 = (arr\n"
        "                 / scale[None, :, None]\n"
        "                 ).astype(np.float16)\n"
        "        np.savez(fp,\n"
        "                 states=arr16,\n"
        "                 scale=scale,\n"
        "                 lens=np.array(\n"
        "                     lens_T[dc][t],\n"
        "                     dtype=np.int32))")
assert src.count(old2) == 1, 'p2 %d' \
    % src.count(old2)
src = src.replace(old2, new2)

# --- 3: RESUME check requires scale ---
old3 = ("                if zt['states'].shape == \\\n"
        "                        (BANK_N, 40, HIDG):\n"
        "                    done = True")
new3 = ("                if 'scale' in zt.files \\\n"
        "                        and zt['states'].shape == \\\n"
        "                        (BANK_N, 40, HIDG):\n"
        "                    done = True")
assert src.count(old3) == 1, 'p3 %d' \
    % src.count(old3)
src = src.replace(old3, new3)

# --- 4: bank load (C) with scale ---
old4 = ("        H_ld[dc][t] = zt['states']\n"
        "        assert H_ld[dc][t].shape == \\\n"
        "            (BANK_N, 40, HIDG)")
new4 = ("        _a = zt['states'] \\\n"
        "            .astype(np.float32)\n"
        "        assert np.isfinite(_a).all(), \\\n"
        "            'bank shard non-finite'\n"
        "        H_ld[dc][t] = _a\n"
        "        H_sc[dc][t] = zt['scale'] \\\n"
        "            .astype(np.float32)\n"
        "        assert H_ld[dc][t].shape == \\\n"
        "            (BANK_N, 40, HIDG)")
assert src.count(old4) == 1, 'p4 %d' \
    % src.count(old4)
src = src.replace(old4, new4)

# --- 5: H_sc init (two places: C+D loops
#     both build H_ld? no: only C loads;
#     D reuses H_ld) ---
old5 = ("H_ld = {}\n"
        "for dc in DIRS:\n"
        "    H_ld[dc] = {}")
new5 = ("H_ld = {}\n"
        "H_sc = {}\n"
        "for dc in DIRS:\n"
        "    H_ld[dc] = {}\n"
        "    H_sc[dc] = {}")
assert src.count(old5) == 1, 'p5 %d' \
    % src.count(old5)
src = src.replace(old5, new5)

# --- 6: per-layer X fill uses scale (C+D,
#     2 occurrences) ---
old6 = ("            X[di, t] = \\\n"
        "                H_ld[dc][t][:, l, :] \\\n"
        "                .astype(np.float32)")
new6 = ("            X[di, t] = \\\n"
        "                H_ld[dc][t][:, l, :] \\\n"
        "                * H_sc[dc][t][l]")
n6 = src.count(old6)
assert n6 == 2, 'p6 %d' % n6
src = src.replace(old6, new6)

io.open(F, 'w', encoding='utf-8').write(src)
chk = io.open(F, encoding='utf-8').read()
for frag in ("'BANK_FP16_SCALED':",
             "scale = np.maximum(",
             "'scale' in zt.files",
             "H_sc[dc][t] = zt['scale']",
             "* H_sc[dc][t][l]"):
    assert chk.count(frag) >= 1, frag
assert chk.count("* H_sc[dc][t][l]") == 2
print('PATCH OK rev3138b (6 segments)')
