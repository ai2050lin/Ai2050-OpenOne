# -*- coding: utf-8 -*-
import io
F = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase3138_omega_p136_statebank_'
     r'bicdecomp_probecontrast.py')
src = io.open(F, encoding='utf-8').read()

# --- 1: C segment (ish + csh + retr) ---
a = src.index(
    '    # identity split-half across template')
b = src.index(
    '    # residual top-1 PCA direction')
new_c = (
    "    # identity split-half across template\n"
    "    # halves (per direction; in-half\n"
    "    # base removed) + cross-half top-1\n"
    "    # identity retrieval (same features)\n"
    "    ish_l = {}\n"
    "    retr_l = []\n"
    "    for di, dc in enumerate(DIRS):\n"
    "        XA = X[di, :2].mean(axis=0)\n"
    "        XB = X[di, 2:].mean(axis=0)\n"
    "        IA = XA - XA.mean(axis=0)[None, :]\n"
    "        IB = XB - XB.mean(axis=0)[None, :]\n"
    "        cs = np.sum(IA * IB, axis=1) / (\n"
    "            np.linalg.norm(IA, axis=1)\n"
    "            * np.linalg.norm(IB, axis=1)\n"
    "            + 1e-12)\n"
    "        ish_l[dc] = float(np.median(cs))\n"
    "        IA_n = IA / (np.linalg.norm(\n"
    "            IA, axis=1, keepdims=True)\n"
    "            + 1e-12)\n"
    "        IB_n = IB / (np.linalg.norm(\n"
    "            IB, axis=1, keepdims=True)\n"
    "            + 1e-12)\n"
    "        sim = IA_n @ IB_n.T\n"
    "        retr_l.append(float(np.mean(\n"
    "            np.argmax(sim, axis=1)\n"
    "            == np.arange(BANK_N))))\n"
    "    C_out['ish'][l] = ish_l\n"
    "    C_out['retr'][l] = float(\n"
    "        np.median(retr_l))\n"
    "    # template split-half across row\n"
    "    # halves (each half uses its OWN\n"
    "    # internal base: no leakage)\n"
    "    BgA = X[:, :, row_half == 0] \\\n"
    "        .mean(axis=(0, 1, 2))\n"
    "    BgB = X[:, :, row_half == 1] \\\n"
    "        .mean(axis=(0, 1, 2))\n"
    "    csh_l = {}\n"
    "    for t in range(N_T):\n"
    "        CA = X[:, t][:, row_half == 0] \\\n"
    "            .mean(axis=(0, 1)) - BgA\n"
    "        CB = X[:, t][:, row_half == 1] \\\n"
    "            .mean(axis=(0, 1)) - BgB\n"
    "        csh_l[t] = float(np.dot(CA, CB) / (\n"
    "            np.linalg.norm(CA)\n"
    "            * np.linalg.norm(CB) + 1e-12))\n"
    "    C_out['csh'][l] = csh_l\n")
src = src[:a] + new_c + src[b:]
assert src.count("C_out['retr'][l] = float") == 1

# --- 2: D segment (comps loop) ---
a2 = src.index('    feats = {')
b2 = src.index(
    '    del X, B, I, C, Hhat, feats')
new_d = (
    "    Hhat = B[None, None, None, :] \\\n"
    "        + I[:, None, :, :] \\\n"
    "        + C[None, :, None, :]\n"
    "    comps = {\n"
    "        'raw': X,\n"
    "        'BI': Hhat,\n"
    "        'I': (B[None, None, None, :]\n"
    "              + I[:, None, :, :])}\n"
    "    a_l = {}\n"
    "    ax_l = {}\n"
    "    for cname, Xc in comps.items():\n"
    "        # within-T0 regime: row split\n"
    "        f_tr = np.concatenate(\n"
    "            [Xc[0, 0][tr_rows],\n"
    "             Xc[1, 0][tr_rows]])\n"
    "        y_tr = np.concatenate(\n"
    "            [np.zeros(n_tr, dtype=np.int64),\n"
    "             np.ones(n_tr, dtype=np.int64)])\n"
    "        f_te = np.concatenate(\n"
    "            [Xc[0, 0][te_rows],\n"
    "             Xc[1, 0][te_rows]])\n"
    "        y_te = np.concatenate(\n"
    "            [np.zeros(BANK_N - n_tr,\n"
    "                      dtype=np.int64),\n"
    "             np.ones(BANK_N - n_tr,\n"
    "                     dtype=np.int64)])\n"
    "        a_l[cname] = _auc(f_tr, y_tr,\n"
    "                          f_te, y_te)\n"
    "        # cross-template regime: train\n"
    "        # on T01 mean feats, test on T23\n"
    "        # mean feats (row split both)\n"
    "        f_tr2 = np.concatenate(\n"
    "            [Xc[0, :2].mean(axis=0)[tr_rows],\n"
    "             Xc[1, :2].mean(axis=0)[tr_rows]])\n"
    "        f_te2 = np.concatenate(\n"
    "            [Xc[0, 2:].mean(axis=0)[te_rows],\n"
    "             Xc[1, 2:].mean(axis=0)[te_rows]])\n"
    "        ax_l[cname] = _auc(f_tr2, y_tr,\n"
    "                           f_te2, y_te)\n"
    "    # C-only: no row info -> chance\n"
    "    a_l['C_only'] = 0.5\n"
    "    ax_l['C_only'] = 0.5\n"
    "    D_out['auc'][l] = a_l\n"
    "    D_out['auc_x'][l] = ax_l\n")
src = src[:a2] + new_d + src[b2:]
assert src.count("a_l['C_only'] = 0.5") == 1

# --- 3: D-SOFT log keys ---
old3 = ("    raw_key = [D_out['auc_x'][l]['raw_x']\n"
        "               for l in KEY_L]")
new3 = ("    raw_key = [D_out['auc_x'][l]['raw']\n"
        "               for l in KEY_L]")
assert src.count(old3) == 1
src = src.replace(old3, new3)
old3b = ("           json.dumps([round(\n"
         "               D_out['auc_x'][l]['I_x'], 4)\n"
         "               for l in KEY_L]),\n"
         "           json.dumps([round(\n"
         "               D_out['auc_x'][l]['BI_x'], 4)\n"
         "               for l in KEY_L])))")
new3b = ("           json.dumps([round(\n"
         "               D_out['auc_x'][l]['I'], 4)\n"
         "               for l in KEY_L]),\n"
         "           json.dumps([round(\n"
         "               D_out['auc_x'][l]['BI'], 4)\n"
         "               for l in KEY_L])))")
assert src.count(old3b) == 1
src = src.replace(old3b, new3b)

# --- 4: npz keys ---
for old_k, new_k in (
        ("auc_raw_x=np.array(\n        [D_out['auc_x'][l]['raw_x']",
         "auc_raw_x=np.array(\n        [D_out['auc_x'][l]['raw']"),
        ("auc_I_x=np.array(\n        [D_out['auc_x'][l]['I_x']",
         "auc_I_x=np.array(\n        [D_out['auc_x'][l]['I']"),
        ("auc_BI_x=np.array(\n        [D_out['auc_x'][l]['BI_x']",
         "auc_BI_x=np.array(\n        [D_out['auc_x'][l]['BI']"),
        ("auc_raw_T0=np.array(\n        [D_out['auc'][l]['raw_T0']",
         "auc_raw_T0=np.array(\n        [D_out['auc'][l]['raw']"),
        ("auc_I_T0=np.array(\n        [D_out['auc'][l]['I_T0']",
         "auc_I_T0=np.array(\n        [D_out['auc'][l]['I']")):
    assert src.count(old_k) == 1, old_k[:40]
    src = src.replace(old_k, new_k)

# --- 5: E mA1 removal ---
old5 = ("mA1 = z26['mlg_s0_A1'][:, -1, 0] \\\n"
        "    if 'mlg_s0_A1' in z26.files else None\n")
assert src.count(old5) == 1
src = src.replace(old5, '')

io.open(F, 'w', encoding='utf-8').write(src)
chk = io.open(F, encoding='utf-8').read()
for frag in ("BgA = X[:, :, row_half == 0]",
             "comps = {",
             "a_l['C_only'] = 0.5",
             "Xc[0, :2].mean(axis=0)[tr_rows]"):
    assert chk.count(frag) == 1, frag
assert chk.count("'raw_x'") == 0
assert chk.count('I[:, :2]') == 0
assert chk.count('if False else') == 0
assert chk.count('mA1') == 0
print('PATCH OK rev3138a (5 segments)')
