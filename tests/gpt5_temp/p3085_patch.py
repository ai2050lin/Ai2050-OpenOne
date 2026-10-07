# -*- coding: utf-8 -*-
"""Generate phase3085_omega_p82_l34_full_
arbitration.py from the 3083 blueprint.
Report: p3085_patch_report.txt"""
import io
import py_compile

SRC = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase3083_omega_p80_third_model_'
       r'arbitration.py')
DST = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
       r'\phase3085_omega_p82_l34_full_'
       r'arbitration.py')
REP = (r'D:\AI2050\Ai2050-OpenOne\tests'
       r'\gpt5_temp\p3085_patch_report.txt')

src = io.open(SRC, encoding='utf-8').read()
o = []


def sub1(old, new, tag):
    global src
    n = src.count(old)
    assert n == 1, (tag, n)
    src = src.replace(old, new)
    o.append('sub1 %s' % tag)


# ---------- S1 docstring wholesale ----------
i = src.index('"""Phase 3083:')
j = src.index('"""', i + 10)
NEW_DOC = '"""Phase 3085: Omega-P82 full-pipeline\nfour-way arbitration at L_INJ=34 on\nqwen2.5-3b-instruct (the 3083 arbitration\nre-opened at the 3084-confirmed loading\nlayer).\n\nQuestion (3085 A, menu of 3084): the 3084\nlayer scan showed that the 3083 family-C\ndegeneration (n_neg=2/16 at L_INJ=33) is a\nLAYER-POSITION effect: L28 and L34 rescue\nall three families, and L34 (2nd from last,\nthe qwen3-4b protocol position) is the\nstrongest layer (n_neg 12/12/11, med_c\n0.22-0.36, R_ALL -0.23..-0.30).  The 3083\nfour-way trunk/dispersed x migrates\narbitration was postponed by the L33\ndegeneration; it is re-opened here by\nrerunning the FULL 3083 pipeline (E1 repV\nladder + E3 per-head scan + focal top8 + E4\n255-subset sweep + spectrum_struct +\nG_DS/f2~T_AB migration gate + cross\nstatistics) at L_INJ=34.\n\nPipeline (3083-identical except L_INJ and\nthe new repro anchor): three frozen 3076\nfamilies A/B/C, 8 bodies x 4 prefixes = 32\nprompts each; V-banks + last-position banks\n(LG/VB/PB/BX/BA/BM/BH); b-anchors\nb0/b1/b3/b4/b6/b7a/b8; E1 repV injection\nladder at L_INJ=34 (24 pairs); E3 per-head\nsingle-swap scan (16 heads x 24 pairs) ->\nr1; per-family focal top8 (3071 criterion);\nE4 ALL 255 non-empty subsets swapped\njointly + full 16-head swap; in-run\nspectrum (3082 E3 bit-identical\ncolumn-centered SVD -> PR / k_eff / top3);\ncross statistics (seed 3085, 20000 perms);\nG_DS main gate (count>=4 AND min sp>0 over\nthe 6 f2 tests); Stouffer z and Bonferroni\nreported.\n\nNEW vs 3083: L34 deterministic reproduction\nanchor vs the 3084 layer-scan npz (same\nmodel, same frozen texts, same layer):\nn_neg and top8 must match EXACTLY and\nmed_c / CS1H / R_ALL within 1e-9 (the\nfp32-roundtrip noise measured by the\n3083-vs-3084 L33 probe is <= 3e-12;\ntolerance is 3 orders above); mismatch ->\nsetup failure; smoke skips the anchor.\nActivation determinism was already\ndemonstrated by the 3084 L33 replay of the\n3083 state (top8 bit-equal, med_c within\n6e-13).\n\nverdict (preregistered, unchanged from\n3083):  setup/b-anchor/repro failure ->\nthird_setup_failed;  any family top8\ndegenerate (n_neg < 8) ->\nthird_top8_degenerate;  spectrum class\nfrom CS top3 (thresholds in the 3082 gap):\nall top3 >= 0.9 -> trunk; all top3 <= 0.5\n-> dispersed; else mixed;  migration\nstate: G_DS AND f2~T_AB p<0.05 -> locked;\nG_DS -> partial; else absent;  combined:\ntrunk + locked/partial ->\nthird_trunk_migrates; dispersed +\nlocked/partial -> third_dispersed_migrates\n(PR diagnosis falsified); trunk + absent\n-> third_trunk_no_migrate; dispersed +\nabsent -> third_dispersed_no_migrate (3082\nprediction holds, 4B-side outlier\ndeepened); mixed -> third_mixed_<state>\n\narbitration reading (frozen, unchanged):\nthird_trunk_migrates -> DS7B was the\noutlier (R1 distill specialization);\nthird_dispersed_migrates -> 3082 PR\ndiagnostic falsified; third_trunk_no_\nmigrate -> qwen3-4b specificity deepened;\nthird_dispersed_no_migrate -> 3082\nprediction holds (4B-side trunk is the\nrare anatomy).\n\nmodel adaptation (frozen before run):\nqwen2.5-3b-instruct (Qwen2ForCausalLM,\nhidden 2048, 36 layers, 16 heads, 2 kv\nheads, inter 11008, vocab 151936, bf16,\n6.18 GB); L_INJ=34, L_POST=35 (injection\nat the 2nd-from-last layer V, observation\nat the last; layer 35 downstream; the 3084\nscan selected L34 as the strongest rescue\nlayer); E2H/lens machinery omitted as in\n3081/3083 (all cosines from forward\nlogits); input embeddings are TIED to the\nunembed in this model\n(tie_word_embeddings=True; forwards/\nlogits protocol unaffected, recorded as\nan adaptation); same frozen 3076 texts.\n\nlimitations (recorded): L_INJ=34 chosen\nfrom the 3084 descriptive layer scan\n(strongest rescue magnitudes); the rescue\nband boundary remains unmapped (coarse\n4-layer grid); qwen2.5-3b shares the Qwen2\nlineage with the DS7B backbone - not a\nmaximally independent model; instruct-\ntuned like qwen3-4b; same 3076 texts; n=24\npairs per family pair; partial spearman\nfirst-order on n=24 is orientation\nevidence; causal-connective paradigm,\nshared syntactic frame; migration defined\non the per-family focal top-8 subset\nframe; spectrum classification thresholds\n(0.9 / 0.5) chosen in the 3082 empirical\ngap BEFORE this run.\n\nmemory discipline: single model; families\nsequential; big banks fp64 CPU freed after\neach family (del + gc + empty_cache); no\nW32, no Wo34/Wo35 (E2H omitted).\n"""'
src = src[:i] + NEW_DOC + src[j + 3:]
o.append('docstring wholesale replaced')

# ---------- constants ----------
sub1('PHASE = 3083', 'PHASE = 3085', 'PHASE')
sub1("NAME = 'omega_p80_third_model_"
     "arbitration'",
     "NAME = 'omega_p82_l34_full_"
     "arbitration'", 'NAME')
sub1('SEED_MAIN = 3083', 'SEED_MAIN = 3085',
     'SEED_MAIN')
sub1('L_INJ = 33\nL_POST = 34',
     'L_INJ = 34\nL_POST = 35', 'layers')
sub1('SEED = 3083',
     'SEED = 3085\nREPRO_TOL = 1e-9', 'SEED')
sub1("'phase3083', NAME)", "'phase3085', NAME)",
     'OUT')
n = src.count('seed 3083')
assert n == 3, n
src = src.replace('seed 3083', 'seed 3085')
o.append('replace_all seed 3083 -> seed 3085 (3)')

# ---------- PREREG mode ----------
sub1("'ladder at L_INJ=33 (24 pairs, '",
     "'ladder at L_INJ=34 (24 pairs, '",
     'mode ladder')
sub1("'spectrum (3082 E3 bit-identical '\n"
     "            'column-centered SVD of CS and '\n"
     "            'CS1H: PR / k_eff / top3); smoke '\n"
     "            'mode optional (SMOKE=1: 12 '\n"
     "            'masks, K3=4 pairs, NP_USE=8, '\n"
     "            'cross stats off)',",
     "'spectrum (3082 E3 bit-identical '\n"
     "            'column-centered SVD of CS and '\n"
     "            'CS1H: PR / k_eff / top3); NEW '\n"
     "            'vs 3083: L34 deterministic '\n"
     "            'reproduction anchor vs the '\n"
     "            '3084 layer-scan npz (n_neg/'\n"
     "            'top8 exact; med_c/CS1H/R_ALL '\n"
     "            'within 1e-9; feeds setup_ok; '\n"
     "            'smoke skips); smoke mode '\n"
     "            'optional (SMOKE=1: 12 masks, '\n"
     "            'K3=4 pairs, NP_USE=8, cross '\n"
     "            'stats off)',",
     'mode tail')

# ---------- PREREG question ----------
sub1("    'question': '3083 A (menu of 3082): 3082 '\n"
     "                'anatomized the DS7B migration '\n"
     "                'absence into a SPECTRAL split: '\n"
     "                'qwen3-4b carries one shared '\n"
     "                'low-rank causal trunk per '\n"
     "                'family (CS top3 0.957-0.964) '\n"
     "                'while DS7B is high-rank '\n"
     "                'dispersed (top3 0.27-0.32) '\n"
     "                'with random head-level focal '\n"
     "                'overlap.  Preregistered '\n"
     "                'falsifiable prediction: '\n"
     "                'trunk-type models should show '\n"
     "                'head migration / cos lock, '\n"
     "                'dispersed-type should not.  '\n"
     "                'Run the full 3081 protocol on '\n"
     "                'qwen2.5-3b-instruct (Qwen2 '\n"
     "                'architecture like the DS7B '\n"
     "                'backbone, instruct-tuned like '\n"
     "                'qwen3-4b, NOT an R1 distill) '\n"
     "                'and combine the in-run spectrum '\n"
     "                'class with the migration gate '\n"
     "                'into the verdict.',",
     "    'question': '3085 A (menu of 3084): the '\n"
     "                '3084 layer scan located the '\n"
     "                '3B causal loading machinery: '\n"
     "                'L33 (the 3083 position) is a '\n"
     "                'valley (family C n_neg=2/16, '\n"
     "                'med_c lowest, R_ALL near '\n"
     "                'zero) while L34 is the '\n"
     "                'strongest rescue layer (n_neg '\n"
     "                '12/12/11, med_c 0.22-0.36, '\n"
     "                'R_ALL -0.23..-0.30, the '\n"
     "                'qwen3-4b protocol position). '\n"
     "                ' The 3083 four-way '\n"
     "                'trunk/dispersed x migrates '\n"
     "                'arbitration was postponed by '\n"
     "                'the L33 degeneration - it is '\n"
     "                'RE-OPENED here: rerun the '\n"
     "                'full 3083 pipeline (E1 + E3 + '\n"
     "                'E4 + spectrum + cross '\n"
     "                'statistics) at L_INJ=34 and '\n"
     "                'combine the in-run spectrum '\n"
     "                'class with the migration gate '\n"
     "                'into the four-way verdict.',",
     'question')

# ---------- PREREG layer_mapping ----------
sub1("    'layer_mapping': 'L_INJ=33 (V injection + '\n"
     "                     'attn swaps), L_POST=34 '\n"
     "                     '(block-output continuity '\n"
     "                     'check); 36-layer stack, '\n"
     "                     'layers 34-35 downstream of '\n"
     "                     'the injection; DS7B used '\n"
     "                     'L25/L26 of 28 (3rd/2nd '\n"
     "                     'from last), qwen3-4b used '\n"
     "                     'L34/L35 of 36 (2nd from '\n"
     "                     'last / last)',",
     "    'layer_mapping': 'L_INJ=34 (V injection + '\n"
     "                     'attn swaps), L_POST=35 '\n"
     "                     '(block-output continuity '\n"
     "                     'check); 36-layer stack, '\n"
     "                     'layer 35 downstream of '\n"
     "                     'the injection; selected by '\n"
     "                     'the 3084 layer scan '\n"
     "                     '(L28/31/33/34: L34 strongest '\n"
     "                     'rescue, L33 the 3083 '\n"
     "                     'valley); DS7B used L25/L26 '\n"
     "                     'of 28 (3rd/2nd from last), '\n"
     "                     'qwen3-4b used L34/L35 of '\n"
     "                     '36 (2nd from last / last)',",
     'layer_mapping')

# ---------- PREREG independence + repro_anchor ----------
sub1("    'independence': 'f2 values and CS spectra '\n"
     "                    'on qwen2.5-3b are new data '\n"
     "                    '(different model, new '\n"
     "                    'forwards); the gate G_DS, '\n"
     "                    'the spectrum thresholds '\n"
     "                    '(0.9 / 0.5) and the '\n"
     "                    'combined verdict tree were '\n"
     "                    'fixed in this prereg before '\n"
     "                    'any qwen2.5-3b observation; '\n"
     "                    'family texts are the same '\n"
     "                    'frozen 3076 texts - '\n"
     "                    'independence is across '\n"
     "                    'models, not texts',",
     "    'independence': 'f2 values and CS spectra '\n"
     "                    'at L34 are new data (new '\n"
     "                    'forwards, seed 3085); the '\n"
     "                    'gate G_DS, the spectrum '\n"
     "                    'thresholds (0.9 / 0.5) and '\n"
     "                    'the combined verdict tree '\n"
     "                    'were fixed before any L34 '\n"
     "                    'observation; 3084 L34 '\n"
     "                    'priors (n_neg/med_c/R_ALL '\n"
     "                    'descriptives) guided the '\n"
     "                    'LAYER CHOICE ONLY - all '\n"
     "                    'verdict inputs (f2/T/U, '\n"
     "                    'spectrum class, gates) are '\n"
     "                    'computed fresh in-run; '\n"
     "                    'family texts are the same '\n"
     "                    'frozen 3076 texts',\n"
     "    'repro_anchor': 'activation collection is '\n"
     "                    'deterministic: same model, '\n"
     "                    'same frozen texts, same '\n"
     "                    'layer -> L34 values must '\n"
     "                    'match the 3084 npz per '\n"
     "                    'family (n_neg exact, top8 '\n"
     "                    'exact, med_c/CS1H/R_ALL '\n"
     "                    'abs diff <= 1e-9); the 1e-9 '\n"
     "                    'tolerance covers the fp32-'\n"
     "                    'roundtrip noise measured by '\n"
     "                    'the 3083-vs-3084 L33 probe '\n"
     "                    '(max 5.8e-13 med_c, 3.0e-12 '\n"
     "                    'CS1H, 1.4e-13 R_ALL); '\n"
     "                    'mismatch -> '\n"
     "                    'third_setup_failed; smoke '\n"
     "                    'skips the anchor (K3/NP_USE '\n"
     "                    'differ)',",
     'independence+repro_anchor')

# ---------- PREREG limitations ----------
sub1("'True) - the forwards/logits-only '\n"
     "        'protocol is unaffected, recorded '\n"
     "        'as an adaptation',",
     "'True) - the forwards/logits-only '\n"
     "        'protocol is unaffected, recorded '\n"
     "        'as an adaptation; L_INJ=34 was '\n"
     "        'selected from the 3084 '\n"
     "        'descriptive layer scan (strongest '\n"
     "        'rescue magnitudes, n_neg 12/12/'\n"
     "        '11) - layer-choice dependence is '\n"
     "        'handled explicitly but the '\n"
     "        'rescue-band boundary remains '\n"
     "        'unmapped (coarse 4-layer grid)',",
     'limitations')

# ---------- repro code block ----------
sub1("# ==== per-family causal spectrum (3082 E3\n"
     "# bit-identical: column-centered SVD) ====\n"
     "SPEC = {}",
     "# ==== L34 deterministic reproduction\n"
     "# anchor (vs 3084 layer-scan npz) ====\n"
     "REPRO = {}\n"
     "REPRO_OK = True\n"
     "if SMOKE:\n"
     "    log('repro anchor skipped (smoke: '\n"
     "        'K3/NP_USE differ from the 3084 '\n"
     "        'reference)')\n"
     "else:\n"
     "    z84 = np.load(os.path.join(\n"
     "        ROOT, 'tests', 'glm5', 'result',\n"
     "        'rdc_query_construction_20260913',\n"
     "        'phase3084',\n"
     "        'omega_p81_3b_layer_scan',\n"
     "        'omega_p81_3b_layer_scan.npz'),\n"
     "        allow_pickle=False)\n"
     "    for fk in FKEYS:\n"
     "        d_mc = abs(float(RES_F[fk]['med_c'])\n"
     "                   - float(z84['MED_C_L34_'\n"
     "                                + fk]))\n"
     "        nn_ok = bool(\n"
     "            int(RES_F[fk]['n_neg'])\n"
     "            == int(z84['N_NEG_L34_' + fk]))\n"
     "        t8_ok = bool(\n"
     "            [int(h) for h\n"
     "             in RES_F[fk]['top8']]\n"
     "            == [int(h) for h\n"
     "                in z84['TOP8_L34_' + fk]])\n"
     "        d_cs = float(np.max(np.abs(\n"
     "            RES_F[fk]['CS1H']\n"
     "            - z84['CS1H_L34_' + fk])))\n"
     "        d_ra = abs(float(RES_F[fk]['R_ALL'])\n"
     "                   - float(z84['R_ALL_L34_'\n"
     "                                + fk]))\n"
     "        ok = bool(d_mc <= REPRO_TOL\n"
     "                  and nn_ok and t8_ok\n"
     "                  and d_cs <= REPRO_TOL\n"
     "                  and d_ra <= REPRO_TOL)\n"
     "        REPRO[fk] = {'medc_diff': d_mc,\n"
     "                     'nneg_ok': nn_ok,\n"
     "                     'top8_ok': t8_ok,\n"
     "                     'cs1h_diff': d_cs,\n"
     "                     'rall_diff': d_ra,\n"
     "                     'ok': ok}\n"
     "        REPRO_OK = REPRO_OK and ok\n"
     "        log('repro[%s] vs 3084 L34: '\n"
     "            'medc_diff=%.3e nneg_ok=%s '\n"
     "            'top8_ok=%s cs1h_diff=%.3e '\n"
     "            'rall_diff=%.3e ok=%s'\n"
     "            % (fk, d_mc, nn_ok, t8_ok,\n"
     "               d_cs, d_ra, ok))\n"
     "\n"
     "# ==== per-family causal spectrum (3082 E3\n"
     "# bit-identical: column-centered SVD) ====\n"
     "SPEC = {}",
     'repro block')

# ---------- setup_ok_all ----------
sub1("setup_ok_all = bool(all(\n"
     "    RES_F[fk]['setup_ok_f'] for fk in FKEYS))",
     "setup_ok_all = bool(all(\n"
     "    RES_F[fk]['setup_ok_f'] for fk in FKEYS)\n"
     "    and REPRO_OK)",
     'setup_ok_all')

# ---------- npz keys ----------
sub1("    'TIED': np.bool_(TIED),\n"
     "}\n"
     "for fk in FKEYS:\n"
     "    R = RES_F[fk]",
     "    'TIED': np.bool_(TIED),\n"
     "    'REPRO_OK': np.bool_(REPRO_OK),\n"
     "}\n"
     "for fk in FKEYS:\n"
     "    if fk in REPRO:\n"
     "        save['REPRO_MEDC_DIFF_' + fk] = \\\n"
     "            np.float64(\n"
     "                REPRO[fk]['medc_diff'])\n"
     "        save['REPRO_CS1H_DIFF_' + fk] = \\\n"
     "            np.float64(\n"
     "                REPRO[fk]['cs1h_diff'])\n"
     "        save['REPRO_RALL_DIFF_' + fk] = \\\n"
     "            np.float64(\n"
     "                REPRO[fk]['rall_diff'])\n"
     "        save['REPRO_NNEG_OK_' + fk] = \\\n"
     "            np.bool_(REPRO[fk]['nneg_ok'])\n"
     "        save['REPRO_TOP8_OK_' + fk] = \\\n"
     "            np.bool_(REPRO[fk]['top8_ok'])\n"
     "    R = RES_F[fk]",
     'npz repro keys')

# ---------- result stats repro ----------
sub1("    'stats': {\n"
     "        'families': fam_stats,",
     "    'stats': {\n"
     "        'repro': (None if SMOKE\n"
     "                  else REPRO),\n"
     "        'families': fam_stats,",
     'result repro')

# ---------- sanity scan ----------
bad = ('phase3083', 'omega_p80', 'L_INJ = 33',
       'L_POST = 34', 'L_POST=34', 'seed 3083')
hits = []
for b in bad:
    if b in src:
        hits.append(b)
assert not hits, hits
n33 = src.count('L_INJ=33')
assert n33 == 1, n33
o.append('sanity scan clean (L_INJ=33 x1 = '
         'docstring historical ref)')

counts = {
    'REPRO_TOL': 4,
    'phase3084': 1,
    'phase3085': 1,
    'spectrum_struct': 5,
    'third_top8_degenerate': 6,
    'omega_p82_l34_full_arbitration': 1,
    'seed 3085': 5,
    'repro_anchor': 1,
    'REPRO_OK': 6,
}
for k, want in counts.items():
    got = src.count(k)
    assert got == want, (k, got, want)
o.append('counts ok ' + str(counts))

io.open(DST, 'w', encoding='utf-8').write(src)
py_compile.compile(DST, doraise=True)
o.append('PY_COMPILE OK')
o.append('written %d chars' % len(src))
io.open(REP, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('PATCH_OK')
