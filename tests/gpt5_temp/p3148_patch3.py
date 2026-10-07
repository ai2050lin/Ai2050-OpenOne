# -*- coding: utf-8 -*-
"""p3148 patch3: prune 3147-skeleton
PART S/X/N3 sections not needed by 3148;
fix bit-count logs /11 -> /N_BIT_TOT.
Run AFTER patch2."""
import io

FP = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
      r'\phase3148_omega_p146_xcross_'
      r'headsrc_uncancel_v1sym.py')

s = io.open(FP, encoding='utf-8').read()
if 'p3148 patch3 applied' in s:
    print('ALREADY_APPLIED')
    raise SystemExit
if 'p3148 patch2 applied' not in s:
    raise SystemExit('PATCH2_MISSING')

n_sub = 0


def rep(old, new, tag):
    global s, n_sub
    cnt = s.count(old)
    assert cnt == 1, (tag, cnt)
    s = s.replace(old, new)
    n_sub += 1
    print('OK sub %s' % tag)


# --- S1: drop sign-matrix trials -------
rep(
    "# sign matrix: TAIL25 pos d{1,2} fill\n"
    "# (neg d{1,2,4} + pos_d4 known bits)\n"
    "_run_coord_trial('s_tailpos_d1.0', TAIL25,\n"
    "                 1, 1.0, rows_A1, base12_A1)\n"
    "_run_coord_trial('s_tailpos_d2.0', TAIL25,\n"
    "                 1, 2.0, rows_A1, base12_A1)\n"
    "# bot15 sign matrix: +/- x d{1,2}\n"
    "_run_coord_trial('s_tb15_pos_d1.0', TBOT15,\n"
    "                 1, 1.0, rows_A1, base12_A1)\n"
    "_run_coord_trial('s_tb15_pos_d2.0', TBOT15,\n"
    "                 1, 2.0, rows_A1, base12_A1)\n"
    "_run_coord_trial('s_tb15_neg_d1.0', TBOT15,\n"
    "                 -1, 1.0, rows_A1, base12_A1)\n"
    "_run_coord_trial('s_tb15_neg_d2.0', TBOT15,\n"
    "                 -1, 2.0, rows_A1, base12_A1)\n"
    "# bot15 subdivision @d4 sgn-1\n"
    "_run_coord_trial('s_tb5_neg_d4.0', TB5,\n"
    "                 -1, 4.0, rows_A1, base12_A1)\n"
    "_run_coord_trial('s_tb10_neg_d4.0', TB10,\n"
    "                 -1, 4.0, rows_A1, base12_A1)\n",
    "# 3148: tail sign curve moved to\n"
    "# PART X2 (s2_* namespace, full\n"
    "# d{1,2,2.5,3,3.5,4} both signs)\n",
    'S1 sign-matrix trials')

# --- S2: drop tail/tb15/sub gates ------
i0 = s.index("# tail (TAIL25) sign curve:")
i1 = s.index("# ================================================================\n"
             "# PART X: co50ex superlinear anatomy")
s = s[:i0] + s[i1:]
n_sub += 1
print('OK sub S2 tail gates')

# --- X1: drop d0.5 + k-subsets + gates -
i0 = s.index("_run_coord_trial('x_co50ex_d0.5', co50ex,")
i1 = s.index("# ================================================================\n"
             "# PART T: pc1 instant replay")
NEW_X = (
    "if not SMOKE:\n"
    "    for tn in ('d_co50ex_d1.0',\n"
    "               'd_co50ex_d2.0',\n"
    "               'd_co50ex_d4.0'):\n"
    "        got = E_res[tn]['chg']\n"
    "        m = abs(got - BIT11[tn]) "  "< 1e-9\n"
    "        n_bit_all += int(m)\n"
    "        bit_anchors[tn] = {\n"
    "            'got': got,\n"
    "            'want': BIT11[tn],\n"
    "            'match': bool(m)}\n"
    "    log('X replay: +3 bit anchors '\n"
    "        '(total %d/%d)'\n"
    "        % (n_bit_all, N_BIT_TOT))\n"
    "assert [int(c) for c in order_ex[:2]] "
    "== ORDER_EX2, list(order_ex[:2])\n"
    "log('order_ex top2 == 3147 %s "
    "(enrichment desc)' % ORDER_EX2)\n\n")
s = s[:i0] + NEW_X + s[i1:]
n_sub += 1
print('OK sub X1 co50ex gates')

# --- N1: flip rows frozen-list check ---
i0 = s.index("# flip rows\nflip_rows = []")
i1 = s.index("# ================================================================\n"
             "# PART X2: tail sign-crossing mid-dose fill")
NEW_N = (
    "# flip rows (3147 frozen list)\n"
    "flip_rows = []\n"
    "for j in range(NCAP):\n"
    "    if fstep_store['n_neg_d0.5'][j] >= 0:\n"
    "        flip_rows.append(j)\n"
    "NF = len(flip_rows)\n"
    "log('N2: %d flip rows %s'\n"
    "    % (NF, flip_rows))\n"
    "if not SMOKE:\n"
    "    assert flip_rows == N47['flip_rows'], \\\n"
    "        flip_rows\n"
    "    assert [int(fstep_store['n_neg_d0.5'][j])\n"
    "            for j in flip_rows] == \\\n"
    "        N47['fsteps'], 'fsteps drift'\n"
    "    log('N2 frozen-list match ok (3147)')\n\n")
s = s[:i0] + NEW_N + s[i1:]
n_sub += 1
print('OK sub N1 flip frozen')

# --- B1: log counts /18 ----------------
rep(
    "    log('S replay: +2 bit anchors (total '\n"
    "        '%d/11)' % n_bit_all)",
    "    log('S replay: +2 bit anchors (total '\n"
    "        '%d/%d)' % (n_bit_all,\n"
    "                    N_BIT_TOT))",
    'B1 S log count')
rep(
    "    log('N replay: +2 bit anchors (total '\n"
    "        '%d/11)' % n_bit_all)",
    "    log('N replay: +2 bit anchors (total '\n"
    "        '%d/%d)' % (n_bit_all,\n"
    "                    N_BIT_TOT))",
    'B1 N log count')

s = s.replace(
    '# p3148 patch2 applied',
    '# p3148 patch2 applied\n'
    '# p3148 patch3 applied')
io.open(FP, 'w', encoding='utf-8',
        newline='\n').write(s)
print('PATCH3_OK subs=%d len=%d'
      % (n_sub, len(s)))
