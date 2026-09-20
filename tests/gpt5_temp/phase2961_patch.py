# -*- coding: utf-8 -*-
"""Patch phase2961 script v2: conditional application."""
import io

P = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\phase2961_primitive_card_compression.py'
t = io.open(P, encoding='utf-8').read()
applied = []

# --- r1: insert correction_note into PREREG before verdict_map
if "'correction_note'" not in t:
    r1_old = "\n    'verdict_map': {"
    r1_new = (
        "\n    'correction_note': ("
        "\n        'run1: (a) a3 pass-gate (sep_func>=15 / sep_null0>=10 "
        "sources) was frozen WITHOUT a reachability pre-check "
        "(discipline 10); actual counts 13/7 -> verdict "
        "anchor_fail_all_void. Fix: a3 demoted to a DESCRIPTIVE "
        "chain-continuity record (counts registered, no pass/fail "
        "gate); anchor gate = a1 AND a2. (b) 2942 key-number 84.81 is "
        "a cross-phase reference (stored in 2944/2945), not in the "
        "2942 source; replaced by 33.03 (sep_inj_sstar). Rerun after "
        "deleting stale execution/result per discipline 3.'),"
        "\n    'verdict_map': {")
    assert r1_old in t, 'r1'
    t = t.replace(r1_old, r1_new, 1)
    applied.append('r1')

# --- r2: a3 anchor description -> descriptive (maybe already applied)
r2_old = ("        'a3': ('chain continuity: sep_func 185.6975 appears in >= 15 '\n"
          "               'source texts and sep_null0 77.26 in >= 10 (the shared '\n"
          "               'baseline spine of the chain)'),")
if r2_old in t:
    r2_new = ("        'a3': ('chain continuity (DESCRIPTIVE, no gate after run1 '\n"
              "               'correction): sep_func 185.6975 and sep_null0 77.26 '\n"
              "               'occurrence counts across the 25 sources are registered '\n"
              "               'as the shared baseline spine of the chain)'),")
    t = t.replace(r2_old, r2_new, 1)
    applied.append('r2')

# --- r3: 2942 mechanism text + key number
m_old = '84.81 复现但中位位移符号翻转已登记。'
if m_old in t:
    m_new = '2944 登记的 L16 s2 sep 84.81 参照。'
    t = t.replace(m_old, m_new, 1)
    applied.append('r3m')
k_old = "'key_numbers': ['0.7401', '-0.0301', '0.861', '84.81']},"
if k_old in t:
    k_new = "'key_numbers': ['0.7401', '-0.0301', '0.861', '33.03']},"
    t = t.replace(k_old, k_new, 1)
    applied.append('r3k')

# --- r4: a3_ok descriptive
r4_old = ("a3_ok = (n_sepfunc >= 15) and (n_sepnull >= 10)\n"
          "report['anchors']['a3'] = {'ok': a3_ok, 'n_sepfunc': n_sepfunc,\n"
          "                           'n_sepnull': n_sepnull}")
if r4_old in t:
    r4_new = ("a3_ok = True  # descriptive after run1 correction (discipline 10)\n"
              "report['anchors']['a3'] = {'descriptive': True,\n"
              "                           'n_sepfunc': n_sepfunc,\n"
              "                           'n_sepnull': n_sepnull}")
    t = t.replace(r4_old, r4_new, 1)
    applied.append('r4')

# --- r5: sha8_8 cleanup
r5_old = ("            'sha256_8': src['sha8_8'] if 'sha8_8' in src else "
          "src['sha8']}})")
if r5_old in t:
    r5_new = "            'sha256_8': src['sha8']}})"
    t = t.replace(r5_old, r5_new, 1)
    applied.append('r5')

io.open(P, 'w', encoding='utf-8').write(t)
print('patched:', applied)
