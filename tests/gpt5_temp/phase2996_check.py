# -*- coding: utf-8 -*-
"""Static structure check for phase2996 script."""
import io
import py_compile

P = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
     r'\phase2996_omega_f2a_registry_caliber_audit.py')
py_compile.compile(P, doraise=True)
s = io.open(P, encoding='utf-8').read()
o = []
o.append('compile=OK')
o.append('freeze_first=%s' % (
    s.find('execution.json frozen')
    < s.find('cells (2977 exec verbatim') if
    'cells (2977 exec verbatim' in s else
    s.find('execution.json frozen') < s.find('ARM 1')))
o.append('verdicts=%s' % all(v in s for v in [
    'anchor_fail_all_void',
    'registry_replicated_at_matched_caliber',
    'qwen_registry_act_side_only_glm4_absent',
    'registry_robust_across_calibers',
    'registry_not_replicated_confirmed']))
o.append('list_hooks_q=%s' % (
    'ai_cap = {li: [] for li in range(NL_Q)}' in s))
o.append('list_hooks_g=%s' % (
    'g_ai = {li: [] for li in range(NL_G)}' in s))
o.append('no_single_slot_sweep=%s' % (
    "A_ai[:, i, :] = ai[0, 1, :]" not in s))
o.append('preinit=%s' % (
    'verdict = None' in s and 'T1 = T2 = T3 = None' in s))
o.append('single_perm=%s' % (
    'rng.permutation(lab)' in s and
    'pool[pm]' not in s))
o.append('ga5_l39=%s' % (
    'dirs_mlp_g[NL_G - 1]' in s))
o.append('deg_anchor=%s' % (
    'non39_diff > 1e-3' in s))
o.append('silu_recompute=%s' % (
    'g / (1.0 + np.exp(-g)) * u' in s and
    'g * _t.sigmoid(g) * u' in s))
o.append('fused_slice=%s' % (
    'W64[:INT_G]' in s and 'W64[INT_G:]' in s))
o.append('unload_between=%s' % (
    s.find('qwen unloaded') < s.find('ARM 2')))
o.append('int_cast_none_needed=%s' % (
    'int((' not in s))
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_chk2996.txt', 'w',
        encoding='utf-8').write('\n'.join(o))
print('ok')
