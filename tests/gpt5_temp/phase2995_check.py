# -*- coding: utf-8 -*-
import io

s = io.open(r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
            r'\phase2995_omega_f1_glm4_panel.py',
            encoding='utf-8').read()
o = []
o.append('freeze_first=%s' % (
    0 < s.find('execution.json frozen')
    < s.find('cells (runtime filter')))
o.append('no_tautology=%s' % (
    'Rf @ dirs_mlp[10] - aligns' not in s))
o.append('abs5=%s' % ('np.abs(Rf @ dirs_mlp[10])' in s))
o.append('preinit=%s' % ('proj_top = np.zeros(n_cells)' in s))
o.append('batch_assert=%s' % (
    "cap_ai['x'].shape[0] == 1" in s))
o.append('verdicts=%s' % all(
    v in s for v in ['anchor_fail_all_void',
                     'omega_f1_panel_absent',
                     'omega_f1_panel_partial',
                     'omega_f1_panel_replicated']))
o.append('grades=%s' % ('replicated' in s
                        and 'directional' in s
                        and 'not_replicated' in s))
o.append('no_upproj=%s' % ('up_proj' not in s))
o.append('INT_half=%s' % ('Wl[INT:, :]' in s
                          and 'Rf[INT:, :]' in s))
io.open(r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\tmp_chk2995c.txt',
        'w', encoding='utf-8').write('\n'.join(o))
print('written')
