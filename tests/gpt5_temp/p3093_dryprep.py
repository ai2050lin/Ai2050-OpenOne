# -*- coding: utf-8 -*-
"""Dry-run prep for p3093_patch_a2.py:
build mock A1 result.json + a patched copy of
the patch script that reads the mock and
writes to a dry-run output path."""
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'

mock = {
    'verdict': 'layer_rescue',
    'stats': {'rescue_best': 38,
              'families_layers': {}}}
for L in (31, 34, 37, 38):
    mock['stats']['families_layers'][str(L)] = {
        f: {'n_neg': 10 + (L % 5) + i,
            'med_c': 0.05 + 0.01 * i,
            'r_all': -0.05 + 0.02 * i}
        for i, f in enumerate(('A', 'B', 'C'))}
mp = ROOT + (r'\tests\gpt5_temp'
             r'\p3093_mock_a1_result.json')
io.open(mp, 'w', encoding='utf-8').write(
    json.dumps(mock))

src = io.open(ROOT + r'\tests\gpt5_temp'
              r'\p3093_patch_a2.py',
              encoding='utf-8').read()

OLD_RES = ("A1RES = (ROOT + r'\\tests\\glm5"
           "\\result'\n"
           "         r'\\rdc_query_construction_"
           "20260913'\n"
           "         r'\\phase3093\\omega_p90_"
           "qwen14b_layer_'\n"
           "         r'scan\\result.json')")
NEW_RES = ("A1RES = (ROOT + r'\\tests\\gpt5_temp"
           "\\p3093_mock_a1_result.json')")
assert src.count(OLD_RES) == 1, 'A1RES block'
src = src.replace(OLD_RES, NEW_RES)

OLD_DST = ("DST = (ROOT + r'\\tests\\glm5'\n"
           "       r'\\phase3093_omega_p91_"
           "qwen14b_l%d_full_'\n"
           "       r'arbitration.py')  # % L")
NEW_DST = ("DST = (ROOT + r'\\tests\\gpt5_temp"
           "\\p3093_dry_a2_l%d_full_'\n"
           "       r'arbitration.py')  # % L")
assert src.count(OLD_DST) == 1, 'DST block'
src = src.replace(OLD_DST, NEW_DST)

# context dump before BAD assert
OLD_AS = ("bad_left = [b for b in BAD if b "
          "in src]\n"
          "assert not bad_left, ('BAD "
          "remain', bad_left)")
NEW_AS = ("bad_left = [b for b in BAD if b "
          "in src]\n"
          "import io as _io\n"
          "_lns = src.splitlines()\n"
          "_ctx = []\n"
          "for _b in bad_left:\n"
          "    _hit = [ln for ln in _lns "
          "if _b in ln][:3]\n"
          "    _ctx.append('BAD %s => %s'\n"
          "              % (_b, ' ## '\n"
          "                 .join(_hit)))\n"
          "_io.open(r'D:\\AI2050\\Ai2050-"
          "OpenOne\\tests\\gpt5_temp"
          "\\p3093_bad_ctx.txt', 'w',\n"
          "         encoding='utf-8').write("
          "'\\n'.join(_ctx) + '\\n')\n"
          "assert not bad_left, ('BAD "
          "remain', bad_left)")
assert src.count(OLD_AS) == 1, 'AS block'
src = src.replace(OLD_AS, NEW_AS)

dp = ROOT + (r'\tests\gpt5_temp'
             r'\p3093_patch_a2_dryrun.py')
io.open(dp, 'w', encoding='utf-8').write(src)
print('DRYPREP_OK')
