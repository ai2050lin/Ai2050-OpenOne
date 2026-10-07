# -*- coding: utf-8 -*-
"""Phase 3140 closeout remaining steps:
workspace log + MEMORY.md only.
(Ledger n=277 and MEMO block already
persisted and verified.)"""
import io
import os

ROOT = r'D:\AI2050\Ai2050-OpenOne'
WLOG = (ROOT + r'\.workbuddy\memory'
        r'\2026-09-29.md')
WMEM = (ROOT + r'\.workbuddy\memory'
        r'\MEMORY.md')

wlog_add = '''

## Phase 3140 (Omega-P138) 闭环 [06:35]
- 正式跑：主体 33560s（GPU 降频 231→588s/试验）+ rev-3140b 续跑 83s；56 谱试验+7 WR 试验+84 对 V1/V2。
- verdict=a_3139_ok|spec_repro_3139_ok|iinj_peak_l19|cinj_peak_l38|peaks_separated|spec_dose_monotone|own_below_nb|own_rank_chance|co_enriched|wr_active|wr_pcvar_recorded|retr_v2_flat|retr_v2_below|retr_v1_repro_ok|xphase_ok|coverage_full
- 关键发现：(1) 身份峰区 L17-19（L19 0.344>L17 0.328）vs 模板平台 L23-27（L38 0.977=depth-0 伪象）；(2) own 反特异 rank_frac 0.77-0.91 比随机差（端口类第 5 次确证）；(3) co36 z=6.85 身份富集 vs co50 z=0.23 独立通路；(4) WR PC1 剂量单调行为通路（L29 0.086→0.203→0.547）而 PC2 行特定全 0；(5) 检索失败=写入历史缺失（V2 同族模板无提升）。
- 9/9 与 3139 位级复现 + xphase 1.0 (128/128) + V1 检索位级复现。
- closeout：ledger n=277 sha8=6d699902/seal 8a3ee208；MEMO 追加（T4 第23Phase）；3141 预注册（L24-28 精细化+WR 联合+co36 因果对照+新材料写入诱导）。
'''
existed = os.path.exists(WLOG)
t = io.open(WLOG, encoding='utf-8').read() \
    if existed else '# 2026-09-29 工作日志\n'
if 'Omega-P138) 闭环' not in t:
    with io.open(WLOG, 'a',
                 encoding='utf-8') as f:
        f.write(wlog_add)
t2 = io.open(WLOG, encoding='utf-8').read()
assert 'Omega-P138) 闭环' in t2
print('wlog ok (existed=%s)' % existed)

wm = io.open(WMEM, encoding='utf-8').read()
changed = False
lines = wm.splitlines(keepends=True)
out_lines = []
for l in lines:
    if l.startswith('- max=') \
            and 'max=3140' not in l:
        l = '- max=3140，下一 3141：L24-28 cinj 功能峰精细化(step1 x dose{1,2,4}) + WR PC1 x dvec 联合注入叠加律 + co36 高低身份能量子集因果对照 + 新材料真值绑定写入诱导重测 retr_same_s。**层位谱分离=身份 L17-19 / 模板 L23-27（L38 depth-0 伪象）；own 反特异 rank 0.77-0.91=端口类第 5 次确证；co36 z=6.85 身份富集 vs co50 独立；WR PC1 全局行为通路/PC2 行特异全 0；检索失败=写入历史缺失。**\n'
        changed = True
    out_lines.append(l)
wm_new = ''.join(out_lines)
anchor_3140 = ('- 3140（T4）：层位谱分离正式化——iinj 峰区 L17-19（L19 0.344）'
               'vs cinj 平台 L23-27（L38 0.977=depth-0 伪象）；own 反特异 '
               'rank 0.77-0.91=端口类第 5 次确证；co36 z=6.85 身份富集 vs '
               'co50 z=0.23 独立通路；WR PC1 剂量单调行为通路（L29 '
               '0.086→0.203→0.547）/PC2 行特定全 0=双通路分离；检索失败='
               '写入历史缺失（V2 同族模板无提升）。\n')
key = '\n## 本机环境缺陷（Windows，必读）'
if anchor_3140 not in wm_new:
    assert key in wm_new
    wm_new = wm_new.replace(
        key, '\n' + anchor_3140 + key, 1)
    changed = True
io.open(WMEM, 'w',
        encoding='utf-8').write(wm_new)
wm2 = io.open(WMEM, encoding='utf-8').read()
assert '3140（T4）：层位谱分离正式化' in wm2
assert '- max=3140' in wm2
print('MEMORY.md updated OK (changed=%s)'
      % changed)
print('CLOSEOUT REMAINING DONE')
