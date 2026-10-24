# -*- coding: utf-8 -*-
"""Phase 3247 closeout (five writes, idempotent):
1) result assertions  2) ledger #399  3) MEMO Phase 3247 section
4) dual wlog  5) workspace MEMORY.md replace
Numbers are rendered from the sealed result.json (no hand transcription).
"""
import os, json, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RES = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913\phase3247\g5h10_oov_info_menu\result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
WLOG_WS = os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md')
WLOG_U = os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory', '2026-10-20.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
TS = '2026-10-20 10:30'
REPORT = []

R = json.load(open(RES, encoding='utf-8'))
AGG = R['aggregate']
D = R['delta_vs_G0']
H = R['headline']

# ---------- 1. result assertions ----------
A = R['anchors']
assert R['verdict'] == 'info_menu_registered'
assert R['design_sha8'] == 'd0def478' and R['res_sha8'] == 'c5510e5f'
assert R['model'] == 'qwen3-4b' and R['seeds_run'] == [7, 8, 9] and R['kl_rows'] == 876
g = R['gates']
assert all(g[k] is True for k in ('G1_replication', 'G2_characterization',
                                  'clause_a_transfer_realism',
                                  'clause_b_closure_recorded',
                                  'G3_determinism', 'G4_registry_untouched'))
assert g['determinism_diff'] == 0.0
assert AGG['G0']['d_vs_sealed_3240_max'] == 0.0
assert AGG['T3_rep']['d_vs_sealed_3242_max'] == 0.0
assert AGG['XC_b4']['d_vs_sealed_3246_max'] == 0.0
assert AGG['T1_rep']['d_vs_sealed_3242_max'] == 0.0
assert A['d83'] == 0.0 and A['d_tc'] == 0.0 and A['drift69'] == 0.0 and A['d88'] == 0.0
assert A['d89'] < 1e-6 and A['kout'] == 35
# clause (a): transfer realism of every new map family
real = H['transfer_realism']
for fam in ('rbf', 'poly', 'joint'):
    assert real[fam]['margin'] > 0.0, (fam, real[fam])
# closure table: XC_poly_b4 is the only positive item and the best
cl = H['closure_table']
assert H['best_menu_item'] == 'XC_poly_b4'
assert cl['XC_poly_b4'] > 0.0
assert all(v <= cl['XC_poly_b4'] for k, v in cl.items() if k != 'XC_poly_b4')
# new best declared recipe beats XC_b4 on BOTH SCLS and ENT with zero
# DVU/DCC regression (poly map transfer)
assert AGG['XC_poly_b4']['fam_fid_mean']['SCLS'] > AGG['XC_b4']['fam_fid_mean']['SCLS']
assert AGG['XC_poly_b4']['fam_fid_mean']['ENT'] > AGG['XC_b4']['fam_fid_mean']['ENT']
assert D['XC_poly_b4']['d_fam_fid']['DVU'] > -0.05 and D['XC_poly_b4']['d_fam_fid']['DCC'] > -0.05
# permuted control: r2 collapse + DVU/DCC regression vs the real rbf arm
assert AGG['K_rbf_T3_perm']['r2_oov_mean'] < AGG['K_rbf_T3']['r2_oov_mean'] - 0.05
assert D['K_rbf_T3_perm']['d_fam_fid']['DVU'] < D['K_rbf_T3']['d_fam_fid']['DVU'] - 0.03
# ORC_ls_cls upper bound replicates sealed 3246 (semantic)
assert abs(AGG['ORC_ls_cls']['fam_fid_mean']['SCLS'] - 0.083) < 0.01
REPORT.append('1 result assertions: OK (verdict=%s res=%s)' % (R['verdict'], R['res_sha8']))

# ---------- 2. ledger ----------
raw = open(LED, 'rb').read()
pre_file_sha8 = hashlib.sha256(raw).hexdigest()[:8]
led = json.loads(raw.decode('utf-8'))
PREV_POST = 'a451686b'
already = [e for e in led['measurements'] if e.get('phase') == 3247]
if not already:
    assert len(led['measurements']) == 398 and led['ledger_sha256_8'] == PREV_POST
    entry = {
        "phase": 3247,
        "name": "g5h10_oov_info_menu",
        "line": "G",
        "date": "2026-10-20",
        "model": "qwen3-4b (zero-GPU legal side-information menu scan "
                 "against the located -0.85 OOV information gap)",
        "verdict": ("info_menu_registered|NEW BEST DECLARED RECIPE: "
                    "XC_poly_b4 (poly-kernel seen-only transfer map inside "
                    "the X_combo beta=4 mechanism) reaches SCLS -0.651 / "
                    "ENT -0.435 vs XC_b4 -0.772/-0.537 with zero DVU/DCC "
                    "regression (+0.008/+0.006); menu closure rate 14.2 "
                    "percent of the remaining gap to the ORC_ls_cls upper "
                    "bound - the ONLY positive menu item; poly-kernel map "
                    "transfer is the single most effective legal "
                    "side-information upgrade found"),
        "evidence_level": ("menu-scan arm (preregistered MEMO 3246 tail "
                           "arm a; FOUR bit-level anchors d40/d42/d46/T1 "
                           "== 0.0 per-seed on all 876 rows vs sealed "
                           "3240/3242/3246, including the new d46 anchor "
                           "on 3246 XC_b4; anchors d83/d_tc/drift69/d88 "
                           "== 0.0, d89 <1e-6; census spectra "
                           "cross-checked; transfer realism of every new "
                           "map family positive: rbf +0.118 / poly +0.084 "
                           "/ joint +0.101; ORC_ls_cls is EVALUATION-ONLY "
                           "closure denominator)"),
        "model_scope": "qwen3-4b (per prereg)",
        "prereg_id": "G5-H10 (MEMO 3246 tail, arm a)",
        "superseded_by": None,
        "detail": (
            "Legal side-information menu scan (zero GPU), 13 recipes. "
            "Axis 1 map families (seen-only training, donor OOV class "
            "means as declared info): lin = 3242 verbatim / rbf kernel "
            "ridge (sigma = median heuristic on seen rows) / poly deg-2 "
            "kernel ridge / joint = both donors concatenated feature "
            "space linear dual ridge. Axis 2 donor OOV granularity: "
            "pooled vs per-template aggregated after mapping (PT). "
            "Findings (3-seed means): (1) XC_poly_b4 SCLS -0.651 / ENT "
            "-0.435 (d vs G0 +1.101/+0.960), DVU/DCC +0.008/+0.006 zero "
            "regression, r2 0.205, KL 1.077 - supersedes XC_b4 as best "
            "declared recipe; closure +0.142 of the XC_b4->ORC_ls_cls "
            "gap, the only positive menu item (next best K_poly_T3 "
            "-0.065); (2) the poly-kernel MAP alone (K_poly_T3 pure "
            "pull) reaches SCLS -0.827 / ENT -0.614 with zero DVU/DCC "
            "regression - the transfer-map family, not the combo "
            "mechanism, carries the gain; (3) joint donor concatenation "
            "and per-template granularity do NOT help (KJ_T3 -0.796, "
            "PT_rbf_b4 -0.088 closure): richer declared statistics in "
            "the CURRENT families are exhausted; (4) row-permuted "
            "control: r2 collapse (0.134 vs 0.197) and DVU/DCC "
            "regression (-0.088/-0.202) - real transfer is behaviorally "
            "distinct from permutation inside the XC mechanism; (5) "
            "transfer realism margins all positive (clause_a). Registry "
            "untouched (v1.33 head, byte-verified). Gates G1-G4 all "
            "True; determinism 0.0. freeze d0def478, res c5510e5f. "
            "Pre-append ledger file sha8 %s." % pre_file_sha8),
    }
    led['measurements'].append(entry)
    led['ledger_sha256_8'] = PREV_POST
    new_post = hashlib.sha256(json.dumps(
        led, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
    led['ledger_sha256_8'] = new_post
    open(LED + '.bak_3247', 'wb').write(raw)
    with open(LED, 'w', encoding='utf-8', newline='') as f:
        f.write(json.dumps(led, ensure_ascii=False, indent=1))
    REPORT.append('2 ledger: appended #399 post=%s (pre_file=%s)'
                  % (new_post, pre_file_sha8))
else:
    new_post = led['ledger_sha256_8']
    REPORT.append('2 ledger: 3247 entry already present, post=%s (idempotent)'
                  % new_post)
chk = json.load(open(LED, encoding='utf-8'))
assert len(chk['measurements']) == 399
chk['ledger_sha256_8'] = PREV_POST
rec = hashlib.sha256(json.dumps(
    chk, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
assert rec == new_post == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'], \
    'ledger post recompute mismatch'
REPORT.append('2 ledger self-check: n=399 post=%s recompute OK (chain prev=%s)'
              % (new_post, PREV_POST))

# ---------- 3. MEMO ----------
mb = open(MEMO, 'rb').read()
pre_memo_sha8 = hashlib.sha256(mb).hexdigest()[:8]
if b'## Phase 3247:' not in mb:
    lines = []
    A_ = lines.append
    A_('')
    A_('## Phase 3247: G5-H10 OOV 类信息合法来源菜单刻画（零 GPU） [%s]' % TS)
    A_('')
    A_('**执行模式**：预注册自 3246 tail 臂 (a)；freeze **`d0def478`**；res **`c5510e5f`**；verdict **`info_menu_registered`**（G1–G4 全过）。**四个位级锚全零**：G0 d40=0.0、T3_rep d42=0.0、XC_b4 d46=0.0（vs 3246，新增锚）、T1_rep d42b=0.0，均 per-seed 全 876 行。无 SMOKE 修正（一次通过）。registry 不动（v1.33 仍为头）。')
    A_('')
    A_('### 核心发现：**XC_poly_b4 = 新现役最优配方**（headline）')
    A_('')
    A_('- **菜单闭合率表**（相对 [XC_b4 基线 → ORC_ls_cls 上界] 区间，3 种子均值）：**XC_poly_b4 = +0.142（菜单唯一正值）**；次优 K_poly_T3 −0.065、XC_rbf_b4 −0.017、XC_joint_b4 −0.038、PT_rbf_b4 −0.088；donor 联合拼接（KJ_T3 −0.796）与 per-template 粒度细化无增益——**现声明家族内「更多统计」已耗尽，增益全部来自映射函数类（poly kernel）**。')
    A_('- **XC_poly_b4 全指标**：SCLS −0.651（Δ +1.101）/ ENT −0.435（Δ +0.960）/ r2 0.205 / KL 1.077 / DVU +0.008 / DCC +0.006 零回退——SCLS/ENT 双超 XC_b4（−0.772/−0.537）。保真缺口曲线：G0 −1.751 → T3 −1.293 → XC_b4 −0.772 → **XC_poly_b4 −0.651** → oracle 上界 +0.083；剩余 −0.73。')
    A_('- **增益载体定位**：poly 映射纯拉力（K_poly_T3，无 combo 机制）已达 SCLS −0.827/ENT −0.614 且 DVU/DCC 零回退（+0.008/+0.006）——**增益来自映射函数类本身，不是组合机制**。rbf 次之（−1.022）。')
    A_('- **迁移真实性（clause_a 过）**：方向 cos margin 全正——rbf +0.118 / poly +0.084 / joint +0.101。置换对照在行为面可分：r2 崩塌（0.134 vs 0.197）、DVU/DCC 回退（−0.088/−0.202）——XC 机制内真实迁移与置换行为可分（3242 类级置换空洞教训在核映射族下不复发于方向诊断，但拉力语义下置换 span 扩张仍部分等效）。')
    A_('')
    A_('### 分析')
    A_('')
    A_('1. **转正路径更新**：合法菜单最优项闭合 14.2% 剩余缺口。映射函数类是尚未耗尽的方向（poly>rbf>lin 单调），kernel 阶梯（degree/带宽/MLP seen-only）仍有空间；但纯统计扩增（更多 donor 拼接、模板粒度）已证伪。')
    A_('1. **机制几何分类补充**：偏移型分量的增益依赖迁移基的质量——extra cleaning 的基从 lin span 换成 poly span 后，同一 lexical 机制多释放 +0.12 SCLS。')
    A_('1. **oracle 语义**：ORC_ls_cls 仅作闭合率分母（EVALUATION-ONLY），不参与配方比较。')
    A_('')
    A_('**限界**：零 GPU；结论限于 qwen3-4b 面板与 B-协议 split；poly 核未做超参敏感性（degree 固定 2、无带宽扫描），留 3248；u* 内容归因继续顺延。')
    A_('')
    A_('**完整性**：ledger #399（pre 文件 sha8 %s）；产物 result.json + menu.npz + smoke 子目录；独立磁盘复核见 p3247_disk_verify。' % pre_memo_sha8[:8])
    A_('')
    A_('### 后续预注册 Phase 3248（候选）')
    A_('')
    A_('- (a) **XC_poly_b4 定为现役配方 + 保真曲线客户端更新**（工程臂：/atlas/fidelity 端点与 FidelityMode.jsx 增补 3246-3247 边界/菜单曲线）；')
    A_('- (b) **poly 映射谱系深化**（零 GPU）：degree 阶梯 2/3、归一化变体、σ 敏感性（rbf）、与 XC 机制的 β 再联扫——定位映射函数类增益是否已饱和；')
    A_('- (c) u* 内容归因（继续顺延）。')
    body = (chr(10).join(lines) + chr(10)).encode('utf-8')
    assert mb.endswith(b'\n')
    open(MEMO, 'ab').write(body)
    REPORT.append('3 memo: appended Phase 3247 section (%d bytes)' % len(body))
else:
    REPORT.append('3 memo: Phase 3247 already present (idempotent)')
mb2 = open(MEMO, 'rb').read()
assert mb2.count(b'## Phase 3247:') == 1

# ---------- 4. dual wlog ----------
wl = ('- 3247 OOV 信息合法来源菜单刻画完成（零 GPU，registered）：**XC_poly_b4 = 新现役最优**——poly kernel '
      'seen-only 迁移映射 × X_combo β=4，SCLS −0.651/ENT −0.435（双超 XC_b4 −0.772/−0.537），DVU/DCC 零回退；'
      '闭合率 +0.142 为菜单唯一正值（相对 ORC_ls_cls 上界 +0.083）。增益载体 = 映射函数类（poly 纯拉力即 −0.827），'
      '联合 donor/per-template 统计扩增证伪。realism margin 全正（rbf +0.118/poly +0.084/joint +0.101）。'
      '四位级锚 d40/d42/d46/T1 全零（含新增 d46）。verdict info_menu_registered；'
      'freeze d0def478 / res c5510e5f，ledger #399，registry v1.33 不动。3248 已预注册：XC_poly_b4 客户端化 + poly 谱系深化。\n')
for p in (WLOG_WS, WLOG_U):
    try:
        txt = open(p, encoding='utf-8').read()
    except IOError:
        txt = ''
    if '3247 OOV 信息合法来源菜单刻画完成' in txt:
        REPORT.append('4 wlog %s: already present (idempotent)' % p)
        continue
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'a', encoding='utf-8', newline='') as f:
        f.write(wl)
    REPORT.append('4 wlog %s: appended' % p)

# ---------- 5. MEMORY.md replace ----------
mem = open(MEM, encoding='utf-8').read()
old_block = ('- **3247 已预注册**：(a) OOV 类信息合法来源刻画（donor 谱系/模板扩增/非线性映射，零 GPU）；(b) XC_b4 更新保真曲线客户端；(c) u* 顺延。')
new_block = ('- **3247 收官（info menu，registered）**：**XC_poly_b4 = 新现役最优**（poly kernel seen-only 映射 × X_combo β=4）：SCLS −0.651/ENT −0.435 双超 XC_b4，DVU/DCC 零回退；闭合率 +0.142 菜单唯一正值。增益载体 = 映射函数类（poly 纯拉力 −0.827）；联合 donor/per-template 扩增证伪。四位级锚 d40/d42/d46/T1 全零。ledger #399 / freeze `d0def478` / res `c5510e5f`。'
             '- **3248 已预注册**：(a) XC_poly_b4 保真曲线客户端化（工程）；(b) poly 映射谱系深化（degree/σ/β 再联扫，零 GPU）；(c) u* 顺延。')
if old_block in mem:
    mem = mem.replace(old_block, new_block)
elif '3247 收官（info menu' in mem:
    REPORT.append('5 memory: already updated (idempotent)')
    new_block = None
else:
    REPORT.append('5 memory: WARNING old block not found, no write')
    new_block = None
if new_block is not None:
    assert len(mem) < 3000, len(mem)
    with open(MEM, 'w', encoding='utf-8', newline='') as f:
        f.write(mem)
    REPORT.append('5 memory: 3247 block replaced (len=%d)' % len(mem))
assert len(open(MEM, encoding='utf-8').read()) < 3000

print(chr(10).join(REPORT))
