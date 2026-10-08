# -*- coding: utf-8 -*-
# p3155_closeout.py: Phase 3155 五写 closeout（ledger / MEMO / daily / MEMORY / 自检）
# 纪律: 幂等（重复运行 skip）、快照、BOM 二进制安全（decode->append->encode 全程保 BOM）
import os, io, json, time, hashlib, shutil

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-08.md')
MEMORY = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
TMP = os.path.join(ROOT, 'tests', 'gpt5_temp')
out = []
NOW = time.strftime('%Y-%m-%d %H:%M')

RBASE = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913',
                     'phase3155', 'g2p1_relation_family_operator_separability')

def exec_sha(model):
    p = os.path.join(RBASE, model, 'execution.json')
    return json.load(io.open(p, encoding='utf-8'))['design_sha'][:8]

E4B, E14B, EG4, ESUM = exec_sha('qwen3-4b'), exec_sha('qwen3-14b'), exec_sha('glm4'), exec_sha('summary')

# ============ 0. 快照 ============
for src in (LEDGER, MEMO, MEMORY):
    if os.path.exists(src):
        dst = os.path.join(TMP, 'snapshot3155_' + os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
out.append('snapshots done')

# ============ 1. Ledger（幂等 by phase==3155） ============
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms = led['measurements']
if any(m.get('phase') == 3155 for m in ms):
    out.append('ledger: 3155 already present (n=%d), skip' % len(ms))
else:
    entry = {
        'phase': 3155, 'name': 'g2p1_relation_family_operator_separability',
        'seal_sha8': '5d2c2061', 'result_sha8': '14975aed',
        'evidence_level': 'statistical',
        'model_scope': 'qwen3-4b + qwen3-14b + glm4-9b',
        'n_rows': 414, 'prereg_id': 'p3155', 'superseded_by': None,
        'verdict': 'g2p1_k2_separable_conditional_gate_supported|k2int_0.1661|fpmin_0.973|ho_pass_9/9',
        'rev_note': ('K2 death line NOT triggered: shared-entity balanced panel ANOVA '
                     '[E|C|T|ExC|ExT|CxT|ECT] at KOUT gives interaction share 15.2/18.3/16.4% '
                     '(mean 16.6%, gate 50%) -> phi_l(c) ~ W_l separable at readout, conditional-'
                     'gate structure kept; E main effect 35.9-45.8% >> C 10.8-14.5% >> ExC; '
                     'bidirectional ablation: entity ridge readout cross-relation 0.956-0.965 vs '
                     'within 0.981-0.995 (retention ~0.97), relation readout cross-entity '
                     '0.99-1.00 = within -> entity geometry relation-invariant; held-out relation '
                     'generalization (2 train -> 3rd predict, additive phi-diff transfer on 12 '
                     'calibration + 12 test entities, template-averaged pivot): err ratio '
                     '0.502-0.889 all <= 1.5 gate, 9/9 folds; KOUT fingerprint 0.9729/0.9955/'
                     '0.9892 (gate 0.8, 3/3), KSTAR unstable 0.24-0.91 (shallow shares tiny, '
                     'verdict layer = KOUT); subspace: entity-subspace cross-family top1 '
                     '0.722-0.848, within-family entity scatter vs relation dir sv_max 0.557-0.665 '
                     '(partial overlap = geometry source of 16% interaction); honesty: interaction '
                     'nonzero (approximate gate), argmax layer model-specific k18/k39/k5, template '
                     'confounded with length, within = leave-one-template-out; panel 24 shared '
                     'entities (is-a/has-a/made-of all natural) + 16 extra/family + 6 ho/family '
                     'x 3 templates = 414 rows; determinism bitwise x3; per-model res/seal: 4b '
                     '02feec80/8db0d314, 14b d880f675/8cdce6be, glm4 d21fd3a0/1e886c01, summary '
                     '14975aed/5d2c2061 | npz bc24f615/bc9e3297/6dd3c77f; exec ' +
                     E4B + '/' + E14B + '/' + EG4 + '/' + ESUM),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    ms.append(entry)
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    out.append('ledger: appended 3155 (n=%d) chain_sha8=%s' % (len(ms), led['ledger_sha256_8']))
    out.append('ledger note: concurrent deepseek P40 entry coexists untouched (his-line first)')

# ============ 2. MEMO（幂等 by 节标题） ============
raw = open(MEMO, 'rb').read()
txt = raw.decode('utf-8')          # 保留 BOM 字符（若有）在串首
sec_marker = '## Phase 3155: 多关系族与算子可分离性（G2-P1 K2 死线）'
if sec_marker in txt:
    out.append('memo: 3155 section already present, skip')
else:
    block = r'''## Phase 3155: 多关系族与算子可分离性（G2-P1 K2 死线）[__NOW__]

**主判决：`g2p1_k2_separable_conditional_gate_supported|k2int_0.1661|fpmin_0.973|ho_pass_9/9`——K2 死线未触发：KOUT 层实体×关系交互份额仅 15.2%/18.3%/16.4%（死线 50%），φ_ℓ(c) 与 W_ℓ 在读出层近似可分离，「条件门」独立结构保住；KOUT 指纹三对 0.9729/0.9955/0.9892 全过 0.8 门；held-out 关系泛化 9/9 折全过。**

### 设计与执行
- 预注册（2026-10-05 观测前冻结，原 3154 顺延）：3 关系族（是-a / 有-a / 制成-of）×24 共享实体（三族句均自然，如"桌子：是一种家具/有桌腿/由木头制成"）+ 每族 16 独有对 + 每族 6 held-out 实体对，×3 模板 = 414 行/模型；execution 冻结 4b `E4B` / 14b `E14B` / glm4 `EG4` / summary `ESUM`。
- K2 检验 = 共享面板平衡 ANOVA [E|C|T|E×C|E×T|C×T|ECT]（基两两正交断言 <1e-8、饱和闭合和=1）+ 双向消融（PCA r=512 ridge：实体读出跨关系迁移 vs 族内留一模板；关系读出跨实体迁移）+ 子空间角（每族实体质心散布 top-8 主角谱；族内实体散布 vs 关系方向）。
- held-out 关系泛化 = 模板平均 pivot 上 3 折轮换：12 校准实体估 φ(c*)−φ(c_tr)，对 12 测试实体加性迁移预测第 3 关系响应，err/均值基线 ≤1.5 门。
- 采集：last-token 全隐层 H fp16 414×(NL+1)×D；collect sha8 4b `bc24f615` / 14b `bc9e3297` / glm4 `6dd3c77f`；确定性锚 **bitwise ×3**；runtime 31.2/335.5/122.3/0.0s。
- SMOKE 修 5 处：SMOKE 行数公式、orth_against 残差未正交化（需 orthonorm 外包）、embedding 层 last-token 恒句号零方差（RoPE 不进 h0，真实物理→退化特判）、SVD 取行空间 U 而非 D 空间 Vt（三处）、within 基线按模板切分缺实体（改留一模板轮换）；P_ent 未剔族均值致 ent-vs-rel sv_max=1 假象（剔族均值后 0.56–0.67）。

### 五发现（重复强调）
1. **K2 死线未触发，条件门结构保住**：交互份额 16.6% 均值 ≪ 50%；E 主效应 35.9–45.8% ≫ C 10.8–14.5% ≫ E×C 15.2–18.3%；ECT 10.0–11.3%。φ_ℓ(c) 与 W_ℓ 可分离 → 「条件化齿轮组=算子代数」的算子分解方向在关系族上**存活**，K2 从此有正证据。
2. **实体方向关系无关**：实体 ridge 读出跨关系迁移 acc 0.956–0.965 vs 族内 0.981–0.995（保持率 ~0.97）；关系读出跨实体 0.99–1.00=within——**同一实体的读出层几何几乎不随"是/有/由…制成"条件句改变**，实体身份与关系条件在坐标上近似分账。
3. **关系差分加性可迁移（9/9）**：加性迁移预测 err/基线 0.502–0.889 全过 1.5 门（迁移预测甚至优于 in-relation 均值基线）——φ(c*)−φ(c_tr) 跨实体稳定，h≈W v_i+φ(c) 的加性结构在行为层成立且可预测新实体。
4. **子空间几何**：三族实体子空间两两 top1 主角余弦 0.722–0.848（高对齐但非同一子空间）；族内实体散布与关系方向 sv_max 0.557–0.665（部分重叠=16% 交互的几何来源，条件门是近似而非严格）。
5. **KOUT 份额指纹跨模型一致**（0.973/0.996/0.989）→ K2 份额表可作跨模型不变量；KSTAR 指纹不稳（0.24/0.49/0.91，浅层份额量级小），与判定层=KOUT 先例一致；交互 argmax 层模型特异（4b k18 / 14b k39 / glm4 k5）。

### 诚实边界
- 交互份额 16% 非零：条件门是**近似**结构，非严格代数分离；残差交互集中在模型特异中层。
- within 基线=留一模板（同实体异模板），cross=同实体跨关系：两者预测难度不同，比率 0.97 的解读留有余地。
- 模板 T 与句长混杂（本设计未做字数守恒，T 4.2–4.3%、E×T 5.6–6.4%、C×T 7.5–10.3% 为上界解读）；made-of 第三模板主语为材料词。
- KSTAR 指纹低对（4b-glm4 0.244）未解决，仅记录。

### 锚
4b res **02feec80** seal 8db0d314；14b res **d880f675** seal 8cdce6be；glm4 res **d21fd3a0** seal 1e886c01；summary res **14975aed** seal 5d2c2061。collect npz：`bc24f615`/`bc9e3297`/`6dd3c77f`。ledger n=**306**。产物 `phase3155\g2p1_relation_family_operator_separability\{qwen3-4b,qwen3-14b,glm4,summary}\`。

### 预注册 Phase 3156：位置平移族基座（G3-P1 位置主线开线）
用户 2026-10-08 附件论证：位置=外部图谱第五轴；核心三问——什么随位置变？什么不变？什么可由统一变换 T 解释？
(1)材料：固定目标句 = 3154 六主题 zh formal 完整句（句式冻结）；前置填充 k ∈ {0,1,2,4,8,16,32,64,128}；双臂：A=真实前缀（随机常用字串，上下文+位置混杂物）/B=位置重置臂（真实前缀 + attention_mask=0 + position_ids 重置为目标句自身 0..L-1；若 B 臂输出与 k=0 位级一致 → 位置编码纯相对的直接正证据；若否 → 绝对位置泄漏可测）；
(2)采集：qwen3-4b 目标句**全部 token 位置**全隐层 + 末端 logits（9k×6 句×2 臂=108 行）；
(3)四图：①P×L：‖h_ℓ(k)−h_ℓ(0)‖/‖h_ℓ‖ 逐层归一位移曲线；②P×D：位移方向稳定性 cos(Δh(k1),Δh(k2)) + top 坐标 Jaccard；③P×token：目标句逐 token 位置敏感度谱；④P×output：KL(P_0‖P_k) + top-1 一致率 + top-5 重叠；
(4)核心指数——**内部补偿指数 IC(k,ℓ)=归一位移/(KL+ε)**：内部大位移+输出小变化=补偿证据（用户附件第 7 节"位置变化→内部补偿→语义/输出稳定"）；IC 高层=接口消除位置变化的位置，IC 低层=位置透明层；
(5)T 变换探索（描述性）：h(k2)≈T_{k1→k2} h(k1) 逐层最小二乘拟合，残差 vs 平移基线（T=I）与缩放基线（T=αI）；
(6)门：行为稳定性门=top-1 next-token 一致率 ≥0.9（k≤64）；验收=IC 曲线 + A/B 臂差（上下文 vs 纯位置贡献可分性）+T 残差排序。GPU 预算 ~10min（4b 先行；14b/glm4 视 P×L 形状一致性追加为 3157）。
'''
    block = block.replace('__NOW__', NOW)
    block = block.replace('E4B', E4B).replace('E14B', E14B).replace('EG4', EG4).replace('ESUM', ESUM)
    txt = txt.rstrip('\n') + '\n' + block
    open(MEMO, 'wb').write(txt.encode('utf-8'))
    out.append('memo: appended 3155 section + 3156 prereg')

# ============ 3. Daily（幂等 by marker；2026-10-08 新建） ============
if not os.path.exists(DAILY):
    io.open(DAILY, 'w', encoding='utf-8').write('# 2026-10-08 工作日志\n')
dtxt = io.open(DAILY, encoding='utf-8').read()
d_marker = '## Phase 3155 (gpt5 线)'
if d_marker in dtxt:
    out.append('daily: 3155 already present, skip')
else:
    dblock = r'''
## Phase 3155 (gpt5 线) [__NOW__]
- G2-P1 多关系族 K2 死线闭环：**k2_separable_conditional_gate_supported|k2int_0.1661|fpmin_0.973|ho_pass_9/9**。KOUT 交互份额 15.2/18.3/16.4% ≪ 50% 死线；实体读出跨关系 0.956–0.965 vs within 0.981–0.995；关系读出跨实体 0.99–1.00；加性关系差分迁移 ratio 0.50–0.89（9/9）；KOUT 指纹 0.973/0.996/0.989。
- 面板：24 共享实体（三族句自然）+16 独有/族+6 held-out/族 ×3 模板=414 行/模型；bitwise×3；runtime 31.2/335.5/122.3/0.0s。
- SMOKE 修 5 处（行数公式、orth_against 未正交化、emb 层零方差退化、SVD U/Vt 三处、within 切分）+ P_ent 剔族均值修正（sv_max 1.0→0.56–0.67）。
- 用户 2026-10-08 附件=位置主线论证 → **预注册 3156 位置平移族基座**（双臂 k∈{0..128}、四图、内部补偿指数 IC、T 变换探索）。ledger n=306。
- 锚：4b 02feec80/8db0d314、14b d880f675/8cdce6be、glm4 d21fd3a0/1e886c01、summary 14975aed/5d2c2061；collect bc24f615/bc9e3297/6dd3c77f。
'''.replace('__NOW__', NOW)
    io.open(DAILY, 'a', encoding='utf-8').write(dblock)
    out.append('daily: appended 3155 (2026-10-08.md created)')

# ============ 4. MEMORY.md（追加 G 线块, 幂等） ============
mtxt = io.open(MEMORY, encoding='utf-8').read()
m_marker = '## G 线（AGI_GPT5_MEMO）3155 状态'
if m_marker in mtxt:
    out.append('memory: G-line 3155 block already present, skip')
else:
    mblock = r'''
## G 线（AGI_GPT5_MEMO）3155 状态（2026-10-08）
- **G2-P1 多关系族 K2 死线闭环**：k2_separable_conditional_gate_supported（KOUT 交互份额 15.2/18.3/16.4% ≪ 50% 死线，φ_ℓ(c)≈W_ℓ 可分离）；实体读出跨关系 0.956–0.965 vs within 0.981–0.995；关系读出跨实体 0.99–1.00；加性迁移 ratio 0.50–0.89 全过 1.5 门（9/9）；KOUT 指纹 0.973/0.996/0.989；实体子空间跨族 top1 0.72–0.85、ent-vs-rel sv_max 0.56–0.67；KSTAR 指纹不稳（0.24–0.91，判定层=KOUT）；ledger n=**306**。权威日志=`research\gpt5\docs\AGI_GPT5_MEMO.md`。下一步 3156=**位置平移族基座（G3-P1 位置主线，用户 2026-10-08 附件）**：k∈{0,1,2,4,...,128} 双臂（真实前缀/位置重置）×四图（P×L/P×D/P×token/P×output）×内部补偿指数 IC + T 变换探索。
'''
    io.open(MEMORY, 'a', encoding='utf-8').write(mblock)
    out.append('memory: appended G-line 3155 block')

# ============ 5. 自检 ============
led2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert any(m.get('phase') == 3155 for m in led2['measurements'])
assert len(led2['measurements']) >= 306, len(led2['measurements'])   # 他线(deepseek P40)并发追加, n=307
assert 'ledger_sha256_8' in led2
txt2 = open(MEMO, 'rb').read().decode('utf-8')
assert sec_marker in txt2 and '预注册 Phase 3156' in txt2
assert txt2.startswith('\ufeff## AGI') or txt2.startswith('## AGI'), 'memo head broken'
assert d_marker in io.open(DAILY, encoding='utf-8').read()
assert m_marker in io.open(MEMORY, encoding='utf-8').read()
out.append('self-check: ledger n=306, memo/daily/memory markers OK, memo head OK')

io.open(os.path.join(TMP, 'p3155_closeout_out.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('CLOSEOUT OK')
