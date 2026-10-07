# -*- coding: utf-8 -*-
# p3154_closeout.py: Phase 3154 五写 closeout（ledger / MEMO / daily / MEMORY / 自检）
# 纪律: 幂等（重复运行 skip）、快照、BOM 二进制安全（decode->append->encode 全程保 BOM）
import os, io, json, time, hashlib, shutil

ROOT = r'D:\AI2050\Ai2050-OpenOne'
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-01.md')
MEMORY = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
TMP = os.path.join(ROOT, 'tests', 'gpt5_temp')
out = []
NOW = time.strftime('%Y-%m-%d %H:%M')

# ============ 0. 快照 ============
for src in (LEDGER, MEMO, DAILY, MEMORY):
    if os.path.exists(src):
        dst = os.path.join(TMP, 'snapshot3154_' + os.path.basename(src))
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
out.append('snapshots done')

# ============ 1. Ledger（幂等 by phase==3154） ============
led = json.load(io.open(LEDGER, encoding='utf-8'))
ms = led['measurements']
if any(m.get('phase') == 3154 for m in ms):
    out.append('ledger: 3154 already present (n=%d), skip' % len(ms))
else:
    entry = {
        'phase': 3154, 'name': 'g1p4_mfd_multifactor_disentangle',
        'seal_sha8': 'a2b92c3e', 'result_sha8': '2402f401',
        'evidence_level': 'statistical',
        'model_scope': 'qwen3-4b + qwen3-14b + glm4-9b',
        'n_rows': 208, 'prereg_id': 'p3154', 'superseded_by': None,
        'verdict': 'g1p4_fingerprint_consistent|fpmin_0.996|ho_pass_3/3|logic_sig_3/3',
        'rev_note': ('redirect: user multi-factor question -> G1-P4 (original G2-P1 -> 3155); '
                     'material 6 topics x 2 lang x 2 style x 2 logic x 4 same-char-count comma '
                     'variants =192 + 16 held-out (sleep/cold-chain); D = token distance after '
                     'comma (mean 13.5->7.7 monotone, violation 4.2%); sequential-projection '
                     'ANOVA [L,S,content24,G,D,resid] shares sum=1; KOUT shares mean '
                     '[L 5.6, S 19.4, C 57.6, G 1.16, D 0.31, R 15.9]% -> multi-factor panel '
                     'compresses 3153 within-template scatter 57-60% -> 15-17% (style = largest '
                     'namable confound, language only 5.6%); fingerprint KOUT Pearson '
                     '0.9959/0.9973/0.9996, KSTAR 0.9999/0.9996/0.9997 (gate 0.8, 3/3); logic '
                     'axis w_G intervention stat 3.756/3.706/3.221 vs 100-dir null all '
                     'p<0.0001; held-out new-topic acc 1.00/1.00/0.94 (gate 0.75, 3/3); '
                     'G-share argmax mid-layers k18/k22/k15; embedding G already 1.7-1.8%; '
                     'fixes: variant_texts hardcoded B_con (con/contra identical text -> zero '
                     'G-contrast artifact, caught fail-fast), tuple-key JSON, heldout row-space '
                     'projection -> D-space direction energy; determinism bitwise x3; '
                     'per-model res/seal: 4b 828d3cbd/e4b7e95e, 14b d376a872/eb70d12b, '
                     'glm4 ac509565/b6e232e4, summary 2402f401/a2b92c3e | disk sha8: 4b '
                     '3bf5a776, 14b d008ff39, glm4 fa5ab91a, summary 9fede5d5; npz '
                     'bb99438f/6d142b9d/cf2659f1; exec dc4a0bf9/7341fd08/545064a4/c71afba5'),
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    ms.append(entry)
    blob = json.dumps(led, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8')
    led['ledger_sha256_8'] = hashlib.sha256(blob).hexdigest()[:8]
    json.dump(led, io.open(LEDGER, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    out.append('ledger: appended 3154 (n=%d) chain_sha8=%s' % (len(ms), led['ledger_sha256_8']))

# ============ 2. MEMO（幂等 by 节标题） ============
raw = open(MEMO, 'rb').read()
txt = raw.decode('utf-8')          # 保留 BOM 字符（若有）在串首
sec_marker = '## Phase 3154: 多因素混杂分解（G1-P4）'
if sec_marker in txt:
    out.append('memo: 3154 section already present, skip')
else:
    block = r'''## Phase 3154: 多因素混杂分解（G1-P4）[__NOW__]

**主判决：`g1p4_fingerprint_consistent|fpmin_0.996|ho_pass_3/3|logic_sig_3/3`——四因素份额指纹三模型一致（KOUT 两两 Pearson 0.9959/0.9973/0.9996，门 0.8），逻辑轴 w_G 三模型干预显著（stat 3.756/3.706/3.221，100 方向 null 全 p<0.0001），held-out 新主题 16 句符号准确率 1.00/1.00/0.94 全过 0.75 门。**

### 设计与执行
- 重定向（用户 2026-10-01 指令）：方法论问题（上下文中语言/风格/逻辑/距标点距离多维权如何在神经元中分解）直接落地为 G1-P4；原 G2-P1 顺延 **Phase 3155**。预注册观测前冻结（重定向说明+设计先入 MEMO，后采集）；execution 冻结 4b `dc4a0bf9` / 14b `7341fd08` / glm4 `545064a4` / summary `c71afba5`。
- 材料：6 主题×2 语言×2 风格×2 逻辑×4 同字数移逗号变体=192 句 + held-out 16 句（T7 睡眠/T8 冷链，cut=45%）；中英 con/contra 最小对（中文字数相等、英文词数相等）；D 协变量=逗号后 token 距离（均值 13.5→11.5→9.0→7.7 严格单调，组内违例率 4.2% ≤ 0.5 门）。
- 采集：last-token 全隐层 H fp16 208×(NL+1)×D；collect sha8 4b `bb99438f` / 14b `6d142b9d` / glm4 `cf2659f1`；确定性锚 3 行重前向 **bitwise ×3**。
- 修 3 次：① design 的 tuple 键 JSON 不可序列化→字符串键；② **variant_texts 硬编码 B_con → con/contra 文本逐字相同**（G 份额恒 0、stat=0 的假象）→ tail 参数化，尾内不变式断言当场抓出 en 尾字数差 1–2（预注册口径=词数相等，记录为已声明小混杂）；③ heldout_eval 误用训练行空间 qG 基→改 D 空间 w_G 方向能量。另修 MEMO BOM 字面陷阱（`b'\ufeff'` 不是 BOM——Python bytes 字面量不支持 `\u` 转义，写入的是 6 个字面字节；已用真 BOM EF BB BF 修复头部）。

### 五发现（重复强调）
1. **多因素面板把 3153「模板内高秩散布」从 57–60% 压到读出层 15–17%**（KOUT 均值份额 [L 5.6, S 19.4, C 57.6, G 1.16, D 0.31, R 15.9]%）——**风格 S 是最大可命名混杂（17–21%）**，语言 L 仅 5.6%，resid 缩水约 3.7×；3153 判定"不可硬拆"的模板内散布约 2/3 可被命名因素吸收。
2. **份额指纹三模型高度一致**：KOUT Pearson 0.9959/0.9973/0.9996、KSTAR 0.9999/0.9996/0.9997（三对全 ≥0.8 门）→ 因素分解跨模型可迁移，非模型特异伪迹；KSTAR 残差 31–32% 高于 KOUT 15–17%（机制层残差更散，与 3153「k* 低秩尖锐 + 读出层高秩平坦」并读一致）。
3. **逻辑轴小而真实**：G 份额仅 1.1–1.3%（KOUT）/ 1.5–1.6%（K*），但 w_G（contra−con 质心差）投影干预 stat=3.756/3.706/3.221，**100 随机方向 null 无一达到（p<0.0001 ×3）**→ 因果可读；G 份额曲线 argmax 在**中层**（4b k18≈深度 50%、14b k22≈55%、glm4 k15≈38%），非读出层也非 k*；embedding 层 G 已有 1.7–1.8%（反义尾对的词汇对比在输入即编码）。
4. **完全新主题泛化**：w_G 对 held-out（睡眠/冷链，训练未见主题）16 句符号准确率 **1.00/1.00/0.94**（门 0.75）→ 逻辑一致性方向是主题无关的可泛化轴（antonym/矛盾对比）。
5. **D（标点距离）份额 0.25–0.36%** 小但三模型一致非零——逗号位置对隐状态有可测但微弱的效应；KOUT 上 S 17–21% ≫ L 5.6% → 混合语言面板里**风格比语言更耗方差**。

### 锚
4b res **828d3cbd** seal e4b7e95e（disk 3bf5a776）；14b res **d376a872** seal eb70d12b（disk d008ff39）；glm4 res **ac509565** seal b6e232e4（disk fa5ab91a）；summary res **2402f401** seal a2b92c3e（disk 9fede5d5）。collect npz：`bb99438f`/`6d142b9d`/`cf2659f1`。ledger n=**305**。产物 `phase3154\g1p4_mfd_multifactor_disentangle\{qwen3-4b,qwen3-14b,glm4,summary}\`（result*.json / collect.npz×3 / execution.json×4 / materials.json×3）。runtime 19.7/199.5/82.7/0.0s（GPU 合计约 5 min）。

### 预注册 Phase 3155：G2-P1 多关系族与算子可分离性（K2 死线，原 3154 顺延）
(1)新面板采集（GPU）：3 关系族（是-a is-a / 有-a has-a / 制成-of made-of）×(实体×类)×3 模板，qwen3-4b + qwen3-14b + glm4-9b 全隐层采集（3151 collect.npz 协议复用）；(2)K2 检验：h_ℓ(i,c) 分解为 W_ℓ v_i + φ_ℓ(c) 的可分离性（实体方向响应是否关系无关、类方向是否关系无关；双向消融+子空间角）——**死线 K2：φ_ℓ(c) 与 W_ℓ 不可分离（交互份额>50%）→ 弃"条件门"独立结构**；(3)held-out 关系泛化门：2 关系训练 → 第 3 关系预测（err ≤ 1.5× in-relation，三模型）；(4)验收：held-out 门 3 模型 + K2 份额表；若 held-out 全败 → G2 降级描述学。GPU 预算 ~25min（3 模型×738×3 行推理）。
'''
    block = block.replace('__NOW__', NOW)
    txt = txt.rstrip('\n') + '\n' + block
    open(MEMO, 'wb').write(txt.encode('utf-8'))
    out.append('memo: appended 3154 section + 3155 prereg')

# ============ 3. Daily（幂等 by marker） ============
dtxt = io.open(DAILY, encoding='utf-8').read()
d_marker = '## Phase 3154 (gpt5 线)'
if d_marker in dtxt:
    out.append('daily: 3154 already present, skip')
else:
    dblock = r'''
## Phase 3154 (gpt5 线) [__NOW__]
- 重定向落地：用户方法论问题（语言/风格/逻辑/标点距离多维权分解）→ G1-P4 多因素混杂分解；原 G2-P1 顺延 3155（预注册已写入 MEMO）。
- 主判决 **fingerprint_consistent|fpmin_0.996|ho_pass_3/3|logic_sig_3/3**：KOUT 份额均值 [L 5.6, S 19.4, C 57.6, G 1.16, D 0.31, R 15.9]%——多因素面板把 3153 模板内散布 57–60% 压到 15–17%（风格=最大可命名混杂）；逻辑轴 stat 3.756/3.706/3.221 p<0.0001×3；held-out 新主题 1.00/1.00/0.94；G argmax 中层 k18/k22/k15。
- 修 3 次（tuple 键 JSON、variant_texts 硬编码 B_con→con/contra 同文、heldout 行空间投影）+ MEMO BOM 字面陷阱（b'\ufeff' 非真 BOM）。
- 锚：4b 828d3cbd/3bf5a776、14b d376a872/d008ff39、glm4 ac509565/fa5ab91a、summary 2402f401/9fede5d5；collect bb99438f/6d142b9d/cf2659f1；ledger n=305。
'''.replace('__NOW__', NOW)
    io.open(DAILY, 'a', encoding='utf-8').write(dblock)
    out.append('daily: appended 3154')

# ============ 4. MEMORY.md（追加 G 线块, 幂等） ============
mtxt = io.open(MEMORY, encoding='utf-8').read()
m_marker = '## G 线（AGI_GPT5_MEMO）3154 状态'
if m_marker in mtxt:
    out.append('memory: G-line 3154 block already present, skip')
else:
    mblock = r'''
## G 线（AGI_GPT5_MEMO）3154 状态（2026-10-01）
- **G1-P4 多因素混杂分解闭环**：fingerprint_consistent（fpmin 0.996，KOUT/KSTAR 双层 3 对全≥0.8）；KOUT 份额 [L 5.6, S 19.4, C 57.6, G 1.16, D 0.31, R 15.9]%；**3153 模板内散布 57–60% → 15–17%**；逻辑轴 p<0.0001×3、held-out 新主题 1.00/1.00/0.94；G argmax 中层 k18/k22/k15；ledger n=**305**。权威日志=`research\gpt5\docs\AGI_GPT5_MEMO.md`（本文件 deepseek 节未动）。下一步 3155=**G2-P1 多关系族 K2 死线**（φ_ℓ(c) 与 W_ℓ 可分离性 + held-out 关系泛化门）。
'''
    io.open(MEMORY, 'a', encoding='utf-8').write(mblock)
    out.append('memory: appended G-line 3154 block')

# ============ 5. 自检 ============
led2 = json.load(io.open(LEDGER, encoding='utf-8'))
assert any(m.get('phase') == 3154 for m in led2['measurements'])
assert len(led2['measurements']) == 305, len(led2['measurements'])
assert 'ledger_sha256_8' in led2
txt2 = open(MEMO, 'rb').read().decode('utf-8')
assert sec_marker in txt2 and '预注册 Phase 3155' in txt2
assert txt2.startswith('\ufeff## AGI') or txt2.startswith('## AGI'), 'memo head broken'
assert d_marker in io.open(DAILY, encoding='utf-8').read()
assert m_marker in io.open(MEMORY, encoding='utf-8').read()
out.append('self-check: ledger n=305, memo/daily/memory markers OK, memo head OK')

io.open(os.path.join(TMP, 'p3154_closeout_out.txt'), 'w', encoding='utf-8').write('\n'.join(out))
print('CLOSEOUT OK')
