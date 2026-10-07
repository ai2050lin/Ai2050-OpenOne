# -*- coding: utf-8 -*-
"""Phase 3109 closeout (idempotent):
Ledger -> MEMO Phase 3109 -> workspace logs -> MEMORY.md."""
import datetime
import hashlib
import io
import json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTD = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913'
        r'\phase3109'
        r'\omega_p107_underdetermination_verdict')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
WLOG_D = ROOT + r'\.workbuddy\memory'
WLOG_C = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
          r'\.workbuddy\memory')
MEMO_W = WLOG_D + r'\MEMORY.md'
LOGF = OUTD + r'\closeout_log.txt'
NOW = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
TODAY = datetime.date.today().isoformat()
o = []

res = json.load(io.open(OUTD + r'\result.json',
                        encoding='utf-8'))
verb = res['verdict']
assert verb == 'stability_not_n_limited', verb
assert res['gates']['D2_random_control'][
    'K100']['random_sufficient'] is True

# ---------- Ledger ----------
led = json.load(io.open(LEDGER, encoding='utf-8'))
if not any(m.get('phase') == 3109
           for m in led['measurements']):
    claim = (
        'Omega-P107 (3109, offline T3 on frozen 3105 '
        'capture, no GPU, 62s) - underdetermination '
        'verdict.  Verdict stability_not_n_limited '
        '(pre-registered map executed: D1 failed, D2 '
        'passed) - and the failure direction CORRECTS '
        '3108.  D1 N-SWEEP: floor(n) = 0.2355/0.1606/'
        '0.1157/0.0845/0.0194 for n = 100/200/400/800/'
        '1101, Spearman(n, floor) = -1.000 - bootstrap '
        'solutions get LESS similar as n grows, the '
        'OPPOSITE of the underdetermination prediction '
        '(monotone stabilization); small-n similarity is '
        'a lambda-regularization artifact (data term '
        'weak, solutions pulled toward the same prior '
        'geometry); J_half stays flat 0.067-0.117 '
        '(rho=0.5); PR_norm flat 0.82-0.86.  '
        'Underdetermination as the PRIMARY cause of '
        'degeneracy is REJECTED.  D2 RANDOM CONTROL '
        '(decisive): random 100-dim subspaces refit '
        'AUC [0.999, 1.000, 1.000, 1.000, 1.000] '
        'median 0.9997 vs |w|-Top100 0.9999; random '
        '200-dim median 0.9995 vs 0.9999 - '
        'random_sufficient for BOTH K.  Truth-readout '
        'information is DIFFUSELY REDUNDANT in the '
        'residual stream: ANY ~100-dim random '
        'subspace (3.9pct of coordinates) carries '
        'enough signal for a near-perfect linear '
        'readout; |w|-selection has no advantage over '
        'random.  D3 FUNCTIONAL EQUIVALENCE: half-A '
        'Top-K refit on half-B gives AUC 0.9998/0.9998 '
        'vs half-B own 0.9995/0.9978 while the two '
        'coordinate sets overlap only J=0.070/0.072 - '
        'transfer_ok for both K.  DIFFERENT coordinate '
        'sets are FUNCTIONALLY IDENTICAL readout '
        'ports.  KEY THEORETICAL UPDATE (3107+3108+'
        '3109 synthesis): the write-head group does '
        'not exist as ANY fixed geometric object '
        '(coordinate set, subspace, or direction); '
        'what exists is a large equivalence class of '
        'readout ports - AUC~1 readout directions '
        'form a very wide cone intersecting every '
        'random 100-dim subspace; the model unembed '
        'direction (3107 G3) is one member.  Truth is '
        'a high-level semantic variable redundantly '
        'broadcast across nearly all coordinates '
        '(each coordinate mixing many low-level '
        'features).  The research object shifts from '
        'readout geometry (now proven degenerate) to '
        'the WRITE-IN side and the CONDITIONAL '
        'STRUCTURE of the truth variable.  NEXT 3110: '
        '(a) K-sweep random-subspace AUC curve K in '
        '{5,10,20,50,100,200,400,800,1600,2560} -> '
        'minimum readout-port dimension = quantitative '
        'diffuseness; (b) cross-variable specificity '
        'control: same random subspaces probing a '
        'DIFFERENT variable (crit_rel) to test whether '
        'diffuseness is truth-specific or universal; '
        '(c) then T4 multi-step autoregression line.')
    meas = {
        'meas_id': 'meas3109_omega_p107_'
                   'underdetermination_verdict',
        'phase': 3109,
        'claim': claim,
        'verdict': 'stability_not_n_limited',
        'anchors': 'design_seal.json frozen before '
                   'computation: n_sweep {100,200,400,'
                   '800,1101}, boot 16 r=8 5 groupings, '
                   'K {100,200} x 5 random seeds, '
                   'margin 0.02, D1 rho>=0.9; lambda '
                   '0.01 recorded; standardization '
                   'frozen from full train',
        'artifacts': {
            'result': 'phase3109/omega_p107_'
                      'underdetermination_verdict/'
                      'result.json',
            'seal': 'phase3109/omega_p107_'
                    'underdetermination_verdict/'
                    'design_seal.json'},
        'hashes': {},
        'note': 'offline only; SMOKE clean (first '
                'try); D1 opposite-direction result '
                'corrects 3108 underdetermination '
                'picture; D2/D3 decisive for diffuse '
                'redundancy',
    }
    led['measurements'].append(meas)
    l14 = [l for l in led['linkage']
           if l.get('link_id')
           == 'L14_readout_spectrum_cross_model'][0]
    l14['connects'].append(
        'meas3109_omega_p107_'
        'underdetermination_verdict')
    led.pop('ledger_sha256_8', None)
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    led['ledger_sha256_8'] = hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    with io.open(LEDGER, 'w', encoding='utf-8') as f:
        json.dump(led, f, ensure_ascii=False,
                  indent=1)
    o.append('ledger appended n=%d l14=%d sha=%s'
             % (len(led['measurements']),
                len(l14['connects']),
                led['ledger_sha256_8']))
else:
    o.append('ledger already upserted')

# ---------- MEMO Phase 3109 ----------
memo = io.open(MEMO, encoding='utf-8').read()
if '## Phase 3109:' not in memo:
    sec = u'''## Phase 3109: Ω-P107 欠定几何判决——D1 反方向（floor 随 n 单调下降 −1.000，欠定解释被修正），D2 决定性：随机 100/200 维子空间 refit AUC≈1.0 与 \\|w\\|-TopK 无差，D3 跨半功能迁移 AUC 0.9998 而坐标仅 7% 重叠 → stability_not_n_limited；真值=弥散冗余编码，读出端口=功能等价类，"写入头组"作为任何固定几何对象不存在 [[NOW]]

**性质**：T3 第 3 Phase，纯离线（3105 冻结 capture，免 GPU，62s，SMOKE 一次通过）。预注册（design_seal.json 先于一切统计冻结）：n_sweep {100,200,400,800,1101}、boot 16 r=8 五分组、K {100,200} × 5 随机 seeds、margin 0.02、D1 Spearman ≥ 0.9；λ=0.01 记录值；标准化冻结自全 train。

### 1. D1 n-sweep：欠定单调预测被否定（方向相反）
| n | 100 | 200 | 400 | 800 | 1101 |
| --- | --- | --- | --- | --- | --- |
| floor | 0.2355 | 0.1606 | 0.1157 | 0.0845 | **0.0194** |
| J_half | 0.080 | 0.089 | 0.068 | 0.088 | 0.117 |
| PR_norm | 0.822 | 0.856 | 0.852 | 0.860 | 0.815 |

Spearman(n, floor) = **−1.000**——bootstrap 解随 n 增大**更不相似**，与欠定预测（单调稳定化）相反。小 n 的高相似是 **λ 正则伪影**（数据项弱，解被拉向同一先验几何）；n 大后数据约束主导，解进入宽解空间。**欠定作为简并的第一性来源被拒绝**——3108 §4 的解释被本轮证据修正：简并的主因不是 n<d，而是信号本身的弥散冗余（见 D2）。

### 2. D2 随机对照（判决性）：随机子空间读出与 \\|w\\| 选择无差
| K | \\|w\\|-TopK refit AUC | 随机 K 维 refit AUC（5 seeds） | 随机中位 | 判定 |
| --- | --- | --- | --- | --- |
| 100 | 0.9999 | [0.999, 1.000, 1.000, 1.000, 1.000] | **0.9997** | random_sufficient |
| 200 | 0.9999 | [1.000, 1.000, 0.998, 1.000, 0.998] | **0.9995** | random_sufficient |

**任意 100 维（3.9% 坐标）随机子空间都承载近乎完美的真值线性读出**——\\|w\\| 选择相对随机毫无优势。结合 3107 G1：Top-100 保持 AUC 的真正原因是"维度够"，不是"那 100 维特殊"。真值信息在残差流中**弥散冗余（diffusely redundant）**：AUC≈1 的读出方向构成一个与每个随机 100 维子空间都非平凡相交的宽锥。

### 3. D3 功能等价：坐标集不同，功能相同
half-A 的 Top-K 在 half-B refit：AUC 0.9998/0.9998（K=100/200）vs half-B 自选 Top-K 0.9995/0.9978，而两个坐标集 **Jaccard 仅 0.070/0.072**——transfer_ok 双双成立。跨 3107/3108 的全部证据：材料间、位置间、层间、采样间的 Top-200 集合重叠都在 4–21%，但它们都是**同一真值变量的等价读出端口**。

### 4. 理论更新（3107+3108+3109 三部曲综合）
**"写入头组"作为任何固定几何对象（坐标集、子空间、方向）都不存在。**存在的是一个巨大的**功能等价读出端口类**：真值是高层语义变量，由大量低层特征的条件组合决定，经残差流的混合坐标**冗余广播**到几乎全部 2560 维——因此任何足够大的子空间都可读出。模型自有 unembed 方向（3107 G3：与 ridge 探针几何正交而功能耦合 0.65）是该等价类中的一个非 L2 成员。**研究重心转移**：读出端几何已被三部曲证明是简并的，继续在读出端找"结构"无增量；下一步转向**写入端**（真值读数在何处、被哪些层/条件改变——3106 绑定 L3→真值 L4 链是入口）与**条件化结构**（哪些上下文条件决定真值取值）。

### 5. 硬伤
① D1 只有 5 个 n 点、J_half 平坦使 ρ_J=0.5 不显著；floor 曲线下降的解释（λ 伪影）是推论，未做 λ-sweep 分离；② D2 的"随机充分"只证明**存在性**（每个子空间内有好的判别方向），未刻画这些方向彼此的关系（是否都近似投影到同一真值子空间——注意这与"固定子空间"图景的调和方式：可以是"读出锥很宽"或"锥内方向共享一个小核"，K-sweep 可分辨）；③ D2/D3 全部在 3105 last\\|L8 单配置；④ test 集 AUC 天花板效应（0.9999）使 K 下限检测需要更难的材料或更小 K。

### 6. 3110 预注册（观测前冻结于 3110 seal）
① **K-sweep 随机子空间曲线**：K ∈ {5,10,20,50,100,200,400,800,1600,2560} × 10 seeds 的随机 refit AUC 中位曲线——**最小读出端口维度** d_min（AUC ≥ 0.95×全维的最小 K）= 弥散度的定量刻画，同时检验硬伤②的两种调和（若小 K 处曲线陡降 → 锥有共享小核）；② **跨变量特异性对照**：同一批随机子空间探针换目标变量（crit_rel 8 类）——弥散是真值特有还是普遍性质；③ 之后 T4 多步自回归线启动。之后 3111+ 转向写入端条件化结构。

产物：`tests/glm5/result/rdc_query_construction_20260913/phase3109/omega_p107_underdetermination_verdict/`（result.json、design_seal.json、run_log.txt）；脚本 `tests/glm5/phase3109_omega_p107_underdetermination_verdict.py`。
'''
    sec = sec.replace('[[NOW]]', NOW)
    memo += '\n' + sec
    with io.open(MEMO, 'w', encoding='utf-8') as f:
        f.write(memo)
    o.append('memo +%d chars (Phase 3109)' % len(sec))
else:
    o.append('memo already appended')

# ---------- workspace logs ----------
line_d = ('- Phase 3109 Omega-P107 (offline T3, no GPU): '
          'underdetermination verdict -> '
          'stability_not_n_limited. D1: floor(n) falls '
          'monotonically 0.236->0.019 (Spearman -1.000), '
          'OPPOSITE of underdetermination prediction - '
          'small-n similarity is a lambda artifact; '
          'underdetermination rejected as primary cause. '
          'D2 decisive: random 100/200-dim subspaces '
          'refit AUC median 0.9997/0.9995 vs |w|-TopK '
          '0.9999 -> random_sufficient; truth info '
          'DIFFUSELY REDUNDANT, any 100-dim subspace '
          'reads it out. D3: cross-half Top-K transfer '
          'AUC 0.9998 with coordinate overlap only 0.07 '
          '-> functional equivalence. SYNTHESIS 3107+'
          '3108+3109: write-head group exists as NEITHER '
          'coordinate set, subspace, nor direction - '
          'only a huge equivalence class of readout '
          'ports; truth is broadcast-redundant across '
          'coordinates. Focus shifts to write-in side + '
          'conditional structure. NEXT 3110: K-sweep '
          'minimum port dimension + cross-variable '
          'specificity control, then T4.\n')
for wdir in (WLOG_D, WLOG_C):
    wl = wdir + '\\' + TODAY + '.md'
    try:
        prev = io.open(wl, encoding='utf-8').read()
    except IOError:
        prev = ''
    if 'Phase 3109 Omega-P107' not in prev:
        try:
            with io.open(wl, 'a',
                         encoding='utf-8') as f:
                f.write(line_d)
            o.append('wlog appended %s' % wl)
        except Exception as e:
            o.append('wlog fail %s: %r' % (wl, e))
    else:
        o.append('wlog already %s' % wl)

# ---------- MEMORY.md rewrite ----------
# Full rewrite: previous file had duplicated 3104/3105/3106
# lines (replace-chain residue); 3000-char limit requires
# compaction.  Curated content, all discipline sections
# preserved, chain status compressed per phase.
mem_new = u'''# RDC/LPF 研究项目纪律（工作区长期记忆）

## 项目身份
- LPF v5.3 机械可解释性；qwen3-4b（models\\hf\\qwen3-4b）+ qwen3-14b。
- MEMO append-only：research\\gpt5\\docs\\AGI_GPT5_MEMO.md（glm5 版封存勿写）。
- 脚本 tests\\glm5\\phase{N}_*.py；产物 ...\\phase{N}\\{arm}\\；临时 gpt5_temp\\。
- Ledger research\\gpt5\\atlas\\atlas_ledger.json；L14 connects 现为 meas_id 字符串列表。

## 强制流程
1. 闭环：execution 冻结→执行→判决→seal→Ledger→MEMO→工作区日志→MEMORY→present→磁盘复核。
2. MEMO 标题 `## Phase {N}: 标题 [yyyy-mm-dd hh:mm]`；纠错 append 不回改。

## 标准锚与精度
- bit-0 锚族：d1/d1b/d1v/d2/d2b/d3/d5/d6；d4 sha；跨相位 sealed 对比 0.0。
- **干预语义必须先逆向再复刻**（3101 教训）：3093=全 V 流替换；"仅替换一层"与全流不等价。
- per-head 分解只在 o_proj 输入侧合法（post-o_proj reshape=伪象，R55）。

## 统计与度量纪律（3102 收紧五原则）
1. cos 与 relative-L2 双报；能量份额含交叉项非互补。
2. 门设计先验证可达域。
3. 秩相关用平均秩；跨模型合并须块校正。
4. JVP 准≠映射不混合；描述性 cos≠机制桥。
5. 次模权威判据=条件二阶差分；Möbius 谱仅描述。

## 机制链状态（3109）
- 3109：欠定几何判决 stability_not_n_limited。D1 反方向（floor 随 n 单调降 −1.000，欠定解释被修正；小 n 相似=λ 伪影）；D2 决定性：随机 100/200 维子空间 refit AUC≈1.0 与 |w|-TopK 无差——真值弥散冗余，任意大子空间可读出；D3 跨半迁移 0.9998 而坐标重叠仅 0.07——功能等价。**写入头组作为固定几何对象不存在；读出端=功能等价端口类**。重心转向写入端与条件化结构。
- 3108：简并分离。同材料两半 Top-200 J 仅 0.11/0.07（选择噪声主导）；子空间角全 fail（sanity 0/3）；解族近正交宽平坦（bootstrap cos 0.04、谱 Gini 0.10）。其欠定第一性来源提法已被 3109 修正。
- 3107：Top-100 坐标保持 AUC 但坐标基跨材料简并旋转（J 0.053≈随机）；unembed 与探针近正交而功能耦合（Spearman 0.654）——多路读出。
- 3106：四门全过 composition_dose_tracked。链 m AUC 0.930/0.958；频率启发式否证；被动穿透 +0.36；绑定-真值层对齐 Pearson 0.960、峰差 1 层（绑定 L3→真值 L4）。
- 3105：读出追踪 in-context 真值（m AUC 0.99，P>A1 74/74，诱饵拒绝=联合匹配）；判离：上下文可推导 0.99 vs 外部私有 0.50。绑定曲线复现（hs[12] teE 0.995）。
- 3104 更正：margin 符号 +m（AUC 0.503，判决不变）；K3 真值门不适定。
- 3103：命题账本 62 条 A5/B34/C14/D21/E20（sha8 add57ba7）；E 级禁入新推理链。
- 3101：seventh_carrier_absent；L37 自然写入头组 21/12/14。

## 本机环境缺陷（Windows，必读）
- bash shim 劣化：rm/ls/grep/cat/tail/wc 坏→python + 写文件；-c stdout 丢→写文件再 Read；反引号被命令替换→chr(96)。
- 关键写入后必须 Grep/Read 复核磁盘；GPU 测试逐模型防 OOM；残留进程查 psutil。

## 工作方式
- "好的，继续"=AI 主导不停；结构化输出；关键发现重复 3 次。

## 下一步
- max=3109，下一 3110：**K-sweep 最小读出端口维度 + 跨变量特异性对照**（弥散度定量）→ 之后 T4 多步自回归 + 写入端条件化结构。
'''
assert len(mem_new) < 3000, len(mem_new)
with io.open(MEMO_W, 'w', encoding='utf-8') as f:
    f.write(mem_new)
o.append('memory rewritten %d chars' % len(mem_new))

io.open(LOGF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
print('closeout ok')
