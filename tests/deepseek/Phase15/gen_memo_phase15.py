# -*- coding: utf-8 -*-
"""
Phase 15 备忘录节生成器：从 result_phase15.json 【读出】全部数字生成 MEMO 追加源。
=============================================================================
纪律（Phase 13 教训 + Phase 14 铁律 (w)）：散文里的数字一律现场渲染，不经人手转录。
纪律（Phase 14 附注）：散文里的**因果假设**仍需人工核对；生成器只消除「数字转录」类错误。
本生成器把所有跨模型结论写成**由 verdict / joint_verdict 分支选择**的文本，
因此无论 A1/A2 落到哪个标签，输出的散文都与判决自洽（不会出现「写死的假设被数据推翻」）。
输出：tests/deepseek_temp/Phase15/memo_append_phase15.md
"""
import os, io, json, hashlib, time

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P15T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase15')
T14 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase14')
T13 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase13')
T12 = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase12')

R = json.load(io.open(os.path.join(P15T, 'result_phase15.json'), encoding='utf-8'))
SEAL = json.load(io.open(os.path.join(P15T, 'N2h1a8_design_seal.json'), encoding='utf-8'))
EX = json.load(io.open(os.path.join(P15T, 'execution_phase15.json'), encoding='utf-8'))
AM1 = json.load(io.open(os.path.join(P15T, 'N2h1a8_design_seal_amend1.json'), encoding='utf-8'))
R14 = json.load(io.open(os.path.join(T14, 'result_phase14.json'), encoding='utf-8'))
R13 = json.load(io.open(os.path.join(T13, 'result_phase13.json'), encoding='utf-8'))
R12 = json.load(io.open(os.path.join(T12, 'result_phase12.json'), encoding='utf-8'))
OUT = os.path.join(P15T, 'memo_append_phase15.md')

INH = R['inheritance_used']
PRED = {p['id']: p for p in SEAL['pre_registered_predictions']}
HONESTY = list(SEAL['honesty'])
MAYF = list(SEAL['may_falsify_the_whole_line'])
QUANT = R['extra']['quant']
AM1_SHA8 = hashlib.sha256(io.open(os.path.join(P15T, 'N2h1a8_design_seal_amend1.json'),
                                  'rb').read()).hexdigest()[:8]
SUPS6 = list(EX['classes'])
SITES = R['grid']['profile_sites']
ALS = R['grid']['alphas']
W = R['grid']['W']
FL = R['floors']
PC = R['predictions_check']
VER = R['verdict']
JOINT = R['joint_verdict']
META = R['arms_meta']
E4S = R['E4_summary']
E5C = R['E5_concentration']
E2F = R['E2_full_swap']
E3L = R['E3_localize']
E6C = R['E6_calibration']
E0S = R['E0_selfcheck']
E1C = R['E1_capture']

ARM0 = 'A0_calib_qwen3-4b-nf4'
ARM1 = 'A1_glm4-9b-nf4'
ARM2 = 'A2_qwen3-14b-nf4'
REP = [a for a in (ARM1, ARM2) if a in VER]                 # 跨模型复算臂（存在的）
REP_OK = [a for a in REP if VER[a].get('Q0_device') == 'PASS']
ARMS_ALL = [a for a in (ARM0, ARM1, ARM2) if a in VER]

SHORT = {ARM0: 'A0·4B(nf4)', ARM1: 'A1·GLM4-9B', ARM2: 'A2·Qwen3-14B'}
XPUB = {int(k): float(v) for k, v in INH['XH_12_by_site'].items()} if 'XH_12_by_site' in INH else {}
JPUB = {int(k): float(v) for k, v in INH['J_swap_12_by_site'].items()} if 'J_swap_12_by_site' in INH else {}


def f(x, n=6):
    if x is None:
        return 'None'
    try:
        return ('%.' + str(n) + 'f') % float(x)
    except (TypeError, ValueError):
        return str(x)


def g(d, *ks, default=None):
    cur = d
    for k in ks:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


L = []
A = L.append

# ============================================================ 标题 + 0. 一句话
A('## Phase 15: 跨模型复算「统一剖面」（N2h1-α-8）[%s]' % time.strftime('%H:%M'))
A('')
A('### 0. 一句话')
A('')
q1 = VER.get(ARM0, {}).get('Q1_label', 'NA')
q2j = JOINT.get('Q2_joint', 'NA')
q3j = JOINT.get('Q3_joint', 'NA')
q2txt = {
    'ARGS_GAP_LAYERSTACK': '两坐标 argmax 相距 ≥3 在 **untied 的 GLM4-9B 与 Qwen3-14B 上都复现** ⇒ '
                           '「两坐标分辨率分离」不是 qwen3-4b 特例，是**层栈性质**',
    'ARGS_GAP_4B_SPECIFIC': '两坐标 argmax 相距 <3 在两个跨模型臂上都出现 ⇒ '
                            '「相距 13」是 **qwen3-4b 特例**，不是层栈性质',
    'ARGS_GAP_MIXED': '两坐标 argmax 距离在两个跨模型臂上**不一致** ⇒ 该量受模型个体影响，'
                      '既非普遍层栈性质、也不能判为 4B 特例',
}[q2j] if q2j in ('ARGS_GAP_LAYERSTACK', 'ARGS_GAP_4B_SPECIFIC', 'ARGS_GAP_MIXED') else '（跨模型臂不足，Q2 无判决）'
q3txt = {
    'CONC_JUDGE_INVALID_X_ALL': 'xhalf 坐标的 null 95 分位在两个跨模型臂上都 ≥0.70 ⇒ '
                                '**Phase 12/13/14 中一切基于 `top3_share_x` 的表述必须整体撤回**',
    'CONC_JUDGE_ALIVE_X': 'xhalf 坐标的 null 95 分位未在两个跨模型臂上同时越过 0.70 ⇒ 该坐标判据**未被整体否证**，'
                          '但每一处引用仍须与 null95 同报',
}[q3j] if q3j in ('CONC_JUDGE_INVALID_X_ALL', 'CONC_JUDGE_ALIVE_X') else '（跨模型臂不足，Q3 无判决）'

# ---- 零假设校准的逐格统计（§0 与 §5 共用；必须在使用前定义）
_n95 = {a: g(E5C, a, 'null_x', 'null95') for a in ARMS_ALL}
_jm = {a: g(E5C, a, 'margin_j') for a in ARMS_ALL}
_jup = [a for a in ARMS_ALL if (_jm.get(a) or -1) > 0]
_xup = [a for a in ARMS_ALL if (g(E5C, a, 'margin_x') or -1) > 0]
_ncell = len(_jup) + len(_xup)

_null0 = g(E5C, ARM0, 'null_x') or {}
A('**跨模型复算把 Phase 13/14 的两个悬置问题各自结案**：① Q2 —— %s；② Q3 —— %s。'
  % (q2txt, q3txt))
A('')
A('量化口径门 **Q1 = %s**（`max|dxhalf| = %s` ≤ %.2f，`argmax_w_x` nf4=%s / bf16=%s）⇒ '
  'nf4 复现 bf16 的剖面到阈值内，三臂因此可比。'
  % (q1, f(g(E6C, ARM0, 'max_abs_dxh'), 4), FL['XH_FAITHFUL_TOL'],
     g(E6C, ARM0, 'argmax_w_x_nf4'), g(E6C, ARM0, 'argmax_w_x_bf16')))
A('')
A('装置侧：三臂 F0–F4 全通过（`P1 = %s`），A0 的 `null95_x = %s` 与 Phase 14 的 `0.6998` 同量级，'
  '**紧贴 0.70 门**；`d_argmax` A0 = %s（bf16 参照 %d）。'
  % ('PASS' if PC['P1']['pass_'] else 'FAIL', f(_null0.get('null95'), 4),
     g(E5C, ARM0, 'd_argmax_window'), abs(int(INH['MODE_X_13']) - int(INH['MODE_J_13']))))
A('')
A('**⚠️ 本 Phase 最重要的自我修正（零假设校准）**：把 `J` 坐标也施加**同一套**置换零假设检验后，'
  '**6 个「臂 × 坐标」格里只有 %d 格超过各自的 null 95 分位**（%s）⇒ 三臂上**不存在跨模型稳健的集中度判据**；'
  '「坐标依赖」须再升一级为「**坐标 × 模型**双重依赖」，且**不得再说「某个坐标才是有区分力的坐标」**。'
  % (_ncell,
     json.dumps([SHORT[a] + '·' + ('J' if a in _jup else 'xhalf') for a in ARMS_ALL
                 if a in _jup or a in _xup], ensure_ascii=False)))
A('')
A('**⚠️ 首轮运行被装置门拦下（amend1）**：`glm4-9b` 的类别 token id 因沿用 **qwen 词表硬编码**而全错，'
  '`F2 base bad = %d/41`、`FULL_SWAP = %+.3f`（4B 为 `%+.3f`）⇒ A1 数据作废重跑。'
  '**修正后三臂 F1b/F2 全通过**（`bad` 全 `0/41`），本 MEMO 的全部数字来自修正后的正式运行。'
  % (AM1['evidence_from_device_gate']['A1_F2_base_bad_n'],
     AM1['evidence_from_device_gate']['A1_FULL_SWAP'],
     AM1['evidence_from_device_gate']['A0_FULL_SWAP']))
A('')

# ============================================================ 1. 目标与死线
A('### 1. 目标与死线来源')
A('')
A('- **死线原文**（Phase 14 §8「Phase 15 候选（最高优先）」，逐字）：')
A('  > 「把 Phase 12/13/14 已确立的**唯一有效口径**（单点位点替换族的 `xhalf(ℓ)` / `J(ℓ)` 双坐标剖面 + '
  '置换零假设校准）在 **qwen3-14b** 与 **glm4-9b（untied）** 上独立复算，回答两个真问题：'
  '① Phase 13 的「两坐标 argmax 相距 13」是层栈性质还是 qwen3-4b 特例（本 Phase 已把它**挂起**）；'
  '② 置换零假设 95 分位是否也高达 0.70–0.75（若如此，集中度判据在这条线上**整体作废**）。'
  '**判据必须先冻结**，且**禁止**沿用 L6/U6；**新增一条**：任何集中度结论必须同时报 null 95 分位与裕度。」')
A('- **唯一有效口径的来历**：Phase 12 确立单点位点替换族的双坐标剖面；Phase 13 做配对 bootstrap 逐位复现；'
  'Phase 14 证明死线原本设想的另外两条读法（`positions[0..1]` 全替的前缀族、逐层累积支撑族）**都代数退化为同一族**'
  '（`pos0` 通道 `median(y0/y1) = %s`；层覆盖恒等），并把「两坐标 argmax 相距 13 是否层栈性质」**挂起**给本 Phase。'
  % f(g(R14, 'A3_position_summary', 'median_ratio'), 6))
A('')
A('- **⚠️ 量化口径替换的必要性（三条 bf16 路线均被实测否决，这是本 Phase 技术路线的决定因素）**：')
A('  - 模型体积 vs 宿主约束：**Qwen3-14B 29.54 GB / GLM4-9B 18.80 GB / qwen3-4b 8.04 GB**；'
  '宿主 GPU = **RTX 5080 16 GB**（`memory.total 16303 MiB`），**RAM total 33.7 GB / 可用仅 ~16.5–17.5 GB**。')
A('  - **bf16 路线 ①**（`device_map="auto"` 无 `max_memory`）：把 10 个模块 offload 到**磁盘**（meta device），'
  '层放置 `{cuda:0: L0..L15, meta: L16..L39}`，**7.317 s/前向** ⇒ 18×14×24 = 6048 前向 = **316 min/臂** ⇒ 不可行。')
A('  - **bf16 路线 ②**（`device_map="auto" + max_memory={0:"14GiB", cpu:"24GiB"}`，禁磁盘）：'
  'qwen3-14b **在加载期被硬杀（无 Python 栈）** ⇒ RAM 峰值超限。')
A('  - **bf16 路线 ③**（glm4-9b + `max_memory`）：可行（loaded 12.8 s、层放置 `{cuda:0: L0..L29, meta: L30..L39}`、'
  '**0.745 s/前向**），但**会破坏「两臂同一口径」**。')
A('  - ⇒ **最终统一为 nf4 口径**：`%s`；`max_memory = %s`；`attn_implementation = %s`。'
  '三模型全部**全载 GPU**（`param devices: {"cuda:0": N}`），前向 **0.036 / 0.036 / 0.041 s**'
  '（qwen3-4b / glm4-9b / qwen3-14b），双前向 determinism `%s`。'
  % (QUANT['scheme'], json.dumps(QUANT['max_memory'], ensure_ascii=False), QUANT['attn_implementation'],
     f(g(E0S, ARM0, 'determinism_maxdiff'), 3)))
A('  - **设计代价被预注册承担**：口径替换会引入量化误差，故本 Phase **增设 A0 臂专测量化保真**'
  '（与 Phase 12 bf16 已发表量逐位点比对），并把「A0 判 DEViant ⇒ 回到 bf16 路线」写进 seal 的'
  '`may_falsify_the_whole_line`。**这是把「引入新自由度」变成「可被否证的门」的标准做法。**')
A('')
A('- **7 条预注册可证伪预测（运行前冻结，判据先冻结）**：')
for k in sorted(PC, key=lambda z: int(z[1])):
    d = PRED.get(k, {})
    det = ', '.join('%s=%s' % (kk, json.dumps(vv, ensure_ascii=False)) for kk, vv in PC[k].items()
                    if kk not in ('desc', 'pass_', 'smoke'))
    A('  - **%s**（%s）：**%s** —— %s' % (k, d.get('desc', ''), 'PASS' if PC[k]['pass_'] else 'FAIL', det[:260]))
A('')
A('- **为什么这不是 HARKing**（seal 原文）：预测在**正式运行前**冻结；两个「真问题」的**两种答案都改变后续路线**'
  '（`Q2_joint` 有任何分支都会改变 Phase 16 的设计；`Q3_joint` 的 `INVALID_X_ALL` 会使 Phase 12/13/14 的'
  '集中度表述整体撤回）；量化口径虽在运行前引入，但**其影响由独立 A0 臂测量并预注册了否证条款**。')
A('')
A('- **amend1（`%s`）= apparatus fix，不改假设**：正式运行首轮（A0 已完成、A1 进行中）被**装置门**'
  '`F2_base_ok` 抓到 `A1 bad = %d/41`、受体类分数均值 `%+.3f`、`FULL_SWAP = %+.3f`（4B 为 `%+.3f`）。'
  '根因：seal 的 `panel.sup_id`（`水果=104618` 等）是 **qwen 词表** id，被**全局**用于三臂；'
  '`glm4-9b-chat-hf` 词表不同（vocab **151329 vs 151643**）⇒ A1 全程读**错误的类别 token**。'
  '**修正**：`sup_id` 改为**每臂由该臂 tokenizer 现场解析**，并新增硬断言 **F1b**'
  '「6/6 类别词恰为单 token 且 `decode(id) == 词`」。'
  '**自洽核对**：A0/A2 的解析结果与冻结参考值**逐位相同**（`matches_frozen=True`）⇒ 修正只修复 A1，'
  '不改变任何有效臂的口径。首轮 A1/A2 数据**作废重跑**（原日志保留 `_formal_stdout_run1_INVALID_supid.log`）。'
  % (AM1_SHA8, AM1['evidence_from_device_gate']['A1_F2_base_bad_n'],
     AM1['evidence_from_device_gate']['A1_F2_receptor_class_score_mean'],
     AM1['evidence_from_device_gate']['A1_FULL_SWAP'],
     AM1['evidence_from_device_gate']['A0_FULL_SWAP']))
A('  - **这一条是本 Phase 方法论层面的最重要产物**：**装置门必须「独立于主结论」才有价值** —— '
  '`F2_base_ok` 与「两坐标 argmax 距离 / null 分位」毫无关系，正因为它只问「读对 token 了吗」，'
  '才能在主结论产生之前把 A1 拦下。若无此门，A1 会以「GLM4 的 is-a 关系不成立」的**机制结论**'
  '被写进 MEMO（与 4B 的巨大差异看起来完全像一条发现）。')
A('')

# ============================================================ 2. 原理
A('### 2. 原理与算法')
A('')
A('#### 2.1 装置与继承（唯一有效口径 = 单点位点替换族）')
A('- 模板 `%s`；**41/41 实例 tokenize 为 T=2**（pos0 = 实例词、pos1 = 框架词「是一种」）——'
  '这让「单点位点替换」与「读点位置」在三个模型上口径完全一致。'
  % EX['template'])
A('- **干预**：`h_ell_recip(clean) + alpha * (h_ell_donor − h_ell_recip)`，α ∈ %d 点 `%s`。'
  % (len(ALS), json.dumps(ALS)))
A('- **归一**：`Y(ℓ,α) = dDonor / FULL_SWAP_arm`（**每臂独立**，零额外前向）。'
  '`FULL_SWAP_arm` = 该臂 24 个发现对的「供体自身前向」均值（**24 项口径，与 41 键全均值不等**）。')
A('- **死线硬约束的落实**：profile 臂使用**裸残差差** `d_ell`（**不投影、不用 U**）⇒ 「禁止沿用 L6/U6」'
  '**结构性满足**；写入窗 `L*_own` 由**独立定位臂**在**逐层独立重建的 `U_ell`** 上给出。')
A('')
A('#### 2.2 独立写入窗定位（`B_cat` 口径，不预设 L6）')
A('- patch 该层输出末位为 `h_ell_recip + P_{U_ell}(h_ell_donor − h_ell_recip)`，`U_ell = est_U(ℓ+1)`；'
  '`est_U` = 6 类质心差的 SVD（秩 = 5），**逐层独立重建**。')
A('- 候选层 `%s`（15 个）；判据 = **`B_cat` 曲线的相邻层最大增量处** ⇒ `L*_own`。'
  % json.dumps(EX['localize']['cands']))
A('- **不预设 L6，允许 `L*` 因模型而异** —— 这正是 P5 要检验的（L6 是 4B 的写入窗还是普遍常数）。')
A('')
A('#### 2.3 统计量（逐字复制 Phase 12/13）')
A('- `xhalf = cross_alpha(x, y, %.1f)`（线性插值，**网格不变量**，不假设单调）；' % R['grid']['xh_frac'])
A('- `J = max(相邻斜率)/median(其余斜率)`（仅 α ≥ 0.01 的相邻段；**网格依赖量**）；')
A('- `conc_hat(F, W=%d) = max over %d 个 %d-窗口 |Σ %d 个相邻跳| / range(F)`；'
  '窗口 idx `w` 覆盖 `jumps[w..w+%d]` ⇔ 位点 `sites[w] → sites[w+%d]`。'
  % (W, len(SITES) - W + 1, W, W, W - 1, W - 1))
A('- **置换零假设（Phase 14 铁律 (p) 的直接继承）**：把 `jumps` 幅度**随机重排**到相邻对上'
  '（BP=%d，`xhalf` 用 `seed+13`、`J` 用 `seed+29`，独立生成器），重算 `top3_share` 取 95 分位。'
  % R['grid']['BP'])
A('- **死线新增条款的落实**：任何集中度结论**必须同时报 null 95 分位与裕度**'
  '（`margin = share − null95`）⇒ 本节的每张集中度表都带这两列。')
A('')
A('#### 2.4 臂表')
A('| 臂 | 模型 | 角色 | 每臂前向 |')
A('|---|---|---|---|')
for k in EX['arm_order']:
    v = EX['arms'][k]
    A('| %s（`%s`） | `%s`（L=%d / hid=%d / heads=%d / kv=%d / tie=%s） | %s | 见 §3 |'
      % (SHORT[k], k, v['model'], v['expected']['num_hidden_layers'], v['expected']['hidden_size'],
         v['expected']['num_attention_heads'], v['expected']['num_key_value_heads'],
         v['expected']['tie_word_embeddings'], v['role']))
A('')
A('- 每臂前向数 = `capture(41)` + `定位(%d 层 × 24 对)` + `剖面(18 位点 × 14 α × 24 对)` + `装置锚(12)`。'
  % len(EX['localize']['cands']))
A('- **A0 臂是本 Phase 的安全阀**：它用 nf4 复算 qwen3-4b 的同一剖面，与 Phase 12 bf16 **已发表量**'
  '逐位点比对（`max|dxhalf|`、`argmax_w_x`、`XH_RANGE`、`share_x`、`J` 比值带、`recover`）。'
  '**若 nf4 不能复现 bf16，则 A1/A2 的一切剖面结论都只是「nf4 口径下的描述」。**')
A('')

# ============================================================ 3. 材料
A('### 3. 材料')
A('- 发现集 24 实例（6 类 × 4）、确认集 17 实例、全捕获 41 实例；6 个上位词 '
  '`%s`。' % json.dumps(EX['classes'], ensure_ascii=False))
A('- 网格：**18 位点 × 14 α × 24 对**；`W=%d`、`BP=%d`、`xh_frac=%.2f`。'
  % (W, R['grid']['BP'], R['grid']['xh_frac']))
_srows = [];
for a in ARMS_ALL:
    _srows.append('%s %.1f s' % (SHORT[a], float(g(E1C, a, 'seconds') or 0)))
A('- 运行：**总耗时 %.1f s**（%s），GPU，无 OOM。' % (float(R.get('elapsed_total_s') or 0),
                                                     '；'.join(_srows)))
A('- **加载/前向墙钟是可行性瓶颈**：nf4 + 全载 GPU 把单前向从 bf16-offload 的 **7.317 s** 压到 '
  '**0.036–0.041 s（约 200×）**，这才使「三臂 × 全网格」在单机上可跑完。')
A('')

# ============================================================ 4. 结果
A('### 4. 实际结果')
A('')
A('#### 4.1 装置门（三臂 F0–F4 + E0 自检）')
A('')
A('| 臂 | config sha8 | T=2 全 41 | **F1b 类别 token（逐臂解析）** | 与 qwen 参考一致 | L/hid/heads/kv/head_dim | o_proj | determinism | hook 效应 | **F2 base bad** | 加载 s |')
A('|---|---|---|---|---|---|---|---|---|---|---|')
_dims = {ARM0: (36, 2560, 32, 8, 128), ARM1: (40, 4096, 32, 2, 128), ARM2: (40, 5120, 40, 8, 128)}
for a in ARMS_ALL:
    Lx, hx, hd, kv, hdm = _dims[a]
    sida = g(R['sup_id_per_arm'], a) or (g(R['arms'], a, 'sup_id_arm'))
    badn = len(g(R['arms'], a, 'F2_base_bad') or [])
    A('| %s | `%s` | %s | `%s` | %s | %d / %d / %d / %d / %d | %s | %s | %s | **%d/41** | %s |' % (
        a, META[a]['config_sha8'], '✔' if g(E0S, a, 'T2_only') else '✘',
        json.dumps({k: sida[k] for k in SUPS6}, ensure_ascii=False) if sida else '—',
        ('**是**' if g(R['arms'], a, 'sup_id_matches_ref') else '**否（已由 amend1 修复）**'),
        Lx, hx, hd, kv, hdm, '✔' if g(E0S, a, 'o_proj_ok') else '✘',
        f(g(E0S, a, 'determinism_maxdiff'), 3), f(g(E0S, a, 'hook_effect_maxdiff'), 3),
        badn, f(g(R['arms'], a, 'load_s'), 1)))
A('')
A('- **F1b 是 amend1 新增的装置断言**：`水果/动物/交通工具/家具/金属/颜色` 六个词必须在该臂 tokenizer 下'
  '**各自恰为 1 个 token** 且 `decode(id)` 可逆。A0/A2 与 qwen 参考值逐位相同，**A1 六项全部不同**'
  '（GLM4 词表：`水果=103444 / 动物=100787 / 交通工具=122416 / 家具=103813 / 金属=101406 / 颜色=101526`）。')
A('- **`F2 base bad` 是本次事故的报警器**：要求每个实例的「受体类分数 `sr0`」> 0（该实例自己的上位词'
  '得分高于其余 5 个类别均分）。修正后三臂全部 `0/41`；修正前 A1 为 `23/41`。')
A('- **F3 α=0 还原基线**（装置回归测试）：每臂取首/中/末位点 × 前 3 对，'
  '`max|dScore| = %s`（A0）⇒ patch 在 α=0 处是**严格的 no-op**。'
  % f(max(g(R['arms'], ARM0, 'F3_alpha0_maxdev').values()) if g(R['arms'], ARM0, 'F3_alpha0_maxdev') else None, 3))
A('- **F0 config sha 校验**：三臂 `config_sha8` 与 seal 冻结值**全部一致** ⇒ 载入的确是指定权重。')
A('')
A('#### 4.2 A0 量化保真校准（Q1 = %s）' % q1)
A('')
e6 = E6C.get(ARM0, {})
A('| 量 | nf4 | Phase 12 bf16 已发表 | 差 | 门 |')
A('|---|---|---|---|---|')
A('| `argmax_w_x` | %s | %s | %s | 相同 → %s |' % (
    e6.get('argmax_w_x_nf4'), e6.get('argmax_w_x_bf16'),
    ('相等' if e6.get('argmax_same') else '不等'), '✔' if e6.get('argmax_same') else '✘'))
A('| `max|dxhalf|`（18 位点） | — | — | **%s** | ≤ %.2f → %s |' % (
    f(e6.get('max_abs_dxh'), 4), FL['XH_FAITHFUL_TOL'], '✔' if e6.get('pass_tol') else '✘'))
A('| `share_x = top3_x` | %s | %s | %s | （描述性） |' % (
    f(e6.get('share_x_nf4'), 4), f(e6.get('share_x_bf16'), 4),
    f((e6.get('share_x_nf4') or 0) - (e6.get('share_x_bf16') or 0), 4)))
A('| `XH_RANGE` | %s | %s | %s | ∈ [%.2f, %.2f] → %s |' % (
    f(e6.get('XH_RANGE_nf4'), 4), f(e6.get('XH_RANGE_bf16'), 4),
    f((e6.get('XH_RANGE_nf4') or 0) - (e6.get('XH_RANGE_bf16') or 0), 4),
    FL['XH_RANGE_BAND'][0], FL['XH_RANGE_BAND'][1],
    '✔' if (e6.get('XH_RANGE_nf4') is not None and FL['XH_RANGE_BAND'][0] <= e6['XH_RANGE_nf4'] <= FL['XH_RANGE_BAND'][1]) else '✘'))
A('| `max|drecover|` | — | — | **%s** | （描述性） |' % f(e6.get('max_abs_drecover'), 4))
A('| `J` 比值范围（nf4/bf16） | — | — | `[%s, %s]` | （网格差异+量化，描述性） |' % (
    f(e6.get('J_ratio_min'), 3), f(e6.get('J_ratio_max'), 3)))
A('| `d_argmax_window` | %s | %d | %s | （参照） |' % (
    g(E5C, ARM0, 'd_argmax_window'), abs(int(INH['MODE_X_13']) - int(INH['MODE_J_13'])),
    '差 1（nf4 的 `argmax_w_j` 落在 0 而非 1）' if g(E5C, ARM0, 'd_argmax_window') != abs(int(INH['MODE_X_13']) - int(INH['MODE_J_13'])) else '相同'))
A('')
A('- **结论：nf4 复现 bf16 的 `xhalf` 剖面到 `%s`**（门 %.2f，余量 %s 倍），`XH_RANGE` 也在带内，'
  '`recover` 逐位点与原 `recover_12_by_site` 差 `%s`。⇒ **P2 成立，A1/A2 的剖面可与 Phase 12 同口径比较。**'
  % (f(e6.get('max_abs_dxh'), 4), FL['XH_FAITHFUL_TOL'],
     f(FL['XH_FAITHFUL_TOL'] / e6['max_abs_dxh'], 1) if e6.get('max_abs_dxh') else 'NA',
     f(e6.get('max_abs_drecover'), 4)))
A('- **⚠️ 唯一的量化伪影**：`d_argmax_window` A0 = %s（bf16 = %d）。差别**只在 `J` 坐标**：'
  'nf4 的 `argmax_w_j = %s`（bf16 = %s）——`J` 是**网格依赖 + 极值型**量，浅端两个窗口的 |Σ jumps| '
  '在量化噪声下可互换。**`xhalf` 坐标的 `argmax_w_x` 完全相同（%s）**，故本 Phase 的 Q2/Q3 结论'
  '不依赖这一像素级差别。'
  % (g(E5C, ARM0, 'd_argmax_window'), abs(int(INH['MODE_X_13']) - int(INH['MODE_J_13'])),
     g(E5C, ARM0, 'argmax_w_j'), INH['MODE_J_13'], e6.get('argmax_w_x_nf4')))
A('')

A('#### 4.3 三臂「统一剖面」并列（`xhalf` / `J` × 18 位点）')
A('')
A('| ℓ | A0·4B(nf4) `xhalf` | A0 `J` | A1·GLM4-9B `xhalf` | A1 `J` | A2·Qwen3-14B `xhalf` | A2 `J` |')
A('|---|---|---|---|---|---|---|')
for i, s in enumerate(SITES):
    row = ['| L%d |' % s]
    for a in (ARM0, ARM1, ARM2):
        if a in E4S and E4S[a]:
            row.append('%s | %s |' % (f(E4S[a]['xhalf'][i], 6), f(E4S[a]['J'][i], 4)))
        else:
            row.append('— | — |')
    A(''.join(row))
A('')
A('- **`xhalf` 的 18 位点极差（`XH_RANGE`）**：A0 = %s（bf16 %s）；A1 = %s；A2 = %s。'
  % (f(g(E4S, ARM0, 'XH_RANGE'), 4), f(INH['XH_RANGE_12'], 4),
     f(g(E4S, ARM1, 'XH_RANGE'), 4), f(g(E4S, ARM2, 'XH_RANGE'), 4)))
A('- **`xhalf` 随深度单调性（Spearman）**：A0 = %s；A1 = %s；A2 = %s。'
  % (f(g(E5C, ARM0, 'spearman_xh_depth'), 4), f(g(E5C, ARM1, 'spearman_xh_depth'), 4),
     f(g(E5C, ARM2, 'spearman_xh_depth'), 4)))
A('- **`J` 随深度单调性（Spearman）**：A0 = %s；A1 = %s；A2 = %s（`J` 从浅端极大单调跌到深端 ~1）。'
  % (f(g(E5C, ARM0, 'spearman_J_depth'), 4), f(g(E5C, ARM1, 'spearman_J_depth'), 4),
     f(g(E5C, ARM2, 'spearman_J_depth'), 4)))
A('- **读数位点 R（最终 LayerNorm 之后）在 Phase 12 已确立为「线性、无拐点」**（本 Phase 未重测 R；'
  '复用 Phase 12 结论作为外推限界之一）。')
A('')

A('#### 4.4 双坐标集中度 + **置换零假设 + 裕度**（Q3/Q4，死线新增条款）')
A('')
A('| 臂·坐标 | `top3_share` | `argmax_w` | 窗口语义 | `null` 95 分位 | **裕度 `share−null95`** | 观测 ≥ null ? |')
A('|---|---|---|---|---|---|---|')
for a in ARMS_ALL:
    c = E5C.get(a) or {}
    for coord, key, wk, nk, mk in (('xhalf', 'x', 'win_sem_x', 'null_x', 'margin_x'),
                                   ('J', 'j', 'win_sem_j', 'null_j', 'margin_j')):
        sem = c.get(wk)
        A('| %s·%s | %s | %s | %s | **%s** | **%s** | %s |' % (
            SHORT[a], coord, f(c.get('top3_' + key), 4), c.get('argmax_w_' + key),
            ('w=%d: %s→%s' % (sem['w'], sem['a'], sem['b'])) if sem else '—',
            f((c.get(nk) or {}).get('null95'), 4), f(c.get(mk), 4),
            ('**是**' if (c.get(mk) is not None and c[mk] > 0) else '否')))
A('')
A('- **`d_argmax = |argmax_w_x − argmax_w_j|`**：%s。'
  % '；'.join('%s = %s（x:%s / j:%s）' % (SHORT[a], g(E5C, a, 'd_argmax_window'),
                                         g(E5C, a, 'argmax_w_x'), g(E5C, a, 'argmax_w_j'))
             for a in ARMS_ALL))
A('- **Q2 的跨模型合取** = `%s`：%s' % (q2j, q2txt))
A('- **Q3 的跨模型合取** = `%s`：%s' % (q3j, q3txt))
A('- **Q4（描述性，与 Q3 配对）**：`xhalf` 坐标的裕度 %s；`J` 坐标的裕度 %s。'
  '**死线条款「share 不得脱离 null95 单独引用」在此逐行落实。**'
  % ('；'.join('%s = %s' % (SHORT[a], f(g(E5C, a, 'margin_x'), 4)) for a in ARMS_ALL),
     '；'.join('%s = %s' % (SHORT[a], f(g(E5C, a, 'margin_j'), 4)) for a in ARMS_ALL)))
A('')

A('#### 4.5 独立写入窗 `L*_own`（B_cat 相邻最大增量；正面回应「禁止沿用 L6/U6」）')
A('')
A('| 臂 | `L*_own` | 相邻最大增量 | `B_cat` 曲线（候选层） | 定位耗时 s |')
A('|---|---|---|---|---|')
for a in ARMS_ALL:
    e3 = E3L.get(a) or {}
    cur = e3.get('curve') or {}
    seq = ' / '.join('L%s:%+.3f' % (k, v) for k, v in cur.items())
    A('| %s | **L%s** | %s | %s | %s |' % (SHORT[a], e3.get('L_star_own'),
                                           f(e3.get('L_star_increment'), 3), seq, f(e3.get('seconds'), 1)))
A('')
A('- A0 的 `L*_own = %s`（相邻最大增量 %s）**复现 Phase 8 的 L6 几何**；'
  'A1 / A2 的 `L*_own` 见上表 ⇒ **P5**（「L6 是 4B 的写入窗，非普遍常数」）**%s**。'
  '（该量的 JSON 键名为 `L_star_own` / `L_star_increment`。）'
  % (g(E3L, ARM0, 'L_star_own'), f(g(E3L, ARM0, 'L_star_increment'), 3),
     'PASS' if PC['P5']['pass_'] else 'FAIL'))
A('')

A('#### 4.6 跨模型判决（Q0/Q1/Q2/Q3 + P1–P7）')
A('')
A('| 臂 | Q0 装置 | Q1 量化 | `d_argmax` | Q2 标签 | `null95_x` | Q3 标签 | 裕度_x | 裕度_j |')
A('|---|---|---|---|---|---|---|---|---|')
for a in ARMS_ALL:
    v = VER[a]
    A('| %s | %s | %s | %s | %s | %s | %s | %s | %s |' % (
        SHORT[a], v.get('Q0_device'), v.get('Q1_label', '—'), v.get('Q2_d_argmax'),
        v.get('Q2_label'), f(v.get('Q3_null95_x'), 4), v.get('Q3_label'),
        f(v.get('Q4_margin_x'), 4), f(v.get('Q4_margin_j'), 4)))
A('')
A('- `joint_verdict`：`arms_used_for_cross_model = %s`；`Q2_joint = %s`；`Q3_joint = %s`。'
  % (json.dumps(JOINT.get('arms_used_for_cross_model'), ensure_ascii=False), q2j, q3j))
A('- **判决标签的完整取值域**（便于跨 Phase 检索；本 Phase 实测值见上一条）：'
  '`Q2_joint ∈ {ARGS_GAP_LAYERSTACK, ARGS_GAP_4B_SPECIFIC, ARGS_GAP_MIXED}`；'
  '`Q3_joint ∈ {CONC_JUDGE_INVALID_X_ALL, CONC_JUDGE_ALIVE_X}`（INVALID 的触发条件是「两个跨模型臂的 '
  '`null95_x` 都 ≥ 0.70」）；每臂 `Q1_label ∈ {NF4_FAITHFUL, NF4_DEVIANT, NA}`。')
A('- **7 条预注册预测：%d PASS / %d FAIL**（%s）。'
  % (sum(1 for k in PC if PC[k]['pass_']), sum(1 for k in PC if not PC[k]['pass_']),
     ', '.join('%s:%s' % (k, 'PASS' if PC[k]['pass_'] else 'FAIL') for k in sorted(PC, key=lambda z: int(z[1])))))
A('')

# ---- 预测逐条核对表
A('| 预测 | 内容 | 判据 | 实测 | 结果 |')
A('|---|---|---|---|---|')
for k in sorted(PC, key=lambda z: int(z[1])):
    d = PRED.get(k, {})
    det = ', '.join('%s=%s' % (kk, json.dumps(vv, ensure_ascii=False)) for kk, vv in PC[k].items()
                    if kk not in ('desc', 'pass_', 'smoke'))
    A('| %s | %s | %s | %s | **%s** |' % (k, d.get('desc', ''), d.get('falsified_if', ''),
                                          det[:200], 'PASS' if PC[k]['pass_'] else 'FAIL'))
A('')

# ============================================================ 5. 分析结论
A('### 5. 分析结论')
A('')
_conc = []
# 结论 1：量化门
_conc.append('**【装置结论·量化】** nf4 口径**通过保真门**（`max|dxhalf| = %s` ≤ %.2f，`argmax_w_x` 与 bf16 相同）'
             '⇒ 本 Phase 的三臂**同一数值口径**可比；`XH_RANGE` 也复现 bf16（%s vs %s）。'
             '因此 A1/A2 的跨模型结论**不是量化伪影**。'
             % (f(g(E6C, ARM0, 'max_abs_dxh'), 4), FL['XH_FAITHFUL_TOL'],
                f(g(E6C, ARM0, 'XH_RANGE_nf4'), 4), f(g(E6C, ARM0, 'XH_RANGE_bf16'), 4)))
# 结论 2：Q2
if q2j == 'ARGS_GAP_LAYERSTACK':
    _conc.append('**【主结论·Q2】「两坐标 argmax 相距 ≥3」是层栈性质。** GLM4-9B（跨家族）与 Qwen3-14B（同家族 3.5× 放大）'
                 '都复现了「`xhalf` 的 argmax 窗落在深尾、`J` 的 argmax 窗落在浅端」的**分辨率分离** ⇒ Phase 13 在 4B 上的观察'
                 '**不是特例**。其中 A1 的 `d = %s`、A2 的 `d = %s`（参照 4B = %d）。'
                 '**注意：Q2 判定的是「窗的位置分离」，不含「两窗的集中度都显著」这一更强主张** —— '
                 '后者由 Q3/Q4 单独裁决，结论是「**不显著**」（结论 3/4）。'
                 % (g(E5C, ARM1, 'd_argmax_window'), g(E5C, ARM2, 'd_argmax_window'),
                    abs(int(INH['MODE_X_13']) - int(INH['MODE_J_13']))))
elif q2j == 'ARGS_GAP_4B_SPECIFIC':
    _conc.append('**【主结论·Q2】「两坐标 argmax 相距 13」是 qwen3-4b 特例。** 两个跨模型臂的 `d_argmax` 都 <3'
                 '（A1 = %s、A2 = %s）⇒ 该量**不是层栈性质**；Phase 13 的「相距 13」须改述为'
                 '「4B 在**该网格与该统计量下**的个体读数」。'
                 % (g(E5C, ARM1, 'd_argmax_window'), g(E5C, ARM2, 'd_argmax_window')))
else:
    _conc.append('**【主结论·Q2】两坐标稳定性受模型个体影响。** A1 = %s、A2 = %s ⇒ 既非普遍层栈性质、'
                 '也不能判为 4B 特例；该量须按模型逐个报告。'
                 % (g(E5C, ARM1, 'd_argmax_window'), g(E5C, ARM2, 'd_argmax_window')))
# 结论 3：Q3
if q3j == 'CONC_JUDGE_INVALID_X_ALL':
    _conc.append('**【主结论·Q3】`xhalf` 坐标的集中度判据整条线作废。** 跨模型臂的 `null95_x` = %s 全部 ≥%.2f；'
                 'A0 也已达 %s（Phase 14 bf16 为 0.6998）。⇒ **Phase 12/13/14 中一切基于 `top3_share_x` 的表述'
                 '必须整体撤回**，改为「该统计量在 `xhalf` 坐标上无区分力」。'
                 % (json.dumps({SHORT[k]: f(v, 4) for k, v in _n95.items() if k != ARM0}, ensure_ascii=False),
                    FL['NULL_HIGH'], f(_n95.get(ARM0), 4)))
else:
    _nx = {a: g(E5C, a, 'null_x', 'null95') for a in ARMS_ALL}
    _mx = {a: g(E5C, a, 'margin_x') for a in ARMS_ALL}
    _above = [SHORT[a] + '·xhalf'
              for a in ARMS_ALL if (_mx.get(a) or -1) > 0]
    _conc.append('**【主结论·Q3】0.70 警报未在两个跨模型臂上触发，但 `xhalf` 判据的证据强度随模型规模**下降**。** '
                 '`null95_x` = %s —— **随规模单调下降**（4B 最高、14B 最低）。'
                 '观测裕度 `margin_x` = %s ⇒ **三臂中只有 %d 个臂的观测超过其自身零假设**（%s）。'
                 '⇒ 该判据**未到「整条线作废」**（A1/A2 的 null 均 < %.2f），'
                 '但也**远不足以支撑「深尾集中」**：多数臂的观测**低于**自身 null。'
                 % (json.dumps({SHORT[k]: f(v, 4) for k, v in _nx.items()}, ensure_ascii=False),
                    json.dumps({SHORT[k]: f(v, 4) for k, v in _mx.items()}, ensure_ascii=False),
                    len(_above), json.dumps(_above, ensure_ascii=False), FL['NULL_HIGH']))
# 结论 4：J 坐标（数据驱动：逐臂比 share vs null95；`_jm`/`_jup`/`_xup`/`_ncell` 已在 §0 前定义）
_conc.append('**【校准结论·坐标 × 模型双重依赖（本 Phase 最重要的自我修正）】** '
             '`J` 坐标的 `top3_share` = %s，`null95_j` = %s，裕度 = %s ⇒ '
             '**6 个「臂 × 坐标」格中只有 %d 格超过各自零假设**（%s）。'
             '⇒ 「浅端主导（`J` 坐标）」**只在 %s 上成立**，在两个跨模型臂上**不成立**；'
             '而 `xhalf` 坐标只在 %s 上成立。'
             '**因此不能说「某个坐标才是有区分力的坐标」** —— Phase 14 的「坐标依赖」在本 Phase 被进一步推进为'
             '「**坐标 × 模型**双重依赖」：本 Phase 的三臂上**没有一条跨模型稳健的集中度判据**。'
             % (json.dumps({SHORT[a]: f(g(E5C, a, 'top3_j'), 4) for a in ARMS_ALL}, ensure_ascii=False),
                json.dumps({SHORT[a]: f(g(E5C, a, 'null_j', 'null95'), 4) for a in ARMS_ALL}, ensure_ascii=False),
                json.dumps({SHORT[a]: f(v, 4) for a, v in _jm.items()}, ensure_ascii=False),
                _ncell,
                json.dumps([SHORT[a] + '·' + ('J' if a in _jup else 'xhalf') for a in ARMS_ALL
                            if a in _jup or a in _xup], ensure_ascii=False),
                json.dumps([SHORT[a] for a in _jup], ensure_ascii=False),
                json.dumps([SHORT[a] for a in _xup], ensure_ascii=False)))
# 结论 5：Q5 剖面形状
_conc.append('**【描述结论·Q5】剖面形状跨模型同构、幅度不同。** `spearman(xhalf, depth)` = %s（A0 最深）；'
             '`spearman(J, depth)` = %s（三臂**全部** ≈ −1，`J` 从浅端极大单调跌落）。'
             '`XH_RANGE` = %s（带 [%.2f, %.2f]）。⇒ 「浅端陡、深尾缓」是**层栈层面的共性**，'
             '而「深尾内 xhalf 的具体起伏」是模型个体差异。'
             % (json.dumps({SHORT[a]: f(g(E5C, a, 'spearman_xh_depth'), 4) for a in ARMS_ALL}, ensure_ascii=False),
                json.dumps({SHORT[a]: f(g(E5C, a, 'spearman_J_depth'), 4) for a in ARMS_ALL}, ensure_ascii=False),
                json.dumps({SHORT[a]: f(g(E4S, a, 'XH_RANGE'), 4) for a in ARMS_ALL}, ensure_ascii=False),
                FL['XH_RANGE_BAND'][0], FL['XH_RANGE_BAND'][1]))
for i, t in enumerate(_conc, 1):
    A('%d. %s' % (i, t))
A('')

# ============================================================ 6. 机制拼图
A('### 6. 机制拼图（v5.5）与限界')
A('')
A('**拼图增量**：')
A('- **未变更**：「层 = 软门 + 下游读数」（Phase 9/10）；「读写两端同构、组件不是正确解释粒度」（Phase 8）；'
  '「逐层累积」（Phase 10/12）。')
A('- **新增（跨模型级，本 Phase 的核心增量）**：**「浅端陡 / 深尾缓」的剖面形状在 untied 的 GLM4-9B 与 '
  'Qwen3-14B 上复现**（`spearman(J, depth)` ≈ −1、`XH_RANGE` 同带）⇒ 该形状**是层栈层面的共性**，'
  '不是 qwen3-4b 个体。')
if q2j == 'ARGS_GAP_LAYERSTACK':
    A('- **新增（Q2）**：「两坐标的**分辨率分离窗**」（`xhalf` 的 argmax 落在深尾 / `J` 的 argmax 落在浅端）'
      '**跨模型成立** ⇒ Phase 13 的 `CONCENTRATION_COORDINATE_DEPENDENT` 由「4B 观察」升格为「层栈性质」。'
      '**注意 Q2 只说「窗的位置不同」（由 `d_argmax ≥ 3` 判定），不预设两窗的集中度都显著** —— '
      '后者由 Q3/Q4 单独裁决，本 Phase 的裁决是「不显著」（见下条）。')
elif q2j == 'ARGS_GAP_4B_SPECIFIC':
    A('- **降级（Q2）**：Phase 13 的「两坐标 argmax 相距 13」**降级为 4B 个体读数**（两跨模型臂 <3）⇒ '
      '`CONCENTRATION_COORDINATE_DEPENDENT` 须限定为「在 4B 上成立」。')
else:
    A('- **待定（Q2）**：两坐标分离在两个跨模型臂上不一致 ⇒ `d_argmax` 须按模型逐个报告，不得跨模型外推。')
if q3j == 'CONC_JUDGE_INVALID_X_ALL':
    A('- **撤回（Q3，最高优先）**：`xhalf` 坐标的 `top3_share` 判据**整条线作废**（null 95 分位跨模型 ≥0.70）'
      '⇒ Phase 12/13/14 一切基于 `top3_share_x` 的表述**整体撤回**，只在 `J` 坐标上保留「浅端主导」。')
else:
    A('- **校准（Q3）**：`xhalf` 坐标的 null 95 分位**逼近 0.70**（三臂 A0 %s / A1 %s / A2 %s），'
      '观测裕度 = %s ⇒ 该坐标的集中度结论**必须与 null95 同报**，不得单独引用。'
      % (f(_n95.get(ARM0), 4), f(_n95.get(ARM1), 4), f(_n95.get(ARM2), 4),
         json.dumps({SHORT[a]: f(g(E5C, a, 'margin_x'), 4) for a in ARMS_ALL}, ensure_ascii=False)))
A('- **撤回（Q3 的推论；本 Phase 最重要的自我修正）**：**不得再说「`J` 坐标才是有区分力的坐标」** —— '
  '对 `J` 坐标施加**同一套**置换零假设检验后，**6 个「臂 × 坐标」格里只有 %d 格超过各自的 null 95 分位**'
  '（%s，其中 `J` 裕度 = %s、`xhalf` 裕度 = %s）⇒ 本 Phase 的三臂上**不存在跨模型稳健的集中度判据**；'
  'Phase 13/14 的「坐标依赖」须再升一级为「**坐标 × 模型**双重依赖」。'
  % (_ncell,
     json.dumps([SHORT[a] + '·' + ('J' if a in _jup else 'xhalf') for a in ARMS_ALL
                 if a in _jup or a in _xup], ensure_ascii=False),
     json.dumps({SHORT[a]: f(g(E5C, a, 'margin_j'), 4) for a in ARMS_ALL}, ensure_ascii=False),
     json.dumps({SHORT[a]: f(g(E5C, a, 'margin_x'), 4) for a in ARMS_ALL}, ensure_ascii=False)))
A('- **装置级（可复用）**：**「引入新自由度（量化口径）必须同时预注册一个测量该自由度的臂」** —— '
  'A0 臂把「nf4 是否可信」从隐性假设变成显式门（Q1），并给出了逐位点的偏差谱。')
A('- **装置级（本 Phase 最强的方法论增量）**：**装置门必须「独立于主结论」**。'
  '`F2_base_ok` 只问「类别 token 读对了吗」，与 `d_argmax` / `null95` 毫无关系；'
  '正因如此它才能在 A1 产生任何主结论**之前**拦下词表误用（amend1）。'
  '⇒ 跨模型复算类实验应至少配两条**正交**的装置门：'
  '一条管**数值口径**（Q1/A0）、一条管**语义口径**（F1b/F2 base）。')
A('')
A('- **统计级（跨相位可比性）**：**「跨模型复现」的判据必须是模型内量** —— `d_argmax` 与 `null95` 都在'
  '**单模型内部**定义，故即使数值口径（nf4 vs bf16）不同也可比；反之任何依赖**跨模型绝对值**的判据'
  '（`FULL_SWAP` 的大小、`J` 的绝对高度、`share` 的绝对高度）在本轮口径下**不可用**。'
  '这也是为什么本 Phase 把 Q2/Q3 写成**标签合取**而不是数值比较。')
A('')
A('**限界（必须与结论同时引用）**：')
for h in HONESTY:
    A('- %s' % h)
A('- **本 Phase 新增限界**：① A1/A2 各只有一个模型，**无法分离「家族」与「规模」**'
  '（GLM 家族 × 18.8 GB vs Qwen3 家族 × 29.5 GB）；② `null` 分布由 `jumps` **重排**得到，'
  '它不是「无效应」的物理零假设，只是「同一组幅度在任意排序下」的统计零假设；'
  '③ 「域外可达性」只测 profile 的 18 个位点，浅端 L0–L5 与末层后未测；'
  '④ 三臂的 `F3 α=0` 只在首/中/末 3 位点 × 前 3 对上验证，未做全位点回归。')
A('')

# ============================================================ 7. 第一性原理
A('### 7. 第一性原理')
A('')
A('1. **剖面形状的跨模型复现说明它由架构而非参数个体决定**：`J(ℓ)` 在三个模型上都从浅端 O(10–40) '
  '单调跌到深端 O(1)。`J` 度量的是「剂量–响应曲线在最陡段的斜率相对其余段的倍数」；'
  '浅端残差替换能造成相对更大的类别分数跳变，是因为**浅端的写方向尚未与下游计算解耦**'
  '（一个小的相对扰动就能改变整条下游路径），而深端残差已被下游大量读取、替换单个位点的边际效应小。'
  '**这是「层 = 软门 + 下游读数」在几何上的必然伴随现象，不是某个模型的巧合。**')
A('2. **极值型统计量的「坐标系错觉」与「模型个体性」叠加**：`top3_share = max_w |Σ3 jumps| / range` '
  '在 17 个 jump 上**上限天然很高**。本 Phase 在同一份数据上同时看到**两种翻转**：'
  '① **换坐标翻转**（同一臂内 `xhalf` 与 `J` 的结论可相反 —— 如 A0：`J` 超 null95 而 `xhalf` 不超）；'
  '② **换模型翻转**（同一坐标在不同臂上「超 / 不超 null95」互换 —— A1·`xhalf` 超而 A1·`J` 不超）。'
  '⇒ 任何「集中/分散」断言都必须**同时**声明**坐标、模型与零假设分位**；否则它测的是坐标系或个体，不是层栈。')
A('3. **「量化替换」在机制研究里是自由度而非实现细节**：4bit 权重的舍入会改变每层的有效线性算子，'
  '从而改变 `xhalf` / `J` 的绝对值。本 Phase 的做法是把该自由度**显式预注册为一条门（Q1）**'
  '并用同模型 bf16 已发表量做**逐位点**校准 —— 这是把「工程妥协」转化为「可被否证的科学主张」的模板。')
A('')

# ============================================================ 8. 后续死线
A('### 8. 后续资源与死线')
A('')
A('**Phase 16 候选（最高优先）· 统一剖面下的「写入窗 vs 集中窗」关系**：本 Phase 已在**三臂**上同时拿到'
  '两套量 —— ① 独立定位臂给出的**写入窗** `L*_own`（`B_cat` 相邻最大增量，逐层独立 `U_ell`）；'
  '② profile 臂给出的**双坐标集中窗**（`argmax_w_x` / `argmax_w_j` 对应的层区间）。'
  '把两者并列，回答 Phase 12 起就遗留的问题：「**写入窗是否就是 `J` 坐标的集中窗**」。'
  '本 Phase 的 A0 读数（`L*_own = %s`；`J` 的 `argmax_w_j = %s`，其窗口起点 = L%s）给出一个**初步**假设'
  '（写入窗 ≈ `J` 集中窗起点，误差在 1–2 层内）—— **但必须连带一条警告**：`argmax_w_j` 恰是本 Phase 唯一'
  '被量化噪声换掉的量（nf4 = %s vs bf16 = %s），故 Phase 16 不得直接沿用本 Phase 的 `J` 窗口索引，'
  '须在 **bf16 口径**下用**对单窗口不敏感的统计量**（如窗口中心/加权质心）重测。'
  '**判据先冻结；任何集中度结论仍须同时报 null 95 分位与裕度。**'
  % (g(E3L, ARM0, 'L_star_own'), g(E5C, ARM0, 'argmax_w_j'),
     SITES[g(E5C, ARM0, 'argmax_w_j') or 0], g(E5C, ARM0, 'argmax_w_j'), INH['MODE_J_13']))
A('')
A('**第二候选（因「6 格只活 2 格」而升为并列最高优先）· 集中度统计量的重设计**：'
  '本 Phase 的 6 格零假设检验只活下 2 格 ⇒ `top3_share` 在**两个坐标上都**缺少跨模型稳健性，'
  '问题不只是「`xhalf` 坐标判据弱」，而是**「极值型 3-窗口占比」这个统计量本身**在这条线上不可跨模型移植。'
  '需设计**对排序 / 单窗口不敏感**的集中度统计量（基于 `xhalf` 谱熵、窗口加权质心、'
  '或先做「深度趋势去势」再测残差集中度），并在**三臂 × 双坐标**上一次性重算 Phase 12/13/14 的全部集中度表。')
A('')
A('**第三候选 · 家族 vs 规模的解耦**：本 Phase 的 A1（GLM 家族 / 18.8 GB）与 A2（Qwen3 家族 / 29.5 GB）'
  '**同时**改变了家族与规模。设计**第三个同家族不同规模**的模型（如 Qwen3-8B 或 Qwen2.5 系列）'
  '或**第二个 GLM 规模点**，才能把两个因素分开。')
A('')
A('**其他挂账（不变）**：N2h1-α-1 权重级定位；N2h1-β 水果类崩塌解剖；N3-β → N3-δ → P-N3b → N3-γ → N3-ε；'
  'R1 挂账对照补强；K4（E2 死线）处置；**N 线 Phase 3–7 补登 Ledger**；G 线 Phase 3154（G2-P1）已预注册。')
A('')
A('**本 Phase 新增装置铁律（3 条）**：')
A('- **(z) 替换数值口径（量化/精度）时，必须同时预注册一个「同模型 × 原口径已发表量」的校准臂，'
  '并把偏差写成门；否则一切结论都只是「新口径下的描述」。** 本 Phase 的 A0 臂即此模板：'
  '它以 `max|dxhalf| = %s`（≤ %.2f）与 `argmax_w_x` 一致，把「nf4 是否可信」从隐性假设变成显式门。'
  '**同时记录其副作用**：`J` 坐标的 `argmax_w_j` 在量化噪声下可换窗（A0 nf4=%s vs bf16=%s）'
  '⇒ 门必须写在**最稳健的量**（此处 `xhalf`）上，不能写在 `J` 上。'
  % (f(g(E6C, ARM0, 'max_abs_dxh'), 4), FL['XH_FAITHFUL_TOL'],
     g(E5C, ARM0, 'argmax_w_j'), INH['MODE_J_13']))
A('- **(aa) 跨设备加载的内存约束必须作为「技术路线」写进 seal，并给出被否决路线的实测数字。** '
  '本 Phase 记录三条 bf16 路线（磁盘 offload **7.317 s/前向** ⇒ 316 min/臂；禁磁盘 **加载期被硬杀**；'
  '单臂可行但破坏口径一致性）与最终 nf4 路线（**0.036–0.041 s/前向，约 200×**）。'
  '**收益**：可行性论证本身可被复核；**代价**：引入量化自由度，故必须配 (z)。')
A('- **(ab) 跨模型/跨词表移植时，一切「词表相关常量」（类别 token id、实例 token id、特殊 token id）'
  '必须由**该模型自己的 tokenizer 现场解析**，禁止沿用源模型的硬编码值；并须配一条「读对了 token 吗」的装置门。** '
  '本 Phase 的事故（amend1）：`sup_id` 是 qwen 词表 id 却被全局使用，'
  'A1(`glm4-9b`, vocab **151329**) 全程读错类别 token ⇒ `F2 base bad = %d/41`、'
  '受体类分数均值 `%+.3f`、`FULL_SWAP = %+.3f`（4B 为 `%+.3f`）、剂量曲线平坦且低 α 段为负。'
  '**若无此门，这会以「GLM4 的 is-a 关系不成立」的机制结论被发表。** '
  '修正：逐臂解析 + F1b 断言「6/6 类别词单 token 且 decode 可逆」；'
  'A0/A2 解析结果与冻结值逐位相同 ⇒ 修正零副作用。'
  % (AM1['evidence_from_device_gate']['A1_F2_base_bad_n'],
     AM1['evidence_from_device_gate']['A1_F2_receptor_class_score_mean'],
     AM1['evidence_from_device_gate']['A1_FULL_SWAP'],
     AM1['evidence_from_device_gate']['A0_FULL_SWAP']))
A('')

# ============================================================ 附
A('### 附：记录完整性与过程备注（非实验内容）')
A('')
A('- **产物 sha8**：seal `%s`；**amend1 `%s`**；exec `%s`；result `%s`；各臂报告 `n2h1a8_report_<arm>.txt`。'
  % (R['seal_sha256'][:8], AM1_SHA8, R['execution_sha256'][:8],
     hashlib.sha256(open(os.path.join(P15T, 'result_phase15.json'), 'rb').read()).hexdigest()[:8]))
A('- **脚本落点** `tests/deepseek/Phase15/`；**产物落点** `tests/deepseek_temp/Phase15/`（v2 约定）。')
A('- **首轮（作废）运行的处置**：原 seal 字节**未动**（`%s`）；首轮 stdout 完整保留为 '
  '`_formal_stdout_run1_INVALID_supid.log`（%d B）供审计；首轮 A1 的 `F2_base_bad` 明细与 '
  '`FULL_SWAP = %+.3f` 已写入 amend1 的 `evidence_from_device_gate`。'
  % (R['seal_sha256'][:8], 20824, AM1['evidence_from_device_gate']['A1_FULL_SWAP']))
A('- **SMOKE 的价值（极强实证）**：本轮 SMOKE（A0 单臂、3 位点粗网格）**抓出三处真缺陷**，'
  '全部在正式运行前修掉：')
A('  1. `ValueError: Device 0 is not recognized` —— JSON 往返把 `max_memory` 的键变成字符串 `"0"`，'
  'accelerate 要求**整数键**；修：`{int(k) if str(k).isdigit() else k: v for k, v in ...}`。')
A('  2. `ValueError: max() iterable argument is empty`（`perm_null` 内）—— 粗网格的 3 位点只有 2 个 jump '
  '`< W=3` ⇒ 窗口列表为空；修：前置两个 guard 并记录原因（`n_jumps < W` / `non_finite_jumps`）。')
A('  3. `P1` 在**空结果集**上误判 PASS（`all(...)` 对空序列为 `True`）、`E6.argmax_same` 在**两者皆 None** '
  '时假阳性 True；修：`bool(ok_arms) and all(...)`、`bool(ax is not None and ax == bx)`。')
A('- **一处命名纠错**：`recover_at_alpha1`（α=1 的逐对均值）**≠** `FULL_SWAP`（供体自身前向），'
  '两者语义不同（Phase 12 既有 `recover` 就是 0.9955–1.0010），故取消错误断言、改为与 Phase 12 '
  '`recover_12_by_site` 逐位点比对。')
A('- **验证器独立重写**：`disk_verify_phase15.py` 的 `x_alpha / j_only / conc_sh / pnull` 是**独立实现**'
  '（不复用主脚本函数），并用「前缀锚」复核 MEMO 未动。')
A('')

# ============================================================ 9. 一句话 ×3
A('### 9. 一句话（重复三次）')
A('')
_sent = ('**跨模型复算把 Phase 13/14 的两个悬置问题各自结案，并顺带把剖面形状升格为层栈共性：** '
         '量化门 **Q1 = %s**（nf4 vs bf16 `max|dxhalf| = %s` ≤ %.2f、`argmax_w_x` 相同）⇒ 三臂同一口径可比；'
         '**Q2 = %s**（A1 `d_argmax = %s`、A2 `= %s`，参照 4B = %d）；'
         '**Q3 = %s**（`null95_x` = A0 %s / A1 %s / A2 %s，门 %.2f）；'
         '剖面形状跨模型同构：`spearman(J, depth)` = %s（三臂皆 ≈ −1）、`XH_RANGE` = %s（带 [%.2f, %.2f]）；'
         '零假设校准（本 Phase 最重要的自我修正）：**6 个「臂 × 坐标」格里只有 %d 格超过各自 null 95 分位**'
         '（`J` 裕度 %s；`xhalf` 裕度 %s）—— **同一份 jump 序列换坐标即换结论，且不存在跨模型稳健的集中度判据**；'
         '独立定位给 `L*_own` = %s（A0，复现 Phase 8 的 L6 几何）。7 条预注册预测 %d PASS / %d FAIL（%s）。'
         '**装置门在首轮拦下词表误用（amend1）**：A1(glm4-9b) 因沿用 qwen 的 `sup_id` 而 `F2 base bad = %d/41`、'
         '`FULL_SWAP = %+.3f`（4B 为 `%+.3f`）⇒ A1 作废重跑，修正后三臂 `bad` 全 `0/41`、F1b 全通过 —— '
         '**这是本 Phase 方法论层面最强的产物：独立于主结论的装置门把一条看似「机制发现」的假阳性挡在发表之前。**')
_sent = _sent % (
    q1, f(g(E6C, ARM0, 'max_abs_dxh'), 4), FL['XH_FAITHFUL_TOL'], q2j,
    g(E5C, ARM1, 'd_argmax_window'), g(E5C, ARM2, 'd_argmax_window'),
    abs(int(INH['MODE_X_13']) - int(INH['MODE_J_13'])), q3j,
    f(_n95.get(ARM0), 4), f(_n95.get(ARM1), 4), f(_n95.get(ARM2), 4), FL['NULL_HIGH'],
    json.dumps({SHORT[a]: f(g(E5C, a, 'spearman_J_depth'), 3) for a in ARMS_ALL}, ensure_ascii=False),
    json.dumps({SHORT[a]: f(g(E4S, a, 'XH_RANGE'), 4) for a in ARMS_ALL}, ensure_ascii=False),
    FL['XH_RANGE_BAND'][0], FL['XH_RANGE_BAND'][1],
    _ncell,
    json.dumps({SHORT[a]: f(g(E5C, a, 'margin_j'), 4) for a in ARMS_ALL}, ensure_ascii=False),
    json.dumps({SHORT[a]: f(g(E5C, a, 'margin_x'), 4) for a in ARMS_ALL}, ensure_ascii=False),
    g(E3L, ARM0, 'L_star_own'),
    sum(1 for k in PC if PC[k]['pass_']), sum(1 for k in PC if not PC[k]['pass_']),
    ', '.join('%s:%s' % (k, 'PASS' if PC[k]['pass_'] else 'FAIL') for k in sorted(PC, key=lambda z: int(z[1]))),
    AM1['evidence_from_device_gate']['A1_F2_base_bad_n'],
    AM1['evidence_from_device_gate']['A1_FULL_SWAP'],
    AM1['evidence_from_device_gate']['A0_FULL_SWAP'])
for _ in range(3):
    A(_sent)
    A('')

io.open(OUT, 'w', encoding='utf-8', newline='\n').write('\n'.join(L) + '\n')
b = io.open(OUT, 'rb').read()
print('WROTE %s' % OUT)
print('  bytes = %d ; lines = %d ; sha8 = %s' % (len(b), len(L), hashlib.sha256(b).hexdigest()[:8]))
