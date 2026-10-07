# -*- coding: utf-8 -*-
"""Q04 落账（日志侧）：AGI_DEEPSEEK_MEMO.md 追加 Phase 38（append-only，BOM+CRLF，前缀逐字节不动）。
数字一律从 q04_smoke_result.json / q04_smoke_execution.json / metric_dict.json 现场渲染。"""
import os, json, time, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
SMO = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q04_smoke_result.json')
SMEX = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q04_smoke_execution.json')
MD = os.path.join(ROOT, 'research', 'deepseek', 'atlas', 'metric_dict.json')
BOM = b'\xef\xbb\xbf'
TS = time.strftime('%Y-%m-%d %H:%M')

q = json.loads(open(SMO, 'rb').read().decode('utf-8-sig'))
ex = json.loads(open(SMEX, 'rb').read().decode('utf-8-sig'))
md = json.loads(open(MD, 'rb').read().decode('utf-8-sig'))
K = q['K']
PS = q['per_seed']

L = []
def A(s):
    L.append(s)

A('## Phase 38: Q04 E_ar(k) 装置建造 —— k 步自回归 logit-margin 误差曲线装置 + K=16 + 预注册冻结，'
  'SMOKE 4/4 装置门通过（B 闸门；gpu=mid；qwen3-4b）[__TS__]')
A('')
A('> 类型：**装置建造 + 预注册冻结 + SMOKE**（新模型观测，但仅子面板；非正式测量）。依据宪法 §1 (I1) + '
  '`research/deepseek/atlas/metric_dict.json` → `global_kpis.E_ar`（v2 中 status=`not_built`）。队列 **Q04**（B KPI，登记 gpu=mid）。')
A('> 产物：`tests/deepseek/result/q04_smoke_execution.json`（预注册；design_sha `__DS__`）、`q04_smoke_result.json`'
  '（res_sha8 `__RS__`）、`q04_smoke_report.txt`；装置脚本 `tests/deepseek/q04_e_ar_device.py`；'
  '独立复核 `verify_q04.txt` = **14 PASS / 0 FAIL / ALL_PASS**。')
A('> 登记：E_ar 口径写回 `metric_dict.json` **v2→v3**（file `03887e51`→`733ea24f`，content `5ce974ee`→`72a3d2c4`，v2 已备份）；'
  '队列 Q04 → `device_built`（`d0d2e208`→`631c1ba3`）。')
A('')
A('### 1. 目标与状态')
A('')
A('**已执行（装置层）**。E_ar(k) 是宪法 §1 三 KPI 之一，v2 中只有口径、**从未被建造过**（历史 MEMO 中 `E_ar` 出现 0 次）。'
  '本 Phase 把它从"口径预注册"推进到"可运行装置"：冻结面板族、冻结 K、冻结 margin 语义、冻结预测器族、冻结可失败的装置门，'
  '并在 qwen3-4b 子面板上跑通 SMOKE。**正式曲线（三模型 × 全面板）留给 Q05。**')
A('')
A('### 2. 原理与算法（通俗）')
A('')
A('- **要测什么**：模型在"自己续写 k 步之后的语境里"，对「目标类 vs 竞争类」的 logit 差距（margin）还能不能被一个**加性模型**预测。')
A('- **为什么这样定义**：`E_read` 测的是"未见组合的读出层 hidden 能不能被加性预测"；`E_ar(k)` 用**同一 B4 加性族**、'
  '**同一 held-out 折**，只把被预测对象换成"k 步自回归之后的 logit margin"。两者同族 ⇒ 可并排读，构成 §1 的 KPI 曲线。')
A('- **margin = logit(t_target) − logit(t_competitor)**：`t_target` = 该 cell 真实类名首 token；'
  '`t_competitor` = 在 k=0（纯模板前缀）处 logit 最高的"非该类"类首 token，**每 cell 冻结**，不随 k 变。')
A('- **自回归 rollout**：上下文从模板前缀开始（如「苹果是一种」），每步读 last-position 的 6 类 logit，'
  '然后把 **模型自己的 argmax token 回喂**——无 teacher forcing，纯自生成。k=0 即前缀本身。')
A('- **k=0 是装置锚**：两臂在 k=0 处同一上下文 ⇒ 用来做确定性自检（D1）。')
A('')
A('### 3. 装置规格（冻结）')
A('')
A('- **面板族**：与 E_read **同一 held-out 面板族**（41 实体 × 6 类 = 246 pairs × 3 模板 = 738 行；'
  'S1 seeds=[7,8,9]，frac=0.2 ⇒ 49 test pairs × 3 = **147 held-out 行/seed**）。复核 V3 已证 test_pairs_sha8 与 Q03 指纹逐项相同。')
A('- **K = %d**（报告 k=0..%d；k=0 为装置锚）。SMOKE 用 K=%d。'
  % (ex['device']['K'], ex['device']['K'], q['K']))
A('- **预测器**：B4 加性 = `ridge_primal(one-hot[entity 41] + one-hot[class 6] + one-hot[template 3] + bias, λ=1e-3)`，与 E_read 同实现。')
A('- **单位**：按冻结公式取**原始 L1（logit，不归一）**；另报 `rel = E_ar/scale`（scale = held-out margin_true 的 std）为派生参照。')
A('- **装置门**：D1 k=0 逐位确定性 / D2 margin 非退化（std>0）/ D3 全有限 / S1 `max_{k>=1} E_ar(k) ≥ 0.05`。')
A('')
A('### 4. 预注册（观测前冻结）')
A('')
A('- `design_sha = `**`__DS__`**（content-excluding-self，可复算；复核 V1 PASS）。')
A('- 冻结时刻 `design` 内**不含时间戳**：首次冻结曾因把 `created` 放进 design 导致哈希不可复算（`DESIGN DRIFT`），'
  '已改为"易变字段不入哈希域"——这是本 Phase 的一处工程教训（见 §8）。')
A('- 冻结前无任何模型观测；SMOKE 观测在冻结之后。')
A('')
A('### 5. SMOKE 实际结果（%s，%d cells，%d forwards）'
  % (q['model'], q['n_cells'], q['sum_fwd']))
A('')
A('| k | E_ar(B4) | null(const) | scale | rel=err/scale | drift(k)−drift(0) |')
A('|---|---|---|---|---|---|')
for k in range(K + 1):
    A('| %d | %.4f | %.4f | %.4f | %.4f | %+.4f |'
      % (k, q['E_ar'][str(k)], q['E_ar_const'][str(k)], q['scale'][str(k)],
         q['E_ar_rel'][str(k)], q['drift'][str(k)]))
A('')
A('**per-seed（原始 L1, k=0..%d）**：' % K)
A('')
for s in ['7', '8', '9']:
    A('- seed %s：%s' % (s, ' / '.join('%.4f' % PS[s][str(k)]['mae_b4'] for k in range(K + 1))))
A('')
A('### 6. 装置门判定')
A('')
A('| 门 | 判据 | 结果 |')
A('|---|---|---|')
A('| D1 | k=0 逐位确定性 | **%s**（同 cell 重跑 6 类 logit 逐位相同） |' % ('PASS' if q['gates']['D1'] else 'FAIL'))
A('| D2 | margin 非退化 | **%s**（held-out margin std(k=0,seed7)=%.4f） |'
  % ('PASS' if q['gates']['D2'] else 'FAIL', q['heldout_margin_std_k0_seed7']))
A('| D3 | 全有限 | **%s** |' % ('PASS' if q['gates']['D3'] else 'FAIL'))
A('| S1 | max_{k>=1} E_ar ≥ %.2f | **%s**（max=%.4f） |'
  % (q['gates']['S1_thr'], 'PASS' if q['gates']['S1'] else 'FAIL', q['gates']['S1_max_k_ge1']))
A('')
A('**判决 = `%s`**。跨进程复现：三次独立运行 `res_sha8` 恒为 `__RS__`，per-seed 数字逐位相同 ⇒ '
  '装置确定性成立（不止"同进程内 D1"）。' % q['verdict'])
A('')
A('### 7. ⚠ 口径观察：S1 门被平凡满足（不追溯改门，另立 Q05 科学门）')
A('')
A('- **观察到**：冻结的 S1 阈 0.05 以**原始 logit** 为单位，而实测 max=%.4f ⇒ 超阈 **%.0f×**，'
  '门"形式上可失败、实际上必过"，**不具科学否证力**。' % (q['gates']['S1_max_k_ge1'], q['gates']['S1_max_k_ge1'] / q['gates']['S1_thr']))
A('- **纪律处置**：按"阈值冻结后不得事后调门"，**不回头修改** S1；S1 保留为**装置灵敏度门**'
  '（它证明的是"装置能看见漂移"，不是"漂移在科学上成立"）。')
A('- **Q05 科学门已在本 Phase 观测后、Q05 观测前预注册**（写入 `result.q05_prereg`，随结果封存）：')
A('  - `S_rel: min_{k=1..K} E_ar_rel(k) ≤ 0.05`，其中 `E_ar_rel(k) = E_ar(k)/scale(k)`，scale = held-out margin_true 的 std。')
A('  - `shape`: 按 `DRIFT(k)=E_ar(k)−E_ar(0)` 的符号与单调性判决 {linear, saturating, diverging}。')
A('')
A('### 8. 工程教训（并入 closeout 技能候选）')
A('')
A('1. **冻结哈希域只含不变量**：design 内混入 `created` 等时间戳 ⇒ 每次重跑 `DESIGN DRIFT`，预注册形同废纸。'
  '固定做法：易变字段（时刻/运行机/耗时）一律置于哈希域之外。')
A('2. **阈值必须带单位体检**：新 KPI 首次建装置时，要问"这个阈值的**量纲**是否使门可失败"。'
  'E_ar 是 logit 量纲，套用 E_read 的 0.05（比值族）会得到"必过门"。')
A('3. **哈希比对宽度**：`content_sha256_8` 是 8 位截断，比较函数必须同样截断（本轮一度因 64 位 vs 8 位误报 self-hash 不符）。')
A('')
A('### 9. SMOKE 局限（数字不可科学解读）')
A('')
A('1. 子面板取 `PAIRS[:60]` = **仅前 10 个实体** × 6 类；41 个 entity one-hot 中 **31 列为零** ⇒ '
  '未见实体处 B4 回退到 class+template，E_ar 偏大、per-seed 方差大（seed7/8/9 在 k=0 为 %.4f/%.4f/%.4f）'
  % (PS['7']['0']['mae_b4'], PS['8']['0']['mae_b4'], PS['9']['0']['mae_b4']))
A('   —— 这是子面板伪影，**正式曲线须看 Q05 全面板**。')
A('2. SMOKE 曲线非单调（k=1 抬升后回落），在 K=4 + 10 实体下不足以判决形状；形状判决留给 Q05。')
A('3. rollout 样例显示前缀相同的 cell 得到相同续写（如前 3 个 cell 均为「水果，对吗」）——'
  '因 `P0`（模板前缀）只依赖实体与模板、不依赖类，故同实体同模板的 6 个类共享同一 rollout，'
  '差异只来自读出哪一对 (target, competitor)。此为设计性质，非缺陷。')
A('')
A('### 10. Q05 资源前置（登记为风险）')
A('')
A('- `qwen3-14b` bf16 ≈ **29.6 GB**、`glm4-9b` bf16 ≈ **18.8 GB**，均 **> 本机 16 GB 显存**；'
  '`qwen3-4b`（8.1 GB）可原样跑。')
A('- Q05 正式测量**必须先定**量化/offload 方案，并**显式声明**其与 E_read（bf16 采集）之间的精度差异，'
  '否则跨 KPI 比较会引入未声明偏差。')
A('')
A('### 11. 硬伤')
A('')
A('1. E_ar 与 E_read **量纲不同**（原始 L1 vs 归一化 MSE），不可直接数值比较；并排读时只能用 `rel` 作桥。')
A('2. 竞争类 `t_competitor` 冻结在 k=0：若 rollout 强烈改变类 logits，冻结的竞争类会**过时**，'
  '该变化被计入 E_ar（属测得的漂移的一部分，非 bug），但使"margin 漂移"与"竞争类漂移"不可分离。')
A('3. `E_ar` 的 null 对照是"训练均值常数"；B4 相对 null 有下降（如 k=0：%.4f vs %.4f），但**未做显著性检验**'
  '（装置阶段无此要求）。' % (q['E_ar'][str(0)], q['E_ar_const'][str(0)]))
A('4. 装置只在 **1 个模型**上 SMOKE；跨模型可运行性（尤其 14B/9B 的显存问题）**未验证**。')
A('')
A('### 12. 后续')
A('')
A('- B 闸门下一项：**Q05 `E_ar(k)` 正式测量** —— 三模型 × 全面板 738 行 × k=0..16 曲线，'
  '按预注册 `S_rel` + shape 判据判决。**前置**：定量化方案。')
A('- 之后的 Q06 `C_steer` 基座测量（登记 gpu=mid）。')
A('- 挂账不变：跨线账本补丁（C4/C6）施加确认、N 线 P3–P7 补 Ledger、N2h1-α-1 权重级、水果类、K4；'
  '以及 R9 遗留的"Phase 36 段 45 个裸 LF 是否规范化"。')
A('')
A('### 13. 一句话 ×3')
A('')
A('1. **E_ar(k) 装置已建成并可复现**：design_sha `__DS__`、res_sha8 `__RS__`（三次运行恒等），SMOKE 4/4 装置门通过。')
A('2. **口径已从"纸面"落到"可运行"并登记**：metric_dict v2→v3 记录 E_ar 的 margin 语义、K=16、B4 预测器、'
  '装置门与 Q05 预注册门。')
A('3. **诚实边界**：SMOKE 数字是 10 实体子面板的装置验证，**不可科学解读**；正式曲线与形状判决属 Q05，'
  '且 Q05 面临 14B/9B 显存与精度声明的硬前置。')
A('')

entry_lf = '\n'.join(L)
entry_lf = (entry_lf.replace('__TS__', TS)
            .replace('__DS__', ex['design_sha'])
            .replace('__RS__', q['res_sha8']))
assert '__TS__' not in entry_lf and '__DS__' not in entry_lf and '__RS__' not in entry_lf, 'placeholder residue'

raw = open(MEMO, 'rb').read()
assert raw[:3] == BOM, 'BOM missing'
body = entry_lf.replace('\n', '\r\n').encode('utf-8')
new = raw + (b'' if raw.endswith(b'\r\n') else b'\r\n') + body
with open(MEMO, 'wb') as f:
    f.write(new)
rb = open(MEMO, 'rb').read()
assert rb[:len(raw)] == raw, 'PREFIX CHANGED!'
t = rb.decode('utf-8-sig')
print('MEMO before=%d B/%s  after=%d B/%s  prefix_ok=%s' %
      (len(raw), hashlib.sha256(raw).hexdigest()[:8], len(rb),
       hashlib.sha256(rb).hexdigest()[:8], rb[:len(raw)] == raw))
print('        Phase 38 in place = %s ; bare_lf now = %d'
      % ('## Phase 38' in t, t.count('\n') - t.count('\r\n')))
print('DONE')
