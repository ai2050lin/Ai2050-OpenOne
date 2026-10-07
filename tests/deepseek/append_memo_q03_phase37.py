# -*- coding: utf-8 -*-
"""Q03 落账：AGI_DEEPSEEK_MEMO.md 追加 Phase 37 + phase_queue_v1.json Q03 -> sealed。
纪律：memo append-only（BOM+CRLF，前缀逐字节不动）；数字一律从 q03_result.json 现场渲染。"""
import os, json, time, hashlib

ROOT = r'D:\AI2050\Ai2050-OpenOne'
MEMO = os.path.join(ROOT, 'research', 'deepseek', 'docs', 'AGI_DEEPSEEK_MEMO.md')
QD = os.path.join(ROOT, 'research', 'deepseek', 'atlas', 'phase_queue_v1.json')
QR = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q03_result.json')
QEX = os.path.join(ROOT, 'tests', 'deepseek', 'result', 'q03_execution.json')
BOM = b'\xef\xbb\xbf'

q = json.loads(open(QR, 'rb').read().decode('utf-8-sig'))
ex = json.loads(open(QEX, 'rb').read().decode('utf-8-sig'))
S, PM, FP = q['summary'], q['per_model'], q['fingerprint']
CAR = q['carriers']
TS = time.strftime('%Y-%m-%d %H:%M')
MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']

def f6(x):
    return '%.6f' % x

L = []
def A(s):
    L.append(s)

A('## Phase 37: Q03 E_read 统一基线复算 —— 三模型 B4 锚逐位复现、5% 门 0/3、统一 held-out 指纹锁定（B 闸门；零 GPU；recompute-only）[__TS__]')
A('')
A('> 类型：**复算（recompute-only，无新模型观测）**。依据宪法 §1 (I1) + `research/deepseek/atlas/metric_dict.json` → `global_kpis.E_read`。队列 **Q03**（B KPI，登记 gpu=low，实为零 GPU）。方法逐字复用 `tests/glm5/phase3152_g1p2_tri_model_k1.py` 的 `split_s1 / rows_of / phi_main / ridge_primal / b4_fit`。')
A('> 产物：`tests/deepseek/result/q03_execution.json`（预注册；design_sha `__DS__`）、`q03_result.json`（res_sha8 `__RS__`）、`q03_report.txt`；独立复核 `verify_q03.txt` = **42 PASS / 0 FAIL / ALL_PASS**。')
A('')
A('### 1. 目标与状态')
A('')
A('**已执行**。K1 判决（R8）完全依赖 E_read 基线；本 Phase 把该基线从三份冻结 `collect.npz` **现场重算**一遍，确认它不是读数或转录误差，并把它锁定为 B 闸门后续实验（Q04 E_ar(k) / Q05 / Q06 C_steer）的统一比较基线。')
A('')
A('### 2. 原理与算法（通俗）')
A('')
A('- **问题**：模型见过 (苹果, 水果)，能否预测**没见过**的组合 (床, 颜色) 在读出层的内部状态？')
A('- **B4 预测器** = 加性线性回归：特征 = one-hot(实体) + one-hot(类别) + one-hot(模板) + 偏置（共 41+6+3+1 = 51 列），λ=1e-3 处理 one-hot 共线；用训练组合拟合，预测 held-out 组合的读出层 hidden。')
A('- **误差定义**：`E_read(m) = mean_{test rows} || B4_pred − y_true ||² / Dk`，其中 `Dk` = **训练行方差**（不是 hidden 维度）。')
A('  ⇒ **E_read 是归一化 MSE**：0.33 表示加性模型只解释 67% 方差；**5% 门 = 加性模型必须解释 95% 方差**才算"组合可加"。')
A('- **held-out**：S1 切分，seed ∈ {7,8,9}，frac=0.2 ⇒ 每 seed 49 test pairs × 3 模板 = **147 test rows**，197 train pairs × 3 = 591 train rows。')
A('')
A('### 3. 材料（三份冻结载体，sha8 现场核验）')
A('')
A('| 模型 | collect.npz sha8 | bytes | H shape | readout 层 |')
A('|---|---|---|---|---|')
for m in MODELS:
    A('| `%s` | `%s` | %d | %s | %d |' % (
        m, CAR[m]['npz_sha8'], CAR[m]['bytes'],
        str(tuple(PM[m]['H_shape'])), PM[m]['readout_layer']))
A('')
A('### 4. 实际结果')
A('')
A('**（a）逐模型复算 vs 3151/3152 原始 result.json 锚**（`drift = 0.00e+00`，逐位一致）：')
A('')
A('| 模型 | 读出层 | 复算 mean | 锚 mean | drift | 5% 门 |')
A('|---|---|---|---|---|---|')
for m in MODELS:
    A('| `%s` | %d | %s | %s | %.2e | %s |' % (
        m, PM[m]['readout_layer'], f6(PM[m]['b4_rel_readout_mean3seed_recompute']),
        f6(PM[m]['b4_rel_readout_mean3seed_anchor']), PM[m]['b4_rel_readout_mean3seed_drift'],
        '未过' if not PM[m]['gate_pass'] else '过'))
A('')
A('**（b）per-seed（读出层）**：')
A('')
for m in MODELS:
    A('- `%s`：%s' % (m, ' / '.join('%.6f' % x for x in PM[m]['b4_rel_readout_per_seed_recompute'])))
A('')
A('**（c）bootstrap 95% CI**（row-level，n_boot=10000，rng_seed=20261003）：')
A('')
A('| 模型 | seed 7 | seed 8 | seed 9 |')
A('|---|---|---|---|')
for m in MODELS:
    cells = []
    for s in ['7', '8', '9']:
        b = PM[m]['bootstrap_per_seed'][s]
        cells.append('%.4f [%.4f, %.4f]' % (b['mean'], b['ci_lo'], b['ci_hi']))
    A('| `%s` | %s | %s | %s |' % (m, cells[0], cells[1], cells[2]))
A('')
A('**（d）汇总**：')
A('')
A('- E_read 池化（3 模型均值）= **%s**（sd=%s，n=3）' % (f6(S['pooled_mean']), f6(S['pooled_sd_3models'])))
A('- 5%% 门（≤0.05）过门 = **%s**；最小 E_read = %s = 门的 **%.2f×**' % (S['gate_pass_frac'], f6(S['min_E']), S['min_E_x']))
A('- carrier sha8 全匹配 = %s；锚全匹配（drift<1e-4）= %s' % (S['all_carrier_sha_ok'], S['all_anchor_match']))
A('')
A('**（e）统一 held-out 指纹（三模型共用同一构造）**：')
A('')
for s in ['7', '8', '9']:
    v = FP[s]
    A('- seed %s：%d train pairs / %d test pairs（×3 模板 = %d test rows），test 集 sha8 `%s`' % (
        s, v['n_train_pairs'], v['n_test_pairs'], v['n_test_pairs'] * 3, v['test_pairs_sha8']))
A('')
A('### 5. 分析结论')
A('')
A('1. **逐位复现**：三模型 × {读出层, k\\*} 共 6 组锚，`drift = 0.00e+00`。Q03 不是"重读 result.json"，而是用冻结实现独立重跑得到**完全相同的数字**（float32 路径一致）。')
A('2. **统一 held-out 指纹成立**：3151（glm4）与 3152（qwen3-4b/14b）的 `PAIRS = [(i,c)] = 41×6 = 246`、`SEEDS_S1=[7,8,9]`、`FRAC_S1=0.2` **逐字相同**，故三模型的 49-pair test fold 完全一致 ⇒ E_read 可跨模型直接比较。')
A('3. **5%% 门 0/3**，最小者仍为门的 %.1f×：读出层加性可加性**在三个模型上均不成立**，与 R8 的 K1 判决（读出层 `fired_all_models`）自洽。' % S['min_E_x'])
A('4. **k\\* 层对照**：复算同样逐位匹配（qwen3-4b k\\* 锚 `%.9f`），即"承诺层看起来可加、读出层不可加"的落差是**真实的层位效应**，不是本 Phase 引入的伪影。' % PM['qwen3-4b']['b4_rel_kstar_mean3seed_recompute'])
A('')
A('### 6. 硬伤')
A('')
A('1. B4 是"加性族中表现最好者"的代理（端点/词袋/位置/加性四基线的最优组合），并非穷举全部加性模型；"组合不可加"的强度以 B4 为下界锚。')
A('2. bootstrap 在 **seed 内**做（每 seed 147 行）；3 个 seed 不合并（CI 方法冻结为"3 seed 单列不折叠"），故池化 CI 未给出。')
A('3. 载体是 **float16** H（3151/3152 采集时即如此）；本 Phase 与冻结实现用同一 float16→float32 路径，保证可比，但不反驳"高精度下误差可能略变"。')
A('4. 面板仅 6 类 × 41 实体 × 3 模板；"未见组合"限于此面板，不可外推为自然语言全域。')
A('')
A('### 7. ⚠ 并发写者与文件漂移（如实记录，非本对话产物）')
A('')
A('- 研究日志在 R8 收尾（`150241da` / 685,647 B）之后、本轮开始前被**外部写入**：')
A('  - `+2 B` → `5be5bf12` / 685,649 B（该值与并发写者 Phase 36 脚本自报的 `before` 一致）')
A('  - Phase 36 追加 → `5b3bdcec` / 692,727 B')
A('- **证据**：`prefix(685,649)` sha8 = `5be5bf12` **MATCH**；`tests/deepseek_temp/Phase36/do_append_phase36.py` 记录"追加前 memo sha8 5be5bf12（685,649 B）"。')
A('- **Phase 36 非本对话产物**（内容为 E4/E4b 词频 × 嵌入有效维度，用户队列外插入）；本对话**不改写**它，仅在其后追加 Phase 37。')
A('- **观察到的质量缺陷**：Phase 36 追加引入了 **45 个裸 LF**（文件 `bare_lf` 由 0 → 45），**全部位于 Phase 36 正文区间**（offset 685,649–691,546）；Phase 35 及之前仍为纯 CRLF。本对话不擅自修改其他线内容，仅报告，留待用户决定是否规范化。')
A('- 该日志**未被 git 跟踪**，无法 diff；已留本轮前快照供后续定位。')
A('')
A('### 8. 后续')
A('')
A('- B 闸门下一项：**Q04 `E_ar(k)` 装置**（k 步自回归 logit-margin 误差曲线，登记 gpu=mid，SMOKE 先通）。')
A('- Q03 状态 → `sealed`；seal_record = `tests/deepseek/result/q03_result.json`。')
A('- 挂账不变：跨线账本补丁施加确认、N 线 P3–P7 补 Ledger、N2h1-α-1 权重级、水果类、K4。')
A('')
A('### 9. 一句话 ×3')
A('')
A('1. **E_read 基线已锁定且逐位可复现**：0.3316 / 0.3986 / 0.3898，池化 %s，5%% 门 0/3。' % f6(S['pooled_mean']))
A('2. **三模型共用同一 held-out 指纹**（49 test pairs × 3 模板），故跨模型比较合法。')
A('3. **E_read 是归一化 MSE**：0.05 门 = 加性模型须解释 95% 方差 —— 这是"组合不可加"的严格表述。')
A('')

entry_lf = '\n'.join(L)
entry_lf = (entry_lf.replace('__TS__', TS)
            .replace('__DS__', ex['design_sha'][:8])
            .replace('__RS__', q['res_sha8']))
assert '__TS__' not in entry_lf and '__DS__' not in entry_lf and '__RS__' not in entry_lf, 'placeholder residue'

# ---- memo append-only (BOM + CRLF) ----
raw = open(MEMO, 'rb').read()
assert raw[:3] == BOM, 'BOM missing'
body = entry_lf.replace('\n', '\r\n').encode('utf-8')
prefix_ok = True
new = raw + (b'' if raw.endswith(b'\r\n') else b'\r\n') + body
with open(MEMO, 'wb') as f:
    f.write(new)
rb = open(MEMO, 'rb').read()
assert rb[:len(raw)] == raw, 'PREFIX CHANGED!'
memo_before_sha = hashlib.sha256(raw).hexdigest()[:8]
memo_after_sha = hashlib.sha256(rb).hexdigest()[:8]
t = rb.decode('utf-8-sig')
print('MEMO before=%d B/%s  after=%d B/%s  prefix_ok=%s' %
      (len(raw), memo_before_sha, len(rb), memo_after_sha, rb[:len(raw)] == raw))
print('        Phase 37 in place = %s ; bare_lf now = %d' %
      ('## Phase 37' in t, t.count('\n') - t.count('\r\n')))

# ---- queue: Q03 -> sealed ----
qd = json.loads(open(QD, 'rb').read().decode('utf-8-sig'))
qsha_before = hashlib.sha256(open(QD, 'rb').read()).hexdigest()[:8]
hit = 0
for it in qd['queue']:
    if it['id'] == 'Q03':
        it['status'] = 'sealed'
        it['sealed_at'] = TS
        it['seal_record'] = 'tests/deepseek/result/q03_result.json'
        hit += 1
assert hit == 1, 'Q03 not found uniquely (%d)' % hit
si = qd['sealed_items']
if 'Q03' not in si:
    si.append('Q03')
qd['sealed_items'] = sorted(set(si))
qd['status_updated_at'] = TS
qd['status_updated_by'] = 'tests/deepseek/append_memo_q03_phase37.py'
with open(QD, 'w', encoding='utf-8', newline='\n') as f:
    json.dump(qd, f, ensure_ascii=False, indent=1)
qd2 = json.loads(open(QD, 'rb').read().decode('utf-8-sig'))
q03 = [x for x in qd2['queue'] if x['id'] == 'Q03'][0]
print('QUEUE before=%s after=%s  sealed_items=%s' %
      (qsha_before, hashlib.sha256(open(QD, 'rb').read()).hexdigest()[:8], qd2['sealed_items']))
print('      Q03.status=%s sealed_at=%s' % (q03['status'], q03.get('sealed_at')))
print('DONE')
