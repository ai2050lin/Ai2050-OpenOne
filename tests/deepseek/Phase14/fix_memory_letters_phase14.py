# -*- coding: utf-8 -*-
"""回正 MEMORY.md §1 的铁律字母错位（本轮重写引入的回归）。

权威映射（从 deepseek MEMO 逐条验证）：
  (j) 绝对/相对双剂量坐标  (k) 判据符号与物理方向一致  (l) r_ℓ 差 3–17 倍先查方向旋转
  (m) 读数位点探针  (n) 固定基报 overlap  (p) 非线性统计量须给误差带+零假设  (q) 构造决定位点写硬断言
  (r) 端点构造饱和 → 降级  (s) 归一化方向须与物理方向一致  (t) 集中度 ≥2 坐标 + argmax
  (u) 不可排序须写区间口径  (v) 独立口径须先证可分  (w) 面板级 vs 逐对作用域分离
  (x) 预注册预测符号自洽  (y) GQA 禁反推 head_dim
  未变：(a)–(i) 沿用（h 冒烟产物落 smoke/、i 冒烟截断保留 α=1、o 同消息多 Edit 静默丢失）
逐处 assert count==1 + 回读。
"""
import io
import hashlib
import os

P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md'
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\deepseek_temp\Phase14\fix_memory_letters_phase14.txt'

R = []

R.append((
    "- (a) 份额用**精确可加量（向量预算）**；(b) SMOKE 必做且必看数字，产物落 `smoke/`；(c) 门裕度写进报告；"
    "(d) 预注册/修正案都记 sha8，修正案须**运行前**冻结 + `why_not_a_HARKing_violation`；"
    "(e) 行为端与层内端同测（`amp`）；(f)「充分」用放大臂、「必要」用撤除臂；"
    "(g) 曲线判据**参数化**（半饱和点/陡度/饱和值），冒烟截断网格须保留 α=1。",
    "- (a) 份额判据必须用**精确可加量（向量预算）**，效应份额不可加；(b) SMOKE 必做**且必看数字**；"
    "(c) 门裕度必须写进报告；(d) 预注册与修正案都落盘记 sha8，修正案须**运行前**冻结 + `why_not_a_HARKing_violation`；"
    "(e) 行为端与层内端同时测（`amp`）；(f)「充分」用放大臂、「必要」用撤除对偶臂；"
    "(g) 曲线判据必须**参数化**（半饱和点/陡度/饱和值）；(h) 冒烟产物落 `smoke/`；(i) 冒烟截断 α 网格必须保留定义点 α=1。"
))

R.append((
    "- (h) 跨位点分**绝对/相对双剂量坐标**，`x_star·r_ℓ`；(i) 判据符号须与物理方向一致；"
    "(j) `r_ℓ` 可差 3–17 倍，「打不动」先查方向旋转；(k) 读数端解释机制必做**读数位点探针**；"
    "(l) 固定基须报 `overlap(U_ℓ, U_ref)`；(m)「应由构造决定的位点」写成**硬断言**；"
    "(n) 端点量若构造饱和（α=1 ≡ 满干预）降级为正向判据、另立形状量作主量。",
    "- (j) 跨位点比较必须分**绝对/相对双剂量坐标**，半饱和点换算 `x*·r_ℓ`；(k) 判据符号须与物理方向一致；"
    "(l) 位点 `r_ℓ` 可差 3–17 倍，「打不动」先自检方向旋转（自基臂）；"
    "(m) 以读数端解释机制必做**读数位点探针**（最终 norm 之后）；(n) 深部位点固定基必须报 `overlap(U_ℓ, U_ref)`。"
))

R.append((
    "- (o) **同消息多条 Edit 会静默丢失** ⇒ 关键改动走 Python 补丁，逐处 `assert count==1` + 回读；"
    "(p) 非线性/极值型统计量判决必须**同时给误差带与零假设校准**；"
    "(q)「首次达比例」型判据归一化方向须与物理方向一致；(r) 集中度判据须 ≥2 独立坐标 + 报 `argmax` 窗口；"
    "(s)「不可排序」型结论须写明区间口径（边际 vs 配对）。",
    "- (o) **同一消息多条 Edit 会静默丢失** ⇒ 关键改动走 Python 补丁脚本，逐处 `assert count==1` + 回读；"
    "(p) 非线性 / 极值型统计量判决必须**同时给误差带与零假设校准**；(q) 探针族「应由构造决定的位点」写成**硬断言**；"
    "(r) 端点量若由构造饱和（α=1 ≡ 满干预）降级为独立正向判据、另立形状量作主量；"
    "(s)「首次达到比例」型判据的归一化方向须与物理方向一致（先断言符号）。"
))

R.append((
    "- **(t)「另一个独立口径」必须在 seal 前用 SMOKE 证明其与既有口径**可分**；重合即降级为阴性对照。**\n"
    "- **(u) 面板级恒等式与逐对恒等式的作用域必须分离**（子集臂 6 对 / 全集分母 24 对 ⇒ 伪偏差 1.147991）；"
    "写**逐对**形式 + `full_panel` 旁路。\n"
    "- **(v) 预注册预测的符号必须与自身 `rationale`/`falsified_if` 一致**"
    "（P14 的 P4 `desc` 与 rationale 反向，把已满足的预测机械判 FAIL）。\n"
    "- **(w) GQA 禁用 `hidden_size/n_heads` 反推 `head_dim`**（qwen3-4b `heads=32/kv=8/head_dim=128`，"
    "`o_proj.in=4096≠2560`）；配置字段直读 config 并与投影维度交叉断言。",
    "- **(t) 集中度 / 离散度型判据必须在 ≥2 个独立坐标上同时报告，并报 `argmax` 位置**"
    "（P13：`xhalf` 未决 / `J` 成立，两 argmax 相距 13）。\n"
    "- **(u)「不可排序 / 不可分辨」型结论必须写明区间口径（边际 vs 配对）**（P13：2/17 → 10/17）。\n"
    "- **(v)「另一个独立口径」必须在 seal 冻结前用 SMOKE 证明其与既有口径**可分**；重合即降级为阴性对照并另找剂量轴。**\n"
    "- **(w) 面板级恒等式与逐对恒等式的作用域必须分离**（子集臂 6 对 / 全集分母 24 对 ⇒ 伪偏差 1.147991）；"
    "写**逐对**形式 + `full_panel` 旁路。\n"
    "- **(x) 预注册预测的符号必须与其自身 `rationale` / `falsified_if` 一致**"
    "（P14 的 P4 `desc` 与 rationale 反向，把已满足的预测机械判 FAIL）。\n"
    "- **(y) GQA 禁用 `hidden_size / n_heads` 反推 `head_dim`**（qwen3-4b `heads=32/kv=8/head_dim=128`，"
    "`o_proj.in=4096≠2560`）；配置字段一律直读 config 并与投影维度交叉断言。"
))

b0 = open(P, 'rb').read()
t = b0.decode('utf-8')
L = ['=== fix_memory_letters_phase14 ===']
for i, (old, new) in enumerate(R, 1):
    c = t.count(old)
    L.append('  [%d] count=%d %s' % (i, c, 'OK' if c == 1 else '**FAIL**'))
    assert c == 1, 'R%d count=%d' % (i, c)
    t = t.replace(old, new)
open(P, 'wb').write(t.encode('utf-8'))
b1 = open(P, 'rb').read()

r = b1.decode('utf-8')
must = ['(j) 跨位点比较必须分', '(k) 判据符号须与物理方向一致', '(l) 位点 `r_ℓ` 可差 3–17 倍',
        '(m) 以读数端解释机制必做', '(n) 深部位点固定基必须报', '(o) **同一消息多条 Edit',
        '(p) 非线性 / 极值型统计量', '(q) 探针族「应由构造决定的位点」', '(r) 端点量若由构造饱和',
        '(s)「首次达到比例」型判据', '(t) 集中度 / 离散度型判据', '(u)「不可排序 / 不可分辨」型结论',
        '(v)「另一个独立口径」', '(w) 面板级恒等式与逐对恒等式', '(x) 预注册预测的符号必须与其自身',
        '(y) GQA 禁用']
bad = ['(h) 跨位点分', '(i) 判据符号须', '(j) `r_ℓ` 可差', '(k) 读数端解释机制',
       '(l) 固定基须报', '(m)「应由构造决定的位点」', '(n) 端点量若构造饱和',
       '(q)「首次达比例」型判据', '(r) 集中度判据须', '(s)「不可排序」型结论须',
       '(t)「另一个独立口径」', '(u) 面板级恒等式', '(v) 预注册预测的符号', '(w) GQA 禁用']
L.append('')
for a in must:
    L.append('  MUST   %-36s %d' % (a[:36], r.count(a)))
for a in bad:
    L.append('  BAD    %-36s %d (须 0)' % (a[:36], r.count(a)))
ok = all(r.count(a) >= 1 for a in must) and all(r.count(a) == 0 for a in bad)
L += ['', 'bytes %d -> %d ; sha8 %s -> %s' % (len(b0), len(b1), hashlib.sha256(b0).hexdigest()[:8], hashlib.sha256(b1).hexdigest()[:8]),
      'ALL OK' if ok else 'HAS FAIL']
io.open(OUT, 'w', encoding='utf-8').write('\n'.join(L) + '\n')
print('\n'.join(L))
assert ok
