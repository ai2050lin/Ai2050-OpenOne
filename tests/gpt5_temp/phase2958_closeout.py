# -*- coding: utf-8 -*-
"""Phase 2958 closeout: seal + Ledger + MEMO + worklog + MEMORY
+ strategy section (Plan v2). Idempotent."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2958',
                   'imprint_dose_response')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-19.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2958_imprint_dose_response.py')

res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))
verdict = res['final_verdict']
created = json.load(open(os.path.join(OUT, 'execution.json'),
                         encoding='utf-8'))['created']


def sha8(path):
    return hashlib.sha256(
        open(path, 'rb').read()).hexdigest()[:8]


npz_name = 'imprint_dose_response.npz'
hashes = {
    'execution.json': sha8(os.path.join(OUT, 'execution.json')),
    'result.json': sha8(os.path.join(OUT, 'result.json')),
    npz_name: sha8(os.path.join(OUT, npz_name)),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2958_imprint_dose_response'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'forward_family_dose_response',
        'verdict': verdict,
        'source': ('phase2958/imprint_dose_response; patch '
                   'curve k*x_orig on ablated slices: '
                   'rebalancing is imprint-magnitude-driven '
                   'and strictly monotone (rho=1.0 both '
                   'families), no threshold switch; k=1 '
                   'closes bit-level to I0. Families split '
                   'on all three axes: L17 linear dose/'
                   'rotating profile/linear readout, L16 '
                   'nonlinear (tail-accelerated)/fixed '
                   'profile/nonlinear readout - consistent '
                   'with the 2946/2947 switch-type vs '
                   'gradient-type layer dichotomy'),
    })
    l14 = [lk for lk in led['linkage']
           if lk['link_id'] == 'L14_readout_spectrum_cross_model'][0]
    if meas_id not in l14['connects']:
        l14['connects'].append(meas_id)
led.pop('ledger_sha256_8', None)
new_h = hashlib.sha256(json.dumps(
    led, sort_keys=True, ensure_ascii=False)
    .encode('utf-8')).hexdigest()[:8]
led['ledger_sha256_8'] = new_h
json.dump(led, open(LEDGER, 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
chk = new_h

# ---------------- MEMO ----------------
memo = open(MEMO, encoding='utf-8').read()
if '## Phase 2958' not in memo:
    section = '''## Phase 2958: 印记剂量-响应判决 [@STAMP@]

**判决：`@VERDICT@`** —— 消融重平衡由**被动损失印记幅度驱动且严格单调**（双族 spearman(k, R)=1.000，无阈值开关），k=1 印记完全还原时 bit 级闭合回 I0；但两族在全部三轴分裂——L17（开关型层）：线性剂量响应（dev 0.079）/剖面随剂量旋转（cos 0.877）/读出线性（dev 0.066）；L16（渐变型层）：尾部加速的非线性（dev 0.196，k=0.75 处 −1.14 vs 预测 −4.40）/剖面固定（cos 0.968）/读出非线性（dev 0.121）——与 2946/2947 的开关型/渐变型层二分法自洽。

**设计（2956/2957 verbatim + patch 曲线，runtime @RT@s）**：把消融头的 o_proj 输入按 k·x_orig 部分还原（o_proj 线性 → 输出印记精确缩放 k·Δa），k∈{0,0.25,0.5,0.75,1.0}，K=2；R(k) = Σ_{l>dose} csep(m^k − m^I0)。锚 **a14 patch 闭合 bit 0**（k=1 fin == I0 fin；k=0 fin == I1 fin——切片级还原在 bf16 下精确）、**a19 非消融切片 I0/I1 bit 0**、a15 vs 2957 **bit 0**、a21 被动线性链 0.011（<1.0；o_proj 线性下 S_att[dose](k) = −(1−k)·D_abl 解析成立）。

**剂量-响应曲线（D1）**：
| 族 | R(k=0) | R(0.25) | R(0.5) | R(0.75) | R(1) | T1 | T2 | T3 |
|---|---|---|---|---|---|---|---|---|
| L17 | −15.16 | −10.21 | −7.60 | −4.98 | 0 | linear（dev 0.079） | rotating（0.877） | linear（0.066） |
| L16 | −17.60 | −11.05 | −5.35 | −1.14 | 0 | nonlinear（dev 0.196） | fixed（0.968） | nonlinear（0.121） |

**关键发现**：
1. **重平衡是印记幅度的函数，不是头身份的函数**：R(k) 沿印记比例单调回归到零，无滞回/阈值/符号翻转——2950"竞争重平衡"的操作本质 = 对印记幅度的（近）线性放大器，L17 族放大近理想线性，L16 族在中段超线性、尾部加速衰减。
2. **剖面旋转 vs 固定 = 层类型签名**：开关型层 L17 的响应剖面随剂量旋转（不同剂量征用不同下游层组合），渐变型层 L16 剖面方向锁定（同一组下游层按比例伸缩）——层类型（2944/2945/2947 链）决定重平衡的"几何模式"。
3. **读出曲线分解**：sep(k) = 被动线性 + 活性级联；L17 活性部分近线性（该层上游效应被 sigmoid 陡区饱和主导），L16 活性部分显著非线性——为 2946 分配干扰提供剂量维度解释。
4. **锚体系新件**：a14 bit 级闭合锚证明"部分还原"干预的精确性；a21 解析被动链（|S_att[dose](k)+(1−k)D_abl|<1.0 实测 0.011）。

**结论（重复 3 次）**：**消融竞争重平衡由被动损失印记幅度驱动、严格单调、无阈值；层类型决定其几何模式（开关型=线性放大+剖面旋转，渐变型=非线性放大+剖面锁定）；机制链的"重平衡"分支至此完整：载体（MLP 主导 2956）→ 特异性（消融特异 2957）→ 驱动律（印记剂量 2958）。**

**硬伤与勘误（2 轮）**：run1 a15 参考错（k=0 捕获对比 I1 而非 I0，差恒 0；观测 9.37=D_abl 恰为该错误签名）+ save 未初始化（anchor_fail 路径 UnboundLocalError）。run2 权威。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / npz @HNPZ@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2958/imprint_dose_response/`。Ledger 97 条 / L14 connects 65 / ledger @LEDHASH@。

**接续**：机制链第二十一环（剂量-响应环）闭合，重平衡分支完整。候选 2959：A（主选）进入研究方案 v2 阶段二首战——词类机制签名矩阵（57 词按具体名词/抽象概念/功能词分组，测路由增益 A11(s) 头分布 × 读出 SVD 坐标 × 承重带剖面的组间/组内差异，置换 null 校准）；B 交叉项代数结构（2955 遗留）；C 承重带跨模型复现（glm4）；D 剖面旋转定位（L17 剖面旋转的逐层分解，2958 遗留）。
'''
    stamp = created.replace('T', ' ')[:16]
    reps = [('@STAMP@', stamp), ('@VERDICT@', verdict),
            ('@RT@', str(res['runtime_s'])),
            ('@LEDHASH@', chk),
            ('@HEXE@', hashes['execution.json']),
            ('@HRES@', hashes['result.json']),
            ('@HNPZ@', hashes[npz_name]),
            ('@HSCR@', hashes['script'])]
    for k, v in reps:
        section = section.replace(k, v)
    memo += section

# ---------------- strategy section ----------------
if '## 研究方案 v2' not in memo:
    stamp2 = created.replace('T', ' ')[:16]
    strategy = '''

---

## 研究方案 v2：从单线机制链到词类机制图谱（思路一整合）[@STAMP2@]

**思路一评估**（用户提出：W_U 反嵌入+自回归给每个 token 独立特征指纹 → 每条脉络机制不同，少数参数高值/大量参数低值、浅层/深层各异 → 画图谱看语言模式如何映射到 LLM 机制、如何少参数高效实现复杂能力）：

- **正确且有本线路实证支撑**：(1) 杠杆异质性——2944 浓度、2947 头级集中（h22/h19/h0 高杠杆）、2949-2951 组重平衡；(2) 深度分层——2932/2935 L6-L12 承重带 vs L28+ 反向带、2940 层剖面双极、2956/2958 重平衡深层增强；(3) 少参数高效——2938 低秩子空间（top-8 能量 91.9%）、2939 三方向 S_IDX 张成注入轴、2948 W_ov 线性秩序；(4) 图谱路线——atlas_ledger/L14 即其雏形。
- **需修正**：(1) "每 token 单独齿轮"过强——2938 证明词身份是**共享子空间内重编码**（0/57 词离开子空间、重编码方向上下文无关固定 2939 δc cos 0.997），token 特异性在**坐标与路由增益**（2952 A11 路由跳变、2953 sigmoid），不在电路拓扑；2949 组水平反转证明"每头重要"≠"头组可移除"，一一对应指纹-齿轮图会在组水平失效；(2) "抽象 vs 具体不同机制"必须在**路由/增益层**而非读出方向层寻找（2940 方向词属性盲）；(3) 判据必须可证伪——词类×机制签名差异需置换 null 校准（纪律 11），词表需 n≳40（纪律 8）。

**三阶段方案**：
- **阶段一（2959-2962）机制原语完型**：B 交叉项代数（2955）；D 剖面旋转定位（2958）；把 21 环机制链压缩为"原语卡片"——每卡 = (层带, 模块, 头集, 读出方向, 剂量-响应律, lin_r 稳健性)。
- **阶段二（2963-2975）词类机制签名矩阵（思路一主战场）**：57 词扩至 n≳40 按具体名词/抽象概念/功能词分组；每类测三件套：路由增益头分布（2953 法）、读出 SVD 基坐标（2939 法）、承重带剖面（2932 法）；判决=组间签名差异 > 组内（置换 null 校准）；产出词类×机制签名矩阵 → 图谱第一版。
- **阶段三（2976+）跨模型图谱对齐**：glm4 复现承重带/开关头/词类签名；图谱跨模型保守性 = 机制原语可迁移性检验；汇入 L14 跨模型谱系。

'''
    strategy = strategy.replace('@STAMP2@', stamp2)
    memo += strategy
    open(MEMO, 'w', encoding='utf-8').write(memo)

# ---------------- worklog ----------------
wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2958' not in wl:
    wl += ('- Phase 2958 闭环：mixed_dose_response_mixed_profile_'
           'mixed_readout。patch 曲线（k·x_orig 部分还原）证明重平衡'
           '由印记幅度驱动、双族严格单调（rho=1.0）、无阈值开关，k=1 '
           'bit 级闭合回 I0；层类型决定几何模式：L17 开关型=线性放大'
           '+剖面旋转+读出线性，L16 渐变型=非线性放大+剖面锁定+读出'
           '非线性。重平衡分支完整：载体(2956)→特异性(2957)→驱动律'
           '(2958)。勘误：a15 参考错（9.37=D_abl 签名）+ save 未初始'
           '化。另：MEMO 追加研究方案 v2（思路一整合，三阶段）。'
           'Ledger 97 / hash ' + chk + '。\n')
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
if '当前 max=**2957**' in mem:
    mem = mem.replace(
        '当前 max=**2957**，下一个 **2958**（候选 A 消融特异性来源定位）',
        '当前 max=**2958**，下一个 **2959**（方案 v2 阶段一首战）')
if '驱动律 印记剂量单调(2958)' not in mem:
    old = '总量巧合非剖面恒等(2957)。'
    if old in mem:
        mem = mem.replace(old, old[:-1] + '→驱动律：印记剂量单调无阈值，层类型定几何模式(2958)。')
if '## 机制链状态（20 环）' in mem:
    mem = mem.replace('## 机制链状态（20 环）',
                      '## 机制链状态（21 环）')
if '连续 28 次前向锚定' in mem:
    mem = mem.replace('连续 28 次前向锚定',
                      '连续 29 次前向锚定')
add = ('- 部分还原干预规范（2958）：剂量型干预用"切片级部分还原"'
       '（消融头 o_proj 输入 := k·x_orig）而非输出端加法——o_proj '
       '线性保证输出印记精确缩放且 k=1 bit 级闭合（a14 锚标准件）；'
       '解析被动链 S_att[dose](k)=−(1−k)·D_abl 可作免费锚；参考对'
       '比必须先写明"vs 哪个条件"（a15 9.37=D_abl 签名教训）；'
       'anchor_fail 路径所有输出容器（save/curves 等）必须预先初始'
       '化。研究方案 v2 已入 MEMO（三阶段：原语完型→词类机制签名'
       '矩阵→跨模型对齐）。\n')
if '部分还原干预规范（2958）' not in mem:
    anchor_line = '- 跨相位 bit 级锚规范（2957）：'
    i = mem.find(anchor_line)
    if i >= 0:
        mem = mem[:i] + add + mem[i:]
    else:
        mem = mem.rstrip() + '\n' + add
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
