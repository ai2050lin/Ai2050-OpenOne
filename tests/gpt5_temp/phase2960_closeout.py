# -*- coding: utf-8 -*-
"""Phase 2960 closeout: seal + Ledger + MEMO + worklog + MEMORY.
Idempotent, placeholder replacement."""
import hashlib
import json
import os

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2960',
                   'profile_rotation_geometry')
LEDGER = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
WLOG = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\2026-09-20.md')
WMEM = (r'C:\Users\Admin\WorkBuddy\2026-09-17-01-30-05'
        r'\.workbuddy\memory\MEMORY.md')
SCRIPT = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
          r'\phase2960_profile_rotation_geometry.py')

res = json.load(open(os.path.join(OUT, 'result.json'),
                     encoding='utf-8'))
verdict = res['final_verdict']
created = json.load(open(os.path.join(OUT, 'execution.json'),
                         encoding='utf-8'))['created']


def sha8(path):
    return hashlib.sha256(
        open(path, 'rb').read()).hexdigest()[:8]


npz_name = 'profile_rotation_geometry.npz'
hashes = {
    'execution.json': sha8(os.path.join(OUT, 'execution.json')),
    'result.json': sha8(os.path.join(OUT, 'result.json')),
    npz_name: sha8(os.path.join(OUT, npz_name)),
    'script': sha8(SCRIPT),
}

# ---------------- Ledger ----------------
led = json.load(open(LEDGER, encoding='utf-8'))
meas_id = 'meas2960_profile_rotation_geometry'
if not any(m['meas_id'] == meas_id
           for m in led['measurements']):
    led['measurements'].append({
        'meas_id': meas_id,
        'type': 'forward_family_profile_geometry',
        'verdict': verdict,
        'source': ('phase2960/profile_rotation_geometry; '
                   'the 2958 S_mlp(k) profile families '
                   'decompose as fixed component dominant '
                   '(97.5 pct L17 / 98.8 pct L16) plus a '
                   'rank-1 deviation (top-1 energy 0.897 / '
                   '0.965) with a curved trajectory in k '
                   '(linear-fit rel resid 0.341 / 0.194); '
                   'rotation axes concentrate at the dose-'
                   'layer flank (L14-18) while deep layers '
                   'stay fixed - the 2958 L17/L16 profile '
                   'split is a deviation-magnitude '
                   'difference, not a mechanism difference'),
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
if '## Phase 2960' not in memo:
    section = '''## Phase 2960: 剖面旋转几何——固定分量+秩1偏差统一分解 [@STAMP@]

**判决：`@VERDICT@`** —— 2958 的"剖面旋转"签名完成几何分解：两族的下游 S_mlp(k) 剖面族都是**固定分量主导（‖均值‖/‖剖面‖ = 97.5% L17 / 98.8% L16）+ 秩 1 偏差（top-1 偏差能量 0.897 / 0.965）+ k 轨迹弯曲（逐层线性拟合相对残差 0.341 / 0.194，超 0.15 阈）**。**2958 的 L17/L16 profile 轴分裂（cos 0.9 线两侧）是偏差幅度差异，不是机制差异——两族共用同一几何模板：固定剖面 + 单旋转轴 + 非线性幅度**。

**设计（2958 verbatim 协议，runtime @RT@s）**：B0 + {I0,I1}×{L17,L16} K=3 + patch 曲线 k∈{0,0.25,0.5,0.75,1.0} K=2（切片级部分还原 k·x_orig），下游 S_mlp(k) ∈ R^18/19 做均值+偏差 SVD。判据冻结：T1 top-1 偏差能量≥0.8→rank1_rotation；T2 线性拟合残差≤0.15→trajectory_linear；T3 固定份额≥0.5→fixed_dominant。

**锚（run3 权威）**：a1 dirs 重建 2.17e-08（**第 31 次连续前向锚定**）、a3/a8/a13/a14/a15/a18/a19 **bit 0**、a16 bf16 界归一 0.979、a17 恒等链 rel 4.19e-03、a21 被动链 0.011、**a22 vs 2958 npz**：R/S_mlp05 4.98e-04（3dp 舍入门 1e-3）、sep 4.66e-03（2dp 舍入门 5.01e-3）。

**主检验**：
| 检验 | 冻结阈 | L17 | L16 | 判定 |
|---|---|---|---|---|
| T1 旋转秩 | top-1 能量≥0.8 | **0.897** | **0.965** | 双 rank1_rotation |
| T2 轨迹形状 | rel≤0.15 | **0.341** | **0.194** | 双 trajectory_curved |
| T3 固定分量 | ≥0.5 | **0.975** | **0.988** | 双 fixed_dominant |

**关键发现**：
1. **偏差奇异值谱高度秩 1**：L17 σ=[5.07, 1.66, 0.39, 0.13]（σ2/σ1=0.33）、L16 σ=[8.38, 1.52, 0.47, 0.10]（σ2/σ1=0.18）——旋转由**单一轴**支配，第二轴仅 1/3 与 1/5。
2. **旋转轴空间定位在剂量层侧翼**：|Vt[0]| 权重 top 层 L17={17,14,15,2,16}、L16={18,17,15,16,0}——旋转集中在 L14-18（剂量层±侧翼），深层（L30-35）几乎不动（mean_profile 深层值主导：L35 = −2.21/−5.13）。与 2957"两端相近、中层分化"自洽。
3. **轨迹弯曲与 2958 剂量非线性同源**：L16 rel_lin 0.194 与其尾部加速 R(k)（k=0.75 处 −1.14 vs 线性 −4.40）对应；L17 0.341 更弯但仍秩 1——弯曲的是幅度律，不是方向。
4. 原语卡片推进：剖面族可参数化为（固定剖面 m̄，旋转轴 v_rot，幅度律 R(k) 三点标定）三元组——图谱签名维度从 18 层压缩到 2+3 参数。

**结论（重复 3 次）**：**消融 MLP 剖面族 = 固定分量主导（~98%）+ 秩 1 旋转 + 弯曲幅度律；2958 的层类型 profile 分裂是偏差幅度差异而非机制差异；旋转轴集中于剂量层侧翼（L14-18），深层不动。**

**硬伤与勘误（run1→run3）**：run1 a22 用 bit 阈 1e-6 不可达——2958 npz 存的是**舍入值**（sep_k round 2dp、R/S_mlp05 round 3dp），4.66e-03 恰落 2dp 舍入界（2948 教训跨相位复现）；run2 a22 拆门通过（R/S05 4.98e-04 / sep 4.66e-03）但 T2 的 lstsq 维度写反（b=Sfam.T (n_l,5) vs A (5,2)，正确调用 lstsq(A, Sfam)）崩溃；run3 权威一次通过。**新教训入 MEMORY**：跨相位复现锚必须先检上游产物的存储精度（round 位数），bit 级仅限"上游存全精度"或同文件链。

**文件+SHA256-8**：execution @HEXE@ / result @HRES@ / npz @HNPZ@ / script @HSCR@。产物 `tests/glm5/result/rdc_query_construction_20260913/phase2960/profile_rotation_geometry/`。Ledger 99 条 / L14 connects 67 / ledger @LEDHASH@。

**接续**：机制链第二十三环（旋转几何环）闭合，方案 v2 阶段一接近收官。候选 2961：A（主选）原语卡片压缩——把 23 环机制链压缩为结构化"原语卡片"表（层带×模块×头集×读出方向×剂量律×lin_r 稳健性，纯文档 Phase，阶段一收官）；B 词类机制签名矩阵预研（阶段二启动：词表扩容 n≳40 分组设计预注册）；C 旋转轴功能身份（v_rot 与剂量层路由头 W_ov/u35 的代数关系，一次前向）；D v3 解码器方向重启（2940 遗留）。
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
    open(MEMO, 'w', encoding='utf-8').write(memo)

# ---------------- worklog ----------------
entry = ('- Phase 2960 闭环：rank1_rotation_trajectory_curved_'
         'fixed_dominant。2958 的剖面旋转完成几何分解：两族 S_mlp(k) '
         '剖面族 = 固定分量主导（97.5%/98.8%）+ 秩 1 偏差（top-1 能量 '
         '0.897/0.965）+ k 轨迹弯曲（rel 0.341/0.194）——2958 的 L17/L16 '
         'profile 分裂是偏差幅度差异非机制差异；旋转轴集中剂量层侧翼 '
         '（L14-18），深层不动。勘误：run1 a22 bit 阈不可达（2958 npz '
         '2dp/3dp 舍入存储）→ 按存储精度拆门；run2 lstsq 维度崩 → run3 '
         '权威。16 锚（a1 第 31 次连续前向锚定）。Ledger 99 / hash '
         + chk + '。\n')
wl = ''
if os.path.exists(WLOG):
    wl = open(WLOG, encoding='utf-8').read()
if 'Phase 2960' not in wl:
    wl += entry
    open(WLOG, 'w', encoding='utf-8').write(wl)

# ---------------- MEMORY ----------------
mem = open(WMEM, encoding='utf-8').read()
mem = mem.replace(
    '当前 max=**2959**，下一个 **2960**（候选 A 剖面旋转定位）',
    '当前 max=**2960**，下一个 **2961**（候选 A 原语卡片压缩）')
old = '→交叉项代数：方向锁定幅度饱和(2959)。'
if old in mem:
    mem = mem.replace(old, old[:-1]
                      + '→剖面旋转几何：固定主导+秩1偏差，'
                        '2958 分裂统一(2960)。')
if '## 机制链状态（22 环）' in mem:
    mem = mem.replace('## 机制链状态（22 环）',
                      '## 机制链状态（23 环）')
if '连续 30 次前向锚定' in mem:
    mem = mem.replace('连续 30 次前向锚定',
                      '连续 31 次前向锚定')
add = ('- 跨相位存储精度锚规范（2960）：跨相位复现锚必须先检上游产物的'
       '存储精度——若上游 npz/json 存的是 round 值（如 sep 2dp、剖面 3dp），'
       '复现门按存储精度设（2dp→5.01e-3、3dp→1e-3），bit 级仅限"上游存'
       '全精度"或同文件链（2948 教训推广）。科学结论：剖面族"旋转"统一'
       '分解 = 固定分量主导（~98%）+ 秩 1 偏差 + 弯曲幅度律——层类型'
       'cos 分裂（0.9 线）是偏差幅度差异非机制差异；剖面签名可压缩为'
       '（固定剖面， 旋转轴， 幅度三点标定）三元组，旋转轴集中剂量层'
       '侧翼 L14-18。\n')
if '跨相位存储精度锚规范（2960）' not in mem:
    anchor_line = '- 判决记账规范（2959）：'
    i = mem.find(anchor_line)
    if i >= 0:
        mem = mem[:i] + add + mem[i:]
    else:
        mem = mem.rstrip() + '\n' + add
open(WMEM, 'w', encoding='utf-8').write(mem)

print('closeout done: ledger=%s n_meas=%s' % (
    chk, len(led['measurements'])))
