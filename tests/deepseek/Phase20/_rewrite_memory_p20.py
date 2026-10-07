# -*- coding: utf-8 -*-
"""Phase 20 收尾：MEMORY.md 深度精简 + 回填（合入 §5 P20 三条教训 / §7 死线 / §8 技能计数）。

目标：把跨轮索引压到最小可检索形态（细节回查 MEMO / 技能），LF-only、无 BOM。
"""

P = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\MEMORY.md'

CONTENT = r'''# RDC/LPF 项目纪律（工作区长期记忆）
> 权威记录：`research\deepseek\docs\AGI_DEEPSEEK_MEMO.md`（N 线）、`research\gpt5\docs\AGI_GPT5_MEMO.md`（G 线）。本文件仅跨轮索引，细节一律回查 MEMO / 技能。

## 0 约定与落点
- deepseek 线只写 deepseek 备忘录（append-only、UTF-8+**BOM**+**CRLF**、`bare_lf 0`）。
- 产物 v2：脚本→`tests\deepseek\Phase{N}\`；报告/seal/`verify_*`/memo·wlog 节源码→`tests\deepseek_temp\Phase{N}\`；非 Phase→`_review\`/`_infra\`。**禁堆根下**。
- 收尾链（**十三次 P8–P20**）：探针→seal→amend→exec→SMOKE→正式→判决→Ledger→MEMO→基线→wlog→MEMORY→技能→独立复核→present。
- 标题 `## Phase {N}: 短标题（线-P）[hh:mm]`；纠错 append 不回改（**唯一例外：交付前生成件缺陷⇒逐字节回滚基线重跑链**）。
- Ledger n=**303**（P8–P20 各 1；**N 线 P3–P7 待补**）。基线 `post-append-phase20`=**488,772 B/4,674 行/20 标题**（`sha8 68eddd46`；旧 `post-append-phase19` 标 `stale`+`drift_events`；`sections` 键=完整标题行）。

## 1 装置铁律（26 条全文在 skill `rdc-main-axis-probe` 坑集）
最易违反：(a) 份额只用**精确可加向量预算**。(b) SMOKE 必做**必看数字**。(o) 同消息多 Edit 静默丢⇒Python 补丁+`assert count==1`+回读。(ad) 实现与 seal **逐字一致**。(ae) 数字**一律 result 现场渲染**。(ac) 插值读数不设单一硬门⇒分层。(af) **冻结锚重验**、如实报「陈旧」；基线带 `stale`/`drift_events`；**字典键不得截断生成**。
其余：patch 剔末层；`h_ℓ=hidden_states[ℓ+1]`；写入窗=相邻最大增量；`jump/max|effect|≥0.5`；剔实例 token 同剔类别 token；绝对/相对双剂量 `x*·r_ℓ`；固定基报 `overlap`；GQA 禁 `hidden/n_heads`；换口径=新自由度⇒校准臂；cos 与 rel-L2 双报；平均秩；paired `margin` 负=候选优。

## 2 N 线主线（P4→P20）
- **P4–P7**：主轴三段（嵌入=词典/层=开关/**权重绑定定 is-a 落点**）；单头 share_max **3.0%**；读位槽=**G−1=5 维**充分必要（随机 170–6000×）；**跨族近正交⇒无通用类别算子**；R1 K_d 降级。
- **P8–P11**：写入端**分布式**（向量预算 MLP 0.472/单头 0.074）；L6 内**无阈值增益**、非线性在其**之后**（S 形 `x*`≈0.6）⇒ **栈=软门**；深端塌陷主因**方向失配**。
- **P12–P16**：集中度 `rho(xhalf,depth)=−0.783`；P14 第三口径不存在；P15 nf4 `NF4_FAITHFUL` 但**无跨模型稳健判据**；**P16 预算否证（A2 反号 −0.716）⇒ P12/13/14 物理深度表述整体撤回**；P16 主域 `REACH={ℓ:ρ≥0.10}`、旧量→`com_layer`/`span_k`。
- **P17**：逐层 `w_ℓ`+质心 `com_V`（**区间求和**）**26.15/26.70/26.68** ≫ median(REACH) 17/14/14；`MLP_DOMINANT` **0.740/0.975/0.824**；`spearman(w,J)` **−0.55/−0.79/−0.60**。
- **P18**：行为预算 `b`；`share_mlp_beh` **0.66/0.96/0.75**；`com_B` **21.1/20.1/24.2**；`spearman(w,|b|)` **正**（与 P17 **反号⇒P17 P6 对象错配**）；`P2`/`P6` FAIL。
- **P19**：向量侧跨精度 **5/5 PASS**（Δ`com_V` **0.093/0.070**、`spearman(w)` **0.9924/0.9988**）⇒「深端集中/MLP 主导」非 nf4 地板效应；A2（29.5GB）bf16 **segfault@19%**。
- **P20**：行为量+写入窗剖面跨精度 **9 预测 8 PASS**（Δ`com_B` **0.056/0.103**、ρ(b) **0.9732/0.9844**、Δ`com_layer(x)` **0.006/0.183**）⇒ **P18 行为侧+P16 剖面侧结论都不是 nf4 地板效应**。**P9 FAIL=域歧义**（as-coded `max|Δxhalf|` **0.2645/0.0338** vs 容差 0.05；容差只在 **ℓ≥6** 标定 ⇒ 冻结 REACH 域 **0.006171/0.003713 皆 PASS**）；**不改判**，另报 post-hoc 域分解。勘误 `E-comv`/`E-xhdom`/`E-rho`。

## 3 挂账与限界
- R1/R2：10 条成立；P1 纠错 1（K_d）+降级 4+挂账 5。
- **限界**：① 否定臂基线不成立；② 留一只覆盖「未见实例」，「未见类别」仍崩（水果 0.04/0.05）；③ **激活级干预 ≠ 权重级证明**；④ 跨深度固定基衰减须扣基旋转；⑤ P13 确认集只 3 对；⑥ P14/P15 集中度 n≤18 无区分力（同报 null95）；⑦ P15 A1/A2 家族×规模混杂；⑧ P17 `com_V`≠`com_layer`、`w_ℓ` 只臂内可比；⑨ P16 exec `bootstrap.seeds` 错记；⑩ P18 `b` 不可加、nb 只 2 位点；⑪ P19 只覆盖向量侧、含 offload 第三源；⑫ **P20** 只两模型+offload、**不**升格 P16 `com_layer`（`P6` FAIL）、P9 是**域歧义**非物理不稳定。
- **外部方案裁决**：三图谱骨架/统一实验卡/三集划分/竞争性解释/「预测未见现象」→ 全采纳；「每族一组特征+拼图还原+统一阈值+通用算子」→ 全改写。

## 4 G 线（gpt5，3150–3154）
3151 k3_only；3152 k1_not_triggered（k* 定律）；3153 fingerprint_consistent_coverage_partial；3105–3150：真值=记录级一阶矩广播、写入头组 L20–28 主写/L32 擦除；**判决符号=rev-3151b（负=优）**。**3154 已预注册**。

## 5 本机缺陷（Windows）
- bash shim 劣化：`ls`/`rm`/`tail` 坏、内联 `python -c` stdout 常丢、**反引号被吃** ⇒ Python 文件化+写 `.txt` 再 Read；`python s.py>log` 产 UTF-16 ⇒ 用 `run_<phase>.py` 自写 UTF-8。
- **`Edit` 幻影**+IME 吞汉字 ⇒ 大段中文走 Python 补丁；调用 `cd <root> && .venv/Scripts/python.exe tests/...`。关键写入后 Grep 复核；GPU 逐模型防 OOM。
- **P16–P20 教训**：详见 skill 教训 26–33。要点：P16 改渲染器必重跑 `do_append_*`；P17 **MERGE 不重算臂内量⇒改口径须重跑臂**；P18 **判据域须与标定域一致**、**复核不得写死「预期成功」**；P19 **换口径=新自由度⇒校准臂**、**配对集定义也是口径**；P20 **E-comv 空检查**（冻结中间量按构造与臂无关⇒不可作证据，须另报随臂重算版）、**探针/生产三对齐面=键空间·网格·配对**、**落盘名须与 seal 声明一致**、**E-baseline 陈旧检测**、**E-xhdom 判据须写明域**、**E-rho 取值器按类型分层**、**交付前自查⇒生成件缺陷可回滚重跑**。

## 6 工作方式
「好的，继续」= AI 主导不停、深度自主续研；结构化输出；**关键发现重复 3 次**。

## 7 下一步（死线优先级）
- **Phase 21 = 把跨精度推进到组件级向量预算与权重实现级**（bf16 下复算 P8 `share_v` 与 N2h1-α-1 权重级定位）。
- **并列**=邻域 ±2 敏感性；**P9 判据补域**（ℓ≥6 标定域下可否改判 ⇒ 需新 Phase 预注册）；**第三**=P17 `P6` 的 MEMO 改判。
- **其他挂账**：N2h1-α-1 权重级；N2h1-β 水果类崩塌；N3-β→ε；R1 补强；K4；**P3–P7 补登 Ledger**。**G 线**：3154 已预注册。

## 8 技能
`rdc-main-axis-probe`（15 臂+**61 坑**）、`rdc-phase-closeout`（**33 教训**/十三次链）、`rdc-dual-arm-phase-template`。
'''


def main():
    import hashlib
    old = open(P, 'rb').read()
    t = CONTENT
    assert '\r' not in t
    out = t.encode('utf-8')
    assert not out.startswith(b'\xef\xbb\xbf')
    for key in ['## 8 技能', '33 教训', 'Phase 21 = 把跨精度推进到组件级',
                'E-rho 取值器按类型分层', '交付前自查⇒生成件缺陷可回滚重跑']:
        assert key in t, 'missing %s' % key
    open(P, 'wb').write(out)
    chk = open(P, 'rb').read()
    ct = chk.decode('utf-8')
    print('bytes %d -> %d (%+d)' % (len(old), len(chk), len(chk) - len(old)))
    print('chars', len(ct))
    print('crlf', ct.count('\r\n'), 'cr', ct.count('\r'), 'bom', chk[:3] == b'\xef\xbb\xbf')
    print('sha8', hashlib.sha256(chk).hexdigest()[:8])
    print('DONE')


if __name__ == '__main__':
    main()
