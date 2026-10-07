"""Document the five-stage explanation; no model execution or new experiment."""
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
OUT = ROOT / "tests/glm5/result/rdc_pipeline_explainer_20260923"
PHASE = 2756

BODY = r"""
### 1. 状态、范围与读法

**已完成教学解释及既有证据核对；未运行模型、未新增GPU实验。** 本文回答“外部关系与角色→共享参数下的条件化响应→跨层传播与交互→全词表读出→自回归接续”的原理、计算过程和证据边界。

五项是分析视角，不是模型中五个互不重叠的层区。条件化响应和跨层传播在每个Transformer块中重复；读出和自回归接续在每个生成步重复。外部语言图是研究者的描述，不能预先当成模型内部已经存在的符号图。

三种证据等级：A为架构/实现可确定的计算；B为本项目在限定材料中的实测；C为待验证的语义机制或推广假设。A并不自动证明我们已经理解计算所实现的语言规则；B不因小模型存在幅值误差就失效，也不因此自动升级为普适定律。

### 2. 贯穿的教学示例

输入：

> 甲盒在乙盒内，乙盒在丙盒内。这里的“在……内”指严格包含关系，而且可以传递。请判断“甲盒不在丙盒内”是否正确，先回答“正确”或“错误”，再用一句话解释。

逻辑上的目标答案：

> 错误。甲盒在乙盒内，乙盒在丙盒内，所以甲盒在丙盒内。

**这是新构造的中文教学例子，以上为逻辑目标，不是任何模型的实测输出。** 没有为它报告token ID、分词边界、概率、激活值或具体执行推理的层/头/神经元。Phase2754包含相关的严格包含材料，但实际为英文判断协议，不能当作已经运行了此中文例子。

### 3. 第一阶段：外部关系与角色

人的任务描述包含对象、关系类型、关系方向、查询中的角色、否定和输出要求。外部图写为甲→乙→丙，箭头仅表示“严格位于……内部”。每条边的源是被包含者，目标是容器；同一个乙在第一条边是容器，在第二条边是被包含者，因此角色是关系中的位置，不是这个词永远固定的身份。

本例已明确可传递，所以从两条事实得到inside(甲,丙)=真；查询的是其否定，故查询命题为假，应回答“错误”。这是外部逻辑基准。若将关系换为“喜欢”，同样两条箭头不能直接传递；编码研究必须区分关系类型，不能只看图形相同。

模型实际收到的首先是文本的分词ID序列；聊天协议还可能添加模板和特殊token。token不是词或概念的一一对应单位。固定权重推理时同一ID的嵌入表行通常固定：
$$
h_i^{(0)}=E[\mathrm{id}_i].
$$
E是嵌入表，i为实际输入位置。角色与语境的影响还要通过之后的计算体现。

**A：** 输入ID、位置、模板及嵌入查表可直接核对。**B：** Phase2754通过明确的源/目标角色注释、同世界变换及独立世界划分，发现角色位置上的可解码关系信号。**C：** 模型是否形成与外部图同构的内部图、是否把“包含者/被包含者”绑定到稳定参数结构、能否自动找出角色，都未由该结果证明。

### 4. 第二阶段：共享参数下的条件化响应

直观上是同一套计算规则处理不同输入。固定权重不意味着固定输出：矩阵乘法依赖输入，softmax、归一化和MLP门控又依赖当前状态。同一层参数跨token位置与生成步复用；不同层及不同头不必共享同一组权重。常规推理不需要现场修改训练好的参数。

以一个attention头示意，对当前层归一化后的状态n_i，先由投影得到query、key和value。Qwen3实际还对Q/K做头内RMSNorm及RoPE位置处理。令q_i、k_j表示已做这些处理的向量，v_j表示value，则：
$$
\alpha_{ij}=\frac{\exp(q_i^\top k_j/\sqrt{d_k})}
{\sum_{s\leq i}\exp(q_i^\top k_s/\sqrt{d_k})},\quad j\leq i,
\qquad
a_i=W_O\left(\sum_{j\leq i}\alpha_{ij}v_j\right).
$$
公式省略额外mask细节，并用单头输出投影块示意；多头实际拼接后投影。因果mask禁止读未来位置。Qwen的RoPE作用于Q/K，不是把整个残差场统一旋转。

q与k共同决定当前如何聚合历史位置；v及输出投影决定写入什么。可以将其类比为“从哪些位置取材料、怎样写回”，但q不是已经标注为“要找包含关系”的语言指令，attention权重也不等于完整因果重要性。

在本例中，文本从“甲盒在丙盒内”变为“甲盒不在丙盒内”，输入和后续状态都会变化，许多头的聚合权重和写入可能随之改变。没有证据指定某个头专门读取“不”，也不能从一个大attention权重直接断言它完成了否定。

MLP在每个位置应用相同的非线性变换。对Qwen3的SwiGLU形式：
$$
M(x)=W_d[\operatorname{SiLU}(W_gx)\odot(W_ux)].
$$
两组投影产生的数值逐元素相乘，再投影回残差空间；SiLU不是一个事先标注了“真/假”的开关，也不被限制为0到1。由于输入已含上下文，按位置计算的MLP同样可以参与语境依赖计算。

**A：** 上述计算、参数复用及输入依赖可从实现核查。固定attention权重时V路径线性；完整attention随输入改变路由，因此不能称整体纯线性。**B：** GPT3072在固定路由的条件下支持V路径的线性重构；GLM2754保存了实际attention/MLP写入及选定MLP层的全部单元贡献。**C：** 哪些参数连接实现“角色绑定、传递、否定”，以及同一组参数怎样在这些操作间复用，尚未完整确定；“现场共振点名”不是已确认物理机制。

### 5. 第三阶段：跨层传播与交互

每层是在已有状态上继续计算和更新。对一整组位置状态H_l，Qwen3的主干形式为：
$$
R_l=H_l+A_l(N_l(H_l)),\qquad
H_{l+1}=R_l+M_l(N'_l(R_l)).
$$
N和N'为归一化，A包含跨位置attention，M为位置内MLP。残差相加保留了一条恒等路径，但不保证语义永远不受破坏；状态里可能有增强、抵消及重新编码。

本例需要信息能够在后面的位置共同起作用：两个事实、当前查询的端点、否定和输出要求。在因果模型里，较早位置的状态不能读后面尚不可见的词；末尾位置可以读取完整前缀的历史位置。因而不能把它解释成“早期甲盒位置被后面所有文字即时改写”。多个层的计算会让后面位置获得依赖更复杂前缀的信息，但传递关系究竟在哪些路径上被算出，仍需测量。

可以在同一读出位置上用四种条件检查交互，例如因素A为查询端点交换、因素B为肯定/否定：
$$
I=F_{11}-F_{10}-F_{01}+F_{00}.
$$
F可以是对齐的隐藏状态或指定输出分数，下标表示两个因素是否切换。I不为零表示组合效应不能用两种单独效应直接相加解释。要控制词身份、位置和长度等影响；非零交互本身不证明模型计算了逻辑规则。

**B中最值得保留的证据：** GLM2754在4B的192个独立确认世界、3072条提示上，用训练后冻结的外部解码器，从第16层末位及查询两角色位置读出基础关系。新实体、新表述、深度3三个轴的基础谓词符号均正确。这支持“当前简单有向链材料存在可复用的角色关系信号”。解码器使用实际第16层状态和外部角色注释，不能冒充从文字直接预测未来内部状态的模型内部算法。

加入否定后的整句真值解码分别为93.36%、73.14%、91.70%；事实倒序另使该真值解码从93.75%降至68.16%。基础关系信号的符号较稳，幅度及条件组合仍有差异。对当前示例而言，这是“基础包含关系与否定合成应分开研究”的依据，而不是已经证明本例怎样被算出。

关键反证：删除一个探针可读方向后，探针绝对分数均值从0.528降至0.00924，但模型的平均真值响应系数几乎不变。可能存在冗余、重建或外部探针读了模型不用的方向，当前尚不能区分。故可读信息≠已定位必要因果电路；干预失败也不抹掉该受限域中的可读规律。

全层原生来源形状相似度约0.97–0.98，但相对幅值误差约0.31–0.52，是可以继续追踪的近似稳定现象；它是事后分析，还不是未见组合的全链预测成功。

### 6. 第四阶段：全词表读出

经过最后一个block后，末位置状态仍是数值向量。以当前Qwen实现为例，最终RMSNorm和无偏置输出矩阵计算：
$$
\tilde h=\frac{\gamma\odot h}{\sqrt{\frac1d\sum_{j=1}^{d}h_j^2+\epsilon}},
\qquad z_v=u_v^\top\tilde h,\qquad
p_v=\frac{\exp z_v}{\sum_{w\in\mathcal V}\exp z_w}.
$$
h是归一化前末位向量；gamma为学习到的坐标增益；u_v是输出矩阵中token v对应行；z为logit；V为词表。分母先约束向量整体尺度，gamma再逐坐标重加权。它不自动去相关或令数据协方差等于单位阵，所以不能叫已证实的白化管道，更不能解释成直接压低常用词、放大高级逻辑词。

整个词表获得下一token分数，解码器再按greedy或采样等规则选token。温度、top-p等属于额外解码配置；它们会影响实际选择。概率是模型对续写的分配，不是命题客观为真的校准概率。

本例理想上应进入“错误”的回答序列，但“正确/错误”可能各对应多个token，必须实际分词；不能虚构它们是一对单token。只比较两个候选序列的教师强制分数，也不等于模型会自然生成其中之一，它还可能先输出格式或其他词。

对于同一个前缀下任意两个token a、b，有精确关系：
$$
\log(p_a/p_b)=z_a-z_b=(u_a-u_b)^\top\tilde h.
$$
二者的相对优势由一个差方向投影决定；其他token仍影响它们各自的绝对概率及全局胜者。这个代数恒等式解释了为什么整体状态很相似，关键答案仍可能变化：关键差方向上的微小变化可以翻转排序。它不是“所有语言能力已被一个差向量解释”的定理。

**A：** 全词表矩阵投影、softmax、同前缀token的log-odds差恒等式，以及实际数值精度均可核对。**B：** GLM2754对真实残差、各块写入、MLP参数列及最终读出做了含数值误差的对账；GPT3057反对“gamma实现严格白化”的旧说法。**C：** 一个关系信号如何必然控制正确答案、风格与内容为何在某些条件竞争、哪个环节导致单条错误，还需要路径和行为联合验证。

### 7. 第五阶段：自回归接续

一个token选出后加入前缀，模型再以“原输入+已生成内容”为条件预测下一token。整个序列的概率可分解为：
$$
P(y_{1:T}\mid x)=\prod_{t=1}^{T}P(y_t\mid x,y_{<t}).
$$
它不是一次读出完整答案。每个新token仍经过各层计算；启用KV cache时，各层复用历史位置的K/V，避免把整个旧前缀重新算一遍。cache和位置是计算状态的一部分，末位一个HiddenState不足以替代它们。

如果模型逐步生成了“错误。”，接下来预测解释时会同时受问题及自己刚写的答案影响；如果先写成“正确。”，后文也可能受错误答案影响，但并不保证永远无法纠正。无需预设吸引子或流形坍缩，就已经存在可观察的反馈与误差累积问题。

生成的一段通顺解释不一定忠实揭示了生成第一答案时的内部计算。模型可能首答正确但解释错误，也可能格式不符但内容正确，还可能继续不停生成。应分别检查首个判断、第一次分叉、完整解释及EOS/外部长度停止；教师强制候选评分与自由生成分别报告。

**A：** 追加token、因果条件分解、KV更新和EOS/解码配置可核查。**B：** GLM2754只执行了有界短续写：4B最多8个新token、14B最多4个；有格式诊断价值，不是完整长生成验证。**C：** 当前中文例子的完整输出、跨步关系信号的维护、长程约束、错误形成与恢复机制尚待专门采集；不能声称已闭合自回归机制。

### 8. 从流程到编码机制：当前基础与阶段性任务

稳定规律的正确目标不要求每个激活值一致或小模型所有题都答对。优先检查实体替换、角色交换、关系方向、表述顺序和组合深度变化时，哪些响应方向、变换关系及计算来源可重现，再定位幅度和输出差异。规模、训练、模板、精度与任务能力分开控制，不能把全部失败归因于小模型。

当前最可信的研究接口是“有范围的角色关系可读信号→条件合成差异→真实计算来源及词表读出”。尚未完成的是从外部类型化条件，提前预测未见组合怎样改变内部作用，并证明模型实际使用了对应路径，而非外部探针读出或事后投影。

下一大阶段可围绕同一问题组织，以下均为计划而非已执行：

1. 扩展同一套冻结关系材料，加入断边、分叉、无关事实、同端点不同路径及不可传递关系，排除只识别端点/模板的捷径。
2. 在自然前向全坐标和原生参数路径中追踪角色绑定与否定合成，允许冗余、协同和重建；不把单方向删除失败当成唯一裁决。
3. 预先明确预测可用输入，在独立组合及完整自由生成上评估关系信号、输出选择和停止；避免用目标层状态或未来量作输入后声称“事先预测”。
4. 资源允许时顺序比较本地4B/14B/GLM4，以可重复方向和失效范围为主，同时报告数值与行为误差，不预设更大即完美。

本轮没有新数学定理或RDC公式改进。沿用Phase2754的候选研究接口：
$$
\mathcal T^D_{\ell,\tau}:
(\mathcal L(p),\mathbf W_\ell(p),\mathcal X_\ell(p))
\rightharpoonup\mathbf W_{\ell+1}(p),\qquad
\operatorname{Atlas}_D=(G_{\mathrm{external}},G_{\mathrm{internal}},
E^D_{\mathrm{association}}).
$$
其中L是类型化语言条件，W专指类型化析因响应束（不是权重矩阵），X为指定边界观测场，D限定模型与输入域，tau标识条件/角色类型；偏映射符号表示只在已覆盖范围定义。该式是待填充、验证的提取接口，不是已求出的通用转移定律。具体计算仍以上述Transformer前向方程为底座。

三图谱在本轮的新增内容是说明接口和证据状态；没有新增实验边。外部图记录对象/类型/角色/否定，内部图记录位置/层/模块/坐标/参数，关联图分别标记统计可读、可推广预测及因果证据；不能把这三种边混为“已破解”。

### 9. 可追溯来源与交付

- 实现：.venv/Lib/site-packages/transformers/models/qwen3/modeling_qwen3.py，RMSNorm第50–65行，MLP第70–83行，attention第222–291行，decoder第294–335行，输入/cache/最终norm第389–438行，lm_head第451行及logits第505行。实际文件身份见evidence_manifest.json；行号仅作导航。
- 实测来源：tests/glm5/result/rdc_relation_stability_20260923/phase_report.md（GLM2754）；不把历史指标当作本轮重跑。
- 历史主张审查：tests/glm5/result/rdc_framework_audit_20260923/phase_report.md及claim_ledger.json（GLM2755，包含GPT3072、3057等来源定位）。
- 一手背景：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)、[Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467)。具体Qwen差异以本地实现为准。
- 本轮交付：phase_report.md、evidence_manifest.json、append_receipt.json。脚本位于tests/glm5_temp/phase2756_explainer_delivery.py；仅追加GLM统一记录，未改写GPT及任何历史Phase，未清理数据。
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    before = MEMO.read_bytes()
    phases = re.findall(r"^## Phase (\d+):", before.decode("utf-8"), re.M)
    assert phases and int(phases[-1]) == PHASE - 1, "Reconcile concurrent memo changes first."
    assert not OUT.exists(), "Do not replace an existing delivery."
    now = datetime.now().astimezone()
    report = (
        f"## Phase {PHASE}: 五阶段语言计算流程、贯穿示例与证据边界 "
        f"[{now:%Y-%m-%d %H:%M}]\n\n"
        + BODY.strip() + "\n"
    )
    assert not any(ord(c) < 32 and c not in "\n\r\t" for c in report)
    sources = [
        ".venv/Lib/site-packages/transformers/models/qwen3/modeling_qwen3.py",
        "tests/glm5/result/rdc_relation_stability_20260923/phase_report.md",
        "tests/glm5/result/rdc_framework_audit_20260923/phase_report.md",
        "tests/glm5/result/rdc_framework_audit_20260923/claim_ledger.json",
    ]
    manifest = {
        "phase": PHASE, "created_local": now.isoformat(), "new_model_runs": 0,
        "example_status": "constructed teaching example; logical target, not a recorded model output",
        "sources": [
            {"path": p, "sha256": sha((ROOT / p).read_bytes())} for p in sources
        ],
        "primary_background": [
            "https://arxiv.org/abs/1706.03762",
            "https://arxiv.org/abs/1910.07467",
        ],
        "script_sha256": sha(Path(__file__).read_bytes()),
    }
    OUT.mkdir(parents=True)
    (OUT / "phase_report.md").write_text(report, encoding="utf-8")
    (OUT / "evidence_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    assert MEMO.read_bytes() == before, "Memo changed while preparing delivery."
    prefix = "\n" if before.endswith(b"\n") else "\n\n"
    addition = (prefix + report).encode("utf-8")
    with MEMO.open("ab") as stream:
        stream.write(addition)
    after = MEMO.read_bytes()
    assert after == before + addition, "Append verification failed."
    receipt = {
        "phase": PHASE, "created_local": now.isoformat(), "memo": str(MEMO),
        "memo_line": before.count(b"\n") + prefix.count("\n") + 1,
        "before_sha256": sha(before), "after_sha256": sha(after),
        "append_sha256": sha(addition), "append_only_verified": True,
        "report_sha256": sha(report.encode("utf-8")),
    }
    (OUT / "append_receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
