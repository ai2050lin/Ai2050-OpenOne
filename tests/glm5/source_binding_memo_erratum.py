"""Append a formula-encoding correction without altering the frozen experiment."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import source_binding_design as D


BODY = r'''

**Phase 2906 公式排版勘误 [{stamp}]。** 上文部分 LaTeX 命令的反斜杠在报告文本写入时丢失。以下完整公式替代对应的损坏显示；实验算法、冻结源码、数据、门槛、数字与结论均不改变。原记录及原始 sources.zip 逐字节保留。本补充属于同一 Phase，不新增实验。

原架构 SwiGLU 计算式为

$$
A_{l,j}=\operatorname{SiLU}(G_{l,j})V_{l,j},\qquad
m_l=W_l^{\mathrm{down}}A_l.
$$

这里 \(l\) 是原模型块索引，\(j\) 是 MLP 中间单元索引，\(G,V\) 分别是该单元 gate/value 线性投影的输出，\(A\) 是实际 down_proj 的输入，\(m_l\) 是写入残差流的向量。

对每个已选头，QK 坐标选择算法为

$$
E_j=\frac{1}{N_{\mathrm{tok}}}
\sum_{i\in D}\sum_{t\in P_i}(q_{i,j}k_{i,t,j})^2,\qquad
J=\operatorname{Top32}_{j}(E_j).
$$

\(D\) 为发现样本集，\(P_i\) 为第 \(i\) 条输入的有效前缀位置集合，\(N_{\mathrm{tok}}=\sum_i|P_i|\)。因此该均值按有效 token 加权，不是先将各句等权平均。\(q,k\) 是实际 Q/K 归一化及 RoPE 后、对应查询头与 GQA 键头的向量；这里的 \(j\) 是头内坐标，不能与前式的 MLP 单元混称。

所选坐标的局部注意力与 VO 写入预测为

$$
\widehat{\alpha}_{i,t}=
\frac{\exp\!\left(\sum_{j\in J}q_{i,j}k_{i,t,j}/\sqrt{128}\right)}
{\sum_{u\in P_i}\exp\!\left(\sum_{j\in J}q_{i,j}k_{i,u,j}/\sqrt{128}\right)},
\qquad
\widehat d_i=W^O_h\sum_{t\in P_i}\widehat{\alpha}_{i,t}v_{i,t}.
$$

\(P_i\) 保持原因果与填充掩码；\(v_{i,t}\) 是同层原生 value，\(W^O_h\) 是该头对应的原生输出投影列块。该式使用原模型当前层 Q/K/V，既不是从文本独立预测这些量，也不是新语言定律。

MLP 的块评分与单元评分为

$$
s_{l,j}=\operatorname{Var}_{i\in D}(A_{i,l,j})
\left\|W^{\mathrm{down}}_{l,:,j}\right\|_2^2,\qquad
S_l=\sum_j s_{l,j}.
$$

按 \(S_l\) 选两块，再在各块按 \(s_{l,j}\) 选256单元构成集合 \(U_l\)。局部写入预测为

$$
\widehat m_{i,l}=\overline m_{D,l}
+\sum_{j\in U_l}W^{\mathrm{down}}_{l,:,j}
(A_{i,l,j}-\overline A_{D,l,j}).
$$

横线量只由发现侧估计，Down 列为原生参数。局部误差改善率的定义为

$$
R=1-\frac{\sum_{i\in H}\|y_i-\widehat y_i\|_2^2}
{\sum_{i\in H}\|y_i-\overline y_D\|_2^2}.
$$

\(H\) 为对应留出面板，\(y\) 是被预测头或 MLP 的完整输出坐标向量，\(\overline y_D\) 是该组件的发现侧均值。\(R\) 衡量相对此均值基线的平方误差降低，不是全模型解释率。

探索性递推模型的计算形式为

$$
\widehat h_{l+1}=\widehat h_l+
f_l(\widehat h_l,\mathcal K_{l,<t},\mathcal V_{l,<t}),
\qquad \widehat h_{27}=h_{27}.
$$

\(h_l\) 是末查询位置、块 \(l\) 之前的残差状态；\(f_l\) 是发现侧拟合的增量预测器，不同方法忽略或摘要前缀 K/V，均值方法仅使用发现侧平均增量。前缀 K/V 来自原模型预计算，严格排除末查询自己的 K/V；递推后续仅使用自身预测的末状态。该式是本轮受检验且未达到输出保真的拟合模型，不是已证闭合关系。
'''


def main():
    target = D.ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
    record = D.OUT / "memo-formula-erratum.json"
    assert not record.exists(), "Correction already recorded; inspect before recovery"
    before = target.read_bytes()
    closeout = D.read(D.OUT / "closeout.json")
    assert hashlib.sha256(before).hexdigest() == closeout["main_after_sha256"]
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    body = BODY.replace("{stamp}", stamp)
    assert body.count("$$") == 14
    assert not any(ord(c) < 32 and c not in "\n\t" for c in body)
    assert r"\operatorname" in body and r"\widehat" in body
    payload = body.encode("utf-8")
    D.check_write(target, len(payload) + (16 << 10), operation="Phase2906公式排版追加勘误")
    with target.open("ab") as stream:
        stream.write(payload)
    after = target.read_bytes()
    assert after == before + payload
    D.save("memo-formula-erratum.json", {
        "phase": 2906,
        "timestamp": stamp,
        "scope": "formula display only; no experiment or conclusion changes",
        "before_sha256": hashlib.sha256(before).hexdigest(),
        "after_sha256": hashlib.sha256(after).hexdigest(),
        "append_sha256": hashlib.sha256(payload).hexdigest(),
        "prefix_bytes_preserved": len(before),
        "source": str(Path(__file__).relative_to(D.ROOT)),
        "source_sha256": D.sha(Path(__file__)),
        "frozen_archive_sha256": D.sha(D.OUT / "sources.zip"),
        "line": before.count(b"\n") + 3,
    })
    print(json.dumps({"appended": True, "phase": 2906, "bytes": len(payload), "timestamp": stamp}))


if __name__ == "__main__":
    main()
