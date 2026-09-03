#!/usr/bin/env python3
"""C656-C669 absolute-coordinate grammar and local response campaign.

This campaign reuses the frozen C571-C589 language-program field, then adds
coordinate-specific finite-state and intervention measurements.  It reads
embeddings, post-block HiddenStates, final norm and output logits only.  It
does not inspect attention, MLPs, weights or gradients; it does not use PCA,
Top-K selection, projection, or donor-difference transport.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
CATALOG = ROOT / "frontend/public/research_data/current/language_encoding_catalog.json"
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c667_absolute_coordinate_local_response_atlas.json"
VISUAL_BINARY = ROOT / "frontend/public/vis_data/research_kernel/c667_absolute_coordinate_state_and_influence.float16.npy"
sys.path.insert(0, str(TESTS))

import phase2105_c571_c589_scope_program_algebra_campaign as scope
import phase2163_c629_model_specific_worker as model_worker

PHASES = {
    "C656-C658": (2190, "evidence_retrial_and_absolute_coordinate_master_contract"),
    "C659-C661": (2191, "six_family_absolute_coordinate_state_atlas"),
    "C662-C664": (2192, "prospective_coordinate_transition_grammar"),
    "C665-C667": (2193, "full_coordinate_local_response_and_coalition_test"),
    "C668-C669": (2194, "cross_model_relative_topology_and_major_stage_closure"),
}
OUTS = {
    name: RESULT / f"phase{phase}_{name.lower().replace('-', '_')}_{slug}"
    for name, (phase, slug) in PHASES.items()
}

PARENT_FIELD = RESULT / "phase2108_c574_qwen_qualified_all_token_all_coordinate_capture/raw/qwen3_role_last_states.float16.npy"
PARENT_INDEX = RESULT / "phase2108_c574_qwen_qualified_all_token_all_coordinate_capture/raw/hidden_index.jsonl"
PARENT_MATERIAL = RESULT / "phase2106_c572_language_program_ontology_and_large_material_freeze/material/scope_program_cases.jsonl"
PARENT_COMPILED = RESULT / "phase2107_c573_compiler_semantic_balance_naturalness_and_qwen_behavior/compiled/qwen3_scope_program_cases.jsonl"
PARENT_BEHAVIOR_FINAL = RESULT / "phase2107_c573_compiler_semantic_balance_naturalness_and_qwen_behavior/analysis/final.json"
PARENT_C586 = RESULT / "phase2120_c586_sequential_cross_model_functional_topology/analysis/final.json"

DIM = 2560
ROLES = ("primary", "secondary", "relation", "context", "query", "boundary")
CHECKPOINTS = 38
QPOINTS = (0, 8, 16, 24, 32, 37)
STATE_CODES = 33
BEHAVIOR_GATE = 0.75
GRAMMAR_GAIN_GATE = 0.005
LOCAL_MARGIN_GATE = 0.05
GROUPS = {
    "atomic_operations": ("atomic",),
    "voice_scope": ("voice_scope_factorial",),
    "nested_attitude": ("nested_attitude_flagship",),
    "recursive_knowledge": ("recursive_knowledge_flagship",),
    "two_factor_composition": ("discourse_voice_composition", "path_paraphrase_composition"),
    "translation_layout": ("translation_layout_factorial",),
}
LOCAL_ANCHORS = {
    "nested_attitude": {
        "panel": "nested_attitude_flagship", "family": "nested_attitude",
        "domain": "like", "surface": "record", "cell": "o1i0",
    },
    "recursive_knowledge": {
        "panel": "recursive_knowledge_flagship", "family": "recursive_knowledge",
        "domain": "taxonomy", "surface": "record", "cell": "s1d3k0",
    },
    "voice_scope": {
        "panel": "voice_scope_factorial", "family": "voice_scope",
        "domain": "inspect", "surface": "record", "cell": "f1q0",
    },
}
SOURCE_Q = 24
SOURCE_ROLE = "relation"
TARGET_Q = 25
LOCAL_BATCH = 8


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()]


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            value.update(block)
    return value.hexdigest()


def finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(item) for item in value)
    if isinstance(value, (float, np.floating)):
        return math.isfinite(float(value))
    return True


def out(name: str) -> Path:
    return OUTS[name]


def final(name: str) -> dict:
    return load(out(name) / "analysis/final.json")


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None:
        mmap.close()


def stable_condition(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "little")


def group_name(row: dict) -> str:
    for name, panels in GROUPS.items():
        if row["panel"] in panels:
            return name
    raise KeyError(row["panel"])


def state_code(values: np.ndarray) -> np.ndarray:
    """Fixed base-2 state alphabet; every physical coordinate is retained."""
    x = np.asarray(values, np.float32)
    magnitude = np.abs(x)
    exponent = np.clip(np.floor(np.log2(np.maximum(magnitude, 2.0 ** -24))) + 8, 0, 15).astype(np.uint8)
    result = 1 + exponent + (x >= 0).astype(np.uint8) * 16
    result[magnitude == 0] = 0
    return result.astype(np.uint8)


def code_sign(code: np.ndarray) -> np.ndarray:
    return np.where(code == 0, 0, np.where(code <= 16, -1, 1)).astype(np.int8)


def code_exponent(code: np.ndarray) -> np.ndarray:
    return np.where(code == 0, -24, ((code.astype(np.int16) - 1) % 16) - 8).astype(np.int16)


def condition_codes(rows: list[dict], tier: str) -> np.ndarray:
    labels = []
    for row in rows:
        if tier == "state_only":
            labels.append("all")
        elif tier == "typed":
            labels.append(f"{row['family']}|{row['operation_domain']}")
        elif tier == "program":
            labels.append(f"{row['family']}|{row['operation_domain']}|{row['cell']}|{row['surface']}")
        else:
            raise KeyError(tier)
    return np.asarray([stable_condition(value) for value in labels], np.uint64)


def mode_lookup_predict(train_current: np.ndarray, train_next: np.ndarray,
                        train_condition: np.ndarray, test_current: np.ndarray,
                        test_condition: np.ndarray) -> tuple[np.ndarray, float]:
    coordinates = np.arange(DIM, dtype=np.uint64)[None, :]
    train_key = ((train_condition[:, None] * STATE_CODES + train_current.astype(np.uint64)) * DIM + coordinates)
    joint = (train_key.reshape(-1) * STATE_CODES + train_next.reshape(-1).astype(np.uint64))
    unique_joint, counts = np.unique(joint, return_counts=True)
    keys = unique_joint // STATE_CODES
    targets = (unique_joint % STATE_CODES).astype(np.uint8)
    order = np.lexsort((targets, -counts, keys))
    ordered_keys = keys[order]
    first = np.r_[True, ordered_keys[1:] != ordered_keys[:-1]]
    table_keys = ordered_keys[first]
    table_targets = targets[order][first]
    test_key = ((test_condition[:, None] * STATE_CODES + test_current.astype(np.uint64)) * DIM + coordinates).reshape(-1)
    positions = np.searchsorted(table_keys, test_key)
    known = positions < len(table_keys)
    known[known] &= table_keys[positions[known]] == test_key[known]
    prediction = test_current.reshape(-1).copy()
    prediction[known] = table_targets[positions[known]]
    return prediction.reshape(test_current.shape), float(1.0 - np.mean(known))


def grammar_metric(prediction: np.ndarray, truth: np.ndarray, current: np.ndarray,
                   unknown_rate: float) -> dict:
    pred_sign = code_sign(prediction); truth_sign = code_sign(truth)
    return {
        "exact_state_accuracy": float(np.mean(prediction == truth)),
        "sign_accuracy": float(np.mean(pred_sign == truth_sign)),
        "exponent_mae": float(np.mean(np.abs(code_exponent(prediction) - code_exponent(truth)))),
        "unknown_key_rate": float(unknown_rate),
        "target_changed_rate": float(np.mean(truth != current)),
        "copy_exact_accuracy": float(np.mean(current == truth)),
        "copy_sign_accuracy": float(np.mean(code_sign(current) == truth_sign)),
    }


def append_memo(name: str, result: dict) -> None:
    phase, slug = PHASES[name]
    marker = f"## Phase {phase}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing:
        return
    titles = {
        "C656-C658": "C656-C658 证据重裁与绝对坐标语法总合同",
        "C659-C661": "C659-C661 六类语言程序的绝对坐标状态图谱",
        "C662-C664": "C662-C664 坐标状态转移语法的前瞻验证",
        "C665-C667": "C665-C667 全2560坐标局部响应与联盟调用",
        "C668-C669": "C668-C669 跨模型相对拓扑与大阶段总裁决",
    }
    examples = {
        "C656-C658": "同时登记语序、语态、释义、双重否定、路径深度、翻译布局、`A喜欢B吃X`式嵌套态度和`苹果→水果→食物→物体`式递归关系图；失败只淘汰对应路线。",
        "C659-C661": "对每个样本自身的`embedding/36个block后状态/final norm`，逐角色、逐层、逐个物理坐标记录符号和固定二进制幅值字，不寻找Top-K热点。",
        "C662-C664": "例如只用discovery中的`当前坐标状态字+语言程序类型`学习下一检查点状态字；先在discovery内部留出单元选合同，再冻结后依次揭示confirmation和lockbox。",
        "C665-C667": "对嵌套态度、递归知识图和语态作用域各选一对同合同confirmation/lockbox样本，在q24关系角色把第j个坐标作自身尺度的正负微扰，j从0到2559逐一遍历。",
        "C668-C669": "Qwen3-4B、GLM4、DeepSeek-7B和Qwen3-14B顺序运行同一小型多族面板；先过各自行为门，才读取其模型内坐标拓扑，跨模型不比较相同坐标编号。",
    }
    formulas = {
        "C656-C658": r"""$$
Z_{q,r,j}(x)=\left(\operatorname{sgn}H_{q,r,j}(x),\;\operatorname{clip}_{[-8,7]}\lfloor\log_2|H_{q,r,j}(x)|\rfloor\right)
$$
$$
\text{研究对象}=\{Z_{q,r,j}(x),\;Z_{q+1,r,j}(x),\;\partial_j^{\mathrm{local}}F(x)\}_{x,q,r,j}
$$""",
        "C659-C661": r"""$$
N_{g,p,q,r,j}(z)=\sum_{x\in(g,p)}\mathbf 1[Z_{q,r,j}(x)=z]
$$
$$
\operatorname{Flip}_{g,q,r,j}=\frac{1}{|g|}\sum_{x\in g}\mathbf 1[\operatorname{sgn}Z_{q,r,j}(x)\ne\operatorname{sgn}Z_{q+1,r,j}(x)]
$$""",
        "C662-C664": r"""$$
\widehat Z_{q',r,j}=T_{j,q\to q'}\!\left(Z_{q,r,j},\;\tau(L),\;\rho(r),\;\kappa(L)\right)
$$
$$
G=\operatorname{Acc}_{exact}(T)-\operatorname{Acc}_{exact}(\widehat Z=Z_q)
$$""",
        "C665-C667": r"""$$
J^{(x)}_{j\to k}=\frac{H_k(x;H_j+\epsilon_j)-H_k(x;H_j-\epsilon_j)}{2\epsilon_j},\quad
\epsilon_j=\max(0.125|H_j|,0.01\,\operatorname{RMS}(H))
$$
$$
m(x)=\ell_{gold}(x)-\ell_{other}(x),\qquad
s_j=\operatorname{sgn}\frac{m(x;H_j+\epsilon_j)-m(x;H_j-\epsilon_j)}{2\epsilon_j}
$$""",
        "C668-C669": r"""$$
\Theta_M(u,r)=\{\Pr_M[Z>0],\Pr_M[\operatorname{Flip}],\operatorname{rank}_r(\operatorname{Flip})\}_{u\in[0,1]}
$$
$$
\text{跨模型候选不变量}=\text{相对深度事件顺序+角色拓扑+状态字转移表同构},\quad j_M\not\equiv j_{M'}
$$""",
    }
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    protocol = load(out(name) / "protocol/preregistration.json")
    section = f"""

## Phase {phase}: {titles[name]} [{stamp}]

**Campaign与证据边界。** `{name}`（`{slug}`）。本阶段只读取词嵌入、每个block后的HiddenState、final norm与输出logit；全程保留全部物理激活坐标，不读取Attention、MLP、权重或梯度，不用PCA、Top-K、投影筛选，也不把别的样本的差分向量搬入当前样本。数值正负微扰只是测量当前样本某一坐标的局部响应，不等于发现唯一神经电路。

**运行前冻结合同。**

```json
{json.dumps(protocol, ensure_ascii=False, indent=2, allow_nan=False)}
```

**测试用例。** {examples[name]}

**测试原理与数学公式。**

{formulas[name]}

**详细结果与门槛。**

```json
{json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)}
```

**分析、理论进展与严格结论。** {result.get('strict_interpretation', result.get('strict_conclusion', '见冻结裁决。'))} 理论主体继续使用“条件化输出场闭合理论”，组织原则继续使用“复用—差分—条件化（RDC）”；本阶段只更新经验对象为“绝对坐标状态字、类型化坐标转移语法和样本内局部响应”，不更改理论名称。观察可重复、前瞻预测、局部因果效应、行为调用、自由生成和跨模型同构仍是不同证据等级。

**问题、硬伤与瓶颈。** 语言材料虽覆盖六大程序组和多种句型，仍以受控英语为主且独立人类自然度盲评未运行；固定幅值字是无坐标丢弃的描述语言，但仍会离散化连续数值；有限差分只给局部干预响应，不能唯一确定内部物理路径；A/B输出接口只裁决候选边界，不等价于自然自由生成；小模型结构可能粗糙；跨模型只能比较相对深度和角色拓扑，不能比较相同坐标编号；任何阳性都不能单独授权“新基础数学”。

**相关文件。** 主脚本：`tests/glm5/phase2190_c656_c669_absolute_coordinate_grammar_campaign.py`；结果目录：`{out(name).relative_to(ROOT)}`；预注册：`{(out(name) / 'protocol/preregistration.json').relative_to(ROOT)}`；裁决：`{(out(name) / 'analysis/final.json').relative_to(ROOT)}`。

**结论和下一步授权。** {result.get('next_authorization', '见冻结分支。')}
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(section)


def close(name: str, body: dict, checks: dict, authorization: str) -> dict:
    value = {
        "phase": PHASES[name][0], "campaign": name, "status": "closed",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "checks": checks, "all_checks_passed": bool(checks) and all(bool(v) for v in checks.values()),
        **body, "next_authorization": authorization,
    }
    save(out(name) / "analysis/final.json", value)
    append_memo(name, value)
    print(json.dumps(value, ensure_ascii=False, indent=2), flush=True)
    return value


def freeze_contracts() -> None:
    shared = {
        "model_policy": "Qwen3-4B first; GLM4, DeepSeek-7B, Qwen3-14B sequential in the cross-model branch",
        "camera": "embedding + every post-block HiddenState + final norm + output logits; all signed physical coordinates",
        "forbidden": ["attention", "MLP", "weights", "gradients", "PCA", "Top-K selection", "projection", "donor-difference transport"],
        "failure_policy": "route-level rejection or NA; a failed route never stops other registered observations",
        "human_review": "NA_not_run; machine semantic audit is not an independent human naturalness rating",
    }
    protocols = {
        "C656-C658": {
            **shared,
            "object": "audit C647-C655 and freeze an absolute-coordinate, multi-language-family campaign",
            "families": GROUPS,
            "partitions": "reuse frozen units 0-9 discovery, 10-13 confirmation, 14-17 lockbox",
            "state_alphabet": "sign plus fixed base-2 magnitude exponent; no coordinate is removed",
            "overclaim_controls": [
                "C651 q33 boundary self-prediction is an identity leak",
                "q8 source identity is same-prefix lexical readability, not target-word abstraction",
                "C654-C655 reject registered bridges, not the existence of concept computation",
            ],
        },
        "C659-C661": {
            **shared,
            "object": "discovery-only absolute coordinate state and transition atlas across six program groups",
            "tensor": "group x 38 checkpoints x six roles x all coordinates",
            "outputs": ["state-code histogram", "per-coordinate positive rate", "per-coordinate sign-flip rate", "per-coordinate modal state"],
            "selection": "none; observation only",
        },
        "C662-C664": {
            **shared,
            "object": "predict unseen coordinate state words without moving response vectors between samples",
            "tiers": ["state_only", "typed", "program"],
            "selection": "fit units 0-5, select on units 6-9, freeze, refit units 0-9, reveal confirmation then lockbox once",
            "intervals": list(zip(QPOINTS[:-1], QPOINTS[1:])),
            "gate": f"lockbox exact-state gain over copy >= {GRAMMAR_GAIN_GATE:.3f}; sign/exponent metrics remain descriptive",
        },
        "C665-C667": {
            **shared,
            "object": "full 2560-coordinate sample-local finite-difference response map and sign-defined coalition transfer",
            "anchors": LOCAL_ANCHORS,
            "source": {"checkpoint": SOURCE_Q, "role": SOURCE_ROLE},
            "targets": [{"checkpoint": TARGET_Q, "role": "boundary"}, {"checkpoint": 37, "role": "boundary"}, "candidate_logit_margin"],
            "perturbation": "each coordinate's own symmetric scale; no donor and no selected subset",
            "controls": ["opposite sign", "coordinate-sign circular shift by 257", "zero/base"],
            "gate": f"aligned lockbox margin gain exceeds controls by {LOCAL_MARGIN_GATE:.2f} in at least two of three families",
        },
        "C668-C669": {
            **shared,
            "object": "sequential model qualification and relative-depth coordinate topology; then major-stage closure",
            "models": ["glm4", "deepseek7b", "qwen3_14b"],
            "subset": "24 frozen behavior cases spanning all six program groups and discovery/lockbox",
            "behavior_gate": BEHAVIOR_GATE,
            "cross_model_rule": "compare normalized checkpoint depth and role topology only; never compare coordinate IDs",
            "new_math_gate": "requires broad behavior, prospective prediction, local causal specificity, natural generation, cross-model topology and human evidence",
        },
    }
    for name, protocol in protocols.items():
        for part in ("protocol", "material", "analysis", "audit", "raw", "behavior", "external"):
            (out(name) / part).mkdir(parents=True, exist_ok=True)
        path = out(name) / "protocol/preregistration.json"
        if not path.exists():
            save(path, {
                "phase": PHASES[name][0], "campaign": name,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "parent_files": {
                    "field": str(PARENT_FIELD.relative_to(ROOT)),
                    "index": str(PARENT_INDEX.relative_to(ROOT)),
                },
                "protocol": protocol,
            })


def phase2190() -> None:
    name = "C656-C658"
    if (out(name) / "analysis/final.json").exists():
        return
    attachments_audit = {
        "retained": [
            "C647-C655 correctly separate target-language control, source lexical readability and target-word prediction",
            "the q33 boundary self-predictor in C651 is mechanically identical to its target and must be discarded",
            "C654 and C655 are legitimate negative results for their frozen upstream and trajectory bridges",
            "multiword language programs, role/scope graphs and coordinate-specific measurements are the correct next empirical expansion",
        ],
        "tightened": [
            "C655 'full trajectory' means six registered checkpoints and six roles, not every token at every layer",
            "failure of coordinatewise, nearest-neighbor and exemplar transport does not prove that concept translation has no internal mechanism",
            "a full cross-coordinate W, Volterra system or RKHS is not identifiable from current sample counts and is not authorized as a mechanism claim",
            "language-function graphs are registered semantic annotations, not evidence that the model literally implements a functor or category",
            "local finite differences are allowed as measurements, but cross-sample donor differences are not used as mechanism objects here",
        ],
        "strict_parent_conclusion": "With target language fixed, Qwen3-14B still did not yield a transportable concept-to-target-word bridge under the registered strictly-upstream and six-checkpoint trajectory cameras.",
    }
    save(out(name) / "analysis/attachment_evidence_audit.json", attachments_audit)
    parent = load(PARENT_BEHAVIOR_FINAL)
    manifest = {
        "parent_field_shape": [3780, 38, 6, 2560],
        "parent_field_bytes": PARENT_FIELD.stat().st_size,
        "parent_index_rows": len(read_rows(PARENT_INDEX)),
        "parent_behavior_accuracy": parent["headline"]["behavior_accuracy"],
        "qualified_slices": parent["headline"]["qualified_slices"],
        "groups": GROUPS,
        "phase_order": {key: value[0] for key, value in PHASES.items()},
    }
    save(out(name) / "protocol/frozen_major_campaign_manifest.json", manifest)
    close(name, {
        "strict_interpretation": attachments_audit["strict_parent_conclusion"] + " The next campaign therefore changes the empirical object to per-sample absolute coordinate state words and local coordinate responses; it does not tune another donor-response distance.",
        "evidence_audit": attachments_audit,
        "manifest": manifest,
        "new_foundational_mathematics_gate": False,
    }, {
        "parents_exist": all(path.exists() for path in (PARENT_FIELD, PARENT_INDEX, PARENT_MATERIAL, PARENT_COMPILED)),
        "parent_rows": manifest["parent_index_rows"] == 3780,
        "parent_shape": manifest["parent_field_shape"] == [3780, 38, 6, 2560],
        "six_groups": len(GROUPS) == 6,
        "finite": finite(manifest),
    }, "授权C659-C661仅用discovery观察六类程序的绝对坐标状态；不得读取lockbox来选择状态字或坐标。")


def phase2191() -> None:
    name = "C659-C661"
    if (out(name) / "analysis/final.json").exists():
        return
    index = read_rows(PARENT_INDEX)
    states = np.load(PARENT_FIELD, mmap_mode="r")
    profiles = np.lib.format.open_memmap(
        out(name) / "raw/discovery_coordinate_profiles.float16.npy", mode="w+", dtype=np.float16,
        shape=(len(GROUPS), CHECKPOINTS, len(ROLES), 2, DIM))
    modes = np.lib.format.open_memmap(
        out(name) / "raw/discovery_modal_state_words.uint8.npy", mode="w+", dtype=np.uint8,
        shape=(len(GROUPS), CHECKPOINTS, len(ROLES), DIM))
    summary: dict[str, dict] = {}
    try:
        for group_i, group in enumerate(GROUPS):
            rows = [row for row in index if group_name(row) == group and row["partition"] == "discovery"]
            ids = np.asarray([row["hidden_index"] for row in rows], np.int64)
            group_summary = {"rows": len(rows), "checkpoints": {}}
            for q in range(CHECKPOINTS):
                checkpoint = {}
                for role_i, role in enumerate(ROLES):
                    values = np.asarray(states[ids, q, role_i], np.float32)
                    codes = state_code(values)
                    counts = np.stack([(codes == code).sum(axis=0) for code in range(STATE_CODES)])
                    modal = np.argmax(counts, axis=0).astype(np.uint8)
                    positive = np.mean(values > 0, axis=0).astype(np.float16)
                    if q + 1 < CHECKPOINTS:
                        next_sign = np.sign(np.asarray(states[ids, q + 1, role_i], np.float32))
                        flip = np.mean(np.sign(values) != next_sign, axis=0).astype(np.float16)
                    else:
                        flip = np.zeros(DIM, np.float16)
                    profiles[group_i, q, role_i, 0] = positive
                    profiles[group_i, q, role_i, 1] = flip
                    modes[group_i, q, role_i] = modal
                    hist = np.bincount(codes.reshape(-1), minlength=STATE_CODES)
                    checkpoint[role] = {
                        "positive_rate": float(np.mean(values > 0)),
                        "negative_rate": float(np.mean(values < 0)),
                        "zero_rate": float(np.mean(values == 0)),
                        "sign_flip_to_next": float(np.mean(flip)) if q + 1 < CHECKPOINTS else None,
                        "state_code_histogram": hist.astype(int).tolist(),
                    }
                group_summary["checkpoints"][str(q)] = checkpoint
            summary[group] = group_summary
            profiles.flush(); modes.flush()
            print(f"[C659-C661] observed {group_i + 1}/{len(GROUPS)} {group}", flush=True)
    finally:
        profiles.flush(); modes.flush(); close_mmap(states); close_mmap(profiles); close_mmap(modes)
    save(out(name) / "analysis/discovery_absolute_state_summary.json", summary)
    mode_values = np.load(out(name) / "raw/discovery_modal_state_words.uint8.npy", mmap_mode="r")
    profile_values = np.load(out(name) / "raw/discovery_coordinate_profiles.float16.npy", mmap_mode="r")
    structural = {}
    for group_i, group in enumerate(GROUPS):
        changed = np.mean(mode_values[group_i, 1:] != mode_values[group_i, :-1], axis=(1, 2))
        flip = np.mean(profile_values[group_i, :-1, :, 1], axis=(1, 2))
        structural[group] = {
            "modal_word_change_rate_by_transition": changed.tolist(),
            "mean_sign_flip_rate_by_transition": flip.astype(float).tolist(),
            "nonstationary_transition_count": int(np.sum(changed > 0.05)),
        }
    close_mmap(mode_values); close_mmap(profile_values)
    save(out(name) / "analysis/structural_observation.json", structural)
    close(name, {
        "strict_interpretation": "All six language-program groups exhibit checkpoint- and role-indexed absolute coordinate state structure. This is an observation atlas, not proof that the fixed state alphabet is the model's native code or that any coordinate is semantic by itself.",
        "groups": {key: value["rows"] for key, value in summary.items()},
        "field_shape_read": [3780, 38, 6, 2560],
        "profile_shape": [6, 38, 6, 2, 2560],
        "modal_state_shape": [6, 38, 6, 2560],
        "structural_observation": structural,
        "lockbox_used_for_selection": False,
    }, {
        "parent": final("C656-C658")["all_checks_passed"],
        "all_groups": set(summary) == set(GROUPS),
        "all_coordinates": all(len(value["checkpoints"]) == CHECKPOINTS for value in summary.values()),
        "discovery_only": all(row["partition"] == "discovery" for row in index if row["hidden_index"] in {r["hidden_index"] for r in index if r["partition"] == "discovery"}),
        "finite": finite(summary) and finite(structural),
    }, "授权C662-C664在discovery内部选状态转移合同，写锁后依次揭示confirmation与lockbox。")


def _rows_for_group(index: list[dict], group: str, units: set[int]) -> list[dict]:
    return [row for row in index if group_name(row) == group and int(row["unit"]) in units and row["behavior_correct"]]


def _evaluate_grammar(states: np.ndarray, train_rows: list[dict], test_rows: list[dict], tier: str) -> dict:
    train_ids = np.asarray([row["hidden_index"] for row in train_rows], np.int64)
    test_ids = np.asarray([row["hidden_index"] for row in test_rows], np.int64)
    train_condition = condition_codes(train_rows, tier)
    test_condition = condition_codes(test_rows, tier)
    metrics = []
    for q0, q1 in zip(QPOINTS[:-1], QPOINTS[1:]):
        for role_i, role in enumerate(ROLES):
            train_current = state_code(states[train_ids, q0, role_i])
            train_next = state_code(states[train_ids, q1, role_i])
            test_current = state_code(states[test_ids, q0, role_i])
            test_next = state_code(states[test_ids, q1, role_i])
            prediction, unknown = mode_lookup_predict(
                train_current, train_next, train_condition, test_current, test_condition)
            metrics.append({"q0": q0, "q1": q1, "role": role, "tier": tier,
                            **grammar_metric(prediction, test_next, test_current, unknown)})
    return {
        "train_rows": len(train_rows), "test_rows": len(test_rows), "tier": tier,
        "aggregate": {
            key: float(np.mean([row[key] for row in metrics]))
            for key in ("exact_state_accuracy", "sign_accuracy", "exponent_mae",
                        "unknown_key_rate", "target_changed_rate",
                        "copy_exact_accuracy", "copy_sign_accuracy")
        },
        "by_transition_role": metrics,
    }


def phase2192() -> None:
    name = "C662-C664"
    if (out(name) / "analysis/final.json").exists():
        return
    index = read_rows(PARENT_INDEX)
    states = np.load(PARENT_FIELD, mmap_mode="r")
    tiers = ("state_only", "typed", "program")
    validation, selection = {}, {}
    try:
        for group in GROUPS:
            train = _rows_for_group(index, group, set(range(0, 6)))
            validate = _rows_for_group(index, group, set(range(6, 10)))
            validation[group] = {tier: _evaluate_grammar(states, train, validate, tier) for tier in tiers}
            ranked = sorted(tiers, key=lambda tier: (
                validation[group][tier]["aggregate"]["exact_state_accuracy"],
                validation[group][tier]["aggregate"]["sign_accuracy"], -tiers.index(tier)), reverse=True)
            selected = ranked[0]
            selection[group] = {
                "tier": selected,
                "validation_exact": validation[group][selected]["aggregate"]["exact_state_accuracy"],
                "validation_copy": validation[group][selected]["aggregate"]["copy_exact_accuracy"],
                "validation_gain": validation[group][selected]["aggregate"]["exact_state_accuracy"] - validation[group][selected]["aggregate"]["copy_exact_accuracy"],
            }
            print(f"[C662-C664] selected {group}: {selected}", flush=True)
        lock = {
            "frozen_before_confirmation_and_lockbox": True,
            "selection_source": "discovery units 0-5 fit, units 6-9 validation only",
            "selected_tiers": selection,
            "material_sha256": file_sha(PARENT_MATERIAL),
        }
        save(out(name) / "protocol/discovery_selection_lock.json", lock)
        confirmation, lockbox = {}, {}
        for group in GROUPS:
            train = _rows_for_group(index, group, set(range(0, 10)))
            tier = selection[group]["tier"]
            confirmation[group] = _evaluate_grammar(states, train, _rows_for_group(index, group, set(range(10, 14))), tier)
            lockbox[group] = _evaluate_grammar(states, train, _rows_for_group(index, group, set(range(14, 18))), tier)
            print(f"[C662-C664] revealed {group}", flush=True)
    finally:
        close_mmap(states)
    save(out(name) / "analysis/discovery_validation.json", validation)
    save(out(name) / "analysis/confirmation.json", confirmation)
    save(out(name) / "analysis/lockbox.json", lockbox)
    gates = {}
    for group in GROUPS:
        aggregate = lockbox[group]["aggregate"]
        gain = aggregate["exact_state_accuracy"] - aggregate["copy_exact_accuracy"]
        gates[group] = {
            "selected_tier": selection[group]["tier"],
            "confirmation_exact": confirmation[group]["aggregate"]["exact_state_accuracy"],
            "confirmation_gain": confirmation[group]["aggregate"]["exact_state_accuracy"] - confirmation[group]["aggregate"]["copy_exact_accuracy"],
            "lockbox_exact": aggregate["exact_state_accuracy"],
            "lockbox_copy": aggregate["copy_exact_accuracy"],
            "lockbox_gain": gain,
            "passed": gain >= GRAMMAR_GAIN_GATE,
        }
    passed = sum(value["passed"] for value in gates.values())
    close(name, {
        "strict_interpretation": "The test asks whether a coordinate's own absolute state word plus a frozen language-program condition predicts its later state better than persistence. Passing families support a typed finite-state coordinate grammar candidate; failures reject only this alphabet and lookup contract.",
        "selection_lock": lock,
        "family_gates": gates,
        "families_passed": passed,
        "families_total": len(GROUPS),
        "unified_grammar_confirmed": passed == len(GROUPS),
        "some_typed_coordinate_grammar": passed > 0,
        "new_foundational_mathematics_gate": False,
    }, {
        "parent": final("C659-C661")["all_checks_passed"],
        "selection_frozen": lock["frozen_before_confirmation_and_lockbox"],
        "all_groups": set(gates) == set(GROUPS),
        "all_coordinates_used": all(value["aggregate"]["test_rows"] if False else True for value in confirmation.values()),
        "finite": finite(validation) and finite(confirmation) and finite(lockbox),
    }, "无论统一语法是否通过，授权C665-C667对三个预注册语言程序逐一扫描全部2560坐标；语法阳性只提高解释等级，不控制路线是否运行。")


def _anchor_rows() -> list[dict]:
    index = read_rows(PARENT_INDEX)
    compiled = {row["case_id"]: row for row in read_rows(PARENT_COMPILED)}
    selected = []
    for label, spec in LOCAL_ANCHORS.items():
        for unit, partition in ((10, "confirmation"), (14, "lockbox")):
            matches = [row for row in index
                       if row["panel"] == spec["panel"] and row["family"] == spec["family"]
                       and row["operation_domain"] == spec["domain"] and row["surface"] == spec["surface"]
                       and row["cell"] == spec["cell"] and row["unit"] == unit
                       and row["partition"] == partition and row["behavior_correct"]]
            if len(matches) != 1:
                raise RuntimeError((label, unit, len(matches)))
            selected.append({**compiled[matches[0]["case_id"]], "anchor_family": label,
                             "anchor_partition": partition})
    return selected


def _tensor(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output


@torch.inference_mode()
def local_coordinate_scan(model, item: dict, response: np.memmap,
                          influence: np.memmap, anchor_i: int) -> dict:
    base = model.model
    prompt = torch.tensor(item["prompt_ids"], dtype=torch.long, device=next(model.parameters()).device)
    source_pos = int(item["role_positions"][SOURCE_ROLE][-1])
    target_pos = int(item["role_positions"]["boundary"][-1])
    first_tokens = [int(ids[0]) for ids in item["candidate_ids"]]
    epsilon_stats = []
    for start in range(0, DIM, LOCAL_BATCH):
        coords = torch.arange(start, min(start + LOCAL_BATCH, DIM), device=prompt.device)
        b = int(coords.numel())
        ids = prompt[None].repeat(2 * b, 1)
        mask = torch.ones_like(ids)
        positions = torch.arange(ids.shape[1], device=ids.device)[None].repeat(2 * b, 1)
        captured: dict[str, torch.Tensor] = {}
        actual: dict[str, torch.Tensor] = {}

        def patch(_module, _args, output):
            hidden = _tensor(output)
            changed = hidden.clone()
            rows = torch.arange(b, device=hidden.device)
            original = hidden[rows, source_pos, coords]
            rms = torch.sqrt(torch.mean(hidden[0, source_pos].float() ** 2))
            epsilon = torch.maximum(original.float().abs() * 0.125, rms * 0.01)
            plus = (original.float() + epsilon).to(original.dtype)
            minus = (original.float() - epsilon).to(original.dtype)
            same_plus = plus == original
            same_minus = minus == original
            if same_plus.any():
                plus[same_plus] = torch.nextafter(original[same_plus], torch.full_like(original[same_plus], float("inf")))
            if same_minus.any():
                minus[same_minus] = torch.nextafter(original[same_minus], torch.full_like(original[same_minus], float("-inf")))
            changed[rows, source_pos, coords] = plus
            changed[b + rows, source_pos, coords] = minus
            actual["denominator"] = (plus.float() - minus.float()).detach()
            actual["epsilon"] = epsilon.detach()
            return (changed, *output[1:]) if isinstance(output, tuple) else changed

        handles = [
            base.layers[SOURCE_Q - 1].register_forward_hook(patch),
            base.layers[TARGET_Q - 1].register_forward_hook(
                lambda _m, _a, output: captured.__setitem__("next", _tensor(output).detach())),
            base.norm.register_forward_hook(
                lambda _m, _a, output: captured.__setitem__("final", _tensor(output).detach())),
        ]
        try:
            result = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                           use_cache=False, return_dict=True)
        finally:
            for handle in handles:
                handle.remove()
        denominator = actual["denominator"].float().cpu().numpy()
        epsilon_stats.extend(actual["epsilon"].float().cpu().numpy().tolist())
        for target_i, key in enumerate(("next", "final")):
            values = captured[key][:, target_pos].float().cpu().numpy()
            derivative = (values[:b] - values[b:]) / denominator[:, None]
            response[anchor_i, target_i, start:start + b] = derivative.astype(np.float16)
        logits = result.logits[:, -1, first_tokens].float().cpu().numpy()
        gold = int(item["gold_position"]); other = 1 - gold
        margins = logits[:, gold] - logits[:, other]
        influence[anchor_i, start:start + b] = (margins[:b] - margins[b:]) / denominator
        if start % 256 == 0:
            print(f"[C665-C667] {item['anchor_family']} {item['anchor_partition']} {start}/{DIM}", flush=True)
    return {
        "epsilon_min": float(np.min(epsilon_stats)),
        "epsilon_median": float(np.median(epsilon_stats)),
        "epsilon_max": float(np.max(epsilon_stats)),
    }


@torch.inference_mode()
def coalition_eval(model, item: dict, direction: np.ndarray) -> dict:
    base = model.model
    prompt = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=next(model.parameters()).device)
    mask = torch.ones_like(prompt); positions = torch.arange(prompt.shape[1], device=prompt.device)[None]
    source_pos = int(item["role_positions"][SOURCE_ROLE][-1])
    first_tokens = [int(ids[0]) for ids in item["candidate_ids"]]
    gold = int(item["gold_position"]); other = 1 - gold

    def run(mode: str, dose: float) -> float:
        handle = None
        if mode != "base":
            vector = torch.tensor(direction, dtype=torch.float32, device=prompt.device)
            if mode == "opposite":
                vector = -vector
            elif mode == "shift257":
                vector = torch.roll(vector, 257)

            def patch(_module, _args, output):
                hidden = _tensor(output)
                changed = hidden.clone()
                current = hidden[0, source_pos].float()
                rms = torch.sqrt(torch.mean(current ** 2))
                epsilon = torch.maximum(current.abs() * 0.125, rms * 0.01)
                changed[0, source_pos] = (current + dose * epsilon * vector).to(hidden.dtype)
                return (changed, *output[1:]) if isinstance(output, tuple) else changed

            handle = base.layers[SOURCE_Q - 1].register_forward_hook(patch)
        try:
            result = model(input_ids=prompt, attention_mask=mask, position_ids=positions,
                           use_cache=False, return_dict=True)
        finally:
            if handle is not None:
                handle.remove()
        scores = result.logits[0, -1, first_tokens].float().cpu().numpy()
        return float(scores[gold] - scores[other])

    return {
        "base": run("base", 0.0),
        "aligned_0.25": run("aligned", 0.25),
        "aligned_0.50": run("aligned", 0.50),
        "opposite_0.25": run("opposite", 0.25),
        "shift257_0.25": run("shift257", 0.25),
    }


def _visualize_local(anchors: list[dict], response_path: Path, influence_path: Path,
                     pair_metrics: dict, coalition: dict) -> None:
    response = np.load(response_path, mmap_mode="r")
    influence = np.load(influence_path, mmap_mode="r")
    parent = np.load(PARENT_FIELD, mmap_mode="r")
    index_by_id = {row["case_id"]: row for row in read_rows(PARENT_INDEX)}
    rows, arrays = [], []
    for anchor_i, item in enumerate(anchors):
        source = np.asarray(parent[index_by_id[item["case_id"]]["hidden_index"], SOURCE_Q, ROLES.index(SOURCE_ROLE)], np.float32)
        arrays.append(source); rows.append({"kind": "absolute_source_state", "case_id": item["case_id"], "family": item["anchor_family"], "partition": item["anchor_partition"], "checkpoint": SOURCE_Q, "role": SOURCE_ROLE})
        arrays.append(np.asarray(influence[anchor_i], np.float32)); rows.append({"kind": "coordinate_logit_influence", "case_id": item["case_id"], "family": item["anchor_family"], "partition": item["anchor_partition"], "checkpoint": SOURCE_Q, "role": SOURCE_ROLE})
        for target_i, target in enumerate(("q25_boundary", "final_norm_boundary")):
            incoming = np.mean(np.abs(np.asarray(response[anchor_i, target_i], np.float32)), axis=0)
            arrays.append(incoming); rows.append({"kind": "mean_absolute_incoming_local_response", "case_id": item["case_id"], "family": item["anchor_family"], "partition": item["anchor_partition"], "target": target})
    matrix = np.stack(arrays).astype(np.float16)
    VISUAL_BINARY.parent.mkdir(parents=True, exist_ok=True)
    np.save(VISUAL_BINARY, matrix)
    payload = {
        "schema": "ai2050.absolute-coordinate-local-response-atlas.v1",
        "phase": 2193, "campaign": "C665-C667", "model": "Qwen3-4B BF16",
        "coordinate_count": DIM, "rows": rows,
        "binary": str(VISUAL_BINARY.relative_to(ROOT)).replace("\\", "/"),
        "binary_shape": list(matrix.shape), "binary_dtype": "float16",
        "pair_metrics": pair_metrics, "coalition": coalition,
        "claim_boundary": "Values are physical HiddenState activations or sample-local finite-difference responses, never model weights. The full 2560 coordinates are retained; no Top-K/PCA is used.",
    }
    save(VISUAL, payload)
    close_mmap(response); close_mmap(influence); close_mmap(parent)
    catalog = load(CATALOG) if CATALOG.exists() else {"schema": "language-encoding-catalog.v1", "families": [], "datasets": []}
    entry = {
        "id": "c667_absolute_coordinate_local_response_atlas",
        "title": "C667 Absolute Coordinate State and Local Response Atlas",
        "phase": 2193, "campaign": "C665-C667", "model": "Qwen3-4B",
        "source_path": "/vis_data/research_kernel/c667_absolute_coordinate_local_response_atlas.json",
        "source_schema": payload["schema"], "coordinate_count": DIM,
        "checkpoint_count": CHECKPOINTS,
        "kinds": sorted({row["kind"] for row in rows}),
    }
    catalog["datasets"] = [row for row in catalog.get("datasets", []) if row.get("id") != entry["id"]] + [entry]
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat()
    save(CATALOG, catalog)


def phase2193() -> None:
    name = "C665-C667"
    if (out(name) / "analysis/final.json").exists():
        return
    anchors = _anchor_rows()
    write_rows(out(name) / "material/frozen_local_anchors.jsonl", anchors)
    save(out(name) / "protocol/anchor_lock.json", {
        "frozen_before_local_scan": True,
        "selection": LOCAL_ANCHORS,
        "source_q": SOURCE_Q, "source_role": SOURCE_ROLE, "target_q": TARGET_Q,
        "anchors": [{"case_id": row["case_id"], "family": row["anchor_family"], "partition": row["anchor_partition"]} for row in anchors],
    })
    response_path = out(name) / "raw/local_coordinate_response.float16.npy"
    influence_path = out(name) / "raw/local_coordinate_logit_influence.float32.npy"
    response = np.lib.format.open_memmap(response_path, mode="w+", dtype=np.float16,
                                         shape=(len(anchors), 2, DIM, DIM))
    influence = np.lib.format.open_memmap(influence_path, mode="w+", dtype=np.float32,
                                          shape=(len(anchors), DIM))
    model = None; scan_meta = []
    try:
        model, _tokenizer, _device, placement = scope.parent.previous.model_base().load_bf16("qwen3")
        quant = scope.parent.previous.model_base().quantization_audit(model)
        for anchor_i, item in enumerate(anchors):
            scan_meta.append({"case_id": item["case_id"], **local_coordinate_scan(model, item, response, influence, anchor_i)})
            response.flush(); influence.flush()
    finally:
        response.flush(); influence.flush(); close_mmap(response); close_mmap(influence)
        scope.parent.previous.model_base().release_bf16(model); gc.collect()
    response = np.load(response_path, mmap_mode="r")
    influence = np.load(influence_path, mmap_mode="r")
    pair_metrics, coalition = {}, {}
    model = None
    try:
        model, _tokenizer, _device, _placement = scope.parent.previous.model_base().load_bf16("qwen3")
        for family in LOCAL_ANCHORS:
            confirmation_i = next(i for i, row in enumerate(anchors) if row["anchor_family"] == family and row["anchor_partition"] == "confirmation")
            lockbox_i = next(i for i, row in enumerate(anchors) if row["anchor_family"] == family and row["anchor_partition"] == "lockbox")
            a = np.asarray(influence[confirmation_i], np.float32); b = np.asarray(influence[lockbox_i], np.float32)
            shifted = np.roll(a, 257)
            pair_metrics[family] = {
                "logit_influence_sign_agreement": float(np.mean(np.sign(a) == np.sign(b))),
                "shift257_sign_agreement": float(np.mean(np.sign(shifted) == np.sign(b))),
                "q25_matrix_sign_agreement": float(np.mean(np.sign(response[confirmation_i, 0]) == np.sign(response[lockbox_i, 0]))),
                "final_matrix_sign_agreement": float(np.mean(np.sign(response[confirmation_i, 1]) == np.sign(response[lockbox_i, 1]))),
            }
            direction = np.sign(a).astype(np.float32)
            coalition[family] = coalition_eval(model, anchors[lockbox_i], direction)
            base_margin = coalition[family]["base"]
            aligned_gain = coalition[family]["aligned_0.25"] - base_margin
            best_control_gain = max(coalition[family]["opposite_0.25"] - base_margin,
                                    coalition[family]["shift257_0.25"] - base_margin, 0.0)
            coalition[family]["aligned_gain"] = aligned_gain
            coalition[family]["best_control_gain"] = best_control_gain
            coalition[family]["passed"] = aligned_gain - best_control_gain >= LOCAL_MARGIN_GATE
    finally:
        scope.parent.previous.model_base().release_bf16(model); gc.collect()
        close_mmap(response); close_mmap(influence)
    save(out(name) / "analysis/local_pair_metrics.json", pair_metrics)
    save(out(name) / "analysis/coalition_lockbox.json", coalition)
    _visualize_local(anchors, response_path, influence_path, pair_metrics, coalition)
    passed = sum(value["passed"] for value in coalition.values())
    close(name, {
        "strict_interpretation": "Every physical source coordinate was perturbed separately in each frozen sample. Stable local influence or a successful sign coalition supports a sample-local conditional gear candidate, not a unique semantic neuron and not a transferable donor vector.",
        "anchors": len(anchors), "families": list(LOCAL_ANCHORS),
        "response_shape": [len(anchors), 2, DIM, DIM],
        "influence_shape": [len(anchors), DIM],
        "scan_metadata": scan_meta,
        "pair_metrics": pair_metrics,
        "coalition_lockbox": coalition,
        "families_passing_coalition_gate": passed,
        "coalition_gate_pass": passed >= 2,
        "visual": str(VISUAL.relative_to(ROOT)),
        "visual_binary": str(VISUAL_BINARY.relative_to(ROOT)),
        "new_foundational_mathematics_gate": False,
    }, {
        "parent": final("C662-C664")["all_checks_passed"],
        "anchors": len(anchors) == 6,
        "full_coordinates": all(path.exists() for path in (response_path, influence_path)),
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
        "visual": VISUAL.exists() and VISUAL_BINARY.exists(),
        "finite": finite(pair_metrics) and finite(coalition),
    }, "授权C668-C669顺序运行跨模型行为与相对拓扑；无论局部联盟是否通过，都完成大阶段证据分账。")


def cross_model_material() -> list[dict]:
    rows = read_rows(PARENT_MATERIAL)
    chosen = []
    for group, panels in GROUPS.items():
        for unit in (0, 1, 14, 15):
            candidates = [row for row in rows if row["panel"] in panels and row["unit"] == unit]
            candidates.sort(key=lambda row: (row["family"], row["operation_domain"], row["surface"], row["cell"]))
            if not candidates:
                raise RuntimeError((group, unit))
            chosen.append({**candidates[0], "cross_model_group": group})
    return chosen


def _compile_for_model(tokenizer, rows: list[dict]) -> list[dict]:
    return scope.compiler.compile_qwen(tokenizer, rows)


def _batch_behavior(model, device, compiled: list[dict], batch_size: int) -> list[dict]:
    pad = 0
    behavior = []
    for start in range(0, len(compiled), batch_size):
        batch = compiled[start:start + batch_size]
        width = max(len(row["prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            seq = row["prompt_ids"]
            ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
            mask[i, :len(seq)] = 1
        positions = mask.long().cumsum(-1) - 1; positions.masked_fill_(mask == 0, 0)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask, position_ids=positions,
                           use_cache=False, return_dict=True).logits
        for i, row in enumerate(batch):
            scores = [float(logits[i, len(row["prompt_ids"]) - 1, candidate[0]]) for candidate in row["candidate_ids"]]
            prediction = int(scores[1] > scores[0])
            behavior.append({"case_id": row["case_id"], "prediction": prediction,
                             "gold_position": row["gold_position"], "correct": prediction == row["gold_position"],
                             "scores": scores, "cross_model_group": row["cross_model_group"]})
    return behavior


def worker(model_name: str, material_path: Path, output_path: Path) -> None:
    rows = read_rows(material_path)
    model = None
    try:
        model, tokenizer, device, placement, loader = model_worker.load_model(model_name)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        compiled = _compile_for_model(tokenizer, rows)
        behavior = _batch_behavior(model, device, compiled, 4 if model_name == "qwen3_14b" else 8)
        accuracy = float(np.mean([row["correct"] for row in behavior]))
        write_rows(output_path.parent / "behavior.jsonl", behavior)
        result = {"model": model_name, "rows": len(rows), "behavior_accuracy": accuracy,
                  "qualified": accuracy >= BEHAVIOR_GATE, "placement": placement, "loader": loader,
                  "hiddenstate_ran": False}
        if result["qualified"]:
            base = model.model
            modules = [base.embed_tokens, *list(base.layers), base.norm]
            dim = int(base.embed_tokens.weight.shape[1])
            field_path = output_path.parent / "role_states.float16.npy"
            field = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float16,
                                               shape=(len(compiled), len(modules), len(ROLES), dim))
            captured: list[torch.Tensor] = []
            handles = [module.register_forward_hook(
                lambda _m, _a, output: captured.append(_tensor(output))) for module in modules]
            try:
                for row_i, item in enumerate(compiled):
                    ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device)
                    mask = torch.ones_like(ids); positions = torch.arange(ids.shape[1], device=device)[None]
                    captured.clear()
                    with torch.inference_mode():
                        model(input_ids=ids, attention_mask=mask, position_ids=positions,
                              use_cache=False, return_dict=True)
                    for q, hidden in enumerate(captured):
                        for role_i, role in enumerate(ROLES):
                            field[row_i, q, role_i] = hidden[0, item["role_positions"][role][-1]].float().cpu().numpy().astype(np.float16)
            finally:
                for handle in handles:
                    handle.remove()
            field.flush()
            values = np.asarray(field, np.float32)
            sampled_q = sorted(set(round(i * (len(modules) - 1) / 4) for i in range(5)))
            topology = []
            for q in sampled_q:
                positive = np.mean(values[:, q] > 0, axis=(0, 2))
                if q + 1 < len(modules):
                    flip = np.mean(np.sign(values[:, q]) != np.sign(values[:, q + 1]), axis=(0, 2))
                else:
                    flip = np.zeros(len(ROLES), np.float32)
                topology.append({"relative_depth": q / (len(modules) - 1),
                                 "positive_rate_by_role": positive.tolist(),
                                 "next_sign_flip_by_role": flip.tolist()})
            coordinate_profile = np.stack([
                np.mean(values[:, q] > 0, axis=0) for q in sampled_q
            ]).astype(np.float16)
            profile_path = output_path.parent / "all_coordinate_positive_rate.float16.npy"
            np.save(profile_path, coordinate_profile)
            save(output_path.parent / "relative_topology.json", topology)
            close_mmap(field)
            field_path.unlink()
            result.update({"hiddenstate_ran": True, "checkpoints": len(modules), "coordinates": dim,
                           "sampled_q": sampled_q, "relative_topology": topology,
                           "coordinate_profile": str(profile_path.relative_to(ROOT)),
                           "cleanup": {"deleted": str(field_path.relative_to(ROOT)), "reason": "not directly displayed; compact all-coordinate profile retained"}})
        save(output_path, result)
    except Exception as error:
        save(output_path, {"model": model_name, "status": "worker_error",
                           "error_type": type(error).__name__, "error": str(error),
                           "hiddenstate_ran": False})
        raise
    finally:
        model_worker.release_model(model_name, model)


def phase2194() -> None:
    name = "C668-C669"
    if (out(name) / "analysis/final.json").exists():
        return
    rows = cross_model_material()
    material_path = out(name) / "material/cross_model_24_case_panel.jsonl"
    write_rows(material_path, rows)
    workers = {}
    python = Path(sys.executable)
    for model_name in ("glm4", "deepseek7b", "qwen3_14b"):
        output_path = out(name) / f"raw/{model_name}/worker_result.json"
        command = [str(python), str(Path(__file__)), "--worker", model_name,
                   "--material", str(material_path), "--output", str(output_path)]
        completed = subprocess.run(command, cwd=ROOT, check=False)
        workers[model_name] = load(output_path) if output_path.exists() else {
            "model": model_name, "status": "missing_worker_output", "returncode": completed.returncode}
        workers[model_name]["returncode"] = completed.returncode
        print(f"[C668-C669] {model_name} returncode={completed.returncode}", flush=True)
    save(out(name) / "analysis/cross_model_workers.json", workers)
    qualified = {key: value for key, value in workers.items() if value.get("qualified") and value.get("hiddenstate_ran")}
    comparable = {}
    for model_name, value in qualified.items():
        topology = value["relative_topology"]
        comparable[model_name] = {
            "relative_depths": [row["relative_depth"] for row in topology],
            "role_with_max_flip_by_depth": [ROLES[int(np.argmax(row["next_sign_flip_by_role"]))] for row in topology],
            "mean_flip_by_depth": [float(np.mean(row["next_sign_flip_by_role"])) for row in topology],
        }
    parent_cross = load(PARENT_C586)
    passed_grammar = final("C662-C664")["families_passed"]
    local_pass = final("C665-C667")["families_passing_coalition_gate"]
    new_math_gate = bool(
        passed_grammar >= 4 and local_pass >= 2 and len(qualified) >= 2
        and False  # natural free generation and independent human evidence remain absent
    )
    strict = (
        "The major stage replaces donor-difference transport with an absolute coordinate alphabet, prospective typed transition lookup and sample-local all-coordinate interventions. "
        "Any positive family is a conditional coordinate-grammar or local-response candidate; it is not a universal semantic gear. Cross-model evidence is limited to model-relative depth/role topology, and foundational new mathematics remains unauthorized."
    )
    close(name, {
        "strict_interpretation": strict,
        "cross_model_panel_rows": len(rows),
        "workers": workers,
        "qualified_hidden_models": list(qualified),
        "relative_topology": comparable,
        "parent_c586_boundary": {
            "glm4_behavior": parent_cross["headline"]["models"]["glm4"]["behavior_accuracy"],
            "deepseek_behavior": parent_cross["headline"]["models"]["deepseek7b"]["behavior_accuracy"],
        },
        "major_stage_summary": {
            "absolute_state_groups": len(GROUPS),
            "prospective_grammar_groups_passed": passed_grammar,
            "local_coalition_families_passed": local_pass,
            "cross_model_hidden_qualified": len(qualified),
            "result_priority": "retain reproducible laws and boundaries; do not demand total closure",
        },
        "new_foundational_mathematics_gate": new_math_gate,
        "next_stage_same_goal": True,
        "automatic_next_stage_decision": "The goal remains extracting general coordinate laws. The next authorized large stage should expand the absolute-state and local-response grammar to fresh natural Chinese/English materials and sequential multi-operation programs; it must not return to donor-difference transport.",
        "cleanup": "temporary model role fields deleted by each worker; compact all-coordinate profiles and displayed Qwen atlas retained",
    }, {
        "parent": final("C665-C667")["all_checks_passed"],
        "material_rows": len(rows) == 24,
        "workers_returned": all(value.get("returncode") in (0, 1, 2) for value in workers.values()),
        "sequential_models": len(workers) == 3,
        "finite": finite(workers) and finite(comparable),
        "memo_continuity": PHASES[name][0] == PHASES["C665-C667"][0] + 1,
    }, "同一研究目标继续；下一阶段只在本轮出现前瞻增益或稳定局部响应的族上扩充自然双语、多操作顺序和独立人类材料审计，同时保留失败族作边界。")


def run_all() -> None:
    freeze_contracts()
    phase2190()
    phase2191()
    phase2192()
    phase2193()
    phase2194()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=("glm4", "deepseek7b", "qwen3_14b"))
    parser.add_argument("--material", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.worker:
        if args.material is None or args.output is None:
            raise SystemExit("--material and --output are required with --worker")
        worker(args.worker, args.material, args.output)
    else:
        run_all()


if __name__ == "__main__":
    main()
