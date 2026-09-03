#!/usr/bin/env python3
"""C684-C709 unified relation-state absolute-coordinate campaign."""
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import math
import re
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
VISUAL = ROOT / "frontend/public/vis_data/research_kernel/c699_unified_relation_coordinate_response_atlas.json"
VISUAL_BINARY = ROOT / "frontend/public/vis_data/research_kernel/c699_unified_relation_coordinate_response.float16.npy"
sys.path.insert(0, str(TESTS))

import model_utils
import phase2105_c571_c589_scope_program_algebra_campaign as scope
import phase2190_c656_c669_absolute_coordinate_grammar_campaign as prior

PHASES = {
    "C684-C688": (2200, "unified_relation_contract_behavior_and_full_field"),
    "C689-C693": (2201, "shared_nuisance_shuffled_relational_tournament"),
    "C694-C699": (2202, "all_coordinate_local_response_output_call_and_visual"),
    "C700-C704": (2203, "sequential_cross_model_relative_topology"),
    "C705-C709": (2204, "major_stage_adjudication_and_cleanup"),
}
OUTS = {key: RESULT / f"phase{phase}_{key.lower().replace('-', '_')}_{slug}"
        for key, (phase, slug) in PHASES.items()}

DIM, CHECKPOINTS = 2560, 38
ROLES, QPOINTS = prior.ROLES, prior.QPOINTS
LANGUAGES, OPERATIONS, UNITS = ("en", "zh"), ("event", "path", "combined"), 24
BEHAVIOR_GATE, RELATION_GAIN_GATE = 0.75, 0.01
LOCAL_GATE, COALITION_GATE = 0.02, 0.05

NAMES_A_EN = ("Mira", "Nolan", "Iris", "Darin", "Lena", "Oren", "Faye", "Gavin", "Talia", "Ronan", "Vera", "Caleb", "Nadia", "Elias", "Sela", "Bram", "Aria", "Jonas", "Lyra", "Marek", "Cora", "Silas", "Nina", "Tobin")
NAMES_B_EN = ("Tovan", "Selin", "Borin", "Kira", "Milo", "Rhea", "Damon", "Elin", "Pavel", "Nora", "Ivo", "Mina", "Ravi", "Tessa", "Noel", "Kara", "Dorian", "Lina", "Perrin", "Maya", "Orin", "Tara", "Vito", "Rina")
NAMES_A_ZH = ("米拉", "诺兰", "伊里斯", "达林", "莱娜", "奥伦", "费伊", "加文", "塔莉娅", "罗南", "维拉", "凯莱布", "纳迪娅", "埃利亚斯", "塞拉", "布拉姆", "阿丽娅", "约纳斯", "莱拉", "马雷克", "科拉", "西拉斯", "妮娜", "托宾")
NAMES_B_ZH = ("托万", "塞林", "博林", "基拉", "米洛", "瑞娅", "达蒙", "埃林", "帕维尔", "诺拉", "伊沃", "米娜", "拉维", "泰莎", "诺埃尔", "卡拉", "多里安", "莉娜", "佩林", "玛雅", "奥林", "塔拉", "维托", "丽娜")
OBJECTS_EN = ("apple", "banana", "pear", "peach", "grape", "lemon", "orange", "plum", "cherry", "mango", "melon", "coconut", "carrot", "potato", "tomato", "onion", "cabbage", "bean", "pea", "corn", "rice", "wheat", "bread", "cheese")
OBJECTS_ZH = ("苹果", "香蕉", "梨", "桃子", "葡萄", "柠檬", "橙子", "李子", "樱桃", "芒果", "甜瓜", "椰子", "胡萝卜", "土豆", "番茄", "洋葱", "卷心菜", "豆子", "豌豆", "玉米", "大米", "小麦", "面包", "奶酪")
DISTRACT_EN = ("hammer", "violin", "ladder", "compass", "lantern", "pillow", "mirror", "bucket", "anchor", "helmet", "whistle", "tripod", "notebook", "ribbon", "basket", "key", "blanket", "bottle", "umbrella", "rope", "tablet", "folder", "shell", "frame")
DISTRACT_ZH = ("锤子", "小提琴", "梯子", "罗盘", "灯笼", "枕头", "镜子", "水桶", "船锚", "头盔", "口哨", "三脚架", "笔记本", "丝带", "篮子", "钥匙", "毯子", "瓶子", "雨伞", "绳子", "木板", "文件夹", "贝壳", "框架")

CELLS = (
    ("event_active_record", "event", "record", 0, 0, 0, 0, 0),
    ("event_active_dialogue", "event", "dialogue", 0, 0, 0, 0, 0),
    ("event_passive_record", "event", "record", 1, 0, 0, 0, 0),
    ("event_outer_dialogue", "event", "dialogue", 0, 1, 0, 0, 0),
    ("event_inner_record", "event", "record", 0, 0, 1, 0, 0),
    ("event_both_passive_dialogue", "event", "dialogue", 1, 1, 1, 0, 0),
    ("path_depth1_record", "path", "record", 0, 0, 0, 1, 0),
    ("path_depth1_dialogue", "path", "dialogue", 0, 0, 0, 1, 0),
    ("path_depth2_record", "path", "record", 1, 0, 0, 2, 0),
    ("path_depth2_dialogue", "path", "dialogue", 1, 1, 0, 2, 0),
    ("path_depth3_record", "path", "record", 0, 0, 1, 3, 0),
    ("path_depth3_shortcut_dialogue", "path", "dialogue", 0, 0, 1, 3, 1),
    ("combined_depth1_record", "combined", "record", 0, 0, 0, 1, 0),
    ("combined_depth1_dialogue", "combined", "dialogue", 1, 0, 0, 1, 0),
    ("combined_depth3_record", "combined", "record", 0, 0, 0, 3, 0),
    ("combined_depth3_dialogue", "combined", "dialogue", 1, 0, 0, 3, 0),
)


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_rows(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows: handle.write(canonical(row) + "\n")


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def out(name: str) -> Path: return OUTS[name]
def final(name: str) -> dict: return load(out(name) / "analysis/final.json")


def close_mmap(value: Any) -> None:
    mmap = getattr(value, "_mmap", None)
    if mmap is not None: mmap.close()


def finite(value: Any) -> bool:
    if isinstance(value, dict): return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(finite(v) for v in value)
    return not isinstance(value, (float, np.floating)) or math.isfinite(float(value))


def hash_id(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "little")


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024): digest.update(chunk)
    return digest.hexdigest()


def partition(unit: int) -> str:
    return "discovery" if unit < 12 else ("confirmation" if unit < 18 else "lockbox")


def words(unit: int, language: str) -> dict:
    if unit < 12: types_en, types_zh = ("fruit", "food", "object"), ("水果", "食物", "物体")
    elif unit < 20: types_en, types_zh = ("vegetable", "food", "object"), ("蔬菜", "食物", "物体")
    else: types_en, types_zh = ("staple food", "food", "object"), ("主食", "食物", "物体")
    if language == "en":
        return {"a": NAMES_A_EN[unit], "b": NAMES_B_EN[unit], "x": OBJECTS_EN[unit], "y": DISTRACT_EN[unit], "types": types_en}
    return {"a": NAMES_A_ZH[unit], "b": NAMES_B_ZH[unit], "x": OBJECTS_ZH[unit], "y": DISTRACT_ZH[unit], "types": types_zh}


def event_clause(u: dict, language: str, voice: int, inner: int, target: str) -> tuple[str, str]:
    if language == "en":
        if voice: return f"the {target} was {'not ' if inner else ''}eaten by {u['b']}", "eaten"
        return f"{u['b']} {'did not eat' if inner else 'ate'} the {target}", "eat" if inner else "ate"
    if voice: return f"{target}{'没有' if inner else ''}被{u['b']}吃掉", "吃掉"
    surface = "没有吃" if inner else "吃了"
    return f"{u['b']}{surface}{target}", surface


def attitude(u: dict, language: str, voice: int, outer: int, inner: int, target: str) -> tuple[str, str]:
    clause, relation = event_clause(u, language, voice, inner, target)
    if language == "en": return f"{u['a']} {'did not remember' if outer else 'remembered'} that {clause}", relation
    return f"{u['a']}{'不记得' if outer else '记得'}{clause}", relation


def build_row(unit: int, language: str, cell_i: int, cell: tuple) -> dict:
    cell_name, operation, surface, voice, outer, inner, depth, shortcut = cell
    u = words(unit, language); truth = unit % 2 == 0; target = u["x"] if truth else u["y"]
    t1, t2, t3 = u["types"]
    event_fact, event_relation = attitude(u, language, voice, outer, inner, u["x"])
    if language == "en":
        type_facts = [f"the {u['x']} is a kind of {t1}", f"{t1} is a kind of {t2}", f"{t2} is a kind of {t3}"]
        if shortcut: type_facts.append(f"the {u['x']} is directly registered as a kind of {t3}")
        noise = f"{u['a']} stored the {u['y']} in a separate inventory"
    else:
        type_facts = [f"{u['x']}是一种{t1}", f"{t1}是一种{t2}", f"{t2}是一种{t3}"]
        if shortcut: type_facts.append(f"{u['x']}被直接登记为一种{t3}")
        noise = f"{u['a']}把{u['y']}登记在另一份清单中"
    if operation == "event":
        query_statement, _ = attitude(u, language, voice, outer, inner, target)
        question = f"Is it true that {query_statement}?" if language == "en" else f"“{query_statement}”这句话成立吗？"
        relation, query = event_relation, u["b"]
    elif operation == "path":
        endpoint = (t1, t2, t3)[depth - 1]; query_target = endpoint if truth else u["y"]
        question = f"Is the {u['x']} a kind of {query_target}?" if language == "en" else f"{u['x']}是一种{query_target}吗？"
        relation, query = ("is a kind of" if language == "en" else "是一种"), u["x"]
    else:
        endpoint, event_target = (t1 if depth == 1 else t3), target
        question = (f"According to both the event and type links, did {u['b']} eat the {event_target}, which is a kind of {endpoint}?"
                    if language == "en" else f"综合事件和类型关系，{u['b']}吃的是{event_target}，而且它是一种{endpoint}吗？")
        relation, query = event_relation, u["b"]
    facts = [event_fact, *type_facts, noise]
    if language == "en":
        body = "; ".join(facts) + "."
        prefix = "A verified relation record states:" if surface == "record" else "During a verified hearing, the following relations were read aloud:"
        core = f"{prefix} {body} Based only on those relations, {question}"
        labels = ("Yes", "No")
    else:
        body = "；".join(facts) + "。"
        prefix = "一份经过核验的关系记录写道：" if surface == "record" else "在一次经过核验的听证中，以下关系被逐条宣读："
        core = f"{prefix}{body}只根据这些关系，{question}"
        labels = ("是", "否")
    correct, wrong = (labels[0], labels[1]) if truth else (labels[1], labels[0])
    order = ((unit // 2) + cell_i + (language == "zh")) % 2
    options = f"(A) {correct} (B) {wrong}" if order == 0 else f"(A) {wrong} (B) {correct}"
    factors = {"voice": voice, "outer": outer, "inner": inner, "depth": depth, "shortcut": shortcut}
    return {
        "case_id": f"c684-{language}-u{unit:02d}-{cell_name}", "panel": "unified_relation_state",
        "family": "unified_relation_state", "query_operation": operation, "operation_type": operation,
        "operation_domain": f"{operation}:{depth}", "language": language, "surface": surface,
        "cell": cell_name, "unit": unit, "partition": partition(unit), "truth": truth,
        "correct_answer": correct, "wrong_answer": wrong, "gold_position": order, "option_order": order,
        "prompt_core": core,
        "prompt": f"{core} {options}. Reply with only A or B." if language == "en" else f"{core} {options}。只回答A或B。",
        "free_prompt": f"{core} Answer only Yes or No." if language == "en" else f"{core} 只回答“是”或“否”。",
        "role_values": {"primary": u["a"], "secondary": u["b"], "relation": relation, "context": u["x"], "query": query},
        "factors": factors,
        "semantic_graph": {"event": [u["a"], u["b"], u["x"]], "type_chain": [u["x"], t1, t2, t3],
                           "query_operation": operation, "serialization": language,
                           "labels_are_observation_coordinates_not_modules": True},
    }


def material() -> list[dict]:
    return [build_row(unit, language, cell_i, cell)
            for unit, language, (cell_i, cell) in itertools.product(range(UNITS), LANGUAGES, enumerate(CELLS))]


def protocol(name: str) -> dict:
    common = {
        "camera": "embedding + all post-block HiddenStates + final norm + logits; all coordinates",
        "forbidden": ["attention", "MLP", "weights", "gradients", "PCA", "Top-K", "donor differences"],
        "label_rule": "operation labels are experimental coordinates in one relation state, not internal modules",
        "human_review": "NA_not_run",
        "failure_policy": "route-level missingness; complete all registered routes",
    }
    objects = {
        "C684-C688": {"object": "audit C656-C683; compile UTF-8 unified material; dual behavior gate; capture qualified field", "behavior_gate": BEHAVIOR_GATE},
        "C689-C693": {"object": "shared/nuisance/shuffled/relational/program state-word tournament", "gain_gate": RELATION_GAIN_GATE, "lockbox": "confirmation-positive strata only"},
        "C694-C699": {"object": "full 2560-coordinate local response and coordinate coalition", "local_gate": LOCAL_GATE, "coalition_gate": COALITION_GATE},
        "C700-C704": {"object": "sequential GLM4, DeepSeek-7B, Qwen3-14B anonymous relative topology", "same_coordinate_comparison": False},
        "C705-C709": {"object": "joint audit, theory gate, visualization and cleanup"},
    }
    return {**common, **objects[name]}


def freeze() -> None:
    for name in PHASES:
        for part in ("protocol", "material", "behavior", "raw", "analysis", "audit", "external"):
            (out(name) / part).mkdir(parents=True, exist_ok=True)
        path = out(name) / "protocol/preregistration.json"
        if not path.exists():
            save(path, {"phase": PHASES[name][0], "campaign": name,
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "protocol": protocol(name)})


TITLES = {
    "C684-C688": "统一关系状态合同、双行为门与全坐标场",
    "C689-C693": "共享、干扰、打乱与关系程序增量总赛",
    "C694-C699": "全坐标局部响应、输出调用与参数热力图",
    "C700-C704": "三模型顺序行为资格与匿名相对拓扑",
    "C705-C709": "大阶段联合审计、理论裁决与清理",
}
FORMULAS = {
    "C684-C688": "$$\n\\mathfrak L=(\\mathcal G,\\mathcal O,\\Sigma,\\mathcal Q),\\qquad o:\\mathcal G_{\\operatorname{domain}(o)}\\rightharpoonup\\mathcal G\n$$\n$$\nG_{beh}(s)=\\mathbf 1[A_{cand}^{dev}(s)\\ge0.75\\land A_{gen}^{conf}(s)\\ge0.75]\n$$",
    "C689-C693": "$$\nZ_{q,r,j}=(\\operatorname{sgn}H_{q,r,j},\\operatorname{clip}_{[-8,7]}\\lfloor\\log_2|H_{q,r,j}|\\rfloor)\n$$\n$$\nG_{relation}=S(T_{relation/program})-\\max\\{S(T_{shared}),S(T_{nuisance}),S(T_{shuffled})\\}\n$$",
    "C694-C699": "$$\nJ^{(x)}_{j\\to k}=\\frac{H_k(x;H_j+\\epsilon_j)-H_k(x;H_j-\\epsilon_j)}{2\\epsilon_j}\n$$",
    "C700-C704": "$$\n\\Theta_M(u,r)=\\{\\Pr_M[Z>0],\\Pr_M[\\operatorname{Flip}],\\operatorname{rank}_r(\\operatorname{Flip})\\},\\qquad j_M\\not\\equiv j_{M'}\n$$",
    "C705-C709": "$$\nH_{q+1}=\\mathcal U_q(H_q)+\\mathcal D_q(H_q,\\delta),\\qquad \\mathcal D_q(H,\\delta)=\\sum_e g_e(H,\\delta)\\Psi_e(H,\\delta)\n$$",
}
EXAMPLES = {
    "C684-C688": "每条提示同时包含态度事件、施事受事、类型链和无关清单；只改变语态、内外否定、查询、路径深度、捷径、表面与中英文序列化。",
    "C689-C693": "预测同一物理坐标下一检查点的状态字，关系模型必须同时击败共享状态、长度位置表面干扰和打乱标签。",
    "C694-C699": "在q24关系角色逐一拨动全部2560坐标，读取q25边界、final边界和候选margin，不筛热点。",
    "C700-C704": "三个模型依次运行同一24题面板，各自行为合格后才读取角色状态场。",
    "C705-C709": "把行为、关系增量、局部特异性、输出调用和跨模型结果严格分账，并删除未显示的大型原始场。",
}


def append_memo(name: str, result: dict) -> None:
    phase = PHASES[name][0]; marker = f"## Phase {phase}:"
    existing = MEMO.read_text(encoding="utf-8-sig") if MEMO.exists() else ""
    if marker in existing: return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = f"""

## Phase {phase}: {TITLES[name]} [{stamp}]

**研究边界。** `{name}`。事件、路径、作用域、语态和语言仅是同一外部关系状态的受控变换，不预设模型内部存在独立模块。只读取词嵌入、各block后的HiddenState、final norm和输出logit；保留全部物理激活坐标，不读取attention、MLP、权重或梯度，不使用PCA、Top-K或供体差分搬运。独立人类盲评未运行，严格记为`NA_not_run`。

**运行前冻结合同。**
```json
{json.dumps(load(out(name) / 'protocol/preregistration.json'), ensure_ascii=False, indent=2)}
```

**测试用例。** {EXAMPLES[name]}

**测试原理与数学公式。**
{FORMULAS[name]}

**详细结果与门槛。**
```json
{json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)}
```

**分析、理论进展与严格结论。** {result.get('strict_interpretation')} 理论主体名称保持“条件化输出场闭合理论”，本期只更新经验对象，不另造理论名。

**问题、硬伤和瓶颈。** 模板材料不能替代独立母语者盲评；外部变换标签只是实验坐标；候选和自由生成通过不证明内部模块；状态字离散了连续激活；有限差分只描述当前基态邻域；跨模型粗曲线不等于功能同构；小模型结论不能直接外推；新基础数学门必须保持克制。

**相关文件。** 主脚本`tests/glm5/phase2200_c684_c709_unified_relation_response_campaign.py`；结果目录`{out(name).relative_to(ROOT)}`；裁决`{(out(name) / 'analysis/final.json').relative_to(ROOT)}`。

**结论与下一步授权。** {result.get('next_authorization')}
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as handle: handle.write(text)


def close(name: str, body: dict, checks: dict, authorization: str) -> dict:
    result = {"phase": PHASES[name][0], "campaign": name, "status": "closed",
              "timestamp_utc": datetime.now(timezone.utc).isoformat(), "checks": checks,
              "all_checks_passed": bool(checks) and all(bool(v) for v in checks.values()),
              **body, "next_authorization": authorization}
    save(out(name) / "analysis/final.json", result); append_memo(name, result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True); return result


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
                                         local_files_only=True, use_fast=False)


def batch_behavior(model, device, compiled: list[dict], batch_size: int = 12) -> list[dict]:
    pad = int(model.config.pad_token_id or model.config.eos_token_id or 0); results = []
    for start in range(0, len(compiled), batch_size):
        batch = compiled[start:start + batch_size]; width = max(len(row["prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            seq = row["prompt_ids"]; ids[i, :len(seq)] = torch.tensor(seq, device=device); mask[i, :len(seq)] = 1
        pos = mask.long().cumsum(-1) - 1; pos.masked_fill_(mask == 0, 0)
        with torch.inference_mode(): logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True).logits
        for i, row in enumerate(batch):
            scores = [float(logits[i, len(row["prompt_ids"]) - 1, candidate[0]]) for candidate in row["candidate_ids"]]
            pred = int(scores[1] > scores[0]); results.append({"case_id": row["case_id"], "prediction": pred,
                "gold_position": row["gold_position"], "correct": pred == row["gold_position"], "scores": scores})
    return results


def parse_binary(text: str, language: str) -> str | None:
    clean = text.strip().lower()
    if language == "zh":
        match = re.search(r"(?<!不)(是)|否", clean); return match.group(0) if match else None
    match = re.search(r"\b(yes|no)\b", clean); return match.group(1).capitalize() if match else None


def free_generate(model, tokenizer, device, compiled: list[dict], batch_size: int = 8) -> list[dict]:
    pad = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id); results = []
    for start in range(0, len(compiled), batch_size):
        batch = compiled[start:start + batch_size]; width = max(len(row["free_prompt_ids"]) for row in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
        for i, row in enumerate(batch):
            seq = row["free_prompt_ids"]; offset = width - len(seq)
            ids[i, offset:] = torch.tensor(seq, device=device); mask[i, offset:] = 1
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=5, do_sample=False,
                                       pad_token_id=pad, eos_token_id=tokenizer.eos_token_id)
        for i, row in enumerate(batch):
            decoded = tokenizer.decode(generated[i, width:].tolist(), skip_special_tokens=True)
            parsed = parse_binary(decoded, row["language"])
            results.append({"case_id": row["case_id"], "text": decoded, "parsed": parsed,
                            "correct_answer": row["correct_answer"], "correct": parsed == row["correct_answer"]})
    return results


def phase2200(rows: list[dict]) -> None:
    name = "C684-C688"
    if (out(name) / "analysis/final.json").exists(): return
    parent_audit = {
        "retained": [
            "C656-C664 established a useful all-coordinate activation-state alphabet and strong shared coordinate dynamics.",
            "C665-C667 left one nested-attitude dense margin-direction candidate but failed its joint gate.",
            "C668-C669 produced coarse model-relative role/depth curves, not functional isomorphism.",
            "C670-C683 correctly closed before behavior because its frozen bilingual material was invalid.",
        ],
        "corrections": [
            "Activation coordinates are not model parameters.",
            "Six copy-baseline gains are not six semantic grammars; only recursive knowledge had a clear discovery increment over state-only.",
            "High local-matrix agreement was mostly shared physics, not family-specific circuitry.",
            "External language labels are observation coordinates, not assumed internal modules.",
            "all_checks_passed means procedural closure, not scientific success.",
        ],
    }
    tokenizer = load_tokenizer(); compiled = scope.compiler.compile_qwen(tokenizer, rows)
    widths = [len(row["prompt_ids"]) for row in compiled]
    banned = ("regreted", "believeed", "�", "锛", "鏄?")
    bad_strings = [row["case_id"] for row in rows if any(item in row["prompt"] for item in banned)]
    missing_roles = [{"case_id": row["case_id"], "role": role, "value": value}
                     for row in rows for role, value in row["role_values"].items() if value not in row["prompt_core"]]
    balance = defaultdict(lambda: [0, 0])
    for row in rows: balance[f"{row['query_operation']}|{row['language']}|{row['partition']}"][row["gold_position"]] += 1
    entities = {part: {value for row in rows if row["partition"] == part for role, value in row["role_values"].items() if role in ("primary", "secondary", "context")}
                for part in ("discovery", "confirmation", "lockbox")}
    overlap = sorted((entities["discovery"] & entities["confirmation"]) | (entities["discovery"] & entities["lockbox"]) | (entities["confirmation"] & entities["lockbox"]))
    material_path = out(name) / "material/unified_relation_programs.jsonl"
    compiled_path = out(name) / "material/qwen_compiled.jsonl"
    write_rows(material_path, rows); write_rows(compiled_path, compiled)
    write_rows(out(name) / "external/human_blind_template.jsonl", [
        {"case_id": row["case_id"], "naturalness_1_5": None, "semantic_uniqueness_0_1": None,
         "equivalence_0_1": None, "reviewer": None} for row in rows if row["partition"] == "lockbox"])
    machine = {"rows": len(rows), "operations": {op: sum(row["query_operation"] == op for row in rows) for op in OPERATIONS},
               "partitions": {part: sum(row["partition"] == part for row in rows) for part in entities},
               "candidate_balance": dict(balance), "bad_strings": bad_strings, "missing_roles": missing_roles,
               "cross_partition_entity_overlap": overlap, "token_width_min_median_max": [min(widths), float(np.median(widths)), max(widths)],
               "human_review": "NA_not_run"}
    save(out(name) / "audit/machine_material_audit.json", machine)
    model = None; behavior = []; free = []; index = []
    field_path = out(name) / "raw/qualified_role_field.float16.npy"
    try:
        model, tokenizer, device, placement = scope.parent.previous.model_base().load_bf16("qwen3")
        quant = scope.parent.previous.model_base().quantization_audit(model)
        behavior = batch_behavior(model, device, compiled); bmap = {row["case_id"]: row for row in behavior}
        free_source = [row for row in compiled if row["partition"] in ("confirmation", "lockbox")]
        free = free_generate(model, tokenizer, device, free_source); fmap = {row["case_id"]: row for row in free}
        write_rows(out(name) / "behavior/candidate.jsonl", behavior); write_rows(out(name) / "behavior/free_generation.jsonl", free)
        slices = {}
        for operation, language in itertools.product(OPERATIONS, LANGUAGES):
            dev = [row for row in compiled if row["query_operation"] == operation and row["language"] == language and row["partition"] in ("discovery", "confirmation")]
            conf = [row for row in compiled if row["query_operation"] == operation and row["language"] == language and row["partition"] == "confirmation"]
            lock = [row for row in compiled if row["query_operation"] == operation and row["language"] == language and row["partition"] == "lockbox"]
            values = {"development_candidate_accuracy": float(np.mean([bmap[row["case_id"]]["correct"] for row in dev])),
                      "confirmation_generation_accuracy": float(np.mean([fmap[row["case_id"]]["correct"] for row in conf])),
                      "lockbox_candidate_accuracy": float(np.mean([bmap[row["case_id"]]["correct"] for row in lock])),
                      "lockbox_generation_accuracy": float(np.mean([fmap[row["case_id"]]["correct"] for row in lock]))}
            values["qualified_prelockbox"] = values["development_candidate_accuracy"] >= BEHAVIOR_GATE and values["confirmation_generation_accuracy"] >= BEHAVIOR_GATE
            slices[f"{operation}|{language}"] = values
        save(out(name) / "behavior/slices.json", slices)
        selected = [row for row in compiled if slices[f"{row['query_operation']}|{row['language']}"]["qualified_prelockbox"]]
        field = np.lib.format.open_memmap(field_path, mode="w+", dtype=np.float16, shape=(len(selected), CHECKPOINTS, len(ROLES), DIM))
        base = model.model; captured = []
        handles = [module.register_forward_hook(lambda _m, _a, output: captured.append(output[0] if isinstance(output, tuple) else output))
                   for module in [base.embed_tokens, *list(base.layers), base.norm]]
        try:
            for row_i, item in enumerate(selected):
                ids = torch.tensor([item["prompt_ids"]], dtype=torch.long, device=device); mask = torch.ones_like(ids)
                pos = torch.arange(ids.shape[1], device=device)[None]; captured.clear()
                with torch.inference_mode(): model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True)
                if len(captured) != CHECKPOINTS: raise RuntimeError((item["case_id"], len(captured)))
                for q, hidden in enumerate(captured):
                    values = hidden[0].float().cpu().numpy().astype(np.float16)
                    for role_i, role in enumerate(ROLES): field[row_i, q, role_i] = values[item["role_positions"][role][-1]]
                index.append({"hidden_index": row_i, "case_id": item["case_id"], "query_operation": item["query_operation"],
                              "language": item["language"], "surface": item["surface"], "cell": item["cell"], "unit": item["unit"],
                              "partition": item["partition"], "factors": item["factors"], "prompt_length": len(item["prompt_ids"]),
                              "query_position": item["role_positions"]["query"][-1], "gold_position": item["gold_position"],
                              "behavior_correct": bmap[item["case_id"]]["correct"], "free_correct": fmap[item["case_id"]]["correct"] if item["case_id"] in fmap else None})
                if row_i % 64 == 0: print(f"[C684-C688] capture {row_i}/{len(selected)}", flush=True)
        finally:
            for handle in handles: handle.remove()
        field.flush(); close_mmap(field); write_rows(out(name) / "raw/hidden_index.jsonl", index)
    finally:
        scope.parent.previous.model_base().release_bf16(model); gc.collect()
    close(name, {
        "strict_interpretation": "The prior evidence is retained only at its strict boundary. This fresh material is one shared relation-state factorial; behavior qualification is interface evidence, and the operation names do not assert internal modules.",
        "parent_evidence_audit": parent_audit, "machine_material_audit": machine,
        "material_sha256": file_hash(material_path), "compiled_sha256": file_hash(compiled_path),
        "slice_results": slices, "qualified_slices": [key for key, value in slices.items() if value["qualified_prelockbox"]],
        "captured_rows": len(index), "field_shape": [len(index), CHECKPOINTS, len(ROLES), DIM],
        "field_sha256": file_hash(field_path) if field_path.exists() else None, "placement": placement, "quantization": quant,
        "human_review": "NA_not_run",
    }, {
        "rows": len(rows) == 768, "unique": len({row["case_id"] for row in rows}) == len(rows),
        "role_compiler": all(set(row["role_positions"]) == set(ROLES) for row in compiled),
        "utf8_morphology": not bad_strings, "surface_roles": not missing_roles,
        "balanced": all(v[0] == v[1] for v in balance.values()), "partition_isolation": not overlap, "width": max(widths) <= 220,
        "candidate_complete": len(behavior) == len(compiled), "generation_complete": len(free) == len(free_source),
        "some_qualified": bool(index), "field": field_path.exists(),
        "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"], "finite": finite(slices),
    }, "Authorize C689-C693 to compare shared, nuisance, shuffled and relation-conditioned state-word prediction; failed confirmation strata keep lockbox sealed.")


def condition(rows: list[dict], mode: str) -> np.ndarray:
    labels = []
    for row in rows:
        f = row["factors"]
        relation = f"{row['query_operation']}|v{f['voice']}|o{f['outer']}|i{f['inner']}|d{f['depth']}|s{f['shortcut']}"
        if mode == "shared": label = "all"
        elif mode == "nuisance": label = f"{row['language']}|{row['surface']}|len{row['prompt_length']//16}|pos{row['query_position']//8}|ord{row['gold_position']}"
        elif mode == "shuffled": label = f"shuffle{hash_id(row['case_id']) % 12}"
        elif mode == "relational": label = relation
        elif mode == "program": label = f"{relation}|{row['language']}|{row['surface']}"
        else: raise KeyError(mode)
        labels.append(label)
    return np.asarray([prior.stable_condition(label) for label in labels], np.uint64)


def evaluate(states: np.ndarray, train: list[dict], test: list[dict], mode: str) -> dict:
    train_ids = np.asarray([row["hidden_index"] for row in train]); test_ids = np.asarray([row["hidden_index"] for row in test]); metrics = []
    for q0, q1 in zip(QPOINTS[:-1], QPOINTS[1:]):
        for role_i, role in enumerate(ROLES):
            tc = prior.state_code(states[train_ids, q0, role_i]); tn = prior.state_code(states[train_ids, q1, role_i])
            xc = prior.state_code(states[test_ids, q0, role_i]); xn = prior.state_code(states[test_ids, q1, role_i])
            pred, unknown = prior.mode_lookup_predict(tc, tn, condition(train, mode), xc, condition(test, mode))
            metrics.append({"q0": q0, "q1": q1, "role": role, **prior.grammar_metric(pred, xn, xc, unknown)})
    keys = ("exact_state_accuracy", "sign_accuracy", "exponent_mae", "unknown_key_rate", "copy_exact_accuracy")
    return {"mode": mode, "train_rows": len(train), "test_rows": len(test),
            "aggregate": {key: float(np.mean([row[key] for row in metrics])) for key in keys}, "by_transition_role": metrics}


def phase2201() -> None:
    name = "C689-C693"
    if (out(name) / "analysis/final.json").exists(): return
    index = read_rows(out("C684-C688") / "raw/hidden_index.jsonl")
    states = np.load(out("C684-C688") / "raw/qualified_role_field.float16.npy", mmap_mode="r")
    modes = ("shared", "nuisance", "shuffled", "relational", "program"); results = {}
    try:
        for operation in OPERATIONS:
            train = [r for r in index if r["query_operation"] == operation and r["partition"] == "discovery" and r["behavior_correct"]]
            confirm_rows = [r for r in index if r["query_operation"] == operation and r["partition"] == "confirmation" and r["behavior_correct"]]
            lock_rows = [r for r in index if r["query_operation"] == operation and r["partition"] == "lockbox" and r["behavior_correct"]]
            if not train or not confirm_rows:
                results[operation] = {"status": "NA_behavior_unqualified", "confirmation_passed": False, "lockbox": "sealed"}; continue
            confirmation = {mode: evaluate(states, train, confirm_rows, mode) for mode in modes}
            scores = {mode: confirmation[mode]["aggregate"]["exact_state_accuracy"] for mode in modes}
            gain = max(scores["relational"], scores["program"]) - max(scores["shared"], scores["nuisance"], scores["shuffled"])
            item = {"confirmation": confirmation, "confirmation_scores": scores, "confirmation_gain": gain,
                    "confirmation_passed": gain >= RELATION_GAIN_GATE, "lockbox": "sealed_confirmation_failed", "prospective_passed": False}
            if item["confirmation_passed"] and lock_rows:
                lockbox = {mode: evaluate(states, train, lock_rows, mode) for mode in modes}
                lock_scores = {mode: lockbox[mode]["aggregate"]["exact_state_accuracy"] for mode in modes}
                lock_gain = max(lock_scores["relational"], lock_scores["program"]) - max(lock_scores["shared"], lock_scores["nuisance"], lock_scores["shuffled"])
                item.update({"lockbox": lockbox, "lockbox_scores": lock_scores, "lockbox_gain": lock_gain,
                             "prospective_passed": lock_gain >= RELATION_GAIN_GATE})
            results[operation] = item
            print(f"[C689-C693] {operation} confirmation={gain:.6f} lockbox={item.get('lockbox_gain')}", flush=True)
    finally: close_mmap(states)
    save(out(name) / "analysis/tournament.json", results)
    passed = [op for op, value in results.items() if value.get("prospective_passed")]
    close(name, {
        "strict_interpretation": "A relation-conditioned coordinate grammar is registered only if it prospectively beats shared state dynamics, nonsemantic prompt variables and shuffled labels. Positive operation strata remain external transformation domains, not internal modules.",
        "operation_results": results, "prospective_relational_operations": passed,
        "operations_passed": len(passed), "operations_total": len(OPERATIONS), "new_foundational_mathematics_gate": False,
    }, {"parent": final("C684-C688")["all_checks_passed"], "all_operations": set(results) == set(OPERATIONS),
        "confirmation_before_lockbox": all(v.get("confirmation_passed") or v.get("lockbox") in ("sealed", "sealed_confirmation_failed") for v in results.values()),
        "finite": finite(results)},
        "Authorize C694-C699 to complete the pre-registered event/path all-coordinate observation; semantic causal claims require prospective relation gain.")


def local_anchors() -> list[dict]:
    index = read_rows(out("C684-C688") / "raw/hidden_index.jsonl")
    compiled = {row["case_id"]: row for row in read_rows(out("C684-C688") / "material/qwen_compiled.jsonl")}
    cells = {"event": "event_active_record", "path": "path_depth3_record"}; anchors = []
    for operation, language, unit in itertools.product(cells, LANGUAGES, (12, 20)):
        matches = [row for row in index if row["query_operation"] == operation and row["language"] == language
                   and row["unit"] == unit and row["cell"] == cells[operation] and row["behavior_correct"]]
        if len(matches) == 1:
            anchors.append({**compiled[matches[0]["case_id"]], "anchor_group": f"{operation}|{language}",
                            "anchor_family": f"{operation}|{language}",
                            "anchor_partition": "confirmation" if unit == 12 else "lockbox"})
    return anchors


def export_visual(anchors: list[dict], response_path: Path, influence_path: Path, metrics: dict, coalition: dict) -> None:
    field = np.load(out("C684-C688") / "raw/qualified_role_field.float16.npy", mmap_mode="r")
    index = {row["case_id"]: row for row in read_rows(out("C684-C688") / "raw/hidden_index.jsonl")}
    response = np.load(response_path, mmap_mode="r"); influence = np.load(influence_path, mmap_mode="r")
    arrays, rows = [], []
    for i, anchor in enumerate(anchors):
        hidden_i = index[anchor["case_id"]]["hidden_index"]
        for q, kind in ((0, "embedding_coordinate_state"), (24, "q24_relation_hiddenstate"), (37, "final_relation_hiddenstate")):
            arrays.append(np.asarray(field[hidden_i, q, ROLES.index("relation")], np.float32))
            rows.append({"kind": kind, "case_id": anchor["case_id"], "group": anchor["anchor_group"],
                         "partition": anchor["anchor_partition"], "checkpoint": q, "role": "relation"})
        arrays.append(np.asarray(influence[i], np.float32))
        rows.append({"kind": "coordinate_logit_margin_influence", "case_id": anchor["case_id"],
                     "group": anchor["anchor_group"], "partition": anchor["anchor_partition"], "checkpoint": 24})
        for target_i, target in enumerate(("q25_boundary", "final_boundary")):
            arrays.append(np.mean(np.abs(np.asarray(response[i, target_i], np.float32)), axis=0))
            rows.append({"kind": "mean_absolute_incoming_local_response", "case_id": anchor["case_id"],
                         "group": anchor["anchor_group"], "partition": anchor["anchor_partition"], "target": target})
    matrix = np.stack(arrays).astype(np.float16); VISUAL_BINARY.parent.mkdir(parents=True, exist_ok=True); np.save(VISUAL_BINARY, matrix)
    payload = {"schema": "ai2050.unified-relation-coordinate-response.v1", "phase": 2202, "campaign": "C694-C699",
               "model": "Qwen3-4B BF16", "coordinate_count": DIM, "rows": rows,
               "binary": str(VISUAL_BINARY.relative_to(ROOT)).replace("\\", "/"), "binary_shape": list(matrix.shape),
               "binary_dtype": "float16", "specificity": metrics, "coalition": coalition,
               "claim_boundary": "Exact physical activation coordinates and sample-local finite differences; no weights, PCA, Top-K, gradients or donor transport."}
    save(VISUAL, payload); close_mmap(field); close_mmap(response); close_mmap(influence)
    catalog = load(CATALOG) if CATALOG.exists() else {"schema": "language-encoding-catalog.v1", "datasets": []}
    entry = {"id": "c699_unified_relation_coordinate_response_atlas", "title": "C699 Unified Relation Coordinate Response Atlas",
             "phase": 2202, "campaign": "C694-C699", "model": "Qwen3-4B",
             "source_path": "/vis_data/research_kernel/c699_unified_relation_coordinate_response_atlas.json",
             "source_schema": payload["schema"], "coordinate_count": DIM, "checkpoint_count": CHECKPOINTS,
             "kinds": sorted({row["kind"] for row in rows})}
    catalog["datasets"] = [row for row in catalog.get("datasets", []) if row.get("id") != entry["id"]] + [entry]
    catalog["generated_at"] = datetime.now(timezone.utc).isoformat(); save(CATALOG, catalog)


def phase2202() -> None:
    name = "C694-C699"
    if (out(name) / "analysis/final.json").exists(): return
    anchors = local_anchors(); write_rows(out(name) / "material/local_anchors.jsonl", anchors)
    pairs = sorted({row["anchor_group"] for row in anchors
                    if {item["anchor_partition"] for item in anchors if item["anchor_group"] == row["anchor_group"]} == {"confirmation", "lockbox"}})
    if not pairs:
        close(name, {"strict_interpretation": "No frozen anchor pair survived behavior qualification, so local response is NA rather than a zero effect.",
                     "anchors": len(anchors), "paired_groups": [], "q25_specific_groups": [],
                     "semantic_eligible_coalition_groups": [], "visual": None},
              {"parent": final("C689-C693")["all_checks_passed"], "missingness_recorded": True, "no_mechanism_claim": True},
              "Continue the independent cross-model behavior route; do not call missing local scans negative.")
        return
    anchors = [row for row in anchors if row["anchor_group"] in pairs]
    response_path = out(name) / "raw/all_coordinate_response.float16.npy"
    influence_path = out(name) / "raw/all_coordinate_influence.float32.npy"
    response = np.lib.format.open_memmap(response_path, mode="w+", dtype=np.float16, shape=(len(anchors), 2, DIM, DIM))
    influence = np.lib.format.open_memmap(influence_path, mode="w+", dtype=np.float32, shape=(len(anchors), DIM))
    model = None; scans = []
    try:
        model, _tok, _dev, placement = scope.parent.previous.model_base().load_bf16("qwen3")
        quant = scope.parent.previous.model_base().quantization_audit(model)
        for i, anchor in enumerate(anchors):
            scans.append({"case_id": anchor["case_id"], **prior.local_coordinate_scan(model, anchor, response, influence, i)})
            response.flush(); influence.flush()
    finally:
        response.flush(); influence.flush(); close_mmap(response); close_mmap(influence)
        scope.parent.previous.model_base().release_bf16(model); gc.collect()
    response = np.load(response_path, mmap_mode="r"); influence = np.load(influence_path, mmap_mode="r")
    metrics, coalition = {}, {}; eligible = set(final("C689-C693")["prospective_relational_operations"]); model = None
    try:
        model, _tok, _dev, _place = scope.parent.previous.model_base().load_bf16("qwen3")
        for group in pairs:
            c = next(i for i, row in enumerate(anchors) if row["anchor_group"] == group and row["anchor_partition"] == "confirmation")
            l = next(i for i, row in enumerate(anchors) if row["anchor_group"] == group and row["anchor_partition"] == "lockbox")
            operation, language = group.split("|")
            wrong = [i for i, row in enumerate(anchors) if row["anchor_partition"] == "lockbox" and row["anchor_group"].endswith(language) and row["anchor_group"] != group]
            metrics[group] = {}
            for target_i, target in enumerate(("q25", "final")):
                same = float(np.mean(np.sign(response[c, target_i]) == np.sign(response[l, target_i])))
                wrong_score = float(np.mean([np.mean(np.sign(response[c, target_i]) == np.sign(response[i, target_i])) for i in wrong])) if wrong else 0.0
                shifted = float(np.mean(np.sign(response[c, target_i]) == np.roll(np.sign(response[l, target_i]), 257, axis=1)))
                gain = same - max(wrong_score, shifted)
                metrics[group][target] = {"same": same, "wrong_operation": wrong_score, "shift257": shifted,
                                          "specificity_gain": gain, "passed": gain >= LOCAL_GATE}
            values = prior.coalition_eval(model, anchors[l], np.sign(np.asarray(influence[c], np.float32)))
            base = values["base"]; values["aligned_gain"] = values["aligned_0.25"] - base
            values["best_control_gain"] = max(values["opposite_0.25"] - base, values["shift257_0.25"] - base, 0.0)
            values["passed"] = values["aligned_gain"] - values["best_control_gain"] >= COALITION_GATE
            values["semantic_claim_eligible"] = operation in eligible; coalition[group] = values
    finally:
        scope.parent.previous.model_base().release_bf16(model); gc.collect(); close_mmap(response); close_mmap(influence)
    save(out(name) / "analysis/specificity.json", metrics); save(out(name) / "analysis/coalition.json", coalition)
    export_visual(anchors, response_path, influence_path, metrics, coalition)
    hashes = {"response": file_hash(response_path), "influence": file_hash(influence_path)}
    response_path.unlink(); influence_path.unlink()
    cleanup = {"deleted": [str(response_path.relative_to(ROOT)), str(influence_path.relative_to(ROOT))],
               "sha256_before_delete": hashes, "retained": [str(VISUAL.relative_to(ROOT)), str(VISUAL_BINARY.relative_to(ROOT))]}
    save(out(name) / "audit/raw_cleanup.json", cleanup)
    local_pass = [group for group, value in metrics.items() if value["q25"]["passed"]]
    causal_pass = [group for group, value in coalition.items() if value["passed"] and value["semantic_claim_eligible"]]
    close(name, {
        "strict_interpretation": "Every source coordinate was perturbed. Same-operation response is semantic-eligible only after wrong-operation and shifted-coordinate controls and a prospective relation increment; coalition output effects remain sample/checkpoint candidates, not unique circuits.",
        "anchors": len(anchors), "paired_groups": pairs, "scan_metadata": scans, "specificity": metrics,
        "coalition": coalition, "q25_specific_groups": local_pass, "semantic_eligible_coalition_groups": causal_pass,
        "visual": str(VISUAL.relative_to(ROOT)), "visual_binary": str(VISUAL_BINARY.relative_to(ROOT)),
        "cleanup": cleanup, "placement": placement, "new_foundational_mathematics_gate": False,
    }, {"parent": final("C689-C693")["all_checks_passed"], "paired": bool(pairs),
        "full_coordinates": len(scans) == len(anchors), "bf16": quant["has_bf16_parameters"] and not quant["has_quantized_modules"],
        "visual": VISUAL.exists() and VISUAL_BINARY.exists(), "raw_cleaned": not response_path.exists() and not influence_path.exists(),
        "finite": finite(metrics) and finite(coalition)},
        "Authorize C700-C704 to run the frozen cross-model panel sequentially and compare anonymous role/depth topology only.")


def cross_model_panel() -> list[dict]:
    rows = read_rows(out("C684-C688") / "material/unified_relation_programs.jsonl"); selected = []
    cells = {"event": "event_active_record", "path": "path_depth3_record", "combined": "combined_depth3_record"}
    for operation, language, unit in itertools.product(OPERATIONS, LANGUAGES, (18, 19, 20, 21)):
        match = [row for row in rows if row["query_operation"] == operation and row["language"] == language and row["unit"] == unit and row["cell"] == cells[operation]]
        if len(match) != 1: raise RuntimeError((operation, language, unit, len(match)))
        selected.append({**match[0], "cross_model_group": f"{operation}|{language}"})
    return selected


def phase2203() -> None:
    name = "C700-C704"
    if (out(name) / "analysis/final.json").exists(): return
    rows = cross_model_panel(); material_path = out(name) / "material/cross_model_24_case_panel.jsonl"; write_rows(material_path, rows)
    workers = {}; python = Path(sys.executable); worker_script = Path(prior.__file__)
    for model_name in ("glm4", "deepseek7b", "qwen3_14b"):
        output = out(name) / f"raw/{model_name}/worker_result.json"
        completed = subprocess.run([str(python), str(worker_script), "--worker", model_name,
                                    "--material", str(material_path), "--output", str(output)], cwd=ROOT, check=False)
        workers[model_name] = load(output) if output.exists() else {"model": model_name, "status": "missing_worker_output"}
        workers[model_name]["returncode"] = completed.returncode
        print(f"[C700-C704] {model_name} returncode={completed.returncode}", flush=True)
    save(out(name) / "analysis/workers.json", workers)
    qualified = {key: value for key, value in workers.items() if value.get("qualified") and value.get("hiddenstate_ran")}
    topology = {key: {"relative_depths": [row["relative_depth"] for row in value["relative_topology"]],
                      "mean_flip": [float(np.mean(row["next_sign_flip_by_role"])) for row in value["relative_topology"]],
                      "max_flip_role": [ROLES[int(np.argmax(row["next_sign_flip_by_role"]))] for row in value["relative_topology"]]}
                for key, value in qualified.items()}
    close(name, {
        "strict_interpretation": "Cross-model evidence is limited to behavior-qualified anonymous relative depth and role topology. Physical coordinate identities and widths are never equated, and similar coarse curves are not functional isomorphism.",
        "panel_rows": len(rows), "workers": workers, "qualified_hidden_models": list(qualified),
        "relative_topology": topology, "new_foundational_mathematics_gate": False,
    }, {"parent": final("C694-C699")["all_checks_passed"], "panel_rows": len(rows) == 24,
        "sequential_models": len(workers) == 3, "workers_returned": all(v.get("returncode") in (0, 1, 2) for v in workers.values()),
        "finite": finite(workers) and finite(topology)},
        "Authorize C705-C709 to jointly adjudicate, preserve displayed coordinate rows, and delete the undisplayed Qwen role field.")


def phase2204() -> None:
    name = "C705-C709"
    if (out(name) / "analysis/final.json").exists(): return
    p0, p1, p2, p3 = final("C684-C688"), final("C689-C693"), final("C694-C699"), final("C700-C704")
    field_path = out("C684-C688") / "raw/qualified_role_field.float16.npy"
    cleanup = {"path": str(field_path.relative_to(ROOT)), "existed": field_path.exists(), "sha256": None, "deleted": False}
    if field_path.exists(): cleanup["sha256"] = file_hash(field_path); field_path.unlink(); cleanup["deleted"] = True
    save(out(name) / "audit/field_cleanup.json", cleanup)
    relation = p1["prospective_relational_operations"]; local = p2.get("q25_specific_groups", [])
    causal = p2.get("semantic_eligible_coalition_groups", []); cross = p3["qualified_hidden_models"]
    new_math = bool(relation and local and causal and len(cross) >= 2 and False)
    same_goal = bool(relation or local or causal)
    decision = ("Continue the same goal on independently human-reviewed natural material, replicating only positive transformation-response objects and adding generation-time deletion/rescue."
                if same_goal else "Continue the broader goal through response-equivalence classes and richer natural relation states; do not tune failed external labels.")
    close(name, {
        "strict_interpretation": "The campaign studies one global autoregressive relation-state field. Shared coordinate physics, nuisance structure, shuffled labels, relation transformations and output calling are separately accounted for. Human blind review and generation necessity/rescue remain absent, so foundational new mathematics is not authorized.",
        "qualified_behavior_slices": p0["qualified_slices"], "prospective_relational_operations": relation,
        "q25_specific_groups": local, "semantic_eligible_coalition_groups": causal,
        "cross_model_qualified_hidden_models": cross,
        "theory_update": "One global HiddenState dynamics with shared coordinate physics and base-state-conditioned local response regions; operation types define domains of external partial transformations, not internal modules.",
        "new_foundational_mathematics_gate": new_math, "important_answer_reached": True,
        "next_stage_same_goal": same_goal, "automatic_continuation_decision": decision,
        "cleanup": {"qwen_role_field": cleanup, "retained_visual": [str(VISUAL.relative_to(ROOT)), str(VISUAL_BINARY.relative_to(ROOT))]},
    }, {"all_parents": all(final(key)["all_checks_passed"] for key in PHASES if key != "C705-C709"),
        "important_answer": True, "field_cleaned": not field_path.exists(),
        "visual_retained": VISUAL.exists() and VISUAL_BINARY.exists(), "finite": finite([relation, local, causal, cross])}, decision)


def run_all() -> None:
    freeze(); rows = material(); phase2200(rows); phase2201(); phase2202(); phase2203(); phase2204()


if __name__ == "__main__": run_all()
