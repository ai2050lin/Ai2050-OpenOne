#!/usr/bin/env python3
"""Audit Phase 2368-2377 claims and freeze a label-free natural binding campaign."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
OUT = TESTS / "result/phase2378_c15601_c15920_label_free_binding_contract"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
MATERIAL = OUT / "material/label_free_natural_binding.jsonl"
PHASE = 2378
CAMPAIGN = "C15601-C15920"
ATTACHMENTS = (
    Path(r"C:\Users\Admin\.codex\attachments\ce31b180-142e-47f6-84fe-6720284712d4\pasted-text.txt"),
    Path(r"C:\Users\Admin\.codex\attachments\973e186d-13fa-40ad-8f86-ac6714627a3f\pasted-text.txt"),
)
FAMILIES = (
    "temporal_narrative", "causal_process", "taxonomy_chain", "spatial_route",
    "procedure", "comparison", "dialogue_coreference", "argument_structure",
)
LANGUAGES = ("en", "zh")
SURFACES = ("distinct_opening", "shared_opening")
SOURCE_PERMS = (
    (0, 1, 2, 3), (3, 2, 1, 0), (1, 3, 0, 2),
    (2, 0, 3, 1), (1, 0, 3, 2), (2, 3, 0, 1),
)
NAMES = ("Mira", "Jonas", "Lena", "Omar", "Priya", "Tomas", "Nadia", "Felix",
         "Asha", "Ruben", "Inez", "Caleb", "Mei", "Arun", "Sara", "Noah")
TOPICS = ("wetland survey", "orchard review", "harbor inspection", "museum audit",
          "forest census", "clinic trial", "bridge survey", "archive review")


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def partition(unit: int, source_index: int) -> str:
    if unit <= 3 and source_index <= 2:
        return "discovery"
    if unit in (4, 5) and source_index == 3:
        return "confirmation"
    if unit in (6, 7) and source_index in (4, 5):
        return "fresh_joint_lockbox"
    return "atlas_only"


def family_bodies(family: str, unit: int, language: str) -> tuple[list[str], list[str]]:
    """Return four naturally ordered sentence bodies and independent paraphrases."""
    a, b, c, d = [NAMES[(unit * 3 + j) % len(NAMES)] for j in range(4)]
    topic = TOPICS[unit]
    value = 11 + unit * 3
    if language == "en":
        data = {
            "temporal_narrative": [
                f"before sunrise, {a} opened the {topic} and photographed the sealed gate",
                f"near midday, {b} measured the eastern channel after the gate inspection",
                f"later that afternoon, {c} compared the channel notes with the field map, adding a careful margin note",
                f"after sunset, {d} filed the signed report once every earlier observation was checked",
            ],
            "causal_process": [
                f"a blocked intake first reduced the flow through the {topic}",
                f"once {a} cleared the intake, pressure returned to the narrow pipe",
                f"the restored pressure then moved water into {b}'s sampling chamber, where the indicator changed color",
                f"because the indicator changed, {c} approved the final reading for {d}",
            ],
            "taxonomy_chain": [
                f"the catalog begins with a physical object as the broadest class used in the {topic}",
                f"within that class, a crafted artifact is the narrower class selected by {a}",
                f"within crafted artifacts, a measuring instrument is the more specific group examined by {b}, despite an older label",
                f"within measuring instruments, the brass gauge is the particular item entered by {c}",
            ],
            "spatial_route": [
                f"at the entrance to the {topic}, {a} passed the stone arch",
                f"beyond the arch, {b} crossed the narrow bridge beside the reeds",
                f"past that bridge, {c} followed the curved path around the old orchard, keeping the river on the left",
                f"at the end of the curved path, {d} reached the observation tower",
            ],
            "procedure": [
                f"before any mixing, {a} rinsed the glass vessel used for the {topic}",
                f"after the vessel dried, {b} combined the clear solution with the blue reagent",
                f"once the liquids were combined, {c} warmed the mixture until a thin silver line appeared, without boiling it",
                f"after warming, {d} sealed the sample and entered it in the register",
            ],
            "comparison": [
                f"{a}'s first sample measured {value} units in the {topic}",
                f"{b}'s second sample measured {value + 9} units under the same conditions",
                f"{c}'s carefully repeated sample measured {value + 23} units after the instrument was recalibrated",
                f"{d}'s final sample measured {value + 41} units under the same conditions",
            ],
            "dialogue_coreference": [
                f"{a} introduced the proposal during the {topic} and handed a red folder to {b}",
                f"the recipient of that folder asked {c} to check its central estimate",
                f"the analyst just asked to check the estimate found a missing assumption and told {d} about it in private",
                f"the person who heard about that omission returned the corrected folder to {a}",
            ],
            "argument_structure": [
                f"the {topic} report opens by claiming that the northern route is safer",
                f"it next supports that claim with {a}'s measurements from three dry mornings",
                f"the report then acknowledges {b}'s objection that winter ice could reverse the result, a limitation the measurements did not cover",
                f"after weighing the claim, evidence, and objection, it concludes that a seasonal trial is still required",
            ],
        }
        para = [
            text.replace("before sunrise", "prior to dawn").replace("near midday", "around noon")
                .replace("later that afternoon", "during the later afternoon").replace("after sunset", "once night had fallen")
                .replace("first", "initially").replace("then", "subsequently").replace("after", "following")
            for text in data[family]
        ]
    else:
        zh_topic = ("湿地调查", "果园复核", "港口检查", "博物馆审计", "森林普查", "诊所试验", "桥梁勘测", "档案复核")[unit]
        data = {
            "temporal_narrative": [
                f"日出前，{a}开启{zh_topic}并拍摄了封闭的大门",
                f"接近中午时，{b}在检查大门后测量了东侧水道",
                f"当天下午稍晚，{c}把水道记录与现场地图比较，并认真补写了一条边注",
                f"日落以后，{d}在核对全部早先观察后提交了签字报告",
            ],
            "causal_process": [
                f"堵塞的入口最初降低了{zh_topic}中的流量",
                f"{a}清理入口以后，狭窄管道的压力恢复了",
                f"恢复的压力随后把水送进{b}的取样室，里面的指示剂改变了颜色",
                f"因为指示剂变色，{c}为{d}批准了最终读数",
            ],
            "taxonomy_chain": [
                f"目录先以物理物体作为{zh_topic}使用的最宽类别",
                f"在该类别之内，人工制品是{a}选择的较窄类别",
                f"在人工制品之内，测量仪器是{b}检查的更具体类别，尽管旧标签使用了另一种写法",
                f"在测量仪器之内，黄铜压力表是{c}登记的具体项目",
            ],
            "spatial_route": [
                f"在{zh_topic}入口处，{a}穿过石拱门",
                f"越过拱门以后，{b}走过芦苇旁的窄桥",
                f"经过窄桥以后，{c}沿弯路绕过旧果园，并一直让河流位于左侧",
                f"在弯路尽头，{d}到达观测塔",
            ],
            "procedure": [
                f"开始混合以前，{a}清洗了{zh_topic}使用的玻璃容器",
                f"容器干燥以后，{b}把透明溶液与蓝色试剂混合",
                f"两种液体混合以后，{c}加热混合物直到出现细银线，但没有让它沸腾",
                f"加热结束以后，{d}密封样品并把它登记入册",
            ],
            "comparison": [
                f"{a}的第一份样品在{zh_topic}中测得{value}单位",
                f"{b}的第二份样品在相同条件下测得{value + 9}单位",
                f"仪器重新校准以后，{c}认真复测的样品测得{value + 23}单位",
                f"{d}的最终样品在相同条件下测得{value + 41}单位",
            ],
            "dialogue_coreference": [
                f"{a}在{zh_topic}时提出方案，并把红色文件夹交给{b}",
                f"收到文件夹的人请{c}核对其中的核心估计",
                f"刚被要求核对估计的分析者发现一项遗漏假设，并私下把它告诉{d}",
                f"听到这项遗漏的人把改正后的文件夹还给{a}",
            ],
            "argument_structure": [
                f"{zh_topic}报告开头主张北侧路线更安全",
                f"报告接着用{a}在三个干燥早晨取得的测量支持该主张",
                f"随后报告承认{b}的反对意见：冬季结冰可能逆转结果，而现有测量没有覆盖这一限制",
                f"权衡主张、证据和反对意见后，报告总结仍需进行季节性试验",
            ],
        }
        para = [text.replace("日出前", "黎明以前").replace("接近中午时", "中午前后")
                .replace("随后", "接下来").replace("以后", "之后").replace("最初", "起初") for text in data[family]]
    return data[family], para


def instruction(family: str, language: str, reverse: bool, task: str) -> str:
    direction_en = {
        "temporal_narrative": ("earliest to latest", "latest to earliest"),
        "causal_process": ("cause to final consequence", "final consequence back to its cause"),
        "taxonomy_chain": ("broadest category to most specific item", "most specific item to broadest category"),
        "spatial_route": ("entrance to destination", "destination back to entrance"),
        "procedure": ("prerequisite to completed procedure", "completed procedure back to its prerequisite"),
        "comparison": ("smallest measurement to largest", "largest measurement to smallest"),
        "dialogue_coreference": ("first utterance to final response", "final response back to first utterance"),
        "argument_structure": ("opening claim to conclusion", "conclusion back to opening claim"),
    }[family][int(reverse)]
    direction_zh = {
        "temporal_narrative": ("从最早到最晚", "从最晚到最早"),
        "causal_process": ("从起因到最终结果", "从最终结果反推到起因"),
        "taxonomy_chain": ("从最宽类别到最具体项目", "从最具体项目到最宽类别"),
        "spatial_route": ("从入口到终点", "从终点返回入口"),
        "procedure": ("从前提步骤到完成步骤", "从完成步骤返回前提步骤"),
        "comparison": ("从最小测量值到最大测量值", "从最大测量值到最小测量值"),
        "dialogue_coreference": ("从首次发言到最后回应", "从最后回应回到首次发言"),
        "argument_structure": ("从开篇主张到结论", "从结论回到开篇主张"),
    }[family][int(reverse)]
    if language == "en":
        verb = "Restate every sentence without changing its facts" if task == "paraphrase" else "Preserve every sentence exactly"
        return f"Rewrite the paragraph from {direction_en}. {verb}; output only the four sentences, one per line."
    verb = "不改变事实地改写每个句子" if task == "paraphrase" else "逐字保留每个句子"
    return f"请按{direction_zh}重写这段文字。{verb}，只输出四个句子，每行一句。"


def compile_material(tokenizer) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    for family in FAMILIES:
        for unit in range(8):
            for language in LANGUAGES:
                bodies, paras = family_bodies(family, unit, language)
                for surface in SURFACES:
                    if language == "en":
                        sentences = [(("During the review, " if surface == "shared_opening" else "") + body + ".") for body in bodies]
                        paraphrases = [(("In the record, " if surface == "shared_opening" else "") + body + ".") for body in paras]
                        prefix = "Here is a shuffled natural paragraph:\n"
                        separator, bridge = "\n", "\n\n"
                    else:
                        sentences = [(("复核期间，" if surface == "shared_opening" else "") + body + "。") for body in bodies]
                        paraphrases = [(("记录显示，" if surface == "shared_opening" else "") + body + "。") for body in paras]
                        prefix = "下面是一段顺序被打乱的自然文字：\n"
                        separator, bridge = "\n", "\n\n"
                    for reverse in (False, True):
                        target_order = tuple(reversed(range(4))) if reverse else tuple(range(4))
                        for source_index, source_perm in enumerate(SOURCE_PERMS):
                            tasks = ("exact_copy", "paraphrase") if source_index == 0 else ("exact_copy",)
                            for task in tasks:
                                source_lines = [sentences[sid] for sid in source_perm]
                                prompt_text = prefix + separator.join(source_lines) + bridge + instruction(family, language, reverse, task) + "\n"
                                prompt_ids: list[int] = []
                                source_spans: list[list[int]] = []
                                prompt_ids += tokenizer.encode(prefix, add_special_tokens=False)
                                for slot, sentence in enumerate(source_lines):
                                    start = len(prompt_ids)
                                    prompt_ids += tokenizer.encode(sentence, add_special_tokens=False)
                                    source_spans.append([start, len(prompt_ids)])
                                    if slot < 3:
                                        prompt_ids += tokenizer.encode(separator, add_special_tokens=False)
                                prompt_ids += tokenizer.encode(bridge + instruction(family, language, reverse, task) + "\n", add_special_tokens=False)

                                target_sentences = paraphrases if task == "paraphrase" else sentences
                                target_ids: list[int] = []
                                target_spans: list[list[int]] = []
                                for slot, sid in enumerate(target_order):
                                    start = len(target_ids)
                                    target_ids += tokenizer.encode(target_sentences[sid], add_special_tokens=False)
                                    target_spans.append([start, len(target_ids)])
                                    if slot < 3:
                                        target_ids += tokenizer.encode(separator, add_special_tokens=False)
                                foil_order = target_order[1:] + target_order[:1]
                                foil_ids: list[int] = []
                                for slot, sid in enumerate(foil_order):
                                    foil_ids += tokenizer.encode(target_sentences[sid], add_special_tokens=False)
                                    if slot < 3:
                                        foil_ids += tokenizer.encode(separator, add_special_tokens=False)
                                rows.append({
                                    "case_id": f"c15601-{family}-u{unit}-{language}-{surface}-d{int(reverse)}-s{source_index}-{task}",
                                    "design_index": len(rows), "family": family, "unit": unit,
                                    "language": language, "surface": surface, "reverse": reverse,
                                    "source_index": source_index, "source_perm": list(source_perm),
                                    "target_order": list(target_order), "foil_order": list(foil_order),
                                    "partition": partition(unit, source_index), "task": task,
                                    "sentences": sentences, "paraphrases": paraphrases,
                                    "source_lines": source_lines, "target_sentences": [target_sentences[s] for s in target_order],
                                    "prompt": prompt_text, "prompt_ids": prompt_ids, "source_spans": source_spans,
                                    "target": separator.join(target_sentences[s] for s in target_order),
                                    "target_ids": target_ids, "target_spans": target_spans,
                                    "foil": separator.join(target_sentences[s] for s in foil_order), "foil_ids": foil_ids,
                                })
    exact = [row for row in rows if row["task"] == "exact_copy"]
    audit = {
        "rows": len(rows), "expected_rows": 3584, "exact_rows": len(exact), "expected_exact": 3072,
        "paraphrase_rows": len(rows) - len(exact), "families": dict(Counter(r["family"] for r in rows)),
        "languages": dict(Counter(r["language"] for r in rows)), "surfaces": dict(Counter(r["surface"] for r in rows)),
        "directions": dict(Counter(str(r["reverse"]) for r in rows)), "partitions": dict(Counter(r["partition"] for r in rows)),
        "unique_case_ids": len({r["case_id"] for r in rows}) == len(rows),
        "unique_prompt_target_pairs": len({(tuple(r["prompt_ids"]), tuple(r["target_ids"])) for r in rows}) == len(rows),
        "no_source_number_labels": all("Sentence 1" not in r["prompt"] and "句1" not in r["prompt"] for r in rows),
        "no_requested_marker_list": all("marker" not in r["prompt"].lower() and "标记" not in r["prompt"] for r in rows),
        "shared_opening_verified": all(len({s.split("，")[0] if r["language"] == "zh" else " ".join(s.split()[:3]) for s in r["sentences"]}) == 1
                                       for r in rows if r["surface"] == "shared_opening"),
        "unequal_sentence_lengths": all(len({len(tokenizer.encode(s, add_special_tokens=False)) for s in r["sentences"]}) >= 2 for r in rows),
        "prompt_token_range": [min(len(r["prompt_ids"]) for r in rows), max(len(r["prompt_ids"]) for r in rows)],
        "target_token_range": [min(len(r["target_ids"]) for r in rows), max(len(r["target_ids"]) for r in rows)],
    }
    return rows, audit


def evidence_audit() -> dict:
    return {
        "attachment_sha256": [{"path": str(path), "sha256": sha256(path)} for path in ATTACHMENTS],
        "retained": [
            "Phase2368-2377 validly separates target-order planning from sentence-content reconstruction.",
            "Behavior-qualified source-slot decodability in Qwen4B/Qwen14B/GLM4 is the strongest cross-model descriptive result.",
            "Qwen4B has a positive late-layer diagonal-affine S4/S5 response law under the tested explicit-marker materials.",
            "Group Fourier, OT, HOSVD, conditional information and H0 supplied bounded diagnostics, not a closed language mechanism.",
            "The next decisive confound is whether source-slot readability survives without numbering, nonce labels or requested marker lists.",
        ],
        "corrected_overclaims": [
            "91% source-slot decoding does not prove sentence-object packaging, an autonomous pointer, or content transport; explicit markers and indices were present.",
            "The tested high-mathematics estimators lost their lockboxes; this does not refute all group, topology, tensor or transport structures.",
            "A diagonal-affine activation response does not identify an MLP gating coefficient and does not prove scalar gating is the true mechanism.",
            "Attention-as-copy-machine, MLP-as-gate, key-value-memory truth, standing waves and diffeomorphic routing graphs are hypotheses absent component evidence.",
            "The 128-token copy result does not show that the model generally loses long content; truncation and prompt design remain confounds.",
            "Static HiddenState atlases remain necessary observational evidence; component attribution becomes justified only after a label-free signal passes.",
        ],
        "frozen_claim_ladder": {
            "behavior": "complete semantic order plus exact sentence/content preservation",
            "descriptive": "full-coordinate state decodes or matches source identity/slot on fresh joint lockbox",
            "dynamic_candidate": "attention or MLP component adds held-out identity/slot information above residual and lexical controls",
            "mechanism": "requires selective causal intervention and rescue; this campaign does not presuppose closure",
        },
    }


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 附件证据审查与无显式标签自然绑定总合同（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 逐项对照Phase2368–2377最终审计和两份附件，保留“顺序计划/内容重建分离、来源槽位可读、Qwen4B局部对角仿射响应”等有边界结果；否决把可解码性写成对象封装/自主指针、把对角仿射写成MLP门控真相、把Attention写成复制机的过度结论。新合同包含8个语言模式族×8个独立unit×中英×同首词/不同首词×正逆语义方向×6个来源排列，主任务3072条exact-copy，另有512条paraphrase；prompt取消`Sentence 1`、数字来源索引、nonce首标记和目标marker清单，并加入变长、重复开头和跨句共指。

$$x=(f,u,\ell,s,d,\sigma),\qquad \pi^*(x)=\operatorname{{semantic\_order}}(f,d),\qquad y=\bigoplus_{{k=1}}^4 S_{{\pi^*_k}}.$$

**结果汇总。** 附件审查 `{json.dumps(result['evidence_audit'], ensure_ascii=False)}`；材料预检 `{json.dumps(result['material_audit'], ensure_ascii=False)}`；检查 `{json.dumps(result['checks'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2378_c15601_c15920_label_free_binding_contract.py`；材料和审计位于 `tests/glm5/result/phase2378_c15601_c15920_label_free_binding_contract`。未修改其他Markdown。

**理论进展、问题硬伤与结论。** 这一步把“来源槽位高可读”从结论降回待检验现象，并冻结三层标签：样本内句对象、它在输入中的来源槽位、它在输出中的进度位置。自然名字仍可成为词汇匹配线索，因此后续必须同时做同首词、unit锁箱、source-permutation锁箱与配对错误源句负控。下一Phase采集全部物理激活坐标；只有无显式标签锁箱通过后，才允许定点分析Attention/MLP，不先验宣称残差路由机制。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main() -> None:
    final = OUT / "analysis/final.json"
    if final.exists():
        result = json.loads(final.read_text(encoding="utf-8")); append_memo(result)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return
    sys.path.insert(0, str(TESTS))
    import model_utils
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_utils.MODEL_CONFIGS["qwen3"]["path"], trust_remote_code=True,
                                               local_files_only=True, use_fast=False)
    rows, material_audit = compile_material(tokenizer)
    write_rows(MATERIAL, rows)
    audit = evidence_audit()
    checks = {
        "attachment_hashes_match": [x["sha256"] for x in audit["attachment_sha256"]] == [
            "d6e5342f81725294859ab4c77889ab1a1cfac50fbd9aee6eca20f7ea3b500c84",
            "0e084a84da8eb632b837fb4dd61bddbd516f19e78814c451d9a880fec40e55fb"],
        "material_count": material_audit["rows"] == material_audit["expected_rows"],
        "exact_count": material_audit["exact_rows"] == material_audit["expected_exact"],
        "unique": material_audit["unique_case_ids"] and material_audit["unique_prompt_target_pairs"],
        "label_free_contract": material_audit["no_source_number_labels"] and material_audit["no_requested_marker_list"],
        "stressors_present": material_audit["shared_opening_verified"] and material_audit["unequal_sentence_lengths"],
    }
    result = {"phase": PHASE, "campaign": CAMPAIGN, "evidence_audit": audit, "material_audit": material_audit,
              "checks": checks, "all_checks_passed": all(checks.values())}
    save(final, result); append_memo(result)
    if not result["all_checks_passed"]:
        raise RuntimeError(checks)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
