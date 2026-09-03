#!/usr/bin/env python3
"""Fresh lexical flagship correction plus teacher-forced/autonomous long-range realization."""
from __future__ import annotations

import gc
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2368 = RESULT / "phase2368_c12481_c12720_longrange_operator_contract"
P2369 = RESULT / "phase2369_c12721_c13040_qwen_longrange_full_field"
OUT = RESULT / "phase2373_c14001_c14320_fresh_flagship_generation"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
LONG_MATERIAL = P2368 / "material/long_sentence_permutation.jsonl"
FRESH_MATERIAL = OUT / "material/fresh_lexical_flagship.jsonl"
FRESH_STATES = OUT / "raw/qwen4b_fresh_flagship_boundary.float16.npy"
FRESH_DECISIONS = OUT / "raw/qwen4b_fresh_flagship_decisions.float32.npy"
FRESH_PROGRESS = OUT / "raw/fresh_flagship_progress.json"
TEACHER_RESULT = OUT / "raw/index_teacher_forced.float32.npy"
TEACHER_PROGRESS = OUT / "raw/teacher_progress.json"
AUTONOMOUS_ROWS = OUT / "material/autonomous_index_rows.jsonl"
AUTONOMOUS_RESULT = OUT / "raw/autonomous_index_results.jsonl"
TRAJECTORY = OUT / "raw/qwen4b_index_generation_trajectory.float16.npy"
TRAJECTORY_ROWS = OUT / "material/trajectory_rows.jsonl"
TRAJECTORY_RESULT = OUT / "raw/trajectory_generation_results.jsonl"
COPY_RESULT = OUT / "raw/exact_copy_generation_results.jsonl"
PHASE = 2373
CAMPAIGN = "C14001-C14320"
SYSTEMS = ("attitude_role", "taxonomy_chain")
SUBJECTS = (("Mira", "米拉"), ("Noah", "诺亚"), ("Lena", "莉娜"), ("Owen", "欧文"), ("Rina", "瑞娜"),
            ("Tariq", "塔里克"), ("Uma", "乌玛"), ("Vera", "维拉"), ("Wade", "韦德"), ("Yara", "雅拉"),
            ("Zane", "赞恩"), ("Bora", "博拉"), ("Cora", "科拉"), ("Davi", "达维"), ("Eira", "艾拉"))
OBJECTS = (("apricots", "杏"), ("plums", "李子"), ("melons", "甜瓜"), ("walnuts", "核桃"), ("figs", "无花果"),
           ("carrots", "胡萝卜"), ("olives", "橄榄"), ("beans", "豆子"), ("mangos", "芒果"), ("dates", "椰枣"),
           ("peaches", "桃子"), ("lemons", "柠檬"), ("grapes", "葡萄"), ("cherries", "樱桃"), ("berries", "浆果"))
CHAIN_EN = (("sparrow", "bird", "animal", "entity"), ("salmon", "fish", "animal", "entity"),
            ("violin", "instrument", "artifact", "object"), ("cedar", "tree", "plant", "organism"),
            ("ruby", "gem", "mineral", "material"), ("canoe", "boat", "vehicle", "artifact"),
            ("sonnet", "poem", "text", "work"), ("copper", "metal", "material", "substance"),
            ("falcon", "bird", "animal", "organism"), ("tulip", "flower", "plant", "organism"),
            ("cello", "instrument", "artifact", "object"), ("basalt", "rock", "mineral", "material"),
            ("schooner", "boat", "vehicle", "artifact"), ("haiku", "poem", "text", "work"),
            ("oak", "tree", "plant", "organism"))
CHAIN_ZH = (("麻雀", "鸟", "动物", "实体"), ("鲑鱼", "鱼", "动物", "实体"), ("小提琴", "乐器", "制品", "物体"),
            ("雪松", "树", "植物", "生物"), ("红宝石", "宝石", "矿物", "材料"), ("独木舟", "船", "交通工具", "制品"),
            ("十四行诗", "诗", "文本", "作品"), ("铜", "金属", "材料", "物质"), ("猎鹰", "鸟", "动物", "生物"),
            ("郁金香", "花", "植物", "生物"), ("大提琴", "乐器", "制品", "物体"), ("玄武岩", "岩石", "矿物", "材料"),
            ("纵帆船", "船", "交通工具", "制品"), ("俳句", "诗", "文本", "作品"), ("橡树", "树", "植物", "生物"))

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402
import phase2369_c12721_c13040_qwen_longrange_full_field as collect  # noqa: E402
import phase2372_c13681_c14000_dual_flagship_atlas as atlas  # noqa: E402


def save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Counter): return dict(value)
    raise TypeError(type(value).__name__)


def read_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8-sig").splitlines() if x.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def local(pair: tuple[str, str], language: str) -> str: return pair[0] if language == "en" else pair[1]


def fresh_partition(unit: int) -> str: return "train" if unit < 5 else "confirmation" if unit < 10 else "lockbox"


def compile_fresh(tokenizer) -> tuple[list[dict], dict]:
    rows = []
    roles = ("subject", "polarity", "attitude", "action", "object")
    for unit in range(15):
        for language in ("en", "zh"):
            for surface in ("direct", "independent_prose"):
                for cell in range(32):
                    bits = [(cell >> i) & 1 for i in range(5)]
                    subject_pair = SUBJECTS[unit], SUBJECTS[(unit + 4) % 15]
                    object_pair = OBJECTS[unit], OBJECTS[(unit + 7) % 15]
                    subject = local(subject_pair[bits[0]], language); polarity = bits[1]
                    attitude = local((("likes", "喜欢"), ("prefers", "偏爱"))[bits[2]], language)
                    action = local((("taste", "品尝"), ("purchase", "购买"))[bits[3]], language)
                    obj = local(object_pair[bits[4]], language); query = roles[unit % 5]
                    values = {"subject": subject, "polarity": local((("affirmative", "肯定"), ("negative", "否定"))[polarity], language),
                              "attitude": attitude, "action": action, "object": obj}
                    neg = local(("does not ", "不"), language) if polarity else ""
                    if language == "en":
                        sentence = f"{subject} {neg}{attitude} to {action} {obj} during archive visit {unit + 1}."
                        prompt = sentence if surface == "direct" else f"A separate archivist preserved this exact preference report: '{sentence}'"
                        prompt += f"\nReturn only the {query}. Answer:"
                    else:
                        sentence = f"{subject}在第{unit + 1}次档案访问中{neg}{attitude}{action}{obj}。"
                        prompt = sentence if surface == "direct" else f"一名独立档案员保留了这句偏好报告：“{sentence}”"
                        prompt += f"\n只返回{query}。答案："
                    target, foil = " " + values[query], " " + local(("unknown", "未知"), language)
                    ti, fi = tokenizer.encode(target, add_special_tokens=False), tokenizer.encode(foil, add_special_tokens=False)
                    rows.append({"case_id": f"c14001-fresh-attitude-u{unit}-{language}-{surface}-c{cell:02d}", "design_index": len(rows),
                                 "system": "attitude_role", "unit": unit, "partition": fresh_partition(unit), "language": language,
                                 "surface": surface, "cell": cell, "bits": bits, "query": query, "prompt": prompt,
                                 "prompt_ids": tokenizer.encode(prompt, add_special_tokens=False), "target": target, "foil": foil,
                                 "target_ids": ti, "foil_ids": fi, "target_first_id": ti[0], "foil_first_id": fi[0]})
    for unit in range(15):
        for language in ("en", "zh"):
            for surface in ("facts", "independent_prose"):
                for cell in range(32):
                    bits = [(cell >> i) & 1 for i in range(5)]
                    base = CHAIN_EN if language == "en" else CHAIN_ZH
                    nodes = list(base[(unit + 5 * bits[0]) % 15])
                    rel = local((("is a", "是一种"), ("belongs to", "属于"))[bits[1]], language)
                    edges = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3])]
                    if bits[2]: edges.pop(1)
                    if bits[3]: edges.append((nodes[0], local((f"archive item {unit + 1}", f"档案项目{unit + 1}"), language)))
                    depth = 3 if bits[4] else 1
                    unknown = local(("unknown", "未知"), language)
                    target_value = nodes[depth] if not (bits[2] and depth == 3) else unknown
                    foil_value = unknown if target_value != unknown else nodes[3]
                    if language == "en":
                        facts = "; ".join(f"{a} {rel} {b}" for a, b in edges) + "."
                        prompt = facts if surface == "facts" else f"Independent catalog entry {unit + 1} states: {facts}"
                        prompt += f"\nUsing only this entry, give the depth-{depth} category of {nodes[0]}. Answer:"
                    else:
                        facts = "；".join(f"{a}{rel}{b}" for a, b in edges) + "。"
                        prompt = facts if surface == "facts" else f"独立目录条目{unit + 1}记录：{facts}"
                        prompt += f"\n只根据该条目，给出{nodes[0]}的第{depth}层类别。答案："
                    target, foil = " " + target_value, " " + foil_value
                    ti, fi = tokenizer.encode(target, add_special_tokens=False), tokenizer.encode(foil, add_special_tokens=False)
                    rows.append({"case_id": f"c14001-fresh-chain-u{unit}-{language}-{surface}-c{cell:02d}", "design_index": len(rows),
                                 "system": "taxonomy_chain", "unit": unit, "partition": fresh_partition(unit), "language": language,
                                 "surface": surface, "cell": cell, "bits": bits, "query": f"depth_{depth}", "prompt": prompt,
                                 "prompt_ids": tokenizer.encode(prompt, add_special_tokens=False), "target": target, "foil": foil,
                                 "target_ids": ti, "foil_ids": fi, "target_first_id": ti[0], "foil_first_id": fi[0]})
    audit = {"rows": len(rows), "expected": 2 * 15 * 2 * 2 * 32, "unique_prompts": len(set(r["prompt"] for r in rows)),
             "unique_case_ids": len(set(r["case_id"] for r in rows)) == len(rows),
             "first_token_distinct": all(r["target_first_id"] != r["foil_first_id"] for r in rows),
             "splits": dict(Counter(r["partition"] for r in rows)), "systems": dict(Counter(r["system"] for r in rows))}
    return rows, audit


def fresh_splits(keys: list[tuple], system: str) -> dict[str, np.ndarray]:
    return {"train": np.asarray([i for i, k in enumerate(keys) if k[0] == system and k[1] < 5]),
            "confirmation": np.asarray([i for i, k in enumerate(keys) if k[0] == system and 5 <= k[1] < 10]),
            "lockbox": np.asarray([i for i, k in enumerate(keys) if k[0] == system and k[1] >= 10])}


def analyze_fresh(rows: list[dict]) -> dict:
    states = np.load(FRESH_STATES, mmap_mode="r"); keys, row_index = atlas.build_group_index(rows)
    splits = {system: fresh_splits(keys, system) for system in SYSTEMS}
    candidates, orders = ("global", "language", "surface", "language_surface", "other_system_global"), (1, 2, 3, 5)
    layer_rows = {system: [] for system in SYSTEMS}
    for qpoint in range(38):
        field = np.asarray(states[row_index, qpoint], dtype=np.float32)
        for si, system in enumerate(SYSTEMS):
            entries = []; other = splits[SYSTEMS[1 - si]]["train"]
            for candidate in candidates:
                for order in orders:
                    rc, _ = atlas.evaluate(field, keys, splits[system], other, candidate, order, "confirmation")
                    rl, _ = atlas.evaluate(field, keys, splits[system], other, candidate, order, "lockbox")
                    entries.append({"candidate": candidate, "order": order, "confirmation_response_r2": rc, "lockbox_response_r2": rl})
            layer_rows[system].append({"qpoint": qpoint, "entries": entries})
        print(f"[phase2373 fresh analysis] qpoint {qpoint}/37", flush=True)
    summary = {}
    for system in SYSTEMS:
        choices = [(e["confirmation_response_r2"], layer["qpoint"], e) for layer in layer_rows[system] for e in layer["entries"] if np.isfinite(e["confirmation_response_r2"])]
        _, q, selected = max(choices, key=lambda x: x[0])
        summary[system] = {"selected_qpoint": q, "selected_candidate": selected["candidate"], "selected_order": selected["order"],
                           "confirmation_response_r2": selected["confirmation_response_r2"], "lockbox_response_r2": selected["lockbox_response_r2"],
                           "passed_fresh_lexical_lockbox": selected["lockbox_response_r2"] > 0}
    save(OUT / "analysis/fresh_flagship_layers.json", layer_rows)
    return {"splits": {s: {k: len(v) for k, v in splits[s].items()} for s in SYSTEMS}, "summary": summary}


def left_pad(sequences: list[list[int]], device: torch.device, pad: int):
    width = max(map(len, sequences)); ids = torch.full((len(sequences), width), pad, dtype=torch.long, device=device); mask = torch.zeros_like(ids)
    for i, seq in enumerate(sequences): ids[i, -len(seq):] = torch.tensor(seq, device=device); mask[i, -len(seq):] = 1
    return ids, mask, (mask.cumsum(1) - 1).clamp_min(0)


def teacher_forced(model, rows: list[dict], batch_size: int = 4) -> dict:
    selected = [r for r in rows if r["task"] == "index_only"]
    if TEACHER_RESULT.exists() and TEACHER_PROGRESS.exists():
        completed = int(json.loads(TEACHER_PROGRESS.read_text(encoding="utf-8"))["completed"]); out = np.lib.format.open_memmap(TEACHER_RESULT, mode="r+")
    else:
        completed = 0; TEACHER_RESULT.parent.mkdir(parents=True, exist_ok=True)
        out = np.lib.format.open_memmap(TEACHER_RESULT, mode="w+", dtype=np.float32, shape=(len(selected), 4))
    device = next(model.parameters()).device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    with torch.inference_mode():
        for start in range(completed, len(selected), batch_size):
            batch = selected[start:start + batch_size]; seqs = [r["prompt_ids"] + r["target_ids"] for r in batch]
            ids, mask, pos = left_pad(seqs, device, pad)
            logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True).logits
            for local_index, row in enumerate(batch):
                target = row["target_ids"]; offset = ids.shape[1] - len(seqs[local_index]) + len(row["prompt_ids"]) - 1
                selected_logits = logits[local_index, offset:offset + len(target)].float()
                logp = torch.log_softmax(selected_logits, -1)[torch.arange(len(target), device=device), torch.tensor(target, device=device)]
                predicted = selected_logits.argmax(-1).cpu().tolist()
                divergence = next((i for i, (a, b) in enumerate(zip(predicted, target)) if a != b), len(target))
                out[start + local_index] = [float(logp.sum()), float(logp.mean()), float(predicted == target), float(divergence)]
            out.flush(); save(TEACHER_PROGRESS, {"completed": start + len(batch)})
            if (start + len(batch)) % 256 == 0 or start + len(batch) == len(selected): print(f"[phase2373 teacher] {start + len(batch)}/{len(selected)}", flush=True)
    result = {"rows": len(selected), "mean_sequence_logprob": float(np.asarray(out[:, 0]).mean()),
              "mean_token_logprob": float(np.asarray(out[:, 1]).mean()), "teacher_forced_argmax_sequence_exact": float(np.asarray(out[:, 2]).mean()),
              "mean_first_divergence_position": float(np.asarray(out[:, 3]).mean())}
    del out; return result


def strip_reasoning(text: str) -> str:
    return text.split("</think>", 1)[-1].strip()


def generate_text(model, tokenizer, rows: list[dict], max_new_tokens: int, batch_size: int) -> list[dict]:
    device = next(model.parameters()).device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0); results = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]; ids, mask, _ = left_pad([r["prompt_ids"] for r in batch], device, pad)
        with torch.inference_mode():
            output = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=max_new_tokens, do_sample=False,
                                    pad_token_id=pad, eos_token_id=model.config.eos_token_id)
        texts = tokenizer.batch_decode(output[:, ids.shape[1]:], skip_special_tokens=True)
        for row, text in zip(batch, texts): results.append({"case_id": row["case_id"], "generated": text})
        print(f"[phase2373 generate] {min(start + len(batch), len(rows))}/{len(rows)} max={max_new_tokens}", flush=True)
    return results


def autonomous_index(model, tokenizer, rows: list[dict]) -> dict:
    chosen_perms = {(0, 1, 2, 3), (1, 0, 2, 3), (1, 2, 0, 3), (3, 2, 1, 0)}
    selected = [r for r in rows if r["task"] == "index_only" and tuple(r["target_perm"]) in chosen_perms]
    write_rows(AUTONOMOUS_ROWS, selected)
    generated = generate_text(model, tokenizer, selected, 10, 8)
    by_id = {r["case_id"]: r for r in selected}; valid = exact = 0; enriched = []
    for item in generated:
        row = by_id[item["case_id"]]; clean = strip_reasoning(item["generated"]); digits = [int(x) for x in re.findall(r"(?<!\d)([1-4])(?!\d)", clean)[:4]]
        target = [int(x) for x in row["target"].split(",")]
        is_valid = len(digits) == 4 and sorted(digits) == [1, 2, 3, 4]
        is_exact = digits == target; valid += is_valid; exact += is_exact
        enriched.append({**item, "parsed_first_four": digits, "target_order": target, "valid_permutation": is_valid, "order_exact": is_exact})
    write_rows(AUTONOMOUS_RESULT, enriched)
    return {"rows": len(selected), "valid_permutation_rate": valid / len(selected), "order_exact_rate": exact / len(selected),
            "boundary": "Whitespace and optional <think> text are ignored before parsing the first four standalone source indices."}


def qmodules(model):
    return [model.model.embed_tokens, *list(model.model.layers), model.model.norm]


def generation_trajectory(model, tokenizer, rows: list[dict]) -> dict:
    target_perm = (3, 2, 1, 0)
    selected = [r for r in rows if r["task"] == "index_only" and r["unit"] >= 4 and tuple(r["target_perm"]) == target_perm]
    if len(selected) != 64: raise RuntimeError(len(selected))
    write_rows(TRAJECTORY_ROWS, selected); modules = qmodules(model); dim = int(model.config.hidden_size); steps = 8
    trajectory = np.lib.format.open_memmap(TRAJECTORY, mode="w+", dtype=np.float16, shape=(len(selected), steps, len(modules), dim))
    device = next(model.parameters()).device; pad = int(model.config.pad_token_id or model.config.eos_token_id or 0)
    captures = {}; handles = []
    for qi, module in enumerate(modules):
        def hook(_module, _inputs, output, qi=qi): captures[qi] = (output[0] if isinstance(output, tuple) else output)[:, -1].detach()
        handles.append(module.register_forward_hook(hook))
    generated_ids = [[] for _ in selected]
    try:
        for step in range(steps):
            for start in range(0, len(selected), 8):
                batch = selected[start:start + 8]; seqs = [r["prompt_ids"] + generated_ids[start + i] for i, r in enumerate(batch)]
                ids, mask, pos = left_pad(seqs, device, pad); captures.clear()
                with torch.inference_mode(): logits = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=False, return_dict=True).logits[:, -1]
                for qi in range(len(modules)): trajectory[start:start + len(batch), step, qi] = captures[qi].float().cpu().numpy().astype(np.float16)
                next_ids = logits.argmax(-1).cpu().tolist()
                for i, token in enumerate(next_ids): generated_ids[start + i].append(token)
            trajectory.flush(); print(f"[phase2373 trajectory] step {step + 1}/{steps}", flush=True)
    finally:
        for handle in handles: handle.remove()
    results = [{"case_id": row["case_id"], "generated_ids": ids, "generated": tokenizer.decode(ids, skip_special_tokens=True)} for row, ids in zip(selected, generated_ids)]
    write_rows(TRAJECTORY_RESULT, results)
    return {"shape": list(trajectory.shape), "rows": len(selected), "steps": steps}


def exact_copy_generation(model, tokenizer, rows: list[dict]) -> dict:
    selected = [r for r in rows if r["task"] == "exact_copy" and r["unit"] >= 4 and r["source_perm"] == [2, 0, 3, 1]
                and r["target_perm"] == [3, 2, 1, 0]]
    if len(selected) != 32: raise RuntimeError(len(selected))
    generated = generate_text(model, tokenizer, selected, 128, 4); by_id = {r["case_id"]: r for r in selected}; enriched = []
    order_exact = content_total = verbatim = 0
    for item in generated:
        row = by_id[item["case_id"]]; clean = strip_reasoning(item["generated"])
        positions = [(clean.find(marker), sid) for sid, marker in enumerate(row["markers"])]
        observed = [sid for pos, sid in sorted(positions) if pos >= 0]
        expected = row["target_perm"]; oe = observed == expected
        codes = [re.search(r"[A-Z]{3}\d\d", sentence).group(0) for sentence in row["source_sentences"]]
        preservation = sum(code in clean for code in codes) / 4
        ve = " ".join(clean.split()).startswith(" ".join(row["target"].strip().split()))
        order_exact += oe; content_total += preservation; verbatim += ve
        enriched.append({**item, "observed_marker_order": observed, "target_marker_order": expected,
                         "marker_order_exact": oe, "code_content_preservation": preservation, "verbatim_exact": ve})
    write_rows(COPY_RESULT, enriched)
    return {"rows": len(selected), "marker_order_exact_rate": order_exact / len(selected),
            "mean_code_content_preservation": content_total / len(selected), "verbatim_exact_rate": verbatim / len(selected),
            "max_new_tokens": 128, "boundary": "128 tokens may truncate four long sentences; code preservation and observed marker order remain separately reported."}


def append_memo(result: dict) -> None:
    if f"## Phase {PHASE}:" in MEMO.read_text(encoding="utf-8"): return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""

## Phase {PHASE}: 真正fresh旗舰纠错、未来序列概率与自主长距离重排（{CAMPAIGN}） [{stamp}]

**测试原理与测试用例。** 审计发现Phase2372旗舰unit未进入文本，所谓fresh锁箱含重复prompt；因此该$R^2=1$撤回。本Phase在模型前冻结15个真正不同的主体/对象/链节点unit，0–4训练、5–9确认、10–14锁箱，共{result['fresh_material']['rows']}条，再采集38×2560全坐标并重跑层×条件×阶数竞赛。同时对全部4608条索引任务计算教师强制全序列概率，对平衡的768条做自主生成，对64条锁箱保存8步×38×2560生成轨迹，并对32条长句原样重排生成128 token。

$$
\log P(y_{{1:T}}\mid x)=\sum_{{t=1}}^T\log P(y_t\mid x,y_{{<t}}),\qquad
t^*=\min\{{t:\arg\max_v P(v\mid x,y_{{<t}})\ne y_t\}}.
$$

**结果汇总。** fresh材料 `{json.dumps(result['fresh_material'], ensure_ascii=False)}`；fresh旗舰纠错结果 `{json.dumps(result['fresh_analysis'], ensure_ascii=False)}`；教师强制序列 `{json.dumps(result['teacher_forced'], ensure_ascii=False)}`；自主索引 `{json.dumps(result['autonomous_index'], ensure_ascii=False)}`；生成轨迹 `{json.dumps(result['trajectory'], ensure_ascii=False)}`；原样长句 `{json.dumps(result['exact_copy'], ensure_ascii=False)}`。

**相关文件。** 脚本 `tests/glm5/phase2373_c14001_c14320_fresh_flagship_generation.py`；fresh材料/全场、序列概率、自主结果和生成轨迹位于 `tests/glm5/result/phase2373_c14001_c14320_fresh_flagship_generation`。

**理论进展、问题硬伤与结论。** 只有fresh词汇锁箱结果可用于判断双旗舰复用，Phase2372重复prompt只保留为材料错误示例。教师强制概率、教师强制argmax、自主索引与长句逐字复制是四个不同门；任何一个通过都不能替代其余门。生成会受Qwen思考模板、空格格式和128-token截断影响，因此同时报告有效排列、顺序、标记、内容代码与逐字指标。生成轨迹是激活观测而不是训练参数或因果必要坐标。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as f: f.write(text)


def main() -> None:
    final_path = OUT / "analysis/final.json"
    if final_path.exists():
        result = json.loads(final_path.read_text(encoding="utf-8")); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2)); return
    long_rows = read_rows(LONG_MATERIAL)
    model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
    try:
        fresh_rows, fresh_audit = compile_fresh(tokenizer); write_rows(FRESH_MATERIAL, fresh_rows)
        collect.collect_boundary(model, fresh_rows, FRESH_STATES, FRESH_DECISIONS, FRESH_PROGRESS, "phase2373 fresh", 16)
        teacher = teacher_forced(model, long_rows)
        autonomous = autonomous_index(model, tokenizer, long_rows)
        trajectory = generation_trajectory(model, tokenizer, long_rows)
        copy_result = exact_copy_generation(model, tokenizer, long_rows)
    finally:
        model_utils.release_model(model); del model; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    fresh_analysis = analyze_fresh(fresh_rows)
    result = {"phase": PHASE, "campaign": CAMPAIGN, "phase2372_retraction": "unit labels did not alter flagship prompts; q1 R2=1 was duplicate-prompt leakage",
              "fresh_material": fresh_audit, "fresh_behavior": collect.summarize_behavior(fresh_rows, FRESH_DECISIONS, ["system", "language", "partition"]),
              "fresh_analysis": fresh_analysis, "teacher_forced": teacher, "autonomous_index": autonomous,
              "trajectory": trajectory, "exact_copy": copy_result}
    save(final_path, result); append_memo(result); print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
