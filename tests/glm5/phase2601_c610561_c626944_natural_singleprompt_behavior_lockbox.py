#!/usr/bin/env python3
"""Bilingual candidate-free behavior lockbox for six genuine language operations."""
from __future__ import annotations

import gc
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
TESTS = ROOT / "tests/glm5"
RESULT = TESTS / "result"
P2600 = RESULT / "phase2600_c602369_c610560_attachment_audit_endogenous_contract"
OUT = RESULT / "phase2601_c610561_c626944_natural_singleprompt_behavior_lockbox"
MEMO = ROOT / "research/glm5/docs/AGI_GLM5_MEMO.md"
PHASE, CAMPAIGN = 2601, "C610561-C626944"
FAMILIES = ("reference", "negation", "chronology", "sentence_reorder", "syntax_role", "taxonomy_chain")
LANGUAGES = ("en", "zh")
PAIRS_PER_GROUP = 50

sys.path.insert(0, str(TESTS))
import model_utils  # noqa: E402


EN_NAMES = ("Lina", "Mira", "Nora", "Tessa", "Omar", "Pavel", "Ravi", "Soren", "Yara", "Zane",
            "Elena", "Freya", "Hugo", "Iris", "Jonas", "Kira", "Leo", "Maya", "Noah", "Rina")
ZH_NAMES = ("林娜", "米拉", "诺拉", "泰莎", "奥马", "帕维", "拉维", "索伦", "雅拉", "赞恩",
            "艾琳", "芙蕾", "雨果", "艾瑞", "乔纳", "琪拉", "里奥", "玛雅", "诺亚", "瑞娜")
EN_OBJECTS = ("amber key", "silver coin", "blue ribbon", "green folder", "brass token", "violet card",
              "wooden seal", "glass bead", "linen map", "copper badge", "ivory comb", "scarlet note",
              "ceramic cup", "paper crane", "velvet pouch", "granite tile", "golden pin", "orange ticket",
              "marble disk", "indigo scarf")
ZH_OBJECTS = ("琥珀钥匙", "银色硬币", "蓝色丝带", "绿色文件", "黄铜令牌", "紫色卡片", "木质印章", "玻璃珠子",
              "亚麻地图", "铜制徽章", "象牙梳子", "红色便笺", "陶瓷杯子", "纸制仙鹤", "丝绒袋子", "花岗石片",
              "金色别针", "橙色票据", "大理石盘", "靛蓝围巾")
EN_EVENTS = ("the bell rang", "the gate opened", "the lamp lit", "the rain stopped", "the train arrived",
             "the clerk called", "the flag rose", "the music ended", "the oven cooled", "the clock chimed")
ZH_EVENTS = ("铃声响起", "大门打开", "灯光亮起", "雨水停止", "列车到达", "职员呼叫", "旗帜升起", "音乐结束", "烤箱冷却", "时钟报时")
EN_ENTITIES = ("amber tern", "silver mole", "violet lynx", "copper ibis", "linen fox", "marble hare",
               "golden yak", "indigo seal", "scarlet vole", "ceramic owl")
ZH_ENTITIES = ("琥珀燕", "银色鼹", "紫色猞猁", "铜色鹮", "亚麻狐", "大理石兔", "金色牦牛", "靛蓝海豹", "绯红田鼠", "陶瓷猫头鹰")
EN_MID = ("talven", "morlic", "senvar", "dulven", "parnic", "kelmor", "ravlen", "tormic", "vespar", "nolven")
ZH_MID = ("塔文类", "莫里类", "森瓦类", "杜文类", "帕尼类", "凯莫类", "拉文类", "托米类", "维斯类", "诺文类")
EN_SUPER = ("vornic", "selmic", "darven", "polric", "kenthal", "mirven", "sorlic", "falven", "bernic", "yulmar")
ZH_SUPER = ("沃尼类", "塞尔类", "达文类", "波里类", "肯塔类", "米尔类", "索里类", "法文类", "贝尼类", "尤玛类")


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.casefold())


def split_for(index: int) -> str:
    return "discovery" if index < 20 else "confirmation" if index < 40 else "external"


def prompt_and_target(family: str, language: str, index: int, variant: int):
    names = EN_NAMES if language == "en" else ZH_NAMES
    objects = EN_OBJECTS if language == "en" else ZH_OBJECTS
    events = EN_EVENTS if language == "en" else ZH_EVENTS
    entities = EN_ENTITIES if language == "en" else ZH_ENTITIES
    mids = EN_MID if language == "en" else ZH_MID
    supers = EN_SUPER if language == "en" else ZH_SUPER
    a, b, c = names[index % len(names)], names[(index * 3 + 5) % len(names)], names[(index * 7 + 9) % len(names)]
    if len({a, b, c}) < 3:
        c = names[(index + 11) % len(names)]
    o0, o1 = objects[(2 * index) % len(objects)], objects[(2 * index + 1) % len(objects)]
    e0, e1, e2, e3 = (events[(index + shift) % len(events)] for shift in (0, 3, 6, 8))
    if family == "reference":
        source = o0 if variant == 0 else o1
        target = source
        if language == "en":
            prompt = (f"Context: {a} placed the {source} beside {b}. After {b} left, {a} picked it up and locked it away. "
                      f"Question: What object did {a} pick up? Reply with only the exact object phrase and no explanation. Answer:")
        else:
            prompt = (f"上下文：{a}把{source}放在{b}旁边。{b}离开后，{a}把它拿起来并锁好。"
                      f"问题：{a}拿起了什么物品？只回答准确的物品名称，不要解释。答案：")
    elif family == "negation":
        source = a if variant == 0 else b
        target = source
        if language == "en":
            prompt = (f"At the inspection, {a}, {b}, and {c} were listed. The report states that everyone except {source} "
                      "submitted a signed copy. Who did not submit a signed copy? Reply with only the name. Answer:")
        else:
            prompt = (f"检查名单中有{a}、{b}和{c}。报告说明，除{source}以外的每个人都提交了签字副本。"
                      "谁没有提交签字副本？只回答姓名。答案：")
    elif family == "chronology":
        source = "before" if variant == 0 else "after"
        target = a if variant == 0 else b
        if language == "en":
            prompt = (f"{a} arrived {source} {b}. {c} arrived after both of them. Who arrived first? "
                      "Reply with only the name. Answer:")
        else:
            source = "之前" if variant == 0 else "之后"
            prompt = (f"{a}在{b}{source}到达。{c}在他们两人之后到达。谁最先到达？只回答姓名。答案：")
    elif family == "sentence_reorder":
        source = "from earliest to latest" if variant == 0 else "from latest to earliest"
        target = "B-D-A-C" if variant == 0 else "C-A-D-B"
        if language == "en":
            prompt = (f"Four intact sentences are out of order: [A] On the third morning, {e0}. [B] On the first morning, {e1}. "
                      f"[C] On the fourth morning, {e2}. [D] On the second morning, {e3}. Arrange all labels {source}. "
                      "Preserve every sentence; reply only as A-B-C-D style labels. Answer:")
        else:
            source = "由早到晚" if variant == 0 else "由晚到早"
            prompt = (f"四个完整句子的顺序被打乱：[A]第三天早晨，{e0}。[B]第一天早晨，{e1}。"
                      f"[C]第四天早晨，{e2}。[D]第二天早晨，{e3}。请{source}排列全部标签，"
                      "保持每个句子内容不变；只用A-B-C-D形式回答。答案：")
    elif family == "syntax_role":
        patient = c
        source = a if variant == 0 else b
        target = source
        if language == "en":
            prompt = (f"During the ceremony, {patient} was quietly praised by {source}. Who performed the praising? "
                      "Reply with only the agent's name. Answer:")
        else:
            prompt = (f"仪式期间，{patient}被{source}轻声表扬了。谁是实施表扬动作的人？只回答施事者姓名。答案：")
    elif family == "taxonomy_chain":
        entity = entities[index % len(entities)]
        mid = mids[(index * 3) % len(mids)]
        s0, s1 = supers[(2 * index) % len(supers)], supers[(2 * index + 1) % len(supers)]
        source = s0 if variant == 0 else s1
        target = source
        if language == "en":
            prompt = (f"A {entity} is a type of {mid}. Every {mid} belongs to the {source} category. "
                      f"Which category does a {entity} ultimately belong to? Reply with only the category name. Answer:")
        else:
            prompt = (f"{entity}属于{mid}。每个{mid}都属于{source}。{entity}最终属于哪一类？只回答类别名称。答案：")
    else:
        raise KeyError(family)
    return prompt, target, source


def diff_span(ids0, ids1):
    if len(ids0) != len(ids1):
        return None
    left = 0
    while left < len(ids0) and ids0[left] == ids1[left]:
        left += 1
    right = 0
    while right < len(ids0) - left and ids0[-1 - right] == ids1[-1 - right]:
        right += 1
    stop = len(ids0) - right
    if stop <= left:
        return None
    return list(range(left, stop))


def compile_material(tokenizer):
    material = []
    rejected = defaultdict(int)
    for family in FAMILIES:
        for language in LANGUAGES:
            accepted = 0
            candidate_index = 0
            while accepted < PAIRS_PER_GROUP and candidate_index < 1000:
                prompts = [prompt_and_target(family, language, candidate_index, variant) for variant in (0, 1)]
                ids = [tokenizer.encode(item[0], add_special_tokens=False) for item in prompts]
                span = diff_span(ids[0], ids[1])
                if span is None or len(span) > 8:
                    rejected[f"{family}/{language}"] += 1
                    candidate_index += 1
                    continue
                pair_id = f"{family}_{language}_{accepted:03d}"
                split = split_for(accepted)
                for variant in (0, 1):
                    prompt, target, source = prompts[variant]
                    alternate = prompts[1 - variant][1]
                    material.append({
                        "case_id": f"{pair_id}_v{variant}", "pair_id": pair_id, "pair_index": accepted,
                        "family": family, "language": language, "split": split, "variant": variant,
                        "prompt": prompt, "target": target, "alternate": alternate, "source_text": source,
                        "prompt_ids": ids[variant], "source_token_positions": span,
                        "answer_boundary_token": len(ids[variant]) - 1, "candidate_list_in_prompt": False,
                    })
                accepted += 1
                candidate_index += 1
            if accepted != PAIRS_PER_GROUP:
                raise RuntimeError((family, language, accepted, dict(rejected)))
    return material, dict(rejected)


def candidate_token_ids(tokenizer, prompt: str, answer: str):
    base = tokenizer.encode(prompt, add_special_tokens=False)
    for separator in (" ", "\n"):
        full = tokenizer.encode(prompt + separator + answer, add_special_tokens=False)
        if full[:len(base)] == base and len(full) > len(base):
            return full, list(range(len(base), len(full)))
    raise RuntimeError("candidate continuation changed prompt tokenization")


def score_binary(model, tokenizer, material, batch_size=24):
    device = model.get_input_embeddings().weight.device
    jobs = []
    for row in material:
        for label, answer in (("target", row["target"]), ("alternate", row["alternate"])):
            ids, answer_positions = candidate_token_ids(tokenizer, row["prompt"], answer)
            jobs.append((row["case_id"], label, ids, answer_positions))
    scores = defaultdict(dict)
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        width = max(len(job[2]) for job in batch)
        ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        label_mask = torch.zeros_like(ids, dtype=torch.bool)
        for index, (_, _, seq, answer_positions) in enumerate(batch):
            ids[index, :len(seq)] = torch.tensor(seq, device=device)
            mask[index, :len(seq)] = 1
            label_mask[index, answer_positions] = True
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask).logits.float()
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            token_lp = logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        for index, (case_id, label, _, _) in enumerate(batch):
            active = label_mask[index, 1:]
            values = token_lp[index][active]
            scores[case_id][label] = float(values.mean().item())
        if start + len(batch) == len(jobs) or (start + len(batch)) % 480 == 0:
            print(f"[phase2601 score] {start + len(batch)}/{len(jobs)}", flush=True)
    return [{"case_id": row["case_id"], "target_mean_logp": scores[row["case_id"]]["target"],
             "alternate_mean_logp": scores[row["case_id"]]["alternate"],
             "target_margin": scores[row["case_id"]]["target"] - scores[row["case_id"]]["alternate"],
             "correct": scores[row["case_id"]]["target"] > scores[row["case_id"]]["alternate"]}
            for row in material]


def left_pad(sequences, pad_id, device):
    width = max(map(len, sequences))
    ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(sequences):
        ids[index, width - len(sequence):] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, width - len(sequence):] = 1
    return ids, mask


def parse_generation(row, text: str):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
    answer_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    if row["family"] == "sentence_reorder":
        letters = re.findall(r"(?i)(?<![A-Za-z])[ABCD](?![A-Za-z])", answer_line)
        parsed = "-".join(letter.upper() for letter in letters[:4]) if len(letters) >= 4 else None
        return parsed, parsed == row["target"]
    target_hit = norm(row["target"]) in norm(answer_line)
    alternate_hit = norm(row["alternate"]) in norm(answer_line)
    parsed = row["target"] if target_hit and not alternate_hit else row["alternate"] if alternate_hit and not target_hit else None
    return parsed, parsed == row["target"]


def generate(model, tokenizer, material, batch_size=12):
    device = model.get_input_embeddings().weight.device
    output = []
    for start in range(0, len(material), batch_size):
        batch = material[start:start + batch_size]
        ids, mask = left_pad([row["prompt_ids"] for row in batch], tokenizer.pad_token_id, device)
        width = ids.shape[1]
        with torch.inference_mode():
            generated = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=16, do_sample=False,
                                       use_cache=True, pad_token_id=tokenizer.pad_token_id,
                                       eos_token_id=tokenizer.eos_token_id)
        for row, sequence in zip(batch, generated):
            generated_ids = sequence[width:].detach().cpu().tolist()
            text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            parsed, correct = parse_generation(row, text)
            cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I).strip()
            answer_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
            output.append({"case_id": row["case_id"], "generated_token_ids": generated_ids,
                           "generated": text, "parsed": parsed, "parsed_correct": correct,
                           "strict_exact": norm(answer_line) == norm(row["target"])})
        if start + len(batch) == len(material) or (start + len(batch)) % 240 == 0:
            print(f"[phase2601 generate] {start + len(batch)}/{len(material)}", flush=True)
    return output


def summarize(material, candidate, generated):
    cidx = {row["case_id"]: row for row in candidate}
    gidx = {row["case_id"]: row for row in generated}

    def metrics(rows):
        return {"n": len(rows),
                "candidate_accuracy": float(np.mean([cidx[row["case_id"]]["correct"] for row in rows])),
                "greedy_parsed_accuracy": float(np.mean([gidx[row["case_id"]]["parsed_correct"] for row in rows])),
                "greedy_strict_accuracy": float(np.mean([gidx[row["case_id"]]["strict_exact"] for row in rows])),
                "mean_candidate_margin": float(np.mean([cidx[row["case_id"]]["target_margin"] for row in rows]))}
    by_group = {}
    for family in FAMILIES:
        for language in LANGUAGES:
            subset = [row for row in material if row["family"] == family and row["language"] == language]
            by_group[f"{family}/{language}"] = metrics(subset)
    by_split = {split: metrics([row for row in material if row["split"] == split])
                for split in ("discovery", "confirmation", "external")}
    eligible = []
    pair_index = defaultdict(list)
    for row in material:
        pair_index[row["pair_id"]].append(row)
    for pair_id, rows in pair_index.items():
        rows = sorted(rows, key=lambda row: row["variant"])
        if len(rows) == 2 and all(gidx[row["case_id"]]["parsed_correct"] for row in rows):
            eligible.append({"pair_id": pair_id, "family": rows[0]["family"], "language": rows[0]["language"],
                             "split": rows[0]["split"], "source_token_positions": rows[0]["source_token_positions"]})
    return metrics(material), by_group, by_split, eligible


def append_memo(result):
    heading = f"## Phase {PHASE}: 六类真实语言操作1200条无候选单提示行为锁箱（{CAMPAIGN}）"
    if heading in MEMO.read_text(encoding="utf-8-sig"):
        return
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
    text = rf"""


{heading} [{stamp}]

**测试原理。** 脱离四事实二维查表，为指代、否定、时序、四句重排、句法施事和二跳分类各构造最小自然语言任务。每个pair只有一个局部source span发生变化并导致目标答案变化；模型prompt内没有候选列表，同时用评估器的二目标完整序列似然和真实greedy区分“可判别”与“能生成”：

$$m(x)=\bar\ell(y^*\mid x)-\bar\ell(y^{{cf}}\mid x),\qquad
\hat y_t=\arg\max_vp(v\mid x,\hat y_{{<t}}).$$

**测试用例。** 六族×中英×50最小pair×2变体=1200条prompt，每族200条、中英各100；每组前20 pair为发现、20为确认、10为外测。所有pair在Qwen tokenizer下总长度相同，记录全部变化token位置；Qwen3-4B BF16 CUDA非量化。无候选greedy最多16 token；另评分2400条未展示给模型的完整目标/反事实序列。

**结果汇总。** overall=`{json.dumps(result['overall'], ensure_ascii=False)}`；12族/语言=`{json.dumps(result['by_family_language'], ensure_ascii=False)}`；三分割=`{json.dumps(result['by_split'], ensure_ascii=False)}`；双变体greedy均正确pair共{result['eligible_pairs']}个，分布=`{json.dumps(result['eligible_by_group_split'], ensure_ascii=False)}`；通过75%预注册门的组=`{json.dumps(result['qualified_groups'], ensure_ascii=False)}`。

**相关文件。** 脚本`tests/glm5/phase2601_c610561_c626944_natural_singleprompt_behavior_lockbox.py`；1200条prompt、2400候选得分、1200真实生成、eligible清单、哈希与final位于`{OUT}`。

**分析与理论进展。** 本Phase首先测量模型是否真的执行不同外部语言操作，不把共同输出格式当机制。pair级对齐为后续单recipient source patch提供物理token位置；失败样本、高低margin与全部split完整保留。通过行为门只授权进入内部测量，不证明存在齿轮。

**问题硬伤。** 指代、否定、句法和分类仍有复制成分；重排输出标签而非复写长句；自然材料由有限模板生成；二目标似然不是开放词表；字符串解析可能宽于严格生成。没有任何内部机制结论。

**结论。** `{result['claim_boundary']}`；检查=`{json.dumps(result['checks'], ensure_ascii=False)}`；语言编码机制未闭合。
"""
    with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def main():
    model = tokenizer = None
    try:
        model, tokenizer, _ = model_utils.load_model("qwen3", dtype=torch.bfloat16, use_8bit=False)
        material, rejected = compile_material(tokenizer)
        candidate = score_binary(model, tokenizer, material)
        generated = generate(model, tokenizer, material)
    finally:
        if model is not None:
            model_utils.release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
    material_path = OUT / "material/cases.jsonl"
    candidate_path = OUT / "behavior/candidate_scores.jsonl"
    generated_path = OUT / "behavior/greedy_generation.jsonl"
    write_jsonl(material_path, material)
    write_jsonl(candidate_path, candidate)
    write_jsonl(generated_path, generated)
    overall, by_group, by_split, eligible = summarize(material, candidate, generated)
    eligible_path = OUT / "material/eligible_pairs.json"
    save_json(eligible_path, eligible)
    eligible_counts = defaultdict(int)
    for row in eligible:
        eligible_counts[f"{row['family']}/{row['language']}/{row['split']}"] += 1
    qualified = [key for key, value in by_group.items() if value["greedy_parsed_accuracy"] >= 0.75]
    contract = json.loads((P2600 / "protocol/frozen_contract.json").read_text(encoding="utf-8"))
    result = {
        "phase": PHASE, "campaign": CAMPAIGN, "timestamp": datetime.now().astimezone().isoformat(),
        "model": "Qwen3-4B BF16 CUDA nonquantized", "design": contract,
        "token_alignment_rejections": rejected, "overall": overall,
        "by_family_language": by_group, "by_split": by_split,
        "qualified_groups": qualified, "eligible_pairs": len(eligible),
        "eligible_by_group_split": dict(eligible_counts),
        "claim_boundary": "behavior qualification for six templated natural operations; no internal mechanism or open-world generalization",
        "files": {"material": str(material_path.relative_to(OUT)).replace("\\", "/"),
                  "candidate": str(candidate_path.relative_to(OUT)).replace("\\", "/"),
                  "generated": str(generated_path.relative_to(OUT)).replace("\\", "/"),
                  "eligible": str(eligible_path.relative_to(OUT)).replace("\\", "/")},
        "hashes": {"material": sha256(material_path), "candidate": sha256(candidate_path),
                   "generated": sha256(generated_path), "eligible": sha256(eligible_path)},
        "language_mechanism_closed": False,
    }
    result["checks"] = {
        "phase2600_complete": json.loads((P2600 / "analysis/final.json").read_text(encoding="utf-8"))["all_checks_passed"],
        "all_1200_prompts": len(material) == 1200,
        "balanced_100_per_family_language": all(sum(row["family"] == family and row["language"] == language for row in material) == 100
                                                    for family in FAMILIES for language in LANGUAGES),
        "all_600_equal_length_pairs": all(len(rows) == 2 and len(rows[0]["prompt_ids"]) == len(rows[1]["prompt_ids"])
                                           for pair_id in {row["pair_id"] for row in material}
                                           for rows in [[row for row in material if row["pair_id"] == pair_id]]),
        "all_source_spans_recorded": all(row["source_token_positions"] for row in material),
        "all_2400_candidate_sequences": len(candidate) * 2 == 2400,
        "all_1200_greedy": len(generated) == 1200,
        "no_candidate_lists": all(not row["candidate_list_in_prompt"] for row in material),
        "all_splits_present": set(row["split"] for row in material) == {"discovery", "confirmation", "external"},
        "scientific_result_does_not_abort": True,
        "claim_boundary": True,
    }
    result["all_checks_passed"] = all(result["checks"].values())
    save_json(OUT / "analysis/final.json", result)
    append_memo(result)
    correction = "**Phase2601解析合同勘误（append-only）**"
    memo_text = MEMO.read_text(encoding="utf-8-sig")
    if correction not in memo_text and '"greedy_parsed_accuracy": 0.655' in memo_text:
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M")
        with MEMO.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                f"\n\n{correction} [{stamp}] 初版唯一字符串解析器扫描了全部16-token生成，"
                "把首行已经按指令给出正确答案、后续却重复题目的样本误判为双实体歧义；"
                "严格首行指标反而高于解析率暴露了该错误。现统一只解析去除闭合think段后的首个非空答案行，"
                f"修正overall与族级结果为`{json.dumps({'overall': result['overall'], 'by_family_language': result['by_family_language'], 'qualified_groups': result['qualified_groups']}, ensure_ascii=False)}`。"
                "原始generation token未作人工改写，75%行为门不变。\n"
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["all_checks_passed"]:
        raise RuntimeError(result["checks"])


if __name__ == "__main__":
    main()
