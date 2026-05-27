from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hf_probe_env import load_probe_model, release_loaded
from model_registry import REPO_ROOT, all_model_keys


DATASET_VERSION = "v2_unique_20260526"


@dataclass(frozen=True)
class ChoiceCase:
    case_id: str
    category: str
    prompt: str
    choices: tuple[str, str]
    answer_index: int
    note: str = ""


def en(text: str) -> str:
    return " " + text


def take_unique(cases: list[ChoiceCase], category: str, target_count: int) -> list[ChoiceCase]:
    seen: set[tuple[str, tuple[str, str]]] = set()
    out: list[ChoiceCase] = []
    for case in cases:
        key = (case.prompt, case.choices)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            ChoiceCase(
                f"{category}_{len(out):03d}",
                category,
                case.prompt,
                case.choices,
                case.answer_index,
                case.note,
            )
        )
        if len(out) >= target_count:
            return out
    raise ValueError(f"Could only build {len(out)} unique cases for {category}, need {target_count}")


def build_cases(cases_per_category: int) -> list[ChoiceCase]:
    agents = [
        "dog",
        "cat",
        "wolf",
        "sheep",
        "teacher",
        "student",
        "king",
        "queen",
        "driver",
        "pilot",
        "doctor",
        "nurse",
        "artist",
        "child",
        "farmer",
        "soldier",
        "chef",
        "guard",
        "robot",
        "bird",
    ]
    objects = [
        "car",
        "city",
        "bridge",
        "apple",
        "door",
        "book",
        "stone",
        "boat",
        "lamp",
        "box",
        "garden",
        "river",
        "school",
        "house",
        "key",
        "table",
        "road",
        "room",
        "bell",
        "flag",
    ]
    actions = [
        "chases",
        "follows",
        "helps",
        "watches",
        "guides",
        "finds",
        "moves",
        "guards",
        "pushes",
        "carries",
    ]
    passive_actions = [
        "chased",
        "followed",
        "helped",
        "watched",
        "guided",
        "found",
        "moved",
        "guarded",
        "pushed",
        "carried",
    ]

    cases: list[ChoiceCase] = []

    svo_candidates: list[ChoiceCase] = []
    for subject in agents:
        for obj in agents:
            if obj == subject:
                continue
            for action in actions:
                sentence = f"the {subject} {action} the {obj}"
                svo_candidates.append(
                    ChoiceCase(
                        "",
                        "svo_agent",
                        f'In the sentence "{sentence}", the doer is the',
                        (en(subject), en(obj)),
                        0,
                        sentence,
                    )
                )
                sentence2 = f"the {obj} is the one that the {subject} {action}"
                svo_candidates.append(
                    ChoiceCase(
                        "",
                        "svo_agent",
                        f'In the sentence "{sentence2}", the doer is the',
                        (en(subject), en(obj)),
                        0,
                        sentence2,
                    )
                )
    cases.extend(take_unique(svo_candidates, "svo_agent", cases_per_category))

    passive_candidates: list[ChoiceCase] = []
    for agent in agents:
        for patient in objects:
            for action in passive_actions:
                sentence = f"the {patient} is {action} by the {agent}"
                passive_candidates.append(
                    ChoiceCase(
                        "",
                        "passive_agent",
                        f'In the sentence "{sentence}", the doer is the',
                        (en(agent), en(patient)),
                        0,
                        sentence,
                    )
                )
                sentence2 = f"the {patient} was {action} by the {agent}"
                passive_candidates.append(
                    ChoiceCase(
                        "",
                        "passive_agent",
                        f'In the sentence "{sentence2}", the actor is the',
                        (en(agent), en(patient)),
                        0,
                        sentence2,
                    )
                )
    cases.extend(take_unique(passive_candidates, "passive_agent", cases_per_category))

    adjectives = [
        "open",
        "bright",
        "happy",
        "working",
        "safe",
        "empty",
        "correct",
        "ready",
        "clean",
        "quiet",
        "warm",
        "broken",
        "visible",
        "heavy",
        "fresh",
        "simple",
        "sharp",
        "awake",
        "true",
        "useful",
    ]
    entities = [
        "door",
        "light",
        "person",
        "machine",
        "place",
        "bottle",
        "answer",
        "team",
        "room",
        "street",
        "water",
        "window",
        "mark",
        "bag",
        "bread",
        "plan",
        "knife",
        "child",
        "claim",
        "tool",
    ]
    negation_candidates: list[ChoiceCase] = []
    for entity in entities:
        for adjective in adjectives:
            negation_candidates.extend(
                [
                    ChoiceCase(
                        "",
                        "negation_yesno",
                        f"The {entity} is not {adjective}. Is the {entity} {adjective}? Answer yes or no:",
                        (en("no"), en("yes")),
                        0,
                    ),
                    ChoiceCase(
                        "",
                        "negation_yesno",
                        f"It is false that the {entity} is {adjective}. Is the {entity} {adjective}? Answer yes or no:",
                        (en("no"), en("yes")),
                        0,
                    ),
                    ChoiceCase(
                        "",
                        "negation_yesno",
                        f"The {entity} is {adjective}. Is the {entity} {adjective}? Answer yes or no:",
                        (en("yes"), en("no")),
                        0,
                    ),
                    ChoiceCase(
                        "",
                        "negation_yesno",
                        f"It is true that the {entity} is {adjective}. Is the {entity} {adjective}? Answer yes or no:",
                        (en("yes"), en("no")),
                        0,
                    ),
                ]
            )
    cases.extend(take_unique(negation_candidates, "negation_yesno", cases_per_category))

    condition_pairs = [
        ("it rains", "the ground gets wet", "wet", "dry"),
        ("the alarm rings", "the guard wakes up", "awake", "asleep"),
        ("the switch is off", "the lamp becomes dark", "dark", "bright"),
        ("the ice melts", "the water flows", "flows", "stops"),
        ("the key turns", "the door opens", "open", "closed"),
        ("the seed grows", "the plant appears", "appears", "vanishes"),
        ("the bell rings", "the class starts", "starts", "ends"),
        ("the fire burns", "the room gets warm", "warm", "cold"),
        ("the button is pressed", "the machine starts", "starts", "stops"),
        ("the gate opens", "the car enters", "enters", "leaves"),
        ("the sun rises", "the sky gets bright", "bright", "dark"),
        ("the cup falls", "the water spills", "spills", "stays"),
        ("the child studies", "the score improves", "improves", "drops"),
        ("the wind stops", "the flag hangs still", "still", "moving"),
        ("the battery dies", "the screen turns black", "black", "bright"),
        ("the train arrives", "the passengers board", "board", "wait"),
        ("the cook heats soup", "the soup gets hot", "hot", "cold"),
        ("the farmer plants seeds", "the crops grow", "grow", "disappear"),
        ("the doctor treats the patient", "the patient recovers", "recovers", "worsens"),
        ("the lock breaks", "the door stays open", "open", "locked"),
        ("the snow melts", "the road gets wet", "wet", "dry"),
        ("the phone rings", "the clerk answers", "answers", "ignores"),
        ("the rope snaps", "the bucket falls", "falls", "rises"),
        ("the engine starts", "the car moves", "moves", "parks"),
        ("the cloud covers the sun", "the field gets dim", "dim", "bright"),
    ]
    conditional_candidates: list[ChoiceCase] = []
    for cause, result, target, wrong in condition_pairs:
        conditional_candidates.extend(
            [
                ChoiceCase(
                    "",
                    "conditional",
                    f"If {cause}, {result}. {cause.capitalize()}. The result is",
                    (en(target), en(wrong)),
                    0,
                ),
                ChoiceCase(
                    "",
                    "conditional",
                    f"Whenever {cause}, {result}. Now {cause}. Therefore the result is",
                    (en(target), en(wrong)),
                    0,
                ),
                ChoiceCase(
                    "",
                    "conditional",
                    f"{result.capitalize()} if {cause}. {cause.capitalize()}. So the correct result is",
                    (en(target), en(wrong)),
                    0,
                ),
                ChoiceCase(
                    "",
                    "conditional",
                    f"Rule: when {cause}, {result}. Fact: {cause}. The outcome is",
                    (en(target), en(wrong)),
                    0,
                ),
            ]
        )
    cases.extend(take_unique(conditional_candidates, "conditional", cases_per_category))

    comparison_templates = [
        ("Alice", "Bob", "taller", "shorter", "person"),
        ("red box", "blue box", "heavier", "lighter", "box"),
        ("Paris", "village", "larger", "smaller", "place"),
        ("river", "road", "longer", "shorter", "thing"),
        ("sun", "moon", "brighter", "dimmer", "thing"),
        ("tea", "ice", "hotter", "colder", "thing"),
        ("train", "bicycle", "faster", "slower", "thing"),
        ("tower", "house", "higher", "lower", "thing"),
        ("drum", "bell", "louder", "quieter", "thing"),
        ("lake", "pond", "deeper", "shallower", "place"),
        ("mountain", "hill", "higher", "lower", "place"),
        ("stone", "leaf", "harder", "softer", "thing"),
        ("blanket", "paper", "warmer", "cooler", "thing"),
        ("truck", "cart", "stronger", "weaker", "thing"),
        ("ocean", "pool", "wider", "narrower", "place"),
        ("needle", "stick", "sharper", "duller", "thing"),
        ("library", "room", "quieter", "louder", "place"),
        ("winter", "spring", "colder", "warmer", "time"),
        ("gold", "silver", "costlier", "cheaper", "metal"),
        ("eagle", "sparrow", "larger", "smaller", "bird"),
    ]
    comparison_candidates: list[ChoiceCase] = []
    for a, b, greater, lesser, noun in comparison_templates:
        comparison_candidates.extend(
            [
                ChoiceCase("", "comparison", f"{a} is {greater} than {b}. The {greater} {noun} is", (en(a), en(b)), 0),
                ChoiceCase("", "comparison", f"{b} is {lesser} than {a}. The {greater} {noun} is", (en(a), en(b)), 0),
                ChoiceCase("", "comparison", f"Compared with {b}, {a} is {greater}. The {greater} {noun} is", (en(a), en(b)), 0),
                ChoiceCase("", "comparison", f"Compared with {a}, {b} is {lesser}. The {greater} {noun} is", (en(a), en(b)), 0),
                ChoiceCase("", "comparison", f"Between {a} and {b}, the one that is {greater} is", (en(a), en(b)), 0),
            ]
        )
    cases.extend(take_unique(comparison_candidates, "comparison", cases_per_category))

    event_forms = [
        ("Maria", "walk", "walked", "will walk", "is walking", "walking"),
        ("the tree", "grow", "grew", "will grow", "is growing", "growing"),
        ("Sam", "wash his hands", "washed his hands", "will wash his hands", "is washing his hands", "washing"),
        ("the guard", "open the gate", "opened the gate", "will open the gate", "is opening the gate", "opening"),
        ("the cook", "prepare dinner", "prepared dinner", "will prepare dinner", "is preparing dinner", "preparing"),
        ("the child", "read a book", "read a book", "will read a book", "is reading a book", "reading"),
        ("the train", "leave the station", "left the station", "will leave the station", "is leaving the station", "leaving"),
        ("the singer", "perform", "performed", "will perform", "is performing", "performing"),
        ("the worker", "fix the pipe", "fixed the pipe", "will fix the pipe", "is fixing the pipe", "fixing"),
        ("the farmer", "water the plant", "watered the plant", "will water the plant", "is watering the plant", "watering"),
        ("the pilot", "land the plane", "landed the plane", "will land the plane", "is landing the plane", "landing"),
        ("the nurse", "check the patient", "checked the patient", "will check the patient", "is checking the patient", "checking"),
        ("the artist", "paint the wall", "painted the wall", "will paint the wall", "is painting the wall", "painting"),
        ("the driver", "park the car", "parked the car", "will park the car", "is parking the car", "parking"),
        ("the teacher", "grade the test", "graded the test", "will grade the test", "is grading the test", "grading"),
        ("the robot", "move the box", "moved the box", "will move the box", "is moving the box", "moving"),
        ("the doctor", "write a note", "wrote a note", "will write a note", "is writing a note", "writing"),
        ("the visitor", "enter the room", "entered the room", "will enter the room", "is entering the room", "entering"),
        ("the captain", "raise the flag", "raised the flag", "will raise the flag", "is raising the flag", "raising"),
        ("the baker", "heat the oven", "heated the oven", "will heat the oven", "is heating the oven", "heating"),
    ]
    temporal_candidates: list[ChoiceCase] = []
    for subject, base, past, future, present, gerund in event_forms:
        temporal_candidates.extend(
            [
                ChoiceCase("", "temporal", f"Yesterday, {subject} {past}. The {gerund} happened in the", (en("past"), en("future")), 0),
                ChoiceCase("", "temporal", f"Tomorrow, {subject} {future}. The {gerund} happens in the", (en("future"), en("past")), 0),
                ChoiceCase("", "temporal", f"Right now, {subject} {present}. The {gerund} is happening in the", (en("present"), en("past")), 0),
                ChoiceCase("", "temporal", f"Before dinner, {subject} {past}. The {gerund} happened", (en("before"), en("after")), 0),
                ChoiceCase("", "temporal", f"After lunch, {subject} {past}. The {gerund} happened", (en("after"), en("before")), 0),
                ChoiceCase("", "temporal", f"Last week, {subject} {past}. The {gerund} happened in the", (en("past"), en("future")), 0),
            ]
        )
    cases.extend(take_unique(temporal_candidates, "temporal", cases_per_category))

    colors = [
        "brown",
        "black",
        "polite",
        "kind",
        "small",
        "old",
        "young",
        "quick",
        "quiet",
        "brave",
        "green",
        "silver",
        "round",
        "soft",
        "large",
        "clean",
        "warm",
        "bright",
        "thin",
        "strong",
    ]
    recursive_candidates: list[ChoiceCase] = []
    for a in agents:
        for b in agents:
            if a == b:
                continue
            for action in passive_actions:
                adj = colors[(len(recursive_candidates) + len(a) + len(b)) % len(colors)]
                recursive_candidates.extend(
                    [
                        ChoiceCase("", "recursive_binding", f"The {a} that {action} the {b} was {adj}. The {adj} one was the", (en(a), en(b)), 0),
                        ChoiceCase("", "recursive_binding", f"The {b} that the {a} {action} was {adj}. The {adj} one was the", (en(b), en(a)), 0),
                        ChoiceCase("", "recursive_binding", f"The {a} near the {b} was {adj}. The {adj} one was the", (en(a), en(b)), 0),
                        ChoiceCase("", "recursive_binding", f"The {b} beside the {a} was {adj}. The {adj} one was the", (en(b), en(a)), 0),
                    ]
                )
                if len(recursive_candidates) >= cases_per_category * 2:
                    break
            if len(recursive_candidates) >= cases_per_category * 2:
                break
        if len(recursive_candidates) >= cases_per_category * 2:
            break
    cases.extend(take_unique(recursive_candidates, "recursive_binding", cases_per_category))

    animal_pairs = [
        ("bird", "robin", "fly"),
        ("fish", "salmon", "swim"),
        ("dog", "puppy", "bark"),
        ("student", "learner", "study"),
        ("key", "small key", "open a door"),
        ("stone", "pebble", "float"),
        ("visitor", "guest", "arrive"),
        ("worker", "builder", "work"),
        ("fruit", "apple", "grow on a plant"),
        ("vehicle", "car", "move"),
        ("tool", "hammer", "break glass"),
        ("plant", "rose", "need water"),
        ("animal", "horse", "sleep"),
        ("book", "novel", "contain pages"),
        ("device", "phone", "use power"),
        ("boat", "canoe", "carry people"),
        ("insect", "ant", "crawl"),
        ("musician", "pianist", "play music"),
        ("container", "box", "hold items"),
        ("machine", "printer", "make noise"),
    ]
    quantifier_candidates: list[ChoiceCase] = []
    for group, item, predicate in animal_pairs:
        plural = group + "s"
        quantifier_candidates.extend(
            [
                ChoiceCase("", "quantifier", f"All {plural} in this story can {predicate}. A {item} is a {group} in this story. Can the {item} {predicate}? Answer yes or no:", (en("yes"), en("no")), 0),
                ChoiceCase("", "quantifier", f"No {plural} in this story can {predicate}. A {item} is a {group} in this story. Can the {item} {predicate}? Answer yes or no:", (en("no"), en("yes")), 0),
                ChoiceCase("", "quantifier", f"Some {plural} passed the test. Did at least one {group} pass? Answer yes or no:", (en("yes"), en("no")), 0),
                ChoiceCase("", "quantifier", f"No {plural} arrived. Did any {group} arrive? Answer yes or no:", (en("no"), en("yes")), 0),
                ChoiceCase("", "quantifier", f"Few {plural} came to the room. Did many {plural} come? Answer yes or no:", (en("no"), en("yes")), 0),
                ChoiceCase("", "quantifier", f"Not all {plural} were selected. Were all {plural} selected? Answer yes or no:", (en("no"), en("yes")), 0),
                ChoiceCase("", "quantifier", f"Exactly one {group} entered. Did at least one {group} enter? Answer yes or no:", (en("yes"), en("no")), 0),
                ChoiceCase("", "quantifier", f"At least two {plural} arrived. Did any {group} arrive? Answer yes or no:", (en("yes"), en("no")), 0),
            ]
        )
    cases.extend(take_unique(quantifier_candidates, "quantifier", cases_per_category))

    translations = [
        ("apple", "苹果", "香蕉"),
        ("banana", "香蕉", "苹果"),
        ("cat", "猫", "狗"),
        ("dog", "狗", "猫"),
        ("water", "水", "火"),
        ("fire", "火", "水"),
        ("sun", "太阳", "月亮"),
        ("moon", "月亮", "太阳"),
        ("mountain", "山", "河"),
        ("river", "河", "山"),
        ("book", "书", "笔"),
        ("pen", "笔", "书"),
        ("door", "门", "窗"),
        ("window", "窗", "门"),
        ("tree", "树", "花"),
        ("flower", "花", "树"),
        ("road", "路", "桥"),
        ("bridge", "桥", "路"),
        ("sky", "天空", "地面"),
        ("ground", "地面", "天空"),
        ("hand", "手", "脚"),
        ("foot", "脚", "手"),
        ("eye", "眼睛", "耳朵"),
        ("ear", "耳朵", "眼睛"),
        ("red", "红色", "蓝色"),
        ("blue", "蓝色", "红色"),
        ("hot", "热", "冷"),
        ("cold", "冷", "热"),
        ("big", "大", "小"),
        ("small", "小", "大"),
        ("school", "学校", "医院"),
        ("hospital", "医院", "学校"),
        ("teacher", "老师", "学生"),
        ("student", "学生", "老师"),
        ("city", "城市", "村庄"),
        ("village", "村庄", "城市"),
        ("king", "国王", "女王"),
        ("queen", "女王", "国王"),
        ("car", "汽车", "火车"),
        ("train", "火车", "汽车"),
        ("bird", "鸟", "鱼"),
        ("fish", "鱼", "鸟"),
        ("food", "食物", "水"),
        ("milk", "牛奶", "茶"),
        ("tea", "茶", "牛奶"),
        ("house", "房子", "车"),
        ("stone", "石头", "木头"),
        ("wood", "木头", "石头"),
        ("rain", "雨", "雪"),
        ("snow", "雪", "雨"),
        ("computer", "电脑", "手机"),
        ("phone", "手机", "电脑"),
        ("paper", "纸", "布"),
        ("cloth", "布", "纸"),
        ("salt", "盐", "糖"),
        ("sugar", "糖", "盐"),
        ("coffee", "咖啡", "茶"),
        ("family", "家庭", "公司"),
        ("company", "公司", "家庭"),
        ("music", "音乐", "噪音"),
    ]
    translation_candidates: list[ChoiceCase] = []
    for idx, (english, chinese, wrong) in enumerate(translations):
        wrong_english = translations[(idx + 1) % len(translations)][0]
        translation_candidates.extend(
            [
                ChoiceCase("", "translation", f"Translate to Chinese: {english}\nChinese:", (chinese, wrong), 0),
                ChoiceCase("", "translation", f"English word: {english}\nChinese meaning:", (chinese, wrong), 0),
                ChoiceCase("", "translation", f"Translate to English: {chinese}\nEnglish:", (en(english), en(wrong_english)), 0),
                ChoiceCase("", "translation", f"Chinese word: {chinese}\nEnglish meaning:", (en(english), en(wrong_english)), 0),
            ]
        )
    cases.extend(take_unique(translation_candidates, "translation", cases_per_category))

    validate_case_uniqueness(cases, cases_per_category)
    return cases


def validate_case_uniqueness(cases: list[ChoiceCase], cases_per_category: int) -> None:
    grouped: dict[str, list[ChoiceCase]] = defaultdict(list)
    for case in cases:
        grouped[case.category].append(case)
    errors = []
    for category, rows in sorted(grouped.items()):
        unique = {(row.prompt, row.choices) for row in rows}
        if len(rows) != cases_per_category:
            errors.append(f"{category}: n={len(rows)} expected={cases_per_category}")
        if len(unique) != len(rows):
            errors.append(f"{category}: unique_prompt_choices={len(unique)} n={len(rows)}")
    if errors:
        raise ValueError("Case uniqueness validation failed:\n" + "\n".join(errors))

    expected_categories = {
        "svo_agent",
        "passive_agent",
        "negation_yesno",
        "conditional",
        "comparison",
        "temporal",
        "recursive_binding",
        "quantifier",
        "translation",
    }
    missing = expected_categories.difference(grouped)
    extra = set(grouped).difference(expected_categories)
    if missing or extra:
        raise ValueError(f"Category mismatch: missing={sorted(missing)} extra={sorted(extra)}")

    return


def token_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Empty tokenization for {text!r}")
    return [int(x) for x in ids]


def make_candidate_records(loaded: Any, cases: list[ChoiceCase]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        prompt_ids = token_ids(loaded.tokenizer, case.prompt)
        for choice_index, choice in enumerate(case.choices):
            choice_ids = token_ids(loaded.tokenizer, choice)
            input_ids = prompt_ids + choice_ids
            records.append(
                {
                    "case_index": case_index,
                    "choice_index": choice_index,
                    "input_ids": input_ids,
                    "choice_start": len(prompt_ids),
                    "choice_ids": choice_ids,
                }
            )
    return records


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


@torch.inference_mode()
def score_candidates(
    loaded: Any,
    records: list[dict[str, Any]],
    batch_size: int,
    progress_every: int,
) -> list[dict[str, float]]:
    pad_id = loaded.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = loaded.tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0

    scores: list[dict[str, float]] = []
    num_batches = (len(records) + batch_size - 1) // batch_size
    for batch_index, start in enumerate(range(0, len(records), batch_size), start=1):
        batch_records = records[start : start + batch_size]
        max_len = max(len(r["input_ids"]) for r in batch_records)
        input_ids = torch.full((len(batch_records), max_len), int(pad_id), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, record in enumerate(batch_records):
            ids = torch.tensor(record["input_ids"], dtype=torch.long)
            input_ids[row, : len(ids)] = ids
            attention_mask[row, : len(ids)] = 1
        input_ids = input_ids.to(loaded.input_device)
        attention_mask = attention_mask.to(loaded.input_device)
        out = loaded.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits.float()

        for row, record in enumerate(batch_records):
            choice_start = int(record["choice_start"])
            choice_ids = record["choice_ids"]
            logprob = 0.0
            first_logprob = None
            for offset, token_id in enumerate(choice_ids):
                pred_pos = choice_start + offset - 1
                token_logits = logits[row, pred_pos]
                token_logprob = float(token_logits[token_id] - torch.logsumexp(token_logits, dim=-1))
                if offset == 0:
                    first_logprob = token_logprob
                logprob += token_logprob
            scores.append(
                {
                    "case_index": int(record["case_index"]),
                    "choice_index": int(record["choice_index"]),
                    "full_logprob": logprob,
                    "mean_logprob": logprob / max(1, len(choice_ids)),
                    "first_token_logprob": float(first_logprob if first_logprob is not None else logprob),
                    "num_choice_tokens": len(choice_ids),
                }
            )
        del out, logits, input_ids, attention_mask
        if progress_every > 0 and (batch_index == 1 or batch_index % progress_every == 0 or batch_index == num_batches):
            print(
                f"[systematic-language] {loaded.key} batch {batch_index}/{num_batches}",
                flush=True,
            )
    return scores


def score_cases(
    loaded: Any,
    cases: list[ChoiceCase],
    batch_size: int,
    progress_every: int,
) -> list[dict[str, Any]]:
    records = make_candidate_records(loaded, cases)
    candidate_scores = score_candidates(
        loaded,
        records,
        batch_size=batch_size,
        progress_every=progress_every,
    )
    by_case: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for score in candidate_scores:
        by_case[int(score["case_index"])].append(score)

    rows: list[dict[str, Any]] = []
    for case_index, case in enumerate(cases):
        scores = sorted(by_case[case_index], key=lambda x: int(x["choice_index"]))
        answer = scores[case.answer_index]
        distractors = [s for s in scores if int(s["choice_index"]) != case.answer_index]
        best_full = max(d["full_logprob"] for d in distractors)
        best_mean = max(d["mean_logprob"] for d in distractors)
        best_first = max(d["first_token_logprob"] for d in distractors)
        full_margin = answer["full_logprob"] - best_full
        mean_margin = answer["mean_logprob"] - best_mean
        first_margin = answer["first_token_logprob"] - best_first
        rows.append(
            {
                "case_id": case.case_id,
                "category": case.category,
                "prompt": case.prompt,
                "choices": list(case.choices),
                "answer_index": case.answer_index,
                "answer": case.choices[case.answer_index],
                "note": case.note,
                "choice_scores": scores,
                "full_margin": full_margin,
                "mean_margin": mean_margin,
                "first_token_margin": first_margin,
                "full_correct": full_margin > 0,
                "mean_correct": mean_margin > 0,
                "first_token_correct": first_margin > 0,
            }
        )
    return rows


def group_cases_by_category(cases: list[ChoiceCase]) -> dict[str, list[ChoiceCase]]:
    grouped: dict[str, list[ChoiceCase]] = defaultdict(list)
    for case in cases:
        grouped[case.category].append(case)
    return dict(grouped)


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)

    def metric(vals: list[dict[str, Any]], prefix: str) -> dict[str, float | int]:
        margins = torch.tensor([float(v[f"{prefix}_margin"]) for v in vals], dtype=torch.float32)
        correct = torch.tensor([1.0 if v[f"{prefix}_correct"] else 0.0 for v in vals])
        return {
            "n": len(vals),
            "accuracy": float(correct.mean()),
            "mean_margin": float(margins.mean()),
            "median_margin": float(margins.median()),
            "min_margin": float(margins.min()),
        }

    per_category = {}
    for category, vals in sorted(grouped.items()):
        per_category[category] = {
            "full": metric(vals, "full"),
            "mean": metric(vals, "mean"),
            "first_token": metric(vals, "first_token"),
            "first_full_disagreements": sum(
                1 for v in vals if bool(v["full_correct"]) != bool(v["first_token_correct"])
            ),
        }
    return {
        "overall": {
            "full": metric(rows, "full"),
            "mean": metric(rows, "mean"),
            "first_token": metric(rows, "first_token"),
            "first_full_disagreements": sum(
                1 for v in rows if bool(v["full_correct"]) != bool(v["first_token_correct"])
            ),
        },
        "per_category": per_category,
    }


def category_checkpoint_path(output_dir: Path, model_key: str, category: str) -> Path:
    return output_dir / "checkpoints" / model_key / f"{category}.json"


def load_category_checkpoint(
    path: Path,
    model_key: str,
    category: str,
    cases_per_category: int,
    allow_partial: bool = False,
) -> list[dict[str, Any]] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("model") != model_key:
        return None
    if data.get("category") != category:
        return None
    if data.get("dataset_version") != DATASET_VERSION:
        return None
    if int(data.get("cases_per_category", -1)) != cases_per_category:
        return None
    rows = data.get("cases")
    if not isinstance(rows, list):
        return None
    if allow_partial:
        if len(rows) > cases_per_category:
            return None
    elif len(rows) != cases_per_category:
        return None
    return rows


def make_result(
    loaded: Any,
    rows: list[dict[str, Any]],
    cases_per_category: int,
    batch_size: int,
    complete: bool,
) -> dict[str, Any]:
    return {
        "model": loaded.key,
        "dataset_version": DATASET_VERSION,
        "class": type(loaded.model).__name__,
        "num_cases": len(rows),
        "cases_per_category": cases_per_category,
        "batch_size": batch_size,
        "complete": complete,
        "aggregate": aggregate(rows),
        "cases": rows,
    }


def write_model_outputs(
    output_dir: Path,
    result: dict[str, Any],
    complete: bool,
) -> None:
    model_key = str(result["model"])
    suffix = "systematic_language.json" if complete else "systematic_language.partial.json"
    out_path = output_dir / f"{model_key}_{suffix}"
    atomic_write_json(out_path, result)


def run_model(
    model_key: str,
    cases_per_category: int,
    batch_size: int,
    progress_every: int,
    output_dir: Path,
    resume: bool,
    categories: set[str] | None,
    case_chunk_size: int,
    release_model: bool,
) -> dict[str, Any]:
    loaded = None
    try:
        loaded = load_probe_model(model_key)
        grouped_cases = group_cases_by_category(build_cases(cases_per_category))
        if categories is not None:
            unknown = sorted(categories.difference(grouped_cases))
            if unknown:
                valid = ", ".join(sorted(grouped_cases))
                raise SystemExit(f"Unknown categories {unknown}. Valid categories: {valid}")
            grouped_cases = {
                category: cases
                for category, cases in grouped_cases.items()
                if category in categories
            }
        rows_by_category: dict[str, list[dict[str, Any]]] = {}

        for category, category_cases in grouped_cases.items():
            checkpoint_path = category_checkpoint_path(output_dir, model_key, category)
            cached_rows = None
            if resume:
                cached_rows = load_category_checkpoint(
                    checkpoint_path,
                    model_key=model_key,
                    category=category,
                    cases_per_category=cases_per_category,
                )
            if cached_rows is not None:
                print(f"[systematic-language] {model_key} resume {category}", flush=True)
                rows_by_category[category] = cached_rows
            else:
                category_rows = []
                if resume:
                    category_rows = load_category_checkpoint(
                        checkpoint_path,
                        model_key=model_key,
                        category=category,
                        cases_per_category=cases_per_category,
                        allow_partial=True,
                    ) or []
                    if category_rows:
                        print(
                            f"[systematic-language] {model_key} resume partial "
                            f"{category} {len(category_rows)}/{cases_per_category}",
                            flush=True,
                        )

                chunk_size = case_chunk_size if case_chunk_size > 0 else len(category_cases)
                while len(category_rows) < len(category_cases):
                    start = len(category_rows)
                    end = min(start + chunk_size, len(category_cases))
                    print(
                        f"[systematic-language] {model_key} category {category} "
                        f"cases {start}:{end}/{len(category_cases)}",
                        flush=True,
                    )
                    category_rows.extend(
                        score_cases(
                            loaded,
                            category_cases[start:end],
                            batch_size=batch_size,
                            progress_every=progress_every,
                        )
                    )
                    checkpoint = {
                        "model": model_key,
                        "dataset_version": DATASET_VERSION,
                        "class": type(loaded.model).__name__,
                        "category": category,
                        "cases_per_category": cases_per_category,
                        "batch_size": batch_size,
                        "case_chunk_size": case_chunk_size,
                        "complete": len(category_rows) == len(category_cases),
                        "num_cases": len(category_rows),
                        "aggregate": aggregate(category_rows),
                        "cases": category_rows,
                    }
                    atomic_write_json(checkpoint_path, checkpoint)

                rows_by_category[category] = category_rows

            partial_rows = [
                row
                for key in grouped_cases
                for row in rows_by_category.get(key, [])
            ]
            partial_result = make_result(
                loaded,
                partial_rows,
                cases_per_category=cases_per_category,
                batch_size=batch_size,
                complete=len(rows_by_category) == len(grouped_cases),
            )
            write_model_outputs(
                output_dir,
                partial_result,
                complete=False,
            )

        rows = [
            row
            for category in grouped_cases
            for row in rows_by_category[category]
        ]
        result = make_result(
            loaded,
            rows,
            cases_per_category=cases_per_category,
            batch_size=batch_size,
            complete=True,
        )
        write_model_outputs(output_dir, result, complete=True)
        return result
    finally:
        if release_model:
            release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=["qwen3"])
    parser.add_argument("--cases-per-category", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--case-chunk-size", type=int, default=25)
    parser.add_argument(
        "--hard-exit-after-model",
        action="store_true",
        help=(
            "For driver-stability debugging: only with one model. Write outputs, "
            "skip explicit CUDA cleanup, then os._exit(0) to avoid hangs during "
            "model release."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "gpt5_systematic_language_benchmark"),
    )
    args = parser.parse_args()

    model_keys = all_model_keys() if args.models == ["all"] else args.models
    if args.hard_exit_after_model and len(model_keys) != 1:
        raise SystemExit("--hard-exit-after-model requires exactly one model")
    categories = set(args.categories) if args.categories else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = {}
    for path in output_dir.glob("*_systematic_language.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            existing[data["model"]] = {
                "model": data["model"],
                "dataset_version": data.get("dataset_version"),
                "class": data["class"],
                "num_cases": data["num_cases"],
                "cases_per_category": data["cases_per_category"],
                "batch_size": data["batch_size"],
                "complete": data.get("complete", True),
                "aggregate": data["aggregate"],
            }
        except Exception:
            pass

    for model_key in model_keys:
        print(f"[systematic-language] {model_key}", flush=True)
        result = run_model(
            model_key,
            args.cases_per_category,
            args.batch_size,
            args.progress_every,
            output_dir=output_dir,
            resume=args.resume,
            categories=categories,
            case_chunk_size=args.case_chunk_size,
            release_model=not args.hard_exit_after_model,
        )
        existing[model_key] = {
            "model": result["model"],
            "dataset_version": result["dataset_version"],
            "class": result["class"],
            "num_cases": result["num_cases"],
            "cases_per_category": result["cases_per_category"],
            "batch_size": result["batch_size"],
            "complete": result["complete"],
            "aggregate": result["aggregate"],
        }
        print(json.dumps(result["aggregate"]["overall"], ensure_ascii=False, indent=2), flush=True)

    summary = {"results": [existing[key] for key in sorted(existing)]}
    summary_path = output_dir / "summary.json"
    atomic_write_json(summary_path, summary)
    print(f"[systematic-language] summary: {summary_path}")
    if args.hard_exit_after_model:
        os._exit(0)


if __name__ == "__main__":
    main()
