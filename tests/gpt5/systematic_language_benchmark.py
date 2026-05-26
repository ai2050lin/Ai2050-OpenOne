from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hf_probe_env import load_probe_model, release_loaded
from model_registry import REPO_ROOT, all_model_keys


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


def build_cases(cases_per_category: int) -> list[ChoiceCase]:
    cases: list[ChoiceCase] = []

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

    for i in range(cases_per_category):
        subject = agents[i % len(agents)]
        obj = agents[(i * 7 + 3) % len(agents)]
        if obj == subject:
            obj = agents[(i * 7 + 4) % len(agents)]
        action = actions[(i * 3) % len(actions)]
        sentence = f"the {subject} {action} the {obj}"
        cases.append(
            ChoiceCase(
                f"svo_agent_{i:03d}",
                "svo_agent",
                f'In the sentence "{sentence}", the doer is the',
                (en(subject), en(obj)),
                0,
                sentence,
            )
        )

    for i in range(cases_per_category):
        agent = agents[(i * 5 + 1) % len(agents)]
        patient = objects[(i * 7 + 2) % len(objects)]
        action = passive_actions[(i * 3 + 1) % len(passive_actions)]
        sentence = f"the {patient} is {action} by the {agent}"
        cases.append(
            ChoiceCase(
                f"passive_agent_{i:03d}",
                "passive_agent",
                f'In the sentence "{sentence}", the doer is the',
                (en(agent), en(patient)),
                0,
                sentence,
            )
        )

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
    for i in range(cases_per_category):
        entity = entities[i % len(entities)]
        adjective = adjectives[(i * 5) % len(adjectives)]
        is_negative = i % 2 == 0
        if is_negative:
            prompt = f"The {entity} is not {adjective}. Is the {entity} {adjective}? Answer yes or no:"
            choices = (en("no"), en("yes"))
        else:
            prompt = f"The {entity} is {adjective}. Is the {entity} {adjective}? Answer yes or no:"
            choices = (en("yes"), en("no"))
        cases.append(ChoiceCase(f"negation_yesno_{i:03d}", "negation_yesno", prompt, choices, 0))

    condition_pairs = [
        ("it rains", "ground gets wet", "dry"),
        ("the alarm rings", "guard wakes up", "asleep"),
        ("the switch is off", "lamp becomes dark", "bright"),
        ("the ice melts", "water flows", "stops"),
        ("the key turns", "door opens", "closes"),
        ("the seed grows", "plant appears", "vanishes"),
        ("the bell rings", "class starts", "ends"),
        ("the fire burns", "room gets warm", "cold"),
        ("the button is pressed", "machine starts", "stops"),
        ("the gate opens", "car enters", "leaves"),
        ("the sun rises", "sky gets bright", "dark"),
        ("the cup falls", "water spills", "stays"),
        ("the child studies", "score improves", "drops"),
        ("the wind stops", "flag hangs still", "moves"),
        ("the battery dies", "screen turns black", "bright"),
        ("the train arrives", "passengers board", "wait"),
        ("the cook heats soup", "soup gets hot", "cold"),
        ("the farmer plants seeds", "crops grow", "disappear"),
        ("the doctor treats the patient", "patient recovers", "worsens"),
        ("the lock breaks", "door stays open", "locked"),
    ]
    for i in range(cases_per_category):
        cause, result, wrong = condition_pairs[i % len(condition_pairs)]
        target = result.split()[-1]
        prompt = f"If {cause}, the {result}. {cause.capitalize()}. The result is"
        cases.append(ChoiceCase(f"conditional_{i:03d}", "conditional", prompt, (en(target), en(wrong)), 0))

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
    ]
    for i in range(cases_per_category):
        a, b, greater, lesser, noun = comparison_templates[i % len(comparison_templates)]
        direct = (i // len(comparison_templates)) % 2 == 0
        if direct:
            prompt = f"{a} is {greater} than {b}. The {greater} {noun} is"
        else:
            prompt = f"{b} is {lesser} than {a}. The {greater} {noun} is"
        cases.append(ChoiceCase(f"comparison_{i:03d}", "comparison", prompt, (en(a), en(b)), 0))

    events = [
        ("Maria walked to school", "walking"),
        ("the tree grew", "growing"),
        ("Sam washed his hands", "washing"),
        ("the guard opened the gate", "opening"),
        ("the cook prepared dinner", "preparing"),
        ("the child reads a book", "reading"),
        ("the train leaves the station", "leaving"),
        ("the singer performs", "performing"),
        ("the worker fixes the pipe", "fixing"),
        ("the farmer waters the plant", "watering"),
    ]
    for i in range(cases_per_category):
        event, gerund = events[i % len(events)]
        mode = i % 5
        if mode == 0:
            prompt = f"Yesterday, {event}. The {gerund} happened in the"
            choices = (en("past"), en("future"))
        elif mode == 1:
            prompt = f"Tomorrow, {event}. The {gerund} happens in the"
            choices = (en("future"), en("past"))
        elif mode == 2:
            prompt = f"Now, {event}. The {gerund} is happening in the"
            choices = (en("present"), en("past"))
        elif mode == 3:
            prompt = f"Before dinner, {event}. The {gerund} happened"
            choices = (en("before"), en("after"))
        else:
            prompt = f"After lunch, {event}. The {gerund} happened"
            choices = (en("after"), en("before"))
        cases.append(ChoiceCase(f"temporal_{i:03d}", "temporal", prompt, choices, 0))

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
    for i in range(cases_per_category):
        a = agents[i % len(agents)]
        b = agents[(i * 7 + 2) % len(agents)]
        if a == b:
            b = agents[(i * 7 + 3) % len(agents)]
        action = passive_actions[(i * 3) % len(passive_actions)]
        adj = colors[(i * 5) % len(colors)]
        subject_relative = i % 2 == 0
        if subject_relative:
            prompt = f"The {a} that {action} the {b} was {adj}. The {adj} one was the"
            choices = (en(a), en(b))
        else:
            prompt = f"The {b} that the {a} {action} was {adj}. The {adj} one was the"
            choices = (en(b), en(a))
        cases.append(ChoiceCase(f"recursive_binding_{i:03d}", "recursive_binding", prompt, choices, 0))

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
    ]
    for i in range(cases_per_category):
        group, item, predicate = animal_pairs[i % len(animal_pairs)]
        mode = i % 5
        if mode == 0:
            prompt = f"All {group}s in this story can {predicate}. A {item} is a {group} in this story. Can the {item} {predicate}? Answer yes or no:"
            choices = (en("yes"), en("no"))
        elif mode == 1:
            prompt = f"No {group}s in this story can {predicate}. A {item} is a {group} in this story. Can the {item} {predicate}? Answer yes or no:"
            choices = (en("no"), en("yes"))
        elif mode == 2:
            prompt = f"Some {group}s passed the test. Did at least one {group} pass? Answer yes or no:"
            choices = (en("yes"), en("no"))
        elif mode == 3:
            prompt = f"No {group}s arrived. Did any {group} arrive? Answer yes or no:"
            choices = (en("no"), en("yes"))
        else:
            prompt = f"Few {group}s came to the room. Did many {group}s come? Answer yes or no:"
            choices = (en("no"), en("yes"))
        cases.append(ChoiceCase(f"quantifier_{i:03d}", "quantifier", prompt, choices, 0))

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
    ]
    for i in range(cases_per_category):
        english, chinese, wrong = translations[i % len(translations)]
        to_chinese = i % 2 == 0
        if to_chinese:
            prompt = f"Translate to Chinese: {english}\nChinese:"
            choices = (chinese, wrong)
        else:
            prompt = f"Translate to English: {chinese}\nEnglish:"
            choices = (en(english), en(translations[(i + 1) % len(translations)][0]))
        cases.append(ChoiceCase(f"translation_{i:03d}", "translation", prompt, choices, 0))

    return cases


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
    if int(data.get("cases_per_category", -1)) != cases_per_category:
        return None
    rows = data.get("cases")
    if not isinstance(rows, list) or len(rows) != cases_per_category:
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
) -> dict[str, Any]:
    loaded = None
    try:
        loaded = load_probe_model(model_key)
        grouped_cases = group_cases_by_category(build_cases(cases_per_category))
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
                print(f"[systematic-language] {model_key} category {category}", flush=True)
                category_rows = score_cases(
                    loaded,
                    category_cases,
                    batch_size=batch_size,
                    progress_every=progress_every,
                )
                rows_by_category[category] = category_rows
                checkpoint = {
                    "model": model_key,
                    "class": type(loaded.model).__name__,
                    "category": category,
                    "cases_per_category": cases_per_category,
                    "batch_size": batch_size,
                    "num_cases": len(category_rows),
                    "aggregate": aggregate(category_rows),
                    "cases": category_rows,
                }
                atomic_write_json(checkpoint_path, checkpoint)

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
        release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=["qwen3"])
    parser.add_argument("--cases-per-category", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results" / "gpt5_systematic_language_benchmark"),
    )
    args = parser.parse_args()

    model_keys = all_model_keys() if args.models == ["all"] else args.models
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    existing = {}
    for path in output_dir.glob("*_systematic_language.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            existing[data["model"]] = {
                "model": data["model"],
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
        )
        existing[model_key] = {
            "model": result["model"],
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


if __name__ == "__main__":
    main()
