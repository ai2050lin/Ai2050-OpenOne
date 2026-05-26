from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from hf_probe_env import encode, first_token_id, load_probe_model, release_loaded
from model_registry import REPO_ROOT, all_model_keys


@dataclass(frozen=True)
class Case:
    category: str
    prompt: str
    target: str
    distractors: tuple[str, ...]
    note: str = ""


def build_cases() -> list[Case]:
    cases: list[Case] = []

    svo = [
        ("the dog chases the cat", "dog", "cat"),
        ("the cat chases the dog", "cat", "dog"),
        ("the wolf hunts the sheep", "wolf", "sheep"),
        ("the sheep follows the wolf", "sheep", "wolf"),
        ("the teacher helps the student", "teacher", "student"),
        ("the student thanks the teacher", "student", "teacher"),
        ("the king rules the city", "king", "city"),
        ("the driver controls the car", "driver", "car"),
    ]
    for sentence, target, wrong in svo:
        cases.append(
            Case(
                "svo_agent",
                f"In the sentence \"{sentence}\", the doer is the",
                f" {target}",
                (f" {wrong}",),
                sentence,
            )
        )

    passive = [
        ("the cat is chased by the dog", "dog", "cat"),
        ("the student is helped by the teacher", "teacher", "student"),
        ("the city is ruled by the king", "king", "city"),
        ("the sheep is hunted by the wolf", "wolf", "sheep"),
        ("the apple is eaten by the child", "child", "apple"),
        ("the bridge is guarded by the soldier", "soldier", "bridge"),
    ]
    for sentence, target, wrong in passive:
        cases.append(
            Case(
                "passive_agent",
                f"In the sentence \"{sentence}\", the doer is the",
                f" {target}",
                (f" {wrong}",),
                sentence,
            )
        )

    negation = [
        ("The door is not open. Is the door open? Answer yes or no:", "no", "yes"),
        ("The light is not on. Is the light on? Answer yes or no:", "no", "yes"),
        ("The person is not happy. Is the person happy? Answer yes or no:", "no", "yes"),
        ("The machine is not working. Is the machine working? Answer yes or no:", "no", "yes"),
        ("There is no reason. Is there a reason? Answer yes or no:", "no", "yes"),
        ("The answer is not correct. Is the answer correct? Answer yes or no:", "no", "yes"),
        ("The place is safe. Is the place safe? Answer yes or no:", "yes", "no"),
        ("The bottle is empty. Is the bottle empty? Answer yes or no:", "yes", "no"),
    ]
    for prompt, target, wrong in negation:
        cases.append(
            Case(
                "negation_yesno",
                prompt,
                f" {target}",
                (f" {wrong}",),
                prompt,
            )
        )

    conditional = [
        ("If it rains, the ground gets wet. It rains. The ground gets", "wet", "dry"),
        ("If the alarm rings, the guard wakes up. The alarm rings. The guard wakes", "up", "down"),
        ("If the switch is off, the lamp is dark. The switch is off. The lamp is", "dark", "bright"),
        ("If the ice melts, the water flows. The ice melts. The water", "flows", "stops"),
        ("If the key turns, the door opens. The key turns. The door", "opens", "closes"),
        ("If the seed grows, the plant appears. The seed grows. The plant", "appears", "vanishes"),
    ]
    for prompt, target, wrong in conditional:
        cases.append(Case("conditional", prompt, f" {target}", (f" {wrong}",)))

    comparison = [
        ("Alice is taller than Bob. The taller person is", "Alice", "Bob"),
        ("Bob is shorter than Alice. The taller person is", "Alice", "Bob"),
        ("The red box is heavier than the blue box. The heavier box is", "red", "blue"),
        ("The blue box is lighter than the red box. The heavier box is", "red", "blue"),
        ("Paris is larger than the village. The larger place is", "Paris", "village"),
        ("The river is longer than the road. The longer thing is the", "river", "road"),
    ]
    for prompt, target, wrong in comparison:
        cases.append(Case("comparison", prompt, f" {target}", (f" {wrong}",)))

    temporal = [
        ("Yesterday, Maria walked to school. The walking happened in the", "past", "future"),
        ("Tomorrow, Maria will walk to school. The walking happens in the", "future", "past"),
        ("Last year, the tree grew. The growing happened in the", "past", "future"),
        ("Next year, the tree will grow. The growing happens in the", "future", "past"),
        ("Now, the child is eating. The eating is happening in the", "present", "past"),
        ("Before dinner, Sam washed his hands. The washing happened", "before", "after"),
    ]
    for prompt, target, wrong in temporal:
        cases.append(Case("temporal", prompt, f" {target}", (f" {wrong}",)))

    recursion = [
        ("The dog that chased the cat was brown. The brown animal was the", "dog", "cat"),
        ("The cat that the dog chased was black. The black animal was the", "cat", "dog"),
        ("The student who thanked the teacher was polite. The polite person was the", "student", "teacher"),
        ("The teacher who helped the student was kind. The kind person was the", "teacher", "student"),
        ("The key that opened the door was small. The small thing was the", "key", "door"),
        ("The city that the king ruled was old. The old thing was the", "city", "king"),
    ]
    for prompt, target, wrong in recursion:
        cases.append(Case("recursive_binding", prompt, f" {target}", (f" {wrong}",)))

    quantifier = [
        ("All birds in this story can fly. A robin is a bird in this story. Can the robin fly? Answer yes or no:", "yes", "no"),
        ("No stones in this story can swim. A pebble is a stone in this story. Can the pebble swim? Answer yes or no:", "no", "yes"),
        ("Some students passed the test. Did at least one student pass? Answer yes or no:", "yes", "no"),
        ("No visitors arrived. Did any visitor arrive? Answer yes or no:", "no", "yes"),
        ("Every key opened a door. Did each key open a door? Answer yes or no:", "yes", "no"),
        ("Few people came to the room. Did many people come? Answer yes or no:", "no", "yes"),
    ]
    for prompt, target, wrong in quantifier:
        cases.append(Case("quantifier", prompt, f" {target}", (f" {wrong}",)))

    translation = [
        ("Translate to Chinese: apple\nChinese:", "苹果", "香蕉"),
        ("Translate to Chinese: cat\nChinese:", "猫", "狗"),
        ("Translate to Chinese: water\nChinese:", "水", "火"),
        ("Translate to Chinese: sun\nChinese:", "太阳", "月亮"),
        ("Translate to English: 苹果\nEnglish:", "apple", "banana"),
        ("Translate to English: 狗\nEnglish:", "dog", "cat"),
        ("Translate to English: 水\nEnglish:", "water", "fire"),
        ("Translate to English: 太阳\nEnglish:", "sun", "moon"),
    ]
    for prompt, target, wrong in translation:
        prefix = "" if target[0] in "苹果猫狗水火太阳月亮" else " "
        wrong_prefix = "" if wrong[0] in "苹果猫狗水火太阳月亮" else " "
        cases.append(Case("translation", prompt, prefix + target, (wrong_prefix + wrong,)))

    return cases


def target_ids(tokenizer: Any, text: str) -> list[int]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        raise ValueError(f"Empty tokenization for {text!r}")
    return [int(x) for x in ids]


@torch.no_grad()
def score_case(loaded: Any, case: Case) -> dict:
    batch = encode(loaded, case.prompt)
    out = loaded.model(**batch, use_cache=False)
    logits = out.logits[0, -1].float().cpu()

    target = target_ids(loaded.tokenizer, case.target)
    distractors = [target_ids(loaded.tokenizer, d) for d in case.distractors]
    target_logit = float(logits[target[0]])
    distractor_logits = [float(logits[d[0]]) for d in distractors]
    best_distractor = max(distractor_logits)
    margin = target_logit - best_distractor
    return {
        "category": case.category,
        "prompt": case.prompt,
        "target": case.target,
        "distractors": list(case.distractors),
        "target_first_token_id": target[0],
        "distractor_first_token_ids": [d[0] for d in distractors],
        "target_first_token_piece": loaded.tokenizer.decode([target[0]]),
        "distractor_first_token_pieces": [loaded.tokenizer.decode([d[0]]) for d in distractors],
        "target_logit": target_logit,
        "best_distractor_logit": best_distractor,
        "margin": margin,
        "correct": margin > 0,
        "note": case.note,
    }


def aggregate(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["category"]].append(row)
    per_category = {}
    for category, vals in sorted(grouped.items()):
        margins = torch.tensor([v["margin"] for v in vals], dtype=torch.float32)
        correct = torch.tensor([1.0 if v["correct"] else 0.0 for v in vals])
        per_category[category] = {
            "n": len(vals),
            "accuracy": float(correct.mean()),
            "mean_margin": float(margins.mean()),
            "median_margin": float(margins.median()),
            "min_margin": float(margins.min()),
        }
    all_margins = torch.tensor([v["margin"] for v in rows], dtype=torch.float32)
    all_correct = torch.tensor([1.0 if v["correct"] else 0.0 for v in rows])
    return {
        "overall": {
            "n": len(rows),
            "accuracy": float(all_correct.mean()),
            "mean_margin": float(all_margins.mean()),
            "median_margin": float(all_margins.median()),
            "min_margin": float(all_margins.min()),
        },
        "per_category": per_category,
    }


def run_model(model_key: str) -> dict:
    loaded = None
    try:
        loaded = load_probe_model(model_key)
        cases = build_cases()
        rows = [score_case(loaded, case) for case in cases]
        return {
            "model": model_key,
            "class": type(loaded.model).__name__,
            "num_cases": len(cases),
            "aggregate": aggregate(rows),
            "cases": rows,
        }
    finally:
        release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("models", nargs="*", default=["qwen3"])
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_language_behavior_anchor"))
    args = parser.parse_args()

    model_keys = all_model_keys() if args.models == ["all"] else args.models
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {"results": []}
    existing = {}
    for path in output_dir.glob("*_language_behavior.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            existing[data["model"]] = {
                "model": data["model"],
                "class": data["class"],
                "num_cases": data["num_cases"],
                "aggregate": data["aggregate"],
            }
        except Exception:
            pass
    for model_key in model_keys:
        print(f"[language-anchor] {model_key}", flush=True)
        result = run_model(model_key)
        existing[model_key] = {
            "model": result["model"],
            "class": result["class"],
            "num_cases": result["num_cases"],
            "aggregate": result["aggregate"],
        }
        (output_dir / f"{model_key}_language_behavior.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2), flush=True)

    summary["results"] = [existing[key] for key in sorted(existing)]
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[language-anchor] summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
