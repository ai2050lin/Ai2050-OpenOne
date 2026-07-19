#!/usr/bin/env python3
"""Phase 976: frozen external confirmation of Qwen3 final/no-think mode markers.

Phase 975 selected the Qwen3 empty ``<think>...</think>`` mode span and layer
L34 using the old 160 short-completion items.  This script never re-selects a
token group, layer, corruption method, or threshold.  It tests the frozen
candidate on 80 newly authored instructions across eight task/mode families.

Primary controls preserve token count, attention mask, and absolute positions:
the mode-span embeddings are either zeroed or replaced by a newline embedding.
Natural generation is greedy, has no EOS bias, and the intervention is applied
only once during prompt prefill.  A frozen L34 clean-residual rescue is also
tested on the completed-period state.
"""
from __future__ import annotations

import gc
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, load_model, release_model
from phase951_protocol_atlas import ensure_dir
from phase966_natural_stop import log
from phase973_conditional_trajectory import get_eos_ids
from phase975_protocol_causal_transfer import (
    chat_template_text, neutral_id, protocol_manifest, run_snapshot, token_ids,
)


OUT = Path("tests/glm5/result/phase976_qwen_mode_external")
MODEL = "qwen3"
FROZEN_GROUP = "mode_marker"
FROZEN_LAYER = 34


def build_external_dataset():
    groups = {
        "new_fact": [
            ("Name the capital of Canada.", "Ottawa", ["Ottawa"]),
            ("Give the chemical symbol for silver.", "Ag", ["Ag"]),
            ("Which ocean is the largest on Earth?", "the Pacific Ocean", ["Pacific"]),
            ("Who wrote Pride and Prejudice?", "Jane Austen", ["Jane Austen"]),
            ("At sea level, water boils at how many degrees Celsius?", "100 degrees Celsius", ["100"]),
            ("What is the smallest prime number?", "two", ["two", "2"]),
            ("How many bones are in a typical adult human skeleton?", "206", ["206"]),
            ("Which planet is famous for its visible rings?", "Saturn", ["Saturn"]),
            ("What is the main currency of the United Kingdom?", "the pound sterling", ["pound", "sterling"]),
            ("What is the largest living mammal?", "the blue whale", ["blue whale"]),
        ],
        "yes_no": [
            ("Answer only yes or no: Is seven a prime number?", "yes", ["yes"]),
            ("Answer only yes or no: Does a square have three sides?", "no", ["no"]),
            ("Answer only yes or no: Are whales mammals?", "yes", ["yes"]),
            ("Answer only yes or no: Is nine an even number?", "no", ["no"]),
            ("Answer only yes or no: Is the chemical formula of water H2O?", "yes", ["yes"]),
            ("Answer only yes or no: Is Paris in Italy?", "no", ["no"]),
            ("Answer only yes or no: In Euclidean geometry, do a triangle's angles sum to 180 degrees?", "yes", ["yes"]),
            ("Answer only yes or no: Are bats birds?", "no", ["no"]),
            ("Answer only yes or no: Is zero a positive number?", "no", ["no"]),
            ("Answer only yes or no: Is oxygen a chemical element?", "yes", ["yes"]),
        ],
        "arithmetic": [
            ("Return only the result: 17 + 8.", "25", ["25"]),
            ("Return only the result: 12 times 7.", "84", ["84"]),
            ("Return only the result: 81 divided by 9.", "9", ["9"]),
            ("Return only the result: 15 minus 28.", "-13", ["-13"]),
            ("Return only the result: 6 squared.", "36", ["36"]),
            ("Return only the result: the average of 10 and 20.", "15", ["15"]),
            ("Return only the result: three quarters of 20.", "15", ["15"]),
            ("Return only the result: the perimeter of a square with side 9.", "36", ["36"]),
            ("Return only the result: 2 to the fifth power.", "32", ["32"]),
            ("Return only the result: the square root of 144.", "12", ["12"]),
        ],
        "translation": [
            ("Translate 'apple' into French. Give one word.", "pomme", ["pomme"]),
            ("Translate 'chair' into Spanish. Give one word.", "silla", ["silla"]),
            ("Translate 'milk' into German. Give one word.", "Milch", ["Milch"]),
            ("Translate 'window' into Italian. Give one word.", "finestra", ["finestra"]),
            ("Translate 'river' into French. Give one word.", "rivière", ["rivière", "riviere"]),
            ("Translate 'black' into Spanish. Give one word.", "negro", ["negro"]),
            ("Translate 'child' into German. Give one word.", "Kind", ["Kind"]),
            ("Translate 'city' into Italian. Give one word.", "città", ["città", "citta"]),
            ("Translate 'flower' into French. Give one word.", "fleur", ["fleur"]),
            ("Translate 'door' into Spanish. Give one word.", "puerta", ["puerta"]),
        ],
        "definition": [
            ("Define an adjective in one short clause.", "a word that describes a noun", ["describes a noun", "modifies a noun"]),
            ("Define osmosis in one short clause.", "the movement of water across a semipermeable membrane", ["water", "semipermeable membrane"]),
            ("Expand the abbreviation CPU.", "central processing unit", ["central processing unit"]),
            ("Define a metaphor in one short clause.", "a figure of speech that makes an implicit comparison", ["figure of speech", "comparison"]),
            ("Define an isotope in one short clause.", "an atom with the same proton count but a different neutron count", ["neutron", "proton"]),
            ("Define a hemisphere in one short clause.", "one half of a sphere", ["half of a sphere", "half a sphere"]),
            ("Define an antonym in one short clause.", "a word with an opposite meaning", ["opposite meaning"]),
            ("Define inertia in one short clause.", "resistance to a change in motion", ["change in motion", "state of motion"]),
            ("Define a byte in one short clause.", "a unit of digital information usually containing eight bits", ["eight bits", "8 bits"]),
            ("Define pollination in one short clause.", "the transfer of pollen between flower structures", ["transfer of pollen", "pollen transfer"]),
        ],
        "classification": [
            ("Classify a whale as fish, mammal, reptile, or bird.", "mammal", ["mammal"]),
            ("Classify granite as igneous, sedimentary, or metamorphic rock.", "igneous", ["igneous"]),
            ("Classify 29 as prime or composite.", "prime", ["prime"]),
            ("Classify a robin as mammal, bird, fish, or reptile.", "bird", ["bird"]),
            ("Classify copper as metal or nonmetal.", "metal", ["metal"]),
            ("Classify a cube as two-dimensional or three-dimensional.", "three-dimensional", ["three-dimensional", "3-dimensional"]),
            ("Classify evaporation as a physical or chemical change.", "physical change", ["physical"]),
            ("Classify the Sun as a planet, moon, or star.", "star", ["star"]),
            ("Classify carbon dioxide as an element or compound.", "compound", ["compound"]),
            ("Classify a frog as amphibian, reptile, or mammal.", "amphibian", ["amphibian"]),
        ],
        "format_transform": [
            ("Write the word ocean in uppercase and nothing else.", "OCEAN", ["OCEAN"]),
            ("Give the plural of mouse and nothing else.", "mice", ["mice"]),
            ("Give the past tense of go and nothing else.", "went", ["went"]),
            ("Give an antonym of ancient in one word.", "modern", ["modern", "new"]),
            ("Alphabetize these two words: pear, apple.", "apple, pear", ["apple, pear", "apple pear"]),
            ("Write decimal five in binary and nothing else.", "101", ["101"]),
            ("Write nine as a Roman numeral and nothing else.", "IX", ["IX"]),
            ("Write HELLO in lowercase and nothing else.", "hello", ["hello"]),
            ("Give only the first letter of zebra.", "z", ["z"]),
            ("Give the standard acronym for World Health Organization.", "WHO", ["WHO"]),
        ],
        "causal_sentence": [
            ("In one short sentence, why does a metal spoon become hot in soup?", "Heat moves through the spoon by conduction", ["conduction"]),
            ("In one short sentence, why does dew form on cool grass?", "Water vapor condenses on the cool grass", ["condens"]),
            ("In one short sentence, why does a balloon expand when warmed?", "The gas particles move faster and spread out", ["particles", "gas"]),
            ("In one short sentence, why does ice float on liquid water?", "Ice is less dense than liquid water", ["less dense"]),
            ("In one short sentence, why do stars appear to twinkle?", "Earth's atmosphere bends their light unevenly", ["atmosphere", "refract"]),
            ("In one short sentence, why does sugar disappear when stirred into tea?", "The sugar dissolves in the tea", ["dissolv"]),
            ("In one short sentence, why does a bicycle slow without pedaling?", "Friction and air resistance oppose its motion", ["friction", "air resistance"]),
            ("In one short sentence, why can we see the Moon?", "The Moon reflects sunlight", ["reflect", "sunlight"]),
            ("In one short sentence, why does an echo occur?", "Sound waves reflect from a distant surface", ["reflect", "sound"]),
            ("In one short sentence, why does a compass point north?", "Its needle aligns with Earth's magnetic field", ["magnetic field", "magnet"]),
        ],
    }
    rows = []
    for task, values in groups.items():
        assert len(values) == 10
        for i, (prompt, answer, aliases) in enumerate(values):
            rows.append({"id": f"ext_{task}_{i:02d}", "task": task,
                         "prompt_template": f"external_{task}", "prompt": prompt,
                         "answer": answer, "aliases": aliases})
    assert len(rows) == 80
    return rows


def assistant_states(item):
    answer = re.sub(r"[\s.!?;:]+$", "", item["answer"])
    words = answer.split()
    partial = " ".join(words[:max(1, len(words)//2)]) if len(words) > 1 else ""
    return {"U": partial, "C": answer, "P": answer + ".", "K": answer + ","}


def semantic_match(item, text):
    value = text.casefold()
    for alias in item["aliases"]:
        a = alias.casefold()
        if len(a) <= 3 and a.isalnum():
            if re.search(r"(?<![\w])" + re.escape(a) + r"(?![\w])", value):
                return True
        elif a in value:
            return True
    return False


def teacher_forced(model, tok, layers, device, eos_ids, items):
    rows = []
    for idx, item in enumerate(items):
        manifest = protocol_manifest(tok, MODEL, item["prompt"], teacher_final=True)
        positions = manifest["groups"][FROZEN_GROUP]
        for state, content in assistant_states(item).items():
            ids = token_ids(tok, manifest["prefix_text"] + content, add_special_tokens=False)
            clean = run_snapshot(model, layers, device, eos_ids, ids,
                                 capture_layers=[FROZEN_LAYER + 1] if state == "P" else None)
            for method in ["zero", "neutral"]:
                corrupt = run_snapshot(model, layers, device, eos_ids, ids,
                                       capture_layers=[FROZEN_LAYER + 1] if state == "P" else None,
                                       embed_positions=positions, embed_mode=method,
                                       neutral_token_id=neutral_id(tok))
                rows.append({"id": item["id"], "task": item["task"], "state": state,
                             "condition": method, "clean_gap": clean["gap"],
                             "patched_gap": corrupt["gap"],
                             "delta_gap": corrupt["gap"] - clean["gap"],
                             "clean_eos_won": clean["eos_won"],
                             "patched_eos_won": corrupt["eos_won"]})
                if state == "P" and method == "neutral":
                    rescue = run_snapshot(model, layers, device, eos_ids, ids,
                                          embed_positions=positions, embed_mode="neutral",
                                          neutral_token_id=neutral_id(tok),
                                          patch_layer=FROZEN_LAYER,
                                          patch_vector=clean["vectors"][FROZEN_LAYER + 1])
                    rows.append({"id": item["id"], "task": item["task"], "state": state,
                                 "condition": "neutral_plus_clean_L34_rescue",
                                 "clean_gap": clean["gap"], "patched_gap": rescue["gap"],
                                 "delta_gap": rescue["gap"] - clean["gap"],
                                 "clean_eos_won": clean["eos_won"],
                                 "patched_eos_won": rescue["eos_won"],
                                 "corrupt_gap": corrupt["gap"],
                                 "recovery_fraction": 1 - abs(rescue["gap"]-clean["gap"])
                                 / max(abs(corrupt["gap"]-clean["gap"]), 1e-6)})
        if (idx + 1) % 20 == 0:
            log(f"  Phase976 teacher frozen external {idx+1}/80")
    summary = {}
    for condition in sorted({r["condition"] for r in rows}):
        summary[condition] = {}
        for state in ["U", "C", "P", "K"]:
            vals = [r for r in rows if r["condition"] == condition and r["state"] == state]
            if not vals:
                continue
            summary[condition][state] = {
                "n": len(vals), "mean_delta_gap": float(np.mean([r["delta_gap"] for r in vals])),
                "positive_delta_rate": float(np.mean([r["delta_gap"] > 0 for r in vals])),
                "patched_eos_win_rate": float(np.mean([r["patched_eos_won"] for r in vals])),
                "mean_recovery_fraction": float(np.mean([r.get("recovery_fraction", 0) for r in vals]))
                    if condition.endswith("rescue") else None,
            }
    return rows, summary


def natural_condition(model, tok, device, eos_ids, items, method):
    rows = []
    emb = model.get_input_embeddings()
    neutral = emb.weight[neutral_id(tok)].detach()
    for idx, item in enumerate(items):
        manifest = protocol_manifest(tok, MODEL, item["prompt"], teacher_final=False)
        ids = manifest["prefix_ids"]
        positions = manifest["groups"][FROZEN_GROUP]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        mask = torch.ones_like(x)
        used = [False]

        def hook(module, args, output):
            z = output.clone()
            if not used[0] and z.shape[1] == len(ids):
                if method == "zero":
                    z[:, positions, :] = 0
                else:
                    z[:, positions, :] = neutral.to(device=z.device, dtype=z.dtype)
                used[0] = True
            return z

        h = None if method == "clean" else emb.register_forward_hook(hook)
        try:
            with torch.no_grad():
                out = model.generate(input_ids=x, attention_mask=mask, max_new_tokens=32,
                                     do_sample=False, pad_token_id=tok.pad_token_id,
                                     eos_token_id=eos_ids, return_dict_in_generate=True)
        finally:
            if h is not None:
                h.remove()
        if method != "clean" and not used[0]:
            raise RuntimeError("prefill intervention did not run")
        gen = out.sequences[0, len(ids):]
        id_list = gen.tolist()
        plain = tok.decode(gen, skip_special_tokens=True)
        has_eos = any(int(x) in eos_ids for x in id_list)
        rows.append({"id": item["id"], "task": item["task"], "condition": method,
                     "generated": tok.decode(gen, skip_special_tokens=False), "plain": plain,
                     "semantic_match": semantic_match(item, plain), "has_eos": has_eos,
                     "valid_eos": has_eos and semantic_match(item, plain),
                     "n_tokens": len(id_list)})
        if (idx + 1) % 20 == 0:
            log(f"  Phase976 natural {method} {idx+1}/80")
    return {"n": len(rows), "semantic_rate": float(np.mean([r["semantic_match"] for r in rows])),
            "eos_rate": float(np.mean([r["has_eos"] for r in rows])),
            "valid_eos_rate": float(np.mean([r["valid_eos"] for r in rows])),
            "mean_tokens": float(np.mean([r["n_tokens"] for r in rows])), "rows": rows}


def run():
    ensure_dir(OUT)
    t0 = time.time()
    items = build_external_dataset()
    model, tok, device = load_model(MODEL)
    layers = get_layers(model)
    eos_ids = get_eos_ids(model, tok)
    if len(layers) <= FROZEN_LAYER:
        raise RuntimeError("frozen layer absent")
    raw, summary = teacher_forced(model, tok, layers, device, eos_ids, items)
    result = {"phase": 976, "model": MODEL, "candidate_source": "Phase975 frozen before these data",
              "frozen_group": FROZEN_GROUP, "frozen_layer": FROZEN_LAYER,
              "n_items": len(items), "n_tasks": 8, "dataset": items,
              "teacher_summary": summary, "teacher_rows": raw,
              "natural": {}}
    path = OUT / "qwen3_result.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for method in ["clean", "zero", "neutral"]:
        result["natural"][method] = natural_condition(model, tok, device, eos_ids, items, method)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["elapsed_seconds"] = time.time() - t0
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    release_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"Saved {path}; elapsed={result['elapsed_seconds']/60:.1f} min")


if __name__ == "__main__":
    run()
