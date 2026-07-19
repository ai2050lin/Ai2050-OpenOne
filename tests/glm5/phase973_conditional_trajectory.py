#!/usr/bin/env python3
"""Phase 973: conditional trajectory and termination-boundary atlas.

Design principles
-----------------
* 160 hand-checkable items, 8 task families, 20 items per family.
* Five aligned textual states per item: unfinished, just_complete,
  punctuation_complete, continuation_incomplete, and continuation_complete.
* Explicit attention masks in every forward/generation call.
* Full-layer forward ablation is performed on a balanced 32-item discovery
  subset. Candidate components are then re-tested on a disjoint 128-item set.
* Only elementary summaries are used: means, counts, sign rates, and token
  change rates. No fitted statistical model is used.

Punctuation and semantic completion are kept as orthogonal labels. No curated
state is called an actual latent "EOS-pre" state.
"""
from __future__ import annotations

import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import get_layers, get_model_info, load_model, release_model
from phase951_protocol_atlas import ensure_dir
from phase966_natural_stop import log

PHASE = 973
RESULT_DIR = Path("tests/glm5/result/phase973_conditional_trajectory")
STATES = ["unfinished", "just_complete", "punctuation_complete",
          "continuation_incomplete", "continuation_complete"]
STATE_AXES = {
    "unfinished": {"semantic_state": "incomplete", "position_type": "content_end"},
    "just_complete": {"semantic_state": "just_complete", "position_type": "content_end"},
    "punctuation_complete": {"semantic_state": "just_complete", "position_type": "punctuation_end"},
    "continuation_incomplete": {"semantic_state": "incomplete_new_clause", "position_type": "content_end"},
    "continuation_complete": {"semantic_state": "complete_new_clause", "position_type": "punctuation_end"},
}


def build_dataset():
    groups = {
        "short_fact": [
            ("The capital of France is", "Paris"), ("The largest planet is", "Jupiter"),
            ("Water freezes at", "zero degrees Celsius"), ("The chemical symbol for gold is", "Au"),
            ("The author of Hamlet was", "William Shakespeare"), ("The fastest land animal is", "the cheetah"),
            ("The Pacific is an", "ocean"), ("The square root of eighty-one is", "nine"),
            ("The Red Planet is", "Mars"), ("Plants absorb carbon dioxide through their", "leaves"),
            ("The currency of Japan is the", "yen"), ("The human heart has", "four chambers"),
            ("The opposite of north is", "south"), ("A triangle has", "three sides"),
            ("The primary language of Brazil is", "Portuguese"),
            ("The tallest mountain above sea level is", "Mount Everest"),
            ("The gas most abundant in Earth's atmosphere is", "nitrogen"),
            ("The instrument used to measure temperature is a", "thermometer"),
            ("The first month of the year is", "January"),
            ("The center of our solar system is", "the Sun"),
        ],
        "definition": [
            ("A mammal is", "an animal that feeds milk to its young"),
            ("Photosynthesis is", "the process by which plants convert light into chemical energy"),
            ("A prime number is", "an integer greater than one with exactly two positive divisors"),
            ("Gravity is", "the attraction between objects with mass"),
            ("A verb is", "a word that expresses an action or state"),
            ("Democracy is", "government in which people choose their representatives"),
            ("Evaporation is", "the change of a liquid into a gas"),
            ("An ecosystem is", "a community of organisms and their environment"),
            ("A polygon is", "a closed plane shape with straight sides"),
            ("Friction is", "a force that resists motion between surfaces"),
            ("A synonym is", "a word with the same or a similar meaning"),
            ("An algorithm is", "a finite sequence of steps for solving a problem"),
            ("A molecule is", "two or more atoms chemically bonded together"),
            ("Inflation is", "a sustained increase in the general price level"),
            ("A habitat is", "the natural home of an organism"),
            ("A hypothesis is", "a testable proposed explanation"),
            ("Condensation is", "the change of a gas into a liquid"),
            ("A fraction is", "a number representing part of a whole"),
            ("Velocity is", "speed in a specified direction"),
            ("A database is", "an organized collection of data"),
        ],
        "translation": [
            ("Translate 'cat' into French:", "chat"), ("Translate 'house' into Spanish:", "casa"),
            ("Translate 'water' into German:", "Wasser"), ("Translate 'book' into Italian:", "libro"),
            ("Translate 'hello' into Japanese:", "konnichiwa"), ("Translate 'thank you' into French:", "merci"),
            ("Translate 'red' into Spanish:", "rojo"), ("Translate 'sun' into German:", "Sonne"),
            ("Translate 'friend' into Italian:", "amico"), ("Translate 'good morning' into Spanish:", "buenos días"),
            ("Translate 'school' into French:", "école"), ("Translate 'moon' into German:", "Mond"),
            ("Translate 'food' into Italian:", "cibo"), ("Translate 'green' into French:", "vert"),
            ("Translate 'night' into Spanish:", "noche"),
            ("Translate 'dog' into French:", "chien"),
            ("Translate 'blue' into German:", "blau"),
            ("Translate 'bread' into Spanish:", "pan"),
            ("Translate 'mother' into Italian:", "madre"),
            ("Translate 'goodbye' into French:", "au revoir"),
        ],
        "logic": [
            ("If all birds have wings and a robin is a bird, then a robin has", "wings"),
            ("If no squares are circles and this shape is a square, then it is not", "a circle"),
            ("If A is taller than B and B is taller than C, then A is taller than", "C"),
            ("If every doctor studied medicine and Lee is a doctor, then Lee studied", "medicine"),
            ("If the switch is off, the lamp is dark. The switch is off, so the lamp is", "dark"),
            ("If all metals conduct electricity and copper is a metal, copper", "conducts electricity"),
            ("If today is Monday, tomorrow is", "Tuesday"),
            ("If x is greater than five, then x is greater than", "four"),
            ("If every whale is a mammal and Moby is a whale, Moby is", "a mammal"),
            ("If a statement and its negation cannot both be true, they are", "contradictory"),
            ("If the door is locked, it cannot be opened without", "a key"),
            ("If all roses are flowers and some flowers fade, it does not follow that all roses", "fade"),
            ("If P implies Q and P is true, then Q is", "true"),
            ("If P implies Q and Q is false, then P is", "false"),
            ("If two sets have no common members, their intersection is", "empty"),
            ("If all cats are animals and Luna is a cat, Luna is", "an animal"),
            ("If three is less than five and five is less than eight, three is less than", "eight"),
            ("If a number is divisible by four, it is", "even"),
            ("If P and Q are both true, then P and Q is", "true"),
            ("If no reptiles have fur and a snake is a reptile, a snake does not have", "fur"),
        ],
        "causal": [
            ("Plants wilt without water because", "their cells lose pressure"),
            ("Ice melts when heated because", "its molecules gain energy"),
            ("A shadow forms because", "an object blocks light"),
            ("Metal expands when heated because", "its particles move farther apart"),
            ("We see lightning before hearing thunder because", "light travels faster than sound"),
            ("Salt dissolves in water because", "water molecules separate its ions"),
            ("A ball falls when released because", "gravity pulls it downward"),
            ("Wet clothes dry in sunlight because", "the water evaporates"),
            ("A sealed bag inflates at high altitude because", "outside air pressure decreases"),
            ("Leaves often look green because", "chlorophyll reflects green light"),
            ("Iron rusts in moist air because", "it reacts with oxygen and water"),
            ("A candle goes out under a jar because", "it uses up the available oxygen"),
            ("Sound becomes quieter with distance because", "its energy spreads out"),
            ("Roads become slippery in rain because", "water reduces friction"),
            ("The Moon has phases because", "we see changing portions of its sunlit half"),
            ("Bread rises during baking because", "gas bubbles expand in the dough"),
            ("A mirror reflects an image because", "light bounces from its surface"),
            ("Sweating cools the body because", "evaporation removes heat"),
            ("A rainbow forms because", "water droplets refract and reflect sunlight"),
            ("Batteries eventually run down because", "their stored chemical reactants are consumed"),
        ],
        "enumeration": [
            ("The primary colors of light are", "red, green, and blue"),
            ("The three common states of matter are", "solid, liquid, and gas"),
            ("The four seasons are", "spring, summer, autumn, and winter"),
            ("The first three positive integers are", "one, two, and three"),
            ("The vowels in English are", "a, e, i, o, and u"),
            ("The two poles of Earth are", "the North Pole and the South Pole"),
            ("The three sides of a right triangle are often called", "the legs and the hypotenuse"),
            ("The colors on a standard traffic light are", "red, yellow, and green"),
            ("The three branches of the United States government are", "legislative, executive, and judicial"),
            ("The basic arithmetic operations are", "addition, subtraction, multiplication, and division"),
            ("The inner rocky planets are", "Mercury, Venus, Earth, and Mars"),
            ("The two main kinds of electric charge are", "positive and negative"),
            ("The cardinal directions are", "north, east, south, and west"),
            ("The three dimensions of a box are", "length, width, and height"),
            ("The two binary digits are", "zero and one"),
            ("The layers of Earth are", "crust, mantle, outer core, and inner core"),
            ("The three angles of a triangle sum to", "one hundred eighty degrees"),
            ("The two houses of the United States Congress are", "the Senate and the House of Representatives"),
            ("The five senses are", "sight, hearing, smell, taste, and touch"),
            ("The three main cloud types are", "cirrus, cumulus, and stratus"),
        ],
        "grammar_style": [
            ("Complete grammatically: She ___ to school every day.", "goes"),
            ("Complete grammatically: They ___ dinner yesterday.", "ate"),
            ("Complete grammatically: I have ___ the letter.", "written"),
            ("Complete grammatically: Neither answer ___ correct.", "is"),
            ("Complete grammatically: The dogs ___ loudly at night.", "bark"),
            ("Rewrite formally: Kids need sleep.", "Children require adequate sleep"),
            ("Rewrite politely: Give me the report.", "Please send me the report"),
            ("Rewrite actively: The ball was kicked by Mia.", "Mia kicked the ball"),
            ("Rewrite concisely: Due to the fact that it rained, we stayed inside.", "Because it rained, we stayed inside"),
            ("Complete grammatically: Each student ___ a book.", "has"),
            ("Complete grammatically: We ___ waiting for an hour.", "have been"),
            ("Rewrite in past tense: He walks home.", "He walked home"),
            ("Rewrite in plural: The child is playing.", "The children are playing"),
            ("Complete grammatically: There ___ many reasons.", "are"),
            ("Rewrite clearly: The experiment, it failed.", "The experiment failed"),
            ("Complete grammatically: She has ___ her work.", "finished"),
            ("Rewrite in future tense: They arrive today.", "They will arrive today"),
            ("Complete grammatically: The team ___ ready.", "is"),
            ("Rewrite politely: Close the window.", "Please close the window"),
            ("Rewrite actively: The song was performed by Ana.", "Ana performed the song"),
        ],
        "arithmetic_reasoning": [
            ("Seven plus eight equals", "fifteen"), ("Twelve divided by three equals", "four"),
            ("If five books cost ten dollars equally, one book costs", "two dollars"),
            ("A dozen eggs minus three eggs leaves", "nine eggs"),
            ("The next even number after fourteen is", "sixteen"),
            ("A square with side length four has area", "sixteen square units"),
            ("Half of fifty is", "twenty-five"), ("Three groups of six contain", "eighteen items"),
            ("If a train travels sixty miles in one hour, its average speed is", "sixty miles per hour"),
            ("Ten percent of two hundred is", "twenty"),
            ("A clock showing 3:00 has the minute hand pointing at", "twelve"),
            ("If nine candies are shared equally by three children, each receives", "three candies"),
            ("The perimeter of a rectangle of length five and width two is", "fourteen"),
            ("Two squared plus three squared equals", "thirteen"),
            ("If a sequence starts 2, 4, 6, 8, the next term is", "ten"),
            ("Twenty minus seven equals", "thirteen"),
            ("Five squared equals", "twenty-five"),
            ("If four pencils cost twelve dollars equally, one pencil costs", "three dollars"),
            ("A triangle with base six and height four has area", "twelve square units"),
            ("One quarter of eighty is", "twenty"),
        ],
    }
    rows = []
    for task, pairs in groups.items():
        assert len(pairs) == 20
        explanations = [
            (" This directly answers the", " This directly answers the question."),
            (" This result follows from the", " This result follows from the given information."),
            (" In other words, the response", " In other words, the response is complete."),
            (" This is the requested", " This is the requested short answer."),
        ]
        for i, (stem, answer) in enumerate(pairs):
            if i < 12:
                prompt_template="shared_A"; prompt=f"Complete with a short answer: {stem}"
            elif i < 16:
                prompt_template="unseen_B"; prompt=f"Give the shortest correct completion: {stem}"
            else:
                prompt_template="unseen_C"; prompt=f"Answer briefly: {stem}"
            complete = prompt + " " + answer
            boundary = complete if re.search(r"[.!?]$", complete) else complete + "."
            words = answer.split()
            partial = " ".join(words[:max(1, len(words)//2)]) if len(words) > 1 else ""
            unfinished = prompt + (" " + partial if partial else "")
            x, xc = explanations[i % len(explanations)]
            rows.append({"id": f"{task}_{i:02d}", "task": task, "prompt": prompt,
                         "prompt_template": prompt_template,
                         "answer": answer, "full_trajectory": boundary + xc,
                         "states": {"unfinished": unfinished, "just_complete": complete,
                                    "punctuation_complete": boundary,
                                    "continuation_incomplete": boundary + x,
                                    "continuation_complete": boundary + xc}})
    assert len(rows) == 160
    return rows


def zero_last_hook():
    def hook(module, args, output):
        is_tuple = isinstance(output, tuple)
        y = output[0] if is_tuple else output
        z = y.clone(); z[:, -1, :] = 0
        return (z,) + output[1:] if is_tuple else z
    return hook


def zero_positions_hook(last_positions):
    """Zero each batch row's last valid (non-padding) position."""
    def hook(module, args, output):
        is_tuple=isinstance(output, tuple)
        y=output[0] if is_tuple else output
        z=y.clone(); rows=torch.arange(z.shape[0], device=z.device)
        z[rows, last_positions.to(z.device), :]=0
        return (z,) + output[1:] if is_tuple else z
    return hook


def batch_iter(rows, size):
    for i in range(0, len(rows), size):
        yield rows[i:i+size]


def encode_batch(tokenizer, texts, device):
    old = tokenizer.padding_side; tokenizer.padding_side = "right"
    enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=True,
                    return_attention_mask=True)
    tokenizer.padding_side = old
    return {k: v.to(device) for k, v in enc.items()}


def get_eos_ids(model, tokenizer):
    ids=[]
    for value in [getattr(tokenizer, "eos_token_id", None),
                  getattr(getattr(model, "generation_config", None), "eos_token_id", None),
                  getattr(getattr(model, "config", None), "eos_token_id", None)]:
        if value is None: continue
        vals=value if isinstance(value, (list, tuple, set)) else [value]
        for x in vals:
            if x is not None and int(x) not in ids: ids.append(int(x))
    if not ids: raise ValueError("No EOS token ids found")
    return ids


def summarize_logits(logits, eos_ids, last_positions=None):
    if last_positions is None:
        v=logits[:, -1].float()
    else:
        rows=torch.arange(logits.shape[0], device=logits.device)
        v=logits[rows, last_positions.to(logits.device)].float()
    competitors = v.clone(); competitors[:, eos_ids] = -torch.inf
    topv, topi = competitors.max(-1)
    eos_values=v[:, eos_ids]; eos, eos_choice=eos_values.max(-1)
    chosen_eos=torch.tensor(eos_ids, device=v.device)[eos_choice]
    return {"gap": (topv-eos).cpu().tolist(), "top_id": topi.cpu().tolist(),
            "top_logit": topv.cpu().tolist(), "eos_logit": eos.cpu().tolist(),
            "eos_id": chosen_eos.cpu().tolist(),
            "eos_rank": (1+(v > eos[:, None]).sum(-1)).cpu().tolist()}


def flat_states(items, states=STATES):
    return [{"id": x["id"], "task": x["task"], "answer": x["answer"],
             "prompt_template": x["prompt_template"],
             "state": s, "text": x["states"][s]} for x in items for s in states]


def token_prefix_audit(tokenizer, items):
    rows=[]
    for item in items:
        full=tokenizer.encode(item["full_trajectory"], add_special_tokens=True)
        for state, text in item["states"].items():
            ids=tokenizer.encode(text, add_special_tokens=True)
            rows.append({"id": item["id"], "state": state,
                         "is_exact_token_prefix": full[:len(ids)] == ids,
                         "state_tokens": len(ids), "full_tokens": len(full)})
    return {"n": len(rows),
            "exact_prefix_rate": float(np.mean([r["is_exact_token_prefix"] for r in rows])),
            "mismatches": [r for r in rows if not r["is_exact_token_prefix"]]}


def regression_checks(model, tokenizer, device, layers, eos_ids, items):
    """Small checks required before expensive atlas execution."""
    # Explicit-mask invariance: an item alone versus left-padded beside a longer item.
    short=items[0]["states"]["unfinished"]
    long=items[1]["states"]["continuation_complete"]
    e1=encode_batch(tokenizer, [short], device)
    e2=encode_batch(tokenizer, [short, long], device)
    with torch.no_grad():
        o1=model(**e1, use_cache=False).logits
        o2=model(**e2, use_cache=False).logits
        p1=int(e1["attention_mask"][0].sum()-1); p2=int(e2["attention_mask"][0].sum()-1)
        l1=o1[0, p1].float(); l2=o2[0, p2].float()
    mask_max_abs=float((l1-l2).abs().max())
    s1=summarize_logits(o1[:1], eos_ids, torch.tensor([p1],device=device))
    s2=summarize_logits(o2[:1], eos_ids, torch.tensor([p2],device=device))
    mask_gap_abs=abs(s1["gap"][0]-s2["gap"][0])
    mask_eos_abs=abs(s1["eos_logit"][0]-s2["eos_logit"][0])
    mask_competitor_same=s1["top_id"][0] == s2["top_id"][0]

    # Pure tensor check: the component hook leaves every non-final position unchanged.
    fake=torch.randn(2, 4, 7, device=device)
    changed=zero_last_hook()(None, (), fake)
    hook_prefix_unchanged=bool(torch.equal(fake[:, :-1], changed[:, :-1]))
    hook_last_zero=bool(torch.count_nonzero(changed[:, -1]).item() == 0)

    # Formula check: EOS above every non-EOS token must produce a negative gap.
    vocab=max(eos_ids)+3
    synthetic=torch.zeros(1, 1, vocab, device=device)
    synthetic[0, 0, eos_ids[0]]=2.0; synthetic[0, 0, 0]=1.0
    synthetic_gap=summarize_logits(synthetic, eos_ids)["gap"][0]
    return {"mask_single_vs_padded_max_abs_logit_diff": mask_max_abs,
            "mask_single_vs_padded_gap_abs_diff": mask_gap_abs,
            "mask_single_vs_padded_eos_abs_diff": mask_eos_abs,
            "mask_single_vs_padded_competitor_same": mask_competitor_same,
            "mask_consistency_pass_le_0.25": mask_gap_abs <= 0.25 and mask_eos_abs <= 0.25 and mask_competitor_same,
            "all_eos_ids": eos_ids, "n_eos_ids": len(eos_ids),
            "hook_prefix_unchanged": hook_prefix_unchanged,
            "hook_last_position_zero": hook_last_zero,
            "negative_gap_when_eos_wins": synthetic_gap < 0,
            "synthetic_gap": synthetic_gap,
            "canonical_answer_metric_requires_full_answer": True}


def base_atlas(model, tokenizer, device, eos_ids, items, batch_size):
    rows = flat_states(items); out = []
    for batch in batch_iter(rows, batch_size):
        enc = encode_batch(tokenizer, [r["text"] for r in batch], device)
        with torch.no_grad(): logits = model(**enc, use_cache=False).logits
        pos=enc["attention_mask"].sum(-1)-1
        sm = summarize_logits(logits, eos_ids, pos)
        for i, row in enumerate(batch):
            out.append({**{k: row[k] for k in ["id", "task", "prompt_template", "state"]},
                        "top_id": sm["top_id"][i],
                        "top_token": tokenizer.decode([sm["top_id"][i]]),
                        "gap": sm["gap"][i], "eos_logit": sm["eos_logit"][i],
                        "eos_id": sm["eos_id"][i], "eos_rank": sm["eos_rank"][i],
                        "eos_won": sm["gap"][i] < 0})
    return out


def causal_rows(model, tokenizer, device, layers, eos_ids, rows, component, layer_idx,
                batch_size, include_raw=False, base_lookup=None):
    module = layers[layer_idx].mlp if component == "mlp" else layers[layer_idx].self_attn
    collected = []
    for batch in batch_iter(rows, batch_size):
        enc = encode_batch(tokenizer, [r["text"] for r in batch], device)
        pos=enc["attention_mask"].sum(-1)-1
        if base_lookup is None:
            with torch.no_grad(): base = model(**enc, use_cache=False).logits
            b = summarize_logits(base, eos_ids, pos)
        else:
            vals=[base_lookup[(r["id"], r["state"])] for r in batch]
            b={"gap":[r["gap"] for r in vals], "top_id":[r["top_id"] for r in vals],
               "eos_logit":[r["eos_logit"] for r in vals], "eos_id":[r["eos_id"] for r in vals]}
        h = module.register_forward_hook(zero_positions_hook(pos))
        try:
            with torch.no_grad(): patched = model(**enc, use_cache=False).logits
        finally:
            h.remove()
        p = summarize_logits(patched, eos_ids, pos)
        rows_idx=torch.arange(patched.shape[0], device=patched.device)
        pv=patched[rows_idx, pos].float()
        for i, row in enumerate(batch):
            fixed_id=b["top_id"][i]
            patched_eos=float(pv[i, eos_ids].max())
            base_fixed=float(b["gap"][i])
            patched_fixed=float(pv[i, fixed_id]-patched_eos)
            rec = {"id": row["id"], "task": row["task"], "prompt_template": row["prompt_template"],
                   "state": row["state"],
                   "delta_gap": p["gap"][i]-b["gap"][i],
                   "delta_fixed_competitor_gap": patched_fixed-base_fixed,
                   "delta_eos": p["eos_logit"][i]-b["eos_logit"][i],
                   "eos_identity_changed": p["eos_id"][i] != b["eos_id"][i],
                   "competitor_changed": p["top_id"][i] != b["top_id"][i],
                   "eos_won": p["gap"][i] < 0}
            if include_raw:
                rec.update({"base_gap": b["gap"][i], "patched_gap": p["gap"][i],
                            "base_top_id": b["top_id"][i], "patched_top_id": p["top_id"][i]})
            collected.append(rec)
    return collected


def aggregate(rows):
    by = defaultdict(list)
    for r in rows: by[r["state"]].append(r)
    out = {}
    for state, vals in by.items():
        out[state] = {"n": len(vals),
                      "mean_delta_gap": float(np.mean([r["delta_gap"] for r in vals])),
                      "mean_delta_eos": float(np.mean([r["delta_eos"] for r in vals])),
                      "negative_delta_gap_rate": float(np.mean([r["delta_gap"] < 0 for r in vals])),
                      "mean_delta_fixed_competitor_gap": float(np.mean([r["delta_fixed_competitor_gap"] for r in vals])),
                      "competitor_changed_rate": float(np.mean([r["competitor_changed"] for r in vals])),
                      "eos_win_rate": float(np.mean([r["eos_won"] for r in vals]))}
    return out


def choose_candidates(discovery):
    ranked = []
    for comp, layers in discovery.items():
        for L, a in enumerate(layers):
            u = a["unfinished"]; b = a["punctuation_complete"]
            # Desired conditional ablation: lowers gap at a strong boundary, but not before answering.
            selectivity = u["mean_delta_gap"] - b["mean_delta_gap"]
            ranked.append({"component": comp, "layer": L, "selectivity": selectivity,
                           "unfinished_delta_gap": u["mean_delta_gap"],
                           "boundary_delta_gap": b["mean_delta_gap"],
                           "boundary_negative_rate": b["negative_delta_gap_rate"],
                           "boundary_competitor_changed_rate": b["competitor_changed_rate"]})
    # Do not pre-assert a mechanism: rank by boundary help plus state selectivity.
    ranked.sort(key=lambda r: (r["boundary_delta_gap"], -r["selectivity"]))
    # Freeze the best MLP and best attention candidate before holdout.
    selected=[]
    for comp in ["mlp", "attention"]:
        selected.extend([r for r in ranked if r["component"] == comp][:1])
    return selected, ranked


def validate_candidates(model, tokenizer, device, layers, eos_ids, items, candidates, batch_size, base_lookup):
    rows = flat_states(items); out = []
    for c in candidates:
        raw = causal_rows(model, tokenizer, device, layers, eos_ids, rows,
                          c["component"], c["layer"], batch_size, include_raw=True,
                          base_lookup=base_lookup)
        same=[r for r in raw if r["prompt_template"] == "shared_A"]
        unseen=[r for r in raw if r["prompt_template"] != "shared_A"]
        out.append({**c, "validation": aggregate(raw),
                    "validation_shared_template": aggregate(same),
                    "validation_unseen_templates": aggregate(unseen), "rows": raw})
    return out


class GenerationGate:
    def __init__(self, tokenizer, mode):
        self.tokenizer = tokenizer; self.mode = mode
        self.active = mode == "unconditional"; self.calls = 0; self.activation_call = None
        self.pending_boundary = False
    def pre_hook(self, module, args, kwargs):
        ids = kwargs.get("input_ids")
        if ids is None and args: ids = args[0]
        if self.mode == "early" and self.calls >= 1 and not self.active:
            self.active=True; self.activation_call=self.calls
        if self.mode == "fixed_step_4" and self.calls >= 4 and not self.active:
            self.active=True; self.activation_call=self.calls
        if self.mode == "delayed_boundary" and self.pending_boundary and not self.active:
            self.active=True; self.activation_call=self.calls; self.pending_boundary=False
        if self.mode in ("boundary", "delayed_boundary") and self.calls > 0 and ids is not None:
            token = self.tokenizer.decode([int(ids[0, -1])])
            if re.search(r"[.!?]\s*$", token):
                if self.mode == "boundary":
                    self.active = True
                    if self.activation_call is None: self.activation_call = self.calls
                elif not self.active:
                    self.pending_boundary = True
        self.calls += 1
    def component_hook(self, module, args, output):
        if not self.active: return output
        return zero_last_hook()(module, args, output)


def generation_test(model, tokenizer, device, layers, eos_ids, items, candidate, n=40):
    results = {}
    module = layers[candidate["layer"]].mlp if candidate["component"] == "mlp" else layers[candidate["layer"]].self_attn
    for mode in ["baseline", "unconditional", "early", "fixed_step_4", "boundary", "delayed_boundary"]:
        rows = []
        for item in items[:n]:
            enc = encode_batch(tokenizer, [item["prompt"]], device)
            handles=[]; gate=None
            if mode != "baseline":
                gate = GenerationGate(tokenizer, mode)
                handles.append(model.register_forward_pre_hook(gate.pre_hook, with_kwargs=True))
                handles.append(module.register_forward_hook(gate.component_hook))
            try:
                with torch.no_grad():
                    oid = model.generate(**enc, max_new_tokens=24, do_sample=False,
                                         pad_token_id=tokenizer.pad_token_id,
                                         eos_token_id=eos_ids)
            finally:
                for h in handles: h.remove()
            gen_ids = oid[0, enc["input_ids"].shape[1]:]
            gen = tokenizer.decode(gen_ids, skip_special_tokens=False)
            plain = tokenizer.decode(gen_ids, skip_special_tokens=True)
            expected = item["answer"].lower() in plain.lower()
            has_eos = any(bool((gen_ids == eid).any().item()) for eid in eos_ids)
            rows.append({"id": item["id"], "task": item["task"], "generated": gen,
                         "has_expected": expected, "has_eos": has_eos,
                         "n_tokens": int(len(gen_ids)),
                         "boundary_triggered": gate.activation_call is not None if gate else False})
        results[mode] = {"n": len(rows),
                         "expected_rate": float(np.mean([r["has_expected"] for r in rows])),
                         "eos_rate": float(np.mean([r["has_eos"] for r in rows])),
                         "joint_rate": float(np.mean([r["has_expected"] and r["has_eos"] for r in rows])),
                         "mean_tokens": float(np.mean([r["n_tokens"] for r in rows])),
                         "boundary_trigger_rate": float(np.mean([r["boundary_triggered"] for r in rows])),
                         "rows": rows}
    return results


def run_model(model_name):
    ensure_dir(RESULT_DIR); t0=time.time(); items=build_dataset()
    model, tokenizer, device = load_model(model_name); layers=get_layers(model)
    info=get_model_info(model, model_name); eos_ids=get_eos_ids(model, tokenizer)
    batch_size = 8 if model_name == "glm4" else 16
    out=RESULT_DIR/f"{model_name}_result.json"
    checkpoint=None
    if out.exists():
        try: checkpoint=json.loads(out.read_text(encoding="utf-8"))
        except Exception: checkpoint=None
    if checkpoint and len(checkpoint.get("base_atlas", [])) == len(items)*len(STATES) \
       and checkpoint.get("eos_token_ids") == eos_ids:
        result=checkpoint; log(f"Phase973 {model_name}: resumed 160x5 base atlas checkpoint")
    else:
        result = {"phase": PHASE, "model": model_name, "n_items": len(items),
                  "n_tasks": 8, "states": STATES,
                  "state_axes": STATE_AXES,
                  "eos_token_ids": eos_ids,
                  "label_warning": "curated semantic/position labels; no state is asserted to be a latent EOS-pre state",
                  "attention_mask_explicit": True, "dataset": items}
        result["token_prefix_audit"] = token_prefix_audit(tokenizer, items)
        result["regression_checks"] = regression_checks(model, tokenizer, device, layers, eos_ids, items)
        if not all([result["regression_checks"]["mask_consistency_pass_le_0.25"],
                    result["regression_checks"]["hook_prefix_unchanged"],
                    result["regression_checks"]["hook_last_position_zero"],
                    result["regression_checks"]["negative_gap_when_eos_wins"]]):
            raise RuntimeError(f"Regression check failed: {result['regression_checks']}")
        log(f"Phase973 {model_name}: base atlas 160x5")
        result["base_atlas"] = base_atlas(model, tokenizer, device, eos_ids, items, batch_size)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    base_lookup={(r["id"], r["state"]):r for r in result["base_atlas"]}

    # One balanced item per family for exhaustive screening; i=1..3 stay unused.
    # The frozen candidates are validated only on i=4..19 (128 disjoint items).
    discovery_items=[]; holdout_items=[]; unused_development=[]
    for task in sorted(set(x["task"] for x in items)):
        task_items=[x for x in items if x["task"] == task]
        discovery_items.extend(task_items[:1]); unused_development.extend(task_items[1:4]); holdout_items.extend(task_items[4:])
    discovery_states=["unfinished", "punctuation_complete", "continuation_incomplete"]
    discovery_rows=flat_states(discovery_items, discovery_states); discovery={}
    for comp in ["mlp", "attention"]:
        discovery[comp]=[]
        for L in range(len(layers)):
            raw=causal_rows(model, tokenizer, device, layers, eos_ids, discovery_rows,
                            comp, L, batch_size, base_lookup=base_lookup)
            discovery[comp].append(aggregate(raw))
            if (L+1) % 8 == 0: log(f"  {comp} {L+1}/{len(layers)}")
    result["full_layer_discovery_n_items"] = len(discovery_items)
    result["full_layer_discovery_states"] = discovery_states
    result["unused_development_n_items"] = len(unused_development)
    result["full_layer_discovery"] = discovery
    top, all_ranked = choose_candidates(discovery)
    result["candidate_ranking"] = all_ranked; result["top_candidates"] = top
    result["holdout_n_items"] = len(holdout_items)
    result["holdout_template_counts"] = dict((t, sum(x["prompt_template"] == t for x in holdout_items))
                                               for t in ["shared_A", "unseen_B", "unseen_C"])
    result["split_overlap"] = bool(set(x["id"] for x in discovery_items) & set(x["id"] for x in holdout_items))
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"  validating 2 frozen candidates on disjoint {len(holdout_items)}-item holdout")
    validated=validate_candidates(model, tokenizer, device, layers, eos_ids, holdout_items, top, batch_size, base_lookup)
    result["candidate_validation_holdout_128"] = validated
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # The best candidate is an empirical test target, not a claimed mechanism.
    best=min(validated, key=lambda c: c["validation"]["punctuation_complete"]["mean_delta_gap"])
    result["selected_for_rollout"]={k:best[k] for k in ["component","layer","selectivity"]}
    if model_name == "glm4":
        rollout40=[]
        for task in sorted(set(x["task"] for x in holdout_items)):
            task_rows=[x for x in holdout_items if x["task"] == task]
            rollout40.extend(task_rows[:2] + task_rows[-3:])
        log(f"  balanced holdout rollout 40 items with {best['component']} L{best['layer']}")
        roll40=generation_test(model, tokenizer, device, layers, eos_ids, rollout40, best, n=40)
        result["rollout_40"] = roll40
        # Expand only a positive early signal, preventing a fourth small-sample illusion.
        improve=roll40["boundary"]["joint_rate"]-roll40["baseline"]["joint_rate"]
        if improve >= 0.05:
            log("  positive >=5pp signal; expanding rollout to disjoint holdout 128")
            result["rollout_holdout_128"] = generation_test(model, tokenizer, device, layers, eos_ids, holdout_items, best, n=128)
        else:
            result["rollout_holdout_128"]={"not_run": True, "reason": f"40-item boundary joint improvement {improve:+.3f} < +0.05"}
    result["elapsed_seconds"] = time.time()-t0
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    release_model(model); gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    log(f"Saved {out}; elapsed={result['elapsed_seconds']/60:.1f} min")


if __name__ == "__main__":
    target=sys.argv[1] if len(sys.argv)>1 else "glm4"
    run_model(target)
