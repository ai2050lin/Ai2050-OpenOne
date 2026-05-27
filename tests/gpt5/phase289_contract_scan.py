from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hf_probe_env import get_layers, load_probe_model, release_loaded


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Pair:
    name: str
    category: str
    subtype: str
    a: str
    b: str


def log(msg: str) -> None:
    print(f"[phase289] {msg}", flush=True)


def build_pairs() -> list[Pair]:
    pairs: list[Pair] = []

    negation_groups = {
        "lexical_not_adj": [
            ("happy", "she is happy", "she is not happy"),
            ("open", "the door is open", "the door is not open"),
            ("possible", "victory is possible", "victory is not possible"),
            ("ready", "they are ready", "they are not ready"),
            ("important", "this is important", "this is not important"),
            ("clear", "the answer is clear", "the answer is not clear"),
            ("safe", "the area is safe", "the area is not safe"),
            ("fair", "the decision is fair", "the decision is not fair"),
        ],
        "syntactic_do_not": [
            ("agree", "they agree with the proposal", "they do not agree with the proposal"),
            ("remember", "i remember the meeting", "i do not remember the meeting"),
            ("understand", "we understand the problem", "we do not understand the problem"),
            ("know", "she knows the answer", "she does not know the answer"),
            ("believe", "he believes the story", "he does not believe the story"),
            ("support", "they support the plan", "they do not support the plan"),
            ("accept", "she accepts the offer", "she does not accept the offer"),
            ("expect", "we expect rain", "we do not expect rain"),
        ],
        "existential_no": [
            ("found_nothing", "he found something interesting", "he found nothing interesting"),
            ("no_one_came", "someone came to the party", "no one came to the party"),
            ("no_food", "there was some food left", "there was no food left"),
            ("no_idea", "she had some idea what to do", "she had no idea what to do"),
            ("no_reason", "there is a reason to worry", "there is no reason to worry"),
            ("no_choice", "they had a choice in the matter", "they had no choice in the matter"),
        ],
        "never": [
            ("seen_before", "i have seen it before", "i have never seen it before"),
            ("been_paris", "she has been to Paris", "she has never been to Paris"),
            ("told_secret", "he told someone the secret", "he never told anyone the secret"),
            ("gives_up", "she sometimes gives up", "she never gives up"),
            ("forgets_face", "he sometimes forgets names", "he never forgets a face"),
            ("late", "she is sometimes late", "she is never late"),
        ],
        "morphological_neg": [
            ("impossible", "the task is possible", "the task is impossible"),
            ("unacceptable", "the proposal is acceptable", "the proposal is unacceptable"),
            ("incomplete", "the report is complete", "the report is incomplete"),
            ("irrelevant", "the comment is relevant", "the comment is irrelevant"),
            ("dishonest", "the person is honest", "the person is dishonest"),
            ("incorrect", "the assumption is correct", "the assumption is incorrect"),
        ],
        "scope_quantifier": [
            ("not_all", "all birds can fly", "not all birds can fly"),
            ("not_everyone", "everyone agreed", "not everyone agreed"),
            ("not_always", "she always tells the truth", "she does not always tell the truth"),
            ("not_entirely", "the plan is entirely successful", "the plan is not entirely successful"),
            ("not_necessarily", "wealth means happiness", "wealth does not necessarily mean happiness"),
            ("not_completely", "the glass is full", "the glass is not completely full"),
        ],
    }
    for subtype, rows in negation_groups.items():
        for name, a, b in rows:
            pairs.append(Pair(f"neg_{name}", "negation", subtype, a, b))

    logical_rows = [
        ("and_or_catdog", "the cat and the dog are sleeping", "the cat or the dog is sleeping", "and_or"),
        ("and_or_tea", "tea and coffee are served", "tea or coffee is served", "and_or"),
        ("and_or_keys", "the keys and the wallet are missing", "the keys or the wallet is missing", "and_or"),
        ("and_or_phone", "the phone and the charger are in the bag", "the phone or the charger is in the bag", "and_or"),
        ("and_or_milk", "milk and bread are on the list", "milk or bread is on the list", "and_or"),
        ("and_or_judge", "the judge and the jury listened carefully", "the judge or the jury listened carefully", "and_or"),
        ("and_or_train", "the train and the bus arrive tonight", "the train or the bus arrives tonight", "and_or"),
        ("and_or_report", "the report and the chart explain the result", "the report or the chart explains the result", "and_or"),
        ("if_rain", "if it rains we will stay home", "we will stay home if it rains", "conditional"),
        ("if_hungry", "if you are hungry eat something", "eat something if you are hungry", "conditional"),
        ("if_alarm", "if the alarm rings leave the building", "leave the building if the alarm rings", "conditional"),
        ("if_late", "if the train is late call the office", "call the office if the train is late", "conditional"),
        ("if_ready", "if the team is ready start the test", "start the test if the team is ready", "conditional"),
        ("if_sick", "if she feels sick she should rest", "she should rest if she feels sick", "conditional"),
        ("if_safe", "if the road is safe we can drive", "we can drive if the road is safe", "conditional"),
        ("if_needed", "if more water is needed add some", "add more water if it is needed", "conditional"),
        ("because_rain", "because it rained we stayed home", "we stayed home because it rained", "causal"),
        ("because_tired", "because he was tired he went to bed", "he went to bed because he was tired", "causal"),
        ("because_noise", "because the room was noisy she closed the door", "she closed the door because the room was noisy", "causal"),
        ("because_dark", "because it was dark they used a flashlight", "they used a flashlight because it was dark", "causal"),
        ("because_sale", "because demand increased prices rose", "prices rose because demand increased", "causal"),
        ("because_error", "because the file was corrupted the program stopped", "the program stopped because the file was corrupted", "causal"),
        ("because_wind", "because the wind was strong the flight was delayed", "the flight was delayed because the wind was strong", "causal"),
        ("because_busy", "because the store was busy the line moved slowly", "the line moved slowly because the store was busy", "causal"),
        ("although_rain", "although it rained they went out", "they went out although it rained", "contrast"),
        ("although_tired", "although she was tired she finished the work", "she finished the work although she was tired", "contrast"),
        ("although_expensive", "although the ticket was expensive he bought it", "he bought the ticket although it was expensive", "contrast"),
        ("although_small", "although the room was small it felt comfortable", "the room felt comfortable although it was small", "contrast"),
        ("although_late", "although the answer was late it was accepted", "the answer was accepted although it was late", "contrast"),
        ("although_new", "although the method was new it worked well", "the method worked well although it was new", "contrast"),
        ("although_cold", "although the water was cold they swam", "they swam although the water was cold", "contrast"),
        ("although_busy", "although he was busy he replied quickly", "he replied quickly although he was busy", "contrast"),
        ("therefore_rain", "it rained all night therefore the ground is wet", "the ground is wet because it rained all night", "inference"),
        ("therefore_study", "she studied hard therefore she passed the exam", "she passed the exam because she studied hard", "inference"),
        ("therefore_heat", "the metal was heated therefore it expanded", "the metal expanded because it was heated", "inference"),
        ("therefore_save", "they saved money therefore they could travel", "they could travel because they saved money", "inference"),
        ("therefore_alarm", "the alarm sounded therefore everyone left", "everyone left because the alarm sounded", "inference"),
        ("therefore_traffic", "traffic was heavy therefore the bus was late", "the bus was late because traffic was heavy", "inference"),
        ("therefore_battery", "the battery was empty therefore the phone died", "the phone died because the battery was empty", "inference"),
        ("therefore_training", "the model trained longer therefore accuracy improved", "accuracy improved because the model trained longer", "inference"),
    ]
    for name, a, b, subtype in logical_rows:
        pairs.append(Pair(f"logic_{name}", "logical", subtype, a, b))

    passive_rows = [
        ("dog_cat", "the dog chases the cat", "the cat is chased by the dog", "by_phrase"),
        ("teacher_student", "the teacher teaches the student", "the student is taught by the teacher", "by_phrase"),
        ("author_book", "the author wrote the book", "the book was written by the author", "by_phrase"),
        ("workers_bridge", "the workers built the bridge", "the bridge was built by the workers", "by_phrase"),
        ("chef_meal", "the chef prepared the meal", "the meal was prepared by the chef", "by_phrase"),
        ("artist_picture", "the artist painted the picture", "the picture was painted by the artist", "by_phrase"),
        ("storm_roof", "the storm damaged the roof", "the roof was damaged by the storm", "by_phrase"),
        ("police_driver", "the police stopped the driver", "the driver was stopped by the police", "by_phrase"),
        ("window_broken", "someone broke the window", "the window was broken", "no_agent"),
        ("letter_sent", "someone sent the letter", "the letter was sent", "no_agent"),
        ("cake_eaten", "someone ate the cake", "the cake was eaten", "no_agent"),
        ("door_closed", "someone closed the door", "the door was closed", "no_agent"),
        ("mistake_made", "someone made a mistake", "a mistake was made", "no_agent"),
        ("meeting_cancelled", "someone cancelled the meeting", "the meeting was cancelled", "no_agent"),
        ("car_repaired", "someone repaired the car", "the car was repaired", "no_agent"),
        ("package_delivered", "someone delivered the package", "the package was delivered", "no_agent"),
        ("get_hired", "the company hired her", "she got hired by the company", "get_passive"),
        ("get_promoted", "the manager promoted him", "he got promoted by the manager", "get_passive"),
        ("get_paid", "the client paid the worker", "the worker got paid by the client", "get_passive"),
        ("get_selected", "the committee selected the proposal", "the proposal got selected by the committee", "get_passive"),
        ("get_invited", "the host invited them", "they got invited by the host", "get_passive"),
        ("get_caught", "the guard caught the thief", "the thief got caught by the guard", "get_passive"),
        ("get_helped", "the nurse helped the patient", "the patient got helped by the nurse", "get_passive"),
        ("get_moved", "the movers moved the piano", "the piano got moved by the movers", "get_passive"),
        ("gift_given", "the teacher gave the student a book", "the student was given a book by the teacher", "dative_passive"),
        ("award_given", "the committee gave the scientist an award", "the scientist was given an award by the committee", "dative_passive"),
        ("offer_made", "the company made her an offer", "she was made an offer by the company", "dative_passive"),
        ("task_assigned", "the manager assigned him a task", "he was assigned a task by the manager", "dative_passive"),
        ("ticket_sent", "the agent sent the customer a ticket", "the customer was sent a ticket by the agent", "dative_passive"),
        ("chance_given", "the coach gave the player a chance", "the player was given a chance by the coach", "dative_passive"),
        ("warning_given", "the officer gave the driver a warning", "the driver was given a warning by the officer", "dative_passive"),
        ("role_offered", "the director offered the actor a role", "the actor was offered a role by the director", "dative_passive"),
    ]
    for name, a, b, subtype in passive_rows:
        pairs.append(Pair(f"pass_{name}", "passive", subtype, a, b))

    recursive_rows = [
        ("rel_man_dog", "the man saw the dog", "the man who saw the dog waved", "relative_clause"),
        ("rel_book_shelf", "the book was on the shelf", "the book that was on the shelf fell", "relative_clause"),
        ("rel_student_question", "the student answered the question", "the student who answered the question smiled", "relative_clause"),
        ("rel_woman_car", "the woman bought the car", "the woman who bought the car arrived", "relative_clause"),
        ("rel_city_river", "the city lies near the river", "the city that lies near the river grew", "relative_clause"),
        ("rel_song_radio", "the song played on the radio", "the song that played on the radio ended", "relative_clause"),
        ("rel_box_table", "the box sat on the table", "the box that sat on the table opened", "relative_clause"),
        ("rel_teacher_lesson", "the teacher explained the lesson", "the teacher who explained the lesson left", "relative_clause"),
        ("pp_key_drawer", "the key is in the drawer", "the key in the drawer under the map is old", "pp_chain"),
        ("pp_cup_table", "the cup is on the table", "the cup on the table near the window is blue", "pp_chain"),
        ("pp_photo_album", "the photo is in the album", "the photo in the album on the desk is faded", "pp_chain"),
        ("pp_bag_chair", "the bag is beside the chair", "the bag beside the chair in the hall is heavy", "pp_chain"),
        ("pp_path_forest", "the path goes through the forest", "the path through the forest behind the school is narrow", "pp_chain"),
        ("pp_note_book", "the note is inside the book", "the note inside the book on the shelf is short", "pp_chain"),
        ("pp_lamp_sofa", "the lamp is behind the sofa", "the lamp behind the sofa near the door is bright", "pp_chain"),
        ("pp_ring_box", "the ring is in the box", "the ring in the box under the cloth is silver", "pp_chain"),
        ("comp_think_leave", "i think she left", "i think that he said that she left", "complement_clause"),
        ("comp_believe_win", "they believe the team won", "they believe that the coach said that the team won", "complement_clause"),
        ("comp_know_safe", "we know the road is safe", "we know that she reported that the road is safe", "complement_clause"),
        ("comp_hear_ready", "he heard the guests were ready", "he heard that they claimed that the guests were ready", "complement_clause"),
        ("comp_expect_arrive", "she expects the train to arrive", "she expects that the sign says that the train will arrive", "complement_clause"),
        ("comp_say_return", "they say he returned", "they say that she believes that he returned", "complement_clause"),
        ("comp_notice_missing", "i noticed the file was missing", "i noticed that the log showed that the file was missing", "complement_clause"),
        ("comp_fear_lost", "he fears the map is lost", "he fears that she thinks that the map is lost", "complement_clause"),
        ("poss_jacket", "the man's jacket is black", "the man's friend's jacket is black", "possessive_chain"),
        ("poss_car", "the woman's car is red", "the woman's brother's car is red", "possessive_chain"),
        ("poss_book", "the teacher's book is open", "the teacher's student's book is open", "possessive_chain"),
        ("poss_phone", "the manager's phone is ringing", "the manager's assistant's phone is ringing", "possessive_chain"),
        ("poss_house", "the artist's house is old", "the artist's neighbor's house is old", "possessive_chain"),
        ("poss_plan", "the engineer's plan is simple", "the engineer's team's plan is simple", "possessive_chain"),
        ("poss_note", "the doctor's note is brief", "the doctor's nurse's note is brief", "possessive_chain"),
        ("poss_song", "the singer's song is popular", "the singer's friend's song is popular", "possessive_chain"),
    ]
    for name, a, b, subtype in recursive_rows:
        pairs.append(Pair(f"rec_{name}", "recursive", subtype, a, b))

    return pairs


def select_pairs(
    pairs: list[Pair],
    categories: set[str] | None,
    subtypes: set[str] | None,
    max_pairs_per_subtype: int,
) -> list[Pair]:
    grouped: dict[tuple[str, str], list[Pair]] = defaultdict(list)
    for pair in pairs:
        if categories and pair.category not in categories:
            continue
        if subtypes and pair.subtype not in subtypes:
            continue
        grouped[(pair.category, pair.subtype)].append(pair)

    selected: list[Pair] = []
    for key in sorted(grouped):
        selected.extend(grouped[key][:max_pairs_per_subtype])
    return selected


def parse_csv(value: str | None) -> list[str]:
    if value is None or value.strip() == "":
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def parse_alphas(value: str) -> list[float]:
    out = []
    for item in parse_csv(value):
        out.append(float(item))
    return out


def choose_layers(n_layers: int, layer_stride: int, explicit_layers: str | None) -> list[int]:
    if explicit_layers:
        layers = sorted({int(x) for x in parse_csv(explicit_layers)})
        return [x for x in layers if 0 <= x < n_layers]
    layers = list(range(0, n_layers, layer_stride))
    if n_layers - 1 not in layers:
        layers.append(n_layers - 1)
    return sorted(set(layers))


def tokenize(loaded: Any, text: str, seq_len: int) -> dict[str, torch.Tensor]:
    batch = loaded.tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
    actual = batch["input_ids"].shape[1]
    if actual < seq_len:
        pad_len = seq_len - actual
        batch["input_ids"] = F.pad(batch["input_ids"], (0, pad_len), value=loaded.tokenizer.pad_token_id)
        batch["attention_mask"] = F.pad(batch["attention_mask"], (0, pad_len), value=0)
    return {k: v.to(loaded.input_device) for k, v in batch.items()}


def capture_outputs(loaded: Any, text: str, target_layers: list[int], seq_len: int) -> dict[int, dict[str, torch.Tensor]]:
    layers = get_layers(loaded.model)
    captured: dict[int, dict[str, torch.Tensor]] = {}
    hooks = []

    def make_hook(layer_idx: int, name: str):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            val = output[0] if isinstance(output, tuple) else output
            captured.setdefault(layer_idx, {})[name] = val.detach().cpu().clone()
        return hook

    for layer_idx in target_layers:
        hooks.append(layers[layer_idx].self_attn.register_forward_hook(make_hook(layer_idx, "attn")))
        hooks.append(layers[layer_idx].mlp.register_forward_hook(make_hook(layer_idx, "mlp")))
        hooks.append(layers[layer_idx].register_forward_hook(make_hook(layer_idx, "resid")))

    with torch.no_grad():
        loaded.model(**tokenize(loaded, text, seq_len))

    for hook in hooks:
        hook.remove()
    return captured


def baseline_logits(loaded: Any, text: str, seq_len: int) -> torch.Tensor:
    with torch.no_grad():
        out = loaded.model(**tokenize(loaded, text, seq_len))
    return out.logits[0, -1, :].detach().cpu().float().clone()


def module_device_dtype(module: Any) -> tuple[torch.device, torch.dtype]:
    param = next(module.parameters())
    return param.device, param.dtype


def interp(a: torch.Tensor, b: torch.Tensor, alpha: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    a_t = a.to(device=device, dtype=dtype)
    b_t = b.to(device=device, dtype=dtype)
    seq = min(a_t.shape[1], b_t.shape[1])
    return (1.0 - alpha) * a_t[:, :seq, :] + alpha * b_t[:, :seq, :]


def forward_patch(
    loaded: Any,
    text: str,
    seq_len: int,
    layer_idx: int,
    patch_type: str,
    alpha: float,
    a_out: dict[str, torch.Tensor],
    b_out: dict[str, torch.Tensor],
) -> tuple[torch.Tensor | None, dict[str, float]]:
    layers = get_layers(loaded.model)
    layer = layers[layer_idx]
    hooks = []
    natural: dict[str, float] = {}

    def patch_hook(value: torch.Tensor):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            ref = output[0] if isinstance(output, tuple) else output
            patched = ref.clone()
            seq = min(value.shape[1], patched.shape[1])
            patched[:, :seq, :] = value[:, :seq, :]
            return (patched,) + output[1:] if isinstance(output, tuple) else patched
        return hook

    if patch_type in {"attn", "both", "cross_battn_amlp", "cross_aattn_bmlp"}:
        device, dtype = module_device_dtype(layer.self_attn.o_proj)
        if patch_type == "cross_aattn_bmlp":
            attn_value = interp(a_out["attn"], b_out["attn"], 0.0, device, dtype)
        else:
            attn_alpha = 1.0 if patch_type == "cross_battn_amlp" else alpha
            attn_value = interp(a_out["attn"], b_out["attn"], attn_alpha, device, dtype)
        hooks.append(layer.self_attn.register_forward_hook(patch_hook(attn_value)))
        natural["patch_attn_norm"] = float(attn_value.float().norm())

    if patch_type in {"mlp", "both", "cross_battn_amlp", "cross_aattn_bmlp"}:
        device, dtype = module_device_dtype(layer.mlp)
        if patch_type == "cross_battn_amlp":
            mlp_value = interp(a_out["mlp"], b_out["mlp"], 0.0, device, dtype)
        else:
            mlp_alpha = 1.0 if patch_type == "cross_aattn_bmlp" else alpha
            mlp_value = interp(a_out["mlp"], b_out["mlp"], mlp_alpha, device, dtype)
        hooks.append(layer.mlp.register_forward_hook(patch_hook(mlp_value)))
        natural["patch_mlp_norm"] = float(mlp_value.float().norm())

    if patch_type == "resid":
        device = next(layer.parameters()).device
        dtype = next(layer.parameters()).dtype
        resid_value = interp(a_out["resid"], b_out["resid"], alpha, device, dtype)
        hooks.append(layer.register_forward_hook(patch_hook(resid_value)))
        natural["patch_resid_norm"] = float(resid_value.float().norm())

    if layer_idx + 1 < len(layers):
        def next_layer_hook(_module: Any, inputs: Any, output: Any) -> None:
            if isinstance(inputs, tuple) and inputs:
                natural["next_resid_in_norm"] = float(inputs[0].detach().float().norm())
            val = output[0] if isinstance(output, tuple) else output
            natural["next_layer_out_norm"] = float(val.detach().float().norm())
        hooks.append(layers[layer_idx + 1].register_forward_hook(next_layer_hook))

    result = None
    try:
        with torch.no_grad():
            out = loaded.model(**tokenize(loaded, text, seq_len))
        result = out.logits[0, -1, :].detach().cpu().float().clone()
    finally:
        for hook in hooks:
            hook.remove()
    return result, natural


def compute_metrics(
    patched: torch.Tensor | None,
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    kl_ab: float,
) -> dict[str, float] | None:
    if patched is None or not torch.isfinite(patched).all():
        return None
    kl_p = float(F.kl_div(F.log_softmax(patched, dim=-1), F.softmax(logits_b, dim=-1), reduction="sum"))
    delta_b = logits_b - logits_a
    delta_p = patched - logits_a
    norm_b = float(delta_b.norm())
    norm_p = float(delta_p.norm())
    if norm_b <= 1e-8 or norm_p <= 1e-8:
        cos_dir = 0.0
        progress = 0.0
        logit_delta_ratio = 0.0
    else:
        cos_dir = float(torch.dot(delta_p, delta_b) / (delta_p.norm() * delta_b.norm()))
        logit_delta_ratio = norm_p / norm_b
        progress = cos_dir * min(logit_delta_ratio, 2.0)
    return {
        "kl_ratio": kl_p / max(kl_ab, 1e-6),
        "progress": progress,
        "cos_dir": cos_dir,
        "logit_delta_ratio": logit_delta_ratio,
        "finite": 1.0,
    }


def mean(values: list[float]) -> float:
    finite_values = [float(v) for v in values if math.isfinite(float(v))]
    return float(sum(finite_values) / len(finite_values)) if finite_values else float("nan")


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_layer_patch: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    by_subtype_patch: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    by_layer_alpha_patch: dict[tuple[int, float, str], list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_layer_alpha_patch[(int(row["layer"]), float(row["alpha"]), str(row["patch_type"]))].append(row)
        if abs(float(row["alpha"]) - 1.0) < 1e-9:
            by_layer_patch[(int(row["layer"]), str(row["patch_type"]))].append(row)
            by_subtype_patch[(str(row["subtype"]), str(row["patch_type"]))].append(row)

    layer_curve: dict[str, dict[str, float]] = {}
    for (layer, patch_type), rows in sorted(by_layer_patch.items()):
        slot = layer_curve.setdefault(str(layer), {})
        slot[f"{patch_type}_progress"] = mean([float(r["progress"]) for r in rows])
        slot[f"{patch_type}_kl_ratio"] = mean([float(r["kl_ratio"]) for r in rows])
        slot[f"{patch_type}_logit_delta_ratio"] = mean([float(r["logit_delta_ratio"]) for r in rows])

    alpha_curve: dict[str, dict[str, dict[str, float]]] = {}
    for (layer, alpha, patch_type), rows in sorted(by_layer_alpha_patch.items()):
        layer_slot = alpha_curve.setdefault(str(layer), {})
        alpha_slot = layer_slot.setdefault(str(alpha), {})
        alpha_slot[f"{patch_type}_progress"] = mean([float(r["progress"]) for r in rows])
        alpha_slot[f"{patch_type}_kl_ratio"] = mean([float(r["kl_ratio"]) for r in rows])
        alpha_slot[f"{patch_type}_logit_delta_ratio"] = mean([float(r["logit_delta_ratio"]) for r in rows])

    subtype_summary: dict[str, dict[str, float]] = {}
    for (subtype, patch_type), rows in sorted(by_subtype_patch.items()):
        slot = subtype_summary.setdefault(subtype, {})
        slot[f"{patch_type}_progress"] = mean([float(r["progress"]) for r in rows])
        slot[f"{patch_type}_kl_ratio"] = mean([float(r["kl_ratio"]) for r in rows])
        slot[f"{patch_type}_logit_delta_ratio"] = mean([float(r["logit_delta_ratio"]) for r in rows])

    contract_events = []
    for layer, vals in layer_curve.items():
        both = vals.get("both_kl_ratio", 0.0)
        both_progress = vals.get("both_progress", 0.0)
        for cross_name in ("cross_battn_amlp", "cross_aattn_bmlp"):
            cross_kl = vals.get(f"{cross_name}_kl_ratio", 0.0)
            cross_progress = vals.get(f"{cross_name}_progress", 0.0)
            cross_delta = vals.get(f"{cross_name}_logit_delta_ratio", 0.0)
            ratio = cross_kl / max(both, 1e-6)
            progress_drop = both_progress - cross_progress
            if (
                ratio >= 2.0
                and cross_kl >= 0.5
                and progress_drop >= 0.25
                and cross_delta >= 0.15
            ):
                contract_events.append({
                    "layer": int(layer),
                    "cross_type": cross_name,
                    "cross_kl_ratio": cross_kl,
                    "both_kl_ratio": both,
                    "kl_ratio_vs_both": ratio,
                    "cross_progress": cross_progress,
                    "both_progress": both_progress,
                    "progress_drop": progress_drop,
                    "cross_logit_delta_ratio": cross_delta,
                })

    best_layer = None
    best_value = -math.inf
    for layer, vals in layer_curve.items():
        val = vals.get("both_progress", 0.0)
        if val > best_value:
            best_value = val
            best_layer = int(layer)

    return {
        "layer_curve": layer_curve,
        "alpha_curve": alpha_curve,
        "subtype_summary": subtype_summary,
        "contract_events": contract_events,
        "contract_broken_layers": sorted({event["layer"] for event in contract_events}),
        "best_layer_by_both_progress": best_layer,
        "nonfinite_rows": sum(1 for row in results if not bool(row.get("finite", 1.0))),
    }


def checkpoint_path(output_dir: Path, model: str, category: str, pilot: bool) -> Path:
    suffix = "pilot" if pilot else "full"
    return output_dir / "checkpoints" / model / f"{category}_{suffix}.json"


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_pairs = build_pairs()
    categories = set(parse_csv(args.categories)) if args.categories else None
    subtypes = set(parse_csv(args.subtypes)) if args.subtypes else None
    max_pairs = args.max_pairs_per_subtype
    pairs = select_pairs(all_pairs, categories, subtypes, max_pairs)
    if not pairs:
        raise SystemExit("No pairs selected")

    category_label = "-".join(sorted({p.category for p in pairs}))
    ckpt = checkpoint_path(output_dir, args.model, category_label, args.pilot)
    if args.resume and ckpt.exists():
        data = json.loads(ckpt.read_text(encoding="utf-8"))
        if data.get("complete"):
            log(f"checkpoint already complete: {ckpt}")
            return data

    loaded = None
    try:
        loaded = load_probe_model(args.model)
        layers = get_layers(loaded.model)
        n_layers = len(layers)
        target_layers = choose_layers(n_layers, args.layer_stride, args.layers)
        alphas = parse_alphas(args.alphas)
        patch_types = parse_csv(args.patch_types)

        log(f"model={args.model} class={type(loaded.model).__name__} layers={n_layers}")
        log(f"pairs={len(pairs)} categories={sorted({p.category for p in pairs})}")
        log(f"target_layers={target_layers}")
        log(f"alphas={alphas} patch_types={patch_types}")

        results: list[dict[str, Any]] = []
        start = time.time()
        for pair_index, pair in enumerate(pairs):
            toks_a = len(loaded.tokenizer.encode(pair.a, add_special_tokens=True))
            toks_b = len(loaded.tokenizer.encode(pair.b, add_special_tokens=True))
            seq_len = min(max(toks_a, toks_b), args.max_seq_len)

            out_a = capture_outputs(loaded, pair.a, target_layers, seq_len)
            out_b = capture_outputs(loaded, pair.b, target_layers, seq_len)
            logits_a = baseline_logits(loaded, pair.a, seq_len)
            logits_b = baseline_logits(loaded, pair.b, seq_len)
            kl_ab = float(F.kl_div(F.log_softmax(logits_a, dim=-1), F.softmax(logits_b, dim=-1), reduction="sum"))
            if kl_ab < 1e-8:
                continue

            for layer_idx in target_layers:
                if layer_idx not in out_a or layer_idx not in out_b:
                    continue
                if not {"attn", "mlp", "resid"}.issubset(out_a[layer_idx]):
                    continue
                if not {"attn", "mlp", "resid"}.issubset(out_b[layer_idx]):
                    continue
                for alpha in alphas:
                    for patch_type in patch_types:
                        patched, natural = forward_patch(
                            loaded,
                            pair.a,
                            seq_len,
                            layer_idx,
                            patch_type,
                            alpha,
                            out_a[layer_idx],
                            out_b[layer_idx],
                        )
                        metrics = compute_metrics(patched, logits_a, logits_b, kl_ab)
                        if metrics is None:
                            metrics = {
                                "kl_ratio": float("nan"),
                                "progress": float("nan"),
                                "cos_dir": float("nan"),
                                "logit_delta_ratio": float("nan"),
                                "finite": 0.0,
                            }
                        results.append({
                            "pair": pair.name,
                            "category": pair.category,
                            "subtype": pair.subtype,
                            "layer": layer_idx,
                            "alpha": alpha,
                            "patch_type": patch_type,
                            "kl_ab": kl_ab,
                            **metrics,
                            **natural,
                        })

            if (pair_index + 1) % args.progress_every == 0:
                elapsed = time.time() - start
                log(f"progress pairs={pair_index + 1}/{len(pairs)} results={len(results)} elapsed={elapsed:.1f}s")
                partial = {
                    "model": args.model,
                    "complete": False,
                    "num_pairs": len(pairs),
                    "num_results": len(results),
                    "target_layers": target_layers,
                    "alphas": alphas,
                    "patch_types": patch_types,
                    "results": results,
                    "summary": summarize(results),
                }
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                ckpt.write_text(json.dumps(partial, indent=2), encoding="utf-8")

        data = {
            "model": args.model,
            "class": type(loaded.model).__name__,
            "complete": True,
            "pilot": bool(args.pilot),
            "num_pairs": len(pairs),
            "num_results": len(results),
            "categories": sorted({p.category for p in pairs}),
            "subtypes": sorted({p.subtype for p in pairs}),
            "target_layers": target_layers,
            "alphas": alphas,
            "patch_types": patch_types,
            "results": results,
            "summary": summarize(results),
        }
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        ckpt.write_text(json.dumps(data, indent=2), encoding="utf-8")
        (output_dir / f"{args.model}_phase289_contract_scan.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )
        log(f"saved checkpoint={ckpt}")
        return data
    finally:
        if not args.hard_exit_after_model:
            release_loaded(loaded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "results" / "gpt5_phase289_contract_scan"))
    parser.add_argument("--categories", default="negation")
    parser.add_argument("--subtypes", default="")
    parser.add_argument("--max-pairs-per-subtype", type=int, default=2)
    parser.add_argument("--layer-stride", type=int, default=12)
    parser.add_argument("--layers", default="")
    parser.add_argument("--alphas", default="0,0.5,1.0")
    parser.add_argument("--patch-types", default="attn,mlp,both,resid,cross_battn_amlp,cross_aattn_bmlp")
    parser.add_argument("--max-seq-len", type=int, default=48)
    parser.add_argument("--progress-every", type=int, default=2)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--hard-exit-after-model", action="store_true")
    args = parser.parse_args()

    data = run(args)
    summary = data.get("summary", {})
    log(
        "done "
        f"model={data.get('model')} pairs={data.get('num_pairs')} results={data.get('num_results')} "
        f"best_layer={summary.get('best_layer_by_both_progress')} "
        f"broken_layers={summary.get('contract_broken_layers')}"
    )
    if args.hard_exit_after_model:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
