"""
Phase 288v2: Attention vs MLP Causal Decomposition (Activation Patching)
=========================================================================
REVISED APPROACH: Skip AW@V reconstruction entirely.
Instead, use standard activation patching at module-output level:

  1. Hook attention block output (o_proj output, before residual add)
  2. Hook MLP block output
  3. Patch B's attention/MLP outputs into A's forward
  4. Measure KL effect relative to natural A→B gap

This is the reliable activation patching approach, decomposed into
attention-path and MLP-path contributions.

PER-MODEL: ~250 pairs × 3 patch types × 5 layer configs ≈ 3750 forwards
Each forward ~0.2s → ~12 min per model

Usage:
  python tests/glm5/phase288_attn_mlp_decomp.py qwen3
  python tests/glm5/phase288_attn_mlp_decomp.py glm4
  python tests/glm5/phase288_attn_mlp_decomp.py deepseek7b
"""
import sys, os, json, gc, time, warnings
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
from model_utils import MODEL_CONFIGS, get_model_info, get_layers, release_model

RESULT_DIR = Path("results/phase288_attn_mlp")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None

def log_time(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        try:
            with open(_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except:
            pass

# ==============================================================================
# DATASET (same as original Phase 288)
# ==============================================================================

def build_all_pairs():
    """250 pairs across 6 functions."""
    pairs = []

    # Negation: 80 pairs
    for name, pos, neg in [
        ("not_happy", "she is happy", "she is not happy"),
        ("not_open", "the door is open", "the door is not open"),
        ("not_possible", "victory is possible", "victory is not possible"),
        ("not_ready", "they are ready", "they are not ready"),
        ("not_important", "this is important", "this is not important"),
        ("not_clear", "the answer is clear", "the answer is not clear"),
        ("not_safe", "the area is safe", "the area is not safe"),
        ("not_fair", "the decision is fair", "the decision is not fair"),
        ("not_simple", "the problem is simple", "the problem is not simple"),
        ("not_correct", "your answer is correct", "your answer is not correct"),
        ("do_not_agree", "they agree with the proposal", "they do not agree with the proposal"),
        ("do_not_remember", "i remember the meeting", "i do not remember the meeting"),
        ("do_not_understand", "we understand the problem", "we do not understand the problem"),
        ("does_not_know", "she knows the answer", "she does not know the answer"),
        ("does_not_believe", "he believes the story", "he does not believe the story"),
        ("do_not_support", "they support the plan", "they do not support the plan"),
        ("does_not_accept", "she accepts the offer", "she does not accept the offer"),
        ("do_not_expect", "we expect rain", "we do not expect rain"),
        ("does_not_trust", "he trusts the source", "he does not trust the source"),
        ("do_not_follow", "they follow the rules", "they do not follow the rules"),
        ("found_nothing", "he found something interesting", "he found nothing interesting"),
        ("no_one_came", "someone came to the party", "no one came to the party"),
        ("no_food", "there was some food left", "there was no food left"),
        ("no_idea", "she had some idea what to do", "she had no idea what to do"),
        ("no_reason", "there is a reason to worry", "there is no reason to worry"),
        ("never_seen", "i have seen it before", "i have never seen it before"),
        ("never_been", "she has been to Paris", "she has never been to Paris"),
        ("never_told", "he told someone the secret", "he never told anyone the secret"),
        ("never_gives_up", "she sometimes gives up", "she never gives up"),
        ("never_forgets", "he sometimes forgets names", "he never forgets a face"),
        ("impossible", "the task is possible", "the task is impossible"),
        ("unacceptable", "the proposal is acceptable", "the proposal is unacceptable"),
        ("incomplete", "the report is complete", "the report is incomplete"),
        ("irrelevant", "the comment is relevant", "the comment is irrelevant"),
        ("dishonest", "the person is honest", "the person is dishonest"),
        ("not_all", "all birds can fly", "not all birds can fly"),
        ("not_everyone", "everyone agreed", "not everyone agreed"),
        ("not_always", "she always tells the truth", "she does not always tell the truth"),
        ("not_entirely", "the plan is entirely successful", "the plan is not entirely successful"),
        ("not_necessarily", "wealth means happiness", "wealth does not necessarily mean happiness"),
    ]:
        pairs.append({"name": f"neg_{name}", "A": pos, "B": neg, "category": "negation"})

    # Translation: 40 pairs
    for name, en, zh in [
        ("dog_cat", "the dog chases the cat", "狗追猫"),
        ("sun_east", "the sun rises in the east", "太阳从东方升起"),
        ("teacher", "the teacher teaches the student", "老师教学生"),
        ("bird_sky", "the bird flies in the sky", "鸟在天空中飞翔"),
        ("water_cold", "the water is very cold", "水非常冷"),
        ("love_overcome", "love overcomes everything", "爱战胜一切"),
        ("child_happy", "the child is very happy", "孩子非常快乐"),
        ("mountain_high", "the mountain is extremely high", "这座山非常高"),
        ("horse_white", "the horse is white", "这匹马是白色的"),
        ("book_interest", "the book is very interesting", "这本书非常有趣"),
        ("city_busy", "the city is very busy", "这个城市非常繁忙"),
        ("food_delicious", "the food is delicious", "食物很美味"),
        ("music_beautiful", "the music is beautiful", "音乐很美"),
        ("garden_small", "the garden is small", "花园很小"),
        ("house_large", "the house is very large", "这栋房子非常大"),
        ("wind_strong", "the wind is strong today", "今天的风很大"),
        ("rain_heavy", "the rain is heavy", "雨下得很大"),
        ("sky_blue", "the sky is blue", "天空是蓝色的"),
        ("flower_red", "the flower is red", "花是红色的"),
        ("fish_fresh", "the fish is fresh", "鱼很新鲜"),
    ]:
        pairs.append({"name": f"trans_{name}", "A": en, "B": zh, "category": "translation"})

    # Logical: 40 pairs
    for name, a, b in [
        ("and_or_catdog", "the cat and the dog are sleeping", "the cat or the dog is sleeping"),
        ("and_or_birds", "birds and bees are pollinators", "birds or bees are pollinators"),
        ("and_or_sun", "the sun and the moon are visible", "the sun or the moon is visible"),
        ("and_or_tea", "tea and coffee are served", "tea or coffee is served"),
        ("and_or_apples", "apples and oranges are fruits", "apples or oranges are fruits"),
        ("if_rain", "if it rains we will stay home", "we will stay home if it rains"),
        ("if_hungry", "if you are hungry eat something", "eat something if you are hungry"),
        ("if_tired", "if she is tired she will rest", "she will rest if she is tired"),
        ("if_cold", "if it gets cold turn on the heater", "turn on the heater if it gets cold"),
        ("if_ready", "if they are ready we can go", "we can go if they are ready"),
        ("because_rain", "because it rained we stayed home", "we stayed home because it rained"),
        ("because_hungry", "because he was hungry he ate", "he ate because he was hungry"),
        ("because_sick", "because she was sick she rested", "she rested because she was sick"),
        ("because_late", "because he was late he ran", "he ran because he was late"),
        ("because_cold", "because it was cold they lit a fire", "they lit a fire because it was cold"),
        ("although_rain", "although it rained they went out", "they went out although it rained"),
        ("although_tired", "although she was tired she continued", "she continued although she was tired"),
        ("although_small", "although the dog was small it was brave", "the dog was brave although it was small"),
        ("therefore_rain", "it is raining therefore we will stay home", "we will stay home therefore it is raining"),
        ("therefore_late", "he overslept therefore he was late", "he was late therefore he overslept"),
    ]:
        pairs.append({"name": f"logic_{name}", "A": a, "B": b, "category": "logical"})

    # Passive: 30 pairs
    for name, active, passive in [
        ("dog_cat", "the dog chases the cat", "the cat is chased by the dog"),
        ("teacher_student", "the teacher teaches the student", "the student is taught by the teacher"),
        ("author_book", "the author wrote the book", "the book was written by the author"),
        ("workers_bridge", "the workers built the bridge", "the bridge was built by the workers"),
        ("detective_clue", "the detective found the clue", "the clue was found by the detective"),
        ("cat_fish", "the cat ate the fish", "the fish was eaten by the cat"),
        ("musician_song", "the musician wrote the song", "the song was written by the musician"),
        ("chef_meal", "the chef prepared the meal", "the meal was prepared by the chef"),
        ("gardener_trees", "the gardener planted the trees", "the trees were planted by the gardener"),
        ("police_thief", "the police caught the thief", "the thief was caught by the police"),
        ("judge_case", "the judge reviewed the case", "the case was reviewed by the judge"),
        ("doctor_patient", "the doctor treated the patient", "the patient was treated by the doctor"),
        ("driver_car", "the driver parked the car", "the car was parked by the driver"),
        ("student_essay", "the student wrote the essay", "the essay was written by the student"),
        ("waiter_order", "the waiter took the order", "the order was taken by the waiter"),
        ("coach_team", "the coach trained the team", "the team was trained by the coach"),
        ("editor_article", "the editor revised the article", "the article was revised by the editor"),
        ("director_film", "the director produced the film", "the film was produced by the director"),
        ("tailor_suit", "the tailor made the suit", "the suit was made by the tailor"),
        ("baker_bread", "the baker made the bread", "the bread was made by the baker"),
    ]:
        pairs.append({"name": f"pass_{name}", "A": active, "B": passive, "category": "passive"})

    # Comparative: 30 pairs
    for name, a, b in [
        ("bigger_smaller", "the elephant is bigger than the mouse", "the mouse is smaller than the elephant"),
        ("taller_shorter", "John is taller than Mary", "Mary is shorter than John"),
        ("more_expensive", "gold is more expensive than silver", "silver is less expensive than gold"),
        ("faster_slower", "the train is faster than the bicycle", "the bicycle is slower than the train"),
        ("stronger_weaker", "steel is stronger than wood", "wood is weaker than steel"),
        ("smarter_dumber", "Alice is smarter than Bob", "Bob is less smart than Alice"),
        ("older_younger", "the grandfather is older than the father", "the father is younger than the grandfather"),
        ("more_fewer", "she has more books than he does", "he has fewer books than she does"),
        ("hotter_colder", "summer is hotter than spring", "spring is colder than summer"),
        ("darker_lighter", "midnight is darker than dusk", "dusk is lighter than midnight"),
        ("richer_poorer", "the king is richer than the merchant", "the merchant is poorer than the king"),
        ("louder_quieter", "thunder is louder than rain", "rain is quieter than thunder"),
        ("heavier_lighter", "iron is heavier than aluminum", "aluminum is lighter than iron"),
        ("sweeter_sour", "honey is sweeter than lemon", "lemon is sourer than honey"),
        ("longer_shorter", "the Nile is longer than the Thames", "the Thames is shorter than the Nile"),
        ("wider_narrower", "the highway is wider than the alley", "the alley is narrower than the highway"),
        ("brighter_dimmer", "the sun is brighter than the moon", "the moon is dimmer than the sun"),
        ("quicker_slower", "email is quicker than postal mail", "postal mail is slower than email"),
        ("warmer_cooler", "the equator is warmer than the poles", "the poles are cooler than the equator"),
        ("easier_harder", "addition is easier than multiplication", "multiplication is harder than addition"),
    ]:
        pairs.append({"name": f"comp_{name}", "A": a, "B": b, "category": "comparative"})

    # Recursive: 30 pairs
    for name, a, b in [
        ("dog_barked", "the dog that barked ran away", "the barking dog ran away"),
        ("woman_wrote", "the woman who wrote the letter smiled", "the letter-writing woman smiled"),
        ("king_said", "the king said that the queen is wise", "the queen is wise said the king"),
        ("door_painted", "the door that was painted red opened", "the red door opened"),
        ("book_which", "the book which i read yesterday was great", "yesterday i read a great book"),
        ("person_who", "the person who called left a message", "a message was left by the caller"),
        ("nested_if", "the man who said that if it rains he will leave arrived", "the man arrived who said he will leave if it rains"),
        ("teacher_think", "the teacher thinks that the student learned well", "the student is thought by the teacher to have learned well"),
        ("fact_surprise", "the fact that he lied surprised everyone", "everyone was surprised that he lied"),
        ("cat_slept", "the cat that slept on the mat was fluffy", "the fluffy cat slept on the mat"),
        ("man_hat", "the man wearing a hat entered the room", "a man with a hat entered the room"),
        ("boy_cried", "the boy who cried wolf was ignored", "the crying-wolf boy was ignored"),
        ("car_broken", "the car that broke down was towed", "the broken-down car was towed"),
        ("child_lost", "the child who got lost was helped", "the lost child was helped"),
        ("house_built", "the house that Jack built stands tall", "Jack's house stands tall"),
        ("story_told", "the story that grandma told was scary", "grandma's scary story"),
        ("dog_found", "the dog that found the bone was happy", "the bone-finding dog was happy"),
        ("tree_fell", "the tree that fell during the storm blocked the road", "the storm-fallen tree blocked the road"),
        ("letter_arrived", "the letter that arrived yesterday was important", "yesterday's letter was important"),
        ("dog_chased_cat", "the dog that chased the cat that caught the mouse barked", "the dog barked after chasing the cat who caught the mouse"),
    ]:
        pairs.append({"name": f"rec_{name}", "A": a, "B": b, "category": "recursive"})

    return pairs


# ==============================================================================
# MODEL LOADING
# ==============================================================================

def load_model_bf16(model_name: str):
    """Load with bfloat16 + device_map="auto" (consistent with model_demo_bf16.py)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bf16 + device_map=auto, eager)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"  Loaded: device={device}, GPU={gpu_mem:.2f}GB, class={type(model).__name__}")
    return model, tokenizer, device


# ==============================================================================
# CORE: Capture Attention and MLP outputs via hooks
# ==============================================================================

def capture_layer_outputs(model, tokenizer, sentence, device, target_layers, common_len):
    """
    Forward sentence and capture attention block output + MLP output at target layers.
    
    Attention block output = o_proj output (the "attention contribution" to residual)
    MLP output = mlp output (the "MLP contribution" to residual)
    
    Returns: {"L{li}": {"attn": tensor[1,seq,d_model], "mlp": tensor[1,seq,d_model]}}
    """
    layers = get_layers(model)
    n_layers = len(layers)

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=common_len).to(device)
    seq_len = inputs["input_ids"].shape[1]
    if seq_len < common_len:
        pad_len = common_len - seq_len
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, pad_len), value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, pad_len), value=0)

    captured = {}
    hooks = []

    def make_attn_hook(li):
        # Hook self_attn output (after o_proj)
        def hook(module, input_t, output_t):
            if isinstance(output_t, tuple):
                val = output_t[0].detach().cpu().clone()
            else:
                val = output_t.detach().cpu().clone()
            captured.setdefault(f"L{li}", {})["attn"] = val
        return hook

    def make_mlp_hook(li):
        def hook(module, input_t, output_t):
            if isinstance(output_t, tuple):
                val = output_t[0].detach().cpu().clone()
            else:
                val = output_t.detach().cpu().clone()
            captured.setdefault(f"L{li}", {})["mlp"] = val
        return hook

    for li in target_layers:
        if li < n_layers:
            hooks.append(layers[li].self_attn.register_forward_hook(make_attn_hook(li)))
            hooks.append(layers[li].mlp.register_forward_hook(make_mlp_hook(li)))

    with torch.no_grad():
        try:
            model(**inputs)
        except Exception as e:
            log_time(f"  capture_layer_outputs FAILED: {e}")
            captured = {}

    for h in hooks:
        h.remove()

    return captured


# ==============================================================================
# PATCHING: Forward A with B's outputs injected
# ==============================================================================

def forward_with_patches(model, tokenizer, sentence, device, common_len,
                         attn_patches, mlp_patches):
    """
    Forward sentence with attention and/or MLP outputs replaced.
    
    attn_patches: dict {layer_idx: tensor[1,seq,d_model]} — B's attention outputs
    mlp_patches: dict {layer_idx: tensor[1,seq,d_model]} — B's MLP outputs
    
    Returns: logits [vocab_size] at last token
    """
    layers = get_layers(model)
    n_layers = len(layers)

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=common_len).to(device)
    seq_len = inputs["input_ids"].shape[1]
    if seq_len < common_len:
        pad_len = common_len - seq_len
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, pad_len), value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, pad_len), value=0)

    hooks = []

    for li in attn_patches:
        if li >= n_layers:
            continue
        patch_val = attn_patches[li]
        # Pre-get o_proj device/dtype (hook fires on self_attn, so module IS self_attn)
        o_proj = layers[li].self_attn.o_proj
        target_device = o_proj.weight.device
        target_dtype = o_proj.weight.dtype

        def make_attn_patch_hook(patch_val, target_device, target_dtype):
            def hook(module, input_t, output_t):
                pv = patch_val.to(target_device).to(target_dtype)
                if isinstance(output_t, tuple):
                    out_ref = output_t[0]
                    min_s = min(pv.shape[1], out_ref.shape[1])
                    new_out = (out_ref.clone(),) + output_t[1:]
                    new_out[0][:, :min_s, :] = pv[:, :min_s, :]
                    return new_out
                else:
                    min_s = min(pv.shape[1], output_t.shape[1])
                    new_out = output_t.clone()
                    new_out[:, :min_s, :] = pv[:, :min_s, :]
                    return new_out
            return hook

        hooks.append(layers[li].self_attn.register_forward_hook(
            make_attn_patch_hook(patch_val, target_device, target_dtype)))

    for li in mlp_patches:
        if li >= n_layers:
            continue
        patch_val = mlp_patches[li]
        # Pre-get mlp output device/dtype from first param
        mlp_module = layers[li].mlp
        target_device_mlp = next(mlp_module.parameters()).device
        target_dtype_mlp = next(mlp_module.parameters()).dtype

        def make_mlp_patch_hook(patch_val, target_device_mlp, target_dtype_mlp):
            def hook(module, input_t, output_t):
                pv = patch_val.to(target_device_mlp).to(target_dtype_mlp)
                if isinstance(output_t, tuple):
                    min_s = min(pv.shape[1], output_t[0].shape[1])
                    new_out = (output_t[0].clone(),) + output_t[1:]
                    new_out[0][:, :min_s, :] = pv[:, :min_s, :]
                    return new_out
                else:
                    min_s = min(pv.shape[1], output_t.shape[1])
                    new_out = output_t.clone()
                    new_out[:, :min_s, :] = pv[:, :min_s, :]
                    return new_out
            return hook

        hooks.append(layers[li].mlp.register_forward_hook(
            make_mlp_patch_hook(patch_val, target_device_mlp, target_dtype_mlp)))

    with torch.no_grad():
        try:
            out = model(**inputs)
            result = out.logits[0, -1, :].detach().cpu().float().clone()
        except Exception as e:
            result = None

    for h in hooks:
        h.remove()

    return result


def compute_effect(patched_logits, logits_a, logits_b, kl_ab):
    """Compute KL ratio and progress towards B."""
    if patched_logits is None or logits_a is None or logits_b is None:
        return None
    
    kl_p = float(F.kl_div(
        F.log_softmax(patched_logits, dim=-1),
        F.softmax(logits_b, dim=-1),
        reduction='sum'
    ))
    kl_ratio = kl_p / max(kl_ab, 1e-6)
    
    # Direction projection: how much toward B?
    delta_B = logits_b - logits_a
    delta_patch = patched_logits - logits_a
    norm_B = float(torch.norm(delta_B))
    norm_p = float(torch.norm(delta_patch))
    if norm_B > 1e-8 and norm_p > 1e-8:
        cos_dir = float(torch.dot(delta_patch, delta_B) / (norm_B * norm_p))
        progress = cos_dir * min(norm_p / norm_B, 2.0)
    else:
        cos_dir = 0.0
        progress = 0.0
    
    return {"kl_ratio": kl_ratio, "cos_dir": cos_dir, "progress": progress}


# ==============================================================================
# 主流程
# ==============================================================================

def run_phase288v2(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase288v2_{model_name}.txt")

    log_time(f"{'='*60}")
    log_time(f"Phase 288v2: Attention vs MLP Causal Decomposition — {model_name}")
    log_time(f"{'='*60}")

    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    log_time(f"Model: {model_info.model_class}, L={n_layers}, d_model={model_info.d_model}")

    # Warmup
    wu = tokenizer("warmup test", return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            model(**wu)
        except:
            pass

    # Layer configurations
    layer_configs = [
        ("early", [0, 1, 2]),
        ("mid", [n_layers // 2 - 1, n_layers // 2, n_layers // 2 + 1]),
        ("late", [n_layers - 3, n_layers - 2, n_layers - 1]),
        ("all3", [0, n_layers // 2, n_layers - 1]),
    ]
    
    all_target_layers = sorted(set(sum([lc[1] for lc in layer_configs], [])))
    all_target_layers = [li for li in all_target_layers if 0 <= li < n_layers]

    # Build dataset
    all_pairs = build_all_pairs()
    log_time(f"Dataset: {len(all_pairs)} pairs")
    for cat in sorted(set(p["category"] for p in all_pairs)):
        n = sum(1 for p in all_pairs if p["category"] == cat)
        log_time(f"  {cat}: {n} pairs")

    # For speed: subsample to ~50 pairs total (ensure per-category coverage)
    # But for key functions (negation), use all
    patching_pairs = all_pairs  # Use all pairs
    log_time(f"Using all {len(patching_pairs)} pairs for patching")

    # ============================================================
    # PHASE 1: Capture outputs for all pairs
    # ============================================================
    log_time(f"\n{'='*60}")
    log_time("PHASE 1: Capture attention & MLP outputs")
    log_time(f"{'='*60}")

    pair_outputs = {}  # {pname: {"A": {L{li}: {"attn":..., "mlp":...}}, "B": ...}}
    pair_baselines = {}

    max_len = 48
    t0 = time.time()

    for pidx, pair in enumerate(patching_pairs):
        pname = pair["name"]
        sent_a, sent_b = pair["A"], pair["B"]
        cat = pair["category"]
        pair["_cl"] = None  # will be set below
        
        toks_a = len(tokenizer.encode(sent_a, add_special_tokens=True))
        toks_b = len(tokenizer.encode(sent_b, add_special_tokens=True))
        cl = min(max(toks_a, toks_b), max_len)
        
        # Capture A outputs
        out_a = capture_layer_outputs(model, tokenizer, sent_a, device, all_target_layers, cl)
        # Capture B outputs
        out_b = capture_layer_outputs(model, tokenizer, sent_b, device, all_target_layers, cl)
        
        if out_a and out_b:
            pair_outputs[pname] = {"A": out_a, "B": out_b, "category": cat, "seq_len": cl}
        
        # Baseline logits
        ia = tokenizer(sent_a, return_tensors="pt", truncation=True, max_length=cl).to(device)
        ib = tokenizer(sent_b, return_tensors="pt", truncation=True, max_length=cl).to(device)
        with torch.no_grad():
            logits_a = model(**ia).logits[0, -1, :].detach().cpu().float()
            logits_b = model(**ib).logits[0, -1, :].detach().cpu().float()
        kl_ab = float(F.kl_div(F.log_softmax(logits_a, dim=-1), F.softmax(logits_b, dim=-1), reduction='sum'))
        pair_baselines[pname] = {"logits_a": logits_a, "logits_b": logits_b, "kl_ab": kl_ab,
                                  "sent_a": sent_a, "sent_b": sent_b, "category": cat, "seq_len": cl}

        if (pidx + 1) % 40 == 0:
            elapsed = time.time() - t0
            rate = (pidx + 1) / max(elapsed, 1) * 3600
            log_time(f"  [{pidx+1}/{len(patching_pairs)}] Captured: {elapsed:.0f}s, "
                     f"~{rate:.0f} pairs/h, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

    t_capture = time.time() - t0
    log_time(f"  Capture done: {len(pair_outputs)} pairs, {t_capture:.0f}s ({t_capture/60:.1f}min)")

    # ============================================================
    # PHASE 2: Patching — Attention vs MLP
    # ============================================================
    log_time(f"\n{'='*60}")
    log_time("PHASE 2: Activation Patching — Attn vs MLP")
    log_time(f"{'='*60}")

    all_results = []
    t0_patch = time.time()
    n_done = 0

    for lc_name, lc_layers in layer_configs:
        lc_layers = [li for li in lc_layers if 0 <= li < n_layers]
        log_time(f"\n  Layer config: {lc_name} → {lc_layers}")
        
        for pidx, pair in enumerate(patching_pairs):
            pname = pair["name"]
            if pname not in pair_outputs or pname not in pair_baselines:
                continue
            
            pb = pair_baselines[pname]
            po = pair_outputs[pname]
            sent_a = pb["sent_a"]
            logits_a = pb["logits_a"]
            logits_b = pb["logits_b"]
            kl_ab = pb["kl_ab"]
            cl = pb["seq_len"]
            
            if kl_ab < 1e-6:
                continue
            
            # Prepare B's outputs for patching
            b_attn = {}
            b_mlp = {}
            for li in lc_layers:
                lk = f"L{li}"
                if lk in po.get("B", {}):
                    b_data = po["B"][lk]
                    if "attn" in b_data:
                        b_attn[li] = b_data["attn"]
                    if "mlp" in b_data:
                        b_mlp[li] = b_data["mlp"]
            
            if not b_attn and not b_mlp:
                continue
            
            # Test 1: Patch attention only
            patched_attn = forward_with_patches(model, tokenizer, sent_a, device, cl, b_attn, {})
            effect_attn = compute_effect(patched_attn, logits_a, logits_b, kl_ab)
            if effect_attn:
                all_results.append({
                    "pname": pname, "category": pb["category"],
                    "layer_config": lc_name, "patch_type": "attn",
                    "kl_ab": kl_ab, **effect_attn,
                })
            
            # Test 2: Patch MLP only
            patched_mlp = forward_with_patches(model, tokenizer, sent_a, device, cl, {}, b_mlp)
            effect_mlp = compute_effect(patched_mlp, logits_a, logits_b, kl_ab)
            if effect_mlp:
                all_results.append({
                    "pname": pname, "category": pb["category"],
                    "layer_config": lc_name, "patch_type": "mlp",
                    "kl_ab": kl_ab, **effect_mlp,
                })
            
            # Test 3: Patch BOTH
            patched_both = forward_with_patches(model, tokenizer, sent_a, device, cl, b_attn, b_mlp)
            effect_both = compute_effect(patched_both, logits_a, logits_b, kl_ab)
            if effect_both:
                all_results.append({
                    "pname": pname, "category": pb["category"],
                    "layer_config": lc_name, "patch_type": "both",
                    "kl_ab": kl_ab, **effect_both,
                })
            
            n_done += 1
        
        elapsed = time.time() - t0_patch
        log_time(f"    {lc_name} done: {n_done} pairs processed, {elapsed:.0f}s")

    t_patch = time.time() - t0_patch
    log_time(f"\n  Patching done: {len(all_results)} results, {t_patch:.0f}s ({t_patch/60:.1f}min)")

    # ============================================================
    # ANALYSIS
    # ============================================================
    log_time(f"\n{'='*60}")
    log_time("ANALYSIS")
    log_time(f"{'='*60}")

    # Per-category, per-layer-config analysis
    log_time(f"\n  {'Category':>16} {'LC':>8} {'Attn_KR':>8} {'Attn_Prog':>9} "
             f"{'MLP_KR':>8} {'MLP_Prog':>9} {'Both_KR':>8} {'Both_Prog':>9} {'Dominant':>12}")

    cat_lc_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in all_results:
        cat_lc_agg[r["category"]][r["layer_config"]][r["patch_type"]].append(r)

    cat_summary = {}
    for cat in sorted(cat_lc_agg.keys()):
        for lc in sorted(cat_lc_agg[cat].keys()):
            d = cat_lc_agg[cat][lc]
            
            a_vals = d.get("attn", [])
            m_vals = d.get("mlp", [])
            b_vals = d.get("both", [])
            
            a_kr = np.mean([x["kl_ratio"] for x in a_vals]) if a_vals else 0
            a_prog = np.mean([x["progress"] for x in a_vals]) if a_vals else 0
            m_kr = np.mean([x["kl_ratio"] for x in m_vals]) if m_vals else 0
            m_prog = np.mean([x["progress"] for x in m_vals]) if m_vals else 0
            b_kr = np.mean([x["kl_ratio"] for x in b_vals]) if b_vals else 0
            b_prog = np.mean([x["progress"] for x in b_vals]) if b_vals else 0
            
            # Dominant: which patch type has highest progress?
            max_prog = max(a_prog, m_prog, b_prog)
            if max_prog < 0.01:
                dom = "no_effect"
            elif a_prog >= max_prog:
                dom = "ATTN"
            elif m_prog >= max_prog:
                dom = "MLP"
            else:
                dom = "BOTH"
            
            key = f"{cat}/{lc}"
            cat_summary[key] = {
                "category": cat, "layer_config": lc,
                "attn_kl_ratio": float(a_kr), "attn_progress": float(a_prog),
                "mlp_kl_ratio": float(m_kr), "mlp_progress": float(m_prog),
                "both_kl_ratio": float(b_kr), "both_progress": float(b_prog),
                "dominant": dom,
                "n_attn": len(a_vals), "n_mlp": len(m_vals), "n_both": len(b_vals),
            }
            
            log_time(f"  {cat:>16} {lc:>8} {a_kr:8.3f} {a_prog:9.4f} "
                     f"{m_kr:8.3f} {m_prog:9.4f} {b_kr:8.3f} {b_prog:9.4f} {dom:>12}")

    # Global: per-patch-type averages
    log_time(f"\n  GLOBAL AVERAGES by patch type:")
    for ptype in ["attn", "mlp", "both"]:
        vals = [r for r in all_results if r["patch_type"] == ptype]
        if vals:
            avg_kr = np.mean([v["kl_ratio"] for v in vals])
            avg_prog = np.mean([v["progress"] for v in vals])
            avg_cos = np.mean([v["cos_dir"] for v in vals])
            n_gt1 = sum(1 for v in vals if v["kl_ratio"] > 1.0)
            log_time(f"    {ptype}: N={len(vals)}, kl_ratio={avg_kr:.3f}, "
                     f"progress={avg_prog:.4f}, cos_dir={avg_cos:.4f}, >1x={n_gt1}/{len(vals)} ({100*n_gt1/len(vals):.1f}%)")

    # ============================================================
    # SAVE
    # ============================================================
    save_data = {
        "model": model_name,
        "model_info": {"class": model_info.model_class, "n_layers": n_layers,
                       "d_model": model_info.d_model, "vocab_size": model_info.vocab_size},
        "n_pairs": len(patching_pairs),
        "n_pairs_captured": len(pair_outputs),
        "n_results": len(all_results),
        "layer_configs": layer_configs,
        "cat_lc_summary": cat_summary,
    }

    save_path = RESULT_DIR / f"{model_name}_decomp.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    log_time(f"\nResults saved to {save_path}")

    # Detail
    detail_path = RESULT_DIR / f"{model_name}_detail.json"
    with open(detail_path, "w") as f:
        json.dump(all_results, f, indent=1, default=str)
    log_time(f"Detail saved to {detail_path}")

    # Release
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    return save_data


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            try:
                r = run_phase288v2(name)
                log_time(f"\n{name} DONE: {len(r['cat_lc_summary'])} category/layer configs")
            except Exception as e:
                log_time(f"!!! {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_phase288v2(model_name)
