"""
Phase 289: Layer-Scan Attention-MLP Contract Decomposition
===========================================================
CORE UPGRADES over Phase 288:
  1. ALL LAYERS (not just 3 sampled) → functional formation curves
  2. CONTINUOUS α INTERPOLATION [0, 0.33, 0.67, 1.0] → smooth vs abrupt detection
  3. CROSS PATCHING (A_attn+B_mlp, B_attn+A_mlp) → contract compatibility
  4. NATURALNESS METRICS (norm_ratio, downstream_amplification)
  5. EXPANDED negation pairs (80, covering 6 subtypes)

FOCUS: Negation (most informative function in Phase 288)
  - 80 pairs × ALL layers × 4α × 5 patch types

Usage:
  python tests/glm5/phase289_layer_scan.py qwen3
  python tests/glm5/phase289_layer_scan.py glm4
  python tests/glm5/phase289_layer_scan.py deepseek7b
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

RESULT_DIR = Path("results/phase289_layer_scan")
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
# DATASET: Expanded Negation (80 pairs, 6 subtypes)
# ==============================================================================

def build_negation_pairs():
    """80 negation pairs covering 6 subtypes."""
    pairs = []

    # Type 1: not + adjective (15 pairs)
    adj = [
        ("happy", "she is happy", "she is not happy"),
        ("open", "the door is open", "the door is not open"),
        ("possible", "victory is possible", "victory is not possible"),
        ("ready", "they are ready", "they are not ready"),
        ("important", "this is important", "this is not important"),
        ("clear", "the answer is clear", "the answer is not clear"),
        ("safe", "the area is safe", "the area is not safe"),
        ("fair", "the decision is fair", "the decision is not fair"),
        ("simple", "the problem is simple", "the problem is not simple"),
        ("correct", "your answer is correct", "your answer is not correct"),
        ("reasonable", "the price is reasonable", "the price is not reasonable"),
        ("visible", "the star is visible", "the star is not visible"),
        ("normal", "the situation is normal", "the situation is not normal"),
        ("common", "this bird is common", "this bird is not common"),
        ("stable", "the system is stable", "the system is not stable"),
    ]
    for name, pos, neg in adj:
        pairs.append({"name": f"neg_adj_{name}", "A": pos, "B": neg, "category": "negation", "subtype": "lexical_not_adj"})

    # Type 2: do/does/did not + verb (15 pairs)
    verb_pairs = [
        ("agree", "they agree with the proposal", "they do not agree with the proposal"),
        ("remember", "i remember the meeting", "i do not remember the meeting"),
        ("understand", "we understand the problem", "we do not understand the problem"),
        ("know", "she knows the answer", "she does not know the answer"),
        ("believe", "he believes the story", "he does not believe the story"),
        ("support", "they support the plan", "they do not support the plan"),
        ("accept", "she accepts the offer", "she does not accept the offer"),
        ("expect", "we expect rain", "we do not expect rain"),
        ("trust", "he trusts the source", "he does not trust the source"),
        ("follow", "they follow the rules", "they do not follow the rules"),
        ("recognize", "she recognizes the face", "she does not recognize the face"),
        ("recommend", "i recommend this book", "i do not recommend this book"),
        ("require", "the task requires effort", "the task does not require effort"),
        ("guarantee", "this guarantees success", "this does not guarantee success"),
        ("confirm", "the test confirms the theory", "the test does not confirm the theory"),
    ]
    for name, pos, neg in verb_pairs:
        pairs.append({"name": f"neg_verb_{name}", "A": pos, "B": neg, "category": "negation", "subtype": "syntactic_do_not"})

    # Type 3: no/nothing/no one (12 pairs)
    no_pairs = [
        ("found_nothing", "he found something interesting", "he found nothing interesting"),
        ("came_no_one", "someone came to the party", "no one came to the party"),
        ("no_food", "there was some food left", "there was no food left"),
        ("no_idea", "she had some idea what to do", "she had no idea what to do"),
        ("no_reason", "there is a reason to worry", "there is no reason to worry"),
        ("no_choice", "they had a choice in the matter", "they had no choice in the matter"),
        ("no_doubt", "there is some doubt about it", "there is no doubt about it"),
        ("no_evidence", "there is evidence of fraud", "there is no evidence of fraud"),
        ("no_hope", "there is some hope left", "there is no hope left"),
        ("no_sign", "there is a sign of life", "there is no sign of life"),
        ("no_animal", "an animal crossed the road", "no animal crossed the road"),
        ("no_visitor", "a visitor arrived today", "no visitor arrived today"),
    ]
    for name, pos, neg in no_pairs:
        pairs.append({"name": f"neg_no_{name}", "A": pos, "B": neg, "category": "negation", "subtype": "existential_no"})

    # Type 4: never (10 pairs)
    never_pairs = [
        ("seen_before", "i have seen it before", "i have never seen it before"),
        ("been_paris", "she has been to Paris", "she has never been to Paris"),
        ("told_secret", "he told someone the secret", "he never told anyone the secret"),
        ("gives_up", "she sometimes gives up", "she never gives up"),
        ("forgets_face", "he sometimes forgets names", "he never forgets a face"),
        ("complains", "she often complains", "she never complains"),
        ("tells_truth", "he sometimes tells the truth", "he never tells the truth"),
        ("late", "she is sometimes late", "she is never late"),
        ("apologizes", "he sometimes apologizes", "he never apologizes"),
        ("admits", "she sometimes admits mistakes", "she never admits mistakes"),
    ]
    for name, pos, neg in never_pairs:
        pairs.append({"name": f"neg_never_{name}", "A": pos, "B": neg, "category": "negation", "subtype": "never"})

    # Type 5: negative prefix/suffix (12 pairs)
    prefix_pairs = [
        ("impossible", "the task is possible", "the task is impossible"),
        ("unacceptable", "the proposal is acceptable", "the proposal is unacceptable"),
        ("incomplete", "the report is complete", "the report is incomplete"),
        ("irrelevant", "the comment is relevant", "the comment is irrelevant"),
        ("dishonest", "the person is honest", "the person is dishonest"),
        ("unfair", "the treatment was fair", "the treatment was unfair"),
        ("unlikely", "the outcome is likely", "the outcome is unlikely"),
        ("incorrect", "the assumption is correct", "the assumption is incorrect"),
        ("irregular", "the pattern is regular", "the pattern is irregular"),
        ("disagree", "they agree on the terms", "they disagree on the terms"),
        ("uncertain", "the result is certain", "the result is uncertain"),
        ("disobey", "the soldiers obey orders", "the soldiers disobey orders"),
    ]
    for name, pos, neg in prefix_pairs:
        pairs.append({"name": f"neg_prefix_{name}", "A": pos, "B": neg, "category": "negation", "subtype": "morphological_neg"})

    # Type 6: scope/quantifier negation (16 pairs)
    scope_pairs = [
        ("not_all", "all birds can fly", "not all birds can fly"),
        ("not_everyone", "everyone agreed", "not everyone agreed"),
        ("not_always", "she always tells the truth", "she does not always tell the truth"),
        ("not_entirely", "the plan is entirely successful", "the plan is not entirely successful"),
        ("not_necessarily", "wealth means happiness", "wealth does not necessarily mean happiness"),
        ("not_only", "he is rich", "he is not only rich but also kind"),
        ("not_exactly", "that is exactly what i meant", "that is not exactly what i meant"),
        ("not_quite", "the work is finished", "the work is not quite finished"),
        ("not_particularly", "the movie was interesting", "the movie was not particularly interesting"),
        ("not_completely", "the glass is full", "the glass is not completely full"),
        ("not_because", "he left because he was angry", "he did not leave because he was angry"),
        ("not_if", "she will come if invited", "she will not come if invited"),
        ("not_a_single", "a single person helped", "not a single person helped"),
        ("not_even_one", "he ate one cookie", "he did not eat even one cookie"),
        ("not_any", "there are some problems", "there are not any problems"),
        ("not_once", "she called once", "she did not call once"),
    ]
    for name, pos, neg in scope_pairs:
        pairs.append({"name": f"neg_scope_{name}", "A": pos, "B": neg, "category": "negation", "subtype": "scope_quantifier"})

    return pairs


# Also include translation and logical (reduced, for cross-function comparison)
def build_other_pairs():
    pairs = []
    # Translation: 20 pairs
    for name, en, zh in [
        ("dog_cat", "the dog chases the cat", "狗追猫"),
        ("sun_east", "the sun rises in the east", "太阳从东方升起"),
        ("teacher", "the teacher teaches the student", "老师教学生"),
        ("bird_sky", "the bird flies in the sky", "鸟在天空中飞翔"),
        ("water_cold", "the water is very cold", "水非常冷"),
        ("child_happy", "the child is very happy", "孩子非常快乐"),
        ("city_busy", "the city is very busy", "这个城市非常繁忙"),
        ("food_delicious", "the food is delicious", "食物很美味"),
        ("music_beautiful", "the music is beautiful", "音乐很美"),
        ("sky_blue", "the sky is blue", "天空是蓝色的"),
        ("flower_red", "the flower is red", "花是红色的"),
        ("fish_fresh", "the fish is fresh", "鱼很新鲜"),
        ("wind_strong", "the wind is strong today", "今天的风很大"),
        ("rain_heavy", "the rain is heavy", "雨下得很大"),
        ("house_large", "the house is very large", "这栋房子非常大"),
        ("garden_small", "the garden is small", "花园很小"),
        ("horse_white", "the horse is white", "这匹马是白色的"),
        ("book_interest", "the book is very interesting", "这本书非常有趣"),
        ("mountain_high", "the mountain is extremely high", "这座山非常高"),
        ("love_overcome", "love overcomes everything", "爱战胜一切"),
    ]:
        pairs.append({"name": f"trans_{name}", "A": en, "B": zh, "category": "translation", "subtype": "sent"})
    
    # Logical: 20 pairs
    for name, a, b in [
        ("and_or_catdog", "the cat and the dog are sleeping", "the cat or the dog is sleeping"),
        ("and_or_birds", "birds and bees are pollinators", "birds or bees are pollinators"),
        ("and_or_tea", "tea and coffee are served", "tea or coffee is served"),
        ("and_or_apples", "apples and oranges are fruits", "apples or oranges are fruits"),
        ("if_rain", "if it rains we will stay home", "we will stay home if it rains"),
        ("if_hungry", "if you are hungry eat something", "eat something if you are hungry"),
        ("if_tired", "if she is tired she will rest", "she will rest if she is tired"),
        ("if_cold", "if it gets cold turn on the heater", "turn on the heater if it gets cold"),
        ("because_rain", "because it rained we stayed home", "we stayed home because it rained"),
        ("because_hungry", "because he was hungry he ate", "he ate because he was hungry"),
        ("because_sick", "because she was sick she rested", "she rested because she was sick"),
        ("because_late", "because he was late he ran", "he ran because he was late"),
        ("although_rain", "although it rained they went out", "they went out although it rained"),
        ("although_tired", "although she was tired she continued", "she continued although she was tired"),
        ("although_small", "although the dog was small it was brave", "the dog was brave although it was small"),
        ("therefore_rain", "it is raining therefore we will stay home", "we will stay home therefore it is raining"),
        ("therefore_late", "he overslept therefore he was late", "he was late therefore he overslept"),
        ("therefore_study", "she studied hard therefore she passed", "she passed therefore she studied hard"),
        ("and_or_sunmoon", "the sun and the moon are visible", "the sun or the moon is visible"),
        ("if_ready", "if they are ready we can go", "we can go if they are ready"),
    ]:
        pairs.append({"name": f"logic_{name}", "A": a, "B": b, "category": "logical", "subtype": "basic"})
    
    return pairs


# ==============================================================================
# MODEL LOADING (single eager model — proven reliable)
# ==============================================================================

def load_model_bf16(model_name: str):
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
# CAPTURE: Cache all-layer attn & mlp outputs
# ==============================================================================

def capture_all_layers(model, tokenizer, sentence, device, max_len=48, n_layers=36):
    """Forward sentence and capture ALL layers' attn + mlp outputs."""
    layers = get_layers(model)
    actual_layers = min(n_layers, len(layers))

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    seq_len = inputs["input_ids"].shape[1]
    if seq_len < max_len:
        pad_len = max_len - seq_len
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, pad_len), value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, pad_len), value=0)

    captured = {}
    hooks = []

    def make_attn_hook(li):
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

    for li in range(actual_layers):
        hooks.append(layers[li].self_attn.register_forward_hook(make_attn_hook(li)))
        hooks.append(layers[li].mlp.register_forward_hook(make_mlp_hook(li)))

    with torch.no_grad():
        try:
            model(**inputs)
        except Exception as e:
            log_time(f"  capture_all_layers FAILED: {e}")
            captured = {}

    for h in hooks:
        h.remove()

    return captured


# ==============================================================================
# PATCHING WITH α INTERPOLATION + CROSS
# ==============================================================================

def forward_with_interp_patches(model, tokenizer, sentence, device, max_len,
                                 attn_a, attn_b, mlp_a, mlp_b,
                                 alpha, patch_type, target_layer, n_layers):
    """
    Forward with interpolated patches.
    
    patch_type:
      "attn":   (1-α)*attn_a + α*attn_b → only attention block at target_layer
      "mlp":    (1-α)*mlp_a + α*mlp_b → only MLP block at target_layer
      "both":   both attn + mlp interpolated at target_layer
      "cross_am": A's attn + B's mlp at target_layer (contract test)
      "cross_ma": B's attn + A's mlp at target_layer (contract test)
    
    Returns: logits, attn_post_norm, mlp_post_norm (for naturalness check)
    """
    layers = get_layers(model)
    actual_layers = min(n_layers, len(layers))
    if target_layer >= actual_layers:
        return None, None, None

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=max_len).to(device)
    seq_len = inputs["input_ids"].shape[1]
    if seq_len < max_len:
        pad_len = max_len - seq_len
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, pad_len), value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, pad_len), value=0)

    # Get device/dtype from target layer
    o_proj = layers[target_layer].self_attn.o_proj
    target_device = o_proj.weight.device
    target_dtype = o_proj.weight.dtype
    
    mlp_module = layers[target_layer].mlp
    mlp_device = next(mlp_module.parameters()).device
    mlp_dtype = next(mlp_module.parameters()).dtype

    # Compute interpolated vectors
    def interp(a, b, alpha, dev, dtype):
        a_t = a.to(dev).to(dtype)
        b_t = b.to(dev).to(dtype)
        min_s = min(min(a_t.shape[1], b_t.shape[1]), max_len)
        return (1 - alpha) * a_t[:, :min_s, :] + alpha * b_t[:, :min_s, :]

    attn_interp = None
    mlp_interp = None
    attn_a_interp = None
    mlp_a_interp = None
    
    if patch_type in ("attn", "both", "cross_am"):
        attn_interp = interp(attn_a, attn_b, alpha, target_device, target_dtype) if patch_type != "cross_am" else interp(attn_a, attn_b, 0.0, target_device, target_dtype)
    if patch_type == "cross_ma":
        attn_interp = interp(attn_a, attn_b, 1.0, target_device, target_dtype)  # B's attn
    
    if patch_type in ("mlp", "both", "cross_ma"):
        mlp_interp = interp(mlp_a, mlp_b, alpha, mlp_device, mlp_dtype) if patch_type != "cross_ma" else interp(mlp_a, mlp_b, 0.0, mlp_device, mlp_dtype)
    if patch_type == "cross_am":
        mlp_interp = interp(mlp_a, mlp_b, 1.0, mlp_device, mlp_dtype)  # B's mlp

    # Register hooks for current patching
    hooks = []
    down_norms = {}

    if attn_interp is not None:
        def make_attn_hook(pv):
            def hook(module, input_t, output_t):
                if isinstance(output_t, tuple):
                    new_out = (output_t[0].clone(),) + output_t[1:]
                    ms = min(pv.shape[1], new_out[0].shape[1])
                    new_out[0][:, :ms, :] = pv[:, :ms, :]
                    return new_out
                else:
                    new_out = output_t.clone()
                    ms = min(pv.shape[1], new_out.shape[1])
                    new_out[:, :ms, :] = pv[:, :ms, :]
                    return new_out
            return hook
        hooks.append(layers[target_layer].self_attn.register_forward_hook(make_attn_hook(attn_interp)))

    if mlp_interp is not None:
        def make_mlp_hook(pv):
            def hook(module, input_t, output_t):
                if isinstance(output_t, tuple):
                    new_out = (output_t[0].clone(),) + output_t[1:]
                    ms = min(pv.shape[1], new_out[0].shape[1])
                    new_out[0][:, :ms, :] = pv[:, :ms, :]
                    return new_out
                else:
                    new_out = output_t.clone()
                    ms = min(pv.shape[1], new_out.shape[1])
                    new_out[:, :ms, :] = pv[:, :ms, :]
                    return new_out
            return hook
        hooks.append(layers[target_layer].mlp.register_forward_hook(make_mlp_hook(mlp_interp)))

    # Also capture downstream norms for naturalness check
    if target_layer + 1 < actual_layers:
        def make_down_hook(li):
            def hook(module, input_t, output_t):
                if isinstance(input_t, tuple) and len(input_t) > 0:
                    down_norms["resid_in"] = float(input_t[0].float().norm())
                if isinstance(output_t, tuple):
                    down_norms["layer_out"] = float(output_t[0].float().norm())
                else:
                    down_norms["layer_out"] = float(output_t.float().norm())
            return hook
        hooks.append(layers[target_layer + 1].register_forward_hook(make_down_hook(target_layer + 1)))

    with torch.no_grad():
        try:
            out = model(**inputs)
            result = out.logits[0, -1, :].detach().cpu().float().clone()
        except Exception as e:
            result = None
            down_norms = {}

    for h in hooks:
        h.remove()

    return result, down_norms, (attn_interp, mlp_interp)


# ==============================================================================
# ANALYSIS METRICS
# ==============================================================================

def compute_metrics_289(patched_logits, logits_a, logits_b, kl_ab):
    """Extended metrics: kl_ratio, progress, norm_ratio."""
    if patched_logits is None or logits_a is None or logits_b is None:
        return None

    kl_p = float(F.kl_div(
        F.log_softmax(patched_logits, dim=-1),
        F.softmax(logits_b, dim=-1),
        reduction='sum'
    ))
    kl_ratio = min(kl_p / max(kl_ab, 1e-6), 50.0)

    delta_B = logits_b - logits_a
    delta_patch = patched_logits - logits_a
    norm_B = float(torch.norm(delta_B))
    norm_p = float(torch.norm(delta_patch))
    if norm_B > 1e-8 and norm_p > 1e-8:
        cos_dir = float(torch.dot(delta_patch, delta_B) / (norm_B * norm_p))
        mag_ratio = norm_p / norm_B
        progress = cos_dir * min(mag_ratio, 2.0)
        norm_ratio = norm_p / norm_B
    else:
        cos_dir = 0.0
        progress = 0.0
        norm_ratio = 0.0

    return {
        "kl_ratio": kl_ratio,
        "progress": progress,
        "cos_dir": cos_dir,
        "norm_ratio": norm_ratio,
    }


# ==============================================================================
# MAIN PHASE 289
# ==============================================================================

def run_phase289(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase289_{model_name}.txt")

    log_time(f"{'='*60}")
    log_time(f"Phase 289: Layer-Scan Contract Decomposition — {model_name}")
    log_time(f"{'='*60}")

    # Load
    model, tokenizer, device = load_model_bf16(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    log_time(f"Model: {model_info.model_class}, L={n_layers}, d_model={model_info.d_model}")

    # Warmup
    wu = tokenizer("warmup test", return_tensors="pt").to(device)
    with torch.no_grad():
        try: model(**wu)
        except: pass

    # Dataset: negation 80 + translation 20 + logical 20 = 120 pairs
    all_pairs = build_negation_pairs() + build_other_pairs()
    log_time(f"Dataset: {len(all_pairs)} pairs")
    log_time(f"  negation: 80 (6 subtypes)")
    log_time(f"  translation: 20")
    log_time(f"  logical: 20")

    max_len = 48
    ALPHAS = [0.0, 0.33, 0.67, 1.0]
    PATCH_TYPES = ["attn", "mlp", "both", "cross_am", "cross_ma"]

    # ============================================================
    # PHASE 1: Capture all layers for all pairs
    # ============================================================
    log_time(f"\n{'='*60}")
    log_time(f"PHASE 1: Capture all-layer outputs ({len(all_pairs)} pairs × {n_layers} layers)")
    log_time(f"{'='*60}")

    pair_outputs = {}
    pair_baselines = {}
    pair_base_norms = {}  # {pname: {"A": {L{li}: {"attn":..., "mlp":...} norm info}}
    layers = get_layers(model)  # get layers once for reuse

    t0 = time.time()
    for pidx, pair in enumerate(all_pairs):
        pname = pair["name"]
        sent_a, sent_b = pair["A"], pair["B"]
        cat = pair["category"]
        subtype = pair.get("subtype", "")

        toks_a = len(tokenizer.encode(sent_a, add_special_tokens=True))
        toks_b = len(tokenizer.encode(sent_b, add_special_tokens=True))
        cl = min(max(toks_a, toks_b), max_len)

        out_a = capture_all_layers(model, tokenizer, sent_a, device, cl, n_layers)
        out_b = capture_all_layers(model, tokenizer, sent_b, device, cl, n_layers)

        if out_a and out_b:
            pair_outputs[pname] = {"A": out_a, "B": out_b, "category": cat, "subtype": subtype, "seq_len": cl}

        # Baseline logits
        ia = tokenizer(sent_a, return_tensors="pt", truncation=True, max_length=cl).to(device)
        ib = tokenizer(sent_b, return_tensors="pt", truncation=True, max_length=cl).to(device)
        with torch.no_grad():
            logits_a = model(**ia).logits[0, -1, :].detach().cpu().float()
            logits_b = model(**ib).logits[0, -1, :].detach().cpu().float()
        kl_ab = float(F.kl_div(F.log_softmax(logits_a, dim=-1), F.softmax(logits_b, dim=-1), reduction='sum'))
        pair_baselines[pname] = {"logits_a": logits_a, "logits_b": logits_b, "kl_ab": kl_ab,
                                  "sent_a": sent_a, "sent_b": sent_b, "category": cat,
                                  "subtype": subtype, "seq_len": cl}

        if (pidx + 1) % 30 == 0:
            elapsed = time.time() - t0
            rate = (pidx + 1) / max(elapsed, 1) * 3600
            log_time(f"  [{pidx+1}/{len(all_pairs)}] Captured: {elapsed:.0f}s, "
                     f"~{rate:.0f} pairs/h, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

    t_capture = time.time() - t0
    log_time(f"  Capture done: {len(pair_outputs)} pairs, {t_capture:.0f}s ({t_capture/60:.1f}min)")

    # ============================================================
    # PHASE 2: Layer-scan patching with α interpolation
    # ============================================================
    log_time(f"\n{'='*60}")
    log_time(f"PHASE 2: Layer-Scan Patching ({n_layers} layers × {len(ALPHAS)}α × {len(PATCH_TYPES)} types)")
    log_time(f"{'='*60}")

    # layers already defined above

    all_results = []
    n_total = len(pair_outputs) * n_layers * len(ALPHAS) * len(PATCH_TYPES)
    n_done = 0
    t0_patch = time.time()

    for pidx, pname in enumerate(pair_outputs.keys()):
        pb = pair_baselines[pname]
        po = pair_outputs[pname]
        sent_a = pb["sent_a"]
        logits_a = pb["logits_a"]
        logits_b = pb["logits_b"]
        kl_ab = pb["kl_ab"]
        cl = pb["seq_len"]
        cat = pb["category"]
        subtype = pb.get("subtype", "")

        if kl_ab < 1e-6:
            n_done += n_layers * len(ALPHAS) * len(PATCH_TYPES)
            continue

        # For non-negation functions, only test step=4 layers
        if cat == "negation":
            test_layers = list(range(n_layers))
        else:
            test_layers = list(range(0, n_layers, 4)) + [n_layers - 1]
            test_layers = sorted(set(test_layers))

        for target_layer in test_layers:
            lk = f"L{target_layer}"
            if lk not in po.get("A", {}) or lk not in po.get("B", {}):
                n_done += len(ALPHAS) * len(PATCH_TYPES)
                continue

            a_data = po["A"][lk]
            b_data = po["B"][lk]
            attn_a = a_data.get("attn")
            attn_b = b_data.get("attn")
            mlp_a = a_data.get("mlp")
            mlp_b = b_data.get("mlp")

            if attn_a is None or attn_b is None or mlp_a is None or mlp_b is None:
                n_done += len(ALPHAS) * len(PATCH_TYPES)
                continue

            for alpha in ALPHAS:
                for patch_type in PATCH_TYPES:
                    patched_logits, down_norms, _ = forward_with_interp_patches(
                        model, tokenizer, sent_a, device, cl,
                        attn_a, attn_b, mlp_a, mlp_b,
                        alpha, patch_type, target_layer, n_layers
                    )

                    if patched_logits is not None:
                        metrics = compute_metrics_289(patched_logits, logits_a, logits_b, kl_ab)
                        if metrics:
                            all_results.append({
                                "pname": pname, "category": cat, "subtype": subtype,
                                "layer": target_layer, "alpha": alpha,
                                "patch_type": patch_type,
                                "kl_ab": kl_ab, **metrics,
                            })

            n_done += len(ALPHAS) * len(PATCH_TYPES)

        # Progress logging every 10 pairs
        if (pidx + 1) % 10 == 0:
            elapsed = time.time() - t0_patch
            total_m = n_total / 60  # total forwards in millions? no, just count
            progress_pct = 100 * n_done / max(n_total, 1)
            eta = elapsed / max(n_done, 1) * (n_total - n_done)
            log_time(f"  [{pidx+1}/{len(pair_outputs)}] Patching: {progress_pct:.1f}%, "
                     f"{elapsed:.0f}s elapsed, ETA={eta/60:.0f}min, "
                     f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB, {len(all_results)} results")

    t_patch = time.time() - t0_patch
    log_time(f"\n  Patching done: {len(all_results)} results, {t_patch:.0f}s ({t_patch/60:.1f}min)")

    # ============================================================
    # ANALYSIS
    # ============================================================
    log_time(f"\n{'='*60}")
    log_time("ANALYSIS")
    log_time(f"{'='*60}")

    # Per-layer, per-patch-type, per-alpha aggregation
    log_time(f"\n  === LAYER-SCAN CURVES (α=1.0, full replacement) ===")
    log_time(f"  {'Layer':>6} {'Attn_KR':>9} {'Attn_Prog':>9} {'MLP_KR':>9} {'MLP_Prog':>9} "
             f"{'Both_KR':>9} {'Both_Prog':>9} {'CrossAM_KR':>9} {'CrossMA_KR':>9}")

    layer_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in all_results:
        if r["category"] == "negation" and r["alpha"] == 1.0:
            layer_agg[r["layer"]][r["patch_type"]]["kl_ratio"].append(r["kl_ratio"])
            layer_agg[r["layer"]][r["patch_type"]]["progress"].append(r["progress"])
            layer_agg[r["layer"]][r["patch_type"]]["norm_ratio"].append(r["norm_ratio"])

    layer_curve = {}
    for li in sorted(layer_agg.keys()):
        d = layer_agg[li]
        vals = {}
        for pt in PATCH_TYPES:
            kr_vals = d[pt].get("kl_ratio", [])
            prog_vals = d[pt].get("progress", [])
            nr_vals = d[pt].get("norm_ratio", [])
            vals[f"{pt}_kr"] = float(np.mean(kr_vals)) if kr_vals else 0
            vals[f"{pt}_prog"] = float(np.mean(prog_vals)) if prog_vals else 0
            vals[f"{pt}_nr"] = float(np.mean(nr_vals)) if nr_vals else 0
        layer_curve[str(li)] = vals

        log_time(f"  {li:>6} {vals['attn_kr']:9.3f} {vals['attn_prog']:9.4f} "
                 f"{vals['mlp_kr']:9.3f} {vals['mlp_prog']:9.4f} "
                 f"{vals['both_kr']:9.3f} {vals['both_prog']:9.4f} "
                 f"{vals['cross_am_kr']:9.3f} {vals['cross_ma_kr']:9.3f}")

    # α interpolation analysis: at the LAYER with strongest effect, show α curve
    # Find strongest layer by combined progress
    best_layer = max(layer_curve.keys(), key=lambda l: layer_curve[l].get("both_prog", 0) + layer_curve[l].get("both_kr", 0))
    best_li = int(best_layer)
    log_time(f"\n  === α INTERPOLATION at strongest layer L{best_li} ===")
    log_time(f"  {'α':>6} {'Attn_KR':>9} {'Attn_Prog':>9} {'MLP_KR':>9} {'MLP_Prog':>9} "
             f"{'Both_KR':>9} {'Both_Prog':>9} {'CrossAM_KR':>9} {'Attn_NR':>9} {'Attn_DA':>9}")

    alpha_curve = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in all_results:
        if r["category"] == "negation" and r["layer"] == best_li:
            alpha_curve[r["alpha"]][r["patch_type"]]["kl_ratio"].append(r["kl_ratio"])
            alpha_curve[r["alpha"]][r["patch_type"]]["progress"].append(r["progress"])
            alpha_curve[r["alpha"]][r["patch_type"]]["norm_ratio"].append(r["norm_ratio"])

    alpha_data = {}
    for alpha in sorted(alpha_curve.keys()):
        d = alpha_curve[alpha]
        vals = {}
        for pt in ["attn", "mlp", "both", "cross_am"]:
            kr_vals = d[pt].get("kl_ratio", [])
            prog_vals = d[pt].get("progress", [])
            nr_vals = d[pt].get("norm_ratio", [])
            vals[f"{pt}_kr"] = float(np.mean(kr_vals)) if kr_vals else 0
            vals[f"{pt}_prog"] = float(np.mean(prog_vals)) if prog_vals else 0
            vals[f"{pt}_nr"] = float(np.mean(nr_vals)) if nr_vals else 0
        alpha_data[str(alpha)] = vals

        log_time(f"  {alpha:>6.2f} {vals['attn_kr']:9.3f} {vals['attn_prog']:9.4f} "
                 f"{vals['mlp_kr']:9.3f} {vals['mlp_prog']:9.4f} "
                 f"{vals['both_kr']:9.3f} {vals['both_prog']:9.4f} "
                 f"{vals['cross_am_kr']:9.3f} {vals['attn_nr']:9.3f}")

    # Contract breakdown: at which layers does cross_am explode?
    log_time(f"\n  === CONTRACT COMPATIBILITY (α=1.0 cross_am vs both) ===")
    n_broken = 0
    for li in sorted(layer_curve.keys()):
        cross_kr = layer_curve[li].get("cross_am_kr", 0)
        both_kr = layer_curve[li].get("both_kr", 1)
        ratio = cross_kr / max(both_kr, 1e-6)
        if ratio > 2.0:
            n_broken += 1
            log_time(f"    L{li}: cross/both={ratio:.1f}x CONTRACT BROKEN (cross={cross_kr:.2f}, both={both_kr:.2f})")
    if n_broken == 0:
        log_time(f"    No broken contract layers (all cross/both < 2x)")

    # Subtype analysis for negation
    log_time(f"\n  === NEGATION SUBTYPE BREAKDOWN (α=1.0, top-5 effect layers) ===")
    neg_results = [r for r in all_results if r["category"] == "negation" and r["alpha"] == 1.0]
    top5_layers = sorted(layer_curve.keys(), key=lambda l: layer_curve[l].get("both_prog", 0), reverse=True)[:5]
    
    subtype_agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in neg_results:
        if str(r["layer"]) in top5_layers:
            subtype_agg[r["subtype"]][r["layer"]][r["patch_type"]].append(r)
    
    for st in sorted(subtype_agg.keys()):
        line_parts = [f"  {st:>25}"]
        for tl in top5_layers:
            tl = int(tl)
            d = subtype_agg[st].get(tl, {})
            a_prog = np.mean([x["progress"] for x in d.get("attn", [])]) if d.get("attn") else 0
            m_prog = np.mean([x["progress"] for x in d.get("mlp", [])]) if d.get("mlp") else 0
            b_prog = np.mean([x["progress"] for x in d.get("both", [])]) if d.get("both") else 0
            best_p = max(a_prog, m_prog, b_prog)
            best_pt = "A" if a_prog >= best_p else ("M" if m_prog >= best_p else "B")
            line_parts.append(f" L{tl}:{best_pt}={best_p:.2f}")
        log_time(" ".join(line_parts))

    # ============================================================
    # SAVE
    # ============================================================
    save_data = {
        "model": model_name,
        "model_info": {"class": model_info.model_class, "n_layers": n_layers,
                       "d_model": model_info.d_model},
        "n_negation_pairs": 80,
        "n_total_pairs": len(all_pairs),
        "n_results": len(all_results),
        "alphas": ALPHAS,
        "patch_types": PATCH_TYPES,
        "layer_curve": layer_curve,
        "alpha_curve": alpha_data,
        "best_layer": best_li,
        "contract_broken_layers": n_broken,
    }

    save_path = RESULT_DIR / f"{model_name}_layer_scan.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    log_time(f"\nResults saved to {save_path}")

    detail_path = RESULT_DIR / f"{model_name}_detail.json"
    with open(detail_path, "w") as f:
        json.dump(all_results, f, indent=1, default=str)
    log_time(f"Detail saved to {detail_path}")

    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    return save_data


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            try:
                r = run_phase289(name)
                log_time(f"\n{name} DONE: {r['n_results']} results")
            except Exception as e:
                log_time(f"!!! {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_phase289(model_name)
