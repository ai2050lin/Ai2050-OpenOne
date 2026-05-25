"""
Phase 285: Real Forward Activation Patching (Causal Component Decomposition)
===========================================================================

GOAL: Replace manual attention patching with real forward activation patching.
This eliminates the manual↔real gap — the #1 hard bottleneck since Phase 282.

METHOD:
1. CACHE: For each sentence pair (A, B), run real forward passes, cache per-layer:
   - attn_out: self_attn module output (after W_o projection)
   - mlp_out:  mlp module output  
   - resid:    full layer output (= residual stream after attn+mlp)
2. PATCH: For each layer L and component C, run sentence B forward, 
   hook-replace component C at layer L with A's cached activation.
3. MEASURE: Logit distribution shift toward A 
   effect = KL(P_patched || P_B) / KL(P_A || P_B)
   0 = no effect, 1 = full conversion, >1 = overshoot

ADVANTAGES OVER MANUAL ATTENTION (Phase 282-284):
- No manual Q/K/V computation or hand-coded RoPE
- Uses model's OWN attention mechanism (SDPA/flash/eager)
- Captures QK norm, attention mask, GQA grouping, etc. natively
- Direct causal measurement: "if component C at layer L were from sentence A..."

COMPONENTS PATCHED:
- attn_out: self_attn output — routing + content combined
- mlp_out:  mlp output — content transformation
- resid:    layer output — accumulated residual stream delta

PATCHING SCALE (per model):
  15 pairs × ~15 sampled layers × 3 components = 675 patching forwards
  + 30 cache forwards = ~705 forwards
  At ~2s/forward → ~23 min per model

Usage:
  python tests/glm5/phase285_real_forward_patching.py qwen3
  python tests/glm5/phase285_real_forward_patching.py glm4
  python tests/glm5/phase285_real_forward_patching.py deepseek7b
"""

import sys, os, json, gc, time, warnings, glob as fileglob
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

RESULT_DIR = Path("results/phase285_real_patching")
RESULT_DIR.mkdir(parents=True, exist_ok=True)
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(parents=True, exist_ok=True)

_log_file = None
def log_time(msg):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    if _log_file:
        with open(_log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# =============================================================================
# Model Loading (bf16 + flash_attention_2 + device_map="auto")
# =============================================================================

def load_model_flash(model_name: str):
    """Load model: bf16 + flash_attn_2 + device_map='auto' for GLM4/DS7B."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bf16 + flash_attn_2 + device_map='auto')...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Try flash_attn first
    attn_impl = "flash_attention_2"
    try:
        import flash_attn
        log_time("  flash_attn available")
    except ImportError:
        attn_impl = "eager"
        log_time("  flash_attn not found, using eager (slower)")

    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=attn_impl,
    )
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"  {model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB, attn={attn_impl}")
    return model, tokenizer, device, attn_impl

# =============================================================================
# Sentence Pair Builder (reuse Phase 284's 121 pairs)
# =============================================================================

def build_phase285_pairs():
    """Same 121 pairs from Phase 284, 14 categories."""
    pairs = []

    # Category 1: animal SVO
    for name, a, b in [
        ("svo_dog_cat", "the dog chases the cat", "the cat chases the dog"),
        ("svo_wolf_sheep", "the wolf hunts the sheep", "the sheep hunts the wolf"),
        ("svo_lion_deer", "the lion stalks the deer", "the deer stalks the lion"),
        ("svo_bird_snake", "the bird pecks the snake", "the snake pecks the bird"),
        ("svo_cat_mouse", "the cat catches the mouse", "the mouse catches the cat"),
        ("svo_fox_rabbit", "the fox chases the rabbit", "the rabbit chases the fox"),
        ("svo_eagle_fish", "the eagle catches the fish", "the fish catches the eagle"),
        ("svo_bear_salmon", "the bear catches the salmon", "the salmon catches the bear"),
        ("svo_shark_seal", "the shark hunts the seal", "the seal hunts the shark"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "animal"})

    # Category 2: human SVO
    for name, a, b in [
        ("svo_man_woman", "the man greets the woman", "the woman greets the man"),
        ("svo_boy_girl", "the boy calls the girl", "the girl calls the boy"),
        ("svo_teacher_student", "the teacher helps the student", "the student helps the teacher"),
        ("svo_mother_child", "the mother feeds the child", "the child feeds the mother"),
        ("svo_father_son", "the father teaches the son", "the son teaches the father"),
        ("svo_doctor_patient", "the doctor examines the patient", "the patient examines the doctor"),
        ("svo_king_queen", "the king commands the queen", "the queen commands the king"),
        ("svo_brother_sister", "the brother protects the sister", "the sister protects the brother"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "human"})

    # Category 3: human_object
    for name, a, b in [
        ("svo_child_apple", "the child eats the apple", "the apple is eaten by the child"),
        ("svo_chef_knife", "the chef uses the knife", "the knife is used by the chef"),
        ("svo_painter_brush", "the painter holds the brush", "the brush is held by the painter"),
        ("svo_driver_car", "the driver starts the car", "the car is started by the driver"),
        ("svo_writer_pen", "the writer lifts the pen", "the pen is lifted by the writer"),
        ("svo_guard_key", "the guard holds the key", "the key is held by the guard"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "human_object"})

    # Category 4: place
    for name, a, b in [
        ("svo_king_city", "the king rules the city", "the city is ruled by the king"),
        ("svo_explorer_island", "the explorer discovers the island", "the island is discovered by the explorer"),
        ("svo_tourist_museum", "the tourist visits the museum", "the museum is visited by the tourist"),
        ("svo_guard_prison", "the guard watches the prison", "the prison is watched by the guard"),
        ("svo_soldier_bridge", "the soldier defends the bridge", "the bridge is defended by the soldier"),
        ("svo_mayor_town", "the mayor governs the town", "the town is governed by the mayor"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "place"})

    # Category 5: passive
    for name, a, b in [
        ("pass_dog_cat", "the dog chases the cat", "the cat is chased by the dog"),
        ("pass_teacher_student", "the teacher teaches the student", "the student is taught by the teacher"),
        ("pass_author_book", "the author wrote the book", "the book was written by the author"),
        ("pass_wife_cake", "the wife baked the cake", "the cake was baked by the wife"),
        ("pass_workers_bridge", "the workers built the bridge", "the bridge was built by the workers"),
        ("pass_detective_clue", "the detective found the clue", "the clue was found by the detective"),
        ("pass_everyone_teacher", "everyone loves the teacher", "the teacher is loved by everyone"),
        ("pass_cat_fish", "the cat ate the fish", "the fish was eaten by the cat"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "passive"})

    # Category 6: negation
    for name, a, b in [
        ("neg_happy", "she is happy", "she is not happy"),
        ("neg_agree", "they agree with the proposal", "they do not agree with the proposal"),
        ("neg_found", "he found something interesting", "he found nothing interesting"),
        ("neg_remember", "i remember the meeting", "i do not remember the meeting"),
        ("neg_possible", "victory is possible", "victory is not possible"),
        ("neg_understand", "we understand the problem", "we do not understand the problem"),
        ("neg_anyone_came", "someone came to the party", "no one came to the party"),
        ("neg_ever_seen", "i have seen it before", "i have never seen it before"),
        ("neg_notonly_smart", "she is smart", "she is not only smart but also kind"),
        ("neg_hardly_works", "he works hard", "he hardly works"),
        ("neg_scarcely_enough", "they had enough food", "they had scarcely enough food"),
        ("neg_unacceptable", "the proposal is acceptable", "the proposal is not unacceptable"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "negation"})

    # Category 7: quantifier
    for name, a, b in [
        ("quant_few_many", "few people attended", "many people attended"),
        ("quant_some_all", "some birds can fly", "all birds can fly"),
        ("quant_slow_fast", "the car is slow", "the car is fast"),
        ("quant_small_large", "the house is small", "the house is large"),
        ("quant_quiet_loud", "the music is quiet", "the music is loud"),
        ("quant_cold_hot", "the water is cold", "the water is hot"),
        ("quant_most_few", "most students passed", "few students passed"),
        ("quant_always_never", "she always tells the truth", "she never tells the truth"),
        ("quant_more_fewer", "more people came than expected", "fewer people came than expected"),
        ("quant_only_even", "only John came", "even John came"),
        ("quant_almost_all", "almost everyone agreed", "almost no one agreed"),
        ("quant_each_both", "each student brought a book", "both students brought a book"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "quantifier"})

    # Category 8: conditional
    for name, a, b in [
        ("cond_if_rain", "if it rains we will stay home", "we will stay home if it rains"),
        ("cond_if_hungry", "if you are hungry eat something", "eat something if you are hungry"),
        ("cond_if_tired", "if she is tired she will sleep", "she will sleep if she is tired"),
        ("cond_if_cold", "if it gets cold turn on the heater", "turn on the heater if it gets cold"),
        ("cond_if_ready", "if they are ready we can go", "we can go if they are ready"),
        ("cond_if_sunny", "if tomorrow is sunny visit the park", "visit the park if tomorrow is sunny"),
        ("cond_unless", "unless you study you will fail", "you will fail unless you study"),
        ("cond_because", "because it rained we stayed home", "we stayed home because it rained"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "conditional"})

    # Category 9: recursive
    for name, a, b in [
        ("rec_believe_theory", "the scientist believes that the theory is correct",
         "the theory is believed by the scientist to be correct"),
        ("rec_dog_barked", "the dog that barked ran away", "the barking dog ran away"),
        ("rec_woman_wrote", "the woman who wrote the letter smiled", "the letter-writing woman smiled"),
        ("rec_king_said", "the king said that the queen is wise", "the queen is wise said the king"),
        ("rec_teacher_think", "the teacher thinks that the student learned well",
         "the student is thought by the teacher to have learned well"),
        ("rec_door_painted", "the door that was painted red opened", "the red door opened"),
        ("rec_book_which", "the book which i read yesterday was great", "yesterday i read a book which was great"),
        ("rec_person_who", "the person who called left a message", "a message was left by the person who called"),
        ("rec_that_fact", "the fact that he lied surprised everyone",
         "everyone was surprised by the fact that he lied"),
        ("rec_nested_if", "the man who said that if it rains he will leave arrived",
         "the man arrived who said that he will leave if it rains"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "recursive"})

    # Category 10: translation
    for name, a, b in [
        ("trans_dog_cat", "the dog chases the cat", "\u72d7\u8ffd\u732b"),  # 狗追猫
        ("trans_sun_east", "the sun rises in the east", "\u592a\u9633\u4ece\u4e1c\u65b9\u5347\u8d77"),
        ("trans_teacher_teach", "the teacher teaches the student", "\u8001\u5e08\u6559\u5b66\u751f"),
        ("trans_bird_sky", "the bird flies in the sky", "\u9e1f\u5728\u5929\u7a7a\u4e2d\u98de\u7fd4"),
        ("trans_king_rules", "the king rules the kingdom", "\u56fd\u738b\u7edf\u6cbb\u738b\u56fd"),
        ("trans_water_cold", "the water is very cold", "\u6c34\u975e\u5e38\u51b7"),
        ("trans_love_overcome", "love overcomes everything", "\u7231\u6218\u80dc\u4e00\u5207"),
        ("trans_fox_hunt", "the fox hunts the rabbit", "\u72d0\u72f8\u730e\u5154"),
        ("trans_child_happy", "the child is very happy", "\u5b69\u5b50\u975e\u5e38\u5feb\u4e50"),
        ("trans_mountain_high", "the mountain is extremely high", "\u8fd9\u5ea7\u5c71\u975e\u5e38\u9ad8"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "translation"})

    # Category 11: comparative
    for name, a, b in [
        ("comp_bigger_than", "the elephant is bigger than the mouse", "the mouse is smaller than the elephant"),
        ("comp_taller_than", "John is taller than Mary", "Mary is shorter than John"),
        ("comp_more_expensive", "gold is more expensive than silver", "silver is less expensive than gold"),
        ("comp_faster_than", "the train is faster than the bicycle", "the bicycle is slower than the train"),
        ("comp_stronger_than", "steel is stronger than wood", "wood is weaker than steel"),
        ("comp_smarter_than", "Alice is smarter than Bob", "Bob is less smart than Alice"),
        ("comp_older_than", "the grandfather is older than the father", "the father is younger than the grandfather"),
        ("comp_more_than", "she has more books than he does", "he has fewer books than she does"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "comparative"})

    # Category 12: temporal
    for name, a, b in [
        ("temp_before_after", "before eating i washed my hands", "after washing my hands i ate"),
        ("temp_was_is", "the cat was hungry", "the cat is hungry"),
        ("temp_will_did", "the team will win the game", "the team won the game"),
        ("temp_is_going", "she is reading a book", "she was reading a book"),
        ("temp_since_until", "the shop has been open since morning", "the shop was open until evening"),
        ("temp_already_yet", "he has already finished his work", "he has not yet finished his work"),
        ("temp_while_when", "while i was cooking the phone rang", "when the phone rang i was cooking"),
        ("temp_still_no", "she is still working on the project", "she is no longer working on the project"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "temporal"})

    # Category 13: logical
    for name, a, b in [
        ("logic_and_or", "the cat and the dog are sleeping", "the cat or the dog is sleeping"),
        ("logic_both_either", "both Alice and Bob agreed", "either Alice or Bob agreed"),
        ("logic_not_only", "he is not only rich but also generous", "he is neither rich nor generous"),
        ("logic_although", "although it rained they went out", "they went out although it rained"),
        ("logic_therefore", "it is raining therefore we will stay home", "we will stay home therefore it is raining"),
        ("logic_however", "the test was hard however she passed", "she passed however the test was hard"),
        ("logic_moreover", "the food is delicious moreover it is healthy", "the food is healthy moreover it is delicious"),
        ("logic_nevertheless", "he was tired nevertheless he continued", "he continued nevertheless he was tired"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "logical"})

    # Category 14: abstract
    for name, a, b in [
        ("abs_justice_corruption", "justice prevails over corruption", "corruption prevails over justice"),
        ("abs_wisdom_folly", "wisdom overcomes folly", "folly overcomes wisdom"),
        ("abs_courage_fear", "courage defeats fear", "fear defeats courage"),
        ("abs_love_hate", "love conquers hate", "hate conquers love"),
        ("abs_truth_lie", "truth defeats lies", "lies defeat truth"),
        ("abs_hope_despair", "hope overcomes despair", "despair overcomes hope"),
        ("abs_freedom_tyranny", "freedom resists tyranny", "tyranny resists freedom"),
        ("abs_knowledge_ignorance", "knowledge dispels ignorance", "ignorance dispels knowledge"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "abstract"})

    return pairs


def sample_pairs_for_patching(all_pairs, max_per_category=2, n_categories=14):
    """Sample up to max_per_category pairs from each category for efficient patching."""
    by_cat = defaultdict(list)
    for p in all_pairs:
        by_cat[p["category"]].append(p)
    
    sampled = []
    for cat in sorted(by_cat.keys()):
        cat_pairs = by_cat[cat]
        # Take evenly spaced pairs
        if len(cat_pairs) <= max_per_category:
            sampled.extend(cat_pairs)
        else:
            step = len(cat_pairs) / max_per_category
            for i in range(max_per_category):
                idx = min(int(i * step), len(cat_pairs) - 1)
                sampled.append(cat_pairs[idx])
    
    # Deduplicate
    seen = set()
    unique = []
    for p in sampled:
        if p["name"] not in seen:
            seen.add(p["name"])
            unique.append(p)
    
    return unique


# =============================================================================
# Core Patching Logic
# =============================================================================

def cache_activations(model, tokenizer, sentence, device, n_layers, common_len):
    """
    Forward pass sentence, cache per-layer activations.
    
    Returns:
        dict: {
            "L{li}_attn": tensor[1, common_len, d_model],  # self_attn output
            "L{li}_mlp":  tensor[1, common_len, d_model],  # mlp output
            "L{li}_resid": tensor[1, common_len, d_model], # layer output (residual)
        }
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=common_len).to(device)
    # Pad if needed
    seq_len = inputs["input_ids"].shape[1]
    if seq_len < common_len:
        pad_len = common_len - seq_len
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, pad_len), value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, pad_len), value=0)
        actual_len = seq_len
    else:
        actual_len = common_len
    
    captured = {}
    layers = get_layers(model)
    
    # Hook self_attn output
    def make_attn_hook(li):
        def hook(module, input_t, output_t):
            if isinstance(output_t, tuple):
                captured[f"L{li}_attn"] = output_t[0].detach().cpu().clone()
            else:
                captured[f"L{li}_attn"] = output_t.detach().cpu().clone()
        return hook
    
    # Hook mlp output
    def make_mlp_hook(li):
        def hook(module, input_t, output_t):
            if isinstance(output_t, tuple):
                captured[f"L{li}_mlp"] = output_t[0].detach().cpu().clone()
            else:
                captured[f"L{li}_mlp"] = output_t.detach().cpu().clone()
        return hook
    
    # Hook layer residual output
    def make_layer_hook(li):
        def hook(module, input_t, output_t):
            if isinstance(output_t, tuple):
                captured[f"L{li}_resid"] = output_t[0].detach().cpu().clone()
            else:
                captured[f"L{li}_resid"] = output_t.detach().cpu().clone()
        return hook
    
    hooks = []
    for li in range(n_layers):
        if li < len(layers):
            hooks.append(layers[li].self_attn.register_forward_hook(make_attn_hook(li)))
            hooks.append(layers[li].mlp.register_forward_hook(make_mlp_hook(li)))
            hooks.append(layers[li].register_forward_hook(make_layer_hook(li)))
    
    with torch.no_grad():
        try:
            out = model(**inputs)
            last_logits = out.logits[0, -1, :].detach().cpu().float().clone()
        except Exception as e:
            log_time(f"  Cache forward failed for '{sentence[:40]}': {e}")
            last_logits = None
    
    for h in hooks:
        h.remove()
    
    return {
        "activations": captured,
        "last_logits": last_logits,
        "actual_len": actual_len,
    }


def patch_component_and_measure(model, tokenizer, sentence, device, n_layers,
                                 cached_activations, layer_idx, component,
                                 common_len, return_logits=True):
    """
    Forward sentence with component at layer_idx replaced by cached_activations.
    
    Args:
        component: "attn", "mlp", or "resid"
        cached_activations: dict from cache_activations (key = f"L{li}_{component}")
    
    Returns:
        last_logits (or None if failed)
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=common_len).to(device)
    seq_len = inputs["input_ids"].shape[1]
    if seq_len < common_len:
        pad_len = common_len - seq_len
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, pad_len), value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, pad_len), value=0)
    
    cached_key = f"L{layer_idx}_{component}"
    if cached_key not in cached_activations:
        return None
    
    cached_tensor = cached_activations[cached_key]  # [1, common_len, d_model] on CPU
    
    layers = get_layers(model)
    if layer_idx >= len(layers):
        return None
    
    # Select target module
    if component == "attn":
        target = layers[layer_idx].self_attn
    elif component == "mlp":
        target = layers[layer_idx].mlp
    elif component == "resid":
        target = layers[layer_idx]
    else:
        return None
    
    # Create patching hook
    def make_patch_hook(cached):
        def hook(module, input_t, output_t):
            # cached: [1, common_len, d_model] on CPU
            # Need to return it in the right dtype/device/shape matching original output
            if isinstance(output_t, tuple):
                orig = output_t[0]
                # Slice cached to match sequence length
                patched = cached[:, :orig.shape[1], :].to(orig.device, dtype=orig.dtype)
                return (patched,) + output_t[1:]
            else:
                patched = cached[:, :output_t.shape[1], :].to(output_t.device, dtype=output_t.dtype)
                return patched
        return hook
    
    # For resid component, we need to be more careful — the hook should replace the layer's output
    # which is the residual stream after attn+mlp
    patch_hook = target.register_forward_hook(make_patch_hook(cached_tensor))
    
    with torch.no_grad():
        try:
            out = model(**inputs)
            if return_logits:
                result = out.logits[0, -1, :].detach().cpu().float().clone()
            else:
                result = out.hidden_states[-1][0, -1, :].detach().cpu().float().clone()
        except Exception as e:
            result = None
    
    patch_hook.remove()
    return result

def measure_kl_divergence(logits_a, logits_b):
    """KL(P_a || P_b). Measures how much P_a diverges from P_b."""
    if logits_a is None or logits_b is None:
        return None
    log_p_a = F.log_softmax(logits_a, dim=-1)
    p_b = F.softmax(logits_b, dim=-1)
    return float(F.kl_div(log_p_a, p_b, reduction='sum'))

def measure_cosine_shift(logits_a, logits_b):
    """Cosine similarity between logit distributions."""
    p_a = F.softmax(logits_a, dim=-1).numpy().astype(np.float64)
    p_b = F.softmax(logits_b, dim=-1).numpy().astype(np.float64)
    dot = np.dot(p_a, p_b)
    na, nb = np.linalg.norm(p_a), np.linalg.norm(p_b)
    if na < 1e-10 or nb < 1e-10:
        return 1.0
    return float(dot / (na * nb))


# =============================================================================
# Full Patching Pipeline
# =============================================================================

def run_phase285(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase285_{model_name}.txt")

    log_time(f"{'='*60}")
    log_time(f"Phase 285: Real Forward Activation Patching — {model_name}")
    log_time(f"{'='*60}")

    # Build and sample pairs
    all_pairs = build_phase285_pairs()
    sampled_pairs = sample_pairs_for_patching(all_pairs, max_per_category=2)
    cat_counts = defaultdict(int)
    for p in sampled_pairs:
        cat_counts[p["category"]] += 1
    log_time(f"Dataset: {len(sampled_pairs)} sampled pairs from {len(cat_counts)} categories")
    for cat, cnt in sorted(cat_counts.items()):
        log_time(f"  {cat}: {cnt}")

    # Load model
    model, tokenizer, device, attn_impl = load_model_flash(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    log_time(f"Model: {model_info.model_class}, L={n_layers}, d={d_model}, mlp_type={model_info.mlp_type}")

    # Determine sample layers: ~15 evenly spaced
    if n_layers <= 20:
        sample_layers = list(range(0, n_layers, max(1, n_layers // 14))) 
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)
        sample_layers = sorted(set(sample_layers))[:15]
    else:
        step = max(1, n_layers // 14)
        sample_layers = list(range(0, n_layers, step))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)
        sample_layers = sorted(set(sample_layers))[:15]

    log_time(f"Testing {len(sample_layers)} layers: {sample_layers}")

    # Warmup
    log_time("Global warmup...")
    wu = tokenizer("warmup test for activation patching", return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            model(**wu)
        except: pass
    log_time(f"Warmup done, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

    # Verify hooks work
    log_time("Hook verification...")
    test_activations = {}
    layers = get_layers(model)

    test_attn_hooked = [False]
    test_mlp_hooked = [False]
    
    def test_attn_hook(module, input_t, output_t):
        test_attn_hooked[0] = True
    
    def test_mlp_hook(module, input_t, output_t):
        test_mlp_hooked[0] = True
    
    h1 = layers[0].self_attn.register_forward_hook(test_attn_hook)
    h2 = layers[0].mlp.register_forward_hook(test_mlp_hook)
    
    v_inputs = tokenizer("test verification sentence here", return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            model(**v_inputs)
        except: pass
    
    h1.remove(); h2.remove()
    
    log_time(f"  self_attn hook: {'FIRED' if test_attn_hooked[0] else 'MISSED'}")
    log_time(f"  mlp hook:      {'FIRED' if test_mlp_hooked[0] else 'MISSED'}")

    components_to_test = []
    if test_attn_hooked[0]:
        components_to_test.append("attn")
    if test_mlp_hooked[0]:
        components_to_test.append("mlp")
    # resid always available (layer-level hook always works)
    components_to_test.append("resid")
    
    if not test_attn_hooked[0] and not test_mlp_hooked[0]:
        log_time("  WARNING: Only resid patching available (sub-module hooks didn't fire)")
        log_time("  Likely flash_attention_2 issue — switching to eager attention")
        # We'll still proceed with resid patching
    else:
        log_time(f"  Components to patch: {components_to_test}")

    # For efficiency, limit pairs further if needed
    max_pairs = min(len(sampled_pairs), 28)  # Cap at 28 pairs total
    if len(sampled_pairs) > max_pairs:
        # Take even spread
        step = len(sampled_pairs) / max_pairs
        final_pairs = [sampled_pairs[min(int(i*step), len(sampled_pairs)-1)] for i in range(max_pairs)]
        final_pairs = list({p["name"]: p for p in final_pairs}.values())  # dedup
    else:
        final_pairs = sampled_pairs
    
    n_pairs = len(final_pairs)
    n_layers_test = len(sample_layers)
    n_components = len(components_to_test)
    n_patches = n_pairs * n_layers_test * n_components
    log_time(f"Patching plan: {n_pairs} pairs × {n_layers_test} layers × {n_components} comps = {n_patches} forwards")

    # === Main Patching Loop ===
    t_start = time.time()
    
    # Results structure: per_pair[pidx] → layers[li] = {comp: effect}
    all_pair_results = []
    n_done = 0
    n_errors = 0

    for pidx, pair in enumerate(final_pairs):
        pname = pair["name"]
        sent_a, sent_b = pair["A"], pair["B"]
        category = pair["category"]

        # Determine common sequence length
        toks_a = len(tokenizer.encode(sent_a, add_special_tokens=True))
        toks_b = len(tokenizer.encode(sent_b, add_special_tokens=True))
        common_len = max(toks_a, toks_b)
        common_len = min(common_len, 32)  # Cap at 32 tokens

        # === Step 1: Cache activations ===
        cache_a = cache_activations(model, tokenizer, sent_a, device, n_layers, common_len)
        cache_b = cache_activations(model, tokenizer, sent_b, device, n_layers, common_len)

        logits_a = cache_a.get("last_logits")
        logits_b = cache_b.get("last_logits")

        if logits_a is None or logits_b is None:
            log_time(f"  [{pidx+1}/{n_pairs}] {pname}: CACHE FAILED, skipping")
            n_errors += 1
            continue

        # Baseline: KL divergence between A and B
        kl_ab = measure_kl_divergence(logits_a, logits_b)
        if kl_ab is None or kl_ab < 1e-10:
            log_time(f"  [{pidx+1}/{n_pairs}] {pname}: KL(A||B)=0, skipping (identical outputs)")
            continue

        pair_result = {"name": pname, "category": category, "kl_ab": kl_ab, "layers": {}}

        # === Step 2: Patch each component at each layer ===
        for li in sample_layers:
            layer_effects = {}
            
            for comp in components_to_test:
                # Patch B → A: forward sentence B, replace component from A
                patched_logits = patch_component_and_measure(
                    model, tokenizer, sent_b, device, n_layers,
                    cache_a["activations"], li, comp, common_len
                )

                if patched_logits is None:
                    layer_effects[comp] = None
                    n_errors += 1
                    continue

                # Effect: how much did patching move B's output toward A?
                kl_patched_b = measure_kl_divergence(patched_logits, logits_b)
                if kl_patched_b is not None:
                    effect_ratio = min(kl_patched_b / kl_ab, 5.0)  # cap at 5x
                else:
                    effect_ratio = None

                layer_effects[comp] = {
                    "kl_patched_b": kl_patched_b,
                    "effect_ratio": effect_ratio,
                }

            pair_result["layers"][str(li)] = layer_effects

        all_pair_results.append(pair_result)
        n_done += 1

        # Progress logging
        if (pidx + 1) % 5 == 0 or pidx == 0:
            elapsed = time.time() - t_start
            rate = (pidx + 1) / max(elapsed, 1) * 3600  # pairs/hour
            eta = (n_pairs - pidx - 1) * elapsed / max(pidx + 1, 1)
            log_time(f"  [{pidx+1}/{n_pairs}] {pname[:30]:<30} {elapsed:.0f}s elapsed, "
                     f"~{rate:.0f} pair/h, ETA={eta/60:.0f}min, errors={n_errors}")

    t_total = time.time() - t_start
    log_time(f"\nPatching complete: {n_done}/{n_pairs} pairs, {n_errors} errors, {t_total:.0f}s ({t_total/60:.1f}min)")

    # === AGGREGATION ===
    log_time(f"\n{'='*60}")
    log_time(f"AGGREGATION: Per-Component, Per-Layer Effects")
    log_time(f"{'='*60}")

    # Aggregate per component per layer across all pairs
    # per_comp_layer[comp][str(li)] = [effect_ratios]
    per_comp_layer = defaultdict(lambda: defaultdict(list))
    per_cat_comp_layer = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for pr in all_pair_results:
        cat = pr["category"]
        for lk, comps in pr["layers"].items():
            for comp, data in comps.items():
                if data and data.get("effect_ratio") is not None:
                    per_comp_layer[comp][lk].append(data["effect_ratio"])
                    per_cat_comp_layer[cat][comp][lk].append(data["effect_ratio"])

    # Display per-component per-layer summary
    for comp in components_to_test:
        if comp not in per_comp_layer:
            continue
        log_time(f"\n  Component: {comp.upper()}")
        log_time(f"  {'L':>4} {'mean_eff':>8} {'std_eff':>8} {'n_valid':>8} {'condensed':>12}")
        
        comp_means = {}
        for lk in sorted(per_comp_layer[comp].keys(), key=lambda x: int(x)):
            vals = per_comp_layer[comp][lk]
            if not vals:
                continue
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            n_val = len(vals)
            comp_means[lk] = mean_val
            
            # Condensed interpretation
            if mean_val < 0.1:
                interp = "NEGLIGIBLE"
            elif mean_val < 0.3:
                interp = "WEAK"
            elif mean_val < 0.6:
                interp = "MODERATE"
            elif mean_val < 0.9:
                interp = "STRONG"
            else:
                interp = "DOMINANT"
            
            log_time(f"  L{int(lk):>3} {mean_val:8.3f} {std_val:8.3f} {n_val:8} {interp:>12}")

    # === Compare components: which is strongest at each layer? ===
    log_time(f"\n  Cross-Component Comparison (All Pairs):")
    header = f"  {'L':>4} "
    for comp in components_to_test:
        header += f"{comp:>12} "
    header += f"{'LEADER':>12}"
    log_time(header)

    for li in sample_layers:
        lk = str(li)
        means = {}
        line = f"  L{li:>3} "
        for comp in components_to_test:
            if comp in per_comp_layer and lk in per_comp_layer[comp]:
                m = float(np.mean(per_comp_layer[comp][lk]))
                means[comp] = m
                line += f"{m:12.3f} "
            else:
                line += f"{'N/A':>12} "
        
        if means:
            leader = max(means, key=means.get)
            line += f"{leader:>12}"
        log_time(line)

    # === Per-category analysis ===
    log_time(f"\n  Per-Category Component Dominance:")
    for cat in sorted(per_cat_comp_layer.keys()):
        cat_data = per_cat_comp_layer[cat]
        # Count which component dominates most layers
        comp_wins = defaultdict(int)
        for comp in cat_data:
            for lk, vals in cat_data[comp].items():
                comp_wins[comp] += len(vals)
        
        total = sum(comp_wins.values())
        if total > 0:
            parts = [f"{comp}={cnt}" for comp, cnt in sorted(comp_wins.items(), key=lambda x: -x[1])]
            log_time(f"    {cat:<18}: {', '.join(parts)}")

    # === Save results ===
    save_data = {
        "model": model_name,
        "attn_impl": attn_impl,
        "n_pairs_tested": n_done,
        "n_layers_tested": n_layers_test,
        "n_components": n_components,
        "components_tested": components_to_test,
        "total_patches": n_patches,
        "total_time_s": round(t_total, 1),
        "self_attn_hook_worked": test_attn_hooked[0],
        "mlp_hook_worked": test_mlp_hooked[0],
        "per_component_layer": {
            comp: {lk: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
                   for lk, vals in comp_data.items()}
            for comp, comp_data in per_comp_layer.items()
        },
        "per_category": {
            cat: {
                comp: {lk: {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
                       for lk, vals in comp_data.items()}
                for comp, comp_data in cat_data.items()
            }
            for cat, cat_data in per_cat_comp_layer.items()
        },
        "per_pair": [
            {
                "name": pr["name"],
                "category": pr["category"],
                "kl_ab": pr["kl_ab"],
                "layers": {
                    lk: {
                        comp: data for comp, data in comps.items() if data
                    }
                    for lk, comps in pr["layers"].items()
                }
            }
            for pr in all_pair_results
        ],
    }

    result_path = RESULT_DIR / f"{model_name}_real_patching.json"
    with open(result_path, "w") as f:
        json.dump(save_data, f, indent=2)
    log_time(f"\nResults saved to {result_path}")

    # === Cleanup ===
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()

    return save_data


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            try:
                r = run_phase285(name)
                log_time(f"\n{name} DONE: {r['n_pairs_tested']} pairs tested")
            except Exception as e:
                log_time(f"!!! {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_phase285(model_name)
