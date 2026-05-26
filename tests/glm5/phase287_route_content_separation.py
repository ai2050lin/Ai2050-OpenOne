"""
Phase 287: Route-Content Separation for Attention Heads
========================================================

GOAL: Decompose each attention head into two independent causal channels:
  - ROUTING: attention weights (QK pattern) — "which tokens to attend to"
  - CONTENT: value vectors (V) — "what information to transmit"

This directly tests the core hypothesis from Phase 280:
  "Function binding happens via VALUE content transformation,
   NOT via attention map rewiring."

For each head, we construct 4 hybrid outputs:
  1. AW_A @ V_A  → Full A patch (Phase 286 equivalent)
  2. AW_B @ V_B  → Full B patch (baseline, should be ~0 effect)
  3. AW_A @ V_B  → A's ROUTING on B's CONTENT (pure routing effect)
  4. AW_B @ V_A  → B's ROUTING on A's CONTENT (pure content effect)

Then we measure:
  - routing_ratio = effect(AW_A@V_B) / effect(AW_A@V_A)
  - content_ratio = effect(AW_B@V_A) / effect(AW_A@V_A)

Interpretation:
  - routing_ratio > 0.7: Function is ROUTING-dominant (attention rewiring)
  - content_ratio > 0.7: Function is CONTENT-dominant (value transformation)
  - both < 0.3: Routing and content are COUPLED (need both together)
  - both > 0.5: Function can be achieved via EITHER routing OR content

METHOD:
  PART 1 (CACHE): Load model with eager attention, cache AW and V per head
  PART 2 (PATCH): Load model with flash_attn_2, construct hybrid outputs,
                   pre_hook o_proj input, measure KL effects
  PART 3 (ANALYZE): Per-head routing vs content attribution

Usage:
  python tests/glm5/phase287_route_content_separation.py qwen3
  python tests/glm5/phase287_route_content_separation.py glm4
  python tests/glm5/phase287_route_content_separation.py deepseek7b
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

RESULT_DIR = Path("results/phase287_route_content")
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
# Model Loading
# =============================================================================

def load_model_eager(model_name: str):
    """Load model with EAGER attention (needed for output_attentions=True)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bf16 + EAGER + output_attentions)...")
    
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
    model.config.output_attentions = True
    model.eval()
    
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    log_time(f"  Loaded: device={device}, GPU={gpu_mem:.2f}GB, mode=EAGER+output_attn")
    return model, tokenizer, device


def load_model_flash(model_name: str):
    """Load model with FLASH_ATTN_2 for fast patching."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bf16 + flash_attn_2)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    attn_impl = "flash_attention_2"
    try:
        import flash_attn
    except ImportError:
        attn_impl = "eager"
        log_time("  flash_attn not found, using eager")
    
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
    log_time(f"  Loaded: device={device}, GPU={gpu_mem:.2f}GB, attn={attn_impl}")
    return model, tokenizer, device


# =============================================================================
# Sentence Pairs
# =============================================================================

def build_all_pairs():
    pairs = []
    
    def add_pair(name, a, b, category):
        pairs.append({"name": name, "A": a, "B": b, "category": category})
    
    # animal SVO (9)
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
        add_pair(name, a, b, "animal")
    
    # human SVO (8)
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
        add_pair(name, a, b, "human")
    
    # human_object (6)
    for name, a, b in [
        ("svo_child_apple", "the child eats the apple", "the apple is eaten by the child"),
        ("svo_chef_knife", "the chef uses the knife", "the knife is used by the chef"),
        ("svo_painter_brush", "the painter holds the brush", "the brush is held by the painter"),
        ("svo_driver_car", "the driver starts the car", "the car is started by the driver"),
        ("svo_writer_pen", "the writer lifts the pen", "the pen is lifted by the writer"),
        ("svo_guard_key", "the guard holds the key", "the key is held by the guard"),
    ]:
        add_pair(name, a, b, "human_object")
    
    # place (6)
    for name, a, b in [
        ("svo_king_city", "the king rules the city", "the city is ruled by the king"),
        ("svo_explorer_island", "the explorer discovers the island", "the island is discovered by the explorer"),
        ("svo_tourist_museum", "the tourist visits the museum", "the museum is visited by the tourist"),
        ("svo_guard_prison", "the guard watches the prison", "the prison is watched by the guard"),
        ("svo_soldier_bridge", "the soldier defends the bridge", "the bridge is defended by the soldier"),
        ("svo_mayor_town", "the mayor governs the town", "the town is governed by the mayor"),
    ]:
        add_pair(name, a, b, "place")
    
    # passive (8)
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
        add_pair(name, a, b, "passive")
    
    # negation (12)
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
        add_pair(name, a, b, "negation")
    
    # quantifier (12)
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
        add_pair(name, a, b, "quantifier")
    
    # conditional (8)
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
        add_pair(name, a, b, "conditional")
    
    # recursive (10)
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
        add_pair(name, a, b, "recursive")
    
    # translation (10)
    for name, a, b in [
        ("trans_dog_cat", "the dog chases the cat", "狗追猫"),
        ("trans_sun_east", "the sun rises in the east", "太阳从东方升起"),
        ("trans_teacher_teach", "the teacher teaches the student", "老师教学生"),
        ("trans_bird_sky", "the bird flies in the sky", "鸟在天空中飞翔"),
        ("trans_king_rules", "the king rules the kingdom", "国王统治王国"),
        ("trans_water_cold", "the water is very cold", "水非常冷"),
        ("trans_love_overcome", "love overcomes everything", "爱战胜一切"),
        ("trans_fox_hunt", "the fox hunts the rabbit", "狐狸猎兔"),
        ("trans_child_happy", "the child is very happy", "孩子非常快乐"),
        ("trans_mountain_high", "the mountain is extremely high", "这座山非常高"),
    ]:
        add_pair(name, a, b, "translation")
    
    # comparative (8)
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
        add_pair(name, a, b, "comparative")
    
    # temporal (8)
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
        add_pair(name, a, b, "temporal")
    
    # logical (8)
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
        add_pair(name, a, b, "logical")
    
    # abstract (8)
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
        add_pair(name, a, b, "abstract")
    
    return pairs


# =============================================================================
# PART 1: Cache AW and V per head for all pairs
# =============================================================================

def get_head_config(model):
    """Get attention head configuration, handling GQA."""
    layers = get_layers(model)
    layer0 = layers[0]
    o_proj = layer0.self_attn.o_proj
    
    n_heads = model.config.num_attention_heads
    n_kv_heads = getattr(model.config, 'num_key_value_heads', n_heads)
    concat_dim = o_proj.weight.shape[1]
    head_dim = concat_dim // n_heads
    
    kv_head_dim = head_dim  # KV head dim = Q head dim for standard models
    
    return n_heads, n_kv_heads, head_dim, concat_dim


def cache_aw_and_v(model, tokenizer, pairs, device, n_layers, head_config,
                   target_layers=None, max_len=32):
    """
    Cache attention weights and V values for all pairs.
    
    Only caches the specified target_layers (or all if None).
    
    Returns: {
        pname: {
            "A": {"L{li}": {"AW": tensor, "V": tensor}},
            "B": {"L{li}": {"AW": tensor, "V": tensor}},
            "category": str,
            "seq_len": int,
        }
    }
    """
    n_heads, n_kv_heads, head_dim, concat_dim = head_config
    layers = get_layers(model)
    
    if target_layers is None:
        target_layers = list(range(n_layers))
    target_set = set(target_layers)
    
    all_cached = {}
    t0 = time.time()
    
    for pidx, pair in enumerate(pairs):
        pname = pair["name"]
        sent_a, sent_b = pair["A"], pair["B"]
        category = pair["category"]
        
        toks_a = len(tokenizer.encode(sent_a, add_special_tokens=True))
        toks_b = len(tokenizer.encode(sent_b, add_special_tokens=True))
        common_len = min(max(toks_a, toks_b), max_len)
        
        pair_cache = {"category": category, "A": {}, "B": {}, "seq_len": common_len}
        
        for sent_key, sent in [("A", sent_a), ("B", sent_b)]:
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=common_len).to(device)
            seq_len = inputs["input_ids"].shape[1]
            if seq_len < common_len:
                pad_len = common_len - seq_len
                inputs["input_ids"] = F.pad(inputs["input_ids"], (0, pad_len), value=tokenizer.pad_token_id)
                inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, pad_len), value=0)
            
            v_cache = {}
            v_hooks = []
            for li in sorted(target_set):
                if li < len(layers):
                    def make_v_hook(li):
                        def hook(module, input_t, output_t):
                            v_cache[f"L{li}"] = output_t.detach().cpu().clone()
                        return hook
                    v_hooks.append(layers[li].self_attn.v_proj.register_forward_hook(make_v_hook(li)))
            
            with torch.no_grad():
                try:
                    out = model(**inputs)
                except Exception as e:
                    log_time(f"  [{pidx+1}/{len(pairs)}] {pname} {sent_key}: FORWARD FAILED {e}")
                    for h in v_hooks: h.remove()
                    v_cache = None
                    break
            
            for h in v_hooks:
                h.remove()
            
            if v_cache is not None and hasattr(out, 'attentions') and out.attentions is not None:
                attn_weights = out.attentions  # tuple of (n_layers) tensors
                layer_data = {}
                for li in target_set:
                    if li < len(attn_weights) and f"L{li}" in v_cache:
                        layer_data[f"L{li}"] = {
                            "AW": attn_weights[li].detach().cpu().clone(),  # [1, n_heads, seq, seq]
                            "V": v_cache[f"L{li}"],
                        }
                pair_cache[sent_key] = layer_data
        
        if pair_cache["A"] and pair_cache["B"]:
            all_cached[pname] = pair_cache
        
        if (pidx + 1) % 20 == 0 or pidx == 0:
            elapsed = time.time() - t0
            rate = (pidx + 1) / max(elapsed, 1) * 3600
            eta = (len(pairs) - pidx - 1) * elapsed / max(pidx + 1, 1)
            log_time(f"  [{pidx+1}/{len(pairs)}] Caching: {elapsed:.0f}s, "
                     f"~{rate:.0f} pairs/h, ETA={eta/60:.0f}min, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    t_total = time.time() - t0
    log_time(f"  Caching done: {len(all_cached)}/{len(pairs)} pairs, {t_total:.0f}s ({t_total/60:.1f}min)")
    return all_cached


# =============================================================================
# PART 2: Construct Hybrid Head Outputs
# =============================================================================

def get_kv_head_idx(head_idx, n_heads, n_kv_heads):
    """Map query head index to KV head index (for GQA)."""
    if n_heads == n_kv_heads:
        return head_idx
    n_rep = n_heads // n_kv_heads
    return head_idx // n_rep


def compute_head_output(AW, V_kv, head_idx, n_heads, n_kv_heads, head_dim):
    """
    Compute head output: AW_h @ V_h
    
    Args:
        AW: [batch, n_heads, seq_q, seq_kv] — attention weights
        V_kv: [batch, seq_kv, n_kv_heads * head_dim] — V values (before v_proj reshape)
        head_idx: which query head
        n_heads, n_kv_heads, head_dim: head config
    
    Returns:
        head_output: [batch, seq_q, head_dim]
    """
    kv_idx = get_kv_head_idx(head_idx, n_heads, n_kv_heads)
    
    # Extract V for this KV head
    v_start = kv_idx * head_dim
    v_end = (kv_idx + 1) * head_dim
    V_h = V_kv[:, :, v_start:v_end]  # [batch, seq_kv, head_dim]
    
    # Extract AW for this query head
    AW_h = AW[:, head_idx, :, :]  # [batch, seq_q, seq_kv]
    
    # head_output = AW_h @ V_h
    head_output = torch.bmm(AW_h.float(), V_h.float())  # [batch, seq_q, head_dim]
    
    return head_output


def construct_hybrid_outputs(cached_pair_data, li, hi, head_config, device, dtype):
    """
    Construct 4 hybrid head outputs for route-content separation.
    
    Returns dict:
      "AW_A_V_A": A's routing + A's content (full A)
      "AW_B_V_B": B's routing + B's content (full B)
      "AW_A_V_B": A's routing + B's content (routing effect)
      "AW_B_V_A": B's routing + A's content (content effect)
    """
    n_heads, n_kv_heads, head_dim, concat_dim = head_config
    lk = f"L{li}"
    
    AW_A = cached_pair_data["A"][lk]["AW"]  # [1, n_heads, seq, seq]
    V_A = cached_pair_data["A"][lk]["V"]    # [1, seq, n_kv_heads*head_dim]
    AW_B = cached_pair_data["B"][lk]["AW"]
    V_B = cached_pair_data["B"][lk]["V"]
    
    # Move to device
    AW_A = AW_A.to(device)
    V_A = V_A.to(device)
    AW_B = AW_B.to(device)
    V_B = V_B.to(device)
    
    ho_AA = compute_head_output(AW_A, V_A, hi, n_heads, n_kv_heads, head_dim)
    ho_BB = compute_head_output(AW_B, V_B, hi, n_heads, n_kv_heads, head_dim)
    ho_AB = compute_head_output(AW_A, V_B, hi, n_heads, n_kv_heads, head_dim)
    ho_BA = compute_head_output(AW_B, V_A, hi, n_heads, n_kv_heads, head_dim)
    
    return {
        "AW_A_V_A": ho_AA.to(dtype).cpu(),
        "AW_B_V_B": ho_BB.to(dtype).cpu(),
        "AW_A_V_B": ho_AB.to(dtype).cpu(),
        "AW_B_V_A": ho_BA.to(dtype).cpu(),
    }


# =============================================================================
# PART 3: Route-Content Patching
# =============================================================================

def patch_head_o_proj(model, tokenizer, sentence, device, n_layers,
                       patch_vector, target_layer, target_head, head_config, common_len):
    """
    Forward with a specific head's o_proj input slot replaced.
    
    Args:
        patch_vector: [seq, head_dim] tensor (on CPU) — replacement for the head slot
        target_layer, target_head: which head to replace
    
    Returns:
        last_logits [vocab_size] or None
    """
    n_heads, n_kv_heads, head_dim, concat_dim = head_config
    layers = get_layers(model)
    if target_layer >= len(layers):
        return None
    
    # Tokenize
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=common_len).to(device)
    seq_len = inputs["input_ids"].shape[1]
    if seq_len < common_len:
        pad_len = common_len - seq_len
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, pad_len), value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, pad_len), value=0)
    
    o_proj = layers[target_layer].self_attn.o_proj
    pv = patch_vector.clone().to(device).to(torch.bfloat16)  # [seq, head_dim]
    
    def pre_hook(module, args):
        new_input = args[0].clone()  # [batch, seq, concat_dim]
        s = target_head * head_dim
        e = (target_head + 1) * head_dim
        # pv is [seq, head_dim], need to broadcast to [batch, seq, head_dim]
        new_input[0, :, s:e] = pv
        return (new_input,)
    
    hook_handle = o_proj.register_forward_pre_hook(pre_hook)
    
    with torch.no_grad():
        try:
            out = model(**inputs)
            result = out.logits[0, -1, :].detach().cpu().float().clone()
        except Exception as e:
            result = None
    
    hook_handle.remove()
    return result, seq_len


def measure_kl(logits_a, logits_b):
    """KL(P_b || P_a) — how different B is from A."""
    if logits_a is None or logits_b is None:
        return None
    log_p_a = F.log_softmax(logits_a, dim=-1)
    p_b = F.softmax(logits_b, dim=-1)
    return float(F.kl_div(log_p_a, p_b, reduction='sum'))


# =============================================================================
# Main Pipeline
# =============================================================================

def run_phase287(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase287_{model_name}.txt")
    
    log_time(f"{'='*60}")
    log_time(f"Phase 287: Route-Content Separation — {model_name}")
    log_time(f"{'='*60}")
    
    # ======== SELECT HEADS (based on Phase 286 results) — must be before caching ========
    head_selection = {
        "qwen3": [
            (16, 27, "L16_H27"),   # top causal head (0.077)
            (12, 25, "L12_H25"),   # 2nd (0.049)
            (12, 0, "L12_H0"),     # 3rd (0.042)
            (35, 0, "L35_H0"),     # 4th (0.040)
            (28, 14, "L28_H14"),   # 5th (0.034)
        ],
        "glm4": [
            (16, 7, "L16_H7"),     # translation top (2.310)
            (8, 16, "L8_H16"),     # translation 2nd (2.294)
            (28, 27, "L28_H27"),   # translation 3rd (2.291)
            (16, 30, "L16_H30"),   # quantifier (0.179)
            (8, 26, "L8_H26"),     # human/comparative (0.100)
        ],
        "deepseek7b": [
            (0, 10, "L0_H10"),     # top causal (0.789)
            (0, 3, "L0_H3"),       # universal head (0.776)
            (21, 9, "L21_H9"),     # 3rd (0.637)
            (18, 0, "L18_H0"),     # 4th (0.572)
            (24, 15, "L24_H15"),   # 5th (0.507)
        ],
    }
    
    selected_heads = head_selection.get(model_name, head_selection["qwen3"])
    target_layers = sorted(set(li for li, hi, label in selected_heads))
    
    log_time(f"Target: {len(selected_heads)} heads across {len(target_layers)} layers")
    for li, hi, label in selected_heads:
        log_time(f"  {label} (layer={li}, head={hi})")
    
    # ==================================================================
    # PART 1: Cache AW + V (EAGER attention)
    # ==================================================================
    log_time(f"\n{'='*60}")
    log_time("PART 1: Caching AW + V (EAGER attention)")
    log_time(f"{'='*60}")
    
    model_eager, tokenizer, device = load_model_eager(model_name)
    model_info = get_model_info(model_eager, model_name)
    n_layers = model_info.n_layers
    head_config = get_head_config(model_eager)
    n_heads, n_kv_heads, head_dim, concat_dim = head_config
    
    log_time(f"Model: {model_info.model_class}, L={n_layers}, "
             f"n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}, "
             f"d_model={model_info.d_model}")
    log_time(f"GQA ratio: {n_heads // n_kv_heads}x" if n_heads > n_kv_heads else "MHA: 1 head = 1 KV")
    
    all_pairs = build_all_pairs()
    log_time(f"Dataset: {len(all_pairs)} pairs")
    
    # Warmup
    log_time("Warmup (eager)...")
    wu = tokenizer("warmup", return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            model_eager(**wu)
        except:
            pass
    
    # Cache AW + V (only for target layers to save time/memory)
    cached_data = cache_aw_and_v(model_eager, tokenizer, all_pairs, device, n_layers, head_config,
                                  target_layers=target_layers)
    n_cached = len(cached_data)
    
    # Release eager model
    release_model(model_eager)
    del model_eager
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)
    log_time(f"Cached {n_cached} pairs. Eager model released. GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # ==================================================================
    # PART 2: Route-Content Patching (FLASH_ATTN_2)
    # ==================================================================
    log_time(f"\n{'='*60}")
    log_time("PART 2: Route-Content Patching (flash_attn_2)")
    log_time(f"{'='*60}")
    
    model_flash, _, device = load_model_flash(model_name)
    log_time(f"Flash model loaded. GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # Warmup flash
    wu_f = tokenizer("warmup flash", return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            model_flash(**wu_f)
        except:
            pass
    
    # ======== Select pairs with strong KL signal ========
    # First compute baseline KLs for all pairs
    log_time("\nComputing baseline logits for all pairs...")
    log_time(f"  (This reduces noise by filtering out low-KL pairs)")
    
    pair_baselines = {}
    t0_baseline = time.time()
    
    for pidx, pair in enumerate(all_pairs):
        pname = pair["name"]
        if pname not in cached_data:
            continue
        
        sent_a, sent_b = pair["A"], pair["B"]
        common_len = cached_data[pname]["seq_len"]
        
        # Forward A
        inputs_a = tokenizer(sent_a, return_tensors="pt", truncation=True, max_length=common_len).to(device)
        with torch.no_grad():
            out_a = model_flash(**inputs_a)
            logits_a = out_a.logits[0, -1, :].detach().cpu().float()
        
        # Forward B
        inputs_b = tokenizer(sent_b, return_tensors="pt", truncation=True, max_length=common_len).to(device)
        with torch.no_grad():
            out_b = model_flash(**inputs_b)
            logits_b = out_b.logits[0, -1, :].detach().cpu().float()
        
        kl_ab = measure_kl(logits_a, logits_b)
        
        pair_baselines[pname] = {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "kl_ab": kl_ab,
            "sent_a": sent_a,
            "sent_b": sent_b,
            "category": pair["category"],
        }
        
        if (pidx + 1) % 30 == 0:
            elapsed = time.time() - t0_baseline
            log_time(f"  [{pidx+1}/{len(all_pairs)}] Baselining: {elapsed:.0f}s")
    
    t_baseline = time.time() - t0_baseline
    log_time(f"  Baselining done: {len(pair_baselines)} pairs, {t_baseline:.0f}s")
    
    # Determine per-model KL threshold to get meaningful pairs
    all_kls = [pb["kl_ab"] for pb in pair_baselines.values() if pb["kl_ab"] is not None and pb["kl_ab"] > 0]
    all_kls_sorted = sorted(all_kls, reverse=True)
    if len(all_kls_sorted) > 50:
        kl_threshold = all_kls_sorted[49]  # top 50
    elif len(all_kls_sorted) > 20:
        kl_threshold = all_kls_sorted[min(49, len(all_kls_sorted) - 1)]
    else:
        kl_threshold = 0
    
    patching_pairs = []
    for pname, pb in pair_baselines.items():
        if pb["kl_ab"] is not None and pb["kl_ab"] >= kl_threshold:
            patching_pairs.append(pname)
    
    # Ensure per-category coverage (at least 2 per category)
    cats_covered = defaultdict(int)
    for pname in patching_pairs:
        cat = pair_baselines[pname]["category"]
        cats_covered[cat] += 1
    
    # Add additional pairs for under-represented categories
    for pname, pb in pair_baselines.items():
        cat = pb["category"]
        if cats_covered[cat] < 2 and pb["kl_ab"] is not None and pb["kl_ab"] > 0:
            if pname not in patching_pairs:
                patching_pairs.append(pname)
                cats_covered[cat] += 1
    
    patching_pairs = list(set(patching_pairs))
    log_time(f"Patching pairs: {len(patching_pairs)} (KL >= {kl_threshold:.3f})")
    for cat in sorted(cats_covered.keys()):
        log_time(f"  {cat}: {cats_covered[cat]} pairs")
    
    # ======== Run Route-Content Patching ========
    n_total_patches = len(selected_heads) * len(patching_pairs) * 4  # 4 conditions
    log_time(f"\nTotal patching forwards: {len(selected_heads)} heads × "
             f"{len(patching_pairs)} pairs × 4 conditions = {n_total_patches}")
    
    # STRUCTURE: results[(pname, li, hi)] = {
    #     "AA_effect": effect of AW_A@V_A,
    #     "AB_effect": effect of AW_A@V_B (routing),
    #     "BA_effect": effect of AW_B@V_A (content),
    #     "BB_effect": effect of AW_B@V_B (sanity, should be ~0),
    #     "kl_ab": baseline KL,
    # }
    results = {}
    t0_patch = time.time()
    n_done = 0
    n_errors = 0
    
    # Pre-compute all hybrid outputs (avoids repeated AW@V computation)
    log_time("Pre-computing hybrid head outputs...")
    hybrid_cache = {}  # {(pname, li, hi): {condition: tensor}}
    
    for (li, hi, label) in selected_heads:
        for pname in patching_pairs:
            if pname not in cached_data or f"L{li}" not in cached_data[pname].get("A", {}):
                continue
            try:
                hy = construct_hybrid_outputs(
                    cached_data[pname], li, hi, head_config, device, torch.bfloat16
                )
                hybrid_cache[(pname, li, hi)] = hy
            except Exception as e:
                log_time(f"  Hybrid construction failed ({pname}, L{li}_H{hi}): {e}")
    
    log_time(f"Pre-computed {len(hybrid_cache)} hybrid outputs")
    
    # Now run patching
    condition_names = ["AW_A_V_A", "AW_A_V_B", "AW_B_V_A", "AW_B_V_B"]
    condition_types = ["full_A", "routing_A_content_B", "content_A_routing_B", "full_B"]
    
    for (li, hi, label) in selected_heads:
        for pidx, pname in enumerate(patching_pairs):
            key = (pname, li, hi)
            if key not in hybrid_cache:
                continue
            
            pb = pair_baselines[pname]
            sent_b = pb["sent_b"]
            logits_b = pb["logits_b"]
            kl_ab = pb["kl_ab"]
            common_len = cached_data[pname]["seq_len"]
            hy = hybrid_cache[key]
            
            if kl_ab is None or kl_ab < 1e-6:
                continue
            
            pair_results = {"kl_ab": kl_ab, "category": pb["category"]}
            
            for cond_name, cond_type in zip(condition_names, condition_types):
                patch_vec = hy[cond_name][0]  # [seq, head_dim], remove batch dim
                
                patched_logits, _ = patch_head_o_proj(
                    model_flash, tokenizer, sent_b, device, n_layers,
                    patch_vec, li, hi, head_config, common_len
                )
                
                if patched_logits is not None:
                    kl_patched = measure_kl(patched_logits, logits_b)
                    if kl_patched is not None:
                        pair_results[f"{cond_type}_kl"] = kl_patched
                        # effect ratio
                        aa_kl = pair_results.get("full_A_kl", kl_ab)
                        if aa_kl > 1e-6:
                            pair_results[f"{cond_type}_ratio"] = min(kl_patched / aa_kl, 5.0)
                        else:
                            pair_results[f"{cond_type}_ratio"] = None
                    else:
                        n_errors += 1
                else:
                    n_errors += 1
            
            if "full_A_kl" in pair_results:
                results[key] = pair_results
            n_done += 1
        
        # Progress
        elapsed = time.time() - t0_patch
        done_this_head = sum(1 for k in results if k[1] == li and k[2] == hi)
        log_time(f"  {label}: {done_this_head}/{len(patching_pairs)} done, "
                 f"{elapsed:.0f}s, errors={n_errors}")
    
    t_total_patch = time.time() - t0_patch
    log_time(f"\nPatching complete: {len(results)} results, {n_errors} errors, "
             f"{t_total_patch:.0f}s ({t_total_patch/60:.1f}min)")
    
    # ==================================================================
    # PART 3: Analyze route vs content
    # ==================================================================
    log_time(f"\n{'='*60}")
    log_time("PART 3: Route-Content Analysis")
    log_time(f"{'='*60}")
    
    # Aggregate per head
    log_time(f"\n  Per-Head Route-Content Decomposition:")
    log_time(f"  {'Head':>12} {'N':>5} {'Full A':>8} {'Routing':>8} {'Content':>8} {'Full B':>8} {'Interpretation':>24}")
    
    head_agg = defaultdict(list)
    for (pname, li, hi), pr in results.items():
        key = f"L{li}_H{hi}"
        head_agg[key].append(pr)
    
    per_head_analysis = {}
    for hlabel, prs in sorted(head_agg.items()):
        full_a_ratios = [p.get("full_A_ratio", None) for p in prs]
        routing_ratios = [p.get("routing_A_content_B_ratio", None) for p in prs]
        content_ratios = [p.get("content_A_routing_B_ratio", None) for p in prs]
        full_b_ratios = [p.get("full_B_ratio", None) for p in prs]
        
        # Filter None
        fa = [r for r in full_a_ratios if r is not None]
        rr = [r for r in routing_ratios if r is not None]
        cr = [r for r in content_ratios if r is not None]
        fb = [r for r in full_b_ratios if r is not None]
        
        if len(fa) == 0:
            continue
        
        mean_fa = np.mean(fa)
        mean_rr = np.mean(rr) if rr else 0
        mean_cr = np.mean(cr) if cr else 0
        mean_fb = np.mean(fb) if fb else 0
        
        # Interpretation
        if mean_rr > 0.7 * mean_fa and mean_cr < 0.3 * mean_fa:
            interp = "ROUTING-dominant"
        elif mean_cr > 0.7 * mean_fa and mean_rr < 0.3 * mean_fa:
            interp = "CONTENT-dominant"
        elif mean_rr < 0.3 * mean_fa and mean_cr < 0.3 * mean_fa:
            interp = "COUPLED (need both)"
        elif mean_rr > 0.5 * mean_fa and mean_cr > 0.5 * mean_fa:
            interp = "EITHER (separable)"
        elif mean_rr > mean_cr:
            interp = f"routing-leaning ({mean_rr/mean_fa:.2f}v{mean_cr/mean_fa:.2f})"
        else:
            interp = f"content-leaning ({mean_cr/mean_fa:.2f}v{mean_rr/mean_fa:.2f})"
        
        per_head_analysis[hlabel] = {
            "n": len(prs),
            "mean_full_A_ratio": float(mean_fa),
            "mean_routing_ratio": float(mean_rr),
            "mean_content_ratio": float(mean_cr),
            "mean_full_B_ratio": float(mean_fb),
            "interpretation": interp,
        }
        
        log_time(f"  {hlabel:>12} {len(prs):>5} {mean_fa:8.3f} {mean_rr:8.3f} {mean_cr:8.3f} {mean_fb:8.3f} {interp:>24}")
    
    # Aggregate per category
    log_time(f"\n  Per-Category Route-Content Breakdown:")
    cat_agg = defaultdict(lambda: defaultdict(list))
    for (pname, li, hi), pr in results.items():
        cat = pr.get("category", "unknown")
        hlabel = f"L{li}_H{hi}"
        for key in ["full_A_ratio", "routing_A_content_B_ratio", "content_A_routing_B_ratio", "full_B_ratio"]:
            val = pr.get(key, None)
            if val is not None:
                cat_agg[cat][key].append(val)
    
    per_cat_analysis = {}
    for cat in sorted(cat_agg.keys()):
        datum = cat_agg[cat]
        m_fa = np.mean(datum["full_A_ratio"]) if datum["full_A_ratio"] else 0
        m_rr = np.mean(datum["routing_A_content_B_ratio"]) if datum["routing_A_content_B_ratio"] else 0
        m_cr = np.mean(datum["content_A_routing_B_ratio"]) if datum["content_A_routing_B_ratio"] else 0
        
        if m_fa > 0.001:
            per_cat_analysis[cat] = {
                "full_A": float(m_fa),
                "routing": float(m_rr),
                "content": float(m_cr),
                "n_points": len(datum["full_A_ratio"]),
            }
            log_time(f"    {cat:<18}: full_A={m_fa:.3f}, routing={m_rr:.3f}, content={m_cr:.3f}")
    
    # Cross-head routing-content balance
    log_time(f"\n  Cross-Head Routing-vs-Content Balance:")
    for hlabel, pa in sorted(per_head_analysis.items()):
        rr = pa["mean_routing_ratio"]
        cr = pa["mean_content_ratio"]
        fa = pa["mean_full_A_ratio"]
        if fa > 0.001:
            log_time(f"    {hlabel}: routing/content ratio = {rr/cr:.2f}" if cr > 0.001 else f"    {hlabel}: routing/content ratio = INF")
        else:
            log_time(f"    {hlabel}: no significant full_A effect")
    
    # ==================================================================
    # PART 4: Save results
    # ==================================================================
    save_data = {
        "model": model_name,
        "head_config": {
            "n_heads": n_heads,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "concat_dim": concat_dim,
        },
        "n_pairs_cached": n_cached,
        "n_pairs_patched": len(patching_pairs),
        "n_heads_tested": len(selected_heads),
        "n_patches_total": len(results),
        "per_head_analysis": per_head_analysis,
        "per_category_analysis": per_cat_analysis,
        "selected_heads": [(li, hi, label) for li, hi, label in selected_heads],
        "summary": {
            "routing_dominant_heads": [h for h, pa in per_head_analysis.items() if pa["interpretation"] == "ROUTING-dominant"],
            "content_dominant_heads": [h for h, pa in per_head_analysis.items() if pa["interpretation"] == "CONTENT-dominant"],
            "coupled_heads": [h for h, pa in per_head_analysis.items() if "COUPLED" in pa["interpretation"]],
        }
    }
    
    save_path = RESULT_DIR / f"{model_name}_route_content.json"
    with open(save_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    log_time(f"\nResults saved to {save_path}")
    
    # Cleanup
    release_model(model_flash)
    gc.collect()
    torch.cuda.empty_cache()
    
    return save_data


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            try:
                r = run_phase287(name)
                log_time(f"\n{name} DONE: {len(r['per_head_analysis'])} heads analyzed")
            except Exception as e:
                log_time(f"!!! {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_phase287(model_name)
