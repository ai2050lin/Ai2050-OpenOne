"""
Phase 286: Head-Level Real Forward Patching (Causal Head Decomposition)
======================================================================

GOAL: Descend from component-level (Phase 285: attn/mlp/resid) to head-level
causal decomposition. Directly answer: which heads implement which function?
Which heads are reused across functions? Which are function-specific?

METHOD (Three-Part Pipeline):
  PART 1 — CACHE: Forward all 121 sentence pairs, hook self_attn.o_proj INPUT
          (the concatenated head outputs before W_o projection).
          Shape: [batch, seq, n_heads * head_dim]
          
  PART 2 — ANALYZE: Compute per-head diff norm ||input_A_h - input_B_h||.
          Rank heads by importance. Build function × head diff matrix.
          
  PART 3 — PATCH: For top-ranked heads, do real forward patching
          (pre_hook on o_proj, replace head slot from sentence A).
          Measure causal effect via KL divergence.
          
  PART 4 — AGGREGATE: Head reuse matrix, universal vs specialized heads,
          function × head causal map.

KEY ADVANTAGES OVER COMPONENT-LEVEL:
- Head is the natural unit of attention computation
- Each head = one QK routing + one V content computation
- Head-level directly addresses "which heads do negation/translation/recursion share?"
- Uses real forward (no manual QK/RoPE computation)

CAVEATS:
- Only patches attention heads, not MLP neurons (Phase 287)
- o_proj input = head outputs BEFORE W_o → captures both routing AND content
- Head_dim determined dynamically from o_proj weight shape (not d_model/n_heads)

Usage:
  python tests/glm5/phase286_head_level_patching.py qwen3
  python tests/glm5/phase286_head_level_patching.py glm4
  python tests/glm5/phase286_head_level_patching.py deepseek7b
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

RESULT_DIR = Path("results/phase286_head_patching")
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

def load_model_flash(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    log_time(f"Loading {model_name} (bf16 + flash_attn_2 + device_map='auto')...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    attn_impl = "flash_attention_2"
    try:
        import flash_attn
        log_time("  flash_attn available")
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
    return model, tokenizer, device, attn_impl

# =============================================================================
# Sentence Pair Builder (121 pairs from Phase 284)
# =============================================================================

def build_all_pairs():
    pairs = []
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
        pairs.append({"name": name, "A": a, "B": b, "category": "animal"})
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
        pairs.append({"name": name, "A": a, "B": b, "category": "human"})
    # human_object (6)
    for name, a, b in [
        ("svo_child_apple", "the child eats the apple", "the apple is eaten by the child"),
        ("svo_chef_knife", "the chef uses the knife", "the knife is used by the chef"),
        ("svo_painter_brush", "the painter holds the brush", "the brush is held by the painter"),
        ("svo_driver_car", "the driver starts the car", "the car is started by the driver"),
        ("svo_writer_pen", "the writer lifts the pen", "the pen is lifted by the writer"),
        ("svo_guard_key", "the guard holds the key", "the key is held by the guard"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "human_object"})
    # place (6)
    for name, a, b in [
        ("svo_king_city", "the king rules the city", "the city is ruled by the king"),
        ("svo_explorer_island", "the explorer discovers the island", "the island is discovered by the explorer"),
        ("svo_tourist_museum", "the tourist visits the museum", "the museum is visited by the tourist"),
        ("svo_guard_prison", "the guard watches the prison", "the prison is watched by the guard"),
        ("svo_soldier_bridge", "the soldier defends the bridge", "the bridge is defended by the soldier"),
        ("svo_mayor_town", "the mayor governs the town", "the town is governed by the mayor"),
    ]:
        pairs.append({"name": name, "A": a, "B": b, "category": "place"})
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
        pairs.append({"name": name, "A": a, "B": b, "category": "passive"})
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
        pairs.append({"name": name, "A": a, "B": b, "category": "negation"})
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
        pairs.append({"name": name, "A": a, "B": b, "category": "quantifier"})
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
        pairs.append({"name": name, "A": a, "B": b, "category": "conditional"})
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
        pairs.append({"name": name, "A": a, "B": b, "category": "recursive"})
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
        pairs.append({"name": name, "A": a, "B": b, "category": "translation"})
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
        pairs.append({"name": name, "A": a, "B": b, "category": "comparative"})
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
        pairs.append({"name": name, "A": a, "B": b, "category": "temporal"})
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
        pairs.append({"name": name, "A": a, "B": b, "category": "logical"})
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
        pairs.append({"name": name, "A": a, "B": b, "category": "abstract"})
    return pairs

# =============================================================================
# PART 1: Cache o_proj inputs for all pairs
# =============================================================================

def get_head_dim_and_heads(model):
    """Dynamically determine head_dim and n_heads from o_proj weight shape."""
    layers = get_layers(model)
    layer0 = layers[0]
    o_proj = layer0.self_attn.o_proj
    # o_proj.weight shape: [d_model, n_heads * head_dim]
    n_heads = model.config.num_attention_heads
    concat_dim = o_proj.weight.shape[1]
    head_dim = concat_dim // n_heads
    return head_dim, n_heads, concat_dim

def cache_o_proj_inputs(model, tokenizer, pairs, device, n_layers, max_len=32):
    """
    For each pair, forward A and B, hook self_attn.o_proj INPUT at every layer.
    Returns: {
        pair_name: {
            "A": {"L{li}": tensor[1, seq, concat_dim] on CPU},
            "B": {"L{li}": tensor[1, seq, concat_dim] on CPU},
            "category": str,
        }
    }
    """
    layers = get_layers(model)
    
    all_cached = {}
    t0 = time.time()
    
    for pidx, pair in enumerate(pairs):
        pname = pair["name"]
        sent_a, sent_b = pair["A"], pair["B"]
        category = pair["category"]
        
        # Determine common length
        toks_a = len(tokenizer.encode(sent_a, add_special_tokens=True))
        toks_b = len(tokenizer.encode(sent_b, add_special_tokens=True))
        common_len = min(max(toks_a, toks_b), max_len)
        
        pair_cache = {"category": category, "A": {}, "B": {}}
        
        for sent_key, sent in [("A", sent_a), ("B", sent_b)]:
            # Tokenize
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=common_len).to(device)
            seq_len = inputs["input_ids"].shape[1]
            if seq_len < common_len:
                pad_len = common_len - seq_len
                inputs["input_ids"] = F.pad(inputs["input_ids"], (0, pad_len), value=tokenizer.pad_token_id)
                inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, pad_len), value=0)
            
            captured = {}
            
            def make_hook(li):
                def hook(module, input_t, output_t):
                    # input_t[0] is the concatenated heads [batch, seq, n_heads*head_dim]
                    captured[f"L{li}"] = input_t[0].detach().cpu().clone()
                return hook
            
            hooks = []
            for li in range(n_layers):
                if li < len(layers):
                    hooks.append(layers[li].self_attn.o_proj.register_forward_hook(make_hook(li)))
            
            with torch.no_grad():
                try:
                    model(**inputs)
                except Exception as e:
                    log_time(f"  [{pidx+1}/{len(pairs)}] {pname} {sent_key}: FORWARD FAILED {e}")
                    for h in hooks: h.remove()
                    captured = None
                    break
            
            for h in hooks:
                h.remove()
            
            if captured is not None:
                # Only keep last-token activations for efficiency
                last_token = {}
                for lk, t in captured.items():
                    last_token[lk] = t[0, -1, :]  # [concat_dim]
                pair_cache[sent_key] = last_token
        
        if pair_cache["A"] and pair_cache["B"]:
            all_cached[pname] = pair_cache
        
        # Progress
        if (pidx + 1) % 20 == 0 or pidx == 0:
            elapsed = time.time() - t0
            rate = (pidx + 1) / max(elapsed, 1) * 3600
            eta = (len(pairs) - pidx - 1) * elapsed / max(pidx + 1, 1)
            log_time(f"  [{pidx+1}/{len(pairs)}] Caching o_proj inputs: {elapsed:.0f}s, "
                     f"~{rate:.0f} pairs/h, ETA={eta/60:.0f}min")
    
    t_total = time.time() - t0
    log_time(f"  Caching done: {len(all_cached)}/{len(pairs)} pairs, {t_total:.0f}s ({t_total/60:.1f}min)")
    return all_cached

# =============================================================================
# PART 2: Analyze head diffs
# =============================================================================

def compute_head_diffs(cached_data, n_layers, n_heads, head_dim):
    """
    For each pair, compute per-head diff norm:
    ||input_A_h - input_B_h||_2
    
    Aggregate by pair category.
    Returns:
        per_pair: {pname: {category, layers: {li: [head_diffs]}}}
        per_category: {cat: {li: {hi: mean_diff}}}  
        global_ranking: [(layer, head, mean_diff), ...] sorted desc
    """
    per_pair = {}
    per_category = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    all_heads = defaultdict(list)  # (li, hi) -> [diffs]
    
    for pname, pdata in cached_data.items():
        cat = pdata["category"]
        pair_result = {"category": cat, "layers": {}}
        
        for lk in pdata["A"]:
            if lk not in pdata["B"]:
                continue
            li = int(lk[1:])
            
            a_vec = pdata["A"][lk].float().numpy()  # [concat_dim]
            b_vec = pdata["B"][lk].float().numpy()  # [concat_dim]
            
            head_diffs = []
            for hi in range(n_heads):
                start = hi * head_dim
                end = (hi + 1) * head_dim
                diff = a_vec[start:end] - b_vec[start:end]
                diff_norm = float(np.linalg.norm(diff))
                head_diffs.append(diff_norm)
                all_heads[(li, hi)].append(diff_norm)
                per_category[cat][li][hi].append(diff_norm)
            
            pair_result["layers"][li] = head_diffs
        
        per_pair[pname] = pair_result
    
    # Global ranking
    global_ranking = []
    for (li, hi), diffs in all_heads.items():
        mean_diff = float(np.mean(diffs))
        std_diff = float(np.std(diffs))
        global_ranking.append((li, hi, mean_diff, std_diff, len(diffs)))
    global_ranking.sort(key=lambda x: -x[2])
    
    # Per-category aggregation
    per_cat_agg = {}
    for cat in per_category:
        cat_heads = {}
        for li in per_category[cat]:
            for hi in per_category[cat][li]:
                diffs = per_category[cat][li][hi]
                cat_heads[f"L{li}_H{hi}"] = {
                    "mean": float(np.mean(diffs)),
                    "std": float(np.std(diffs)),
                    "n": len(diffs),
                }
        per_cat_agg[cat] = cat_heads
    
    return {
        "per_pair": per_pair,
        "per_category": per_cat_agg,
        "global_ranking": global_ranking,
    }

def print_head_diff_summary(diff_results, n_layers, n_heads, n_show=20):
    """Print top heads by diff norm."""
    ranking = diff_results["global_ranking"]
    
    log_time(f"\n  Top {n_show} Heads by Diff Norm (all pairs aggregated):")
    log_time(f"  {'Rank':>4} {'Layer':>5} {'Head':>5} {'Mean_Diff':>10} {'Std':>10} {'N':>6}")
    for rank, (li, hi, mean_diff, std_diff, n) in enumerate(ranking[:n_show]):
        log_time(f"  {rank+1:>4} L{li:>4} H{hi:>4} {mean_diff:10.3f} {std_diff:10.3f} {n:>6}")
    
    # Layer-wise total diff
    log_time(f"\n  Layer-wise Total Head Diff Norm:")
    log_time(f"  {'Layer':>5} {'Total_Diff':>12} {'Mean_Head_Diff':>12} {'Top3_Heads':>20}")
    for li in range(n_layers):
        layer_diffs = []
        for hi in range(n_heads):
            key = (li, hi)
            for r in ranking:
                if r[0] == li and r[1] == hi:
                    layer_diffs.append(r[2])
                    break
        if layer_diffs:
            total = sum(layer_diffs)
            mean_h = np.mean(layer_diffs)
            # Find top 3 heads
            top3 = sorted([(hi, d) for hi, d in enumerate(layer_diffs)], key=lambda x: -x[1])[:3]
            top3_str = ", ".join([f"H{h}={d:.2f}" for h, d in top3])
            log_time(f"  L{li:>4} {total:12.2f} {mean_h:12.3f} {top3_str:>20}")

# =============================================================================
# PART 3: Targeted Head Patching (Causal Verification)
# =============================================================================

def patch_single_head(model, tokenizer, sentence, device, n_layers,
                       cached_inputs, target_layer, target_head, head_dim,
                       common_len):
    """
    Forward sentence with ONE head at target_layer replaced by cached_inputs.
    Uses pre_hook on o_proj to modify the input before W_o projection.
    
    Args:
        cached_inputs: dict of {"L{li}": tensor[concat_dim]} from sentence A
        target_head: which head to replace
    
    Returns:
        last_logits or None if failed
    """
    layers = get_layers(model)
    if target_layer >= len(layers):
        return None
    
    lk = f"L{target_layer}"
    if lk not in cached_inputs:
        return None
    
    cached_vec = cached_inputs[lk]  # [concat_dim] on CPU
    
    # Tokenize
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=common_len).to(device)
    seq_len = inputs["input_ids"].shape[1]
    if seq_len < common_len:
        pad_len = common_len - seq_len
        inputs["input_ids"] = F.pad(inputs["input_ids"], (0, pad_len), value=tokenizer.pad_token_id)
        inputs["attention_mask"] = F.pad(inputs["attention_mask"], (0, pad_len), value=0)
    
    o_proj = layers[target_layer].self_attn.o_proj
    
    def pre_hook(module, args):
        # args[0]: [batch, seq, concat_dim]
        new_input = args[0].clone()
        s = target_head * head_dim
        e = (target_head + 1) * head_dim
        # cached_vec: [concat_dim], need to broadcast to [batch, seq, concat_dim]
        new_input[:, :, s:e] = cached_vec[s:e].to(new_input.device, new_input.dtype).unsqueeze(0).unsqueeze(0)
        return (new_input,)
    
    hook_handle = o_proj.register_forward_pre_hook(pre_hook)
    
    with torch.no_grad():
        try:
            out = model(**inputs)
            result = out.logits[0, -1, :].detach().cpu().float().clone()
        except Exception as e:
            result = None
    
    hook_handle.remove()
    return result

def measure_kl_divergence(logits_a, logits_b):
    if logits_a is None or logits_b is None:
        return None
    log_p_a = F.log_softmax(logits_a, dim=-1)
    p_b = F.softmax(logits_b, dim=-1)
    return float(F.kl_div(log_p_a, p_b, reduction='sum'))

def run_head_patching(model, tokenizer, device, n_layers, n_heads, head_dim,
                       cached_all, selected_heads, patching_pairs, common_lens):
    """
    Patch selected (layer, head) pairs on patching_pairs.
    Measure causal effect via KL divergence.
    
    Args:
        selected_heads: list of (layer, head)
        patching_pairs: list of pair names
        common_lens: dict {pname: common_len}
    
    Returns:
        patching_results: {(pname, li, hi): effect_ratio}
    """
    patching_results = {}
    n_patches = len(selected_heads) * len(patching_pairs)
    t0 = time.time()
    n_done = 0
    n_errors = 0
    
    for idx, (li, hi) in enumerate(selected_heads):
        lk = f"L{li}"
        head_label = f"L{li}_H{hi}"
        
        for pname in patching_pairs:
            if pname not in cached_all:
                continue
            if lk not in cached_all[pname]["A"] or lk not in cached_all[pname]["B"]:
                continue
            
            pair_data = cached_all[pname]
            sent_b = [p for p in build_all_pairs() if p["name"] == pname]
            if not sent_b:
                continue
            sent_b = sent_b[0]["B"]
            
            common_len = common_lens.get(pname, 32)
            
            # Get baseline logits for A and B
            # We need to cache these separately — for now, we'll compute on the fly
            # Actually we need logits_A and logits_B for KL computation
            # Let's compute these first
            
            # ... This is getting complex. Let me redesign.
            # For efficiency, we compute baselines once per pair.
            pass  # Placeholder — actual implementation below
    
    return patching_results

# =============================================================================
# Main Pipeline
# =============================================================================

def run_phase286(model_name):
    global _log_file
    _log_file = str(TMP_DIR / f"phase286_{model_name}.txt")
    
    log_time(f"{'='*60}")
    log_time(f"Phase 286: Head-Level Real Forward Patching — {model_name}")
    log_time(f"{'='*60}")
    
    # Load model
    model, tokenizer, device, attn_impl = load_model_flash(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    head_dim, n_heads, concat_dim = get_head_dim_and_heads(model)
    log_time(f"Model: {model_info.model_class}, L={n_layers}, d={d_model}, "
             f"n_heads={n_heads}, head_dim={head_dim}, concat_dim={concat_dim}, "
             f"mlp_type={model_info.mlp_type}")
    
    # Build pairs
    all_pairs = build_all_pairs()
    cat_counts = defaultdict(int)
    for p in all_pairs:
        cat_counts[p["category"]] += 1
    log_time(f"Dataset: {len(all_pairs)} pairs, {len(cat_counts)} categories")
    for cat, cnt in sorted(cat_counts.items()):
        log_time(f"  {cat}: {cnt}")
    
    # Determine key layers for head patching (based on Phase 285 results)
    if n_layers <= 20:
        key_layers = list(range(0, n_layers, max(1, n_layers // 9)))
        if n_layers - 1 not in key_layers:
            key_layers.append(n_layers - 1)
        key_layers = sorted(set(key_layers))[:10]
    else:
        step = max(1, n_layers // 9)
        key_layers = list(range(0, n_layers, step))
        if n_layers - 1 not in key_layers:
            key_layers.append(n_layers - 1)
        key_layers = sorted(set(key_layers))[:10]
    
    # Ensure L0 is always included (Phase 285 showed L0 is often crucial)
    if 0 not in key_layers:
        key_layers = [0] + key_layers[:9]
    log_time(f"Key layers for head patching: {key_layers}")
    
    # Warmup
    log_time("Global warmup...")
    wu = tokenizer("warmup test for head patching", return_tensors="pt").to(device)
    with torch.no_grad():
        try:
            model(**wu)
        except:
            pass
    log_time(f"Warmup done, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # ==================================================================
    # PART 1: Cache o_proj inputs for ALL pairs
    # ==================================================================
    log_time(f"\n{'='*60}")
    log_time("PART 1: Caching o_proj inputs for all 121 pairs")
    log_time(f"{'='*60}")
    
    cached_all = cache_o_proj_inputs(model, tokenizer, all_pairs, device, n_layers, max_len=32)
    n_cached = len(cached_all)
    log_time(f"Cached {n_cached} pairs. GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # Save intermediate cache
    cache_path = RESULT_DIR / f"{model_name}_o_proj_cache.json"
    # Only save metadata, not the full tensors
    cache_meta = {
        "model": model_name,
        "n_pairs_cached": n_cached,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "concat_dim": concat_dim,
        "pairs": list(cached_all.keys()),
    }
    with open(cache_path, "w") as f:
        json.dump(cache_meta, f, indent=2)
    log_time(f"Cache metadata saved to {cache_path}")
    
    # ==================================================================
    # PART 2: Analyze head diffs
    # ==================================================================
    log_time(f"\n{'='*60}")
    log_time("PART 2: Head Diff Analysis")
    log_time(f"{'='*60}")
    
    diff_results = compute_head_diffs(cached_all, n_layers, n_heads, head_dim)
    print_head_diff_summary(diff_results, n_layers, n_heads, n_show=30)
    
    # Per-category top heads
    log_time(f"\n  Per-Category Top-5 Heads:")
    for cat in sorted(diff_results["per_category"].keys()):
        cat_heads = diff_results["per_category"][cat]
        top5 = sorted(cat_heads.items(), key=lambda x: -x[1]["mean"])[:5]
        top5_str = ", ".join([f"{h}={d['mean']:.3f}" for h, d in top5])
        log_time(f"    {cat:<18}: {top5_str}")
    
    # ==================================================================
    # PART 3: Targeted Head Patching (Causal Verification)
    # ==================================================================
    log_time(f"\n{'='*60}")
    log_time("PART 3: Targeted Head Patching (Causal Verification)")
    log_time(f"{'='*60}")
    
    # Select top heads to patch: top-8 heads per key layer
    selected_for_patch = []
    ranking = diff_results["global_ranking"]
    
    # Group by key layer and take top-5 per layer
    per_layer_heads = defaultdict(list)
    for li, hi, mean_diff, std_diff, n in ranking:
        if li in key_layers:
            per_layer_heads[li].append((hi, mean_diff, std_diff))
    
    for li in key_layers:
        top_heads = sorted(per_layer_heads.get(li, []), key=lambda x: -x[1])[:5]
        for hi, mean_diff, std_diff in top_heads:
            selected_for_patch.append((li, hi, mean_diff))
    
    selected_for_patch.sort(key=lambda x: -x[2])  # sort by mean diff desc
    # Limit total to 40 patches
    selected_for_patch = selected_for_patch[:40]
    
    log_time(f"Selected {len(selected_for_patch)} (layer, head) pairs for causal patching:")
    for li, hi, md in selected_for_patch:
        log_time(f"  L{li:>3} H{hi:>3}  (diff_norm={md:.3f})")
    
    # Select representative pairs for patching: 1 per category
    patching_pair_names = []
    seen_cats = set()
    for p in all_pairs:
        if p["name"] in cached_all and p["category"] not in seen_cats:
            patching_pair_names.append(p["name"])
            seen_cats.add(p["category"])
    log_time(f"Patching pairs: {len(patching_pair_names)} (1 per category)")
    
    # Compute common lengths
    common_lens = {}
    for p in all_pairs:
        if p["name"] in patching_pair_names:
            toks_a = len(tokenizer.encode(p["A"], add_special_tokens=True))
            toks_b = len(tokenizer.encode(p["B"], add_special_tokens=True))
            common_lens[p["name"]] = min(max(toks_a, toks_b), 32)
    
    # First, cache baseline logits for all patching pairs
    log_time("Computing baseline logits for patching pairs...")
    pair_baselines = {}
    for pname in patching_pair_names:
        pair_data = cached_all[pname]
        # Find the pair in all_pairs
        pair_info = None
        for p in all_pairs:
            if p["name"] == pname:
                pair_info = p
                break
        if not pair_info:
            continue
        
        sent_a, sent_b = pair_info["A"], pair_info["B"]
        common_len = common_lens[pname]
        
        # Forward A
        inputs_a = tokenizer(sent_a, return_tensors="pt", truncation=True, max_length=common_len).to(device)
        with torch.no_grad():
            out_a = model(**inputs_a)
            logits_a = out_a.logits[0, -1, :].detach().cpu().float()
        
        # Forward B
        inputs_b = tokenizer(sent_b, return_tensors="pt", truncation=True, max_length=common_len).to(device)
        with torch.no_grad():
            out_b = model(**inputs_b)
            logits_b = out_b.logits[0, -1, :].detach().cpu().float()
        
        kl_ab = measure_kl_divergence(logits_a, logits_b)
        
        pair_baselines[pname] = {
            "logits_a": logits_a,
            "logits_b": logits_b,
            "kl_ab": kl_ab,
            "sent_b": sent_b,
        }
    
    log_time(f"Baselines computed for {len(pair_baselines)} pairs")
    
    # Now patch each selected head
    n_patches = len(selected_for_patch) * len(patching_pair_names)
    log_time(f"Total head patching forwards: {len(selected_for_patch)} heads × {len(patching_pair_names)} pairs = {n_patches}")
    
    head_patch_results = {}  # {(pname, li, hi): effect_ratio}
    t0 = time.time()
    n_done = 0
    n_errors = 0
    n_skip_kl0 = 0
    
    for (li, hi, _) in selected_for_patch:
        lk = f"L{li}"
        
        for pname in patching_pair_names:
            if pname not in pair_baselines:
                continue
            if pair_baselines[pname]["kl_ab"] is None or pair_baselines[pname]["kl_ab"] < 1e-8:
                n_skip_kl0 += 1
                continue
            
            bl = pair_baselines[pname]
            sent_b = bl["sent_b"]
            logits_b = bl["logits_b"]
            kl_ab = bl["kl_ab"]
            common_len = common_lens.get(pname, 32)
            
            # Patch: forward B with head replaced from A
            patched_logits = patch_single_head(
                model, tokenizer, sent_b, device, n_layers,
                cached_all[pname]["A"], li, hi, head_dim,
                common_len
            )
            
            if patched_logits is None:
                n_errors += 1
                continue
            
            kl_patched_b = measure_kl_divergence(patched_logits, logits_b)
            if kl_patched_b is not None and kl_ab > 1e-8:
                effect_ratio = min(kl_patched_b / kl_ab, 5.0)
            else:
                effect_ratio = None
            
            head_patch_results[(pname, li, hi)] = effect_ratio
            n_done += 1
        
        # Progress
        heads_done = sum(1 for k in head_patch_results if k[1] == li)
        elapsed = time.time() - t0
        log_time(f"  L{li:>3} H{hi:>3}: {heads_done}/{len(patching_pair_names)} pairs, "
                 f"{elapsed:.0f}s elapsed, errors={n_errors}")
    
    t_total = time.time() - t0
    log_time(f"\nPatching complete: {n_done} patches, {n_errors} errors, "
             f"{n_skip_kl0} skipped (KL=0), {t_total:.0f}s ({t_total/60:.1f}min)")
    
    # ==================================================================
    # PART 4: Aggregate head patching results
    # ==================================================================
    log_time(f"\n{'='*60}")
    log_time("PART 4: Head Causal Effect Aggregation")
    log_time(f"{'='*60}")
    
    # Aggregate per head across pairs
    head_causal = defaultdict(list)  # (li, hi) -> [effects]
    for (pname, li, hi), effect in head_patch_results.items():
        if effect is not None:
            head_causal[(li, hi)].append(effect)
    
    log_time(f"\n  Head Causal Effects (sorted by mean effect):")
    log_time(f"  {'Layer':>5} {'Head':>5} {'Mean_Effect':>12} {'Std':>10} {'N':>6} {'Interpret':>14}")
    head_causal_sorted = sorted(head_causal.items(), key=lambda x: -float(np.mean(x[1])))
    for (li, hi), effects in head_causal_sorted[:30]:
        mean_eff = float(np.mean(effects))
        std_eff = float(np.std(effects))
        if mean_eff < 0.05:
            interp = "NEGLIGIBLE"
        elif mean_eff < 0.15:
            interp = "WEAK"
        elif mean_eff < 0.30:
            interp = "MODERATE"
        elif mean_eff < 0.50:
            interp = "STRONG"
        else:
            interp = "DOMINANT"
        log_time(f"  L{li:>4} H{hi:>4} {mean_eff:12.4f} {std_eff:10.4f} {len(effects):>6} {interp:>14}")
    
    # Per-category head aggregation
    head_causal_per_cat = defaultdict(lambda: defaultdict(list))
    for (pname, li, hi), effect in head_patch_results.items():
        if effect is not None:
            pair_data = cached_all.get(pname, {})
            cat = pair_data.get("category", "unknown")
            head_causal_per_cat[cat][(li, hi)].append(effect)
    
    log_time(f"\n  Per-Category Top-3 Causal Heads:")
    for cat in sorted(head_causal_per_cat.keys()):
        cat_heads = head_causal_per_cat[cat]
        top3 = sorted(cat_heads.items(), key=lambda x: -float(np.mean(x[1])))[:3]
        top3_str = ", ".join([f"L{li}H{hi}={float(np.mean(eff)):.4f}" for (li, hi), eff in top3])
        log_time(f"    {cat:<18}: {top3_str}")
    
    # Head reuse analysis: cosine similarity between function head profiles
    log_time(f"\n  Head Reuse/Differentiation Matrix:")
    cat_profiles = {}
    for cat in sorted(head_causal_per_cat.keys()):
        profile = np.zeros((n_layers, n_heads))
        for (li, hi), effects in head_causal_per_cat[cat].items():
            profile[li, hi] = float(np.mean(effects))
        cat_profiles[cat] = profile
    
    cats_sorted = sorted(cat_profiles.keys())
    # Cross-category cosine similarity
    header = f"  {'Category':>18} "
    for cat in cats_sorted:
        header += f"{cat[:8]:>8} "
    log_time(header)
    
    for cat_a in cats_sorted:
        line = f"  {cat_a:>18} "
        for cat_b in cats_sorted:
            pa = cat_profiles[cat_a].flatten()
            pb = cat_profiles[cat_b].flatten()
            n_a = np.linalg.norm(pa)
            n_b = np.linalg.norm(pb)
            if n_a > 1e-10 and n_b > 1e-10:
                cos = float(np.dot(pa, pb) / (n_a * n_b))
            else:
                cos = 0.0
            line += f"{cos:8.3f} "
        log_time(line)
    
    # ==================================================================
    # Save results
    # ==================================================================
    # Save diff analysis (lightweight)
    diff_save = {
        "model": model_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "head_dim": head_dim,
        "concat_dim": concat_dim,
        "n_pairs_cached": n_cached,
        "global_ranking": [(li, hi, md, sd, n) for li, hi, md, sd, n in diff_results["global_ranking"][:100]],
        "per_category": diff_results["per_category"],
    }
    with open(RESULT_DIR / f"{model_name}_head_diff.json", "w") as f:
        json.dump(diff_save, f, indent=2)
    
    # Save patching results
    head_causal_save = {
        "model": model_name,
        "n_pairs_patched": len(patching_pair_names),
        "n_heads_patched": len(selected_for_patch),
        "head_causal_effects": {
            f"L{li}_H{hi}": {
                "mean": float(np.mean(effects)),
                "std": float(np.std(effects)),
                "n": len(effects),
            }
            for (li, hi), effects in head_causal.items()
        },
        "per_category_top_heads": {
            cat: {
                f"L{li}_H{hi}": float(np.mean(effects))
                for (li, hi), effects in sorted(cat_heads.items(), key=lambda x: -float(np.mean(x[1])))[:10]
            }
            for cat, cat_heads in head_causal_per_cat.items()
        },
    }
    with open(RESULT_DIR / f"{model_name}_head_patching.json", "w") as f:
        json.dump(head_causal_save, f, indent=2)
    
    log_time(f"\nResults saved to {RESULT_DIR}/")
    
    # Cleanup
    release_model(model)
    gc.collect()
    torch.cuda.empty_cache()
    
    return {
        "diff_results": diff_results,
        "head_patch_results": head_patch_results,
        "head_causal": head_causal,
        "head_causal_per_cat": head_causal_per_cat,
    }


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            try:
                r = run_phase286(name)
                log_time(f"\n{name} DONE: {len(r['head_causal'])} heads tested")
            except Exception as e:
                log_time(f"!!! {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(3)
    else:
        run_phase286(model_name)
