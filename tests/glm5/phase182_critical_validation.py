"""
Phase 182: Critical Validation — 硬伤验证与因果干预 (优化版)
================================================================
Exp1: 只计算2个约束token的logit差值 (省内存, 不做全词表entropy)
Exp2: Running mean + float32 + 及时释放
Exp3/4: float32 + 每对处理后gc.collect()

Usage: python tests/glm5/phase182_critical_validation.py <model_name>
"""

import sys, os, time, json, gc
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'glm5'))
from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS

def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[Phase182] Loading {model_name} (bfloat16 + device_map=auto)...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager")
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[Phase182] {model_name} loaded: device={device}, class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device

GRAMMAR_PAIRS = [
    ("The cat sleeps quietly", "The cat sleep quietly", "sleeps", "sleep"),
    ("The dog runs fast", "The dog run fast", "runs", "run"),
    ("A bird sings beautifully", "A bird sing beautifully", "sings", "sing"),
    ("The child plays outside", "The child play outside", "plays", "play"),
    ("My sister reads books", "My sister read books", "reads", "read"),
    ("The sun shines bright", "The sun shine bright", "shines", "shine"),
    ("A flower grows slowly", "A flower grow slowly", "grows", "grow"),
    ("The river flows south", "The river flow south", "flows", "flow"),
    ("His mother cooks dinner", "His mother cook dinner", "cooks", "cook"),
    ("The student writes well", "The student write well", "writes", "write"),
    ("She walks to school", "She walk to school", "walks", "walk"),
    ("The horse gallops away", "The horse gallop away", "gallops", "gallop"),
    ("A fish swims upstream", "A fish swim upstream", "swims", "swim"),
    ("The wind blows hard", "The wind blow hard", "blows", "blow"),
    ("The rabbit hops around", "The rabbit hop around", "hops", "hop"),
    ("He drives carefully", "He drive carefully", "drives", "drive"),
    ("The girl dances well", "The girl dance well", "dances", "dance"),
    ("A snake slithers slowly", "A snake slither slowly", "slithers", "slither"),
    ("The boat sails fast", "The boat sail fast", "sails", "sail"),
    ("The plane flies high", "The plane fly high", "flies", "fly"),
    ("The teacher speaks clearly", "The teacher speak clearly", "speaks", "speak"),
    ("The doctor works hard", "The doctor work hard", "works", "work"),
    ("The cat jumps high", "The cat jump high", "jumps", "jump"),
    ("The baby cries loud", "The baby cry loud", "cries", "cry"),
    ("The rain falls gently", "The rain fall gently", "falls", "fall"),
    ("The cats sleep quietly", "The cats sleeps quietly", "sleep", "sleeps"),
    ("The dogs run fast", "The dogs runs fast", "run", "runs"),
    ("The birds sing loudly", "The birds sings loudly", "sing", "sings"),
    ("The children play outside", "The children plays outside", "play", "plays"),
]
ANIMACY_PAIRS = [
    ("The dog thought carefully", "The rock thought carefully", "thought", "thought"),
    ("The man remembered everything", "The wall remembered everything", "remembered", "remembered"),
    ("The woman decided quickly", "The table decided quickly", "decided", "decided"),
    ("The child believed the story", "The stone believed the story", "believed", "believed"),
    ("The student understood the lesson", "The desk understood the lesson", "understood", "understood"),
    ("The teacher explained clearly", "The chair explained clearly", "explained", "explained"),
    ("The scientist discovered truth", "The mountain discovered truth", "discovered", "discovered"),
    ("The doctor helped patients", "The building helped patients", "helped", "helped"),
    ("The cat noticed the bird", "The box noticed the bird", "noticed", "noticed"),
    ("The girl imagined a world", "The rock imagined a world", "imagined", "imagined"),
    ("The boy felt happy", "The wall felt happy", "felt", "felt"),
    ("The woman wanted peace", "The table wanted peace", "wanted", "wanted"),
    ("The man feared the dark", "The stone feared the dark", "feared", "feared"),
    ("The child loved animals", "The chair loved animals", "loved", "loved"),
    ("The doctor knew the answer", "The desk knew the answer", "knew", "knew"),
    ("The teacher hoped for rain", "The mountain hoped for rain", "hoped", "hoped"),
    ("The singer sang beautifully", "The rock sang beautifully", "sang", "sang"),
    ("The writer wrote novels", "The table wrote novels", "wrote", "wrote"),
    ("The chef cooked dinner", "The wall cooked dinner", "cooked", "cooked"),
    ("The pilot flew planes", "The stone flew planes", "flew", "flew"),
]
PHYSICS_PAIRS = [
    ("The glass broke into pieces", "The glass melted into pieces", "broke", "melted"),
    ("The ice melted in the sun", "The ice floated in the sun", "melted", "floated"),
    ("The wood floated on water", "The wood sank on water", "floated", "sank"),
    ("The stone sank in the lake", "The stone evaporated in the lake", "sank", "evaporated"),
    ("The metal bent under pressure", "The metal shattered under pressure", "bent", "shattered"),
    ("The rubber stretched easily", "The rubber shattered easily", "stretched", "shattered"),
    ("The paper tore cleanly", "The paper melted cleanly", "tore", "melted"),
    ("The water evaporated quickly", "The water broke quickly", "evaporated", "broke"),
    ("The ball bounced high", "The ball sank high", "bounced", "sank"),
    ("The rope snapped suddenly", "The rope evaporated suddenly", "snapped", "evaporated"),
    ("The candle burned brightly", "The candle floated brightly", "burned", "floated"),
    ("The iron rusted slowly", "The iron melted slowly", "rusted", "melted"),
    ("The snow melted fast", "The snow shattered fast", "melted", "shattered"),
    ("The window cracked loudly", "The window evaporated loudly", "cracked", "evaporated"),
    ("The fabric tore easily", "The fabric sank easily", "tore", "sank"),
]
CONTROL_PAIRS = [
    ("The cat sleeps quietly", "The dog sleeps quietly", "sleeps", "sleeps"),
    ("The bird sings loudly", "The girl sings loudly", "sings", "sings"),
    ("The man walks slowly", "The woman walks slowly", "walks", "walks"),
    ("The red car drives fast", "The blue car drives fast", "drives", "drives"),
    ("The tall tree grows tall", "The small tree grows tall", "grows", "grows"),
    ("A dog runs in the park", "A cat runs in the park", "runs", "runs"),
    ("The old man reads a book", "The young man reads a book", "reads", "reads"),
    ("The warm coffee smells good", "The cold coffee smells good", "smells", "smells"),
    ("His sister works at school", "His brother works at school", "works", "works"),
    ("The heavy box sits there", "The light box sits there", "sits", "sits"),
]
TUNED_LENS_TRAIN_SENTENCES = [
    "The weather is nice today", "She went to the store yesterday",
    "A large dog was running in the park", "The computer screen displays text clearly",
    "Mountains are covered with snow in winter", "He enjoys reading science fiction books",
    "The restaurant serves delicious Italian food", "Students study hard before final exams",
    "The city has many tall buildings", "Music brings people together",
    "The ocean waves crash against the shore", "Scientists conduct experiments in laboratories",
    "The train arrives at noon", "Children play games after school",
    "The garden has beautiful flowers", "She writes letters to her friends",
    "The library contains thousands of books", "Birds build nests in the trees",
    "The river flows through the valley", "He plays the guitar every evening",
    "The movie starts at eight o clock", "Doctors recommend regular exercise",
    "The cake tastes very sweet", "Rain falls from the clouds",
    "The airplane flies above the clouds", "She paints pictures of landscapes",
    "The hotel provides excellent service", "Farmers grow crops in the field",
    "The clock ticks quietly on the wall", "Dolphins swim in the ocean",
    "The bridge crosses over the river", "He speaks three languages fluently",
    "The museum displays ancient artifacts", "The sun sets in the west",
    "She teaches mathematics at the university", "The forest is full of wildlife",
    "They built a house near the lake", "The phone rings loudly",
    "He drives to work every morning", "The book describes historical events",
    "The concert begins at seven", "She prepares dinner for the family",
    "The school organizes annual events", "Wind blows leaves from the trees",
    "He watches television after dinner", "The shop sells fresh vegetables",
    "She listens to classical music", "The baby sleeps in the crib",
    "The river flows toward the sea",
]

def find_differentiating_position(tokenizer, sent_correct, sent_incorrect):
    ids_c = tokenizer.encode(sent_correct, add_special_tokens=True)
    ids_i = tokenizer.encode(sent_incorrect, add_special_tokens=True)
    min_len = min(len(ids_c), len(ids_i))
    for pos in range(min_len):
        if ids_c[pos] != ids_i[pos]:
            return pos
    return min_len - 1

def force_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_hidden_at_pos(model, tokenizer, device, sentence, target_pos, n_layers):
    """获取句子在目标位置所有层的hidden states, 返回 dict{li: np.array[d_model]}"""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        out = model(input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    output_hidden_states=True)
    pos = min(target_pos, out.hidden_states[0].shape[1] - 1)
    result = {}
    for li, hs in enumerate(out.hidden_states):
        result[li] = hs[0, pos].detach().cpu().float().numpy()
    del out
    return result

# =====================================================================
# EXP1: NORM & 2-TOKEN MARGIN (只算2个token, 不做全词表entropy)
# =====================================================================
def exp1_norm_diagnostics(model, tokenizer, device, W_U, n_layers, d_model):
    print("\n" + "="*60)
    print("Exp1: NORM & 2-TOKEN MARGIN (optimized: 2 tokens only)")
    print("="*60)
    W_U_f32 = W_U.astype(np.float32)
    all_results = {}
    
    for ctype, pairs in [("grammar", GRAMMAR_PAIRS), ("animacy", ANIMACY_PAIRS),
                          ("physics", PHYSICS_PAIRS), ("control", CONTROL_PAIRS)]:
        print(f"\n  [{ctype}] Processing {len(pairs)} pairs...")
        norms_c = defaultdict(list)
        norms_i = defaultdict(list)
        margin_raw = defaultdict(list)      # (margin_correct, margin_incorrect)
        margin_normed = defaultdict(list)
        
        for pi, pair in enumerate(pairs):
            if pi % 10 == 0:
                print(f"    Pair {pi+1}/{len(pairs)}", flush=True)
            sent_c, sent_i, tok_c_str, tok_i_str = pair
            target_pos = find_differentiating_position(tokenizer, sent_c, sent_i)
            tok_c_ids = tokenizer.encode(tok_c_str, add_special_tokens=False)
            tok_i_ids = tokenizer.encode(tok_i_str, add_special_tokens=False)
            
            hs_c = get_hidden_at_pos(model, tokenizer, device, sent_c, target_pos, n_layers)
            hs_i = get_hidden_at_pos(model, tokenizer, device, sent_i, target_pos, n_layers)
            
            for li in range(n_layers + 1):
                if li not in hs_c or li not in hs_i:
                    continue
                h_c = hs_c[li].astype(np.float32)
                h_i = hs_i[li].astype(np.float32)
                norm_c = float(np.linalg.norm(h_c))
                norm_i = float(np.linalg.norm(h_i))
                norms_c[li].append(norm_c)
                norms_i[li].append(norm_i)
                
                # ★ 只算2个token的margin
                if len(tok_c_ids) > 0 and len(tok_i_ids) > 0:
                    m_raw_c = float(W_U_f32[tok_c_ids[0]] @ h_c - W_U_f32[tok_i_ids[0]] @ h_c)
                    m_raw_i = float(W_U_f32[tok_c_ids[0]] @ h_i - W_U_f32[tok_i_ids[0]] @ h_i)
                else:
                    m_raw_c = m_raw_i = 0.0
                margin_raw[li].append((m_raw_c, m_raw_i))
                
                # Norm-corrected
                h_c_n = h_c / max(norm_c, 1e-10)
                h_i_n = h_i / max(norm_i, 1e-10)
                if len(tok_c_ids) > 0 and len(tok_i_ids) > 0:
                    m_norm_c = float(W_U_f32[tok_c_ids[0]] @ h_c_n - W_U_f32[tok_i_ids[0]] @ h_c_n)
                    m_norm_i = float(W_U_f32[tok_c_ids[0]] @ h_i_n - W_U_f32[tok_i_ids[0]] @ h_i_n)
                else:
                    m_norm_c = m_norm_i = 0.0
                margin_normed[li].append((m_norm_c, m_norm_i))
            
            del hs_c, hs_i
        
        result = {}
        for li in range(n_layers + 1):
            if li not in norms_c:
                continue
            result[li] = {
                "norm_correct_mean": float(np.mean(norms_c[li])),
                "norm_incorrect_mean": float(np.mean(norms_i[li])),
                "margin_raw_correct_mean": float(np.mean([v[0] for v in margin_raw[li]])),
                "margin_raw_incorrect_mean": float(np.mean([v[1] for v in margin_raw[li]])),
                "margin_normed_correct_mean": float(np.mean([v[0] for v in margin_normed[li]])),
                "margin_normed_incorrect_mean": float(np.mean([v[1] for v in margin_normed[li]])),
            }
        all_results[ctype] = result
        force_cleanup()
    
    del W_U_f32
    return all_results

# =====================================================================
# EXP2: TUNED LENS (running mean + float32)
# =====================================================================
def exp2_tuned_lens(model, tokenizer, device, W_U, n_layers, d_model, vocab_size):
    print("\n" + "="*60)
    print("Exp2: TUNED LENS (running mean + float32)")
    print("="*60)
    W_U_f32 = W_U.astype(np.float32)
    
    # Step 1: Train with running mean
    print("\n  [Training] Computing per-layer bias (running mean)...")
    bias_mean = {}
    bias_count = {}
    layer_norms = defaultdict(list)
    
    for si, sent in enumerate(TUNED_LENS_TRAIN_SENTENCES):
        if si % 10 == 0:
            print(f"    Sentence {si+1}/{len(TUNED_LENS_TRAIN_SENTENCES)}", flush=True)
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            out = model(input_ids=inputs["input_ids"].to(device),
                       attention_mask=inputs["attention_mask"].to(device),
                       output_hidden_states=True)
        final_logits = out.logits[0, -1].detach().cpu().float().numpy().astype(np.float32)
        
        for li in range(n_layers + 1):
            hs = out.hidden_states[li][0, -1].detach().cpu().float().numpy().astype(np.float32)
            layer_norms[li].append(float(np.linalg.norm(hs)))
            raw_logits = W_U_f32 @ hs
            residual = final_logits - raw_logits
            if li not in bias_mean:
                bias_mean[li] = residual.copy()
                bias_count[li] = 1
            else:
                bias_count[li] += 1
                delta = residual - bias_mean[li]
                bias_mean[li] += delta / bias_count[li]
        
        del out, final_logits
        force_cleanup()
    
    for li in range(n_layers + 1):
        print(f"    Layer {li}: bias_norm={float(np.linalg.norm(bias_mean[li])):.2f}, "
              f"mean_norm={float(np.mean(layer_norms[li])):.2f}")
    
    # Step 2: Evaluate on constraint pairs (2-token margin only)
    print("\n  [Evaluation] 2-token margin on constraint pairs...")
    all_results = {}
    
    for ctype, pairs in [("grammar", GRAMMAR_PAIRS), ("animacy", ANIMACY_PAIRS),
                          ("physics", PHYSICS_PAIRS), ("control", CONTROL_PAIRS)]:
        margin_raw = defaultdict(list)
        margin_tuned = defaultdict(list)
        
        for pi, pair in enumerate(pairs):
            if pi % 10 == 0:
                print(f"    [{ctype}] Pair {pi+1}/{len(pairs)}", flush=True)
            sent_c, sent_i, tok_c_str, tok_i_str = pair
            target_pos = find_differentiating_position(tokenizer, sent_c, sent_i)
            tok_c_ids = tokenizer.encode(tok_c_str, add_special_tokens=False)
            tok_i_ids = tokenizer.encode(tok_i_str, add_special_tokens=False)
            
            hs_c = get_hidden_at_pos(model, tokenizer, device, sent_c, target_pos, n_layers)
            hs_i = get_hidden_at_pos(model, tokenizer, device, sent_i, target_pos, n_layers)
            
            for li in range(n_layers + 1):
                if li not in hs_c or li not in hs_i:
                    continue
                h_c = hs_c[li].astype(np.float32)
                h_i = hs_i[li].astype(np.float32)
                
                if len(tok_c_ids) > 0 and len(tok_i_ids) > 0:
                    raw_c = float(W_U_f32[tok_c_ids[0]] @ h_c - W_U_f32[tok_i_ids[0]] @ h_c)
                    raw_i = float(W_U_f32[tok_c_ids[0]] @ h_i - W_U_f32[tok_i_ids[0]] @ h_i)
                    bias_diff = float(bias_mean[li][tok_c_ids[0]] - bias_mean[li][tok_i_ids[0]])
                    tuned_c = raw_c + bias_diff
                    tuned_i = raw_i + bias_diff
                else:
                    raw_c = raw_i = tuned_c = tuned_i = 0.0
                
                margin_raw[li].append((raw_c, raw_i))
                margin_tuned[li].append((tuned_c, tuned_i))
            
            del hs_c, hs_i
        
        result = {}
        for li in range(n_layers + 1):
            if li not in margin_raw:
                continue
            raw_vals = margin_raw[li]
            tuned_vals = margin_tuned[li]
            result[li] = {
                "margin_raw_correct_mean": float(np.mean([v[0] for v in raw_vals])),
                "margin_raw_incorrect_mean": float(np.mean([v[1] for v in raw_vals])),
                "margin_tuned_correct_mean": float(np.mean([v[0] for v in tuned_vals])),
                "margin_tuned_incorrect_mean": float(np.mean([v[1] for v in tuned_vals])),
                "margin_raw_gap": float(np.mean([v[0]-v[1] for v in raw_vals])),
                "margin_tuned_gap": float(np.mean([v[0]-v[1] for v in tuned_vals])),
            }
        all_results[ctype] = result
        force_cleanup()
    
    all_results["_tuned_bias_norms"] = {str(li): float(np.linalg.norm(bias_mean[li])) for li in bias_mean}
    del W_U_f32, bias_mean
    return all_results

# =====================================================================
# EXP3: W_U-FREE DIRECT PROBING (float32)
# =====================================================================
def exp3_wu_free_probing(model, tokenizer, device, n_layers, d_model):
    print("\n" + "="*60)
    print("Exp3: W_U-FREE DIRECT PROBING (float32)")
    print("="*60)
    all_results = {}
    
    for ctype, pairs in [("grammar", GRAMMAR_PAIRS), ("animacy", ANIMACY_PAIRS),
                          ("physics", PHYSICS_PAIRS), ("control", CONTROL_PAIRS)]:
        print(f"\n  [{ctype}] Processing {len(pairs)} pairs...")
        delta_norms = defaultdict(list)
        delta_relative = defaultdict(list)
        delta_cos_next = defaultdict(list)
        delta_cos_final = defaultdict(list)
        h_norms_c = defaultdict(list)
        h_norms_i = defaultdict(list)
        
        for pi, pair in enumerate(pairs):
            if pi % 10 == 0:
                print(f"    Pair {pi+1}/{len(pairs)}", flush=True)
            sent_c, sent_i = pair[0], pair[1]
            target_pos = find_differentiating_position(tokenizer, sent_c, sent_i)
            hs_c = get_hidden_at_pos(model, tokenizer, device, sent_c, target_pos, n_layers)
            hs_i = get_hidden_at_pos(model, tokenizer, device, sent_i, target_pos, n_layers)
            
            deltas = {}
            for li in range(n_layers + 1):
                if li in hs_c and li in hs_i:
                    deltas[li] = (hs_c[li] - hs_i[li]).astype(np.float32)
                    h_norms_c[li].append(float(np.linalg.norm(hs_c[li])))
                    h_norms_i[li].append(float(np.linalg.norm(hs_i[li])))
            
            final_li = n_layers
            for li in sorted(deltas.keys()):
                d = deltas[li]
                dn = float(np.linalg.norm(d))
                hn = (h_norms_c[li][-1] + h_norms_i[li][-1]) / 2
                delta_norms[li].append(dn)
                delta_relative[li].append(dn / max(hn, 1e-10))
                if li+1 in deltas:
                    cn = float(np.dot(d, deltas[li+1]) / max(dn * np.linalg.norm(deltas[li+1]), 1e-10))
                    delta_cos_next[li].append(cn)
                if final_li in deltas and li != final_li:
                    cf = float(np.dot(d, deltas[final_li]) / max(dn * np.linalg.norm(deltas[final_li]), 1e-10))
                    delta_cos_final[li].append(cf)
            
            del hs_c, hs_i, deltas
            force_cleanup()
        
        result = {}
        for li in range(n_layers + 1):
            if li not in delta_norms:
                continue
            entry = {
                "delta_norm_mean": float(np.mean(delta_norms[li])),
                "delta_relative_mean": float(np.mean(delta_relative[li])),
                "h_norm_correct_mean": float(np.mean(h_norms_c[li])),
                "h_norm_incorrect_mean": float(np.mean(h_norms_i[li])),
            }
            if delta_cos_next[li]:
                entry["cos_delta_next_mean"] = float(np.mean(delta_cos_next[li]))
            if delta_cos_final[li]:
                entry["cos_delta_final_mean"] = float(np.mean(delta_cos_final[li]))
            result[li] = entry
        all_results[ctype] = result
    
    return all_results

# =====================================================================
# EXP4: ACTIVATION PATCHING (因果金标准)
# =====================================================================
def exp4_activation_patching(model, tokenizer, device, n_layers):
    print("\n" + "="*60)
    print("Exp4: ACTIVATION PATCHING (CAUSAL)")
    print("="*60)
    all_results = {}
    
    for ctype, pairs in [("grammar", GRAMMAR_PAIRS[:10]), ("animacy", ANIMACY_PAIRS[:10]),
                          ("physics", PHYSICS_PAIRS[:10]), ("control", CONTROL_PAIRS[:5])]:
        print(f"\n  [{ctype}] Processing {len(pairs)} pairs...")
        patch_effects = defaultdict(list)
        
        for pi, pair in enumerate(pairs):
            print(f"    Pair {pi+1}/{len(pairs)}", flush=True)
            sent_c, sent_i, tok_c, tok_i = pair
            tok_c_ids = tokenizer.encode(tok_c, add_special_tokens=False)
            tok_i_ids = tokenizer.encode(tok_i, add_special_tokens=False)
            target_pos = find_differentiating_position(tokenizer, sent_c, sent_i)
            
            # Correct sentence
            inputs_c = tokenizer(sent_c, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                out_c = model(input_ids=inputs_c["input_ids"].to(device),
                             attention_mask=inputs_c["attention_mask"].to(device),
                             output_hidden_states=True)
            correct_h = {}
            for li, hs in enumerate(out_c.hidden_states):
                pos_c = min(target_pos, hs.shape[1] - 1)
                correct_h[li] = hs[0, pos_c].detach().clone().float()
            if len(tok_c_ids) > 0 and len(tok_i_ids) > 0:
                correct_margin = float(out_c.logits[0, -1, tok_c_ids[0]] - out_c.logits[0, -1, tok_i_ids[0]])
            else:
                correct_margin = 0.0
            del out_c
            
            # Incorrect sentence (baseline)
            inputs_i = tokenizer(sent_i, return_tensors="pt", truncation=True, max_length=128)
            input_ids_i = inputs_i["input_ids"].to(device)
            attn_mask_i = inputs_i["attention_mask"].to(device)
            with torch.no_grad():
                out_i = model(input_ids=input_ids_i, attention_mask=attn_mask_i,
                             output_hidden_states=True)
            if len(tok_c_ids) > 0 and len(tok_i_ids) > 0:
                baseline_margin = float(out_i.logits[0, -1, tok_c_ids[0]] - out_i.logits[0, -1, tok_i_ids[0]])
            else:
                baseline_margin = 0.0
            del out_i
            
            # Patch each layer
            layers = get_layers(model)
            for patch_li in range(1, n_layers):
                hook_handle = None
                def make_hook(ch, idx):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            new_out = output[0].detach().clone()
                            p = min(target_pos, new_out.shape[1] - 1)
                            new_out[0, p] += ch[idx].to(new_out.device) - new_out[0, p]
                            return (new_out,) + output[1:]
                        return output
                    return hook_fn
                try:
                    hook_handle = layers[patch_li].register_forward_hook(make_hook(correct_h, patch_li))
                    with torch.no_grad():
                        out_p = model(input_ids=input_ids_i, attention_mask=attn_mask_i)
                    if len(tok_c_ids) > 0 and len(tok_i_ids) > 0:
                        patched_margin = float(out_p.logits[0, -1, tok_c_ids[0]] - out_p.logits[0, -1, tok_i_ids[0]])
                    else:
                        patched_margin = 0.0
                    del out_p
                    hook_handle.remove()
                    patch_effects[patch_li].append(patched_margin - baseline_margin)
                except Exception as e:
                    if hook_handle:
                        hook_handle.remove()
                    patch_effects[patch_li].append(0.0)
                finally:
                    if hook_handle:
                        try: hook_handle.remove()
                        except: pass
            
            del correct_h
            force_cleanup()
        
        result = {}
        for li in sorted(patch_effects.keys()):
            effects = patch_effects[li]
            result[li] = {
                "patch_effect_mean": float(np.mean(effects)),
                "patch_effect_std": float(np.std(effects)),
                "patch_effect_abs_mean": float(np.mean(np.abs(effects))),
                "n_pairs": len(effects),
            }
        all_results[ctype] = result
    
    return all_results

# =====================================================================
# MAIN
# =====================================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    print(f"\n{'#'*70}")
    print(f"# Phase 182: CRITICAL VALIDATION (optimized) — {model_name}")
    print(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers, d_model, vocab_size = info.n_layers, info.d_model, info.vocab_size
    print(f"\nModel: {info.model_class}, Layers={n_layers}, d_model={d_model}, vocab={vocab_size}")
    
    W_U = get_W_U(model, model_name).astype(np.float32)
    print(f"W_U shape: {W_U.shape}, dtype: {W_U.dtype}")
    
    # Exp1
    print(f"\n{'='*70}")
    print("Running Exp1: Norm & 2-Token Margin...")
    exp1_results = exp1_norm_diagnostics(model, tokenizer, device, W_U, n_layers, d_model)
    force_cleanup()
    
    # Exp2
    print(f"\n{'='*70}")
    print("Running Exp2: Tuned Lens...")
    exp2_results = exp2_tuned_lens(model, tokenizer, device, W_U, n_layers, d_model, vocab_size)
    force_cleanup()
    
    # Exp3
    print(f"\n{'='*70}")
    print("Running Exp3: W_U-Free Probing...")
    exp3_results = exp3_wu_free_probing(model, tokenizer, device, n_layers, d_model)
    force_cleanup()
    
    # Exp4
    print(f"\n{'='*70}")
    print("Running Exp4: Activation Patching...")
    exp4_results = exp4_activation_patching(model, tokenizer, device, n_layers)
    force_cleanup()
    
    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = f"tests/glm5_temp/phase182_{model_name}_{timestamp}.json"
    full_results = {
        "model": model_name, "n_layers": n_layers, "d_model": d_model, "vocab_size": vocab_size,
        "timestamp": timestamp,
        "optimization": "Exp1=2token_margin, Exp2=running_mean_float32, Exp3/4=float32+gc",
        "exp1_norm_2token_margin": {k: {str(kk): vv for kk, vv in v.items()} for k, v in exp1_results.items()},
        "exp2_tuned_lens": {k: {str(kk): vv for kk, vv in v.items()} for k, v in exp2_results.items()},
        "exp3_wu_free": {k: {str(kk): vv for kk, vv in v.items()} for k, v in exp3_results.items()},
        "exp4_activation_patching": {k: {str(kk): vv for kk, vv in v.items()} for k, v in exp4_results.items()},
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_path}")
    
    # ===== Print Summary =====
    print(f"\n{'#'*70}")
    print("PHASE 182 SUMMARY (optimized)")
    print(f"{'#'*70}")
    
    # Exp1 Summary
    print("\n★★★ Exp1: Norm & 2-Token Margin ★★★")
    for ctype in ["grammar", "animacy", "physics", "control"]:
        if ctype not in exp1_results: continue
        data = exp1_results[ctype]
        ls = sorted([int(k) for k in data.keys()])
        if len(ls) < 3: continue
        norms = [data[l]["norm_correct_mean"] for l in ls]
        mrc = [data[l]["margin_raw_correct_mean"] for l in ls]
        mri = [data[l]["margin_raw_incorrect_mean"] for l in ls]
        mnc = [data[l]["margin_normed_correct_mean"] for l in ls]
        mni = [data[l]["margin_normed_incorrect_mean"] for l in ls]
        raw_growth = mrc[-1] - mrc[0]
        normed_growth = mnc[-1] - mnc[0]
        print(f"\n  [{ctype}]")
        print(f"    ||h||: L0={norms[0]:.2f} -> L{ls[-1]}={norms[-1]:.2f} (growth={norms[-1]/max(norms[0],0.01):.1f}x)")
        print(f"    Raw margin(correct): L0={mrc[0]:.3f} -> L{ls[-1]}={mrc[-1]:.3f}")
        print(f"    Normed margin(correct): L0={mnc[0]:.3f} -> L{ls[-1]}={mnc[-1]:.3f}")
        print(f"    Raw growth={raw_growth:.3f}, Normed growth={normed_growth:.3f}")
        print(f"    ★ Margin growth is norm artifact? {'YES' if abs(raw_growth) > 0.5 and abs(normed_growth) < 0.3 else 'NEEDS ANALYSIS'}")
    
    # Exp2 Summary
    print("\n★★★ Exp2: Tuned Lens ★★★")
    for ctype in ["grammar", "animacy", "physics", "control"]:
        if ctype not in exp2_results or ctype.startswith("_"): continue
        data = exp2_results[ctype]
        ls = sorted([int(k) for k in data.keys()])
        if len(ls) < 3: continue
        mrc = [data[l]["margin_raw_correct_mean"] for l in ls]
        mtc = [data[l]["margin_tuned_correct_mean"] for l in ls]
        raw_gap = np.mean([data[l]["margin_raw_gap"] for l in ls])
        tuned_gap = np.mean([data[l]["margin_tuned_gap"] for l in ls])
        print(f"\n  [{ctype}]")
        print(f"    Raw margin(correct): L0={mrc[0]:.3f} -> L{ls[-1]}={mrc[-1]:.3f}")
        print(f"    Tuned margin(correct): L0={mtc[0]:.3f} -> L{ls[-1]}={mtc[-1]:.3f}")
        print(f"    Raw gap={raw_gap:.3f}, Tuned gap={tuned_gap:.3f}")
        print(f"    ★ Tuned lens changes pattern? {'YES - logit lens artifact confirmed' if abs(mrc[-1]-mrc[0]) > 0.5 and abs(mtc[-1]-mtc[0]) < 0.3 else 'NEEDS ANALYSIS'}")
    
    # Exp3 Summary
    print("\n★★★ Exp3: W_U-Free Direct Probing ★★★")
    for ctype in ["grammar", "animacy", "physics", "control"]:
        if ctype not in exp3_results: continue
        data = exp3_results[ctype]
        ls = sorted([int(k) for k in data.keys()])
        if len(ls) < 3: continue
        dn = [data[l]["delta_norm_mean"] for l in ls]
        dr = [data[l]["delta_relative_mean"] for l in ls]
        cos_vals = [data[l].get("cos_delta_next_mean", None) for l in ls]
        cos_vals = [c for c in cos_vals if c is not None]
        print(f"\n  [{ctype}]")
        print(f"    ||Delta||: L0={dn[0]:.4f} -> L{ls[-1]}={dn[-1]:.4f} (growth={dn[-1]/max(dn[0],0.001):.1f}x)")
        print(f"    Delta relative: L0={dr[0]:.6f} -> L{ls[-1]}={dr[-1]:.6f}")
        if cos_vals:
            print(f"    cos(Delta_l, Delta_{{l+1}}): mean={np.mean(cos_vals):.3f}, min={np.min(cos_vals):.3f}")
    
    # Exp4 Summary
    print("\n★★★ Exp4: Activation Patching (CAUSAL) ★★★")
    for ctype in ["grammar", "animacy", "physics", "control"]:
        if ctype not in exp4_results: continue
        data = exp4_results[ctype]
        ls = sorted([int(k) for k in data.keys()])
        effects_abs = [data[l]["patch_effect_abs_mean"] for l in ls]
        if not effects_abs: continue
        peak_layer = ls[np.argmax(effects_abs)]
        peak_data = data[peak_layer]
        top3 = sorted(ls, key=lambda l: data[l]["patch_effect_abs_mean"], reverse=True)[:3]
        print(f"\n  [{ctype}]")
        print(f"    Peak causal layer: L{peak_layer} (effect={peak_data['patch_effect_mean']:.4f}, |effect|={peak_data['patch_effect_abs_mean']:.4f})")
        top3_effects = [round(data[l]["patch_effect_mean"], 4) for l in top3]
        print(f"    Top 3 causal layers: {[f'L{l}' for l in top3]}")
        print(f"    Effects: {top3_effects}")
    
    release_model(model)
    print(f"\n{'#'*70}")
    print("Phase 182 COMPLETE!")
    print(f"{'#'*70}")

if __name__ == "__main__":
    main()
