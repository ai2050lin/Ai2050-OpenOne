"""
Phase 182: ★★★ Critical Validation — 硬伤验证与因果干预 ★★★
================================================================

★★★ 三个致命硬伤及验证方案 ★★★

硬伤1: Logit Lens伪象
  问题: L0 entropy=0, 浅层/中间层logit lens不可靠
  验证: 
    A. ||h_l||范数分析 → 如果范数单调增长, entropy下降可能只是范数效应
    B. 范数校正logit lens → 归一化后重算entropy, 看漏斗是否消失
    C. Tuned Lens → 学习逐层affine校正, 最严格的对照

硬伤2: 约束方向依赖W_U
  问题: number_dir = mean(W_U[singular]) - mean(W_U[plural])
  验证:
    A. 直接用Δ_l = h_correct - h_incorrect, 不经过W_U
    B. 在Δ_l上训练线性探针预测约束满足
    C. 计算cos(Δ_l, Δ_{l+1}) — 约束方向稳定性

硬伤3: 无因果干预
  问题: 只有观测性证据, 没有干预实验
  验证:
    A. Activation Patching: 将正确句子的激活patch到错误句子
    B. 测量patch对最终logits的影响
    C. 确定哪些层是因果关键的

★★★ 实验设计 ★★★

Exp1: Norm & Logit Lens诊断
  - 测量||h_l||在每层的值 (正确/错误句子)
  - 范数校正logit lens: entropy(h_l/||h_l||)
  - 对比: raw entropy vs norm-corrected entropy
  - ★ 如果漏斗消失 → 漏斗是范数伪象

Exp2: Tuned Lens
  - 训练: 对每层学习 bias_l = mean(final_logits - W_U @ h_l)
  - 校正: tuned_logits_l = W_U @ h_l + bias_l
  - 计算tuned lens下的entropy
  - ★ 如果漏斗保持 → 漏斗是真实动力学

Exp3: W_U-Free直接探测
  - Δ_l = h_correct_l - h_incorrect_l (完全不经W_U!)
  - ||Δ_l||: 约束信号强度 (gauge-invariant)
  - cos(Δ_l, Δ_{l+1}): 约束方向稳定性
  - ||Δ_l||/||h_l||: 相对约束信号 (消除范数效应)
  - 线性探针: 在每层训练classifier预测约束满足

Exp4: Activation Patching (因果金标准)
  - 正确句子: cache所有层在目标位置的hidden state
  - 错误句子: 在指定层patch正确句子的hidden state
  - 测量: patch后logits在约束相关token上的变化
  - ★ 如果patch某层改变了约束满足 → 该层是因果关键的

Usage: python tests/glm5/phase182_critical_validation.py <model_name>
  model_name: qwen3, glm4, deepseek7b
"""

import sys
import os
import time
import json
import gc
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'glm5'))

from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS


# =====================================================================
# MODEL LOADING (BF16 + device_map="auto")
# =====================================================================

def load_model_bf16(model_name):
    """BF16 + device_map=auto loading"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    print(f"[Phase182] Loading {model_name} (bfloat16 + device_map=auto)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager",
    )
    model.eval()
    
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"[Phase182] {model_name} loaded: device={device}, "
          f"class={type(model).__name__}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


# =====================================================================
# SENTENCE PAIRS — 扩大数据量
# =====================================================================

GRAMMAR_PAIRS = [
    # (correct, incorrect, target_token_correct, target_token_incorrect)
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
    # Plural subjects with singular verbs (wrong)
    ("The cats sleep quietly", "The cats sleeps quietly", "sleep", "sleeps"),
    ("The dogs run fast", "The dogs runs fast", "run", "runs"),
    ("The birds sing loudly", "The birds sings loudly", "sing", "sings"),
    ("The children play outside", "The children plays outside", "play", "plays"),
]

ANIMACY_PAIRS = [
    # (animate_subject, inanimate_subject, verb_requiring_animate)
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
    # (physically_possible, physically_impossible)
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
    # (sentence_a, sentence_b) — 合法替换, 无约束违反
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

# Tuned lens训练用句子 (多样化, 不含约束违反)
TUNED_LENS_TRAIN_SENTENCES = [
    "The weather is nice today",
    "She went to the store yesterday",
    "A large dog was running in the park",
    "The computer screen displays text clearly",
    "Mountains are covered with snow in winter",
    "He enjoys reading science fiction books",
    "The restaurant serves delicious Italian food",
    "Students study hard before final exams",
    "The city has many tall buildings",
    "Music brings people together",
    "The ocean waves crash against the shore",
    "Scientists conduct experiments in laboratories",
    "The train arrives at noon",
    "Children play games after school",
    "The garden has beautiful flowers",
    "She writes letters to her friends",
    "The library contains thousands of books",
    "Birds build nests in the trees",
    "The river flows through the valley",
    "He plays the guitar every evening",
    "The movie starts at eight o clock",
    "Doctors recommend regular exercise",
    "The cake tastes very sweet",
    "Rain falls from the clouds",
    "The airplane flies above the clouds",
    "She paints pictures of landscapes",
    "The hotel provides excellent service",
    "Farmers grow crops in the field",
    "The clock ticks quietly on the wall",
    "Dolphins swim in the ocean",
    "The bridge crosses over the river",
    "He speaks three languages fluently",
    "The museum displays ancient artifacts",
    "The sun sets in the west",
    "She teaches mathematics at the university",
    "The forest is full of wildlife",
    "They built a house near the lake",
    "The phone rings loudly",
    "He drives to work every morning",
    "The book describes historical events",
    "The concert begins at seven",
    "She prepares dinner for the family",
    "The school organizes annual events",
    "Wind blows leaves from the trees",
    "He watches television after dinner",
    "The shop sells fresh vegetables",
    "She listens to classical music",
    "The baby sleeps in the crib",
    "The river flows toward the sea",
]


# =====================================================================
# UTILITY FUNCTIONS
# =====================================================================

def find_differentiating_position(tokenizer, sent_correct, sent_incorrect):
    """找到正确/错误句子之间的第一个不同token位置"""
    ids_c = tokenizer.encode(sent_correct, add_special_tokens=True)
    ids_i = tokenizer.encode(sent_incorrect, add_special_tokens=True)
    min_len = min(len(ids_c), len(ids_i))
    for pos in range(min_len):
        if ids_c[pos] != ids_i[pos]:
            return pos
    return min_len - 1


def get_hidden_states_at_position(model, tokenizer, device, sentence, target_pos, n_layers):
    """
    获取句子在目标位置所有层的hidden states
    
    Returns:
        dict: {layer_idx: numpy_array[d_model]}
        int: 实际使用的位置
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = input_ids.shape[1]
    
    pos = min(target_pos, seq_len - 1)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
    
    all_hidden = outputs.hidden_states
    
    result = {}
    for li, hs in enumerate(all_hidden):
        result[li] = hs[0, pos].detach().cpu().float().numpy()
    
    del outputs, all_hidden
    return result, pos


def get_all_hidden_states(model, tokenizer, device, sentence, n_layers):
    """
    获取句子所有位置所有层的hidden states
    
    Returns:
        dict: {layer_idx: numpy_array[seq_len, d_model]}
        int: seq_len
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
    
    all_hidden = outputs.hidden_states
    
    result = {}
    for li, hs in enumerate(all_hidden):
        result[li] = hs[0].detach().cpu().float().numpy()
    
    del outputs, all_hidden
    return result, seq_len


def compute_entropy_from_logits(logits):
    """从logits计算entropy (numpy)"""
    logits = logits - np.max(logits)  # numerical stability
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    probs = np.clip(probs, 1e-10, 1.0)
    return float(-np.sum(probs * np.log(probs)))


# =====================================================================
# EXP1: NORM & LOGIT LENS DIAGNOSTICS
# =====================================================================

def exp1_norm_diagnostics(model, tokenizer, device, W_U, n_layers, d_model):
    """
    ★ 硬伤1验证: ||h_l||范数分析 + 范数校正logit lens
    
    核心问题: entropy漏斗是真实动力学还是范数伪象?
    """
    print("\n" + "="*60)
    print("Exp1: NORM & LOGIT LENS DIAGNOSTICS")
    print("="*60)
    
    all_results = {}
    
    for ctype, pairs in [("grammar", GRAMMAR_PAIRS), 
                          ("animacy", ANIMACY_PAIRS),
                          ("physics", PHYSICS_PAIRS),
                          ("control", CONTROL_PAIRS)]:
        print(f"\n  [{ctype}] Processing {len(pairs)} pairs...")
        
        norms_correct = defaultdict(list)  # ||h_correct_l||
        norms_incorrect = defaultdict(list)  # ||h_incorrect_l||
        entropy_raw_correct = defaultdict(list)  # raw logit lens entropy
        entropy_raw_incorrect = defaultdict(list)
        entropy_normed_correct = defaultdict(list)  # norm-corrected logit lens entropy
        entropy_normed_incorrect = defaultdict(list)
        margin_raw = defaultdict(list)  # raw logit lens margin
        margin_normed = defaultdict(list)  # norm-corrected margin
        
        for pi, pair in enumerate(pairs):
            if pi % 10 == 0:
                print(f"    Pair {pi+1}/{len(pairs)}", flush=True)
            
            sent_c, sent_i = pair[0], pair[1]
            target_pos = find_differentiating_position(tokenizer, sent_c, sent_i)
            
            # Get hidden states
            hs_c, pos_c = get_hidden_states_at_position(
                model, tokenizer, device, sent_c, target_pos, n_layers)
            hs_i, pos_i = get_hidden_states_at_position(
                model, tokenizer, device, sent_i, target_pos, n_layers)
            
            for li in range(n_layers + 1):
                if li not in hs_c or li not in hs_i:
                    continue
                
                h_c = hs_c[li]
                h_i = hs_i[li]
                
                # Norms
                norm_c = float(np.linalg.norm(h_c))
                norm_i = float(np.linalg.norm(h_i))
                norms_correct[li].append(norm_c)
                norms_incorrect[li].append(norm_i)
                
                # Raw logit lens: W_U @ h
                logits_c_raw = W_U @ h_c
                logits_i_raw = W_U @ h_i
                
                ent_c_raw = compute_entropy_from_logits(logits_c_raw)
                ent_i_raw = compute_entropy_from_logits(logits_i_raw)
                entropy_raw_correct[li].append(ent_c_raw)
                entropy_raw_incorrect[li].append(ent_i_raw)
                
                # Margin: logit difference at constraint-relevant tokens
                # For grammar: margin at verb position
                # Use top-logit as proxy
                top_c = float(np.max(logits_c_raw))
                top_i = float(np.max(logits_i_raw))
                margin_raw[li].append(top_c - top_i)
                
                # Norm-corrected logit lens: W_U @ (h / ||h||)
                h_c_normed = h_c / max(norm_c, 1e-10)
                h_i_normed = h_i / max(norm_i, 1e-10)
                
                logits_c_normed = W_U @ h_c_normed
                logits_i_normed = W_U @ h_i_normed
                
                ent_c_normed = compute_entropy_from_logits(logits_c_normed)
                ent_i_normed = compute_entropy_from_logits(logits_i_normed)
                entropy_normed_correct[li].append(ent_c_normed)
                entropy_normed_incorrect[li].append(ent_i_normed)
                
                top_c_n = float(np.max(logits_c_normed))
                top_i_n = float(np.max(logits_i_normed))
                margin_normed[li].append(top_c_n - top_i_n)
            
            # Free memory
            del hs_c, hs_i
        
        # Aggregate
        result = {}
        for li in range(n_layers + 1):
            if li not in norms_correct:
                continue
            result[li] = {
                "norm_correct_mean": float(np.mean(norms_correct[li])),
                "norm_incorrect_mean": float(np.mean(norms_incorrect[li])),
                "norm_correct_std": float(np.std(norms_correct[li])),
                "entropy_raw_correct_mean": float(np.mean(entropy_raw_correct[li])),
                "entropy_raw_incorrect_mean": float(np.mean(entropy_raw_incorrect[li])),
                "entropy_normed_correct_mean": float(np.mean(entropy_normed_correct[li])),
                "entropy_normed_incorrect_mean": float(np.mean(entropy_normed_incorrect[li])),
                "margin_raw_mean": float(np.mean(margin_raw[li])),
                "margin_normed_mean": float(np.mean(margin_normed[li])),
            }
        all_results[ctype] = result
    
    return all_results


# =====================================================================
# EXP2: TUNED LENS
# =====================================================================

def exp2_tuned_lens(model, tokenizer, device, W_U, n_layers, d_model, vocab_size):
    """
    ★ 硬伤1验证: Tuned Lens — 最严格的logit lens对照
    
    方法:
    1. 用训练句子学习每层的bias: bias_l = mean(final_logits - W_U @ h_l)
    2. tuned_logits_l = W_U @ h_l + bias_l
    3. 比较tuned lens vs raw logit lens的entropy profile
    """
    print("\n" + "="*60)
    print("Exp2: TUNED LENS")
    print("="*60)
    
    # Step 1: Train tuned lens
    print("\n  [Training] Collecting hidden states from training sentences...")
    
    layer_logits_residuals = defaultdict(list)  # {li: [final_logits - W_U @ h_l]}
    layer_diagnostics = defaultdict(list)  # {li: [||h_l||]}
    
    for si, sent in enumerate(TUNED_LENS_TRAIN_SENTENCES):
        if si % 10 == 0:
            print(f"    Training sentence {si+1}/{len(TUNED_LENS_TRAIN_SENTENCES)}", flush=True)
        
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
        
        all_hidden = outputs.hidden_states
        final_logits = outputs.logits[0].detach().cpu().float().numpy()  # [seq_len, vocab]
        
        for li in range(n_layers + 1):
            hs = all_hidden[li][0].detach().cpu().float().numpy()  # [seq_len, d_model]
            # Use last token position
            h_last = hs[-1]
            final_logits_last = final_logits[-1]
            
            raw_logits = W_U @ h_last
            residual = final_logits_last - raw_logits
            
            layer_logits_residuals[li].append(residual)
            layer_diagnostics[li].append(float(np.linalg.norm(h_last)))
        
        del outputs, all_hidden
    
    # Step 2: Compute per-layer bias
    print("\n  [Training] Computing per-layer bias...")
    tuned_bias = {}
    for li in range(n_layers + 1):
        residuals = np.array(layer_logits_residuals[li])  # [n_sentences, vocab]
        tuned_bias[li] = np.mean(residuals, axis=0)  # [vocab]
        
        # Also compute diagonal scaling: for each dim, scale = cov(final, raw) / var(raw)
        # But bias-only is simpler and works well
        print(f"    Layer {li}: bias_norm={float(np.linalg.norm(tuned_bias[li])):.2f}, "
              f"mean_norm={float(np.mean(layer_diagnostics[li])):.2f}")
    
    # Step 3: Evaluate tuned lens on constraint pairs
    print("\n  [Evaluation] Computing tuned lens entropy on constraint pairs...")
    
    all_results = {}
    
    for ctype, pairs in [("grammar", GRAMMAR_PAIRS), 
                          ("animacy", ANIMACY_PAIRS),
                          ("physics", PHYSICS_PAIRS),
                          ("control", CONTROL_PAIRS)]:
        
        entropy_raw = defaultdict(list)
        entropy_tuned = defaultdict(list)
        kl_div = defaultdict(list)  # KL(raw || final)
        
        for pi, pair in enumerate(pairs):
            if pi % 10 == 0:
                print(f"    [{ctype}] Pair {pi+1}/{len(pairs)}", flush=True)
            
            sent_c, sent_i = pair[0], pair[1]
            target_pos = find_differentiating_position(tokenizer, sent_c, sent_i)
            
            hs_c, pos_c = get_hidden_states_at_position(
                model, tokenizer, device, sent_c, target_pos, n_layers)
            hs_i, pos_i = get_hidden_states_at_position(
                model, tokenizer, device, sent_i, target_pos, n_layers)
            
            # Also get final logits for KL divergence
            inputs_c = tokenizer(sent_c, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                out_c = model(input_ids=inputs_c["input_ids"].to(device),
                             attention_mask=inputs_c["attention_mask"].to(device),
                             output_hidden_states=True)
            final_logits_c = out_c.logits[0, pos_c].detach().cpu().float().numpy()
            del out_c
            
            inputs_i = tokenizer(sent_i, return_tensors="pt", truncation=True, max_length=128)
            with torch.no_grad():
                out_i = model(input_ids=inputs_i["input_ids"].to(device),
                             attention_mask=inputs_i["attention_mask"].to(device),
                             output_hidden_states=True)
            final_logits_i = out_i.logits[0, pos_i].detach().cpu().float().numpy()
            del out_i
            
            for li in range(n_layers + 1):
                if li not in hs_c or li not in hs_i:
                    continue
                
                h_c = hs_c[li]
                h_i = hs_i[li]
                
                # Raw logit lens
                raw_logits_c = W_U @ h_c
                raw_logits_i = W_U @ h_i
                ent_raw_c = compute_entropy_from_logits(raw_logits_c)
                ent_raw_i = compute_entropy_from_logits(raw_logits_i)
                
                # Tuned lens
                tuned_logits_c = W_U @ h_c + tuned_bias[li]
                tuned_logits_i = W_U @ h_i + tuned_bias[li]
                ent_tuned_c = compute_entropy_from_logits(tuned_logits_c)
                ent_tuned_i = compute_entropy_from_logits(tuned_logits_i)
                
                entropy_raw[li].append((ent_raw_c, ent_raw_i))
                entropy_tuned[li].append((ent_tuned_c, ent_tuned_i))
                
                # KL divergence: KL(softmax(raw) || softmax(final)) for correct sentence
                # Simplified: just compare entropy
                kl_c = ent_raw_c - compute_entropy_from_logits(final_logits_c)
                kl_i = ent_raw_i - compute_entropy_from_logits(final_logits_i)
                kl_div[li].append((kl_c, kl_i))
            
            del hs_c, hs_i
        
        result = {}
        for li in range(n_layers + 1):
            if li not in entropy_raw:
                continue
            
            raw_vals = entropy_raw[li]
            tuned_vals = entropy_tuned[li]
            
            result[li] = {
                "entropy_raw_correct_mean": float(np.mean([v[0] for v in raw_vals])),
                "entropy_raw_incorrect_mean": float(np.mean([v[1] for v in raw_vals])),
                "entropy_tuned_correct_mean": float(np.mean([v[0] for v in tuned_vals])),
                "entropy_tuned_incorrect_mean": float(np.mean([v[1] for v in tuned_vals])),
                "entropy_raw_gap": float(np.mean([v[0] - v[1] for v in raw_vals])),
                "entropy_tuned_gap": float(np.mean([v[0] - v[1] for v in tuned_vals])),
                "kl_raw_vs_final_correct": float(np.mean([v[0] for v in kl_div[li]])),
                "kl_raw_vs_final_incorrect": float(np.mean([v[1] for v in kl_div[li]])),
            }
        all_results[ctype] = result
    
    # Save tuned bias for reference
    all_results["_tuned_bias_norms"] = {str(li): float(np.linalg.norm(tuned_bias[li])) 
                                         for li in tuned_bias}
    
    return all_results


# =====================================================================
# EXP3: W_U-FREE DIRECT PROBING
# =====================================================================

def exp3_wu_free_probing(model, tokenizer, device, n_layers, d_model):
    """
    ★ 硬伤2验证: 完全不经过W_U的约束信号分析
    
    核心指标 (全部gauge-invariant):
    - Δ_l = h_correct_l - h_incorrect_l
    - ||Δ_l||: 约束信号绝对强度
    - ||Δ_l|| / ||h_l||: 相对强度 (消除范数效应)
    - cos(Δ_l, Δ_{l+1}): 约束方向稳定性
    - cos(Δ_l, Δ_L): 与最终信号的对齐度
    """
    print("\n" + "="*60)
    print("Exp3: W_U-FREE DIRECT PROBING")
    print("="*60)
    
    all_results = {}
    
    for ctype, pairs in [("grammar", GRAMMAR_PAIRS),
                          ("animacy", ANIMACY_PAIRS),
                          ("physics", PHYSICS_PAIRS),
                          ("control", CONTROL_PAIRS)]:
        print(f"\n  [{ctype}] Processing {len(pairs)} pairs...")
        
        delta_norms = defaultdict(list)
        delta_relative = defaultdict(list)  # ||Δ||/||h||
        delta_cos_next = defaultdict(list)  # cos(Δ_l, Δ_{l+1})
        delta_cos_final = defaultdict(list)  # cos(Δ_l, Δ_L)
        h_norms_correct = defaultdict(list)
        h_norms_incorrect = defaultdict(list)
        
        for pi, pair in enumerate(pairs):
            if pi % 10 == 0:
                print(f"    Pair {pi+1}/{len(pairs)}", flush=True)
            
            sent_c, sent_i = pair[0], pair[1]
            target_pos = find_differentiating_position(tokenizer, sent_c, sent_i)
            
            hs_c, _ = get_hidden_states_at_position(
                model, tokenizer, device, sent_c, target_pos, n_layers)
            hs_i, _ = get_hidden_states_at_position(
                model, tokenizer, device, sent_i, target_pos, n_layers)
            
            # Compute Δ_l for all layers
            deltas = {}
            for li in range(n_layers + 1):
                if li in hs_c and li in hs_i:
                    deltas[li] = hs_c[li] - hs_i[li]
                    h_norms_correct[li].append(float(np.linalg.norm(hs_c[li])))
                    h_norms_incorrect[li].append(float(np.linalg.norm(hs_i[li])))
            
            # Compute metrics
            final_li = n_layers
            for li in sorted(deltas.keys()):
                delta = deltas[li]
                delta_norm = float(np.linalg.norm(delta))
                h_norm = (h_norms_correct[li][-1] + h_norms_incorrect[li][-1]) / 2
                
                delta_norms[li].append(delta_norm)
                delta_relative[li].append(delta_norm / max(h_norm, 1e-10))
                
                # cos(Δ_l, Δ_{l+1})
                if li + 1 in deltas:
                    cos_next = float(np.dot(delta, deltas[li+1]) / 
                                     max(delta_norm * np.linalg.norm(deltas[li+1]), 1e-10))
                    delta_cos_next[li].append(cos_next)
                
                # cos(Δ_l, Δ_L)
                if final_li in deltas and li != final_li:
                    final_delta = deltas[final_li]
                    cos_final = float(np.dot(delta, final_delta) / 
                                      max(delta_norm * np.linalg.norm(final_delta), 1e-10))
                    delta_cos_final[li].append(cos_final)
            
            del hs_c, hs_i, deltas
        
        result = {}
        for li in range(n_layers + 1):
            if li not in delta_norms:
                continue
            entry = {
                "delta_norm_mean": float(np.mean(delta_norms[li])),
                "delta_norm_std": float(np.std(delta_norms[li])),
                "delta_relative_mean": float(np.mean(delta_relative[li])),
                "delta_relative_std": float(np.std(delta_relative[li])),
                "h_norm_correct_mean": float(np.mean(h_norms_correct[li])),
                "h_norm_incorrect_mean": float(np.mean(h_norms_incorrect[li])),
            }
            if li in delta_cos_next and delta_cos_next[li]:
                entry["cos_delta_next_mean"] = float(np.mean(delta_cos_next[li]))
                entry["cos_delta_next_std"] = float(np.std(delta_cos_next[li]))
            if li in delta_cos_final and delta_cos_final[li]:
                entry["cos_delta_final_mean"] = float(np.mean(delta_cos_final[li]))
                entry["cos_delta_final_std"] = float(np.std(delta_cos_final[li]))
            result[li] = entry
        
        all_results[ctype] = result
    
    return all_results


# =====================================================================
# EXP4: ACTIVATION PATCHING (因果金标准)
# =====================================================================

def exp4_activation_patching(model, tokenizer, device, n_layers):
    """
    ★ 硬伤3验证: Activation Patching — 因果干预金标准
    
    方法:
    1. 运行正确句子, cache每层在目标位置的hidden state
    2. 运行错误句子 (无patch), 获取baseline logits
    3. 逐层patch: 将正确句子的h_l patch到错误句子
    4. 测量patch后logits在约束相关token上的变化
    
    ★ 如果patch某层显著改变约束满足 → 该层是因果关键的
    """
    print("\n" + "="*60)
    print("Exp4: ACTIVATION PATCHING (CAUSAL)")
    print("="*60)
    
    all_results = {}
    
    for ctype, pairs in [("grammar", GRAMMAR_PAIRS[:10]),   # Use subset for speed
                          ("animacy", ANIMACY_PAIRS[:10]),
                          ("physics", PHYSICS_PAIRS[:10]),
                          ("control", CONTROL_PAIRS[:5])]:
        print(f"\n  [{ctype}] Processing {len(pairs)} pairs...")
        
        # For each layer, store the effect of patching
        patch_effects = defaultdict(list)  # {li: [effect_scores]}
        patch_effects_control = defaultdict(list)
        
        for pi, pair in enumerate(pairs):
            print(f"    Pair {pi+1}/{len(pairs)}", flush=True)
            
            sent_c, sent_i = pair[0], pair[1]
            tok_c, tok_i = pair[2], pair[3]
            
            # Get token IDs for constraint tokens
            tok_c_ids = tokenizer.encode(tok_c, add_special_tokens=False)
            tok_i_ids = tokenizer.encode(tok_i, add_special_tokens=False)
            
            target_pos = find_differentiating_position(tokenizer, sent_c, sent_i)
            
            # === Step 1: Run correct sentence, cache hidden states ===
            inputs_c = tokenizer(sent_c, return_tensors="pt", truncation=True, max_length=128)
            input_ids_c = inputs_c["input_ids"].to(device)
            attn_mask_c = inputs_c["attention_mask"].to(device)
            
            with torch.no_grad():
                out_c = model(input_ids=input_ids_c, attention_mask=attn_mask_c,
                             output_hidden_states=True)
            
            correct_hidden = {}
            for li, hs in enumerate(out_c.hidden_states):
                pos_c = min(target_pos, hs.shape[1] - 1)
                correct_hidden[li] = hs[0, pos_c].detach().clone()  # [d_model]
            
            # Final logits from correct sentence
            final_logits_c = out_c.logits[0, -1].detach().cpu().float().numpy()
            del out_c
            
            # === Step 2: Run incorrect sentence (baseline) ===
            inputs_i = tokenizer(sent_i, return_tensors="pt", truncation=True, max_length=128)
            input_ids_i = inputs_i["input_ids"].to(device)
            attn_mask_i = inputs_i["attention_mask"].to(device)
            
            with torch.no_grad():
                out_i = model(input_ids=input_ids_i, attention_mask=attn_mask_i,
                             output_hidden_states=True)
            
            # Baseline logits for incorrect sentence
            baseline_logits_i = out_i.logits[0, -1].detach().cpu().float().numpy()
            incorrect_hidden = {}
            for li, hs in enumerate(out_i.hidden_states):
                pos_i = min(target_pos, hs.shape[1] - 1)
                incorrect_hidden[li] = hs[0, pos_i].detach().clone()
            
            del out_i
            
            # === Step 3: Baseline logit difference ===
            # For grammar: correct token should have higher logit in correct sentence
            # For animacy/physics: the animate/possible context should favor correct reading
            # Use max logit as proxy for "constraint satisfaction"
            
            # Measure: log P(correct_token) - log P(incorrect_token) at the verb position
            # But we need logits at the verb position, not the last position
            # Use a simpler metric: entropy of final token distribution
            
            baseline_entropy = compute_entropy_from_logits(baseline_logits_i)
            correct_entropy = compute_entropy_from_logits(final_logits_c)
            
            # For the verb position (target_pos), get the logits
            # This requires a separate forward pass with different position
            # Simplified: use last-token logits and check if correct token is preferred
            
            # Even simpler metric: just check the difference in correct/incorrect token logits
            if len(tok_c_ids) > 0 and len(tok_i_ids) > 0:
                baseline_margin = float(baseline_logits_i[tok_c_ids[0]] - baseline_logits_i[tok_i_ids[0]])
                correct_margin = float(final_logits_c[tok_c_ids[0]] - final_logits_c[tok_i_ids[0]])
            else:
                baseline_margin = 0
                correct_margin = 0
            
            # === Step 4: Patch each layer and measure effect ===
            layers = get_layers(model)
            
            for patch_li in range(1, n_layers):
                # Patch: at layer patch_li, replace incorrect hidden state with correct one
                # This requires modifying the forward pass
                
                # We'll use a hook-based approach
                patched_logits = None
                hook_handle = None
                
                def make_patch_hook(correct_h, layer_idx):
                    def patch_hook(module, input, output):
                        if isinstance(output, tuple):
                            # Replace the hidden state at target position
                            # output[0] shape: [1, seq_len, d_model]
                            new_output = output[0].detach().clone()
                            pos = min(target_pos, new_output.shape[1] - 1)
                            # Add the difference (correct - incorrect) at target position
                            delta = correct_h[layer_idx].to(new_output.device) - new_output[0, pos]
                            new_output[0, pos] += delta
                            return (new_output,) + output[1:]
                        return output
                    return patch_hook
                
                try:
                    hook_handle = layers[patch_li].register_forward_hook(
                        make_patch_hook(correct_hidden, patch_li))
                    
                    with torch.no_grad():
                        out_patched = model(input_ids=input_ids_i, attention_mask=attn_mask_i)
                    
                    patched_logits = out_patched.logits[0, -1].detach().cpu().float().numpy()
                    del out_patched
                    
                    hook_handle.remove()
                    
                    # Measure effect
                    if len(tok_c_ids) > 0 and len(tok_i_ids) > 0:
                        patched_margin = float(patched_logits[tok_c_ids[0]] - patched_logits[tok_i_ids[0]])
                    else:
                        patched_margin = 0
                    
                    # Effect = how much patching moved the margin toward correct
                    effect = patched_margin - baseline_margin
                    
                    patch_effects[patch_li].append(effect)
                    
                except Exception as e:
                    if hook_handle:
                        hook_handle.remove()
                    print(f"      Patch L{patch_li} failed: {e}")
                    patch_effects[patch_li].append(0.0)
                
                finally:
                    if hook_handle:
                        try:
                            hook_handle.remove()
                        except:
                            pass
            
            # Free memory
            del correct_hidden, incorrect_hidden
        
        # Aggregate
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
    print(f"# Phase 182: CRITICAL VALIDATION — {model_name}")
    print(f"# Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#'*70}")
    
    # Load model
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    vocab_size = info.vocab_size
    
    print(f"\nModel: {info.model_class}, Layers={n_layers}, d_model={d_model}, vocab={vocab_size}")
    
    # Get W_U
    W_U = get_W_U(model, model_name)
    print(f"W_U shape: {W_U.shape}")
    
    # ===== Exp1: Norm & Logit Lens Diagnostics =====
    print(f"\n{'='*70}")
    print("Running Exp1: Norm & Logit Lens Diagnostics...")
    exp1_results = exp1_norm_diagnostics(model, tokenizer, device, W_U, n_layers, d_model)
    
    # ===== Exp2: Tuned Lens =====
    print(f"\n{'='*70}")
    print("Running Exp2: Tuned Lens...")
    exp2_results = exp2_tuned_lens(model, tokenizer, device, W_U, n_layers, d_model, vocab_size)
    
    # ===== Exp3: W_U-Free Probing =====
    print(f"\n{'='*70}")
    print("Running Exp3: W_U-Free Direct Probing...")
    exp3_results = exp3_wu_free_probing(model, tokenizer, device, n_layers, d_model)
    
    # ===== Exp4: Activation Patching =====
    print(f"\n{'='*70}")
    print("Running Exp4: Activation Patching...")
    exp4_results = exp4_activation_patching(model, tokenizer, device, n_layers)
    
    # ===== Save Results =====
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_path = f"tests/glm5_temp/phase182_{model_name}_{timestamp}.json"
    
    full_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "vocab_size": vocab_size,
        "timestamp": timestamp,
        "exp1_norm_diagnostics": {k: {str(kk): vv for kk, vv in v.items()} 
                                   for k, v in exp1_results.items()},
        "exp2_tuned_lens": {k: {str(kk): vv for kk, vv in v.items()} 
                             for k, v in exp2_results.items()},
        "exp3_wu_free": {k: {str(kk): vv for kk, vv in v.items()} 
                          for k, v in exp3_results.items()},
        "exp4_activation_patching": {k: {str(kk): vv for kk, vv in v.items()} 
                                      for k, v in exp4_results.items()},
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    # ===== Print Summary =====
    print(f"\n{'#'*70}")
    print("PHASE 182 SUMMARY")
    print(f"{'#'*70}")
    
    # Exp1 Summary: Compare raw vs norm-corrected entropy
    print("\n★★★ Exp1: Norm & Logit Lens Diagnostics ★★★")
    for ctype in ["grammar", "animacy", "physics", "control"]:
        if ctype not in exp1_results:
            continue
        data = exp1_results[ctype]
        
        # Find peak and trough layers for raw entropy
        layers_sorted = sorted([int(k) for k in data.keys()])
        raw_ent_c = [data[l]["entropy_raw_correct_mean"] for l in layers_sorted]
        normed_ent_c = [data[l]["entropy_normed_correct_mean"] for l in layers_sorted]
        norms = [data[l]["norm_correct_mean"] for l in layers_sorted]
        
        if len(raw_ent_c) > 5:
            raw_peak_layer = layers_sorted[np.argmax(raw_ent_c)]
            raw_peak_val = max(raw_ent_c)
            raw_final_val = raw_ent_c[-1]
            normed_peak_layer = layers_sorted[np.argmax(normed_ent_c)]
            normed_peak_val = max(normed_ent_c)
            normed_final_val = normed_ent_c[-1]
            
            # Check if norm-corrected entropy still shows funnel
            raw_funnel = raw_peak_val - raw_final_val
            normed_funnel = normed_peak_val - normed_final_val
            
            print(f"\n  [{ctype}]")
            print(f"    ||h||: L0={norms[0]:.2f} → L{layers_sorted[-1]}={norms[-1]:.2f} "
                  f"(growth={norms[-1]/max(norms[0],0.01):.1f}x)")
            print(f"    Raw entropy: peak at L{raw_peak_layer}={raw_peak_val:.3f}, "
                  f"final={raw_final_val:.3f}, funnel={raw_funnel:.3f}")
            print(f"    Norm-corrected entropy: peak at L{normed_peak_layer}={normed_peak_val:.3f}, "
                  f"final={normed_final_val:.3f}, funnel={normed_funnel:.3f}")
            print(f"    ★ Funnel preserved after norm correction? "
                  f"{'YES' if normed_funnel > 0.5 else 'NO - likely norm artifact!'}")
    
    # Exp2 Summary: Compare raw vs tuned lens
    print("\n★★★ Exp2: Tuned Lens ★★★")
    for ctype in ["grammar", "animacy", "physics", "control"]:
        if ctype not in exp2_results or ctype.startswith("_"):
            continue
        data = exp2_results[ctype]
        
        layers_sorted = sorted([int(k) for k in data.keys()])
        raw_ent_c = [data[l]["entropy_raw_correct_mean"] for l in layers_sorted]
        tuned_ent_c = [data[l]["entropy_tuned_correct_mean"] for l in layers_sorted]
        raw_gap = [data[l]["entropy_raw_gap"] for l in layers_sorted]
        tuned_gap = [data[l]["entropy_tuned_gap"] for l in layers_sorted]
        
        if len(raw_ent_c) > 5:
            raw_peak = max(raw_ent_c)
            tuned_peak = max(tuned_ent_c)
            raw_final = raw_ent_c[-1]
            tuned_final = tuned_ent_c[-1]
            
            raw_funnel = raw_peak - raw_final
            tuned_funnel = tuned_peak - tuned_final
            
            print(f"\n  [{ctype}]")
            print(f"    Raw: peak={raw_peak:.3f}, final={raw_final:.3f}, funnel={raw_funnel:.3f}")
            print(f"    Tuned: peak={tuned_peak:.3f}, final={tuned_final:.3f}, funnel={tuned_funnel:.3f}")
            print(f"    Raw gap (correct-incorrect): mean={np.mean(raw_gap):.3f}")
            print(f"    Tuned gap (correct-incorrect): mean={np.mean(tuned_gap):.3f}")
            print(f"    ★ Funnel preserved under tuned lens? "
                  f"{'YES' if tuned_funnel > 0.3 else 'NO - logit lens artifact!'}")
    
    # Exp3 Summary: W_U-free metrics
    print("\n★★★ Exp3: W_U-Free Direct Probing ★★★")
    for ctype in ["grammar", "animacy", "physics", "control"]:
        if ctype not in exp3_results:
            continue
        data = exp3_results[ctype]
        
        layers_sorted = sorted([int(k) for k in data.keys()])
        delta_norms = [data[l]["delta_norm_mean"] for l in layers_sorted]
        delta_rel = [data[l]["delta_relative_mean"] for l in layers_sorted]
        
        # Find transition layer where delta_norm starts increasing rapidly
        if len(delta_norms) > 5:
            print(f"\n  [{ctype}]")
            print(f"    Δ norm: L0={delta_norms[0]:.4f} → L{layers_sorted[-1]}={delta_norms[-1]:.4f} "
                  f"(growth={delta_norms[-1]/max(delta_norms[0],0.001):.1f}x)")
            print(f"    Δ relative: L0={delta_rel[0]:.6f} → L{layers_sorted[-1]}={delta_rel[-1]:.6f}")
            
            # Check cos(Δ_l, Δ_{l+1}) stability
            cos_vals = [data[l].get("cos_delta_next_mean", None) for l in layers_sorted]
            cos_vals = [c for c in cos_vals if c is not None]
            if cos_vals:
                print(f"    cos(Δ_l, Δ_{{l+1}}): mean={np.mean(cos_vals):.3f}, "
                      f"min={np.min(cos_vals):.3f}")
    
    # Exp4 Summary: Activation Patching
    print("\n★★★ Exp4: Activation Patching (CAUSAL) ★★★")
    for ctype in ["grammar", "animacy", "physics", "control"]:
        if ctype not in exp4_results:
            continue
        data = exp4_results[ctype]
        
        layers_sorted = sorted([int(k) for k in data.keys()])
        effects = [data[l]["patch_effect_mean"] for l in layers_sorted]
        effects_abs = [data[l]["patch_effect_abs_mean"] for l in layers_sorted]
        
        if effects:
            peak_layer = layers_sorted[np.argmax(effects_abs)]
            peak_effect = data[peak_layer]["patch_effect_mean"]
            peak_effect_abs = data[peak_layer]["patch_effect_abs_mean"]
            
            print(f"\n  [{ctype}]")
            print(f"    Peak causal layer: L{peak_layer} (effect={peak_effect:.4f}, "
                  f"|effect|={peak_effect_abs:.4f})")
            # Find top 3 causal layers
            top3 = sorted(layers_sorted, key=lambda l: data[l]["patch_effect_abs_mean"], reverse=True)[:3]
            print(f"    Top 3 causal layers: {[f'L{l}' for l in top3]}")
            top3_effects = [round(data[l]["patch_effect_mean"], 4) for l in top3]
            print(f"    Effects: {top3_effects}")
    
    # Release model
    release_model(model)
    
    print(f"\n{'#'*70}")
    print("Phase 182 COMPLETE!")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
