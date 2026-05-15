"""
Phase 180: ★★★ 约束场动力学 — 从"信号"到"可行域" ★★★
================================================================

★★★ 核心理论转变 ★★★
旧框架: 约束 = decoder direction (margin, logits, PCA方向)
新框架: 约束 = 状态空间中的可满足关系

用户的根本洞察 (完全正确):
  1. 真正的约束不是"方向", 而是"哪些状态之间可以互相到达"
  2. Transformer = 约束收缩系统: 每层消灭不兼容自由度
  3. 语言生成 = 约束塌缩 (constraint collapse), 不是"解码"
  4. 语法 = 在线约束求解, 世界知识 = 预训练流形几何
  5. 真正的不变量: 吸引盆, 闭合路径, 兼容图, 分叉结构, 转移算子

★★★ 三个关键实验 ★★★

Exp1: Feasible Region Collapse (可行域塌缩)
  - 核心预测: 每层的"可行token集合"应单调缩小 (约束收缩)
  - 测量: 熵H(l), top-10概率质量, 可行token数, 期望token概率
  - 对比: 合法vs不合法, 世界一致vs不一致
  - 如果熵单调递减 → 约束收缩是真实计算现象, 不是decoder伪象

Exp2: Constraint Bifurcation (约束分叉)
  - 核心预测: correct/incorrect轨迹在特定层突然分离 (相变)
  - 测量: 分离度曲线, 相变层, 分叉锐度
  - 对比: 语法约束 vs 世界约束 vs 生命性约束
  - 如果不同约束类型在不同层发生相变 → 层级传播是真实的

Exp3: Trajectory Topology (轨迹拓扑)
  - 核心预测: 同义句在深层收敛到同一吸引盆 (不变量)
  - 测量: 层间轨迹距离, 收敛/发散率
  - 对比: 释义对 (应收敛) vs 随机对 (应发散或不变)
  - 如果释义对收敛 → 吸引盆是gauge-invariant的数学对象

★★★ 内存优化 ★★★
  - 不存储完整概率分布, 只计算统计量
  - W_U只加载一次, 复用所有句子
  - 每个句子前向传播后立即提取统计量, 释放hidden states

Usage: python tests/glm5/phase180_feasible_region_collapse.py <model_name>
  model_name: qwen3, glm4, deepseek7b
"""

import sys
import os
import time
import json
import gc
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'glm5'))

from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS


# =====================================================================
# MODEL LOADING (BF16 + device_map="auto")
# =====================================================================

def load_model_bf16(model_name):
    """BF16 + device_map=auto loading for all models"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[bf16] Loading {model_name} (bfloat16 + device_map=auto)...", flush=True)

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
    print(f"[bf16] {model_name} loaded: GPU={gpu_mem:.2f}GB", flush=True)

    return model, tokenizer, device


# =====================================================================
# ★★★ 句子定义 — 4个约束类别 × 20对 ★★★
# =====================================================================

# 格式: (correct_sentence, incorrect_sentence, expected_token, unexpected_token, 
#         prediction_position_keyword, category)
# prediction_position_keyword: 用于定位预测位置的关键词 (在它之前预测它)

GRAMMAR_PAIRS = [
    # Subject-verb agreement: singular subject → verb+s
    ("The cat sleeps quietly", "The cat sleep quietly", " sleeps", " sleep", "cat"),
    ("The dog runs fast", "The dog run fast", " runs", " run", "dog"),
    ("The bird sings loudly", "The bird sing loudly", " sings", " sing", "bird"),
    ("The child plays outside", "The child play outside", " plays", " play", "child"),
    ("The man walks slowly", "The man walk slowly", " walks", " walk", "man"),
    ("The girl reads a book", "The girl read a book", " reads", " read", "girl"),
    ("The tree grows tall", "The tree grow tall", " grows", " grow", "tree"),
    ("The car moves forward", "The car move forward", " moves", " move", "car"),
    ("The fish swims upstream", "The fish swim upstream", " swims", " swim", "fish"),
    ("The clock ticks steadily", "The clock tick steadily", " ticks", " tick", "clock"),
    # Subject-verb agreement: plural subject → verb (no s)
    ("The cats sleep quietly", "The cats sleeps quietly", " sleep", " sleeps", "cats"),
    ("The dogs run fast", "The dogs runs fast", " run", " runs", "dogs"),
    ("The birds sing loudly", "The birds sings loudly", " sing", " sings", "birds"),
    ("The children play outside", "The children plays outside", " play", " plays", "children"),
    ("The men walk slowly", "The men walks slowly", " walk", " walks", "men"),
    ("The girls read a book", "The girls reads a book", " read", " reads", "girls"),
    ("The trees grow tall", "The trees grows tall", " grow", " grows", "trees"),
    ("The cars move forward", "The cars moves forward", " move", " moves", "cars"),
    ("The fish swim upstream", "The fish swims upstream", " swim", " swims", "fish"),
    ("The clocks tick steadily", "The clocks ticks steadily", " tick", " ticks", "clocks"),
]

WORLD_PHYSICAL_PAIRS = [
    # Physical affordance: what happens based on physical properties
    ("The glass fell and it broke", "The glass fell and it floated", " broke", " floated", "it"),
    ("The mirror dropped and it shattered", "The mirror dropped and it bounced", " shattered", " bounced", "it"),
    ("The feather drifted and it floated", "The feather drifted and it shattered", " floated", " shattered", "it"),
    ("The leaf fell and it floated", "The leaf fell and it shattered", " floated", " shattered", "it"),
    ("The rock sank because it was heavy", "The rock sank because it was light", " heavy", " light", "was"),
    ("The ice melted because it was warm", "The ice melted because it was cold", " warm", " cold", "was"),
    ("The metal bent because it was soft", "The metal bent because it was rigid", " soft", " rigid", "was"),
    ("The wood burned because it was dry", "The wood burned because it was wet", " dry", " wet", "was"),
    ("The rubber stretched because it was elastic", "The rubber stretched because it was brittle", " elastic", " brittle", "was"),
    ("The cotton absorbed water because it was porous", "The cotton absorbed water because it was waterproof", " porous", " waterproof", "was"),
    # More physical constraints
    ("The ball bounced because it was", "The ball shattered because it was", " round", " fragile", "was"),
    ("The knife cut because it was sharp", "The knife cut because it was blunt", " sharp", " blunt", "was"),
    ("The water froze because it was cold", "The water froze because it was hot", " cold", " hot", "was"),
    ("The balloon popped because it was", "The balloon expanded because it was", " overinflated", " empty", "was"),
    ("The rope snapped because it was", "The rope stretched because it was", " weak", " strong", "was"),
    ("The glass is transparent so light passes through", "The glass is transparent so light is blocked", " through", " blocked", "passes"),
    ("The magnet attracts iron because it is", "The magnet attracts iron because it is", " magnetic", " wooden", "is"),
    ("The sponge expanded because it absorbed", "The sponge expanded because it repelled", " absorbed", " repelled", "it"),
    ("The battery powered the device because it was", "The battery powered the device because it was", " charged", " empty", "was"),
    ("The ice cube melted in the sun because", "The ice cube froze in the sun because", " it", " it", "because"),
]

WORLD_ANIMACY_PAIRS = [
    # Animacy constraints: only animate beings can do certain things
    ("The dog thought about the problem", "The rock thought about the problem", " thought", " thought", "dog/rock"),
    ("The cat decided to leave", "The table decided to leave", " decided", " decided", "cat/table"),
    ("The man felt sad about", "The stone felt sad about", " felt", " felt", "man/stone"),
    ("The woman remembered the event", "The wall remembered the event", " remembered", " remembered", "woman/wall"),
    ("The child believed in magic", "The chair believed in magic", " believed", " believed", "child/chair"),
    ("The teacher explained the concept", "The bookshelf explained the concept", " explained", " explained", "teacher/bookshelf"),
    ("The bird wanted to fly", "The cup wanted to fly", " wanted", " wanted", "bird/cup"),
    ("The fish tried to swim", "The pencil tried to swim", " tried", " tried", "fish/pencil"),
    ("The horse ran across the field", "The bottle ran across the field", " ran", " ran", "horse/bottle"),
    ("The student learned the lesson", "The desk learned the lesson", " learned", " learned", "student/desk"),
    # More animacy - living vs non-living actions
    ("The plant grew toward the light", "The metal grew toward the light", " grew", " grew", "plant/metal"),
    ("The person breathed deeply because", "The statue breathed deeply because", " they", " it", "because"),
    ("The animal survived the winter because", "The building survived the winter because", " it", " it", "because"),
    ("The baby cried because it was", "The pillow cried because it was", " hungry", " soft", "was"),
    ("The scientist discovered the truth", "The rock discovered the truth", " discovered", " discovered", "scientist/rock"),
    ("The patient recovered from the illness", "The chair recovered from the illness", " recovered", " recovered", "patient/chair"),
    ("The dog barked at the stranger", "The shoe barked at the stranger", " barked", " barked", "dog/shoe"),
    ("The cat jumped over the fence", "The rock jumped over the fence", " jumped", " jumped", "cat/rock"),
    ("The singer performed beautifully at", "The lamp performed beautifully at", " the", " the", "at"),
    ("The athlete trained hard before the", "The mountain trained hard before the", " competition", " competition", "the"),
]

WORLD_CAUSAL_PAIRS = [
    # Causal constraints: what follows from cause-effect relations
    ("The ice melted because the temperature rose", "The ice froze because the temperature rose", " melted", " froze", "because"),
    ("The fire spread because it was windy", "The fire stopped because it was windy", " spread", " stopped", "because"),
    ("The plant died because it lacked water", "The plant thrived because it lacked water", " died", " thrived", "because"),
    ("The car stopped because the driver braked", "The car accelerated because the driver braked", " stopped", " accelerated", "because"),
    ("The soup cooled because it was left out", "The soup heated because it was left out", " cooled", " heated", "because"),
    ("The flood happened because it rained heavily", "The drought happened because it rained heavily", " flood", " drought", "because"),
    ("The metal expanded because it was heated", "The metal contracted because it was heated", " expanded", " contracted", "because"),
    ("The patient recovered because the medicine worked", "The patient worsened because the medicine worked", " recovered", " worsened", "because"),
    ("The bridge collapsed because the earthquake struck", "The bridge strengthened because the earthquake struck", " collapsed", " strengthened", "because"),
    ("The battery died because it was overused", "The battery charged because it was overused", " died", " charged", "because"),
    # More causal
    ("The crop failed because the drought was severe", "The crop flourished because the drought was severe", " failed", " flourished", "because"),
    ("The balloon popped because it was overinflated", "The balloon solidified because it was overinflated", " popped", " solidified", "because"),
    ("The glass cracked because the impact was strong", "The glass healed because the impact was strong", " cracked", " healed", "because"),
    ("The food spoiled because it was left in heat", "The food freshened because it was left in heat", " spoiled", " freshened", "because"),
    ("The tire deflated because there was a puncture", "The tire inflated because there was a puncture", " deflated", " inflated", "because"),
    ("The illness spread because the conditions were unsanitary", "The illness disappeared because the conditions were unsanitary", " spread", " disappeared", "because"),
    ("The wood rotted because it was exposed to moisture", "The wood petrified because it was exposed to moisture", " rotted", " petrified", "because"),
    ("The engine overheated because the coolant leaked", "The engine cooled because the coolant leaked", " overheated", " cooled", "because"),
    ("The pipe burst because the water froze inside", "The pipe sealed because the water froze inside", " burst", " sealed", "because"),
    ("The metal rusted because it was exposed to water", "The metal shined because it was exposed to water", " rusted", " shined", "because"),
]

# ★★★ 释义对 — 用于轨迹拓扑实验 ★★★
# 格式: (sentence_A, sentence_B) — 同义但不同表面形式
PARAPHRASE_PAIRS = [
    # Active/passive voice
    ("The cat chased the mouse", "The mouse was chased by the cat"),
    ("The dog bit the man", "The man was bitten by the dog"),
    ("The teacher praised the student", "The student was praised by the teacher"),
    ("The wind destroyed the house", "The house was destroyed by the wind"),
    ("The chef cooked the meal", "The meal was cooked by the chef"),
    # Synonym substitution
    ("The automobile moved quickly", "The car drove fast"),
    ("The feline rested on the rug", "The cat sat on the mat"),
    ("The scientist discovered a new element", "The researcher found a novel chemical"),
    ("The soldier fought bravely", "The warrior battled courageously"),
    ("The musician played beautifully", "The artist performed wonderfully"),
    # Structural variation
    ("It was the glass that broke", "The glass broke"),
    ("It was the cat that chased the mouse", "The cat chased the mouse"),
    ("The fact that ice melts when heated is well known", "Ice melts when heated"),
    ("What the dog did was chase the ball", "The dog chased the ball"),
    ("There was a cat sitting on the mat", "A cat sat on the mat"),
    # Entailment pairs (same essential meaning)
    ("The glass shattered into pieces", "The glass broke completely"),
    ("The water froze solid", "The water turned to ice"),
    ("The fire completely consumed the building", "The fire burned the building down"),
    ("The sun rose at dawn", "Dawn brought sunrise"),
    ("The heavy rain caused flooding", "The flood resulted from heavy rain"),
    # More paraphrase types
    ("The child was afraid of the dark", "The kid feared the darkness"),
    ("The doctor cured the illness", "The physician healed the disease"),
    ("The mountain was very tall", "The mountain had great height"),
    ("The river flowed rapidly", "The river ran swiftly"),
    ("The storm damaged the roof", "The roof was harmed by the storm"),
    ("The student answered correctly", "The learner gave the right response"),
    ("The food was extremely spicy", "The dish had intense heat"),
    ("The room was completely dark", "No light entered the room"),
    ("The engine was very powerful", "The motor had great strength"),
    ("The book described ancient history", "The text told of olden times"),
]

# ★★★ 随机句子对 — 基线对比 ★★★
RANDOM_PAIRS = [
    ("The cat chased the mouse", "The scientist discovered a new element"),
    ("The glass fell and it broke", "The teacher explained the concept"),
    ("The dog ran quickly", "The ice melted because the temperature rose"),
    ("The bird sang loudly", "The metal expanded because it was heated"),
    ("The child played outside", "The flood happened because it rained heavily"),
    ("The man walked slowly", "The plant died because it lacked water"),
    ("The fish swam upstream", "The fire spread because it was windy"),
    ("The clock ticked steadily", "The soup cooled because it was left out"),
    ("The tree grew tall", "The patient recovered because the medicine worked"),
    ("The car moved forward", "The bridge collapsed because the earthquake struck"),
    ("The woman read a book", "The battery died because it was overused"),
    ("The girl sang a song", "The crop failed because the drought was severe"),
    ("The boy jumped high", "The balloon popped because it was overinflated"),
    ("The cat sat on the mat", "The food spoiled because it was left in heat"),
    ("The dog barked at the stranger", "The glass cracked because the impact was strong"),
    ("The fire burned the building", "The illness spread because conditions were unsanitary"),
    ("The water turned to ice", "The engine overheated because the coolant leaked"),
    ("The chef cooked the meal", "The pipe burst because the water froze inside"),
    ("The wind destroyed the house", "The wood rotted because it was exposed to moisture"),
    ("The doctor cured the illness", "The metal rusted because it was exposed to water"),
]


# =====================================================================
# ★★★ 核心计算函数 ★★★
# =====================================================================

def get_final_norm(model):
    """获取模型的最终layer norm层"""
    # Qwen3/Qwen2: model.model.norm
    if hasattr(model, 'model') and hasattr(model.model, 'norm'):
        return model.model.norm
    # GLM4: try multiple paths
    if hasattr(model, 'transformer'):
        if hasattr(model.transformer, 'encoder'):
            if hasattr(model.transformer.encoder, 'final_layernorm'):
                return model.transformer.encoder.final_layernorm
    if hasattr(model, 'model') and hasattr(model.model, 'encoder'):
        if hasattr(model.model.encoder, 'final_layernorm'):
            return model.model.encoder.final_layernorm
    # Fallback: identity
    print("  [WARN] No final norm found, using identity", flush=True)
    return None


def compute_feasible_region_stats(h_l, norm_layer, W_U_torch, device, expected_id=None, unexpected_id=None):
    """
    计算可行域统计量
    
    核心思想: 
    在layer l, hidden state h_l 通过 logit lens 得到概率分布
    这个分布告诉我们"哪些token仍然可行" — 即可行域
    
    Args:
        h_l: numpy array [d_model] — 某层某位置的hidden state
        norm_layer: 最终layer norm层 (或None)
        W_U_torch: lm_head权重 tensor [vocab_size, d_model] on device
        device: torch device
        expected_id: 期望token的id (可选)
        unexpected_id: 非期望token的id (可选)
    
    Returns:
        dict with: entropy, top10_mass, feasible_count, effective_vocab,
                   expected_prob, unexpected_prob, margin
    """
    h_tensor = torch.tensor(h_l, dtype=torch.float32, device=device).unsqueeze(0)
    
    # Apply final layer norm if available
    if norm_layer is not None:
        with torch.no_grad():
            try:
                h_normed = norm_layer(h_tensor)
            except:
                h_normed = h_tensor
    else:
        h_normed = h_tensor
    
    # Compute logits: [1, d_model] @ [d_model, vocab_size] -> [1, vocab_size]
    with torch.no_grad():
        logits = h_normed @ W_U_torch.T  # [1, vocab_size]
        logits = logits.squeeze(0)  # [vocab_size]
    
    # Softmax
    logits_float = logits.float()
    # Numerical stability
    logits_max = logits_float.max()
    exp_logits = torch.exp(logits_float - logits_max)
    probs = exp_logits / exp_logits.sum()
    
    # ★ Stat 1: Entropy H = -Σ p_i log(p_i)
    log_probs = torch.log(probs + 1e-12)
    entropy = float(-(probs * log_probs).sum().cpu().numpy())
    
    # ★ Stat 2: Top-10 probability mass
    top10_probs, _ = torch.topk(probs, min(10, len(probs)))
    top10_mass = float(top10_probs.sum().cpu().numpy())
    
    # ★ Stat 3: Feasible token count (p > 0.001, i.e. 0.1%)
    feasible_count = int((probs > 0.001).sum().cpu().numpy())
    
    # ★ Stat 4: Effective vocabulary = exp(entropy)
    effective_vocab = float(np.exp(entropy))
    
    # ★ Stat 5: Expected/unexpected token probabilities
    expected_prob = 0.0
    unexpected_prob = 0.0
    margin = 0.0
    if expected_id is not None and unexpected_id is not None:
        expected_prob = float(probs[expected_id].cpu().numpy())
        unexpected_prob = float(probs[unexpected_id].cpu().numpy())
        margin = float((logits_float[expected_id] - logits_float[unexpected_id]).cpu().numpy())
    
    # Clean up GPU tensors
    del logits, logits_float, exp_logits, probs, log_probs, h_tensor, h_normed
    if 'top10_probs' in dir():
        del top10_probs
    
    return {
        'entropy': round(entropy, 4),
        'top10_mass': round(top10_mass, 4),
        'feasible_count': feasible_count,
        'effective_vocab': round(effective_vocab, 4),
        'expected_prob': round(expected_prob, 6),
        'unexpected_prob': round(unexpected_prob, 6),
        'margin': round(margin, 4),
    }


def find_prediction_position(tokenizer, sentence, keyword=None):
    """
    找到预测位置
    
    对于 "The cat sleeps", 如果 keyword="cat", 返回 "cat" 的token位置
    模型在此位置预测下一个token (应该是 "sleeps")
    """
    tokens = tokenizer.encode(sentence, add_special_tokens=False)
    
    if keyword is None:
        # 默认: last token之前的位置 (即倒数第二个token末尾)
        return len(tokens) - 1
    
    # 找到keyword所在的token位置
    keyword_ids = tokenizer.encode(keyword, add_special_tokens=False)
    if len(keyword_ids) == 0:
        return len(tokens) - 1
    
    # 搜索keyword_ids在tokens中的位置
    kw_len = len(keyword_ids)
    for i in range(len(tokens) - kw_len + 1):
        if tokens[i:i+kw_len] == keyword_ids:
            return i + kw_len - 1  # 返回keyword最后一个token的位置
    
    # Fallback: last token
    return len(tokens) - 1


def get_hidden_states_at_position(model, tokenizer, device, sentence, position, n_layers):
    """
    获取指定位置在所有层的hidden states
    
    Returns:
        dict: {layer_idx: numpy_array[d_model]}
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
    
    all_hidden = outputs.hidden_states  # tuple of (1, seq_len, d_model)
    
    result = {}
    for li, hs in enumerate(all_hidden):
        if position < hs.shape[1]:
            h = hs[0, position, :].detach().cpu().float().numpy()
            result[li] = h  # li=0: embedding, li=1: after layer 0, etc.
    
    # Clean up
    del outputs, all_hidden
    
    return result


# =====================================================================
# ★★★ Exp1: Feasible Region Collapse (可行域塌缩) ★★★
# =====================================================================

def experiment1_feasible_region(model, tokenizer, device, n_layers, norm_layer, W_U_torch):
    """
    可行域塌缩实验
    
    核心预测: 
    - 熵 H(l) 应随l单调递减 (约束收缩)
    - 可行token数应随l单调递减
    - 正确句子的expected_prob应随l递增
    - 不正确句子的unexpected_prob应随l递减
    """
    print("\n" + "="*70, flush=True)
    print("  Exp1: Feasible Region Collapse (可行域塌缩)", flush=True)
    print("="*70, flush=True)
    
    all_pairs = {
        'grammar': GRAMMAR_PAIRS,
        'physical': WORLD_PHYSICAL_PAIRS,
        'animacy': WORLD_ANIMACY_PAIRS,
        'causal': WORLD_CAUSAL_PAIRS,
    }
    
    # Results structure: {category: {'correct': {layer: stats}, 'incorrect': {layer: stats}}}
    results = {}
    
    for cat_name, pairs in all_pairs.items():
        print(f"\n  --- Category: {cat_name} ({len(pairs)} pairs) ---", flush=True)
        
        correct_entropies = defaultdict(list)  # {layer: [entropy_values]}
        incorrect_entropies = defaultdict(list)
        correct_feasible = defaultdict(list)
        incorrect_feasible = defaultdict(list)
        correct_expected_probs = defaultdict(list)
        incorrect_unexpected_probs = defaultdict(list)
        correct_margins = defaultdict(list)
        incorrect_margins = defaultdict(list)
        
        for pi, (correct_sent, incorrect_sent, exp_tok, unexp_tok, pos_keyword) in enumerate(pairs):
            if pi % 5 == 0:
                print(f"    Processing pair {pi+1}/{len(pairs)}...", flush=True)
            
            # Get token IDs
            exp_ids = tokenizer.encode(exp_tok, add_special_tokens=False)
            unexp_ids = tokenizer.encode(unexp_tok, add_special_tokens=False)
            exp_id = exp_ids[0] if exp_ids else None
            unexp_id = unexp_ids[0] if unexp_ids else None
            
            # Find prediction position
            # For grammar: use subject position
            # For world: use the keyword position
            if cat_name == 'grammar':
                pred_pos = find_prediction_position(tokenizer, correct_sent, pos_keyword)
            else:
                # For world constraints, predict at the position before the target token
                # Use the position of the last token before the expected/unexpected token
                pred_pos = find_prediction_position(tokenizer, correct_sent)
            
            # Get hidden states for both sentences
            for sent_type, sentence in [('correct', correct_sent), ('incorrect', incorrect_sent)]:
                try:
                    hs_dict = get_hidden_states_at_position(
                        model, tokenizer, device, sentence, pred_pos, n_layers
                    )
                    
                    for li, h_l in hs_dict.items():
                        stats = compute_feasible_region_stats(
                            h_l, norm_layer, W_U_torch, device, exp_id, unexp_id
                        )
                        
                        if sent_type == 'correct':
                            correct_entropies[li].append(stats['entropy'])
                            correct_feasible[li].append(stats['feasible_count'])
                            correct_expected_probs[li].append(stats['expected_prob'])
                            correct_margins[li].append(stats['margin'])
                        else:
                            incorrect_entropies[li].append(stats['entropy'])
                            incorrect_feasible[li].append(stats['feasible_count'])
                            incorrect_unexpected_probs[li].append(stats['unexpected_prob'])
                            incorrect_margins[li].append(stats['margin'])
                    
                    del hs_dict
                    
                except Exception as e:
                    print(f"      [WARN] Failed: {e}", flush=True)
                    continue
        
        # Aggregate results
        cat_results = {}
        for li in range(n_layers + 1):
            cat_results[li] = {
                'correct_entropy': round(float(np.mean(correct_entropies[li])), 4) if correct_entropies[li] else 0,
                'incorrect_entropy': round(float(np.mean(incorrect_entropies[li])), 4) if incorrect_entropies[li] else 0,
                'correct_feasible': round(float(np.mean(correct_feasible[li])), 1) if correct_feasible[li] else 0,
                'incorrect_feasible': round(float(np.mean(incorrect_feasible[li])), 1) if incorrect_feasible[li] else 0,
                'correct_expected_prob': round(float(np.mean(correct_expected_probs[li])), 6) if correct_expected_probs[li] else 0,
                'incorrect_unexpected_prob': round(float(np.mean(incorrect_unexpected_probs[li])), 6) if incorrect_unexpected_probs[li] else 0,
                'correct_margin': round(float(np.mean(correct_margins[li])), 4) if correct_margins[li] else 0,
                'incorrect_margin': round(float(np.mean(incorrect_margins[li])), 4) if incorrect_margins[li] else 0,
            }
        
        results[cat_name] = cat_results
        
        # Print summary
        l0 = cat_results.get(0, {})
        l_last = cat_results.get(n_layers, {})
        print(f"\n  ★ {cat_name} Summary:", flush=True)
        print(f"    Correct: Entropy L0={l0.get('correct_entropy',0):.2f} → L_last={l_last.get('correct_entropy',0):.2f}", flush=True)
        print(f"    Incorrect: Entropy L0={l0.get('incorrect_entropy',0):.2f} → L_last={l_last.get('incorrect_entropy',0):.2f}", flush=True)
        print(f"    Correct: Feasible L0={l0.get('correct_feasible',0):.0f} → L_last={l_last.get('correct_feasible',0):.0f}", flush=True)
        print(f"    Incorrect: Feasible L0={l0.get('incorrect_feasible',0):.0f} → L_last={l_last.get('incorrect_feasible',0):.0f}", flush=True)
        print(f"    Correct: Expected prob L0={l0.get('correct_expected_prob',0):.4f} → L_last={l_last.get('correct_expected_prob',0):.4f}", flush=True)
        print(f"    Correct: Margin L0={l0.get('correct_margin',0):.2f} → L_last={l_last.get('correct_margin',0):.2f}", flush=True)
    
    return results


# =====================================================================
# ★★★ Exp2: Constraint Bifurcation (约束分叉) ★★★
# =====================================================================

def experiment2_bifurcation(model, tokenizer, device, n_layers, norm_layer, W_U_torch):
    """
    约束分叉实验
    
    核心预测:
    - correct和incorrect轨迹在某个层突然分离 (相变)
    - 不同约束类型的相变层不同
    - 语法: 浅-中层分叉, 世界: 深层分叉 (或即时)
    
    测量:
    - 分离度 = |margin_correct - margin_incorrect| = 2 * |margin| (因为incorrect的margin与correct相反)
    - 相变层 = 分离度首次超过阈值(2σ)的层
    - 分叉锐度 = 分离度的导数的最大值
    """
    print("\n" + "="*70, flush=True)
    print("  Exp2: Constraint Bifurcation (约束分叉)", flush=True)
    print("="*70, flush=True)
    
    all_pairs = {
        'grammar': GRAMMAR_PAIRS[:10],  # Use subset for speed
        'physical': WORLD_PHYSICAL_PAIRS[:10],
        'animacy': WORLD_ANIMACY_PAIRS[:10],
        'causal': WORLD_CAUSAL_PAIRS[:10],
    }
    
    results = {}
    
    for cat_name, pairs in all_pairs.items():
        print(f"\n  --- Category: {cat_name} ({len(pairs)} pairs) ---", flush=True)
        
        # For each pair, compute margin at each layer
        all_margins = defaultdict(list)  # {layer: [margin_values]}
        
        for pi, (correct_sent, incorrect_sent, exp_tok, unexp_tok, pos_keyword) in enumerate(pairs):
            # Get token IDs
            exp_ids = tokenizer.encode(exp_tok, add_special_tokens=False)
            unexp_ids = tokenizer.encode(unexp_tok, add_special_tokens=False)
            exp_id = exp_ids[0] if exp_ids else None
            unexp_id = unexp_ids[0] if unexp_ids else None
            
            if cat_name == 'grammar':
                pred_pos = find_prediction_position(tokenizer, correct_sent, pos_keyword)
            else:
                pred_pos = find_prediction_position(tokenizer, correct_sent)
            
            # Only need correct sentence's margin (it captures the separation)
            try:
                hs_dict = get_hidden_states_at_position(
                    model, tokenizer, device, correct_sent, pred_pos, n_layers
                )
                
                for li, h_l in hs_dict.items():
                    stats = compute_feasible_region_stats(
                        h_l, norm_layer, W_U_torch, device, exp_id, unexp_id
                    )
                    all_margins[li].append(stats['margin'])
                
                del hs_dict
                
            except Exception as e:
                continue
        
        # Compute separation curve
        avg_margins = {li: float(np.mean(vals)) for li, vals in all_margins.items()}
        std_margins = {li: float(np.std(vals)) for li, vals in all_margins.items()}
        
        # Find phase transition layer
        # Phase transition = first layer where |margin| > 2 * max(std, 0.5)
        layers_sorted = sorted(avg_margins.keys())
        separation = {li: abs(avg_margins[li]) for li in layers_sorted}
        
        # Noise level = average std in early layers (L0-L5)
        early_layers = [li for li in layers_sorted if li <= 5]
        noise_level = float(np.mean([max(std_margins.get(li, 1.0), 0.5) for li in early_layers])) if early_layers else 1.0
        threshold = 2 * noise_level
        
        phase_transition_layer = None
        for li in layers_sorted:
            if separation[li] > threshold:
                phase_transition_layer = li
                break
        
        # Compute bifurcation sharpness (max derivative of separation)
        sharpness = 0
        sharpness_layer = None
        for i in range(1, len(layers_sorted)):
            li_prev = layers_sorted[i-1]
            li_curr = layers_sorted[i]
            dsep = abs(separation[li_curr] - separation[li_prev])
            dl = li_curr - li_prev
            if dl > 0:
                deriv = dsep / dl
                if deriv > sharpness:
                    sharpness = deriv
                    sharpness_layer = li_curr
        
        results[cat_name] = {
            'avg_margins': {str(li): round(avg_margins[li], 4) for li in layers_sorted},
            'separation': {str(li): round(separation[li], 4) for li in layers_sorted},
            'phase_transition_layer': phase_transition_layer,
            'bifurcation_sharpness': round(sharpness, 4),
            'sharpness_layer': sharpness_layer,
            'threshold': round(threshold, 4),
            'noise_level': round(noise_level, 4),
        }
        
        print(f"  ★ {cat_name} Bifurcation:", flush=True)
        print(f"    Phase transition layer: {phase_transition_layer}", flush=True)
        print(f"    Bifurcation sharpness: {sharpness:.4f} at L{sharpness_layer}", flush=True)
        print(f"    Noise level: {noise_level:.4f}, Threshold: {threshold:.4f}", flush=True)
        print(f"    Separation L0={separation.get(0,0):.4f} → L_last={separation.get(n_layers,0):.4f}", flush=True)
    
    return results


# =====================================================================
# ★★★ Exp3: Trajectory Topology (轨迹拓扑) ★★★
# =====================================================================

def experiment3_trajectory_topology(model, tokenizer, device, n_layers):
    """
    轨迹拓扑实验
    
    核心预测:
    - 释义对在深层收敛 (同吸引盆)
    - 随机对不收敛 (不同吸引盆)
    - 收敛 = 层间距离递减
    
    这是gauge-invariance的直接测试:
    如果同义不同形句子在深层收敛 → 吸引盆是真实的不变量
    """
    print("\n" + "="*70, flush=True)
    print("  Exp3: Trajectory Topology (轨迹拓扑)", flush=True)
    print("="*70, flush=True)
    
    # ★★★ 测量层间轨迹距离 ★★★
    # 对于每对句子, 计算它们在每个层的hidden state距离
    
    def compute_pair_distances(pairs, pair_type):
        """计算句子对在每个层的距离"""
        print(f"\n  --- {pair_type}: {len(pairs)} pairs ---", flush=True)
        
        all_distances = defaultdict(list)  # {layer: [distances]}
        all_cosine_sims = defaultdict(list)
        
        for pi, (sent_a, sent_b) in enumerate(pairs):
            if pi % 5 == 0:
                print(f"    Processing pair {pi+1}/{len(pairs)}...", flush=True)
            
            try:
                # Get hidden states for both sentences at the LAST token position
                hs_a = get_hidden_states_at_position(
                    model, tokenizer, device, sent_a, 
                    len(tokenizer.encode(sent_a, add_special_tokens=False)) - 1,
                    n_layers
                )
                hs_b = get_hidden_states_at_position(
                    model, tokenizer, device, sent_b,
                    len(tokenizer.encode(sent_b, add_special_tokens=False)) - 1,
                    n_layers
                )
                
                # Compute distances at each layer
                for li in hs_a:
                    if li not in hs_b:
                        continue
                    h_a = hs_a[li]
                    h_b = hs_b[li]
                    
                    # Euclidean distance
                    dist = float(np.linalg.norm(h_a - h_b))
                    all_distances[li].append(dist)
                    
                    # Cosine similarity
                    norm_a = np.linalg.norm(h_a)
                    norm_b = np.linalg.norm(h_b)
                    if norm_a > 1e-10 and norm_b > 1e-10:
                        cos_sim = float(np.dot(h_a, h_b) / (norm_a * norm_b))
                    else:
                        cos_sim = 0.0
                    all_cosine_sims[li].append(cos_sim)
                
                del hs_a, hs_b
                
            except Exception as e:
                continue
        
        # Aggregate
        avg_distances = {li: float(np.mean(vals)) for li, vals in all_distances.items()}
        avg_cosine = {li: float(np.mean(vals)) for li, vals in all_cosine_sims.items()}
        
        return avg_distances, avg_cosine
    
    # Compute for paraphrase pairs
    para_dists, para_cos = compute_pair_distances(PARAPHRASE_PAIRS, "Paraphrase")
    
    # Compute for random pairs
    rand_dists, rand_cos = compute_pair_distances(RANDOM_PAIRS, "Random")
    
    # ★★★ Analysis: Convergence Rate ★★★
    # Convergence = distance decreasing with layer depth
    # Divergence = distance increasing
    
    layers_sorted = sorted(set(list(para_dists.keys()) + list(rand_dists.keys())))
    
    results = {
        'paraphrase_distances': {str(li): round(para_dists.get(li, 0), 4) for li in layers_sorted},
        'paraphrase_cosine': {str(li): round(para_cos.get(li, 0), 4) for li in layers_sorted},
        'random_distances': {str(li): round(rand_dists.get(li, 0), 4) for li in layers_sorted},
        'random_cosine': {str(li): round(rand_cos.get(li, 0), 4) for li in layers_sorted},
    }
    
    # Compute convergence rate for paraphrase pairs
    para_d_list = [(li, para_dists[li]) for li in layers_sorted if li in para_dists]
    if len(para_d_list) >= 3:
        # Fit linear regression to distance vs layer
        X = np.array([li for li, _ in para_d_list], dtype=np.float64)
        Y = np.array([d for _, d in para_d_list], dtype=np.float64)
        if np.std(X) > 0:
            slope = float(np.polyfit(X, Y, 1)[0])
            results['paraphrase_convergence_slope'] = round(slope, 4)
    
    # Same for random pairs
    rand_d_list = [(li, rand_dists[li]) for li in layers_sorted if li in rand_dists]
    if len(rand_d_list) >= 3:
        X = np.array([li for li, _ in rand_d_list], dtype=np.float64)
        Y = np.array([d for _, d in rand_d_list], dtype=np.float64)
        if np.std(X) > 0:
            slope = float(np.polyfit(X, Y, 1)[0])
            results['random_convergence_slope'] = round(slope, 4)
    
    # Print summary
    print(f"\n  ★★★ Trajectory Topology Summary ★★★", flush=True)
    if para_d_list:
        print(f"    Paraphrase: Distance L0={para_dists.get(0,0):.2f} → L_last={para_dists.get(n_layers,0):.2f}", flush=True)
        print(f"    Paraphrase: Cosine L0={para_cos.get(0,0):.4f} → L_last={para_cos.get(n_layers,0):.4f}", flush=True)
    if rand_d_list:
        print(f"    Random: Distance L0={rand_dists.get(0,0):.2f} → L_last={rand_dists.get(n_layers,0):.2f}", flush=True)
        print(f"    Random: Cosine L0={rand_cos.get(0,0):.4f} → L_last={rand_cos.get(n_layers,0):.4f}", flush=True)
    
    para_slope = results.get('paraphrase_convergence_slope', 0)
    rand_slope = results.get('random_convergence_slope', 0)
    print(f"    Paraphrase convergence slope: {para_slope:.4f} (negative=converge)", flush=True)
    print(f"    Random convergence slope: {rand_slope:.4f} (negative=converge)", flush=True)
    
    if para_slope < 0 and rand_slope >= 0:
        print(f"    ★★★ RESULT: Paraphrases CONVERGE, Random pairs DIVERGE → Attractor basin is REAL ★★★", flush=True)
    elif para_slope < rand_slope:
        print(f"    ★ RESULT: Paraphrases converge MORE than random → Partial attractor structure ★", flush=True)
    else:
        print(f"    ✗ No clear convergence pattern", flush=True)
    
    return results


# =====================================================================
# ★★★ MAIN ★★★
# =====================================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    print(f"\n{'='*70}", flush=True)
    print(f"  Phase 180: 约束场动力学 — 从'信号'到'可行域'", flush=True)
    print(f"  Model: {model_name}", flush=True)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}", flush=True)
    print(f"{'='*70}", flush=True)
    
    # ---- 1. Load Model ----
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    print(f"  Model: {info.model_class}, Layers={n_layers}, d_model={info.d_model}, vocab={info.vocab_size}", flush=True)
    
    # ---- 2. Get W_U and final norm ----
    print("  Loading W_U and final norm...", flush=True)
    W_U_np = get_W_U(model, model_name)  # [vocab_size, d_model]
    W_U_torch = torch.tensor(W_U_np, dtype=torch.float32, device=device)
    norm_layer = get_final_norm(model)
    print(f"  W_U: {W_U_torch.shape}, Norm layer: {type(norm_layer).__name__ if norm_layer else 'None'}", flush=True)
    
    # Free W_U numpy
    del W_U_np
    gc.collect()
    
    # ---- 3. Run Experiments ----
    
    # Exp1: Feasible Region Collapse
    t1 = time.time()
    exp1_results = experiment1_feasible_region(model, tokenizer, device, n_layers, norm_layer, W_U_torch)
    t_exp1 = time.time() - t1
    print(f"\n  Exp1 completed in {t_exp1:.1f}s", flush=True)
    
    # Exp2: Constraint Bifurcation
    t2 = time.time()
    exp2_results = experiment2_bifurcation(model, tokenizer, device, n_layers, norm_layer, W_U_torch)
    t_exp2 = time.time() - t2
    print(f"\n  Exp2 completed in {t_exp2:.1f}s", flush=True)
    
    # Exp3: Trajectory Topology
    t3 = time.time()
    exp3_results = experiment3_trajectory_topology(model, tokenizer, device, n_layers)
    t_exp3 = time.time() - t3
    print(f"\n  Exp3 completed in {t_exp3:.1f}s", flush=True)
    
    # ---- 4. Compile Final Results ----
    final_results = {
        'model': model_name,
        'n_layers': n_layers,
        'd_model': info.d_model,
        'vocab_size': info.vocab_size,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M'),
        'exp1_feasible_region': {},
        'exp2_bifurcation': exp2_results,
        'exp3_topology': exp3_results,
        'timing': {
            'exp1': round(t_exp1, 1),
            'exp2': round(t_exp2, 1),
            'exp3': round(t_exp3, 1),
        }
    }
    
    # Convert exp1 results (layer keys are ints, need to convert to strings for JSON)
    for cat_name, layer_data in exp1_results.items():
        final_results['exp1_feasible_region'][cat_name] = {
            str(li): stats for li, stats in layer_data.items()
        }
    
    # ---- 5. ★★★ Key Findings Summary ★★★
    print(f"\n\n{'='*70}", flush=True)
    print(f"  ★★★ PHASE 180 KEY FINDINGS ★★★", flush=True)
    print(f"{'='*70}", flush=True)
    
    # Exp1: Entropy trend
    print(f"\n  --- Exp1: Feasible Region Collapse ---", flush=True)
    for cat_name in ['grammar', 'physical', 'animacy', 'causal']:
        cat_data = exp1_results.get(cat_name, {})
        l0 = cat_data.get(0, {})
        l_last = cat_data.get(n_layers, {})
        
        e_correct_0 = l0.get('correct_entropy', 0)
        e_correct_last = l_last.get('correct_entropy', 0)
        e_incorrect_0 = l0.get('incorrect_entropy', 0)
        e_incorrect_last = l_last.get('incorrect_entropy', 0)
        
        f_correct_0 = l0.get('correct_feasible', 0)
        f_correct_last = l_last.get('correct_feasible', 0)
        
        m_correct_0 = l0.get('correct_margin', 0)
        m_correct_last = l_last.get('correct_margin', 0)
        
        entropy_trend = "↓ SHRINKING" if e_correct_last < e_correct_0 else "↑ EXPANDING"
        feasible_trend = "↓ SHRINKING" if f_correct_last < f_correct_0 else "↑ EXPANDING"
        
        print(f"    {cat_name}: Entropy L0={e_correct_0:.2f} → L_last={e_correct_last:.2f} ({entropy_trend})", flush=True)
        print(f"    {cat_name}: Feasible L0={f_correct_0:.0f} → L_last={f_correct_last:.0f} ({feasible_trend})", flush=True)
        print(f"    {cat_name}: Margin L0={m_correct_0:.2f} → L_last={m_correct_last:.2f}", flush=True)
    
    # Exp2: Phase transition layers
    print(f"\n  --- Exp2: Constraint Bifurcation ---", flush=True)
    for cat_name in ['grammar', 'physical', 'animacy', 'causal']:
        bif = exp2_results.get(cat_name, {})
        ptl = bif.get('phase_transition_layer', 'N/A')
        sharp = bif.get('bifurcation_sharpness', 0)
        sharp_l = bif.get('sharpness_layer', 'N/A')
        print(f"    {cat_name}: Phase transition at L{ptl}, Sharpness={sharp:.4f} at L{sharp_l}", flush=True)
    
    # Exp3: Trajectory topology
    print(f"\n  --- Exp3: Trajectory Topology ---", flush=True)
    para_slope = exp3_results.get('paraphrase_convergence_slope', 0)
    rand_slope = exp3_results.get('random_convergence_slope', 0)
    para_cos_last = list(exp3_results.get('paraphrase_cosine', {}).values())
    rand_cos_last = list(exp3_results.get('random_cosine', {}).values())
    
    print(f"    Paraphrase convergence slope: {para_slope:.4f}", flush=True)
    print(f"    Random convergence slope: {rand_slope:.4f}", flush=True)
    if para_cos_last:
        print(f"    Paraphrase cosine L_last: {para_cos_last[-1]:.4f}", flush=True)
    if rand_cos_last:
        print(f"    Random cosine L_last: {rand_cos_last[-1]:.4f}", flush=True)
    
    # ---- 6. Save Results ----
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_short = {'qwen3': 'qwen3', 'glm4': 'glm4', 'deepseek7b': 'deepseek7b'}[model_name]
    result_path = f"tests/glm5_temp/phase180_{model_short}_{timestamp}.json"
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {result_path}", flush=True)
    
    # ---- 7. Release Model ----
    del W_U_torch
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    
    total_time = time.time() - t0
    print(f"\n  Total time: {total_time:.1f}s", flush=True)
    print(f"  Phase 180 complete!", flush=True)


if __name__ == "__main__":
    main()
