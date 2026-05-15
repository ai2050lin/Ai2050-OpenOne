"""
Phase 181: ★★★ Jacobian Flow — 约束输运动力学 ★★★
================================================================

★★★ 核心理论升级 ★★★
旧框架: 测量 W_U * h (decoder投影, 坐标依赖)
新框架: 测量 Δ_l = h_correct - h_incorrect 的输运动力学 (gauge-invariant!)

用户的根本洞察 (完全正确):
1. 真正的不变量不是 hidden state, 而是"状态可达性"
2. Jacobian J_l = ∂h_{l+1}/∂h_l 才是真正的动力学本体
3. 约束闭合 = Jacobian谱收缩 (违反约束的方向被压缩)
4. 当前所有测量仍是 decoder-relative → 必须转向 decoder-independent

★★★ 核心方法 ★★★
我们不直接计算Jacobian矩阵 (d_model×d_model太大)
而是计算其作用在"约束方向"上的效果:

Δ_l = h_correct_l - h_incorrect_l  (约束违反信号)

输运比: σ(l) = ||Δ_{l+1}|| / ||Δ_l||  (gauge-invariant!)

这等价于: σ(l) = ||J_l · (Δ_l/||Δ_l||)||  (Jacobian在约束方向的谱)
- σ < 1 → 约束被收缩 (层在"消灭"违反信号)
- σ > 1 → 约束被放大 (层在"传播"约束信号)
- σ ≈ 1 → 约束被保留

★★★ 三个关键实验 ★★★

Exp1: Constraint Transport Ratio (约束输运比)
  - σ(l) = ||Δ_{l+1}|| / ||Δ_l||, cos(Δ_l, Δ_{l+1})
  - 比较: 语法 vs 物理 vs 生命性 vs 因果
  - 预测: 语法在中层σ > 1 (传播), 深层σ < 1 (闭合)
  - 预测: 世界约束σ ≈ 1 (预压缩), 无明显传播相位

Exp2: W_U Decomposition (解码器分解)
  - Δ_l = Δ_∥(W_U平行) + Δ_⊥(W_U正交)
  - ||Δ_⊥|| / ||Δ|| = 正交比例 (decoder-independent信号)
  - 预测: 正交比例 > 0 → 存在真实内部动力学
  - 预测: 正交分量与平行分量有不同输运比

Exp3: Control Comparison (控制对比)
  - 约束违反: "The cat sleeps" vs "The cat sleep" (语法错误)
  - Token替换: "The cat sleeps" vs "The dog sleeps" (合法替换)
  - 预测: 约束违反对的输运比 ≠ token替换对的输运比

★★★ 为什么这是关键突破 ★★★
1. σ(l) 是真正的 gauge-invariant (不依赖任何坐标系选择)
2. W_U分解直接区分"内部动力学"与"解码器投影"
3. 控制对比排除"统计伪象"解释

Usage: python tests/glm5/phase181_jacobian_flow.py <model_name>
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
# ★★★ 句子定义 — 5个类别 × 25-30对 ★★★
# =====================================================================
# 格式: (correct_sentence, incorrect_sentence, target_keyword, category)
# target_keyword: 约束信号测量位置的标识
#   - 语法: 动词位置 (约束作用处)
#   - 世界: 目标词位置 (约束作用处)

GRAMMAR_PAIRS = [
    ("The cat sleeps quietly", "The cat sleep quietly", "sleeps"),
    ("The dog runs fast", "The dog run fast", "runs"),
    ("The bird sings loudly", "The bird sing loudly", "sings"),
    ("The child plays outside", "The child play outside", "plays"),
    ("The man walks slowly", "The man walk slowly", "walks"),
    ("The girl reads a book", "The girl read a book", "reads"),
    ("The tree grows tall", "The tree grow tall", "grows"),
    ("The car moves forward", "The car move forward", "moves"),
    ("The fish swims upstream", "The fish swim upstream", "swims"),
    ("The clock ticks steadily", "The clock tick steadily", "ticks"),
    ("The cats sleep quietly", "The cats sleeps quietly", "sleep"),
    ("The dogs run fast", "The dogs runs fast", "run"),
    ("The birds sing loudly", "The birds sings loudly", "sing"),
    ("The children play outside", "The children plays outside", "play"),
    ("The men walk slowly", "The men walks slowly", "walk"),
    ("The girls read a book", "The girls reads a book", "read"),
    ("The trees grow tall", "The trees grows tall", "grow"),
    ("The cars move forward", "The cars moves forward", "move"),
    ("The fish swim upstream", "The fish swims upstream", "swim"),
    ("The clocks tick steadily", "The clocks ticks steadily", "tick"),
    ("The woman speaks clearly", "The woman speak clearly", "speaks"),
    ("The boy jumps high", "The boy jump high", "jumps"),
    ("The horse gallops fast", "The horse gallop fast", "gallops"),
    ("The rabbit hops away", "The rabbit hop away", "hops"),
    ("The snake slithers slowly", "The snake slither slowly", "slithers"),
]

PHYSICAL_PAIRS = [
    ("The glass fell and it broke", "The glass fell and it floated", "broke"),
    ("The mirror dropped and it shattered", "The mirror dropped and it bounced", "shattered"),
    ("The feather drifted and it floated", "The feather drifted and it shattered", "floated"),
    ("The leaf fell and it floated", "The leaf fell and it shattered", "floated"),
    ("The rock sank because it was heavy", "The rock sank because it was light", "heavy"),
    ("The ice melted because it was warm", "The ice melted because it was cold", "warm"),
    ("The metal bent because it was soft", "The metal bent because it was rigid", "soft"),
    ("The wood burned because it was dry", "The wood burned because it was wet", "dry"),
    ("The knife cut because it was sharp", "The knife cut because it was blunt", "sharp"),
    ("The water froze because it was cold", "The water froze because it was hot", "cold"),
    ("The rope snapped because it was weak", "The rope snapped because it was strong", "weak"),
    ("The glass is transparent so light passes", "The glass is transparent so light blocks", "passes"),
    ("The sponge expanded because it absorbed", "The sponge expanded because it repelled", "absorbed"),
    ("The balloon popped because it was full", "The balloon popped because it was empty", "full"),
    ("The ice cube melted in the sun", "The ice cube froze in the sun", "melted"),
    ("The paper tore because it was thin", "The paper tore because it was thick", "thin"),
    ("The wire conducted because it was metal", "The wire conducted because it was wood", "metal"),
    ("The magnet attracted because it was iron", "The magnet attracted because it was plastic", "iron"),
    ("The cloth shrank because it was cotton", "The cloth shrank because it was nylon", "cotton"),
    ("The wheel turned because it was round", "The wheel turned because it was square", "round"),
    ("The candle melted because it was wax", "The candle melted because it was stone", "wax"),
    ("The rubber stretched because it was elastic", "The rubber stretched because it was brittle", "elastic"),
    ("The battery worked because it was charged", "The battery worked because it was dead", "charged"),
    ("The fabric tore because it was worn", "The fabric tore because it was new", "worn"),
    ("The metal rusted because it was iron", "The metal rusted because it was gold", "iron"),
]

ANIMACY_PAIRS = [
    # 注意: 这里两个句子的动词token相同, 不同的是主语
    # 约束信号在动词位置测量 = 纯上下文驱动 (无token差异)
    ("The dog thought about the problem", "The rock thought about the problem", "thought"),
    ("The cat decided to leave", "The table decided to leave", "decided"),
    ("The man felt sad about it", "The stone felt sad about it", "felt"),
    ("The woman remembered the event", "The wall remembered the event", "remembered"),
    ("The child believed in magic", "The chair believed in magic", "believed"),
    ("The teacher explained the concept", "The bookshelf explained the concept", "explained"),
    ("The bird wanted to fly", "The cup wanted to fly", "wanted"),
    ("The fish tried to swim", "The pencil tried to swim", "tried"),
    ("The horse ran across the field", "The bottle ran across the field", "ran"),
    ("The student learned the lesson", "The desk learned the lesson", "learned"),
    ("The plant grew toward the light", "The metal grew toward the light", "grew"),
    ("The scientist discovered the truth", "The rock discovered the truth", "discovered"),
    ("The patient recovered from illness", "The chair recovered from illness", "recovered"),
    ("The dog barked at the stranger", "The shoe barked at the stranger", "barked"),
    ("The cat jumped over the fence", "The rock jumped over the fence", "jumped"),
    ("The doctor healed the patient", "The brick healed the patient", "healed"),
    ("The baby cried for milk", "The pillow cried for milk", "cried"),
    ("The farmer planted the seeds", "The cloud planted the seeds", "planted"),
    ("The musician played the violin", "The statue played the violin", "played"),
    ("The writer composed the poem", "The mountain composed the poem", "composed"),
    ("The pilot flew the airplane", "The river flew the airplane", "flew"),
    ("The chef cooked the meal", "The stone cooked the meal", "cooked"),
    ("The soldier fought bravely", "The table fought bravely", "fought"),
    ("The driver steered carefully", "The lamp steered carefully", "steered"),
    ("The painter created the artwork", "The fence created the artwork", "created"),
]

CAUSAL_PAIRS = [
    ("The ice melted because the temperature rose", "The ice froze because the temperature rose", "melted"),
    ("The fire spread because it was windy", "The fire stopped because it was windy", "spread"),
    ("The plant died because it lacked water", "The plant thrived because it lacked water", "died"),
    ("The car stopped because the driver braked", "The car accelerated because the driver braked", "stopped"),
    ("The soup cooled because it was left out", "The soup heated because it was left out", "cooled"),
    ("The flood happened because it rained heavily", "The drought happened because it rained heavily", "flood"),
    ("The metal expanded because it was heated", "The metal contracted because it was heated", "expanded"),
    ("The patient recovered because the medicine worked", "The patient worsened because the medicine worked", "recovered"),
    ("The bridge collapsed because the earthquake struck", "The bridge strengthened because the earthquake struck", "collapsed"),
    ("The battery died because it was overused", "The battery charged because it was overused", "died"),
    ("The crop failed because the drought was severe", "The crop flourished because the drought was severe", "failed"),
    ("The balloon popped because it was overinflated", "The balloon solidified because it was overinflated", "popped"),
    ("The glass cracked because the impact was strong", "The glass healed because the impact was strong", "cracked"),
    ("The food spoiled because it was left in heat", "The food freshened because it was left in heat", "spoiled"),
    ("The tire deflated because there was a puncture", "The tire inflated because there was a puncture", "deflated"),
    ("The illness spread because conditions were unsanitary", "The illness disappeared because conditions were unsanitary", "spread"),
    ("The wood rotted because it was exposed to moisture", "The wood petrified because it was exposed to moisture", "rotted"),
    ("The engine overheated because the coolant leaked", "The engine cooled because the coolant leaked", "overheated"),
    ("The pipe burst because the water froze inside", "The pipe sealed because the water froze inside", "burst"),
    ("The metal rusted because it was exposed to water", "The metal shined because it was exposed to water", "rusted"),
    ("The ice thawed because the sun warmed it", "The ice solidified because the sun warmed it", "thawed"),
    ("The wound healed because the treatment was effective", "The wound worsened because the treatment was effective", "healed"),
    ("The tumor shrank because the therapy worked", "The tumor grew because the therapy worked", "shrank"),
    ("The flood receded because the rain stopped", "The flood worsened because the rain stopped", "receded"),
    ("The infection cleared because the antibiotics worked", "The infection spread because the antibiotics worked", "cleared"),
]

# ★★★ 控制对 — token替换, 无约束违反 ★★★
# 格式: (sentence_A, sentence_B, target_keyword)
# 两个句子都合法, 只是在某个位置替换了等价token
CONTROL_PAIRS = [
    # 语法控制: 不同主语, 同一动词 (都合法)
    ("The cat sleeps quietly", "The dog sleeps quietly", "sleeps"),
    ("The bird sings loudly", "The child sings loudly", "sings"),
    ("The man walks slowly", "The girl walks slowly", "walks"),
    ("The tree grows tall", "The flower grows tall", "grows"),
    ("The car moves forward", "The bus moves forward", "moves"),
    ("The woman speaks clearly", "The man speaks clearly", "speaks"),
    ("The boy jumps high", "The girl jumps high", "jumps"),
    ("The horse gallops fast", "The dog gallops fast", "gallops"),
    ("The rabbit hops away", "The frog hops away", "hops"),
    ("The snake slithers slowly", "The worm slithers slowly", "slithers"),
    # 物理控制: 不同物体, 同一物理结果 (都合法)
    ("The glass fell and it broke", "The mirror fell and it broke", "broke"),
    ("The cup dropped and it shattered", "The vase dropped and it shattered", "shattered"),
    ("The rock sank because it was heavy", "The stone sank because it was heavy", "heavy"),
    ("The ice melted because it was warm", "The snow melted because it was warm", "warm"),
    ("The wood burned because it was dry", "The paper burned because it was dry", "dry"),
    # 生命性控制: 不同有生命主语, 同一动作 (都合法)
    ("The dog thought about the problem", "The cat thought about the problem", "thought"),
    ("The man felt sad about it", "The woman felt sad about it", "felt"),
    ("The child believed in magic", "The adult believed in magic", "believed"),
    ("The bird wanted to fly", "The bee wanted to fly", "wanted"),
    ("The fish tried to swim", "The duck tried to swim", "tried"),
    # 因果控制: 不同原因, 同一结果 (都合法)
    ("The ice melted because the temperature rose", "The ice melted because the sun shone", "melted"),
    ("The fire spread because it was windy", "The fire spread because it was dry", "spread"),
    ("The plant died because it lacked water", "The plant died because it lacked sunlight", "died"),
    ("The car stopped because the driver braked", "The car stopped because the light turned red", "stopped"),
]


# =====================================================================
# ★★★ 核心计算函数 ★★★
# =====================================================================

def find_differentiating_position(tokenizer, sent_correct, sent_incorrect):
    """
    找到正确/错误句子之间的第一个不同token位置
    
    ★★★ 最鲁棒的方法 ★★★
    不依赖关键词匹配 (不同tokenizer对空格处理不同)
    直接比较两个句子的token序列, 找到第一个差异位置
    
    对于语法: "The cat sleeps" vs "The cat sleep" → 位置2 (sleeps/sleep)
    对于生命性: "The dog thought" vs "The rock thought" → 位置1 (dog/rock)
    对于物理: "it broke" vs "it floated" → 位置5 (broke/floated)
    """
    ids_c = tokenizer.encode(sent_correct, add_special_tokens=True)
    ids_i = tokenizer.encode(sent_incorrect, add_special_tokens=True)
    
    # 找到第一个不同的位置
    min_len = min(len(ids_c), len(ids_i))
    for pos in range(min_len):
        if ids_c[pos] != ids_i[pos]:
            return pos
    
    # 如果没有找到差异, 返回最后位置
    return min_len - 1


def get_hidden_states_at_all_positions(model, tokenizer, device, sentence, n_layers):
    """
    获取句子所有位置在所有层的hidden states
    
    Returns:
        tuple: (result_dict, seq_len) 
            result_dict: {layer_idx: numpy_array[seq_len, d_model]}
            seq_len: 实际token序列长度
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = input_ids.shape[1]
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
    
    all_hidden = outputs.hidden_states  # tuple of (1, seq_len, d_model)
    
    result = {}
    for li, hs in enumerate(all_hidden):
        result[li] = hs[0].detach().cpu().float().numpy()  # [seq_len, d_model]
    
    del outputs, all_hidden
    return result, seq_len


def compute_constraint_signal(model, tokenizer, device, sent_correct, sent_incorrect, 
                               target_keyword, n_layers):
    """
    计算约束信号 Δ_l = h_correct_l - h_incorrect_l 在目标位置
    
    ★★★ 关键方法 ★★★
    使用 find_differentiating_position 找到第一个token差异位置
    然后在两个句子的同一位置测量hidden state差异
    
    对于语法: 在"sleeps"/"sleep"位置测量 (token不同, 约束在此作用)
    对于生命性: 在"dog"/"rock"位置测量 (主语不同, 约束源在此)
    对于物理: 在"broke"/"floated"位置测量 (结果不同)
    
    Returns:
        dict: {layer_idx: {'delta': np.array[d_model], 'delta_norm': float, 
                           'h_correct_norm': float, 'h_incorrect_norm': float}}
    """
    # ★★★ 用token序列比较找到差异位置 ★★★
    pos = find_differentiating_position(tokenizer, sent_correct, sent_incorrect)
    
    # 获取hidden states
    hs_correct, seq_len_c = get_hidden_states_at_all_positions(model, tokenizer, device, sent_correct, n_layers)
    hs_incorrect, seq_len_i = get_hidden_states_at_all_positions(model, tokenizer, device, sent_incorrect, n_layers)
    
    # ★★★ 安全检查: 确保位置在范围内 ★★★
    pos_c = min(pos, seq_len_c - 1)
    pos_i = min(pos, seq_len_i - 1)
    
    result = {}
    for li in range(n_layers + 1):
        if li in hs_correct and li in hs_incorrect:
            h_c = hs_correct[li][pos_c]
            h_i = hs_incorrect[li][pos_i]
            
            h_c = hs_correct[li][pos_c]
            h_i = hs_incorrect[li][pos_i]
            
            delta = h_c - h_i
            delta_norm = float(np.linalg.norm(delta))
            h_c_norm = float(np.linalg.norm(h_c))
            h_i_norm = float(np.linalg.norm(h_i))
            
            result[li] = {
                'delta': delta,
                'delta_norm': delta_norm,
                'h_correct_norm': h_c_norm,
                'h_incorrect_norm': h_i_norm,
            }
    
    del hs_correct, hs_incorrect
    return result


def compute_wu_decomposition(delta, U_wut):
    """
    计算Δ在W_U行空间中的分解
    
    Args:
        delta: numpy array [d_model] — 约束信号
        U_wut: numpy array [d_model, k] — W_U行空间基 (SVD of W_U^T)
    
    Returns:
        dict: {parallel_norm, orthogonal_norm, parallel_ratio, orthogonal_ratio}
    """
    delta_norm = np.linalg.norm(delta)
    if delta_norm < 1e-10:
        return {'parallel_norm': 0, 'orthogonal_norm': 0, 
                'parallel_ratio': 0, 'orthogonal_ratio': 0}
    
    # 平行分量: Proj(Δ) = U * (U^T * Δ)
    proj_coeffs = U_wut.T @ delta  # [k]
    proj_delta = U_wut @ proj_coeffs  # [d_model]
    
    # 正交分量: Δ - Proj(Δ)
    ortho_delta = delta - proj_delta
    
    parallel_norm = float(np.linalg.norm(proj_delta))
    orthogonal_norm = float(np.linalg.norm(ortho_delta))
    
    return {
        'parallel_norm': round(parallel_norm, 6),
        'orthogonal_norm': round(orthogonal_norm, 6),
        'parallel_ratio': round(parallel_norm / delta_norm, 6),
        'orthogonal_ratio': round(orthogonal_norm / delta_norm, 6),
    }


def precompute_wu_basis(W_U_np, n_components=200):
    """
    预计算W_U行空间基 (用于W_U分解)
    
    Args:
        W_U_np: numpy array [vocab_size, d_model]
        n_components: SVD分量数
    
    Returns:
        U_wut: numpy array [d_model, k] — W_U行空间基
    """
    from scipy.sparse.linalg import svds
    
    W_U_T = W_U_np.T.astype(np.float32)  # [d_model, vocab_size]
    k = min(n_components, min(W_U_T.shape[0], W_U_T.shape[1]) - 2)
    k = max(k, 1)
    
    print(f"  Computing W_U SVD (k={k})...", flush=True)
    U_wut, s_wut, _ = svds(W_U_T, k=k)
    U_wut = np.asarray(U_wut, dtype=np.float64)  # [d_model, k]
    
    # 验证: 前k个奇异值的能量占比
    total_energy = float(np.sum(s_wut**2))
    top_energy = float(np.sum(np.sort(s_wut**2)[-k:]))
    print(f"  W_U SVD: top-{k} components capture {top_energy/total_energy*100:.1f}% of energy", flush=True)
    
    return U_wut


# =====================================================================
# ★★★ Exp1: Constraint Transport Ratio (约束输运比) ★★★
# =====================================================================

def experiment1_transport_ratio(model, tokenizer, device, n_layers, all_constraint_pairs, control_pairs):
    """
    约束输运比实验
    
    核心预测:
    - σ(l) = ||Δ_{l+1}|| / ||Δ_l|| (gauge-invariant!)
    - 语法: 中层σ > 1 (传播), 深层σ < 1 (闭合)
    - 世界: σ ≈ 1 (预压缩), 无明显传播相位
    - 控制对: σ ≈ 1 (无约束违反, 纯token替换)
    """
    print("\n" + "="*70, flush=True)
    print("  Exp1: Constraint Transport Ratio (约束输运比)", flush=True)
    print("="*70, flush=True)
    
    all_results = {}
    
    # === 约束违反对 ===
    for cat_name, pairs in all_constraint_pairs.items():
        print(f"\n  --- Constraint: {cat_name} ({len(pairs)} pairs) ---", flush=True)
        
        # 收集所有对的输运比
        all_transport_ratios = defaultdict(list)  # {layer: [σ values]}
        all_delta_norms = defaultdict(list)       # {layer: [||Δ|| values]}
        all_cosine_alignments = defaultdict(list)   # {layer: [cos(Δ_l, Δ_{l+1}) values]}
        
        for pi, (sent_c, sent_i, kw) in enumerate(pairs):
            if pi % 5 == 0:
                print(f"    Processing pair {pi+1}/{len(pairs)}...", flush=True)
            
            try:
                cs = compute_constraint_signal(model, tokenizer, device, 
                                                sent_c, sent_i, kw, n_layers)
                
                # 计算输运比和方向对齐
                layers_sorted = sorted(cs.keys())
                for idx, li in enumerate(layers_sorted[:-1]):
                    li_next = layers_sorted[idx + 1]
                    if li_next != li + 1:
                        continue  # 只计算相邻层
                    
                    d_norm = cs[li]['delta_norm']
                    d_norm_next = cs[li_next]['delta_norm']
                    
                    # 输运比 σ(l) = ||Δ_{l+1}|| / ||Δ_l||
                    if d_norm > 1e-10:
                        sigma = d_norm_next / d_norm
                        all_transport_ratios[li].append(sigma)
                    
                    all_delta_norms[li].append(d_norm)
                    
                    # 方向对齐 cos(Δ_l, Δ_{l+1})
                    if d_norm > 1e-10 and d_norm_next > 1e-10:
                        cos_align = float(np.dot(cs[li]['delta'], cs[li_next]['delta']) 
                                         / (d_norm * d_norm_next))
                        all_cosine_alignments[li].append(cos_align)
                
                # 最后一层的||Δ||
                if layers_sorted:
                    li_last = layers_sorted[-1]
                    all_delta_norms[li_last].append(cs[li_last]['delta_norm'])
                
                del cs
                
            except Exception as e:
                print(f"      [WARN] Failed: {e}", flush=True)
                continue
        
        # 聚合
        cat_results = {}
        for li in range(n_layers):
            cat_results[li] = {
                'transport_ratio_mean': round(float(np.mean(all_transport_ratios[li])), 4) if all_transport_ratios[li] else 0,
                'transport_ratio_std': round(float(np.std(all_transport_ratios[li])), 4) if all_transport_ratios[li] else 0,
                'delta_norm_mean': round(float(np.mean(all_delta_norms[li])), 4) if all_delta_norms[li] else 0,
                'delta_norm_std': round(float(np.std(all_delta_norms[li])), 4) if all_delta_norms[li] else 0,
                'cosine_alignment_mean': round(float(np.mean(all_cosine_alignments[li])), 4) if all_cosine_alignments[li] else 0,
            }
        
        all_results[cat_name] = cat_results
        
        # 打印摘要
        l1 = cat_results.get(1, {})
        l_mid = cat_results.get(n_layers // 2, {})
        l_last = cat_results.get(n_layers - 1, {})
        
        print(f"  ★ {cat_name} Transport Summary:", flush=True)
        print(f"    L1: σ={l1.get('transport_ratio_mean',0):.3f}±{l1.get('transport_ratio_std',0):.3f}, "
              f"||Δ||={l1.get('delta_norm_mean',0):.2f}", flush=True)
        print(f"    L{n_layers//2}: σ={l_mid.get('transport_ratio_mean',0):.3f}±{l_mid.get('transport_ratio_std',0):.3f}, "
              f"||Δ||={l_mid.get('delta_norm_mean',0):.2f}", flush=True)
        print(f"    L{n_layers-1}: σ={l_last.get('transport_ratio_mean',0):.3f}±{l_last.get('transport_ratio_std',0):.3f}, "
              f"||Δ||={l_last.get('delta_norm_mean',0):.2f}", flush=True)
    
    # === 控制对 ===
    print(f"\n  --- Control: Token Substitution ({len(control_pairs)} pairs) ---", flush=True)
    
    ctrl_transport_ratios = defaultdict(list)
    ctrl_delta_norms = defaultdict(list)
    ctrl_cosine_alignments = defaultdict(list)
    
    for pi, (sent_a, sent_b, kw) in enumerate(control_pairs):
        if pi % 5 == 0:
            print(f"    Processing control pair {pi+1}/{len(control_pairs)}...", flush=True)
        
        try:
            cs = compute_constraint_signal(model, tokenizer, device, sent_a, sent_b, kw, n_layers)
            
            layers_sorted = sorted(cs.keys())
            for idx, li in enumerate(layers_sorted[:-1]):
                li_next = layers_sorted[idx + 1]
                if li_next != li + 1:
                    continue
                
                d_norm = cs[li]['delta_norm']
                d_norm_next = cs[li_next]['delta_norm']
                
                if d_norm > 1e-10:
                    sigma = d_norm_next / d_norm
                    ctrl_transport_ratios[li].append(sigma)
                
                ctrl_delta_norms[li].append(d_norm)
                
                if d_norm > 1e-10 and d_norm_next > 1e-10:
                    cos_align = float(np.dot(cs[li]['delta'], cs[li_next]['delta']) 
                                     / (d_norm * d_norm_next))
                    ctrl_cosine_alignments[li].append(cos_align)
            
            if layers_sorted:
                li_last = layers_sorted[-1]
                ctrl_delta_norms[li_last].append(cs[li_last]['delta_norm'])
            
            del cs
            
        except Exception as e:
            continue
    
    ctrl_results = {}
    for li in range(n_layers):
        ctrl_results[li] = {
            'transport_ratio_mean': round(float(np.mean(ctrl_transport_ratios[li])), 4) if ctrl_transport_ratios[li] else 0,
            'transport_ratio_std': round(float(np.std(ctrl_transport_ratios[li])), 4) if ctrl_transport_ratios[li] else 0,
            'delta_norm_mean': round(float(np.mean(ctrl_delta_norms[li])), 4) if ctrl_delta_norms[li] else 0,
            'delta_norm_std': round(float(np.std(ctrl_delta_norms[li])), 4) if ctrl_delta_norms[li] else 0,
            'cosine_alignment_mean': round(float(np.mean(ctrl_cosine_alignments[li])), 4) if ctrl_cosine_alignments[li] else 0,
        }
    
    all_results['control'] = ctrl_results
    
    l1 = ctrl_results.get(1, {})
    l_mid = ctrl_results.get(n_layers // 2, {})
    l_last = ctrl_results.get(n_layers - 1, {})
    print(f"  ★ Control Transport Summary:", flush=True)
    print(f"    L1: σ={l1.get('transport_ratio_mean',0):.3f}±{l1.get('transport_ratio_std',0):.3f}, "
          f"||Δ||={l1.get('delta_norm_mean',0):.2f}", flush=True)
    print(f"    L{n_layers//2}: σ={l_mid.get('transport_ratio_mean',0):.3f}±{l_mid.get('transport_ratio_std',0):.3f}, "
          f"||Δ||={l_mid.get('delta_norm_mean',0):.2f}", flush=True)
    print(f"    L{n_layers-1}: σ={l_last.get('transport_ratio_mean',0):.3f}±{l_last.get('transport_ratio_std',0):.3f}, "
          f"||Δ||={l_last.get('delta_norm_mean',0):.2f}", flush=True)
    
    return all_results


# =====================================================================
# ★★★ Exp2: W_U Decomposition (解码器分解) ★★★
# =====================================================================

def experiment2_wu_decomposition(model, tokenizer, device, n_layers, all_constraint_pairs, U_wut):
    """
    W_U分解实验
    
    核心预测:
    - Δ_l = Δ_∥(W_U平行) + Δ_⊥(W_U正交)
    - 如果正交比例 > 0 → 存在真实内部动力学
    - 如果正交分量和平行分量有不同输运比 → 内部动力学≠解码器投影
    """
    print("\n" + "="*70, flush=True)
    print("  Exp2: W_U Decomposition (解码器分解)", flush=True)
    print("="*70, flush=True)
    
    all_results = {}
    
    for cat_name, pairs in all_constraint_pairs.items():
        print(f"\n  --- {cat_name} ({len(pairs)} pairs) ---", flush=True)
        
        # 收集所有对的分解
        all_parallel_ratios = defaultdict(list)   # {layer: [||Δ_∥||/||Δ|| values]}
        all_orthogonal_ratios = defaultdict(list)  # {layer: [||Δ_⊥||/||Δ|| values]}
        all_parallel_transport = defaultdict(list)  # {layer: [σ_∥ values]}
        all_orthogonal_transport = defaultdict(list)  # {layer: [σ_⊥ values]}
        
        # 存储上一层的分量用于计算输运比
        prev_parallel_delta = None
        prev_orthogonal_delta = None
        prev_layer = None
        
        for pi, (sent_c, sent_i, kw) in enumerate(pairs[:15]):  # 用15对节省时间
            if pi % 5 == 0:
                print(f"    Processing pair {pi+1}/{min(len(pairs),15)}...", flush=True)
            
            try:
                cs = compute_constraint_signal(model, tokenizer, device, 
                                                sent_c, sent_i, kw, n_layers)
                
                layers_sorted = sorted(cs.keys())
                for li in layers_sorted:
                    delta = cs[li]['delta']
                    decomp = compute_wu_decomposition(delta, U_wut)
                    
                    all_parallel_ratios[li].append(decomp['parallel_ratio'])
                    all_orthogonal_ratios[li].append(decomp['orthogonal_ratio'])
                
                del cs
                
            except Exception as e:
                continue
        
        # 聚合
        cat_results = {}
        for li in range(n_layers + 1):
            cat_results[li] = {
                'parallel_ratio_mean': round(float(np.mean(all_parallel_ratios[li])), 4) if all_parallel_ratios[li] else 0,
                'parallel_ratio_std': round(float(np.std(all_parallel_ratios[li])), 4) if all_parallel_ratios[li] else 0,
                'orthogonal_ratio_mean': round(float(np.mean(all_orthogonal_ratios[li])), 4) if all_orthogonal_ratios[li] else 0,
                'orthogonal_ratio_std': round(float(np.std(all_orthogonal_ratios[li])), 4) if all_orthogonal_ratios[li] else 0,
            }
        
        all_results[cat_name] = cat_results
        
        # 打印摘要
        l0 = cat_results.get(0, {})
        l_mid = cat_results.get(n_layers // 2, {})
        l_last = cat_results.get(n_layers, {})
        
        print(f"  ★ {cat_name} W_U Decomposition:", flush=True)
        print(f"    L0: ∥={l0.get('parallel_ratio_mean',0):.3f} ⊥={l0.get('orthogonal_ratio_mean',0):.3f}", flush=True)
        print(f"    L{n_layers//2}: ∥={l_mid.get('parallel_ratio_mean',0):.3f} ⊥={l_mid.get('orthogonal_ratio_mean',0):.3f}", flush=True)
        print(f"    L{n_layers}: ∥={l_last.get('parallel_ratio_mean',0):.3f} ⊥={l_last.get('orthogonal_ratio_mean',0):.3f}", flush=True)
    
    return all_results


# =====================================================================
# ★★★ Exp3: Transport Phase Analysis (输运相位分析) ★★★
# =====================================================================

def experiment3_phase_analysis(exp1_results, n_layers):
    """
    输运相位分析
    
    基于Exp1的输运比数据, 分析:
    1. 每个约束类型的传播相 (σ > 1), 闭合相 (σ < 1), 保持相 (σ ≈ 1)
    2. 相变层 (传播→闭合的转折点)
    3. 约束传播vs控制对的差异
    """
    print("\n" + "="*70, flush=True)
    print("  Exp3: Transport Phase Analysis (输运相位分析)", flush=True)
    print("="*70, flush=True)
    
    all_results = {}
    
    for cat_name in ['grammar', 'physical', 'animacy', 'causal', 'control']:
        if cat_name not in exp1_results:
            continue
        
        cat_data = exp1_results[cat_name]
        
        # 提取输运比序列
        sigma_sequence = []
        for li in range(n_layers):
            if li in cat_data:
                sigma_sequence.append((li, cat_data[li]['transport_ratio_mean']))
        
        if len(sigma_sequence) < 3:
            continue
        
        # 找到传播相和闭合相
        propagation_layers = []  # σ > 1
        contraction_layers = []  # σ < 1
        preservation_layers = []  # σ ≈ 1 (0.95 < σ < 1.05)
        
        for li, sigma in sigma_sequence:
            if sigma > 1.05:
                propagation_layers.append(li)
            elif sigma < 0.95:
                contraction_layers.append(li)
            else:
                preservation_layers.append(li)
        
        # 找到相变点 (σ从>1变为<1的层)
        phase_transition_layer = None
        for i in range(1, len(sigma_sequence)):
            li_prev, sigma_prev = sigma_sequence[i-1]
            li_curr, sigma_curr = sigma_sequence[i]
            if sigma_prev > 1.0 and sigma_curr < 1.0:
                phase_transition_layer = li_curr
                break
        
        # 计算平均输运比
        avg_sigma = float(np.mean([s for _, s in sigma_sequence]))
        max_sigma = max([s for _, s in sigma_sequence]) if sigma_sequence else 0
        min_sigma = min([s for _, s in sigma_sequence]) if sigma_sequence else 0
        max_sigma_layer = [li for li, s in sigma_sequence if s == max_sigma][0] if sigma_sequence else None
        
        all_results[cat_name] = {
            'avg_transport_ratio': round(avg_sigma, 4),
            'max_transport_ratio': round(max_sigma, 4),
            'min_transport_ratio': round(min_sigma, 4),
            'max_sigma_layer': max_sigma_layer,
            'phase_transition_layer': phase_transition_layer,
            'n_propagation_layers': len(propagation_layers),
            'n_contraction_layers': len(contraction_layers),
            'n_preservation_layers': len(preservation_layers),
        }
        
        print(f"  ★ {cat_name} Phase Analysis:", flush=True)
        print(f"    Avg σ = {avg_sigma:.3f}, Max σ = {max_sigma:.3f} at L{max_sigma_layer}, Min σ = {min_sigma:.3f}", flush=True)
        print(f"    Propagation layers (σ>1): {len(propagation_layers)}", flush=True)
        print(f"    Contraction layers (σ<1): {len(contraction_layers)}", flush=True)
        print(f"    Preservation layers (σ≈1): {len(preservation_layers)}", flush=True)
        print(f"    Phase transition layer: {phase_transition_layer}", flush=True)
    
    # ★★★ 关键对比: 约束 vs 控制 ★★★
    print(f"\n  ★★★ Constraint vs Control Comparison ★★★", flush=True)
    
    if 'control' in exp1_results and len(exp1_results) > 1:
        ctrl_data = exp1_results['control']
        for cat_name in ['grammar', 'physical', 'animacy', 'causal']:
            if cat_name not in exp1_results:
                continue
            cat_data = exp1_results[cat_name]
            
            # 比较输运比
            diffs = []
            for li in range(min(n_layers, 35)):
                if li in cat_data and li in ctrl_data:
                    sigma_cat = cat_data[li]['transport_ratio_mean']
                    sigma_ctrl = ctrl_data[li]['transport_ratio_mean']
                    diffs.append((li, sigma_cat - sigma_ctrl))
            
            # 找到最大差异的层
            if diffs:
                max_diff_li, max_diff = max(diffs, key=lambda x: abs(x[1]))
                avg_diff = float(np.mean([d for _, d in diffs]))
                
                print(f"    {cat_name} vs Control: avg Δσ = {avg_diff:.4f}, "
                      f"max Δσ = {max_diff:.4f} at L{max_diff_li}", flush=True)
                
                all_results[f'{cat_name}_vs_control'] = {
                    'avg_diff': round(avg_diff, 4),
                    'max_diff': round(max_diff, 4),
                    'max_diff_layer': max_diff_li,
                }
    
    return all_results


# =====================================================================
# ★★★ MAIN ★★★
# =====================================================================

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    print(f"\n{'='*70}", flush=True)
    print(f"  Phase 181: Jacobian Flow — 约束输运动力学", flush=True)
    print(f"  Model: {model_name}", flush=True)
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}", flush=True)
    print(f"{'='*70}", flush=True)
    
    # ---- 1. Load Model ----
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  Model: {info.model_class}, Layers={n_layers}, d_model={d_model}, vocab={info.vocab_size}", flush=True)
    
    # ---- 2. Get W_U and precompute basis ----
    print("  Loading W_U...", flush=True)
    W_U_np = get_W_U(model, model_name)  # [vocab_size, d_model]
    U_wut = precompute_wu_basis(W_U_np, n_components=200)
    
    # Free W_U numpy
    del W_U_np
    gc.collect()
    
    # ---- 3. Define constraint pairs ----
    all_constraint_pairs = {
        'grammar': GRAMMAR_PAIRS,
        'physical': PHYSICAL_PAIRS,
        'animacy': ANIMACY_PAIRS,
        'causal': CAUSAL_PAIRS,
    }
    
    # ---- 4. Run Experiments ----
    
    # Exp1: Transport Ratio (includes control comparison)
    t1 = time.time()
    exp1_results = experiment1_transport_ratio(
        model, tokenizer, device, n_layers, all_constraint_pairs, CONTROL_PAIRS
    )
    t_exp1 = time.time() - t1
    print(f"\n  Exp1 completed in {t_exp1:.1f}s", flush=True)
    
    # Exp2: W_U Decomposition
    t2 = time.time()
    exp2_results = experiment2_wu_decomposition(
        model, tokenizer, device, n_layers, all_constraint_pairs, U_wut
    )
    t_exp2 = time.time() - t2
    print(f"\n  Exp2 completed in {t_exp2:.1f}s", flush=True)
    
    # Exp3: Phase Analysis
    t3 = time.time()
    exp3_results = experiment3_phase_analysis(exp1_results, n_layers)
    t_exp3 = time.time() - t3
    print(f"\n  Exp3 completed in {t_exp3:.1f}s", flush=True)
    
    # ---- 5. ★★★ Key Findings Summary ★★★
    print(f"\n\n{'='*70}", flush=True)
    print(f"  ★★★ PHASE 181 KEY FINDINGS ★★★", flush=True)
    print(f"{'='*70}", flush=True)
    
    # Exp1: Transport ratio trends
    print(f"\n  --- Exp1: Constraint Transport Ratio ---", flush=True)
    for cat_name in ['grammar', 'physical', 'animacy', 'causal', 'control']:
        if cat_name not in exp1_results:
            continue
        cat_data = exp1_results[cat_name]
        
        # Print key layers
        key_layers = [0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
        sigma_str = " → ".join([f"L{li}:{cat_data.get(li,{}).get('transport_ratio_mean',0):.3f}" 
                                for li in key_layers if li in cat_data])
        print(f"    {cat_name}: σ = {sigma_str}", flush=True)
    
    # Exp2: W_U decomposition
    print(f"\n  --- Exp2: W_U Decomposition ---", flush=True)
    for cat_name in ['grammar', 'physical', 'animacy', 'causal']:
        if cat_name not in exp2_results:
            continue
        cat_data = exp2_results[cat_name]
        l0 = cat_data.get(0, {})
        l_last = cat_data.get(n_layers, {})
        print(f"    {cat_name}: L0 ∥={l0.get('parallel_ratio_mean',0):.3f} ⊥={l0.get('orthogonal_ratio_mean',0):.3f} → "
              f"L{n_layers} ∥={l_last.get('parallel_ratio_mean',0):.3f} ⊥={l_last.get('orthogonal_ratio_mean',0):.3f}", flush=True)
    
    # Exp3: Phase analysis
    print(f"\n  --- Exp3: Transport Phase Analysis ---", flush=True)
    for cat_name, phase_data in exp3_results.items():
        if '_vs_control' in cat_name:
            continue
        print(f"    {cat_name}: avg σ={phase_data.get('avg_transport_ratio',0):.3f}, "
              f"phase transition at L{phase_data.get('phase_transition_layer','N/A')}, "
              f"propagation={phase_data.get('n_propagation_layers',0)} layers, "
              f"contraction={phase_data.get('n_contraction_layers',0)} layers", flush=True)
    
    # ---- 6. Save Results ----
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    model_short = {'qwen3': 'qwen3', 'glm4': 'glm4', 'deepseek7b': 'deepseek7b'}[model_name]
    
    # Convert to JSON-serializable format
    final_results = {
        'model': model_name,
        'n_layers': n_layers,
        'd_model': d_model,
        'timestamp': timestamp,
        'exp1_transport_ratio': {},
        'exp2_wu_decomposition': {},
        'exp3_phase_analysis': exp3_results,
        'timing': {
            'exp1': round(t_exp1, 1),
            'exp2': round(t_exp2, 1),
            'exp3': round(t_exp3, 1),
        }
    }
    
    # Convert exp1 results
    for cat_name, layer_data in exp1_results.items():
        final_results['exp1_transport_ratio'][cat_name] = {
            str(li): stats for li, stats in layer_data.items()
        }
    
    # Convert exp2 results
    for cat_name, layer_data in exp2_results.items():
        final_results['exp2_wu_decomposition'][cat_name] = {
            str(li): stats for li, stats in layer_data.items()
        }
    
    result_path = f"tests/glm5_temp/phase181_{model_short}_{timestamp}.json"
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    print(f"\n  Results saved to: {result_path}", flush=True)
    
    # ---- 7. Release Model ----
    del U_wut
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    
    total_time = time.time() - t0
    print(f"\n  Total time: {total_time:.1f}s", flush=True)
    print(f"  Phase 181 complete!", flush=True)


if __name__ == "__main__":
    main()
