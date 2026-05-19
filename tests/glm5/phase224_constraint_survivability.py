"""
Phase 224: 约束存活性与传播模态分析
=====================================

核心理论升级（综合两份分析 + Phase 223结果）：

  Phase 223 发现:
    1. Jacobian谱极度稳定 (Pearson>0.98, Spearman=1.0)
    2. 但JVP方向不如Δh稳定（尤其在深层）
    3. Jacobian是"选择性传播算子"：保留约束方向，抑制随机方向

  两份分析的核心共识:
    - 需要从"向量世界观"转向"算子/传播世界观"
    - 真正稳定的是"传播通道结构"（哪些方向允许传播），不是"具体传播向量"
    - 核心问题: "什么样的扰动能够长期传播"？

  Phase 224 三个关键实验:

  实验1 (★★★★★ P0): 约束存活性 (Constraint Survivability)
    对每层l，注入不同类型的扰动v:
      a. 约束方向: v = Δh（语法约束差分）
      b. 随机方向: v = random
      c. 语义方向: v = Δh_semantic（同语法不同语义的差分）
    传播到最后一层L，测量 survival(v) = ||扰动在L的残存|| / ||原始扰动||
    如果: survival(约束) >> survival(随机) → 语言约束是"可传播模态"

  实验2 (★★★★★): 传播模态结构 (Propagation Mode Structure)
    对不同句子的Jacobian，比较top-k奇异子空间的主角(principal angles)
    如果: 子空间高度对齐 → 传播通道是通用的
    如果: 子空间不对齐但谱一致 → 通道结构通用但基旋转
    
  实验3 (★★★★): 约束选择性指数 (Constraint Selectivity Index)  
    CSI = ||J·d_constraint|| / ||J·v_random|| （对同一层，同一句子）
    如果CSI >> 1 → Jacobian选择性放大约束方向
    
跨模型: Qwen3 → GLM4 → DS7B
BF16 + device_map="auto" + sdpa(flash) + 定期GC
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import numpy as np
import torch
from pathlib import Path
from model_utils import (get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS,
                          get_sample_layers)

OUTPUT_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")

# ============================================================
# 句子生成：多种约束类型
# ============================================================

def generate_constraint_sentences(n_per_type=20):
    """生成多种约束类型的句子对"""
    sentences = {
        "number_sva": [],     # 主谓一致-数
        "tense": [],          # 时态
        "negation": [],       # 否定
        "voice": [],          # 语态（主动/被动）
        "semantic_control": [] # 语义控制（同语法不同词汇）
    }
    
    # === 数约束 SVA ===
    singular_subjects = ["cat", "dog", "bird", "girl", "boy", "tree", "car", "child",
                         "man", "woman", "fish", "horse", "student", "teacher", 
                         "flower", "river", "star", "moon", "sun", "cloud"]
    verbs_s = ["chases", "runs", "sings", "reads", "walks", "falls", "moves",
               "plays", "works", "dances", "swims", "sleeps", "barks", "flies",
               "grows", "blooms", "flows", "shines", "rises", "blows"]
    
    for i in range(min(n_per_type, len(singular_subjects))):
        subj = singular_subjects[i]
        verb = verbs_s[i]
        correct = f"The {subj} {verb}"
        wrong = f"The {subj} {verb.rstrip('s')}"  # 去掉s
        # 也加复数版本
        correct_pl = f"The {subj}s {verb.rstrip('s')}"
        wrong_pl = f"The {subj}s {verb}"
        sentences["number_sva"].append({
            "correct": correct, "wrong": wrong,
            "correct_plural": correct_pl, "wrong_plural": wrong_pl,
            "type": "number"
        })
    
    # === 时态约束 ===
    present_sentences = [
        ("The cat sleeps", "The cat slept"),
        ("The dog runs", "The dog ran"),
        ("The bird sings", "The bird sang"),
        ("The girl reads", "The girl read"),
        ("The boy walks", "The boy walked"),
        ("The tree grows", "The tree grew"),
        ("The car moves", "The car moved"),
        ("The child plays", "The child played"),
        ("The man works", "The man worked"),
        ("The woman dances", "The woman danced"),
        ("The fish swims", "The fish swam"),
        ("The student studies", "The student studied"),
        ("The teacher speaks", "The teacher spoke"),
        ("The river flows", "The river flowed"),
        ("The wind blows", "The wind blew"),
        ("The sun shines", "The sun shone"),
        ("The rain falls", "The rain fell"),
        ("The fire burns", "The fire burned"),
        ("The snow melts", "The snow melted"),
        ("The bell rings", "The bell rang"),
    ]
    for present, past in present_sentences[:n_per_type]:
        sentences["tense"].append({
            "correct": present, "wrong": past, "type": "tense"
        })
    
    # === 否定约束 ===
    affirmative_neg = [
        ("The cat can sleep", "The cat cannot sleep"),
        ("The dog will run", "The dog will not run"),
        ("The bird does sing", "The bird does not sing"),
        ("The girl is reading", "The girl is not reading"),
        ("The boy has eaten", "The boy has not eaten"),
        ("The car was moving", "The car was not moving"),
        ("The child should play", "The child should not play"),
        ("The man could work", "The man could not work"),
        ("The woman would dance", "The woman would not dance"),
        ("The fish can swim", "The fish cannot swim"),
        ("The student must study", "The student must not study"),
        ("The teacher will speak", "The teacher will not speak"),
        ("The river is flowing", "The river is not flowing"),
        ("The wind might blow", "The wind might not blow"),
        ("The sun is shining", "The sun is not shining"),
        ("The rain was falling", "The rain was not falling"),
        ("The fire has burned", "The fire has not burned"),
        ("The snow will melt", "The snow will not melt"),
        ("The bell was ringing", "The bell was not ringing"),
        ("The dog can bark", "The dog cannot bark"),
    ]
    for aff, neg in affirmative_neg[:n_per_type]:
        sentences["negation"].append({
            "correct": aff, "wrong": neg, "type": "negation"
        })
    
    # === 语态约束（主动/被动）===
    voice_pairs = [
        ("The cat chases the dog", "The dog is chased by the cat"),
        ("The dog bites the man", "The man is bitten by the dog"),
        ("The girl reads the book", "The book is read by the girl"),
        ("The boy throws the ball", "The ball is thrown by the boy"),
        ("The teacher praised the student", "The student was praised by the teacher"),
        ("The wind blows the leaves", "The leaves are blown by the wind"),
        ("The chef cooks the meal", "The meal is cooked by the chef"),
        ("The artist paints the wall", "The wall is painted by the artist"),
        ("The writer finished the novel", "The novel was finished by the writer"),
        ("The company builds the house", "The house is built by the company"),
        ("The driver stops the car", "The car is stopped by the driver"),
        ("The police caught the thief", "The thief was caught by the police"),
        ("The mother loves the child", "The child is loved by the mother"),
        ("The scientist discovered the element", "The element was discovered by the scientist"),
        ("The river carries the boat", "The boat is carried by the river"),
        ("The fire destroyed the forest", "The forest was destroyed by the fire"),
        ("The sun warms the earth", "The earth is warmed by the sun"),
        ("The musician plays the piano", "The piano is played by the musician"),
        ("The farmer grows the wheat", "The wheat is grown by the farmer"),
        ("The king ruled the kingdom", "The kingdom was ruled by the king"),
    ]
    for active, passive in voice_pairs[:n_per_type]:
        sentences["voice"].append({
            "correct": active, "wrong": passive, "type": "voice"
        })
    
    # === 语义控制（同语法不同词汇）===
    semantic_pairs = [
        ("The cat sleeps", "The dog runs"),
        ("The bird sings", "The fish swims"),
        ("The girl reads", "The boy walks"),
        ("The tree grows", "The car moves"),
        ("The child plays", "The man works"),
        ("The star shines", "The moon glows"),
        ("The river flows", "The wind blows"),
        ("The fire burns", "The snow melts"),
        ("The cat chases", "The dog barks"),
        ("The teacher speaks", "The student listens"),
        ("The flower blooms", "The leaf falls"),
        ("The bell rings", "The horn sounds"),
        ("The rain falls", "The sun rises"),
        ("The horse gallops", "The bird soars"),
        ("The snake crawls", "The rabbit hops"),
        ("The cloud drifts", "The stream rushes"),
        ("The lion roars", "The wolf howls"),
        ("The baby cries", "The child laughs"),
        ("The rocket launches", "The ship sails"),
        ("The lamp glows", "The candle flickers"),
    ]
    for s1, s2 in semantic_pairs[:n_per_type]:
        sentences["semantic_control"].append({
            "correct": s1, "wrong": s2, "type": "semantic_control"
        })
    
    return sentences


# ============================================================
# 核心: 约束存活率测量
# ============================================================

def inject_and_propagate(model, tokenizer, device, text, layer_idx, 
                          perturbation, eps=1.0):
    """
    在layer l注入扰动，传播到最后一层，测量残存信号
    
    Args:
        text: 输入文本
        layer_idx: 注入层
        perturbation: 扰动方向 [d_model] numpy数组
        eps: 扰动幅度
    
    Returns:
        baseline_logits: 无扰动的logits [vocab_size]
        perturbed_logits: 有扰动的logits [vocab_size]  
        residual: 扰动残存 = perturbed_logits - baseline_logits
    """
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
    d_model = perturbation.shape[0]
    
    layers = get_layers(model)
    
    # === 基线前向 ===
    with torch.no_grad():
        baseline_output = model(input_ids)
        baseline_logits = baseline_output.logits[0, -1, :].detach().float().cpu().numpy()
    
    # === 扰动前向 ===
    v = torch.tensor(perturbation, dtype=torch.bfloat16, device=device)
    
    # 自适应eps: 确保扰动可见
    # 先捕获h_l的范数
    h_l_capture = {}
    def capture_h(module, input, output):
        if isinstance(input, tuple):
            h_l_capture['norm'] = input[0][0, -1, :].detach().float().norm().item()
    hook_cap = layers[layer_idx].register_forward_hook(capture_h)
    with torch.no_grad():
        model(input_ids)
    hook_cap.remove()
    
    h_l_norm = h_l_capture.get('norm', 1.0)
    # 确保扰动至少是h_l的1%
    eps_min = max(0.01 * h_l_norm, 0.1)
    actual_eps = max(eps, eps_min)
    
    # 注入扰动
    perturbed_logits_cap = {}
    def make_inject_hook(v_pert, eps_pert):
        def inject_hook(module, input, output):
            hidden = input[0]
            perturbed_hidden = hidden.clone()
            # float32加法避免精度丢失
            perturbation = (eps_pert * v_pert).to(torch.float32)
            last_tok = perturbed_hidden[0, -1, :].to(torch.float32) + perturbation
            perturbed_hidden[0, -1, :] = last_tok.to(perturbed_hidden.dtype)
            return (perturbed_hidden,) + input[1:]
        return inject_hook
    
    hook_inject = layers[layer_idx].register_forward_pre_hook(make_inject_hook(v, actual_eps))
    with torch.no_grad():
        try:
            perturbed_output = model(input_ids)
            perturbed_logits = perturbed_output.logits[0, -1, :].detach().float().cpu().numpy()
        except Exception:
            perturbed_logits = np.zeros_like(baseline_logits)
    hook_inject.remove()
    
    residual = perturbed_logits - baseline_logits
    
    return baseline_logits, perturbed_logits, residual, actual_eps


def compute_constraint_survivability(model, tokenizer, device, sentences_dict, 
                                     sample_layers, n_random=20):
    """
    实验1: 约束存活性
    
    对每种约束类型，在每层注入:
      - 约束方向 Δh
      - 随机方向
    传播到最后一层，比较残存率
    """
    results = {}
    d_model = 2560
    # 从model.config获取d_model
    if hasattr(model.config, 'hidden_size'):
        d_model = model.config.hidden_size
    elif hasattr(model.config, 'd_model'):
        d_model = model.config.d_model
    
    for constraint_type, pairs in sentences_dict.items():
        if not pairs:
            continue
        print(f"  [{time.strftime('%H:%M:%S')}] Constraint type: {constraint_type} ({len(pairs)} pairs)")
        
        type_results = {
            "n_pairs": len(pairs),
            "layers": {}
        }
        
        for layer_idx in sample_layers:
            layer_data = {
                "constraint_survivals": [],
                "random_survivals": [],
                "csi_values": [],  # Constraint Selectivity Index
                "eps_values": [],
                "delta_h_norms": [],
                "random_norms": [],
            }
            
            for pair_idx, pair in enumerate(pairs[:min(len(pairs), 15)]):  # 每种类型最多15对
                correct = pair["correct"]
                wrong = pair["wrong"]
                
                try:
                    # 1. 获取约束方向 Δh = h_correct - h_wrong (在注入层的hidden state)
                    layers = get_layers(model)
                    
                    # 捕获两个句子的h_l
                    h_correct_cap = {}
                    def make_capture(name):
                        def capture(module, input, output):
                            if isinstance(input, tuple):
                                name['h'] = input[0][0, -1, :].detach().float().cpu().numpy()
                        return capture
                    
                    hook1 = layers[layer_idx].register_forward_hook(make_capture(h_correct_cap))
                    with torch.no_grad():
                        model(tokenizer(correct, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device))
                    hook1.remove()
                    
                    h_wrong_cap = {}
                    hook2 = layers[layer_idx].register_forward_hook(make_capture(h_wrong_cap))
                    with torch.no_grad():
                        model(tokenizer(wrong, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device))
                    hook2.remove()
                    
                    if 'h' not in h_correct_cap or 'h' not in h_wrong_cap:
                        continue
                    
                    delta_h = h_correct_cap['h'] - h_wrong_cap['h']
                    delta_h_norm = np.linalg.norm(delta_h)
                    
                    if delta_h_norm < 1e-6:
                        continue
                    
                    # 归一化方向
                    delta_h_dir = delta_h / delta_h_norm
                    
                    # 2. 约束方向传播
                    _, _, constraint_residual, eps_used = inject_and_propagate(
                        model, tokenizer, device, correct, layer_idx, delta_h_dir, eps=1.0
                    )
                    constraint_survival = np.linalg.norm(constraint_residual)
                    
                    # 3. 随机方向传播（多个随机方向取平均）
                    random_survivals = []
                    for r in range(n_random):
                        random_dir = np.random.randn(d_model).astype(np.float32)
                        random_dir /= np.linalg.norm(random_dir)
                        
                        _, _, random_residual, eps_r = inject_and_propagate(
                            model, tokenizer, device, correct, layer_idx, random_dir, eps=1.0
                        )
                        random_survivals.append(np.linalg.norm(random_residual))
                    
                    avg_random_survival = np.mean(random_survivals)
                    
                    # 4. 约束选择性指数
                    csi = constraint_survival / max(avg_random_survival, 1e-10)
                    
                    layer_data["constraint_survivals"].append(float(constraint_survival))
                    layer_data["random_survivals"].append(float(avg_random_survival))
                    layer_data["csi_values"].append(float(csi))
                    layer_data["eps_values"].append(float(eps_used))
                    layer_data["delta_h_norms"].append(float(delta_h_norm))
                    
                except Exception as e:
                    print(f"    Error pair {pair_idx}: {e}")
                    continue
                
                # 每处理5对输出进度
                if (pair_idx + 1) % 5 == 0:
                    print(f"    [{time.strftime('%H:%M:%S')}] L{layer_idx}: {pair_idx+1}/{min(len(pairs), 15)} done")
                
                # 清理GPU
                if pair_idx % 3 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
            
            # 汇总
            if layer_data["csi_values"]:
                layer_data["csi_mean"] = float(np.mean(layer_data["csi_values"]))
                layer_data["csi_std"] = float(np.std(layer_data["csi_values"]))
                layer_data["constraint_survival_mean"] = float(np.mean(layer_data["constraint_survivals"]))
                layer_data["random_survival_mean"] = float(np.mean(layer_data["random_survivals"]))
                layer_data["survival_ratio"] = float(layer_data["constraint_survival_mean"] / max(layer_data["random_survival_mean"], 1e-10))
            
            type_results["layers"][str(layer_idx)] = layer_data
            print(f"    [{time.strftime('%H:%M:%S')}] L{layer_idx}: CSI={layer_data.get('csi_mean', 0):.2f}±{layer_data.get('csi_std', 0):.2f}")
        
        results[constraint_type] = type_results
    
    return results


# ============================================================
# 实验2: 传播模态结构 (Principal Angles)
# ============================================================

def compute_principal_angles(subspace1, subspace2, k=None):
    """
    计算两个子空间之间的主角(principal angles)
    
    Args:
        subspace1: [n_vectors, d] — 第一组基向量
        subspace2: [n_vectors, d] — 第二组基向量
        k: 计算前k个主角
    
    Returns:
        angles: 弧度列表
    """
    # QR分解正交化
    Q1, _ = np.linalg.qr(subspace1.T)  # [d, n]
    Q2, _ = np.linalg.qr(subspace2.T)  # [d, n]
    
    # SVD求主角
    S = Q1.T @ Q2  # [n, n]
    cos_angles = np.linalg.svd(S, compute_uv=False)
    cos_angles = np.clip(cos_angles, -1, 1)
    angles = np.arccos(np.abs(cos_angles))
    
    if k is not None:
        angles = angles[:k]
    
    return angles


def compute_propagation_mode_structure(model, tokenizer, device, n_sentences=30, 
                                        n_jvp_vectors=40, sample_layers=None):
    """
    实验2: 比较不同句子的Jacobian的top-k奇异子空间
    
    对每对句子，计算Jacobian的近似(通过JVPs)，
    然后比较top-k子空间的主角
    """
    if sample_layers is None:
        n_layers_local = len(get_layers(model))
        sample_layers = get_sample_layers(n_layers_local)
    
    d_model = 2560
    if hasattr(model.config, 'hidden_size'):
        d_model = model.config.hidden_size
    elif hasattr(model.config, 'd_model'):
        d_model = model.config.d_model
    
    # 生成随机向量（共享）
    np.random.seed(42)
    V_shared = np.random.randn(n_jvp_vectors, d_model).astype(np.float32)
    # 正交化
    Q_v, R_v = np.linalg.qr(V_shared.T)
    V_shared = Q_v.T[:n_jvp_vectors].astype(np.float32)
    
    # 生成简单句子
    subjects = ["cat", "dog", "bird", "girl", "boy", "tree", "car", "child",
                "man", "woman", "fish", "horse", "student", "teacher",
                "flower", "river", "star", "moon", "sun", "cloud",
                "wind", "rain", "fire", "snow", "bell", "lion",
                "snake", "rabbit", "baby", "king"]
    verbs_s = ["chases", "runs", "sings", "reads", "walks", "falls", "moves",
               "plays", "works", "dances", "swims", "sleeps", "barks", "flies",
               "grows", "blooms", "flows", "shines", "rises", "blows",
               "falls", "melts", "burns", "freezes", "rings", "roars",
               "crawls", "hops", "cries", "rules"]
    
    texts = [f"The {subjects[i]} {verbs_s[i]}" for i in range(min(n_sentences, len(subjects)))]
    
    results = {}
    
    for layer_idx in sample_layers:
        print(f"  [{time.strftime('%H:%M:%S')}] Computing Jacobian mode structure at L{layer_idx}...")
        
        # 对每个句子计算JVPs
        all_jvps = []
        for idx, text in enumerate(texts):
            jvps = compute_jacobian_jvps_simple(model, tokenizer, device, text, 
                                                 layer_idx, V_shared)
            all_jvps.append(jvps)
            
            if (idx + 1) % 5 == 0:
                print(f"    [{time.strftime('%H:%M:%S')}] {idx+1}/{len(texts)} done")
                torch.cuda.empty_cache()
                gc.collect()
        
        # 对每对JVPs计算主角
        n = len(all_jvps)
        if n < 2:
            continue
        
        # 采样比较（不全对全，太多了）
        n_compare = min(n * (n - 1) // 2, 100)
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        if len(pairs) > n_compare:
            indices = np.random.choice(len(pairs), n_compare, replace=False)
            pairs = [pairs[i] for i in indices]
        
        # 计算top-5, top-10, top-20子空间的主角
        k_values = [5, 10, 20]
        principal_angles = {f"top{k}": [] for k in k_values}
        
        for i, j in pairs:
            jvp_i = all_jvps[i]  # [n_vectors, d_model]
            jvp_j = all_jvps[j]
            
            for k in k_values:
                try:
                    angles = compute_principal_angles(jvp_i[:k], jvp_j[:k], k=k)
                    principal_angles[f"top{k}"].append({
                        "mean_angle": float(np.mean(angles)),
                        "min_angle": float(np.min(angles)),
                        "max_angle": float(np.max(angles)),
                        "median_angle": float(np.median(angles)),
                    })
                except:
                    pass
        
        # 汇总
        layer_result = {}
        for k_key in principal_angles:
            if principal_angles[k_key]:
                angles_list = principal_angles[k_key]
                layer_result[k_key] = {
                    "mean_mean_angle": float(np.mean([a["mean_angle"] for a in angles_list])),
                    "mean_median_angle": float(np.mean([a["median_angle"] for a in angles_list])),
                    "std_mean_angle": float(np.std([a["mean_angle"] for a in angles_list])),
                    # 转换为度
                    "mean_mean_angle_deg": float(np.degrees(np.mean([a["mean_angle"] for a in angles_list]))),
                    "n_pairs": len(angles_list),
                }
        
        results[str(layer_idx)] = layer_result
        print(f"    L{layer_idx}: top5 mean angle = {layer_result.get('top5', {}).get('mean_mean_angle_deg', 0):.1f}°")
    
    return results


def compute_jacobian_jvps_simple(model, tokenizer, device, text, layer_idx, V_shared):
    """简化版JVP计算（只返回JVPs，不返回额外信息）"""
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
    n_vectors = V_shared.shape[0]
    d_model = V_shared.shape[1]
    
    layers = get_layers(model)
    layer = layers[layer_idx]
    
    # 基线
    baseline = {}
    def capture_baseline(module, input, output):
        if isinstance(input, tuple):
            baseline['input'] = input[0][0, -1, :].detach().float().cpu().numpy()
        if isinstance(output, tuple):
            baseline['output'] = output[0][0, -1, :].detach().float().cpu().numpy()
        else:
            baseline['output'] = output[0, -1, :].detach().float().cpu().numpy()
    
    hook = layer.register_forward_hook(capture_baseline)
    with torch.no_grad():
        try:
            model(input_ids)
        except:
            pass
    hook.remove()
    
    h_l = baseline.get('input', np.zeros(d_model))
    h_l_plus_1 = baseline.get('output', np.zeros(d_model))
    
    # 自适应eps
    h_l_norm = np.linalg.norm(h_l)
    actual_eps = max(0.01 * h_l_norm, 0.1)
    
    # JVPs
    jvps = np.zeros((n_vectors, d_model), dtype=np.float32)
    
    for vec_idx in range(n_vectors):
        v = torch.tensor(V_shared[vec_idx], dtype=torch.bfloat16, device=device)
        
        perturbed = {}
        def make_pre_hook(v_pert, eps_pert, idx):
            def pre_hook(module, args):
                hidden = args[0]
                perturbed_hidden = hidden.clone()
                perturbation = (eps_pert * v_pert).to(torch.float32)
                last_tok = perturbed_hidden[0, -1, :].to(torch.float32) + perturbation
                perturbed_hidden[0, -1, :] = last_tok.to(perturbed_hidden.dtype)
                return (perturbed_hidden,) + args[1:]
            return pre_hook
        
        hook_p = layers[layer_idx].register_forward_pre_hook(make_pre_hook(v, actual_eps, vec_idx))
        
        def make_post_hook(idx):
            def post_hook(module, input, output):
                if isinstance(output, tuple):
                    perturbed[idx] = output[0][0, -1, :].detach().float().cpu().numpy()
                else:
                    perturbed[idx] = output[0, -1, :].detach().float().cpu().numpy()
            return post_hook
        
        hook_o = layer.register_forward_hook(make_post_hook(vec_idx))
        
        with torch.no_grad():
            try:
                model(input_ids)
            except:
                pass
        
        hook_p.remove()
        hook_o.remove()
        
        if vec_idx in perturbed:
            jvps[vec_idx] = (perturbed[vec_idx] - h_l_plus_1) / actual_eps
    
    return jvps


# ============================================================
# 主实验流程
# ============================================================

def run_experiment(model_name: str):
    """运行Phase 224完整实验"""
    print(f"\n{'='*60}")
    print(f"=== Phase 224: Constraint Survivability & Propagation Modes ({model_name}) ===")
    print(f"{'='*60}")
    
    # 加载模型
    config = MODEL_CONFIGS.get(model_name)
    if config is None:
        print(f"Unknown model: {model_name}")
        return
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name}...")
    
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import os
    os.environ["FLASH_ATTENTION"] = "1"
    
    model_path = config["path"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 加载模型
    load_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    if model_name != "qwen3":
        load_kwargs["device_map"] = "auto"
    
    try:
        model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)
    except Exception as e:
        print(f"  Error loading with default: {e}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, 
                torch_dtype=torch.bfloat16,
                device_map="auto",
                low_cpu_mem_usage=True
            )
        except Exception as e2:
            print(f"  Failed to load model: {e2}")
            return
    
    if model_name == "qwen3":
        model = model.to(device)
    
    model.eval()
    
    # 启用flash attention
    if hasattr(model, 'config') and hasattr(model.config, 'attn_implementation'):
        pass  # 已经在加载时设置
    
    n_layers = len(get_layers(model))
    sample_layers = get_sample_layers(n_layers)
    
    d_model = 2560
    if hasattr(model.config, 'hidden_size'):
        d_model = model.config.hidden_size
    elif hasattr(model.config, 'd_model'):
        d_model = model.config.d_model
    
    print(f"[{time.strftime('%H:%M:%S')}] {model_name} loaded: {n_layers} layers, d_model={d_model}")
    print(f"  Sample layers: {sample_layers}")
    print(f"  GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # === 生成句子 ===
    n_per_type = 20 if model_name == "qwen3" else 12
    sentences = generate_constraint_sentences(n_per_type=n_per_type)
    for k, v in sentences.items():
        print(f"  {k}: {len(v)} pairs")
    
    # === 实验1: 约束存活性 ===
    print(f"\n[{time.strftime('%H:%M:%S')}] === Experiment 1: Constraint Survivability ===")
    
    n_random_dirs = 10 if model_name != "qwen3" else 20
    exp1_results = compute_constraint_survivability(
        model, tokenizer, device, sentences, sample_layers,
        n_random=n_random_dirs
    )
    
    # 输出汇总
    print(f"\n--- Survivability Summary ({model_name}) ---")
    print(f"{'Type':<20} {'Layer':<8} {'CSI':>8} {'C_surv':>10} {'R_surv':>10} {'Ratio':>8}")
    for ctype, cdata in exp1_results.items():
        for layer_str, ldata in cdata.get("layers", {}).items():
            csi = ldata.get("csi_mean", 0)
            c_surv = ldata.get("constraint_survival_mean", 0)
            r_surv = ldata.get("random_survival_mean", 0)
            ratio = ldata.get("survival_ratio", 0)
            print(f"  {ctype:<18} L{layer_str:<6} {csi:>8.2f} {c_surv:>10.2f} {r_surv:>10.2f} {ratio:>8.2f}")
    
    # === 实验2: 传播模态结构 ===
    print(f"\n[{time.strftime('%H:%M:%S')}] === Experiment 2: Propagation Mode Structure ===")
    
    n_mode_sentences = 30 if model_name == "qwen3" else 15
    n_mode_jvps = 40 if model_name == "qwen3" else 25
    
    exp2_results = compute_propagation_mode_structure(
        model, tokenizer, device,
        n_sentences=n_mode_sentences,
        n_jvp_vectors=n_mode_jvps,
        sample_layers=sample_layers
    )
    
    # 输出汇总
    print(f"\n--- Mode Structure Summary ({model_name}) ---")
    for layer_str, ldata in exp2_results.items():
        for k_key, k_data in ldata.items():
            print(f"  L{layer_str} {k_key}: mean_angle={k_data.get('mean_mean_angle_deg', 0):.1f}° ± {np.degrees(k_data.get('std_mean_angle', 0)):.1f}°")
    
    # === 实验3: 约束选择性指数 (在实验1中已计算，这里补充语义控制对比) ===
    # 实验1的CSI已经是 constraint_survival / random_survival
    # 如果语义控制的CSI ≈ 1，而语法约束的CSI >> 1，则证明选择性
    
    # === 保存结果 ===
    results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "sample_layers": sample_layers,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "exp1_survivability": exp1_results,
        "exp2_mode_structure": exp2_results,
    }
    
    out_path = OUTPUT_DIR / f"phase224_{model_name}_results.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[{time.strftime('%H:%M:%S')}] Results saved to {out_path}")
    
    # === 关键结论总结 ===
    print(f"\n{'='*60}")
    print(f"Phase 224 Key Findings ({model_name}):")
    print(f"{'='*60}")
    
    # CSI by constraint type
    print("\n  Constraint Selectivity Index (CSI = constraint_survival / random_survival):")
    for ctype, cdata in exp1_results.items():
        csi_by_layer = []
        for layer_str, ldata in cdata.get("layers", {}).items():
            if "csi_mean" in ldata:
                csi_by_layer.append(ldata["csi_mean"])
        if csi_by_layer:
            print(f"    {ctype}: CSI = {np.mean(csi_by_layer):.2f} ± {np.std(csi_by_layer):.2f}")
    
    # 判断: 语法约束 vs 语义控制
    grammatical_cs = []
    semantic_cs = []
    for ctype in ["number_sva", "tense", "negation", "voice"]:
        if ctype in exp1_results:
            for layer_str, ldata in exp1_results[ctype].get("layers", {}).items():
                if "csi_mean" in ldata:
                    grammatical_cs.append(ldata["csi_mean"])
    if "semantic_control" in exp1_results:
        for layer_str, ldata in exp1_results["semantic_control"].get("layers", {}).items():
            if "csi_mean" in ldata:
                semantic_cs.append(ldata["csi_mean"])
    
    if grammatical_cs and semantic_cs:
        g_mean = np.mean(grammatical_cs)
        s_mean = np.mean(semantic_cs)
        print(f"\n  Grammatical CSI: {g_mean:.2f}")
        print(f"  Semantic CSI:    {s_mean:.2f}")
        if g_mean > s_mean * 1.5:
            print(f"  ★★★ Grammatical constraints have HIGHER survivability than semantic changes!")
            print(f"      → Transformer selectively propagates constraint directions")
        elif g_mean < s_mean * 0.7:
            print(f"  Semantic changes have higher survivability")
        else:
            print(f"  No significant difference between grammatical and semantic CSI")
    
    # 释放模型
    release_model(model)
    torch.cuda.empty_cache()
    gc.collect()
    print(f"\n[{time.strftime('%H:%M:%S')}] Model released. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase224_constraint_survivability.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)
    
    model_name = sys.argv[1]
    run_experiment(model_name)
