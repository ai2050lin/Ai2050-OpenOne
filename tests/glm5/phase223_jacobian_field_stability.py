"""
Phase 223: Jacobian场稳定性——从"方向传播"到"动力学规则"
=========================================================

核心理论升级（来自Phase 222的反思）：
  Δh = h_correct - h_wrong 不是约束本身
  它混合了三种东西：约束修正量 + 语义重构量 + 动力学轨道偏移
  
  真正的数学对象是 Jacobian场 J(h) = ∂h_{l+1}/∂h_l
  
  如果 J_i ≈ J_j (跨句子稳定) 但 Δh_i ≠ Δh_j (跨句子不稳定)
  → 证明语言编码的是"动力学规则"而非"方向"
  → Transformer的本质是"在高维状态流形上维持语言约束稳定性的动力系统"

实验1 (★★★★★ P0最关键): Jacobian谱稳定性
  - 对N个句子，在各层计算 J_l = ∂h_{l+1}/∂h_l (近似)
  - 比较谱结构: 奇异值分布、有效秩、主子空间对齐
  - 如果谱稳定 → 动力学规则是通用对象

实验2 (★★★★★): JVP方向一致性 vs Δh方向一致性
  - 对同一组随机向量V，计算 J_i·v 和 J_j·v
  - 比较 cos(J_i·v, J_j·v) 与 cos(Δh_i, Δh_j)
  - 如果前者远高于后者 → Jacobian比Δh更稳定 → 确认理论预测

实验3 (★★★★): Jacobian校正的约束方向对齐
  - 用J将Δh_i从句子i的局部坐标系"输运"到句子j的局部坐标系
  - Δh_i_corrected = J_j · J_i⁺ · Δh_i  (pseudoinverse校正)
  - 测量 cos(Δh_i_corrected, Δh_j) vs cos(Δh_i, Δh_j)
  - 如果校正后对齐改善 → Jacobian确实捕获了约束的几何结构

跨模型: Qwen3 → GLM4 → DS7B (顺序，避免OOM)
BF16 + device_map="auto" + sdpa(flash) + 定期GC

执行时间: 2026-05-18 08:00
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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 全局日志 =====
_last_log_time = time.time()
_LOG_INTERVAL = 30

def log_status(msg):
    global _last_log_time
    t = time.strftime("%H:%M:%S")
    gpu_mem = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
    print(f"[{t}] GPU={gpu_mem:.1f}GB | {msg}", flush=True)
    _last_log_time = time.time()

def maybe_log(msg, interval=None):
    global _last_log_time
    iv = interval or _LOG_INTERVAL
    if time.time() - _last_log_time > iv:
        log_status(msg)

# ===== 模型加载(sdpa + flash) =====
def load_model_sdpa(model_name: str):
    """BF16 + device_map='auto' + sdpa(flash内存优化)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    log_status(f"Loading {model_name} (bf16 + auto + sdpa)...")
    
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
        attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
    log_status(f"  Loaded: device={device}, GPU={gpu_mem:.1f}GB")
    return model, tokenizer, device

# ===== 句子生成 =====
NOUNS_SG = [
    "cat", "dog", "bird", "girl", "boy", "tree", "car", "child", "man", "woman",
    "fish", "horse", "student", "teacher", "flower", "river", "star", "moon", "sun",
    "cloud", "wind", "rain", "fire", "light", "sound", "door", "window", "book",
    "pen", "phone", "clock", "bell", "flag", "king", "queen", "wolf", "bear",
    "lion", "tiger", "eagle", "snake", "frog", "bee", "ant", "fox", "goat",
    "lamb", "owl", "pig", "rat", "swan", "duck", "deer", "crab", "crow",
    "doctor", "nurse", "lawyer", "judge", "singer", "dancer", "painter", "writer",
    "driver", "rider", "walker", "runner", "swimmer", "priest", "knight",
]

NOUNS_PL = {
    "cat": "cats", "dog": "dogs", "bird": "birds", "girl": "girls", "boy": "boys",
    "tree": "trees", "car": "cars", "child": "children", "man": "men", "woman": "women",
    "fish": "fish", "horse": "horses", "student": "students", "teacher": "teachers",
    "flower": "flowers", "river": "rivers", "star": "stars", "moon": "moons",
    "sun": "suns", "cloud": "clouds", "wind": "winds", "rain": "rains",
    "fire": "fires", "light": "lights", "sound": "sounds", "door": "doors",
    "window": "windows", "book": "books", "pen": "pens", "phone": "phones",
    "clock": "clocks", "bell": "bells", "flag": "flags", "king": "kings",
    "queen": "queens", "wolf": "wolves", "bear": "bears", "lion": "lions",
    "tiger": "tigers", "eagle": "eagles", "snake": "snakes", "frog": "frogs",
    "bee": "bees", "ant": "ants", "fox": "foxes", "goat": "goats",
    "lamb": "lambs", "owl": "owls", "pig": "pigs", "rat": "rats",
    "swan": "swans", "duck": "ducks", "deer": "deer", "crab": "crabs",
    "crow": "crows", "doctor": "doctors", "nurse": "nurses", "lawyer": "lawyers",
    "judge": "judges", "singer": "singers", "dancer": "dancers", "painter": "painters",
    "writer": "writers", "driver": "drivers", "rider": "riders", "walker": "walkers",
    "runner": "runners", "swimmer": "swimmers", "priest": "priests", "knight": "knights",
}

VERBS_SG = [
    "chases", "runs", "sings", "reads", "walks", "falls", "moves", "plays",
    "works", "dances", "swims", "sleeps", "barks", "flies", "grows", "blooms",
    "flows", "shines", "rises", "sets", "blows", "melts", "burns", "opens",
    "breaks", "rings", "stops", "sails", "stands", "sits", "waits", "looks",
    "seems", "feels", "begins", "ends", "helps", "wants", "needs", "loves",
    "knows", "thinks", "hopes", "fears", "trusts", "doubts", "wonders",
    "learns", "teaches", "shows", "tells", "asks", "gives", "takes", "finds",
    "keeps", "holds", "brings", "builds", "defends", "creates", "discovers",
]

VERBS_PL = {
    "chases": "chase", "runs": "run", "sings": "sing", "reads": "read",
    "walks": "walk", "falls": "fall", "moves": "move", "plays": "play",
    "works": "work", "dances": "dance", "swims": "swim", "sleeps": "sleep",
    "barks": "bark", "flies": "fly", "grows": "grow", "blooms": "bloom",
    "flows": "flow", "shines": "shine", "rises": "rise", "sets": "set",
    "blows": "blow", "melts": "melt", "burns": "burn", "opens": "open",
    "breaks": "break", "rings": "ring", "stops": "stop", "sails": "sail",
    "stands": "stand", "sits": "sit", "waits": "wait", "looks": "look",
    "seems": "seem", "feels": "feel", "begins": "begin", "ends": "end",
    "helps": "help", "wants": "want", "needs": "need", "loves": "love",
    "knows": "know", "thinks": "think", "hopes": "hope", "fears": "fear",
    "trusts": "trust", "doubts": "doubt", "wonders": "wonder",
    "learns": "learn", "teaches": "teach", "shows": "show", "tells": "tell",
    "asks": "ask", "gives": "give", "takes": "take", "finds": "find",
    "keeps": "keep", "holds": "hold", "brings": "bring", "builds": "build",
    "defends": "defend", "creates": "create", "discovers": "discover",
}

ADJECTIVES = [
    "small", "large", "happy", "sad", "fast", "slow", "young", "old",
    "red", "blue", "green", "white", "black", "bright", "dark", "warm",
    "cold", "soft", "hard", "quiet", "loud", "strong", "weak", "tall",
    "short", "thin", "thick", "clean", "dirty", "dry", "wet", "rich",
]

NOUNS_SG2 = [n for n in NOUNS_SG if n in NOUNS_PL][:30]


def generate_diverse_sentences(n=80):
    """生成多样化句子——不同复杂度、不同语义域"""
    sentences = []
    
    # 类型1: 简单SVA (40句)
    for i, noun in enumerate(NOUNS_SG[:40]):
        verb = VERBS_SG[i % len(VERBS_SG)]
        s = f"The {noun} {verb}"
        s_wrong = f"The {NOUNS_PL.get(noun, noun + 's')} {VERBS_PL.get(verb, verb)}"
        sentences.append({
            "correct": s, "wrong": s_wrong,
            "type": "simple_sva", "noun": noun, "verb": verb
        })
    
    # 类型2: 带形容词 (20句)
    for i in range(20):
        noun = NOUNS_SG[40 + (i % len(NOUNS_SG) - 40)]
        adj = ADJECTIVES[i % len(ADJECTIVES)]
        verb = VERBS_SG[(i + 10) % len(VERBS_SG)]
        if noun not in NOUNS_PL or verb not in VERBS_PL:
            continue
        s = f"The {adj} {noun} {verb}"
        s_wrong = f"The {adj} {NOUNS_PL[noun]} {VERBS_PL[verb]}"
        sentences.append({
            "correct": s, "wrong": s_wrong,
            "type": "adj_sva", "noun": noun, "verb": verb, "adj": adj
        })
    
    # 类型3: 带介词短语 (20句)
    for i in range(20):
        noun1 = NOUNS_SG[i]
        noun2 = NOUNS_SG2[(i + 5) % len(NOUNS_SG2)]
        verb = VERBS_SG[(i + 20) % len(VERBS_SG)]
        if noun1 not in NOUNS_PL or noun2 not in NOUNS_PL or verb not in VERBS_PL:
            continue
        s = f"The {noun1} near the {noun2} {verb}"
        s_wrong = f"The {NOUNS_PL[noun1]} near the {noun2} {VERBS_PL[verb]}"
        sentences.append({
            "correct": s, "wrong": s_wrong,
            "type": "pp_sva", "noun": noun1, "verb": verb
        })
    
    return sentences[:n]


# ===== Jacobian计算核心 =====
def compute_hidden_states(model, tokenizer, device, text, target_layers):
    """获取指定层的residual stream"""
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
    
    captured = {}
    layers = get_layers(model)
    
    def make_hook(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                h = output[0][0, -1, :].detach().float().cpu().numpy()
            else:
                h = output[0, -1, :].detach().float().cpu().numpy()
            captured[layer_idx] = h
        return hook
    
    hooks = []
    for li in target_layers:
        hooks.append(layers[li].register_forward_hook(make_hook(li)))
    
    with torch.no_grad():
        try:
            model(input_ids)
        except Exception as e:
            log_status(f"  Forward failed for '{text[:30]}...': {e}")
    
    for h in hooks:
        h.remove()
    
    return captured


def compute_jacobian_jvps(model, tokenizer, device, text, layer_idx, 
                           V_shared, eps=0.1, batch_size=3):
    """
    计算Jacobian-Vector Products: J_l · V ≈ [h_{l+1}(h_l+εV) - h_{l+1}(h_l)] / ε
    
    ★★★ 关键修复: bfloat16精度问题
    bfloat16只有~3位十进制精度。如果||h_l||≈10, ε=1e-4的扰动相对大小≈1e-5,
    远低于bfloat16的1e-2精度。扰动会被完全丢弃!
    
    解决方案: 使用自适应ε = max(eps, ||h_l|| * eps_relative)
    确保 ε·||v|| 至少是 ||h_l|| 的 1% (bfloat16可见)
    
    Args:
        model: 模型
        tokenizer: 分词器
        device: 设备
        text: 输入文本
        layer_idx: 目标层索引
        V_shared: 共享随机向量 [n_vectors, d_model] numpy数组
        eps: 扰动大小（会被自适应调整）
        batch_size: 每批处理的向量数
    
    Returns:
        jvps: [n_vectors, d_model] numpy数组
        h_l: 基线h_l [d_model]
        h_l_plus_1: 基线h_{l+1} [d_model]
        actual_eps: 实际使用的eps
    """
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids.to(device)
    n_vectors = V_shared.shape[0]
    d_model = V_shared.shape[1]
    
    layers = get_layers(model)
    layer = layers[layer_idx]
    
    # Step 1: 基线前向传播 - 捕获h_l和h_{l+1}
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
        except Exception:
            pass
    hook.remove()
    
    h_l = baseline.get('input', np.zeros(d_model))
    h_l_plus_1 = baseline.get('output', np.zeros(d_model))
    
    # ★★★ 自适应ε: 确保扰动在bfloat16中可见
    h_l_norm = np.linalg.norm(h_l)
    # 扰动相对大小需要至少1%才能在bfloat16中存活
    # ε * ||v|| / ||h_l|| >= 0.01 → ε >= 0.01 * ||h_l||
    eps_min = max(0.01 * h_l_norm, 0.1)  # 至少1%的相对扰动
    actual_eps = max(eps, eps_min)
    
    # Step 2: 扰动前向传播 - 批处理
    jvps = np.zeros((n_vectors, d_model), dtype=np.float32)
    
    for batch_start in range(0, n_vectors, batch_size):
        batch_end = min(batch_start + batch_size, n_vectors)
        current_batch_size = batch_end - batch_start
        
        for idx_in_batch, vec_idx in enumerate(range(batch_start, batch_end)):
            v = torch.tensor(V_shared[vec_idx], dtype=torch.bfloat16, device=device)
            
            perturbed = {}
            
            def make_pre_hook(v_pert, eps_pert):
                def pre_hook(module, args):
                    hidden = args[0]
                    perturbed_hidden = hidden.clone()
                    # ★ 使用float32做加法，再转回模型精度，避免bfloat16精度丢失
                    perturbation = (eps_pert * v_pert).to(torch.float32)
                    last_tok = perturbed_hidden[0, -1, :].to(torch.float32) + perturbation
                    perturbed_hidden[0, -1, :] = last_tok.to(perturbed_hidden.dtype)
                    return (perturbed_hidden,) + args[1:]
                return pre_hook
            
            def make_post_hook(idx):
                def post_hook(module, input, output):
                    if isinstance(output, tuple):
                        perturbed[idx] = output[0][0, -1, :].detach().float().cpu().numpy()
                    else:
                        perturbed[idx] = output[0, -1, :].detach().float().cpu().numpy()
                return post_hook
            
            pre_h = layer.register_forward_pre_hook(make_pre_hook(v, eps))
            post_h = layer.register_forward_hook(make_post_hook(vec_idx))
            
            with torch.no_grad():
                try:
                    model(input_ids)
                except Exception:
                    pass
            
            pre_h.remove()
            post_h.remove()
            
            if vec_idx in perturbed:
                jvps[vec_idx] = (perturbed[vec_idx] - h_l_plus_1) / actual_eps
        
        maybe_log(f"  JVP batch {batch_start}-{batch_end}/{n_vectors} done (eps={actual_eps:.2f})")
    
    return jvps, h_l, h_l_plus_1, actual_eps


# ===== 谱分析 =====
def compute_spectral_properties(jvps, V, n_components=30):
    """
    从JVPs估计Jacobian的谱性质
    
    jvps: [n_vectors, d_model] — J·v_i 的估计
    V: [n_vectors, d_model] — 输入随机向量 (正交化)
    
    因为 jvps ≈ J · V^T (矩阵形式，每行是J·v_i)
    V已正交化，所以SVD(jvps)的奇异值近似J的奇异值（在V列空间上的投影）
    
    Returns:
        dict with singular values, effective rank, etc.
    """
    # 直接对JVPs做SVD
    U_jvps, S_jvps, Vt_jvps = np.linalg.svd(jvps, full_matrices=False)
    # S_jvps: [min(n, d)] 奇异值
    
    # 有效秩 (能量阈值法)
    total_energy = np.sum(S_jvps ** 2)
    if total_energy > 0:
        cumulative = np.cumsum(S_jvps ** 2) / total_energy
        eff_rank = np.searchsorted(cumulative, 0.9) + 1
        eff_rank_95 = np.searchsorted(cumulative, 0.95) + 1
        eff_rank_99 = np.searchsorted(cumulative, 0.99) + 1
    else:
        eff_rank = eff_rank_95 = eff_rank_99 = 0
    
    # 谱熵 (衡量奇异值分布的均匀性)
    p = S_jvps ** 2 / max(np.sum(S_jvps ** 2), 1e-20)
    entropy = -np.sum(p[p > 0] * np.log(p[p > 0] + 1e-20))
    max_entropy = np.log(len(S_jvps))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    return {
        "singular_values": S_jvps.tolist(),
        "eff_rank_90": int(eff_rank),
        "eff_rank_95": int(eff_rank_95),
        "eff_rank_99": int(eff_rank_99),
        "spectral_entropy": float(normalized_entropy),
        "top1_sv": float(S_jvps[0]) if len(S_jvps) > 0 else 0,
        "top5_sv_ratio": float(np.sum(S_jvps[:5]**2) / total_energy) if total_energy > 0 and len(S_jvps) >= 5 else 0,
        "top10_sv_ratio": float(np.sum(S_jvps[:10]**2) / total_energy) if total_energy > 0 and len(S_jvps) >= 10 else 0,
        "condition_number": float(S_jvps[0] / S_jvps[-1]) if len(S_jvps) > 0 and S_jvps[-1] > 1e-10 else float('inf'),
    }


def compare_spectra(spec1, spec2):
    """比较两个谱的相似度"""
    sv1 = np.array(spec1["singular_values"])
    sv2 = np.array(spec2["singular_values"])
    
    min_len = min(len(sv1), len(sv2))
    sv1 = sv1[:min_len]
    sv2 = sv2[:min_len]
    
    # Pearson相关 (log空间，因为奇异值通常跨越多个数量级)
    log_sv1 = np.log(sv1 + 1e-10)
    log_sv2 = np.log(sv2 + 1e-10)
    
    mean1, mean2 = np.mean(log_sv1), np.mean(log_sv2)
    std1, std2 = np.std(log_sv1), np.std(log_sv2)
    
    if std1 < 1e-10 or std2 < 1e-10:
        pearson = 0.0
    else:
        pearson = float(np.mean((log_sv1 - mean1) * (log_sv2 - mean2)) / (std1 * std2))
    
    # Spearman秩相关
    from scipy.stats import spearmanr
    spearman, _ = spearmanr(sv1, sv2)
    
    # 有效秩差异
    rank_diff = abs(spec1["eff_rank_90"] - spec2["eff_rank_90"])
    
    # 谱熵差异
    entropy_diff = abs(spec1["spectral_entropy"] - spec2["spectral_entropy"])
    
    return {
        "pearson_log_sv": float(pearson),
        "spearman_sv": float(spearman),
        "rank_diff": int(rank_diff),
        "entropy_diff": float(entropy_diff),
    }


def compare_subspaces(jvps1, jvps2, V, top_k=10):
    """
    比较两个Jacobian近似的主子空间对齐度
    
    Returns:
        dict with subspace angles, mean cosine, etc.
    """
    # SVD of JVPs
    U1, S1, _ = np.linalg.svd(jvps1, full_matrices=False)
    U2, S2, _ = np.linalg.svd(jvps2, full_matrices=False)
    
    # U的列是JVP空间的主成分方向
    # 但U的形状是 [n_vectors, n_vectors]，不是 [d_model, ...]
    # 正确的做法：JVPs的右奇异向量才是d_model空间中的方向
    
    _, _, Vt1 = np.linalg.svd(jvps1, full_matrices=False)
    _, _, Vt2 = np.linalg.svd(jvps2, full_matrices=False)
    
    # Vt1[:k] 是 jvps1 的前k个右奇异向量 [k, d_model]
    k = min(top_k, Vt1.shape[0], Vt2.shape[0])
    sub1 = Vt1[:k]  # [k, d_model]
    sub2 = Vt2[:k]  # [k, d_model]
    
    # 子空间投影矩阵
    P1 = sub1.T @ sub1  # [d_model, d_model]
    P2 = sub2.T @ sub2  # [d_model, d_model]
    
    # 子空间重叠 = ||P1 P2||_F / ||P1||_F
    overlap = np.linalg.norm(P1 @ P2) / max(np.linalg.norm(P1), 1e-10)
    
    # Principal angles
    try:
        from scipy.linalg import subspace_angles
        angles = subspace_angles(sub1.T, sub2.T)  # 输入 [d_model, k]
        mean_angle = float(np.mean(angles))
        max_angle = float(np.max(angles))
    except Exception:
        mean_angle = float('nan')
        max_angle = float('nan')
    
    # Mean cosine of projected directions
    cosines = []
    for i in range(min(k, 10)):
        v1 = sub1[i]  # [d_model]
        proj_on_sub2 = sub2.T @ (sub2 @ v1)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(proj_on_sub2)
        if norm1 > 1e-10 and norm2 > 1e-10:
            cosines.append(float(np.dot(v1, proj_on_sub2) / (norm1 * norm2)))
    
    return {
        "subspace_overlap": float(overlap),
        "mean_principal_angle_rad": mean_angle,
        "max_principal_angle_rad": max_angle,
        "mean_cosine_top_k": float(np.mean(cosines)) if cosines else 0.0,
    }


# ===== 主实验 =====
def run_experiment(model_name: str, n_sentences: int = None):
    """运行Phase 223完整实验"""
    log_status(f"=== Phase 223: Jacobian Field Stability ({model_name}) ===")
    
    # 大模型（GLM4/DS7B）使用较少句子，加速测试
    if n_sentences is None:
        n_sentences = 80 if model_name == "qwen3" else 30
    
    model, tokenizer, device = load_model_sdpa(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    log_status(f"  Model: {model_info.model_class}, n_layers={n_layers}, d_model={d_model}")
    
    # 生成句子
    sentences = generate_diverse_sentences(n=n_sentences)
    log_status(f"  Generated {len(sentences)} sentences")
    
    # 采样层
    sample_layers = get_sample_layers(n_layers, 5)  # 5层采样
    log_status(f"  Sample layers: {sample_layers}")
    
    # 生成共享随机向量 (对所有句子使用同一组V)
    n_jvp_vectors = 40
    np.random.seed(42)  # 确保跨句子使用相同的V
    V_shared = np.random.randn(n_jvp_vectors, d_model).astype(np.float32)
    # 正交化
    Q, R = np.linalg.qr(V_shared.T)
    V_shared = Q.T[:n_jvp_vectors].astype(np.float32)  # [n_vectors, d_model]
    
    log_status(f"  Shared random vectors: {V_shared.shape}")
    
    # ===== 实验1+2: Jacobian谱稳定性 + JVP方向一致性 =====
    log_status("=== Experiment 1+2: Jacobian Spectral Stability + JVP Consistency ===")
    
    all_results = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "sample_layers": sample_layers,
        "n_sentences": len(sentences),
        "n_jvp_vectors": n_jvp_vectors,
        "exp1_spectral": {},
        "exp2_jvp_consistency": {},
        "exp3_transport": {},
    }
    
    # 存储每个句子每层的JVPs和hidden states
    per_sentence_data = {}  # {sentence_idx: {layer_idx: {jvps, h_l, h_l_plus_1, delta_h}}}
    
    for s_idx, sent in enumerate(sentences):
        per_sentence_data[s_idx] = {}
        
        # 基线: 正确和错误句子的hidden states
        correct_text = sent["correct"]
        wrong_text = sent["wrong"]
        
        log_status(f"  Sentence {s_idx+1}/{len(sentences)}: '{correct_text[:40]}...'")
        
        # 计算正确和错误句子的hidden states
        hs_correct = compute_hidden_states(model, tokenizer, device, correct_text, sample_layers)
        hs_wrong = compute_hidden_states(model, tokenizer, device, wrong_text, sample_layers)
        
        for l_idx, layer_l in enumerate(sample_layers):
            if layer_l not in hs_correct or layer_l not in hs_wrong:
                continue
            
            h_correct = hs_correct[layer_l]
            h_wrong = hs_wrong[layer_l]
            delta_h = h_correct - h_wrong
            
            # 计算JVPs
            log_status(f"    Layer {layer_l}: computing JVPs...")
            jvps, h_l, h_l_plus_1, actual_eps = compute_jacobian_jvps(
                model, tokenizer, device, correct_text, layer_l, V_shared,
                eps=0.1, batch_size=3
            )
            
            # 谱分析
            spectral = compute_spectral_properties(jvps, V_shared)
            
            per_sentence_data[s_idx][layer_l] = {
                "jvps": jvps,
                "h_l": h_l,
                "h_l_plus_1": h_l_plus_1,
                "delta_h": delta_h,
                "spectral": spectral,
                "sentence_type": sent["type"],
            }
            
            log_status(f"    L{layer_l}: eff_rank={spectral['eff_rank_90']}, "
                       f"top1_sv={spectral['top1_sv']:.3f}, "
                       f"top5_ratio={spectral['top5_sv_ratio']:.3f}, "
                       f"entropy={spectral['spectral_entropy']:.3f}, "
                       f"eps={actual_eps:.2f}")
        
        # 定期GC
        if s_idx % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    log_status("  All sentences processed. Analyzing results...")
    
    # ===== 实验1: 谱稳定性分析 =====
    log_status("=== Analyzing Experiment 1: Spectral Stability ===")
    
    for layer_l in sample_layers:
        spectra = []
        types = []
        for s_idx in range(len(sentences)):
            if layer_l in per_sentence_data[s_idx]:
                spectra.append(per_sentence_data[s_idx][layer_l]["spectral"])
                types.append(per_sentence_data[s_idx][layer_l]["sentence_type"])
        
        if len(spectra) < 2:
            continue
        
        # 两两比较谱
        n_pairs = min(len(spectra), 50)  # 限制比较次数
        pairwise_comparisons = []
        for i in range(n_pairs):
            for j in range(i + 1, min(i + 10, n_pairs)):  # 每个句子与后10个比较
                comp = compare_spectra(spectra[i], spectra[j])
                comp["type_i"] = types[i]
                comp["type_j"] = types[j]
                pairwise_comparisons.append(comp)
        
        # 汇总
        if pairwise_comparisons:
            mean_pearson = np.mean([c["pearson_log_sv"] for c in pairwise_comparisons])
            mean_spearman = np.mean([c["spearman_sv"] for c in pairwise_comparisons])
            mean_rank_diff = np.mean([c["rank_diff"] for c in pairwise_comparisons])
            mean_entropy_diff = np.mean([c["entropy_diff"] for c in pairwise_comparisons])
            
            # 有效秩的方差
            eff_ranks = [s["eff_rank_90"] for s in spectra]
            rank_std = np.std(eff_ranks)
            rank_mean = np.mean(eff_ranks)
            
            all_results["exp1_spectral"][str(layer_l)] = {
                "n_spectra": len(spectra),
                "mean_pearson_log_sv": float(mean_pearson),
                "mean_spearman_sv": float(mean_spearman),
                "mean_rank_diff": float(mean_rank_diff),
                "mean_entropy_diff": float(mean_entropy_diff),
                "eff_rank_mean": float(rank_mean),
                "eff_rank_std": float(rank_std),
                "eff_rank_min": int(min(eff_ranks)),
                "eff_rank_max": int(max(eff_ranks)),
                "top5_sv_ratio_mean": float(np.mean([s["top5_sv_ratio"] for s in spectra])),
                "top5_sv_ratio_std": float(np.std([s["top5_sv_ratio"] for s in spectra])),
                "spectral_entropy_mean": float(np.mean([s["spectral_entropy"] for s in spectra])),
                "spectral_entropy_std": float(np.std([s["spectral_entropy"] for s in spectra])),
            }
            
            log_status(f"  L{layer_l}: pearson={mean_pearson:.3f}, spearman={mean_spearman:.3f}, "
                       f"rank={rank_mean:.1f}±{rank_std:.1f}, entropy_diff={mean_entropy_diff:.4f}")
    
    # ===== 实验2: JVP方向一致性 vs Δh一致性 =====
    log_status("=== Analyzing Experiment 2: JVP Consistency vs Δh Consistency ===")
    
    for layer_l in sample_layers:
        # 收集该层所有句子的JVPs和Δh
        jvps_list = []
        delta_h_list = []
        
        for s_idx in range(len(sentences)):
            if layer_l in per_sentence_data[s_idx]:
                jvps_list.append(per_sentence_data[s_idx][layer_l]["jvps"])
                delta_h_list.append(per_sentence_data[s_idx][layer_l]["delta_h"])
        
        if len(jvps_list) < 2:
            continue
        
        n_compare = min(len(jvps_list), 60)
        
        # 2a: JVP方向一致性: cos(J_i·v, J_j·v) 对随机v
        jvp_cosines = []
        for v_idx in range(min(10, n_jvp_vectors)):
            vectors = [jvps[v_idx] for jvps in jvps_list[:n_compare]]
            vectors = np.array(vectors)  # [n_compare, d_model]
            
            # 两两cos
            for i in range(min(20, len(vectors))):
                for j in range(i + 1, min(i + 5, len(vectors))):
                    v1 = vectors[i]
                    v2 = vectors[j]
                    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        jvp_cosines.append(float(np.dot(v1, v2) / (n1 * n2)))
        
        # 2b: Δh方向一致性: cos(Δh_i, Δh_j)
        delta_h_cosines = []
        delta_h_arr = np.array(delta_h_list[:n_compare])
        for i in range(min(20, len(delta_h_arr))):
            for j in range(i + 1, min(i + 5, len(delta_h_arr))):
                n1, n2 = np.linalg.norm(delta_h_arr[i]), np.linalg.norm(delta_h_arr[j])
                if n1 > 1e-10 and n2 > 1e-10:
                    delta_h_cosines.append(float(np.dot(delta_h_arr[i], delta_h_arr[j]) / (n1 * n2)))
        
        # 2c: JVP范数比 (J的放大/缩小效应)
        jvp_norms = np.array([np.linalg.norm(jvps, axis=1).mean() for jvps in jvps_list[:n_compare]])
        jvp_norm_mean = float(np.mean(jvp_norms))
        jvp_norm_std = float(np.std(jvp_norms))
        jvp_norm_cv = float(jvp_norm_std / max(jvp_norm_mean, 1e-10))  # 变异系数
        
        # 2d: 子空间对齐
        if len(jvps_list) >= 2:
            sub_comp = compare_subspaces(jvps_list[0], jvps_list[1], V_shared, top_k=10)
        else:
            sub_comp = {"subspace_overlap": 0, "mean_principal_angle_rad": 0, "mean_cosine_top_k": 0}
        
        all_results["exp2_jvp_consistency"][str(layer_l)] = {
            "jvp_mean_cos": float(np.mean(jvp_cosines)) if jvp_cosines else 0,
            "jvp_std_cos": float(np.std(jvp_cosines)) if jvp_cosines else 0,
            "jvp_abs_mean_cos": float(np.mean(np.abs(jvp_cosines))) if jvp_cosines else 0,
            "delta_h_mean_cos": float(np.mean(delta_h_cosines)) if delta_h_cosines else 0,
            "delta_h_std_cos": float(np.std(delta_h_cosines)) if delta_h_cosines else 0,
            "delta_h_abs_mean_cos": float(np.mean(np.abs(delta_h_cosines))) if delta_h_cosines else 0,
            "jvp_norm_mean": jvp_norm_mean,
            "jvp_norm_std": jvp_norm_std,
            "jvp_norm_cv": jvp_norm_cv,
            "n_jvp_pairs": len(jvp_cosines),
            "n_delta_h_pairs": len(delta_h_cosines),
            "subspace_overlap_first2": float(sub_comp.get("subspace_overlap", 0)),
            "subspace_angle_first2": float(sub_comp.get("mean_principal_angle_rad", 0)),
        }
        
        # 关键比较: JVP一致性 vs Δh一致性
        jvp_cos_mean = float(np.mean(jvp_cosines)) if jvp_cosines else 0
        dh_cos_mean = float(np.mean(delta_h_cosines)) if delta_h_cosines else 0
        
        verdict = "J_STABLE" if abs(jvp_cos_mean) > abs(dh_cos_mean) + 0.1 else \
                  "SIMILAR" if abs(jvp_cos_mean) > abs(dh_cos_mean) - 0.05 else \
                  "DH_STABLE"
        
        log_status(f"  L{layer_l}: JVP|cos|={abs(jvp_cos_mean):.3f} vs Δh|cos|={abs(dh_cos_mean):.3f} "
                   f"→ {verdict}, norm_cv={jvp_norm_cv:.3f}")
    
    # ===== 实验3: Jacobian校正的约束方向对齐 =====
    log_status("=== Analyzing Experiment 3: Jacobian-corrected Constraint Alignment ===")
    
    for layer_l in sample_layers:
        # 需要JVPs来近似J的伪逆
        jvps_list = []
        delta_h_list = []
        h_l_list = []
        
        for s_idx in range(len(sentences)):
            if layer_l in per_sentence_data[s_idx]:
                jvps_list.append(per_sentence_data[s_idx][layer_l]["jvps"])
                delta_h_list.append(per_sentence_data[s_idx][layer_l]["delta_h"])
                h_l_list.append(per_sentence_data[s_idx][layer_l]["h_l"])
        
        if len(jvps_list) < 2:
            continue
        
        n_transport = min(len(jvps_list), 30)
        
        # 对每对句子，测试Jacobian校正是否改善Δh对齐
        raw_cosines = []
        corrected_cosines = []
        
        for i in range(min(15, n_transport)):
            for j in range(i + 1, min(i + 3, n_transport)):
                jvps_i = jvps_list[i]  # [n_vectors, d_model]
                jvps_j = jvps_list[j]
                dh_i = delta_h_list[i]  # [d_model]
                dh_j = delta_h_list[j]  # [d_model]
                
                # 原始对齐
                n_i, n_j = np.linalg.norm(dh_i), np.linalg.norm(dh_j)
                if n_i < 1e-10 or n_j < 1e-10:
                    continue
                raw_cos = float(np.dot(dh_i, dh_j) / (n_i * n_j))
                raw_cosines.append(raw_cos)
                
                # Jacobian校正: Δh_i_corrected ≈ J_j · J_i⁺ · Δh_i
                # 近似: 用JVPs估计J_i⁺·Δh_i
                # J_i · V ≈ jvps_i, 所以 J_i ≈ jvps_i · V⁺
                # J_i⁺ ≈ V · jvps_i⁺
                
                # 简化方法1: 投影校正
                # 将dh_i投影到J_i的列空间，然后用J_j的列空间重建
                # J_i的列空间 ≈ jvps_i的列空间
                
                try:
                    # jvps_i的SVD
                    U_i, S_i, Vt_i = np.linalg.svd(jvps_i, full_matrices=False)
                    U_j, S_j, Vt_j = np.linalg.svd(jvps_j, full_matrices=False)
                    
                    k = min(15, len(S_i), len(S_j))  # 使用前15个主成分
                    
                    # 方法: Δh_i_corrected = U_j[:k] · U_i[:k]^T · Δh_i
                    # 即: 将Δh_i投影到J_i的输出子空间，然后用J_j的输出子空间重建
                    
                    # 在J_i的输出子空间中的坐标
                    coords_in_i = U_i[:k] @ dh_i  # [k]
                    
                    # 用J_j的输出子空间重建
                    dh_i_corrected = U_j[:k].T @ coords_in_i  # [d_model]
                    
                    n_corr = np.linalg.norm(dh_i_corrected)
                    if n_corr > 1e-10 and n_j > 1e-10:
                        corrected_cos = float(np.dot(dh_i_corrected, dh_j) / (n_corr * n_j))
                        corrected_cosines.append(corrected_cos)
                    else:
                        corrected_cosines.append(0.0)
                
                except Exception as e:
                    corrected_cosines.append(0.0)
        
        all_results["exp3_transport"][str(layer_l)] = {
            "raw_mean_cos": float(np.mean(raw_cosines)) if raw_cosines else 0,
            "raw_abs_mean_cos": float(np.mean(np.abs(raw_cosines))) if raw_cosines else 0,
            "corrected_mean_cos": float(np.mean(corrected_cosines)) if corrected_cosines else 0,
            "corrected_abs_mean_cos": float(np.mean(np.abs(corrected_cosines))) if corrected_cosines else 0,
            "n_pairs": len(raw_cosines),
            "improvement": float(np.mean(np.abs(corrected_cosines)) - np.mean(np.abs(raw_cosines))) if raw_cosines and corrected_cosines else 0,
        }
        
        raw_abs = float(np.mean(np.abs(raw_cosines))) if raw_cosines else 0
        corr_abs = float(np.mean(np.abs(corrected_cosines))) if corrected_cosines else 0
        improvement = corr_abs - raw_abs
        
        verdict3 = "TRANSPORT_WORKS" if improvement > 0.05 else \
                   "MARGINAL" if improvement > -0.02 else \
                   "NO_IMPROVEMENT"
        
        log_status(f"  L{layer_l}: raw|cos|={raw_abs:.3f}, corrected|cos|={corr_abs:.3f}, "
                   f"improvement={improvement:+.3f} → {verdict3}")
    
    # ===== 额外分析: J vs I (identity) =====
    log_status("=== Extra: J vs Identity (how much does J deviate from I?) ===")
    
    for layer_l in sample_layers:
        deviations = []
        for s_idx in range(len(sentences)):
            if layer_l not in per_sentence_data[s_idx]:
                continue
            jvps = per_sentence_data[s_idx][layer_l]["jvps"]  # [n_vectors, d_model]
            V_local = V_shared[:jvps.shape[0]]  # [n_vectors, d_model]
            
            # J·v vs v (如果J=I, 则J·v = v)
            # cos(J·v, v) 对每个v
            cos_with_v = []
            for v_idx in range(jvps.shape[0]):
                jv = jvps[v_idx]
                v = V_local[v_idx]
                n_jv, n_v = np.linalg.norm(jv), np.linalg.norm(v)
                if n_jv > 1e-10 and n_v > 1e-10:
                    cos_with_v.append(float(np.dot(jv, v) / (n_jv * n_v)))
            
            if cos_with_v:
                deviations.append({
                    "mean_cos_Jv_v": float(np.mean(cos_with_v)),
                    "std_cos_Jv_v": float(np.std(cos_with_v)),
                    "mean_jv_norm": float(np.mean([np.linalg.norm(jvps[k]) for k in range(jvps.shape[0])])),
                    "mean_v_norm": float(np.mean([np.linalg.norm(V_local[k]) for k in range(V_local.shape[0])])),
                })
        
        if deviations:
            all_results[f"extra_j_vs_I_L{layer_l}"] = {
                "mean_cos_Jv_v": float(np.mean([d["mean_cos_Jv_v"] for d in deviations])),
                "std_cos_Jv_v": float(np.mean([d["std_cos_Jv_v"] for d in deviations])),
                "mean_jv_norm": float(np.mean([d["mean_jv_norm"] for d in deviations])),
                "mean_v_norm": float(np.mean([d["mean_v_norm"] for d in deviations])),
            }
            
            mean_cos = np.mean([d["mean_cos_Jv_v"] for d in deviations])
            mean_jv_norm = np.mean([d["mean_jv_norm"] for d in deviations])
            mean_v_norm = np.mean([d["mean_v_norm"] for d in deviations])
            
            log_status(f"  L{layer_l}: cos(J·v, v)={mean_cos:.3f}, "
                       f"||J·v||={mean_jv_norm:.2f}, ||v||={mean_v_norm:.2f}")
    
    # ===== 保存结果 =====
    # 将numpy数组转为列表以便JSON序列化
    def sanitize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [sanitize(v) for v in obj]
        return obj
    
    results_clean = sanitize(all_results)
    
    # 移除大的jvps数据，只保留汇总
    for s_idx in per_sentence_data:
        for layer_l in per_sentence_data[s_idx]:
            data = per_sentence_data[s_idx][layer_l]
            # 只保留谱分析和方向信息，不保留完整jvps
            per_sentence_data[s_idx][layer_l] = {
                "spectral": data["spectral"],
                "delta_h_norm": float(np.linalg.norm(data["delta_h"])),
                "h_l_norm": float(np.linalg.norm(data["h_l"])),
                "h_l_plus_1_norm": float(np.linalg.norm(data["h_l_plus_1"])),
                "sentence_type": data["sentence_type"],
            }
    
    results_clean["per_sentence_summary"] = sanitize(per_sentence_data)
    
    output_file = OUTPUT_DIR / f"phase223_{model_name}_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_clean, f, indent=2, ensure_ascii=False)
    
    log_status(f"  Results saved to {output_file}")
    
    # ===== 释放模型 =====
    release_model(model)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # ===== 打印最终汇总 =====
    log_status(f"\n{'='*60}")
    log_status(f"Phase 223 Summary ({model_name})")
    log_status(f"{'='*60}")
    
    log_status("\n--- Experiment 1: Spectral Stability ---")
    for layer_l, data in all_results["exp1_spectral"].items():
        log_status(f"  L{layer_l}: pearson={data['mean_pearson_log_sv']:.3f}, "
                   f"spearman={data['mean_spearman_sv']:.3f}, "
                   f"rank={data['eff_rank_mean']:.1f}±{data['eff_rank_std']:.1f}")
    
    log_status("\n--- Experiment 2: JVP vs Δh Consistency ---")
    for layer_l, data in all_results["exp2_jvp_consistency"].items():
        jvp_cos = data["jvp_abs_mean_cos"]
        dh_cos = data["delta_h_abs_mean_cos"]
        ratio = jvp_cos / max(dh_cos, 1e-10)
        log_status(f"  L{layer_l}: |JVP_cos|={jvp_cos:.3f}, |Δh_cos|={dh_cos:.3f}, "
                   f"ratio={ratio:.2f}x")
    
    log_status("\n--- Experiment 3: Transport Correction ---")
    for layer_l, data in all_results["exp3_transport"].items():
        raw = data["raw_abs_mean_cos"]
        corr = data["corrected_abs_mean_cos"]
        log_status(f"  L{layer_l}: raw|cos|={raw:.3f}, corrected|cos|={corr:.3f}, "
                   f"improvement={data['improvement']:+.3f}")
    
    return results_clean


# ===== 入口 =====
if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            try:
                run_experiment(name)
            except Exception as e:
                log_status(f"!!! {name} FAILED: {e}")
                import traceback; traceback.print_exc()
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(5)
    else:
        run_experiment(model_name)
    
    log_status("Phase 223 complete!")
