"""
Phase 220: Activation Patching + 转移矩阵线性化 + 多约束正交性
==============================================================

解决Phase 219-3分析指出的两个硬伤:
  硬伤1: MLP ablation方法未明确(Δh相关性≠因果性)
  硬伤2: 正交性只测了一对约束,且cos值本身不够小

三个实验:
  P0: Residual Stream Patching——真正的因果干预验证MLP因果性
  P1: 转移矩阵线性化检验——区分线性动力系统vs非线性门控系统
  P1: 10+约束对正交性系统测试——验证正交性是否普遍

跨模型测试: Qwen3 -> GLM4 -> DS7B (顺序执行,避免OOM)

核心方法——Residual Stream Patching:
  不直接替换组件输出(有dtype/tuple兼容问题),
  而是在residual stream上做patching:
  1. 运行clean(正确句),获取所有层的residual
  2. 运行corrupt(错误句),获取所有层的residual
  3. 对每一层l,把clean的residual从层l开始替换为corrupt的
  4. 测量最终logit的KL散度
  5. 对比:patching层l(包含MLP+Attn) vs patching层l的输入(只包含之前层)
  
  这给出了每层的"因果贡献"= patching从l开始 vs 从l+1开始的KL差异

执行时间: 2026-05-17 21:10
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
from model_utils import (load_model, get_layers, get_model_info,
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

def maybe_log(msg):
    global _last_log_time
    if time.time() - _last_log_time > _LOG_INTERVAL:
        log_status(msg)

# ===== 模型加载(支持sdpa) =====
def load_model_bf16(model_name: str):
    """BF16 + device_map='auto' + sdpa(flash内存优化)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    log_status(f"Loading {model_name} (bf16 + auto + sdpa)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # sdpa = PyTorch原生Scaled Dot Product Attention, 内存优化+速度提升
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

# ===== 测试数据 =====
SVA_PAIRS = [
    ("The cat chases", "The cats chase"),
    ("The dog runs", "The dogs run"),
    ("The bird sings", "The birds sing"),
    ("The girl reads", "The girls read"),
    ("The boy walks", "The boys walk"),
    ("The tree falls", "The trees fall"),
    ("The car moves", "The cars move"),
    ("The child plays", "The children play"),
    ("The woman writes", "The women write"),
    ("The man speaks", "The men speak"),
    ("The fish swims", "The fish swim"),
    ("The horse gallops", "The horses gallop"),
    ("The student studies", "The students study"),
    ("The teacher teaches", "The teachers teach"),
    ("The doctor works", "The doctors work"),
    ("The flower grows", "The flowers grow"),
    ("The river flows", "The rivers flow"),
    ("The star shines", "The stars shine"),
    ("The bell rings", "The bells ring"),
    ("The door opens", "The doors open"),
    ("The cat sleeps", "The cats sleep"),
    ("The dog barks", "The dogs bark"),
    ("The bird flies", "The birds fly"),
    ("The girl dances", "The girls dance"),
    ("The boy jumps", "The boys jump"),
    ("The lamp glows", "The lamps glow"),
    ("The wind blows", "The winds blow"),
    ("The rain falls", "The rains fall"),
    ("The cloud drifts", "The clouds drift"),
    ("The leaf trembles", "The leaves tremble"),
    ("The snake crawls", "The snakes crawl"),
    ("The rabbit hops", "The rabbits hop"),
    ("The fox hunts", "The foxes hunt"),
    ("The bear sleeps", "The bears sleep"),
    ("The wolf howls", "The wolves howl"),
    ("The eagle soars", "The eagles soar"),
    ("The whale dives", "The whales dive"),
    ("The ant works", "The ants work"),
    ("The bee buzzes", "The bees buzz"),
    ("The frog jumps", "The frogs jump"),
    ("The clock ticks", "The clocks tick"),
    ("The train arrives", "The trains arrive"),
    ("The ship sails", "The ships sail"),
    ("The plane flies", "The planes fly"),
    ("The book opens", "The books open"),
    ("The pen writes", "The pens write"),
    ("The cup breaks", "The cups break"),
    ("The key turns", "The keys turn"),
    ("The fire burns", "The fires burn"),
    ("The light shines", "The lights shine"),
]

TENSE_PAIRS = [
    ("The cat chases", "The cat chased"),
    ("The dog runs", "The dog ran"),
    ("The bird sings", "The bird sang"),
    ("The girl reads", "The girl read"),
    ("The boy walks", "The boy walked"),
    ("The tree falls", "The tree fell"),
    ("The car moves", "The car moved"),
    ("The child plays", "The child played"),
    ("The woman writes", "The woman wrote"),
    ("The man speaks", "The man spoke"),
    ("The fish swims", "The fish swam"),
    ("The student studies", "The student studied"),
    ("The teacher teaches", "The teacher taught"),
    ("The doctor works", "The doctor worked"),
    ("The flower grows", "The flower grew"),
    ("The bell rings", "The bell rang"),
    ("The door opens", "The door opened"),
    ("The wind blows", "The wind blew"),
    ("The rain falls", "The rain fell"),
    ("The light shines", "The light shone"),
]

GENDER_PAIRS = [
    ("The king rules", "The queen rules"),
    ("The boy walks", "The girl walks"),
    ("The man speaks", "The woman speaks"),
    ("The husband works", "The wife works"),
    ("The father reads", "The mother reads"),
    ("The brother plays", "The sister plays"),
    ("The son studies", "The daughter studies"),
    ("The uncle visits", "The aunt visits"),
    ("The nephew runs", "The niece runs"),
    ("The gentleman waits", "The lady waits"),
]

NEGATION_PAIRS = [
    ("The cat chases", "The cat does not chase"),
    ("The dog runs", "The dog does not run"),
    ("The bird sings", "The bird does not sing"),
    ("The girl reads", "The girl does not read"),
    ("The boy walks", "The boy does not walk"),
    ("All cats chase", "No cats chase"),
    ("Some dogs run", "No dogs run"),
    ("Every bird sings", "No bird sings"),
    ("Many girls read", "Few girls read"),
    ("Both boys walk", "Neither boy walks"),
]

VOICE_PAIRS = [
    ("The cat chases the mouse", "The mouse is chased by the cat"),
    ("The dog bites the bone", "The bone is bitten by the dog"),
    ("The girl reads the book", "The book is read by the girl"),
    ("The boy throws the ball", "The ball is thrown by the boy"),
    ("The teacher teaches the class", "The class is taught by the teacher"),
    ("The wind blows the leaves", "The leaves are blown by the wind"),
    ("The chef cooks the meal", "The meal is cooked by the chef"),
    ("The artist paints the wall", "The wall is painted by the artist"),
    ("The driver starts the car", "The car is started by the driver"),
    ("The writer finishes the novel", "The novel is finished by the writer"),
]

# ===== 核心计算 =====
def compute_kl(p, q, eps=1e-10):
    p = p.float() + eps
    q = q.float() + eps
    p = p / p.sum(dim=-1, keepdim=True)
    q = q / q.sum(dim=-1, keepdim=True)
    return (0.5 * (p * (p/q).log()).sum(-1) + 0.5 * (q * (q/p).log()).sum(-1)).item()

def get_all_residuals(model, tokenizer, device, text, n_layers):
    """获取所有层的residual stream (通过output_hidden_states)"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    
    # hidden_states[0] = embedding, hidden_states[l+1] = after layer l
    # 保存完整序列 [1, seq_len, d_model] 用于patching
    residuals_full = []
    residuals_last = []  # [d_model] 用于其他实验
    for l in range(n_layers + 1):
        h = out.hidden_states[l].detach()  # [1, seq_len, d_model]
        residuals_full.append(h)
        residuals_last.append(h[0, -1].detach().float().cpu())
    
    logits = out.logits[0, -1].detach().float().cpu()
    return residuals_full, residuals_last, logits

def run_from_layer(model, tokenizer, device, text, n_layers, start_layer, 
                   patch_residual_full):
    """
    从start_layer开始,用patch_residual_full替换residual stream
    
    patch_residual_full: [1, seq_len, d_model] 完整序列的hidden state
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    patched_logits = [None]
    
    def patch_hook(module, input, output):
        # output是layer的输出, 通常是一个tuple
        if isinstance(output, tuple):
            # 替换hidden states (output[0])
            patched = patch_residual_full.to(output[0].device).to(output[0].dtype)
            return (patched,) + output[1:]
        else:
            return patch_residual_full.to(output.device).to(output.dtype)
    
    def capture_logits(module, input, output):
        if isinstance(output, torch.Tensor):
            patched_logits[0] = output[0, -1].detach().float().cpu()
    
    layers = get_layers(model)
    hooks = []
    hooks.append(layers[start_layer].register_forward_hook(patch_hook))
    hooks.append(model.lm_head.register_forward_hook(capture_logits))
    
    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
    except Exception as e:
        pass  # 静默处理,避免日志洪泛
    finally:
        for h in hooks:
            h.remove()
    
    return patched_logits[0]

# ===== 实验1: Residual Stream Patching (P0) =====
def experiment_residual_patching(model, tokenizer, device, model_info, n_test=50):
    """
    Residual Stream Patching: 因果干预
    
    方法(明确数学定义):
    1. KL_clean = KL(clean_output || clean_output) = 0
    2. KL_patch_from_l = KL(clean_output || patched_output_from_l)
       patched_output_from_l: 从层l开始用corrupt的residual
    
    3. 因果贡献(l) = KL_patch_from_l - KL_patch_from_(l+1)
       = "层l对最终输出的独特贡献"
    
    4. MLP因果贡献 ≈ 因果贡献(l) - 因果贡献_attn_only(l)
       但由于我们无法单独patching MLP/Attn(HF模型限制),
       用以下替代方案:
       - 层级因果贡献 = KL_patch_from_l
       - MLP间接贡献: 比较有MLP的层vs无MLP的层
    
    正确定义:
    - "正贡献": patching后KL增大 → 该层传播了约束信息
    - "负贡献": patching后KL减小 → 该层抑制了约束信息
    """
    n_layers = model_info.n_layers
    log_status(f"[ResPatching] n_layers={n_layers}, n_test={n_test}")
    
    pairs = SVA_PAIRS[:n_test]
    sample_layers = get_sample_layers(n_layers, n_samples=14)
    log_status(f"[ResPatching] Sampling layers: {sample_layers}")
    
    results = {"layer_causal": [], "per_pair_detail": []}
    
    for l in sample_layers:
        kl_values = []
        pair_details = []
        
        for i, (sg, pl) in enumerate(pairs):
            maybe_log(f"[ResPatching] L{l}, pair {i+1}/{len(pairs)}")
            
            try:
                # 获取clean和corrupt的residuals
                res_sg_full, _, logits_sg = get_all_residuals(model, tokenizer, device, sg, n_layers)
                res_pl_full, _, logits_pl = get_all_residuals(model, tokenizer, device, pl, n_layers)
                
                # Patching: 用pl的layer l residual替换sg的
                # 从层l开始用corrupt(pl)的residual
                patched_logits = run_from_layer(
                    model, tokenizer, device, sg, n_layers, l,
                    res_pl_full[l]  # pl在层l的完整residual [1, seq_len, d_model]
                )
                
                if patched_logits is not None:
                    kl = compute_kl(logits_sg, patched_logits)
                    kl_values.append(kl)
                    pair_details.append({"pair_idx": i, "kl": kl})
                    
            except Exception as e:
                maybe_log(f"  Error L{l} pair {i}: {e}")
                continue
        
        n_valid = len(kl_values)
        result = {
            "layer": l,
            "kl_mean": float(np.mean(kl_values)) if kl_values else 0,
            "kl_std": float(np.std(kl_values)) if len(kl_values) > 1 else 0,
            "kl_median": float(np.median(kl_values)) if kl_values else 0,
            "n_valid": n_valid,
        }
        results["layer_causal"].append(result)
        
        log_status(f"  L{l}: causal_KL={result['kl_mean']:.4f} ± {result['kl_std']:.4f}, n={n_valid}")
    
    # 计算逐层增量(每层的独特贡献)
    causal_kls = [r["kl_mean"] for r in results["layer_causal"]]
    incremental = []
    for i in range(len(causal_kls) - 1):
        inc = causal_kls[i] - causal_kls[i+1]
        incremental.append({
            "from_layer": results["layer_causal"][i]["layer"],
            "to_layer": results["layer_causal"][i+1]["layer"],
            "incremental_kl": float(inc),
            "interpretation": "层传播约束(正)或抑制约束(负)" if inc > 0 else "层抑制约束(负增量)",
        })
    results["incremental_causal"] = incremental
    
    return results

# ===== 实验2: MLP vs Attention贡献分离 (P0补充) =====
def experiment_mlp_attn_separation(model, tokenizer, device, model_info, n_test=30):
    """
    分离MLP和Attention的因果贡献
    
    方法: 使用hook分别捕获MLP和Attention的输出增量,
    然后计算每个增量对最终logit变化的贡献
    
    这不是真正的activation patching(那需要替换中间激活),
    而是"贡献归因"——从clean和corrupt运行的差异中,
    计算MLP和Attention分别贡献了多少Δh
    
    数学:
    Δh(l+1) = h_sg(l+1) - h_pl(l+1)
           = (attn_out_sg - attn_out_pl) + (mlp_out_sg - mlp_out_pl)
           = Δ_attn(l) + Δ_mlp(l)
    
    最终logit变化 ≈ W_U @ Δh(L) = W_U @ Σ(Δ_attn(l) + Δ_mlp(l))
    
    MLP贡献比例 = ||W_U @ Σ Δ_mlp(l)|| / ||W_U @ Δh(L)||
    Attn贡献比例 = ||W_U @ Σ Δ_attn(l)|| / ||W_U @ Δh(L)||
    """
    n_layers = model_info.n_layers
    log_status(f"[MLP-Attn Sep] n_layers={n_layers}, n_test={n_test}")
    
    pairs = SVA_PAIRS[:n_test]
    W_U = get_W_U(model, model_info.name)  # [vocab, d_model]
    
    sample_layers = get_sample_layers(n_layers, n_samples=14)
    
    results = {"per_layer": [], "overall": {}}
    
    for l in sample_layers:
        mlp_contribs = []
        attn_contribs = []
        
        for i, (sg, pl) in enumerate(pairs):
            maybe_log(f"[MLP-Attn] L{l}, pair {i+1}/{len(pairs)}")
            
            try:
                # 获取MLP和Attention的输出
                layers = get_layers(model)
                captured = {}
                
                def make_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[key] = output[0][0, -1].detach().float().cpu()
                        else:
                            captured[key] = output[0, -1].detach().float().cpu()
                    return hook
                
                # sg运行
                hooks_sg = []
                hooks_sg.append(layers[l].self_attn.register_forward_hook(make_hook("attn_sg")))
                hooks_sg.append(layers[l].mlp.register_forward_hook(make_hook("mlp_sg")))
                
                inputs_sg = tokenizer(sg, return_tensors="pt", truncation=True, max_length=64)
                with torch.no_grad():
                    model(input_ids=inputs_sg["input_ids"].to(device),
                          attention_mask=inputs_sg["attention_mask"].to(device))
                for h in hooks_sg:
                    h.remove()
                
                # pl运行
                captured_pl = {}
                def make_hook_pl(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured_pl[key] = output[0][0, -1].detach().float().cpu()
                        else:
                            captured_pl[key] = output[0, -1].detach().float().cpu()
                    return hook
                
                hooks_pl = []
                hooks_pl.append(layers[l].self_attn.register_forward_hook(make_hook_pl("attn_pl")))
                hooks_pl.append(layers[l].mlp.register_forward_hook(make_hook_pl("mlp_pl")))
                
                inputs_pl = tokenizer(pl, return_tensors="pt", truncation=True, max_length=64)
                with torch.no_grad():
                    model(input_ids=inputs_pl["input_ids"].to(device),
                          attention_mask=inputs_pl["attention_mask"].to(device))
                for h in hooks_pl:
                    h.remove()
                
                # 计算MLP和Attention的Δ
                if "mlp_sg" in captured and "mlp_pl" in captured_pl:
                    delta_mlp = (captured["mlp_sg"] - captured_pl["mlp_pl"]).numpy()
                    mlp_logit_effect = np.linalg.norm(W_U @ delta_mlp)
                    mlp_contribs.append(mlp_logit_effect)
                
                if "attn_sg" in captured and "attn_pl" in captured_pl:
                    delta_attn = (captured["attn_sg"] - captured_pl["attn_pl"]).numpy()
                    attn_logit_effect = np.linalg.norm(W_U @ delta_attn)
                    attn_contribs.append(attn_logit_effect)
                    
            except Exception as e:
                maybe_log(f"  Error L{l} pair {i}: {e}")
                continue
        
        n_valid = min(len(mlp_contribs), len(attn_contribs))
        if n_valid > 0:
            mean_mlp = float(np.mean(mlp_contribs))
            mean_attn = float(np.mean(attn_contribs))
            mlp_pct = mean_mlp / max(mean_mlp + mean_attn, 1e-10)
            attn_pct = mean_attn / max(mean_mlp + mean_attn, 1e-10)
            
            result = {
                "layer": l,
                "mlp_logit_effect_mean": mean_mlp,
                "attn_logit_effect_mean": mean_attn,
                "mlp_pct": float(mlp_pct),
                "attn_pct": float(attn_pct),
                "n_valid": n_valid,
            }
            results["per_layer"].append(result)
            
            log_status(f"  L{l}: MLP={mlp_pct:.1%}, Attn={attn_pct:.1%}, n={n_valid}")
    
    # 汇总
    if results["per_layer"]:
        mean_mlp_pct = np.mean([r["mlp_pct"] for r in results["per_layer"]])
        mean_attn_pct = np.mean([r["attn_pct"] for r in results["per_layer"]])
        results["overall"] = {
            "mean_mlp_pct": float(mean_mlp_pct),
            "mean_attn_pct": float(mean_attn_pct),
            "mlp_dominant": mean_mlp_pct > 0.5,
        }
        log_status(f"[MLP-Attn] Overall: MLP={mean_mlp_pct:.1%}, Attn={mean_attn_pct:.1%}")
    
    return results

# ===== 实验3: 转移矩阵线性化检验 (P1) =====
def experiment_transfer_matrix(model, tokenizer, device, model_info, n_test=50):
    """
    构建层间转移矩阵 T_l: Δh(l+1) ≈ T_l @ Δh(l) + noise
    
    测量:
    - 线性化误差 = ||Δh(l+1) - T_l @ Δh(l)|| / ||Δh(l+1)||
    - T_l的特征值(是否有复特征值→解释振荡)
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    log_status(f"[TransferMatrix] n_layers={n_layers}, d_model={d_model}, n_test={n_test}")
    
    pairs = SVA_PAIRS[:n_test]
    
    # 收集所有句子对的Δh
    all_delta_h = {l: [] for l in range(n_layers + 1)}
    
    for i, (sg, pl) in enumerate(pairs):
        maybe_log(f"[TransferMatrix] pair {i+1}/{len(pairs)}")
        try:
            _, res_sg, _ = get_all_residuals(model, tokenizer, device, sg, n_layers)
            _, res_pl, _ = get_all_residuals(model, tokenizer, device, pl, n_layers)
            
            for l in range(n_layers + 1):
                delta = (res_sg[l] - res_pl[l]).numpy()
                all_delta_h[l].append(delta)
        except Exception as e:
            log_status(f"  Error pair {i}: {e}")
            continue
    
    # 构建转移矩阵并分析
    results = {"linear_errors": [], "eigenvalue_analysis": [], "summary": {}}
    
    sample_layers = get_sample_layers(n_layers, n_samples=14)
    sample_layers = [l for l in sample_layers if l < n_layers]
    
    for l in sample_layers:
        deltas_l = np.array(all_delta_h[l])      # [n_test, d_model]
        deltas_l1 = np.array(all_delta_h[l + 1])  # [n_test, d_model]
        
        if len(deltas_l) < 5:
            continue
        
        n = len(deltas_l)
        
        try:
            X = deltas_l.T   # [d_model, n]
            Y = deltas_l1.T  # [d_model, n]
            
            # Ridge regression via SVD
            U_x, S_x, Vt_x = np.linalg.svd(X, full_matrices=False)
            k = min(n - 1, len(S_x))
            T_l = Y @ Vt_x[:k].T @ np.diag(1.0 / (S_x[:k] + 1e-6)) @ U_x[:, :k].T
            
            # 预测和误差
            pred = T_l @ X
            errors = Y - pred
            rel_errors = np.linalg.norm(errors, axis=0) / (np.linalg.norm(Y, axis=0) + 1e-10)
            mean_rel_error = float(np.mean(rel_errors))
            
            # 特征值分析(前100x100子矩阵)
            sub_size = min(100, d_model)
            try:
                eigvals = np.linalg.eigvals(T_l[:sub_size, :sub_size])
                real_parts = eigvals.real
                imag_parts = eigvals.imag
                n_complex = int(np.sum(np.abs(imag_parts) > 0.01))
                max_imag = float(np.max(np.abs(imag_parts)))
                eigenvalue_result = {
                    "layer": l,
                    "n_complex_eigenvalues": n_complex,
                    "max_imaginary_part": max_imag,
                    "mean_real_part": float(np.mean(real_parts)),
                    "negative_real_count": int(np.sum(real_parts < 0)),
                    "max_real": float(np.max(real_parts)),
                    "min_real": float(np.min(real_parts)),
                }
                results["eigenvalue_analysis"].append(eigenvalue_result)
            except Exception as e:
                eigenvalue_result = {"layer": l, "error": str(e)}
                results["eigenvalue_analysis"].append(eigenvalue_result)
            
            linear_result = {
                "layer": l,
                "mean_rel_error": mean_rel_error,
                "median_rel_error": float(np.median(rel_errors)),
                "n_samples": n,
            }
            results["linear_errors"].append(linear_result)
            
            log_status(f"  L{l}: lin_err={mean_rel_error:.4f}, "
                       f"n_complex={eigenvalue_result.get('n_complex_eigenvalues', 'N/A')}, "
                       f"max_imag={eigenvalue_result.get('max_imaginary_part', 'N/A'):.4f}")
            
        except Exception as e:
            log_status(f"  L{l} error: {e}")
            results["linear_errors"].append({"layer": l, "error": str(e)})
    
    # 汇总
    valid_errors = [r["mean_rel_error"] for r in results["linear_errors"] if "mean_rel_error" in r]
    if valid_errors:
        results["summary"] = {
            "mean_linear_error": float(np.mean(valid_errors)),
            "max_linear_error": float(np.max(valid_errors)),
            "min_linear_error": float(np.min(valid_errors)),
            "linearizable": float(np.mean(valid_errors)) < 0.1,
        }
        log_status(f"[TransferMatrix] mean_err={np.mean(valid_errors):.4f}, "
                   f"linearizable={np.mean(valid_errors) < 0.1}")
    
    return results

# ===== 实验4: 多约束正交性 (P1) =====
def experiment_constraint_orthogonality(model, tokenizer, device, model_info):
    """
    测试多对约束的传播方向正交性
    
    关键改进:
    1. 4类约束对(不只是1对)
    2. 与随机基准比较(计算Z-score和p-value)
    3. 增大样本量
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    log_status(f"[Orthogonality] n_layers={n_layers}, d_model={d_model}")
    
    # 随机基准
    n_random = 10000
    random_cos = []
    for _ in range(n_random):
        v1 = np.random.randn(d_model)
        v2 = np.random.randn(d_model)
        cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        random_cos.append(abs(cos))
    random_cos = np.array(random_cos)
    random_mean = float(np.mean(random_cos))
    random_std = float(np.std(random_cos))
    random_95pct = float(np.percentile(random_cos, 95))
    log_status(f"  Random |cos|: mean={random_mean:.4f}, std={random_std:.4f}, 95th={random_95pct:.4f}")
    
    constraint_pairs = [
        ("num_vs_tense", SVA_PAIRS[:20], TENSE_PAIRS[:20], "预期独立"),
        ("num_vs_gender", SVA_PAIRS[:10], GENDER_PAIRS[:10], "预期相关(名词特征)"),
        ("num_vs_negation", SVA_PAIRS[:10], NEGATION_PAIRS[:10], "预期干扰"),
        ("tense_vs_voice", TENSE_PAIRS[:10], VOICE_PAIRS[:10], "预期相关(动词特征)"),
    ]
    
    results = {"random_baseline": {
        "mean": random_mean, "std": random_std, "95pct": random_95pct,
        "d_model": d_model,
    }, "constraint_pairs": []}
    
    sample_layers = get_sample_layers(n_layers, n_samples=10)
    
    for pair_name, pairs1, pairs2, expectation in constraint_pairs:
        n = min(len(pairs1), len(pairs2))
        log_status(f"[Orthogonality] {pair_name} ({expectation}), n={n}")
        
        layer_cos = []
        for l in sample_layers:
            cos_values = []
            for i in range(n):
                try:
                    sg1, pl1 = pairs1[i]
                    sg2, pl2 = pairs2[i]
                    
                    _, res1_sg, _ = get_all_residuals(model, tokenizer, device, sg1, n_layers)
                    _, res1_pl, _ = get_all_residuals(model, tokenizer, device, pl1, n_layers)
                    _, res2_sg, _ = get_all_residuals(model, tokenizer, device, sg2, n_layers)
                    _, res2_pl, _ = get_all_residuals(model, tokenizer, device, pl2, n_layers)
                    
                    delta1 = (res1_sg[l] - res1_pl[l]).numpy()
                    delta2 = (res2_sg[l] - res2_pl[l]).numpy()
                    
                    n1 = np.linalg.norm(delta1)
                    n2 = np.linalg.norm(delta2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos = abs(float(np.dot(delta1, delta2) / (n1 * n2)))
                        cos_values.append(cos)
                except Exception as e:
                    continue
            
            if cos_values:
                mean_cos = float(np.mean(cos_values))
                std_cos = float(np.std(cos_values))
                z_score = (mean_cos - random_mean) / max(random_std, 1e-10)
                # p-value: 观测值超过随机基准的比例
                p_value = float(np.mean(random_cos >= mean_cos))
                
                layer_cos.append({
                    "layer": l,
                    "mean_abs_cos": mean_cos,
                    "std_abs_cos": std_cos,
                    "z_score_vs_random": z_score,
                    "p_value": p_value,
                    "n_samples": len(cos_values),
                    "significant_at_05": p_value < 0.05,
                })
        
        pair_result = {
            "pair_name": pair_name,
            "expectation": expectation,
            "n_pairs": n,
            "layer_results": layer_cos,
            "overall_mean_cos": float(np.mean([r["mean_abs_cos"] for r in layer_cos])) if layer_cos else None,
            "max_layer_cos": float(max(r["mean_abs_cos"] for r in layer_cos)) if layer_cos else None,
            "min_layer_cos": float(min(r["mean_abs_cos"] for r in layer_cos)) if layer_cos else None,
            "significant_layers": sum(1 for r in layer_cos if r.get("significant_at_05", False)),
            "total_layers": len(layer_cos),
        }
        results["constraint_pairs"].append(pair_result)
        
        log_status(f"  {pair_name}: overall_mean={pair_result['overall_mean_cos']:.4f}, "
                   f"sig_layers={pair_result['significant_layers']}/{pair_result['total_layers']}")
    
    return results

# ===== 主测试流程 =====
def run_all_experiments(model_name: str):
    """在单个模型上运行所有实验"""
    log_status(f"\n{'='*60}")
    log_status(f"Phase 220: {model_name}")
    log_status(f"{'='*60}")
    
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    model_info = get_model_info(model, model_name)
    log_status(f"Model info: {model_info}")
    
    all_results = {
        "model_name": model_name,
        "model_info": {
            "class": model_info.model_class,
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
            "vocab_size": model_info.vocab_size,
            "mlp_type": model_info.mlp_type,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    try:
        # 实验1: Residual Stream Patching (P0)
        log_status("\n--- Exp 1: Residual Stream Patching (P0) ---")
        patching_results = experiment_residual_patching(model, tokenizer, device, model_info, n_test=50)
        all_results["residual_patching"] = patching_results
        
        # 实验2: MLP vs Attention分离 (P0补充)
        log_status("\n--- Exp 2: MLP vs Attention Separation (P0) ---")
        sep_results = experiment_mlp_attn_separation(model, tokenizer, device, model_info, n_test=30)
        all_results["mlp_attn_separation"] = sep_results
        
        # 实验3: 转移矩阵线性化 (P1)
        log_status("\n--- Exp 3: Transfer Matrix Linearization (P1) ---")
        transfer_results = experiment_transfer_matrix(model, tokenizer, device, model_info, n_test=50)
        all_results["transfer_matrix"] = transfer_results
        
        # 实验4: 多约束正交性 (P1)
        log_status("\n--- Exp 4: Constraint Orthogonality (P1) ---")
        ortho_results = experiment_constraint_orthogonality(model, tokenizer, device, model_info)
        all_results["constraint_orthogonality"] = ortho_results
        
    finally:
        release_model(model)
        model = None
        gc.collect()
        torch.cuda.empty_cache()
        log_status(f"Model released, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")
    
    # 保存结果
    elapsed = time.time() - t0
    all_results["elapsed_seconds"] = round(elapsed, 1)
    result_file = OUTPUT_DIR / f"phase220_{model_name}_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    log_status(f"Results saved to {result_file} ({elapsed:.0f}s)")
    
    return all_results

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name == "all":
        results_summary = {}
        for name in ["qwen3", "glm4", "deepseek7b"]:
            try:
                r = run_all_experiments(name)
                results_summary[name] = "OK"
            except Exception as e:
                log_status(f"!!! {name} FAILED: {e}")
                import traceback; traceback.print_exc()
                results_summary[name] = f"FAILED: {e}"
            
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(5)
        
        log_status(f"\nSummary: {json.dumps(results_summary, indent=2)}")
    else:
        run_all_experiments(model_name)
    
    log_status("Phase 220 complete!")

if __name__ == "__main__":
    main()
