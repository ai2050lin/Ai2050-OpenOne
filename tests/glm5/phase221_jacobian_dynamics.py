"""
Phase 221: Jacobian动力学——从可解释性到神经语言动力学
=====================================================

核心目标：验证"约束传播是状态依赖的Jacobian场"这一核心假说

三个实验（按优先级）:
  P0: 方向注入Activation Patching——解决"归因≠因果"问题
      注入约束方向到MLP vs Attention，测量哪个对最终KL影响更大
  P1: Jacobian谱追踪——计算J_l = ∂h_{l+1}/∂h_l
      分析谱半径、特征值分布、奇异值谱
  P2: 有效秩分析——计算rank(J_l)和奇异值分布
      测试"语言约束传播是否在低维流形上"

跨模型测试: Qwen3 -> GLM4 -> DS7B (顺序执行,避免OOM)

执行时间: 2026-05-18 01:30
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
    ("The man works", "The men work"),
    ("The woman dances", "The women dance"),
    ("The fish swims", "The fish swim"),
    ("The horse gallops", "The horses gallop"),
    ("The student writes", "The students write"),
    ("The teacher speaks", "The teachers speak"),
    ("The cat sleeps", "The cats sleep"),
    ("The dog barks", "The dogs bark"),
    ("The bird flies", "The birds fly"),
    ("The girl sings", "The girls sing"),
    ("The boy runs", "The boys run"),
    ("The tree grows", "The trees grow"),
    ("The flower blooms", "The flowers bloom"),
    ("The river flows", "The rivers flow"),
    ("The star shines", "The stars shine"),
    ("The moon rises", "The moons rise"),
    ("The sun sets", "The suns set"),
    ("The cloud moves", "The clouds move"),
    ("The wind blows", "The winds blow"),
    ("The rain falls", "The rains fall"),
    ("The snow melts", "The snows melt"),
    ("The fire burns", "The fires burn"),
    ("The light shines", "The lights shine"),
    ("The sound echoes", "The sounds echo"),
    ("The door opens", "The doors open"),
    ("The window breaks", "The windows break"),
    ("The book falls", "The books fall"),
    ("The pen writes", "The pens write"),
    ("The phone rings", "The phones ring"),
    ("The clock ticks", "The clocks tick"),
    ("The bell rings", "The bells ring"),
    ("The flag waves", "The flags wave"),
    ("The candle burns", "The candles burn"),
    ("The glass shatters", "The glasses shatter"),
    ("The key turns", "The keys turn"),
    ("The wheel spins", "The wheels spin"),
    ("The engine roars", "The engines roar"),
    ("The train stops", "The trains stop"),
    ("The boat sails", "The boats sail"),
    ("The plane flies", "The planes fly"),
    ("The rocket launches", "The rockets launch"),
]

# ===== 工具函数 =====
def compute_kl(p_logits, q_logits):
    """KL(p || q)，p和q是logits tensor"""
    p = torch.softmax(p_logits.float(), dim=-1)
    q = torch.softmax(q_logits.float(), dim=-1)
    q = q.clamp(min=1e-10)
    p = p.clamp(min=1e-10)
    return float((p * (p / q).log()).sum().item())

def get_hidden_states(model, tokenizer, device, text, n_layers):
    """获取所有层的hidden states（最后一个token）"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    
    # hidden_states[0]=embedding, hidden_states[l+1]=after layer l
    h_states = []
    for l in range(n_layers + 1):
        h = out.hidden_states[l][0, -1].detach().float().cpu()
        h_states.append(h)
    
    logits = out.logits[0, -1].detach().float().cpu()
    return h_states, logits

def get_full_hidden_states(model, tokenizer, device, text, n_layers):
    """获取所有层的hidden states（完整序列 [1, seq_len, d_model]）"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    
    h_states_full = []
    for l in range(n_layers + 1):
        h = out.hidden_states[l].detach()  # [1, seq_len, d_model]
        h_states_full.append(h)
    
    logits = out.logits[0, -1].detach().float().cpu()
    return h_states_full, logits

# ===== 实验1: 方向注入Activation Patching (P0) =====
def experiment_direction_patching(model, tokenizer, device, model_info, n_test=40):
    """
    方向注入Patching：解决"归因≠因果"问题
    
    核心思路：
    1. 计算约束方向 d_c = h_correct_l - h_wrong_l（在MLP和Attn输出上分别计算）
    2. 在测试句的MLP/Attn输出上注入这个方向
    3. 测量注入MLP vs 注入Attn对最终KL的影响差异
    
    如果MLP注入效果 >> Attn注入 → MLP是因果主导
    如果两者相当 → 归因法高估了MLP
    """
    n_layers = model_info.n_layers
    log_status(f"[DirPatching] n_layers={n_layers}, n_test={n_test}")
    
    pairs = SVA_PAIRS[:n_test]
    sample_layers = get_sample_layers(n_layers, n_samples=10)
    log_status(f"[DirPatching] Sampling layers: {sample_layers}")
    
    results = {"per_layer": [], "overall": {}}
    layers_list = get_layers(model)
    
    for l in sample_layers:
        mlp_kls = []
        attn_kls = []
        
        for i, (sg, pl) in enumerate(pairs):
            maybe_log(f"[DirPatching] L{l}, pair {i+1}/{len(pairs)}")
            
            try:
                # Step 1: 获取约束方向
                # 运行sg和pl，用hook捕获MLP和Attn的输出
                captured_sg = {}
                captured_pl = {}
                
                def make_hook_sg(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured_sg[key] = output[0][0, -1].detach().float()
                        else:
                            captured_sg[key] = output[0, -1].detach().float()
                    return hook
                
                def make_hook_pl(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured_pl[key] = output[0][0, -1].detach().float()
                        else:
                            captured_pl[key] = output[0, -1].detach().float()
                    return hook
                
                # sg运行
                h1 = layers_list[l].self_attn.register_forward_hook(make_hook_sg("attn"))
                h2 = layers_list[l].mlp.register_forward_hook(make_hook_sg("mlp"))
                
                inputs_sg = tokenizer(sg, return_tensors="pt", truncation=True, max_length=64)
                with torch.no_grad():
                    out_sg = model(input_ids=inputs_sg["input_ids"].to(device),
                                  attention_mask=inputs_sg["attention_mask"].to(device))
                logits_sg = out_sg.logits[0, -1].detach().float().cpu()
                h1.remove(); h2.remove()
                
                # pl运行
                h3 = layers_list[l].self_attn.register_forward_hook(make_hook_pl("attn"))
                h4 = layers_list[l].mlp.register_forward_hook(make_hook_pl("mlp"))
                
                inputs_pl = tokenizer(pl, return_tensors="pt", truncation=True, max_length=64)
                with torch.no_grad():
                    out_pl = model(input_ids=inputs_pl["input_ids"].to(device),
                                  attention_mask=inputs_pl["attention_mask"].to(device))
                logits_pl = out_pl.logits[0, -1].detach().float().cpu()
                h3.remove(); h4.remove()
                
                if "attn" not in captured_sg or "mlp" not in captured_sg:
                    continue
                if "attn" not in captured_pl or "mlp" not in captured_pl:
                    continue
                
                # 约束方向
                d_attn = (captured_sg["attn"] - captured_pl["attn"]).to(device)  # [d_model]
                d_mlp = (captured_sg["mlp"] - captured_pl["mlp"]).to(device)     # [d_model]
                
                # Step 2: 在pl句上注入约束方向，测量KL
                # 2a: 注入到MLP输出
                def mlp_inject_hook(module, input, output):
                    if isinstance(output, tuple):
                        out_tensor = output[0].clone()
                        out_tensor[0, -1, :] += d_mlp.to(out_tensor.dtype)
                        return (out_tensor,) + output[1:]
                    else:
                        out_tensor = output.clone()
                        out_tensor[0, -1, :] += d_mlp.to(out_tensor.dtype)
                        return out_tensor
                
                h_mlp = layers_list[l].mlp.register_forward_hook(mlp_inject_hook)
                with torch.no_grad():
                    out_mlp_patched = model(
                        input_ids=inputs_pl["input_ids"].to(device),
                        attention_mask=inputs_pl["attention_mask"].to(device)
                    )
                logits_mlp_patched = out_mlp_patched.logits[0, -1].detach().float().cpu()
                h_mlp.remove()
                
                kl_mlp = compute_kl(logits_sg, logits_mlp_patched)
                mlp_kls.append(kl_mlp)
                
                # 2b: 注入到Attn输出
                def attn_inject_hook(module, input, output):
                    if isinstance(output, tuple):
                        out_tensor = output[0].clone()
                        out_tensor[0, -1, :] += d_attn.to(out_tensor.dtype)
                        return (out_tensor,) + output[1:]
                    else:
                        out_tensor = output.clone()
                        out_tensor[0, -1, :] += d_attn.to(out_tensor.dtype)
                        return out_tensor
                
                h_attn = layers_list[l].self_attn.register_forward_hook(attn_inject_hook)
                with torch.no_grad():
                    out_attn_patched = model(
                        input_ids=inputs_pl["input_ids"].to(device),
                        attention_mask=inputs_pl["attention_mask"].to(device)
                    )
                logits_attn_patched = out_attn_patched.logits[0, -1].detach().float().cpu()
                h_attn.remove()
                
                kl_attn = compute_kl(logits_sg, logits_attn_patched)
                attn_kls.append(kl_attn)
                
                # 清理GPU
                del out_sg, out_pl, out_mlp_patched, out_attn_patched
                del captured_sg, captured_pl
                if i % 5 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                maybe_log(f"  Error L{l} pair {i}: {e}")
                continue
        
        n_valid = min(len(mlp_kls), len(attn_kls))
        if n_valid > 0:
            mean_mlp_kl = float(np.mean(mlp_kls))
            mean_attn_kl = float(np.mean(attn_kls))
            total = mean_mlp_kl + mean_attn_kl
            mlp_pct = mean_mlp_kl / max(total, 1e-10)
            attn_pct = mean_attn_kl / max(total, 1e-10)
            
            # 方向注入因果比: MLP注入/Attn注入
            causal_ratio = mean_mlp_kl / max(mean_attn_kl, 1e-10)
            
            result = {
                "layer": l,
                "mlp_inject_kl_mean": mean_mlp_kl,
                "attn_inject_kl_mean": mean_attn_kl,
                "mlp_causal_pct": float(mlp_pct),
                "attn_causal_pct": float(attn_pct),
                "causal_ratio": float(causal_ratio),
                "n_valid": n_valid,
            }
            results["per_layer"].append(result)
            
            log_status(f"  L{l}: MLP_inject_KL={mean_mlp_kl:.4f}, Attn_inject_KL={mean_attn_kl:.4f}, "
                       f"ratio={causal_ratio:.2f}, n={n_valid}")
    
    # 汇总
    if results["per_layer"]:
        mean_mlp_kl = np.mean([r["mlp_inject_kl_mean"] for r in results["per_layer"]])
        mean_attn_kl = np.mean([r["attn_inject_kl_mean"] for r in results["per_layer"]])
        total = mean_mlp_kl + mean_attn_kl
        mlp_pct = mean_mlp_kl / max(total, 1e-10)
        mean_ratio = np.mean([r["causal_ratio"] for r in results["per_layer"]])
        
        results["overall"] = {
            "mean_mlp_causal_pct": float(mlp_pct),
            "mean_attn_causal_pct": float(1 - mlp_pct),
            "mean_causal_ratio": float(mean_ratio),
            "mlp_causally_dominant": bool(mlp_pct > 0.5),
        }
        log_status(f"[DirPatching] Overall: MLP_causal={mlp_pct:.1%}, "
                   f"ratio={mean_ratio:.2f}, dominant={'MLP' if mlp_pct > 0.5 else 'Attn'}")
    
    return results

# ===== 实验2: Jacobian谱追踪 (P1) =====
def experiment_jacobian_spectrum(model, tokenizer, device, model_info, n_test=30):
    """
    计算层间Jacobian J_l = ∂h_{l+1}/∂h_l
    
    方法：有限差分法
    J_l ≈ (h_{l+1}(h_l + ε·e_i) - h_{l+1}(h_l - ε·e_i)) / (2ε)
    
    但这需要d_model次前向传播，太慢。
    
    替代方案：利用已知的线性近似
    T_l（Phase 220的转移矩阵）就是J_l在平均状态下的近似
    
    更精确的方法：利用自动微分
    J_l = ∂h_{l+1}/∂h_l 可以通过torch.autograd计算
    
    但HF模型不方便做autograd，所以我们用以下方法：
    1. 对同一句子对，计算 Δh(l) 和 Δh(l+1)
    2. 对多个句子对，用最小二乘法估计 T_l ≈ J_l
    3. 分析T_l的奇异值谱（= Jacobian的"有效秩"）
    
    这与Phase 220的转移矩阵方法相同，但现在重点在谱分析。
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    log_status(f"[JacobianSpectrum] n_layers={n_layers}, d_model={d_model}, n_test={n_test}")
    
    pairs = SVA_PAIRS[:n_test]
    
    # 收集所有句子对的Δh
    all_delta_h = {l: [] for l in range(n_layers + 1)}
    
    for i, (sg, pl) in enumerate(pairs):
        maybe_log(f"[Jacobian] pair {i+1}/{len(pairs)}")
        try:
            res_sg, _ = get_hidden_states(model, tokenizer, device, sg, n_layers)
            res_pl, _ = get_hidden_states(model, tokenizer, device, pl, n_layers)
            
            for l in range(n_layers + 1):
                delta = (res_sg[l] - res_pl[l]).numpy()
                all_delta_h[l].append(delta)
        except Exception as e:
            log_status(f"  Error pair {i}: {e}")
            continue
    
    # 对每层构建T_l并做谱分析
    sample_layers = get_sample_layers(n_layers, n_samples=12)
    sample_layers = [l for l in sample_layers if l < n_layers]
    
    results = {"per_layer": [], "overall": {}, "green_function": []}
    
    # 存储所有T_l用于Green Function计算
    T_matrices = {}
    
    for l in sample_layers:
        deltas_l = np.array(all_delta_h[l])      # [n_test, d_model]
        deltas_l1 = np.array(all_delta_h[l + 1])  # [n_test, d_model]
        
        if len(deltas_l) < 5:
            continue
        
        n = len(deltas_l)
        
        try:
            # 构建T_l
            X = deltas_l.T   # [d_model, n]
            Y = deltas_l1.T  # [d_model, n]
            
            U_x, S_x, Vt_x = np.linalg.svd(X, full_matrices=False)
            k = min(n - 1, len(S_x))
            T_l = Y @ Vt_x[:k].T @ np.diag(1.0 / (S_x[:k] + 1e-6)) @ U_x[:, :k].T
            T_matrices[l] = T_l
            
            # 奇异值分解（Jacobian的谱）
            svd_result = np.linalg.svd(T_l, compute_uv=False)
            svd_result = np.sort(svd_result)[::-1]
            
            # 有效秩（Effective Rank）
            # 定义：exp(entropy of normalized singular values)
            sv_norm = svd_result / (np.sum(svd_result) + 1e-10)
            sv_norm = sv_norm[sv_norm > 1e-15]  # 去掉零
            entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-30))
            effective_rank = float(np.exp(entropy))
            
            # 90%能量秩
            cumulative = np.cumsum(svd_result**2)
            total_energy = cumulative[-1]
            rank_90 = int(np.searchsorted(cumulative, 0.9 * total_energy)) + 1
            rank_95 = int(np.searchsorted(cumulative, 0.95 * total_energy)) + 1
            rank_99 = int(np.searchsorted(cumulative, 0.99 * total_energy)) + 1
            
            # 谱半径（最大奇异值）和条件数
            spectral_radius = float(svd_result[0])
            condition_number = float(svd_result[0] / max(svd_result[-1], 1e-10))
            
            # 增强方向：奇异值>1的比例
            n_amplifying = int(np.sum(svd_result > 1.0))
            n_attenuating = int(np.sum((svd_result > 0.01) & (svd_result < 1.0)))
            n_near_zero = int(np.sum(svd_result < 0.01))
            
            # 特征值分析（100x100子矩阵）
            sub_size = min(100, d_model)
            eigvals = np.linalg.eigvals(T_l[:sub_size, :sub_size])
            real_parts = eigvals.real
            imag_parts = eigvals.imag
            n_complex = int(np.sum(np.abs(imag_parts) > 0.01))
            
            # 负实特征值比例
            n_negative_real = int(np.sum((np.abs(imag_parts) < 0.01) & (real_parts < -0.001)))
            
            result = {
                "layer": l,
                "effective_rank": effective_rank,
                "d_model": d_model,
                "rank_ratio": float(effective_rank / d_model),
                "rank_90pct_energy": rank_90,
                "rank_95pct_energy": rank_95,
                "rank_99pct_energy": rank_99,
                "spectral_radius": spectral_radius,
                "condition_number": condition_number,
                "n_amplifying_sv": n_amplifying,
                "n_attenuating_sv": n_attenuating,
                "n_near_zero_sv": n_near_zero,
                "n_complex_eigenvalues": n_complex,
                "n_negative_real_eigenvalues": n_negative_real,
                "top_10_singular_values": [float(x) for x in svd_result[:10]],
                "singular_value_at_10pct": float(svd_result[len(svd_result)//10]) if len(svd_result) > 10 else 0,
                "singular_value_at_50pct": float(svd_result[len(svd_result)//2]),
                "singular_value_at_90pct": float(svd_result[int(len(svd_result)*0.9)]),
            }
            results["per_layer"].append(result)
            
            log_status(f"  L{l}: eff_rank={effective_rank:.0f}/{d_model} "
                       f"({effective_rank/d_model:.1%}), "
                       f"ρ={spectral_radius:.3f}, "
                       f"rank_90%={rank_90}, "
                       f"amplifying={n_amplifying}, "
                       f"complex_eig={n_complex}, "
                       f"neg_real={n_negative_real}")
            
        except Exception as e:
            log_status(f"  Error L{l}: {e}")
            continue
    
    # Green Function计算
    log_status("[Jacobian] Computing Green Function G(l1→l2)...")
    sorted_layers = sorted(T_matrices.keys())
    
    # 选择几个关键层对
    key_layers = sorted_layers[:3] + sorted_layers[len(sorted_layers)//2:len(sorted_layers)//2+1] + sorted_layers[-2:]
    
    for l1 in key_layers:
        for l2 in key_layers:
            if l2 <= l1:
                continue
            # G(l1→l2) = T_{l2-1} · T_{l2-2} · ... · T_{l1}
            layers_between = [l for l in sorted_layers if l1 <= l < l2]
            if not layers_between:
                continue
            
            G = np.eye(d_model)
            for l in layers_between:
                if l in T_matrices:
                    G = T_matrices[l] @ G
            
            # Green Function的谱
            svd_G = np.linalg.svd(G, compute_uv=False)
            svd_G = np.sort(svd_G)[::-1]
            
            # 传播衰减：最大的奇异值
            propagation_strength = float(svd_G[0])
            
            # 有效秩
            sv_norm = svd_G / (np.sum(svd_G) + 1e-10)
            sv_norm = sv_norm[sv_norm > 1e-15]
            if len(sv_norm) > 0:
                entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-30))
                eff_rank_G = float(np.exp(entropy))
            else:
                eff_rank_G = 0
            
            results["green_function"].append({
                "from_layer": l1,
                "to_layer": l2,
                "n_hops": l2 - l1,
                "propagation_strength": propagation_strength,
                "effective_rank": eff_rank_G,
                "rank_ratio": float(eff_rank_G / d_model) if d_model > 0 else 0,
                "top_5_sv": [float(x) for x in svd_G[:5]],
            })
            
            log_status(f"  G({l1}→{l2}): strength={propagation_strength:.4f}, "
                       f"eff_rank={eff_rank_G:.0f}/{d_model}")
    
    # 汇总
    if results["per_layer"]:
        mean_eff_rank = np.mean([r["effective_rank"] for r in results["per_layer"]])
        mean_rank_ratio = np.mean([r["rank_ratio"] for r in results["per_layer"]])
        mean_spectral_radius = np.mean([r["spectral_radius"] for r in results["per_layer"]])
        mean_rank_90 = np.mean([r["rank_90pct_energy"] for r in results["per_layer"]])
        
        results["overall"] = {
            "mean_effective_rank": float(mean_eff_rank),
            "mean_rank_ratio": float(mean_rank_ratio),
            "mean_spectral_radius": float(mean_spectral_radius),
            "mean_rank_90pct_energy": float(mean_rank_90),
            "d_model": d_model,
            "low_dimensional": bool(mean_rank_ratio < 0.5),
        }
        log_status(f"[Jacobian] Overall: eff_rank={mean_eff_rank:.0f}/{d_model} "
                   f"({mean_rank_ratio:.1%}), ρ={mean_spectral_radius:.3f}, "
                   f"rank_90%={mean_rank_90:.0f}")
    
    return results

# ===== 实验3: 约束子空间维度分析 (P2) =====
def experiment_constraint_subspace(model, tokenizer, device, model_info, n_test=40):
    """
    分析约束信号在残差流中占据的维度
    
    方法：
    1. 收集多个约束对的Δh
    2. 对每层的Δh做PCA，分析主成分
    3. 测量约束信号的"有效维度"
    
    如果有效维度 << d_model → 约束在低维子空间传播
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    log_status(f"[SubspaceDim] n_layers={n_layers}, d_model={d_model}, n_test={n_test}")
    
    pairs = SVA_PAIRS[:n_test]
    
    # 收集所有句子对的Δh
    all_delta_h = {l: [] for l in range(n_layers + 1)}
    
    for i, (sg, pl) in enumerate(pairs):
        maybe_log(f"[SubspaceDim] pair {i+1}/{len(pairs)}")
        try:
            res_sg, _ = get_hidden_states(model, tokenizer, device, sg, n_layers)
            res_pl, _ = get_hidden_states(model, tokenizer, device, pl, n_layers)
            
            for l in range(n_layers + 1):
                delta = (res_sg[l] - res_pl[l]).numpy()
                all_delta_h[l].append(delta)
        except Exception as e:
            log_status(f"  Error pair {i}: {e}")
            continue
    
    sample_layers = get_sample_layers(n_layers, n_samples=10)
    results = {"per_layer": [], "overall": {}}
    
    for l in sample_layers:
        deltas = np.array(all_delta_h[l])  # [n_test, d_model]
        
        if len(deltas) < 5:
            continue
        
        try:
            # PCA via SVD
            # 中心化
            deltas_centered = deltas - deltas.mean(axis=0, keepdims=True)
            
            # SVD: deltas_centered = U @ S @ Vt
            # Vt的行就是主成分方向
            U, S, Vt = np.linalg.svd(deltas_centered, full_matrices=False)
            
            # 解释方差比
            explained_var = S**2 / (np.sum(S**2) + 1e-10)
            cumulative_var = np.cumsum(explained_var)
            
            # 90%方差需要的成分数
            n_90 = int(np.searchsorted(cumulative_var, 0.9)) + 1
            n_95 = int(np.searchsorted(cumulative_var, 0.95)) + 1
            n_99 = int(np.searchsorted(cumulative_var, 0.99)) + 1
            
            # 有效秩
            sv_norm = S / (np.sum(S) + 1e-10)
            sv_norm = sv_norm[sv_norm > 1e-15]
            entropy = -np.sum(sv_norm * np.log(sv_norm + 1e-30))
            effective_dim = float(np.exp(entropy))
            
            # 第一个主成分的解释方差
            first_pc_var = float(explained_var[0]) if len(explained_var) > 0 else 0
            
            # 前10个主成分的累计解释方差
            top10_var = float(np.sum(explained_var[:10])) if len(explained_var) >= 10 else float(np.sum(explained_var))
            
            result = {
                "layer": l,
                "effective_dimension": effective_dim,
                "d_model": d_model,
                "dim_ratio": float(effective_dim / d_model),
                "n_components_90pct": n_90,
                "n_components_95pct": n_95,
                "n_components_99pct": n_99,
                "first_pc_variance": first_pc_var,
                "top10_pc_variance": top10_var,
                "top_10_singular_values": [float(x) for x in S[:10]],
                "n_samples": len(deltas),
            }
            results["per_layer"].append(result)
            
            log_status(f"  L{l}: eff_dim={effective_dim:.1f}/{d_model} "
                       f"({effective_dim/d_model:.1%}), "
                       f"90%var={n_90} PCs, "
                       f"1st_pc={first_pc_var:.1%}, "
                       f"top10={top10_var:.1%}")
            
        except Exception as e:
            log_status(f"  Error L{l}: {e}")
            continue
    
    # 汇总
    if results["per_layer"]:
        mean_eff_dim = np.mean([r["effective_dimension"] for r in results["per_layer"]])
        mean_dim_ratio = np.mean([r["dim_ratio"] for r in results["per_layer"]])
        mean_90 = np.mean([r["n_components_90pct"] for r in results["per_layer"]])
        mean_first_pc = np.mean([r["first_pc_variance"] for r in results["per_layer"]])
        mean_top10 = np.mean([r["top10_pc_variance"] for r in results["per_layer"]])
        
        results["overall"] = {
            "mean_effective_dimension": float(mean_eff_dim),
            "mean_dim_ratio": float(mean_dim_ratio),
            "mean_n_components_90pct": float(mean_90),
            "mean_first_pc_variance": float(mean_first_pc),
            "mean_top10_pc_variance": float(mean_top10),
            "d_model": d_model,
            "low_dimensional": bool(mean_dim_ratio < 0.3),
        }
        log_status(f"[SubspaceDim] Overall: eff_dim={mean_eff_dim:.1f}/{d_model} "
                   f"({mean_dim_ratio:.1%}), 90%var={mean_90:.0f} PCs, "
                   f"1st_pc={mean_first_pc:.1%}, top10={mean_top10:.1%}")
    
    return results

# ===== 主函数 =====
def run_all_experiments(model_name: str):
    """运行所有实验"""
    log_status(f"{'='*60}")
    log_status(f"Phase 221: Jacobian Dynamics - {model_name}")
    log_status(f"{'='*60}")
    
    # 加载模型
    model, tokenizer, device = load_model_sdpa(model_name)
    model_info = get_model_info(model, model_name)
    
    log_status(f"Model: {model_info.model_class}, layers={model_info.n_layers}, "
               f"d_model={model_info.d_model}, mlp_type={model_info.mlp_type}")
    
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
    
    # ===== 实验1: 方向注入Patching (P0) =====
    log_status(f"\n{'='*40}")
    log_status("Exp 1: Direction Injection Patching (P0)")
    log_status(f"{'='*40}")
    try:
        all_results["direction_patching"] = experiment_direction_patching(
            model, tokenizer, device, model_info, n_test=40
        )
    except Exception as e:
        log_status(f"Exp 1 failed: {e}")
        import traceback; traceback.print_exc()
    
    # ===== 实验2: Jacobian谱追踪 (P1) =====
    log_status(f"\n{'='*40}")
    log_status("Exp 2: Jacobian Spectrum (P1)")
    log_status(f"{'='*40}")
    try:
        all_results["jacobian_spectrum"] = experiment_jacobian_spectrum(
            model, tokenizer, device, model_info, n_test=30
        )
    except Exception as e:
        log_status(f"Exp 2 failed: {e}")
        import traceback; traceback.print_exc()
    
    # ===== 实验3: 约束子空间维度 (P2) =====
    log_status(f"\n{'='*40}")
    log_status("Exp 3: Constraint Subspace Dimension (P2)")
    log_status(f"{'='*40}")
    try:
        all_results["constraint_subspace"] = experiment_constraint_subspace(
            model, tokenizer, device, model_info, n_test=40
        )
    except Exception as e:
        log_status(f"Exp 3 failed: {e}")
        import traceback; traceback.print_exc()
    
    # 保存结果
    result_path = OUTPUT_DIR / f"phase221_{model_name}_results.json"
    
    # 处理NaN和Inf
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif isinstance(obj, np.floating):
            v = float(obj)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return sanitize(obj.tolist())
        return obj
    
    all_results = sanitize(all_results)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log_status(f"Results saved to {result_path}")
    
    # 释放模型
    release_model(model)
    
    return all_results

if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            log_status(f"\n{'#'*60}")
            log_status(f"# Starting {name}")
            log_status(f"{'#'*60}")
            run_all_experiments(name)
            gc.collect()
            torch.cuda.empty_cache()
            log_status(f"Waiting 30s for GPU cleanup...")
            time.sleep(30)
    else:
        run_all_experiments(model_name)
    
    log_status("Phase 221 complete!")
