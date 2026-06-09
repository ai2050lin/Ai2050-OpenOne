"""
Phase 405: Norm/Entropy Mechanism Localization
================================================

核心问题:
1. 分布压缩/膨胀从哪个处理步骤开始?
2. Qwen3/DS7B的even为负(压缩) vs GLM4的even为正(膨胀)来自哪里?
3. RMSNorm在每一步对候选分布的entropy/variance/gap做了什么?

测试设计:
- 在每个检查点捕获residual stream状态:
  1. pre-RMSNorm (layer input)
  2. post-RMSNorm (after input layernorm)
  3. after attention output (attn residual added)
  4. post-attn-RMSNorm (before MLP)
  5. after MLP output (MLP residual added)
  6. final layer output

- 对每个检查点:
  a. 用lm_head读取候选分布 (logits for 8 speed candidates)
  b. 计算: entropy, variance, top-logit gap, rank correlation
  c. 计算odd/even效应 (patch +direction vs -direction)

- 测试对象: 6个对象, 2层(早+深), 3模型

关键指标:
- entropy_trajectory: 熵在各处理步骤的变化轨迹
- variance_trajectory: 方差轨迹
- top_gap_trajectory: 顶部候选差距轨迹
- rank_corr_trajectory: 排序相关轨迹
- odd/even_at_each_step: 每步的方向效应和范数效应

Usage:
  python tests/glm5/phase405_norm_entropy_mechanism.py qwen3
  python tests/glm5/phase405_norm_entropy_mechanism.py deepseek7b
  python tests/glm5/phase405_norm_entropy_mechanism.py glm4
"""

import sys
import os
import json
import time
import gc
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict, OrderedDict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "glm5"))

from model_utils import MODEL_CONFIGS, get_layers, get_model_info, release_model, get_W_U

# ===== 配置 =====

SPEED_CANDIDATES = OrderedDict([
    ("sluggish", 1), ("slow", 2), ("steady", 3), ("moderate", 4),
    ("quick", 5), ("fast", 6), ("rapid", 7), ("swift", 8),
])

# 6 objects: 2 per type (fast + slow)
SPEED_OBJECTS = OrderedDict([
    ("snail",     {"type": "animal",     "speed_level": 1, "target": "slow",   "comp": "fast"}),
    ("cheetah",   {"type": "animal",     "speed_level": 5, "target": "fast",   "comp": "slow"}),
    ("bicycle",  {"type": "vehicle",    "speed_level": 2, "target": "slow",   "comp": "fast"}),
    ("rocket",   {"type": "vehicle",    "speed_level": 5, "target": "fast",   "comp": "slow"}),
    ("glacier",  {"type": "phenomenon", "speed_level": 1, "target": "slow",   "comp": "fast"}),
    ("lightning", {"type": "phenomenon", "speed_level": 5, "target": "fast",   "comp": "slow"}),
])

# Extended objects for confirmation round (12 total)
SPEED_OBJECTS_EXTENDED = OrderedDict([
    # Animal
    ("snail",     {"type": "animal",     "speed_level": 1, "target": "slow",   "comp": "fast"}),
    ("turtle",    {"type": "animal",     "speed_level": 1, "target": "slow",   "comp": "fast"}),
    ("cheetah",   {"type": "animal",     "speed_level": 5, "target": "fast",   "comp": "slow"}),
    ("falcon",    {"type": "animal",     "speed_level": 5, "target": "fast",   "comp": "slow"}),
    # Vehicle
    ("bicycle",   {"type": "vehicle",    "speed_level": 2, "target": "slow",   "comp": "fast"}),
    ("cart",      {"type": "vehicle",    "speed_level": 1, "target": "slow",   "comp": "fast"}),
    ("rocket",    {"type": "vehicle",    "speed_level": 5, "target": "fast",   "comp": "slow"}),
    ("jet",       {"type": "vehicle",    "speed_level": 5, "target": "fast",   "comp": "slow"}),
    # Phenomenon
    ("glacier",   {"type": "phenomenon", "speed_level": 1, "target": "slow",   "comp": "fast"}),
    ("erosion",   {"type": "phenomenon", "speed_level": 1, "target": "slow",   "comp": "fast"}),
    ("lightning", {"type": "phenomenon", "speed_level": 5, "target": "fast",   "comp": "slow"}),
    ("explosion", {"type": "phenomenon", "speed_level": 5, "target": "fast",   "comp": "slow"}),
])

FRAMES = [
    "The {obj} is {attr}.",
    "An {obj} is {attr}.",
]

CORRUPT_FRAMES = [
    "The item is {attr}.",
    "An item is {attr}.",
]

LAYER_CONFIGS = {
    "qwen3": [4, 28],
    "deepseek7b": [4, 20],
    "glm4": [5, 35],
}

# Number of layers to scan for mechanism localization
SCAN_LAYERS = {
    "qwen3": [0, 4, 12, 20, 28, 35],
    "deepseek7b": [0, 4, 10, 16, 20, 27],
    "glm4": [0, 5, 15, 25, 35, 39],
}


def log_memory():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU: {alloc:.2f}GB alloc, {reserved:.2f}GB reserved"
    return "GPU not available"


def load_model_bf16_safe(model_name):
    """BF16 + device_map=auto 加载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"[{time.strftime('%H:%M:%S')}] Loading {model_name} (BF16+auto)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = None
    for impl in ["eager", "sdpa"]:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
                trust_remote_code=True, local_files_only=True, attn_implementation=impl
            )
            break
        except Exception as e:
            print(f"  attn_implementation={impl} failed: {e}")
            continue
    if model is None:
        raise RuntimeError(f"Failed to load {model_name}")
    model.eval()
    print(f"  Loaded. {log_memory()}")
    return model, tokenizer


def compute_distribution_metrics(logits, candidate_ids, speed_levels):
    """
    计算候选分布的多个指标
    
    Args:
        logits: 全词表logits (numpy array)
        candidate_ids: 候选词token ID列表
        speed_levels: 对应速度等级列表
    
    Returns:
        dict of metrics
    """
    # 候选词logits
    cand_logits = np.array([logits[cid] if cid is not None else float('-inf') for cid in candidate_ids])
    valid_mask = np.array([cid is not None for cid in candidate_ids])
    
    if valid_mask.sum() < 2:
        return {"entropy": 0, "variance": 0, "top_gap": 0, "rank_corr": 0,
                "speed_gradient": 0, "cand_logits": cand_logits.tolist()}
    
    # Softmax to probabilities
    max_logit = np.max(cand_logits[valid_mask])
    exp_logits = np.exp(cand_logits - max_logit)
    exp_logits[~valid_mask] = 0
    total = np.sum(exp_logits)
    probs = exp_logits / total if total > 0 else np.zeros_like(exp_logits)
    
    # 1. Entropy (概率分布熵)
    valid_probs = probs[valid_mask]
    valid_probs = valid_probs[valid_probs > 0]
    entropy = -np.sum(valid_probs * np.log(valid_probs)) if len(valid_probs) > 0 else 0
    
    # 2. Variance (logit方差)
    variance = float(np.var(cand_logits[valid_mask])) if valid_mask.sum() > 1 else 0
    
    # 3. Top-logit gap (最高和第二高的差距)
    sorted_logits = np.sort(cand_logits[valid_mask])[::-1]
    top_gap = float(sorted_logits[0] - sorted_logits[1]) if len(sorted_logits) > 1 else 0
    
    # 4. Rank correlation (Spearman: 速度等级 vs logit排序)
    valid_levels = np.array(speed_levels)[valid_mask]
    valid_cand_logits = cand_logits[valid_mask]
    if len(valid_levels) > 2:
        # Compute Spearman rank correlation
        from scipy.stats import spearmanr
        corr, _ = spearmanr(valid_levels, valid_cand_logits)
        rank_corr = float(corr) if not np.isnan(corr) else 0
    else:
        rank_corr = 0
    
    # 5. Speed-level gradient (速度等级对logit的线性回归斜率)
    if len(valid_levels) > 1:
        slope = np.polyfit(valid_levels, valid_cand_logits, 1)[0]
        speed_gradient = float(slope)
    else:
        speed_gradient = 0
    
    return {
        "entropy": float(entropy),
        "variance": float(variance),
        "top_gap": float(top_gap),
        "rank_corr": float(rank_corr),
        "speed_gradient": float(speed_gradient),
    }


def get_checkpoints_for_layer(layer, mlp_type):
    """
    获取一层的内部检查点模块
    
    返回: {checkpoint_name: module} 的字典
    """
    checkpoints = OrderedDict()
    
    # Checkpoint 1: Input layernorm (pre-RMSNorm -> post-RMSNorm)
    for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
        if hasattr(layer, ln_name):
            checkpoints["post_input_ln"] = getattr(layer, ln_name)
            break
    
    # Checkpoint 2: Self-attention output
    for sa_name in ["self_attn", "attention", "attn"]:
        if hasattr(layer, sa_name):
            sa = getattr(layer, sa_name)
            for oname in ["o_proj", "dense", "out_proj"]:
                if hasattr(sa, oname):
                    checkpoints["attn_out"] = getattr(sa, oname)
                    break
            break
    
    # Checkpoint 3: Post-attention layernorm
    for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
        if hasattr(layer, ln_name):
            checkpoints["post_attn_ln"] = getattr(layer, ln_name)
            break
    
    # Checkpoint 4: MLP output (down_proj)
    if hasattr(layer, "mlp"):
        mlp = layer.mlp
        for dname in ["down_proj", "dense_4h_to_h"]:
            if hasattr(mlp, dname):
                checkpoints["mlp_down"] = getattr(mlp, dname)
                break
    
    return checkpoints


def run_forward_with_hooks(model, tokenizer, device, prompt, W_U_tensor,
                           candidate_ids, speed_levels, target_layer_idx,
                           layer_module, direction_np=None, beta=8.0):
    """
    运行前向传播, 在目标检查点捕获残差流, 计算分布指标
    
    Args:
        model: 模型
        tokenizer: 分词器
        device: 设备
        prompt: 输入文本
        W_U_tensor: lm_head权重 tensor (on device)
        candidate_ids: 候选词ID列表
        speed_levels: 速度等级列表
        target_layer_idx: 目标层索引
        layer_module: 要hook的模块 (layer自身, 或子模块)
        direction_np: 注入方向 (numpy), None=不注入
        beta: 注入强度
    
    Returns:
        dict of metrics at this checkpoint, plus final logits metrics
    """
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # Hook to capture output of target module
    captured = {}
    def make_hook(key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook_fn
    
    handle = layer_module.register_forward_hook(make_hook('checkpoint'))
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    
    handle.remove()
    
    # Get final logits
    final_logits = out.logits[0, -1].float().cpu().numpy()
    
    # Compute metrics at checkpoint
    checkpoint_metrics = {}
    if 'checkpoint' in captured:
        h = captured['checkpoint'][0, -1].numpy()  # [d_model]
        
        # Project to logits via W_U
        checkpoint_logits = (W_U_tensor @ h).cpu().numpy()  # direct projection
        
        checkpoint_metrics = compute_distribution_metrics(
            checkpoint_logits, candidate_ids, speed_levels
        )
    
    # Final logits metrics
    final_metrics = compute_distribution_metrics(final_logits, candidate_ids, speed_levels)
    
    # All hidden states metrics (for trajectory across layers)
    hs = out.hidden_states
    layer_trajectory = {}
    for li, h_state in enumerate(hs):
        h_vec = h_state[0, -1].float().cpu().numpy()
        layer_logits = (W_U_tensor @ h_vec).cpu().numpy()
        layer_metrics = compute_distribution_metrics(layer_logits, candidate_ids, speed_levels)
        layer_trajectory[str(li)] = {
            "entropy": layer_metrics["entropy"],
            "variance": layer_metrics["variance"],
            "top_gap": layer_metrics["top_gap"],
            "speed_gradient": layer_metrics["speed_gradient"],
        }
    
    return {
        "checkpoint_metrics": checkpoint_metrics,
        "final_metrics": final_metrics,
        "layer_trajectory": layer_trajectory,
    }


def compute_direction_at_layer(model, tokenizer, device, layers_list, li,
                                obj_name, obj_data, token_ids):
    """
    计算速度方向 (clean vs corrupt的残差差)
    """
    target = obj_data["target"]
    
    h_correct_list = []
    h_corrupt_list = []
    
    captured = {}
    def make_hook(key):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook_fn
    
    handle = layers_list[li].register_forward_hook(make_hook('h'))
    
    for f_idx in range(2):
        correct_clean = FRAMES[f_idx].format(obj=obj_name, attr=target)
        correct_corrupt = CORRUPT_FRAMES[f_idx].format(attr=target)
        
        captured.clear()
        inputs = tokenizer(correct_clean, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            model(input_ids=inputs["input_ids"].to(device),
                  attention_mask=inputs["attention_mask"].to(device))
        h_correct_list.append(captured['h'][0, -1].numpy())
        
        captured.clear()
        inputs = tokenizer(correct_corrupt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            model(input_ids=inputs["input_ids"].to(device),
                  attention_mask=inputs["attention_mask"].to(device))
        h_corrupt_list.append(captured['h'][0, -1].numpy())
    
    handle.remove()
    dh = np.mean(np.array(h_correct_list) - np.array(h_corrupt_list), axis=0)
    return dh


def run_phase405(model_name, use_extended=False):
    """
    Phase 405主函数: 范数/熵机制定位
    
    策略:
    1. 对每个目标层, 在6个内部检查点注入方向
    2. 记录每步的候选分布指标变化
    3. 追踪entropy/variance/gap的压缩/膨胀轨迹
    """
    timestamp = time.strftime("%Y-%m-%d %H:%M")
    objects = SPEED_OBJECTS_EXTENDED if use_extended else SPEED_OBJECTS
    obj_names = sorted(objects.keys())
    
    print(f"\n{'='*80}")
    print(f"=== Phase 405: Norm/Entropy Mechanism ({model_name}) [{timestamp}] ===")
    print(f"{'='*80}")
    print(f"  Objects: {len(obj_names)}, Extended: {use_extended}")
    
    # Load model
    model, tokenizer = load_model_bf16_safe(model_name)
    layers_list = get_layers(model)
    info = get_model_info(model, model_name)
    mlp_type = info.mlp_type
    device = next(model.parameters()).device
    
    # Get W_U
    W_U_np = get_W_U(model, model_name)  # [vocab, d_model]
    W_U_tensor = torch.tensor(W_U_np, dtype=torch.float32, device="cpu")
    print(f"  W_U: shape={W_U_np.shape}, n_layers={info.n_layers}")
    
    # Resolve token IDs for speed candidates
    candidate_ids = []
    speed_levels = []
    cand_names = []
    for cand_name, level in SPEED_CANDIDATES.items():
        ids = tokenizer.encode(cand_name, add_special_tokens=False)
        tid = ids[0] if ids else None
        candidate_ids.append(tid)
        speed_levels.append(level)
        cand_names.append(cand_name)
    
    print(f"  Candidates: {dict(zip(cand_names, candidate_ids))}")
    
    # Also resolve target/comp token IDs
    token_ids = {}
    for cand_name in SPEED_CANDIDATES:
        ids = tokenizer.encode(cand_name, add_special_tokens=False)
        token_ids[cand_name] = ids[0] if ids else None
    for obj_name, obj_data in objects.items():
        for tok in [obj_data["target"], obj_data["comp"]]:
            if tok not in token_ids:
                ids = tokenizer.encode(tok, add_special_tokens=False)
                token_ids[tok] = ids[0] if ids else None
    
    layer_indices = LAYER_CONFIGS.get(model_name, [4])
    scan_layers = SCAN_LAYERS.get(model_name, [0, info.n_layers//2, info.n_layers-1])
    # Ensure scan layers don't exceed n_layers
    scan_layers = [li for li in scan_layers if li < info.n_layers]
    
    all_results = {
        "model": model_name,
        "timestamp": timestamp,
        "extended": use_extended,
        "n_layers": info.n_layers,
        "d_model": info.d_model,
        "candidate_ids": {n: int(tid) if tid is not None else None for n, tid in zip(cand_names, candidate_ids)},
        "per_layer": {},
        "trajectory": {},
    }
    
    # ====== Part A: Baseline trajectory (no patching) ======
    print(f"\n{'='*70}")
    print(f"=== Part A: Baseline Entropy Trajectory ===")
    
    # For each object, run clean forward and track entropy at every layer
    baseline_trajectories = {}
    for obj_name in obj_names:
        obj_data = objects[obj_name]
        prompt = FRAMES[0].format(obj=obj_name, attr=obj_data["target"])
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            out = model(input_ids=inputs["input_ids"].to(device),
                       attention_mask=inputs["attention_mask"].to(device),
                       output_hidden_states=True)
        
        hs = out.hidden_states
        traj = {}
        for li, h_state in enumerate(hs):
            h_vec = h_state[0, -1].float().cpu().numpy()
            layer_logits = W_U_np @ h_vec
            metrics = compute_distribution_metrics(layer_logits, candidate_ids, speed_levels)
            traj[str(li)] = {
                "entropy": metrics["entropy"],
                "variance": metrics["variance"],
                "top_gap": metrics["top_gap"],
                "speed_gradient": metrics["speed_gradient"],
                "rank_corr": metrics["rank_corr"],
            }
        
        baseline_trajectories[obj_name] = traj
        
        # Print key layers
        key_layers = [0, info.n_layers//2, info.n_layers-1]
        key_layers = [li for li in key_layers if li < len(hs)]
        traj_summary = ", ".join(f"L{li}:H={traj[str(li)]['entropy']:.3f}" for li in key_layers)
        print(f"  {obj_name}: {traj_summary}")
    
    all_results["trajectory"] = baseline_trajectories
    
    # ====== Part B: Checkpoint-level injection ======
    print(f"\n{'='*70}")
    print(f"=== Part B: Checkpoint-Level Injection ===")
    
    for li in layer_indices:
        t0_layer = time.time()
        print(f"\n{'='*70}")
        print(f"--- Layer {li} ---")
        
        layer = layers_list[li]
        checkpoints = get_checkpoints_for_layer(layer, mlp_type)
        print(f"  Checkpoints: {list(checkpoints.keys())}")
        
        # Compute speed directions for representative objects
        print(f"\n  Computing speed directions...")
        obj_dirs = {}
        for obj_name in obj_names:
            obj_data = objects[obj_name]
            dh = compute_direction_at_layer(model, tokenizer, device, layers_list, li,
                                             obj_name, obj_data, token_ids)
            obj_dirs[obj_name] = dh
            print(f"    {obj_name}: |dir|={np.linalg.norm(dh):.4f}")
        
        # For each checkpoint, inject direction and measure distribution change
        # Test objects: 2 per type (fast representative)
        test_objects = [n for n in ["cheetah", "rocket", "lightning", "snail", "bicycle", "glacier"]
                       if n in objects]
        
        layer_results = {
            "checkpoint_analysis": {},
            "baseline_trajectory_at_layer": {},
        }
        
        # Get baseline trajectory at this layer
        for obj_name in test_objects:
            if obj_name in baseline_trajectories:
                layer_results["baseline_trajectory_at_layer"][obj_name] = \
                    baseline_trajectories[obj_name].get(str(li), {})
        
        for cp_name, cp_module in checkpoints.items():
            print(f"\n  Checkpoint: {cp_name}")
            cp_results = {}
            
            for obj_name in test_objects:
                obj_data = objects[obj_name]
                direction = obj_dirs[obj_name]
                target = obj_data["target"]
                comp = obj_data["comp"]
                tid = token_ids.get(target)
                cid = token_ids.get(comp)
                
                if tid is None or cid is None:
                    continue
                
                # Skip if module output dim doesn't match direction dim
                if hasattr(cp_module, 'weight'):
                    w_shape = cp_module.weight.shape
                    module_out_dim = w_shape[0]
                    if module_out_dim != direction.shape[0]:
                        continue
                
                prompt = CORRUPT_FRAMES[0].format(attr=target)
                inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                
                # === +direction patch ===
                delta_pos = torch.tensor(direction, dtype=torch.bfloat16, device=device)
                def make_add_hook_pos(module, input, output):
                    if isinstance(output, tuple):
                        hs = output[0].clone()
                        hs[0, -1, :] += delta_pos
                        return (hs,) + output[1:]
                    else:
                        hs = output.clone()
                        hs[0, -1, :] += delta_pos
                        return hs
                
                # Capture residual stream at this checkpoint AND final output
                captured_cp = {}
                def make_capture_hook(key):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            captured_cp[key] = output[0].detach().float().cpu()
                        else:
                            captured_cp[key] = output.detach().float().cpu()
                    return hook_fn
                
                h_patch = cp_module.register_forward_hook(make_add_hook_pos)
                h_capture = layers_list[li].register_forward_hook(make_capture_hook('residual_after_patch'))
                
                with torch.no_grad():
                    out_plus = model(input_ids=input_ids, attention_mask=attention_mask,
                                    output_hidden_states=True)
                
                h_patch.remove()
                h_capture.remove()
                
                # Metrics from final logits
                final_logits_plus = out_plus.logits[0, -1].float().cpu().numpy()
                final_metrics_plus = compute_distribution_metrics(final_logits_plus, candidate_ids, speed_levels)
                
                # Metrics from checkpoint residual
                if 'residual_after_patch' in captured_cp:
                    h_cp = captured_cp['residual_after_patch'][0, -1].numpy()
                    cp_logits_plus = W_U_np @ h_cp
                    cp_metrics_plus = compute_distribution_metrics(cp_logits_plus, candidate_ids, speed_levels)
                else:
                    cp_metrics_plus = {}
                
                # === -direction patch ===
                delta_neg = torch.tensor(-direction, dtype=torch.bfloat16, device=device)
                def make_add_hook_neg(module, input, output):
                    if isinstance(output, tuple):
                        hs = output[0].clone()
                        hs[0, -1, :] += delta_neg
                        return (hs,) + output[1:]
                    else:
                        hs = output.clone()
                        hs[0, -1, :] += delta_neg
                        return hs
                
                captured_cp.clear()
                h_patch2 = cp_module.register_forward_hook(make_add_hook_neg)
                h_capture2 = layers_list[li].register_forward_hook(make_capture_hook('residual_after_patch'))
                
                with torch.no_grad():
                    out_minus = model(input_ids=input_ids, attention_mask=attention_mask,
                                     output_hidden_states=True)
                
                h_patch2.remove()
                h_capture2.remove()
                
                final_logits_minus = out_minus.logits[0, -1].float().cpu().numpy()
                final_metrics_minus = compute_distribution_metrics(final_logits_minus, candidate_ids, speed_levels)
                
                if 'residual_after_patch' in captured_cp:
                    h_cp = captured_cp['residual_after_patch'][0, -1].numpy()
                    cp_logits_minus = W_U_np @ h_cp
                    cp_metrics_minus = compute_distribution_metrics(cp_logits_minus, candidate_ids, speed_levels)
                else:
                    cp_metrics_minus = {}
                
                # === Baseline (no patch) ===
                captured_cp.clear()
                h_capture3 = layers_list[li].register_forward_hook(make_capture_hook('residual_baseline'))
                
                with torch.no_grad():
                    out_base = model(input_ids=input_ids, attention_mask=attention_mask,
                                    output_hidden_states=True)
                
                h_capture3.remove()
                
                final_logits_base = out_base.logits[0, -1].float().cpu().numpy()
                final_metrics_base = compute_distribution_metrics(final_logits_base, candidate_ids, speed_levels)
                
                if 'residual_baseline' in captured_cp:
                    h_cp = captured_cp['residual_baseline'][0, -1].numpy()
                    cp_logits_base = W_U_np @ h_cp
                    cp_metrics_base = compute_distribution_metrics(cp_logits_base, candidate_ids, speed_levels)
                else:
                    cp_metrics_base = {}
                
                # === Compute odd/even at this checkpoint ===
                # For target vs comp logit difference
                diff_plus = float(final_logits_plus[tid] - final_logits_plus[cid])
                diff_minus = float(final_logits_minus[tid] - final_logits_minus[cid])
                diff_base = float(final_logits_base[tid] - final_logits_base[cid])
                
                eff_plus = diff_plus - diff_base
                eff_minus = diff_minus - diff_base
                odd = (eff_plus - eff_minus) / 2
                even = (eff_plus + eff_minus) / 2
                
                # Entropy change (odd/even decomposition)
                entropy_plus = final_metrics_plus["entropy"]
                entropy_minus = final_metrics_minus["entropy"]
                entropy_base = final_metrics_base["entropy"]
                
                entropy_eff_plus = entropy_plus - entropy_base
                entropy_eff_minus = entropy_minus - entropy_base
                entropy_odd = (entropy_eff_plus - entropy_eff_minus) / 2
                entropy_even = (entropy_eff_plus + entropy_eff_minus) / 2
                
                # Variance change
                var_plus = final_metrics_plus["variance"]
                var_minus = final_metrics_minus["variance"]
                var_base = final_metrics_base["variance"]
                
                var_eff_plus = var_plus - var_base
                var_eff_minus = var_minus - var_base
                var_odd = (var_eff_plus - var_eff_minus) / 2
                var_even = (var_eff_plus + var_eff_minus) / 2
                
                # Speed gradient change
                sg_plus = final_metrics_plus["speed_gradient"]
                sg_minus = final_metrics_minus["speed_gradient"]
                sg_base = final_metrics_base["speed_gradient"]
                
                sg_eff_plus = sg_plus - sg_base
                sg_eff_minus = sg_minus - sg_base
                sg_odd = (sg_eff_plus - sg_eff_minus) / 2
                sg_even = (sg_eff_plus + sg_eff_minus) / 2
                
                # Checkpoint-level metrics (at residual stream after patch)
                cp_entropy_odd = 0
                cp_entropy_even = 0
                if cp_metrics_plus and cp_metrics_minus and cp_metrics_base:
                    cp_e_plus = cp_metrics_plus.get("entropy", 0) - cp_metrics_base.get("entropy", 0)
                    cp_e_minus = cp_metrics_minus.get("entropy", 0) - cp_metrics_base.get("entropy", 0)
                    cp_entropy_odd = (cp_e_plus - cp_e_minus) / 2
                    cp_entropy_even = (cp_e_plus + cp_e_minus) / 2
                
                cp_result = {
                    "logit_odd": float(odd),
                    "logit_even": float(even),
                    "entropy_base": float(entropy_base),
                    "entropy_plus": float(entropy_plus),
                    "entropy_minus": float(entropy_minus),
                    "entropy_odd": float(entropy_odd),
                    "entropy_even": float(entropy_even),
                    "variance_base": float(var_base),
                    "variance_odd": float(var_odd),
                    "variance_even": float(var_even),
                    "speed_gradient_base": float(sg_base),
                    "speed_gradient_odd": float(sg_odd),
                    "speed_gradient_even": float(sg_even),
                    "top_gap_base": float(final_metrics_base["top_gap"]),
                    "top_gap_plus": float(final_metrics_plus["top_gap"]),
                    "top_gap_minus": float(final_metrics_minus["top_gap"]),
                    "rank_corr_base": float(final_metrics_base["rank_corr"]),
                    "rank_corr_plus": float(final_metrics_plus["rank_corr"]),
                    "rank_corr_minus": float(final_metrics_minus["rank_corr"]),
                    "cp_entropy_odd": float(cp_entropy_odd),
                    "cp_entropy_even": float(cp_entropy_even),
                }
                
                cp_results[obj_name] = cp_result
            
            # Aggregate across objects for this checkpoint
            if cp_results:
                agg = defaultdict(list)
                for obj_name, res in cp_results.items():
                    for key, val in res.items():
                        agg[key].append(val)
                
                agg_means = {k: float(np.mean(v)) for k, v in agg.items()}
                cp_summary = {
                    "per_object": cp_results,
                    "aggregate": agg_means,
                }
                layer_results["checkpoint_analysis"][cp_name] = cp_summary
                
                print(f"    Agg: logit_odd={agg_means['logit_odd']:+.4f}, "
                      f"logit_even={agg_means['logit_even']:+.4f}, "
                      f"entropy_odd={agg_means['entropy_odd']:+.4f}, "
                      f"entropy_even={agg_means['entropy_even']:+.4f}, "
                      f"var_even={agg_means['variance_even']:+.4f}")
        
        # ====== Part C: Layer-level injection (residual stream) ======
        print(f"\n  Part C: Layer-level residual injection...")
        residual_results = {}
        
        for obj_name in test_objects:
            obj_data = objects[obj_name]
            direction = obj_dirs[obj_name]
            target = obj_data["target"]
            comp = obj_data["comp"]
            tid = token_ids.get(target)
            cid = token_ids.get(comp)
            
            if tid is None or cid is None:
                continue
            
            prompt = CORRUPT_FRAMES[0].format(attr=target)
            
            # +direction at residual stream
            delta = torch.tensor(direction, dtype=torch.bfloat16, device=device)
            def make_res_hook_pos(module, input, output):
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    hs[0, -1, :] += delta
                    return (hs,) + output[1:]
                else:
                    hs = output.clone()
                    hs[0, -1, :] += delta
                    return hs
            
            h1 = layers_list[li].register_forward_hook(make_res_hook_pos)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                out_p = model(input_ids=inputs["input_ids"].to(device),
                             attention_mask=inputs["attention_mask"].to(device))
            h1.remove()
            
            # -direction at residual stream
            neg_delta = torch.tensor(-direction, dtype=torch.bfloat16, device=device)
            def make_res_hook_neg(module, input, output):
                if isinstance(output, tuple):
                    hs = output[0].clone()
                    hs[0, -1, :] += neg_delta
                    return (hs,) + output[1:]
                else:
                    hs = output.clone()
                    hs[0, -1, :] += neg_delta
                    return hs
            
            h2 = layers_list[li].register_forward_hook(make_res_hook_neg)
            with torch.no_grad():
                out_m = model(input_ids=inputs["input_ids"].to(device),
                             attention_mask=inputs["attention_mask"].to(device))
            h2.remove()
            
            # Baseline
            with torch.no_grad():
                out_b = model(input_ids=inputs["input_ids"].to(device),
                             attention_mask=inputs["attention_mask"].to(device))
            
            logits_p = out_p.logits[0, -1].float().cpu().numpy()
            logits_m = out_m.logits[0, -1].float().cpu().numpy()
            logits_b = out_b.logits[0, -1].float().cpu().numpy()
            
            diff_p = float(logits_p[tid] - logits_p[cid])
            diff_m = float(logits_m[tid] - logits_m[cid])
            diff_b = float(logits_b[tid] - logits_b[cid])
            
            odd = ((diff_p - diff_b) - (diff_m - diff_b)) / 2
            even = ((diff_p - diff_b) + (diff_m - diff_b)) / 2
            
            # Distribution metrics
            metrics_p = compute_distribution_metrics(logits_p, candidate_ids, speed_levels)
            metrics_m = compute_distribution_metrics(logits_m, candidate_ids, speed_levels)
            metrics_b = compute_distribution_metrics(logits_b, candidate_ids, speed_levels)
            
            entropy_even = ((metrics_p["entropy"] - metrics_b["entropy"]) + 
                          (metrics_m["entropy"] - metrics_b["entropy"])) / 2
            var_even = ((metrics_p["variance"] - metrics_b["variance"]) + 
                       (metrics_m["variance"] - metrics_b["variance"])) / 2
            
            residual_results[obj_name] = {
                "logit_odd": float(odd),
                "logit_even": float(even),
                "entropy_base": float(metrics_b["entropy"]),
                "entropy_even": float(entropy_even),
                "variance_base": float(metrics_b["variance"]),
                "variance_even": float(var_even),
                "speed_gradient_base": float(metrics_b["speed_gradient"]),
            }
        
        # Aggregate residual results
        if residual_results:
            agg_res = defaultdict(list)
            for obj_name, res in residual_results.items():
                for key, val in res.items():
                    agg_res[key].append(val)
            layer_results["residual_injection"] = {
                "per_object": residual_results,
                "aggregate": {k: float(np.mean(v)) for k, v in agg_res.items()},
            }
            agg = layer_results["residual_injection"]["aggregate"]
            print(f"  Residual: logit_odd={agg['logit_odd']:+.4f}, "
                  f"logit_even={agg['logit_even']:+.4f}, "
                  f"entropy_even={agg['entropy_even']:+.4f}")
        
        all_results["per_layer"][str(li)] = layer_results
        print(f"\n  L{li} done in {time.time()-t0_layer:.0f}s. {log_memory()}")
    
    # ====== Part D: Cross-layer entropy trajectory comparison ======
    print(f"\n{'='*70}")
    print(f"=== Part D: Cross-Layer Entropy Trajectory ===")
    
    # For each type, compute mean entropy trajectory
    type_trajectories = defaultdict(lambda: defaultdict(list))
    for obj_name, traj in baseline_trajectories.items():
        obj_type = objects[obj_name]["type"]
        for li_str, metrics in traj.items():
            type_trajectories[obj_type][li_str].append(metrics["entropy"])
    
    type_mean_entropy = {}
    for obj_type, li_dict in type_trajectories.items():
        type_mean_entropy[obj_type] = {
            li: float(np.mean(v)) for li, v in li_dict.items()
        }
    
    # Print key layers for each type
    key_layers = [0, info.n_layers//4, info.n_layers//2, 3*info.n_layers//4, info.n_layers-1]
    key_layers = [li for li in key_layers if li < info.n_layers]
    
    print(f"  {'Type':>12s} " + " ".join(f"L{li:>3d}" for li in key_layers))
    for obj_type in ["animal", "vehicle", "phenomenon"]:
        vals = [type_mean_entropy.get(obj_type, {}).get(str(li), 0) for li in key_layers]
        print(f"  {obj_type:>12s} " + " ".join(f"{v:.3f}" for v in vals))
    
    all_results["type_mean_entropy"] = type_mean_entropy
    
    # ====== Part E: RMSNorm effect analysis ======
    print(f"\n{'='*70}")
    print(f"=== Part E: RMSNorm Effect on Distribution ===")
    
    # At each target layer, measure entropy before and after input layernorm
    rmsnorm_effects = {}
    for li in layer_indices:
        layer = layers_list[li]
        
        # Get input layernorm
        input_ln = None
        for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(layer, ln_name):
                input_ln = getattr(layer, ln_name)
                break
        
        if input_ln is None:
            continue
        
        # Hook before and after layernorm
        captured_pre = {}
        captured_post = {}
        
        def make_pre_hook(key):
            def hook_fn(module, input, output):
                if isinstance(input, tuple) and len(input) > 0:
                    captured_pre[key] = input[0].detach().float().cpu()
            return hook_fn
        
        def make_post_hook(key):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    captured_post[key] = output[0].detach().float().cpu()
                else:
                    captured_post[key] = output.detach().float().cpu()
            return hook_fn
        
        ln_effects = {}
        for obj_name in test_objects:
            obj_data = objects[obj_name]
            prompt = FRAMES[0].format(obj=obj_name, attr=obj_data["target"])
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            
            captured_pre.clear()
            captured_post.clear()
            
            h_pre = input_ln.register_forward_hook(make_pre_hook('pre'))
            h_post = input_ln.register_forward_hook(make_post_hook('post'))
            
            # Actually, pre_hook should be on the layer, not on ln
            # Let's capture layer input (pre-ln) and ln output (post-ln)
            h_layer_pre = layers_list[li].register_forward_hook(make_pre_hook('layer_input'))
            
            with torch.no_grad():
                model(input_ids=inputs["input_ids"].to(device),
                      attention_mask=inputs["attention_mask"].to(device))
            
            h_pre.remove()
            h_post.remove()
            h_layer_pre.remove()
            
            # Compute entropy at pre-ln and post-ln
            pre_entropy = 0
            post_entropy = 0
            pre_norm = 0
            post_norm = 0
            
            if 'layer_input' in captured_pre:
                h_pre_vec = captured_pre['layer_input'][0, -1].numpy()
                pre_logits = W_U_np @ h_pre_vec
                pre_metrics = compute_distribution_metrics(pre_logits, candidate_ids, speed_levels)
                pre_entropy = pre_metrics["entropy"]
                pre_norm = float(np.linalg.norm(h_pre_vec))
            
            if 'post' in captured_post:
                h_post_vec = captured_post['post'][0, -1].numpy()
                post_logits = W_U_np @ h_post_vec
                post_metrics = compute_distribution_metrics(post_logits, candidate_ids, speed_levels)
                post_entropy = post_metrics["entropy"]
                post_norm = float(np.linalg.norm(h_post_vec))
            
            delta_entropy = post_entropy - pre_entropy
            delta_norm = post_norm - pre_norm if pre_norm > 0 else 0
            
            ln_effects[obj_name] = {
                "pre_entropy": float(pre_entropy),
                "post_entropy": float(post_entropy),
                "delta_entropy": float(delta_entropy),
                "pre_norm": float(pre_norm),
                "post_norm": float(post_norm),
                "norm_ratio": float(post_norm / pre_norm) if pre_norm > 0 else 0,
            }
        
        rmsnorm_effects[str(li)] = ln_effects
        
        # Aggregate
        if ln_effects:
            mean_delta_e = np.mean([v["delta_entropy"] for v in ln_effects.values()])
            mean_norm_ratio = np.mean([v["norm_ratio"] for v in ln_effects.values()])
            print(f"  L{li}: delta_entropy={mean_delta_e:+.4f}, norm_ratio={mean_norm_ratio:.4f}")
    
    all_results["rmsnorm_effects"] = rmsnorm_effects
    
    # ====== Save results ======
    out_dir = ROOT / "results" / "phase405_norm_entropy"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_extended" if use_extended else ""
    out_path = out_dir / f"{model_name}_phase405{suffix}.json"
    
    import copy
    results_to_save = copy.deepcopy(all_results)
    with open(out_path, 'w') as f:
        json.dump(results_to_save, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")
    
    # ====== Summary ======
    print(f"\n{'='*80}")
    print(f"=== Phase 405 Summary ({model_name}) ===")
    print(f"{'='*80}")
    
    for li in layer_indices:
        lr = all_results["per_layer"].get(str(li), {})
        print(f"\n  L{li}:")
        
        # Checkpoint analysis summary
        cp_analysis = lr.get("checkpoint_analysis", {})
        for cp_name in ["post_input_ln", "attn_out", "post_attn_ln", "mlp_down"]:
            if cp_name in cp_analysis:
                agg = cp_analysis[cp_name].get("aggregate", {})
                print(f"    {cp_name:>16s}: logit_odd={agg.get('logit_odd',0):+.4f}, "
                      f"logit_even={agg.get('logit_even',0):+.4f}, "
                      f"entropy_even={agg.get('entropy_even',0):+.4f}, "
                      f"var_even={agg.get('variance_even',0):+.4f}")
        
        # Residual injection
        res_inj = lr.get("residual_injection", {}).get("aggregate", {})
        if res_inj:
            print(f"    {'residual':>16s}: logit_odd={res_inj.get('logit_odd',0):+.4f}, "
                  f"logit_even={res_inj.get('logit_even',0):+.4f}, "
                  f"entropy_even={res_inj.get('entropy_even',0):+.4f}")
    
    # Cleanup
    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Released. {log_memory()}")
    
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    extended = "--extended" in sys.argv
    
    # Phase 405: First round (6 objects)
    print(f"\n{'#'*80}")
    print(f"# Phase 405 Round 1: Basic test ({model_name})")
    print(f"{'#'*80}")
    results = run_phase405(model_name, use_extended=False)
    
    print(f"\nRound 1 complete for {model_name}")
