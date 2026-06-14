"""
Phase 490: 功能子空间层位连续曲线与末层分解
==============================================

Phase 489关键发现:
1. shared_semantic因果效应模型特异(DS7B/GLM4=刹车, Qwen3=支撑)
2. shared_semantic因果效应层位特异(早层=支撑, 中晚层=刹车/支撑)
3. 末层orth_bc消融一致削弱边界(跨模型一致)
4. orth_bc是混合空间(支撑+抑制+读出共存)
5. DS7B food L13: ablate_competitor→-1.407 (比ablate_bc(-0.127)大一个量级!)

Phase 490核心目标:
- Exp1: shared_semantic + proj_bc + competitor全层扫描 → 找功能转换点
- Exp2: 末层orth_bc子空间分解 → 读出支撑/抑制/格式
- Exp3: 竞争类别边界因果测试 → 验证DS7B food的强competitor效应
- Exp4: 早层vs中晚层shared_semantic对比 → 确认层位功能分化

用法:
  python tests/glm5/phase490_layer_sweep_decomposition.py qwen3 1
  python tests/glm5/phase490_layer_sweep_decomposition.py glm4 1
  python tests/glm5/phase490_layer_sweep_decomposition.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json, math
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS, safe_decode)


def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 数据定义 ====================
CATEGORIES = {
    "fruit":     ["apple", "banana", "orange", "grape", "pear", "peach", "mango", "plum"],
    "animal":    ["dog", "cat", "horse", "lion", "bear", "rabbit", "eagle", "fish"],
    "clothing":  ["shirt", "dress", "hat", "coat", "jacket", "skirt", "scarf", "boot"],
    "food":      ["bread", "rice", "cheese", "pasta", "soup", "cake", "salad", "meat"],
    "vehicle":   ["car", "bus", "bicycle", "truck", "train", "plane", "boat", "motorcycle"],
    "plant":     ["tree", "flower", "grass", "bush", "fern", "moss", "vine", "shrub"],
    "tool":      ["hammer", "saw", "drill", "wrench", "pliers", "chisel", "ruler", "knife"],
    "furniture": ["chair", "table", "desk", "bed", "sofa", "shelf", "cabinet", "bench"],
}

CAT_NAMES = list(CATEGORIES.keys())

TEMPLATE = "The {obj} is a kind of"


def get_model_and_tokenizer(model_name):
    """BF16加载模型"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name} (bfloat16 + device_map=auto)...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True,
        attn_implementation="eager",
    )
    model.eval()
    
    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        plog(f"{model_name}: GPU={gpu_count}, CPU={cpu_count}")
    
    device = next(model.parameters()).device
    return model, tokenizer, device


def encode_prompts(tokenizer, device, objects, template=TEMPLATE):
    """编码提示"""
    texts = [template.format(obj=obj) for obj in objects]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    return {
        "input_ids": enc["input_ids"].to(device),
        "attention_mask": enc["attention_mask"].to(device),
    }


def get_category_centers(model, tokenizer, device, categories, layer_idx, W_U):
    """获取每个类别在某层的hidden state中心"""
    layers = get_layers(model)
    cat_centers = {}
    cat_hidden = {}
    
    for cat_name, objects in categories.items():
        train_objs = objects[:6]
        inputs = encode_prompts(tokenizer, device, train_objs)
        
        captured = {}
        def make_hook(key):
            def hook(module, inp, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook
        
        hook = layers[layer_idx].register_forward_hook(make_hook("h"))
        with torch.no_grad():
            model(**inputs)
        hook.remove()
        
        h = captured["h"]  # (batch, seq, d)
        # 取最后一个非padding token
        mask = inputs["attention_mask"].bool()
        last_idx = mask.sum(dim=1) - 1
        h_last = h[torch.arange(h.size(0)), last_idx.cpu()]  # (batch, d)
        center = h_last.mean(dim=0)
        
        cat_centers[cat_name] = center.numpy()
        cat_hidden[cat_name] = h_last.numpy()
    
    return cat_centers, cat_hidden


def compute_dcf_from_center(h_center, W_U, cat_name, cat_names):
    """计算DCF: 目标类别D值和所有类别D值"""
    logit = W_U @ h_center
    target_idx = cat_names.index(cat_name)
    target_D = logit[target_idx]
    all_D = {cn: logit[cat_names.index(cn)] for cn in cat_names}
    return target_D, all_D


def ablate_direction(h, direction):
    """消融h在direction方向的投影"""
    d_norm = direction / (np.linalg.norm(direction) + 1e-10)
    proj = np.dot(h, d_norm) * d_norm
    return h - proj


def inject_direction(h, direction, scale=1.0):
    """注入direction到h"""
    d_norm = direction / (np.linalg.norm(direction) + 1e-10)
    return h + scale * d_norm


def compute_Bc_and_orth(h_target, h_others, n_dirs=5):
    """计算B_c方向和正交子空间"""
    # B_c: 目标vs其他类别的对比方向
    others_mean = np.mean(h_others, axis=0)
    B_c = h_target - others_mean
    
    # proj_bc: B_c在h上的投影
    Bc_norm = B_c / (np.linalg.norm(B_c) + 1e-10)
    
    # 正交成分: 构造正交于B_c的子空间
    # 用其他类别的对比方向
    contrast_vecs = []
    for h_o in h_others:
        diff = h_o - others_mean
        # 去除B_c分量
        diff_orth = diff - np.dot(diff, Bc_norm) * Bc_norm
        if np.linalg.norm(diff_orth) > 1e-6:
            contrast_vecs.append(diff_orth / np.linalg.norm(diff_orth))
    
    # 加入一些随机正交方向
    for _ in range(max(0, n_dirs - len(contrast_vecs))):
        rand = np.random.randn(len(B_c))
        rand_orth = rand - np.dot(rand, Bc_norm) * Bc_norm
        if np.linalg.norm(rand_orth) > 1e-6:
            contrast_vecs.append(rand_orth / np.linalg.norm(rand_orth))
    
    # SVD获取正交子空间
    if len(contrast_vecs) > 0:
        M = np.stack(contrast_vecs[:n_dirs])
        U, S, Vt = np.linalg.svd(M, full_matrices=False)
        orth_dirs = Vt[:n_dirs]
    else:
        orth_dirs = np.eye(len(B_c))[:n_dirs]
    
    return Bc_norm, orth_dirs


def decompose_orth_at_layer(h_target, h_others, Bc_norm, W_U, cat_names, target_cat, n_dirs=5):
    """分解orth_bc为: shared_semantic, competitor, format, residual"""
    # 1. shared_semantic: 目标与其他所有类别的共享方向
    all_mean = np.mean(np.vstack([h_target[np.newaxis], h_others]), axis=0)
    shared_dir = all_mean / (np.linalg.norm(all_mean) + 1e-10)
    # 去除B_c分量
    shared_dir = shared_dir - np.dot(shared_dir, Bc_norm) * Bc_norm
    if np.linalg.norm(shared_dir) > 1e-6:
        shared_dir = shared_dir / np.linalg.norm(shared_dir)
    shared_dirs = [shared_dir]
    
    # 2. competitor: 每个竞争类别的对比方向(正交于B_c)
    competitor_dirs = []
    others_mean = np.mean(h_others, axis=0)
    for i, h_o in enumerate(h_others):
        comp_dir = h_o - others_mean
        # 去除B_c分量
        comp_dir = comp_dir - np.dot(comp_dir, Bc_norm) * Bc_norm
        # 去除shared分量
        comp_dir = comp_dir - np.dot(comp_dir, shared_dir) * shared_dir
        if np.linalg.norm(comp_dir) > 1e-6:
            competitor_dirs.append(comp_dir / np.linalg.norm(comp_dir))
    
    return shared_dirs, competitor_dirs


# ==================== Exp1: 全层扫描 ====================
def exp1_layer_sweep(model, tokenizer, device, model_name, W_U, round_num):
    """对shared_semantic, proj_bc, competitor做全层扫描"""
    plog(f"=== Exp1: Layer Sweep ===")
    n_layers = get_model_info(model, model_name).n_layers
    
    # 选择扫描层: 每3层采样 + 末3层全扫描
    if n_layers <= 16:
        scan_layers = list(range(n_layers))
    else:
        scan_layers = list(range(0, n_layers, 3)) + list(range(max(0, n_layers-3), n_layers))
        scan_layers = sorted(set(scan_layers))
    
    # 选择类别(每模型2个)
    if model_name == "qwen3":
        test_cats = ["fruit", "clothing"]
    elif model_name == "glm4":
        test_cats = ["fruit", "clothing"]
    else:  # deepseek7b
        test_cats = ["fruit", "food"]
    
    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        cat_results = {}
        target_idx = CAT_NAMES.index(cat_name)
        
        for li in scan_layers:
            t0 = time.time()
            # 获取类别中心
            test_cats_dict = {cn: CATEGORIES[cn] for cn in [cat_name] + [c for c in CAT_NAMES if c != cat_name][:7]}
            cat_centers, cat_hidden = get_category_centers(
                model, tokenizer, device, test_cats_dict, li, W_U
            )
            
            h_target = cat_centers[cat_name]
            other_cats = [c for c in test_cats_dict if c != cat_name]
            h_others = np.array([cat_centers[c] for c in other_cats])
            
            # 计算B_c
            others_mean = np.mean(h_others, axis=0)
            B_c = h_target - others_mean
            Bc_norm = B_c / (np.linalg.norm(B_c) + 1e-10)
            
            # 分解orth_bc
            shared_dirs, competitor_dirs = decompose_orth_at_layer(
                h_target, h_others, Bc_norm, W_U, CAT_NAMES, cat_name
            )
            
            # 测试4种操作: ablate_shared, ablate_proj, ablate_competitor, ablate_orth_bc
            layer_res = {}
            
            # Baseline DCF
            baseline_target_D, _ = compute_dcf_from_center(h_target, W_U, cat_name, CAT_NAMES)
            
            # 1. ablate_shared
            h_mod = h_target.copy()
            for sd in shared_dirs:
                h_mod = ablate_direction(h_mod, sd)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            layer_res["ablate_shared"] = {
                "target_delta": float(target_D_after - baseline_target_D),
                "baseline": float(baseline_target_D),
                "n_shared": len(shared_dirs),
            }
            
            # 2. ablate_proj (消融B_c投影)
            h_mod = ablate_direction(h_target, Bc_norm)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            layer_res["ablate_proj"] = {
                "target_delta": float(target_D_after - baseline_target_D),
                "baseline": float(baseline_target_D),
            }
            
            # 3. ablate_competitor (消融竞争类别方向)
            h_mod = h_target.copy()
            for cd in competitor_dirs[:5]:  # 最多5个
                h_mod = ablate_direction(h_mod, cd)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            layer_res["ablate_competitor"] = {
                "target_delta": float(target_D_after - baseline_target_D),
                "baseline": float(baseline_target_D),
                "n_comp": min(len(competitor_dirs), 5),
            }
            
            # 4. ablate_orth_bc (消融所有正交方向)
            h_mod = h_target.copy()
            # 移除B_c方向的全部正交成分
            proj_bc = np.dot(h_target, Bc_norm) * Bc_norm
            h_mod = proj_bc  # 只保留B_c投影
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            layer_res["ablate_orth_bc"] = {
                "target_delta": float(target_D_after - baseline_target_D),
                "baseline": float(baseline_target_D),
            }
            
            elapsed = time.time() - t0
            layer_res["elapsed"] = elapsed
            
            cat_results[f"L{li}"] = layer_res
            
            if li % 6 == 0 or li >= n_layers - 3:
                plog(f"    L{li}: shared={layer_res['ablate_shared']['target_delta']:+.3f}, "
                     f"proj={layer_res['ablate_proj']['target_delta']:+.3f}, "
                     f"comp={layer_res['ablate_competitor']['target_delta']:+.3f}, "
                     f"orth={layer_res['ablate_orth_bc']['target_delta']:+.3f}")
        
        results[cat_name] = cat_results
    
    return results, scan_layers


# ==================== Exp2: 末层orth_bc分解 ====================
def exp2_late_orth_decomposition(model, tokenizer, device, model_name, W_U, round_num):
    """分解末层orth_bc为读出支撑/抑制/格式等子成分"""
    plog(f"=== Exp2: Late Layer orth_bc Decomposition ===")
    n_layers = get_model_info(model, model_name).n_layers
    
    if model_name == "qwen3":
        test_cats = ["fruit", "clothing"]
    elif model_name == "glm4":
        test_cats = ["fruit", "clothing"]
    else:
        test_cats = ["fruit", "food"]
    
    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        target_idx = CAT_NAMES.index(cat_name)
        
        # 在末层和末层-1做分解
        for li in [n_layers - 1, n_layers - 2]:
            cat_centers, cat_hidden = get_category_centers(
                model, tokenizer, device,
                {cn: CATEGORIES[cn] for cn in [cat_name] + [c for c in CAT_NAMES if c != cat_name][:7]},
                li, W_U
            )
            
            h_target = cat_centers[cat_name]
            other_cats = [c for c in cat_centers if c != cat_name]
            h_others = np.array([cat_centers[c] for c in other_cats])
            
            # B_c方向
            others_mean = np.mean(h_others, axis=0)
            B_c = h_target - others_mean
            Bc_norm = B_c / (np.linalg.norm(B_c) + 1e-10)
            
            # 正交分解
            proj_on_bc = np.dot(h_target, Bc_norm) * Bc_norm
            orth_component = h_target - proj_on_bc  # 整个正交部分
            
            # 分解orth_component:
            # 1. 读出支撑: orth中对目标DCF有正贡献的子方向
            # 2. 抑制方向: orth中对目标DCF有负贡献的子方向
            # 3. 格式/残差: orth中DCF贡献接近零的方向
            
            # 用SVD分解orth_component
            orth_norm = np.linalg.norm(orth_component)
            if orth_norm > 1e-6:
                orth_dir = orth_component / orth_norm
            else:
                orth_dir = np.zeros_like(orth_component)
            
            # 构造orth子空间 (用varimax方法简化)
            # 对8个类别构造对比矩阵, 只取正交于B_c的部分
            contrast_matrix = []
            for cn, h_c in cat_centers.items():
                diff = h_c - others_mean
                diff_orth = diff - np.dot(diff, Bc_norm) * Bc_norm
                if np.linalg.norm(diff_orth) > 1e-6:
                    contrast_matrix.append(diff_orth)
            
            if len(contrast_matrix) > 0:
                M = np.stack(contrast_matrix)
                U, S, Vt = np.linalg.svd(M, full_matrices=False)
                orth_subspace = Vt[:8]  # 取前8个主成分
            else:
                orth_subspace = np.eye(len(B_c))[:8]
            
            # 对每个orth子方向测试DCF贡献
            baseline_target_D, _ = compute_dcf_from_center(h_target, W_U, cat_name, CAT_NAMES)
            
            dir_contributions = []
            for i, d in enumerate(orth_subspace):
                # 消融这个方向后的DCF变化
                h_mod = ablate_direction(h_target, d)
                target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
                delta = target_D_after - baseline_target_D
                
                # 与B_c的对齐度
                alignment = float(np.dot(d, Bc_norm))
                
                dir_contributions.append({
                    "dir_idx": i,
                    "ablate_delta": float(delta),
                    "alignment_with_bc": float(alignment),
                    "singular_value": float(S[i]) if i < len(S) else 0,
                })
            
            # 按ablate_delta排序: 消融后边界下降=支撑方向, 上升=抑制方向
            dir_contributions.sort(key=lambda x: x["ablate_delta"])
            
            # 分为3组
            support_dirs = [d for d in dir_contributions if d["ablate_delta"] < -0.05]
            inhibit_dirs = [d for d in dir_contributions if d["ablate_delta"] > 0.05]
            neutral_dirs = [d for d in dir_contributions if abs(d["ablate_delta"]) <= 0.05]
            
            # 对每组做消融测试
            group_results = {}
            for group_name, group_dirs in [("support", support_dirs), ("inhibit", inhibit_dirs), ("neutral", neutral_dirs)]:
                if len(group_dirs) == 0:
                    group_results[group_name] = {"n_dirs": 0, "target_delta": 0.0}
                    continue
                # 消融整组方向
                h_mod = h_target.copy()
                for d_info in group_dirs:
                    d = orth_subspace[d_info["dir_idx"]]
                    h_mod = ablate_direction(h_mod, d)
                target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
                group_results[group_name] = {
                    "n_dirs": len(group_dirs),
                    "target_delta": float(target_D_after - baseline_target_D),
                    "mean_ablate_delta": float(np.mean([d["ablate_delta"] for d in group_dirs])),
                }
            
            layer_key = f"L{li}"
            results.setdefault(cat_name, {})[layer_key] = {
                "n_orth_dirs": len(orth_subspace),
                "dir_contributions": dir_contributions[:10],  # 只记录前10
                "group_ablate": group_results,
                "orth_norm_ratio": float(orth_norm / (np.linalg.norm(h_target) + 1e-10)),
            }
            
            plog(f"    L{li}: support={group_results.get('support',{})}, "
                 f"inhibit={group_results.get('inhibit',{})}, "
                 f"neutral={group_results.get('neutral',{})}")
    
    return results


# ==================== Exp3: 竞争类别边界因果 ====================
def exp3_competitor_causal(model, tokenizer, device, model_name, W_U, round_num):
    """测试竞争类别边界的因果效应, 特别关注DS7B food的强competitor效应"""
    plog(f"=== Exp3: Competitor Boundary Causal Test ===")
    n_layers = get_model_info(model, model_name).n_layers
    
    if model_name == "qwen3":
        test_cats = ["fruit", "clothing"]
        test_layers = [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    elif model_name == "glm4":
        test_cats = ["fruit", "clothing"]
        test_layers = [n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    else:
        test_cats = ["fruit", "food"]
        test_layers = [7, 14, 21, 27]
    
    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        cat_results = {}
        target_idx = CAT_NAMES.index(cat_name)
        
        # 选择3个最相关竞争类别
        if cat_name == "fruit":
            competitors = ["food", "plant", "animal"]
        elif cat_name == "food":
            competitors = ["fruit", "animal", "plant"]
        elif cat_name == "clothing":
            competitors = ["furniture", "tool", "animal"]
        else:
            competitors = [c for c in CAT_NAMES if c != cat_name][:3]
        
        for li in test_layers:
            t0 = time.time()
            test_cats_dict = {cn: CATEGORIES[cn] for cn in [cat_name] + competitors + 
                             [c for c in CAT_NAMES if c not in [cat_name] + competitors][:4]}
            cat_centers, cat_hidden = get_category_centers(
                model, tokenizer, device, test_cats_dict, li, W_U
            )
            
            h_target = cat_centers[cat_name]
            other_cats = [c for c in cat_centers if c != cat_name]
            h_others = np.array([cat_centers[c] for c in other_cats])
            
            # B_c方向
            others_mean = np.mean(h_others, axis=0)
            B_c = h_target - others_mean
            Bc_norm = B_c / (np.linalg.norm(B_c) + 1e-10)
            
            baseline_target_D, _ = compute_dcf_from_center(h_target, W_U, cat_name, CAT_NAMES)
            
            layer_res = {}
            
            # 对每个竞争类别, 构造其与整体的对比方向
            for comp_cat in competitors:
                if comp_cat not in cat_centers:
                    continue
                h_comp = cat_centers[comp_cat]
                
                # 竞争类别方向(正交于B_c)
                comp_dir = h_comp - others_mean
                comp_dir_orth = comp_dir - np.dot(comp_dir, Bc_norm) * Bc_norm
                if np.linalg.norm(comp_dir_orth) > 1e-6:
                    comp_dir_orth = comp_dir_orth / np.linalg.norm(comp_dir_orth)
                
                # 消融竞争类别方向
                h_mod = ablate_direction(h_target, comp_dir_orth)
                target_D_after, all_D_after = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
                
                # DCF变化
                dcf_delta = {}
                for cn in CAT_NAMES:
                    if cn in all_D_after:
                        baseline_cn_D, _ = compute_dcf_from_center(h_target, W_U, cn, CAT_NAMES)
                        dcf_delta[cn] = float(all_D_after[cn] - baseline_cn_D)
                
                layer_res[comp_cat] = {
                    "target_delta": float(target_D_after - baseline_target_D),
                    "alignment_with_bc": float(np.dot(comp_dir_orth, Bc_norm)),
                    "dcf_delta": dcf_delta,
                }
            
            # 同时消融所有竞争方向
            h_mod = h_target.copy()
            for comp_cat in competitors:
                if comp_cat not in cat_centers:
                    continue
                h_comp = cat_centers[comp_cat]
                comp_dir = h_comp - others_mean
                comp_dir_orth = comp_dir - np.dot(comp_dir, Bc_norm) * Bc_norm
                if np.linalg.norm(comp_dir_orth) > 1e-6:
                    h_mod = ablate_direction(h_mod, comp_dir_orth / np.linalg.norm(comp_dir_orth))
            
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            layer_res["all_competitors"] = {
                "target_delta": float(target_D_after - baseline_target_D),
            }
            
            elapsed = time.time() - t0
            layer_res["elapsed"] = elapsed
            
            cat_results[f"L{li}"] = layer_res
            plog(f"    L{li}: " + ", ".join(
                f"{k}={v['target_delta']:+.3f}" for k, v in layer_res.items() if k != "elapsed"
            ))
        
        results[cat_name] = cat_results
    
    return results


# ==================== Exp4: 早层vs中晚层shared_semantic对比 ====================
def exp4_early_vs_late_shared(model, tokenizer, device, model_name, W_U, round_num):
    """对比早层和中晚层shared_semantic的因果效应"""
    plog(f"=== Exp4: Early vs Late shared_semantic ===")
    n_layers = get_model_info(model, model_name).n_layers
    
    # 选择层位
    early_layers = [1, 2, 3] if n_layers <= 16 else [n_layers//8, n_layers//6, n_layers//4]
    mid_layers = [n_layers//2, 2*n_layers//3]
    late_layers = [3*n_layers//4, n_layers-2, n_layers-1]
    
    if model_name == "qwen3":
        test_cats = ["fruit", "clothing"]
    elif model_name == "glm4":
        test_cats = ["fruit", "clothing"]
    else:
        test_cats = ["fruit", "food"]
    
    results = {}
    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        target_idx = CAT_NAMES.index(cat_name)
        
        all_layers = early_layers + mid_layers + late_layers
        cat_results = {}
        
        for li in all_layers:
            if li >= n_layers:
                continue
            t0 = time.time()
            test_cats_dict = {cn: CATEGORIES[cn] for cn in [cat_name] + [c for c in CAT_NAMES if c != cat_name][:7]}
            cat_centers, cat_hidden = get_category_centers(
                model, tokenizer, device, test_cats_dict, li, W_U
            )
            
            h_target = cat_centers[cat_name]
            other_cats = [c for c in cat_centers if c != cat_name]
            h_others = np.array([cat_centers[c] for c in other_cats])
            
            others_mean = np.mean(h_others, axis=0)
            B_c = h_target - others_mean
            Bc_norm = B_c / (np.linalg.norm(B_c) + 1e-10)
            
            # shared_semantic方向
            shared_dirs, competitor_dirs = decompose_orth_at_layer(
                h_target, h_others, Bc_norm, W_U, CAT_NAMES, cat_name
            )
            
            baseline_target_D, _ = compute_dcf_from_center(h_target, W_U, cat_name, CAT_NAMES)
            
            layer_res = {}
            
            # ablate_shared
            h_mod = h_target.copy()
            for sd in shared_dirs:
                h_mod = ablate_direction(h_mod, sd)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            layer_res["ablate_shared"] = float(target_D_after - baseline_target_D)
            
            # inject_shared (注入shared方向)
            h_mod = h_target.copy()
            for sd in shared_dirs:
                d_norm = sd / (np.linalg.norm(sd) + 1e-10)
                h_mod = h_mod + 0.5 * d_norm
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            layer_res["inject_shared_s0.5"] = float(target_D_after - baseline_target_D)
            
            # double_shared (加倍shared)
            h_mod = h_target.copy()
            for sd in shared_dirs:
                d_norm = sd / (np.linalg.norm(sd) + 1e-10)
                proj_coeff = np.dot(h_target, d_norm)
                h_mod = h_mod + proj_coeff * d_norm  # 等于加倍
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            layer_res["double_shared"] = float(target_D_after - baseline_target_D)
            
            # reverse_shared (反向shared)
            h_mod = h_target.copy()
            for sd in shared_dirs:
                d_norm = sd / (np.linalg.norm(sd) + 1e-10)
                proj_coeff = np.dot(h_target, d_norm)
                h_mod = h_mod - 2 * proj_coeff * d_norm  # 反转
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            layer_res["reverse_shared"] = float(target_D_after - baseline_target_D)
            
            # ablate_proj
            h_mod = ablate_direction(h_target, Bc_norm)
            target_D_after, _ = compute_dcf_from_center(h_mod, W_U, cat_name, CAT_NAMES)
            layer_res["ablate_proj"] = float(target_D_after - baseline_target_D)
            
            # 分类: early/mid/late
            if li in early_layers:
                phase = "early"
            elif li in mid_layers:
                phase = "mid"
            else:
                phase = "late"
            
            cat_results[f"L{li}"] = {
                "phase": phase,
                "results": layer_res,
                "elapsed": time.time() - t0,
            }
            
            plog(f"    L{li}({phase}): shared_abl={layer_res['ablate_shared']:+.3f}, "
                 f"proj_abl={layer_res['ablate_proj']:+.3f}")
        
        results[cat_name] = cat_results
    
    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    
    plog(f"Phase 490: {model_name} R{round_num}")
    
    # 加载模型
    model, tokenizer, device = get_model_and_tokenizer(model_name)
    info = get_model_info(model, model_name)
    W_U_raw = get_W_U(model, model_name)
    W_U = W_U_raw.numpy() if hasattr(W_U_raw, 'numpy') else W_U_raw
    n_layers = info.n_layers
    d_model = info.d_model
    
    plog(f"Model: {info.model_class}, L={n_layers}, d={d_model}")
    
    all_results = {
        "phase": 490,
        "round": round_num,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {
            "class": info.model_class,
            "n_layers": n_layers,
            "d_model": d_model,
        },
    }
    
    try:
        # Exp1: 全层扫描
        plog("=" * 60)
        plog("Starting Exp1: Layer Sweep...")
        exp1_res, scan_layers = exp1_layer_sweep(model, tokenizer, device, model_name, W_U, round_num)
        all_results["exp1_layer_sweep"] = exp1_res
        all_results["scan_layers"] = scan_layers
        plog(f"Exp1 done. {len(scan_layers)} layers scanned.")
        
        # 清理GPU
        gc.collect()
        torch.cuda.empty_cache()
        
        # Exp2: 末层orth_bc分解
        plog("=" * 60)
        plog("Starting Exp2: Late orth_bc Decomposition...")
        exp2_res = exp2_late_orth_decomposition(model, tokenizer, device, model_name, W_U, round_num)
        all_results["exp2_late_orth_decomposition"] = exp2_res
        plog("Exp2 done.")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Exp3: 竞争类别边界因果
        plog("=" * 60)
        plog("Starting Exp3: Competitor Boundary Causal...")
        exp3_res = exp3_competitor_causal(model, tokenizer, device, model_name, W_U, round_num)
        all_results["exp3_competitor_causal"] = exp3_res
        plog("Exp3 done.")
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Exp4: 早层vs中晚层shared_semantic
        plog("=" * 60)
        plog("Starting Exp4: Early vs Late shared_semantic...")
        exp4_res = exp4_early_vs_late_shared(model, tokenizer, device, model_name, W_U, round_num)
        all_results["exp4_early_vs_late_shared"] = exp4_res
        plog("Exp4 done.")
    
    finally:
        # 保存结果
        out_dir = "results/glm5"
        os.makedirs(out_dir, exist_ok=True)
        out_file = os.path.join(out_dir, f"phase490_{model_name}_r{round_num}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        plog(f"Results saved to {out_file}")
        
        # 释放模型
        release_model(model)
        gc.collect()
        torch.cuda.empty_cache()
        plog("Model released.")


if __name__ == "__main__":
    main()
