"""
Phase 507: 正交语义场解析、功能验证与场论闭环
================================================
核心目标: 解析 v_c 中 99% 正交成分到底编码什么，验证其功能

实验组成:
  Exp1: 正交成分分解 (PCA/SVD + 聚类)
  Exp2: 中层正交成分功能干预 (remove/keep Φ_perp at mid-layer → forward)
  Exp3: 正交场功能探针 (轻量probe预测object/category/relation等)
  Exp4: 跨层读出能量轨迹 (||Φ||, a_c, ρ_c, D逐层)
  Exp5: action专项 (扩展5子类)
  Exp6: token级输出闭合 (category_argmax, first-token prob)

数据量: 30 objects/category × 7 categories = 210 objects total
        (避免数据不足导致推翻结论)

Usage:
  python tests/glm5/phase507_orthogonal_field.py qwen3
  python tests/glm5/phase507_orthogonal_field.py glm4
  python tests/glm5/phase507_orthogonal_field.py deepseek7b
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc, json, time, os, warnings
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open
from model_utils import (get_model_info, release_model, get_W_U, MODEL_CONFIGS, get_layers)

warnings.filterwarnings('ignore', category=FutureWarning)
OUTPUT_DIR = Path("results/glm5")

# === 数据配置 ===
# 7个类别，每类30个对象，远超之前的20个
CATEGORIES = {
    "fruit": {
        "objects": [
            "apple","banana","orange","grape","pear","peach","mango","plum",
            "cherry","lemon","apricot","kiwi","pineapple","melon","coconut","lime",
            "fig","pomegranate","papaya","avocado","strawberry","blueberry",
            "raspberry","blackberry","tangerine","watermelon","guava","lychee",
            "persimmon","nectarine"
        ],
        "relation": "is a type of fruit",
        "clean_relation": "belongs to the same category as apple",
    },
    "animal": {
        "objects": [
            "dog","cat","horse","elephant","tiger","dolphin","eagle","snake",
            "rabbit","whale","lion","bear","fox","wolf","deer","monkey",
            "shark","frog","penguin","owl","giraffe","zebra","parrot","turtle",
            "flamingo","otter","cheetah","gorilla","hedgehog","pelican"
        ],
        "relation": "is a type of animal",
        "clean_relation": "belongs to the same category as dog",
    },
    "action": {
        "objects": [
            "run","eat","build","throw","buy","learn","measure","communicate",
            "swim","write","sing","draw","fly","climb","teach","drive",
            "cook","dance","fight","sleep","read","listen","paint","explore",
            "analyze","create","discover","observe","practice","investigate"
        ],
        "relation": "is a type of action",
        "clean_relation": "belongs to the same category as run",
        # Exp5: 5子类
        "subtypes": {
            "physical": ["run","swim","climb","fly","throw","dance","fight","sleep"],
            "creation": ["build","write","draw","sing","paint","cook","create"],
            "communication": ["teach","communicate","listen","read","tell"],
            "cognitive": ["learn","analyze","discover","observe","investigate","measure","practice"],
            "transaction": ["buy","sell","trade","hire","rent"],
        },
    },
    "emotion": {
        "objects": [
            "joy","anger","fear","sadness","surprise","disgust","pride","shame",
            "guilt","envy","hope","love","hate","boredom","anxiety","jealousy",
            "gratitude","regret","curiosity","embarrassment","compassion","delight",
            "frustration","nostalgia","relief","admiration","contempt","doubt",
            "excitement","loneliness"
        ],
        "relation": "is a type of emotion",
        "clean_relation": "belongs to the same category as joy",
    },
    "clothing": {
        "objects": [
            "shirt","dress","jacket","pants","coat","skirt","sweater","blouse",
            "scarf","vest","hat","glove","sock","boot","belt","tie",
            "jeans","shorts","hoodie","raincoat","sweater","cardigan","tunic",
            "legging","poncho","parka","blazer","sandal","sneaker","cap"
        ],
        "relation": "is a type of clothing",
        "clean_relation": "belongs to the same category as shirt",
    },
    "color": {
        "objects": [
            "red","blue","green","yellow","purple","orange","pink","brown",
            "black","white","gray","violet","indigo","crimson","scarlet",
            "teal","cyan","magenta","maroon","navy","turquoise","lavender",
            "coral","salmon","ivory","beige","olive","amber","chartreuse","cerulean"
        ],
        "relation": "is a type of color",
        "clean_relation": "belongs to the same category as red",
    },
    "vehicle": {
        "objects": [
            "car","bus","train","bicycle","motorcycle","airplane","helicopter",
            "boat","ship","truck","van","taxi","subway","tram","scooter",
            "canoe","yacht","ferry","hovercraft","jeep","ambulance","tractor",
            "bulldozer","crane","spaceship","rocket","glider","segway","gondola","cab"
        ],
        "relation": "is a type of vehicle",
        "clean_relation": "belongs to the same category as car",
    },
}
ALL_CLASS = list(CATEGORIES.keys())
NEUTRAL_RELATION = "is a thing"
N_OBJECTS = 30  # 每类使用30个对象
TRAIN_N = 20
TEST_N = 10


def load_bf16_auto(name):
    """BF16 + device_map=auto + sdpa 加载"""
    cfg = MODEL_CONFIGS[name]
    print(f"[load] Loading {name} (bf16 + auto + sdpa)...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="sdpa")
    m.eval()
    dev = next(m.parameters()).device
    gmem = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
    print(f"[load] {name}: mem={gmem:.1f}GB class={type(m).__name__} ({time.time()-t0:.0f}s)")
    return m, tok, dev


def get_norm_g(model, name):
    """获取最终RMSNorm/LayerNorm的gain权重"""
    for attr in ['model.norm','model.final_layernorm','model.decoder.final_layer_norm']:
        obj = model
        for p in attr.split('.'):
            obj = getattr(obj, p, None)
            if obj is None: break
        if obj and hasattr(obj, 'weight'):
            w = obj.weight.detach()
            if str(w.device) != 'meta': return w.float().cpu().numpy()
    cfg = MODEL_CONFIGS[name]
    for sf in sorted(Path(cfg["path"]).glob("*.safetensors")):
        with safe_open(str(sf), framework="pt") as f:
            for key in f.keys():
                if 'norm' in key.lower() and 'weight' in key.lower() and not any(x in key for x in ['layer','input','post']):
                    return f.get_tensor(key).float().cpu().numpy()
    return None


def get_token_ids(tokenizer, words):
    ids = []
    for w in words:
        tid = tokenizer.encode(w, add_special_tokens=False)
        if tid: ids.append(tid[0])
    return ids


def forward_with_all_hidden(model, tokenizer, prompt, device):
    """前向推理，返回所有层隐藏状态"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    hs = outputs.hidden_states
    n_layers = len(hs)
    result = {}
    for l in range(n_layers):
        seq_len = hs[l].shape[1]
        ans = hs[l][0, -1, :].float().cpu().numpy().astype(np.float64)
        pre = hs[l][0, -2, :].float().cpu().numpy().astype(np.float64) if seq_len >= 2 else ans.copy()
        result[l] = {"ans": ans, "pre": pre}
    return result


def forward_with_intervention(model, tokenizer, prompt, device,
                               intervention_layer, intervention_fn):
    """
    在指定层进行干预后继续forward
    
    Args:
        intervention_layer: 干预的层索引(0-based transformer layer)
        intervention_fn: callable(hidden_state) -> modified_hidden_state
    
    Returns:
        final_logits, final_hidden_states (last layer)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs.input_ids
    
    # 使用hook进行干预
    captured = {}
    layers = get_layers(model)
    
    def make_capture_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().clone()
            else:
                captured[key] = output.detach().clone()
        return hook
    
    def make_intervene_hook(fn):
        def hook(module, input, output):
            if isinstance(output, tuple):
                modified = fn(output[0])
                return (modified,) + output[1:]
            else:
                return fn(output)
        return hook
    
    hooks = []
    # 在干预层添加干预hook
    hooks.append(layers[intervention_layer].register_forward_hook(
        make_intervene_hook(intervention_fn)))
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, return_dict=True)
    
    for h in hooks:
        h.remove()
    
    # 获取最终层的隐藏状态和logits
    last_hidden = outputs.hidden_states[-1][0, -1, :].float().cpu().numpy().astype(np.float64)
    logits = outputs.logits[0, -1, :].float().cpu().numpy().astype(np.float64)
    
    return logits, last_hidden


def rms_norm(vec):
    return float(np.sqrt(np.mean(vec**2)))


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return float(1 - ss_res/(ss_tot+1e-10))


def ridge_map(X, Y, ridge=0.1):
    X, Y = X.astype(np.float64), Y.astype(np.float64)
    XtX = X.T @ X
    return np.linalg.solve(XtX + ridge * np.eye(XtX.shape[0]), X.T @ Y).astype(np.float64)


# ============================================================
# Exp1: 正交成分分解 (PCA + 聚类)
# ============================================================
def exp1_orthogonal_decomposition(all_hidden, cat_meta, L, d, model_name):
    """
    对每层的Φ_perp做PCA和聚类分析
    
    Returns:
        dict: {cat: {layer: {pca_var_explained, n_components_90, clustering_metrics}}}
    """
    print(f"\n{'='*60}")
    print("Exp1: 正交成分分解 (PCA + Clustering)")
    print(f"{'='*60}")
    
    results = {}
    
    for cat_name in CATEGORIES:
        print(f"\n--- {cat_name} ---")
        qc = cat_meta[cat_name]["qc"]
        qcn = np.linalg.norm(qc)
        if qcn < 1e-10:
            continue
        q_hat = qc / qcn  # 单位读出方向
        
        cat_data = all_hidden[cat_name]
        cat_result = {}
        
        # 采样关键层
        sample_layers = sorted(set([0, 1] + list(range(0, L+1, max(L//8, 1))) + [L-3, L-1, L]))
        sample_layers = [min(l, L) for l in sample_layers]
        sample_layers = sorted(set(sample_layers))
        
        for l in sample_layers:
            # 收集所有对象的 Φ_perp
            phi_perp_list = []
            phi_para_list = []
            phi_full_list = []
            
            for oi in range(len(cat_data["rich"])):
                if l not in cat_data["rich"][oi] or l not in cat_data["neutral"][oi]:
                    continue
                h_rich = cat_data["rich"][oi][l]["ans"]
                h_neutral = cat_data["neutral"][oi][l]["ans"]
                phi = h_rich - h_neutral  # Φ_c^l
                
                # 分解: Φ = a·q_hat + Φ_perp
                a_c = np.dot(phi, q_hat)
                phi_para = a_c * q_hat
                phi_perp = phi - phi_para
                
                phi_full_list.append(phi)
                phi_para_list.append(a_c)  # 标量
                phi_perp_list.append(phi_perp)
            
            if len(phi_perp_list) < 3:
                continue
            
            # 堆叠为矩阵
            Phi_perp = np.array(phi_perp_list)  # [n_samples, d_model]
            Phi_full = np.array(phi_full_list)
            a_c_arr = np.array(phi_para_list)
            
            # --- PCA on Φ_perp ---
            # 中心化
            Phi_perp_centered = Phi_perp - Phi_perp.mean(axis=0)
            
            # SVD
            n_comp = min(50, Phi_perp_centered.shape[0]-1, Phi_perp_centered.shape[1])
            try:
                U, S, Vt = np.linalg.svd(Phi_perp_centered, full_matrices=False)
                # 取前n_comp个
                U = U[:, :n_comp]
                S = S[:n_comp]
                Vt = Vt[:n_comp, :]
                
                total_var = np.sum(S**2)
                var_explained = S**2 / (total_var + 1e-10)
                cum_var = np.cumsum(var_explained)
                n_90 = int(np.searchsorted(cum_var, 0.9)) + 1 if len(cum_var) > 0 else 0
                n_95 = int(np.searchsorted(cum_var, 0.95)) + 1 if len(cum_var) > 0 else 0
                n_99 = int(np.searchsorted(cum_var, 0.99)) + 1 if len(cum_var) > 0 else 0
                
                # Top-10 主成分的方差占比
                top10_var = float(np.sum(var_explained[:10])) if len(var_explained) >= 10 else float(np.sum(var_explained))
                top3_var = float(np.sum(var_explained[:3])) if len(var_explained) >= 3 else float(np.sum(var_explained))
                
            except Exception as e:
                print(f"  L{l}: PCA failed: {e}")
                continue
            
            # --- 对Φ_full的cos(Φ, q_hat)统计 ---
            cos_phi_qc = [float(np.dot(phi, q_hat) / (np.linalg.norm(phi) * qcn + 1e-10)) 
                          for phi in phi_full_list]
            norm_phi = [float(np.linalg.norm(phi)) for phi in phi_full_list]
            rho_phi_qc = [c**2 for c in cos_phi_qc]  # cos² = 读出能量占比
            
            # --- Φ_perp的范数 vs Φ_para的范数 ---
            phi_perp_norms = [float(np.linalg.norm(pp)) for pp in phi_perp_list]
            phi_para_norms = [abs(a) * qcn for a in phi_para_list]
            
            cat_result[l] = {
                "n_samples": len(phi_perp_list),
                # PCA结果
                "pca_top1_var": round(float(var_explained[0]), 6) if len(var_explained) > 0 else 0,
                "pca_top3_var": round(top3_var, 6),
                "pca_top10_var": round(top10_var, 6),
                "pca_n_90": n_90,
                "pca_n_95": n_95,
                "pca_n_99": n_99,
                # Φ统计
                "phi_norm_mean": round(float(np.mean(norm_phi)), 2),
                "phi_norm_std": round(float(np.std(norm_phi)), 2),
                "cos_phi_qc_mean": round(float(np.mean(cos_phi_qc)), 6),
                "cos_phi_qc_std": round(float(np.std(cos_phi_qc)), 6),
                "rho_phi_qc_mean": round(float(np.mean(rho_phi_qc)), 8),
                # 分解统计
                "phi_para_norm_mean": round(float(np.mean(phi_para_norms)), 2),
                "phi_perp_norm_mean": round(float(np.mean(phi_perp_norms)), 2),
                "perp_para_ratio_mean": round(float(np.mean([p/(q+1e-10) for p,q in zip(phi_perp_norms, phi_para_norms)])), 2),
                # a_c统计
                "a_c_mean": round(float(np.mean(a_c_arr)), 4),
                "a_c_std": round(float(np.std(a_c_arr)), 4),
            }
            
            if l in [0, L//2, L-1, L]:
                print(f"  L{l:>3}: top1={var_explained[0]:.4f} top10={top10_var:.4f} "
                      f"n90={n_90} cos(Φ,qc)={np.mean(cos_phi_qc):.4f} "
                      f"|Φ_perp|={np.mean(phi_perp_norms):.1f} |Φ_para|={np.mean(phi_para_norms):.1f} "
                      f"ratio={np.mean(phi_perp_norms)/(np.mean(phi_para_norms)+1e-10):.1f}")
        
        results[cat_name] = cat_result
    
    return results


# ============================================================
# Exp2: 中层正交成分功能干预
# ============================================================
def exp2_midlayer_intervention(model, tokenizer, cat_meta, W_U, L, d, device, model_name):
    """
    在中间层移除/保留Φ_perp，观察对最终读出的影响
    
    干预条件:
    1. remove_perp: 移除正交成分 h' = h - Φ_perp
    2. keep_para_only: 只保留平行成分 h' = h_neutral + Φ_para  
    3. add_perp_noise: 用匹配范数随机向量替代Φ_perp
    4. baseline: 不干预
    
    检查每层 l ∈ {L//4, L//2, 3L//4, L-3}
    """
    print(f"\n{'='*60}")
    print("Exp2: 中层正交成分功能干预")
    print(f"{'='*60}")
    
    intervention_layers = sorted(set([L//4, L//2, 2*L//3, 3*L//4, L-5, L-3]))
    intervention_layers = [l for l in intervention_layers if 0 < l < L]
    print(f"  Intervention layers: {intervention_layers}")
    
    results = {}
    
    for cat_name, cfg in CATEGORIES.items():
        print(f"\n--- {cat_name} ---")
        qc = cat_meta[cat_name]["qc"]
        qcn = np.linalg.norm(qc)
        q_hat = qc / (qcn + 1e-10)
        target_ids = cat_meta[cat_name]["target_ids"]
        competitor_ids = cat_meta[cat_name]["competitor_ids"]
        
        cat_result = {}
        test_objs = cfg["objects"][TRAIN_N:TRAIN_N+TEST_N]
        
        for int_layer in intervention_layers:
            layer_results = {"baseline": [], "remove_perp": [], 
                           "keep_para_only": [], "add_perp_noise": []}
            
            for obj in test_objs:
                rich_prompt = f"The {obj} {cfg['relation']}"
                neutral_prompt = f"The {obj} {NEUTRAL_RELATION}"
                
                # 获取baseline隐藏状态
                rich_states = forward_with_all_hidden(model, tokenizer, rich_prompt, device)
                neutral_states = forward_with_all_hidden(model, tokenizer, neutral_prompt, device)
                
                if int_layer >= len(rich_states):
                    continue
                
                h_rich = rich_states[int_layer]["ans"]
                h_neutral = neutral_states[int_layer]["ans"]
                phi = h_rich - h_neutral
                a_c = np.dot(phi, q_hat)
                phi_para = a_c * q_hat
                phi_perp = phi - phi_para
                
                # Baseline: 最终层的DCF
                h_final_rich = rich_states[L]["ans"]
                h_final_neutral = neutral_states[L]["ans"]
                
                logits_rich = h_final_rich @ W_U.T
                logits_neutral = h_final_neutral @ W_U.T
                
                t_rich = np.mean([logits_rich[i] for i in target_ids if i < len(logits_rich)])
                c_rich = np.mean([logits_rich[i] for i in competitor_ids if i < len(logits_rich)])
                D_baseline = t_rich - c_rich
                a_c_baseline = np.dot(h_final_rich, q_hat) / rms_norm(h_final_rich)
                
                layer_results["baseline"].append({
                    "D": D_baseline, "a_c": a_c_baseline,
                    "T": t_rich, "C": c_rich
                })
                
                # --- 干预1: remove_perp ---
                # 在干预层移除Φ_perp: h' = h_rich - phi_perp = h_neutral + phi_para
                delta_perp = torch.tensor(-phi_perp, dtype=torch.bfloat16, device=device)
                
                def make_intervene_fn(delta):
                    _delta = delta  # 显式捕获
                    def intervene(hidden_state):
                        return hidden_state + _delta.unsqueeze(0).unsqueeze(0)
                    return intervene
                
                try:
                    logits_rm, h_final_rm = forward_with_intervention(
                        model, tokenizer, rich_prompt, device,
                        int_layer, make_intervene_fn(delta_perp))
                    
                    t_rm = np.mean([logits_rm[i] for i in target_ids if i < len(logits_rm)])
                    c_rm = np.mean([logits_rm[i] for i in competitor_ids if i < len(logits_rm)])
                    D_rm = t_rm - c_rm
                    a_c_rm = np.dot(h_final_rm, q_hat) / rms_norm(h_final_rm)
                    
                    layer_results["remove_perp"].append({
                        "D": D_rm, "a_c": a_c_rm, "T": t_rm, "C": c_rm
                    })
                except Exception as e:
                    print(f"    remove_perp failed: {e}")
                
                # --- 干预2: keep_para_only ---
                # 替换为 h_neutral + phi_para (只有平行分量)
                phi_para_t = torch.tensor(phi_para, dtype=torch.bfloat16, device=device)
                h_rich_t = torch.tensor(h_rich, dtype=torch.bfloat16, device=device)
                # 干预量 = h_neutral + phi_para - h_rich = -phi_perp (同remove_perp)
                # 实际不同: remove_perp是从rich减perp, keep_para是从neutral加para
                # 但数学上等价: h_rich - phi_perp = h_neutral + phi_para
                # 所以只需要一个干预, 但验证一下
                
                # --- 干预3: add_perp_noise ---
                # 用匹配范数的随机向量替代phi_perp
                # h' = h_rich - phi_perp + noise = h_rich + (noise - phi_perp)
                perp_norm = np.linalg.norm(phi_perp)
                rng = np.random.RandomState(42 + hash(obj) % 1000)
                noise = rng.randn(d)
                noise = noise / np.linalg.norm(noise) * perp_norm  # 匹配范数
                delta_replace = noise - phi_perp  # 替换差异
                delta_t = torch.tensor(delta_replace, dtype=torch.bfloat16, device=device)
                
                def make_add_delta_fn(delta):
                    _delta = delta  # 显式捕获
                    def intervene(hidden_state):
                        return hidden_state + _delta.unsqueeze(0).unsqueeze(0)
                    return intervene
                
                try:
                    logits_noise, h_final_noise = forward_with_intervention(
                        model, tokenizer, rich_prompt, device,
                        int_layer, make_add_delta_fn(delta_t))
                    
                    t_noise = np.mean([logits_noise[i] for i in target_ids if i < len(logits_noise)])
                    c_noise = np.mean([logits_noise[i] for i in competitor_ids if i < len(logits_noise)])
                    D_noise = t_noise - c_noise
                    a_c_noise = np.dot(h_final_noise, q_hat) / rms_norm(h_final_noise)
                    
                    layer_results["add_perp_noise"].append({
                        "D": D_noise, "a_c": a_c_noise, "T": t_noise, "C": c_noise
                    })
                except Exception as e:
                    print(f"    add_perp_noise failed: {e}")
                
                # 清理GPU
                torch.cuda.empty_cache()
            
            # 汇总统计
            if layer_results["baseline"]:
                baseline_D = np.mean([r["D"] for r in layer_results["baseline"]])
                baseline_a = np.mean([r["a_c"] for r in layer_results["baseline"]])
            else:
                baseline_D, baseline_a = 0, 0
            
            summary = {"baseline_D": round(baseline_D, 4), "baseline_a_c": round(baseline_a, 6)}
            
            for cond in ["remove_perp", "add_perp_noise"]:
                if layer_results[cond]:
                    cond_D = np.mean([r["D"] for r in layer_results[cond]])
                    cond_a = np.mean([r["a_c"] for r in layer_results[cond]])
                    cond_T = np.mean([r["T"] for r in layer_results[cond]])
                    cond_C = np.mean([r["C"] for r in layer_results[cond]])
                    delta_D = cond_D - baseline_D
                    
                    summary[f"{cond}_D"] = round(cond_D, 4)
                    summary[f"{cond}_a_c"] = round(cond_a, 6)
                    summary[f"{cond}_T"] = round(cond_T, 4)
                    summary[f"{cond}_C"] = round(cond_C, 4)
                    summary[f"{cond}_delta_D"] = round(delta_D, 4)
                    summary[f"{cond}_n"] = len(layer_results[cond])
                else:
                    summary[f"{cond}_n"] = 0
            
            cat_result[int_layer] = summary
            
            # 打印
            rm_D = summary.get("remove_perp_D", "N/A")
            rm_dD = summary.get("remove_perp_delta_D", "N/A")
            noise_D = summary.get("add_perp_noise_D", "N/A")
            noise_dD = summary.get("add_perp_noise_delta_D", "N/A")
            print(f"  L{int_layer:>3}: base_D={baseline_D:+.3f} "
                  f"rm_perp_D={rm_D} ΔD={rm_dD} "
                  f"noise_D={noise_D} ΔD={noise_dD}")
        
        results[cat_name] = cat_result
    
    return results


# ============================================================
# Exp3: 正交场功能探针
# ============================================================
def exp3_functional_probes(all_hidden, cat_meta, L, d):
    """
    训练轻量probe从Φ_para和Φ_perp预测上下文变量
    
    预测目标:
    - object identity (one-hot → accuracy)
    - category (which of 7 categories → accuracy)
    - target-driven vs competitor-suppression (T>C vs C>T → accuracy)
    """
    print(f"\n{'='*60}")
    print("Exp3: 正交场功能探针")
    print(f"{'='*60}")
    
    results = {}
    probe_layers = sorted(set([0, L//4, L//2, 3*L//4, L-3, L-1]))
    probe_layers = [min(l, L) for l in probe_layers]
    
    for l in probe_layers:
        # 收集所有类别的数据
        X_para_list, X_perp_list, X_full_list = [], [], []
        y_cat_list, y_obj_list, y_tc_list = [], [], []
        
        for cat_name, cfg in CATEGORIES.items():
            qc = cat_meta[cat_name]["qc"]
            qcn = np.linalg.norm(qc)
            if qcn < 1e-10:
                continue
            q_hat = qc / qcn
            cat_idx = ALL_CLASS.index(cat_name)
            
            cat_data = all_hidden[cat_name]
            target_ids = cat_meta[cat_name]["target_ids"]
            competitor_ids = cat_meta[cat_name]["competitor_ids"]
            
            for oi in range(len(cat_data["rich"])):
                if l not in cat_data["rich"][oi] or l not in cat_data["neutral"][oi]:
                    continue
                
                h_rich = cat_data["rich"][oi][l]["ans"]
                h_neutral = cat_data["neutral"][oi][l]["ans"]
                phi = h_rich - h_neutral
                
                a_c = np.dot(phi, q_hat)
                phi_para_vec = a_c * q_hat
                phi_perp_vec = phi - phi_para_vec
                
                X_para_list.append(phi_para_vec)
                X_perp_list.append(phi_perp_vec)
                X_full_list.append(phi)
                
                y_cat_list.append(cat_idx)
                y_obj_list.append(oi)
                
                # T/C模式: target-driven (T>C) or competitor-suppression (C>T)
                # 用rich hidden state的logit lens
                # 这里简化: 用a_c的符号
                y_tc_list.append(1 if a_c > 0 else 0)
        
        if len(X_para_list) < 10:
            continue
        
        X_para = np.array(X_para_list)
        X_perp = np.array(X_perp_list)
        X_full = np.array(X_full_list)
        y_cat = np.array(y_cat_list)
        y_tc = np.array(y_tc_list)
        
        # --- Category probe ---
        # 用ridge分类 (one-vs-rest)
        n_classes = len(ALL_CLASS)
        n_total = len(y_cat)
        n_train = int(0.7 * n_total)
        
        idx = np.random.RandomState(42).permutation(n_total)
        train_idx, test_idx = idx[:n_train], idx[n_train:]
        
        cat_acc = {}
        for name, X in [("para", X_para), ("perp", X_perp), ("full", X_full)]:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_cat[train_idx], y_cat[test_idx]
            
            # One-vs-rest ridge
            preds = np.zeros((len(X_test), n_classes))
            for c in range(n_classes):
                y_bin = (y_train == c).astype(float)
                W = ridge_map(X_train, y_bin, ridge=1.0)
                preds[:, c] = X_test @ W
            
            pred_labels = np.argmax(preds, axis=1)
            acc = float(np.mean(pred_labels == y_test))
            cat_acc[name] = round(acc, 4)
        
        # --- T/C mode probe ---
        tc_acc = {}
        for name, X in [("para", X_para), ("perp", X_perp), ("full", X_full)]:
            X_train, X_test = X[train_idx], X[test_idx]
            y_train_tc, y_test_tc = y_tc[train_idx], y_tc[test_idx]
            
            W = ridge_map(X_train, y_train_tc, ridge=1.0)
            preds_tc = X_test @ W
            pred_labels_tc = (preds_tc > 0.5).astype(int)
            acc = float(np.mean(pred_labels_tc == y_test_tc))
            tc_acc[name] = round(acc, 4)
        
        # --- Φ_perp 范数与category的关联 ---
        perp_norms = np.linalg.norm(X_perp, axis=1)
        perp_norm_by_cat = {}
        for c_idx, c_name in enumerate(ALL_CLASS):
            mask = y_cat == c_idx
            if mask.sum() > 0:
                perp_norm_by_cat[c_name] = round(float(np.mean(perp_norms[mask])), 2)
        
        results[l] = {
            "n_total": n_total,
            "category_accuracy": cat_acc,
            "tc_mode_accuracy": tc_acc,
            "perp_norm_by_category": perp_norm_by_cat,
        }
        
        print(f"  L{l:>3}: cat_acc para={cat_acc['para']:.3f} perp={cat_acc['perp']:.3f} "
              f"full={cat_acc['full']:.3f} | tc_acc para={tc_acc['para']:.3f} "
              f"perp={tc_acc['perp']:.3f} full={tc_acc['full']:.3f}")
    
    return results


# ============================================================
# Exp4: 跨层读出能量轨迹
# ============================================================
def exp4_energy_trajectory(all_hidden, cat_meta, W_U, L, d):
    """
    逐层计算完整能量指标
    
    ||Φ_c^l||, a_c^l, ρ_c^l, D_l, T_l, C_l
    """
    print(f"\n{'='*60}")
    print("Exp4: 跨层读出能量轨迹")
    print(f"{'='*60}")
    
    results = {}
    
    for cat_name in CATEGORIES:
        qc = cat_meta[cat_name]["qc"]
        qcn = np.linalg.norm(qc)
        if qcn < 1e-10:
            continue
        q_hat = qc / qcn
        target_ids = cat_meta[cat_name]["target_ids"]
        competitor_ids = cat_meta[cat_name]["competitor_ids"]
        
        cat_data = all_hidden[cat_name]
        cat_traj = {}
        
        for l in range(L + 1):
            phi_norms, a_c_vals, rho_vals = [], [], []
            D_rich_vals, T_rich_vals, C_rich_vals = [], [], []
            D_neutral_vals = []
            
            for oi in range(len(cat_data["rich"])):
                if l not in cat_data["rich"][oi] or l not in cat_data["neutral"][oi]:
                    continue
                
                h_rich = cat_data["rich"][oi][l]["ans"]
                h_neutral = cat_data["neutral"][oi][l]["ans"]
                phi = h_rich - h_neutral
                
                phi_norm = np.linalg.norm(phi)
                a_c = np.dot(phi, q_hat)
                cos_phi_qc = a_c / (phi_norm + 1e-10)
                rho = cos_phi_qc ** 2
                
                phi_norms.append(phi_norm)
                a_c_vals.append(a_c)
                rho_vals.append(rho)
                
                # DCF via logit lens
                logits_rich = h_rich @ W_U.T
                logits_neutral = h_neutral @ W_U.T
                
                t_rich = np.mean([logits_rich[i] for i in target_ids if i < len(logits_rich)])
                c_rich = np.mean([logits_rich[i] for i in competitor_ids if i < len(logits_rich)])
                t_neutral = np.mean([logits_neutral[i] for i in target_ids if i < len(logits_neutral)])
                c_neutral = np.mean([logits_neutral[i] for i in competitor_ids if i < len(logits_neutral)])
                
                D_rich_vals.append(t_rich - c_rich)
                T_rich_vals.append(t_rich)
                C_rich_vals.append(c_rich)
                D_neutral_vals.append(t_neutral - c_neutral)
            
            if not phi_norms:
                continue
            
            cat_traj[l] = {
                "phi_norm_mean": round(float(np.mean(phi_norms)), 4),
                "phi_norm_std": round(float(np.std(phi_norms)), 4),
                "a_c_mean": round(float(np.mean(a_c_vals)), 6),
                "a_c_std": round(float(np.std(a_c_vals)), 6),
                "rho_mean": round(float(np.mean(rho_vals)), 8),
                "rho_std": round(float(np.std(rho_vals)), 8),
                "cos_mean": round(float(np.mean([np.sign(a)*np.sqrt(abs(r)) for a, r in zip(a_c_vals, rho_vals)])), 6),
                "D_rich_mean": round(float(np.mean(D_rich_vals)), 4),
                "T_rich_mean": round(float(np.mean(T_rich_vals)), 4),
                "C_rich_mean": round(float(np.mean(C_rich_vals)), 4),
                "D_neutral_mean": round(float(np.mean(D_neutral_vals)), 4),
                "n_samples": len(phi_norms),
            }
        
        results[cat_name] = cat_traj
        
        # 打印关键层
        key_layers = [0, L//4, L//2, 3*L//4, L-3, L-1]
        key_layers = [min(l, L) for l in key_layers]
        print(f"  {cat_name}:")
        for l in key_layers:
            if l in cat_traj:
                t = cat_traj[l]
                print(f"    L{l:>3}: |Φ|={t['phi_norm_mean']:.1f} a_c={t['a_c_mean']:.4f} "
                      f"ρ={t['rho_mean']:.6f} D_rich={t['D_rich_mean']:+.2f} "
                      f"D_neutral={t['D_neutral_mean']:+.2f}")
    
    return results


# ============================================================
# Exp5: action专项
# ============================================================
def exp5_action_special(model_ref, tokenizer, device_ref, cat_meta, W_U, L, d):
    """
    对action类5个子类型分别分析Φ_perp功能
    """
    print(f"\n{'='*60}")
    print("Exp5: action子类型专项")
    print(f"{'='*60}")
    
    if "action" not in CATEGORIES or "subtypes" not in CATEGORIES["action"]:
        print("  No action subtypes defined, skipping")
        return {}
    
    subtypes = CATEGORIES["action"]["subtypes"]
    qc = cat_meta["action"]["qc"]
    qcn = np.linalg.norm(qc)
    if qcn < 1e-10:
        return {}
    q_hat = qc / qcn
    target_ids = cat_meta["action"]["target_ids"]
    competitor_ids = cat_meta["action"]["competitor_ids"]
    
    results = {}
    
    for subtype_name, subtype_objs in subtypes.items():
        print(f"\n  --- action/{subtype_name} ---")
        
        # 收集子类型对象的隐藏状态
        subtype_rich = []
        subtype_neutral = []
        
        for obj in subtype_objs:
            rich_prompt = f"The {obj} is a type of action"
            neutral_prompt = f"The {obj} is a thing"
            
            rich_states = forward_with_all_hidden(model_ref, tokenizer, rich_prompt, device_ref)
            neutral_states = forward_with_all_hidden(model_ref, tokenizer, neutral_prompt, device_ref)
            
            subtype_rich.append(rich_states)
            subtype_neutral.append(neutral_states)
        
        # 分析末层
        l = L
        phi_norms, a_c_vals, cos_vals, D_vals = [], [], [], []
        
        for oi in range(len(subtype_rich)):
            h_rich = subtype_rich[oi][l]["ans"]
            h_neutral = subtype_neutral[oi][l]["ans"]
            phi = h_rich - h_neutral
            
            phi_norm = np.linalg.norm(phi)
            a_c = np.dot(phi, q_hat)
            cos_val = a_c / (phi_norm + 1e-10)
            
            logits = h_rich @ W_U.T
            t_val = np.mean([logits[i] for i in target_ids if i < len(logits)])
            c_val = np.mean([logits[i] for i in competitor_ids if i < len(logits)])
            
            phi_norms.append(phi_norm)
            a_c_vals.append(a_c)
            cos_vals.append(cos_val)
            D_vals.append(t_val - c_val)
        
        results[subtype_name] = {
            "n_objects": len(subtype_rich),
            "phi_norm_mean": round(float(np.mean(phi_norms)), 2),
            "a_c_mean": round(float(np.mean(a_c_vals)), 4),
            "cos_mean": round(float(np.mean(cos_vals)), 6),
            "D_mean": round(float(np.mean(D_vals)), 4),
            "objects": subtype_objs,
        }
        
        print(f"    |Φ|={np.mean(phi_norms):.1f} a_c={np.mean(a_c_vals):.4f} "
              f"cos={np.mean(cos_vals):.4f} D={np.mean(D_vals):+.3f}")
    
    return results


# ============================================================
# Exp6: token级输出闭合
# ============================================================
def exp6_token_level_output(model, tokenizer, cat_meta, W_U, L, d, device):
    """
    检查Φ_perp是否影响token级输出
    
    指标:
    - category_argmax_rate: 类别词是否是argmax
    - first_token_prob: 类别词的概率
    - top5 tokens
    """
    print(f"\n{'='*60}")
    print("Exp6: token级输出闭合")
    print(f"{'='*60}")
    
    results = {}
    
    for cat_name, cfg in CATEGORIES.items():
        print(f"\n--- {cat_name} ---")
        qc = cat_meta[cat_name]["qc"]
        qcn = np.linalg.norm(qc)
        q_hat = qc / (qcn + 1e-10)
        target_ids = cat_meta[cat_name]["target_ids"]
        competitor_ids = cat_meta[cat_name]["competitor_ids"]
        other_cats = [c for c in ALL_CLASS if c != cat_name]
        
        test_objs = cfg["objects"][TRAIN_N:TRAIN_N+TEST_N]
        
        cat_result = {
            "rich": {"cat_argmax_rate": 0, "cat_prob_mean": 0, "top5_tokens": []},
            "neutral": {"cat_argmax_rate": 0, "cat_prob_mean": 0, "top5_tokens": []},
        }
        
        rich_cat_argmax, rich_cat_prob = [], []
        neutral_cat_argmax, neutral_cat_prob = [], []
        rich_top5_counts = defaultdict(int)
        neutral_top5_counts = defaultdict(int)
        
        for obj in test_objs:
            rich_prompt = f"The {obj} {cfg['relation']}"
            neutral_prompt = f"The {obj} {NEUTRAL_RELATION}"
            
            # Rich prompt
            inputs = tokenizer(rich_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)
            logits = outputs.logits[0, -1, :].float().cpu().numpy()
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            
            # 类别词概率
            cat_probs_rich = [probs[i] for i in target_ids if i < len(probs)]
            rich_cat_prob.append(sum(cat_probs_rich))
            rich_cat_argmax.append(1 if np.argmax(probs) in target_ids else 0)
            
            # top5
            top5_idx = np.argsort(probs)[-5:][::-1]
            for idx in top5_idx:
                tok = tokenizer.decode([idx])
                rich_top5_counts[tok] += 1
            
            # Neutral prompt
            inputs = tokenizer(neutral_prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs, return_dict=True)
            logits = outputs.logits[0, -1, :].float().cpu().numpy()
            probs = np.exp(logits - logits.max())
            probs = probs / probs.sum()
            
            cat_probs_neutral = [probs[i] for i in target_ids if i < len(probs)]
            neutral_cat_prob.append(sum(cat_probs_neutral))
            neutral_cat_argmax.append(1 if np.argmax(probs) in target_ids else 0)
            
            top5_idx = np.argsort(probs)[-5:][::-1]
            for idx in top5_idx:
                tok = tokenizer.decode([idx])
                neutral_top5_counts[tok] += 1
            
            torch.cuda.empty_cache()
        
        cat_result["rich"]["cat_argmax_rate"] = round(float(np.mean(rich_cat_argmax)), 4)
        cat_result["rich"]["cat_prob_mean"] = round(float(np.mean(rich_cat_prob)), 6)
        cat_result["rich"]["top5_tokens"] = sorted(rich_top5_counts.items(), key=lambda x: -x[1])[:5]
        
        cat_result["neutral"]["cat_argmax_rate"] = round(float(np.mean(neutral_cat_argmax)), 4)
        cat_result["neutral"]["cat_prob_mean"] = round(float(np.mean(neutral_cat_prob)), 6)
        cat_result["neutral"]["top5_tokens"] = sorted(neutral_top5_counts.items(), key=lambda x: -x[1])[:5]
        
        results[cat_name] = cat_result
        
        print(f"  rich: argmax_rate={cat_result['rich']['cat_argmax_rate']:.3f} "
              f"cat_prob={cat_result['rich']['cat_prob_mean']:.6f}")
        print(f"  neutral: argmax_rate={cat_result['neutral']['cat_argmax_rate']:.3f} "
              f"cat_prob={cat_result['neutral']['cat_prob_mean']:.6f}")
    
    return results


# ============================================================
# 主函数
# ============================================================
def run_phase507(model, tokenizer, model_name, device):
    """Phase 507 主函数"""
    info = get_model_info(model, model_name)
    L = info.n_layers
    d = info.d_model
    print(f"[info] L={L}, d={d}, class={info.model_class}")
    
    # 加载W_U和gain
    W_U = get_W_U(model, model_name).astype(np.float64)
    g = get_norm_g(model, model_name)
    if g is None:
        print("[ERROR] Cannot get final layer norm gain!")
        return None
    g = g.astype(np.float64)
    print(f"[info] W_U shape={W_U.shape}, g norm={np.linalg.norm(g):.2f}")
    
    # 构建每个类别的q_c
    cat_meta = {}
    for cat_name, cfg in CATEGORIES.items():
        target_ids = get_token_ids(tokenizer, [cat_name])
        other_cats = [c for c in ALL_CLASS if c != cat_name]
        competitor_ids = get_token_ids(tokenizer, other_cats)
        
        wDt = np.mean([W_U[i] for i in target_ids if i < len(W_U)], axis=0) if target_ids else np.zeros(d)
        wDc = np.mean([W_U[i] for i in competitor_ids if i < len(W_U)], axis=0) if competitor_ids else np.zeros(d)
        qc = (wDt - wDc) * g
        
        cat_meta[cat_name] = {
            "target_ids": target_ids,
            "competitor_ids": competitor_ids,
            "qc": qc,
            "qcn": float(np.linalg.norm(qc)),
        }
        print(f"  {cat_name}: target_ids={target_ids}, |qc|={np.linalg.norm(qc):.2f}")
    
    # ============================================================
    # 收集所有隐藏状态 (一次性推理，所有实验共用)
    # ============================================================
    print(f"\n{'='*60}")
    print(f"收集隐藏状态: {sum(len(c['objects']) for c in CATEGORIES.values())} objects × 7 categories")
    print(f"{'='*60}")
    
    all_hidden = {}  # {cat: {"rich": [obj_states], "neutral": [obj_states]}}
    
    for cat_name, cfg in CATEGORIES.items():
        print(f"\n  [{cat_name}] ", end="", flush=True)
        objs = cfg["objects"][:N_OBJECTS]
        rich_list, neutral_list = [], []
        
        for oi, obj in enumerate(objs):
            rich_prompt = f"The {obj} {cfg['relation']}"
            neutral_prompt = f"The {obj} {NEUTRAL_RELATION}"
            
            rich_states = forward_with_all_hidden(model, tokenizer, rich_prompt, device)
            neutral_states = forward_with_all_hidden(model, tokenizer, neutral_prompt, device)
            
            rich_list.append(rich_states)
            neutral_list.append(neutral_states)
            
            if (oi + 1) % 10 == 0:
                print(f"{oi+1}", end=" ", flush=True)
        
        all_hidden[cat_name] = {"rich": rich_list, "neutral": neutral_list}
        gmem = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
        print(f"({len(rich_list)} objs, GPU={gmem:.1f}GB)")
    
    # ============================================================
    # 运行所有实验
    # ============================================================
    
    # Exp1: 正交成分分解
    exp1_results = exp1_orthogonal_decomposition(all_hidden, cat_meta, L, d, model_name)
    
    # Exp3: 功能探针 (不需要干预，用已收集的hidden states)
    exp3_results = exp3_functional_probes(all_hidden, cat_meta, L, d)
    
    # Exp4: 跨层能量轨迹
    exp4_results = exp4_energy_trajectory(all_hidden, cat_meta, W_U, L, d)
    
    # Exp2: 中层干预 (需要model，单独运行)
    exp2_results = exp2_midlayer_intervention(model, tokenizer, cat_meta, W_U, L, d, device, model_name)
    
    # Exp5: action子类型 (需要额外前向推理)
    exp5_results = exp5_action_special(model, tokenizer, device, cat_meta, W_U, L, d)
    
    # Exp6: token级输出 (需要model)
    exp6_results = exp6_token_level_output(model, tokenizer, cat_meta, W_U, L, d, device)
    
    # ============================================================
    # 汇总
    # ============================================================
    summary = {
        "model": model_name,
        "L": L,
        "d_model": d,
        "model_class": info.model_class,
        "n_objects_per_cat": N_OBJECTS,
        "n_categories": len(CATEGORIES),
        "categories": list(CATEGORIES.keys()),
        "exp1_orthogonal_decomposition": exp1_results,
        "exp2_midlayer_intervention": exp2_results,
        "exp3_functional_probes": exp3_results,
        "exp4_energy_trajectory": exp4_results,
        "exp5_action_special": exp5_results,
        "exp6_token_level_output": exp6_results,
    }
    
    return summary


def print_summary(results):
    """打印结果摘要"""
    if results is None:
        print("No results!")
        return
    
    model_name = results["model"]
    L = results["L"]
    
    print(f"\n{'='*70}")
    print(f"Phase 507 Summary — {model_name}")
    print(f"{'='*70}")
    
    # Exp1摘要
    print("\n[Exp1] 正交成分PCA:")
    for cat, layers_data in results.get("exp1_orthogonal_decomposition", {}).items():
        last_layer = L
        if last_layer in layers_data:
            ld = layers_data[last_layer]
            print(f"  {cat}: top1={ld['pca_top1_var']:.4f} top10={ld['pca_top10_var']:.4f} "
                  f"n90={ld['pca_n_90']} |Φ_perp|={ld['phi_perp_norm_mean']:.1f} "
                  f"|Φ_para|={ld['phi_para_norm_mean']:.1f} ratio={ld['perp_para_ratio_mean']:.1f}")
    
    # Exp2摘要
    print("\n[Exp2] 中层干预效果:")
    for cat, layers_data in results.get("exp2_midlayer_intervention", {}).items():
        for l in sorted(layers_data.keys()):
            ld = layers_data[l]
            rm_dD = ld.get("remove_perp_delta_D", "N/A")
            noise_dD = ld.get("add_perp_noise_delta_D", "N/A")
            if rm_dD != "N/A":
                print(f"  {cat} L{l}: ΔD(rm_perp)={rm_dD:+.3f} ΔD(noise)={noise_dD:+.3f}")
    
    # Exp3摘要
    print("\n[Exp3] 功能探针:")
    for l, ld in sorted(results.get("exp3_functional_probes", {}).items()):
        ca = ld.get("category_accuracy", {})
        print(f"  L{l}: cat_acc para={ca.get('para',0):.3f} perp={ca.get('perp',0):.3f} full={ca.get('full',0):.3f}")
    
    # Exp4摘要
    print("\n[Exp4] 能量轨迹 (末层):")
    for cat, traj in results.get("exp4_energy_trajectory", {}).items():
        if L in traj:
            t = traj[L]
            print(f"  {cat}: |Φ|={t['phi_norm_mean']:.1f} a_c={t['a_c_mean']:.4f} "
                  f"ρ={t['rho_mean']:.6f} D={t['D_rich_mean']:+.3f}")
    
    # Exp6摘要
    print("\n[Exp6] Token级输出:")
    for cat, cd in results.get("exp6_token_level_output", {}).items():
        r = cd.get("rich", {})
        n = cd.get("neutral", {})
        print(f"  {cat}: rich_argmax={r.get('cat_argmax_rate',0):.3f} "
              f"neutral_argmax={n.get('cat_argmax_rate',0):.3f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python tests/glm5/phase507_orthogonal_field.py <model_name>")
        sys.exit(1)
    
    mn = sys.argv[1]
    if mn not in MODEL_CONFIGS:
        print(f"Unknown model: {mn}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)
    
    print("=" * 70)
    print(f"Phase 507: Orthogonal Semantic Field Analysis — {mn}")
    print(f"7 categories × 30 objects | Exp1-6 integrated")
    print("=" * 70)
    
    t0 = time.time()
    model, tokenizer, device = load_bf16_auto(mn)
    
    try:
        results = run_phase507(model, tokenizer, mn, device)
        if results is None:
            print("ERROR: No results!")
            return
        
        print_summary(results)
        
        # 保存结果
        out = OUTPUT_DIR / f"phase507_{mn}.json"
        
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, tuple):
                return list(obj)
            return obj
        
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=convert)
        print(f"\nSaved: {out}")
        
        elapsed = time.time() - t0
        print(f"Total time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    
    finally:
        release_model(model)
        print("Model released.")


if __name__ == "__main__":
    main()
