"""Phase 484 Round 2: 确认关键发现
1. Qwen3 clothing边界高度集中 — cos_remove=0.964(k=5)
2. Fruit/animal MLP神经元不是主要写入器 — cos_remove为负
3. 关系模板不影响B_c注入效果 — B_c relation-invariant
4. 异常竞争对的属性解释
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS)

def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# 导入Phase 484主脚本的数据定义和工具
from phase484_writer_reconstruction import (
    CATEGORIES, CATEGORIES_TRAIN, BEST_NEIGHBORS, BEST_LAYERS,
    FAMILY_WORDS_8D, DCF_DIM_NAMES, RELATION_TEMPLATES,
    load_model_bf16, find_token_id, compute_dcf, logit_lens_dcf,
    _make_capture_hook, get_prompt_ids, make_inject_hook, make_remove_hook,
    qr_orthogonalize, compute_selectivity, safe_load_weight,
    get_specific_direction, get_shared_manifold,
    get_category_residuals_at_layer,
)


def confirm_test(model_name):
    """Phase 484 R2确认测试"""
    plog(f"=== Phase 484 R2: {model_name} ===")
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)
    cat_names = list(CATEGORIES.keys())
    
    results = {}
    
    # ===== 确认1: clothing边界集中 — 用更多测试对象验证 =====
    plog("\n--- Confirm 1: Clothing boundary concentration ---")
    for cat_name in ["clothing", "fruit"]:
        best_layer = BEST_LAYERS[model_name][cat_name]
        spec_vec, spec_norm = get_specific_direction(
            model, tokenizer, device, model_name, cat_name, best_layer
        )
        if spec_vec is None or spec_norm < 1e-6:
            plog(f"  Skip {cat_name}")
            continue
        
        b_hat = spec_vec / spec_norm
        target_idx = cat_names.index(cat_name)
        W_down = safe_load_weight(model, model_name, best_layer, "mlp.down_proj")
        if W_down is None:
            continue
        
        y = W_down.T @ b_hat  # [intermediate]
        
        # 用8个测试对象(比R1的4个多)
        template = RELATION_TEMPLATES["kind_of"]
        test_objs = CATEGORIES[cat_name]  # 全部8个
        
        # 获取MLP激活
        cat_acts = []
        for obj in test_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap_mlp = {}
            done = [False]
            def make_hook(key):
                def hook_fn(module, inp, output):
                    if not done[0]:
                        if isinstance(inp, tuple) and len(inp) > 0:
                            t = inp[0]
                            if not t.is_meta:
                                cap_mlp[key] = t.detach().float().cpu()
                        done[0] = True
                return hook_fn
            mlp = layers_list[best_layer].mlp
            h = mlp.down_proj.register_forward_hook(make_hook("mlp_act"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "mlp_act" in cap_mlp:
                cat_acts.append(cap_mlp["mlp_act"][0, pos].numpy())
        
        mean_act = np.mean(cat_acts, axis=0) if cat_acts else None
        if mean_act is None:
            continue
        neuron_contrib = mean_act * y
        abs_contrib = np.abs(neuron_contrib)
        sorted_idx = np.argsort(abs_contrib)[::-1]
        
        # baseline DCF (8 objects)
        baseline_dcfs = []
        for obj in test_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap:
                baseline_dcfs.append(logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer))
        
        mean_baseline = np.mean(baseline_dcfs, axis=0) if baseline_dcfs else np.zeros(8)
        
        # direction remove baseline (8 objects)
        remove_dcfs = []
        for obj in test_objs:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            h = layers_list[best_layer].register_forward_hook(make_remove_hook(spec_vec, pos, scale=1.0))
            cap = {}
            h2 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            h2.remove()
            if "resid" in cap:
                remove_dcfs.append(logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer))
        
        mean_remove = np.mean(remove_dcfs, axis=0) if remove_dcfs else mean_baseline
        remove_delta = mean_remove - mean_baseline
        
        # neuron ablation k=5 and k=10
        ablation_results = {}
        for k in [5, 10]:
            top_k_idx = sorted_idx[:k]
            ablate_dcfs = []
            for obj in test_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                
                cap_resid = {}
                cap_mlp_act = {}
                done_resid = [False]
                done_mlp = [False]
                def make_rh(key):
                    def hook_fn(module, inp, output):
                        if not done_resid[0]:
                            if isinstance(output, tuple):
                                cap_resid[key] = output[0].detach().float().cpu()
                            else:
                                cap_resid[key] = output.detach().float().cpu()
                            done_resid[0] = True
                    return hook_fn
                def make_mh(key, position):
                    def hook_fn(module, inp, output):
                        if not done_mlp[0]:
                            if isinstance(inp, tuple) and len(inp) > 0:
                                t = inp[0]
                                if not t.is_meta:
                                    cap_mlp_act[key] = t.detach().float().cpu()
                            done_mlp[0] = True
                    return hook_fn
                
                mlp = layers_list[best_layer].mlp
                h1 = layers_list[best_layer].register_forward_hook(make_rh("resid"))
                h2 = mlp.down_proj.register_forward_hook(make_mh("mlp_act", pos))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h1.remove()
                h2.remove()
                
                if "resid" not in cap_resid or "mlp_act" not in cap_mlp_act:
                    continue
                
                baseline_resid = cap_resid["resid"][0, pos].numpy()
                mlp_act = cap_mlp_act["mlp_act"][0, pos].numpy()
                
                # 计算消融后残差
                ablated_contribution = np.zeros(info.d_model)
                for j in top_k_idx:
                    ablated_contribution += mlp_act[j] * W_down[:, j]
                ablated_resid = baseline_resid - ablated_contribution
                
                dcf = logit_lens_dcf(ablated_resid, W_U, tokenizer)
                ablate_dcfs.append(dcf)
            
            mean_ablate = np.mean(ablate_dcfs, axis=0) if ablate_dcfs else mean_baseline
            ablate_delta = mean_ablate - mean_baseline
            cos_remove = float(np.dot(ablate_delta, remove_delta) / 
                               (np.linalg.norm(ablate_delta) * np.linalg.norm(remove_delta) + 1e-10))
            
            ablation_results[f"k={k}"] = {
                "target_delta": float(ablate_delta[target_idx]),
                "cos_with_direction_remove": cos_remove,
                "dcf_delta": ablate_delta.tolist(),
                "direction_remove_delta": remove_delta.tolist(),
            }
            plog(f"  {cat_name}/k={k}: target_D={ablate_delta[target_idx]:.2f}, cos_remove={cos_remove:.3f}")
        
        results[cat_name] = {
            "best_layer": best_layer,
            "n_test_objects": len(test_objs),
            "ablation_results": ablation_results,
        }
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ===== 确认2: B_c关系不变性 — 用更细致的测量 =====
    plog("\n--- Confirm 2: B_c relation invariance ---")
    cat_name = "fruit"
    best_layer = BEST_LAYERS[model_name][cat_name]
    spec_vec, spec_norm = get_specific_direction(
        model, tokenizer, device, model_name, cat_name, best_layer
    )
    if spec_vec is not None and spec_norm > 1e-6:
        target_idx = cat_names.index(cat_name)
        test_objs = CATEGORIES_TRAIN[cat_name]
        
        # 对3种关系，分别测量baseline DCF和injection DCF的RAW值
        relation_raw = {}
        for relation in ["kind_of", "used_for", "found_in"]:
            template = RELATION_TEMPLATES[relation]
            
            # baseline
            base_dcfs = []
            for obj in test_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                cap = {}
                h = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                if "resid" in cap:
                    base_dcfs.append(logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer))
            
            # injection
            inj_dcfs = []
            for obj in test_objs:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                ivec = torch.tensor(spec_vec * 1.0, dtype=torch.float32)
                h = layers_list[best_layer].register_forward_hook(make_inject_hook(ivec, pos))
                cap = {}
                h2 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                h2.remove()
                if "resid" in cap:
                    inj_dcfs.append(logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer))
            
            mean_base = np.mean(base_dcfs, axis=0) if base_dcfs else np.zeros(8)
            mean_inj = np.mean(inj_dcfs, axis=0) if inj_dcfs else np.zeros(8)
            delta = mean_inj - mean_base
            
            relation_raw[relation] = {
                "baseline_dcf": mean_base.tolist(),
                "injection_dcf": mean_inj.tolist(),
                "delta_dcf": delta.tolist(),
                "target_delta": float(delta[target_idx]),
            }
            plog(f"  {relation}: baseline_fruit={mean_base[target_idx]:.2f}, "
                  f"inj_fruit={mean_inj[target_idx]:.2f}, delta={delta[target_idx]:.2f}")
        
        results["relation_invariance"] = {
            "category": cat_name,
            "best_layer": best_layer,
            "relation_raw": relation_raw,
        }
    
    # 释放
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    
    # 保存
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj
    
    out_path = f"results/glm5/phase484_{model_name}_r2.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(convert({
            "phase": 484, "round": 2, "model": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "confirm_results": results,
        }), f, indent=2, ensure_ascii=False)
    plog(f"R2 saved to {out_path}")


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    confirm_test(model_name)
