"""Phase 485 R2确认测试 - 聚焦两个关键发现"""
import sys, os, gc, time, json, math
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS)

def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

# ===== 数据定义 =====
CATEGORIES = {
    "fruit":     ["apple", "banana", "orange", "grape", "pear", "peach", "mango", "plum"],
    "animal":    ["dog", "cat", "horse", "lion", "bear", "rabbit", "eagle", "fish"],
    "clothing":  ["shirt", "dress", "hat", "coat", "sock", "glove", "jacket", "scarf"],
    "furniture": ["chair", "table", "desk", "sofa", "bed", "shelf", "cabinet", "stool"],
    "food":      ["bread", "rice", "cheese", "pasta", "soup", "steak", "salad", "cake"],
    "plant":     ["tree", "flower", "grass", "bush", "fern", "cactus", "vine", "shrub"],
    "tool":      ["hammer", "knife", "wrench", "saw", "drill", "axe", "chisel", "pliers"],
    "vehicle":   ["car", "bus", "bicycle", "truck", "train", "boat", "plane", "motorcycle"],
}
CATEGORIES_TRAIN = {
    "fruit":     ["apple", "banana", "orange", "grape"],
    "animal":    ["dog", "cat", "horse", "lion"],
    "clothing":  ["shirt", "dress", "hat", "coat"],
    "furniture": ["chair", "table", "desk", "sofa"],
    "food":      ["bread", "rice", "cheese", "pasta"],
    "plant":     ["tree", "flower", "grass", "bush"],
    "tool":      ["hammer", "knife", "wrench", "saw"],
    "vehicle":   ["car", "bus", "bicycle", "truck"],
}
BEST_NEIGHBORS = {
    "qwen3": {"fruit": ["plant", "food"], "animal": ["food", "clothing"], "tool": ["vehicle", "furniture"],
              "vehicle": ["furniture", "tool"], "clothing": ["furniture", "tool"], "furniture": ["vehicle", "clothing"],
              "food": ["plant", "vehicle"], "plant": ["food", "animal"]},
    "glm4": {"fruit": ["plant", "food"], "animal": ["food", "clothing"], "tool": ["furniture", "vehicle"],
             "vehicle": ["tool", "furniture"], "clothing": ["furniture", "plant"], "furniture": ["vehicle", "clothing"],
             "food": ["plant", "fruit"], "plant": ["vehicle", "clothing"]},
    "deepseek7b": {"fruit": ["plant", "food"], "animal": ["food", "clothing"], "tool": ["vehicle", "furniture"],
                   "vehicle": ["furniture", "tool"], "clothing": ["furniture", "plant"], "furniture": ["tool", "clothing"],
                   "food": ["plant", "fruit"], "plant": ["food", "fruit"]},
}
BEST_LAYERS = {
    "qwen3": {"fruit": 32, "animal": 33, "tool": 23, "vehicle": 29, "clothing": 30, "furniture": 26, "food": 34, "plant": 28},
    "glm4": {"fruit": 27, "animal": 38, "tool": 27, "vehicle": 29, "clothing": 39, "furniture": 34, "food": 38, "plant": 32},
    "deepseek7b": {"fruit": 26, "animal": 27, "tool": 26, "vehicle": 26, "clothing": 23, "furniture": 25, "food": 27, "plant": 25},
}
FAMILY_WORDS_8D = {
    "fruit": ["fruit", "produce", "crop", "berry"], "animal": ["animal", "creature", "beast", "pet"],
    "tool": ["tool", "implement", "device", "instrument"], "vehicle": ["vehicle", "transport", "automobile", "car"],
    "clothing": ["clothing", "attire", "wear", "garment"], "furniture": ["furniture", "furnishing", "fixture", "seat"],
    "food": ["food", "meal", "dish", "snack"], "plant": ["plant", "tree", "vegetation", "flora"],
}
DCF_DIM_NAMES = ["fruit", "animal", "tool", "vehicle", "clothing", "furniture", "food", "plant"]
RELATION_TEMPLATES = {"kind_of": "The {obj} is a kind of", "used_for": "The {obj} is used for", "found_in": "The {obj} is found in"}

def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True, attn_implementation="flash_attention_2")
    except:
        model = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True, attn_implementation="eager")
    model.eval()
    device = next(model.parameters()).device
    plog(f"  {model_name} loaded, device={device}, GPU={torch.cuda.memory_allocated()/1e9:.2f}GB")
    return model, tokenizer, device

def find_token_id(tokenizer, word):
    vocab = tokenizer.get_vocab()
    for c in [word, " " + word, word.lower(), " " + word.lower()]:
        if c in vocab: return vocab[c]
    return None

def compute_dcf(logits, tokenizer, dim_dict, dim_names):
    dcf = []
    for dn in dim_names:
        words = dim_dict.get(dn, [])
        vals = [float(logits[find_token_id(tokenizer,w)]) for w in words if find_token_id(tokenizer,w) is not None and find_token_id(tokenizer,w) < len(logits)]
        dcf.append(float(np.mean(vals)) if vals else 0.0)
    return np.array(dcf)

def logit_lens_dcf(resid, W_U, tokenizer):
    logits = resid @ W_U.T
    return compute_dcf(logits, tokenizer, FAMILY_WORDS_8D, DCF_DIM_NAMES)

def _make_capture_hook(store_dict, key):
    def hook_fn(module, inp, output):
        if isinstance(output, tuple): store_dict[key] = output[0].detach().float().cpu()
        else: store_dict[key] = output.detach().float().cpu()
    return hook_fn

def get_prompt_ids(tokenizer, device, prompt, max_len=128):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    return inputs["input_ids"].to(device), inputs["attention_mask"].to(device), inputs["attention_mask"].sum().item()-1

def make_inject_hook(ivec, position):
    added = [False]
    def hook_fn(module, inp, output):
        if not added[0]:
            out = (output[0].clone() if isinstance(output, tuple) else output.clone())
            out[0, position, :] += ivec.to(out.device).to(out.dtype)
            added[0] = True
            return (out,)+output[1:] if isinstance(output, tuple) else out
        return output
    return hook_fn

def qr_orthogonalize(target_vec, basis_vecs):
    if not basis_vecs: return target_vec.copy()
    Q, _ = np.linalg.qr(np.array(basis_vecs).T)
    return target_vec - Q @ (Q.T @ target_vec)

def get_specific_direction(model, tokenizer, device, model_name, cat_name, target_layer, n_obj=4):
    layers_list = get_layers(model)
    template = RELATION_TEMPLATES["kind_of"]
    raw_dirs = {}
    for cn, objs in CATEGORIES_TRAIN.items():
        resids = []
        for obj in objs[:n_obj]:
            prompt = template.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
            cap = {}
            h = layers_list[target_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
            with torch.no_grad(): model(input_ids=input_ids, attention_mask=attention_mask)
            h.remove()
            if "resid" in cap: resids.append(cap["resid"][0, pos].numpy())
        if resids: raw_dirs[cn] = np.mean(resids, axis=0)
    if cat_name not in raw_dirs: return None, 0.0
    basis = [raw_dirs[n] for n in BEST_NEIGHBORS[model_name][cat_name] if n in raw_dirs]
    sv = qr_orthogonalize(raw_dirs[cat_name], basis) if basis else raw_dirs[cat_name].copy()
    return sv, float(np.linalg.norm(sv))

def compute_selectivity(dcf_delta, target_idx=0):
    t = abs(dcf_delta[target_idx])
    m = max(abs(dcf_delta[i]) for i in range(len(dcf_delta)) if i != target_idx)
    return t / (m + 0.01)

def safe_load_weight(model, model_name, layer_idx, component_path):
    layers_list = get_layers(model)
    layer = layers_list[layer_idx]
    parts = component_path.split('.')
    obj = layer
    for p in parts: obj = getattr(obj, p)
    w = obj.weight
    if not w.is_meta: return w.detach().cpu().float().numpy()
    from safetensors.torch import load_file
    import glob as glob_mod
    model_path = MODEL_CONFIGS[model_name]["path"]
    for sf_file in glob_mod.glob(os.path.join(model_path, '*.safetensors')):
        try:
            st = load_file(sf_file)
            key = f"model.layers.{layer_idx}.{component_path}.weight"
            if key in st: return st[key].float().numpy()
        except: continue
    return None


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    plog(f"Phase 485 R2 Confirm: {model_name}")
    
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    W_U = get_W_U(model, model_name)
    layers_list = get_layers(model)
    cat_names = list(CATEGORIES.keys())
    
    results = {}
    
    # ===== 确认1: Clothing MLP幅度饱和问题 =====
    # R1发现: k=5→k=500, 幅度从15.7%仅增到29.9%，cos始终>0.93
    # 问题: 为什么cos高但幅度饱和? 需要更多测试对象和更多k值
    plog("=== Confirm1: Clothing MLP Amplitude Saturation ===")
    
    cat_name = "clothing"
    best_layer = BEST_LAYERS[model_name][cat_name]
    spec_vec, spec_norm = get_specific_direction(model, tokenizer, device, model_name, cat_name, best_layer)
    if spec_vec is not None and spec_norm > 1e-6:
        b_hat = spec_vec / spec_norm
        target_idx = cat_names.index(cat_name)
        W_down = safe_load_weight(model, model_name, best_layer, "mlp.down_proj")
        
        if W_down is not None:
            y = W_down.T @ b_hat
            template = RELATION_TEMPLATES["kind_of"]
            
            # 获取MLP激活(8个训练对象)
            cat_acts = []
            for obj in CATEGORIES_TRAIN[cat_name]:
                prompt = template.format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                cap_mlp = {}
                done = [False]
                def make_hook(key):
                    def hook_fn(module, inp, output):
                        if not done[0]:
                            if isinstance(inp, tuple) and len(inp) > 0 and not inp[0].is_meta:
                                cap_mlp[key] = inp[0].detach().float().cpu()
                            done[0] = True
                    return hook_fn
                h = layers_list[best_layer].mlp.down_proj.register_forward_hook(make_hook("mlp_act"))
                with torch.no_grad(): model(input_ids=input_ids, attention_mask=attention_mask)
                h.remove()
                if "mlp_act" in cap_mlp: cat_acts.append(cap_mlp["mlp_act"][0, pos].numpy())
            
            if cat_acts:
                mean_act = np.mean(cat_acts, axis=0)
                neuron_contrib = mean_act * y
                abs_contrib = np.abs(neuron_contrib)
                sorted_idx = np.argsort(abs_contrib)[::-1]
                
                # 用8个测试对象(更多数据)
                test_objs = CATEGORIES[cat_name][:8]
                
                # Baseline
                baseline_dcfs = []
                for obj in test_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                    cap = {}
                    h = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
                    with torch.no_grad(): model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()
                    if "resid" in cap: baseline_dcfs.append(logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer))
                mean_baseline = np.mean(baseline_dcfs, axis=0) if baseline_dcfs else np.zeros(8)
                
                # 方向级remove
                remove_dcfs = []
                for obj in test_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                    b_hat_np = spec_vec / (np.linalg.norm(spec_vec) + 1e-10)
                    # 手动remove
                    cap = {}
                    done = [False]
                    def make_remove_hook2(position):
                        def hook_fn(module, inp, output):
                            if not done[0]:
                                out = (output[0].clone() if isinstance(output, tuple) else output.clone())
                                resid_np = out[0, position, :].float().cpu().numpy()
                                proj = np.dot(resid_np, b_hat_np) * b_hat_np
                                out[0, position, :] -= torch.tensor(proj, dtype=out.dtype, device=out.device)
                                done[0] = True
                                cap["resid"] = out[0, position].detach().float().cpu().numpy()
                                return (out,)+output[1:] if isinstance(output, tuple) else out
                            return output
                        return hook_fn
                    h = layers_list[best_layer].register_forward_hook(make_remove_hook2(pos))
                    with torch.no_grad(): model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()
                    if "resid" in cap: remove_dcfs.append(logit_lens_dcf(cap["resid"], W_U, tokenizer))
                
                mean_remove = np.mean(remove_dcfs, axis=0) if remove_dcfs else mean_baseline
                remove_delta = mean_remove - mean_baseline
                dir_target = float(remove_delta[target_idx])
                
                plog(f"  Direction remove: target_D={dir_target:.2f}")
                
                # MLP总输出消融(不是top-k)
                mlp_total_dcfs = []
                for obj in test_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                    cap_resid = {}
                    cap_mlp_out = {}
                    done1 = [False]
                    done2 = [False]
                    def make_resid_h(key):
                        def hook_fn(module, inp, output):
                            if not done1[0]:
                                if isinstance(output, tuple): cap_resid[key] = output[0].detach().float().cpu()
                                else: cap_resid[key] = output.detach().float().cpu()
                                done1[0] = True
                        return hook_fn
                    def make_mlp_h(key):
                        def hook_fn(module, inp, output):
                            if not done2[0]:
                                if isinstance(output, tuple): cap_mlp_out[key] = output[0].detach().float().cpu()
                                else: cap_mlp_out[key] = output.detach().float().cpu()
                                done2[0] = True
                        return hook_fn
                    
                    h1 = layers_list[best_layer].register_forward_hook(make_resid_h("r"))
                    h2 = layers_list[best_layer].mlp.register_forward_hook(make_mlp_h("m"))
                    with torch.no_grad(): model(input_ids=input_ids, attention_mask=attention_mask)
                    h1.remove()
                    h2.remove()
                    
                    if "r" in cap_resid and "m" in cap_mlp_out:
                        resid = cap_resid["r"][0, pos].numpy()
                        mlp_out = cap_mlp_out["m"][0, pos].numpy()
                        ablated = resid - mlp_out
                        dcf = logit_lens_dcf(ablated, W_U, tokenizer)
                        mlp_total_dcfs.append(dcf)
                
                if mlp_total_dcfs:
                    mean_mlp_total = np.mean(mlp_total_dcfs, axis=0)
                    mlp_total_delta = mean_mlp_total - mean_baseline
                    mlp_total_target = float(mlp_total_delta[target_idx])
                    cos_mlp = float(np.dot(mlp_total_delta, remove_delta) / (np.linalg.norm(mlp_total_delta)*np.linalg.norm(remove_delta)+1e-10))
                    amp_mlp = abs(mlp_total_target / (dir_target + 1e-10))
                    plog(f"  Full MLP ablation: target_D={mlp_total_target:.2f}, cos={cos_mlp:.3f}, amp={amp_mlp:.1%}")
                    results["full_mlp_ablation"] = {"target_delta": mlp_total_target, "cos_remove": cos_mlp, "amplitude_ratio": amp_mlp}
                
                # top-k测试(k=1000, 2000, all)
                for k in [1000, 2000, len(sorted_idx)-1]:
                    if k > len(sorted_idx): continue
                    top_k_idx = sorted_idx[:k]
                    ablate_dcfs = []
                    for obj in test_objs:
                        prompt = template.format(obj=obj)
                        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                        cap_resid = {}
                        cap_mlp_act = {}
                        done1 = [False]
                        done2 = [False]
                        def make_rh(key):
                            def hook_fn(m, inp, out):
                                if not done1[0]:
                                    if isinstance(out, tuple): cap_resid[key] = out[0].detach().float().cpu()
                                    else: cap_resid[key] = out.detach().float().cpu()
                                    done1[0] = True
                            return hook_fn
                        def make_mh(key):
                            def hook_fn(m, inp, out):
                                if not done2[0]:
                                    if isinstance(inp, tuple) and len(inp)>0 and not inp[0].is_meta:
                                        cap_mlp_act[key] = inp[0].detach().float().cpu()
                                    done2[0] = True
                            return hook_fn
                        
                        h1 = layers_list[best_layer].register_forward_hook(make_rh("r"))
                        h2 = layers_list[best_layer].mlp.down_proj.register_forward_hook(make_mh("m"))
                        with torch.no_grad(): model(input_ids=input_ids, attention_mask=attention_mask)
                        h1.remove()
                        h2.remove()
                        
                        if "r" in cap_resid and "m" in cap_mlp_act:
                            resid = cap_resid["r"][0, pos].numpy()
                            mlp_act = cap_mlp_act["m"][0, pos].numpy()
                            contrib = np.zeros(info.d_model)
                            for j in top_k_idx: contrib += mlp_act[j] * W_down[:, j]
                            ablated = resid - contrib
                            dcf = logit_lens_dcf(ablated, W_U, tokenizer)
                            ablate_dcfs.append(dcf)
                    
                    if ablate_dcfs:
                        mean_abl = np.mean(ablate_dcfs, axis=0)
                        abl_delta = mean_abl - mean_baseline
                        abl_target = float(abl_delta[target_idx])
                        cos_abl = float(np.dot(abl_delta, remove_delta) / (np.linalg.norm(abl_delta)*np.linalg.norm(remove_delta)+1e-10))
                        amp_abl = abs(abl_target / (dir_target + 1e-10))
                        plog(f"  k={k}: target_D={abl_target:.2f}, cos={cos_abl:.3f}, amp={amp_abl:.1%}")
                        results[f"k={k}"] = {"target_delta": abl_target, "cos_remove": cos_abl, "amplitude_ratio": amp_abl}
    
    # ===== 确认2: B_c关系不变性(小scale) =====
    plog("=== Confirm2: B_c Relation Invariance at Small Scale ===")
    
    for cat_name in ["fruit", "clothing"]:
        best_layer = BEST_LAYERS[model_name][cat_name]
        spec_vec2, spec_norm2 = get_specific_direction(model, tokenizer, device, model_name, cat_name, best_layer)
        if spec_vec2 is None or spec_norm2 < 1e-6: continue
        
        target_idx = cat_names.index(cat_name)
        rel_deltas = {}
        
        for rel_name, template in RELATION_TEMPLATES.items():
            test_objs = CATEGORIES[cat_name][:6]
            for scale in [0.05, 0.1, 0.2]:
                inject_vec = spec_vec2 * scale
                ivec = torch.tensor(inject_vec, dtype=torch.float32)
                
                inject_dcfs = []
                baseline_dcfs2 = []
                for obj in test_objs:
                    prompt = template.format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
                    # baseline
                    cap = {}
                    h = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
                    with torch.no_grad(): model(input_ids=input_ids, attention_mask=attention_mask)
                    h.remove()
                    if "resid" in cap:
                        baseline_dcfs2.append(logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer))
                    
                    # inject
                    cap2 = {}
                    h1 = layers_list[best_layer].register_forward_hook(make_inject_hook(ivec, pos))
                    h2 = layers_list[best_layer].register_forward_hook(_make_capture_hook(cap2, "resid"))
                    with torch.no_grad(): model(input_ids=input_ids, attention_mask=attention_mask)
                    h1.remove()
                    h2.remove()
                    if "resid" in cap2:
                        inject_dcfs.append(logit_lens_dcf(cap2["resid"][0, pos].numpy(), W_U, tokenizer))
                
                if baseline_dcfs2 and inject_dcfs:
                    mean_bl = np.mean(baseline_dcfs2, axis=0)
                    mean_inj = np.mean(inject_dcfs, axis=0)
                    delta = mean_inj - mean_bl
                    target_d = float(delta[target_idx])
                    key = f"{cat_name}_{rel_name}_scale{scale}"
                    rel_deltas[key] = target_d
        
        # 检查跨关系一致性
        for scale in [0.05, 0.1, 0.2]:
            deltas = [v for k, v in rel_deltas.items() if f"scale{scale}" in k]
            if len(deltas) >= 2:
                d_mean = sum(deltas) / len(deltas)
                d_range = max(deltas) - min(deltas)
                plog(f"  {cat_name} scale={scale}: deltas={[f'{d:.2f}' for d in deltas]}, "
                      f"range={d_range:.2f}, rel_range={d_range/(abs(d_mean)+1e-10):.2%}")
    
    # 保存
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase485_{model_name}_r2.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"phase": 485, "round": 2, "model": model_name, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results}, f, indent=2, ensure_ascii=False)
    plog(f"Results saved to {out_path}")
    
    release_model(model)
    plog("Done!")

if __name__ == "__main__":
    main()
