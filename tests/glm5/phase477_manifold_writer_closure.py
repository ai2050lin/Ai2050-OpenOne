"""
Phase 477: 自然流形边界、L30水果写入器完整闭环与格式覆盖头验证
================================================================
核心目标:
1. 确认自然流形边界是否约为 ||δ||/sqrt(d_model)=1
2. 验证 Qwen3 L30 fruit writer 是否只提升 fruit DCF (完整8D)
3. 验证 fruit writer 跨对象、跨模板泛化
4. 扩大消融规模，验证正负神经元协同
5. 确定 fruit writer 的有效注入强度区间
6. 验证 DS7B Head 12 是否是格式覆盖必要组件
7. 验证 DS7B Head 12 是否具有格式覆盖充分性

实验:
1. Exp1: 细粒度beta扫描 (3模型)
2. Exp2: L30 fruit writer完整8D DCF变化 (Qwen3)
3. Exp3: 跨对象+跨模板泛化 (Qwen3)
4. Exp4: 扩大消融规模 (Qwen3)
5. Exp5: fruit writer剂量-响应与流形 (Qwen3)
6. Exp6: DS7B Head 12必要性 (DS7B)
7. Exp7: DS7B Head 12充分性 (DS7B)

用法:
  python tests/glm5/phase477_manifold_writer_closure.py qwen3 1
  python tests/glm5/phase477_manifold_writer_closure.py glm4 1
  python tests/glm5/phase477_manifold_writer_closure.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json, math
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS,
                          get_layer_weights)


def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 数据定义 ====================
CATEGORIES = {
    "fruit":    ["apple", "banana", "orange", "grape", "pear", "peach", "mango", "plum"],
    "animal":   ["dog", "cat", "horse", "lion", "bear", "rabbit"],
    "tool":     ["hammer", "knife", "wrench", "saw", "drill", "axe"],
    "vehicle":  ["car", "bus", "bicycle", "truck", "train", "boat"],
    "clothing": ["shirt", "dress", "hat", "coat", "sock", "glove"],
    "furniture":["chair", "table", "desk", "sofa", "bed", "shelf"],
}

# 完整8D DCF维度词汇
FAMILY_WORDS_8D = {
    "fruit":    ["fruit", "produce", "crop", "berry"],
    "animal":   ["animal", "creature", "beast", "pet"],
    "tool":     ["tool", "implement", "device", "instrument"],
    "vehicle":  ["vehicle", "transport", "automobile", "car"],
    "clothing": ["clothing", "attire", "wear", "garment"],
    "furniture":["furniture", "furnishing", "fixture", "seat"],
    "food":     ["food", "meal", "dish", "snack"],
    "plant":    ["plant", "tree", "vegetation", "flora"],
}

# 8D维度名列表(固定顺序)
DCF_DIM_NAMES = ["fruit", "animal", "tool", "vehicle", "clothing", "furniture", "food", "plant"]

# 多模板
RELATION_TEMPLATES = {
    "kind_of":      "The {obj} is a kind of",
    "belongs_to":   "A {obj} belongs to the category",
    "classified_as":"{obj} is classified as",
    "eaten_as":     "{obj} is usually eaten as",
}

# 训练集/留出集水果对象
FRUIT_TRAIN = ["apple", "banana", "pear", "mango"]
FRUIT_HELDOUT = ["grape", "orange", "peach", "plum"]

FORMAT_TOKENS = [
    "(", ")", "[", "]", "{", "}", "<", ">", ",", ".", ":", ";", "!", "?",
    "-", "=", "+", "*", "/", "\\", "|", "&", "^", "%", "$", "#", "@", "~",
    "`", "'", "\"", "...", "..", "--", "---",
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "to", "for", "with", "on", "at", "by", "from", "as",
    "that", "which", "who", "whom", "this", "these", "those",
    "therefore", "because", "since", "thus", "hence", "so", "consequently",
]

SEMANTIC_TOKENS = [
    "fruit", "apple", "banana", "orange", "grape", "pear", "peach", "produce", "crop", "berry",
    "animal", "dog", "cat", "horse", "lion", "bear", "rabbit", "creature", "beast", "pet",
    "tool", "hammer", "knife", "wrench", "saw", "drill", "axe", "implement", "device", "instrument",
    "vehicle", "car", "bus", "bicycle", "truck", "train", "boat", "transport", "automobile",
    "clothing", "shirt", "dress", "hat", "coat", "sock", "glove", "attire", "wear", "garment",
    "furniture", "chair", "table", "desk", "sofa", "bed", "shelf", "furnishing", "fixture",
    "food", "plant", "tree", "flower", "grass", "leaf", "root", "seed",
]

# 数学/推理触发词
MATH_TOKENS = [
    "step", "calculate", "equation", "solve", "therefore", "let", "assume",
    "given", "since", "thus", "hence", "prove", "show", "define",
    "1.", "2.", "3.", "4.", "5.", "first", "second", "third",
    "reasoning", "logic", "analysis", "conclusion", "argument",
]


# ==================== 模型加载 ====================
def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name} (bfloat16 + device_map=auto + flash)...")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="flash_attention_2",
        )
        plog(f"  Flash Attention 2 enabled")
    except Exception as e:
        plog(f"  Flash Attention 2 failed ({e}), falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="eager",
        )

    model.eval()
    layers_list = get_layers(model)
    n_loaded = len(layers_list)
    plog(f"  Loaded {n_loaded} transformer layers")

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        layer_devices = {}
        for k, v in dmap.items():
            if k.startswith('model.layers.'):
                lid = k.split('.')[2]
                if lid not in layer_devices:
                    layer_devices[lid] = str(v)
        gpu_layers = sum(1 for v in layer_devices.values() if 'cuda' in v)
        cpu_layers = sum(1 for v in layer_devices.values() if 'cpu' in v)
        plog(f"  Layer distribution: {gpu_layers} GPU + {cpu_layers} CPU (total {n_loaded})")
        if cpu_layers > 0:
            gpu_lids = [int(lid) for lid, dev in layer_devices.items() if 'cuda' in dev]
            cpu_lids = [int(lid) for lid, dev in layer_devices.items() if 'cpu' in dev]
            if gpu_lids:
                plog(f"  Last GPU layer: L{max(gpu_lids)}")
            if cpu_lids:
                plog(f"  Last CPU layer: L{max(cpu_lids)}")

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    plog(f"  {model_name}: device={device}, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


# ==================== 基础工具 ====================
def find_token_id(tokenizer, word):
    vocab = tokenizer.get_vocab()
    for candidate in [word, f" {word}", word.lower(), f" {word.lower()}"]:
        if candidate in vocab:
            return vocab[candidate]
    return None


def compute_dcf_from_logits(logits, tokenizer, dim_dict):
    """计算完整8D DCF向量"""
    dcf_vector = []
    for dim_name in DCF_DIM_NAMES:
        words = dim_dict.get(dim_name, [])
        logit_values = []
        for w in words:
            tid = find_token_id(tokenizer, w)
            if tid is not None and tid < len(logits):
                logit_values.append(float(logits[tid]))
        dcf_vector.append(float(np.mean(logit_values)) if logit_values else 0.0)
    return np.array(dcf_vector)  # [8]


def logit_lens_dcf(resid, W_U, tokenizer, dim_dict=None):
    """logit lens获取8D DCF向量"""
    if dim_dict is None:
        dim_dict = FAMILY_WORDS_8D
    logits = resid @ W_U.T
    return compute_dcf_from_logits(logits, tokenizer, dim_dict)


def _make_capture_hook(store_dict, key):
    def hook_fn(module, inp, output):
        if isinstance(output, tuple):
            store_dict[key] = output[0].detach().float().cpu()
        else:
            store_dict[key] = output.detach().float().cpu()
    return hook_fn


def build_dcf_subspace(W_U, tokenizer, dim_dict, d_model):
    """构建DCF敏感子空间的正交基(8维)"""
    basis = []
    for dim_name in DCF_DIM_NAMES:
        words = dim_dict.get(dim_name, [])
        vecs = []
        for w in words:
            tid = find_token_id(tokenizer, w)
            if tid is not None and tid < W_U.shape[0]:
                vecs.append(W_U[tid])
        if vecs:
            basis.append(np.mean(vecs, axis=0))
    basis = np.array(basis)  # [8, d_model]

    # Gram-Schmidt正交化
    ortho = []
    for v in basis:
        w = v.copy()
        for u in ortho:
            proj = np.dot(w, u) / (np.dot(u, u) + 1e-12) * u
            w = w - proj
        norm = np.linalg.norm(w)
        if norm > 1e-10:
            ortho.append(w / norm)

    return np.array(ortho)  # [8, d_model]


def project_dcf_subspace(delta, ortho_basis):
    """投影delta到DCF子空间"""
    dcf_proj = np.zeros_like(delta)
    for basis_vec in ortho_basis:
        proj_len = np.dot(delta, basis_vec)
        dcf_proj += proj_len * basis_vec
    null_proj = delta - dcf_proj
    return dcf_proj, null_proj


def compute_entropy(logits):
    """从logits计算softmax entropy"""
    max_l = np.max(logits)
    exp_l = np.exp(logits - max_l)
    probs = exp_l / np.sum(exp_l)
    probs = probs[probs > 1e-12]
    return -float(np.sum(probs * np.log(probs)))


def get_test_layers(model_name):
    """获取各模型的测试层"""
    if model_name == "qwen3":
        return {
            "mid": 30, "last": 35,
            "sample": [24, 27, 30, 33, 35],
        }
    elif model_name == "glm4":
        return {
            "mid": 30, "last": 37,
            "sample": [24, 27, 30, 33, 35, 37],
        }
    else:  # deepseek7b
        return {
            "mid": 24, "last": 26,
            "sample": [24, 25, 26],
        }


def get_prompt_ids(tokenizer, device, prompt, max_len=128):
    """获取编码后的input_ids和attention_mask"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    seq_len = attention_mask.sum().item()
    pos = seq_len - 1
    return input_ids, attention_mask, pos


# ==================== Exp1: 细粒度Beta扫描 ====================
def exp1_fine_beta_scan(model, tokenizer, device, model_name):
    """
    细粒度beta扫描, 精确定位自然流形边界
    记录: DCF恢复、完整delta、DCF分量、null分量、entropy、norm ratio
    """
    plog(f"=== Exp1: Fine-Grained Beta Scan ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)
    test_layers = get_test_layers(model_name)
    mid_layer = test_layers["mid"]
    last_layer = test_layers["last"]

    betas = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
    n_obj = 3

    # 构建DCF子空间
    ortho_basis = build_dcf_subspace(W_U, tokenizer, FAMILY_WORDS_8D, info.d_model)
    plog(f"  DCF subspace: {ortho_basis.shape[0]} dims")

    # 计算类别方向(用mid层残差)
    fruit_resids = []
    for obj in CATEGORIES["fruit"][:n_obj]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap = {}
        h = layers_list[mid_layer].register_forward_hook(_make_capture_hook(cap, "resid"))
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h.remove()
        if "resid" in cap:
            fruit_resids.append(cap["resid"][0, pos].numpy())

    if not fruit_resids:
        return {"error": "No fruit residuals captured"}

    fruit_mean = np.mean(fruit_resids, axis=0)
    perturb_dir = -fruit_mean / (np.linalg.norm(fruit_mean) + 1e-12)

    results = {}

    for beta in betas:
        plog(f"  Beta={beta}")
        target_norm = beta * math.sqrt(info.d_model)

        recoveries_mid = []
        recoveries_last = []
        entropies_clean = []
        entropies_pert = []
        dcf_fracs = []
        norm_ratios = []

        for obj in CATEGORIES["fruit"][:n_obj]:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

            # Clean baseline: 一次前向捕获mid+last
            clean_data = {}
            hooks_c = []
            hooks_c.append(layers_list[mid_layer].register_forward_hook(_make_capture_hook(clean_data, "mid")))
            hooks_c.append(layers_list[last_layer].register_forward_hook(_make_capture_hook(clean_data, "last")))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            for h in hooks_c:
                h.remove()

            clean_dcfs = {}
            clean_resids_dict = {}
            for lk, li in [("mid", mid_layer), ("last", last_layer)]:
                if lk in clean_data:
                    r = clean_data[lk][0, pos].numpy()
                    clean_resids_dict[lk] = r
                    clean_dcfs[lk] = logit_lens_dcf(r, W_U, tokenizer)

            # Clean entropy
            if "last" in clean_data:
                logits_clean = clean_data["last"][0, pos].numpy() @ W_U.T
                entropies_clean.append(compute_entropy(logits_clean))

            # Perturbed
            perturb_vec = perturb_dir * target_norm
            perturb_tensor = torch.tensor(perturb_vec, dtype=torch.float32)

            pert_data = {}
            hooks_p = []
            hooks_p.append(layers_list[mid_layer].register_forward_hook(_make_capture_hook(pert_data, "mid")))
            hooks_p.append(layers_list[last_layer].register_forward_hook(_make_capture_hook(pert_data, "last")))

            def make_perturb_hook(pvec, position):
                _added = [False]
                def hook_fn(module, inp, output):
                    if not _added[0]:
                        if isinstance(output, tuple):
                            out = output[0].clone()
                        else:
                            out = output.clone()
                        out[0, position, :] += pvec.to(out.device).to(out.dtype)
                        _added[0] = True
                        if isinstance(output, tuple):
                            return (out,) + output[1:]
                        return out
                    return output
                return hook_fn

            h0 = layers_list[0].register_forward_hook(make_perturb_hook(perturb_tensor, pos))

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            h0.remove()
            for h in hooks_p:
                h.remove()

            # 计算指标
            for lk, li in [("mid", mid_layer), ("last", last_layer)]:
                if lk in pert_data and lk in clean_dcfs:
                    pert_r = pert_data[lk][0, pos].numpy()
                    pert_dcf = logit_lens_dcf(pert_r, W_U, tokenizer)
                    cos_sim = float(np.dot(pert_dcf, clean_dcfs[lk]) /
                                   (np.linalg.norm(pert_dcf) * np.linalg.norm(clean_dcfs[lk]) + 1e-12))
                    if lk == "mid":
                        recoveries_mid.append(cos_sim)
                    else:
                        recoveries_last.append(cos_sim)

            # Last层详细分析
            if "last" in pert_data and "last" in clean_resids_dict:
                pert_last_r = pert_data["last"][0, pos].numpy()
                clean_last_r = clean_resids_dict["last"]
                delta = pert_last_r - clean_last_r

                # DCF/null分解
                dcf_comp, null_comp = project_dcf_subspace(delta, ortho_basis)
                dcf_frac = np.linalg.norm(dcf_comp)**2 / (np.linalg.norm(delta)**2 + 1e-12)
                dcf_fracs.append(float(dcf_frac))

                # norm ratio
                nr = np.linalg.norm(delta) / (np.linalg.norm(clean_last_r) + 1e-12)
                norm_ratios.append(float(nr))

                # Pert entropy
                logits_pert = pert_last_r @ W_U.T
                entropies_pert.append(compute_entropy(logits_pert))

        results[f"beta_{beta}"] = {
            "mid_recovery": float(np.mean(recoveries_mid)) if recoveries_mid else 0.0,
            "last_recovery": float(np.mean(recoveries_last)) if recoveries_last else 0.0,
            "target_norm": target_norm,
            "dcf_fraction_mean": float(np.mean(dcf_fracs)) if dcf_fracs else 0.0,
            "norm_ratio_mean": float(np.mean(norm_ratios)) if norm_ratios else 0.0,
            "entropy_clean_mean": float(np.mean(entropies_clean)) if entropies_clean else 0.0,
            "entropy_pert_mean": float(np.mean(entropies_pert)) if entropies_pert else 0.0,
        }
        r = results[f"beta_{beta}"]
        plog(f"    mid_rec={r['mid_recovery']:.4f}, last_rec={r['last_recovery']:.4f}, "
             f"dcf_frac={r['dcf_fraction_mean']:.4f}, norm_ratio={r['norm_ratio_mean']:.4f}")

    return results


# ==================== Exp2: L30 Fruit Writer完整8D DCF ====================
def exp2_fruit_writer_full_8d(model, tokenizer, device, model_name):
    """
    在非fruit对象中注入L30 fruit writer, 记录完整8D DCF变化
    只对Qwen3执行
    """
    if model_name != "qwen3":
        plog(f"  Exp2 skipped (only for qwen3)")
        return {"skipped": True, "reason": "only for qwen3"}

    plog(f"=== Exp2: Fruit Writer Full 8D DCF ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    target_layer = 30
    lw = get_layer_weights(layers_list[target_layer], info.d_model, info.mlp_type)
    W_down = lw.W_down
    if W_down is None:
        W_down = layers_list[target_layer].mlp.down_proj.weight.detach().float().cpu().numpy().T

    n_obj = 4

    # Step1: 找到fruit正贡献神经元(与Phase476相同方法)
    plog(f"  Finding fruit-positive neurons at L{target_layer}...")
    fruit_contributions = {}

    for obj in CATEGORIES["fruit"][:n_obj]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

        cap_mid = {}
        h_mid = layers_list[target_layer].mlp.down_proj.register_forward_hook(
            lambda m, i, o: cap_mid.update({"mid": i[0].detach().float().cpu()}) if isinstance(i, tuple) and len(i) > 0 else None)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h_mid.remove()

        if "mid" in cap_mid:
            mid_act = cap_mid["mid"][0, pos].numpy()
            for idx in range(min(len(mid_act), W_down.shape[1])):
                if idx not in fruit_contributions:
                    fruit_contributions[idx] = 0.0
                write_vec = W_down[:, idx] * mid_act[idx]
                logits = write_vec @ W_U.T
                dcf_dim0 = 0.0
                for w in FAMILY_WORDS_8D["fruit"]:
                    tid = find_token_id(tokenizer, w)
                    if tid is not None and tid < len(logits):
                        dcf_dim0 += float(logits[tid])
                dcf_dim0 /= len(FAMILY_WORDS_8D["fruit"])
                fruit_contributions[idx] += dcf_dim0

    sorted_neurons = sorted(fruit_contributions.items(), key=lambda x: -x[1])
    top20_pos = [int(n[0]) for n in sorted_neurons[:20]]
    plog(f"  Top-20 fruit-positive neurons: {top20_pos}")

    # Step2: 计算mean write vector
    plog(f"  Computing mean fruit write vector...")
    write_vectors = []
    for obj in CATEGORIES["fruit"][:n_obj]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

        cap_mid = {}
        h_mid = layers_list[target_layer].mlp.down_proj.register_forward_hook(
            lambda m, i, o: cap_mid.update({"mid": i[0].detach().float().cpu()}) if isinstance(i, tuple) and len(i) > 0 else None)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h_mid.remove()

        if "mid" in cap_mid:
            mid_act = cap_mid["mid"][0, pos].numpy()
            wv = np.zeros(info.d_model)
            for idx in top20_pos:
                if idx < W_down.shape[1]:
                    wv += W_down[:, idx] * mid_act[idx]
            write_vectors.append(wv)

    mean_write_vec = np.mean(write_vectors, axis=0) if write_vectors else np.zeros(info.d_model)
    write_vec_norm = np.linalg.norm(mean_write_vec)
    plog(f"  Mean fruit write vector norm: {write_vec_norm:.4f}")

    # Step3: 注入测试 — 完整8D DCF
    inject_targets = {
        "animal": CATEGORIES["animal"][:3],
        "vehicle": CATEGORIES["vehicle"][:3],
        "tool": CATEGORIES["tool"][:3],
    }

    amplifications = [0.5, 1.0, 2.0]
    results = {}

    for amp in amplifications:
        plog(f"    Amp={amp}")
        inject_vec = mean_write_vec * amp
        inject_tensor = torch.tensor(inject_vec, dtype=torch.float32)

        cat_results = {}
        for target_cat, objs in inject_targets.items():
            dcf_before_list = []
            dcf_after_list = []
            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                # Clean
                cap_clean = {}
                h_clean = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap_clean, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_clean.remove()

                if "resid" not in cap_clean:
                    continue
                dcf_before = logit_lens_dcf(cap_clean["resid"][0, pos].numpy(), W_U, tokenizer)

                # Injected
                cap_pert = {}
                h_pert = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap_pert, "resid"))

                def make_inject_hook(ivec, position):
                    added = [False]
                    def hook_fn(module, inp, output):
                        if not added[0]:
                            if isinstance(output, tuple):
                                out = output[0].clone()
                            else:
                                out = output.clone()
                            out[0, position, :] += ivec.to(out.device).to(out.dtype)
                            added[0] = True
                            if isinstance(output, tuple):
                                return (out,) + output[1:]
                            return out
                        return output
                    return hook_fn

                h_inj = layers_list[target_layer].register_forward_hook(
                    make_inject_hook(inject_tensor, pos))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                h_pert.remove()
                h_inj.remove()

                if "resid" in cap_pert:
                    dcf_after = logit_lens_dcf(cap_pert["resid"][0, pos].numpy(), W_U, tokenizer)
                    dcf_before_list.append(dcf_before)
                    dcf_after_list.append(dcf_after)

            if dcf_before_list:
                mean_before = np.mean(dcf_before_list, axis=0)
                mean_after = np.mean(dcf_after_list, axis=0)
                mean_delta = mean_after - mean_before

                cat_results[target_cat] = {
                    "before": {DCF_DIM_NAMES[i]: float(mean_before[i]) for i in range(len(DCF_DIM_NAMES))},
                    "after": {DCF_DIM_NAMES[i]: float(mean_after[i]) for i in range(len(DCF_DIM_NAMES))},
                    "delta": {DCF_DIM_NAMES[i]: float(mean_delta[i]) for i in range(len(DCF_DIM_NAMES))},
                }

                # 计算选择性
                fruit_delta = mean_delta[0]  # fruit是第0维
                max_other = max(abs(mean_delta[i]) for i in range(1, len(DCF_DIM_NAMES)))
                selectivity = abs(fruit_delta) / (max_other + 0.01)
                cat_results[target_cat]["selectivity"] = float(selectivity)
                plog(f"      {target_cat}: fruit_Δ={fruit_delta:.3f}, max_other_Δ={max_other:.3f}, "
                     f"selectivity={selectivity:.2f}")

        results[f"amp_{amp}"] = cat_results

    results["top20_neurons"] = top20_pos
    results["write_vec_norm"] = float(write_vec_norm)
    return results


# ==================== Exp3: 跨对象+跨模板泛化 ====================
def exp3_cross_object_template(model, tokenizer, device, model_name):
    """
    测试fruit writer的跨对象和跨模板泛化能力
    只对Qwen3执行
    """
    if model_name != "qwen3":
        plog(f"  Exp3 skipped (only for qwen3)")
        return {"skipped": True, "reason": "only for qwen3"}

    plog(f"=== Exp3: Cross-Object & Cross-Template Generalization ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    target_layer = 30
    lw = get_layer_weights(layers_list[target_layer], info.d_model, info.mlp_type)
    W_down = lw.W_down
    if W_down is None:
        W_down = layers_list[target_layer].mlp.down_proj.weight.detach().float().cpu().numpy().T

    # 使用与Exp2相同的top-20神经元
    # 为简洁, 重新计算(只花几秒)
    n_obj = 4
    fruit_contributions = {}
    for obj in CATEGORIES["fruit"][:n_obj]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap_mid = {}
        h_mid = layers_list[target_layer].mlp.down_proj.register_forward_hook(
            lambda m, i, o: cap_mid.update({"mid": i[0].detach().float().cpu()}) if isinstance(i, tuple) and len(i) > 0 else None)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h_mid.remove()
        if "mid" in cap_mid:
            mid_act = cap_mid["mid"][0, pos].numpy()
            for idx in range(min(len(mid_act), W_down.shape[1])):
                if idx not in fruit_contributions:
                    fruit_contributions[idx] = 0.0
                write_vec = W_down[:, idx] * mid_act[idx]
                logits = write_vec @ W_U.T
                dcf_dim0 = 0.0
                for w in FAMILY_WORDS_8D["fruit"]:
                    tid = find_token_id(tokenizer, w)
                    if tid is not None and tid < len(logits):
                        dcf_dim0 += float(logits[tid])
                dcf_dim0 /= len(FAMILY_WORDS_8D["fruit"])
                fruit_contributions[idx] += dcf_dim0

    sorted_neurons = sorted(fruit_contributions.items(), key=lambda x: -x[1])
    top20_pos = [int(n[0]) for n in sorted_neurons[:20]]

    # 计算mean write vector
    write_vectors = []
    for obj in CATEGORIES["fruit"][:n_obj]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap_mid = {}
        h_mid = layers_list[target_layer].mlp.down_proj.register_forward_hook(
            lambda m, i, o: cap_mid.update({"mid": i[0].detach().float().cpu()}) if isinstance(i, tuple) and len(i) > 0 else None)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h_mid.remove()
        if "mid" in cap_mid:
            mid_act = cap_mid["mid"][0, pos].numpy()
            wv = np.zeros(info.d_model)
            for idx in top20_pos:
                if idx < W_down.shape[1]:
                    wv += W_down[:, idx] * mid_act[idx]
            write_vectors.append(wv)

    mean_write_vec = np.mean(write_vectors, axis=0) if write_vectors else np.zeros(info.d_model)
    inject_tensor = torch.tensor(mean_write_vec, dtype=torch.float32)

    results = {}

    # === Part A: 跨对象泛化 ===
    plog(f"  Part A: Cross-Object Generalization")
    test_objects = {
        "train_fruit": FRUIT_TRAIN,
        "heldout_fruit": FRUIT_HELDOUT,
        "animal": CATEGORIES["animal"][:3],
    }

    for obj_group, objs in test_objects.items():
        dcf_deltas = []
        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

            # Clean
            cap_clean = {}
            h_clean = layers_list[info.n_layers - 1].register_forward_hook(
                _make_capture_hook(cap_clean, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_clean.remove()

            if "resid" not in cap_clean:
                continue
            dcf_before = logit_lens_dcf(cap_clean["resid"][0, pos].numpy(), W_U, tokenizer)

            # Injected
            cap_pert = {}
            h_pert = layers_list[info.n_layers - 1].register_forward_hook(
                _make_capture_hook(cap_pert, "resid"))

            def make_inject_hook(ivec, position):
                added = [False]
                def hook_fn(module, inp, output):
                    if not added[0]:
                        if isinstance(output, tuple):
                            out = output[0].clone()
                        else:
                            out = output.clone()
                        out[0, position, :] += ivec.to(out.device).to(out.dtype)
                        added[0] = True
                        if isinstance(output, tuple):
                            return (out,) + output[1:]
                        return out
                    return output
                return hook_fn

            h_inj = layers_list[target_layer].register_forward_hook(make_inject_hook(inject_tensor, pos))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_pert.remove()
            h_inj.remove()

            if "resid" in cap_pert:
                dcf_after = logit_lens_dcf(cap_pert["resid"][0, pos].numpy(), W_U, tokenizer)
                dcf_deltas.append(dcf_after - dcf_before)

        if dcf_deltas:
            mean_delta = np.mean(dcf_deltas, axis=0)
            results[f"cross_obj_{obj_group}"] = {
                "mean_delta": {DCF_DIM_NAMES[i]: float(mean_delta[i]) for i in range(len(DCF_DIM_NAMES))},
                "fruit_delta": float(mean_delta[0]),
                "n_objects": len(dcf_deltas),
            }
            plog(f"    {obj_group}: fruit_Δ={mean_delta[0]:.3f}, "
                 f"animal_Δ={mean_delta[1]:.3f}, tool_Δ={mean_delta[2]:.3f}")

    # === Part B: 跨模板泛化 ===
    plog(f"  Part B: Cross-Template Generalization")
    test_fruits = ["apple", "banana", "pear"]
    template_names = ["kind_of", "belongs_to", "classified_as", "eaten_as"]

    for tpl_name in template_names:
        tpl = RELATION_TEMPLATES[tpl_name]
        dcf_deltas = []
        for obj in test_fruits:
            prompt = tpl.format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

            # Clean
            cap_clean = {}
            h_clean = layers_list[info.n_layers - 1].register_forward_hook(
                _make_capture_hook(cap_clean, "resid"))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_clean.remove()

            if "resid" not in cap_clean:
                continue
            dcf_before = logit_lens_dcf(cap_clean["resid"][0, pos].numpy(), W_U, tokenizer)

            # Injected
            cap_pert = {}
            h_pert = layers_list[info.n_layers - 1].register_forward_hook(
                _make_capture_hook(cap_pert, "resid"))

            def make_inject_hook(ivec, position):
                added = [False]
                def hook_fn(module, inp, output):
                    if not added[0]:
                        if isinstance(output, tuple):
                            out = output[0].clone()
                        else:
                            out = output.clone()
                        out[0, position, :] += ivec.to(out.device).to(out.dtype)
                        added[0] = True
                        if isinstance(output, tuple):
                            return (out,) + output[1:]
                        return out
                    return output
                return hook_fn

            h_inj = layers_list[target_layer].register_forward_hook(make_inject_hook(inject_tensor, pos))
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_pert.remove()
            h_inj.remove()

            if "resid" in cap_pert:
                dcf_after = logit_lens_dcf(cap_pert["resid"][0, pos].numpy(), W_U, tokenizer)
                dcf_deltas.append(dcf_after - dcf_before)

        if dcf_deltas:
            mean_delta = np.mean(dcf_deltas, axis=0)
            results[f"cross_tpl_{tpl_name}"] = {
                "mean_delta": {DCF_DIM_NAMES[i]: float(mean_delta[i]) for i in range(len(DCF_DIM_NAMES))},
                "fruit_delta": float(mean_delta[0]),
                "n_objects": len(dcf_deltas),
            }
            plog(f"    {tpl_name}: fruit_Δ={mean_delta[0]:.3f}, food_Δ={mean_delta[6]:.3f}, "
                 f"plant_Δ={mean_delta[7]:.3f}")

    return results


# ==================== Exp4: 扩大消融规模 ====================
def exp4_scaled_ablation(model, tokenizer, device, model_name):
    """
    扩大消融规模: top-20, 50, 100, 200
    只对Qwen3执行
    """
    if model_name != "qwen3":
        plog(f"  Exp4 skipped (only for qwen3)")
        return {"skipped": True, "reason": "only for qwen3"}

    plog(f"=== Exp4: Scaled Ablation ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    target_layer = 30
    lw = get_layer_weights(layers_list[target_layer], info.d_model, info.mlp_type)
    W_down = lw.W_down
    if W_down is None:
        W_down = layers_list[target_layer].mlp.down_proj.weight.detach().float().cpu().numpy().T

    n_obj = 4
    cat_list_test = ["fruit", "animal", "vehicle", "tool"]

    # Step1: 找到所有fruit相关神经元(与Exp2相同)
    plog(f"  Finding fruit contributions at L{target_layer}...")
    fruit_contributions = {}
    for obj in CATEGORIES["fruit"][:n_obj]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap_mid = {}
        h_mid = layers_list[target_layer].mlp.down_proj.register_forward_hook(
            lambda m, i, o: cap_mid.update({"mid": i[0].detach().float().cpu()}) if isinstance(i, tuple) and len(i) > 0 else None)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h_mid.remove()
        if "mid" in cap_mid:
            mid_act = cap_mid["mid"][0, pos].numpy()
            for idx in range(min(len(mid_act), W_down.shape[1])):
                if idx not in fruit_contributions:
                    fruit_contributions[idx] = 0.0
                write_vec = W_down[:, idx] * mid_act[idx]
                logits = write_vec @ W_U.T
                dcf_dim0 = 0.0
                for w in FAMILY_WORDS_8D["fruit"]:
                    tid = find_token_id(tokenizer, w)
                    if tid is not None and tid < len(logits):
                        dcf_dim0 += float(logits[tid])
                dcf_dim0 /= len(FAMILY_WORDS_8D["fruit"])
                fruit_contributions[idx] += dcf_dim0

    sorted_neurons = sorted(fruit_contributions.items(), key=lambda x: -x[1])

    # 定义不同规模的消融集合
    ablation_scales = [20, 50, 100, 200]
    max_n = len(sorted_neurons)

    results = {}

    for scale in ablation_scales:
        actual_scale = min(scale, max_n)
        top_pos = set(int(n[0]) for n in sorted_neurons[:actual_scale])
        top_neg = set(int(n[0]) for n in sorted_neurons[-actual_scale:])
        mid_start = max_n // 2 - actual_scale // 2
        random_set = set(int(n[0]) for n in sorted_neurons[mid_start:mid_start + actual_scale])

        ablation_modes = {
            "positive_only": top_pos,
            "negative_only": top_neg,
            "pos_plus_neg": top_pos | top_neg,
            "random": random_set,
        }

        plog(f"  Scale={actual_scale}")
        scale_results = {}

        for mode_name, neuron_set in ablation_modes.items():
            neuron_list = sorted([int(i) for i in neuron_set])

            cat_dcfs = {}
            for cat in cat_list_test:
                objs = CATEGORIES.get(cat, [])[:n_obj]
                dcfs = []
                for obj in objs:
                    prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    # 零掉指定神经元
                    def make_pre_ablation_hook(indices, position):
                        def pre_hook(module, args):
                            if isinstance(args, tuple) and len(args) > 0:
                                x = args[0].clone()
                                for idx in indices:
                                    if idx < x.shape[-1]:
                                        x[0, position, idx] = 0.0
                                return (x,) + args[1:] if len(args) > 1 else (x,)
                            return args
                        return pre_hook

                    h_abl = layers_list[target_layer].mlp.down_proj.register_forward_pre_hook(
                        make_pre_ablation_hook(neuron_list, pos))
                    cap = {}
                    h_resid = layers_list[info.n_layers - 1].register_forward_hook(
                        _make_capture_hook(cap, "resid"))

                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)

                    h_abl.remove()
                    h_resid.remove()

                    if "resid" in cap:
                        dcfs.append(logit_lens_dcf(cap["resid"][0, pos].numpy(), W_U, tokenizer))

                if dcfs:
                    cat_dcfs[cat] = np.mean(dcfs, axis=0)

            # 记录完整8D DCF + 特异性指标
            mode_result = {}
            for cat in cat_list_test:
                if cat in cat_dcfs:
                    mode_result[cat] = {DCF_DIM_NAMES[i]: float(cat_dcfs[cat][i]) for i in range(len(DCF_DIM_NAMES))}

            # 计算fruit margin: fruit DCF fruit维度 - max(other DCF fruit维度)
            if "fruit" in cat_dcfs:
                fruit_fruit_dcf = cat_dcfs["fruit"][0]  # fruit对象的fruit维度
                other_fruit_dcfs = []
                for cat in ["animal", "vehicle", "tool"]:
                    if cat in cat_dcfs:
                        other_fruit_dcfs.append(cat_dcfs[cat][0])
                if other_fruit_dcfs:
                    fruit_margin = fruit_fruit_dcf - max(other_fruit_dcfs)
                    mode_result["fruit_margin"] = float(fruit_margin)

            scale_results[mode_name] = mode_result

            if "fruit" in cat_dcfs:
                plog(f"    {mode_name}: fruit_dim0={cat_dcfs['fruit'][0]:.2f}, "
                     f"animal_dim0={cat_dcfs.get('animal', [0])[0]:.2f}, "
                     f"fruit_margin={mode_result.get('fruit_margin', 0):.2f}")

        results[f"scale_{actual_scale}"] = scale_results

    return results


# ==================== Exp5: Fruit Writer剂量-响应与流形 ====================
def exp5_dose_response_manifold(model, tokenizer, device, model_name):
    """
    Fruit writer不同注入强度: 0.25×, 0.5×, 1.0×, 1.5×, 2.0×
    记录: 完整8D DCF, entropy, norm ratio, top-5预测
    只对Qwen3执行
    """
    if model_name != "qwen3":
        plog(f"  Exp5 skipped (only for qwen3)")
        return {"skipped": True, "reason": "only for qwen3"}

    plog(f"=== Exp5: Dose-Response vs Manifold ({model_name}) ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    target_layer = 30
    lw = get_layer_weights(layers_list[target_layer], info.d_model, info.mlp_type)
    W_down = lw.W_down
    if W_down is None:
        W_down = layers_list[target_layer].mlp.down_proj.weight.detach().float().cpu().numpy().T

    n_obj = 3

    # 计算mean write vector (与Exp2相同)
    fruit_contributions = {}
    for obj in CATEGORIES["fruit"][:n_obj]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap_mid = {}
        h_mid = layers_list[target_layer].mlp.down_proj.register_forward_hook(
            lambda m, i, o: cap_mid.update({"mid": i[0].detach().float().cpu()}) if isinstance(i, tuple) and len(i) > 0 else None)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h_mid.remove()
        if "mid" in cap_mid:
            mid_act = cap_mid["mid"][0, pos].numpy()
            for idx in range(min(len(mid_act), W_down.shape[1])):
                if idx not in fruit_contributions:
                    fruit_contributions[idx] = 0.0
                write_vec = W_down[:, idx] * mid_act[idx]
                logits = write_vec @ W_U.T
                dcf_dim0 = 0.0
                for w in FAMILY_WORDS_8D["fruit"]:
                    tid = find_token_id(tokenizer, w)
                    if tid is not None and tid < len(logits):
                        dcf_dim0 += float(logits[tid])
                dcf_dim0 /= len(FAMILY_WORDS_8D["fruit"])
                fruit_contributions[idx] += dcf_dim0

    sorted_neurons = sorted(fruit_contributions.items(), key=lambda x: -x[1])
    top20_pos = [int(n[0]) for n in sorted_neurons[:20]]

    write_vectors = []
    for obj in CATEGORIES["fruit"][:n_obj]:
        prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
        input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)
        cap_mid = {}
        h_mid = layers_list[target_layer].mlp.down_proj.register_forward_hook(
            lambda m, i, o: cap_mid.update({"mid": i[0].detach().float().cpu()}) if isinstance(i, tuple) and len(i) > 0 else None)
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attention_mask)
        h_mid.remove()
        if "mid" in cap_mid:
            mid_act = cap_mid["mid"][0, pos].numpy()
            wv = np.zeros(info.d_model)
            for idx in top20_pos:
                if idx < W_down.shape[1]:
                    wv += W_down[:, idx] * mid_act[idx]
            write_vectors.append(wv)

    mean_write_vec = np.mean(write_vectors, axis=0) if write_vectors else np.zeros(info.d_model)
    write_vec_norm = np.linalg.norm(mean_write_vec)

    doses = [0.25, 0.5, 1.0, 1.5, 2.0]
    test_objects = {
        "animal": CATEGORIES["animal"][:2],
        "tool": CATEGORIES["tool"][:2],
    }

    results = {}

    for dose in doses:
        plog(f"  Dose={dose}")
        inject_vec = mean_write_vec * dose
        inject_tensor = torch.tensor(inject_vec, dtype=torch.float32)

        # 计算注入范数与流形尺度的比值
        inject_norm = np.linalg.norm(inject_vec)
        manifold_scale = math.sqrt(info.d_model)
        rho = inject_norm / manifold_scale

        dose_results = {}

        for target_cat, objs in test_objects.items():
            dcf_before_list = []
            dcf_after_list = []
            entropy_before = []
            entropy_after = []
            norm_ratios = []
            top5_changes = []

            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                # Clean
                cap_clean = {}
                h_clean = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap_clean, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_clean.remove()

                if "resid" not in cap_clean:
                    continue

                clean_r = cap_clean["resid"][0, pos].numpy()
                dcf_before = logit_lens_dcf(clean_r, W_U, tokenizer)
                logits_clean = clean_r @ W_U.T
                entropy_before.append(compute_entropy(logits_clean))

                # Top-5 clean
                top5_clean_ids = np.argsort(logits_clean)[-5:][::-1]
                top5_clean = [(tokenizer.decode([int(i)]).strip(), float(logits_clean[i])) for i in top5_clean_ids]

                # Injected
                cap_pert = {}
                h_pert = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap_pert, "resid"))

                def make_inject_hook(ivec, position):
                    added = [False]
                    def hook_fn(module, inp, output):
                        if not added[0]:
                            if isinstance(output, tuple):
                                out = output[0].clone()
                            else:
                                out = output.clone()
                            out[0, position, :] += ivec.to(out.device).to(out.dtype)
                            added[0] = True
                            if isinstance(output, tuple):
                                return (out,) + output[1:]
                            return out
                        return output
                    return hook_fn

                h_inj = layers_list[target_layer].register_forward_hook(make_inject_hook(inject_tensor, pos))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_pert.remove()
                h_inj.remove()

                if "resid" in cap_pert:
                    pert_r = cap_pert["resid"][0, pos].numpy()
                    dcf_after = logit_lens_dcf(pert_r, W_U, tokenizer)
                    logits_pert = pert_r @ W_U.T
                    entropy_after.append(compute_entropy(logits_pert))

                    delta = pert_r - clean_r
                    norm_ratios.append(float(np.linalg.norm(delta) / (np.linalg.norm(clean_r) + 1e-12)))

                    # Top-5 pert
                    top5_pert_ids = np.argsort(logits_pert)[-5:][::-1]
                    top5_pert = [(tokenizer.decode([int(i)]).strip(), float(logits_pert[i])) for i in top5_pert_ids]
                    top5_changes.append({"clean": top5_clean, "perturbed": top5_pert})

                    dcf_before_list.append(dcf_before)
                    dcf_after_list.append(dcf_after)

            if dcf_before_list:
                mean_delta = np.mean(dcf_after_list, axis=0) - np.mean(dcf_before_list, axis=0)
                dose_results[target_cat] = {
                    "delta_8d": {DCF_DIM_NAMES[i]: float(mean_delta[i]) for i in range(len(DCF_DIM_NAMES))},
                    "fruit_delta": float(mean_delta[0]),
                    "entropy_change": float(np.mean(entropy_after) - np.mean(entropy_before)) if entropy_before else 0.0,
                    "norm_ratio": float(np.mean(norm_ratios)) if norm_ratios else 0.0,
                    "rho": float(rho),
                    "top5_examples": top5_changes[:1] if top5_changes else [],
                }
                plog(f"    {target_cat}: fruit_Δ={mean_delta[0]:.3f}, ρ={rho:.3f}, "
                     f"entropy_Δ={dose_results[target_cat]['entropy_change']:.2f}")

        results[f"dose_{dose}"] = dose_results

    results["write_vec_norm"] = float(write_vec_norm)
    results["manifold_scale_sqrt_d"] = float(manifold_scale)
    return results


# ==================== Exp6: DS7B Head 12必要性 ====================
def exp6_ds7b_head12_necessity(model, tokenizer, device, model_name):
    """
    关闭DS7B L27 Head 12, 测试format/semantic/DCF变化
    同时测试Head 13(第二格式覆盖head)和Head 0(语义head)作为对照
    只对DS7B执行
    """
    if model_name != "deepseek7b":
        plog(f"  Exp6 skipped (only for deepseek7b)")
        return {"skipped": True, "reason": "only for deepseek7b"}

    plog(f"=== Exp6: DS7B Head 12 Necessity Test ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    target_layer = 27
    layer = layers_list[target_layer]

    # 获取head配置
    if hasattr(layer.self_attn, 'num_heads'):
        n_heads = layer.self_attn.num_heads
    elif hasattr(model.config, 'num_attention_heads'):
        n_heads = model.config.num_attention_heads
    else:
        n_heads = info.d_model // layer.self_attn.head_dim
    head_dim = layer.self_attn.head_dim
    plog(f"  L27: n_heads={n_heads}, head_dim={head_dim}")

    # format/semantic/math token IDs
    format_ids = []
    for t in FORMAT_TOKENS:
        tid = find_token_id(tokenizer, t)
        if tid is not None:
            format_ids.append(tid)
    semantic_ids = []
    for t in SEMANTIC_TOKENS:
        tid = find_token_id(tokenizer, t)
        if tid is not None:
            semantic_ids.append(tid)
    math_ids = []
    for t in MATH_TOKENS:
        tid = find_token_id(tokenizer, t)
        if tid is not None:
            math_ids.append(tid)

    plog(f"  Format tokens: {len(format_ids)}, Semantic: {len(semantic_ids)}, Math: {len(math_ids)}")

    # 测试heads: 12(格式覆盖), 13(第二格式), 0(语义), 21(语义)
    test_heads = [12, 13, 0, 21]
    n_obj = 3

    # 测试对象: fruit + animal + vehicle
    test_cats = {
        "fruit": CATEGORIES["fruit"][:n_obj],
        "animal": CATEGORIES["animal"][:n_obj],
    }

    results = {}

    for head_idx in test_heads:
        plog(f"  Ablating head_{head_idx}...")
        start = head_idx * head_dim
        end = start + head_dim

        head_result = {}

        for cat, objs in test_cats.items():
            format_scores = []
            semantic_scores = []
            math_scores = []
            dcf_before_list = []
            dcf_after_list = []

            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                # Clean baseline
                cap_clean = {}
                h_clean = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap_clean, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_clean.remove()

                if "resid" not in cap_clean:
                    continue

                clean_r = cap_clean["resid"][0, pos].numpy()
                logits_clean = clean_r @ W_U.T
                clean_format = float(np.mean([logits_clean[tid] for tid in format_ids if tid < len(logits_clean)]))
                clean_semantic = float(np.mean([logits_clean[tid] for tid in semantic_ids if tid < len(logits_clean)]))
                clean_math = float(np.mean([logits_clean[tid] for tid in math_ids if tid < len(logits_clean)]))
                dcf_before = logit_lens_dcf(clean_r, W_U, tokenizer)

                # Ablate head
                cap_abl = {}
                h_abl = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap_abl, "resid"))

                def make_head_ablation_hook(h_start, h_end, position):
                    def pre_hook(module, args):
                        if isinstance(args, tuple) and len(args) > 0:
                            x = args[0].clone()
                            x[0, position, h_start:h_end] = 0.0
                            return (x,) + args[1:] if len(args) > 1 else (x,)
                        return args
                    return pre_hook

                h_pre = layer.self_attn.o_proj.register_forward_pre_hook(
                    make_head_ablation_hook(start, end, pos))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                h_pre.remove()
                h_abl.remove()

                if "resid" in cap_abl:
                    abl_r = cap_abl["resid"][0, pos].numpy()
                    logits_abl = abl_r @ W_U.T
                    abl_format = float(np.mean([logits_abl[tid] for tid in format_ids if tid < len(logits_abl)]))
                    abl_semantic = float(np.mean([logits_abl[tid] for tid in semantic_ids if tid < len(logits_abl)]))
                    abl_math = float(np.mean([logits_abl[tid] for tid in math_ids if tid < len(logits_abl)]))
                    dcf_after = logit_lens_dcf(abl_r, W_U, tokenizer)

                    # 差值 = clean - ablation → head贡献
                    format_scores.append(clean_format - abl_format)
                    semantic_scores.append(clean_semantic - abl_semantic)
                    math_scores.append(clean_math - abl_math)
                    dcf_before_list.append(dcf_before)
                    dcf_after_list.append(dcf_after)

            if format_scores:
                mean_dcf_before = np.mean(dcf_before_list, axis=0)
                mean_dcf_after = np.mean(dcf_after_list, axis=0)
                dcf_delta = mean_dcf_after - mean_dcf_before

                head_result[cat] = {
                    "format_score_change": float(np.mean(format_scores)),
                    "semantic_score_change": float(np.mean(semantic_scores)),
                    "math_score_change": float(np.mean(math_scores)),
                    "dcf_delta": {DCF_DIM_NAMES[i]: float(dcf_delta[i]) for i in range(len(DCF_DIM_NAMES))},
                }
                plog(f"    {cat}: format_Δ={np.mean(format_scores):.3f}, "
                     f"semantic_Δ={np.mean(semantic_scores):.3f}, "
                     f"math_Δ={np.mean(math_scores):.3f}")

        results[f"head_{head_idx}"] = head_result

    return results


# ==================== Exp7: DS7B Head 12充分性 ====================
def exp7_ds7b_head12_sufficiency(model, tokenizer, device, model_name):
    """
    在L26语义状态后注入Head 12的输出, 看format score是否上升
    只对DS7B执行

    方法:
    1. 在fruit对象上运行, 捕获L27 o_proj的输入
    2. 提取head_12部分的输出, 乘以W_o得到head_12的residual贡献
    3. 在新对象上, 将这个贡献注入到L26输出后(即L27输入中)
    4. 检查format/semantic/DCF变化

    如果W_o不可用(在meta device上), 用ablation方法:
    1. Clean前向
    2. 在L27 o_proj输入中只保留head_12, 零掉其他heads
    """
    if model_name != "deepseek7b":
        plog(f"  Exp7 skipped (only for deepseek7b)")
        return {"skipped": True, "reason": "only for deepseek7b"}

    plog(f"=== Exp7: DS7B Head 12 Sufficiency Test ===")
    info = get_model_info(model, model_name)
    layers_list = get_layers(model)
    W_U = get_W_U(model, model_name)

    target_layer = 27
    inject_layer = 26  # 注入到L26输出
    layer = layers_list[target_layer]

    # 获取head配置
    if hasattr(layer.self_attn, 'num_heads'):
        n_heads = layer.self_attn.num_heads
    elif hasattr(model.config, 'num_attention_heads'):
        n_heads = model.config.num_attention_heads
    else:
        n_heads = info.d_model // layer.self_attn.head_dim
    head_dim = layer.self_attn.head_dim

    # 尝试获取W_o
    try:
        W_o = layer.self_attn.o_proj.weight.detach().float().cpu().numpy()  # [d_model, d_model]
        has_W_o = True
        plog(f"  W_o available: shape={W_o.shape}")
    except (NotImplementedError, RuntimeError):
        has_W_o = False
        plog(f"  W_o not available (meta device), using isolation method")

    # format/semantic token IDs
    format_ids = [find_token_id(tokenizer, t) for t in FORMAT_TOKENS]
    format_ids = [tid for tid in format_ids if tid is not None]
    semantic_ids = [find_token_id(tokenizer, t) for t in SEMANTIC_TOKENS]
    semantic_ids = [tid for tid in semantic_ids if tid is not None]

    n_obj = 2
    results = {}

    # 方法1: 用W_o直接计算head_12输出, 然后注入到L26后
    # 方法2: 无W_o时, 在L27 o_proj输入中只保留head_12

    if has_W_o:
        # 先从fruit对象上获取head_12的典型输出模式
        plog(f"  Method: W_o-based head_12 extraction and injection")

        # 捕获head_12在fruit上下文中的输出
        head12_outputs = []
        for obj in CATEGORIES["fruit"][:n_obj]:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

            cap_attn = {}
            def attn_output_hook(module, args):
                if isinstance(args, tuple) and len(args) > 0:
                    cap_attn["attn_output"] = args[0].detach().float().cpu()

            h_attn = layer.self_attn.o_proj.register_forward_pre_hook(attn_output_hook)
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)
            h_attn.remove()

            if "attn_output" in cap_attn:
                attn_out = cap_attn["attn_output"][0, pos].numpy()
                # 提取head_12部分
                start = 12 * head_dim
                end = start + head_dim
                head12_part = attn_out[start:end]
                # 计算head_12对residual的贡献
                head12_contribution = W_o[:, start:end] @ head12_part
                head12_outputs.append(head12_contribution)

        if not head12_outputs:
            return {"error": "No head_12 outputs captured"}

        mean_head12_output = np.mean(head12_outputs, axis=0)
        head12_norm = np.linalg.norm(mean_head12_output)
        plog(f"  Mean head_12 output norm: {head12_norm:.4f}")

        # 注入测试: 在不同对象上注入head_12的输出
        inject_amplifications = [0.5, 1.0, 2.0]
        test_cats = {
            "animal": CATEGORIES["animal"][:2],
            "vehicle": CATEGORIES["vehicle"][:2],
        }

        for amp in inject_amplifications:
            inject_vec = mean_head12_output * amp
            inject_tensor = torch.tensor(inject_vec, dtype=torch.float32)

            amp_results = {}
            for cat, objs in test_cats.items():
                format_before_list = []
                format_after_list = []
                semantic_before_list = []
                semantic_after_list = []
                dcf_before_list = []
                dcf_after_list = []

                for obj in objs:
                    prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                    input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                    # Clean
                    cap_clean = {}
                    h_clean = layers_list[info.n_layers - 1].register_forward_hook(
                        _make_capture_hook(cap_clean, "resid"))
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h_clean.remove()

                    if "resid" not in cap_clean:
                        continue

                    clean_r = cap_clean["resid"][0, pos].numpy()
                    logits_clean = clean_r @ W_U.T
                    format_before_list.append(float(np.mean([logits_clean[tid] for tid in format_ids if tid < len(logits_clean)])))
                    semantic_before_list.append(float(np.mean([logits_clean[tid] for tid in semantic_ids if tid < len(logits_clean)])))
                    dcf_before_list.append(logit_lens_dcf(clean_r, W_U, tokenizer))

                    # Inject head_12 output into L26
                    cap_pert = {}
                    h_pert = layers_list[info.n_layers - 1].register_forward_hook(
                        _make_capture_hook(cap_pert, "resid"))

                    def make_inject_hook(ivec, position):
                        added = [False]
                        def hook_fn(module, inp, output):
                            if not added[0]:
                                if isinstance(output, tuple):
                                    out = output[0].clone()
                                else:
                                    out = output.clone()
                                out[0, position, :] += ivec.to(out.device).to(out.dtype)
                                added[0] = True
                                if isinstance(output, tuple):
                                    return (out,) + output[1:]
                                return out
                            return output
                        return hook_fn

                    h_inj = layers_list[inject_layer].register_forward_hook(make_inject_hook(inject_tensor, pos))
                    with torch.no_grad():
                        model(input_ids=input_ids, attention_mask=attention_mask)
                    h_pert.remove()
                    h_inj.remove()

                    if "resid" in cap_pert:
                        pert_r = cap_pert["resid"][0, pos].numpy()
                        logits_pert = pert_r @ W_U.T
                        format_after_list.append(float(np.mean([logits_pert[tid] for tid in format_ids if tid < len(logits_pert)])))
                        semantic_after_list.append(float(np.mean([logits_pert[tid] for tid in semantic_ids if tid < len(logits_pert)])))
                        dcf_after_list.append(logit_lens_dcf(pert_r, W_U, tokenizer))

                if format_before_list:
                    format_change = np.mean(format_after_list) - np.mean(format_before_list)
                    semantic_change = np.mean(semantic_after_list) - np.mean(semantic_before_list)
                    mean_dcf_delta = np.mean(dcf_after_list, axis=0) - np.mean(dcf_before_list, axis=0)

                    amp_results[cat] = {
                        "format_change": float(format_change),
                        "semantic_change": float(semantic_change),
                        "dcf_delta": {DCF_DIM_NAMES[i]: float(mean_dcf_delta[i]) for i in range(len(DCF_DIM_NAMES))},
                    }
                    plog(f"    amp={amp} {cat}: format_Δ={format_change:.3f}, semantic_Δ={semantic_change:.3f}")

            results[f"amp_{amp}"] = amp_results

        results["head12_output_norm"] = float(head12_norm)

    else:
        # 无W_o: 使用isolation方法 — 只保留head_12
        plog(f"  Method: Head isolation (keep only head_12 in L27)")

        test_cats = {
            "fruit": CATEGORIES["fruit"][:n_obj],
            "animal": CATEGORIES["animal"][:n_obj],
        }

        for cat, objs in test_cats.items():
            format_scores = []
            semantic_scores = []
            dcf_before_list = []
            dcf_after_list = []

            for obj in objs:
                prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
                input_ids, attention_mask, pos = get_prompt_ids(tokenizer, device, prompt)

                # Clean
                cap_clean = {}
                h_clean = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap_clean, "resid"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                h_clean.remove()

                if "resid" not in cap_clean:
                    continue

                clean_r = cap_clean["resid"][0, pos].numpy()
                logits_clean = clean_r @ W_U.T
                clean_format = float(np.mean([logits_clean[tid] for tid in format_ids if tid < len(logits_clean)]))
                clean_semantic = float(np.mean([logits_clean[tid] for tid in semantic_ids if tid < len(logits_clean)]))
                dcf_before_list.append(logit_lens_dcf(clean_r, W_U, tokenizer))

                # Isolation: 只保留head_12
                cap_iso = {}
                h_iso = layers_list[info.n_layers - 1].register_forward_hook(
                    _make_capture_hook(cap_iso, "resid"))

                def make_head_isolation_hook(keep_head_idx, n_h, h_dim, position):
                    """只保留指定head, 零掉其他所有heads"""
                    keep_start = keep_head_idx * h_dim
                    keep_end = keep_start + h_dim
                    def pre_hook(module, args):
                        if isinstance(args, tuple) and len(args) > 0:
                            x = args[0].clone()
                            for h_idx in range(n_h):
                                h_start = h_idx * h_dim
                                h_end = h_start + h_dim
                                if h_start != keep_start:
                                    x[0, position, h_start:h_end] = 0.0
                            return (x,) + args[1:] if len(args) > 1 else (x,)
                        return args
                    return pre_hook

                h_pre = layer.self_attn.o_proj.register_forward_pre_hook(
                    make_head_isolation_hook(12, n_heads, head_dim, pos))

                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)

                h_pre.remove()
                h_iso.remove()

                if "resid" in cap_iso:
                    iso_r = cap_iso["resid"][0, pos].numpy()
                    logits_iso = iso_r @ W_U.T
                    iso_format = float(np.mean([logits_iso[tid] for tid in format_ids if tid < len(logits_iso)]))
                    iso_semantic = float(np.mean([logits_iso[tid] for tid in semantic_ids if tid < len(logits_iso)]))
                    dcf_after_list.append(logit_lens_dcf(iso_r, W_U, tokenizer))

                    format_scores.append(iso_format - clean_format)
                    semantic_scores.append(iso_semantic - clean_semantic)

            if format_scores:
                mean_dcf_delta = np.mean(dcf_after_list, axis=0) - np.mean(dcf_before_list, axis=0)
                results[f"isolation_{cat}"] = {
                    "format_change": float(np.mean(format_scores)),
                    "semantic_change": float(np.mean(semantic_scores)),
                    "dcf_delta": {DCF_DIM_NAMES[i]: float(mean_dcf_delta[i]) for i in range(len(DCF_DIM_NAMES))},
                }
                plog(f"    isolation_{cat}: format_Δ={np.mean(format_scores):.3f}, "
                     f"semantic_Δ={np.mean(semantic_scores):.3f}")

    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    plog(f"Phase 477: Manifold Boundary, Writer Closure & Format Head Verification")
    plog(f"Model: {model_name}, Round: {round_num}")

    t_start = time.time()

    # 加载模型
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model info: class={info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    results = {
        "phase": 477,
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "theory": "Manifold Boundary, Writer 8D Closure & Format Head Verification",
        "core_question": "Can we close the loop on manifold boundary, fruit writer specificity, and format head causality?",
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }

    # Exp1: 细粒度Beta扫描 (所有模型)
    try:
        results["exp1_fine_beta_scan"] = exp1_fine_beta_scan(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp1 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp1_fine_beta_scan"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Exp1 done. Elapsed: {time.time()-t_start:.1f}s")

    # Exp2: L30 Fruit Writer完整8D DCF (仅qwen3)
    try:
        results["exp2_fruit_writer_full_8d"] = exp2_fruit_writer_full_8d(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp2 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp2_fruit_writer_full_8d"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Exp2 done. Elapsed: {time.time()-t_start:.1f}s")

    # Exp3: 跨对象+跨模板泛化 (仅qwen3)
    try:
        results["exp3_cross_object_template"] = exp3_cross_object_template(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp3 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp3_cross_object_template"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Exp3 done. Elapsed: {time.time()-t_start:.1f}s")

    # Exp4: 扩大消融规模 (仅qwen3)
    try:
        results["exp4_scaled_ablation"] = exp4_scaled_ablation(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp4 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp4_scaled_ablation"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Exp4 done. Elapsed: {time.time()-t_start:.1f}s")

    # Exp5: 剂量-响应与流形 (仅qwen3)
    try:
        results["exp5_dose_response_manifold"] = exp5_dose_response_manifold(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp5 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp5_dose_response_manifold"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Exp5 done. Elapsed: {time.time()-t_start:.1f}s")

    # Exp6: DS7B Head 12必要性 (仅deepseek7b)
    try:
        results["exp6_ds7b_head12_necessity"] = exp6_ds7b_head12_necessity(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp6 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp6_ds7b_head12_necessity"] = {"error": str(e)}

    gc.collect()
    torch.cuda.empty_cache()
    plog(f"Exp6 done. Elapsed: {time.time()-t_start:.1f}s")

    # Exp7: DS7B Head 12充分性 (仅deepseek7b)
    try:
        results["exp7_ds7b_head12_sufficiency"] = exp7_ds7b_head12_sufficiency(model, tokenizer, device, model_name)
    except Exception as e:
        plog(f"Exp7 FAILED: {e}")
        import traceback; traceback.print_exc()
        results["exp7_ds7b_head12_sufficiency"] = {"error": str(e)}

    # 保存结果
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase477_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    plog(f"Results saved to {out_path}")

    # 释放模型
    release_model(model)
    t_total = time.time() - t_start
    plog(f"Phase 477 {model_name} complete. Total: {t_total:.1f}s ({t_total/60:.1f}min)")


if __name__ == "__main__":
    main()
