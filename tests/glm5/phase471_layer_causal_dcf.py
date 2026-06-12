"""
Phase 471: 层因果DCF追踪 — 分布约束的电路定位
================================================
核心问题: 语义约束在哪一层被写入残差流?

Phase 470发现DCF比残差cos更好地聚类语义类别, 但DCF是最终logit的函数,
无法告诉我们约束在哪层被写入。Phase 471用三种方法定位约束写入层:

Exp1: Logit-Lens DCF — 在每层投影到logit空间计算DCF
  - 约束在哪些层变得可读? (可读性 ≠ 写入, 但强相关)
  - 跨层DCF演变轨迹: 从随机→弱结构→强结构的转变点

Exp2: 因果DCF干预 — 注入DCF方向, 看是否能因果控制输出
  - 在kind_of上下文中, 注入另一个类别的DCF方向
  - 测量: 注入后top-1 token是否切换到目标类别
  - 如果成功 → 我们找到了语义约束的因果方向

Exp3: 扩展DCF维度 — 从8维到20+维语义属性
  - 当前8个类别族太粗糙, 无法区分细粒度语义
  - 新增: color, size, motion, habitat, material等
  - 验证: 扩展DCF是否提供更丰富的语义约束信号

模型加载: bfloat16 + device_map="auto" + flash_attention_2

用法:
  python tests/glm5/phase471_layer_causal_dcf.py qwen3 1
  python tests/glm5/phase471_layer_causal_dcf.py glm4 1
  python tests/glm5/phase471_layer_causal_dcf.py deepseek7b 1
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json, math
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS,
                          get_sample_layers, collect_layer_outputs)


def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ==================== 数据定义 ====================
CATEGORIES = {
    "fruit":    ["apple", "banana", "orange", "grape", "pear", "peach", "lemon", "mango"],
    "animal":   ["dog", "cat", "horse", "lion", "bear", "rabbit", "cow", "tiger"],
    "tool":     ["hammer", "knife", "wrench", "saw", "drill", "axe", "shovel", "scissors"],
    "vehicle":  ["car", "bus", "bicycle", "truck", "train", "boat", "plane", "scooter"],
    "clothing": ["shirt", "dress", "hat", "coat", "sock", "glove", "scarf", "boot"],
    "furniture":["chair", "table", "desk", "sofa", "bed", "shelf", "lamp", "cabinet"],
}

# Phase 470 的8维DCF族词
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

# Phase 471 扩展DCF: 20维语义属性
# 每个维度用3-5个代表性token, 确保token在各模型词表中都能找到
EXTENDED_SEMANTIC_DIMS = {
    # --- 8个类别维度 (继承Phase 470) ---
    "cat_fruit":    ["fruit", "produce", "crop", "berry"],
    "cat_animal":   ["animal", "creature", "beast", "pet"],
    "cat_tool":     ["tool", "implement", "device", "instrument"],
    "cat_vehicle":  ["vehicle", "transport", "automobile", "car"],
    "cat_clothing": ["clothing", "attire", "wear", "garment"],
    "cat_furniture":["furniture", "furnishing", "fixture", "seat"],
    "cat_food":     ["food", "meal", "dish", "snack"],
    "cat_plant":    ["plant", "tree", "vegetation", "flora"],
    # --- 12个属性维度 (新增) ---
    "attr_color":   ["color", "red", "blue", "green", "yellow"],
    "attr_size":    ["size", "large", "small", "big", "tiny"],
    "attr_shape":   ["shape", "round", "square", "flat", "long"],
    "attr_material":["material", "wood", "metal", "fabric", "plastic"],
    "attr_motion":  ["motion", "move", "run", "fly", "drive"],
    "attr_habitat": ["habitat", "home", "forest", "water", "field"],
    "attr_function":["function", "use", "work", "help", "serve"],
    "attr_sound":   ["sound", "noise", "music", "quiet", "loud"],
    "attr_taste":   ["taste", "sweet", "bitter", "sour", "spicy"],
    "attr_texture": ["texture", "soft", "hard", "smooth", "rough"],
    "attr_weight":  ["weight", "heavy", "light", "mass", "burden"],
    "attr_temperature":["temperature", "hot", "cold", "warm", "cool"],
}

# 关系模板
RELATION_TEMPLATES = {
    "kind_of":    "The {obj} is a kind of",
    "used_for":   "The {obj} is commonly used for",
    "found_in":   "The {obj} is typically found in",
}

ROUNDS = {
    1: {k: v[:6] for k, v in CATEGORIES.items()},
    2: {k: v[:8] for k, v in CATEGORIES.items()},
}


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
        plog(f"  flash_attention_2 loaded OK")
    except Exception as e:
        plog(f"  flash_attention_2 failed ({e}), falling back to eager")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, local_files_only=True,
            attn_implementation="eager",
        )

    model.eval()
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
    """
    从logits向量计算DCF (可使用任意维度字典)
    """
    dcf_vector = []
    dcf_details = {}
    for dim_name, words in dim_dict.items():
        logit_values = []
        valid_words = []
        for w in words:
            tid = find_token_id(tokenizer, w)
            if tid is not None and tid < len(logits):
                logit_values.append(float(logits[tid]))
                valid_words.append(w)
        if logit_values:
            mean_logit = float(np.mean(logit_values))
            dcf_vector.append(mean_logit)
            dcf_details[dim_name] = {"mean": round(mean_logit, 4), "n_valid": len(valid_words)}
        else:
            dcf_vector.append(0.0)
            dcf_details[dim_name] = {"mean": 0, "n_valid": 0}
    return np.array(dcf_vector), dcf_details


def compute_dcf_centered(dcf_vectors, labels):
    dcf_matrix = np.array(dcf_vectors)
    global_mean = np.mean(dcf_matrix, axis=0)
    centered = dcf_matrix - global_mean
    return centered, global_mean


def cluster_quality(vectors, labels):
    from scipy.spatial.distance import pdist, squareform
    if len(vectors) < 3 or len(set(labels)) < 2:
        return 0.0
    vectors = np.array(vectors)
    labels = np.array(labels)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    vectors_norm = vectors / norms
    dist_matrix = squareform(pdist(vectors_norm, metric='cosine'))
    silhouette_values = []
    unique_labels = list(set(labels))
    for i in range(len(vectors)):
        own_label = labels[i]
        own_cluster = [j for j in range(len(vectors)) if labels[j] == own_label and j != i]
        other_clusters = {l: [j for j in range(len(vectors)) if labels[j] == l]
                          for l in unique_labels if l != own_label}
        if not own_cluster:
            continue
        a = np.mean([dist_matrix[i, j] for j in own_cluster])
        b = float('inf')
        for l, indices in other_clusters.items():
            if indices:
                mean_dist = np.mean([dist_matrix[i, j] for j in indices])
                b = min(b, mean_dist)
        if b == float('inf'):
            b = 0
        s = (b - a) / max(a, b, 1e-10)
        silhouette_values.append(s)
    return float(np.mean(silhouette_values)) if silhouette_values else 0.0


def discriminability(vectors, labels):
    """计算区分类内/类外cos的差异"""
    vectors = np.array(vectors)
    labels = np.array(labels)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    vectors_norm = vectors / norms

    same_cos, diff_cos = [], []
    for i in range(len(vectors)):
        for j in range(i+1, len(vectors)):
            c = float(np.dot(vectors_norm[i], vectors_norm[j]))
            if labels[i] == labels[j]:
                same_cos.append(c)
            else:
                diff_cos.append(c)

    if not same_cos or not diff_cos:
        return 0.0
    return float(np.mean(same_cos) - np.mean(diff_cos))


# ==================== Exp1: Logit-Lens DCF ====================
def exp1_logit_lens_dcf(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    核心实验: 在每层用logit-lens计算DCF, 追踪语义约束何时可读

    方法:
    1. 对每个对象在kind_of模板下前向传播
    2. 用hook收集每层残差
    3. 在每层投影到logit空间: logits_L = resid_L @ W_U^T
    4. 计算DCF_L
    5. 跟踪DCF聚类质量随层的演变
    """
    plog("=== Exp1: Logit-Lens DCF — 层间约束可读性追踪 ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)  # [vocab_size, d_model]
    plog(f"  W_U shape: {W_U.shape}")

    n_obj = 6 if round_num == 1 else 8
    cat_list = ["fruit", "animal", "vehicle", "tool", "furniture", "clothing"]

    # 采样层: 每3层采样 + 首尾
    sample_layers = get_sample_layers(n_layers, n_samples=12)

    # ---- 1a. 收集所有对象在kind_of下的每层残差 ----
    all_layer_resids = {f"L{l}": [] for l in sample_layers}
    all_labels = []
    all_objects = []

    for cat in cat_list:
        objs = obj_dict.get(cat, [])[:n_obj]
        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)

            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            seq_len = attention_mask.sum().item()
            pos = seq_len - 1

            # 收集每层残差
            layers = get_layers(model)
            captured = {}

            def make_hook(key):
                def hook_fn(module, inp, output):
                    if isinstance(inp, tuple) and len(inp) > 0:
                        captured[key] = inp[0].detach().float().cpu()
                return hook_fn

            hooks = []
            for li in sample_layers:
                hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))

            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attention_mask)

            for h in hooks:
                h.remove()

            # 提取最后一个token的残差
            for li in sample_layers:
                key = f"L{li}"
                if key in captured:
                    resid = captured[key][0, pos].numpy()
                    all_layer_resids[key].append(resid)
                else:
                    # 如果hook没捕获到(某些模型架构), 用None占位
                    all_layer_resids[key].append(None)

            all_labels.append(cat)
            all_objects.append(obj)

    plog(f"  Collected residuals for {len(all_objects)} objects across {len(sample_layers)} layers")

    # ---- 1b. 在每层计算logit-lens DCF ----
    layer_dcf_results = {}

    for li in sample_layers:
        key = f"L{li}"
        resids = all_layer_resids[key]

        if any(r is None for r in resids):
            plog(f"  {key}: missing residuals, skip")
            continue

        dcf_vectors = []
        for resid in resids:
            # Logit lens: logits = resid @ W_U^T
            # W_U shape: [vocab_size, d_model], 所以 logits = resid @ W_U.T
            logits = resid @ W_U.T  # [vocab_size]
            dcf_vec, _ = compute_dcf_from_logits(logits, tokenizer, FAMILY_WORDS_8D)
            dcf_vectors.append(dcf_vec)

        # 聚类质量
        sil = cluster_quality(dcf_vectors, all_labels)
        disc = discriminability(dcf_vectors, all_labels)

        # 也计算残差cos的聚类质量(对比)
        resid_sil = cluster_quality(resids, all_labels)

        layer_dcf_results[key] = {
            "logit_lens_dcf_silhouette": round(sil, 4),
            "logit_lens_dcf_discriminability": round(disc, 4),
            "resid_silhouette": round(resid_sil, 4),
            "layer_idx": li,
            "n_objects": len(dcf_vectors),
        }

        plog(f"  {key} (layer {li}): LL-DCF sil={sil:.4f}, disc={disc:.4f}, resid_sil={resid_sil:.4f}")

    # ---- 1c. 找到约束可读的转变点 ----
    sil_values = {k: v["logit_lens_dcf_silhouette"] for k, v in layer_dcf_results.items()}
    if sil_values:
        # 找到silhouette首次超过0.2的层
        emergence_layer = None
        for k in sorted(sil_values.keys(), key=lambda x: int(x[1:])):
            if sil_values[k] > 0.2:
                emergence_layer = k
                break

        # 找到silhouette最大的层
        peak_layer = max(sil_values.keys(), key=lambda k: sil_values[k])

        summary = {
            "n_layers_tested": len(layer_dcf_results),
            "emergence_layer": emergence_layer,
            "peak_layer": peak_layer,
            "peak_silhouette": round(sil_values[peak_layer], 4) if peak_layer else 0,
            "silhouette_trajectory": {k: round(v, 4) for k, v in sorted(
                sil_values.items(), key=lambda x: int(x[0][1:]))},
        }
    else:
        summary = {"n_layers_tested": 0, "emergence_layer": None, "peak_layer": None}

    plog(f"  Emergence: {summary.get('emergence_layer', 'N/A')}, Peak: {summary.get('peak_layer', 'N/A')}")

    return {"per_layer": layer_dcf_results, "summary": summary}


# ==================== Exp2: 因果DCF干预 ====================
def exp2_causal_dcf_intervention(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    核心实验: 注入DCF方向, 看是否能因果控制输出

    方法:
    1. 计算每个类别的DCF中心向量(在kind_of模板下)
    2. 对每个对象, 计算目标类别与源类别的DCF差异方向
    3. 在源类别的上下文中, 注入目标类别的DCF方向
    4. 测量: top-1 token是否从源类别切换到目标类别

    例如: 在"The apple is a kind of"下, 注入vehicle的DCF方向,
          看top-1是否从fruit切换到vehicle
    """
    plog("=== Exp2: 因果DCF干预 — 能否因果控制语义读出? ===")
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    W_U = get_W_U(model, model_name)

    n_obj = 4 if round_num == 1 else 6
    cat_list = ["fruit", "animal", "vehicle", "tool"]

    # ---- 2a. 计算每个类别的DCF中心 (在kind_of下) ----
    plog("  Step 1: Computing category DCF centers...")
    cat_dcf_centers = {}

    for cat in cat_list:
        objs = obj_dict.get(cat, [])[:n_obj]
        dcf_list = []
        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[0, -1].float().cpu().numpy()
            dcf_vec, _ = compute_dcf_from_logits(logits, tokenizer, FAMILY_WORDS_8D)
            dcf_list.append(dcf_vec)

        cat_dcf_centers[cat] = np.mean(dcf_list, axis=0)

    # 中心化DCF方向
    global_mean = np.mean(list(cat_dcf_centers.values()), axis=0)
    cat_dcf_centered = {cat: center - global_mean for cat, center in cat_dcf_centers.items()}

    plog(f"  Category DCF centers computed for {len(cat_list)} categories")

    # ---- 2b. 对每个对象进行因果干预 ----
    plog("  Step 2: Performing causal DCF intervention...")

    # 选择4个干预对: fruit→vehicle, animal→tool, vehicle→fruit, tool→animal
    intervention_pairs = [
        ("fruit", "vehicle"),
        ("animal", "tool"),
        ("fruit", "animal"),
        ("vehicle", "tool"),
    ]

    intervention_results = []
    sample_layers = get_sample_layers(n_layers, n_samples=8)

    for src_cat, tgt_cat in intervention_pairs:
        src_objs = obj_dict.get(src_cat, [])[:2]
        for obj in src_objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)

            # 基线: 无干预
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                base_out = model(input_ids=input_ids, attention_mask=attention_mask)
            base_logits = base_out.logits[0, -1].float().cpu().numpy()
            base_top5 = np.argsort(base_logits)[-5:][::-1]
            base_top5_tokens = [tokenizer.decode([t]) for t in base_top5]

            # 计算基线DCF
            base_dcf, _ = compute_dcf_from_logits(base_logits, tokenizer, FAMILY_WORDS_8D)
            base_top_dim = list(FAMILY_WORDS_8D.keys())[int(np.argmax(base_dcf))]

            # DCF差异方向: target - source
            dcf_direction = cat_dcf_centered[tgt_cat] - cat_dcf_centered[src_cat]
            dcf_direction_norm = np.linalg.norm(dcf_direction)
            if dcf_direction_norm < 1e-10:
                continue

            # 将DCF方向转换到残差空间
            # DCF是logit空间的8维向量, 需要找到residual space中的方向
            # 方法: 找到W_U中使DCF各维度最大的方向
            # 简化: 直接在embedding层注入, 然后看效果

            # 在不同层注入DCF方向的等价残差方向
            layer_intervention = {}

            for li in sample_layers:
                # 计算注入向量: 在logit空间中, DCF方向 = [Δlogit_fruit, Δlogit_animal, ...]
                # 我们需要找到residual space中的等价方向
                # 方法: 对每个DCF维度的族词, 找到W_U中对应的行, 加权平均
                inject_direction = np.zeros(W_U.shape[1])  # [d_model]
                dim_names = list(FAMILY_WORDS_8D.keys())
                n_valid_dims = 0

                for dim_idx, dim_name in enumerate(dim_names):
                    weight = dcf_direction[dim_idx]
                    if abs(weight) < 0.01:
                        continue
                    # 找到该维度的族词token在W_U中对应的行
                    words = FAMILY_WORDS_8D[dim_name]
                    for w in words:
                        tid = find_token_id(tokenizer, w)
                        if tid is not None and tid < W_U.shape[0]:
                            inject_direction += weight * W_U[tid]
                            n_valid_dims += 1

                if n_valid_dims == 0:
                    continue

                # 归一化
                inj_norm = np.linalg.norm(inject_direction)
                if inj_norm > 1e-10:
                    inject_direction = inject_direction / inj_norm

                # 在指定层注入
                layers = get_layers(model)
                captured_logits = {}

                def make_logits_hook(key):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            captured_logits[key] = output[0].detach().clone()
                    return hook_fn

                # 先正常前向到目标层, 捕获输出
                hook = layers[li].register_forward_hook(make_logits_hook("target"))
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attention_mask)
                hook.remove()

                if "target" not in captured_logits:
                    continue

                # 注入方向
                injected_output = captured_logits["target"].clone()
                beta_values = [2.0, 5.0, 10.0]

                beta_results = {}
                for beta in beta_values:
                    # 在最后一个token位置注入
                    inject_tensor = torch.tensor(
                        inject_direction * beta,
                        dtype=injected_output.dtype,
                        device=device
                    )
                    injected_output_mod = injected_output.clone()
                    seq_len = attention_mask.sum().item()
                    pos = seq_len - 1
                    injected_output_mod[0, pos, :] += inject_tensor

                    # 从注入层继续前向
                    # (简化: 在embedding层注入然后重新前向)
                    pass

                # 简化方法: 在embedding层注入
                embed_layer = model.get_input_embeddings()
                inputs_embeds_base = embed_layer(input_ids).detach().clone()

                beta_results = {}
                for beta in beta_values:
                    inject_tensor = torch.tensor(
                        inject_direction * beta,
                        dtype=inputs_embeds_base.dtype,
                        device=device
                    )
                    inputs_embeds_mod = inputs_embeds_base.clone()
                    inputs_embeds_mod[0, pos, :] += inject_tensor

                    with torch.no_grad():
                        try:
                            position_ids = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
                            mod_out = model(inputs_embeds=inputs_embeds_mod, position_ids=position_ids)
                            mod_logits = mod_out.logits[0, -1].float().cpu().numpy()
                        except:
                            continue

                    mod_top5 = np.argsort(mod_logits)[-5:][::-1]
                    mod_top5_tokens = [tokenizer.decode([t]) for t in mod_top5]

                    # 干预后的DCF
                    mod_dcf, _ = compute_dcf_from_logits(mod_logits, tokenizer, FAMILY_WORDS_8D)
                    mod_top_dim = list(FAMILY_WORDS_8D.keys())[int(np.argmax(mod_dcf))]

                    # 检查top-1是否切换
                    base_top1 = base_top5[0]
                    mod_top1 = mod_top5[0]
                    switched = base_top1 != mod_top1

                    # 检查目标类别的DCF维度是否提升
                    tgt_dim_idx = list(FAMILY_WORDS_8D.keys()).index(tgt_cat) if tgt_cat in FAMILY_WORDS_8D else -1
                    src_dim_idx = list(FAMILY_WORDS_8D.keys()).index(src_cat) if src_cat in FAMILY_WORDS_8D else -1

                    tgt_boost = float(mod_dcf[tgt_dim_idx] - base_dcf[tgt_dim_idx]) if tgt_dim_idx >= 0 else 0
                    src_suppress = float(base_dcf[src_dim_idx] - mod_dcf[src_dim_idx]) if src_dim_idx >= 0 else 0

                    beta_results[f"beta_{beta}"] = {
                        "mod_top1": tokenizer.decode([mod_top1]),
                        "mod_top5": mod_top5_tokens,
                        "mod_top_dim": mod_top_dim,
                        "switched": switched,
                        "target_boost": round(tgt_boost, 4),
                        "source_suppress": round(src_suppress, 4),
                        "target_dim_rank": int(list(np.argsort(mod_dcf)[::-1]).index(tgt_dim_idx)) if tgt_dim_idx >= 0 else -1,
                    }

                layer_intervention[f"L{li}"] = beta_results

            result = {
                "object": obj,
                "source_cat": src_cat,
                "target_cat": tgt_cat,
                "prompt": prompt,
                "base_top1": tokenizer.decode([base_top5[0]]),
                "base_top5": base_top5_tokens,
                "base_top_dim": base_top_dim,
                "intervention_by_layer": layer_intervention,
            }
            intervention_results.append(result)
            plog(f"    {obj} ({src_cat}→{tgt_cat}): base_top_dim={base_top_dim}")

    # ---- 2c. 汇总 ----
    # 在最佳beta下, 各层干预的成功率
    best_beta = 5.0
    success_by_layer = {}
    for result in intervention_results:
        for layer_key, beta_res in result.get("intervention_by_layer", {}).items():
            beta_key = f"beta_{best_beta}"
            if beta_key in beta_res:
                if layer_key not in success_by_layer:
                    success_by_layer[layer_key] = {"total": 0, "target_boosted": 0, "dim_switched": 0}
                success_by_layer[layer_key]["total"] += 1
                if beta_res[beta_key]["target_boost"] > 0:
                    success_by_layer[layer_key]["target_boosted"] += 1
                if beta_res[beta_key]["mod_top_dim"] == result["target_cat"]:
                    success_by_layer[layer_key]["dim_switched"] += 1

    summary = {
        "n_interventions": len(intervention_results),
        "best_beta": best_beta,
        "success_by_layer": success_by_layer,
        "causal_control_possible": any(
            v["target_boosted"] > 0 for v in success_by_layer.values()
        ),
    }

    plog(f"  Causal control: {'possible' if summary['causal_control_possible'] else 'not detected'}")

    return {"interventions": intervention_results, "summary": summary}


# ==================== Exp3: 扩展DCF维度 ====================
def exp3_extended_dcf(model, tokenizer, model_name, device, obj_dict, round_num):
    """
    核心实验: 扩展DCF从8维到20维, 验证更细粒度的语义约束

    方法:
    1. 用20维语义字典计算DCF
    2. 比较扩展DCF vs 8维DCF的聚类质量
    3. 分析哪些属性维度在不同类别间有区分力
    """
    plog("=== Exp3: 扩展DCF维度 — 20维语义属性 ===")
    info = get_model_info(model, model_name)

    n_obj = 6 if round_num == 1 else 8
    cat_list = ["fruit", "animal", "vehicle", "tool", "furniture", "clothing"]

    # ---- 3a. 收集所有对象在kind_of下的logits ----
    all_logits = {}
    all_labels = []
    all_objects = []

    for cat in cat_list:
        objs = obj_dict.get(cat, [])[:n_obj]
        for obj in objs:
            prompt = RELATION_TEMPLATES["kind_of"].format(obj=obj)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[0, -1].float().cpu().numpy()
            all_logits[obj] = logits
            all_labels.append(cat)
            all_objects.append(obj)

    # ---- 3b. 8维 vs 20维DCF聚类对比 ----
    dcf_8d_vectors = []
    dcf_20d_vectors = []

    for obj in all_objects:
        logits = all_logits[obj]
        dcf8, _ = compute_dcf_from_logits(logits, tokenizer, FAMILY_WORDS_8D)
        dcf20, _ = compute_dcf_from_logits(logits, tokenizer, EXTENDED_SEMANTIC_DIMS)
        dcf_8d_vectors.append(dcf8)
        dcf_20d_vectors.append(dcf20)

    sil_8d = cluster_quality(dcf_8d_vectors, all_labels)
    sil_20d = cluster_quality(dcf_20d_vectors, all_labels)
    disc_8d = discriminability(dcf_8d_vectors, all_labels)
    disc_20d = discriminability(dcf_20d_vectors, all_labels)

    plog(f"  8D DCF: sil={sil_8d:.4f}, disc={disc_8d:.4f}")
    plog(f"  20D DCF: sil={sil_20d:.4f}, disc={disc_20d:.4f}")

    # ---- 3c. 20维DCF的属性维度分析 ----
    dcf_20d_matrix = np.array(dcf_20d_vectors)
    dim_names = list(EXTENDED_SEMANTIC_DIMS.keys())

    # 每个维度的方差(区分力)
    dim_variances = np.var(dcf_20d_matrix, axis=0)
    dim_order = np.argsort(dim_variances)[::-1]

    dim_importance = []
    for idx in dim_order:
        dim_importance.append({
            "dimension": dim_names[idx],
            "variance": round(float(dim_variances[idx]), 4),
            "type": "category" if dim_names[idx].startswith("cat_") else "attribute",
        })

    # 类别维度 vs 属性维度的平均方差
    cat_var = [float(dim_variances[i]) for i, n in enumerate(dim_names) if n.startswith("cat_")]
    attr_var = [float(dim_variances[i]) for i, n in enumerate(dim_names) if n.startswith("attr_")]

    # ---- 3d. 每个类别的属性DCF profile ----
    cat_attr_profiles = {}
    for cat in cat_list:
        mask = [i for i, l in enumerate(all_labels) if l == cat]
        if mask:
            cat_mean = np.mean(dcf_20d_matrix[mask], axis=0)
            cat_attr_profiles[cat] = {
                dim_names[i]: round(float(cat_mean[i]), 4) for i in range(len(dim_names))
            }

    # 每个类别的top-3属性维度
    cat_top_attrs = {}
    for cat in cat_list:
        mask = [i for i, l in enumerate(all_labels) if l == cat]
        if mask:
            cat_mean = np.mean(dcf_20d_matrix[mask], axis=0)
            # 只看属性维度(非类别维度)
            attr_indices = [i for i, n in enumerate(dim_names) if n.startswith("attr_")]
            attr_vals = [(dim_names[i], float(cat_mean[i])) for i in attr_indices]
            attr_vals.sort(key=lambda x: x[1], reverse=True)
            cat_top_attrs[cat] = attr_vals[:3]

    results = {
        "comparison": {
            "8d_silhouette": round(sil_8d, 4),
            "20d_silhouette": round(sil_20d, 4),
            "8d_discriminability": round(disc_8d, 4),
            "20d_discriminability": round(disc_20d, 4),
            "improvement_sil": round(sil_20d - sil_8d, 4),
            "improvement_disc": round(disc_20d - disc_8d, 4),
        },
        "dim_importance": dim_importance[:10],  # top-10
        "category_vs_attribute_variance": {
            "category_dim_mean_var": round(float(np.mean(cat_var)), 4),
            "attribute_dim_mean_var": round(float(np.mean(attr_var)), 4),
            "ratio_cat_to_attr": round(float(np.mean(cat_var) / max(np.mean(attr_var), 1e-6)), 4),
        },
        "cat_attr_profiles": cat_attr_profiles,
        "cat_top_attrs": {cat: [(a, round(v, 4)) for a, v in attrs]
                          for cat, attrs in cat_top_attrs.items()},
        "n_dims": len(dim_names),
        "n_objects": len(all_objects),
    }

    plog(f"  Category dim mean var: {results['category_vs_attribute_variance']['category_dim_mean_var']}")
    plog(f"  Attribute dim mean var: {results['category_vs_attribute_variance']['attribute_dim_mean_var']}")

    return results


# ==================== 主函数 ====================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        return

    obj_dict = ROUNDS[round_num]

    plog(f"Phase 471: Layer-Causal DCF — {model_name}, Round {round_num}")
    plog(f"Objects per category: {len(list(obj_dict.values())[0])}")
    plog(f"Core: Where are semantic constraints written?")

    # ---- 1. 加载模型 ----
    t0 = time.time()
    model, tokenizer, device = load_model_bf16(model_name)
    t_load = time.time() - t0
    plog(f"Model loaded in {t_load:.0f}s")

    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, n_layers={info.n_layers}, d_model={info.d_model}")

    # ---- 2. 运行实验 ----
    all_results = {
        "phase": 471,
        "model": model_name,
        "round": round_num,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "theory": "Layer-Causal DCF Tracing",
        "core_question": "At which layer are semantic constraints written into the residual stream?",
        "model_info": {
            "class": info.model_class,
            "n_layers": info.n_layers,
            "d_model": info.d_model,
        },
    }

    # Exp1: Logit-Lens DCF
    t1 = time.time()
    all_results["exp1_logit_lens_dcf"] = exp1_logit_lens_dcf(
        model, tokenizer, model_name, device, obj_dict, round_num)
    plog(f"Exp1 done in {time.time()-t1:.0f}s")

    # Exp2: 因果DCF干预
    t2 = time.time()
    all_results["exp2_causal_intervention"] = exp2_causal_dcf_intervention(
        model, tokenizer, model_name, device, obj_dict, round_num)
    plog(f"Exp2 done in {time.time()-t2:.0f}s")

    # Exp3: 扩展DCF维度
    t3 = time.time()
    all_results["exp3_extended_dcf"] = exp3_extended_dcf(
        model, tokenizer, model_name, device, obj_dict, round_num)
    plog(f"Exp3 done in {time.time()-t3:.0f}s")

    # ---- 3. 保存结果 ----
    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase471_{model_name}_r{round_num}.json"

    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, (int, float, str)):
            return obj
        return str(obj)

    all_results = convert(all_results)

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    plog(f"Results saved to {out_path}")

    # ---- 4. 释放模型 ----
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - t0
    plog(f"Phase 471 {model_name} Round {round_num} complete in {total_time:.0f}s")


if __name__ == "__main__":
    main()
