"""
Phase 496 R2: 确认测试 — MLP的shared vs 非shared方向贡献分解
================================================================

Phase 496 R1的关键发现:
- MLP在L(n-1)对所有类别都提供正向D贡献（ΔD(zeroMLP) < 0）
- 但Phase 495发现释放类的MLP翻转shared方向，刹车类不翻转
- 这两个发现表面矛盾: 如果emotion的MLP不翻转shared方向，为什么零化MLP后D也大幅下降？

解释假设: MLP的D贡献有两个来源:
1. shared方向的贡献（释放类=正，刹车类=负）
2. 非shared方向的贡献（所有类别=正，且可能更大）

本实验验证:
- 在L(n-1)分别消融MLP的shared分量和MLP的非shared分量
- 看各自的D贡献
- 如果非shared分量贡献远大于shared分量，就能解释"表面矛盾"

用法:
  python tests/glm5/phase496_mlp_shared_vs_nonshared.py qwen3 2
  python tests/glm5/phase496_mlp_shared_vs_nonshared.py glm4 2
  python tests/glm5/phase496_mlp_shared_vs_nonshared.py deepseek7b 2
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')
import os, gc, time, json
import numpy as np
import torch
from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS)


def plog(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


CATEGORIES_16 = {
    "fruit":     ["apple", "banana", "orange", "grape", "pear", "peach", "mango", "plum",
                  "cherry", "lemon", "lime", "kiwi", "melon", "fig", "date", "guava"],
    "animal":    ["dog", "cat", "horse", "lion", "bear", "rabbit", "eagle", "fish",
                  "tiger", "deer", "wolf", "fox", "owl", "hawk", "crab", "frog"],
    "clothing":  ["shirt", "dress", "hat", "coat", "jacket", "skirt", "scarf", "boot",
                  "pants", "vest", "glove", "sock", "belt", "tie", "cap", "robe"],
    "food":      ["bread", "rice", "cheese", "pasta", "soup", "cake", "salad", "meat",
                  "pizza", "taco", "stew", "pie", "roll", "ham", "corn", "bean"],
    "vehicle":   ["car", "bus", "bicycle", "truck", "train", "plane", "boat", "motorcycle",
                  "van", "scooter", "taxi", "tram", "ship", "jet", "subway", "sled"],
    "plant":     ["tree", "flower", "grass", "bush", "fern", "moss", "vine", "shrub",
                  "palm", "oak", "pine", "rose", "lily", "weed", "reed", "cactus"],
    "tool":      ["hammer", "saw", "drill", "wrench", "pliers", "chisel", "ruler", "knife",
                  "screwdriver", "axe", "shovel", "mallet", "clamp", "vise", "file", "level"],
    "furniture": ["chair", "table", "desk", "bed", "sofa", "shelf", "cabinet", "bench",
                  "stool", "dresser", "wardrobe", "couch", "hutch", "ottoman", "cot", "armoire"],
    "body_part":   ["hand", "foot", "head", "arm", "leg", "eye", "ear", "nose",
                    "mouth", "finger", "toe", "knee", "elbow", "wrist", "ankle", "hip"],
    "building":    ["house", "church", "school", "store", "factory", "tower", "bridge", "castle",
                    "barn", "temple", "museum", "hotel", "cabin", "palace", "fort", "shrine"],
    "container":   ["box", "bottle", "cup", "jar", "bowl", "basket", "bucket", "pot",
                    "can", "tub", "mug", "flask", "vase", "crate", "barrel", "jug"],
    "device":      ["phone", "radio", "clock", "lamp", "camera", "speaker", "monitor", "printer",
                    "fan", "heater", "scale", "timer", "scanner", "router", "switch", "pager"],
    "place":       ["park", "beach", "forest", "lake", "mountain", "river", "desert", "island",
                    "valley", "cliff", "cave", "swamp", "meadow", "canyon", "bay", "glacier"],
    "material":    ["wood", "stone", "metal", "glass", "paper", "cloth", "rubber", "leather",
                    "plastic", "cotton", "silk", "wool", "clay", "sand", "gold", "iron"],
    "emotion":     ["joy", "fear", "anger", "sadness", "hope", "love", "pride", "shame",
                    "guilt", "envy", "pity", "awe", "grief", "bliss", "rage", "dread"],
    "action":      ["run", "walk", "jump", "swim", "climb", "throw", "catch", "push",
                    "pull", "lift", "carry", "drag", "drop", "kick", "hit", "hold"],
}

CAT_NAMES_16 = list(CATEGORIES_16.keys())


def get_model_and_tokenizer(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    plog(f"Loading {model_name}...")
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


def encode_prompts(tokenizer, device, objects, template="The {obj} is a kind of"):
    texts = [template.format(obj=obj) for obj in objects]
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=64)
    return {"input_ids": enc["input_ids"].to(device), "attention_mask": enc["attention_mask"].to(device)}


def compute_D(h, W_U, cat_name, cat_names):
    """计算DCF: target logit - mean(other category logits)"""
    logit = W_U @ h
    target = logit[cat_names.index(cat_name)]
    others = [logit[i] for i in range(len(cat_names)) if i != cat_names.index(cat_name)]
    return float(target - np.mean(others))


def get_shared_direction(h_target, h_others):
    others_mean = np.mean(h_others, axis=0)
    B_c = h_target - others_mean
    Bc_norm_dir = B_c / (np.linalg.norm(B_c) + 1e-10)
    all_mean = np.mean(np.vstack([h_target[np.newaxis], h_others]), axis=0)
    shared_dir = all_mean / (np.linalg.norm(all_mean) + 1e-10)
    shared_dir = shared_dir - np.dot(shared_dir, Bc_norm_dir) * Bc_norm_dir
    if np.linalg.norm(shared_dir) > 1e-6:
        shared_dir = shared_dir / np.linalg.norm(shared_dir)
    return shared_dir, Bc_norm_dir


def get_layer_hidden(model, tokenizer, device, objects, layer_idx,
                     template="The {obj} is a kind of"):
    layers = get_layers(model)
    inputs = encode_prompts(tokenizer, device, objects, template)
    captured = {}
    def make_hook(key):
        def hook(mod, inp, out):
            if isinstance(out, tuple):
                captured[key] = out[0].detach().float().cpu()
            else:
                captured[key] = out.detach().float().cpu()
        return hook
    hook = layers[layer_idx].register_forward_hook(make_hook("h"))
    with torch.no_grad():
        model(**inputs)
    hook.remove()
    h = captured["h"]
    mask = inputs["attention_mask"].bool()
    last_idx = mask.sum(dim=1) - 1
    h_last = h[torch.arange(h.size(0)), last_idx.cpu()]
    return h_last.numpy(), inputs, last_idx


def exp_mlp_shared_vs_nonshared(model, tokenizer, device, model_name, W_U):
    """
    核心实验: 分解MLP在L(n-1)的D贡献为shared分量和非shared分量
    
    方法:
    1. 获取L(n-1)的shared方向
    2. 捕获MLP output
    3. 将MLP output分解为: shared分量 + 非shared分量
    4. 分别计算两个分量对D的贡献
    5. 同时做真正因果干预: 只消融MLP的shared分量(保留非shared) vs 只保留MLP的shared分量(消融非shared)
    """
    plog(f"=== MLP shared vs non-shared decomposition at L(n-1) ===")
    n_layers = get_model_info(model, model_name).n_layers
    L_n1 = n_layers - 1
    layers = get_layers(model)
    layer_n1 = layers[L_n1]
    mlp_mod = getattr(layer_n1, 'mlp', None) or getattr(layer_n1, 'feed_forward', None)

    test_cats = ["fruit", "clothing", "emotion", "action", "animal"]
    results = {}

    for cat_name in test_cats:
        plog(f"  Cat: {cat_name}")
        t0 = time.time()
        cat_objs = CATEGORIES_16[cat_name][:10]
        other_cats = [c for c in CAT_NAMES_16 if c != cat_name][:7]

        # 获取L(n-1)的shared方向
        h_n1_all, _, _ = get_layer_hidden(model, tokenizer, device, cat_objs, L_n1)
        h_n1_mean = np.mean(h_n1_all, axis=0)
        other_h_n1 = {}
        for oc in other_cats:
            oc_h, _, _ = get_layer_hidden(model, tokenizer, device, CATEGORIES_16[oc][:8], L_n1)
            other_h_n1[oc] = oc_h
        h_others_n1 = np.array([np.mean(other_h_n1[oc], axis=0) for oc in other_cats])
        shared_dir_n1, Bc_dir_n1 = get_shared_direction(h_n1_mean, h_others_n1)

        sample_data = []
        for s_idx, obj in enumerate(cat_objs):
            inputs_single = encode_prompts(tokenizer, device, [obj])
            mask_single = inputs_single["attention_mask"].bool()
            sp = int(mask_single[0].sum() - 1)

            # --- 步骤1: 捕获MLP output和L(n-1) output ---
            cap = {}
            def make_cap(key, store):
                def hook(mod, inp, out):
                    if isinstance(out, tuple):
                        store[key] = out[0].detach().float().cpu()
                    else:
                        store[key] = out.detach().float().cpu()
                return hook

            hooks = [
                mlp_mod.register_forward_hook(make_cap("mlp_out", cap)),
                layer_n1.register_forward_hook(make_cap("layer_out", cap)),
            ]
            with torch.no_grad():
                model(**inputs_single)
            for h in hooks:
                h.remove()

            if "mlp_out" not in cap or "layer_out" not in cap:
                continue

            mlp_out_vec = cap["mlp_out"][0, sp, :].numpy()
            layer_out_vec = cap["layer_out"][0, sp, :].numpy()

            # --- 步骤2: 分解MLP output ---
            proj_shared_mlp = np.dot(mlp_out_vec, shared_dir_n1)  # shared投影
            shared_component = proj_shared_mlp * shared_dir_n1     # shared分量
            nonshared_component = mlp_out_vec - shared_component   # 非shared分量

            # --- 步骤3: 计算各分量对D的贡献 ---
            # D = target_logit - mean(other_logits)
            # 某分量v对D的贡献 = (W_U @ v)[target] - mean((W_U @ v)[others])
            target_idx = CAT_NAMES_16.index(cat_name)
            other_idxs = [i for i in range(len(CAT_NAMES_16)) if i != target_idx]

            logit_mlp = W_U @ mlp_out_vec
            logit_shared = W_U @ shared_component
            logit_nonshared = W_U @ nonshared_component

            D_mlp = float(logit_mlp[target_idx] - np.mean([logit_mlp[i] for i in other_idxs]))
            D_shared = float(logit_shared[target_idx] - np.mean([logit_shared[i] for i in other_idxs]))
            D_nonshared = float(logit_nonshared[target_idx] - np.mean([logit_nonshared[i] for i in other_idxs]))

            # shared方向对目标logit的贡献
            shared_target_logit = float(logit_shared[target_idx])
            nonshared_target_logit = float(logit_nonshared[target_idx])

            # Bc方向的分析
            proj_Bc_mlp = np.dot(mlp_out_vec, Bc_dir_n1)
            Bc_component = proj_Bc_mlp * Bc_dir_n1
            logit_Bc = W_U @ Bc_component
            D_Bc = float(logit_Bc[target_idx] - np.mean([logit_Bc[i] for i in other_idxs]))

            sample_data.append({
                "obj": obj,
                "D_mlp_total": round(D_mlp, 4),
                "D_shared_component": round(D_shared, 4),
                "D_nonshared_component": round(D_nonshared, 4),
                "D_Bc_component": round(D_Bc, 4),
                "shared_target_logit": round(shared_target_logit, 4),
                "nonshared_target_logit": round(nonshared_target_logit, 4),
                "proj_shared_mlp": round(float(proj_shared_mlp), 2),
                "proj_Bc_mlp": round(float(proj_Bc_mlp), 2),
                "mlp_norm": round(float(np.linalg.norm(mlp_out_vec)), 2),
                "shared_norm": round(float(np.linalg.norm(shared_component)), 2),
                "nonshared_norm": round(float(np.linalg.norm(nonshared_component)), 2),
            })

        if not sample_data:
            results[cat_name] = {"error": "no samples"}
            continue

        # 汇总
        mean_D_shared = np.mean([s["D_shared_component"] for s in sample_data])
        mean_D_nonshared = np.mean([s["D_nonshared_component"] for s in sample_data])
        mean_D_mlp = np.mean([s["D_mlp_total"] for s in sample_data])
        mean_D_Bc = np.mean([s["D_Bc_component"] for s in sample_data])

        results[cat_name] = {
            "n_samples": len(sample_data),
            "mean_D_mlp_total": round(float(mean_D_mlp), 4),
            "mean_D_shared": round(float(mean_D_shared), 4),
            "mean_D_nonshared": round(float(mean_D_nonshared), 4),
            "mean_D_Bc": round(float(mean_D_Bc), 4),
            "shared_nonshared_ratio": round(float(mean_D_shared / (abs(mean_D_nonshared) + 1e-10)), 4),
            "shared_dominant": "shared" if abs(mean_D_shared) > abs(mean_D_nonshared) else "nonshared",
            "sample_details": sample_data[:3],  # 只存3个样本
            "elapsed": round(time.time() - t0, 2),
        }
        plog(f"  {cat_name}: D_mlp={mean_D_mlp:.2f}, D_shared={mean_D_shared:.2f}, "
             f"D_nonshared={mean_D_nonshared:.2f}, D_Bc={mean_D_Bc:.2f}, "
             f"dominant={results[cat_name]['shared_dominant']}")

    return results


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    plog(f"Phase 496 R{round_num}: {model_name}")
    model, tokenizer, device = get_model_and_tokenizer(model_name)
    info = get_model_info(model, model_name)
    plog(f"Model: {info.model_class}, {info.n_layers} layers, d_model={info.d_model}")

    W_U = get_W_U(model, model_name)

    results = {
        "phase": 496,
        "round": round_num,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_info": {"class": info.model_class, "n_layers": info.n_layers, "d_model": info.d_model},
        "exp_mlp_shared_vs_nonshared": exp_mlp_shared_vs_nonshared(
            model, tokenizer, device, model_name, W_U
        ),
    }

    os.makedirs("results/glm5", exist_ok=True)
    out_path = f"results/glm5/phase496_{model_name}_r{round_num}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    plog(f"Results saved to {out_path}")

    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    plog("Done!")


if __name__ == "__main__":
    main()
