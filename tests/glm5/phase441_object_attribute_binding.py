"""
Phase 441: GLM4 对象-属性直接绑定验证
============================================

验证GLM4是否采用"对象→属性直接绑定"而非"类别→属性中介"。

核心逻辑:
- Qwen3: 改类别→属性跟着变(Phase 437已验证)
- GLM4假说: 改类别→属性不变, 但改对象→属性跟着变

测试1: 对象identity residual修改 → 属性变化
  方法: 在对象词元位置修改隐藏状态, 使apple→orange, 看属性是否跟着变

测试2: 同对象不同模板 → 属性方向是否稳定
  方法: apple在"An apple is..." vs "The apple has..."中提取属性方向

测试3: 同类别不同对象 → 属性方向是否不共享
  方法: apple的属性方向 vs orange的属性方向, 如果不共享则支持直接绑定
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import os
import time
import json
import numpy as np
import torch
from datetime import datetime
from model_utils import (get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)


def load_model_bf16(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"  [bf16] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"], torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True, local_files_only=True, attn_implementation="eager")
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    print(f"  [bf16] {model_name} loaded, GPU={gpu_mem:.2f}GB")
    return model, tokenizer, device


def get_cat_logit(logits, tokenizer, words):
    """获取某组词的平均logit"""
    ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in words]
    return float(np.mean([logits[i] for i in ids]))


def run_forward(model, input_ids, attention_mask, capture_layers=None):
    """前向传播，可选捕获隐藏状态"""
    layers = get_layers(model)
    captured = {}
    
    if capture_layers:
        def make_cap(li):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[li] = output[0].detach().float().cpu()
            return hook
        hooks = [layers[li].register_forward_hook(make_cap(li)) for li in capture_layers]
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, -1].float().cpu().numpy()
    
    if capture_layers:
        for h in hooks:
            h.remove()
    
    return logits, captured


def run_phase441(model_name, round_num):
    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"Phase 441: {model_name} Round {round_num}")
    print(f"{'='*60}")

    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers = info.n_layers
    d_model = info.d_model
    W_U = get_W_U(model, model_name)

    print(f"  n_layers={n_layers}, d_model={d_model}")

    # 属性词汇
    properties = {
        "color": {
            "fruit": {"red": ["red"], "green": ["green"]},
            "animal": {"brown": ["brown"], "gray": ["gray"]},
            "tool": {"silver": ["silver"], "black": ["black"]},
        },
        "taste": {
            "fruit": {"sweet": ["sweet"], "sour": ["sour"]},
        },
    }

    # ============== TEST 1: 对象identity替换 → 属性变化 ==============
    print(f"\n{'='*40}")
    print("TEST 1: Object Identity Swap → Property Change")
    print(f"{'='*40}")

    # 同类别对象对
    obj_pairs = [
        ("apple", "orange", "fruit"),
        ("knife", "hammer", "tool"),
        ("dog", "cat", "animal"),
    ]

    test1_results = {}

    for src_obj, tgt_obj, cat in obj_pairs:
        # 源对象baseline
        src_prompt = f"An {src_obj} is a kind of"
        src_inputs = tokenizer(src_prompt, return_tensors="pt", truncation=True, max_length=64)
        src_ids = src_inputs["input_ids"].to(device)
        src_mask = src_inputs["attention_mask"].to(device)
        src_tokens = tokenizer.convert_ids_to_tokens(src_ids[0].cpu().numpy())

        # 目标对象baseline
        tgt_prompt = f"An {tgt_obj} is a kind of"
        tgt_inputs = tokenizer(tgt_prompt, return_tensors="pt", truncation=True, max_length=64)
        tgt_ids = tgt_inputs["input_ids"].to(device)
        tgt_mask = tgt_inputs["attention_mask"].to(device)
        tgt_tokens = tokenizer.convert_ids_to_tokens(tgt_ids[0].cpu().numpy())

        # 获取源和目标的隐藏状态
        src_logits, src_hs = run_forward(model, src_ids, src_mask, capture_layers=[0, n_layers // 2, n_layers - 1])
        tgt_logits, tgt_hs = run_forward(model, tgt_ids, tgt_mask, capture_layers=[0, n_layers // 2, n_layers - 1])

        # 对象identity差分: 在每层每个位置计算 src - tgt
        # 关键: 找到对象词元位置
        src_obj_pos = None
        for i, t in enumerate(src_tokens):
            if src_obj[:3].lower() in t.lower():
                src_obj_pos = i
                break
        if src_obj_pos is None:
            src_obj_pos = 1

        # 计算每层对象位置的identity差分
        identity_diffs = {}
        for li in [0, n_layers // 2, n_layers - 1]:
            if li in src_hs and li in tgt_hs:
                # 对象位置的差分
                diff = tgt_hs[li][0, src_obj_pos].numpy() - src_hs[li][0, src_obj_pos].numpy()
                identity_diffs[li] = diff

        # 注入identity差分到源对象，使源对象在内部更像目标对象
        # 在每层注入，看属性是否变化
        inject_results = {}
        for inject_layer in [0, 1, 2, 3, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]:
            if inject_layer >= n_layers:
                continue

            # 在inject_layer注入identity差分
            diff_tensor = torch.tensor(identity_diffs.get(0, np.zeros(d_model)),
                                        dtype=torch.bfloat16, device=device)

            def make_inject_hook(dl, pos, dt):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        out = output[0].clone()
                        out[0, pos] = out[0, pos] + 1.0 * dt.to(out.device, out.dtype)
                        return (out,) + output[1:]
                    return output
                return hook

            inject_hook = layers[inject_layer].register_forward_hook(
                make_inject_hook(inject_layer, src_obj_pos, diff_tensor))

            with torch.no_grad():
                inject_out = model(input_ids=src_ids, attention_mask=src_mask)
                inject_logits = inject_out.logits[0, -1].float().cpu().numpy()

            inject_hook.remove()

            # 测量属性变化
            # fruit: color(red vs green), taste(sweet vs sour)
            if cat == "fruit":
                red_logit = get_cat_logit(inject_logits, tokenizer, ["red"])
                green_logit = get_cat_logit(inject_logits, tokenizer, ["green"])
                sweet_logit = get_cat_logit(inject_logits, tokenizer, ["sweet"])
                sour_logit = get_cat_logit(inject_logits, tokenizer, ["sour"])
                color_shift = red_logit - green_logit
                taste_shift = sweet_logit - sour_logit

                # baseline
                base_red = get_cat_logit(src_logits, tokenizer, ["red"])
                base_green = get_cat_logit(src_logits, tokenizer, ["green"])
                base_sweet = get_cat_logit(src_logits, tokenizer, ["sweet"])
                base_sour = get_cat_logit(src_logits, tokenizer, ["sour"])
                base_color = base_red - base_green
                base_taste = base_sweet - base_sour

                color_delta = color_shift - base_color
                taste_delta = taste_shift - base_taste
            elif cat == "tool":
                metal_logit = get_cat_logit(inject_logits, tokenizer, ["metal"])
                wood_logit = get_cat_logit(inject_logits, tokenizer, ["wood"])
                color_shift = metal_logit - wood_logit
                base_metal = get_cat_logit(src_logits, tokenizer, ["metal"])
                base_wood = get_cat_logit(src_logits, tokenizer, ["wood"])
                color_delta = color_shift - (base_metal - base_wood)
                taste_delta = 0
            else:  # animal
                fur_logit = get_cat_logit(inject_logits, tokenizer, ["fur"])
                feather_logit = get_cat_logit(inject_logits, tokenizer, ["feather"])
                color_shift = fur_logit - feather_logit
                base_fur = get_cat_logit(src_logits, tokenizer, ["fur"])
                base_feather = get_cat_logit(src_logits, tokenizer, ["feather"])
                color_delta = color_shift - (base_fur - base_feather)
                taste_delta = 0

            # 类别logit
            fruit_logit = get_cat_logit(inject_logits, tokenizer, ["fruit"])
            animal_logit = get_cat_logit(inject_logits, tokenizer, ["animal"])
            tool_logit = get_cat_logit(inject_logits, tokenizer, ["tool"])
            vehicle_logit = get_cat_logit(inject_logits, tokenizer, ["vehicle"])

            inject_results[f"L{inject_layer}"] = {
                "color_delta": round(float(color_delta), 4),
                "taste_delta": round(float(taste_delta), 4),
                "fruit_logit": round(float(fruit_logit), 4),
                "animal_logit": round(float(animal_logit), 4),
                "tool_logit": round(float(tool_logit), 4),
                "vehicle_logit": round(float(vehicle_logit), 4),
            }

            print(f"  {src_obj}→{tgt_obj} L{inject_layer}: color_delta={color_delta:.4f}, "
                  f"taste_delta={taste_delta:.4f}")

        test1_results[f"{src_obj}→{tgt_obj}"] = {
            "category": cat,
            "inject_results": inject_results,
        }

    # ============== TEST 2: 同对象不同模板的属性方向稳定性 ==============
    print(f"\n{'='*40}")
    print("TEST 2: Same Object, Different Templates → Attribute Direction Stability")
    print(f"{'='*40}")

    # 两种模板提取颜色属性方向
    templates = {
        "is_a": "An {obj} is a kind of",
        "has": "A {obj} has",
        "is_color": "The color of a {obj} is",
    }

    test_objects = ["apple", "knife", "dog"]
    test2_results = {}

    for obj in test_objects:
        color_dirs = {}
        for tmpl_name, tmpl in templates.items():
            # red vs green context
            red_prompt = tmpl.replace("{obj}", obj) if "{obj}" in tmpl else f"The {obj} is red"
            green_prompt = tmpl.replace("{obj}", obj) if "{obj}" in tmpl else f"The {obj} is green"

            # For "is_color" template, we can't meaningfully get color direction
            # Instead use standard red vs green completion
            if tmpl_name == "is_color":
                red_prompt = f"The color of a {obj} is red"
                green_prompt = f"The color of a {obj} is green"

            red_inputs = tokenizer(red_prompt, return_tensors="pt", truncation=True, max_length=64)
            green_inputs = tokenizer(green_prompt, return_tensors="pt", truncation=True, max_length=64)

            red_ids = red_inputs["input_ids"].to(device)
            red_mask = red_inputs["attention_mask"].to(device)
            green_ids = green_inputs["input_ids"].to(device)
            green_mask = green_inputs["attention_mask"].to(device)

            _, red_hs = run_forward(model, red_ids, red_mask,
                                    capture_layers=[0, n_layers // 2, n_layers - 1])
            _, green_hs = run_forward(model, green_ids, green_mask,
                                       capture_layers=[0, n_layers // 2, n_layers - 1])

            # 最后词元位置的差分
            last_red = len(tokenizer.convert_ids_to_tokens(red_ids[0].cpu().numpy())) - 1
            last_green = len(tokenizer.convert_ids_to_tokens(green_ids[0].cpu().numpy())) - 1

            for li in [n_layers - 1]:
                if li in red_hs and li in green_hs:
                    diff = red_hs[li][0, last_red].numpy() - green_hs[li][0, last_green].numpy()
                    color_dirs[f"{tmpl_name}_L{li}"] = diff

        # 计算不同模板间颜色方向的余弦相似度
        cos_sims = {}
        dir_keys = list(color_dirs.keys())
        for i in range(len(dir_keys)):
            for j in range(i + 1, len(dir_keys)):
                k1, k2 = dir_keys[i], dir_keys[j]
                d1, d2 = color_dirs[k1], color_dirs[k2]
                n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                if n1 > 1e-6 and n2 > 1e-6:
                    cos_sim = float(np.dot(d1, d2) / (n1 * n2))
                else:
                    cos_sim = 0.0
                cos_sims[f"{k1}_vs_{k2}"] = round(cos_sim, 4)

        test2_results[obj] = {
            "cos_sims_across_templates": cos_sims,
        }
        print(f"  {obj}: cross-template cos_sims = {cos_sims}")

    # ============== TEST 3: 同类别不同对象的属性方向共享性 ==============
    print(f"\n{'='*40}")
    print("TEST 3: Same Category, Different Objects → Attribute Direction Sharing")
    print(f"{'='*40}")

    # 同类别对象对的属性方向比较
    cat_obj_pairs = {
        "fruit": ["apple", "orange", "lemon", "grape"],
        "tool": ["knife", "hammer", "spoon", "axe"],
    }

    test3_results = {}

    for cat, objs in cat_obj_pairs.items():
        # 每个对象的颜色属性方向(red vs green上下文)
        obj_color_dirs = {}
        for obj in objs:
            # 使用is_color模板
            red_prompt = f"The color of a {obj} is red"
            green_prompt = f"The color of a {obj} is green"

            red_inputs = tokenizer(red_prompt, return_tensors="pt", truncation=True, max_length=64)
            green_inputs = tokenizer(green_prompt, return_tensors="pt", truncation=True, max_length=64)

            red_ids = red_inputs["input_ids"].to(device)
            green_ids = green_inputs["input_ids"].to(device)
            red_mask = red_inputs["attention_mask"].to(device)
            green_mask = green_inputs["attention_mask"].to(device)

            _, red_hs = run_forward(model, red_ids, red_mask,
                                    capture_layers=[n_layers - 1])
            _, green_hs = run_forward(model, green_ids, green_mask,
                                       capture_layers=[n_layers - 1])

            if n_layers - 1 in red_hs and n_layers - 1 in green_hs:
                last_r = len(tokenizer.convert_ids_to_tokens(red_ids[0].cpu().numpy())) - 1
                last_g = len(tokenizer.convert_ids_to_tokens(green_ids[0].cpu().numpy())) - 1
                diff = red_hs[n_layers - 1][0, last_r].numpy() - green_hs[n_layers - 1][0, last_g].numpy()
                obj_color_dirs[obj] = diff

        # 计算对象间的颜色方向余弦相似度
        obj_names = list(obj_color_dirs.keys())
        pairwise_cos = {}
        for i in range(len(obj_names)):
            for j in range(i + 1, len(obj_names)):
                o1, o2 = obj_names[i], obj_names[j]
                d1, d2 = obj_color_dirs[o1], obj_color_dirs[o2]
                n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
                if n1 > 1e-6 and n2 > 1e-6:
                    cos = float(np.dot(d1, d2) / (n1 * n2))
                else:
                    cos = 0.0
                pairwise_cos[f"{o1}_vs_{o2}"] = round(cos, 4)

        avg_cos = np.mean(list(pairwise_cos.values())) if pairwise_cos else 0.0

        test3_results[cat] = {
            "pairwise_cos": pairwise_cos,
            "avg_cos": round(avg_cos, 4),
        }
        print(f"  {cat}: avg_cross_obj_cos={avg_cos:.4f}, pairwise={pairwise_cos}")

    # Save results
    results = {
        "model": model_name, "round": round_num,
        "n_layers": n_layers, "d_model": d_model,
        "timestamp": datetime.now().isoformat(),
        "test1_identity_swap": test1_results,
        "test2_template_stability": test2_results,
        "test3_cross_object_sharing": test3_results,
    }

    out_dir = "d:/Ai2050/TransformerLens-Project/results/phase441_object_attribute_binding"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/{model_name}_phase441_r{round_num}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved to {out_path}")

    # Summary
    print(f"\n{'='*40}")
    print(f"SUMMARY: {model_name}")
    print(f"{'='*40}")
    print("TEST 1 - Identity Swap → Property Change:")
    for pair_key, pair_r in test1_results.items():
        max_color = max(abs(r["color_delta"]) for r in pair_r["inject_results"].values())
        max_taste = max(abs(r.get("taste_delta", 0)) for r in pair_r["inject_results"].values())
        print(f"  {pair_key}: max|color_delta|={max_color:.4f}, max|taste_delta|={max_taste:.4f}")

    print("\nTEST 2 - Cross-Template Attribute Stability:")
    for obj, r in test2_results.items():
        cos_sims = r.get("cos_sims_across_templates", {})
        avg = np.mean(list(cos_sims.values())) if cos_sims else 0
        print(f"  {obj}: avg_cross_template_cos={avg:.4f}")

    print("\nTEST 3 - Cross-Object Attribute Sharing:")
    for cat, r in test3_results.items():
        print(f"  {cat}: avg_cross_obj_cos={r['avg_cos']:.4f}")

    release_model(model)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    round_num = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run_phase441(model_name, round_num)
