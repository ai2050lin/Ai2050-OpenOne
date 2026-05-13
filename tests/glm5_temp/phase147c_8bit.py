"""Phase 147c: Rollout + Exp2+3 for 8bit models (GLM4/DS7B)
============================================================
修复: 8bit模型不加载W_U, Rollout使用随机扰动(不是null-space方向)
      Exp2: 先释放训练模型再创建随机模型
      Exp3: 非线性交互测试

用法:
  python tests/glm5_temp/phase147c_8bit.py glm4
  python tests/glm5_temp/phase147c_8bit.py deepseek7b
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import numpy as np
import torch
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)

TEST_PROMPTS = [
    "The scientist discovered that the",
    "In the morning, she decided to",
    "The book on the table was about",
    "After the rain stopped, the children",
    "The most important thing about science is",
    "When the sun sets over the ocean,",
    "The relationship between language and thought is",
    "A fundamental principle of mathematics states that",
    "The history of civilization shows that humans",
    "In order to understand consciousness, we must",
    "The economic crisis led to widespread",
    "She walked through the garden and noticed",
    "The algorithm processes data by first",
    "Between the two options, the better choice is",
    "The transformation of energy in this system follows",
]

N_ROLLOUT_TOKENS = 100
INJECT_LAYERS_FRAC = [0.0, 0.25, 0.5, 0.75, 1.0]
EPSILONS = [0.5, 2.0, 5.0]


def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Exp 1: Rollout稳定性 (不依赖W_U)
# ============================================================
def exp1_rollout(model, tokenizer, device, model_info):
    """在指定层注入随机扰动, 自回归生成100 token"""
    print("\n" + "="*60)
    print("Exp 1: Rollout稳定性")
    print("="*60)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    results = {}

    inject_layers = [int(f * (n_layers - 1)) for f in INJECT_LAYERS_FRAC]
    layers = get_layers(model)

    for si in range(min(15, len(TEST_PROMPTS))):
        prompt = TEST_PROMPTS[si]
        print(f"\n  --- Sent {si}: '{prompt[:40]}...' ---")

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)

        # Clean生成
        with torch.no_grad():
            clean_gen = model.generate(
                input_ids, attention_mask=attn_mask,
                max_new_tokens=N_ROLLOUT_TOKENS, do_sample=False,
                repetition_penalty=1.2,
            )
        clean_toks = clean_gen[0][input_ids.shape[1]:].cpu().numpy()
        clean_text = tokenizer.decode(clean_gen[0], skip_special_tokens=True)

        for inj_l in inject_layers:
            for eps in EPSILONS:
                key = f"sent{si}_L{inj_l}_eps{eps:.1f}"
                print(f"    {key}...", end=" ", flush=True)

                try:
                    # 随机扰动方向
                    rand_dir = np.random.randn(d_model).astype(np.float32)
                    rand_dir = rand_dir / np.linalg.norm(rand_dir)
                    delta_np = (eps * rand_dir).astype(np.float32)

                    def inject_hook(mod, inp, out):
                        h = out[0] if isinstance(out, tuple) else out
                        dt = torch.tensor(delta_np, dtype=h.dtype, device=h.device)
                        h_new = h.clone()
                        h_new[0, -1, :] += dt
                        return (h_new,) + out[1:] if isinstance(out, tuple) else h_new

                    h_handle = layers[inj_l].register_forward_hook(inject_hook)

                    try:
                        with torch.no_grad():
                            pert_gen = model.generate(
                                input_ids, attention_mask=attn_mask,
                                max_new_tokens=N_ROLLOUT_TOKENS, do_sample=False,
                                repetition_penalty=1.2,
                            )
                    finally:
                        h_handle.remove()

                    pert_toks = pert_gen[0][input_ids.shape[1]:].cpu().numpy()
                    pert_text = tokenizer.decode(pert_gen[0], skip_special_tokens=True)

                    # Metrics
                    n_gen = min(len(clean_toks), len(pert_toks))
                    overlaps = {}
                    for n in [10, 50, 100]:
                        if n <= n_gen:
                            overlaps[f"overlap_{n}"] = float(np.mean(clean_toks[:n] == pert_toks[:n]))
                        else:
                            overlaps[f"overlap_{n}"] = float(np.mean(clean_toks[:n_gen] == pert_toks[:n_gen]))

                    first_div = n_gen
                    for i in range(n_gen):
                        if clean_toks[i] != pert_toks[i]:
                            first_div = i
                            break

                    # Semantic divergence
                    ids1 = tokenizer.encode(clean_text, add_special_tokens=False)
                    ids2 = tokenizer.encode(pert_text, add_special_tokens=False)
                    if len(ids1) >= 2 and len(ids2) >= 2:
                        bg1 = set(tuple(ids1[i:i+2]) for i in range(len(ids1)-1))
                        bg2 = set(tuple(ids2[i:i+2]) for i in range(len(ids2)-1))
                        sem_div = 1.0 - len(bg1 & bg2) / max(len(bg1 | bg2), 1)
                    else:
                        sem_div = 1.0

                    r = {
                        "inject_layer": inj_l, "eps": eps,
                        "n_gen": int(n_gen), "first_div_pos": int(first_div),
                        **overlaps, "semantic_div": float(sem_div),
                        "clean_text": clean_text[:120], "pert_text": pert_text[:120],
                    }
                    results[key] = r
                    print(f"overlap@50={r['overlap_50']:.2f}, "
                          f"first_div={r['first_div_pos']}, sem_div={sem_div:.3f}")

                except Exception as e:
                    print(f"ERR: {e}")
                    results[key] = {"error": str(e)}

                torch.cuda.empty_cache()

    # 汇总
    print(f"\n  === Rollout汇总 ===")
    for inj_l in inject_layers:
        for eps in EPSILONS:
            keys = [k for k in results
                    if f"_L{inj_l}_" in k and f"_eps{eps:.1f}" in k
                    and "error" not in results[k]]
            if not keys:
                continue
            vals = [results[k] for k in keys]
            o10 = np.mean([v["overlap_10"] for v in vals])
            o50 = np.mean([v["overlap_50"] for v in vals])
            o100 = np.mean([v["overlap_100"] for v in vals])
            fd = np.mean([v["first_div_pos"] for v in vals])
            sd = np.mean([v["semantic_div"] for v in vals])
            print(f"    L{inj_l} eps={eps:.1f}: overlap=[{o10:.2f},{o50:.2f},{o100:.2f}], "
                  f"first_div={fd:.0f}, sem_div={sd:.3f}")

    return results


# ============================================================
# Exp 2: 随机权重对照
# ============================================================
def exp2_random_control(model_name):
    """先释放训练模型, 再加载+随机化, 测量传播性质"""
    print("\n" + "="*60)
    print("Exp 2: 随机权重对照")
    print("="*60)

    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16

    # Step 1: 训练模型
    print("\n  [Step 1] 加载训练模型...")
    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    model_info = get_model_info(model, model_name)

    print("  [Step 1] 测量传播性质...")
    trained = _measure_prop(model, tokenizer, device, model_info.n_layers, model_info.d_model, 6)

    print("  [Step 1] 释放...")
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    import time; time.sleep(3)

    # Step 2: 随机模型
    print("\n  [Step 2] 创建随机模型...")
    try:
        model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
        def init_w(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None: torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.ones_(m.weight); torch.nn.init.zeros_(m.bias)
        model.apply(init_w)
        model.eval()
        print("  [Step 2] 随机模型OK")

        random = _measure_prop(model, tokenizer, device, model_info.n_layers, model_info.d_model, 6)
        release_model(model)
    except Exception as e:
        print(f"  [Step 2] 失败: {e}")
        random = {"error": str(e)}
        if model: release_model(model)

    # 对比
    print("\n  === 训练 vs 随机 ===")
    comparison = {}
    for mk in ["perturbation_growth", "direction_cosines"]:
        t, r = trained.get(mk, {}), random.get(mk, {})
        for k in t:
            if k in r and "error" not in r:
                tv, rv = np.mean(t[k]), np.mean(r[k])
                comparison[k] = {"trained": float(tv), "random": float(rv), "diff": float(tv-rv)}
                print(f"    {k}: trained={tv:.4f}, random={rv:.4f}, diff={tv-rv:.4f}")

    return {"trained": trained, "random": random, "comparison": comparison}


def _measure_prop(model, tokenizer, device, n_layers, d_model, n_sents):
    """测量传播性质"""
    stats = {"perturbation_growth": {}, "direction_cosines": {}}
    sample_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]))
    layers = get_layers(model)
    mdev = get_input_device(model)
    eps = 1.0

    for si in range(min(n_sents, len(TEST_PROMPTS))):
        inputs = tokenizer(TEST_PROMPTS[si], return_tensors="pt", truncation=True, max_length=64)
        ids = inputs["input_ids"].to(mdev)
        mask = inputs["attention_mask"].to(mdev)

        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hs = [h[0, -1, :].detach().cpu().float().numpy() for h in out.hidden_states]
        del out; torch.cuda.empty_cache()

        for il in sample_layers:
            v = np.random.randn(d_model).astype(np.float32)
            v /= np.linalg.norm(v)
            delta = (eps * v).astype(np.float32)

            captured = {}
            def mk_cap(li):
                def hook(m, i, o):
                    h = o[0] if isinstance(o, tuple) else o
                    captured[li] = h[0, -1, :].detach().cpu().float().numpy()
                return hook

            def inj(m, i, o):
                h = o[0] if isinstance(o, tuple) else o
                dt = torch.tensor(delta, dtype=h.dtype, device=h.device)
                hn = h.clone(); hn[0, -1, :] += dt
                return (hn,) + o[1:] if isinstance(o, tuple) else hn

            hooks = [layers[il].register_forward_hook(inj)]
            for li in sample_layers:
                if li > il: hooks.append(layers[li].register_forward_hook(mk_cap(li)))

            try:
                with torch.no_grad():
                    model(input_ids=ids, attention_mask=mask, output_hidden_states=False)
            except: pass
            finally:
                for h in hooks: h.remove()

            for li in captured:
                hp = captured[li]
                hc = hs[li+1]
                d = hp - hc
                g = np.linalg.norm(d) / eps
                c = np.dot(hc, hp) / (np.linalg.norm(hc) * np.linalg.norm(hp) + 1e-10)
                kg = f"L{il}_to_L{li}"
                for dd, kk in [(g, "perturbation_growth"), (c, "direction_cosines")]:
                    if kg not in stats[kk]: stats[kk][kg] = []
                    stats[kk][kg].append(float(dd))

            torch.cuda.empty_cache()
    return stats


# ============================================================
# Exp 3: 非线性交互
# ============================================================
def exp3_nonlinearity(model_name):
    print("\n" + "="*60)
    print("Exp 3: 非线性交互")
    print("="*60)

    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16

    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    mdev = get_input_device(model)

    pairs = [(0, info.n_layers//4), (0, info.n_layers//2),
             (info.n_layers//4, info.n_layers//2),
             (info.n_layers//4, 3*info.n_layers//4),
             (info.n_layers//2, 3*info.n_layers//4)]
    results = {}

    for si in range(min(8, len(TEST_PROMPTS))):
        inputs = tokenizer(TEST_PROMPTS[si], return_tensors="pt", truncation=True, max_length=64)
        ids = inputs["input_ids"].to(mdev)
        mask = inputs["attention_mask"].to(mdev)
        print(f"\n  Sent {si}")

        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        hc = out.hidden_states[-1][0, -1, :].detach().cpu().float().numpy()
        del out; torch.cuda.empty_cache()

        for l1, l2 in pairs:
            v1 = np.random.randn(info.d_model).astype(np.float32)
            v1 = v1 / np.linalg.norm(v1)
            v2 = np.random.randn(info.d_model).astype(np.float32)
            v2 = v2 / np.linalg.norm(v2)

            def inj_get(specs):
                cap = {}
                def mk_inj(d):
                    def h(m, i, o):
                        ho = o[0] if isinstance(o, tuple) else o
                        dt = torch.tensor(d, dtype=ho.dtype, device=ho.device)
                        hn = ho.clone(); hn[0, -1, :] += dt
                        return (hn,) + o[1:] if isinstance(o, tuple) else hn
                    return h
                def capf(m, i, o):
                    ho = o[0] if isinstance(o, tuple) else o
                    cap['f'] = ho[0, -1, :].detach().cpu().float().numpy()

                hh = [layers[li].register_forward_hook(mk_inj(d)) for li, d in specs]
                hh.append(layers[-1].register_forward_hook(capf))
                try:
                    with torch.no_grad():
                        model(input_ids=ids, attention_mask=mask, output_hidden_states=False)
                except: pass
                finally:
                    for h in hh: h.remove()
                return cap.get('f')

            h1 = inj_get([(l1, v1)])
            h2 = inj_get([(l2, v2)])
            hb = inj_get([(l1, v1), (l2, v2)])

            if h1 is None or h2 is None or hb is None:
                continue

            dl1, dl2, db = h1-hc, h2-hc, hb-hc
            dlin = dl1 + dl2
            dnl = db - dlin
            nl_n = np.linalg.norm(dnl)
            lin_n = np.linalg.norm(dlin)
            both_n = np.linalg.norm(db)

            cos_bl = np.dot(db, dlin) / (both_n * lin_n + 1e-10) if both_n > 1e-10 else 1.0
            rnl = nl_n / max(lin_n, 1e-10)

            key = f"sent{si}_L{l1}L{l2}"
            results[key] = {"cos_both_linear": float(cos_bl), "relative_nonlinearity": float(rnl)}
            print(f"    L{l1}+L{l2}: cos={cos_bl:.6f}, rel_nl={rnl:.6f}")
            torch.cuda.empty_cache()

    release_model(model)

    valid = {k: v for k, v in results.items() if "error" not in v}
    if valid:
        ac = np.mean([v["cos_both_linear"] for v in valid.values()])
        an = np.mean([v["relative_nonlinearity"] for v in valid.values()])
        print(f"\n  Avg cos(δ_both, δ_linear) = {ac:.6f}")
        print(f"  Avg ||δ_nl||/||δ_linear|| = {an:.6f}")

    return results


# ============================================================
import time

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "glm4"
    print(f"\nPhase 147c: {model_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    print(f"Mode: {'8bit' if use_8bit else 'bfloat16'}")

    # Exp 1
    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    info = get_model_info(model, model_name)
    print(f"Model: {info.model_class}, {info.n_layers}L, d={info.d_model}")

    exp1 = exp1_rollout(model, tokenizer, device, info)
    release_model(model)
    gc.collect(); torch.cuda.empty_cache(); time.sleep(3)

    # Exp 2
    exp2 = exp2_random_control(model_name)
    gc.collect(); torch.cuda.empty_cache(); time.sleep(3)

    # Exp 3
    exp3 = exp3_nonlinearity(model_name)

    # 保存
    ts = datetime.now().strftime('%Y%m%d_%H%M')
    def to_ser(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.float32, np.float64)): return float(o)
        if isinstance(o, (np.int32, np.int64)): return int(o)
        if isinstance(o, dict): return {k: to_ser(v) for k, v in o.items()}
        if isinstance(o, list): return [to_ser(x) for x in o]
        return o

    full = {"model": model_name, "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M'),
            "exp1_rollout": to_ser(exp1), "exp2_random_control": to_ser(exp2),
            "exp3_nonlinearity": to_ser(exp3)}

    p = Path(f"tests/glm5_temp/phase147c_{model_name}_{ts}.json")
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(full, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
