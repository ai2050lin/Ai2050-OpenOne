"""
Phase 153 精简版: 边界驱动统计动力学 — DS7B/GLM4快速测试
==========================================================

减少样本量加速8bit模型测试:
- Exp 1: 20句 × 20扰动 × 8注入层
- Exp 2: CCA (50句)
- Exp 3: Attention Routing (20句)
- Exp 4: Boundary Crossing (40句 × 5注入层)

用法:
  python tests/glm5_temp/phase153_fast.py deepseek7b
  python tests/glm5_temp/phase153_fast.py glm4
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from model_utils import (get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS)

OUTPUT_DIR = Path("tests/glm5_temp")

TEST_PROMPTS = [
    "The scientist discovered that the",
    "In the morning, she decided to",
    "The book on the table was about",
    "After the rain stopped, the children",
    "The most important thing about science is",
    "When the sun sets over the ocean,",
    "She walked into the room and saw",
    "The professor explained that the theory",
    "Despite the challenges, the team managed",
    "The ancient city was known for its",
    "He realized that the answer was",
    "The relationship between language and thought",
    "Every morning she would read the",
    "The experiment showed that the results",
    "Music has the power to change how",
    "The government announced that the new policy",
    "In the future, artificial intelligence will",
    "The philosopher argued that consciousness is",
    "After years of research, they found that",
    "The key difference between the two approaches is",
    "The cat sat on the windowsill and watched",
    "Through the telescope, they observed a new",
    "The river flowed gently through the valley",
    "She opened the letter and read the",
    "The painting on the wall depicted a",
    "During the concert, the audience was",
    "The invention changed the way people",
    "He wrote a letter to his friend about",
    "The students in the classroom were learning",
    "The old building at the corner had",
    "The doctor told him that he needed",
    "A sudden noise from outside made her",
    "The forest was filled with ancient trees",
    "She picked up the phone and called",
    "The road to the village was long and",
    "They stood at the edge of the cliff and",
    "The novel she was reading described a",
    "At the conference, the speaker presented",
    "The children played in the garden while",
    "The old man smiled and said that",
    "The company decided to invest in new",
    "The train arrived at the station just as",
    "Through the window, she could see the",
    "The puzzle was more difficult than they",
    "The artist carefully mixed the colors to",
    "The report concluded that the main cause",
    "She remembered the day when they first",
    "The mountain was covered with snow and",
    "The debate focused on whether the government",
    "He turned the key and opened the door to",
]


def load_model_custom(model_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    cfg = MODEL_CONFIGS[model_name]
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    print(f"  Loading {model_name} (8bit={use_8bit})...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    gc.collect()
    torch.cuda.empty_cache()
    if use_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
        attn_impl = "sdpa" if model_name == "deepseek7b" else "eager"
        model = AutoModelForCausalLM.from_pretrained(cfg["path"], quantization_config=bnb_config,
            device_map="auto", trust_remote_code=True, local_files_only=True,
            attn_implementation=attn_impl, low_cpu_mem_usage=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["path"], torch_dtype=torch.bfloat16,
            device_map="cpu", trust_remote_code=True, local_files_only=True,
            low_cpu_mem_usage=True, attn_implementation="eager")
        if torch.cuda.is_available():
            model = model.to("cuda")
    model.eval()
    device = next(model.parameters()).device
    return model, tokenizer, device


def get_sample_layers(n_layers):
    if n_layers <= 12:
        return list(range(0, n_layers + 1))
    early = [0, 1, 2, 3, 4]
    mid_step = max(1, (n_layers - 10) // 5)
    mid = list(range(5, n_layers - 4, mid_step))
    late = [n_layers - 4, n_layers - 3, n_layers - 2, n_layers - 1, n_layers]
    return sorted(set(early + mid + late))


def make_inject_hook(last_pos, delta_tensor):
    def hook(module, input, output):
        if isinstance(output, tuple):
            out = output[0].clone()
            out[0, last_pos, :] += delta_tensor.to(out.dtype).to(out.device)
            return (out,) + output[1:]
        else:
            out = output.clone()
            out[0, last_pos, :] += delta_tensor.to(out.dtype).to(out.device)
            return out
    return hook


# ============================================================
# Exp 1: Boundary Propagation (精简版)
# ============================================================
def exp1_boundary_propagation(model, tokenizer, model_name):
    print("\n" + "="*60)
    print("Exp 1: Logit Boundary Propagation (Fast)")
    print("="*60)
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    n_layers = info.n_layers
    d_model = info.d_model

    inject_layers = sorted(set([0, 1, 2, 3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]))[:9]
    n_sents = 20
    n_perturb = 20
    EPSILON = 1.0

    results = {}
    for inject_li in inject_layers:
        margin_changes = []
        stay_count = 0
        total = 0
        for sent_idx in range(n_sents):
            prompt = TEST_PROMPTS[sent_idx]
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            last_pos = input_ids.shape[1] - 1

            with torch.no_grad():
                out_clean = model(input_ids=input_ids, attention_mask=attention_mask)
            clean_logits = out_clean.logits[0, -1, :].float().cpu().numpy()
            top1_id = int(np.argmax(clean_logits))
            sorted_ids = np.argsort(-clean_logits)
            top2_id = sorted_ids[1]
            baseline_margin = clean_logits[top1_id] - clean_logits[top2_id]

            for p_idx in range(n_perturb):
                np.random.seed(inject_li * 10000 + sent_idx * 100 + p_idx)
                delta = np.random.randn(d_model)
                delta = delta / np.linalg.norm(delta) * EPSILON
                delta_tensor = torch.tensor(delta, dtype=torch.float32)

                layers_list = get_layers(model)
                hooks = [layers_list[inject_li].register_forward_hook(make_inject_hook(last_pos, delta_tensor))]
                try:
                    with torch.no_grad():
                        out_p = model(input_ids=input_ids, attention_mask=attention_mask)
                    p_logits = out_p.logits[0, -1, :].float().cpu().numpy()
                    p_top1 = int(np.argmax(p_logits))
                    p_margin = p_logits[top1_id] - p_logits[top2_id]
                    margin_changes.append(abs(baseline_margin - p_margin))
                    if p_top1 == top1_id:
                        stay_count += 1
                    total += 1
                except:
                    pass
                for h in hooks:
                    h.remove()

        if margin_changes:
            results[inject_li] = {
                'mean_margin_change': float(np.mean(margin_changes)),
                'std_margin_change': float(np.std(margin_changes)),
                'stay_rate': float(stay_count / total) if total > 0 else 0,
                'n': len(margin_changes),
            }
            print(f"  L{inject_li:>2d}: margin_chg={results[inject_li]['mean_margin_change']:.4f}, "
                  f"stay={results[inject_li]['stay_rate']:.3f}")
    return results


# ============================================================
# Exp 2: CCA (50句)
# ============================================================
def exp2_cca(model, tokenizer, model_name):
    print("\n" + "="*60)
    print("Exp 2: CCA (sklearn)")
    print("="*60)
    from sklearn.cross_decomposition import CCA as SklearnCCA

    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    n_layers = info.n_layers
    d_model = info.d_model

    sample_layers = get_sample_layers(n_layers)
    n_sents = min(50, len(TEST_PROMPTS))

    all_hs = {li: [] for li in sample_layers}
    for sent_idx in range(n_sents):
        prompt = TEST_PROMPTS[sent_idx]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        for li in sample_layers:
            all_hs[li].append(out.hidden_states[li][0, -1, :].float().cpu().numpy())

    ref_layer = 1
    X_ref = np.array(all_hs[ref_layer])
    X_ref_c = X_ref - X_ref.mean(axis=0)

    n_pca = min(n_sents - 1, d_model)
    U_ref, s_ref, _ = np.linalg.svd(X_ref_c, full_matrices=False)
    n_keep = min(n_sents - 2, 60)
    X_ref_pca = U_ref[:, :n_keep] * s_ref[:n_keep]

    n_cca = min(5, n_keep - 1)
    cca_results = {}

    for li in sample_layers:
        X_li = np.array(all_hs[li])
        X_li_c = X_li - X_li.mean(axis=0)
        U_li, s_li, _ = np.linalg.svd(X_li_c, full_matrices=False)
        X_li_pca = U_li[:, :n_keep] * s_li[:n_keep]

        try:
            cca = SklearnCCA(n_components=n_cca)
            cca.fit(X_ref_pca, X_li_pca)
            X_r, X_l = cca.transform(X_ref_pca, X_li_pca)
            corrs = [abs(np.corrcoef(X_r[:, i], X_l[:, i])[0, 1]) for i in range(n_cca)]
        except:
            corrs = [0.0] * n_cca

        cca_results[li] = {
            'cca1': float(corrs[0]) if corrs else 0,
            'cca_mean': float(np.mean(corrs)),
        }
        print(f"  hs[{li:>3d}]: CCA_1={corrs[0]:.4f}, CCA_mean={np.mean(corrs):.4f}")
    return cca_results


# ============================================================
# Exp 3: Attention Routing (20句)
# ============================================================
def exp3_attention(model, tokenizer, model_name):
    print("\n" + "="*60)
    print("Exp 3: Attention Routing (Fast)")
    print("="*60)
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    n_layers = info.n_layers

    supports = True
    try:
        inputs = tokenizer(TEST_PROMPTS[0], return_tensors="pt", truncation=True, max_length=64)
        with torch.no_grad():
            t = model(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"].to(device),
                      output_attentions=True)
        if t.attentions is None:
            supports = False
    except:
        supports = False

    n_sents = 20
    attn_layers = sorted(set([0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]))[:7]
    head_ent = {li: {} for li in attn_layers}

    for si in range(n_sents):
        inputs = tokenizer(TEST_PROMPTS[si], return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        seq_len = input_ids.shape[1]
        last_pos = seq_len - 1

        if supports:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
            for li in attn_layers:
                if li < len(out.attentions) and out.attentions[li] is not None:
                    aw = out.attentions[li][0, :, last_pos, :last_pos+1].float().cpu().numpy()
                    for h in range(aw.shape[0]):
                        if h not in head_ent[li]:
                            head_ent[li][h] = []
                        ad = aw[h]; ad = ad / max(ad.sum(), 1e-10)
                        ent = -np.sum(ad * np.log(ad + 1e-10))
                        max_ent = np.log(len(ad))
                        head_ent[li][h].append(ent / max_ent if max_ent > 0 else 0)
        else:
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            for li in attn_layers:
                layers_list = get_layers(model)
                sa = layers_list[li].self_attn
                h_in = out.hidden_states[li][0]
                W_q = sa.q_proj.weight.detach().float()
                W_k = sa.k_proj.weight.detach().float()
                Q = (h_in @ W_q.T).cpu().numpy()
                K = (h_in @ W_k.T).cpu().numpy()
                n_h = min(8, Q.shape[1] // 32)
                d_h = Q.shape[1] // n_h if n_h > 0 else Q.shape[1]
                for h in range(n_h):
                    if h not in head_ent[li]:
                        head_ent[li][h] = []
                    Qh = Q[last_pos, h*d_h:(h+1)*d_h]
                    Kh = K[:last_pos+1, h*d_h:(h+1)*d_h]
                    scores = Kh @ Qh / max(np.sqrt(d_h), 1.0)
                    es = np.exp(scores - np.max(scores))
                    ad = es / es.sum()
                    ent = -np.sum(ad * np.log(ad + 1e-10))
                    max_ent = np.log(len(ad)) if len(ad) > 1 else 1
                    head_ent[li][h].append(ent / max_ent if max_ent > 0 else 0)

    spec = {}
    for li in sorted(head_ent.keys()):
        hd = head_ent[li]
        if not hd:
            continue
        focused = sum(1 for e in hd.values() if len(e) >= 3 and np.mean(e) < 0.3)
        diffuse = sum(1 for e in hd.values() if len(e) >= 3 and np.mean(e) > 0.7)
        cvs = [np.std(e)/max(np.mean(e),0.01) for e in hd.values() if len(e)>=3 and np.mean(e)>0.01]
        spec[li] = {'focused': focused, 'diffuse': diffuse, 'cv': float(np.mean(cvs)) if cvs else 0}
        print(f"  L{li}: focused={focused}, diffuse={diffuse}, CV={spec[li]['cv']:.3f}")
    return {'spec': spec, 'supports': supports}


# ============================================================
# Exp 4: Boundary Crossing (40句 × 5层)
# ============================================================
def exp4_crossing(model, tokenizer, model_name):
    print("\n" + "="*60)
    print("Exp 4: Boundary Crossing (Fast)")
    print("="*60)
    info = get_model_info(model, model_name)
    device = next(model.parameters()).device
    n_layers = info.n_layers
    d_model = info.d_model

    eps_scan = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
    inject_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2]))[:5]
    n_sents = 40
    results = {li: [] for li in inject_layers}

    for si in range(n_sents):
        prompt = TEST_PROMPTS[si]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attn_mask = inputs["attention_mask"].to(device)
        last_pos = input_ids.shape[1] - 1

        with torch.no_grad():
            oc = model(input_ids=input_ids, attention_mask=attn_mask)
        cl = oc.logits[0, -1, :].float().cpu().numpy()
        sid = np.argsort(-cl); top1=sid[0]; top2=sid[1]
        bm = cl[top1] - cl[top2]

        np.random.seed(si * 7 + 13)
        dd = np.random.randn(d_model); dd /= np.linalg.norm(dd)

        for li in inject_layers:
            sw = None
            for eps in eps_scan:
                ds = dd * eps
                dt = torch.tensor(ds, dtype=torch.float32)
                ll = get_layers(model)
                hooks = [ll[li].register_forward_hook(make_inject_hook(last_pos, dt))]
                try:
                    with torch.no_grad():
                        op = model(input_ids=input_ids, attention_mask=attn_mask)
                    pt = int(np.argmax(op.logits[0, -1, :].float().cpu().numpy()))
                    if pt != top1 and sw is None:
                        sw = eps
                except:
                    pass
                for h in hooks:
                    h.remove()
            results[li].append({'margin': float(bm), 'sw_eps': sw})

    summary = {}
    for li in inject_layers:
        d = results[li]
        sw = [r for r in d if r['sw_eps'] is not None]
        sr = len(sw)/len(d) if d else 0
        n = [r for r in d if r['margin'] < 1]
        m = [r for r in d if 1 <= r['margin'] < 3]
        w = [r for r in d if r['margin'] >= 3]
        corr = np.corrcoef([r['margin'] for r in sw], [r['sw_eps'] for r in sw])[0,1] if len(sw)>3 else 0
        summary[li] = {'sr': float(sr), 'narrow_sr': sum(1 for r in n if r['sw_eps'] is not None)/len(n) if n else 0,
                       'wide_sr': sum(1 for r in w if r['sw_eps'] is not None)/len(w) if w else 0,
                       'corr': float(corr)}
        print(f"  L{li}: switch_rate={sr:.1%}, narrow_sr={summary[li]['narrow_sr']:.1%}, "
              f"wide_sr={summary[li]['wide_sr']:.1%}, corr={corr:.3f}")
    return summary


def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "deepseek7b"
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    print(f"Phase 153 Fast: {model_name}, {ts}")

    t0 = time.time()
    model, tokenizer, device = load_model_custom(model_name)
    info = get_model_info(model, model_name)
    print(f"Model: {info.model_class}, {info.n_layers}L, d={info.d_model}, load={time.time()-t0:.1f}s")

    e1 = exp1_boundary_propagation(model, tokenizer, model_name)
    e2 = exp2_cca(model, tokenizer, model_name)
    e3 = exp3_attention(model, tokenizer, model_name)
    e4 = exp4_crossing(model, tokenizer, model_name)

    all_r = {"phase": "153_fast", "model": model_name, "timestamp": ts,
             "model_info": {"class": info.model_class, "n_layers": info.n_layers, "d_model": info.d_model},
             "exp1": e1, "exp2_cca": e2, "exp3": e3, "exp4": e4}

    rf = OUTPUT_DIR / f"phase153_{model_name}_{ts}.json"
    def conv(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.float32, np.float64)): return float(o)
        if isinstance(o, (np.int32, np.int64)): return int(o)
        raise TypeError(f"Cannot serialize {type(o)}")
    with open(rf, 'w', encoding='utf-8') as f:
        json.dump(all_r, f, indent=2, default=conv, ensure_ascii=False)
    print(f"\nSaved: {rf}")
    release_model(model)
    gc.collect(); torch.cuda.empty_cache()
    print("Done.")


if __name__ == "__main__":
    main()
