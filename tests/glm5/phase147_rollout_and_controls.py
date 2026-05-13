"""Phase 147: Rollout稳定性 + 随机权重对照 + Jacobian可交换性
================================================================

Phase 146.5理论审查后最关键的3个实验:

Exp 1: Rollout稳定性 (最高优先级)
  - 在指定层注入null-space扰动
  - 自回归生成100 token, 对比clean vs perturbed轨迹
  - 关键问题: null-space扰动50 token后轨迹是否漂移?
  - 如果完全漂移 → "输出等价类稳定"是局部假象

Exp 2: 随机权重对照 (区分架构 vs 训练)
  - 同架构随机初始化模型的传播性质
  - 对比: 逐层SV估计, 扰动传播增长, 方向余弦
  - 如果随机模型也有SV≈1 → 临界性是架构效应

Exp 3: Jacobian非线性交互 (流形存在性判据)
  - 在l1注入δ, 在l2注入δ, 同时注入δ
  - 如果 δ_both ≈ δ_l1 + δ_l2 → 系统近似线性(流形可能存在)
  - 如果 δ_both ≠ δ_l1 + δ_l2 → 非线性交互(流形不存在)

用法:
  python tests/glm5/phase147_rollout_and_controls.py qwen3
  python tests/glm5/phase147_rollout_and_controls.py glm4
  python tests/glm5/phase147_rollout_and_controls.py deepseek7b
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
from scipy.sparse.linalg import svds
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)

# ===== 实验配置 =====
N_ROLLOUT_TOKENS = 100
N_ROLLOUT_SENTENCES = 15       # Rollout测试句子数(加大数据量)
N_SVD_COMPONENTS = 200
INJECT_LAYERS_FRAC = [0.0, 0.25, 0.5, 0.75, 1.0]
EPSILONS_ROLLOUT = [0.5, 2.0, 5.0]

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


def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# Exp 1: Rollout稳定性
# ============================================================
def exp1_rollout_stability(model, tokenizer, device, model_info, W_U=None):
    """
    在指定层注入null-space扰动, 自回归生成100 token, 对比轨迹
    """
    print("\n" + "="*60)
    print("Exp 1: Rollout稳定性 (最高优先级)")
    print("="*60)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    results = {}

    # W_U SVD → null space基
    U_wut = None
    if W_U is not None:
        W_U_T = W_U.T.astype(np.float32)
        k = min(N_SVD_COMPONENTS, min(W_U_T.shape) - 2)
        U_wut, s_wut, _ = svds(W_U_T, k=k)
        U_wut = U_wut.astype(np.float64)
        print(f"  W_U SVD: {k} components, top SV={s_wut[-1]:.2f}")

    inject_layers = [int(f * (n_layers - 1)) for f in INJECT_LAYERS_FRAC]
    layers = get_layers(model)

    for sent_idx in range(min(N_ROLLOUT_SENTENCES, len(TEST_PROMPTS))):
        prompt = TEST_PROMPTS[sent_idx]
        print(f"\n  --- Sent {sent_idx}: '{prompt[:40]}...' ---")

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
            for eps in EPSILONS_ROLLOUT:
                key = f"sent{sent_idx}_L{inj_l}_eps{eps:.1f}"
                print(f"    {key}...", end=" ", flush=True)

                try:
                    r = _single_rollout(
                        model, tokenizer, device, layers, n_layers, d_model,
                        input_ids, attn_mask, clean_toks, clean_text,
                        inj_l, eps, U_wut
                    )
                    results[key] = r
                    print(f"overlap@50={r['overlap_50']:.2f}, "
                          f"first_div={r['first_div_pos']}, "
                          f"sem_div={r['semantic_div']:.3f}")
                except Exception as e:
                    print(f"ERR: {e}")
                    results[key] = {"error": str(e)}

                torch.cuda.empty_cache()

    # 汇总
    _summarize_rollout(results, inject_layers, EPSILONS_ROLLOUT)
    return results


def _single_rollout(model, tokenizer, device, layers, n_layers, d_model,
                    input_ids, attn_mask, clean_toks, clean_text,
                    inject_layer, eps, U_wut):
    """单次rollout: 注入null-space扰动 → 生成 → 对比"""

    # 生成null-space扰动方向
    rand_dir = np.random.randn(d_model).astype(np.float64)
    if U_wut is not None:
        proj = U_wut @ (U_wut.T @ rand_dir)
        null_dir = rand_dir - proj
        norm = np.linalg.norm(null_dir)
        null_dir = null_dir / max(norm, 1e-10)
    else:
        null_dir = rand_dir / np.linalg.norm(rand_dir)
    delta_np = (eps * null_dir).astype(np.float32)

    # Hook: 注入扰动
    def inject_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        delta_t = torch.tensor(delta_np, dtype=h.dtype, device=h.device)
        h_new = h.clone()
        h_new[0, -1, :] += delta_t  # 只对last token
        return (h_new,) + output[1:] if isinstance(output, tuple) else h_new

    h_handle = layers[inject_layer].register_forward_hook(inject_hook)

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

    semantic_div = _semantic_divergence(tokenizer, clean_text, pert_text)

    # Perplexity drift (简化: 只测前200字符)
    try:
        ppl_clean = _quick_ppl(model, tokenizer, device, clean_text[:200])
        ppl_pert = _quick_ppl(model, tokenizer, device, pert_text[:200])
        ppl_ratio = ppl_pert / max(ppl_clean, 1e-10)
    except:
        ppl_clean, ppl_pert, ppl_ratio = 0, 0, 0

    return {
        "inject_layer": inject_layer, "eps": eps,
        "n_gen": int(n_gen), "first_div_pos": int(first_div),
        **overlaps,
        "semantic_div": float(semantic_div),
        "ppl_clean": float(ppl_clean), "ppl_pert": float(ppl_pert),
        "ppl_ratio": float(ppl_ratio),
        "clean_text": clean_text[:120], "pert_text": pert_text[:120],
    }


def _quick_ppl(model, tokenizer, device, text, max_len=128):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model(input_ids=ids)
        ce = F.cross_entropy(
            out.logits[:, :-1, :].reshape(-1, out.logits.size(-1)),
            ids[:, 1:].reshape(-1), reduction='mean'
        )
    return torch.exp(ce).item()


def _semantic_divergence(tokenizer, text1, text2):
    ids1 = tokenizer.encode(text1, add_special_tokens=False)
    ids2 = tokenizer.encode(text2, add_special_tokens=False)
    if len(ids1) < 2 or len(ids2) < 2:
        return 1.0
    bg1 = set(tuple(ids1[i:i+2]) for i in range(len(ids1)-1))
    bg2 = set(tuple(ids2[i:i+2]) for i in range(len(ids2)-1))
    union = bg1 | bg2
    if not union:
        return 1.0
    return 1.0 - len(bg1 & bg2) / len(union)


def _summarize_rollout(results, inject_layers, epsilons):
    """按注入层和eps汇总rollout结果"""
    print(f"\n  === Rollout汇总 ===")
    for inj_l in inject_layers:
        for eps in epsilons:
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
            pr = np.mean([v["ppl_ratio"] for v in vals if v["ppl_ratio"] > 0])
            print(f"    L{inj_l} eps={eps:.1f}: overlap=[{o10:.2f},{o50:.2f},{o100:.2f}], "
                  f"first_div={fd:.0f}, sem_div={sd:.3f}, ppl_ratio={pr:.3f}")


# ============================================================
# Exp 2: 随机权重对照
# ============================================================
def exp2_random_weight_control(model, tokenizer, device, model_info):
    """
    对比训练模型 vs 随机初始化模型的传播性质
    """
    print("\n" + "="*60)
    print("Exp 2: 随机权重对照 (区分架构 vs 训练)")
    print("="*60)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    results = {"trained": {}, "random": {}}

    # 1. 训练模型的传播性质
    print("\n  [1] 训练模型传播性质...")
    trained_stats = _propagation_stats(model, tokenizer, device, n_layers, d_model, n_sents=8)
    results["trained"] = trained_stats

    # 2. 创建随机模型 (CPU上创建避免OOM, 逐层比较)
    print("\n  [2] 创建随机权重模型...")
    try:
        random_model = _make_random_model(model, model_info, device)
        print("  [2] 随机模型创建成功")
    except Exception as e:
        print(f"  [2] 随机模型创建失败: {e}")
        print("  [2] 跳过Exp2, 改用理论分析")
        # 退路: 只用训练模型的数据做理论判断
        results["random"] = {"error": str(e)}
        return results

    # 3. 随机模型的传播性质
    print("\n  [3] 随机模型传播性质...")
    random_stats = _propagation_stats(random_model, tokenizer, device, n_layers, d_model, n_sents=8)
    results["random"] = random_stats

    # 4. 对比
    print("\n  === 训练 vs 随机 对比 ===")
    for metric in ["sv_estimates", "direction_cosines", "perturbation_growth"]:
        t = trained_stats.get(metric, {})
        r = random_stats.get(metric, {})
        for layer_key in t:
            if layer_key in r:
                t_val = np.mean(t[layer_key]) if isinstance(t[layer_key], list) else t[layer_key]
                r_val = np.mean(r[layer_key]) if isinstance(r[layer_key], list) else r[layer_key]
                diff = t_val - r_val
                print(f"    {metric}/{layer_key}: trained={t_val:.4f}, random={r_val:.4f}, diff={diff:.4f}")

    # 释放随机模型
    del random_model
    gc.collect()
    torch.cuda.empty_cache()

    return results


def _make_random_model(model, model_info, device):
    """创建同架构随机初始化模型"""
    config = model.config
    model_class = type(model)
    random_model = model_class(config)

    def init_weights(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    random_model.apply(init_weights)
    random_model.eval()

    # 如果GPU内存不够, 只把小模型放GPU
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    used_mem = torch.cuda.memory_allocated() / 1e9
    available = gpu_mem - used_mem
    model_size = sum(p.numel() * p.element_size() for p in random_model.parameters()) / 1e9

    if model_size < available * 0.8:
        random_model = random_model.to(device)
        print(f"    随机模型已放GPU (size={model_size:.1f}GB, avail={available:.1f}GB)")
    else:
        # CPU上运行(慢但可行)
        random_model = random_model.to("cpu")
        print(f"    随机模型放CPU (size={model_size:.1f}GB > avail={available:.1f}GB)")

    return random_model


def _propagation_stats(model, tokenizer, device, n_layers, d_model, n_sents=8):
    """计算模型的传播统计量: SV估计, 方向余弦, 扰动增长"""
    stats = {
        "sv_estimates": {},       # 每层Jacobian的最大SV估计
        "direction_cosines": {},   # 扰动后方向余弦
        "perturbation_growth": {}, # 扰动幅度增长
    }

    sample_layers = sorted(set([0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]))
    layers = get_layers(model)
    model_device = get_input_device(model)

    for si in range(min(n_sents, len(TEST_PROMPTS))):
        prompt = TEST_PROMPTS[si]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(model_device)
        attn_mask = inputs["attention_mask"].to(model_device)

        # Clean forward
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        hs_clean = [h[0, -1, :].detach().cpu().float().numpy() for h in out.hidden_states]

        # 对每个采样层, 注入扰动并收集后续层hidden state
        eps = 1.0
        for inj_l in sample_layers:
            # 随机扰动方向
            v = np.random.randn(d_model).astype(np.float32)
            v = v / np.linalg.norm(v)
            delta_np = (eps * v).astype(np.float32)

            captured = {}
            def make_capture_hook(layer_idx):
                def hook(module, input, output):
                    h = output[0] if isinstance(output, tuple) else output
                    captured[layer_idx] = h[0, -1, :].detach().cpu().float().numpy()
                return hook

            def inject_hook(module, input, output):
                h = output[0] if isinstance(output, tuple) else output
                delta_t = torch.tensor(delta_np, dtype=h.dtype, device=h.device)
                h_new = h.clone()
                h_new[0, -1, :] += delta_t
                return (h_new,) + output[1:] if isinstance(output, tuple) else h_new

            # 注册hooks
            hooks = []
            hooks.append(layers[inj_l].register_forward_hook(inject_hook))
            for li in sample_layers:
                if li > inj_l:
                    hooks.append(layers[li].register_forward_hook(make_capture_hook(li)))

            try:
                with torch.no_grad():
                    model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
            except:
                pass
            finally:
                for h in hooks:
                    h.remove()

            # 分析
            for li in captured:
                h_pert = captured[li]
                h_c = hs_clean[li + 1]  # +1: hs_clean包含embedding
                delta = h_pert - h_c

                # 扰动增长比
                growth = np.linalg.norm(delta) / eps
                key_growth = f"L{inj_l}_to_L{li}"
                if key_growth not in stats["perturbation_growth"]:
                    stats["perturbation_growth"][key_growth] = []
                stats["perturbation_growth"][key_growth].append(float(growth))

                # 方向余弦
                cos_val = np.dot(h_c, h_pert) / (np.linalg.norm(h_c) * np.linalg.norm(h_pert) + 1e-10)
                key_cos = f"L{inj_l}_to_L{li}"
                if key_cos not in stats["direction_cosines"]:
                    stats["direction_cosines"][key_cos] = []
                stats["direction_cosines"][key_cos].append(float(cos_val))

                # SV估计: ||delta|| / eps ≈ ||Jv|| → max over random v ≈ max SV
                key_sv = f"L{inj_l}_at_L{li}"
                if key_sv not in stats["sv_estimates"]:
                    stats["sv_estimates"][key_sv] = []
                stats["sv_estimates"][key_sv].append(float(growth))

        # 清理
        del out, hs_clean
        torch.cuda.empty_cache()

    return stats


# ============================================================
# Exp 3: Jacobian非线性交互 (流形存在性判据)
# ============================================================
def exp3_jacobian_nonlinearity(model, tokenizer, device, model_info):
    """
    测试非线性交互: δ_both vs δ_l1 + δ_l2
    
    如果系统是线性的(流形上): δ_both = δ_l1 + δ_l2
    如果非线性: δ_both ≠ δ_l1 + δ_l2, 不可交换
    """
    print("\n" + "="*60)
    print("Exp 3: Jacobian非线性交互 (流形存在性判据)")
    print("="*60)

    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    results = {}

    layer_pairs = [
        (0, n_layers // 4),
        (0, n_layers // 2),
        (n_layers // 4, n_layers // 2),
        (n_layers // 4, 3 * n_layers // 4),
        (n_layers // 2, 3 * n_layers // 4),
    ]

    eps = 1.0
    model_device = get_input_device(model)

    for si in range(min(10, len(TEST_PROMPTS))):
        prompt = TEST_PROMPTS[si]
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(model_device)
        attn_mask = inputs["attention_mask"].to(model_device)

        print(f"\n  Sent {si}: '{prompt[:40]}...'")

        # Clean forward
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True)
        h_clean = out.hidden_states[-1][0, -1, :].detach().cpu().float().numpy()
        del out
        torch.cuda.empty_cache()

        for l1, l2 in layer_pairs:
            key = f"sent{si}_L{l1}L{l2}"

            # 随机扰动方向
            v1 = np.random.randn(d_model).astype(np.float32)
            v1 = v1 / np.linalg.norm(v1) * eps
            v2 = np.random.randn(d_model).astype(np.float32)
            v2 = v2 / np.linalg.norm(v2) * eps

            # 三种注入: l1 only, l2 only, both
            h_l1 = _inject_and_get_final_h(model, input_ids, attn_mask, layers, [(l1, v1)])
            h_l2 = _inject_and_get_final_h(model, input_ids, attn_mask, layers, [(l2, v2)])
            h_both = _inject_and_get_final_h(model, input_ids, attn_mask, layers, [(l1, v1), (l2, v2)])

            if h_l1 is None or h_l2 is None or h_both is None:
                results[key] = {"error": "injection failed"}
                continue

            delta_l1 = h_l1 - h_clean
            delta_l2 = h_l2 - h_clean
            delta_both = h_both - h_clean
            delta_linear = delta_l1 + delta_l2

            # 非线性交互
            delta_nl = delta_both - delta_linear
            nl_norm = np.linalg.norm(delta_nl)
            linear_norm = np.linalg.norm(delta_linear)
            both_norm = np.linalg.norm(delta_both)

            # Cosine: delta_both vs delta_linear
            if both_norm > 1e-10 and linear_norm > 1e-10:
                cos_both_linear = np.dot(delta_both, delta_linear) / (both_norm * linear_norm)
            else:
                cos_both_linear = 1.0

            # 相对非线性度
            rel_nl = nl_norm / max(linear_norm, 1e-10)

            results[key] = {
                "l1": l1, "l2": l2,
                "cos_both_linear": float(cos_both_linear),
                "relative_nonlinearity": float(rel_nl),
                "delta_both_norm": float(both_norm),
                "delta_linear_norm": float(linear_norm),
                "nl_norm": float(nl_norm),
            }
            print(f"    L{l1}+L{l2}: cos={cos_both_linear:.6f}, "
                  f"rel_nl={rel_nl:.6f}")

            torch.cuda.empty_cache()

    # 汇总
    valid = {k: v for k, v in results.items() if "error" not in v}
    if valid:
        avg_cos = np.mean([v["cos_both_linear"] for v in valid.values()])
        avg_nl = np.mean([v["relative_nonlinearity"] for v in valid.values()])
        print(f"\n  === 非线性交互汇总 ===")
        print(f"  Avg cos(δ_both, δ_linear) = {avg_cos:.6f}")
        print(f"  Avg ||δ_nl|| / ||δ_linear|| = {avg_nl:.6f}")

        if avg_cos > 0.99:
            print("  >>> 系统近似线性 → 流形可能存在!")
        elif avg_cos > 0.95:
            print("  >>> 弱非线性 → 可能有近似流形")
        else:
            print("  >>> 强非线性 → 流形不存在, 方向场框架正确!")

    return results


def _inject_and_get_final_h(model, input_ids, attn_mask, layers, inject_specs):
    """
    在指定层注入指定扰动, 返回最终层hidden state
    
    inject_specs: [(layer_idx, delta_np), ...]
    """
    captured = {}

    def make_inject_hook(delta_np):
        def hook(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            delta_t = torch.tensor(delta_np, dtype=h.dtype, device=h.device)
            h_new = h.clone()
            h_new[0, -1, :] += delta_t
            return (h_new,) + output[1:] if isinstance(output, tuple) else h_new
        return hook

    def capture_final_hook(module, input, output):
        h = output[0] if isinstance(output, tuple) else output
        captured['final'] = h[0, -1, :].detach().cpu().float().numpy()

    hooks = []
    for li, delta in inject_specs:
        hooks.append(layers[li].register_forward_hook(make_inject_hook(delta)))
    hooks.append(layers[-1].register_forward_hook(capture_final_hook))

    try:
        with torch.no_grad():
            model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=False)
    except:
        pass
    finally:
        for h in hooks:
            h.remove()

    return captured.get('final', None)


# ============================================================
# 主函数
# ============================================================
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"

    print(f"\n{'#'*60}")
    print(f"Phase 147: Rollout稳定性 + 随机权重对照 + Jacobian非线性交互")
    print(f"Model: {model_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'#'*60}")

    cfg = MODEL_CONFIGS[model_name]
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    print(f"Load mode: {'8bit' if use_8bit else 'bfloat16'} (GPU={gpu_mem_gb:.1f}GB)")

    # 加载模型
    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    model_info = get_model_info(model, model_name)
    print(f"Model: {model_info.model_class}, {model_info.n_layers}L, d={model_info.d_model}")

    # W_U
    W_U = None
    try:
        W_U = get_W_U(model, model_name)
        print(f"W_U: shape={W_U.shape}")
    except Exception as e:
        print(f"W_U加载失败: {e}")

    # ===== Exp 1 =====
    exp1 = exp1_rollout_stability(model, tokenizer, device, model_info, W_U)

    # ===== Exp 2 =====
    exp2 = exp2_random_weight_control(model, tokenizer, device, model_info)

    # ===== Exp 3 =====
    exp3 = exp3_jacobian_nonlinearity(model, tokenizer, device, model_info)

    # ===== 保存 =====
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_serializable(x) for x in obj]
        return obj

    full_results = {
        "model": model_name,
        "model_info": {"n_layers": model_info.n_layers, "d_model": model_info.d_model,
                        "model_class": model_info.model_class},
        "use_8bit": use_8bit,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M'),
        "exp1_rollout": to_serializable(exp1),
        "exp2_random_control": to_serializable(exp2),
        "exp3_nonlinearity": to_serializable(exp3),
    }

    out_path = Path(f"tests/glm5_temp/phase147_{model_name}_results_{timestamp}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(full_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")

    # ===== 释放模型 =====
    release_model(model)

    # ===== 最终结论 =====
    print(f"\n{'#'*60}")
    print("Phase 147 最终结论")
    print(f"{'#'*60}")

    # Exp 1
    valid_r = {k: v for k, v in exp1.items() if "error" not in v}
    if valid_r:
        o50s = [v["overlap_50"] for v in valid_r.values()]
        avg_o50 = np.mean(o50s)
        fds = [v["first_div_pos"] for v in valid_r.values()]
        avg_fd = np.mean(fds)
        print(f"\n  Exp1 Rollout: avg overlap@50={avg_o50:.3f}, avg first_div={avg_fd:.0f}")
        if avg_o50 > 0.8:
            print("  → 输出等价类稳定不是幻觉, null-space扰动多步后仍稳定")
        elif avg_o50 > 0.5:
            print("  → 部分稳定, 有泄漏但可控")
        else:
            print("  → Rollout不稳定! 输出等价类稳定是单步假象!")

    # Exp 2
    trained = exp2.get("trained", {})
    random = exp2.get("random", {})
    if "error" not in random and trained and random:
        t_growth = trained.get("perturbation_growth", {})
        r_growth = random.get("perturbation_growth", {})
        t_cos = trained.get("direction_cosines", {})
        r_cos = random.get("direction_cosines", {})
        # 比较关键层对
        for key in t_growth:
            if key in r_growth:
                t_val = np.mean(t_growth[key])
                r_val = np.mean(r_growth[key])
                print(f"  Exp2 {key}: trained={t_val:.3f}, random={r_val:.3f}")
    elif "error" in random:
        print("  Exp2: 随机模型创建失败, 无法区分架构vs训练效应")
        print("  → 残差连接理论上保证SV≈1, 需要后续验证")

    # Exp 3
    valid_nl = {k: v for k, v in exp3.items() if "error" not in v}
    if valid_nl:
        avg_cos = np.mean([v["cos_both_linear"] for v in valid_nl.values()])
        avg_nl = np.mean([v["relative_nonlinearity"] for v in valid_nl.values()])
        print(f"  Exp3 非线性: avg cos(δ_both, δ_linear)={avg_cos:.6f}, "
              f"avg ||δ_nl||/||δ_lin||={avg_nl:.6f}")
        if avg_cos > 0.99:
            print("  → 系统近似线性, 流形可能存在")
        elif avg_cos > 0.95:
            print("  → 弱非线性, 可能有近似流形")
        else:
            print("  → 强非线性, 流形不存在!")


if __name__ == "__main__":
    main()
