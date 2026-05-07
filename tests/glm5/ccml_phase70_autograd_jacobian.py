"""
Phase 70: Autograd Jacobian分析 — Float32 GPT-2 Small
======================================================

Phase 69的核心问题:
1. bf16精度下有限差分不稳定 — Jv_gap≈0.01在bf16噪声边界
2. autograd全模型反向传播OOM (7B模型太大)
3. CKA≈0.1, eff_rank≈1.5可能是bf16噪声伪象
4. "没有线性区"的结论不可信

Phase 70的解决方案:
1. ★★★ 使用GPT-2 Small (124M) float32 — 无数值精度问题
2. ★★★ 用autograd计算精确梯度 ∂logit_gap/∂h_l
   - 不是有限差分! 是真正的导数!
   - Jv = dot(gradient, v) 是精确值
3. ★★★ 验证有限差分→autograd的收敛性
   - 如果FD在ε→0时收敛到autograd → Jacobian存在
   - 如果不收敛 → 真正的非线性
4. ★★★ 梯度矩阵的SVD/CKA
   - 多个输入的梯度构成矩阵 → SVD → spectrum/rank
   - CKA between sing/plur梯度 → 语法是否影响Jacobian

核心数学:
- logit_gap = logits[pv_id] - logits[sv_id]  (标量)
- ∂logit_gap/∂h_l = gradient  (R^{d_model}向量)
- 这就是Jacobian的一行(对应logit_gap方向)
- Jv = dot(gradient, v)  (精确, 无需ε)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np, gc, argparse, time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# GPT-2本地路径
GPT2_PATH = "D:/develop/model/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"

# 语法测试对 (GPT-2 BPE token IDs将在运行时确定)
NVA_PAIRS = [
    ("cat","cats","runs","run"),
    ("dog","dogs","walks","walk"),
    ("bird","birds","flies","fly"),
    ("girl","girls","reads","read"),
    ("boy","boys","sings","sing"),
    ("horse","horses","jumps","jump"),
    ("bear","bears","sleeps","sleep"),
    ("snake","snakes","crawls","crawl"),
    ("fish","fishes","swims","swim"),
    ("frog","frogs","hops","hop"),
    ("star","stars","shines","shine"),
    ("tree","trees","grows","grow"),
    ("clock","clocks","ticks","tick"),
    ("bell","bells","rings","ring"),
    ("fox","foxes","hunts","hunt"),
]


def load_gpt2_float32():
    """加载GPT-2 Small float32 — 124M参数, ~500MB"""
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    print("[Phase70] Loading GPT-2 Small float32...")
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_PATH, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(
        GPT2_PATH,
        torch_dtype=torch.float32,  # ★★★ 关键: float32!
        local_files_only=True,
    )
    if torch.cuda.is_available():
        model = model.to('cuda')
    model.eval()
    device = next(model.parameters()).device
    n_layers = model.config.n_layer
    d_model = model.config.n_embd
    vocab_size = model.config.vocab_size
    print(f"[Phase70] GPT-2 loaded: {n_layers} layers, d_model={d_model}, "
          f"vocab={vocab_size}, dtype=float32, device={device}")
    return model, tokenizer, device


def get_gpt2_layers(model):
    """获取GPT-2的transformer层"""
    return model.transformer.h


def compute_autograd_gradient(model, tokenizer, device, sentence, layer_idx,
                               sv_id, pv_id):
    """
    ★★★ 核心方法: 用autograd计算精确梯度 ∂logit_gap/∂h_l
    
    对于标量输出logit_gap, 梯度就是Jacobian的对应行.
    这是精确的, 不需要有限差分!
    
    Returns:
        gradient: numpy array [d_model] — 精确梯度
        logit_gap_base: float — 基线logit差
    """
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    
    # Step 1: 准备input embeddings with requires_grad
    embed_layer = model.transformer.wte
    pos_layer = model.transformer.wpe
    
    input_embeds = embed_layer(input_ids).detach().clone().requires_grad_(True)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    
    # Step 2: 在目标层注册hook来拦截和替换hidden state
    layers = get_gpt2_layers(model)
    
    h_l_captured = [None]
    
    def capture_hook(module, input, output):
        # 捕获layer的输出 (即h_{l+1}的输入)
        if isinstance(output, tuple):
            h_l_captured[0] = output[0].detach().clone()
        else:
            h_l_captured[0] = output.detach().clone()
        return output
    
    # 先做一次no_grad forward来捕获h_l
    hook = layers[layer_idx].register_forward_hook(capture_hook)
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    hook.remove()
    
    if h_l_captured[0] is None:
        return None, 0.0
    
    h_l = h_l_captured[0]  # [1, seq_len, d_model]
    
    # Step 3: 用h_l做forward (从layer l+1到输出)
    # 创建requires_grad的h_l
    h_l_grad = h_l.detach().clone().requires_grad_(True)
    
    def replace_hook(module, input, output):
        # 替换layer输出为我们requires_grad的版本
        # 需要保持和原始输出相同的格式
        if isinstance(output, tuple):
            return (h_l_grad,) + output[1:]
        return h_l_grad
    
    hook2 = layers[layer_idx].register_forward_hook(replace_hook)
    
    with torch.enable_grad():
        out = model(input_ids=input_ids)
        logits = out.logits[0, -1, :].float()
        logit_gap = logits[pv_id] - logits[sv_id]
        logit_gap.backward()
    
    hook2.remove()
    
    gradient = h_l_grad.grad[0, -1, :].detach().cpu().numpy()  # [d_model]
    logit_gap_val = float(logit_gap.detach().cpu())
    
    return gradient, logit_gap_val


def compute_fd_jv(model, tokenizer, device, sentence, layer_idx,
                   direction, sv_id, pv_id, epsilon):
    """
    有限差分Jv — 用于和autograd比较
    
    Jv ≈ (F(x+εv) - F(x-εv)) / (2ε)  (中心差分)
    """
    toks = tokenizer(sentence, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    
    layers = get_gpt2_layers(model)
    
    # 捕获h_l
    h_l_captured = [None]
    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            h_l_captured[0] = output[0].detach().clone()
        else:
            h_l_captured[0] = output.detach().clone()
        return output
    
    hook = layers[layer_idx].register_forward_hook(capture_hook)
    with torch.no_grad():
        _ = model(input_ids=input_ids)
    hook.remove()
    
    h_l = h_l_captured[0]  # [1, seq_len, d_model]
    
    # 注入 +εv 和 -εv
    direction_t = torch.tensor(direction, dtype=torch.float32, device=device)
    
    results = {}
    for sign, label in [(+1, 'plus'), (-1, 'minus')]:
        h_mod = h_l.detach().clone()
        h_mod[0, -1, :] += sign * epsilon * direction_t
        
        def replace_hook(module, input, output, h_replace=h_mod):
            if isinstance(output, tuple):
                return (h_replace,) + output[1:]
            return h_replace
        
        hook2 = layers[layer_idx].register_forward_hook(replace_hook)
        with torch.no_grad():
            out = model(input_ids=input_ids)
            logits = out.logits[0, -1, :].float().cpu().numpy()
        hook2.remove()
        
        results[label] = logits[pv_id] - logits[sv_id]
    
    fd_jv = (results['plus'] - results['minus']) / (2 * epsilon)
    return fd_jv


def compute_direction(model, tokenizer, device, layer_idx, pairs, n_pairs=8):
    """
    计算语法方向: v = E[h|plural] - E[h|singular]
    """
    layers = get_gpt2_layers(model)
    
    sing_states = []
    plur_states = []
    
    for i, (sn, pn, sv, pv) in enumerate(pairs[:n_pairs]):
        # Singular
        sent_s = f"The {sn}"
        toks = tokenizer(sent_s, return_tensors="pt").to(device)
        
        h_captured = [None]
        def capture_hook(module, input, output):
            if isinstance(output, tuple):
                h_captured[0] = output[0].detach().float().cpu().numpy()
            else:
                h_captured[0] = output.detach().float().cpu().numpy()
            return output
        
        hook = layers[layer_idx].register_forward_hook(capture_hook)
        with torch.no_grad():
            _ = model(**toks)
        hook.remove()
        
        if h_captured[0] is not None:
            sing_states.append(h_captured[0][0, -1, :])
        
        # Plural
        sent_p = f"The {pn}"
        toks = tokenizer(sent_p, return_tensors="pt").to(device)
        
        h_captured[0] = None
        hook = layers[layer_idx].register_forward_hook(capture_hook)
        with torch.no_grad():
            _ = model(**toks)
        hook.remove()
        
        if h_captured[0] is not None:
            plur_states.append(h_captured[0][0, -1, :])
    
    if not sing_states or not plur_states:
        return None
    
    sing_mean = np.mean(sing_states, axis=0)
    plur_mean = np.mean(plur_states, axis=0)
    direction = plur_mean - sing_mean
    
    norm = np.linalg.norm(direction)
    if norm > 1e-10:
        direction = direction / norm
    
    return direction


def linear_cka(X, Y):
    """
    Linear CKA between matrices X [n, d] and Y [m, d]
    """
    X = X - X.mean(axis=0, keepdims=True)
    Y = Y - Y.mean(axis=0, keepdims=True)
    
    # HSIC
    def hsic(A, B):
        # A: [n, d], B: [m, d]
        K = A @ A.T  # [n, n]
        L = B @ B.T  # [m, m]
        # 简化: 用trace公式
        return np.trace(K @ K) * np.trace(L @ L) / (A.shape[0] * B.shape[0])**2
    
    # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
    # 使用更稳定的公式
    XY = np.trace((X.T @ X) @ (Y.T @ Y))
    XX = np.trace((X.T @ X) @ (X.T @ X))
    YY = np.trace((Y.T @ Y) @ (Y.T @ Y))
    
    denom = np.sqrt(max(XX * YY, 1e-30))
    if denom < 1e-30:
        return 0.0
    return XY / denom


# ============================================================
# Experiment A: Autograd vs Finite Difference 收敛验证
# ============================================================
def experiment_a(model, tokenizer, device, n_test=15):
    """
    ★★★ 核心实验: 验证Jacobian是否存在
    
    方法:
    1. 用autograd计算精确梯度 = ∂logit_gap/∂h_l
    2. 用有限差分在ε = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]计算Jv
    3. 检查: FD → autograd as ε → 0?
    
    如果FD收敛 → Jacobian存在, 线性区存在
    如果FD不收敛 → 真正的非线性或数值问题
    """
    print("\n" + "="*70)
    print("Experiment A: Autograd vs Finite Difference Convergence")
    print("="*70)
    
    n_layers = model.config.n_layer
    sample_layers = [0, 3, 6, 9, 11]  # GPT-2 has 12 layers
    sample_layers = [l for l in sample_layers if l < n_layers]
    
    # 获取token IDs
    sv_id = tokenizer.encode(" runs", add_special_tokens=False)[0]
    pv_id = tokenizer.encode(" run", add_special_tokens=False)[0]
    print(f"  sv_id={sv_id} ('{tokenizer.decode([sv_id])}'), "
          f"pv_id={pv_id} ('{tokenizer.decode([pv_id])}')")
    
    epsilons = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1.0]
    
    results = defaultdict(lambda: defaultdict(list))
    
    for li in sample_layers:
        print(f"\n  --- Layer {li} ---")
        
        # 计算方向
        direction = compute_direction(model, tokenizer, device, li, NVA_PAIRS, n_pairs=8)
        if direction is None:
            print(f"    Direction computation failed, skipping")
            continue
        
        for test_idx in range(min(n_test, len(NVA_PAIRS))):
            sn, pn, sv, pv = NVA_PAIRS[test_idx]
            sentence = f"The {sn}"
            
            try:
                # ★★★ Autograd梯度 (精确!)
                gradient, logit_gap_base = compute_autograd_gradient(
                    model, tokenizer, device, sentence, li, sv_id, pv_id
                )
                
                if gradient is None:
                    continue
                
                # Jv_autograd = dot(gradient, direction)  — 精确值!
                jv_autograd = float(np.dot(gradient, direction))
                grad_norm = float(np.linalg.norm(gradient))
                
                # 有限差分
                for eps in epsilons:
                    jv_fd = compute_fd_jv(
                        model, tokenizer, device, sentence, li,
                        direction, sv_id, pv_id, eps
                    )
                    results[li][eps].append({
                        'jv_autograd': jv_autograd,
                        'jv_fd': jv_fd,
                        'grad_norm': grad_norm,
                        'logit_gap': logit_gap_base,
                    })
                
                if test_idx == 0:
                    print(f"    grad_norm={grad_norm:.4f}, "
                          f"jv_autograd={jv_autograd:.4f}, "
                          f"logit_gap={logit_gap_base:.4f}")
                
            except Exception as e:
                print(f"    Error at test {test_idx}: {e}")
                continue
        
        # 打印收敛性
        if li in results and results[li]:
            print(f"\n    Convergence (mean |FD - autograd| / |autograd|):")
            autograd_vals = [r['jv_autograd'] for r in results[li][epsilons[0]]]
            autograd_rms = np.sqrt(np.mean(np.array(autograd_vals)**2)) if autograd_vals else 1.0
            
            for eps in epsilons:
                if eps in results[li] and results[li][eps]:
                    errors = [abs(r['jv_fd'] - r['jv_autograd']) for r in results[li][eps]]
                    mean_err = np.mean(errors)
                    rel_err = mean_err / max(autograd_rms, 1e-10)
                    print(f"      ε={eps:.0e}: mean_err={mean_err:.6f}, "
                          f"rel_err={rel_err:.4f}")
    
    # 汇总
    print("\n" + "="*70)
    print("Experiment A Summary")
    print("="*70)
    
    for li in sample_layers:
        if li not in results or not results[li]:
            continue
        print(f"\n  Layer {li}:")
        
        # autograd统计
        ag = results[li][epsilons[0]]
        jv_ag = [r['jv_autograd'] for r in ag]
        gn = [r['grad_norm'] for r in ag]
        
        print(f"    Autograd Jv: mean={np.mean(jv_ag):.4f}, "
              f"std={np.std(jv_ag):.4f}, "
              f"|mean/std|={abs(np.mean(jv_ag))/max(np.std(jv_ag),1e-10):.2f}")
        print(f"    Gradient norm: mean={np.mean(gn):.4f}, "
              f"std={np.std(gn):.4f}")
        
        # 收敛性
        for eps in epsilons:
            if eps in results[li] and results[li][eps]:
                cors = [np.corrcoef(
                    [r['jv_autograd'] for r in results[li][eps]],
                    [r['jv_fd'] for r in results[li][eps]]
                )[0,1] if len(results[li][eps]) > 1 else 0]
                rel_errs = [abs(r['jv_fd'] - r['jv_autograd']) / max(abs(r['jv_autograd']), 1e-6)
                           for r in results[li][eps]]
                print(f"    ε={eps:.0e}: corr(FD,AG)={cors[0]:.4f}, "
                      f"mean_rel_err={np.mean(rel_errs):.4f}")
    
    return results


# ============================================================
# Experiment B: Gradient Spectrum + CKA
# ============================================================
def experiment_b(model, tokenizer, device, n_test=12):
    """
    ★★★ 梯度矩阵的SVD和CKA
    
    收集多个输入的梯度向量, 构建:
    - G_sing [n_sing, d_model]: singular句子的梯度
    - G_plur [n_plur, d_model]: plural句子的梯度
    
    分析:
    1. SVD of G → spectrum, effective rank
    2. CKA(G_sing, G_plur) → 语法是否影响tangent space
    3. CKA(G_sing_i, G_sing_j) → 同类句子的梯度一致性
    """
    print("\n" + "="*70)
    print("Experiment B: Gradient Spectrum + CKA")
    print("="*70)
    
    n_layers = model.config.n_layer
    sample_layers = [0, 3, 6, 9, 11]
    sample_layers = [l for l in sample_layers if l < n_layers]
    
    sv_id = tokenizer.encode(" runs", add_special_tokens=False)[0]
    pv_id = tokenizer.encode(" run", add_special_tokens=False)[0]
    
    for li in sample_layers:
        print(f"\n  --- Layer {li} ---")
        
        sing_grads = []
        plur_grads = []
        
        for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:n_test]):
            # Singular sentence
            sent_s = f"The {sn}"
            grad_s, gap_s = compute_autograd_gradient(
                model, tokenizer, device, sent_s, li, sv_id, pv_id
            )
            if grad_s is not None:
                sing_grads.append(grad_s)
            
            # Plural sentence
            sent_p = f"The {pn}"
            grad_p, gap_p = compute_autograd_gradient(
                model, tokenizer, device, sent_p, li, sv_id, pv_id
            )
            if grad_p is not None:
                plur_grads.append(grad_p)
        
        if len(sing_grads) < 3 or len(plur_grads) < 3:
            print(f"    Not enough gradients collected ({len(sing_grads)} sing, {len(plur_grads)} plur)")
            continue
        
        G_sing = np.array(sing_grads)  # [n_sing, d_model]
        G_plur = np.array(plur_grads)  # [n_plur, d_model]
        G_all = np.vstack([G_sing, G_plur])  # [n_all, d_model]
        
        # 1. SVD
        U, S, Vt = np.linalg.svd(G_all, full_matrices=False)
        
        # Effective rank
        S_norm = S / S.sum()
        eff_rank = np.exp(-np.sum(S_norm * np.log(S_norm + 1e-30)))
        
        # Spectrum decay
        s_ratios = S[:10] / S[0]
        
        print(f"    SVD: eff_rank={eff_rank:.2f}, "
              f"S[0:5]={S[:5].round(2)}, "
              f"S/S[0]={s_ratios[:5].round(4)}")
        
        # 2. CKA
        cka_ss = linear_cka(G_sing, G_sing)
        cka_pp = linear_cka(G_plur, G_plur)
        cka_sp = linear_cka(G_sing, G_plur)
        
        print(f"    CKA(sing,sing)={cka_ss:.4f}, "
              f"CKA(plur,plur)={cka_pp:.4f}, "
              f"CKA(sing,plur)={cka_sp:.4f}")
        
        # 3. 逐对cosine
        sing_norms = np.linalg.norm(G_sing, axis=1)
        plur_norms = np.linalg.norm(G_plur, axis=1)
        print(f"    Gradient norms: sing={np.mean(sing_norms):.4f}±{np.std(sing_norms):.4f}, "
              f"plur={np.mean(plur_norms):.4f}±{np.std(plur_norms):.4f}")
        
        # 4. 梯度方向的cosine相似度矩阵
        G_sing_norm = G_sing / (np.linalg.norm(G_sing, axis=1, keepdims=True) + 1e-10)
        G_plur_norm = G_plur / (np.linalg.norm(G_plur, axis=1, keepdims=True) + 1e-10)
        
        cos_ss = (G_sing_norm @ G_sing_norm.T)
        cos_pp = (G_plur_norm @ G_plur_norm.T)
        cos_sp = (G_sing_norm @ G_plur_norm.T)
        
        # 取上三角(不含对角线)
        n_s, n_p = len(sing_grads), len(plur_grads)
        cos_ss_vals = cos_ss[np.triu_indices(n_s, k=1)]
        cos_pp_vals = cos_pp[np.triu_indices(n_p, k=1)]
        cos_sp_vals = cos_sp.flatten()
        
        print(f"    Cosine: sing-sing={np.mean(cos_ss_vals):.4f}±{np.std(cos_ss_vals):.4f}, "
              f"plur-plur={np.mean(cos_pp_vals):.4f}±{np.std(cos_pp_vals):.4f}, "
              f"sing-plur={np.mean(cos_sp_vals):.4f}±{np.std(cos_sp_vals):.4f}")
        
        # 5. 梯度投影: sing梯度在plur方向上的投影
        # 即: sing梯度是否也有"推向plur"的效果?
        direction = compute_direction(model, tokenizer, device, li, NVA_PAIRS, n_pairs=8)
        if direction is not None:
            proj_sing = [np.dot(g, direction) for g in sing_grads]
            proj_plur = [np.dot(g, direction) for g in plur_grads]
            print(f"    Jv(direction): sing={np.mean(proj_sing):.4f}±{np.std(proj_sing):.4f}, "
                  f"plur={np.mean(proj_plur):.4f}±{np.std(proj_plur):.4f}")
            print(f"    Jv sign: sing>0: {sum(1 for x in proj_sing if x>0)}/{len(proj_sing)}, "
                  f"plur>0: {sum(1 for x in proj_plur if x>0)}/{len(proj_plur)}")


# ============================================================
# Experiment C: 控制验证 — autograd方向 vs mean difference方向
# ============================================================
def experiment_c(model, tokenizer, device, n_test=12):
    """
    ★★★ 关键验证: autograd方向是否比mean difference方向更适合控制?
    
    对比:
    1. v_mean = E[h|plur] - E[h|sing]  (传统的"语法方向")
    2. v_grad = mean(∂logit_gap/∂h_l)  (梯度方向)
    
    如果v_grad在控制上更有效 → 语法信息在梯度结构中, 不在均值差中
    """
    print("\n" + "="*70)
    print("Experiment C: Autograd Direction vs Mean Difference Direction")
    print("="*70)
    
    n_layers = model.config.n_layer
    sample_layers = [0, 3, 6, 9, 11]
    sample_layers = [l for l in sample_layers if l < n_layers]
    
    sv_id = tokenizer.encode(" runs", add_special_tokens=False)[0]
    pv_id = tokenizer.encode(" run", add_special_tokens=False)[0]
    
    betas = [0.5, 1.0, 2.0, 4.0, 8.0]
    
    for li in sample_layers:
        print(f"\n  --- Layer {li} ---")
        
        # 计算两种方向
        v_mean = compute_direction(model, tokenizer, device, li, NVA_PAIRS, n_pairs=8)
        if v_mean is None:
            continue
        
        # 收集梯度
        all_grads = []
        for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:n_test]):
            sent = f"The {sn}"
            grad, _ = compute_autograd_gradient(
                model, tokenizer, device, sent, li, sv_id, pv_id
            )
            if grad is not None:
                all_grads.append(grad)
        
        if len(all_grads) < 3:
            continue
        
        # v_grad = 平均梯度方向
        v_grad = np.mean(all_grads, axis=0)
        v_grad_norm = np.linalg.norm(v_grad)
        if v_grad_norm > 1e-10:
            v_grad = v_grad / v_grad_norm
        
        # 两种方向的cosine对齐
        cos_align = float(np.dot(v_mean, v_grad))
        print(f"    cos(v_mean, v_grad) = {cos_align:.4f}")
        
        # 测试控制效果
        layers = get_gpt2_layers(model)
        
        for v_name, v_dir in [("v_mean", v_mean), ("v_grad", v_grad)]:
            results_beta = []
            for beta in betas:
                flips = 0
                delta_gaps = []
                
                for i, (sn, pn, sv, pv) in enumerate(NVA_PAIRS[:n_test]):
                    sent = f"The {sn}"
                    toks = tokenizer(sent, return_tensors="pt").to(device)
                    
                    # 捕获h_l
                    h_captured = [None]
                    def capture_hook(module, input, output):
                        if isinstance(output, tuple):
                            h_captured[0] = output[0].detach().clone()
                        else:
                            h_captured[0] = output.detach().clone()
                        return output
                    
                    hook = layers[li].register_forward_hook(capture_hook)
                    with torch.no_grad():
                        out_base = model(**toks)
                        logits_base = out_base.logits[0, -1, :].float().cpu().numpy()
                    hook.remove()
                    
                    h_l = h_captured[0]
                    if h_l is None:
                        continue
                    
                    # 注入方向
                    h_mod = h_l.detach().clone()
                    direction_t = torch.tensor(v_dir, dtype=torch.float32, device=device)
                    h_mod[0, -1, :] += beta * direction_t
                    
                    def replace_hook(module, input, output, h_replace=h_mod):
                        if isinstance(output, tuple):
                            return (h_replace,) + output[1:]
                        return h_replace
                    
                    hook2 = layers[li].register_forward_hook(replace_hook)
                    with torch.no_grad():
                        out_mod = model(**toks)
                        logits_mod = out_mod.logits[0, -1, :].float().cpu().numpy()
                    hook2.remove()
                    
                    gap_base = logits_base[pv_id] - logits_base[sv_id]
                    gap_mod = logits_mod[pv_id] - logits_mod[sv_id]
                    delta_gap = gap_mod - gap_base
                    
                    delta_gaps.append(delta_gap)
                    if gap_base < 0 and gap_mod > 0:
                        flips += 1
                
                n_valid = len(delta_gaps)
                mean_delta = np.mean(delta_gaps) if delta_gaps else 0
                flip_rate = flips / max(n_valid, 1)
                results_beta.append((beta, mean_delta, flip_rate))
            
            print(f"    {v_name}:")
            for beta, mean_d, fr in results_beta:
                print(f"      β={beta:.1f}: Δgap={mean_d:+.4f}, flip_rate={fr:.2f}")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="a", choices=["a", "b", "c", "all"])
    parser.add_argument("--n_test", type=int, default=15)
    args = parser.parse_args()
    
    model, tokenizer, device = load_gpt2_float32()
    
    try:
        if args.exp in ["a", "all"]:
            results_a = experiment_a(model, tokenizer, device, n_test=args.n_test)
        
        if args.exp in ["b", "all"]:
            experiment_b(model, tokenizer, device, n_test=args.n_test)
        
        if args.exp in ["c", "all"]:
            experiment_c(model, tokenizer, device, n_test=args.n_test)
    finally:
        del model
        torch.cuda.empty_cache()
        gc.collect()
        print("\n[Phase70] Done. GPU memory released.")
