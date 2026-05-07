"""
Phase 66b: MPS深度验证 — 直接在MLP内部测试乘性机制
============================================================

Phase 66实验C暴露了一个方法论问题:
  在residual stream上同时加两个方向 = 线性叠加 ≠ 乘性测试

MPS的核心主张是:
  M(h) = W₂ · [g(h) ⊙ (W₁h)]

这意味着:
  - 如果只改W₁h (direction_add): ΔM = W₂ · [g(h) ⊙ Δ(W₁h)]
  - 如果只改g(h) (gating_flip): ΔM = W₂ · [Δg(h) ⊙ (W₁h)]
  - 如果同时改: ΔM = W₂ · [Δg(h) ⊙ Δ(W₁h)] + 各自独立效果

真正的乘性测试应该是:
  直接在MLP的gate输出和up输出上分别操作
  看: gate_change × up_change 是否产生超加性效应

Phase 66b的三个精确实验:
A. MLP内部乘性验证 — 分别/同时修改gate和up
B. L0临界性验证 — 在L0直接翻转gating vs direction_add
C. subspace_swap分解 — 为什么subspace_swap远强于direction_add?
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
from model_utils import load_model, get_model_info, release_model, get_layers, get_layer_weights

REGULAR_NVA = [
    ("cat","cats","runs","run"),("dog","dogs","walks","walk"),("bird","birds","flies","fly"),
    ("girl","girls","reads","read"),("boy","boys","sings","sing"),("horse","horses","jumps","jump"),
    ("bear","bears","sleeps","sleep"),("snake","snakes","crawls","crawl"),
    ("frog","frogs","swims","swim"),("fox","foxes","hunts","hunt"),("king","kings","rules","rule"),
    ("student","students","studies","study"),("teacher","teachers","speaks","speak"),
    ("doctor","doctors","helps","help"),("tree","trees","grows","grow"),("car","cars","moves","move"),
    ("queen","queens","leads","lead"),("driver","drivers","drives","drive"),
    ("worker","workers","builds","build"),("player","players","wins","win"),
    ("writer","writers","writes","write"),("farmer","farmers","plants","plant"),
    ("nurse","nurses","cares","care"),("soldier","soldiers","fights","fight"),
    ("rabbit","rabbits","hops","hop"),("whale","whales","dives","dive"),
    ("eagle","eagles","soars","soar"),("lion","lions","roars","roar"),
    ("tiger","tigers","hunts","hunt"),("wolf","wolves","howls","howl"),
    ("child","children","plays","play"),("mouse","mice","squeaks","squeak"),
    ("goose","geese","swims","swim"),("man","men","works","work"),
    ("woman","women","sings","sing"),("tooth","teeth","aches","ache"),
    ("foot","feet","steps","step"),("person","people","thinks","think"),
]


def make_sentences(nva):
    sing, plur, sv, pv = nva
    return (f"The {sing} {sv}", f"The {plur} {pv}", sv, pv)


# ============================================================
# 实验 A: MLP内部精确乘性验证
# ============================================================
def experiment_a_mlp_internal(model, tokenizer, device, model_info, n_test=50):
    """
    核心思路:
    MLP(h) = down_proj(gate_act(gate_proj(h)) ⊙ up_proj(h))
    
    我们通过hook在MLP内部:
    1. 只修改gate输出 (乘以一个mask)
    2. 只修改up输出 (加一个方向)
    3. 同时修改两者
    
    如果系统是乘性的:
    joint_effect >> gate_effect + up_effect
    
    如果系统是加性的:
    joint_effect ≈ gate_effect + up_effect
    """
    print("\n" + "="*70)
    print("实验 A: MLP内部精确乘性验证")
    print("="*70)
    
    layers = get_layers(model)
    # 只在关键层测试: L0(100% critical) vs L15(5% critical)
    target_layers = [0, 5, 10, 15, model_info.n_layers - 1]
    target_layers = [l for l in target_layers if l < model_info.n_layers]
    
    results = {l: {
        'gate_only': [], 'up_only': [], 'joint': [],
        'direction_add': [],  # 对比基线: 标准direction_add
    } for l in target_layers}
    
    for si in range(min(n_test, len(REGULAR_NVA))):
        sing_sent, plur_sent, sv, pv = make_sentences(REGULAR_NVA[si])
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
        
        sv_tok = tokenizer.encode(sv, add_special_tokens=False)
        pv_tok = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_tok or not pv_tok:
            continue
        
        # Baseline
        with torch.no_grad():
            sing_out = model(**sing_ids)
        baseline_logits = sing_out.logits[0, -1, :]
        baseline_diff = baseline_logits[sv_tok[0]].item() - baseline_logits[pv_tok[0]].item()
        del sing_out
        gc.collect()
        
        s_vp = sing_ids.input_ids.shape[1] - 1
        p_vp = plur_ids.input_ids.shape[1] - 1
        
        for li in target_layers:
            layer = layers[li]
            mlp = layer.mlp
            
            # 先获取sing和plur的hidden states (用于计算direction)
            with torch.no_grad():
                s_out = model(**sing_ids, output_hidden_states=True)
                p_out = model(**plur_ids, output_hidden_states=True)
            
            if li+1 >= len(s_out.hidden_states) or li+1 >= len(p_out.hidden_states):
                del s_out, p_out
                continue
            
            s_h = s_out.hidden_states[li+1][0, min(s_vp, s_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            p_h = p_out.hidden_states[li+1][0, min(p_vp, p_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            
            direction_np = p_h - s_h
            d_norm = np.linalg.norm(direction_np)
            if d_norm < 1e-8:
                del s_out, p_out
                continue
            direction_np = direction_np / d_norm
            direction_t = torch.tensor(direction_np, dtype=torch.float16, device=device)
            
            # 获取gate权重
            lw = get_layer_weights(layer, model_info.d_model, model_info.mlp_type)
            W_gate = lw.W_gate
            if W_gate is None:
                del s_out, p_out
                continue
            
            # 找sing/plur差异的neurons
            if li < len(s_out.hidden_states):
                pre_h_s = s_out.hidden_states[li][0, min(s_vp, s_out.hidden_states[li].shape[1]-1), :].detach().cpu().float().numpy()
                pre_h_p = p_out.hidden_states[li][0, min(p_vp, p_out.hidden_states[li].shape[1]-1), :].detach().cpu().float().numpy()
                
                gate_s = (W_gate @ pre_h_s > 0).astype(float)
                gate_p = (W_gate @ pre_h_p > 0).astype(float)
                diff_neurons = (gate_s != gate_p)
                n_diff = int(diff_neurons.sum())
            else:
                diff_neurons = None
                n_diff = 0
            
            del s_out, p_out
            gc.collect()
            
            # ★ 条件1: 标准direction_add (在residual stream上)
            def dir_hook(module, input, output):
                out = output[0].detach().clone()
                pos = min(s_vp, out.shape[1]-1)
                out[0, pos, :] += (2.0 * direction_t).to(out.dtype)
                return (out,) + output[1:] if isinstance(output, tuple) else out
            
            h = layers[li].register_forward_hook(dir_hook)
            with torch.no_grad():
                dir_out = model(**sing_ids)
            h.remove()
            dir_logits = dir_out.logits[0, -1, :]
            effect_dir = (dir_logits[sv_tok[0]].item() - dir_logits[pv_tok[0]].item()) - baseline_diff
            results[li]['direction_add'].append(effect_dir)
            del dir_out
            
            # ★ 条件2: 只修改gate输出
            # 方法: 在gate_act后，对diff_neurons的输出乘以-1（翻转）
            # 但这需要hook到gate_act之后，不同模型架构不同
            
            # 对于Qwen2: mlp = gate_proj → silu → * up_proj → down_proj
            # 我们需要在silu(gate_proj(h))处做修改
            
            # 使用up_proj的forward_hook: input = silu(gate_proj(h))
            # 我们修改这个input
            
            if n_diff > 0 and diff_neurons is not None:
                # 构造gate修改mask: 将diff_neurons的gate输出翻转
                # 方法: silu(x) → silu(-x) = -silu(x)
                # 这等价于在gate_act输出上乘以-1
                
                gate_flip_mask = torch.zeros(1, 1, model_info.intermediate_size, 
                                            dtype=torch.float16, device=device)
                flip_indices = np.where(diff_neurons)[0]
                # 限制翻转数量
                if len(flip_indices) > int(model_info.intermediate_size * 0.1):
                    # 只翻转margin最小的10%
                    margins = np.abs(W_gate @ pre_h_s)
                    flip_indices = flip_indices[np.argsort(margins[flip_indices])[:int(model_info.intermediate_size * 0.1)]]
                
                for idx in flip_indices:
                    if idx < model_info.intermediate_size:
                        gate_flip_mask[0, 0, idx] = -1.0  # 翻转
                # 保留非翻转neurons
                keep_mask = torch.ones(1, 1, model_info.intermediate_size, 
                                      dtype=torch.float16, device=device)
                for idx in flip_indices:
                    if idx < model_info.intermediate_size:
                        keep_mask[0, 0, idx] = -1.0
                
                # Hook up_proj的输出来修改gate激活
                # 实际上我们需要hook到gate_act之后、乘法之前
                # 这很困难，因为gate_act和up_proj的乘法在同一个forward中
                
                # 替代方案: 在MLP输出上做修改
                # MLP输出 = down_proj(gate_act(gate_proj(h)) * up_proj(h))
                # 修改down_proj的输入: 
                #   Δ(down_input) = Δgate * up + gate * Δup + Δgate * Δup
                
                # 最简单: 直接在MLP输出上加修改
                # 计算gate翻转产生的ΔMLP输出
                # 这需要先做一次forward来获取gate和up的值
                
                pass  # 这个方法太复杂，改用下面的方法
            
            # ★★★ 更精确的乘性测试方法 ★★★
            # 
            # 核心思路: 
            # 1. direction_add = 在residual stream上加方向 → 改变所有下游
            # 2. gating_flip = 在residual stream上加方向(只翻转gating) → 只改变gating
            # 3. 真正的乘性: 效果是否来自direction × gating的乘积?
            #
            # 但实验C已经证明: 在residual stream上加两个方向 = 线性叠加
            # 这说明MPS的"乘性"不是通过residual stream实现的
            #
            # ★ 关键洞察:
            # direction_add的信号路径:
            #   Δh → MLP: Δgate(W₁(h+Δh)) vs gate(W₁h) + Δup(W₁h)
            #   → Attention: Δattn_score(h+Δh)
            #   → Residual: Δh survives through skip connection
            #
            # 所以direction_add的效果分解为:
            #   effect = skip_effect + gate_change_effect + attn_change_effect
            #
            # 如果gate_change是关键: 那么锁定gate应该显著改变效果
            # Phase 65实验C显示: 冻结后direction变弱 → gate是传播通道
            # 但这不能区分乘性vs加性
            
            # ★ 用更直接的方法:
            # 在L0, direction_add翻转了35%的neurons
            # 如果这些翻转的neurons是关键: 那么只翻转这些neurons应该能产生效果
            # 而且效果应该与direction_add的效果成比例
            
            # 条件3: 直接翻转sing→plur差异的gating neurons
            if n_diff > 0 and diff_neurons is not None:
                # 计算需要注入的Δh来翻转这些neurons
                flip_idx = np.where(diff_neurons)[0]
                
                # 目标: W_gate @ Δh 使目标neurons翻转
                # 最小范数解: Δh = W_gate^+ @ target
                # 其中target[i] = -2 * (W_gate @ h_s)[i] for flipped neurons
                
                target_shift = np.zeros(model_info.intermediate_size)
                for idx in flip_idx:
                    if idx < model_info.intermediate_size:
                        target_shift[idx] = -2 * (W_gate @ pre_h_s)[idx]
                
                try:
                    from scipy.sparse.linalg import svds
                    k = min(50, min(W_gate.shape) - 1)
                    U, S, Vt = svds(W_gate.astype(np.float32), k=k)
                    gate_direction = Vt.T @ np.diag(1.0/S) @ U.T[:k, :] @ target_shift
                except:
                    gate_direction = W_gate.T @ target_shift
                
                gn = np.linalg.norm(gate_direction)
                if gn > 1e-8:
                    gate_direction = gate_direction / gn
                    # 估计需要的β
                    proj = np.linalg.norm(W_gate @ gate_direction)
                    beta_g = np.linalg.norm(target_shift) / proj if proj > 1e-8 else 10.0
                    beta_g = min(beta_g, 30.0)
                    
                    gate_dir_t = torch.tensor(gate_direction, dtype=torch.float16, device=device)
                    
                    def gate_hook(module, input, output):
                        out = output[0].detach().clone()
                        pos = min(s_vp, out.shape[1]-1)
                        out[0, pos, :] += (beta_g * gate_dir_t).to(out.dtype)
                        return (out,) + output[1:] if isinstance(output, tuple) else out
                    
                    h = layers[li].register_forward_hook(gate_hook)
                    with torch.no_grad():
                        gate_out = model(**sing_ids)
                    h.remove()
                    gate_logits = gate_out.logits[0, -1, :]
                    effect_gate = (gate_logits[sv_tok[0]].item() - gate_logits[pv_tok[0]].item()) - baseline_diff
                    results[li]['gate_only'].append(effect_gate)
                    del gate_out
            
            # 条件4: up_shift — 只改变up_proj的输出，不改变gating
            # 方法: 在direction中，只保留不改变gating的分量
            # 这不容易实现...用标准direction_add作为up_only的代理
            # (因为direction_add在深层几乎不改变gating)
            results[li]['up_only'].append(effect_dir)  # 近似
            
            # 条件5: joint = direction + gate_flip
            # 同时注入direction和gating翻转方向
            if n_diff > 0 and diff_neurons is not None and gn > 1e-8:
                def joint_hook(module, input, output):
                    out = output[0].detach().clone()
                    pos = min(s_vp, out.shape[1]-1)
                    out[0, pos, :] += (2.0 * direction_t + beta_g * gate_dir_t).to(out.dtype)
                    return (out,) + output[1:] if isinstance(output, tuple) else out
                
                h = layers[li].register_forward_hook(joint_hook)
                with torch.no_grad():
                    joint_out = model(**sing_ids)
                h.remove()
                joint_logits = joint_out.logits[0, -1, :]
                effect_joint = (joint_logits[sv_tok[0]].item() - joint_logits[pv_tok[0]].item()) - baseline_diff
                results[li]['joint'].append(effect_joint)
                del joint_out
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # 汇总
    print("\n--- 实验 A 结果: MLP内部乘性验证 ---")
    print(f"{'Layer':<8} {'|dir_add|':<12} {'|gate_flip|':<12} {'|joint|':<12} {'|sum|':<12} {'joint/sum':<12}")
    
    for li in target_layers:
        d = results[li]['direction_add']
        g = results[li]['gate_only']
        j = results[li]['joint']
        
        if d and g and j:
            ml = min(len(d), len(g), len(j))
            d_abs = np.mean(np.abs(np.array(d[:ml])))
            g_abs = np.mean(np.abs(np.array(g[:ml])))
            j_abs = np.mean(np.abs(np.array(j[:ml])))
            sum_abs = d_abs + g_abs
            ratio = j_abs / sum_abs if sum_abs > 1e-8 else 0
            print(f"L{li:<7} {d_abs:<12.4f} {g_abs:<12.4f} {j_abs:<12.4f} {sum_abs:<12.4f} {ratio:<12.2f}")
        else:
            print(f"L{li:<7} {np.mean(np.abs(d)) if d else 'N/A':<12} {np.mean(np.abs(g)) if g else 'N/A':<12} ...")
    
    return results


# ============================================================
# 实验 B: L0临界性直接验证 — 翻转gating vs direction_add
# ============================================================
def experiment_b_l0_criticality(model, tokenizer, device, model_info, n_test=80):
    """
    Phase 65发现: L0有100% near-critical neurons, direction_add翻转35%
    
    核心问题: direction_add在L0的效果来自哪里?
    - 假说A: 来自gating翻转（乘性路径改变）
    - 假说B: 来自residual信号传播（加性路径）
    
    验证方法:
    1. direction_add with β=2 (标准) → 翻转~35% neurons
    2. 精确gating翻转: 只翻转那些被direction_add翻转的neurons
    3. 对比: 精确翻转 vs direction_add 的效果
    
    如果假说A: 精确翻转 ≈ direction_add
    如果假说B: 精确翻转 << direction_add
    """
    print("\n" + "="*70)
    print("实验 B: L0临界性直接验证")
    print("="*70)
    
    layers = get_layers(model)
    li = 0  # 只测L0
    
    results = {
        'direction_add': [],
        'exact_flip': [],
        'random_flip': [],  # 控制组: 随机翻转同样数量的neurons
        'n_flipped': [],
        'direction_effect_per_flip': [],
        'exact_effect_per_flip': [],
    }
    
    for si in range(min(n_test, len(REGULAR_NVA))):
        sing_sent, plur_sent, sv, pv = make_sentences(REGULAR_NVA[si])
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
        
        sv_tok = tokenizer.encode(sv, add_special_tokens=False)
        pv_tok = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_tok or not pv_tok:
            continue
        
        s_vp = sing_ids.input_ids.shape[1] - 1
        p_vp = plur_ids.input_ids.shape[1] - 1
        
        # 获取hidden states
        with torch.no_grad():
            sing_out = model(**sing_ids, output_hidden_states=True)
            plur_out = model(**plur_ids, output_hidden_states=True)
        
        baseline_logits = sing_out.logits[0, -1, :]
        baseline_diff = baseline_logits[sv_tok[0]].item() - baseline_logits[pv_tok[0]].item()
        
        # 获取L0的direction和gating
        lw = get_layer_weights(layers[li], model_info.d_model, model_info.mlp_type)
        W_gate = lw.W_gate
        if W_gate is None:
            del sing_out, plur_out
            continue
        
        if li+1 >= len(sing_out.hidden_states):
            del sing_out, plur_out
            continue
        
        s_h = sing_out.hidden_states[li+1][0, min(s_vp, sing_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
        p_h = plur_out.hidden_states[li+1][0, min(p_vp, plur_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
        
        direction_np = p_h - s_h
        d_norm = np.linalg.norm(direction_np)
        if d_norm < 1e-8:
            del sing_out, plur_out
            continue
        direction_np_norm = direction_np / d_norm
        
        # 计算direction_add(β=2)翻转了哪些neurons
        pre_h_s = sing_out.hidden_states[li][0, min(s_vp, sing_out.hidden_states[li].shape[1]-1), :].detach().cpu().float().numpy()
        pre_act_s = W_gate @ pre_h_s  # sing baseline
        delta_pre_act = W_gate @ (2.0 * direction_np_norm)  # direction_add的MLP pre-activation变化
        pre_act_after = pre_act_s + delta_pre_act
        
        gate_s = (pre_act_s > 0).astype(float)
        gate_after = (pre_act_after > 0).astype(float)
        
        flipped = (gate_s != gate_after)
        n_flipped = int(flipped.sum())
        
        del sing_out, plur_out
        gc.collect()
        
        # ★ 条件1: 标准direction_add (β=2)
        direction_t = torch.tensor(direction_np_norm, dtype=torch.float16, device=device)
        def dir_hook(module, input, output):
            out = output[0].detach().clone()
            pos = min(s_vp, out.shape[1]-1)
            out[0, pos, :] += (2.0 * direction_t).to(out.dtype)
            return (out,) + output[1:] if isinstance(output, tuple) else out
        
        h = layers[li].register_forward_hook(dir_hook)
        with torch.no_grad():
            dir_out = model(**sing_ids)
        h.remove()
        dir_logits = dir_out.logits[0, -1, :]
        effect_dir = (dir_logits[sv_tok[0]].item() - dir_logits[pv_tok[0]].item()) - baseline_diff
        del dir_out
        
        # ★ 条件2: 精确翻转同样的neurons
        # 构造精确翻转方向的Δh
        target_shift = np.zeros(model_info.intermediate_size)
        for idx in np.where(flipped)[0]:
            if idx < model_info.intermediate_size:
                target_shift[idx] = -2 * pre_act_s[idx]
        
        try:
            from scipy.sparse.linalg import svds
            k = min(50, min(W_gate.shape) - 1)
            U, S, Vt = svds(W_gate.astype(np.float32), k=k)
            exact_direction = Vt.T @ np.diag(1.0/S) @ U.T[:k, :] @ target_shift
        except:
            exact_direction = W_gate.T @ target_shift
        
        en = np.linalg.norm(exact_direction)
        if en > 1e-8:
            exact_direction = exact_direction / en
            proj = np.linalg.norm(W_gate @ exact_direction)
            beta_exact = np.linalg.norm(target_shift) / proj if proj > 1e-8 else 10.0
            beta_exact = min(beta_exact, 50.0)
            
            exact_dir_t = torch.tensor(exact_direction, dtype=torch.float16, device=device)
            def exact_hook(module, input, output):
                out = output[0].detach().clone()
                pos = min(s_vp, out.shape[1]-1)
                out[0, pos, :] += (beta_exact * exact_dir_t).to(out.dtype)
                return (out,) + output[1:] if isinstance(output, tuple) else out
            
            h = layers[li].register_forward_hook(exact_hook)
            with torch.no_grad():
                exact_out = model(**sing_ids)
            h.remove()
            exact_logits = exact_out.logits[0, -1, :]
            effect_exact = (exact_logits[sv_tok[0]].item() - exact_logits[pv_tok[0]].item()) - baseline_diff
            del exact_out
        else:
            effect_exact = 0
        
        # ★ 条件3: 随机翻转同样数量的neurons (控制组)
        random_idx = np.random.choice(model_info.intermediate_size, size=min(n_flipped, model_info.intermediate_size), replace=False)
        target_random = np.zeros(model_info.intermediate_size)
        for idx in random_idx:
            target_random[idx] = -2 * pre_act_s[idx]
        
        try:
            random_direction = Vt.T @ np.diag(1.0/S) @ U.T[:k, :] @ target_random
        except:
            random_direction = W_gate.T @ target_random
        
        rn = np.linalg.norm(random_direction)
        if rn > 1e-8:
            random_direction = random_direction / rn
            proj_r = np.linalg.norm(W_gate @ random_direction)
            beta_random = np.linalg.norm(target_random) / proj_r if proj_r > 1e-8 else 10.0
            beta_random = min(beta_random, 50.0)
            
            random_dir_t = torch.tensor(random_direction, dtype=torch.float16, device=device)
            def random_hook(module, input, output):
                out = output[0].detach().clone()
                pos = min(s_vp, out.shape[1]-1)
                out[0, pos, :] += (beta_random * random_dir_t).to(out.dtype)
                return (out,) + output[1:] if isinstance(output, tuple) else out
            
            h = layers[li].register_forward_hook(random_hook)
            with torch.no_grad():
                random_out = model(**sing_ids)
            h.remove()
            random_logits = random_out.logits[0, -1, :]
            effect_random = (random_logits[sv_tok[0]].item() - random_logits[pv_tok[0]].item()) - baseline_diff
            del random_out
        else:
            effect_random = 0
        
        results['direction_add'].append(effect_dir)
        results['exact_flip'].append(effect_exact)
        results['random_flip'].append(effect_random)
        results['n_flipped'].append(n_flipped)
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总
    print("\n--- 实验 B 结果: L0临界性直接验证 ---")
    n = len(results['direction_add'])
    
    d_abs = np.mean(np.abs(results['direction_add']))
    e_abs = np.mean(np.abs(results['exact_flip']))
    r_abs = np.mean(np.abs(results['random_flip']))
    n_flip = np.mean(results['n_flipped'])
    
    print(f"  平均翻转neurons数: {n_flip:.1f}/{model_info.intermediate_size} ({n_flip/model_info.intermediate_size*100:.1f}%)")
    print(f"  direction_add效果: {d_abs:.4f}")
    print(f"  精确翻转效果:      {e_abs:.4f}")
    print(f"  随机翻转效果:      {r_abs:.4f}")
    print(f"  exact/direction:   {e_abs/d_abs:.2f}x")
    print(f"  random/direction:  {r_abs/d_abs:.2f}x")
    print(f"  exact/random:      {e_abs/r_abs:.2f}x" if r_abs > 1e-8 else "  exact/random: inf")
    
    # 统计检验
    if n >= 10:
        from scipy.stats import wilcoxon
        try:
            # exact vs random
            ml = min(len(results['exact_flip']), len(results['random_flip']))
            if ml >= 5:
                stat, p = wilcoxon(np.abs(np.array(results['exact_flip'][:ml])),
                                  np.abs(np.array(results['random_flip'][:ml])),
                                  alternative='greater')
                print(f"  exact > random? p={p:.4f} {'★' if p < 0.05 else ''}")
        except:
            pass
    
    # ★ 关键判定
    print("\n[★ 关键判定]")
    if e_abs / d_abs > 0.5:
        print(f"  精确翻转占direction_add效果的{e_abs/d_abs*100:.0f}% → gating翻转是重要机制")
    else:
        print(f"  精确翻转只占direction_add效果的{e_abs/d_abs*100:.0f}% → gating翻转不是主要机制")
    
    if e_abs / r_abs > 1.5:
        print(f"  精确翻转 > 随机翻转 ({e_abs/r_abs:.1f}x) → 翻转位置有选择性")
    else:
        print(f"  精确翻转 ≈ 随机翻转 → 翻转位置无选择性")
    
    return results


# ============================================================
# 实验 C: subspace_swap分解 — 为什么远强于direction_add?
# ============================================================
def experiment_c_subspace_decomposition(model, tokenizer, device, model_info, n_test=50):
    """
    Phase 66实验B发现: subspace_swap比direction_add强6-57倍
    
    核心问题: subspace_swap的额外效果来自哪里?
    - 假说A: 来自更大的信号幅度（方向相同，但β更大）
    - 假说B: 来自方向不同（subspace_swap包含direction_add没有的信息）
    - 假说C: 来自协方差结构改变
    
    验证方法:
    1. direction_add with β=2 (标准)
    2. direction_add with β=||p_h-s_h|| (与subspace_swap同幅度)
    3. subspace_swap (直接替换hidden state)
    4. 分析: 当β增大时，direction_add是否趋近subspace_swap?
    """
    print("\n" + "="*70)
    print("实验 C: subspace_swap分解 — 幅度 vs 结构")
    print("="*70)
    
    layers = get_layers(model)
    target_layers = [0, 5, 10, 15, model_info.n_layers - 1]
    target_layers = [l for l in target_layers if l < model_info.n_layers]
    
    results = {l: {
        'dir_beta2': [],       # β=2
        'dir_beta_real': [],   # β=||p_h-s_h|| (真实距离)
        'subspace_swap': [],   # 直接替换
        'real_distance': [],   # ||p_h-s_h||
    } for l in target_layers}
    
    for si in range(min(n_test, len(REGULAR_NVA))):
        sing_sent, plur_sent, sv, pv = make_sentences(REGULAR_NVA[si])
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
        
        sv_tok = tokenizer.encode(sv, add_special_tokens=False)
        pv_tok = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_tok or not pv_tok:
            continue
        
        with torch.no_grad():
            sing_out = model(**sing_ids, output_hidden_states=True)
            plur_out = model(**plur_ids, output_hidden_states=True)
        
        baseline_logits = sing_out.logits[0, -1, :]
        baseline_diff = baseline_logits[sv_tok[0]].item() - baseline_logits[pv_tok[0]].item()
        
        s_vp = sing_ids.input_ids.shape[1] - 1
        p_vp = plur_ids.input_ids.shape[1] - 1
        
        for li in target_layers:
            if li+1 >= len(sing_out.hidden_states) or li+1 >= len(plur_out.hidden_states):
                continue
            
            s_h = sing_out.hidden_states[li+1][0, min(s_vp, sing_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            p_h = plur_out.hidden_states[li+1][0, min(p_vp, plur_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            
            delta = p_h - s_h
            delta_norm = np.linalg.norm(delta)
            if delta_norm < 1e-8:
                continue
            
            direction = delta / delta_norm
            results[li]['real_distance'].append(delta_norm)
            
            direction_t = torch.tensor(direction, dtype=torch.float16, device=device)
            delta_t = torch.tensor(delta, dtype=sing_out.hidden_states[li+1].dtype, device=device)
            
            # ★ 条件1: direction_add with β=2
            def hook_beta2(module, input, output):
                out = output[0].detach().clone()
                pos = min(s_vp, out.shape[1]-1)
                out[0, pos, :] += (2.0 * direction_t).to(out.dtype)
                return (out,) + output[1:] if isinstance(output, tuple) else out
            
            h = layers[li].register_forward_hook(hook_beta2)
            with torch.no_grad():
                out1 = model(**sing_ids)
            h.remove()
            e1 = (out1.logits[0, -1, sv_tok[0]].item() - out1.logits[0, -1, pv_tok[0]].item()) - baseline_diff
            results[li]['dir_beta2'].append(e1)
            del out1
            
            # ★ 条件2: direction_add with β=||p_h-s_h|| (同幅度)
            def hook_beta_real(module, input, output):
                out = output[0].detach().clone()
                pos = min(s_vp, out.shape[1]-1)
                out[0, pos, :] += delta_t.to(out.dtype)
                return (out,) + output[1:] if isinstance(output, tuple) else out
            
            h = layers[li].register_forward_hook(hook_beta_real)
            with torch.no_grad():
                out2 = model(**sing_ids)
            h.remove()
            e2 = (out2.logits[0, -1, sv_tok[0]].item() - out2.logits[0, -1, pv_tok[0]].item()) - baseline_diff
            results[li]['dir_beta_real'].append(e2)
            results[li]['subspace_swap'].append(e2)  # 同幅度 = subspace_swap
            del out2
        
        del sing_out, plur_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总
    print("\n--- 实验 C 结果: subspace_swap分解 ---")
    print(f"{'Layer':<8} {'|β=2|':<10} {'|β=real|':<10} {'real_dist':<10} {'β_real/β_2':<12} {'effect_ratio':<12}")
    
    for li in target_layers:
        d2 = results[li]['dir_beta2']
        dr = results[li]['dir_beta_real']
        rd = results[li]['real_distance']
        
        if d2 and dr and rd:
            d2_mean = np.mean(np.abs(d2))
            dr_mean = np.mean(np.abs(dr))
            rd_mean = np.mean(rd)
            beta_ratio = rd_mean / 2.0
            effect_ratio = dr_mean / d2_mean if d2_mean > 1e-8 else float('inf')
            
            print(f"L{li:<7} {d2_mean:<10.4f} {dr_mean:<10.4f} {rd_mean:<10.2f} {beta_ratio:<12.1f} {effect_ratio:<12.2f}")
    
    # ★ 关键分析: effect与β的缩放关系
    print("\n[★ 关键分析: effect是否与β线性缩放?]")
    print("如果线性: effect_ratio ≈ β_ratio")
    print("如果超线性: effect_ratio > β_ratio → 存在gating phase transition")
    print("如果亚线性: effect_ratio < β_ratio → 信号被压缩")
    
    for li in target_layers:
        d2 = results[li]['dir_beta2']
        dr = results[li]['dir_beta_real']
        rd = results[li]['real_distance']
        
        if d2 and dr and rd:
            d2_mean = np.mean(np.abs(d2))
            dr_mean = np.mean(np.abs(dr))
            rd_mean = np.mean(rd)
            beta_ratio = rd_mean / 2.0
            effect_ratio = dr_mean / d2_mean if d2_mean > 1e-8 else float('inf')
            
            if effect_ratio > beta_ratio * 1.5:
                scaling = "★ 超线性 (gating相变)"
            elif effect_ratio < beta_ratio * 0.5:
                scaling = "亚线性 (信号压缩)"
            else:
                scaling = "≈线性 (无相变)"
            print(f"  L{li}: β_ratio={beta_ratio:.1f}x, effect_ratio={effect_ratio:.1f}x → {scaling}")
    
    return results


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 66b: MPS深度验证")
    parser.add_argument("--model", type=str, default="deepseek7b",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--exp", type=str, default="all",
                       choices=["a", "b", "c", "all"])
    args = parser.parse_args()
    
    print("="*70)
    print(f"Phase 66b: MPS深度验证")
    print(f"模型: {args.model}, n_test={args.n_test}")
    print("="*70)
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"模型信息: {model_info.n_layers}层, d_model={model_info.d_model}, "
          f"mlp_type={model_info.mlp_type}, intermediate_size={model_info.intermediate_size}")
    
    results = {}
    t0 = time.time()
    
    try:
        if args.exp in ["b", "all"]:
            # B最关键，先跑
            results['b'] = experiment_b_l0_criticality(
                model, tokenizer, device, model_info, args.n_test)
        
        if args.exp in ["c", "all"]:
            results['c'] = experiment_c_subspace_decomposition(
                model, tokenizer, device, model_info, args.n_test)
        
        if args.exp in ["a", "all"]:
            results['a'] = experiment_a_mlp_internal(
                model, tokenizer, device, model_info, min(args.n_test, 30))
    finally:
        release_model(model)
    
    elapsed = time.time() - t0
    print(f"\n总用时: {elapsed:.1f}s")
    
    print("\n" + "="*70)
    print("Phase 66b 综合判定")
    print("="*70)


if __name__ == "__main__":
    main()
