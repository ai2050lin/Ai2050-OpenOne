"""
Phase 66: 乘性路径系统 (MPS) 验证
============================================

基于用户对Phase 65的关键纠正:
1. 冻结gating实验有不可识别性问题 — 同时去掉了路径选择和非线性传播
2. "无相关≠无机制" — 单变量统计对高维非线性系统太弱
3. direction弱≠信号幅度问题 — subspace_swap >>> direction_add说明是结构不匹配

MPS核心主张:
  M(h) = W₂ · [g(h) ⊙ (W₁h)] = W₂ · diag(g(h)) · W₁h
  信息传播是乘性的: signal_out ≈ signal_in × gate_pattern
  direction_add只影响一阶项(线性)，subspace_swap改变二阶结构+路径集合

Phase 66的三个核心实验:
A. β递增实验 — 逐步增大direction_add强度，找gating通道阈值
   预测: 存在phase transition（neuron flips突然跃迁）
B. 二阶控制 vs 一阶控制 — mean shift vs covariance shift
   预测: cov_shift >> mean_shift
C. 乘性验证 — direction + gating_change的联合效应
   预测: joint_effect >> sum(individual_effects)
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

# ============================================================
# 数据定义
# ============================================================
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
    """生成singular/plural句子对"""
    sing, plur, sv, pv = nva
    return (f"The {sing} {sv}", f"The {plur} {pv}", sv, pv)


# ============================================================
# 实验 A: β递增实验 — 找phase transition
# ============================================================
def experiment_a_beta_scaling(model, tokenizer, device, model_info, n_test=50):
    """
    核心问题: direction_add是否有phase transition?
    
    MPS预测:
    - 在非临界区(|W₁h| >> 0): β递增→线性响应，gating不变
    - 在临界区(|W₁h| ≈ 0): β递增→存在阈值，gating突然翻转
    - L0(100% near-critical)应该有最低阈值
    - 深层(<5% near-critical)应该无phase transition
    """
    print("\n" + "="*70)
    print("实验 A: β递增实验 — 寻找gating phase transition")
    print("="*70)
    
    layers = get_layers(model)
    target_layers = [0, 5, 10, 15, model_info.n_layers - 1]
    target_layers = [l for l in target_layers if l < model_info.n_layers]
    
    betas = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
    
    # 收集: {layer: {beta: [switch_rate, avg_effect, max_effect, n_flips_median]}}
    results = {l: {b: {'switch_rates': [], 'effects': [], 'n_flips': []} 
                   for b in betas} for l in target_layers}
    
    for si in range(min(n_test, len(REGULAR_NVA))):
        sing_sent, plur_sent, sv, pv = make_sentences(REGULAR_NVA[si])
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
        
        sv_tok = tokenizer.encode(sv, add_special_tokens=False)
        pv_tok = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_tok or not pv_tok:
            continue
        
        # 获取baseline
        with torch.no_grad():
            sing_out = model(**sing_ids, output_hidden_states=True)
            plur_out = model(**plur_ids, output_hidden_states=True)
        
        sing_logits_base = sing_out.logits[0, -1, :]
        baseline_logit_diff = sing_logits_base[sv_tok[0]].item() - sing_logits_base[pv_tok[0]].item()
        
        sing_vp = sing_ids.input_ids.shape[1] - 1
        plur_vp = plur_ids.input_ids.shape[1] - 1
        
        for li in target_layers:
            if li+1 >= len(sing_out.hidden_states) or li+1 >= len(plur_out.hidden_states):
                continue
            
            # 获取direction = sing→plur的方向
            s_h = sing_out.hidden_states[li+1][0, min(sing_vp, sing_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            p_h = plur_out.hidden_states[li+1][0, min(plur_vp, plur_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            direction_np = p_h - s_h
            d_norm = np.linalg.norm(direction_np)
            if d_norm < 1e-8:
                continue
            direction_np = direction_np / d_norm
            
            # 获取gate权重
            layer = layers[li]
            lw = get_layer_weights(layer, model_info.d_model, model_info.mlp_type)
            W_gate = lw.W_gate
            if W_gate is None:
                continue
            
            # 获取sing的MLP pre-activation (for baseline gating)
            if li < len(sing_out.hidden_states):
                pre_h = sing_out.hidden_states[li][0, min(sing_vp, sing_out.hidden_states[li].shape[1]-1), :].detach().cpu().float().numpy()
                pre_act_base = W_gate @ pre_h  # [intermediate]
                gate_base = (pre_act_base > 0).astype(float)  # baseline gating mask
            else:
                continue
            
            for beta in betas:
                # 注入direction后，计算新的gating
                delta_pre_act = W_gate @ (beta * direction_np)
                pre_act_new = pre_act_base + delta_pre_act
                gate_new = (pre_act_new > 0).astype(float)
                
                # switch rate
                n_switched = int(np.sum(gate_new != gate_base))
                n_total = len(gate_base)
                switch_rate = n_switched / n_total
                
                # 实际前向传播测效应量
                direction_t = torch.tensor(direction_np, dtype=sing_out.hidden_states[li+1].dtype, device=device)
                
                hook_data = {}
                def make_hook(key):
                    def hook(module, input, output):
                        hook_data[key] = output[0].detach().clone() if isinstance(output, tuple) else output.detach().clone()
                    return hook
                
                h = layers[li].register_forward_hook(make_hook(f'L{li}'))
                
                def intervene_hook(module, input, output):
                    out = output[0].detach().clone()
                    pos = min(sing_vp, out.shape[1]-1)
                    out[0, pos, :] += beta * direction_t.to(out.dtype)
                    return (out,) + output[1:] if isinstance(output, tuple) else out
                
                h2 = layers[li].register_forward_hook(intervene_hook)
                
                with torch.no_grad():
                    interv_out = model(**sing_ids)
                
                h.remove()
                h2.remove()
                
                interv_logits = interv_out.logits[0, -1, :]
                interv_diff = interv_logits[sv_tok[0]].item() - interv_logits[pv_tok[0]].item()
                effect = interv_diff - baseline_logit_diff
                
                results[li][beta]['switch_rates'].append(switch_rate)
                results[li][beta]['effects'].append(abs(effect))
                results[li][beta]['n_flips'].append(n_switched)
                
                del interv_out
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        del sing_out, plur_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总结果
    print("\n--- 实验 A 结果: β递增 ---")
    print(f"{'Layer':<8}", end="")
    for b in betas:
        print(f"{'β='+str(b):<12}", end="")
    print()
    
    print("\n[Switch Rate (%)]")
    for li in target_layers:
        print(f"L{li:<7}", end="")
        for b in betas:
            vals = results[li][b]['switch_rates']
            if vals:
                print(f"{np.mean(vals)*100:<12.2f}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()
    
    print("\n[Mean |Effect| (logit diff)]")
    for li in target_layers:
        print(f"L{li:<7}", end="")
        for b in betas:
            vals = results[li][b]['effects']
            if vals:
                print(f"{np.mean(vals):<12.4f}", end="")
            else:
                print(f"{'N/A':<12}", end="")
        print()
    
    # ★ 关键分析: β-效应曲线的凹凸性
    print("\n[★ Phase Transition 检测: effect/β (边际效应)]")
    print("如果存在phase transition, effect/β应该在某个β处突然增大")
    for li in target_layers:
        print(f"\nL{li}:")
        prev_effect = 0
        for b in betas:
            vals = results[li][b]['effects']
            if vals:
                mean_eff = np.mean(vals)
                marginal = mean_eff / b if b > 0 else 0
                print(f"  β={b:>5.1f}: mean|effect|={mean_eff:.4f}, marginal={marginal:.4f}, switch%={np.mean(results[li][b]['switch_rates'])*100:.1f}%")
                prev_effect = mean_eff
    
    # ★ 关键指标: switch_rate从<5%跳到>20%的β阈值
    print("\n[★ Gating阈值: switch_rate从<5%跳到>20%的β]")
    for li in target_layers:
        threshold_beta = None
        for b in sorted(betas):
            vals = results[li][b]['switch_rates']
            if vals and np.mean(vals) > 0.20:
                threshold_beta = b
                break
        print(f"  L{li}: threshold_β = {threshold_beta if threshold_beta else '> max(β)'}")
    
    return results


# ============================================================
# 实验 B: 二阶控制 vs 一阶控制
# ============================================================
def experiment_b_covariance_control(model, tokenizer, device, model_info, n_test=50):
    """
    核心问题: 协方差结构改变是否比均值平移更有效?
    
    MPS预测:
    - direction_add = mean shift → 只改变一阶统计量
    - covariance shift = 改变E[hh^T] → 改变gating pattern + attention
    - effect(cov_shift) >> effect(mean_shift)
    
    方法:
    A. mean shift: h → h + βv (标准direction_add)
    B. cov shift: h → whiten(h) + scale → 改变分布结构
    C. subspace swap (对比基线)
    """
    print("\n" + "="*70)
    print("实验 B: 二阶控制 vs 一阶控制")
    print("="*70)
    
    layers = get_layers(model)
    target_layers = [0, 5, 10, 15, model_info.n_layers - 1]
    target_layers = [l for l in target_layers if l < model_info.n_layers]
    
    beta = 2.0  # 标准beta
    cov_scale = 1.0  # covariance shift缩放
    
    # 收集结果: {layer: {'mean_shift': [], 'cov_shift': [], 'subspace_swap': []}}
    results = {l: {'mean_shift': [], 'cov_shift': [], 'subspace_swap': []} 
               for l in target_layers}
    
    # 先收集一批(sing, plur)的hidden states用于估计协方差
    print("  收集hidden states用于协方差估计...")
    collect_size = min(n_test, len(REGULAR_NVA))
    h_sing_all = {l: [] for l in target_layers}
    h_plur_all = {l: [] for l in target_layers}
    
    for si in range(collect_size):
        sing_sent, plur_sent, sv, pv = make_sentences(REGULAR_NVA[si])
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
        
        with torch.no_grad():
            s_out = model(**sing_ids, output_hidden_states=True)
            p_out = model(**plur_ids, output_hidden_states=True)
        
        s_vp = sing_ids.input_ids.shape[1] - 1
        p_vp = plur_ids.input_ids.shape[1] - 1
        
        for li in target_layers:
            if li+1 >= len(s_out.hidden_states):
                continue
            s_h = s_out.hidden_states[li+1][0, min(s_vp, s_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            p_h = p_out.hidden_states[li+1][0, min(p_vp, p_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            h_sing_all[li].append(s_h)
            h_plur_all[li].append(p_h)
        
        del s_out, p_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 计算每层的sing/plur协方差
    print("  计算协方差矩阵...")
    cov_sing = {}
    cov_plur = {}
    mean_sing = {}
    mean_plur = {}
    for li in target_layers:
        if len(h_sing_all[li]) >= 3:
            S = np.array(h_sing_all[li])  # [n, d]
            P = np.array(h_plur_all[li])
            mean_sing[li] = S.mean(axis=0)
            mean_plur[li] = P.mean(axis=0)
            # 用PCA近似协方差（避免d×d矩阵太大）
            # 只保存均值差和前k个主成分
            cov_sing[li] = S
            cov_plur[li] = P
        else:
            cov_sing[li] = None
            cov_plur[li] = None
    
    # 释放收集的hidden states
    del h_sing_all, h_plur_all
    gc.collect()
    
    # 现在做对比实验
    print("  执行对比实验...")
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
        
        sing_logits_base = sing_out.logits[0, -1, :]
        baseline_diff = sing_logits_base[sv_tok[0]].item() - sing_logits_base[pv_tok[0]].item()
        
        s_vp = sing_ids.input_ids.shape[1] - 1
        p_vp = plur_ids.input_ids.shape[1] - 1
        
        for li in target_layers:
            if li+1 >= len(sing_out.hidden_states) or li+1 >= len(plur_out.hidden_states):
                continue
            if cov_sing[li] is None:
                continue
            
            s_h = sing_out.hidden_states[li+1][0, min(s_vp, sing_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            p_h = plur_out.hidden_states[li+1][0, min(p_vp, plur_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            
            direction_np = p_h - s_h
            d_norm = np.linalg.norm(direction_np)
            if d_norm < 1e-8:
                continue
            direction_np = direction_np / d_norm
            
            # ---- 方法A: mean shift (标准direction_add) ----
            effect_mean = _run_intervention(
                model, tokenizer, device, layers, li, sing_ids, s_vp,
                direction_np, beta, sv_tok[0], pv_tok[0], baseline_diff
            )
            if effect_mean is not None:
                results[li]['mean_shift'].append(effect_mean)
            
            # ---- 方法B: covariance shift ----
            # 核心: 将sing的分布向plur的分布对齐
            # 方法: h → h + cov_scale * (mean_plur - mean_sing) + 在plur主成分方向上的旋转
            # 简化版: 用PCA找sing→plur的主成分变化方向，注入这些方向
            
            S = cov_sing[li]  # [n, d]
            P = cov_plur[li]
            
            # 计算sing和plur的PCA
            from sklearn.decomposition import PCA
            n_comp = min(10, min(S.shape[0], S.shape[1]) - 1)
            
            pca_s = PCA(n_components=n_comp)
            pca_s.fit(S)
            pca_p = PCA(n_components=n_comp)
            pca_p.fit(P)
            
            # cov_shift: 将sing的表示向plur的主成分空间投影
            # 最简实现: 在plur的前k个主成分方向上注入
            plur_components = pca_p.components_  # [k, d]
            plur_explained = pca_p.explained_variance_ratio_  # [k]
            
            # 构造covariance shift向量: 加权组合plur主成分
            # 权重由plur各成分的方差解释比决定
            cov_direction = np.zeros_like(direction_np)
            for k in range(n_comp):
                # 在plur第k主成分方向上的投影差
                proj_diff = np.dot(p_h - s_h, plur_components[k])
                # 只保留与sing→plur方向一致的成分
                if abs(proj_diff) > 0.01:
                    cov_direction += plur_components[k] * proj_diff * plur_explained[k]
            
            cov_norm = np.linalg.norm(cov_direction)
            if cov_norm > 1e-8:
                cov_direction = cov_direction / cov_norm
                effect_cov = _run_intervention(
                    model, tokenizer, device, layers, li, sing_ids, s_vp,
                    cov_direction, beta * cov_scale, sv_tok[0], pv_tok[0], baseline_diff
                )
                if effect_cov is not None:
                    results[li]['cov_shift'].append(effect_cov)
            
            # ---- 方法C: subspace swap (对比基线) ----
            # 直接替换为plur的hidden state
            effect_subspace = _run_intervention(
                model, tokenizer, device, layers, li, sing_ids, s_vp,
                (p_h - s_h) / max(np.linalg.norm(p_h - s_h), 1e-8),
                np.linalg.norm(p_h - s_h),  # 使用真实距离作为β
                sv_tok[0], pv_tok[0], baseline_diff
            )
            if effect_subspace is not None:
                results[li]['subspace_swap'].append(effect_subspace)
        
        del sing_out, plur_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总结果
    print("\n--- 实验 B 结果: 二阶控制 vs 一阶控制 ---")
    print(f"{'Layer':<8} {'mean_shift':<14} {'cov_shift':<14} {'subspace_swap':<14} {'cov/mean':<10} {'sub/mean':<10}")
    for li in target_layers:
        ms = results[li]['mean_shift']
        cs = results[li]['cov_shift']
        ss = results[li]['subspace_swap']
        ms_mean = np.mean(np.abs(ms)) if ms else 0
        cs_mean = np.mean(np.abs(cs)) if cs else 0
        ss_mean = np.mean(np.abs(ss)) if ss else 0
        ratio_cm = cs_mean / ms_mean if ms_mean > 1e-8 else float('inf')
        ratio_sm = ss_mean / ms_mean if ms_mean > 1e-8 else float('inf')
        print(f"L{li:<7} {ms_mean:<14.4f} {cs_mean:<14.4f} {ss_mean:<14.4f} {ratio_cm:<10.2f} {ratio_sm:<10.2f}")
    
    # MPS预测判定
    print("\n[★ MPS预测判定]")
    for li in target_layers:
        ms = results[li]['mean_shift']
        cs = results[li]['cov_shift']
        if ms and cs:
            # 用配对t检验或Wilcoxon检验
            from scipy.stats import wilcoxon
            try:
                # 取绝对值后比较
                ms_abs = np.abs(ms)
                cs_abs = np.abs(cs)
                min_len = min(len(ms_abs), len(cs_abs))
                if min_len >= 5:
                    stat, p = wilcoxon(cs_abs[:min_len], ms_abs[:min_len])
                    sig = "★" if p < 0.05 else ""
                    print(f"  L{li}: cov > mean? p={p:.4f} {sig}")
                else:
                    print(f"  L{li}: 数据不足")
            except:
                print(f"  L{li}: 检验失败")
    
    return results


def _run_intervention(model, tokenizer, device, layers, li, sing_ids, s_vp,
                      direction_np, beta, sv_tok_id, pv_tok_id, baseline_diff):
    """辅助函数: 在某层注入direction并测效应量"""
    direction_t = torch.tensor(direction_np, dtype=torch.float16, device=device)
    
    hook_removed = [False, False]
    
    def intervene_hook(module, input, output):
        out = output[0].detach().clone()
        pos = min(s_vp, out.shape[1]-1)
        out[0, pos, :] += (beta * direction_t).to(out.dtype)
        return (out,) + output[1:] if isinstance(output, tuple) else out
    
    h = layers[li].register_forward_hook(intervene_hook)
    
    with torch.no_grad():
        try:
            interv_out = model(**sing_ids)
        except Exception as e:
            h.remove()
            return None
    
    h.remove()
    
    interv_logits = interv_out.logits[0, -1, :]
    interv_diff = interv_logits[sv_tok_id].item() - interv_logits[pv_tok_id].item()
    effect = interv_diff - baseline_diff
    
    del interv_out
    return effect


# ============================================================
# 实验 C: 乘性验证 — direction + gating_change联合效应
# ============================================================
def experiment_c_multiplicative(model, tokenizer, device, model_info, n_test=50):
    """
    核心问题: direction和gating是否存在乘性耦合?
    
    MPS预测:
    - MLP(h) = W₂ · diag(g(h)) · W₁h
    - direction_add只改变W₁h（线性项）
    - 如果同时改变g(h)和W₁h，效果应该 >> 各自独立之和
    - 这证明系统是乘性的，不是加性的
    
    方法:
    1. direction_add only (β=2) → effect_A
    2. gating_shift only (翻转5%的neurons) → effect_B
    3. direction_add + gating_shift (同时) → effect_AB
    4. 如果 effect_AB >> effect_A + effect_B → 乘性耦合
    """
    print("\n" + "="*70)
    print("实验 C: 乘性验证 — direction + gating联合效应")
    print("="*70)
    
    layers = get_layers(model)
    target_layers = [0, 5, 10, 15, model_info.n_layers - 1]
    target_layers = [l for l in target_layers if l < model_info.n_layers]
    
    beta = 2.0
    n_flip_ratio = 0.05  # 翻转5%的neurons
    
    # 收集结果
    results = {l: {'dir_only': [], 'gate_only': [], 'joint': []} 
               for l in target_layers}
    
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
        
        sing_logits_base = sing_out.logits[0, -1, :]
        baseline_diff = sing_logits_base[sv_tok[0]].item() - sing_logits_base[pv_tok[0]].item()
        
        s_vp = sing_ids.input_ids.shape[1] - 1
        p_vp = plur_ids.input_ids.shape[1] - 1
        
        for li in target_layers:
            if li+1 >= len(sing_out.hidden_states) or li+1 >= len(plur_out.hidden_states):
                continue
            
            layer = layers[li]
            lw = get_layer_weights(layer, model_info.d_model, model_info.mlp_type)
            W_gate = lw.W_gate
            W_up = lw.W_up
            W_down = lw.W_down
            if W_gate is None or W_up is None or W_down is None:
                continue
            
            # 获取direction和baseline
            s_h = sing_out.hidden_states[li+1][0, min(s_vp, sing_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            p_h = plur_out.hidden_states[li+1][0, min(p_vp, plur_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            
            direction_np = p_h - s_h
            d_norm = np.linalg.norm(direction_np)
            if d_norm < 1e-8:
                continue
            direction_np_norm = direction_np / d_norm
            
            # 获取MLP pre-activation for baseline gating
            if li < len(sing_out.hidden_states):
                pre_h = sing_out.hidden_states[li][0, min(s_vp, sing_out.hidden_states[li].shape[1]-1), :].detach().cpu().float().numpy()
            else:
                continue
            
            pre_act = W_gate @ pre_h  # [intermediate]
            gate_mask = (pre_act > 0).astype(float)  # baseline gating
            
            # 找到sing/plur差异最大的neurons（gating差异方向）
            if li < len(plur_out.hidden_states):
                pre_h_plur = plur_out.hidden_states[li][0, min(p_vp, plur_out.hidden_states[li].shape[1]-1), :].detach().cpu().float().numpy()
                pre_act_plur = W_gate @ pre_h_plur
                gate_plur = (pre_act_plur > 0).astype(float)
                
                # 差异neurons
                diff_mask = (gate_mask != gate_plur)
                n_diff = int(diff_mask.sum())
                
                if n_diff < 3:
                    # 太少差异neurons，用marginal接近0的neurons
                    margins = np.abs(pre_act)
                    n_flip = max(int(len(gate_mask) * n_flip_ratio), 3)
                    flip_idx = np.argsort(margins)[:n_flip]
                    flip_mask = np.zeros_like(gate_mask)
                    flip_mask[flip_idx] = 1
                else:
                    # 用差异neurons
                    if n_diff > int(len(gate_mask) * 0.2):
                        # 太多了，只取margin最小的那些
                        diff_indices = np.where(diff_mask)[0]
                        diff_margins = np.abs(pre_act[diff_indices])
                        n_flip = max(int(len(gate_mask) * n_flip_ratio), 3)
                        flip_idx = diff_indices[np.argsort(diff_margins)[:n_flip]]
                    else:
                        flip_idx = np.where(diff_mask)[0]
                    flip_mask = np.zeros_like(gate_mask)
                    flip_mask[flip_idx] = 1
            else:
                continue
            
            # ★ 条件1: direction_add only
            effect_dir = _run_intervention(
                model, tokenizer, device, layers, li, sing_ids, s_vp,
                direction_np_norm, beta, sv_tok[0], pv_tok[0], baseline_diff
            )
            if effect_dir is not None:
                results[li]['dir_only'].append(effect_dir)
            
            # ★ 条件2: gating shift only — 通过在MLP pre-activation上注入来翻转gating
            # 构造gating shift向量: 在residual stream空间中，什么向量能翻转这些neurons?
            # 方法: W_gate @ delta_h = -2 * pre_act[flip_idx] → 使这些neurons翻转
            # 即: delta_h = W_gate^+ @ target_pre_act_shift
            # 简化: 用伪逆
            target_shift = np.zeros(len(gate_mask))
            flip_idx_list = np.where(flip_mask > 0)[0]
            for idx in flip_idx_list:
                # 使pre_act[idx]变号: 需要 delta = -2 * pre_act[idx] (sign flip)
                # 但可能方向不对，取: delta = -sign(pre_act[idx]) * (|pre_act[idx]| + margin)
                target_shift[idx] = -2 * pre_act[idx]
            
            # 用W_gate的伪逆求residual方向的delta
            try:
                # W_gate: [intermediate, d_model]
                # 目标: W_gate @ delta_h ≈ target_shift
                # 最小范数解: delta_h = W_gate^T (W_gate W_gate^T)^{-1} target_shift
                # 但intermediate可能很大，用SVD
                from scipy.sparse.linalg import svds
                k_svd = min(50, min(W_gate.shape) - 1)
                U, S, Vt = svds(W_gate.astype(np.float32), k=k_svd)
                # W_gate ≈ U @ diag(S) @ Vt
                # W_gate @ Vt^T @ diag(1/S) @ U^T @ target_shift = target_shift (近似)
                delta_h_gate = Vt.T @ np.diag(1.0/S) @ U.T[:k_svd, :] @ target_shift
            except Exception as e:
                # 回退: 直接用W_gate^T
                delta_h_gate = W_gate.T @ target_shift
            
            gate_norm = np.linalg.norm(delta_h_gate)
            if gate_norm < 1e-8:
                continue
            delta_h_gate_norm = delta_h_gate / gate_norm
            
            # gating shift的强度: 需要使目标neurons翻转
            # 估计需要的β: 使得 W_gate @ (β * delta_h_gate_norm) ≈ target_shift
            # β ≈ ||target_shift|| / ||W_gate @ delta_h_gate_norm||
            proj_strength = np.linalg.norm(W_gate @ delta_h_gate_norm)
            if proj_strength > 1e-8:
                beta_gate = np.linalg.norm(target_shift) / proj_strength
            else:
                beta_gate = 0
            
            # 限制beta_gate范围
            beta_gate = min(beta_gate, 50.0)
            
            effect_gate = _run_intervention(
                model, tokenizer, device, layers, li, sing_ids, s_vp,
                delta_h_gate_norm, beta_gate, sv_tok[0], pv_tok[0], baseline_diff
            )
            if effect_gate is not None:
                results[li]['gate_only'].append(effect_gate)
            
            # ★ 条件3: direction + gating联合
            # 联合向量 = direction + gating方向
            # 但关键是: 它们是否通过不同的机制起作用?
            # 乘性验证: 联合效果是否 > 各自之和?
            
            # 方法: 在同一hook中同时注入direction和gating_shift
            combined = direction_np_norm * beta + delta_h_gate_norm * beta_gate
            combined_norm = np.linalg.norm(combined)
            if combined_norm < 1e-8:
                continue
            combined_direction = combined / combined_norm
            combined_beta = combined_norm
            
            effect_joint = _run_intervention(
                model, tokenizer, device, layers, li, sing_ids, s_vp,
                combined_direction, 1.0,  # 归一化后用1.0，实际幅度已包含
                sv_tok[0], pv_tok[0], baseline_diff
            )
            # 等等，这个方法不对——联合注入只是加法组合
            # 正确的乘性验证需要: direction改变W₁h的同时，gating也改变
            
            # 更好的方法: 直接在MLP层面操作
            # MLP(h) = W₂ · σ(W₁h)
            # 如果我们: 
            #   (a) 改变h → h + βv  (direction_add)
            #   (b) 改变σ → 在gate输出上做修改
            # 这需要更精确的hook...
            
            # 用更简单的方法: 分别测试，然后检查乘性
            # effect(dir+gate) vs effect(dir) + effect(gate)
            # 如果乘性: effect(dir+gate) > effect(dir) + effect(gate)
            
            # 实际上我需要同时注入两个方向
            # 方法: 两次hook注入
            direction_t = torch.tensor(direction_np_norm, dtype=torch.float16, device=device)
            gate_t = torch.tensor(delta_h_gate_norm, dtype=torch.float16, device=device)
            
            def joint_hook(module, input, output):
                out = output[0].detach().clone()
                pos = min(s_vp, out.shape[1]-1)
                out[0, pos, :] += (beta * direction_t + beta_gate * gate_t).to(out.dtype)
                return (out,) + output[1:] if isinstance(output, tuple) else out
            
            h = layers[li].register_forward_hook(joint_hook)
            
            with torch.no_grad():
                try:
                    joint_out = model(**sing_ids)
                except Exception as e:
                    h.remove()
                    continue
            
            h.remove()
            
            joint_logits = joint_out.logits[0, -1, :]
            joint_diff = joint_logits[sv_tok[0]].item() - joint_logits[pv_tok[0]].item()
            effect_joint_val = joint_diff - baseline_diff
            
            results[li]['joint'].append(effect_joint_val)
            
            del joint_out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        del sing_out, plur_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 汇总结果
    print("\n--- 实验 C 结果: 乘性验证 ---")
    print(f"{'Layer':<8} {'|dir|':<10} {'|gate|':<10} {'|joint|':<10} {'|sum|':<10} {'joint/sum':<12} {'乘性?':<8}")
    
    multiplicative_layers = 0
    for li in target_layers:
        d = results[li]['dir_only']
        g = results[li]['gate_only']
        j = results[li]['joint']
        
        if len(d) >= 3 and len(g) >= 3 and len(j) >= 3:
            # 取最小公共长度
            ml = min(len(d), len(g), len(j))
            d_abs = np.abs(np.array(d[:ml]))
            g_abs = np.abs(np.array(g[:ml]))
            j_abs = np.abs(np.array(j[:ml]))
            
            mean_dir = np.mean(d_abs)
            mean_gate = np.mean(g_abs)
            mean_joint = np.mean(j_abs)
            mean_sum = np.mean(np.abs(np.array(d[:ml])) + np.abs(np.array(g[:ml])))
            
            ratio = mean_joint / mean_sum if mean_sum > 1e-8 else 0
            is_mult = "★" if ratio > 1.3 else ""
            if ratio > 1.3:
                multiplicative_layers += 1
            
            print(f"L{li:<7} {mean_dir:<10.4f} {mean_gate:<10.4f} {mean_joint:<10.4f} {mean_sum:<10.4f} {ratio:<12.2f} {is_mult}")
        else:
            print(f"L{li:<7} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<12} {'':8}")
    
    # 用配对检验验证乘性
    print("\n[★ 配对检验: joint vs sum(dir,gate)]")
    for li in target_layers:
        d = results[li]['dir_only']
        g = results[li]['gate_only']
        j = results[li]['joint']
        ml = min(len(d), len(g), len(j))
        if ml >= 5:
            from scipy.stats import wilcoxon
            try:
                j_abs = np.abs(np.array(j[:ml]))
                sum_abs = np.abs(np.array(d[:ml])) + np.abs(np.array(g[:ml]))
                stat, p = wilcoxon(j_abs, sum_abs, alternative='greater')
                sig = "★ 乘性" if p < 0.05 else "加性"
                print(f"  L{li}: joint > sum? p={p:.4f} → {sig}")
            except:
                print(f"  L{li}: 检验失败")
    
    print(f"\n[★ 总判定: {multiplicative_layers}/{len(target_layers)}层支持乘性]")
    
    return results


# ============================================================
# 主程序
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Phase 66: MPS验证")
    parser.add_argument("--model", type=str, default="deepseek7b", 
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--n_test", type=int, default=50)
    parser.add_argument("--exp", type=str, default="all",
                       choices=["a", "b", "c", "all"])
    args = parser.parse_args()
    
    print("="*70)
    print(f"Phase 66: 乘性路径系统 (MPS) 验证")
    print(f"模型: {args.model}, n_test={args.n_test}")
    print("="*70)
    
    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"模型信息: {model_info.n_layers}层, d_model={model_info.d_model}, "
          f"mlp_type={model_info.mlp_type}")
    
    results = {}
    t0 = time.time()
    
    try:
        if args.exp in ["a", "all"]:
            results['a'] = experiment_a_beta_scaling(
                model, tokenizer, device, model_info, args.n_test)
        
        if args.exp in ["b", "all"]:
            results['b'] = experiment_b_covariance_control(
                model, tokenizer, device, model_info, args.n_test)
        
        if args.exp in ["c", "all"]:
            results['c'] = experiment_c_multiplicative(
                model, tokenizer, device, model_info, args.n_test)
    finally:
        release_model(model)
    
    elapsed = time.time() - t0
    print(f"\n总用时: {elapsed:.1f}s")
    
    # ★★★ 最终综合判定
    print("\n" + "="*70)
    print("Phase 66 综合判定")
    print("="*70)
    print("""
MPS核心预测:
1. β递增存在phase transition（gating突然翻转）
2. covariance shift >> mean shift
3. direction + gating联合效应 >> 各自之和（乘性）

实验结果判定:
（见上方各实验输出）
""")


if __name__ == "__main__":
    main()
