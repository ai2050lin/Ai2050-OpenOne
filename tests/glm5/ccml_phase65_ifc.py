"""
Phase 65: IFC理论验证 — Information Flow Constraint Theory
============================================================

基于Phase 64的关键纠正:
1. ARI分析犯了"范畴错误"——用欧几里得聚类检测组合结构
   → 改用argmax/top-k graph提取离散拓扑结构
2. "路由不按number分区 ≠ 路由不是分区结构"
   → routing可能沿其他维度分区
3. "信息被压缩"可能是错的 → 更可能是"信息被旋转混合"

IFC核心主张:
  语法 = 对信息流Φ(x)的约束结构
  如果Φ(x+δ) ≈ Φ(x)，则ΔP(y) ≈ 0

Phase 65的四个核心实验:
A. 组合结构分析 — argmax attention graph（离散拓扑，非softmax值聚类）
B. 临界距离实验 — MLP pre-activation margin vs effect size
C. 冻结gating实验 — 锁定MLP mask后direction_add是否变强
D. 路由间隙分析 — attention top-1/top-2 gap vs effect size

关键区分:
- Phase 64用softmax概率做KMeans → 范畴错误
- Phase 65用argmax/top-k提取graph → 正确的组合分析
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch, numpy as np, gc, argparse, time
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')
from model_utils import load_model, get_model_info, release_model, get_layers

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
    ("writer","writers","writes","write"),("rabbit","rabbits","hops","hop"),
    ("eagle","eagles","soars","soar"),("tiger","tigers","stalks","stalk"),
    ("monkey","monkeys","climbs","climb"),("lion","lions","roars","roar"),
    ("farmer","farmers","plants","plant"),("robot","robots","moves","move"),
    ("wizard","wizards","casts","cast"),("dragon","dragons","breathes","breathe"),
    ("angel","angels","flies","fly"),("soldier","soldiers","marches","march"),
    ("nurse","nurses","cares","care"),("artist","artists","paints","paint"),
    ("dancer","dancers","spins","spin"),("singer","singers","sings","sing"),
    ("hunter","hunters","tracks","track"),("apple","apples","falls","fall"),
    ("river","rivers","flows","flow"),("cloud","clouds","drifts","drift"),
    ("star","stars","shines","shine"),("moon","moons","orbits","orbit"),
    ("book","books","opens","open"),("phone","phones","rings","ring"),
    ("clock","clocks","ticks","tick"),("train","trains","arrives","arrive"),
    ("plane","planes","lands","land"),("ship","ships","sails","sail"),
]

EXTRA_NVA = [
    ("rock","rocks","sits","sit"),("lamp","lamps","glows","glow"),
    ("door","doors","opens","open"),("wall","walls","stands","stand"),
    ("bell","bells","rings","ring"),("cup","cups","holds","hold"),
    ("pen","pens","writes","write"),("map","maps","shows","show"),
    ("road","roads","leads","lead"),("hill","hills","rises","rise"),
    ("lake","lakes","shines","shine"),("wind","winds","blows","blow"),
    ("rain","rains","falls","fall"),("snow","snows","melts","melt"),
    ("fire","fires","burns","burn"),("gold","golds","shines","shine"),
    ("sand","sands","shifts","shift"),("wave","waves","crashes","crash"),
    ("bone","bones","breaks","break"),("seed","seeds","grows","grow"),
    ("rose","roses","blooms","bloom"),("leaf","leaves","falls","fall"),
    ("pear","pears","ripens","ripen"),("bean","beans","sprouts","sprout"),
    ("cake","cakes","bakes","bake"),("ring","rings","spins","spin"),
    ("kite","kites","flies","fly"),("drum","drums","beats","beat"),
    ("flag","flags","waves","wave"),("coin","coins","spins","spin"),
]

ALL_NVA = REGULAR_NVA + EXTRA_NVA

NOUNS_SET = set()
for sn, pn, _, _ in ALL_NVA:
    NOUNS_SET.add(sn.lower())
    NOUNS_SET.add(pn.lower())

TARGET_LAYERS = [0, 5, 10, 15, 27]


def find_noun_pos(tokenizer, tokens):
    for i, t in enumerate(tokens):
        d = tokenizer.decode([t]).strip().lower()
        for n in NOUNS_SET:
            if d == n or d.startswith(n):
                return i + 1
    return None


def find_verb_pos(tokenizer, tokens, sv, pv):
    targets = {sv.lower(), pv.lower()}
    for i, t in enumerate(tokens):
        d = tokenizer.decode([t]).strip().lower()
        for vt in targets:
            if d == vt or d.startswith(vt):
                return i + 1
    return None


# ============================================================
# 实验 A: 组合结构分析 — argmax/top-k attention graph
# ============================================================
def experiment_A_combinatorial_routing(model, tokenizer, device, info, target_layers, n_test=80):
    """
    Phase 64的纠正: 用argmax/top-k提取离散拓扑结构，而非softmax概率聚类
    
    核心思路:
    - softmax概率在欧几里得空间看是连续的
    - 但argmax(top-k)选择是离散的组合结构
    - 这才是"路由"的真正本质
    
    分析内容:
    1. 每个head的argmax routing pattern
    2. routing graph的离散分区
    3. sing/plur是否产生不同的routing graph
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT A: 组合结构分析（argmax/top-k routing graph）")
    print("=" * 70)
    print("  关键纠正: Phase 64用softmax概率做KMeans是范畴错误")
    print("  正确方法: 提取argmax/top-k graph → 离散组合结构")
    
    layers_list = get_layers(model)
    
    # 收集每个样本的routing graph
    routing_data = {li: {'sing_graphs': [], 'plur_graphs': [],
                          'sing_top2_gap': [], 'plur_top2_gap': []}
                    for li in target_layers}
    
    for si, (sn, pn, sv, pv) in enumerate(ALL_NVA[:n_test]):
        if si % 20 == 0 and si > 0:
            print(f"    Collecting {si}/{n_test}...")
        
        sing_sent = f"The {sn} {sv}"
        plur_sent = f"The {pn} {pv}"
        
        sing_toks = tokenizer.encode(sing_sent, add_special_tokens=False)
        plur_toks = tokenizer.encode(plur_sent, add_special_tokens=False)
        
        sing_sp = find_noun_pos(tokenizer, sing_toks)
        sing_vp = find_verb_pos(tokenizer, sing_toks, sv, pv)
        plur_sp = find_noun_pos(tokenizer, plur_toks)
        plur_vp = find_verb_pos(tokenizer, plur_toks, sv, pv)
        
        if sing_sp is None or sing_vp is None or plur_sp is None or plur_vp is None:
            continue
        
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
        
        with torch.no_grad():
            sing_out = model(**sing_ids, output_attentions=True)
            plur_out = model(**plur_ids, output_attentions=True)
        
        for li in target_layers:
            if li >= len(sing_out.attentions) or li >= len(plur_out.attentions):
                continue
            
            s_attn = sing_out.attentions[li][0]  # [heads, seq_q, seq_k]
            p_attn = plur_out.attentions[li][0]
            
            # ===== 核心改变: 提取离散组合结构 =====
            # 对每个head，verb位置的argmax attention target
            if sing_vp < s_attn.shape[1] and plur_vp < p_attn.shape[1]:
                s_verb_attn = s_attn[:, sing_vp, :].float()  # [heads, seq_k]
                p_verb_attn = p_attn[:, plur_vp, :].float()
                
                # 1. argmax routing: 每个head最attend to哪个position
                s_argmax = s_verb_attn.argmax(dim=1).cpu().numpy()  # [heads]
                p_argmax = p_verb_attn.argmax(dim=1).cpu().numpy()
                
                # 2. top-3 routing: 每个head的top-3 positions
                s_top3 = s_verb_attn.topk(3, dim=1).indices.cpu().numpy()  # [heads, 3]
                p_top3 = p_verb_attn.topk(3, dim=1).indices.cpu().numpy()
                
                # 3. routing gap: top-1与top-2的差
                s_sorted = s_verb_attn.sort(dim=1, descending=True)
                p_sorted = p_verb_attn.sort(dim=1, descending=True)
                s_gap = (s_sorted.values[:, 0] - s_sorted.values[:, 1]).cpu().numpy()  # [heads]
                p_gap = (p_sorted.values[:, 0] - p_sorted.values[:, 1]).cpu().numpy()
                
                routing_data[li]['sing_graphs'].append({
                    'argmax': s_argmax,
                    'top3': s_top3,
                    'sent_idx': si,
                })
                routing_data[li]['plur_graphs'].append({
                    'argmax': p_argmax,
                    'top3': p_top3,
                    'sent_idx': si,
                })
                routing_data[li]['sing_top2_gap'].append(s_gap)
                routing_data[li]['plur_top2_gap'].append(p_gap)
        
        del sing_out, plur_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ============================================================
    # 分析组合结构
    # ============================================================
    print("\n" + "-" * 60)
    print("组合结构分析结果")
    print("-" * 60)
    
    results = {}
    
    for li in target_layers:
        print(f"\n=== Layer {li} ===")
        
        s_graphs = routing_data[li]['sing_graphs']
        p_graphs = routing_data[li]['plur_graphs']
        s_gaps = routing_data[li]['sing_top2_gap']
        p_gaps = routing_data[li]['plur_top2_gap']
        
        if len(s_graphs) < 5 or len(p_graphs) < 5:
            print(f"  Not enough data ({len(s_graphs)} sing, {len(p_graphs)} plur)")
            continue
        
        n_sing = len(s_graphs)
        n_plur = len(p_graphs)
        
        # ---- 1. argmax routing变化率 ----
        # 对每个head，计算sing和plur之间的argmax一致率
        n_heads = s_graphs[0]['argmax'].shape[0]
        
        head_switch_rate = []  # 每个head在sing/plur之间切换argmax的频率
        for h in range(n_heads):
            switch_count = 0
            total_pairs = 0
            for sg, pg in zip(s_graphs, p_graphs):
                if sg['argmax'][h] != pg['argmax'][h]:
                    switch_count += 1
                total_pairs += 1
            switch_rate = switch_count / max(total_pairs, 1)
            head_switch_rate.append(switch_rate)
        
        avg_switch_rate = np.mean(head_switch_rate)
        max_switch_rate = np.max(head_switch_rate)
        max_switch_head = np.argmax(head_switch_rate)
        
        print(f"  Argmax routing switch rate (sing→plur):")
        print(f"    Mean: {avg_switch_rate:.4f}")
        print(f"    Max:  {max_switch_rate:.4f} (head {max_switch_head})")
        print(f"    Per-head rates: {[f'{r:.3f}' for r in head_switch_rate[:8]]}...")
        
        # ---- 2. top-3 routing graph重叠度 ----
        # 计算sing和plur的top-3集合的重叠
        top3_overlap_rates = []
        for h in range(n_heads):
            overlaps = []
            for sg, pg in zip(s_graphs, p_graphs):
                s_set = set(sg['top3'][h].tolist())
                p_set = set(pg['top3'][h].tolist())
                overlap = len(s_set & p_set) / 3.0
                overlaps.append(overlap)
            top3_overlap_rates.append(np.mean(overlaps))
        
        avg_top3_overlap = np.mean(top3_overlap_rates)
        min_top3_overlap = np.min(top3_overlap_rates)
        min_overlap_head = np.argmin(top3_overlap_rates)
        
        print(f"  Top-3 routing overlap (sing vs plur):")
        print(f"    Mean overlap: {avg_top3_overlap:.4f}")
        print(f"    Min overlap:  {min_top3_overlap:.4f} (head {min_overlap_head})")
        
        # ---- 3. argmax routing的离散分布 ----
        # 关键: 在组合空间中，argmax pattern有多少种？
        all_argmax_patterns = []
        for sg in s_graphs:
            all_argmax_patterns.append(tuple(sg['argmax'].tolist()))
        for pg in p_graphs:
            all_argmax_patterns.append(tuple(pg['argmax'].tolist()))
        
        pattern_counter = Counter(all_argmax_patterns)
        n_unique_patterns = len(pattern_counter)
        n_total = len(all_argmax_patterns)
        
        # 最常见的pattern覆盖多少样本？
        most_common_count = pattern_counter.most_common(1)[0][1]
        coverage = most_common_count / n_total
        
        print(f"  Argmax pattern diversity:")
        print(f"    Unique patterns: {n_unique_patterns} / {n_total} samples")
        print(f"    Top-1 pattern coverage: {coverage:.4f}")
        print(f"    Top-5 patterns coverage: {sum(c for _, c in pattern_counter.most_common(5)) / n_total:.4f}")
        
        # ---- 4. routing gap（临界性指标）----
        # gap越小 → 越接近attention routing相变
        s_gap_arr = np.array(s_gaps)  # [n_sing, heads]
        p_gap_arr = np.array(p_gaps)  # [n_plur, heads]
        all_gaps = np.vstack([s_gap_arr, p_gap_arr])
        
        print(f"  Routing gap (top1 - top2 attention weight):")
        print(f"    Mean gap: {all_gaps.mean():.4f}")
        print(f"    Min gap:  {all_gaps.min():.4f}")
        print(f"    Gap < 0.1 (near-critical): {(all_gaps < 0.1).mean():.4f}")
        print(f"    Gap < 0.05 (very near-critical): {(all_gaps < 0.05).mean():.4f}")
        
        # ---- 5. sing/plur是否产生不同routing pattern ----
        # 对每个pattern，看它是否主要出现在sing或plur中
        sing_patterns = [tuple(sg['argmax'].tolist()) for sg in s_graphs]
        plur_patterns = [tuple(pg['argmax'].tolist()) for pg in p_graphs]
        
        # 用pattern作为特征，做ARI
        from sklearn.metrics import adjusted_rand_score
        all_pattern_labels = [0]*n_sing + [1]*n_plur
        
        # 方法1: 直接用unique pattern ID做"聚类"
        pattern_to_id = {p: i for i, p in enumerate(pattern_counter.keys())}
        pattern_ids = [pattern_to_id[p] for p in all_argmax_patterns]
        
        # 如果每个pattern都不同，ARI没意义
        if n_unique_patterns < n_total * 0.8:
            ari_pattern = adjusted_rand_score(all_pattern_labels, pattern_ids)
            print(f"  Pattern-based ARI (number vs pattern_id): {ari_pattern:.4f}")
        else:
            ari_pattern = 0
            print(f"  Pattern-based ARI: N/A (too many unique patterns)")
        
        results[li] = {
            'avg_switch_rate': avg_switch_rate,
            'max_switch_rate': max_switch_rate,
            'max_switch_head': max_switch_head,
            'avg_top3_overlap': avg_top3_overlap,
            'min_top3_overlap': min_top3_overlap,
            'n_unique_patterns': n_unique_patterns,
            'n_total': n_total,
            'coverage': coverage,
            'mean_gap': all_gaps.mean(),
            'near_critical_rate': (all_gaps < 0.1).mean(),
            'head_switch_rates': head_switch_rate,
            'head_top3_overlaps': top3_overlap_rates,
        }
    
    return results


# ============================================================
# 实验 B: 临界距离实验 — MLP margin vs effect size
# ============================================================
def experiment_B_critical_margin(model, tokenizer, device, info, target_layers, n_test=60):
    """
    IFC核心预测: 
    - 如果Φ(x+δ) ≈ Φ(x)，则ΔP(y) ≈ 0
    - MLP gating的"临界距离"= pre-activation到0的距离
    - 越接近0 → 越容易切换 → effect size应该越大
    
    同时提取MLP pre-activation (W_gate @ x) 而非 post-activation
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B: 临界距离 vs 效应量")
    print("=" * 70)
    print("  IFC预测: 越接近gating边界 → 扰动效果越大")
    
    layers_list = get_layers(model)
    
    # 需要捕获MLP gate pre-activation
    results = {li: {'margins': [], 'effects': [], 'n_switched': 0, 'n_total': 0}
               for li in target_layers}
    
    for si, (sn, pn, sv, pv) in enumerate(ALL_NVA[:n_test]):
        if si % 15 == 0 and si > 0:
            print(f"    Testing {si}/{n_test}...")
        
        sing_sent = f"The {sn} {sv}"
        plur_sent = f"The {pn} {pv}"
        
        sing_toks = tokenizer.encode(sing_sent, add_special_tokens=False)
        plur_toks = tokenizer.encode(plur_sent, add_special_tokens=False)
        
        sing_vp = find_verb_pos(tokenizer, sing_toks, sv, pv)
        plur_vp = find_verb_pos(tokenizer, plur_toks, sv, pv)
        
        if sing_vp is None or plur_vp is None:
            continue
        
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
        
        # Step 1: 前向传播，获取sing的logits（baseline）和hidden states
        with torch.no_grad():
            sing_out = model(**sing_ids, output_hidden_states=True)
            plur_out = model(**plur_ids, output_hidden_states=True)
        sing_logits = sing_out.logits[0, -1, :]
        sv_tok = tokenizer.encode(sv, add_special_tokens=False)
        pv_tok = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_tok or not pv_tok:
            del sing_out, plur_out
            continue
        baseline_diff = sing_logits[sv_tok[0]].item() - sing_logits[pv_tok[0]].item()
        
        for li in target_layers:
            if li+1 >= len(sing_out.hidden_states) or li+1 >= len(plur_out.hidden_states):
                continue
            
            # 计算direction
            s_h = sing_out.hidden_states[li+1][0, min(sing_vp, sing_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            p_h = plur_out.hidden_states[li+1][0, min(plur_vp, plur_out.hidden_states[li+1].shape[1]-1), :].detach().cpu().float().numpy()
            
            direction_np = p_h - s_h
            direction_norm = np.linalg.norm(direction_np)
            if direction_norm < 1e-8:
                continue
            direction_np = direction_np / direction_norm
            
            # Step 3: 在该层注入direction，测效应量
            # 使用hook注入
            direction_t = torch.tensor(direction_np, dtype=sing_out.hidden_states[li+1].dtype, device=device)
            beta = 2.0  # 注入强度
            
            hook_added = [False]
            effect_data = {}
            
            def inject_hook(m, inp, out, _li=li, _vp=sing_vp, _dir=direction_t, _beta=beta):
                if not hook_added[0]:
                    new_out = out[0].clone() if isinstance(out, tuple) else out.clone()
                    vp = min(_vp, new_out.shape[1]-1)
                    new_out[0, vp, :] += (_beta * _dir).to(new_out.dtype)
                    hook_added[0] = True
                    if isinstance(out, tuple):
                        return (new_out,) + out[1:]
                    return new_out
                return out
            
            hook = layers_list[li].register_forward_hook(inject_hook)
            
            with torch.no_grad():
                interv_out = model(**sing_ids)
            
            hook.remove()
            
            interv_logits = interv_out.logits[0, -1, :]
            interv_logit_s = interv_logits[tokenizer.encode(sv, add_special_tokens=False)[0]].item()
            interv_logit_p = interv_logits[tokenizer.encode(pv, add_special_tokens=False)[0]].item()
            interv_diff = interv_logit_s - interv_logit_p
            
            effect = interv_diff - baseline_diff  # 正值=朝plur方向移动
            results[li]['effects'].append(effect)
            
            # Step 4: 计算MLP gate的临界距离
            # 需要获取gate pre-activation: W_gate @ x + b
            # 这里简化：用hidden state和gate权重计算
            # 实际中gate pre-activation ≈ W_gate @ LN(x)
            # 我们用前一层的hidden state作为proxy
            
            if li < len(sing_out.hidden_states):
                # pre-LN hidden state
                pre_h = sing_out.hidden_states[li][0, min(sing_vp, sing_out.hidden_states[li].shape[1]-1), :].detach()
                
                # 获取gate权重
                mlp = layers_list[li].mlp
                if hasattr(mlp, 'gate_proj'):
                    W_gate = mlp.gate_proj.weight.detach()  # [intermediate, d_model]
                    b_gate = mlp.gate_proj.bias.detach() if mlp.gate_proj.bias is not None else torch.zeros(W_gate.shape[0], device=device)
                    
                    # pre-activation
                    pre_act = (W_gate @ pre_h.to(W_gate.dtype) + b_gate).float()
                    
                    # 临界距离 = min |pre_act_i| （到0的距离）
                    margins = pre_act.abs().cpu().numpy()
                    
                    # 最小的margin
                    min_margin = margins.min()
                    # 中位数margin
                    median_margin = np.median(margins)
                    # 接近0的neuron比例（|margin| < 0.5）
                    near_critical = (margins < 0.5).mean()
                    
                    # 哪些neurons会被direction翻转？
                    direction_tensor = direction_t.to(W_gate.dtype)
                    delta_pre_act = (W_gate @ direction_tensor).float().cpu().numpy()
                    
                    # 检查：sign(pre_act) vs sign(pre_act + beta * delta_pre_act)
                    pre_act_np = pre_act.cpu().numpy()
                    post_act = pre_act_np + beta * delta_pre_act
                    switched = (np.sign(pre_act_np) != np.sign(post_act)).sum()
                    
                    results[li]['margins'].append({
                        'min_margin': min_margin,
                        'median_margin': median_margin,
                        'near_critical_rate': near_critical,
                        'n_switched': switched,
                        'total_neurons': len(margins),
                    })
                    results[li]['n_switched'] += switched
                    results[li]['n_total'] += len(margins)
            
            # 清理循环内变量
            if 'interv_out' in dir():
                del interv_out
        
        del sing_out, plur_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ============================================================
    # 分析临界距离 vs 效应量
    # ============================================================
    print("\n" + "-" * 60)
    print("临界距离 vs 效应量结果")
    print("-" * 60)
    
    analysis = {}
    for li in target_layers:
        effects = results[li]['effects']
        margins = results[li]['margins']
        
        if len(effects) < 5:
            print(f"\nLayer {li}: Not enough data ({len(effects)})")
            continue
        
        print(f"\n=== Layer {li} (n={len(effects)}) ===")
        
        # 基本统计
        effects_arr = np.array(effects)
        print(f"  Effect: mean={effects_arr.mean():.4f}, σ={effects_arr.std():.4f}")
        print(f"  |Effect|: mean={np.abs(effects_arr).mean():.4f}")
        
        if margins:
            min_margins = np.array([m['min_margin'] for m in margins])
            median_margins = np.array([m['median_margin'] for m in margins])
            near_critical = np.array([m['near_critical_rate'] for m in margins])
            n_switched = np.array([m['n_switched'] for m in margins])
            total_neurons = margins[0]['total_neurons']
            switch_rate = n_switched / total_neurons
            
            print(f"  MLP gate margins:")
            print(f"    Min margin: mean={min_margins.mean():.4f}, min={min_margins.min():.6f}")
            print(f"    Median margin: {median_margins.mean():.4f}")
            print(f"    Near-critical rate (|m|<0.5): {near_critical.mean():.4f}")
            print(f"    Neurons switched by direction_add: {n_switched.mean():.1f}/{total_neurons} ({switch_rate.mean():.4f})")
            
            # ★★★ 核心相关性: margin vs effect
            # 预测: min_margin越小 → |effect|越大
            from scipy.stats import spearmanr, pearsonr
            
            abs_effects = np.abs(effects_arr)
            
            if len(min_margins) == len(abs_effects):
                corr_min, p_min = spearmanr(min_margins, abs_effects)
                print(f"  ★ Spearman(min_margin, |effect|): ρ={corr_min:.4f}, p={p_min:.4f}")
                
                corr_switch, p_switch = spearmanr(switch_rate, abs_effects)
                print(f"  ★ Spearman(switch_rate, |effect|): ρ={corr_switch:.4f}, p={p_switch:.4f}")
                
                # 按临界性分组
                critical_mask = near_critical > np.median(near_critical)
                if critical_mask.sum() > 3 and (~critical_mask).sum() > 3:
                    crit_effect = abs_effects[critical_mask].mean()
                    noncrit_effect = abs_effects[~critical_mask].mean()
                    print(f"  ★ Near-critical group |effect|: {crit_effect:.4f}")
                    print(f"  ★ Far-from-critical group |effect|: {noncrit_effect:.4f}")
                    print(f"  ★ Ratio: {crit_effect / max(noncrit_effect, 1e-8):.2f}x")
            
            analysis[li] = {
                'mean_effect': effects_arr.mean(),
                'std_effect': effects_arr.std(),
                'mean_min_margin': min_margins.mean(),
                'mean_switch_rate': switch_rate.mean(),
                'corr_margin_effect': corr_min if len(min_margins) == len(abs_effects) else 0,
                'corr_switch_effect': corr_switch if len(min_margins) == len(abs_effects) else 0,
            }
        else:
            print(f"  No margin data available")
    
    return analysis


# ============================================================
# 实验 C: 冻结gating实验 — 锁定MLP mask后direction_add是否变强
# ============================================================
def experiment_C_freeze_gating(model, tokenizer, device, info, target_layers, n_test=50):
    """
    IFC最关键预测:
    - 如果固定g(x)和r(x)，则direction_add应该变强
    - 因为direction_add无效的原因是它改变了gating，导致信息流Φ变化不可控
    - 固定gating后，信息流不变，direction可以在固定路径内有效传播
    
    实现:
    1. 获取sing句子的MLP gating mask
    2. 在direction_add过程中，强制MLP输出不变（冻结gating）
    3. 比较冻结 vs 不冻结的direction_add效果
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C: 冻结gating实验")
    print("=" * 70)
    print("  IFC预测: 固定Φ(x)后，direction_add应该变强")
    print("  方法: 锁定MLP输出 → 防止信息流结构改变")
    
    layers_list = get_layers(model)
    
    results = {li: {'free_effect': [], 'frozen_effect': []}
               for li in target_layers}
    
    for si, (sn, pn, sv, pv) in enumerate(ALL_NVA[:n_test]):
        if si % 10 == 0 and si > 0:
            print(f"    Testing {si}/{n_test}...")
        
        sing_sent = f"The {sn} {sv}"
        plur_sent = f"The {pn} {pv}"
        
        sing_toks = tokenizer.encode(sing_sent, add_special_tokens=False)
        plur_toks = tokenizer.encode(plur_sent, add_special_tokens=False)
        
        sing_vp = find_verb_pos(tokenizer, sing_toks, sv, pv)
        plur_vp = find_verb_pos(tokenizer, plur_toks, sv, pv)
        
        if sing_vp is None or plur_vp is None:
            continue
        
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
        
        # Baseline: 正常forward
        with torch.no_grad():
            sing_out = model(**sing_ids)
        sing_logits = sing_out.logits[0, -1, :]
        sv_tok = tokenizer.encode(sv, add_special_tokens=False)
        pv_tok = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_tok or not pv_tok:
            continue
        baseline_diff = sing_logits[sv_tok[0]].item() - sing_logits[pv_tok[0]].item()
        
        # 获取hidden states
        with torch.no_grad():
            sing_full = model(**sing_ids, output_hidden_states=True)
            plur_full = model(**plur_ids, output_hidden_states=True)
        
        for li in target_layers:
            if li+1 >= len(sing_full.hidden_states):
                continue
            
            # Direction
            s_h = sing_full.hidden_states[li+1][0, min(sing_vp, sing_full.hidden_states[li+1].shape[1]-1), :]
            p_h = plur_full.hidden_states[li+1][0, min(plur_vp, plur_full.hidden_states[li+1].shape[1]-1), :]
            direction_t = (p_h - s_h).detach()
            direction_norm = direction_t.norm()
            if direction_norm < 1e-8:
                continue
            direction_t = direction_t / direction_norm
            
            beta = 2.0
            
            # ===== 方法1: Free direction_add（正常） =====
            hook_added_free = [False]
            def free_hook(m, inp, out, _vp=sing_vp, _dir=direction_t, _beta=beta):
                if not hook_added_free[0]:
                    new_out = out[0].clone() if isinstance(out, tuple) else out.clone()
                    vp = min(_vp, new_out.shape[1]-1)
                    new_out[0, vp, :] += (_beta * _dir).to(new_out.dtype)
                    hook_added_free[0] = True
                    return (new_out,) + out[1:] if isinstance(out, tuple) else new_out
                return out
            
            h1 = layers_list[li].register_forward_hook(free_hook)
            with torch.no_grad():
                free_out = model(**sing_ids)
            h1.remove()
            
            free_logits = free_out.logits[0, -1, :]
            free_diff = free_logits[sv_tok[0]].item() - free_logits[pv_tok[0]].item()
            free_effect = free_diff - baseline_diff
            
            # ===== 方法2: Frozen MLP（冻结所有后续层的MLP输出） =====
            # 先获取正常的MLP输出（作为frozen reference）
            captured_mlp = {}
            def capture_mlp_hook(m, inp, out, _li=li):
                out_t = out[0] if isinstance(out, tuple) else out
                captured_mlp[f'L{_li}'] = out_t.detach().clone()
            
            hooks_capture = []
            for future_li in range(li+1, len(layers_list)):
                if future_li in target_layers or future_li == len(layers_list)-1:
                    h = layers_list[future_li].mlp.register_forward_hook(
                        lambda m, inp, out, _fli=future_li: capture_mlp_hook(m, inp, out, _fli)
                    )
                    hooks_capture.append(h)
            
            # 先做一次正常forward获取MLP输出
            with torch.no_grad():
                _ = model(**sing_ids)
            
            for h in hooks_capture:
                h.remove()
            
            # 现在注入direction + 冻结MLP
            hook_added_frozen = [False]
            frozen_count = [0]
            target_future_layers = list(captured_mlp.keys())
            
            def frozen_inject_hook(m, inp, out, _vp=sing_vp, _dir=direction_t, _beta=beta):
                if not hook_added_frozen[0]:
                    new_out = out[0].clone() if isinstance(out, tuple) else out.clone()
                    vp = min(_vp, new_out.shape[1]-1)
                    new_out[0, vp, :] += (_beta * _dir).to(new_out.dtype)
                    hook_added_frozen[0] = True
                    return (new_out,) + out[1:] if isinstance(out, tuple) else new_out
                return out
            
            def freeze_mlp_hook(m, inp, out, _fli=None):
                """用之前捕获的MLP输出替换当前输出"""
                key = f'L{_fli}'
                if key in captured_mlp:
                    frozen_count[0] += 1
                    ref = captured_mlp[key]
                    if isinstance(out, tuple):
                        return (ref,) + out[1:]
                    return ref
                return out
            
            hooks_frozen = []
            hooks_frozen.append(layers_list[li].register_forward_hook(frozen_inject_hook))
            for fli_str in target_future_layers:
                fli = int(fli_str[1:])
                hooks_frozen.append(layers_list[fli].mlp.register_forward_hook(
                    lambda m, inp, out, _fli=fli: freeze_mlp_hook(m, inp, out, _fli)
                ))
            
            with torch.no_grad():
                frozen_out = model(**sing_ids)
            
            for h in hooks_frozen:
                h.remove()
            
            frozen_logits = frozen_out.logits[0, -1, :]
            frozen_diff = frozen_logits[sv_tok[0]].item() - frozen_logits[pv_tok[0]].item()
            frozen_effect = frozen_diff - baseline_diff
            
            results[li]['free_effect'].append(free_effect)
            results[li]['frozen_effect'].append(frozen_effect)
        
        del sing_out, sing_full, plur_full, free_out, frozen_out, captured_mlp
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ============================================================
    # 分析冻结效果
    # ============================================================
    print("\n" + "-" * 60)
    print("冻结gating实验结果")
    print("-" * 60)
    
    analysis = {}
    for li in target_layers:
        free_eff = np.array(results[li]['free_effect'])
        frozen_eff = np.array(results[li]['frozen_effect'])
        
        if len(free_eff) < 5:
            print(f"\nLayer {li}: Not enough data ({len(free_eff)})")
            continue
        
        print(f"\n=== Layer {li} (n={len(free_eff)}) ===")
        print(f"  Free direction_add: mean={free_eff.mean():.4f}, |mean|={np.abs(free_eff).mean():.4f}, σ={free_eff.std():.4f}")
        print(f"  Frozen MLP:         mean={frozen_eff.mean():.4f}, |mean|={np.abs(frozen_eff).mean():.4f}, σ={frozen_eff.std():.4f}")
        
        # SNR
        free_snr = np.abs(free_eff.mean()) / max(free_eff.std(), 1e-8)
        frozen_snr = np.abs(frozen_eff.mean()) / max(frozen_eff.std(), 1e-8)
        print(f"  Free SNR: {free_snr:.4f}")
        print(f"  Frozen SNR: {frozen_snr:.4f}")
        
        ratio = frozen_snr / max(free_snr, 1e-8)
        print(f"  ★ SNR improvement (frozen/free): {ratio:.2f}x")
        
        if ratio > 1.5:
            print(f"  ★★★ IFC PREDICTION CONFIRMED: 冻结gating后direction显著变强!")
        elif ratio > 1.0:
            print(f"  ★ IFC弱支持: 冻结gating后direction略强")
        else:
            print(f"  ✗ IFC不支持: 冻结gating后direction未变强")
        
        analysis[li] = {
            'free_mean': free_eff.mean(),
            'frozen_mean': frozen_eff.mean(),
            'free_abs_mean': np.abs(free_eff).mean(),
            'frozen_abs_mean': np.abs(frozen_eff).mean(),
            'free_snr': free_snr,
            'frozen_snr': frozen_snr,
            'snr_ratio': ratio,
        }
    
    return analysis


# ============================================================
# 实验 D: 路由间隙 vs 效应量（attention critical point）
# ============================================================
def experiment_D_routing_gap(model, tokenizer, device, info, target_layers, n_test=60):
    """
    IFPT预测: 
    - attention routing也有"临界点"
    - 当top-1和top-2的attention分数接近时，小扰动可能切换routing
    - routing gap越小 → 扰动效果越大
    
    同时测试: 最小相变扰动
    - 找到使argmax切换的最小δ
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT D: 路由间隙 vs 效应量（attention相变）")
    print("=" * 70)
    print("  IFPT预测: routing gap越小 → 扰动效果越大")
    
    layers_list = get_layers(model)
    
    results = {li: {'gaps': [], 'effects': []} for li in target_layers}
    
    for si, (sn, pn, sv, pv) in enumerate(ALL_NVA[:n_test]):
        if si % 15 == 0 and si > 0:
            print(f"    Testing {si}/{n_test}...")
        
        sing_sent = f"The {sn} {sv}"
        plur_sent = f"The {pn} {pv}"
        
        sing_toks = tokenizer.encode(sing_sent, add_special_tokens=False)
        plur_toks = tokenizer.encode(plur_sent, add_special_tokens=False)
        
        sing_vp = find_verb_pos(tokenizer, sing_toks, sv, pv)
        plur_vp = find_verb_pos(tokenizer, plur_toks, sv, pv)
        
        if sing_vp is None or plur_vp is None:
            continue
        
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
        
        # 获取baseline
        with torch.no_grad():
            sing_out = model(**sing_ids)
        sing_logits = sing_out.logits[0, -1, :]
        sv_tok = tokenizer.encode(sv, add_special_tokens=False)
        pv_tok = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_tok or not pv_tok:
            continue
        baseline_diff = sing_logits[sv_tok[0]].item() - sing_logits[pv_tok[0]].item()
        
        # 获取hidden states和attentions
        with torch.no_grad():
            sing_full = model(**sing_ids, output_hidden_states=True, output_attentions=True)
        
        for li in target_layers:
            if li+1 >= len(sing_full.hidden_states) or li >= len(sing_full.attentions):
                continue
            
            # 计算routing gap（sing句子verb位置的attention）
            s_attn = sing_full.attentions[li][0, :, min(sing_vp, sing_full.attentions[li].shape[2]-1), :].float()  # [heads, seq_k]
            
            # 每个head的routing gap
            sorted_attn = s_attn.sort(dim=1, descending=True)
            gaps = (sorted_attn.values[:, 0] - sorted_attn.values[:, 1]).cpu().numpy()  # [heads]
            
            # 最小gap（最接近相变的head）
            min_gap = gaps.min()
            mean_gap = gaps.mean()
            
            # direction_add效应量
            s_h = sing_full.hidden_states[li+1][0, min(sing_vp, sing_full.hidden_states[li+1].shape[1]-1), :]
            
            with torch.no_grad():
                plur_full = model(**plur_ids, output_hidden_states=True)
            p_h = plur_full.hidden_states[li+1][0, min(plur_vp, plur_full.hidden_states[li+1].shape[1]-1), :]
            
            direction_t = (p_h - s_h).detach()
            direction_norm = direction_t.norm()
            if direction_norm < 1e-8:
                del plur_full
                continue
            direction_t = direction_t / direction_norm
            
            beta = 2.0
            hook_added = [False]
            def inject_hook(m, inp, out, _vp=sing_vp, _dir=direction_t, _beta=beta):
                if not hook_added[0]:
                    new_out = out[0].clone() if isinstance(out, tuple) else out.clone()
                    vp = min(_vp, new_out.shape[1]-1)
                    new_out[0, vp, :] += (_beta * _dir).to(new_out.dtype)
                    hook_added[0] = True
                    return (new_out,) + out[1:] if isinstance(out, tuple) else new_out
                return out
            
            h = layers_list[li].register_forward_hook(inject_hook)
            with torch.no_grad():
                interv_out = model(**sing_ids)
            h.remove()
            
            interv_logits = interv_out.logits[0, -1, :]
            interv_diff = interv_logits[sv_tok[0]].item() - interv_logits[pv_tok[0]].item()
            effect = interv_diff - baseline_diff
            
            results[li]['gaps'].append({
                'min_gap': min_gap,
                'mean_gap': mean_gap,
            })
            results[li]['effects'].append(effect)
            
            del plur_full, interv_out
            gc.collect()
        
        del sing_out, sing_full
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ============================================================
    # 分析routing gap vs effect
    # ============================================================
    print("\n" + "-" * 60)
    print("路由间隙 vs 效应量结果")
    print("-" * 60)
    
    analysis = {}
    for li in target_layers:
        gaps = results[li]['gaps']
        effects = np.array(results[li]['effects'])
        
        if len(gaps) < 5:
            print(f"\nLayer {li}: Not enough data ({len(gaps)})")
            continue
        
        min_gaps = np.array([g['min_gap'] for g in gaps])
        mean_gaps = np.array([g['mean_gap'] for g in gaps])
        
        print(f"\n=== Layer {li} (n={len(gaps)}) ===")
        print(f"  Routing gaps: min_gap mean={min_gaps.mean():.4f}, mean_gap mean={mean_gaps.mean():.4f}")
        print(f"  Near-critical (min_gap < 0.1): {(min_gaps < 0.1).mean():.4f}")
        print(f"  Effect: mean={effects.mean():.4f}, |mean|={np.abs(effects).mean():.4f}")
        
        from scipy.stats import spearmanr
        abs_effects = np.abs(effects)
        
        corr_min, p_min = spearmanr(min_gaps, abs_effects)
        print(f"  ★ Spearman(min_routing_gap, |effect|): ρ={corr_min:.4f}, p={p_min:.4f}")
        
        corr_mean, p_mean = spearmanr(mean_gaps, abs_effects)
        print(f"  ★ Spearman(mean_routing_gap, |effect|): ρ={corr_mean:.4f}, p={p_mean:.4f}")
        
        # 按gap分组
        if (min_gaps < 0.1).sum() > 3:
            critical_mask = min_gaps < 0.1
            crit_effect = abs_effects[critical_mask].mean()
            noncrit_effect = abs_effects[~critical_mask].mean()
            print(f"  ★ Near-critical group |effect|: {crit_effect:.4f}")
            print(f"  ★ Far-from-critical group |effect|: {noncrit_effect:.4f}")
            print(f"  ★ Ratio: {crit_effect / max(noncrit_effect, 1e-8):.2f}x")
        else:
            # 用median分
            median_gap = np.median(min_gaps)
            low_gap = min_gaps < median_gap
            if low_gap.sum() > 3 and (~low_gap).sum() > 3:
                print(f"  ★ Low-gap group |effect|: {abs_effects[low_gap].mean():.4f}")
                print(f"  ★ High-gap group |effect|: {abs_effects[~low_gap].mean():.4f}")
        
        analysis[li] = {
            'mean_min_gap': min_gaps.mean(),
            'corr_gap_effect': corr_min,
            'p_gap_effect': p_min,
        }
    
    return analysis


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b")
    parser.add_argument("--n_test", type=int, default=60)
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Phase 65: IFC理论验证 (Information Flow Constraint)")
    print(f"Model: {args.model}")
    print("=" * 70)
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"  Model: {info.model_class}, Layers: {info.n_layers}, d_model: {info.d_model}")
    print(f"  Target layers: {TARGET_LAYERS}")
    
    t0 = time.time()
    
    # 实验 A: 组合结构分析
    t_a = time.time()
    result_A = experiment_A_combinatorial_routing(model, tokenizer, device, info, TARGET_LAYERS, n_test=80)
    print(f"\n[Time] Experiment A: {time.time()-t_a:.1f}s")
    
    # 实验 B: 临界距离
    t_b = time.time()
    result_B = experiment_B_critical_margin(model, tokenizer, device, info, TARGET_LAYERS, n_test=args.n_test)
    print(f"\n[Time] Experiment B: {time.time()-t_b:.1f}s")
    
    # 实验 C: 冻结gating
    t_c = time.time()
    result_C = experiment_C_freeze_gating(model, tokenizer, device, info, TARGET_LAYERS, n_test=args.n_test)
    print(f"\n[Time] Experiment C: {time.time()-t_c:.1f}s")
    
    # 实验 D: 路由间隙
    t_d = time.time()
    result_D = experiment_D_routing_gap(model, tokenizer, device, info, TARGET_LAYERS, n_test=args.n_test)
    print(f"\n[Time] Experiment D: {time.time()-t_d:.1f}s")
    
    # ============================================================
    # 综合判断
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 65 综合判断: IFC理论验证")
    print("=" * 70)
    
    print("\n--- 实验 A: 组合结构（纠正Phase 64的范畴错误）---")
    for li in TARGET_LAYERS:
        if li in result_A:
            r = result_A[li]
            print(f"  L{li}: switch_rate={r['avg_switch_rate']:.4f}, "
                  f"top3_overlap={r['avg_top3_overlap']:.4f}, "
                  f"unique_patterns={r['n_unique_patterns']}/{r['n_total']}, "
                  f"near_critical={r['near_critical_rate']:.4f}")
    
    print("\n--- 实验 B: 临界距离 ---")
    for li in TARGET_LAYERS:
        if li in result_B:
            r = result_B[li]
            print(f"  L{li}: corr(margin,|effect|)={r.get('corr_margin_effect',0):.4f}, "
                  f"corr(switch,|effect|)={r.get('corr_switch_effect',0):.4f}, "
                  f"switch_rate={r.get('mean_switch_rate',0):.4f}")
    
    print("\n--- 实验 C: 冻结gating ---")
    ifc_confirmed = 0
    for li in TARGET_LAYERS:
        if li in result_C:
            r = result_C[li]
            print(f"  L{li}: free_SNR={r['free_snr']:.4f}, frozen_SNR={r['frozen_snr']:.4f}, "
                  f"ratio={r['snr_ratio']:.2f}x")
            if r['snr_ratio'] > 1.5:
                ifc_confirmed += 1
    
    print(f"\n  IFC confirmed in {ifc_confirmed}/{len([li for li in TARGET_LAYERS if li in result_C])} layers")
    
    print("\n--- 实验 D: 路由间隙 ---")
    for li in TARGET_LAYERS:
        if li in result_D:
            r = result_D[li]
            print(f"  L{li}: corr(gap,|effect|)={r['corr_gap_effect']:.4f}, p={r['p_gap_effect']:.4f}")
    
    print(f"\n[Time] Total: {time.time()-t0:.1f}s")
    
    release_model(model)
    print("\nPhase 65 完成!")
