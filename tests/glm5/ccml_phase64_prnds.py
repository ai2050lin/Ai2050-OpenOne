"""
Phase 64: PRNDS验证 — Piecewise-Routed Nonlinear Dynamical System
=================================================================

核心假设（来自用户的理论升级）：
1. 语法不是固定方向/子空间，而是条件子空间S(x)
2. direction_add失败是因为不改变路由分支
3. 控制变量是"激活路径"而非"向量值"
4. 系统是分段动力系统，由离散路由状态控制

三个核心实验：
A. 路由状态提取与聚类 — 验证"离散路由状态"是否存在
B. 分支切换测试 — 验证"改变路由"是否比"改变值"更有效
C. 条件方向测试 — 验证"条件子空间"是否比全局子空间更一致

关键区分：
- 如果系统是piecewise → 路由状态离散，切换效果大
- 如果系统是smooth → 路由状态连续，切换效果平滑
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

# 额外增加的名词对，用于增大样本量
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

TARGET_LAYERS = [0, 5, 10, 15, 18]


def find_noun_pos(tokenizer, tokens):
    for i, t in enumerate(tokens):
        d = tokenizer.decode([t]).strip().lower()
        for n in NOUNS_SET:
            if d == n or d.startswith(n):
                return i + 1  # +1 for BOS
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
# 实验 A: 路由状态提取与聚类
# ============================================================
def experiment_A_routing_states(model, tokenizer, device, info, target_layers, n_test=80):
    """
    提取每层的路由状态（attention pattern），分析：
    1. 路由状态是否离散？
    2. 路由状态是否与语法属性（单/复数）对齐？
    3. MLP gating pattern是否也离散？
    
    关键指标：
    - 路由状态ARI（与number/position的对齐）
    - 路由状态的熵（离散 vs 连续）
    - head specialization程度
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT A: 路由状态提取与离散性分析")
    print("=" * 70)
    print("  假设: 如果系统是PRNDS，则路由状态应形成离散聚类")
    print("  反假设: 如果系统是smooth，则路由状态应是连续分布")
    
    layers_list = get_layers(model)
    n_layers = len(layers_list)
    
    # 收集数据
    routing_data = {li: {'sing_attn': [], 'plur_attn': [], 
                          'sing_mlp': [], 'plur_mlp': [],
                          'sing_labels': [], 'plur_labels': []} 
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
        
        # 前向传播，获取attention和hidden states
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
        
        # Hook来捕获MLP中间激活 (使用post-hook捕获MLP输出)
        captured_data = {}
        
        def make_capture_fn(prefix, _si=si):
            def fn(module, input, output):
                # MLP输出 (post-GELU)
                out_tensor = output[0] if isinstance(output, tuple) else output
                captured_data[f'{prefix}_mlp_output'] = out_tensor.detach().cpu()
            return fn
        
        hooks = []
        for li in target_layers:
            if hasattr(layers_list[li], 'mlp'):
                h = layers_list[li].mlp.register_forward_hook(make_capture_fn(f'sing_L{li}'))
                hooks.append(h)
        
        with torch.no_grad():
            sing_out = model(**sing_ids, output_attentions=True, output_hidden_states=True)
        
        for h in hooks:
            h.remove()
        
        # 存储sing数据
        sing_attn = sing_out.attentions if hasattr(sing_out, 'attentions') and sing_out.attentions else None
        sing_hidden = sing_out.hidden_states if hasattr(sing_out, 'hidden_states') else None
        
        # 对plural重复
        captured_data2 = {}
        def make_capture_fn2(prefix, _si=si):
            def fn(module, input, output):
                out_tensor = output[0] if isinstance(output, tuple) else output
                captured_data2[f'{prefix}_mlp_output'] = out_tensor.detach().cpu()
            return fn
        
        hooks = []
        for li in target_layers:
            if hasattr(layers_list[li], 'mlp'):
                h = layers_list[li].mlp.register_forward_hook(make_capture_fn2(f'plur_L{li}'))
                hooks.append(h)
        
        with torch.no_grad():
            plur_out = model(**plur_ids, output_attentions=True, output_hidden_states=True)
        
        for h in hooks:
            h.remove()
        
        plur_attn = plur_out.attentions if hasattr(plur_out, 'attentions') and plur_out.attentions else None
        plur_hidden = plur_out.hidden_states if hasattr(plur_out, 'hidden_states') else None
        
        if sing_attn is None or plur_attn is None:
            continue
        
        for li in target_layers:
            if li >= len(sing_attn) or li >= len(plur_attn):
                continue
            
            s_a = sing_attn[li].detach().cpu()  # [1, heads, seq_q, seq_k]
            p_a = plur_attn[li].detach().cpu()
            
            # 提取verb位置和noun位置的attention pattern
            # pattern = attention FROM verb TO all positions (归一化后已经是概率分布)
            if sing_vp < s_a.shape[2] and plur_vp < p_a.shape[2]:
                # verb的attention分布: [heads, seq_k]
                s_verb_attn = s_a[0, :, sing_vp, :].float().numpy()  # [heads, seq_k]
                p_verb_attn = p_a[0, :, plur_vp, :].float().numpy()  # [heads, seq_k]
                routing_data[li]['sing_attn'].append(s_verb_attn)
                routing_data[li]['plur_attn'].append(p_verb_attn)
                routing_data[li]['sing_labels'].append(0)  # singular
                routing_data[li]['plur_labels'].append(1)  # plural
            
            # MLP output pattern: use as proxy for gating
            sing_key = f'sing_L{li}_mlp_output'
            plur_key = f'plur_L{li}_mlp_output'
            if sing_key in captured_data and plur_key in captured_data2:
                s_mlp = captured_data[sing_key]  # [1, seq, d_model]
                p_mlp = captured_data2[plur_key]
                if sing_vp < s_mlp.shape[1] and plur_vp < p_mlp.shape[1]:
                    # verb位置的MLP输出 (用sign作为gate proxy)
                    s_gate = (s_mlp[0, sing_vp, :].float().numpy() > 0).astype(float)  # binary pattern
                    p_gate = (p_mlp[0, plur_vp, :].float().numpy() > 0).astype(float)
                    routing_data[li]['sing_mlp'].append(s_gate)
                    routing_data[li]['plur_mlp'].append(p_gate)
        
        del sing_out, plur_out, captured_data, captured_data2
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ============================================================
    # 分析路由状态的离散性
    # ============================================================
    print("\n" + "-" * 60)
    print("路由状态分析结果")
    print("-" * 60)
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, silhouette_score
    
    results = {}
    
    for li in target_layers:
        print(f"\n=== Layer {li} ===")
        
        # ---- Attention routing ----
        s_attns = routing_data[li]['sing_attn']
        p_attns = routing_data[li]['plur_attn']
        
        if len(s_attns) < 5 or len(p_attns) < 5:
            print(f"  Not enough data ({len(s_attns)} sing, {len(p_attns)} plur)")
            continue
        
        # 统一维度: 找到公共seq_k长度，截断到最短的
        min_seq_k = min(min(a.shape[1] for a in s_attns), min(a.shape[1] for a in p_attns))
        n_heads = s_attns[0].shape[0]
        
        # 截断到公共长度
        s_attns_trim = [a[:, :min_seq_k] for a in s_attns]
        p_attns_trim = [a[:, :min_seq_k] for a in p_attns]
        
        # Flatten: [n_samples, heads * seq_k]
        s_flat = np.array([a.flatten() for a in s_attns_trim])
        p_flat = np.array([a.flatten() for a in p_attns_trim])
        all_attn = np.vstack([s_flat, p_flat])
        all_labels = np.array([0]*len(s_flat) + [1]*len(p_flat))
        
        # KMeans with 2 clusters
        km = KMeans(n_clusters=2, random_state=42, n_init=10)
        pred = km.fit_predict(all_attn)
        attn_ari = adjusted_rand_score(all_labels, pred)
        
        # Silhouette (衡量聚类紧密度)
        sil = silhouette_score(all_attn, all_labels) if len(all_attn) > 2 else 0
        
        # Key test: 计算attention pattern的离散性
        # 方法: 计算sing和plur之间的JS散度
        # 如果离散: sing内一致, plur内一致, sing/plur之间差异大
        # 如果连续: sing和plur之间有大量重叠
        
        # 对于每个head，计算sing和plur的平均attention分布
        seq_k = min_seq_k
        
        head_ari = []
        head_separation = []
        for h in range(n_heads):
            s_ha = np.array([a[h, :min_seq_k] for a in s_attns_trim])  # [n_sing, seq_k]
            p_ha = np.array([a[h, :min_seq_k] for a in p_attns_trim])   # [n_plur, seq_k]
            
            # 每个head独立聚类
            all_ha = np.vstack([s_ha, p_ha])
            km_h = KMeans(n_clusters=2, random_state=42, n_init=10)
            pred_h = km_h.fit_predict(all_ha)
            ari_h = adjusted_rand_score(all_labels[:len(all_ha)], pred_h)
            head_ari.append(ari_h)
            
            # 平均分布的Jensen-Shannon散度
            s_mean = s_ha.mean(axis=0)
            p_mean = p_ha.mean(axis=0)
            # 简单KL散度估计
            eps = 1e-10
            s_mean = np.clip(s_mean, eps, None)
            s_mean = s_mean / s_mean.sum()
            p_mean = np.clip(p_mean, eps, None)
            p_mean = p_mean / p_mean.sum()
            js = 0.5 * np.sum(s_mean * np.log(s_mean / (0.5*(s_mean+p_mean)+eps))) + \
                 0.5 * np.sum(p_mean * np.log(p_mean / (0.5*(s_mean+p_mean)+eps)))
            head_separation.append(js)
        
        # 离散性度量: within-class variance vs between-class variance
        s_center = s_flat.mean(axis=0)
        p_center = p_flat.mean(axis=0)
        global_center = all_attn.mean(axis=0)
        
        within_var = np.mean(np.sum((s_flat - s_center)**2, axis=1)) + \
                     np.mean(np.sum((p_flat - p_center)**2, axis=1))
        between_var = np.sum((s_center - global_center)**2) + np.sum((p_center - global_center)**2)
        
        # F-like ratio: between/within (越大=越离散)
        f_ratio = between_var / (within_var + 1e-10)
        
        print(f"  Attention routing:")
        print(f"    Overall ARI (2-cluster vs number): {attn_ari:.4f}")
        print(f"    Silhouette: {sil:.4f}")
        print(f"    F-ratio (between/within): {f_ratio:.6f}")
        print(f"    Head-level ARI: max={max(head_ari):.4f}, mean={np.mean(head_ari):.4f}")
        print(f"    Head JS-divergence: max={max(head_separation):.4f}, mean={np.mean(head_separation):.4f}")
        
        # 最discriminative head
        best_head = np.argmax(head_ari)
        print(f"    Best head for number: h{best_head} (ARI={head_ari[best_head]:.4f}, JS={head_separation[best_head]:.4f})")
        
        # ---- MLP gating pattern ----
        s_mlps = routing_data[li]['sing_mlp']
        p_mlps = routing_data[li]['plur_mlp']
        
        if len(s_mlps) >= 5 and len(p_mlps) >= 5:
            s_mlp_arr = np.array(s_mlps)  # [n, intermediate]
            p_mlp_arr = np.array(p_mlps)
            all_mlp = np.vstack([s_mlp_arr, p_mlp_arr])
            all_mlp_labels = np.array([0]*len(s_mlp_arr) + [1]*len(p_mlp_arr))
            
            # MLP gate pattern: 每个neuron是开/关
            # 计算sing和plur的gate overlap
            s_mean_gate = s_mlp_arr.mean(axis=0)  # 每个neuron的开启概率(sing)
            p_mean_gate = p_mlp_arr.mean(axis=0)  # 每个neuron的开启概率(plur)
            
            # 差异neurons: |p(sing) - p(plur)| > threshold
            gate_diff = np.abs(s_mean_gate - p_mean_gate)
            n_diff_neurons = np.sum(gate_diff > 0.1)  # 差异>10%的neurons
            total_neurons = len(gate_diff)
            diff_fraction = n_diff_neurons / total_neurons
            
            # KMeans on MLP gate
            km_mlp = KMeans(n_clusters=2, random_state=42, n_init=10)
            pred_mlp = km_mlp.fit_predict(all_mlp)
            mlp_ari = adjusted_rand_score(all_mlp_labels, pred_mlp)
            
            # Hamming distance: sing内 vs sing-plur间
            hamming_within_s = np.mean([np.mean(s_mlps[i] != s_mlps[j]) 
                                        for i in range(min(len(s_mlps), 30)) 
                                        for j in range(i+1, min(len(s_mlps), 30))]) if len(s_mlps) > 1 else 0
            hamming_between = np.mean([np.mean(s_mlps[i] != p_mlps[j]) 
                                       for i in range(min(len(s_mlps), 30)) 
                                       for j in range(min(len(p_mlps), 30))])
            
            print(f"  MLP gating:")
            print(f"    ARI (2-cluster vs number): {mlp_ari:.4f}")
            print(f"    Differential neurons (>10% gate diff): {n_diff_neurons}/{total_neurons} ({diff_fraction:.2%})")
            print(f"    Hamming: within(sing)={hamming_within_s:.4f}, between={hamming_between:.4f}")
            print(f"    Hamming ratio (between/within): {hamming_between/(hamming_within_s+1e-6):.4f}")
        
        results[li] = {
            'attn_ari': attn_ari,
            'silhouette': sil,
            'f_ratio': f_ratio,
            'head_ari': head_ari,
            'head_js': head_separation,
            'best_head': int(best_head),
        }
    
    # ============================================================
    # 关键判断: 离散 vs 连续
    # ============================================================
    print("\n" + "=" * 70)
    print("路由状态离散性判断")
    print("=" * 70)
    
    all_aris = [r['attn_ari'] for r in results.values()]
    all_f_ratios = [r['f_ratio'] for r in results.values()]
    all_head_aris = [max(r['head_ari']) for r in results.values()]
    
    print(f"\n  Attention routing ARI: {all_aris}")
    print(f"  Best head ARI: {all_head_aris}")
    print(f"  F-ratio: {all_f_ratios}")
    
    # 判断标准:
    # 离散: ARI > 0.3, F-ratio > 0.01
    # 连续: ARI < 0.1, F-ratio < 0.001
    # 混合: 介于之间
    mean_ari = np.mean(all_aris) if all_aris else 0
    mean_f = np.mean(all_f_ratios) if all_f_ratios else 0
    max_head_ari = max(all_head_aris) if all_head_aris else 0
    
    if mean_ari > 0.3:
        print(f"\n  ★ 判断: 路由状态倾向于离散 (mean ARI={mean_ari:.3f})")
        print(f"    → 支持PRNDS框架")
    elif mean_ari < 0.1:
        print(f"\n  ★ 判断: 路由状态倾向于连续 (mean ARI={mean_ari:.3f})")
        print(f"    → 不支持piecewise假设")
        print(f"    → 但个别head可能有离散性 (max head ARI={max_head_ari:.3f})")
    else:
        print(f"\n  ★ 判断: 路由状态是混合的 (mean ARI={mean_ari:.3f})")
        print(f"    → 部分head有离散路由，整体是soft routing")
    
    return results


# ============================================================
# 实验 B: 分支切换测试
# ============================================================
def experiment_B_branch_switching(model, tokenizer, device, info, target_layers, n_test=50):
    """
    核心测试: "改变路由" vs "改变值" 的效果对比
    
    三种干预方式:
    1. direction_add: x + αv (改变值，不改变路由)
    2. routing_swap: 替换attention输出 (改变路由)
    3. combined: 同时改变值和路由
    
    如果PRNDS正确:
    - routing_swap >> direction_add
    - combined ≈ routing_swap (路由是主导因素)
    
    如果smooth:
    - routing_swap ≈ direction_add (两者等价)
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT B: 分支切换测试")
    print("=" * 70)
    print("  核心测试: 改变路由 vs 改变值 vs 同时改变")
    
    layers_list = get_layers(model)
    
    results = {li: {'direction': [], 'routing': [], 'combined': [], 
                     'attn_delta': [], 'resid_delta': []} 
               for li in target_layers}
    
    for si, (sn, pn, sv, pv) in enumerate(ALL_NVA[:n_test]):
        if si % 10 == 0 and si > 0:
            print(f"    Testing {si}/{n_test}...")
        
        sing_sent = f"The {sn} {sv}"
        plur_sent = f"The {pn} {pv}"
        
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
        
        sing_toks = tokenizer.encode(sing_sent, add_special_tokens=False)
        plur_toks = tokenizer.encode(plur_sent, add_special_tokens=False)
        
        sing_vp = find_verb_pos(tokenizer, sing_toks, sv, pv)
        plur_vp = find_verb_pos(tokenizer, plur_toks, sv, pv)
        
        if sing_vp is None or plur_vp is None:
            continue
        
        sv_ids = tokenizer.encode(sv, add_special_tokens=False)
        pv_ids = tokenizer.encode(pv, add_special_tokens=False)
        if not sv_ids or not pv_ids:
            continue
        
        # Baseline agreement for singular
        with torch.no_grad():
            base_out = model(**sing_ids)
            base_logits = base_out.logits.detach().cpu()
        base_agr = (base_logits[0, sing_vp, sv_ids[0]] - base_logits[0, sing_vp, pv_ids[0]]).item()
        
        for li in target_layers:
            # === 捕获sing和plur的attention输出和residual ===
            sing_attn_out = {}
            plur_attn_out = {}
            sing_resid_pre = {}
            plur_resid_pre = {}
            sing_ln_out = {}
            plur_ln_out = {}
            
            def cap_sing_attn(m, inp, out, _li=li):
                sing_attn_out[_li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
            def cap_plur_attn(m, inp, out, _li=li):
                plur_attn_out[_li] = (out[0] if isinstance(out, tuple) else out).detach().cpu()
            
            # 获取LN输入(即residual)
            ln = None
            for ln_name in ["input_layernorm", "ln_1"]:
                if hasattr(layers_list[li], ln_name):
                    ln = getattr(layers_list[li], ln_name)
                    break
            
            def cap_sing_ln(m, inp, out, _li=li):
                sing_ln_out[_li] = out.detach().cpu() if not isinstance(out, tuple) else out[0].detach().cpu()
                sing_resid_pre[_li] = inp[0].detach().cpu() if isinstance(inp, tuple) else inp.detach().cpu()
            def cap_plur_ln(m, inp, out, _li=li):
                plur_ln_out[_li] = out.detach().cpu() if not isinstance(out, tuple) else out[0].detach().cpu()
                plur_resid_pre[_li] = inp[0].detach().cpu() if isinstance(inp, tuple) else inp.detach().cpu()
            
            # Capture sing
            hooks = []
            hooks.append(layers_list[li].self_attn.register_forward_hook(cap_sing_attn))
            if ln is not None:
                hooks.append(ln.register_forward_hook(cap_sing_ln))
            with torch.no_grad():
                model(**sing_ids)
            for h in hooks:
                h.remove()
            
            # Capture plur
            hooks = []
            hooks.append(layers_list[li].self_attn.register_forward_hook(cap_plur_attn))
            if ln is not None:
                hooks.append(ln.register_forward_hook(cap_plur_ln))
            with torch.no_grad():
                model(**plur_ids)
            for h in hooks:
                h.remove()
            
            if li not in sing_attn_out or li not in plur_attn_out:
                continue
            if li not in sing_resid_pre or li not in plur_resid_pre:
                continue
            
            # === 计算差异 (只用verb位置，避免序列长度不匹配) ===
            # attention输出差异: routing component (verb位置)
            s_attn_verb = sing_attn_out[li][0, min(sing_vp, sing_attn_out[li].shape[1]-1), :].to(device)  # [d]
            p_attn_verb = plur_attn_out[li][0, min(plur_vp, plur_attn_out[li].shape[1]-1), :].to(device)  # [d]
            attn_delta_verb = (p_attn_verb - s_attn_verb).unsqueeze(0).unsqueeze(0)  # [1, 1, d]
            
            # residual差异: value component (verb位置)
            s_resid_verb = sing_resid_pre[li][0, min(sing_vp, sing_resid_pre[li].shape[1]-1), :].to(device)  # [d]
            p_resid_verb = plur_resid_pre[li][0, min(plur_vp, plur_resid_pre[li].shape[1]-1), :].to(device)  # [d]
            resid_delta_verb = (p_resid_verb - s_resid_verb).unsqueeze(0).unsqueeze(0)  # [1, 1, d]
            
            # 方向 (从residual差异提取)
            direction_np = resid_delta_verb[0, 0].float().cpu().numpy()
            direction_np = direction_np / (np.linalg.norm(direction_np) + 1e-10)
            direction_t = torch.tensor(direction_np, dtype=attn_delta_verb.dtype, device=device)
            
            # === 三种干预 ===
            
            # 1. direction_add: 只在residual上加方向
            #    在LN输入处注入方向
            alpha = 2.0
            direction_hook_added = [False]
            def direction_hook(m, inp, out):
                if not direction_hook_added[0]:
                    if isinstance(out, tuple):
                        new_out = out[0].clone()
                    else:
                        new_out = out.clone()
                    # 在verb位置加方向
                    if sing_vp < new_out.shape[1]:
                        new_out[0, sing_vp, :] += alpha * direction_t.to(new_out.dtype)
                    direction_hook_added[0] = True
                    return (new_out,) + out[1:] if isinstance(out, tuple) else new_out
                return out
            
            hook = ln.register_forward_hook(direction_hook) if ln is not None else None
            with torch.no_grad():
                out1 = model(**sing_ids)
            if hook:
                hook.remove()
            logits1 = out1.logits.detach().cpu()
            agr1 = (logits1[0, sing_vp, sv_ids[0]] - logits1[0, sing_vp, pv_ids[0]]).item()
            delta_dir = agr1 - base_agr
            
            # 2. routing_swap: 只替换attention输出（verb位置）
            routing_hook_added = [False]
            def routing_hook(m, inp, out, _attn_delta=attn_delta_verb, _vp=sing_vp):
                if not routing_hook_added[0]:
                    if isinstance(out, tuple):
                        new_out = out[0].clone()
                    else:
                        new_out = out.clone()
                    # 在verb位置加routing delta
                    vp = min(_vp, new_out.shape[1]-1)
                    new_out[0, vp, :] = new_out[0, vp, :] + _attn_delta[0, 0, :].to(new_out.dtype)
                    routing_hook_added[0] = True
                    return (new_out,) + out[1:] if isinstance(out, tuple) else new_out
                return out
            
            hook = layers_list[li].self_attn.register_forward_hook(routing_hook)
            with torch.no_grad():
                out2 = model(**sing_ids)
            hook.remove()
            logits2 = out2.logits.detach().cpu()
            agr2 = (logits2[0, sing_vp, sv_ids[0]] - logits2[0, sing_vp, pv_ids[0]]).item()
            delta_routing = agr2 - base_agr
            
            # 3. combined: 同时替换attention和加方向
            combined_hook_added = [False]
            def combined_hook_attn(m, inp, out, _attn_delta=attn_delta_verb, _vp=sing_vp):
                if not combined_hook_added[0]:
                    if isinstance(out, tuple):
                        new_out = out[0].clone()
                    else:
                        new_out = out.clone()
                    vp = min(_vp, new_out.shape[1]-1)
                    new_out[0, vp, :] = new_out[0, vp, :] + _attn_delta[0, 0, :].to(new_out.dtype)
                    combined_hook_added[0] = True
                    return (new_out,) + out[1:] if isinstance(out, tuple) else new_out
                return out
            
            # 对combined，我们只修改attention输出（已经包含了routing+value的变化）
            # 因为attn_delta = plur_attn - sing_attn，这本身已经是"完整路由交换"
            hook = layers_list[li].self_attn.register_forward_hook(combined_hook_attn)
            with torch.no_grad():
                out3 = model(**sing_ids)
            hook.remove()
            logits3 = out3.logits.detach().cpu()
            agr3 = (logits3[0, sing_vp, sv_ids[0]] - logits3[0, sing_vp, pv_ids[0]]).item()
            delta_combined = agr3 - base_agr
            
            results[li]['direction'].append(delta_dir)
            results[li]['routing'].append(delta_routing)
            results[li]['combined'].append(delta_combined)
            
            # 记录attn_delta和resid_delta的幅度
            attn_norm = attn_delta_verb[0, 0, :].float().norm().item()
            resid_norm = resid_delta_verb[0, 0, :].float().norm().item()
            results[li]['attn_delta'].append(attn_norm)
            results[li]['resid_delta'].append(resid_norm)
            
            del attn_delta_verb, resid_delta_verb, direction_t
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # ============================================================
    # 结果分析
    # ============================================================
    print("\n" + "-" * 60)
    print("分支切换测试结果")
    print("-" * 60)
    
    for li in target_layers:
        dirs = results[li]['direction']
        routs = results[li]['routing']
        combs = results[li]['combined']
        attn_norms = results[li]['attn_delta']
        resid_norms = results[li]['resid_delta']
        
        if len(dirs) < 3:
            continue
        
        d_mean, d_std = np.mean(dirs), np.std(dirs)
        r_mean, r_std = np.mean(routs), np.std(routs)
        c_mean, c_std = np.mean(combs), np.std(combs)
        a_norm = np.mean(attn_norms) if attn_norms else 0
        r_norm = np.mean(resid_norms) if resid_norms else 0
        
        print(f"\n  Layer {li} (n={len(dirs)}):")
        print(f"    direction_add: Δ={d_mean:+.4f} σ={d_std:.4f}")
        print(f"    routing_swap:  Δ={r_mean:+.4f} σ={r_std:.4f}")
        print(f"    combined:      Δ={c_mean:+.4f} σ={c_std:.4f}")
        print(f"    |routing|/|direction| = {abs(r_mean)/(abs(d_mean)+1e-6):.2f}")
        print(f"    attn_delta norm: {a_norm:.4f}")
        print(f"    resid_delta norm: {r_norm:.4f}")
        print(f"    |attn_delta|/|resid_delta| = {a_norm/(r_norm+1e-6):.4f}")
        
        # 关键判断
        if abs(r_mean) > 2 * abs(d_mean):
            print(f"    ★ routing >> direction → 支持PRNDS")
        elif abs(r_mean) < 0.5 * abs(d_mean):
            print(f"    ★ direction >> routing → 不支持PRNDS")
        else:
            print(f"    ★ routing ≈ direction → 混合模式")
    
    return results


# ============================================================
# 实验 C: 条件方向测试
# ============================================================
def experiment_C_conditional_directions(model, tokenizer, device, info, target_layers, n_test=60):
    """
    测试条件子空间 S(x) vs 全局子空间
    
    方法:
    1. 全局: 所有sing样本提取一个方向, 所有plur提取一个方向
    2. 条件: 按routing state分组, 每组内提取方向
    3. 比较: 条件方向是否比全局方向更一致(cos更高)
    
    如果PRNDS正确:
    - 条件方向cos >> 全局方向cos
    - 因为在同一个routing branch内, 语法信号更纯
    
    如果smooth:
    - 条件方向cos ≈ 全局方向cos
    - 因为没有离散分支，分组无意义
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT C: 条件方向测试")
    print("=" * 70)
    print("  假设: 如果条件子空间存在，同一routing组内方向更一致")
    
    layers_list = get_layers(model)
    
    # 收集数据: hidden states + attention patterns
    all_data = {li: {'sing_h': [], 'plur_h': [], 
                      'sing_attn': [], 'plur_attn': []}
                for li in target_layers}
    
    for si, (sn, pn, sv, pv) in enumerate(ALL_NVA[:n_test]):
        if si % 20 == 0 and si > 0:
            print(f"    Collecting {si}/{n_test}...")
        
        sing_sent = f"The {sn} {sv}"
        plur_sent = f"The {pn} {pv}"
        
        sing_ids = tokenizer(sing_sent, return_tensors="pt").to(device)
        plur_ids = tokenizer(plur_sent, return_tensors="pt").to(device)
        
        sing_toks = tokenizer.encode(sing_sent, add_special_tokens=False)
        sing_vp = find_verb_pos(tokenizer, sing_toks, sv, pv)
        plur_toks = tokenizer.encode(plur_sent, add_special_tokens=False)
        plur_vp = find_verb_pos(tokenizer, plur_toks, sv, pv)
        
        if sing_vp is None or plur_vp is None:
            continue
        
        with torch.no_grad():
            sing_out = model(**sing_ids, output_attentions=True, output_hidden_states=True)
            plur_out = model(**plur_ids, output_attentions=True, output_hidden_states=True)
        
        for li in target_layers:
            if li+1 >= len(sing_out.hidden_states):
                continue
            
            # Hidden state at verb position
            s_h = sing_out.hidden_states[li+1][0, sing_vp, :].detach().cpu().float().numpy()  # post-layer
            p_h = plur_out.hidden_states[li+1][0, plur_vp, :].detach().cpu().float().numpy()
            all_data[li]['sing_h'].append(s_h)
            all_data[li]['plur_h'].append(p_h)
            
            # Attention pattern at verb position
            if sing_out.attentions and li < len(sing_out.attentions):
                s_a = sing_out.attentions[li][0, :, sing_vp, :].detach().cpu().float().numpy()  # [heads, seq_k]
                p_a = plur_out.attentions[li][0, :, plur_vp, :].detach().cpu().float().numpy()
                all_data[li]['sing_attn'].append(s_a.flatten())
                all_data[li]['plur_attn'].append(p_a.flatten())
        
        del sing_out, plur_out
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ============================================================
    # 分析
    # ============================================================
    print("\n" + "-" * 60)
    print("条件方向分析结果")
    print("-" * 60)
    
    from sklearn.cluster import KMeans
    
    for li in target_layers:
        s_h = np.array(all_data[li]['sing_h'])  # [n_sing, d]
        p_h = np.array(all_data[li]['plur_h'])   # [n_plur, d]
        s_a = all_data[li]['sing_attn']
        p_a = all_data[li]['plur_attn']
        
        if len(s_h) < 10 or len(p_h) < 10:
            print(f"  Layer {li}: Not enough data")
            continue
        
        # === 全局方向 ===
        # sing的均值方向 vs plur的均值方向
        s_center = s_h.mean(axis=0)
        p_center = p_h.mean(axis=0)
        global_diff = p_center - s_center
        global_diff_norm = global_diff / (np.linalg.norm(global_diff) + 1e-10)
        
        # 全局cos: 每个样本与全局方向的对齐度
        s_cos_global = [np.dot(s_h[i] - s_center, global_diff_norm) / (np.linalg.norm(s_h[i] - s_center) + 1e-10) 
                        for i in range(len(s_h))]
        p_cos_global = [np.dot(p_h[i] - p_center, global_diff_norm) / (np.linalg.norm(p_h[i] - p_center) + 1e-10) 
                        for i in range(len(p_h))]
        global_alignment = np.mean(np.abs(s_cos_global + p_cos_global)) / 2
        
        # === 条件方向 ===
        # 用attention pattern聚类，然后每组内提取方向
        if len(s_a) >= 5 and len(p_a) >= 5:
            # 统一维度
            min_len = min(min(len(a) for a in s_a), min(len(a) for a in p_a))
            s_a_trim = np.array([a[:min_len] for a in s_a])
            p_a_trim = np.array([a[:min_len] for a in p_a])
            all_attn = np.vstack([s_a_trim, p_a_trim])
            all_h = np.vstack([s_h, p_h])
            
            # 聚成2-4组
            n_clusters = min(4, len(all_attn) // 5)
            n_clusters = max(n_clusters, 2)
            
            km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = km.fit_predict(all_attn)
            
            # 每组内提取方向
            conditional_cos = []
            conditional_sizes = []
            
            for c in range(n_clusters):
                mask = clusters == c
                group_h = all_h[mask]
                
                # 在组内，区分sing和plur
                n_sing_in_group = np.sum(mask[:len(s_h)])
                n_plur_in_group = np.sum(mask[len(s_h):])
                
                if n_sing_in_group < 3 or n_plur_in_group < 3:
                    continue
                
                sing_in_group = all_h[mask][:n_sing_in_group] if n_sing_in_group > 0 else None
                plur_in_group = all_h[mask][n_sing_in_group:] if n_plur_in_group > 0 else None
                
                if sing_in_group is None or plur_in_group is None:
                    continue
                
                # 组内方向
                s_cg = sing_in_group.mean(axis=0)
                p_cg = plur_in_group.mean(axis=0)
                cond_diff = p_cg - s_cg
                cond_diff_norm = cond_diff / (np.linalg.norm(cond_diff) + 1e-10)
                
                # 组内cos
                s_cos_c = [np.dot(sing_in_group[i] - s_cg, cond_diff_norm) / 
                           (np.linalg.norm(sing_in_group[i] - s_cg) + 1e-10)
                           for i in range(len(sing_in_group))]
                p_cos_c = [np.dot(plur_in_group[i] - p_cg, cond_diff_norm) / 
                           (np.linalg.norm(plur_in_group[i] - p_cg) + 1e-10)
                           for i in range(len(plur_in_group))]
                
                mean_cos_c = np.mean(np.abs(s_cos_c + p_cos_c)) / 2
                conditional_cos.append(mean_cos_c)
                conditional_sizes.append(n_sing_in_group + n_plur_in_group)
            
            if conditional_cos:
                mean_cond_cos = np.mean(conditional_cos)
                # 加权平均
                weighted_cond_cos = np.average(conditional_cos, weights=conditional_sizes)
                
                print(f"\n  Layer {li} (n_clusters={n_clusters}):")
                print(f"    Global alignment: {global_alignment:.4f}")
                print(f"    Conditional alignment: {mean_cond_cos:.4f}")
                print(f"    Weighted conditional: {weighted_cond_cos:.4f}")
                print(f"    Improvement: {mean_cond_cos/(global_alignment+1e-6):.2f}x")
                
                # 每个cluster的方向一致性
                for ci, (cc, sz) in enumerate(zip(conditional_cos, conditional_sizes)):
                    print(f"      Cluster {ci}: cos={cc:.4f} n={sz}")
                
                if mean_cond_cos > 1.5 * global_alignment:
                    print(f"    ★ 条件方向 >> 全局方向 → 支持条件子空间")
                elif mean_cond_cos < 0.8 * global_alignment:
                    print(f"    ★ 全局方向 >> 条件方向 → 不支持条件子空间")
                else:
                    print(f"    ★ 条件方向 ≈ 全局方向 → 条件效应不显著")
            else:
                print(f"\n  Layer {li}: Not enough data for conditional analysis")
        else:
            print(f"\n  Layer {li}: Not enough attention data")
    
    return all_data


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="deepseek7b", 
                       choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"Phase 64: PRNDS验证 (Piecewise-Routed Nonlinear Dynamical System)")
    print(f"Model: {args.model}")
    print("=" * 70)
    
    model, tokenizer, device = load_model(args.model)
    info = get_model_info(model, args.model)
    print(f"  Model: {info.model_class}, Layers: {info.n_layers}, d_model: {info.d_model}")
    
    # 调整target_layers
    global TARGET_LAYERS
    TARGET_LAYERS = [0, min(5, info.n_layers-1), min(10, info.n_layers-1), 
                     min(15, info.n_layers-1), info.n_layers-1]
    TARGET_LAYERS = sorted(set(TARGET_LAYERS))
    print(f"  Target layers: {TARGET_LAYERS}")
    
    t0 = time.time()
    
    # 实验 A: 路由状态提取
    results_A = experiment_A_routing_states(model, tokenizer, device, info, TARGET_LAYERS, n_test=80)
    
    t1 = time.time()
    print(f"\n[Time] Experiment A: {t1-t0:.1f}s")
    
    # 实验 B: 分支切换
    results_B = experiment_B_branch_switching(model, tokenizer, device, info, TARGET_LAYERS, n_test=50)
    
    t2 = time.time()
    print(f"\n[Time] Experiment B: {t2-t1:.1f}s")
    
    # 实验 C: 条件方向
    results_C = experiment_C_conditional_directions(model, tokenizer, device, info, TARGET_LAYERS, n_test=60)
    
    t3 = time.time()
    print(f"\n[Time] Experiment C: {t3-t2:.1f}s")
    print(f"\n[Time] Total: {t3-t0:.1f}s")
    
    # ============================================================
    # 综合判断
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 64 综合判断: PRNDS vs Smooth")
    print("=" * 70)
    
    # 从三个实验汇总证据
    evidence_prnds = 0
    evidence_smooth = 0
    
    # A: 路由状态离散性
    if results_A:
        aris = [r['attn_ari'] for r in results_A.values()]
        mean_ari = np.mean(aris) if aris else 0
        if mean_ari > 0.2:
            evidence_prnds += 2
            print(f"  A: 路由状态ARI={mean_ari:.3f} → 离散 (+2 PRNDS)")
        elif mean_ari > 0.1:
            evidence_prnds += 1
            print(f"  A: 路由状态ARI={mean_ari:.3f} → 弱离散 (+1 PRNDS)")
        else:
            evidence_smooth += 2
            print(f"  A: 路由状态ARI={mean_ari:.3f} → 连续 (+2 Smooth)")
    
    # B: 分支切换效果
    for li in TARGET_LAYERS:
        dirs = results_B[li]['direction']
        routs = results_B[li]['routing']
        if len(dirs) >= 3:
            r_mean = abs(np.mean(routs))
            d_mean = abs(np.mean(dirs))
            if r_mean > 2 * d_mean:
                evidence_prnds += 1
                print(f"  B-L{li}: |routing|/|direction|={r_mean/(d_mean+1e-6):.1f} → routing主导 (+1 PRNDS)")
            elif d_mean > 2 * r_mean:
                evidence_smooth += 1
                print(f"  B-L{li}: |direction|/|routing|={d_mean/(r_mean+1e-6):.1f} → direction主导 (+1 Smooth)")
    
    # C: 条件方向
    print(f"\n  C: 见上方条件方向分析结果")
    
    print(f"\n  === 总评 ===")
    print(f"  PRNDS证据: {evidence_prnds}")
    print(f"  Smooth证据: {evidence_smooth}")
    
    if evidence_prnds > evidence_smooth + 2:
        print(f"  ★ 结论: 强支持PRNDS — 系统倾向于分段动力系统")
    elif evidence_prnds > evidence_smooth:
        print(f"  ★ 结论: 弱支持PRNDS — 有分段特征但不纯粹")
    elif evidence_smooth > evidence_prnds + 2:
        print(f"  ★ 结论: 不支持PRNDS — 系统倾向于连续动力系统")
    else:
        print(f"  ★ 结论: 混合 — 系统可能介于分段和连续之间")
    
    release_model(model)
    print("\nPhase 64 完成!")


if __name__ == "__main__":
    main()
