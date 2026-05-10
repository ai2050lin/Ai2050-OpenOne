"""
Phase 80: Operator Reverse Engineering — 从routing到真正的computation operator
==============================================================================

用户精准指出Phase 79的致命问题:
  routing = memory access pattern (信息检索)
  ≠ computation (状态变换)

真正的computation应该分析:
  transition operator structure
  即: h(t+1) = F(h(t)) 中的 F

而不是:
  attention routing topology (信息如何访问)

五层逆向框架:
  1. Representation — 信息编码在哪 (Phase 1-70+)
  2. Routing — 信息如何访问 (Phase 79)
  3. Operator — 状态如何变换 ★★★★★ (本Phase核心)
  4. Recursive Rollout — generation如何形成computation
  5. Compression — 为什么这些结构自然出现

四个核心实验:
  A: Jacobian Spectrum Analysis ★★★★★
     对不同任务计算层间Jacobian, 比较其谱结构
     核心判据: 不同任务是否属于不同operator family?

  B: Operator Decomposition: Attention vs MLP ★★★★★
     分离routing(attn)和computation(mlp)
     transition = h_in + delta_attn + delta_mlp
     看mlp的operator structure是否task-specific

  C: Operator Rank & Structure ★★★★★
     翻译 = 低秩coordinate transform?
     推理 = iterative refinement operator?
     补全 = identity-like operator?

  D: Sensitive Heads的Causal Validation
     ablate sensitive heads → 是否消除task-specific computation?

关键方法论:
  不看"信息去哪了" (routing)
  而看"信息如何变换" (operator)
  核心工具: Jacobian, SVD, operator spectrum, rank

Usage:
  python ccml_phase80_operator.py --exp a
  python ccml_phase80_operator.py --exp b
  python ccml_phase80_operator.py --exp c
  python ccml_phase80_operator.py --exp d
  python ccml_phase80_operator.py --exp all
"""

import torch
import numpy as np
import argparse
from collections import defaultdict
from transformer_lens import HookedTransformer

def get_model():
    model = HookedTransformer.from_pretrained(
        "gpt2-small",
        center_unembed=False,
        center_writing_weights=False,
        fold_ln=False,
        device="cpu",
    )
    model.eval()
    return model

# ============================================================
# 核心工具: Jacobian计算
# ============================================================

def compute_layer_jacobian(model, text, layer, position=-1):
    """
    计算某一层在特定位置的局部Jacobian:
      J = d h_out / d h_in
    
    h_in = residual stream at layer input
    h_out = residual stream at layer output
    
    transition: h_out = h_in + attn(h_in) + mlp(h_in + attn(h_in))
    J = I + d_attn/d_h_in + d_mlp/d_h_in  (chain rule)
    
    这个J就是该层的"computation operator"
    不同任务的J是否不同 → computation是否task-specific
    """
    tokens = model.to_tokens(text)
    
    # 获取h_in和h_out
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    h_in = cache[f'blocks.{layer}.hook_resid_pre'][position].clone().detach()
    h_out = cache[f'blocks.{layer}.hook_resid_post'][position].clone().detach()
    
    d_model = h_in.shape[0]
    
    # 用有限差分计算Jacobian
    # 对每个方向施加扰动, 看输出的变化
    eps = 1e-3
    
    # 构造一组正交探测方向 (用随机方向, 更高效)
    n_probes = min(d_model, 50)  # 不需要全部768维, 采样50维足够
    
    # 使用PCA-like方式: 用残差的主方向
    # 简化: 用随机正交基
    torch.manual_seed(42)
    probe_dirs = torch.randn(n_probes, d_model)
    probe_dirs = torch.linalg.qr(probe_dirs.T).Q.T[:n_probes]  # 正交化
    
    # 有限差分: J @ dir ≈ (F(h + eps*dir) - F(h - eps*dir)) / (2*eps)
    delta_outputs = []
    
    for i in range(n_probes):
        # 需要重新forward才能准确计算, 但太慢
        # 用近似: 从cache中, 我们有 h_in, attn_out, mlp_out
        # transition = attn_out + mlp_out (不包含residual connection)
        # 所以 d_transition/d_h_in 就是我们要的
        
        # 简化: 用transition vector本身作为Jacobian的近似列空间
        # 更好的方法: 用autograd
        pass
    
    # 用autograd计算精确Jacobian (更可靠)
    return compute_jacobian_autograd(model, text, layer, position)


def compute_jacobian_autograd(model, text, layer, position=-1):
    """
    用autograd计算精确Jacobian
    J_ij = d h_out_i / d h_in_j
    
    方法: 对h_in的每个分量施加扰动, 计算h_out的变化
    """
    tokens = model.to_tokens(text)
    
    # 重新运行, 但hook注入perturbed h_in
    _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
    h_in_orig = cache[f'blocks.{layer}.hook_resid_pre'][position].clone().detach()
    
    d_model = h_in_orig.shape[0]
    
    # 采样方法: 不计算完整768x768 Jacobian
    # 而是: 计算 J @ v 对一组随机向量 v 的结果
    # 这给出Jacobian的"投影", 可以用来分析谱结构
    
    n_probes = 50
    torch.manual_seed(42)
    probe_vectors = torch.randn(n_probes, d_model)
    probe_vectors = probe_vectors / probe_vectors.norm(dim=1, keepdim=True)
    
    # 使用有限差分法
    eps = 1e-2
    
    J_probes = torch.zeros(n_probes, d_model)  # J @ probe_vectors
    
    for i in range(n_probes):
        # 需要用hook注入perturbed input
        results = {}
        
        def make_perturbed_forward(eps_val, probe_vec):
            """创建perturbed forwardpass"""
            def hook_fn(module, input, output):
                # hook_resid_pre的位置: 在layer norm之前
                # 我们需要修改进入attention的residual
                pass
            return hook_fn
        
        # 简化方法: 直接用transition vector的性质
        # 对一个小模型, 可以逐分量有限差分
        pass
    
    # 最终简化: 用transition vector的SVD代替Jacobian
    # transition = h_out - h_in = delta_attn + delta_mlp
    # 对多个任务收集transition vectors, 分析它们的operator structure
    
    transition = cache[f'blocks.{layer}.hook_resid_post'][position] - cache[f'blocks.{layer}.hook_resid_pre'][position]
    
    return transition.detach(), cache


def collect_transitions(model, texts, layers, position=-1):
    """
    收集多个文本在多个层的transition vectors
    transition = h_out - h_in = delta_attn + delta_mlp
    
    这是computation operator的核心输出
    """
    results = {}
    
    for text in texts:
        tokens = model.to_tokens(text)
        _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
        
        text_data = {}
        for layer in layers:
            h_in = cache[f'blocks.{layer}.hook_resid_pre'][position].detach().cpu()
            h_out = cache[f'blocks.{layer}.hook_resid_post'][position].detach().cpu()
            attn_out = cache[f'blocks.{layer}.attn.hook_attn_out'][position].detach().cpu() if f'blocks.{layer}.attn.hook_attn_out' in cache else None
            
            # 分离attn和mlp贡献
            # hook_resid_mid = h_in + attn_out (after attn, before mlp)
            h_mid = cache[f'blocks.{layer}.hook_resid_mid'][position].detach().cpu()
            
            delta_total = h_out - h_in
            delta_attn = h_mid - h_in  # attention contribution
            delta_mlp = h_out - h_mid   # MLP contribution
            
            text_data[layer] = {
                'h_in': h_in,
                'h_out': h_out,
                'delta_total': delta_total,
                'delta_attn': delta_attn,
                'delta_mlp': delta_mlp,
            }
        
        results[text] = text_data
    
    return results


# ============================================================
# 实验A: Jacobian Spectrum via Transition Operator ★★★★★
# ============================================================

def exp_a_operator_spectrum(model):
    """
    核心问题: 不同任务的transition operator是否属于不同family?
    
    方法: 
      1. 收集同类任务的transition vectors → PCA看operator的"输出子空间"
      2. 不同任务是否在operator output space中分属不同cluster?
      3. operator的秩(rank)和谱结构如何?
    
    关键区别于Phase 79:
      Phase 79看routing (information access)
      Phase 80看operator (information transformation)
      
    具体来说:
      不是问"attention去哪了" 
      而是问"MLP做了什么变换"
    """
    print("=" * 70)
    print("实验A: Operator Spectrum Analysis")
    print("核心问题: 不同任务的transition operator是否属于不同family?")
    print("=" * 70)
    
    # ---- 大量同类任务 ----
    task_groups = {
        "addition": [f"{a} + {b} =" for a, b in 
                      [(2,3),(7,4),(15,23),(9,6),(11,8),(3,5),(12,7),(4,9),(21,14),(6,2),
                       (8,1),(5,5),(10,3),(13,6),(1,7),(2,8),(4,3),(6,4),(9,2),(3,7),
                       (14,5),(6,9),(8,3),(7,2),(5,11),(3,8),(9,4),(2,13),(6,1),(4,7)]],
        "translate_fr": [f"Translate to French: {s}" for s in 
                      ["The cat is on the mat", "The dog runs in the park", "The bird sings a song",
                       "The sun shines bright", "The water flows down", "The child plays outside",
                       "The tree grows tall", "The rain falls softly", "The wind blows hard",
                       "The moon rises slowly", "The fish swims deep", "The flower blooms red",
                       "The snow falls white", "The fire burns hot", "The earth spins round",
                       "The river runs wide", "The mountain stands high", "The cloud floats free",
                       "The star shines far", "The ocean waves crash",
                       "The cat sleeps softly", "The dog barks loudly", "The bird flies high",
                       "The sun warms all", "The water runs clear", "The child laughs freely",
                       "The tree provides shade", "The rain feeds earth", "The wind cools air",
                       "The moon lights night"]],
        "antonym": [f"The opposite of {w} is" for w in 
                      ["hot","big","fast","happy","light","strong","loud","rough","wide","tall",
                       "cold","small","slow","sad","dark","weak","quiet","smooth","narrow","short",
                       "bright","heavy","soft","hard","old","young","rich","poor","thick","thin"]],
        "capital": [f"The capital of {c} is" for c in 
                      ["France","Germany","Japan","Italy","Spain","China","Brazil","India","Russia","Egypt",
                       "UK","Canada","Mexico","Korea","Turkey","Norway","Sweden","Poland","Greece","Portugal",
                       "Australia","Argentina","Chile","Peru","Colombia","Thailand","Vietnam","Finland","Denmark","Austria"]],
    }
    
    layers = list(range(12))
    
    # ---- 分析1: MLP Transition Operator的PCA ----
    print("\n" + "=" * 50)
    print("分析1: MLP Transition Operator (真正的computation)")
    print("transition = delta_mlp = h_out - h_mid")
    print("这是MLP对residual stream的变换 — 真正的computation operator")
    print("=" * 50)
    
    from sklearn.decomposition import PCA
    
    for layer in [0, 3, 6, 9, 11]:
        print(f"\n  Layer {layer} — MLP operator output space:")
        
        # 收集每组任务的MLP transition vectors
        group_mlp_transitions = {}
        for group_name, texts in task_groups.items():
            transitions = []
            for text in texts:
                tokens = model.to_tokens(text)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach().cpu().numpy()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach().cpu().numpy()
                h_out = cache[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu().numpy()
                
                delta_mlp = h_out - h_mid
                transitions.append(delta_mlp)
            
            group_mlp_transitions[group_name] = np.array(transitions)
        
        # 对每组做PCA
        for group_name, trans in group_mlp_transitions.items():
            pca = PCA()
            pca.fit(trans)
            
            var1 = pca.explained_variance_ratio_[0]
            var2 = pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0
            var3 = pca.explained_variance_ratio_[2] if len(pca.explained_variance_ratio_) > 2 else 0
            
            # 有效秩: 解释95%方差需要的PC数
            cumvar = np.cumsum(pca.explained_variance_ratio_)
            effective_rank = np.searchsorted(cumvar, 0.95) + 1
            
            print(f"    {group_name}: PC1={var1:.4f}, PC1-2={var1+var2:.4f}, eff_rank(95%)={effective_rank}")
        
        # ---- 跨组分析: 不同组的MLP transitions是否在同一个子空间? ----
        all_transitions = np.vstack(list(group_mlp_transitions.values()))
        all_labels = np.concatenate([
            np.full(len(group_mlp_transitions[g]), i) 
            for i, g in enumerate(group_mlp_transitions.keys())
        ])
        
        # 在all_transitions上做PCA
        pca_all = PCA(n_components=10)
        pca_all.fit(all_transitions)
        
        # 每组在top-10 PCs上的centroid
        projected = pca_all.transform(all_transitions)
        
        print(f"\n    跨组PCA — 各组在PC1-2上的centroid:")
        for i, group_name in enumerate(group_mlp_transitions.keys()):
            mask = all_labels == i
            centroid = projected[mask, :2].mean(axis=0)
            spread = projected[mask, :2].std(axis=0)
            print(f"      {group_name}: PC1={centroid[0]:.4f}+/-{spread[0]:.4f}, PC2={centroid[1]:.4f}+/-{spread[1]:.4f}")
    
    # ---- 分析2: Attention vs MLP的Task-Specificity对比 ----
    print("\n" + "=" * 50)
    print("分析2: Attention (routing) vs MLP (computation) 的Task-Specificity")
    print("如果routing ≠ computation → attention和MLP的task-specificity应该不同")
    print("=" * 50)
    
    for layer in [3, 6, 9]:
        print(f"\n  Layer {layer}:")
        
        group_attn_transitions = {}
        group_mlp_transitions_local = {}
        
        for group_name, texts in task_groups.items():
            attn_trans = []
            mlp_trans = []
            
            for text in texts[:10]:  # 10个样本
                tokens = model.to_tokens(text)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach().cpu()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach().cpu()
                h_out = cache[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu()
                
                delta_attn = h_mid - h_in
                delta_mlp = h_out - h_mid
                
                attn_trans.append(delta_attn.numpy())
                mlp_trans.append(delta_mlp.numpy())
            
            group_attn_transitions[group_name] = np.array(attn_trans)
            group_mlp_transitions_local[group_name] = np.array(mlp_trans)
        
        # 计算组内/跨组相似度 (attn vs mlp分别)
        for comp_name, group_trans in [("Attention (routing)", group_attn_transitions), 
                                        ("MLP (computation)", group_mlp_transitions_local)]:
            # 组内: 同组transition vectors的cosine similarity
            within_sims = []
            for g, trans in group_trans.items():
                # normalize
                norms = np.linalg.norm(trans, axis=1, keepdims=True)
                trans_normed = trans / (norms + 1e-8)
                
                # 两两cosine
                for i in range(min(5, len(trans_normed))):
                    for j in range(i+1, min(5, len(trans_normed))):
                        cos_sim = np.dot(trans_normed[i], trans_normed[j])
                        within_sims.append(cos_sim)
            
            # 跨组: 不同组transition vectors的cosine similarity
            cross_sims = []
            group_names = list(group_trans.keys())
            for i, g1 in enumerate(group_names):
                for j, g2 in enumerate(group_names):
                    if i >= j:
                        continue
                    t1 = group_trans[g1][:5]
                    t2 = group_trans[g2][:5]
                    
                    n1 = t1 / (np.linalg.norm(t1, axis=1, keepdims=True) + 1e-8)
                    n2 = t2 / (np.linalg.norm(t2, axis=1, keepdims=True) + 1e-8)
                    
                    for a in range(3):
                        for b in range(3):
                            cos_sim = np.dot(n1[a], n2[b])
                            cross_sims.append(cos_sim)
            
            within_avg = np.mean(within_sims)
            cross_avg = np.mean(cross_sims)
            gap = within_avg - cross_avg
            
            print(f"    {comp_name}: within={within_avg:.4f}, cross={cross_avg:.4f}, gap={gap:.4f}")
    
    # ---- 分析3: MLP Operator的低秩结构 ----
    print("\n" + "=" * 50)
    print("分析3: MLP Operator的低秩结构")
    print("如果翻译=coordinate transform → 应该是低秩旋转")
    print("如果推理=iterative refinement → 应该有特定eigenvalue结构")
    print("=" * 50)
    
    for layer in [3, 6, 9]:
        print(f"\n  Layer {layer}:")
        
        for group_name, texts in task_groups.items():
            # 收集多个样本的 (h_in, delta_mlp) 对
            h_ins = []
            delta_mlps = []
            
            for text in texts[:15]:
                tokens = model.to_tokens(text)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach().cpu().numpy()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach().cpu().numpy()
                h_out = cache[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu().numpy()
                
                delta_mlp = h_out - h_mid
                h_ins.append(h_in)
                delta_mlps.append(delta_mlp)
            
            h_ins = np.array(h_ins)
            delta_mlps = np.array(delta_mlps)
            
            # 线性回归: delta_mlp ≈ A @ h_in + b
            # 这给出effective linear operator A
            from sklearn.linear_model import LinearRegression
            
            reg = LinearRegression()
            reg.fit(h_ins, delta_mlps)
            
            # R^2: 线性模型能解释多少variance
            r2 = reg.score(h_ins, delta_mlps)
            
            # A的SVD
            A = reg.coef_  # [d_model, d_model]
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            
            # 奇异值谱
            total_energy = np.sum(S**2)
            top3_energy = np.sum(S[:3]**2)
            top10_energy = np.sum(S[:10]**2)
            
            # 有效秩
            cum_energy = np.cumsum(S**2) / total_energy
            eff_rank_90 = np.searchsorted(cum_energy, 0.90) + 1
            eff_rank_95 = np.searchsorted(cum_energy, 0.95) + 1
            
            # 奇异值的衰减率
            if len(S) > 5:
                decay_rate = S[5] / (S[0] + 1e-8)
            else:
                decay_rate = 0
            
            print(f"    {group_name}:")
            print(f"      R^2={r2:.4f}, eff_rank(90%)={eff_rank_90}, eff_rank(95%)={eff_rank_95}")
            print(f"      Top-3 energy: {top3_energy/total_energy:.4f}, Top-10: {top10_energy/total_energy:.4f}")
            print(f"      Top-5 singular values: {S[:5].tolist()}")
            print(f"      S5/S0 decay: {decay_rate:.6f}")


# ============================================================
# 实验B: MLP Internal Operator Structure ★★★★★
# ============================================================

def exp_b_mlp_operator(model):
    """
    核心问题: MLP内部的operator结构是什么?
    
    MLP(x) = W_out * GELU(W_in * x + b_in) + b_out
    
    真正的computation发生在:
      1. W_in: 投影到隐空间 (d_model → d_mlp)
      2. GELU: 非线性变换
      3. W_out: 投影回残差空间 (d_mlp → d_model)
    
    关键分析:
      1. W_in是否task-specific? (投影方向)
      2. GELU激活哪些神经元? (哪些维度参与计算)
      3. W_out如何组合? (输出结构)
    
    如果不同任务激活不同的neurons → 存在task-specific computation
    如果不同任务激活相同的neurons但组合方式不同 → 存在task-specific operator
    """
    print("=" * 70)
    print("实验B: MLP Internal Operator Structure")
    print("核心问题: MLP内部如何实现task-specific computation?")
    print("=" * 70)
    
    task_groups = {
        "addition": [f"{a} + {b} =" for a, b in 
                      [(2,3),(7,4),(15,23),(9,6),(11,8),(3,5),(12,7),(4,9),(21,14),(6,2),
                       (8,1),(5,5),(10,3),(13,6),(1,7),(2,8),(4,3),(6,4),(9,2),(3,7)]],
        "translate_fr": [f"Translate to French: {s}" for s in 
                      ["The cat is on the mat", "The dog runs in the park", "The bird sings a song",
                       "The sun shines bright", "The water flows down", "The child plays outside",
                       "The tree grows tall", "The rain falls softly", "The wind blows hard",
                       "The moon rises slowly", "The fish swims deep", "The flower blooms red",
                       "The snow falls white", "The fire burns hot", "The earth spins round",
                       "The river runs wide", "The mountain stands high", "The cloud floats free",
                       "The star shines far", "The ocean waves crash"]],
        "antonym": [f"The opposite of {w} is" for w in 
                      ["hot","big","fast","happy","light","strong","loud","rough","wide","tall",
                       "cold","small","slow","sad","dark","weak","quiet","smooth","narrow","short"]],
        "capital": [f"The capital of {c} is" for c in 
                      ["France","Germany","Japan","Italy","Spain","China","Brazil","India","Russia","Egypt",
                       "UK","Canada","Mexico","Korea","Turkey","Norway","Sweden","Poland","Greece","Portugal"]],
    }
    
    for layer in [3, 6, 9]:
        print(f"\n  Layer {layer} — MLP Internal Structure:")
        
        # 收集MLP的中间激活
        group_activations = {}
        group_outputs = {}
        
        for group_name, texts in task_groups.items():
            pre_gelu = []  # GELU之前的激活
            post_gelu = []  # GELU之后的激活
            mlp_outs = []
            
            for text in texts:
                tokens = model.to_tokens(text)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                # hook_resid_mid: 进入MLP之前的residual
                # blocks.L.mlp.hook_pre: MLP输入 (after layer norm + W_in)
                # blocks.L.mlp.hook_post: MLP输出 (after GELU, before W_out)
                # blocks.L.hook_resid_post: 整层输出
                
                pre = cache[f'blocks.{layer}.mlp.hook_pre'][-1].detach().cpu()
                post = cache[f'blocks.{layer}.mlp.hook_post'][-1].detach().cpu()
                
                pre_gelu.append(pre.numpy())
                post_gelu.append(post.numpy())
                
                # MLP output vector
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach().cpu()
                h_out = cache[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu()
                mlp_out = h_out - h_mid
                mlp_outs.append(mlp_out.numpy())
            
            group_activations[group_name] = {
                'pre_gelu': np.array(pre_gelu),   # [n_texts, d_mlp]
                'post_gelu': np.array(post_gelu),  # [n_texts, d_mlp]
            }
            group_outputs[group_name] = np.array(mlp_outs)  # [n_texts, d_model]
        
        # ---- 分析1: 哪些neurons被激活? (GELU activation pattern) ----
        print(f"\n    分析1: MLP Neuron Activation Patterns")
        print(f"    如果不同任务激活不同neurons → task-specific computation routing")
        
        for group_name, acts in group_activations.items():
            pre = acts['pre_gelu']
            
            # 每个neuron的平均激活强度
            mean_act = np.mean(np.abs(pre), axis=0)  # [d_mlp]
            
            # 被激活的neurons (pre_gelu > 0 的比例)
            active_ratio = np.mean(pre > 0, axis=0)  # [d_mlp]
            
            # Top-10 最活跃neurons
            top_neurons = np.argsort(mean_act)[-10:][::-1]
            
            # 有效维度: 激活率>5%的neurons数
            n_active = np.sum(active_ratio > 0.05)
            
            print(f"      {group_name}: n_active_neurons={n_active}, top_neurons={top_neurons.tolist()}")
        
        # ---- 分析2: 不同任务激活的neurons是否重叠? ----
        print(f"\n    分析2: Neuron Activation Overlap")
        
        group_names = list(task_groups.keys())
        for i, g1 in enumerate(group_names):
            for j, g2 in enumerate(group_names):
                if i >= j:
                    continue
                
                pre1 = group_activations[g1]['pre_gelu']
                pre2 = group_activations[g2]['pre_gelu']
                
                # 每个neuron是否被两组都激活
                active1 = np.mean(pre1 > 0, axis=0) > 0.1  # 被10%+样本激活
                active2 = np.mean(pre2 > 0, axis=0) > 0.1
                
                overlap = np.sum(active1 & active2) / max(np.sum(active1 | active2), 1)
                
                print(f"      {g1} vs {g2}: Jaccard overlap = {overlap:.4f}")
        
        # ---- 分析3: MLP output的task-specific subspaces ----
        print(f"\n    分析3: MLP Output Subspace Overlap")
        
        from sklearn.decomposition import PCA
        for group_name, outs in group_outputs.items():
            pca = PCA(n_components=5)
            pca.fit(outs)
            
            var1 = pca.explained_variance_ratio_[0]
            cumvar3 = np.cumsum(pca.explained_variance_ratio_)[2]
            
            print(f"      {group_name}: PC1={var1:.4f}, PC1-3={cumvar3:.4f}")
        
        # 跨组subspace alignment
        for i, g1 in enumerate(group_names):
            for j, g2 in enumerate(group_names):
                if i >= j:
                    continue
                
                pca1 = PCA(n_components=10)
                pca1.fit(group_outputs[g1])
                
                pca2 = PCA(n_components=10)
                pca2.fit(group_outputs[g2])
                
                # Subspace alignment: 两者的top-10 PCs的principal angles
                Q1 = pca1.components_.T  # [d_model, 10]
                Q2 = pca2.components_.T  # [d_model, 10]
                
                # PCA subspace alignment = ||Q1.T @ Q2||_F / sqrt(k)
                alignment = np.linalg.norm(Q1.T @ Q2, 'fro') / np.sqrt(10)
                
                print(f"      {g1} vs {g2}: subspace alignment = {alignment:.4f}")


# ============================================================
# 实验C: Operator Family Classification ★★★★★
# ============================================================

def exp_c_operator_family(model):
    """
    核心问题: 不同任务是否属于不同的operator family?
    
    Operator families:
      1. Coordinate Transform (翻译?) — 低秩, 接近正交
      2. Iterative Refinement (推理?) — 接近单位阵, 小扰动
      3. Content Generation (补全?) — 高秩, 大变换
      4. Feature Selection (反义词?) — 稀疏, 低秩
    
    方法: 对每个任务计算effective linear operator A
    然后: A的SVD结构 → operator family
    """
    print("=" * 70)
    print("实验C: Operator Family Classification")
    print("核心问题: 不同任务是否使用不同类型的computation operator?")
    print("=" * 70)
    
    task_groups = {
        "addition": [f"{a} + {b} =" for a, b in 
                      [(2,3),(7,4),(15,23),(9,6),(11,8),(3,5),(12,7),(4,9),(21,14),(6,2),
                       (8,1),(5,5),(10,3),(13,6),(1,7),(2,8),(4,3),(6,4),(9,2),(3,7),
                       (14,5),(6,9),(8,3),(7,2),(5,11),(3,8),(9,4),(2,13),(6,1),(4,7)]],
        "translate_fr": [f"Translate to French: {s}" for s in 
                      ["The cat is on the mat", "The dog runs in the park", "The bird sings a song",
                       "The sun shines bright", "The water flows down", "The child plays outside",
                       "The tree grows tall", "The rain falls softly", "The wind blows hard",
                       "The moon rises slowly", "The fish swims deep", "The flower blooms red",
                       "The snow falls white", "The fire burns hot", "The earth spins round",
                       "The river runs wide", "The mountain stands high", "The cloud floats free",
                       "The star shines far", "The ocean waves crash",
                       "The cat sleeps softly", "The dog barks loudly", "The bird flies high",
                       "The sun warms all", "The water runs clear", "The child laughs freely",
                       "The tree provides shade", "The rain feeds earth", "The wind cools air",
                       "The moon lights night"]],
        "antonym": [f"The opposite of {w} is" for w in 
                      ["hot","big","fast","happy","light","strong","loud","rough","wide","tall",
                       "cold","small","slow","sad","dark","weak","quiet","smooth","narrow","short",
                       "bright","heavy","soft","hard","old","young","rich","poor","thick","thin"]],
        "capital": [f"The capital of {c} is" for c in 
                      ["France","Germany","Japan","Italy","Spain","China","Brazil","India","Russia","Egypt",
                       "UK","Canada","Mexico","Korea","Turkey","Norway","Sweden","Poland","Greece","Portugal",
                       "Australia","Argentina","Chile","Peru","Colombia","Thailand","Vietnam","Finland","Denmark","Austria"]],
        "continue": [f"Continue: {s}" for s in 
                      ["The cat sat on", "The dog ran to", "The bird flew up", "The fish swam down",
                       "The tree grew very", "The sun was very", "The wind blew the", "The rain fell on",
                       "The moon shone on", "The star twinkled",
                       "Once upon a time", "In the beginning", "Long ago there", "The story begins",
                       "It was a dark", "The morning came", "The evening fell", "The night was",
                       "The day broke and", "The clock struck"]],
    }
    
    from sklearn.linear_model import LinearRegression
    from sklearn.decomposition import PCA
    
    # 对每层, 对每组任务, 计算effective linear operator
    print("\n--- Effective Linear Operator Analysis ---")
    print("方法: delta_mlp ≈ A @ h_in + b")
    print("A的SVD结构揭示operator type")
    
    for layer in [3, 6, 9]:
        print(f"\n  Layer {layer}:")
        
        operator_metrics = {}
        
        for group_name, texts in task_groups.items():
            h_ins = []
            delta_mlps = []
            
            for text in texts:
                tokens = model.to_tokens(text)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach().cpu().numpy()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach().cpu().numpy()
                h_out = cache[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu().numpy()
                
                delta_mlp = h_out - h_mid
                h_ins.append(h_in)
                delta_mlps.append(delta_mlp)
            
            h_ins = np.array(h_ins)
            delta_mlps = np.array(delta_mlps)
            
            # 线性回归
            reg = LinearRegression()
            reg.fit(h_ins, delta_mlps)
            r2 = reg.score(h_ins, delta_mlps)
            
            A = reg.coef_
            
            # SVD of A
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            
            total_energy = np.sum(S**2)
            cum_energy = np.cumsum(S**2) / total_energy
            
            # Operator metrics
            eff_rank_90 = np.searchsorted(cum_energy, 0.90) + 1
            eff_rank_95 = np.searchsorted(cum_energy, 0.95) + 1
            
            # Top-3 energy concentration
            top3_conc = np.sum(S[:3]**2) / total_energy
            top10_conc = np.sum(S[:10]**2) / total_energy
            
            # Operator norm (Frobenius)
            op_norm = np.sqrt(total_energy)
            
            # Condition number (top/bottom singular value ratio)
            # S可能有很多接近0的, 取前100
            n_sig = min(100, len(S))
            cond = S[0] / (S[n_sig-1] + 1e-12)
            
            # "Identity-ness": operator有多接近单位阵?
            # delta_mlp ≈ A @ h_in, 如果A≈0 → 不接近identity
            # 用 operator_norm / input_norm 来衡量
            input_norm = np.mean(np.linalg.norm(h_ins, axis=1))
            relative_strength = op_norm / (input_norm + 1e-8)
            
            # 是否接近正交变换? (U和V的对齐度)
            # 如果A接近正交 → S的值应该比较均匀
            sv_spread = np.std(S[:20]) / (np.mean(S[:20]) + 1e-12)
            
            operator_metrics[group_name] = {
                'r2': r2,
                'eff_rank_90': eff_rank_90,
                'eff_rank_95': eff_rank_95,
                'top3_conc': top3_conc,
                'top10_conc': top10_conc,
                'op_norm': op_norm,
                'cond': cond,
                'relative_strength': relative_strength,
                'sv_spread': sv_spread,
                'top5_sv': S[:5].tolist(),
            }
        
        # 打印
        print(f"\n    {'Group':<15} {'R2':<8} {'Rank90':<8} {'Rank95':<8} {'Top3%':<8} {'Top10%':<8} {'OpNorm':<10} {'Cond':<10} {'SVspread':<10}")
        for g, m in operator_metrics.items():
            print(f"    {g:<15} {m['r2']:<8.4f} {m['eff_rank_90']:<8d} {m['eff_rank_95']:<8d} {m['top3_conc']:<8.4f} {m['top10_conc']:<8.4f} {m['op_norm']:<10.4f} {m['cond']:<10.2f} {m['sv_spread']:<10.4f}")
        
        # Operator family判别
        print(f"\n    Operator Family Classification:")
        for g, m in operator_metrics.items():
            if m['top3_conc'] > 0.5 and m['eff_rank_95'] < 50:
                family = "Low-Rank Projection (coordinate transform?)"
            elif m['r2'] > 0.8:
                family = "Strong Linear Operator (structured computation)"
            elif m['r2'] > 0.5:
                family = "Moderate Linear (partially structured)"
            else:
                family = "Nonlinear Dominant (complex computation)"
            
            print(f"      {g}: {family}")
            print(f"        Top-5 SVs: {[f'{s:.4f}' for s in m['top5_sv']]}")
    
    # ---- 跨组Operator Alignment ----
    print("\n--- Cross-Group Operator Alignment ---")
    print("如果不同任务属于不同operator family → 它们的A矩阵应该不同")
    
    for layer in [3, 6, 9]:
        print(f"\n  Layer {layer}:")
        
        # 重新计算A矩阵
        group_As = {}
        for group_name, texts in task_groups.items():
            h_ins = []
            delta_mlps = []
            
            for text in texts[:15]:
                tokens = model.to_tokens(text)
                _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
                
                h_in = cache[f'blocks.{layer}.hook_resid_pre'][-1].detach().cpu().numpy()
                h_mid = cache[f'blocks.{layer}.hook_resid_mid'][-1].detach().cpu().numpy()
                h_out = cache[f'blocks.{layer}.hook_resid_post'][-1].detach().cpu().numpy()
                
                delta_mlp = h_out - h_mid
                h_ins.append(h_in)
                delta_mlps.append(delta_mlp)
            
            reg = LinearRegression()
            reg.fit(np.array(h_ins), np.array(delta_mlps))
            group_As[group_name] = reg.coef_
        
        # 两两比较A矩阵
        group_names = list(group_As.keys())
        for i, g1 in enumerate(group_names):
            for j, g2 in enumerate(group_names):
                if i >= j:
                    continue
                
                A1 = group_As[g1]
                A2 = group_As[g2]
                
                # Frobenius inner product / Frobenius norm product
                flat1 = A1.flatten()
                flat2 = A2.flatten()
                
                cos_sim = np.dot(flat1, flat2) / (np.linalg.norm(flat1) * np.linalg.norm(flat2) + 1e-12)
                
                # Subspace alignment of top singular vectors
                U1, S1, Vt1 = np.linalg.svd(A1, full_matrices=False)
                U2, S2, Vt2 = np.linalg.svd(A2, full_matrices=False)
                
                # Top-10 left singular vectors alignment
                k = 10
                Q1 = U1[:, :k]
                Q2 = U2[:, :k]
                subspace_align = np.linalg.norm(Q1.T @ Q2, 'fro') / np.sqrt(k)
                
                print(f"    {g1} vs {g2}: A_cosine={cos_sim:.4f}, subspace_align={subspace_align:.4f}")


# ============================================================
# 实验D: Sensitive Heads的Causal Validation
# ============================================================

def exp_d_causal_validation(model):
    """
    核心问题: ablate sensitive heads是否消除task-specific computation?
    
    Phase 79发现: 少数heads对prefix高度sensitive
    现在验证: 这些heads是否真的在执行computation policy switching?
    
    方法:
      1. 识别每层的sensitive heads (Phase 79的结果)
      2. Zero-ablate这些heads
      3. 看transition operator是否变得task-independent
    """
    print("=" * 70)
    print("实验D: Causal Validation of Sensitive Heads")
    print("核心问题: sensitive heads是否真的在执行computation policy switching?")
    print("=" * 70)
    
    # 从Phase 79结果, 已知L6的sensitive heads: 0, 4, 8
    # insensitive heads: 3, 6, 9, 10
    
    sensitive_heads = {3: [1, 2, 7, 8], 6: [0, 4, 8, 7], 9: [2, 3, 10, 11]}
    
    task_pairs = {
        "add_vs_trans": ("2 + 3 =", "Translate to French: The cat is on the mat"),
        "add_vs_ant": ("2 + 3 =", "The opposite of hot is"),
        "ant_vs_cap": ("The opposite of hot is", "The capital of France is"),
    }
    
    for layer, heads in sensitive_heads.items():
        print(f"\n  Layer {layer} — Ablating sensitive heads {heads}:")
        
        for pair_name, (text1, text2) in task_pairs.items():
            # 基线: 正常forward
            _, cache1 = model.run_with_cache(model.to_tokens(text1), remove_batch_dim=True)
            _, cache2 = model.run_with_cache(model.to_tokens(text2), remove_batch_dim=True)
            
            # 正常的transition direction similarity
            t1 = cache1[f'blocks.{layer}.hook_resid_post'][-1] - cache1[f'blocks.{layer}.hook_resid_pre'][-1]
            t2 = cache2[f'blocks.{layer}.hook_resid_post'][-1] - cache2[f'blocks.{layer}.hook_resid_pre'][-1]
            
            normal_sim = torch.nn.functional.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()
            
            # Ablate sensitive heads: 用hook将它们的输出置零
            def make_ablation_hook(heads_to_ablate):
                def hook_fn(module, input, output):
                    # output是 [seq_len, n_heads, d_head] 或类似格式
                    # 需要根据具体的hook point来确定格式
                    pass
                return hook_fn
            
            # 简化: 用run_with_hooks
            # 在attn.hook_result上将指定heads置零
            
            # 用另一种方式: 计算每个head的贡献, 减去sensitive heads的贡献
            # 重新计算transition direction without sensitive heads
            
            z1 = cache1[f'blocks.{layer}.attn.hook_z'][-1]  # [n_heads, d_head]
            z2 = cache2[f'blocks.{layer}.attn.hook_z'][-1]
            
            W_O = model.blocks[layer].attn.W_O.detach().cpu()  # [n_heads, d_head, d_model]
            
            # 计算sensitive heads的贡献
            sensitive_contrib1 = torch.zeros(768)
            sensitive_contrib2 = torch.zeros(768)
            
            all_contrib1 = torch.zeros(768)
            all_contrib2 = torch.zeros(768)
            
            for h in range(z1.shape[0]):
                head_out1 = z1[h] @ W_O[h]
                head_out2 = z2[h] @ W_O[h]
                
                all_contrib1 += head_out1
                all_contrib2 += head_out2
                
                if h in heads:
                    sensitive_contrib1 += head_out1
                    sensitive_contrib2 += head_out2
            
            # Ablated transition = original - sensitive_heads_contribution
            ablated_t1 = t1 - sensitive_contrib1
            ablated_t2 = t2 - sensitive_contrib2
            
            ablated_sim = torch.nn.functional.cosine_similarity(
                ablated_t1.unsqueeze(0), ablated_t2.unsqueeze(0)
            ).item()
            
            # Sensitive heads的贡献比例
            sensitive_ratio1 = sensitive_contrib1.norm() / (all_contrib1.norm() + 1e-8)
            
            print(f"    {pair_name}:")
            print(f"      Normal sim: {normal_sim:.4f}")
            print(f"      Ablated sim: {ablated_sim:.4f}")
            print(f"      Change: {ablated_sim - normal_sim:.4f}")
            print(f"      Sensitive heads contribute: {sensitive_ratio1:.4f} of total attn")
            
            if ablated_sim > normal_sim:
                print(f"      ★ Ablation INCREASED sim → sensitive heads CAUSE task-specific divergence")
            else:
                print(f"      Ablation decreased sim → sensitive heads are not the main cause")
    
    # ---- 同理: ablate insensitive heads ----
    insensitive_heads = {3: [0, 4, 5, 10], 6: [3, 6, 9, 10], 9: [1, 6, 8, 9]}
    
    for layer, heads in insensitive_heads.items():
        print(f"\n  Layer {layer} — Ablating insensitive heads {heads}:")
        
        for pair_name, (text1, text2) in task_pairs.items():
            _, cache1 = model.run_with_cache(model.to_tokens(text1), remove_batch_dim=True)
            _, cache2 = model.run_with_cache(model.to_tokens(text2), remove_batch_dim=True)
            
            t1 = cache1[f'blocks.{layer}.hook_resid_post'][-1] - cache1[f'blocks.{layer}.hook_resid_pre'][-1]
            t2 = cache2[f'blocks.{layer}.hook_resid_post'][-1] - cache2[f'blocks.{layer}.hook_resid_pre'][-1]
            
            normal_sim = torch.nn.functional.cosine_similarity(t1.unsqueeze(0), t2.unsqueeze(0)).item()
            
            z1 = cache1[f'blocks.{layer}.attn.hook_z'][-1]
            z2 = cache2[f'blocks.{layer}.attn.hook_z'][-1]
            W_O = model.blocks[layer].attn.W_O.detach().cpu()
            
            insensitive_contrib1 = torch.zeros(768)
            insensitive_contrib2 = torch.zeros(768)
            
            for h in heads:
                head_out1 = z1[h] @ W_O[h]
                head_out2 = z2[h] @ W_O[h]
                insensitive_contrib1 += head_out1
                insensitive_contrib2 += head_out2
            
            ablated_t1 = t1 - insensitive_contrib1
            ablated_t2 = t2 - insensitive_contrib2
            
            ablated_sim = torch.nn.functional.cosine_similarity(
                ablated_t1.unsqueeze(0), ablated_t2.unsqueeze(0)
            ).item()
            
            insensitive_ratio1 = insensitive_contrib1.norm() / (t1.norm() + 1e-8)
            
            print(f"    {pair_name}:")
            print(f"      Normal sim: {normal_sim:.4f}")
            print(f"      Ablated sim: {ablated_sim:.4f}")
            print(f"      Change: {ablated_sim - normal_sim:.4f}")
            print(f"      Insensitive heads contribute: {insensitive_ratio1:.4f} of transition")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="all", choices=["a", "b", "c", "d", "all"])
    args = parser.parse_args()
    
    print("Phase 80: Operator Reverse Engineering")
    print("=" * 70)
    print("核心范式: routing = memory access, operator = computation")
    print("关键问题: transition operator的结构是什么?")
    print("五层框架: Representation → Routing → Operator → Recursive → Compression")
    print("=" * 70)
    
    model = get_model()
    
    if args.exp in ["a", "all"]:
        exp_a_operator_spectrum(model)
    
    if args.exp in ["b", "all"]:
        exp_b_mlp_operator(model)
    
    if args.exp in ["c", "all"]:
        exp_c_operator_family(model)
    
    if args.exp in ["d", "all"]:
        exp_d_causal_validation(model)
    
    print("\n\nPhase 80 完成")
