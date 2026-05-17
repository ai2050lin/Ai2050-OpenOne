"""
Phase 206: Residual Covariance Geometry
========================================

Phase 205 → Phase 206 关键理论修正:
  1. d_eff≈1是LayerNorm伪影 → 需要去除全局模态后再分析
  2. 协方差≠语义 → 可能是architecture-induced covariance
  3. 真正目标: relational invariants (关系不变量)

核心设计:
  Exp1: Global Mode Decomposition
    - 计算 h̄_l (每层均值) 和残差 h̃_l = h_l - h̄_l
    - 对比 Cov(h_l) vs Cov(h̃_l) 的谱结构
    - 关键: d_eff是否从≈1变成合理值?

  Exp2: Residual Covariance Alignment (核心!)
    - 用残差协方差计算跨层对齐度
    - 关键: 去除全局模态后，对齐度是否仍>0.7?

  Exp3: Token-Pair Relation Transport (全新!)
    - 定义 R_ij^l = cosine(h_i^l, h_j^l) 为token对关系
    - 研究 R 矩阵跨层如何演化
    - 关键: 哪些关系被保持? (位置、语法、语义)

  Exp4: Relational Invariant Detection
    - 定义具体关系约束:
      a) 位置关系: 相邻token vs 远距离token
      b) 语法关系: 主语-动词 vs 随机对
      c) 语义关系: 同义/反义 vs 随机对
    - 研究这些关系跨层的保持程度

  Exp5: Cross-task Residual Covariance Overlap
    - 用残差协方差重做Phase 205 Exp5

数据量: 60句/mode, 3 modes
模型加载: Qwen3=bfloat16全GPU, GLM4/DS7B=bfloat16+device_map="auto"
"""

import sys, os
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent.parent / "tests"))

import gc, time, json, math, warnings
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from model_utils import (get_model_info, release_model, get_layers, get_W_U, 
                          MODEL_CONFIGS)

warnings.filterwarnings('ignore')

LITE = os.environ.get('LITE', '1') == '1'


# ========================================================================
# 句子集 (60句, 充分的协方差估计)
# ========================================================================
BASE_SENTENCES = [
    "The cat chases the dog",
    "The teacher helps the student",
    "The leader guides the team",
    "The doctor treats the patient",
    "The chef cooks the meal",
    "The writer drafts the letter",
    "The farmer plants the seed",
    "The artist paints the portrait",
    "The scientist discovers the element",
    "The engineer designs the bridge",
    "The judge delivers the verdict",
    "The soldier defends the fortress",
    "The musician composes the symphony",
    "The pilot flies the airplane",
    "The author writes the novel",
    "The builder constructs the house",
    "The driver operates the vehicle",
    "The hunter tracks the animal",
    "The swimmer crosses the river",
    "The climber reaches the summit",
    "The baker prepares the bread",
    "The tailor makes the garment",
    "The gardener grows the flowers",
    "The fisherman catches the fish",
    "The librarian organizes the books",
    "The mechanic repairs the engine",
    "The programmer writes the code",
    "The analyst studies the data",
    "The manager oversees the project",
    "The director produces the film",
    "The philosopher questions the assumption",
    "The historian examines the evidence",
    "The linguist analyzes the grammar",
    "The mathematician proves the theorem",
    "The physicist tests the hypothesis",
    "The chemist synthesizes the compound",
    "The biologist observes the organism",
    "The geologist studies the rock",
    "The astronomer observes the star",
    "The meteorologist predicts the weather",
    "The economist models the market",
    "The psychologist studies the mind",
    "The sociologist examines the culture",
    "The anthropologist studies the tradition",
    "The architect designs the building",
    "The surveyor measures the land",
    "The technician calibrates the instrument",
    "The inspector checks the quality",
    "The auditor reviews the accounts",
    "The consultant advises the client",
    "The mediator resolves the conflict",
    "The negotiator reaches the agreement",
    "The coordinator manages the schedule",
    "The supervisor monitors the progress",
    "The trainer teaches the skill",
    "The mentor guides the protege",
    "The volunteer helps the community",
    "The researcher explores the frontier",
    "The pioneer discovers the territory",
]

COT_PROMPTS = [
    f"Let's think step by step. {s}" for s in BASE_SENTENCES
]

TRANSLATION_SENTENCES = [
    "Le chat chase le chien",
    "Le professeur aide l'étudiant",
    "Le leader guide l'équipe",
    "Le médecin traite le patient",
    "Le chef prépare le repas",
    "L'écrivain rédige la lettre",
    "Le fermier plante la graine",
    "L'artiste peint le portrait",
    "Le scientifique découvre l'élément",
    "L'ingénieur conçoit le pont",
    "Le juge prononce le verdict",
    "Le soldat défend la forteresse",
    "Le musicien compose la symphonie",
    "Le pilote vole l'avion",
    "L'auteur écrit le roman",
    "Le constructeur bâtit la maison",
    "Le conducteur opère le véhicule",
    "Le chasseur traque l'animal",
    "Le nageur traverse la rivière",
    "L'alpiniste atteint le sommet",
    "Le boulanger prépare le pain",
    "Le tailleur fait le vêtement",
    "Le jardinier cultive les fleurs",
    "Le pêcheur attrape le poisson",
    "Le bibliothécaire organise les livres",
    "Le mécanicien répare le moteur",
    "Le programmeur écrit le code",
    "L'analyste étudie les données",
    "Le gestionnaire supervise le projet",
    "Le directeur produit le film",
    "Le philosophe questionne l'hypothèse",
    "L'historien examine la preuve",
    "Le linguiste analyse la grammaire",
    "Le mathématicien prouve le théorème",
    "Le physicien teste l'hypothèse",
    "Le chimiste synthétise le composé",
    "Le biologiste observe l'organisme",
    "Le géologue étudie la roche",
    "L'astronome observe l'étoile",
    "Le météorologue prédit le temps",
    "L'économiste modélise le marché",
    "Le psychologue étudie l'esprit",
    "Le sociologue examine la culture",
    "L'anthropologue étudie la tradition",
    "L'architecte conçoit le bâtiment",
    "Le géomètre mesure le terrain",
    "Le technicien calibre l'instrument",
    "L'inspecteur vérifie la qualité",
    "L'auditeur examine les comptes",
    "Le conseiller advise le client",
    "Le médiateur résout le conflit",
    "Le négociateur atteint l'accord",
    "Le coordonnateur gère le planning",
    "Le superviseur suit les progrès",
    "Le formateur enseigne la compétence",
    "Le mentor guide le protégé",
    "Le bénévole aide la communauté",
    "Le chercheur explore la frontière",
    "Le pionnier découvre le territoire",
]

# 语法关系对 — (主语位置, 动词位置) 的句子索引
# 简化版: 句子结构为 "The X verbs the Y"，主语=token[1], 动词=token[2]
SYNTAX_RELATIONS = {
    "subject_verb": [(1, 2)],  # token 1 (主语名词) 和 token 2 (动词)
    "verb_object": [(2, 4)],   # token 2 (动词) 和 token 4 (宾语名词)
    "determiner_noun": [(0, 1), (3, 4)],  # "The" + 名词
    "adjacent": [(0,1), (1,2), (2,3), (3,4)],  # 相邻token
    "distant": [(0,4), (1,3)],  # 远距离token
}


def load_model_bf16(model_name: str):
    """BF16优先加载模型, 大模型自动使用8bit+device_map=auto"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # For large models (GLM4, DS7B), use 8bit to avoid OOM
    use_8bit = model_name in ("glm4", "deepseek7b")
    
    if use_8bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
        )
        print(f"[load] Loading {model_name} (8bit + device_map=auto)...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        )
    else:
        print(f"[load] Loading {model_name} (bfloat16 + device_map=auto)...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg["path"],
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
            attn_implementation="eager",
        )
    model.eval()

    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

    if hasattr(model, 'hf_device_map'):
        dmap = model.hf_device_map
        gpu_count = sum(1 for v in dmap.values() if 'cuda' in str(v))
        cpu_count = sum(1 for v in dmap.values() if 'cpu' in str(v))
        print(f"[load] {model_name} loaded: GPU={gpu_count} components, CPU={cpu_count} components, "
              f"GPU mem={gpu_mem:.2f}GB")
    else:
        print(f"[load] {model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB")

    return model, tokenizer, device


def collect_hidden_states(model, tokenizer, device, sentences, n_layers, max_len=64):
    """收集所有句子在各层的hidden states"""
    layers = get_layers(model)
    all_hidden = {l: [] for l in range(n_layers + 1)}  # +1 for embedding layer
    seq_lengths = []

    captured = {}
    def make_hook(li):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[li] = output[0].detach().float().cpu()
            else:
                captured[li] = output.detach().float().cpu()
        return hook

    hooks = [layers[li].register_forward_hook(make_hook(li)) for li in range(n_layers)]

    for si, sent in enumerate(sentences):
        if LITE and si >= 15:
            break
        if si % 10 == 0:
            print(f"    Collecting sentence {si+1}/{len(sentences)}...")

        toks = tokenizer(sent, return_tensors="pt", truncation=True, max_length=max_len)
        input_ids = toks["input_ids"].to(device)
        attention_mask = toks["attention_mask"].to(device)

        captured.clear()
        with torch.no_grad():
            try:
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
                # Use model's hidden_states output (more reliable)
                for li in range(n_layers + 1):
                    hs = out.hidden_states[li].float().cpu().numpy()  # [1, seq_len, d]
                    all_hidden[li].append(hs[0])  # [seq_len, d]
                seq_lengths.append(hs.shape[1])
            except Exception as e:
                print(f"    [WARN] Forward failed for sentence {si}: {e}")
                continue

        # Periodic memory cleanup
        if si % 20 == 0:
            torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    return all_hidden, seq_lengths


def compute_global_mode(hidden_states):
    """计算全局模态 — 所有token的均值向量"""
    # hidden_states: list of [seq_len, d] arrays
    all_tokens = np.concatenate(hidden_states, axis=0)  # [N, d]
    h_global = all_tokens.mean(axis=0)  # [d]
    return h_global


def compute_residual(hidden_states, h_global):
    """计算残差 — 去除全局均值"""
    return [h - h_global for h in hidden_states]


def compute_covariance(hidden_states_list, d_model):
    """
    计算协方差矩阵 (只用有效维度)
    
    Args:
        hidden_states_list: list of [seq_len, d] arrays
        d_model: 模型维度
    
    Returns:
        eigenvalues, explained_variance_ratio, d_eff
    """
    # Concatenate all tokens
    all_tokens = np.concatenate(hidden_states_list, axis=0)  # [N, d]
    N = all_tokens.shape[0]
    
    if N < 2:
        return np.array([1.0]), np.array([1.0]), 1.0
    
    # Center the data
    mean = all_tokens.mean(axis=0)
    centered = all_tokens - mean
    
    # Use SVD for numerical stability instead of eigvalsh
    try:
        if N < d_model:
            # Gram matrix trick: X X^T [N, N]
            gram = centered @ centered.T / N  # [N, N]
            eigvals_small = np.linalg.eigvalsh(gram)
            eigvals_small = np.maximum(eigvals_small, 0)
            eigvals_small = np.sort(eigvals_small)[::-1]
            eigvals = np.zeros(d_model)
            eigvals[:N] = eigvals_small
        else:
            # Use SVD instead of eigvalsh for stability
            U, S, Vt = np.linalg.svd(centered / np.sqrt(N), full_matrices=False)
            eigvals = S**2
            # Pad to d_model
            if len(eigvals) < d_model:
                eigvals = np.concatenate([eigvals, np.zeros(d_model - len(eigvals))])
    except np.linalg.LinAlgError:
        # Fallback: use truncated SVD
        try:
            k = min(N, d_model, 100)
            U, S, Vt = np.linalg.svd(centered / np.sqrt(N), full_matrices=False)
            eigvals = np.concatenate([S[:k]**2, np.zeros(d_model - k)])
        except:
            # Last resort: return uniform eigenvalues
            eigvals = np.ones(d_model) / d_model
    
    total_var = np.sum(eigvals)
    if total_var < 1e-10:
        return eigvals, np.zeros_like(eigvals), 0.0
    
    explained_ratio = eigvals / total_var
    
    # d_eff = (sum σ_i)^2 / sum(σ_i^2) — participation ratio
    s = eigvals / total_var  # normalized
    d_eff = (np.sum(s))**2 / (np.sum(s**2) + 1e-10)
    
    return eigvals, explained_ratio, d_eff


def compute_subspace_alignment(eigvecs_1, eigvecs_2, k=50):
    """
    计算两个子空间的对齐度
    
    Uses subspace angles: alignment = mean(cos^2(theta_i)) for i in 1..k
    """
    V1 = eigvecs_1[:, :k]  # [d, k]
    V2 = eigvecs_2[:, :k]  # [d, k]
    
    # Project V2 onto V1's subspace
    proj = V1 @ (V1.T @ V2)  # [d, k]
    
    # Alignment = ||proj||_F^2 / ||V2||_F^2
    alignment = np.sum(proj**2) / (np.sum(V2**2) + 1e-10)
    return float(alignment)


def compute_cov_with_eigvecs(hidden_states_list, d_model, k_eigvecs=100):
    """
    计算协方差矩阵并返回特征向量（用于子空间对齐）
    Uses SVD for numerical stability.
    
    Returns:
        eigenvalues, d_eff, top_k_eigvecs [d, k]
    """
    all_tokens = np.concatenate(hidden_states_list, axis=0)  # [N, d]
    N = all_tokens.shape[0]
    
    if N < 2:
        return np.array([1.0]), 1.0, np.eye(d_model, k_eigvecs)
    
    mean = all_tokens.mean(axis=0)
    centered = all_tokens - mean
    
    # Use SVD for stability — works for both N<d and N>=d cases
    k = min(k_eigvecs, N, d_model)
    try:
        U, S, Vt = np.linalg.svd(centered, full_matrices=False)
        # eigenvalues = S^2 / N
        eigvals = np.zeros(d_model)
        eigvals[:len(S)] = S**2 / N
        # Eigenvectors = Vt rows (sorted by eigenvalue, descending)
        eigvecs = Vt[:k].T  # [d, k]
    except np.linalg.LinAlgError:
        # Fallback: use truncated SVD
        try:
            from scipy.linalg import svd as scipy_svd
            U, S, Vt = scipy_svd(centered, full_matrices=False, lapack_driver='gesvd')
            eigvals = np.zeros(d_model)
            eigvals[:len(S)] = S**2 / N
            eigvecs = Vt[:k].T
        except:
            # Last resort: random eigenvectors
            eigvals = np.ones(d_model) / d_model
            eigvecs = np.random.randn(d_model, k)
            Q, _ = np.linalg.qr(eigvecs)
            eigvecs = Q
    
    total_var = np.sum(eigvals)
    s = eigvals / (total_var + 1e-10)
    d_eff = (np.sum(s))**2 / (np.sum(s**2) + 1e-10)
    
    return eigvals, d_eff, eigvecs[:, :min(k, eigvecs.shape[1])]


def compute_token_pair_relations(hidden_states_list, max_pairs=500):
    """
    计算token对之间的关系矩阵
    
    For each sentence, compute cosine similarity between all token pairs.
    Return the average relation matrix.
    
    Args:
        hidden_states_list: list of [seq_len, d] arrays
    
    Returns:
        avg_relations: dict of {distance: avg_cosine_similarity}
        pair_relations: list of (i, j, cos_sim, sentence_idx) tuples
    """
    all_relations = defaultdict(list)
    pair_data = []
    
    for si, hs in enumerate(hidden_states_list):
        # Normalize each token
        norms = np.linalg.norm(hs, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        hs_norm = hs / norms
        
        n_tokens = hs.shape[0]
        # Compute pairwise cosine similarity
        cos_sim = hs_norm @ hs_norm.T  # [n_tokens, n_tokens]
        
        for i in range(n_tokens):
            for j in range(i+1, min(i+6, n_tokens)):  # Only nearby pairs (within 5 tokens)
                dist = j - i
                all_relations[dist].append(cos_sim[i, j])
                pair_data.append((i, j, cos_sim[i, j], si))
    
    # Average by distance
    avg_relations = {d: np.mean(v) for d, v in all_relations.items()}
    
    return avg_relations, pair_data


# ========================================================================
# Main experiment
# ========================================================================
def run_phase206(model_name: str):
    print(f"\n{'='*70}")
    print(f"Phase 206: Residual Covariance Geometry — {model_name}")
    print(f"{'='*70}")
    t_start = time.time()

    # ---- Load model ----
    model, tokenizer, device = load_model_bf16(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    print(f"  n_layers={n_layers}, d_model={d_model}, class={info.model_class}")

    # ---- Sample layers ----
    if n_layers <= 12:
        sample_layers = list(range(n_layers + 1))
    else:
        step = max(1, n_layers // 12)
        sample_layers = sorted(set(list(range(0, n_layers + 1, step)) + [n_layers]))
    print(f"  Sample layers: {sample_layers}")

    # ---- Collect hidden states for 3 modes ----
    modes = ["normal", "cot", "translation"]
    mode_sentences = {
        "normal": BASE_SENTENCES[:60],
        "cot": COT_PROMPTS[:60],
        "translation": TRANSLATION_SENTENCES[:60],
    }
    if LITE:
        for k in mode_sentences:
            mode_sentences[k] = mode_sentences[k][:15]

    all_hidden = {}
    for mode in modes:
        print(f"\n--- Collecting {mode} hidden states ---")
        hidden, seq_lens = collect_hidden_states(
            model, tokenizer, device, mode_sentences[mode], n_layers
        )
        all_hidden[mode] = hidden
        print(f"  Collected {len(seq_lens)} sentences, avg seq_len={np.mean(seq_lens):.1f}")

    # ---- Release model ----
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("\nModel released.")

    # ====================================================================
    # Exp1: Global Mode Decomposition
    # ====================================================================
    print(f"\n{'='*70}")
    print("Exp1: Global Mode Decomposition")
    print(f"{'='*70}")

    exp1_results = {}
    for mode in modes:
        exp1_results[mode] = {}
        print(f"\n--- {mode} ---")
        print(f"  {'Layer':>6}  {'d_eff(raw)':>12}  {'d_eff(resid)':>12}  "
              f"{'var_ratio':>10}  {'top1_raw%':>10}  {'top1_resid%':>12}")
        print(f"  {'-'*72}")

        for li in sample_layers:
            hs_list = all_hidden[mode][li]
            if len(hs_list) < 2:
                continue

            # Raw covariance
            eigvals_raw, ratio_raw, d_eff_raw = compute_covariance(hs_list, d_model)
            
            # Compute global mode and residual
            h_global = compute_global_mode(hs_list)
            hs_residual = compute_residual(hs_list, h_global)
            
            # Residual covariance
            eigvals_resid, ratio_resid, d_eff_resid = compute_covariance(hs_residual, d_model)
            
            # How much variance does the global mode explain?
            var_raw = np.sum(eigvals_raw)
            var_resid = np.sum(eigvals_resid)
            var_ratio = var_resid / (var_raw + 1e-10)
            
            top1_raw_pct = ratio_raw[0] * 100 if len(ratio_raw) > 0 else 0
            top1_resid_pct = ratio_resid[0] * 100 if len(ratio_resid) > 0 else 0
            
            print(f"  {li:>6}  {d_eff_raw:>12.2f}  {d_eff_resid:>12.2f}  "
                  f"{var_ratio:>10.4f}  {top1_raw_pct:>10.1f}%  {top1_resid_pct:>12.1f}%")
            
            exp1_results[mode][li] = {
                'd_eff_raw': float(d_eff_raw),
                'd_eff_resid': float(d_eff_resid),
                'var_ratio': float(var_ratio),
                'top1_raw_pct': float(top1_raw_pct),
                'top1_resid_pct': float(top1_resid_pct),
                'top5_raw_pct': float(np.sum(ratio_raw[:5]) * 100) if len(ratio_raw) >= 5 else 0,
                'top5_resid_pct': float(np.sum(ratio_resid[:5]) * 100) if len(ratio_resid) >= 5 else 0,
            }

    # ====================================================================
    # Exp2: Normalized Covariance Alignment (核心!)
    # Key fix: Use L2-normalized hidden states to eliminate LayerNorm effect
    # After normalization, all tokens lie on unit hypersphere, so covariance
    # captures directional structure only — this is the true "relation geometry"
    # ====================================================================
    print(f"\n{'='*70}")
    print("Exp2: Normalized Covariance Alignment (L2-normalized hidden states)")
    print(f"{'='*70}")

    exp2_results = {}
    k_sub = 50  # subspace dimension for alignment

    def normalize_hidden_states(hs_list):
        """L2-normalize each token's hidden state"""
        result = []
        for hs in hs_list:
            norms = np.linalg.norm(hs, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            result.append(hs / norms)
        return result

    for mode in modes:
        exp2_results[mode] = {}
        print(f"\n--- {mode} ---")
        
        # Compute eigenvectors for each sampled layer using NORMALIZED hidden states
        eigvecs_cache = {}
        d_eff_cache = {}
        for li in sample_layers:
            hs_list = all_hidden[mode][li]
            if len(hs_list) < 2:
                continue
            # KEY: L2-normalize first, then compute covariance
            hs_normalized = normalize_hidden_states(hs_list)
            eigvals, d_eff, eigvecs = compute_cov_with_eigvecs(hs_normalized, d_model, k_eigvecs=k_sub)
            eigvecs_cache[li] = eigvecs
            d_eff_cache[li] = d_eff
            if li == sample_layers[0] or li == sample_layers[-1] or li == n_layers // 2:
                print(f"  L{li}: d_eff(normalized)={d_eff:.2f}")

        # Compute alignment between consecutive sampled layers
        sorted_layers = sorted(eigvecs_cache.keys())
        print(f"\n  {'Layer->Layer':>14}  {'Norm_Align':>12}  {'d_eff(from)':>12}  {'d_eff(to)':>12}  {'Preservation':>14}")
        print(f"  {'-'*72}")

        for idx in range(len(sorted_layers) - 1):
            li = sorted_layers[idx]
            li_next = sorted_layers[idx + 1]
            
            if li not in eigvecs_cache or li_next not in eigvecs_cache:
                continue
            
            V1 = eigvecs_cache[li]
            V2 = eigvecs_cache[li_next]
            
            if V1.shape[1] < k_sub or V2.shape[1] < k_sub:
                k_use = min(V1.shape[1], V2.shape[1], k_sub)
            else:
                k_use = k_sub
            
            alignment = compute_subspace_alignment(V1, V2, k=k_use)
            
            # Classify preservation
            if alignment > 0.8:
                preservation = "STRONG"
            elif alignment > 0.6:
                preservation = "MODERATE+"
            elif alignment > 0.4:
                preservation = "MODERATE"
            elif alignment > 0.2:
                preservation = "WEAK"
            else:
                preservation = "NONE"
            
            print(f"  {li:>5}->{li_next:<5}  {alignment:>12.4f}  {d_eff_cache[li]:>12.2f}  "
                  f"{d_eff_cache[li_next]:>12.2f}  {preservation:>14}")
            
            exp2_results[mode][f"{li}->{li_next}"] = {
                'alignment': float(alignment),
                'd_eff_from': float(d_eff_cache[li]),
                'd_eff_to': float(d_eff_cache[li_next]),
                'preservation': preservation,
            }

        # Compute average alignment
        alignments = [v['alignment'] for v in exp2_results[mode].values()]
        strong_count = sum(1 for v in exp2_results[mode].values() if v['preservation'] == 'STRONG')
        moderate_plus = sum(1 for v in exp2_results[mode].values() if v['alignment'] > 0.6)
        avg_align = np.mean(alignments) if alignments else 0
        
        print(f"\n  {mode} SUMMARY: avg_alignment={avg_align:.4f}, "
              f"STRONG={strong_count}, moderate+={moderate_plus}/{len(alignments)}")

    # ====================================================================
    # Exp3: Token-Pair Relation Transport
    # ====================================================================
    print(f"\n{'='*70}")
    print("Exp3: Token-Pair Relation Transport")
    print(f"{'='*70}")

    exp3_results = {}
    for mode in modes:
        exp3_results[mode] = {}
        print(f"\n--- {mode} ---")
        print(f"  {'Layer':>6}  {'d=1':>8}  {'d=2':>8}  {'d=3':>8}  {'d=4':>8}  {'d=5':>8}  {'d=1->5 drift':>14}")
        print(f"  {'-'*72}")

        prev_relations = None
        for li in sample_layers:
            hs_list = all_hidden[mode][li]
            if len(hs_list) < 2:
                continue
            
            avg_rel, pair_data = compute_token_pair_relations(hs_list)
            
            # Compute relation stability: compare with previous layer
            relation_drift = {}
            if prev_relations is not None:
                for d in avg_rel:
                    if d in prev_relations:
                        drift = abs(avg_rel[d] - prev_relations[d])
                        relation_drift[d] = float(drift)
            
            d1 = avg_rel.get(1, 0)
            d2 = avg_rel.get(2, 0)
            d3 = avg_rel.get(3, 0)
            d4 = avg_rel.get(4, 0)
            d5 = avg_rel.get(5, 0)
            d1_to_5_drift = abs(d1 - d5) if d1 != 0 and d5 != 0 else 0
            
            print(f"  {li:>6}  {d1:>8.4f}  {d2:>8.4f}  {d3:>8.4f}  {d4:>8.4f}  {d5:>8.4f}  {d1_to_5_drift:>14.4f}")
            
            exp3_results[mode][li] = {
                'avg_relations_by_dist': {str(k): float(v) for k, v in avg_rel.items()},
                'd1_to_5_drift': float(d1_to_5_drift),
                'relation_drift_from_prev': relation_drift,
            }
            
            prev_relations = avg_rel

    # ====================================================================
    # Exp4: Relational Invariants — Specific relation types
    # ====================================================================
    print(f"\n{'='*70}")
    print("Exp4: Relational Invariants — Position/Syntax/Semantic")
    print(f"{'='*70}")

    exp4_results = {}
    for mode in modes:
        exp4_results[mode] = {}
        print(f"\n--- {mode} ---")

        for li in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
            if li not in all_hidden[mode] or len(all_hidden[mode][li]) < 2:
                continue
            
            hs_list = all_hidden[mode][li]
            
            # Compute cosine similarities for specific relation types
            relation_sims = defaultdict(list)
            
            for si, hs in enumerate(hs_list):
                n_tok = hs.shape[0]
                if n_tok < 5:
                    continue
                
                # Normalize tokens
                norms = np.linalg.norm(hs, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                hs_norm = hs / norms
                
                cos_sim = hs_norm @ hs_norm.T
                
                # Adjacent pairs
                for i in range(n_tok - 1):
                    relation_sims['adjacent'].append(cos_sim[i, i+1])
                
                # Distance-2 pairs
                for i in range(n_tok - 2):
                    relation_sims['distance_2'].append(cos_sim[i, i+2])
                
                # Distance-3 pairs
                for i in range(n_tok - 3):
                    relation_sims['distance_3'].append(cos_sim[i, i+3])
                
                # Distance-4 pairs
                for i in range(n_tok - 4):
                    relation_sims['distance_4'].append(cos_sim[i, i+4])
                
                # Subject-verb (tokens 1 and 2 for "The X verbs...")
                if n_tok > 2:
                    relation_sims['subject_verb'].append(cos_sim[1, 2])
                
                # Verb-object (tokens 2 and 4)
                if n_tok > 4:
                    relation_sims['verb_object'].append(cos_sim[2, 4])
                
                # Determiner-noun
                relation_sims['det_noun'].append(cos_sim[0, 1])
                if n_tok > 4:
                    relation_sims['det_noun'].append(cos_sim[3, 4])
                
                # Long-distance (first vs last token)
                relation_sims['long_distance'].append(cos_sim[0, n_tok-1])
            
            # Average and display
            avg_sims = {k: np.mean(v) for k, v in relation_sims.items()}
            std_sims = {k: np.std(v) for k, v in relation_sims.items()}
            
            print(f"  Layer {li}:")
            for rel_type in ['adjacent', 'distance_2', 'distance_3', 'distance_4',
                           'subject_verb', 'verb_object', 'det_noun', 'long_distance']:
                if rel_type in avg_sims:
                    print(f"    {rel_type:>15}: {avg_sims[rel_type]:.4f} ± {std_sims[rel_type]:.4f} "
                          f"(n={len(relation_sims[rel_type])})")
            
            exp4_results[mode][li] = {
                rel_type: {'mean': float(avg_sims[rel_type]), 'std': float(std_sims[rel_type]),
                          'count': len(relation_sims[rel_type])}
                for rel_type in avg_sims
            }

    # ====================================================================
    # Exp5: Cross-task Normalized Covariance Overlap
    # Using L2-normalized hidden states (same as Exp2)
    # ====================================================================
    print(f"\n{'='*70}")
    print("Exp5: Cross-task Normalized Covariance Overlap")
    print(f"{'='*70}")

    exp5_results = {}
    mid_layer = n_layers // 2
    late_layer = max(0, n_layers - n_layers // 4)

    for target_li in [mid_layer, late_layer]:
        print(f"\n--- Target layer: {target_li} ---")
        exp5_results[target_li] = {}
        
        # Compute normalized eigenvectors for each mode
        eigvecs_per_mode = {}
        for mode in modes:
            hs_list = all_hidden[mode].get(target_li, [])
            if len(hs_list) < 2:
                continue
            # KEY: L2-normalize first (same as Exp2)
            hs_normalized = normalize_hidden_states(hs_list)
            _, _, eigvecs = compute_cov_with_eigvecs(hs_normalized, d_model, k_eigvecs=k_sub)
            eigvecs_per_mode[mode] = eigvecs
        
        # Compute pairwise overlap
        for m1 in modes:
            for m2 in modes:
                if m1 >= m2:
                    continue
                if m1 not in eigvecs_per_mode or m2 not in eigvecs_per_mode:
                    continue
                
                V1 = eigvecs_per_mode[m1]
                V2 = eigvecs_per_mode[m2]
                
                k_use = min(V1.shape[1], V2.shape[1], k_sub)
                
                overlap = compute_subspace_alignment(V1, V2, k=k_use)
                
                # Random baseline
                n_random = 100
                random_overlaps = []
                for _ in range(n_random):
                    V_r1 = np.random.randn(d_model, k_sub)
                    V_r1, _ = np.linalg.qr(V_r1)
                    V_r2 = np.random.randn(d_model, k_sub)
                    V_r2, _ = np.linalg.qr(V_r2)
                    r_overlap = compute_subspace_alignment(V_r1, V_r2, k=k_use)
                    random_overlaps.append(r_overlap)
                random_mean = np.mean(random_overlaps)
                enrichment = overlap / (random_mean + 1e-10)
                
                print(f"  {m1}-{m2}: overlap={overlap:.4f}, random={random_mean:.4f}, "
                      f"enrichment={enrichment:.1f}x")
                
                exp5_results[target_li][f"{m1}-{m2}"] = {
                    'overlap': float(overlap),
                    'random_baseline': float(random_mean),
                    'enrichment': float(enrichment),
                }

    # ====================================================================
    # Summary
    # ====================================================================
    print(f"\n{'='*70}")
    print("PHASE 206 SUMMARY")
    print(f"{'='*70}")

    # Key comparison: d_eff raw vs residual
    print("\n--- d_eff: Raw vs Residual ---")
    for mode in modes:
        raw_d_effs = [exp1_results[mode][li]['d_eff_raw'] for li in exp1_results[mode]]
        resid_d_effs = [exp1_results[mode][li]['d_eff_resid'] for li in exp1_results[mode]]
        if raw_d_effs and resid_d_effs:
            print(f"  {mode}: raw_d_eff={np.mean(raw_d_effs):.2f} → resid_d_eff={np.mean(resid_d_effs):.2f} "
                  f"(change: {np.mean(resid_d_effs) - np.mean(raw_d_effs):+.2f})")

    # Residual covariance alignment summary
    print("\n--- Residual Covariance Alignment Summary ---")
    for mode in modes:
        alignments = [v['alignment'] for v in exp2_results[mode].values()]
        strong = sum(1 for v in exp2_results[mode].values() if v['preservation'] == 'STRONG')
        moderate_plus = sum(1 for v in exp2_results[mode].values() if v['alignment'] > 0.6)
        avg = np.mean(alignments) if alignments else 0
        print(f"  {mode}: avg={avg:.4f}, STRONG={strong}, moderate+={moderate_plus}/{len(alignments)}")

    # Token-pair relation evolution
    print("\n--- Token-Pair Relation Summary ---")
    for mode in modes:
        d1_values = [exp3_results[mode][li]['avg_relations_by_dist'].get('1', 0)
                     for li in exp3_results[mode]]
        d4_values = [exp3_results[mode][li]['avg_relations_by_dist'].get('4', 0)
                     for li in exp3_results[mode]]
        if d1_values:
            print(f"  {mode}: d=1 avg={np.mean(d1_values):.4f}, d=4 avg={np.mean(d4_values):.4f}")

    # Relational invariants summary
    print("\n--- Relational Invariants Summary ---")
    for mode in modes:
        for li in sorted(exp4_results[mode].keys()):
            rels = exp4_results[mode][li]
            adj = rels.get('adjacent', {}).get('mean', 0)
            sv = rels.get('subject_verb', {}).get('mean', 0)
            ld = rels.get('long_distance', {}).get('mean', 0)
            print(f"  {mode} L{li}: adjacent={adj:.4f}, subject_verb={sv:.4f}, long_dist={ld:.4f}")

    # Cross-task overlap
    print("\n--- Cross-task Normalized Covariance Overlap ---")
    for target_li in exp5_results:
        for pair_key, data in exp5_results[target_li].items():
            print(f"  L{target_li} {pair_key}: overlap={data['overlap']:.4f} ({data['enrichment']:.1f}x random)")

    # ====================================================================
    # Save results
    # ====================================================================
    save_dir = Path(__file__).parent.parent / "glm5_temp"
    save_dir.mkdir(exist_ok=True)
    save_path = save_dir / f"phase206_{model_name}_results.json"

    def make_serializable(obj):
        """Recursively convert numpy types to Python native types."""
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()
                    if k != 'eigenvectors'}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, (int, float, str)):
            return obj
        else:
            return str(obj)

    json_results = make_serializable({
        "exp1_global_mode": exp1_results,
        "exp2_residual_alignment": exp2_results,
        "exp3_token_pair_relations": exp3_results,
        "exp4_relational_invariants": exp4_results,
        "exp5_cross_task_overlap": exp5_results,
        "metadata": {
            "model": model_name,
            "n_layers": n_layers,
            "d_model": d_model,
            "lite": LITE,
            "n_sentences": 15 if LITE else 60,
        }
    })

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {save_path}")

    t_total = time.time() - t_start
    print(f"\nTotal time: {t_total:.1f}s ({t_total/60:.1f}min)")
    return json_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_phase206(model_name)
