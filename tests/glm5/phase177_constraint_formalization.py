"""
Phase 177: ★★★ 约束形式化 — 从语义标签到数学约束 ★★★
========================================================

用户核心洞察 (完全正确):
  Phase 176 的 "约束" (fruit/edible/round) 仍然是人类语言标签，不是数学约束。
  真正的约束应该是：可满足条件 (satisfiable condition)

  关键区分:
  | 类型       | 本质       | 例子                           |
  |-----------|-----------|-------------------------------|
  | constraint | 可满足条件 | number(subj) = number(verb)   |
  | concept    | 稳定吸引子 | apple = ∩{edible,plant,round} |
  | feature    | 局部自由度 | color=red                     |
  | token      | 语言接口   | "apple"                       |

★★★ 约束形式化的三个标准 ★★★
1. 可验证 (verifiable): 可以检查约束是否被满足
2. 可满足 (satisfiable): 可以构造满足/违反约束的输入
3. 可传播 (propagatable): 可以追踪约束如何在层间传递

★★★ 四大实验 ★★★

Phase A: ★★★ 形式语法约束验证 (Functional Invariants) ★★★
  - 主谓一致: number(subject) = number(verb)
  - 指代一致: gender(pronoun) = gender(antecedent)
  - 时态一致: tense(clause_1) = tense(clause_2)
  - ★★★ 这是功能不变量: 无论模型内部如何实现, 约束必须被满足 ★★★
  - 大数据量: 40+ 主谓一致, 30+ 指代一致, 20+ 时态一致

Phase B: ★★★ 约束依赖图 (Topological Invariants) ★★★
  - number约束是否影响verb morphology约束?
  - gender约束是否独立于number约束?
  - 方法: Hook因果干预 — ablate一个约束子空间, 测量其他约束变化
  - ★★★ 约束依赖图 G_l 是拓扑不变量 ★★★

Phase C: ★★★ 约束能量景观 (Dynamical Invariants) ★★★
  - 定义: E(l) = -Σ_C σ_C(h_l) (负的约束满足度之和)
  - 追踪: E(L0) → E(L_last)
  - 假设: E递减 → 系统收敛到约束满足态
  - ★★★ 能量递减是动力学不变量 ★★★

Phase D: ★★★ 概念 = 约束交集 (Concept = Constraint Intersection) ★★★
  - 形式定义: apple = {edible=true, plant=true, round=true, sweet=true}
  - 必要约束删除 → 概念崩塌?
  - 非必要约束删除 → 概念存活?
  - ★★★ 区分必要约束 vs 旁观特征 ★★★

Usage: python tests/glm5/phase177_constraint_formalization.py <model_name>
  model_name: qwen3, glm4, deepseek7b
"""

import sys
import os
import time
import json
import gc
import numpy as np
import torch
from datetime import datetime
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8', line_buffering=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'glm5'))

from model_utils import get_layers, get_model_info, release_model, get_W_U, MODEL_CONFIGS


# =====================================================================
# MODEL LOADING (BF16 + device_map="auto") — 参考 model_demo_bf16.py
# =====================================================================

def load_model_bf16(model_name):
    """BF16 + device_map=auto loading for all models"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[bf16] Loading {model_name} (bfloat16 + device_map=auto)...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=True, local_files_only=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    print(f"[bf16] {model_name} loaded: GPU={gpu_mem:.2f}GB", flush=True)

    return model, tokenizer, device


# =====================================================================
# ★★★ 形式约束定义 — 核心理论升级 ★★★
# =====================================================================

# Phase A: 形式语法约束句子对
# 每个约束由 (满足句子, 违反句子) 对定义
# 约束 = 可满足条件, 不是语义标签!

# ---------- 主谓一致: number(subject) = number(verb) ----------
NUMBER_AGREEMENT = {
    # ★★★ 单数主语 + 正确单数动词 vs 错误复数动词 ★★★
    "singular_correct": [
        "The cat sleeps on the mat",
        "The dog runs to the park",
        "The bird flies over the tree",
        "The child reads a book",
        "The man walks to work",
        "The woman sings a song",
        "The fish swims in the pond",
        "The horse gallops across",
        "The student writes an essay",
        "The teacher explains the lesson",
        "The flower grows in spring",
        "The river flows downhill",
        "The cloud drifts slowly",
        "The lamp shines brightly",
        "The clock ticks quietly",
        "The bell rings loudly",
        "The door opens slowly",
        "The wind blows gently",
        "The rain falls softly",
        "The sun rises early",
    ],
    "singular_wrong": [
        "The cat sleep on the mat",
        "The dog run to the park",
        "The bird fly over the tree",
        "The child read a book",
        "The man walk to work",
        "The woman sing a song",
        "The fish swim in the pond",
        "The horse gallop across",
        "The student write an essay",
        "The teacher explain the lesson",
        "The flower grow in spring",
        "The river flow downhill",
        "The cloud drift slowly",
        "The lamp shine brightly",
        "The clock tick quietly",
        "The bell ring loudly",
        "The door open slowly",
        "The wind blow gently",
        "The rain fall softly",
        "The sun rise early",
    ],
    # ★★★ 复数主语 + 正确复数动词 vs 错误单数动词 ★★★
    "plural_correct": [
        "The cats sleep on the mat",
        "The dogs run to the park",
        "The birds fly over the tree",
        "The children read a book",
        "The men walk to work",
        "The women sing a song",
        "The fish swim in the pond",
        "The horses gallop across",
        "The students write an essay",
        "The teachers explain the lesson",
        "The flowers grow in spring",
        "The rivers flow downhill",
        "The clouds drift slowly",
        "The lamps shine brightly",
        "The clocks tick quietly",
        "The bells ring loudly",
        "The doors open slowly",
        "The winds blow gently",
        "The rains fall softly",
        "The suns rise early",
    ],
    "plural_wrong": [
        "The cats sleeps on the mat",
        "The dogs runs to the park",
        "The birds flies over the tree",
        "The children reads a book",
        "The men walks to work",
        "The women sings a song",
        "The fish swims in the pond",
        "The horses gallops across",
        "The students writes an essay",
        "The teachers explains the lesson",
        "The flowers grows in spring",
        "The rivers flows downhill",
        "The clouds drifts slowly",
        "The lamps shines brightly",
        "The clocks ticks quietly",
        "The bells rings loudly",
        "The doors opens slowly",
        "The winds blows gently",
        "The rains falls softly",
        "The suns rises early",
    ],
    # ★★★ 复杂结构: 介词短语插入 (attraction error) ★★★
    "complex_singular_correct": [
        "The cat near the dogs sleeps on the",
        "The dog behind the trees runs to the",
        "The bird above the cats flies over the",
        "The student with the friends reads a",
        "The woman among the children sings a",
        "The horse beside the cows gallops",
        "The flower near the trees grows in",
        "The lamp between the clocks shines",
    ],
    "complex_singular_wrong": [
        "The cat near the dogs sleep on the",
        "The dog behind the trees run to the",
        "The bird above the cats fly over the",
        "The student with the friends read a",
        "The woman among the children sing a",
        "The horse beside the cows gallop",
        "The flower near the trees grow in",
        "The lamp between the clocks shine",
    ],
}

# ---------- 指代一致: gender(pronoun) = gender(antecedent) ----------
GENDER_AGREEMENT = {
    "masculine_correct": [
        "The actor said he was tired",
        "The king ruled that he would",
        "The boy knew that he should",
        "The man believed that he could",
        "The father told him that he",
        "The brother realized that he",
        "The uncle mentioned that he",
        "The hero declared that he",
        "The prince announced that he",
        "The gentleman insisted that he",
        "The monk explained that he",
        "The knight promised that he",
        "The wizard claimed that he",
        "The emperor commanded that he",
        "The soldier reported that he",
    ],
    "masculine_wrong": [
        "The actor said she was tired",
        "The king ruled that she would",
        "The boy knew that she should",
        "The man believed that she could",
        "The father told her that he",
        "The brother realized that she",
        "The uncle mentioned that she",
        "The hero declared that she",
        "The prince announced that she",
        "The gentleman insisted that she",
        "The monk explained that she",
        "The knight promised that she",
        "The wizard claimed that she",
        "The emperor commanded that she",
        "The soldier reported that she",
    ],
    "feminine_correct": [
        "The actress said she was tired",
        "The queen ruled that she would",
        "The girl knew that she should",
        "The woman believed that she could",
        "The mother told her that she",
        "The sister realized that she",
        "The aunt mentioned that she",
        "The heroine declared that she",
        "The princess announced that she",
        "The lady insisted that she",
        "The nun explained that she",
        "The witch claimed that she",
        "The empress commanded that she",
        "The nurse reported that she",
        "The goddess promised that she",
    ],
    "feminine_wrong": [
        "The actress said he was tired",
        "The queen ruled that he would",
        "The girl knew that he should",
        "The woman believed that he could",
        "The mother told him that she",
        "The sister realized that he",
        "The aunt mentioned that he",
        "The heroine declared that he",
        "The princess announced that he",
        "The lady insisted that he",
        "The nun explained that he",
        "The witch claimed that he",
        "The empress commanded that he",
        "The nurse reported that he",
        "The goddess promised that he",
    ],
}

# ---------- 时态一致: tense(clause_1) = tense(clause_2) ----------
TENSE_AGREEMENT = {
    "past_correct": [
        "She walked and talked about",
        "He ran and jumped over",
        "They cooked and ate the",
        "We studied and passed the",
        "I wrote and sent the",
        "The dog barked and ran",
        "The cat slept and dreamed",
        "The boy laughed and played",
        "The girl cried and hugged",
        "The man worked and earned",
    ],
    "past_wrong": [
        "She walked and talks about",
        "He ran and jumps over",
        "They cooked and eat the",
        "We studied and pass the",
        "I wrote and send the",
        "The dog barked and runs",
        "The cat slept and dreams",
        "The boy laughed and plays",
        "The girl cried and hugs",
        "The man worked and earns",
    ],
    "present_correct": [
        "She walks and talks about",
        "He runs and jumps over",
        "They cook and eat the",
        "We study and pass the",
        "I write and send the",
        "The dog barks and runs",
        "The cat sleeps and dreams",
        "The boy laughs and plays",
        "The girl cries and hugs",
        "The man works and earns",
    ],
    "present_wrong": [
        "She walks and talked about",
        "He runs and jumped over",
        "They cook and ate the",
        "We study and passed the",
        "I write and sent the",
        "The dog barks and ran",
        "The cat sleeps and dreamed",
        "The boy laughs and played",
        "The girl cries and hugged",
        "The man works and earned",
    ],
}

# ---------- 跨语言主谓一致 (FR/ES) ----------
CROSS_LING_NUMBER = {
    "fr_singular_correct": [
        "Le chat dort sur le",
        "Le chien court au parc",
        "L'oiseau vole au-dessus",
        "Le enfant lit un livre",
        "Le homme marche au travail",
        "La femme chante une chanson",
        "Le fleuve coule en bas",
        "La fleur pousse au printemps",
        "Le soleil se lève tôt",
        "La porte s'ouvre lentement",
    ],
    "fr_singular_wrong": [
        "Le chat dorment sur le",
        "Le chien courent au parc",
        "L'oiseau volent au-dessus",
        "Le enfant lisent un livre",
        "Le homme marchent au travail",
        "La femme chantent une chanson",
        "Le fleuve coulent en bas",
        "La fleur poussent au printemps",
        "Le soleil se lèvent tôt",
        "La porte s'ouvrent lentement",
    ],
    "fr_plural_correct": [
        "Les chats dorment sur le",
        "Les chiens courent au parc",
        "Les oiseaux volent au-dessus",
        "Les enfants lisent un livre",
        "Les hommes marchent au travail",
        "Les femmes chantent une chanson",
        "Les fleuves coulent en bas",
        "Les fleurs poussent au printemps",
        "Les soleils se lèvent tôt",
        "Les portes s'ouvrent lentement",
    ],
    "fr_plural_wrong": [
        "Les chats dort sur le",
        "Les chiens court au parc",
        "Les oiseaux vole au-dessus",
        "Les enfants lit un livre",
        "Les hommes marche au travail",
        "Les femmes chante une chanson",
        "Les fleuves coule en bas",
        "Les fleurs pousse au printemps",
        "Les soleils se lève tôt",
        "Les portes s'ouvre lentement",
    ],
    "es_singular_correct": [
        "El gato duerme en la",
        "El perro corre al parque",
        "El niño lee un libro",
        "El hombre camina al trabajo",
        "La mujer canta una canción",
        "El río fluye cuesta abajo",
        "La flor crece en primavera",
        "El sol se levanta temprano",
        "La puerta se abre lentamente",
        "El pájaro vuela sobre el",
    ],
    "es_singular_wrong": [
        "El gato duermen en la",
        "El perro corren al parque",
        "El niño leen un libro",
        "El hombre caminan al trabajo",
        "La mujer cantan una canción",
        "El río fluyen cuesta abajo",
        "La flor crecen en primavera",
        "El sol se levantan temprano",
        "La puerta se abren lentamente",
        "El pájaro vuelan sobre el",
    ],
    "es_plural_correct": [
        "Los gatos duermen en la",
        "Los perros corren al parque",
        "Los niños leen un libro",
        "Los hombres caminan al trabajo",
        "Las mujeres cantan una canción",
        "Los ríos fluyen cuesta abajo",
        "Las flores crecen en primavera",
        "Los soles se levantan temprano",
        "Las puertas se abren lentamente",
        "Los pájaros vuelan sobre el",
    ],
    "es_plural_wrong": [
        "Los gatos duerme en la",
        "Los perros corre al parque",
        "Los niños lee un libro",
        "Los hombres camina al trabajo",
        "Las mujeres canta una canción",
        "Los ríos fluye cuesta abajo",
        "Las flores crece en primavera",
        "Los soles se levanta temprano",
        "Las puertas se abre lentamente",
        "Los pájaros vuela sobre el",
    ],
}


# ---------- Phase D: 概念 = 约束交集 ----------
# 形式定义: concept = {constraint_name: is_necessary}
# 必要约束: 删除 → 概念崩塌
# 非必要约束: 删除 → 概念存活

CONCEPT_FORMAL_CONSTRAINTS = {
    "apple": {
        "necessary": {
            "edible": ["eat", "food", "delicious", "taste", "cook", "meal", "dish"],
            "plant_origin": ["plant", "tree", "grow", "leaf", "root", "seed", "garden"],
        },
        "unnecessary": {
            "round": ["round", "circle", "sphere", "ball", "shape", "curved", "globe"],
            "red": ["red", "crimson", "scarlet", "ruby", "cherry", "maroon"],
            "mechanical": ["engine", "machine", "motor", "metal", "mechanical", "gear"],
        },
        "test_sentences": [
            "The apple is",
            "I ate an apple",
            "An apple tree grows",
            "She picked an apple",
            "The apple was fresh",
            "Fresh apple juice is",
        ],
    },
    "cat": {
        "necessary": {
            "animal": ["animal", "wild", "pet", "creature", "alive", "species", "mammal"],
            "living": ["alive", "life", "grow", "born", "breathe", "live", "organism"],
        },
        "unnecessary": {
            "small": ["small", "tiny", "little", "mini", "petite", "compact"],
            "black": ["black", "dark", "noir", "ebony", "midnight", "shadow"],
            "mechanical": ["engine", "machine", "motor", "metal", "mechanical", "gear"],
        },
        "test_sentences": [
            "The cat is",
            "I saw a cat",
            "A cat sat on the",
            "She fed the cat",
            "The cat was sleeping",
            "My cat is cute",
        ],
    },
    "car": {
        "necessary": {
            "vehicle": ["vehicle", "drive", "engine", "road", "transport", "wheel", "fast"],
            "mechanical": ["engine", "machine", "motor", "metal", "mechanical", "gear", "power"],
        },
        "unnecessary": {
            "edible": ["eat", "food", "delicious", "taste", "cook", "meal", "dish"],
            "living": ["alive", "life", "grow", "born", "breathe", "live", "organism"],
            "red": ["red", "crimson", "scarlet", "ruby", "cherry", "maroon"],
        },
        "test_sentences": [
            "The car is",
            "I drove a car",
            "A car on the road",
            "She bought a car",
            "The car was fast",
            "My car is new",
        ],
    },
}


# =====================================================================
# ★★★ 核心方法: 约束满足函数 (基于logits, 不是hidden state) ★★★
# =====================================================================

def get_word_logit(logits, word, tokenizer):
    """Get logit value for a word from logits vector."""
    tok_ids = tokenizer.encode(word, add_special_tokens=False)
    if not tok_ids:
        return 0.0
    if len(tok_ids) == 1:
        return float(logits[tok_ids[0]])
    return float(np.mean([logits[tid] for tid in tok_ids if tid < len(logits)]))


def compute_number_signal(logits, tokenizer):
    """
    ★★★ Number约束满足信号 ★★★
    定义: σ_N(logits) = mean(logits[singular_words]) - mean(logits[plural_words])
    
    > 0: 表示singular约束被满足
    < 0: 表示plural约束被满足
    ≈ 0: number约束未被编码
    
    这是功能不变量: 模型必须区分singular和plural!
    """
    singular_words = ["is", "was", "runs", "walks", "sleeps", "has", "one", "it"]
    plural_words = ["are", "were", "run", "walk", "sleep", "have", "many", "they"]
    
    s_scores = [get_word_logit(logits, w, tokenizer) for w in singular_words]
    p_scores = [get_word_logit(logits, w, tokenizer) for w in plural_words]
    
    return float(np.mean(s_scores) - np.mean(p_scores))


def compute_gender_signal(logits, tokenizer):
    """
    ★★★ Gender约束满足信号 ★★★
    定义: σ_G(logits) = mean(logits[masculine_words]) - mean(logits[feminine_words])
    
    > 0: 表示masculine约束被满足
    < 0: 表示feminine约束被满足
    """
    masc_words = ["he", "him", "his", "himself", "man", "boy", "father"]
    fem_words = ["she", "her", "hers", "herself", "woman", "girl", "mother"]
    
    m_scores = [get_word_logit(logits, w, tokenizer) for w in masc_words]
    f_scores = [get_word_logit(logits, w, tokenizer) for w in fem_words]
    
    return float(np.mean(m_scores) - np.mean(f_scores))


def compute_tense_signal(logits, tokenizer):
    """
    ★★★ Tense约束满足信号 ★★★
    定义: σ_T(logits) = mean(logits[past_words]) - mean(logits[present_words])
    
    > 0: 表示past约束被满足
    < 0: 表示present约束被满足
    """
    past_words = ["was", "were", "had", "did", "went", "came", "said", "took"]
    present_words = ["is", "are", "has", "does", "goes", "comes", "says", "takes"]
    
    pa_scores = [get_word_logit(logits, w, tokenizer) for w in past_words]
    pr_scores = [get_word_logit(logits, w, tokenizer) for w in present_words]
    
    return float(np.mean(pa_scores) - np.mean(pr_scores))


def compute_constraint_signal(logits, tokenizer, constraint_words):
    """Generic constraint signal: mean logit of constraint words."""
    scores = [get_word_logit(logits, w, tokenizer) for w in constraint_words]
    return float(np.mean(scores)) if scores else 0.0


# =====================================================================
# Phase A: ★★★ 形式语法约束验证 (Functional Invariants) ★★★
# =====================================================================

def run_formal_constraint_verification(model, tokenizer, device, model_info, W_U):
    """
    ★★★ Phase A: 形式语法约束是否在模型内部被编码和执行? ★★★
    
    方法:
    1. 对每对 (满足, 违反) 句子, 在每一层计算约束满足信号
    2. 比较满足 vs 违反: 模型是否区分?
    3. 追踪: 约束信号在哪一层开始显著?
    
    ★★★ 这是功能不变量: 模型必须区分满足和违反! ★★★
    """
    n_layers = model_info.n_layers
    sample_layers = list(range(0, n_layers + 1, max(1, n_layers // 12)))
    if n_layers not in sample_layers:
        sample_layers.append(n_layers)
    sample_layers = sorted(set(sample_layers))
    
    print("\n" + "="*70, flush=True)
    print("Phase A: ★★★ 形式语法约束验证 (Functional Invariants) ★★★", flush=True)
    print("="*70, flush=True)
    
    results = {}
    
    # ---- A1: Number Agreement ----
    print("\n  A1: Number Agreement — number(subject) = number(verb)", flush=True)
    
    number_results = {}
    
    for pair_type in ["singular", "plural", "complex_singular"]:
        correct_key = f"{pair_type}_correct"
        wrong_key = f"{pair_type}_wrong"
        
        if correct_key not in NUMBER_AGREEMENT or wrong_key not in NUMBER_AGREEMENT:
            continue
        
        correct_sents = NUMBER_AGREEMENT[correct_key]
        wrong_sents = NUMBER_AGREEMENT[wrong_key]
        n_pairs = min(len(correct_sents), len(wrong_sents))
        
        # For each layer, collect number signals
        correct_signals = {li: [] for li in sample_layers}
        wrong_signals = {li: [] for li in sample_layers}
        
        for i in range(n_pairs):
            for label, sents, signal_dict in [
                ("correct", correct_sents, correct_signals),
                ("wrong", wrong_sents, wrong_signals),
            ]:
                sent = sents[i]
                input_device = next(model.parameters()).device
                inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
                
                with torch.no_grad():
                    out = model(
                        input_ids=inputs["input_ids"].to(input_device),
                        attention_mask=inputs["attention_mask"].to(input_device),
                        output_hidden_states=True,
                    )
                
                hs = out.hidden_states
                for li in sample_layers:
                    if li < len(hs):
                        # Use the VERB position (typically position 2-3 for "The cat sleeps")
                        # Use last token for prediction context
                        h_verb = hs[li][0, -1].float().cpu().numpy()
                        logits_l = W_U @ h_verb
                        signal = compute_number_signal(logits_l, tokenizer)
                        signal_dict[li].append(signal)
        
        # Compute average signals per layer
        avg_correct = {li: float(np.mean(correct_signals[li])) for li in sample_layers if correct_signals[li]}
        avg_wrong = {li: float(np.mean(wrong_signals[li])) for li in sample_layers if wrong_signals[li]}
        
        # ★★★ Constraint violation detection signal ★★★
        # = correct_signal - wrong_signal
        # > 0: model distinguishes correct from wrong (constraint is enforced)
        violation_signal = {}
        for li in sample_layers:
            if li in avg_correct and li in avg_wrong:
                violation_signal[li] = round(avg_correct[li] - avg_wrong[li], 4)
        
        number_results[pair_type] = {
            "n_pairs": n_pairs,
            "avg_correct_signal": {str(li): round(v, 4) for li, v in avg_correct.items()},
            "avg_wrong_signal": {str(li): round(v, 4) for li, v in avg_wrong.items()},
            "violation_signal": {str(li): v for li, v in violation_signal.items()},
        }
        
        # Find the "constraint enforcement layer": first layer where violation_signal > threshold
        threshold = 0.1
        enforcement_layer = None
        for li in sorted(violation_signal.keys()):
            if abs(violation_signal[li]) > threshold:
                enforcement_layer = li
                break
        
        # Print key results
        for li in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]:
            li_str = str(li)
            if li_str in number_results[pair_type]["violation_signal"]:
                vs = number_results[pair_type]["violation_signal"][li_str]
                cs = number_results[pair_type]["avg_correct_signal"].get(li_str, 0)
                ws = number_results[pair_type]["avg_wrong_signal"].get(li_str, 0)
                print(f"    {pair_type} L{li}: correct_σN={cs:.4f}, wrong_σN={ws:.4f}, "
                      f"violation={vs:.4f}", flush=True)
        
        if enforcement_layer is not None:
            print(f"    → Constraint enforcement starts at L{enforcement_layer}", flush=True)
    
    results["number_agreement"] = number_results
    
    # ---- A2: Gender Agreement ----
    print("\n  A2: Gender Agreement — gender(pronoun) = gender(antecedent)", flush=True)
    
    gender_results = {}
    
    for pair_type in ["masculine", "feminine"]:
        correct_key = f"{pair_type}_correct"
        wrong_key = f"{pair_type}_wrong"
        
        if correct_key not in GENDER_AGREEMENT or wrong_key not in GENDER_AGREEMENT:
            continue
        
        correct_sents = GENDER_AGREEMENT[correct_key]
        wrong_sents = GENDER_AGREEMENT[wrong_key]
        n_pairs = min(len(correct_sents), len(wrong_sents))
        
        correct_signals = {li: [] for li in sample_layers}
        wrong_signals = {li: [] for li in sample_layers}
        
        for i in range(n_pairs):
            for label, sents, signal_dict in [
                ("correct", correct_sents, correct_signals),
                ("wrong", wrong_sents, wrong_signals),
            ]:
                sent = sents[i]
                input_device = next(model.parameters()).device
                inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
                
                with torch.no_grad():
                    out = model(
                        input_ids=inputs["input_ids"].to(input_device),
                        attention_mask=inputs["attention_mask"].to(input_device),
                        output_hidden_states=True,
                    )
                
                hs = out.hidden_states
                for li in sample_layers:
                    if li < len(hs):
                        h = hs[li][0, -1].float().cpu().numpy()
                        logits_l = W_U @ h
                        signal = compute_gender_signal(logits_l, tokenizer)
                        signal_dict[li].append(signal)
        
        avg_correct = {li: float(np.mean(correct_signals[li])) for li in sample_layers if correct_signals[li]}
        avg_wrong = {li: float(np.mean(wrong_signals[li])) for li in sample_layers if wrong_signals[li]}
        
        violation_signal = {}
        for li in sample_layers:
            if li in avg_correct and li in avg_wrong:
                violation_signal[li] = round(avg_correct[li] - avg_wrong[li], 4)
        
        gender_results[pair_type] = {
            "n_pairs": n_pairs,
            "avg_correct_signal": {str(li): round(v, 4) for li, v in avg_correct.items()},
            "avg_wrong_signal": {str(li): round(v, 4) for li, v in avg_wrong.items()},
            "violation_signal": {str(li): v for li, v in violation_signal.items()},
        }
        
        for li in [0, n_layers // 2, n_layers]:
            li_str = str(li)
            if li_str in gender_results[pair_type]["violation_signal"]:
                vs = gender_results[pair_type]["violation_signal"][li_str]
                print(f"    {pair_type} L{li}: violation_signal={vs:.4f}", flush=True)
    
    results["gender_agreement"] = gender_results
    
    # ---- A3: Tense Consistency ----
    print("\n  A3: Tense Consistency — tense(clause_1) = tense(clause_2)", flush=True)
    
    tense_results = {}
    
    for pair_type in ["past", "present"]:
        correct_key = f"{pair_type}_correct"
        wrong_key = f"{pair_type}_wrong"
        
        if correct_key not in TENSE_AGREEMENT or wrong_key not in TENSE_AGREEMENT:
            continue
        
        correct_sents = TENSE_AGREEMENT[correct_key]
        wrong_sents = TENSE_AGREEMENT[wrong_key]
        n_pairs = min(len(correct_sents), len(wrong_sents))
        
        correct_signals = {li: [] for li in sample_layers}
        wrong_signals = {li: [] for li in sample_layers}
        
        for i in range(n_pairs):
            for label, sents, signal_dict in [
                ("correct", correct_sents, correct_signals),
                ("wrong", wrong_sents, wrong_signals),
            ]:
                sent = sents[i]
                input_device = next(model.parameters()).device
                inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
                
                with torch.no_grad():
                    out = model(
                        input_ids=inputs["input_ids"].to(input_device),
                        attention_mask=inputs["attention_mask"].to(input_device),
                        output_hidden_states=True,
                    )
                
                hs = out.hidden_states
                for li in sample_layers:
                    if li < len(hs):
                        h = hs[li][0, -1].float().cpu().numpy()
                        logits_l = W_U @ h
                        signal = compute_tense_signal(logits_l, tokenizer)
                        signal_dict[li].append(signal)
        
        avg_correct = {li: float(np.mean(correct_signals[li])) for li in sample_layers if correct_signals[li]}
        avg_wrong = {li: float(np.mean(wrong_signals[li])) for li in sample_layers if wrong_signals[li]}
        
        violation_signal = {}
        for li in sample_layers:
            if li in avg_correct and li in avg_wrong:
                violation_signal[li] = round(avg_correct[li] - avg_wrong[li], 4)
        
        tense_results[pair_type] = {
            "n_pairs": n_pairs,
            "violation_signal": {str(li): v for li, v in violation_signal.items()},
        }
        
        for li in [0, n_layers // 2, n_layers]:
            li_str = str(li)
            if li_str in tense_results[pair_type]["violation_signal"]:
                vs = tense_results[pair_type]["violation_signal"][li_str]
                print(f"    {pair_type} L{li}: violation_signal={vs:.4f}", flush=True)
    
    results["tense_consistency"] = tense_results
    
    # ---- A4: Cross-linguistic Number Agreement (FR/ES) ----
    print("\n  A4: Cross-linguistic Number Agreement (FR/ES)", flush=True)
    
    cross_ling_results = {}
    
    for lang_prefix in ["fr", "es"]:
        for number in ["singular", "plural"]:
            correct_key = f"{lang_prefix}_{number}_correct"
            wrong_key = f"{lang_prefix}_{number}_wrong"
            
            if correct_key not in CROSS_LING_NUMBER or wrong_key not in CROSS_LING_NUMBER:
                continue
            
            correct_sents = CROSS_LING_NUMBER[correct_key]
            wrong_sents = CROSS_LING_NUMBER[wrong_key]
            n_pairs = min(len(correct_sents), len(wrong_sents))
            
            correct_signals = {li: [] for li in sample_layers}
            wrong_signals = {li: [] for li in sample_layers}
            
            for i in range(n_pairs):
                for label, sents, signal_dict in [
                    ("correct", correct_sents, correct_signals),
                    ("wrong", wrong_sents, wrong_signals),
                ]:
                    sent = sents[i]
                    try:
                        input_device = next(model.parameters()).device
                        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
                        
                        with torch.no_grad():
                            out = model(
                                input_ids=inputs["input_ids"].to(input_device),
                                attention_mask=inputs["attention_mask"].to(input_device),
                                output_hidden_states=True,
                            )
                        
                        hs = out.hidden_states
                        for li in sample_layers:
                            if li < len(hs):
                                h = hs[li][0, -1].float().cpu().numpy()
                                logits_l = W_U @ h
                                signal = compute_number_signal(logits_l, tokenizer)
                                signal_dict[li].append(signal)
                    except Exception as e:
                        continue
            
            avg_correct = {li: float(np.mean(correct_signals[li])) for li in sample_layers if correct_signals[li]}
            avg_wrong = {li: float(np.mean(wrong_signals[li])) for li in sample_layers if wrong_signals[li]}
            
            violation_signal = {}
            for li in sample_layers:
                if li in avg_correct and li in avg_wrong:
                    violation_signal[li] = round(avg_correct[li] - avg_wrong[li], 4)
            
            key = f"{lang_prefix}_{number}"
            cross_ling_results[key] = {
                "n_pairs": n_pairs,
                "violation_signal": {str(li): v for li, v in violation_signal.items()},
            }
            
            for li in [0, n_layers // 2, n_layers]:
                li_str = str(li)
                if li_str in cross_ling_results[key]["violation_signal"]:
                    vs = cross_ling_results[key]["violation_signal"][li_str]
                    print(f"    {key} L{li}: violation={vs:.4f}", flush=True)
    
    results["cross_linguistic_number"] = cross_ling_results
    
    return results


# =====================================================================
# Phase B: ★★★ 约束依赖图 (Topological Invariants) ★★★
# =====================================================================

def run_constraint_dependency_graph(model, tokenizer, device, model_info, W_U):
    """
    ★★★ Phase B: 约束之间的依赖关系是什么? ★★★
    
    方法:
    1. 定义约束方向: number_dir, gender_dir, tense_dir (在W_U空间中)
    2. Hook因果干预: 在某层ablate一个约束方向
    3. 测量: 其他约束信号是否受影响?
    4. 如果ablate number → gender不受影响 → 约束独立
    5. 如果ablate number → verb_morphology受影响 → 约束依赖
    
    ★★★ 约束依赖图 G_l 是拓扑不变量! ★★★
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    
    print("\n" + "="*70, flush=True)
    print("Phase B: ★★★ 约束依赖图 (Topological Invariants) ★★★", flush=True)
    print("="*70, flush=True)
    
    # Step 1: Compute constraint directions in W_U space
    print("  Step 1: Computing constraint directions...", flush=True)
    
    # Number direction: singular_words - plural_words in W_U space
    singular_words = ["is", "was", "runs", "walks", "sleeps", "has", "one"]
    plural_words = ["are", "were", "run", "walk", "sleep", "have", "many"]
    
    masc_words = ["he", "him", "his", "himself", "man", "boy"]
    fem_words = ["she", "her", "hers", "herself", "woman", "girl"]
    
    past_words = ["was", "were", "had", "did", "went", "came"]
    present_words = ["is", "are", "has", "does", "goes", "comes"]
    
    def compute_direction(pos_words, neg_words, W_U, tokenizer):
        """Compute a constraint direction = mean(W_U[pos]) - mean(W_U[neg])"""
        pos_vecs = []
        for w in pos_words:
            ids = tokenizer.encode(w, add_special_tokens=False)
            for tid in ids:
                if tid < W_U.shape[0]:
                    pos_vecs.append(W_U[tid])
        
        neg_vecs = []
        for w in neg_words:
            ids = tokenizer.encode(w, add_special_tokens=False)
            for tid in ids:
                if tid < W_U.shape[0]:
                    neg_vecs.append(W_U[tid])
        
        if not pos_vecs or not neg_vecs:
            return np.zeros(d_model)
        
        pos_mean = np.mean(pos_vecs, axis=0)
        neg_mean = np.mean(neg_vecs, axis=0)
        direction = pos_mean - neg_mean
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        return direction
    
    number_dir = compute_direction(singular_words, plural_words, W_U, tokenizer)
    gender_dir = compute_direction(masc_words, fem_words, W_U, tokenizer)
    tense_dir = compute_direction(past_words, present_words, W_U, tokenizer)
    
    # Compute direction independence
    cos_ng = float(np.dot(number_dir, gender_dir))
    cos_nt = float(np.dot(number_dir, tense_dir))
    cos_gt = float(np.dot(gender_dir, tense_dir))
    
    print(f"    Number-Gender cosine: {cos_ng:.4f}", flush=True)
    print(f"    Number-Tense cosine: {cos_nt:.4f}", flush=True)
    print(f"    Gender-Tense cosine: {cos_gt:.4f}", flush=True)
    
    # Step 2: Causal intervention — ablate one constraint, measure others
    print("\n  Step 2: Causal intervention — ablate constraint directions...", flush=True)
    
    test_sentences = [
        "The cat sleeps on the",
        "The actor said he was",
        "She walked and talked",
        "The dogs run to the",
        "The actress said she was",
        "He runs and jumps",
    ]
    
    intervention_layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]
    
    constraint_dirs = {
        "number": number_dir,
        "gender": gender_dir,
        "tense": tense_dir,
    }
    
    dependency_results = {}
    
    for ablated_constraint, ablate_dir in constraint_dirs.items():
        print(f"\n  Ablating '{ablated_constraint}' direction...", flush=True)
        
        for li in intervention_layers:
            # For each test sentence, ablate the constraint direction and measure all signals
            effects = defaultdict(list)
            
            for sent in test_sentences:
                try:
                    input_device = next(model.parameters()).device
                    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
                    input_ids_dev = inputs["input_ids"].to(input_device)
                    attn_mask_dev = inputs["attention_mask"].to(input_device)
                    
                    # Normal forward
                    with torch.no_grad():
                        normal_out = model(input_ids=input_ids_dev, attention_mask=attn_mask_dev)
                    normal_logits = normal_out.logits[0, -1].float().cpu().numpy()
                    
                    # Normal signals
                    normal_number = compute_number_signal(normal_logits, tokenizer)
                    normal_gender = compute_gender_signal(normal_logits, tokenizer)
                    normal_tense = compute_tense_signal(normal_logits, tokenizer)
                    
                    # Ablated forward: project out the ablated constraint direction
                    subspace_t = torch.tensor(ablate_dir.reshape(1, -1), dtype=torch.float32)
                    
                    def make_ablation_hook(subspace):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                h = output[0]
                            else:
                                h = output
                            subspace_dev = subspace.to(h.device).to(h.dtype)
                            proj = torch.matmul(subspace_dev, h.transpose(-1, -2))
                            recon = torch.matmul(subspace_dev.T, proj)
                            h_ablated = h - recon.transpose(-1, -2)
                            if isinstance(output, tuple):
                                return (h_ablated,) + output[1:]
                            return h_ablated
                        return hook
                    
                    hooks = [layers[li].register_forward_hook(make_ablation_hook(subspace_t))]
                    
                    with torch.no_grad():
                        ablated_out = model(input_ids=input_ids_dev, attention_mask=attn_mask_dev)
                    
                    for h in hooks:
                        h.remove()
                    
                    ablated_logits = ablated_out.logits[0, -1].float().cpu().numpy()
                    
                    # Ablated signals
                    ablated_number = compute_number_signal(ablated_logits, tokenizer)
                    ablated_gender = compute_gender_signal(ablated_logits, tokenizer)
                    ablated_tense = compute_tense_signal(ablated_logits, tokenizer)
                    
                    # Changes
                    effects["number"].append(ablated_number - normal_number)
                    effects["gender"].append(ablated_gender - normal_gender)
                    effects["tense"].append(ablated_tense - normal_tense)
                    
                except Exception as e:
                    continue
            
            # Average effects
            avg_effects = {}
            for constraint_name, deltas in effects.items():
                avg_effects[constraint_name] = {
                    "mean_delta": round(float(np.mean(deltas)), 4),
                    "std_delta": round(float(np.std(deltas)), 4),
                    "is_ablated": constraint_name == ablated_constraint,
                }
            
            key = f"ablate_{ablated_constraint}_L{li}"
            dependency_results[key] = avg_effects
            
            # Print: ablated constraint effect vs other constraints
            ablated_effect = avg_effects.get(ablated_constraint, {}).get("mean_delta", 0)
            other_effects = [v["mean_delta"] for k, v in avg_effects.items() if k != ablated_constraint]
            avg_other = float(np.mean(other_effects)) if other_effects else 0
            
            print(f"    L{li}: ablated_{ablated_constraint}_Δ={ablated_effect:.4f}, "
                  f"other_avg_Δ={avg_other:.4f}", flush=True)
    
    # Step 3: Random direction control
    print("\n  Step 3: Random direction control...", flush=True)
    
    random_results = {}
    n_random_trials = 5
    
    for trial in range(n_random_trials):
        random_dir = np.random.randn(d_model)
        random_dir = random_dir / np.linalg.norm(random_dir)
        
        for li in [n_layers // 2, n_layers - 1]:
            effects = defaultdict(list)
            
            for sent in test_sentences[:3]:
                try:
                    input_device = next(model.parameters()).device
                    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
                    input_ids_dev = inputs["input_ids"].to(input_device)
                    attn_mask_dev = inputs["attention_mask"].to(input_device)
                    
                    with torch.no_grad():
                        normal_out = model(input_ids=input_ids_dev, attention_mask=attn_mask_dev)
                    normal_logits = normal_out.logits[0, -1].float().cpu().numpy()
                    
                    subspace_t = torch.tensor(random_dir.reshape(1, -1), dtype=torch.float32)
                    
                    def make_rand_hook(subspace):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                h = output[0]
                            else:
                                h = output
                            subspace_dev = subspace.to(h.device).to(h.dtype)
                            proj = torch.matmul(subspace_dev, h.transpose(-1, -2))
                            recon = torch.matmul(subspace_dev.T, proj)
                            h_ablated = h - recon.transpose(-1, -2)
                            if isinstance(output, tuple):
                                return (h_ablated,) + output[1:]
                            return h_ablated
                        return hook
                    
                    hooks = [layers[li].register_forward_hook(make_rand_hook(subspace_t))]
                    
                    with torch.no_grad():
                        ablated_out = model(input_ids=input_ids_dev, attention_mask=attn_mask_dev)
                    
                    for h in hooks:
                        h.remove()
                    
                    ablated_logits = ablated_out.logits[0, -1].float().cpu().numpy()
                    
                    effects["number"].append(compute_number_signal(ablated_logits, tokenizer) - 
                                            compute_number_signal(normal_logits, tokenizer))
                    effects["gender"].append(compute_gender_signal(ablated_logits, tokenizer) - 
                                            compute_gender_signal(normal_logits, tokenizer))
                    effects["tense"].append(compute_tense_signal(ablated_logits, tokenizer) - 
                                            compute_tense_signal(normal_logits, tokenizer))
                except:
                    continue
            
            key = f"random_trial{trial}_L{li}"
            random_results[key] = {
                cname: round(float(np.mean(deltas)), 4)
                for cname, deltas in effects.items() if deltas
            }
    
    # Average random effects
    random_avg = defaultdict(list)
    for key, vals in random_results.items():
        for cname, delta in vals.items():
            random_avg[cname].append(delta)
    
    random_baseline = {cname: round(float(np.mean(deltas)), 4) for cname, deltas in random_avg.items()}
    print(f"    Random ablation baseline: {random_baseline}", flush=True)
    
    return {
        "direction_cosines": {
            "number_gender": round(cos_ng, 4),
            "number_tense": round(cos_nt, 4),
            "gender_tense": round(cos_gt, 4),
        },
        "dependency_graph": dependency_results,
        "random_baseline": random_baseline,
        "random_details": random_results,
    }


# =====================================================================
# Phase C: ★★★ 约束能量景观 (Dynamical Invariants) ★★★
# =====================================================================

def run_constraint_energy_landscape(model, tokenizer, device, model_info, W_U):
    """
    ★★★ Phase C: 约束能量是否逐层递减? ★★★
    
    定义:
    - E(l) = -Σ_C σ_C(h_l) (负的约束满足度之和)
    - 约束闭合率 = -dE/dl
    
    假设:
    - E应该递减 → 系统收敛到约束满足态
    - 语法约束比语义约束更早闭合
    - 能量递减率是动力学不变量
    """
    n_layers = model_info.n_layers
    all_layers = list(range(n_layers + 1))
    
    print("\n" + "="*70, flush=True)
    print("Phase C: ★★★ 约束能量景观 (Dynamical Invariants) ★★★", flush=True)
    print("="*70, flush=True)
    
    # Test sentences: both syntactically correct and semantically meaningful
    test_sentences = [
        ("The cat sleeps on the mat", "syntax_correct"),
        ("The cat sleep on the mat", "syntax_wrong"),
        ("The actor said he was tired", "syntax_correct"),
        ("The actor said she was tired", "syntax_wrong"),
        ("She walked and talked about", "syntax_correct"),
        ("She walked and talks about", "syntax_wrong"),
        ("The apple is sweet and fresh", "semantic"),
        ("The car drives fast on the road", "semantic"),
        ("The cat is a small animal", "semantic"),
    ]
    
    energy_results = {}
    
    for sent, sent_type in test_sentences:
        input_device = next(model.parameters()).device
        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
        
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"].to(input_device),
                attention_mask=inputs["attention_mask"].to(input_device),
                output_hidden_states=True,
            )
        
        hs = out.hidden_states
        
        # Compute all constraint signals at each layer
        layer_signals = {}
        for li in all_layers:
            if li >= len(hs):
                continue
            h = hs[li][0, -1].float().cpu().numpy()
            logits_l = W_U @ h
            
            signals = {
                "number": compute_number_signal(logits_l, tokenizer),
                "gender": compute_gender_signal(logits_l, tokenizer),
                "tense": compute_tense_signal(logits_l, tokenizer),
            }
            layer_signals[li] = signals
        
        # Compute energy E(l) = -Σ |σ_C|
        # We use absolute value because both positive and negative signals indicate constraint encoding
        energy = {}
        for li, signals in layer_signals.items():
            E = -sum(abs(v) for v in signals.values())
            energy[li] = round(E, 4)
        
        # Compute constraint closure rate -dE/dl
        closure_rates = {}
        sorted_layers = sorted(energy.keys())
        for i in range(1, len(sorted_layers)):
            li_prev = sorted_layers[i-1]
            li_curr = sorted_layers[i]
            dE = energy[li_curr] - energy[li_prev]
            dl = li_curr - li_prev
            closure_rates[li_curr] = round(-dE / dl, 4)
        
        # Find the layer of maximum closure rate (fastest constraint enforcement)
        if closure_rates:
            max_closure_layer = max(closure_rates, key=closure_rates.get)
            max_closure_rate = closure_rates[max_closure_layer]
        else:
            max_closure_layer = None
            max_closure_rate = 0
        
        energy_results[sent_type + "_" + sent[:20]] = {
            "sentence": sent,
            "type": sent_type,
            "energy": {str(li): v for li, v in energy.items()},
            "closure_rates": {str(li): v for li, v in closure_rates.items()},
            "max_closure_layer": max_closure_layer,
            "max_closure_rate": max_closure_rate,
            "E_L0": energy.get(0, 0),
            "E_L_last": energy.get(n_layers, 0),
            "total_energy_change": round(energy.get(n_layers, 0) - energy.get(0, 0), 4),
        }
        
        print(f"  [{sent_type}] '{sent[:30]}...'", flush=True)
        print(f"    E(L0)={energy.get(0, 0):.4f}, E(L_last)={energy.get(n_layers, 0):.4f}, "
              f"ΔE={energy.get(n_layers, 0) - energy.get(0, 0):.4f}", flush=True)
        if max_closure_layer is not None:
            print(f"    Max closure rate at L{max_closure_layer}: {max_closure_rate:.4f}", flush=True)
    
    # Compare: syntax_correct vs syntax_wrong energy profiles
    print("\n  Comparison: syntax_correct vs syntax_wrong energy profiles", flush=True)
    
    syntax_comparison = {}
    for key, data in energy_results.items():
        if data["type"] in ["syntax_correct", "syntax_wrong"]:
            syntax_comparison[key] = {
                "type": data["type"],
                "E_L0": data["E_L0"],
                "E_L_last": data["E_L_last"],
                "total_change": data["total_energy_change"],
            }
    
    # Key insight: correct sentences should have LOWER final energy (more constrained)
    correct_energies = [d["E_L_last"] for d in syntax_comparison.values() if d["type"] == "syntax_correct"]
    wrong_energies = [d["E_L_last"] for d in syntax_comparison.values() if d["type"] == "syntax_wrong"]
    
    if correct_energies and wrong_energies:
        avg_correct = float(np.mean(correct_energies))
        avg_wrong = float(np.mean(wrong_energies))
        print(f"    Avg E(L_last) correct: {avg_correct:.4f}", flush=True)
        print(f"    Avg E(L_last) wrong: {avg_wrong:.4f}", flush=True)
        print(f"    → Correct sentences have {'lower' if avg_correct < avg_wrong else 'higher'} "
              f"final energy", flush=True)
    
    return {
        "energy_profiles": energy_results,
        "syntax_comparison": syntax_comparison,
        "avg_correct_final_energy": float(np.mean(correct_energies)) if correct_energies else 0,
        "avg_wrong_final_energy": float(np.mean(wrong_energies)) if wrong_energies else 0,
    }


# =====================================================================
# Phase D: ★★★ 概念 = 约束交集 (Concept = Constraint Intersection) ★★★
# =====================================================================

def run_concept_constraint_intersection(model, tokenizer, device, model_info, W_U):
    """
    ★★★ Phase D: 概念 = 必要约束的交集 ★★★
    
    核心区分:
    - 必要约束: 删除 → 概念崩塌 (如: apple的edible约束)
    - 非必要约束: 删除 → 概念存活 (如: apple的round约束 — banana没有但仍然是水果)
    - 无关约束: 删除 → 概念不受影响 (如: apple的mechanical约束)
    
    测试方法:
    1. 对每个概念, 通过hook删除一个约束方向
    2. 测量: 概念词本身的logit是否大幅下降?
    3. 比较必要 vs 非必要 vs 无关约束的删除效果
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers = get_layers(model)
    
    print("\n" + "="*70, flush=True)
    print("Phase D: ★★★ 概念 = 约束交集 (Concept = Constraint Intersection) ★★★", flush=True)
    print("="*70, flush=True)
    
    intersection_results = {}
    
    for concept_name, concept_data in CONCEPT_FORMAL_CONSTRAINTS.items():
        print(f"\n  Concept: '{concept_name}'", flush=True)
        
        necessary = concept_data["necessary"]
        unnecessary = concept_data.get("unnecessary", {})
        test_sents = concept_data["test_sentences"]
        
        # Compute constraint directions for all constraints
        all_constraints = {**necessary, **unnecessary}
        constraint_dirs = {}
        
        for cname, cwords in all_constraints.items():
            c_tok_ids = []
            for w in cwords:
                ids = tokenizer.encode(w, add_special_tokens=False)
                c_tok_ids.extend([tid for tid in ids if tid < W_U.shape[0]])
            
            if c_tok_ids:
                c_dir = np.mean(W_U[c_tok_ids], axis=0)
                c_norm = np.linalg.norm(c_dir)
                if c_norm > 1e-10:
                    constraint_dirs[cname] = c_dir / c_norm
        
        # For each test sentence, compute concept word logit under each ablation
        for sent in test_sents:
            input_device = next(model.parameters()).device
            inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
            input_ids_dev = inputs["input_ids"].to(input_device)
            attn_mask_dev = inputs["attention_mask"].to(input_device)
            
            # Normal forward — get concept word logit
            with torch.no_grad():
                normal_out = model(input_ids=input_ids_dev, attention_mask=attn_mask_dev)
            normal_logits = normal_out.logits[0, -1].float().cpu().numpy()
            
            # Concept word logit (the word itself)
            concept_tok_ids = tokenizer.encode(concept_name, add_special_tokens=False)
            concept_logit_ids = [tid for tid in concept_tok_ids if tid < len(normal_logits)]
            
            if not concept_logit_ids:
                continue
            
            normal_concept_logit = float(np.mean([normal_logits[tid] for tid in concept_logit_ids]))
            
            # Also compute all constraint satisfactions from logits
            normal_cs = {}
            for cname, cwords in all_constraints.items():
                normal_cs[cname] = compute_constraint_signal(normal_logits, tokenizer, cwords)
            
            # Ablate each constraint direction and measure impact
            for li in [n_layers // 2, n_layers - 1]:
                ablation_effects = {}
                
                for cname, c_dir in constraint_dirs.items():
                    is_necessary = cname in necessary
                    
                    subspace_t = torch.tensor(c_dir.reshape(1, -1), dtype=torch.float32)
                    
                    def make_ablation_hook(subspace):
                        def hook(module, input, output):
                            if isinstance(output, tuple):
                                h = output[0]
                            else:
                                h = output
                            subspace_dev = subspace.to(h.device).to(h.dtype)
                            proj = torch.matmul(subspace_dev, h.transpose(-1, -2))
                            recon = torch.matmul(subspace_dev.T, proj)
                            h_ablated = h - recon.transpose(-1, -2)
                            if isinstance(output, tuple):
                                return (h_ablated,) + output[1:]
                            return h_ablated
                        return hook
                    
                    hooks = [layers[li].register_forward_hook(make_ablation_hook(subspace_t))]
                    
                    with torch.no_grad():
                        ablated_out = model(input_ids=input_ids_dev, attention_mask=attn_mask_dev)
                    
                    for h in hooks:
                        h.remove()
                    
                    ablated_logits = ablated_out.logits[0, -1].float().cpu().numpy()
                    
                    # Concept word logit change
                    ablated_concept_logit = float(np.mean([ablated_logits[tid] for tid in concept_logit_ids]))
                    concept_delta = ablated_concept_logit - normal_concept_logit
                    
                    # Target constraint satisfaction change
                    ablated_cs = compute_constraint_signal(ablated_logits, tokenizer, all_constraints[cname])
                    target_delta = ablated_cs - normal_cs[cname]
                    
                    ablation_effects[cname] = {
                        "is_necessary": is_necessary,
                        "concept_logit_delta": round(float(concept_delta), 4),
                        "target_constraint_delta": round(float(target_delta), 4),
                    }
                
                # Summarize: necessary vs unnecessary vs irrelevant
                necessary_deltas = [v["concept_logit_delta"] for k, v in ablation_effects.items() if v["is_necessary"]]
                unnecessary_deltas = [v["concept_logit_delta"] for k, v in ablation_effects.items() if not v["is_necessary"]]
                
                avg_necessary = float(np.mean([abs(d) for d in necessary_deltas])) if necessary_deltas else 0
                avg_unnecessary = float(np.mean([abs(d) for d in unnecessary_deltas])) if unnecessary_deltas else 0
                
                key = f"{concept_name}_L{li}"
                intersection_results[key] = {
                    "concept": concept_name,
                    "layer": li,
                    "sentence": sent[:30],
                    "ablation_effects": ablation_effects,
                    "avg_necessary_concept_impact": round(avg_necessary, 4),
                    "avg_unnecessary_concept_impact": round(avg_unnecessary, 4),
                    "necessary_larger": avg_necessary > avg_unnecessary,
                }
                
                print(f"    L{li}: necessary_impact={avg_necessary:.4f}, "
                      f"unnecessary_impact={avg_unnecessary:.4f}, "
                      f"necessary>{'✓' if avg_necessary > avg_unnecessary else '✗'}", flush=True)
    
    # Overall summary
    necessary_larger_count = sum(1 for v in intersection_results.values() if v["necessary_larger"])
    total_count = len(intersection_results)
    
    print(f"\n  Overall: necessary impact > unnecessary in "
          f"{necessary_larger_count}/{total_count} cases", flush=True)
    
    return {
        "intersection_results": intersection_results,
        "necessary_larger_ratio": round(necessary_larger_count / max(total_count, 1), 4),
    }


# =====================================================================
# MAIN
# =====================================================================

def run_phase177(model_name):
    print(f"\n{'='*70}", flush=True)
    print(f"Phase 177: ★★★ 约束形式化 — 从语义标签到数学约束 ★★★", flush=True)
    print(f"Model: {model_name}", flush=True)
    print(f"{'='*70}", flush=True)

    t_start = time.time()

    # Load model (BF16 + device_map="auto")
    model, tokenizer, device = load_model_bf16(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    print(f"  Model: {model_info.model_class}, L={n_layers}, d={d_model}", flush=True)

    # Load W_U
    print("  Loading W_U (unembedding matrix)...", flush=True)
    W_U = get_W_U(model, model_name)
    print(f"  W_U shape: {W_U.shape}", flush=True)

    # =====================================================================
    # Run all experiments
    # =====================================================================

    # Phase A: Formal Syntactic Constraint Verification
    exp_a = run_formal_constraint_verification(model, tokenizer, device, model_info, W_U)

    # Phase B: Constraint Dependency Graph
    exp_b = run_constraint_dependency_graph(model, tokenizer, device, model_info, W_U)

    # Phase C: Constraint Energy Landscape
    exp_c = run_constraint_energy_landscape(model, tokenizer, device, model_info, W_U)

    # Phase D: Concept = Constraint Intersection
    exp_d = run_concept_constraint_intersection(model, tokenizer, device, model_info, W_U)

    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    output = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": d_model,
        "timestamp": timestamp,
        "phase_A_formal_constraints": exp_a,
        "phase_B_dependency_graph": exp_b,
        "phase_C_energy_landscape": exp_c,
        "phase_D_concept_intersection": exp_d,
    }

    out_path = f"tests/glm5_temp/phase177_{model_name}_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_path}", flush=True)

    # Release model
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    print(f"\nPhase 177 ({model_name}) completed in {elapsed:.1f}s", flush=True)

    return output


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python phase177_constraint_formalization.py <model_name>")
        print("  model_name: qwen3, glm4, deepseek7b")
        sys.exit(1)

    model_name = sys.argv[1]
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        sys.exit(1)

    run_phase177(model_name)
