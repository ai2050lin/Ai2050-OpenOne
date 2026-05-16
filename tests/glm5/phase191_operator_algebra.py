"""
Phase 191: 回路代数与角色绑定 — 从"向量几何"到"算子结构"
==========================================================

用户核心洞察:
1. 语义 = transformation (算子), 不是 vector (点)
2. NOT, PAST, PLURAL 是 operators, 不是 embeddings
3. 关键问题: NOT(PAST(x)) ≈ PAST(NOT(x))? (交换律)
4. 关键问题: "dog bites man" vs "man bites dog" 的角色绑定机制
5. 语言编码可能是 fiber bundle: base=objects, fibers=transformations

实验设计:
- Exp1: 算子代数 — NOT/PAST/PLURAL/QUESTION的组合律 (交换律, 结合律, 加法性)
- Exp2: 角色绑定 — "dog bites man" vs "man bites dog" 的head路由差异
- Exp3: 注意力路由图 — 不同句型的信息流模式
- Exp4: 跨层状态精化 — 各层如何逐步更新语义状态

用法:
  python tests/glm5/phase191_operator_algebra.py qwen3
  python tests/glm5/phase191_operator_algebra.py glm4
  python tests/glm5/phase191_operator_algebra.py deepseek7b
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import time
import json
import numpy as np
import torch
from collections import defaultdict
from itertools import combinations

from model_utils import (get_layers, get_model_info, release_model,
                          get_W_U, MODEL_CONFIGS)


# ===== Exp1: 算子组合测试集 =====
# 每个base sentence有8个变体: base, NOT, PAST, PLURAL, NOT+PAST, NOT+PLURAL, PAST+PLURAL, NOT+PAST+PLURAL
# 这允许我们测试算子的交换律和加法性

OPERATOR_TEST_SETS = [
    # (base, NOT, PAST, PLURAL, NOT+PAST, NOT+PLURAL, PAST+PLURAL, NOT+PAST+PLURAL)
    {
        "base": "The cat sleeps",
        "not": "The cat does not sleep",
        "past": "The cat slept",
        "plural": "The cats sleep",
        "not_past": "The cat did not sleep",
        "not_plural": "The cats do not sleep",
        "past_plural": "The cats slept",
        "not_past_plural": "The cats did not sleep",
    },
    {
        "base": "She walks home",
        "not": "She does not walk home",
        "past": "She walked home",
        "plural": "They walk home",
        "not_past": "She did not walk home",
        "not_plural": "They do not walk home",
        "past_plural": "They walked home",
        "not_past_plural": "They did not walk home",
    },
    {
        "base": "The bird sings",
        "not": "The bird does not sing",
        "past": "The bird sang",
        "plural": "The birds sing",
        "not_past": "The bird did not sing",
        "not_plural": "The birds do not sing",
        "past_plural": "The birds sang",
        "not_past_plural": "The birds did not sing",
    },
    {
        "base": "He runs fast",
        "not": "He does not run fast",
        "past": "He ran fast",
        "plural": "They run fast",
        "not_past": "He did not run fast",
        "not_plural": "They do not run fast",
        "past_plural": "They ran fast",
        "not_past_plural": "They did not run fast",
    },
    {
        "base": "The river flows east",
        "not": "The river does not flow east",
        "past": "The river flowed east",
        "plural": "The rivers flow east",
        "not_past": "The river did not flow east",
        "not_plural": "The rivers do not flow east",
        "past_plural": "The rivers flowed east",
        "not_past_plural": "The rivers did not flow east",
    },
    {
        "base": "The student reads",
        "not": "The student does not read",
        "past": "The student read",
        "plural": "The students read",
        "not_past": "The student did not read",
        "not_plural": "The students do not read",
        "past_plural": "The students read",
        "not_past_plural": "The students did not read",
    },
    {
        "base": "The wind blows hard",
        "not": "The wind does not blow hard",
        "past": "The wind blew hard",
        "plural": "The winds blow hard",
        "not_past": "The wind did not blow hard",
        "not_plural": "The winds do not blow hard",
        "past_plural": "The winds blew hard",
        "not_past_plural": "The winds did not blow hard",
    },
    {
        "base": "She cooks dinner",
        "not": "She does not cook dinner",
        "past": "She cooked dinner",
        "plural": "They cook dinner",
        "not_past": "She did not cook dinner",
        "not_plural": "They do not cook dinner",
        "past_plural": "They cooked dinner",
        "not_past_plural": "They did not cook dinner",
    },
    {
        "base": "The light shines bright",
        "not": "The light does not shine bright",
        "past": "The light shone bright",
        "plural": "The lights shine bright",
        "not_past": "The light did not shine bright",
        "not_plural": "The lights do not shine bright",
        "past_plural": "The lights shone bright",
        "not_past_plural": "The lights did not shine bright",
    },
    {
        "base": "The dog barks loudly",
        "not": "The dog does not bark loudly",
        "past": "The dog barked loudly",
        "plural": "The dogs bark loudly",
        "not_past": "The dog did not bark loudly",
        "not_plural": "The dogs do not bark loudly",
        "past_plural": "The dogs barked loudly",
        "not_past_plural": "The dogs did not bark loudly",
    },
    # Extra sentences for larger sample
    {
        "base": "The fish swims deep",
        "not": "The fish does not swim deep",
        "past": "The fish swam deep",
        "plural": "The fish swim deep",
        "not_past": "The fish did not swim deep",
        "not_plural": "The fish do not swim deep",
        "past_plural": "The fish swam deep",
        "not_past_plural": "The fish did not swim deep",
    },
    {
        "base": "The tree grows tall",
        "not": "The tree does not grow tall",
        "past": "The tree grew tall",
        "plural": "The trees grow tall",
        "not_past": "The tree did not grow tall",
        "not_plural": "The trees do not grow tall",
        "past_plural": "The trees grew tall",
        "not_past_plural": "The trees did not grow tall",
    },
    {
        "base": "He drives carefully",
        "not": "He does not drive carefully",
        "past": "He drove carefully",
        "plural": "They drive carefully",
        "not_past": "He did not drive carefully",
        "not_plural": "They do not drive carefully",
        "past_plural": "They drove carefully",
        "not_past_plural": "They did not drive carefully",
    },
    {
        "base": "The bell rings loudly",
        "not": "The bell does not ring loudly",
        "past": "The bell rang loudly",
        "plural": "The bells ring loudly",
        "not_past": "The bell did not ring loudly",
        "not_plural": "The bells do not ring loudly",
        "past_plural": "The bells rang loudly",
        "not_past_plural": "The bells did not ring loudly",
    },
    {
        "base": "She writes poetry",
        "not": "She does not write poetry",
        "past": "She wrote poetry",
        "plural": "They write poetry",
        "not_past": "She did not write poetry",
        "not_plural": "They do not write poetry",
        "past_plural": "They wrote poetry",
        "not_past_plural": "They did not write poetry",
    },
]


# ===== Exp2: 角色绑定测试集 =====
# "dog bites man" vs "man bites dog" — 相同token, 不同角色
# 还包括被动语态对比: "the man is bitten by the dog"

ROLE_BINDING_SETS = [
    {
        "agent_action_patient": "The dog bites the man",
        "patient_action_agent": "The man bites the dog",
        "passive": "The man is bitten by the dog",
    },
    {
        "agent_action_patient": "The teacher praised the student",
        "patient_action_agent": "The student praised the teacher",
        "passive": "The student was praised by the teacher",
    },
    {
        "agent_action_patient": "The king punished the servant",
        "patient_action_agent": "The servant punished the king",
        "passive": "The servant was punished by the king",
    },
    {
        "agent_action_patient": "The mother hugged the child",
        "patient_action_agent": "The child hugged the mother",
        "passive": "The child was hugged by the mother",
    },
    {
        "agent_action_patient": "The cat chased the mouse",
        "patient_action_agent": "The mouse chased the cat",
        "passive": "The mouse was chased by the cat",
    },
    {
        "agent_action_patient": "The boy helped the girl",
        "patient_action_agent": "The girl helped the boy",
        "passive": "The girl was helped by the boy",
    },
    {
        "agent_action_patient": "The doctor treated the patient",
        "patient_action_agent": "The patient treated the doctor",
        "passive": "The patient was treated by the doctor",
    },
    {
        "agent_action_patient": "The police arrested the thief",
        "patient_action_agent": "The thief arrested the police",
        "passive": "The thief was arrested by the police",
    },
    {
        "agent_action_patient": "The judge sentenced the criminal",
        "patient_action_agent": "The criminal sentenced the judge",
        "passive": "The criminal was sentenced by the judge",
    },
    {
        "agent_action_patient": "The boss fired the worker",
        "patient_action_agent": "The worker fired the boss",
        "passive": "The worker was fired by the boss",
    },
    {
        "agent_action_patient": "The hunter killed the deer",
        "patient_action_agent": "The deer killed the hunter",
        "passive": "The deer was killed by the hunter",
    },
    {
        "agent_action_patient": "The manager hired the employee",
        "patient_action_agent": "The employee hired the manager",
        "passive": "The employee was hired by the manager",
    },
    {
        "agent_action_patient": "The chef cooked the meal",
        "patient_action_agent": "The meal cooked the chef",
        "passive": "The meal was cooked by the chef",
    },
    {
        "agent_action_patient": "The wind destroyed the house",
        "patient_action_agent": "The house destroyed the wind",
        "passive": "The house was destroyed by the wind",
    },
    {
        "agent_action_patient": "The soldier protected the civilian",
        "patient_action_agent": "The civilian protected the soldier",
        "passive": "The civilian was protected by the soldier",
    },
    # Extra pairs for robustness
    {
        "agent_action_patient": "The wolf attacked the sheep",
        "patient_action_agent": "The sheep attacked the wolf",
        "passive": "The sheep was attacked by the wolf",
    },
    {
        "agent_action_patient": "The writer published the book",
        "patient_action_agent": "The book published the writer",
        "passive": "The book was published by the writer",
    },
    {
        "agent_action_patient": "The coach trained the athlete",
        "patient_action_agent": "The athlete trained the coach",
        "passive": "The athlete was trained by the coach",
    },
    {
        "agent_action_patient": "The sun warmed the earth",
        "patient_action_agent": "The earth warmed the sun",
        "passive": "The earth was warmed by the sun",
    },
    {
        "agent_action_patient": "The leader guided the team",
        "patient_action_agent": "The team guided the leader",
        "passive": "The team was guided by the leader",
    },
]


# ===== Exp3: 注意力路由图模式 =====
# 不同句型的信息流模式

ROUTING_SENTENCE_SETS = {
    "simple_declarative": [
        "The cat sleeps on the mat",
        "She walks to the store",
        "He reads the book carefully",
        "The bird sings in the tree",
        "They play soccer every day",
    ],
    "negation": [
        "The cat does not sleep on the mat",
        "She does not walk to the store",
        "He does not read the book carefully",
        "The bird does not sing in the tree",
        "They do not play soccer every day",
    ],
    "question": [
        "Does the cat sleep on the mat?",
        "Does she walk to the store?",
        "Does he read the book carefully?",
        "Does the bird sing in the tree?",
        "Do they play soccer every day?",
    ],
    "conditional": [
        "If the cat sleeps on the mat, it is happy",
        "If she walks to the store, she buys milk",
        "If he reads the book carefully, he learns",
        "If the bird sings in the tree, spring has come",
        "If they play soccer every day, they improve",
    ],
    "causal": [
        "Because the cat sleeps on the mat, it is warm",
        "Because she walks to the store, she is healthy",
        "Because he reads the book carefully, he understands",
        "Because the bird sings in the tree, the morning is beautiful",
        "Because they play soccer every day, they are fit",
    ],
}


# ===== Exp4: 跨层状态精化 =====
# 追踪各层如何区分语义功能

STATE_TRACKING_PAIRS = {
    "negation": [
        ("The cat is sleeping", "The cat is not sleeping"),
        ("She likes the movie", "She does not like the movie"),
        ("He can swim well", "He cannot swim well"),
        ("The door was open", "The door was not open"),
        ("Birds can fly high", "Birds cannot fly high"),
    ],
    "tense": [
        ("She walks to school", "She walked to school"),
        ("He eats breakfast", "He ate breakfast"),
        ("They play soccer", "They played soccer"),
        ("The bird sings", "The bird sang"),
        ("She writes letters", "She wrote letters"),
    ],
    "role_binding": [
        ("The dog bites the man", "The man bites the dog"),
        ("The teacher praised the student", "The student praised the teacher"),
        ("The king punished the servant", "The servant punished the king"),
        ("The cat chased the mouse", "The mouse chased the cat"),
        ("The boy helped the girl", "The girl helped the boy"),
    ],
}


def load_model_bf16_auto(model_name: str):
    """BF16 + device_map=auto 加载 (参考 model_demo_bf16.py)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = MODEL_CONFIGS[model_name]
    print(f"[P191] Loading {model_name} (bfloat16 + device_map=auto)...")
    print(f"[P191] Path: {cfg['path']}")

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
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
    print(f"[P191] {model_name} loaded: device={device}, GPU={gpu_mem:.2f}GB")

    return model, tokenizer, device


def get_input_device(model):
    """获取输入tensor应放的设备"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_hidden_states_all_layers(model, tokenizer, sentences, desc=""):
    """提取所有层的隐藏状态 (最后一个token)"""
    input_device = get_input_device(model)
    results = {}

    for i, sent in enumerate(sentences):
        if desc and i % 5 == 0:
            print(f"  [{desc}] {i}/{len(sentences)}: {sent[:50]}...")

        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)

        last_pos = attention_mask.sum().item() - 1
        hs = [layer_hs[0, last_pos].float().cpu().numpy()
              for layer_hs in out.hidden_states]
        results[sent] = hs

    return results


def extract_attn_patterns(model, tokenizer, sentences, n_layers, desc=""):
    """提取注意力模式 (每个head对最后一个token的注意力分布)"""
    input_device = get_input_device(model)
    results = {}

    for i, sent in enumerate(sentences):
        if desc and i % 5 == 0:
            print(f"  [{desc}] {i}/{len(sentences)}: {sent[:50]}...")

        inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(input_device)
        attention_mask = inputs["attention_mask"].to(input_device)

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_attentions=True)

        if out.attentions is not None:
            last_pos = attention_mask.sum().item() - 1
            attn = torch.stack([a[0, :, last_pos, :].float().cpu()
                                for a in out.attentions])
            results[sent] = attn.numpy()
        else:
            results[sent] = None

    return results


def cosine_sim(a, b):
    """余弦相似度"""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# ===== Exp1: 算子代数 =====
def exp1_operator_algebra(model, tokenizer, device, model_name):
    """
    测试算子的代数性质:
    1. 交换律: NOT(PAST(x)) ≈ PAST(NOT(x))?
    2. 加法性: d_NOT+PAST ≈ d_NOT + d_PAST?
    3. 算子正交性: d_NOT ⊥ d_PAST ⊥ d_PLURAL?
    4. 算子独立性: 各算子在不同base上是否稳定?
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    print(f"\n{'='*70}")
    print(f"Exp1: 算子代数 — {model_name}")
    print(f"{'='*70}")

    # 收集所有句子
    all_sentences = []
    for s in OPERATOR_TEST_SETS:
        for key in ["base", "not", "past", "plural", "not_past", "not_plural",
                     "past_plural", "not_past_plural"]:
            all_sentences.append(s[key])

    all_sentences = list(set(all_sentences))
    print(f"  总句数: {len(all_sentences)}")

    # 提取hidden states
    print(f"  提取hidden states...")
    hs_dict = extract_hidden_states_all_layers(model, tokenizer, all_sentences, desc="Exp1")

    # 分析层: 用中间层 (避免太浅或太深的层)
    analysis_layer = n_layers // 2
    print(f"  分析层: L{analysis_layer}")

    # 1. 算子方向提取
    operator_directions = defaultdict(list)  # {op_name: [d_op for each base]}

    for s in OPERATOR_TEST_SETS:
        h_base = hs_dict[s["base"]][analysis_layer]
        for op_name, op_key in [("NOT", "not"), ("PAST", "past"), ("PLURAL", "plural")]:
            h_op = hs_dict[s[op_key]][analysis_layer]
            d_op = h_op - h_base
            operator_directions[op_name].append(d_op)

    # 2. 算子间正交性
    print(f"\n  --- 算子间正交性 ---")
    orthogonality_results = {}
    for op1, op2 in combinations(["NOT", "PAST", "PLURAL"], 2):
        cos_vals = []
        for i in range(len(operator_directions[op1])):
            cos_vals.append(cosine_sim(operator_directions[op1][i],
                                        operator_directions[op2][i]))
        mean_cos = np.mean(cos_vals)
        orthogonality_results[f"{op1}_vs_{op2}"] = {
            "mean_cos": mean_cos,
            "std_cos": float(np.std(cos_vals)),
            "individual": [float(c) for c in cos_vals],
        }
        print(f"  cos(d_{op1}, d_{op2}): mean={mean_cos:.4f} ± {np.std(cos_vals):.4f}")

    # 3. 算子方向跨base的稳定性 (同一算子在不同base上是否一致)
    print(f"\n  --- 算子方向跨base稳定性 ---")
    stability_results = {}
    for op_name in ["NOT", "PAST", "PLURAL"]:
        dirs = operator_directions[op_name]
        cos_pairs = []
        for i, j in combinations(range(len(dirs)), 2):
            cos_pairs.append(cosine_sim(dirs[i], dirs[j]))
        mean_cos = np.mean(cos_pairs) if cos_pairs else 0
        stability_results[op_name] = {
            "mean_cos_across_bases": mean_cos,
            "n_pairs": len(cos_pairs),
        }
        print(f"  {op_name} 跨base稳定性: mean_cos={mean_cos:.4f} ({len(cos_pairs)} pairs)")

    # 4. 加法性测试: d_NOT+PAST ≈ d_NOT + d_PAST?
    print(f"\n  --- 算子加法性测试 ---")
    additivity_results = {}

    for pair_name, op1_key, op2_key, combined_key in [
        ("NOT+PAST", "not", "past", "not_past"),
        ("NOT+PLURAL", "not", "plural", "not_plural"),
        ("PAST+PLURAL", "past", "plural", "past_plural"),
    ]:
        cos_vals = []
        residual_ratios = []
        for s in OPERATOR_TEST_SETS:
            h_base = hs_dict[s["base"]][analysis_layer]
            h_op1 = hs_dict[s[op1_key]][analysis_layer]
            h_op2 = hs_dict[s[op2_key]][analysis_layer]
            h_combined = hs_dict[s[combined_key]][analysis_layer]

            d_op1 = h_op1 - h_base
            d_op2 = h_op2 - h_base
            d_combined = h_combined - h_base
            d_sum = d_op1 + d_op2

            cos_combined_sum = cosine_sim(d_combined, d_sum)
            residual = np.linalg.norm(d_combined - d_sum)
            combined_norm = np.linalg.norm(d_combined)
            residual_ratio = residual / max(combined_norm, 1e-10)

            cos_vals.append(cos_combined_sum)
            residual_ratios.append(residual_ratio)

        mean_cos = np.mean(cos_vals)
        mean_residual = np.mean(residual_ratios)
        additivity_results[pair_name] = {
            "mean_cos_d_combined_vs_d1_plus_d2": mean_cos,
            "mean_residual_ratio": mean_residual,
            "n_samples": len(cos_vals),
        }
        print(f"  {pair_name}: cos(d_combined, d1+d2)={mean_cos:.4f}, "
              f"residual_ratio={mean_residual:.4f}")

    # 5. 交换律测试: d_NOT(PAST) ≈ d_PAST(NOT)?
    # 在英语中 NOT(PAST(x)) 和 PAST(NOT(x)) 的表面形式相同
    # 但内部的算子作用顺序可能不同
    # 我们用一种间接方式测试: 比较 d_NOT|PAST_base vs d_NOT|base
    # 即: 在"已经过去时"的base上施加NOT, vs 在"原base"上施加NOT
    print(f"\n  --- 算子交换律 (间接) ---")
    commutativity_results = {}

    for op1_name, op2_name, op1_key, op2_key, combined_key in [
        ("NOT", "PAST", "not", "past", "not_past"),
        ("NOT", "PLURAL", "not", "plural", "not_plural"),
        ("PAST", "PLURAL", "past", "plural", "past_plural"),
    ]:
        # d_op1(base) = h[op1(base)] - h[base]
        # d_op1(op2(base)) = h[op1(op2(base))] - h[op2(base)]
        # 如果交换律成立: d_op1(base) ≈ d_op1(op2(base))
        # 即: 算子1的效果不依赖于算子2是否已施加
        cos_same_context = []
        cos_diff_context = []

        for s in OPERATOR_TEST_SETS:
            h_base = hs_dict[s["base"]][analysis_layer]
            h_op1_on_base = hs_dict[s[op1_key]][analysis_layer]
            h_op2_on_base = hs_dict[s[op2_key]][analysis_layer]
            h_combined = hs_dict[s[combined_key]][analysis_layer]

            d_op1_on_base = h_op1_on_base - h_base           # op1施加在base上
            d_op1_on_op2_base = h_combined - h_op2_on_base   # op1施加在op2(base)上

            cos_ctx = cosine_sim(d_op1_on_base, d_op1_on_op2_base)
            cos_same_context.append(cos_ctx)

            # 也测试反方向: d_op2(base) vs d_op2(op1(base))
            d_op2_on_base = h_op2_on_base - h_base
            d_op2_on_op1_base = h_combined - h_op1_on_base
            cos_ctx2 = cosine_sim(d_op2_on_base, d_op2_on_op1_base)
            cos_diff_context.append(cos_ctx2)

        mean_cos1 = np.mean(cos_same_context)
        mean_cos2 = np.mean(cos_diff_context)
        commutativity_results[f"{op1_name}_on_{op2_name}_base"] = {
            "cos_d_op1_base_vs_d_op1_op2base": mean_cos1,
            "cos_d_op2_base_vs_d_op2_op1base": mean_cos2,
            "interpretation": "If ≈1.0, operator effect is context-independent (commutative)"
        }
        print(f"  d_{op1_name}(base) vs d_{op1_name}({op2_name}(base)): cos={mean_cos1:.4f}")
        print(f"  d_{op2_name}(base) vs d_{op2_name}({op1_name}(base)): cos={mean_cos2:.4f}")

    # 6. 三算子组合测试
    print(f"\n  --- 三算子组合 ---")
    triple_cos = []
    triple_residual = []
    for s in OPERATOR_TEST_SETS:
        h_base = hs_dict[s["base"]][analysis_layer]
        h_not = hs_dict[s["not"]][analysis_layer]
        h_past = hs_dict[s["past"]][analysis_layer]
        h_plural = hs_dict[s["plural"]][analysis_layer]
        h_all = hs_dict[s["not_past_plural"]][analysis_layer]

        d_not = h_not - h_base
        d_past = h_past - h_base
        d_plural = h_plural - h_base
        d_all = h_all - h_base
        d_sum3 = d_not + d_past + d_plural

        cos_3 = cosine_sim(d_all, d_sum3)
        residual = np.linalg.norm(d_all - d_sum3) / max(np.linalg.norm(d_all), 1e-10)
        triple_cos.append(cos_3)
        triple_residual.append(residual)

    print(f"  NOT+PAST+PLURAL: cos(d_all, d_not+d_past+d_plural)={np.mean(triple_cos):.4f}, "
          f"residual_ratio={np.mean(triple_residual):.4f}")

    return {
        "orthogonality": orthogonality_results,
        "stability": stability_results,
        "additivity": additivity_results,
        "commutativity": commutativity_results,
        "triple_cos_mean": float(np.mean(triple_cos)),
        "triple_residual_mean": float(np.mean(triple_residual)),
    }


# ===== Exp2: 角色绑定 =====
def exp2_role_binding(model, tokenizer, device, model_name):
    """
    研究"dog bites man" vs "man bites dog"的编码差异:
    1. hidden state差异的层间演化
    2. 哪些head对角色反转最敏感
    3. 主动/被动语态的编码差异
    4. 角色差异与语义差异的解耦
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    print(f"\n{'='*70}")
    print(f"Exp2: 角色绑定 — {model_name}")
    print(f"{'='*70}")

    # 收集所有句子
    all_sentences = []
    for s in ROLE_BINDING_SETS:
        all_sentences.extend([s["agent_action_patient"], s["patient_action_agent"], s["passive"]])
    all_sentences = list(set(all_sentences))
    print(f"  总句数: {len(all_sentences)}")

    # 提取hidden states
    print(f"  提取hidden states...")
    hs_dict = extract_hidden_states_all_layers(model, tokenizer, all_sentences, desc="Exp2")

    # 提取attention patterns
    print(f"  提取attention patterns...")
    attn_dict = extract_attn_patterns(model, tokenizer, all_sentences, n_layers, desc="Exp2-attn")

    # 1. 角色反转的层间差异演化
    print(f"\n  --- 角色反转的层间差异演化 ---")
    role_diff_by_layer = defaultdict(list)  # {layer: [||h_active - h_passive|| for each pair]}

    for s in ROLE_BINDING_SETS:
        h_a = hs_dict[s["agent_action_patient"]]
        h_p = hs_dict[s["patient_action_agent"]]
        for li in range(n_layers + 1):  # +1 for embedding layer
            diff_norm = np.linalg.norm(h_a[li] - h_p[li])
            role_diff_by_layer[li].append(diff_norm)

    # 打印关键层的统计
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]
    print(f"  角色反转差异 ||h(agent→patient) - h(patient→agent)|| by layer:")
    for li in key_layers:
        if li in role_diff_by_layer:
            mean_diff = np.mean(role_diff_by_layer[li])
            print(f"    L{li}: mean={mean_diff:.4f}")

    # 2. 角色差异 vs 内容差异的解耦
    print(f"\n  --- 角色差异 vs 内容差异 ---")
    # 对每对, 计算角色差异方向和内容差异方向的cosine
    # 内容差异: 用不同动词但相同角色结构的句子
    # 这里我们用间接方式: 比较角色差异方向在各层的大小

    role_directions = []  # 角色差异方向 (平均)
    for s in ROLE_BINDING_SETS:
        h_a = hs_dict[s["agent_action_patient"]][n_layers]
        h_p = hs_dict[s["patient_action_agent"]][n_layers]
        role_directions.append(h_a - h_p)

    # 角色方向间的cosine (不同句对的角色差异是否一致?)
    role_cos_pairs = []
    for i, j in combinations(range(len(role_directions)), 2):
        role_cos_pairs.append(cosine_sim(role_directions[i], role_directions[j]))
    mean_role_cos = np.mean(role_cos_pairs) if role_cos_pairs else 0
    print(f"  角色差异方向跨句对一致性: mean_cos={mean_role_cos:.4f}")

    # 3. 主动 vs 被动的编码差异
    print(f"\n  --- 主动 vs 被动语态 ---")
    active_passive_diff = []
    active_reversed_diff = []
    for s in ROLE_BINDING_SETS:
        h_active = hs_dict[s["agent_action_patient"]][n_layers]
        h_passive = hs_dict[s["passive"]][n_layers]
        h_reversed = hs_dict[s["patient_action_agent"]][n_layers]

        # 主动 vs 被动: 语义相同但结构不同
        cos_ap = cosine_sim(h_active - h_passive, h_active - h_reversed)
        active_passive_diff.append(cosine_sim(h_active, h_passive))
        active_reversed_diff.append(cosine_sim(h_active, h_reversed))

    print(f"  cos(主动, 被动[语义等价]): mean={np.mean(active_passive_diff):.4f}")
    print(f"  cos(主动, 角色反转[语义不同]): mean={np.mean(active_reversed_diff):.4f}")

    # 4. 角色敏感的attention heads
    print(f"\n  --- 角色敏感的attention heads ---")
    role_sensitive_heads = defaultdict(list)

    for s in ROLE_BINDING_SETS:
        a_key = s["agent_action_patient"]
        p_key = s["patient_action_agent"]

        if attn_dict.get(a_key) is None or attn_dict.get(p_key) is None:
            continue

        attn_a = attn_dict[a_key]  # (n_layers, n_heads, seq_len)
        attn_p = attn_dict[p_key]

        for li in range(min(attn_a.shape[0], attn_p.shape[0])):
            for hi in range(attn_a.shape[1]):
                # 对每个head, 计算注意力模式的差异
                diff = np.linalg.norm(attn_a[li, hi] - attn_p[li, hi])
                role_sensitive_heads[(li, hi)].append(diff)

    # 排序找最敏感的heads
    head_sensitivity = {}
    for (li, hi), diffs in role_sensitive_heads.items():
        head_sensitivity[(li, hi)] = np.mean(diffs)

    sorted_heads = sorted(head_sensitivity.items(), key=lambda x: x[1], reverse=True)
    print(f"  Top-10 角色敏感 heads:")
    for (li, hi), sens in sorted_heads[:10]:
        print(f"    L{li}H{hi}: mean_diff={sens:.4f}")

    # 5. 角色绑定与fiber structure
    print(f"\n  --- Fiber structure: 角色绑定作为base上的fiber ---")
    # 如果角色绑定是fiber, 那么:
    # - 主动和被动应该在base (对象内容) 上相似, 但在fiber (关系) 上不同
    # - 角色差异方向应该与内容方向正交
    content_directions = []
    for s in ROLE_BINDING_SETS:
        # 用主语作为content方向的代理
        # 简单方式: 主动句与被动句的均值代表"content", 差异代表"role fiber"
        h_a = hs_dict[s["agent_action_patient"]][n_layers]
        h_p = hs_dict[s["passive"]][n_layers]
        content_center = (h_a + h_p) / 2
        role_fiber = h_a - h_p
        content_directions.append(content_center)
        # role_directions already collected above

    # 内容中心之间的cosine (不同句对的内容是否正交?)
    content_cos = []
    for i, j in combinations(range(len(content_directions)), 2):
        content_cos.append(cosine_sim(content_directions[i], content_directions[j]))

    # 内容方向 vs 角色方向的正交性
    orth_pairs = []
    for i in range(len(content_directions)):
        c = content_cos[i] if i < len(content_cos) else 0
        orth_pairs.append(cosine_sim(content_directions[i], role_directions[i]))

    mean_orth = np.mean(orth_pairs) if orth_pairs else 0
    print(f"  cos(content_center, role_fiber): mean={mean_orth:.4f}")
    print(f"  → {'正交(支持fiber structure)' if abs(mean_orth) < 0.3 else '非正交(角色绑定与内容耦合)'}")

    return {
        "role_diff_by_layer_key": {
            f"L{li}": float(np.mean(role_diff_by_layer[li]))
            for li in key_layers if li in role_diff_by_layer
        },
        "role_direction_consistency": float(mean_role_cos),
        "active_passive_cos": float(np.mean(active_passive_diff)),
        "active_reversed_cos": float(np.mean(active_reversed_diff)),
        "top_role_sensitive_heads": [
            {"layer": li, "head": hi, "sensitivity": float(sens)}
            for (li, hi), sens in sorted_heads[:10]
        ],
        "content_role_orthogonality": float(mean_orth),
    }


# ===== Exp3: 注意力路由图 =====
def exp3_routing_topology(model, tokenizer, device, model_name):
    """
    分析不同句型的信息路由模式:
    1. 不同句型的attention graph结构差异
    2. "谁在与谁通信"的模式
    3. 否定/疑问/因果/条件句的特殊路由
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    print(f"\n{'='*70}")
    print(f"Exp3: 注意力路由拓扑 — {model_name}")
    print(f"{'='*70}")

    # 收集所有句子
    all_sentences = []
    for sent_type, sents in ROUTING_SENTENCE_SETS.items():
        all_sentences.extend(sents)
    all_sentences = list(set(all_sentences))
    print(f"  总句数: {len(all_sentences)}")

    # 提取attention patterns
    print(f"  提取attention patterns...")
    attn_dict = extract_attn_patterns(model, tokenizer, all_sentences, n_layers, desc="Exp3")

    # 分析每个句型的路由模式
    routing_results = {}

    for sent_type, sents in ROUTING_SENTENCE_SETS.items():
        print(f"\n  --- {sent_type} ---")

        # 收集该类型的所有注意力模式
        type_attns = []
        for sent in sents:
            if attn_dict.get(sent) is not None:
                type_attns.append(attn_dict[sent])

        if not type_attns:
            print(f"    无数据, 跳过")
            continue

        # 1. 注意力熵 (信息分散程度)
        attn_entropy = []
        for attn in type_attns:  # (n_layers, n_heads, seq_len)
            for li in range(attn.shape[0]):
                for hi in range(attn.shape[1]):
                    p = attn[li, hi]
                    p = p[p > 0]
                    if len(p) > 0:
                        entropy = -np.sum(p * np.log(p + 1e-10))
                        attn_entropy.append(entropy)

        mean_entropy = np.mean(attn_entropy) if attn_entropy else 0
        print(f"    注意力熵: mean={mean_entropy:.4f}")

        # 2. 首token注意力比例 (许多模型倾向于关注首token)
        first_token_attn = []
        for attn in type_attns:
            for li in range(attn.shape[0]):
                for hi in range(attn.shape[1]):
                    first_token_attn.append(attn[li, hi, 0])

        mean_first = np.mean(first_token_attn) if first_token_attn else 0
        print(f"    首token注意力比: mean={mean_first:.4f}")

        # 3. 层间路由差异: 前半层 vs 后半层的注意力模式
        if type_attns[0].shape[0] > 1:
            half = type_attns[0].shape[0] // 2
            early_attn = np.mean([a[:half].mean() for a in type_attns])
            late_attn = np.mean([a[half:].mean() for a in type_attns])
            print(f"    前半层均值: {early_attn:.6f}, 后半层均值: {late_attn:.6f}")
        else:
            early_attn = 0
            late_attn = 0

        routing_results[sent_type] = {
            "mean_attn_entropy": float(mean_entropy),
            "first_token_attn_ratio": float(mean_first),
            "early_layer_attn_mean": float(early_attn),
            "late_layer_attn_mean": float(late_attn),
        }

    # 4. 跨句型路由差异
    print(f"\n  --- 跨句型路由差异 ---")
    type_names = list(ROUTING_SENTENCE_SETS.keys())
    routing_diff = {}
    for t1, t2 in combinations(type_names, 2):
        if t1 in routing_results and t2 in routing_results:
            r1 = routing_results[t1]
            r2 = routing_results[t2]
            entropy_diff = abs(r1["mean_attn_entropy"] - r2["mean_attn_entropy"])
            first_diff = abs(r1["first_token_attn_ratio"] - r2["first_token_attn_ratio"])
            routing_diff[f"{t1}_vs_{t2}"] = {
                "entropy_diff": float(entropy_diff),
                "first_token_diff": float(first_diff),
            }
            print(f"  {t1} vs {t2}: entropy_diff={entropy_diff:.4f}, "
                  f"first_token_diff={first_diff:.4f}")

    return {
        "routing_by_type": routing_results,
        "cross_type_diff": routing_diff,
    }


# ===== Exp4: 跨层状态精化 =====
def exp4_cross_layer_state(model, tokenizer, device, model_name):
    """
    追踪各层如何逐步区分语义功能:
    1. 各层的区分能力 (negation/tense/role_binding)
    2. 区分能力的层间演化曲线
    3. 语义功能"何时被写入"残差流
    """
    info = get_model_info(model, model_name)
    n_layers = info.n_layers

    print(f"\n{'='*70}")
    print(f"Exp4: 跨层状态精化 — {model_name}")
    print(f"{'='*70}")

    # 收集所有句子
    all_sentences = []
    for func_type, pairs in STATE_TRACKING_PAIRS.items():
        for s1, s2 in pairs:
            all_sentences.extend([s1, s2])
    all_sentences = list(set(all_sentences))
    print(f"  总句数: {len(all_sentences)}")

    # 提取hidden states
    print(f"  提取hidden states...")
    hs_dict = extract_hidden_states_all_layers(model, tokenizer, all_sentences, desc="Exp4")

    # 分析各层的区分能力
    discrimination_by_layer = defaultdict(lambda: defaultdict(list))

    for func_type, pairs in STATE_TRACKING_PAIRS.items():
        for s1, s2 in pairs:
            h1 = hs_dict[s1]
            h2 = hs_dict[s2]
            for li in range(n_layers + 1):
                # 区分能力 = 差异方向与随机方向的信噪比
                # 简化: 直接用||h1 - h2||
                diff_norm = np.linalg.norm(h1[li] - h2[li])
                # 也计算cosine差异
                cos_diff = cosine_sim(h1[li], h2[li])
                discrimination_by_layer[func_type][li].append({
                    "diff_norm": diff_norm,
                    "cosine": cos_diff,
                })

    # 打印关键结果
    print(f"\n  --- 各功能在各层的区分能力 ---")
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]

    discrimination_summary = {}
    for func_type in STATE_TRACKING_PAIRS.keys():
        print(f"\n  {func_type}:")
        func_data = {}
        for li in key_layers:
            data = discrimination_by_layer[func_type].get(li, [])
            if data:
                mean_diff = np.mean([d["diff_norm"] for d in data])
                mean_cos = np.mean([d["cosine"] for d in data])
                print(f"    L{li}: ||Δh||={mean_diff:.4f}, cos={mean_cos:.4f}")
                func_data[f"L{li}"] = {
                    "mean_diff_norm": float(mean_diff),
                    "mean_cosine": float(mean_cos),
                }
        discrimination_summary[func_type] = func_data

    # 找到各功能"最显著区分"的层
    print(f"\n  --- 各功能最显著区分的层 ---")
    peak_layers = {}
    for func_type in STATE_TRACKING_PAIRS.keys():
        max_diff = 0
        peak_li = 0
        for li in range(n_layers + 1):
            data = discrimination_by_layer[func_type].get(li, [])
            if data:
                mean_diff = np.mean([d["diff_norm"] for d in data])
                if mean_diff > max_diff:
                    max_diff = mean_diff
                    peak_li = li
        peak_layers[func_type] = {"peak_layer": peak_li, "peak_diff": float(max_diff)}
        print(f"  {func_type}: peak at L{peak_li} (||Δh||={max_diff:.4f})")

    # 各功能的区分曲线形状 (递增? 钟形? 阶梯?)
    print(f"\n  --- 区分曲线形状 ---")
    curve_shapes = {}
    for func_type in STATE_TRACKING_PAIRS.keys():
        diffs = []
        for li in range(n_layers + 1):
            data = discrimination_by_layer[func_type].get(li, [])
            if data:
                diffs.append(np.mean([d["diff_norm"] for d in data]))
            else:
                diffs.append(0)

        # 简单分类: 递增/递减/钟形/阶梯
        if len(diffs) > 2:
            early = np.mean(diffs[:len(diffs)//3])
            mid = np.mean(diffs[len(diffs)//3:2*len(diffs)//3])
            late = np.mean(diffs[2*len(diffs)//3:])

            if late > mid > early:
                shape = "monotone_increasing"
            elif early > mid > late:
                shape = "monotone_decreasing"
            elif mid > early and mid > late:
                shape = "bell_shaped"
            elif late > early and late > mid:
                shape = "late_rising"
            else:
                shape = "other"

            curve_shapes[func_type] = {
                "shape": shape,
                "early_mean": float(early),
                "mid_mean": float(mid),
                "late_mean": float(late),
            }
            print(f"  {func_type}: {shape} (early={early:.4f}, mid={mid:.4f}, late={late:.4f})")

    return {
        "discrimination_summary": discrimination_summary,
        "peak_layers": peak_layers,
        "curve_shapes": curve_shapes,
    }


# ===== 主函数 =====
def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}, available: {list(MODEL_CONFIGS.keys())}")
        return

    t_start = time.time()
    print(f"\n{'='*70}")
    print(f"Phase 191: 回路代数与角色绑定 — {model_name}")
    print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # 加载模型
    model, tokenizer, device = load_model_bf16_auto(model_name)
    info = get_model_info(model, model_name)
    print(f"  n_layers={info.n_layers}, d_model={info.d_model}, vocab={info.vocab_size}")

    # 运行所有实验
    results = {
        "model": model_name,
        "model_info": {
            "n_layers": info.n_layers,
            "d_model": info.d_model,
            "vocab_size": info.vocab_size,
            "model_class": info.model_class,
        },
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    # Exp1: 算子代数
    print(f"\n{'='*70}")
    print(f"运行 Exp1: 算子代数...")
    t1 = time.time()
    results["exp1"] = exp1_operator_algebra(model, tokenizer, device, model_name)
    print(f"  Exp1 完成: {time.time()-t1:.1f}s")

    # Exp2: 角色绑定
    print(f"\n{'='*70}")
    print(f"运行 Exp2: 角色绑定...")
    t2 = time.time()
    results["exp2"] = exp2_role_binding(model, tokenizer, device, model_name)
    print(f"  Exp2 完成: {time.time()-t2:.1f}s")

    # Exp3: 路由拓扑
    print(f"\n{'='*70}")
    print(f"运行 Exp3: 路由拓扑...")
    t3 = time.time()
    results["exp3"] = exp3_routing_topology(model, tokenizer, device, model_name)
    print(f"  Exp3 完成: {time.time()-t3:.1f}s")

    # Exp4: 跨层状态
    print(f"\n{'='*70}")
    print(f"运行 Exp4: 跨层状态精化...")
    t4 = time.time()
    results["exp4"] = exp4_cross_layer_state(model, tokenizer, device, model_name)
    print(f"  Exp4 完成: {time.time()-t4:.1f}s")

    # 保存结果
    t_stamp = time.strftime('%Y%m%d_%H%M')
    out_path = f"tests/glm5_temp/phase191_{model_name}_{t_stamp}.json"
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n结果保存到: {out_path}")

    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Phase 191 COMPLETE — {model_name}")
    print(f"总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"结束时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
