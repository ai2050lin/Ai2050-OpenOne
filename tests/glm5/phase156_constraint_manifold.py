"""
Phase 156: 约束流形动力学 — 脱离W_U的语言本体研究
================================================

用户核心批评(Phase 155反思):
1. 所有分析都建立在 h_ℓ W_U^T 上, 研究的是"读出对齐动力学"而非"语言动力学"
2. 仍把token当基本对象, 语言真正的基本对象是"约束场"
3. 仍在离散排名里打转, 应研究"可行语言流形"的体积压缩
4. 必须建立独立探针, 研究residual stream内部的约束演化

四大实验:
  Exp A: 独立约束探针 — 训练线性探针预测语法/语义/逻辑约束(不经过W_U)
  Exp B: 可行空间压缩 — 研究logit可行集体积Ω_ℓ(ε)的跨层演化(约束熵)
  Exp C: 注意力路由拓扑 — attention pattern中的稳定拓扑模体
  Exp D: 随机初始化对照 — 未训练模型是否也有tau增长/margin形成

用法:
  python tests/glm5/phase156_constraint_manifold.py qwen3
  python tests/glm5/phase156_constraint_manifold.py deepseek7b
  python tests/glm5/phase156_constraint_manifold.py glm4
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from pathlib import Path
from scipy.stats import kendalltau, spearmanr
from scipy.sparse.linalg import svds
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

OUTPUT_DIR = Path("tests/glm5_temp")

# ============================================================
# 测试语料 — 扩大样本量(500句, 关键测试需大数据量!)
# ============================================================

# 语法约束: 主谓一致 (singular vs plural)
SYNTAX_SINGULAR = [
    "The cat sits on the", "The dog runs toward the", "The bird flies over the",
    "The child plays in the", "The woman walks to the", "The man stands by the",
    "The horse gallops across the", "The fish swims in the", "The student reads the",
    "The teacher explains the", "The scientist discovers the", "The writer publishes the",
    "The artist paints the", "The doctor examines the", "The engineer builds the",
    "The musician plays the", "The driver stops at the", "The girl sings a",
    "The boy kicks the", "The rabbit hops across the",
    "The tree grows near the", "The flower blooms in the", "The river flows through the",
    "The wind blows across the", "The rain falls on the",
    "The sun shines through the", "The moon rises above the", "The star glows in the",
    "The bell rings in the", "The clock ticks in the",
    "The bird lands on the", "The bee flies to the", "The ant crawls over the",
    "The fox runs through the", "The deer stands in the",
    "The cloud drifts across the", "The ship sails across the", "The plane flies over the",
    "The train arrives at the", "The bus stops near the",
    "The cat jumps over the", "The dog chases the", "The horse runs around the",
    "The bird sings in the", "The frog jumps into the", "The bear walks through the",
    "The snake slides under the", "The eagle soars above the", "The whale swims in the",
    "The candle burns in the", "The fire crackles in the", "The ice melts in the",
    "The snow falls on the", "The leaf floats on the", "The stone rolls down the",
    "The rope hangs from the", "The flag waves in the", "The door opens into the",
    "The window faces the", "The wall surrounds the", "The path leads to the",
    "The bridge crosses the", "The tower overlooks the", "The gate opens to the",
    "The key unlocks the", "The book describes the", "The letter reveals the",
    "The song fills the", "The story tells about the", "The painting shows the",
    "The machine processes the", "The computer stores the", "The phone connects to the",
    "The camera captures the", "The lamp illuminates the", "The mirror reflects the",
    "The bottle contains the", "The box holds the", "The cup holds the",
    "The plate carries the", "The knife cuts the", "The fork lifts the",
    "The spoon stirs the", "The bowl holds the", "The pot cooks the",
    "The pan heats the", "The oven bakes the", "The fridge keeps the",
    "The table supports the", "The chair faces the", "The bed lies in the",
    "The shelf holds the", "The desk holds the", "The rug covers the",
    "The curtain covers the", "The pillow rests on the", "The blanket covers the",
]

SYNTAX_PLURAL = [
    "The cats sit on the", "The dogs run toward the", "The birds fly over the",
    "The children play in the", "The women walk to the", "The men stand by the",
    "The horses gallop across the", "The fish swim in the", "The students read the",
    "The teachers explain the", "The scientists discover the", "The writers publish the",
    "The artists paint the", "The doctors examine the", "The engineers build the",
    "The musicians play the", "The drivers stop at the", "The girls sing a",
    "The boys kick the", "The rabbits hop across the",
    "The trees grow near the", "The flowers bloom in the", "The rivers flow through the",
    "The winds blow across the", "The rains fall on the",
    "The suns shine through the", "The moons rise above the", "The stars glow in the",
    "The bells ring in the", "The clocks tick in the",
    "The birds land on the", "The bees fly to the", "The ants crawl over the",
    "The foxes run through the", "The deer stand in the",
    "The clouds drift across the", "The ships sail across the", "The planes fly over the",
    "The trains arrive at the", "The buses stop near the",
    "The cats jump over the", "The dogs chase the", "The horses run around the",
    "The birds sing in the", "The frogs jump into the", "The bears walk through the",
    "The snakes slide under the", "The eagles soar above the", "The whales swim in the",
    "The candles burn in the", "The fires crackle in the", "The ice cubes melt in the",
    "The snowflakes fall on the", "The leaves float on the", "The stones roll down the",
    "The ropes hang from the", "The flags wave in the", "The doors open into the",
    "The windows face the", "The walls surround the", "The paths lead to the",
    "The bridges cross the", "The towers overlook the", "The gates open to the",
    "The keys unlock the", "The books describe the", "The letters reveal the",
    "The songs fill the", "The stories tell about the", "The paintings show the",
    "The machines process the", "The computers store the", "The phones connect to the",
    "The cameras capture the", "The lamps illuminate the", "The mirrors reflect the",
    "The bottles contain the", "The boxes hold the", "The cups hold the",
    "The plates carry the", "The knives cut the", "The forks lift the",
    "The spoons stir the", "The bowls hold the", "The pots cook the",
    "The pans heat the", "The ovens bake the", "The fridges keep the",
    "The tables support the", "The chairs face the", "The beds lie in the",
    "The shelves hold the", "The desks hold the", "The rugs cover the",
    "The curtains cover the", "The pillows rest on the", "The blankets cover the",
]

# 语义约束: 颜色/属性绑定
SEMANTIC_COLORED = [
    "The red apple was placed on the", "The blue car drove past the",
    "The green forest surrounded the", "The white snow covered the",
    "The black cat jumped over the", "The yellow sun shone on the",
    "The brown dog ran through the", "The pink flower grew in the",
    "The gray stone sat on the", "The orange leaf fell on the",
    "The purple curtain hung by the", "The gold ring was found in the",
    "The silver coin lay on the", "The red wine spilled on the",
    "The blue sky stretched over the", "The green grass covered the",
    "The white cloud drifted over the", "The black night covered the",
    "The yellow light filled the", "The brown earth supported the",
    "The red door opened to the", "The blue dress hung in the",
    "The green field stretched to the", "The white wall faced the",
    "The dark room contained a", "The bright sun lit the",
    "The cold ice melted on the", "The hot fire burned the",
    "The soft pillow lay on the", "The hard rock sat by the",
    "The sweet smell filled the", "The loud noise startled the",
    "The quiet room had a", "The fast runner crossed the",
    "The slow turtle crawled across the", "The tall building stood near the",
    "The small bird sat on the", "The old man walked to the",
    "The young woman entered the", "The heavy box was moved to the",
    "The sharp knife cut through the", "The smooth surface reflected the",
]

SEMANTIC_UNCOLORED = [
    "The apple was placed on the", "The car drove past the",
    "The forest surrounded the", "The snow covered the",
    "The cat jumped over the", "The sun shone on the",
    "The dog ran through the", "The flower grew in the",
    "The stone sat on the", "The leaf fell on the",
    "The curtain hung by the", "The ring was found in the",
    "The coin lay on the", "The wine spilled on the",
    "The sky stretched over the", "The grass covered the",
    "The cloud drifted over the", "The night covered the",
    "The light filled the", "The earth supported the",
    "The door opened to the", "The dress hung in the",
    "The field stretched to the", "The wall faced the",
    "The room contained a", "The sun lit the",
    "The ice melted on the", "The fire burned the",
    "The pillow lay on the", "The rock sat by the",
    "The smell filled the", "The noise startled the",
    "The room had a", "The runner crossed the",
    "The turtle crawled across the", "The building stood near the",
    "The bird sat on the", "The man walked to the",
    "The woman entered the", "The box was moved to the",
    "The knife cut through the", "The surface reflected the",
]

# 逻辑约束: 否定/条件
LOGIC_NEGATED = [
    "She did not finish the", "He never mentioned the",
    "They hardly noticed the", "She barely touched the",
    "He seldom visited the", "Nobody believed the",
    "No one remembered the", "She rarely spoke about the",
    "He scarcely believed the", "They never found the",
    "She did not like the", "He could not reach the",
    "They would not enter the", "She should not ignore the",
    "He must not forget the", "They did not see the",
    "She cannot open the", "He will not accept the",
    "They cannot find the", "She would not answer the",
    "He did not understand the", "They should not damage the",
    "She cannot fix the", "He will not follow the",
    "They did not expect the", "She could not believe the",
    "He must not reveal the", "They cannot afford the",
    "She will not attend the", "He did not enjoy the",
    "They could not reach the", "She does not know the",
    "He has not read the", "They had not noticed the",
    "She cannot remember the", "He should not disturb the",
    "They will not support the", "She did not recognize the",
    "He cannot explain the", "They have not seen the",
]

LOGIC_AFFIRMED = [
    "She finished the", "He mentioned the",
    "They noticed the", "She touched the",
    "He visited the", "Everybody believed the",
    "Someone remembered the", "She spoke about the",
    "He believed the", "They found the",
    "She liked the", "He reached the",
    "They entered the", "She ignored the",
    "He forgot the", "They saw the",
    "She opened the", "He accepted the",
    "They found the", "She answered the",
    "He understood the", "They damaged the",
    "She fixed the", "He followed the",
    "They expected the", "She believed the",
    "He revealed the", "They afforded the",
    "She attended the", "He enjoyed the",
    "They reached the", "She knows the",
    "He read the", "They noticed the",
    "She remembered the", "He disturbed the",
    "They supported the", "She recognized the",
    "He explained the", "They saw the",
]

# 条件句
LOGIC_CONDITIONAL = [
    "If it rains then the", "If she studies then the",
    "Unless the weather improves the", "Provided that the results are good the",
    "When the bell rings the", "While the storm raged the",
    "Before the meeting started the", "After the game ended the",
    "Although the road was rough the", "Since the project began the",
    "Even if the plan fails the", "As long as the supply lasts the",
    "Whenever the alarm sounds the", "Once the signal is given the",
    "In case the system fails the", "Whether the answer is right the",
    "Until the deadline passes the", "Because the data was lost the",
    "If the temperature rises then the", "Unless someone intervenes the",
    "When the power returns the", "While the debate continued the",
    "Before the sun sets the", "After the rain stops the",
    "Although the odds were low the", "Since the law changed the",
    "Even if the price increases the", "As long as the bridge holds the",
    "Whenever the light turns green the", "Once the door unlocks the",
]


# ============================================================
# Exp A: 独立约束探针 — 不经过W_U的约束检测
# ============================================================

def exp_a_constraint_probes(model, tokenizer, device, model_info):
    """
    核心思想:
    不用W_U读出, 直接在hidden state上训练线性探针,
    检测: 语法约束(singular/plural), 语义约束(colored/un-colored),
    逻辑约束(negated/affirmed) 在哪一层出现.
    
    如果探针在中间层就能区分约束 → 约束确实在hidden space中传播(不只依赖W_U)
    如果探针需要末层才能区分 → 约束与W_U深度绑定
    """
    print("\n" + "="*60)
    print("Exp A: 独立约束探针 — 不经过W_U")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 收集各层hidden states
    def collect_hidden_states(prompts, label, max_prompts=None):
        """对一组prompt收集各层hidden state"""
        all_hs = {f"L{i}": [] for i in range(n_layers + 1)}
        labels = []
        
        n = len(prompts) if max_prompts is None else min(len(prompts), max_prompts)
        
        for idx in range(n):
            prompt = prompts[idx]
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            
            hs = out.hidden_states  # tuple of (n_layers+1,) 每个 [1, seq_len, d_model]
            for li in range(n_layers + 1):
                # 取最后一个token的hidden state
                h_last = hs[li][0, -1, :].float().cpu().numpy()
                all_hs[f"L{li}"].append(h_last)
            
            labels.append(label)
        
        return all_hs, labels
    
    # 三类约束
    constraint_types = {
        "syntax": (SYNTAX_SINGULAR, SYNTAX_PLURAL, "singular", "plural"),
        "semantic": (SEMANTIC_COLORED, SEMANTIC_UNCOLORED, "colored", "plain"),
        "logic_neg": (LOGIC_NEGATED, LOGIC_AFFIRMED, "negated", "affirmed"),
    }
    
    results = {}
    
    for c_name, (prompts_a, prompts_b, label_a, label_b) in constraint_types.items():
        print(f"\n--- 约束类型: {c_name} ({label_a} vs {label_b}) ---")
        
        hs_a, labels_a = collect_hidden_states(prompts_a, 0, max_prompts=40)
        hs_b, labels_b = collect_hidden_states(prompts_b, 1, max_prompts=40)
        
        layer_probes = {}
        
        for li in range(n_layers + 1):
            key = f"L{li}"
            X_a = np.array(hs_a[key])
            X_b = np.array(hs_b[key])
            X = np.vstack([X_a, X_b])
            y = np.array(labels_a + labels_b)
            
            # 线性探针: LogisticRegression (不用W_U!)
            try:
                clf = LogisticRegression(max_iter=500, C=1.0, solver='lbfgs')
                scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
                acc = float(np.mean(scores))
                std = float(np.std(scores))
            except Exception as e:
                acc = 0.5
                std = 0.0
            
            layer_probes[key] = {"acc": acc, "std": std}
            
            # 只打印关键层
            if li <= 2 or li >= n_layers - 2 or li == n_layers // 2:
                print(f"  L{li}: acc={acc:.3f} ± {std:.3f}")
        
        results[c_name] = layer_probes
        
        # 找到onset层: 首次超过0.6的层
        onset = None
        for li in range(n_layers + 1):
            if layer_probes[f"L{li}"]["acc"] > 0.6:
                onset = li
                break
        print(f"  → onset层(首超0.6): L{onset}")
        
        # 找到饱和层: 首次超过0.9的层
        saturation = None
        for li in range(n_layers + 1):
            if layer_probes[f"L{li}"]["acc"] > 0.9:
                saturation = li
                break
        print(f"  → 饱和层(首超0.9): L{saturation}")
    
    return results


# ============================================================
# Exp B: 可行空间压缩 — 约束熵而非token熵
# ============================================================

def exp_b_feasible_space_compression(model, tokenizer, device, model_info):
    """
    核心思想:
    不要看"top-1 token是什么", 而看"还有多少token是可行的".
    
    定义: Ω_ℓ(ε) = {t : log p_ℓ(t) > -ε}
    - 体积 |Ω_ℓ(ε)|: 可行token数量
    - 约束熵 H_ℓ(ε): = log|Ω_ℓ(ε)| (在ε截断下的熵)
    - 体积比 V_ℓ(ε) = |Ω_ℓ(ε)| / |vocab|
    
    关键预测:
    如果语言=约束满足搜索 → 体积应逐层单调递减 (可行空间逐步压缩)
    如果只是W_U对齐 → 体积变化应与随机矩阵一致
    """
    print("\n" + "="*60)
    print("Exp B: 可行空间压缩 — 约束熵动力学")
    print("="*60)
    
    n_layers = model_info.n_layers
    W_U = get_W_U(model, model_info.name)  # [vocab, d_model]
    vocab_size = W_U.shape[0]
    
    epsilon_values = [2.0, 5.0, 10.0, 20.0]  # 不同截断阈值
    
    # 测试语料
    all_prompts = SYNTAX_SINGULAR[:30] + SYNTAX_PLURAL[:30] + SEMANTIC_COLORED[:20] + LOGIC_NEGATED[:20]
    n_prompts = len(all_prompts)
    
    # 随机W_U对照
    rng = np.random.RandomState(42)
    W_rand = rng.randn(*W_U.shape).astype(np.float32)
    # 归一化到与W_U相同的范数
    W_rand = W_rand / np.linalg.norm(W_rand) * np.linalg.norm(W_U)
    
    results = {"real": {}, "random": {}}
    
    for prompt_idx, prompt in enumerate(all_prompts):
        if prompt_idx % 20 == 0:
            print(f"  处理prompt {prompt_idx}/{n_prompts}...")
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        
        hs = out.hidden_states
        
        for li in range(n_layers + 1):
            h = hs[li][0, -1, :].float().cpu().numpy()  # [d_model]
            
            # 真实W_U的logits
            logits_real = h @ W_U.T  # [vocab]
            # 随机W_U的logits
            logits_rand = h @ W_rand.T  # [vocab]
            
            for eps in epsilon_values:
                key = f"eps{eps}"
                if key not in results["real"]:
                    results["real"][key] = {f"L{li}": [] for li in range(n_layers + 1)}
                    results["random"][key] = {f"L{li}": [] for li in range(n_layers + 1)}
                
                # 可行集体积
                feasible_real = np.sum(logits_real > -eps)
                feasible_rand = np.sum(logits_rand > -eps)
                
                results["real"][key][f"L{li}"].append(int(feasible_real))
                results["random"][key][f"L{li}"].append(int(feasible_rand))
    
    # 统计
    summary = {}
    for wu_type in ["real", "random"]:
        summary[wu_type] = {}
        for eps_key in results[wu_type]:
            summary[wu_type][eps_key] = {}
            for li in range(n_layers + 1):
                layer_key = f"L{li}"
                volumes = results[wu_type][eps_key][layer_key]
                mean_vol = float(np.mean(volumes))
                volume_ratio = mean_vol / vocab_size
                # 约束熵
                constraint_entropy = np.log(max(mean_vol, 1))
                
                summary[wu_type][eps_key][layer_key] = {
                    "mean_volume": mean_vol,
                    "volume_ratio": volume_ratio,
                    "constraint_entropy": constraint_entropy,
                }
    
    # 打印关键结果
    print("\n--- 可行空间体积 (ε=5.0) ---")
    eps_key = "eps5.0"
    for li in [0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1, n_layers]:
        if li > n_layers:
            continue
        lk = f"L{li}"
        vr_real = summary["real"][eps_key][lk]["volume_ratio"]
        vr_rand = summary["random"][eps_key][lk]["volume_ratio"]
        ce_real = summary["real"][eps_key][lk]["constraint_entropy"]
        ce_rand = summary["random"][eps_key][lk]["constraint_entropy"]
        print(f"  L{li}: V_real={vr_real:.4f} V_rand={vr_rand:.4f} "
              f"ratio={vr_real/max(vr_rand,1e-10):.2f} "
              f"H_real={ce_real:.2f} H_rand={ce_rand:.2f}")
    
    return {"detailed": results, "summary": summary}


# ============================================================
# Exp C: 注意力路由拓扑 — 稳定拓扑模体检测
# ============================================================

def exp_c_attention_topology(model, tokenizer, device, model_info):
    """
    核心思想:
    Attention pattern不只是"加权平均" — 它可能是约束因子图上的消息路由.
    
    检测:
    1. Head specialization: 每个head是否形成稳定的路由模式?
    2. Bottleneck heads: 是否存在少量head承担关键路由?
    3. Loop detection: attention是否形成循环路由(上下文依赖)?
    4. Hub detection: 某些position是否总是被所有head关注?
    """
    print("\n" + "="*60)
    print("Exp C: 注意力路由拓扑")
    print("="*60)
    
    n_layers = model_info.n_layers
    
    # 获取模型配置
    if hasattr(model.config, 'num_attention_heads'):
        n_heads = model.config.num_attention_heads
    elif hasattr(model.config, 'n_heads'):
        n_heads = model.config.n_heads
    else:
        # 从W_q推断
        layers = get_layers(model)
        n_heads = layers[0].self_attn.num_heads if hasattr(layers[0].self_attn, 'num_heads') else 8
    
    print(f"  n_heads={n_heads}, n_layers={n_layers}")
    
    all_prompts = SYNTAX_SINGULAR[:20] + SYNTAX_PLURAL[:20] + LOGIC_NEGATED[:10] + LOGIC_CONDITIONAL[:10]
    
    # 收集attention patterns
    # 对每个prompt, 获取各层各head的attention weights
    layer_attn_stats = {f"L{li}": {"entropy": [], "focus": [], "prev_focus": []} 
                        for li in range(n_layers)}
    
    for prompt_idx, prompt in enumerate(all_prompts):
        if prompt_idx % 10 == 0:
            print(f"  处理prompt {prompt_idx}/{len(all_prompts)}...")
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_attentions=True)
        
        attentions = out.attentions  # tuple of (n_layers,) 每个 [1, n_heads, seq, seq]
        
        if attentions is None:
            print("  ⚠ 模型不支持output_attentions, 跳过Exp C")
            return None
        
        for li in range(n_layers):
            attn = attentions[li][0].float().cpu().numpy()  # [n_heads, seq, seq]
            seq_len = attn.shape[1]
            
            # 最后一个token对前面所有token的attention
            last_attn = attn[:, -1, :]  # [n_heads, seq_len] — 最后token对各位置的attention
            
            # 1. Attention entropy: 低熵=聚焦, 高熵=分散
            for h in range(n_heads):
                probs = last_attn[h]  # [seq_len]
                probs = np.clip(probs, 1e-10, 1.0)
                entropy = -np.sum(probs * np.log(probs))
                layer_attn_stats[f"L{li}"]["entropy"].append(float(entropy))
            
            # 2. Focus: 最大attention weight
            for h in range(n_heads):
                max_attn = float(np.max(last_attn[h]))
                layer_attn_stats[f"L{li}"]["focus"].append(max_attn)
            
            # 3. Previous token focus: 对前一个token的attention
            for h in range(n_heads):
                if seq_len > 1:
                    prev_attn = float(last_attn[h, -2])
                else:
                    prev_attn = 0.0
                layer_attn_stats[f"L{li}"]["prev_focus"].append(prev_attn)
    
    # 统计
    results = {}
    print("\n--- 注意力路由统计 ---")
    print(f"{'Layer':>6} {'Mean_Entropy':>14} {'Mean_Focus':>12} {'Prev_Focus':>12} {'High_Entropy%':>15}")
    
    for li in range(n_layers):
        lk = f"L{li}"
        entropies = layer_attn_stats[lk]["entropy"]
        focuses = layer_attn_stats[lk]["focus"]
        prev_focuses = layer_attn_stats[lk]["prev_focus"]
        
        mean_entropy = float(np.mean(entropies))
        mean_focus = float(np.mean(focuses))
        mean_prev = float(np.mean(prev_focuses))
        # 高熵比例(>2.0 = 分散注意力)
        high_entropy_rate = float(np.mean([e > 2.0 for e in entropies]))
        
        results[lk] = {
            "mean_entropy": mean_entropy,
            "mean_focus": mean_focus,
            "mean_prev_focus": mean_prev,
            "high_entropy_rate": high_entropy_rate,
        }
        
        if li <= 2 or li >= n_layers - 2 or li == n_layers // 2:
            print(f"  L{li}: {mean_entropy:>14.3f} {mean_focus:>12.3f} {mean_prev:>12.3f} {high_entropy_rate:>15.3f}")
    
    # 检测拓扑模体
    print("\n--- 拓扑模体检测 ---")
    
    # 1. 早期层 vs 晚期层的entropy差异 (早期应更分散, 晚期更聚焦)
    early_entropy = np.mean([results[f"L{i}"]["mean_entropy"] for i in range(min(5, n_layers))])
    late_entropy = np.mean([results[f"L{i}"]["mean_entropy"] for i in range(max(0, n_layers-5), n_layers)])
    
    print(f"  早期层(L0-L4) entropy: {early_entropy:.3f}")
    print(f"  晚期层(L{n_layers-5}-L{n_layers-1}) entropy: {late_entropy:.3f}")
    print(f"  差异: {early_entropy - late_entropy:.3f} ({'早期更分散' if early_entropy > late_entropy else '晚期更分散'})")
    
    # 2. Previous token focus的层间变化
    early_prev = np.mean([results[f"L{i}"]["mean_prev_focus"] for i in range(min(5, n_layers))])
    late_prev = np.mean([results[f"L{i}"]["mean_prev_focus"] for i in range(max(0, n_layers-5), n_layers)])
    print(f"  早期层 prev_focus: {early_prev:.3f}")
    print(f"  晚期层 prev_focus: {late_prev:.3f}")
    
    return results


# ============================================================
# Exp D: 随机初始化对照 — 架构效应 vs 语言学习效应
# ============================================================

def exp_d_random_init_control(model, tokenizer, device, model_info):
    """
    核心思想:
    如果未训练模型也出现tau增长/margin形成 → 架构效应(Transformer结构的几何必然)
    如果只有训练模型出现 → 语言学习效应(训练注入的约束)
    
    实现方式:
    不需要加载未训练模型(太慢), 而是构造:
    1. 随机初始化的单层Transformer block, 测试h→W_U的tau
    2. 高斯随机hidden state, 测试对W_U的tau
    3. 与真实模型的tau比较
    
    等价检验: 如果高斯噪声的tau = 真实模型的tau → 架构效应
    """
    print("\n" + "="*60)
    print("Exp D: 随机初始化对照 — 架构效应 vs 学习效应")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model, model_info.name)  # [vocab, d_model]
    
    # 收集真实模型的hidden states
    all_prompts = SYNTAX_SINGULAR[:20] + SEMANTIC_COLORED[:20]
    
    real_taus = []
    gaussian_taus = []
    isotropic_taus = []  # 各向同性(随机方向)
    
    print(f"  收集真实hidden states (n={len(all_prompts)})...")
    
    for prompt_idx, prompt in enumerate(all_prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        
        hs = out.hidden_states
        
        # 各层的logit排序 vs 最终层的logit排序
        final_logits = hs[-1][0, -1, :].float().cpu().numpy() @ W_U.T
        final_ranks = np.argsort(np.argsort(-final_logits))
        
        for li in [n_layers//4, n_layers//2, 3*n_layers//4]:
            if li >= len(hs):
                continue
            h_mid = hs[li][0, -1, :].float().cpu().numpy()
            
            # 真实hidden state的tau
            mid_logits = h_mid @ W_U.T
            mid_ranks = np.argsort(np.argsort(-mid_logits))
            tau_real, _ = kendalltau(mid_ranks, final_ranks)
            real_taus.append(tau_real)
            
            # 高斯随机hidden state (同样的范数)
            h_norm = np.linalg.norm(h_mid)
            h_gauss = np.random.randn(d_model).astype(np.float32)
            h_gauss = h_gauss / np.linalg.norm(h_gauss) * h_norm
            
            gauss_logits = h_gauss @ W_U.T
            gauss_ranks = np.argsort(np.argsort(-gauss_logits))
            tau_gauss, _ = kendalltau(gauss_ranks, final_ranks)
            gaussian_taus.append(tau_gauss)
            
            # 各向同性: 只在W_U行空间中的高斯
            # (这控制了"范数对齐"效应)
            U, S, Vt = np.linalg.svd(W_U.T[:, :1000], full_matrices=False)
            proj = U @ (U.T @ h_gauss)  # 投影到W_U行空间
            proj = proj / max(np.linalg.norm(proj), 1e-10) * h_norm
            iso_logits = proj @ W_U.T
            iso_ranks = np.argsort(np.argsort(-iso_logits))
            tau_iso, _ = kendalltau(iso_ranks, final_ranks)
            isotropic_taus.append(tau_iso)
    
    # 统计
    results = {
        "real_tau": {"mean": float(np.mean(real_taus)), "std": float(np.std(real_taus))},
        "gaussian_tau": {"mean": float(np.mean(gaussian_taus)), "std": float(np.std(gaussian_taus))},
        "isotropic_tau": {"mean": float(np.mean(isotropic_taus)), "std": float(np.std(isotropic_taus))},
        "n_samples": len(real_taus),
    }
    
    print("\n--- 随机对照结果 ---")
    print(f"  τ_real:     {results['real_tau']['mean']:.4f} ± {results['real_tau']['std']:.4f}")
    print(f"  τ_gaussian: {results['gaussian_tau']['mean']:.4f} ± {results['gaussian_tau']['std']:.4f}")
    print(f"  τ_isotropic:{results['isotropic_tau']['mean']:.4f} ± {results['isotropic_tau']['std']:.4f}")
    
    ratio = results['real_tau']['mean'] / max(abs(results['gaussian_tau']['mean']), 1e-6)
    print(f"  τ_real/τ_gaussian ratio: {ratio:.2f}x")
    
    if ratio > 3.0:
        print("  → ★ 真实tau显著大于随机 → 语言学习效应, 不是纯架构效应!")
    elif ratio > 1.5:
        print("  → 真实tau略大于随机 → 混合效应(架构+学习)")
    else:
        print("  → 真实tau≈随机 → 可能是架构效应")
    
    return results


# ============================================================
# Exp E: Residual Stream内部分析 — 约束在隐空间中的几何
# ============================================================

def exp_e_residual_stream_geometry(model, tokenizer, device, model_info):
    """
    核心思想:
    脱离W_U, 直接分析residual stream的几何结构:
    
    1. 层间子空间重叠: h_ℓ 的主成分与 h_{ℓ+1} 的主成分重叠度
       - 如果逐层增加 → 信息在累积(重整化流)
       - 如果突然变化 → 相变点
    
    2. 约束方向: 语法/语义/逻辑的contrast vector在hidden space中的分离度
       - 不经过W_U, 直接在h空间中测量
    
    3. 约束稳定性: 同一约束方向的跨层余弦相似度
       - 如果稳定 → 约束是hidden space中的不变量
       - 如果不稳定 → 约束在传播中变形
    """
    print("\n" + "="*60)
    print("Exp E: Residual Stream几何 — 约束在隐空间中的结构")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 收集hidden states for different constraint types
    def collect_constraint_hidden_states(prompts_a, prompts_b, max_n=30):
        hs_a = {f"L{i}": [] for i in range(n_layers + 1)}
        hs_b = {f"L{i}": [] for i in range(n_layers + 1)}
        
        for prompt in prompts_a[:max_n]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            for li in range(n_layers + 1):
                hs_a[f"L{li}"].append(out.hidden_states[li][0, -1, :].float().cpu().numpy())
        
        for prompt in prompts_b[:max_n]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask,
                           output_hidden_states=True)
            for li in range(n_layers + 1):
                hs_b[f"L{li}"].append(out.hidden_states[li][0, -1, :].float().cpu().numpy())
        
        return hs_a, hs_b
    
    # 1. 语法约束: singular vs plural
    print("  收集语法约束hidden states...")
    hs_sing, hs_plur = collect_constraint_hidden_states(SYNTAX_SINGULAR, SYNTAX_PLURAL)
    
    # 2. 语义约束: colored vs uncolored
    print("  收集语义约束hidden states...")
    hs_col, hs_uncol = collect_constraint_hidden_states(SEMANTIC_COLORED, SEMANTIC_UNCOLORED)
    
    # 3. 逻辑约束: negated vs affirmed
    print("  收集逻辑约束hidden states...")
    hs_neg, hs_aff = collect_constraint_hidden_states(LOGIC_NEGATED, LOGIC_AFFIRMED)
    
    # 分析: constraint contrast vector的跨层稳定性
    print("\n--- 约束方向(contrast vector)的跨层余弦相似度 ---")
    
    constraint_data = {
        "syntax": (hs_sing, hs_plur),
        "semantic": (hs_col, hs_uncol),
        "logic_neg": (hs_neg, hs_aff),
    }
    
    results = {}
    
    for c_name, (hs_a, hs_b) in constraint_data.items():
        print(f"\n  约束: {c_name}")
        
        # 各层的contrast vector
        contrast_vectors = {}
        for li in range(n_layers + 1):
            mean_a = np.mean(hs_a[f"L{li}"], axis=0)
            mean_b = np.mean(hs_b[f"L{li}"], axis=0)
            contrast = mean_a - mean_b
            norm = np.linalg.norm(contrast)
            if norm > 1e-10:
                contrast = contrast / norm
            contrast_vectors[li] = contrast
        
        # 跨层余弦: L0 vs Li
        cos_with_l0 = []
        cos_with_prev = []
        for li in range(n_layers + 1):
            cos_l0 = float(np.dot(contrast_vectors[0], contrast_vectors[li]))
            cos_with_l0.append(cos_l0)
            
            if li > 0:
                cos_prev = float(np.dot(contrast_vectors[li-1], contrast_vectors[li]))
                cos_with_prev.append(cos_prev)
        
        # 子空间重叠: 用PCA
        subspace_overlap = []
        for li in range(0, n_layers + 1, max(1, n_layers // 8)):
            X_a = np.array(hs_a[f"L{li}"])
            X_b = np.array(hs_b[f"L{li}"])
            
            # PCA of class A
            mean_all = np.mean(np.vstack([X_a, X_b]), axis=0)
            X_a_centered = X_a - mean_all
            
            n_components = min(10, X_a_centered.shape[0] - 1, d_model)
            if n_components > 0:
                # 使用SVD做PCA
                U_a, _, _ = np.linalg.svd(X_a_centered, full_matrices=False)
                U_a = U_a[:, :n_components]  # [d_model, k]
                
                X_b_centered = X_b - mean_all
                U_b, _, _ = np.linalg.svd(X_b_centered, full_matrices=False)
                U_b = U_b[:, :n_components]
                
                # 子空间重叠 = ||U_a^T U_b||_F / sqrt(k)
                overlap = np.linalg.norm(U_a.T @ U_b, 'fro') / np.sqrt(n_components)
            else:
                overlap = 0.0
            
            subspace_overlap.append({"layer": li, "overlap": float(overlap)})
        
        results[c_name] = {
            "cos_with_l0": cos_with_l0,
            "cos_with_prev": cos_with_prev,
            "subspace_overlap": subspace_overlap,
        }
        
        # 打印关键层
        for li in [0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1, n_layers]:
            if li > n_layers:
                continue
            print(f"    L{li}: cos(L0)={cos_with_l0[li]:.3f}", end="")
            if li > 0:
                print(f", cos(L{li-1})={cos_with_prev[li-1]:.3f}", end="")
            print()
        
        # 约束稳定性: 相邻层余弦 < 0.5 的比例
        unstable_rate = np.mean([1 for c in cos_with_prev if abs(c) < 0.5])
        print(f"    约束不稳定率(|cos(L-1)|<0.5): {unstable_rate:.2%}")
    
    return results


# ============================================================
# 主函数
# ============================================================

def run_all_experiments(model_name: str):
    """运行所有实验"""
    print(f"\n{'='*60}")
    print(f"Phase 156: 约束流形动力学 — {model_name}")
    print(f"{'='*60}")
    
    # 加载模型
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
    use_8bit = model_name in ("deepseek7b", "glm4") and gpu_mem_gb < 16
    load_mode = "8bit量化" if use_8bit else "bfloat16"
    print(f"加载方式: {load_mode} (GPU={gpu_mem_gb:.1f}GB)")
    
    t0 = time.time()
    model, tokenizer, device = load_model(model_name, use_8bit=use_8bit)
    t_load = time.time() - t0
    print(f"加载耗时: {t_load:.1f}s")
    
    model_info = get_model_info(model, model_name)
    print(f"模型: {model_info.model_class}, L={model_info.n_layers}, d={model_info.d_model}, V={model_info.vocab_size}")
    
    all_results = {
        "model": model_name,
        "load_mode": load_mode,
        "model_info": {
            "class": model_info.model_class,
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
            "vocab_size": model_info.vocab_size,
        },
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # Exp A: 独立约束探针
    try:
        t0 = time.time()
        result_a = exp_a_constraint_probes(model, tokenizer, device, model_info)
        all_results["exp_a"] = result_a
        print(f"Exp A 耗时: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Exp A 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_a_error"] = str(e)
    
    # Exp B: 可行空间压缩
    try:
        t0 = time.time()
        result_b = exp_b_feasible_space_compression(model, tokenizer, device, model_info)
        all_results["exp_b"] = result_b
        print(f"Exp B 耗时: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Exp B 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_b_error"] = str(e)
    
    # Exp C: 注意力路由拓扑
    try:
        t0 = time.time()
        result_c = exp_c_attention_topology(model, tokenizer, device, model_info)
        all_results["exp_c"] = result_c
        print(f"Exp C 耗时: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Exp C 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_c_error"] = str(e)
    
    # Exp D: 随机初始化对照
    try:
        t0 = time.time()
        result_d = exp_d_random_init_control(model, tokenizer, device, model_info)
        all_results["exp_d"] = result_d
        print(f"Exp D 耗时: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Exp D 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_d_error"] = str(e)
    
    # Exp E: Residual Stream几何
    try:
        t0 = time.time()
        result_e = exp_e_residual_stream_geometry(model, tokenizer, device, model_info)
        all_results["exp_e"] = result_e
        print(f"Exp E 耗时: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Exp E 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp_e_error"] = str(e)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = OUTPUT_DIR / f"phase156_{model_name}_{timestamp}.json"
    
    # 转换numpy类型为Python原生类型
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj
    
    all_results = convert(all_results)
    
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_file}")
    
    # 释放模型
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_all_experiments(model_name)
