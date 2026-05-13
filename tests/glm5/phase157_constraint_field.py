"""
Phase 157: 约束场动力学 — 发现语言系统的状态变量
==============================================

用户核心批评(Phase 156反思):
1. 探针用的仍是人类语言学标签(语法/语义/逻辑), 没发现模型的原生约束基底
2. tau/entropy/margin都是"观测量", 不是"状态变量"
3. 缺少"守恒量" — 哪些量跨层守恒? 哪些量在推理时爆炸?
4. 需要研究Δh_ℓ的输运动力学, 而非静态probe

核心目标:
  Exp 1: Δh输运分析 — 层间增量在低维流形上transport?
  Exp 2: 原生约束基底发现 — 数据驱动, 不用人类标签
  Exp 3: 约束干涉实验 — 冲突约束的cancellation/rotation/bifurcation
  Exp 4: 约束相变检测 — d²Ω/dℓ²的二阶变化
  Exp 5: 语言守恒量搜索 — 跨层守恒的几何量

用法:
  python tests/glm5/phase157_constraint_field.py qwen3
  python tests/glm5/phase157_constraint_field.py deepseek7b
  python tests/glm5/phase157_constraint_field.py glm4
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
from scipy.stats import kendalltau
from scipy.sparse.linalg import svds
from model_utils import (load_model, get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS)
from collections import defaultdict

OUTPUT_DIR = Path("tests/glm5_temp")


# ============================================================
# 大规模语料 — 每类200+句, 关键测试加大数据量!
# ============================================================

# 普通句子(无特殊约束冲突) — 200句
NORMAL_PROMPTS = [
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
    "Yesterday she went to the", "Tomorrow they will visit the", "Now he is reading the",
    "Last week the team completed the", "Next year the company will launch the",
    "The red apple was placed on the", "The blue car drove past the",
    "The tall building stood near the", "The small bird sat on the",
    "The old man walked to the", "The young woman entered the",
    "The hot coffee spilled on the", "The cold wind blew through the",
    "The bright light shone on the", "The dark room contained a",
    "The heavy box was moved to the", "The soft pillow lay on the",
    "The sharp knife cut through the", "The smooth surface reflected the",
    "The sweet smell filled the", "The bitter taste lingered in the",
    "The loud noise startled the", "The quiet room had a",
    "The fast runner crossed the", "The slow turtle crawled across the",
    "She opened the door to the", "He closed the window near the",
    "They built a house by the", "We found the key near the",
    "The book was on the", "The pen lay beside the",
    "Water flows down the", "Fire burns the",
    "Snow covered the", "Ice melted on the",
    "The teacher wrote on the", "The student sat at the",
    "Morning light filled the", "Evening shadows crossed the",
    "Spring flowers bloomed in the", "Winter snow fell on the",
    "The bridge crossed the", "The tunnel went under the",
    "The mountain towered above the", "The valley stretched below the",
    "The ocean waves crashed on the", "The lake reflected the",
    "The forest path led to the", "The garden gate opened to the",
    "The kitchen smelled of the", "The bedroom faced the",
    "The library held the", "The museum displayed the",
    "The church stood near the", "The school was next to the",
    "The hospital treated the", "The restaurant served the",
    "The train arrived at the", "The plane landed at the",
    "The boat sailed across the", "The bus stopped near the",
    "The phone rang in the", "The computer displayed the",
    "The camera captured the", "The screen showed the",
    "The music played in the", "The painting hung on the",
    "The poem described the", "The story told of the",
    "The equation solved the", "The experiment tested the",
    "The theory explained the", "The data supported the",
    "The proof confirmed the", "The argument convinced the",
    "The evidence revealed the", "The witness described the",
    "The judge considered the", "The jury decided the",
    "The lawyer argued the", "The defendant claimed the",
    "The president announced the", "The committee discussed the",
    "The report detailed the", "The survey found the",
    "The analysis showed the", "The review criticized the",
    "The proposal suggested the", "The plan included the",
    "The project required the", "The budget funded the",
    "The strategy focused on the", "The goal aimed at the",
    "The result exceeded the", "The failure taught the",
    "The success inspired the", "The challenge tested the",
    "The opportunity offered the", "The risk threatened the",
    "The benefit improved the", "The cost reduced the",
    "The quality enhanced the", "The speed increased the",
    "The size mattered for the", "The shape influenced the",
    "The color matched the", "The texture affected the",
    "The temperature changed the", "The pressure altered the",
    "The volume filled the", "The weight pressed on the",
    "The height reached the", "The depth measured the",
    "The length spanned the", "The width covered the",
    "The area included the", "The distance separated the",
    "The direction pointed to the", "The angle formed the",
    "The position determined the", "The movement created the",
    "The force pushed the", "The energy powered the",
    "The power controlled the", "The signal triggered the",
    "The message conveyed the", "The meaning expressed the",
    "The language described the", "The word referred to the",
    "The sentence stated the", "The paragraph developed the",
    "The chapter covered the", "The section addressed the",
    "The topic focused on the", "The theme explored the",
    "The idea suggested the", "The concept defined the",
    "The principle guided the", "The rule governed the",
    "The law required the", "The standard set the",
    "The pattern repeated the", "The structure organized the",
    "The system processed the", "The network connected the",
    "The circuit powered the", "The device measured the",
    "The instrument recorded the", "The tool shaped the",
    "The machine built the", "The engine drove the",
    "The motor turned the", "The gear moved the",
    "The wheel rotated the", "The lever lifted the",
    "The spring pushed the", "The bolt secured the",
    "The nail held the", "The screw fastened the",
    "The glue bonded the", "The weld joined the",
    "The rope tied the", "The chain linked the",
    "The wire connected the", "The pipe carried the",
    "The tube delivered the", "The channel directed the",
    "The path led to the", "The route followed the",
    "The track guided the", "The trail wound through the",
    "The road crossed the", "The street passed the",
    "The avenue lined the", "The boulevard bordered the",
    "The highway connected the", "The intersection joined the",
    "The roundabout circled the", "The bridge spanned the",
    "The tunnel pierced the", "The overpass crossed the",
    "The underpass went below the", "The ramp led to the",
    "The slope descended to the", "The hill rose above the",
    "The cliff overlooked the", "The gorge cut through the",
    "The canyon stretched through the", "The plateau extended across the",
    "The plain spread across the", "The desert stretched across the",
    "The oasis provided the", "The island sat in the",
    "The peninsula jutted into the", "The bay curved into the",
    "The cove sheltered the", "The harbor protected the",
    "The dock extended into the", "The pier reached over the",
    "The wharf lined the", "The quay bordered the",
    "The marina held the", "The port received the",
]

# 约束冲突句子 — 语法正确但语义荒谬
CONFLICT_SEMANTIC = [
    "The invisible elephant sat on the", "The dry water flowed through the",
    "The silent thunder shook the", "The cold fire burned the",
    "The dark light illuminated the", "The heavy feather crushed the",
    "The tiny mountain blocked the", "The empty crowd filled the",
    "The slow lightning struck the", "The quiet explosion shattered the",
    "The smooth sandpaper polished the", "The blunt razor cut the",
    "The flexible steel bent the", "The rigid rubber stretched the",
    "The transparent wall blocked the", "The liquid rock flowed through the",
    "The growing fossil changed the", "The dying immortal lost the",
    "The moving mountain approached the", "The frozen volcano erupted the",
    "The sleeping alarm woke the", "The blind spotlight illuminated the",
    "The deaf siren alerted the", "The numb pain hurt the",
    "The tasteless spice flavored the", "The odorless perfume scented the",
    "The colorless rainbow painted the", "The soundless music played the",
    "The weightless anchor held the", "The wingless bird flew over the",
    "The legless table stood on the", "The rootless tree grew in the",
    "The bookless library contained the", "The waterless ocean covered the",
    "The treeless forest surrounded the", "The starless sky lit the",
    "The peopleless city populated the", "The wordless story told the",
    "The numberless equation solved the", "The soundless echo returned the",
    "The heatless fire warmed the", "The lightless day brightened the",
    "The motionless race finished the", "The thoughtless genius created the",
    "The effortless struggle challenged the", "The painless agony suffered the",
    "The hopeless optimist believed the", "The fearless coward faced the",
    "The truthful liar told the", "The honest cheat revealed the",
    "The generous miser shared the", "The humble bragger announced the",
    "The patient hurry waited for the", "The careful accident caused the",
    "The quiet shout echoed the", "The gentle violence hurt the",
    "The peaceful war destroyed the", "The loving hatred consumed the",
    "The wise fool understood the", "The clear confusion explained the",
    "The simple complexity confused the", "The certain doubt questioned the",
    "The full emptiness contained the", "The ordered chaos organized the",
    "The stable change shifted the", "The constant variable altered the",
    "The finite infinity reached the", "The temporary permanence lasted the",
    "The probable impossibility occurred the", "The necessary option chose the",
]

# 约束冲突句子 — 逻辑矛盾
CONFLICT_LOGIC = [
    "She did not not finish the", "He never never mentioned the",
    "Nobody did not believe the", "Nothing was not found in the",
    "The statement is not untrue about the", "It is impossible that the result is not the",
    "She cannot not open the", "He will not never accept the",
    "They both agreed and disagreed about the", "The answer is both right and wrong for the",
    "The event happened before and after the", "The object is inside and outside the",
    "Everything and nothing filled the", "Always and never described the",
    "The full empty container held the", "The open closed door blocked the",
    "The living dead walked toward the", "The known secret revealed the",
    "The certain uncertainty confused the", "The exact approximation measured the",
    "The original copy of the document was found in the",
    "The same different result appeared in the",
    "The increasing decrease affected the", "The growing shrinkage reduced the",
    "The accelerating deceleration slowed the", "The brightening darkness hid the",
    "The rising fall dropped the", "The advancing retreat moved the",
    "The improving deterioration worsened the", "The strengthening weakness undermined the",
]

# 长距离依赖句子
LONG_RANGE_DEP = [
    "The cat that the dog that the boy liked chased sat on the",
    "The students that the teacher that the principal hired taught passed the",
    "The key that the woman that the man trusted gave him opened the",
    "The book that the author that the publisher rejected wrote contained the",
    "The song that the singer that the audience loved performed became the",
    "The law that the congress that the people elected passed affected the",
    "The plan that the team that the manager led developed solved the",
    "The drug that the company that the FDA approved produced cured the",
    "The project that the engineer that the company hired designed needed the",
    "The theory that the scientist that the university employed proposed explained the",
    "The machine that the factory that the corporation owned built produced the",
    "The painting that the artist that the gallery exhibited sold fetched the",
    "The medicine that the doctor that the hospital employed prescribed cured the",
    "The car that the dealer that the manufacturer supplied sold had the",
    "The movie that the director that the studio hired made won the",
]

# 多约束叠加句子
MULTI_CONSTRAINT = [
    "The three red cats quickly sat on the", "The five blue dogs slowly ran toward the",
    "The seven tall birds silently flew over the", "The two small children happily played in the",
    "The four old women carefully walked to the", "The six young men proudly stood by the",
    "The eight fast horses gracefully galloped across the",
    "The nine slow fish peacefully swam in the",
    "The ten smart students quietly read the",
    "The three kind teachers patiently explained the",
    "The five brave scientists carefully discovered the",
    "The seven young writers proudly published the",
    "The two talented artists beautifully painted the",
    "The four experienced doctors carefully examined the",
    "The six skilled engineers precisely built the",
    "The three loud musicians enthusiastically played the",
    "The five careful drivers safely stopped at the",
    "She never quickly finished the", "He always slowly mentioned the",
    "They rarely carefully noticed the", "She seldom quietly touched the",
    "He barely carefully visited the", "Nobody ever really believed the",
    "No one ever actually remembered the", "She rarely clearly spoke about the",
]


# ============================================================
# Exp 1: Δh输运分析 — 约束场输运
# ============================================================

def exp1_delta_h_transport(model, tokenizer, device, model_info):
    """
    核心思想:
    研究 Δh_ℓ = h_{ℓ+1} - h_ℓ 的动力学.
    
    如果Δh_ℓ在低维流形上transport → 约束沿特定方向传播
    如果Δh_ℓ是高维噪声 → 无结构化传播
    
    分析:
    1. Δh_ℓ的SVD谱 — 有效维度是多少?
    2. Δh_ℓ的主方向与约束方向的align度
    3. Δh_ℓ的层间相关性 — 输运是否连贯?
    """
    print("\n" + "="*60)
    print("Exp 1: Δh输运分析 — 约束场输运动力学")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 收集各层hidden states
    prompts = NORMAL_PROMPTS[:80]  # 80句大样本
    print(f"  收集{len(prompts)}句的hidden states...")
    
    all_h = {f"L{i}": [] for i in range(n_layers + 1)}
    
    for idx, prompt in enumerate(prompts):
        if idx % 20 == 0:
            print(f"  处理prompt {idx}/{len(prompts)}...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            all_h[f"L{li}"].append(h)
    
    # 转换为numpy数组
    H = {}
    for li in range(n_layers + 1):
        H[f"L{li}"] = np.array(all_h[f"L{li}"])  # [n_prompts, d_model]
    
    # 计算 Δh
    DeltaH = {}
    for li in range(n_layers):
        DeltaH[f"L{li}"] = H[f"L{li+1}"] - H[f"L{li}"]  # [n_prompts, d_model]
    
    # 分析1: Δh的SVD谱 — 每层的有效维度
    print("\n--- Δh的SVD谱分析 ---")
    delta_ranks = {}
    
    for li in range(n_layers):
        dh = DeltaH[f"L{li}"]  # [n, d]
        # 中心化
        dh_mean = np.mean(dh, axis=0)
        dh_centered = dh - dh_mean
        
        # SVD
        n_samples = dh_centered.shape[0]
        n_features = dh_centered.shape[1]
        
        # 用小SVD避免OOM
        if n_samples < n_features:
            U, S, Vt = np.linalg.svd(dh_centered @ dh_centered.T, full_matrices=False)
            S = np.sqrt(np.maximum(S, 0))
        else:
            U, S, Vt = np.linalg.svd(dh_centered, full_matrices=False)
        
        # 有效维度: 90%能量对应的维度
        total_energy = np.sum(S**2)
        cumulative = np.cumsum(S**2) / max(total_energy, 1e-10)
        eff_dim_90 = int(np.searchsorted(cumulative, 0.9)) + 1
        eff_dim_50 = int(np.searchsorted(cumulative, 0.5)) + 1
        
        delta_ranks[f"L{li}"] = {
            "eff_dim_90": eff_dim_90,
            "eff_dim_50": eff_dim_50,
            "top1_energy": float(S[0]**2 / max(total_energy, 1e-10)),
            "top5_energy": float(np.sum(S[:5]**2) / max(total_energy, 1e-10)),
            "top10_energy": float(np.sum(S[:10]**2) / max(total_energy, 1e-10)),
        }
        
        if li <= 2 or li >= n_layers - 3 or li == n_layers // 2:
            print(f"  L{li}→L{li+1}: eff_dim_90={eff_dim_90}, eff_dim_50={eff_dim_50}, "
                  f"top1_energy={delta_ranks[f'L{li}']['top1_energy']:.3f}, "
                  f"top10_energy={delta_ranks[f'L{li}']['top10_energy']:.3f}")
    
    # 分析2: Δh的主方向与约束方向的align度
    print("\n--- Δh主方向分析 ---")
    
    # 用语法(singular/plural)的contrast作为约束方向参考
    sing_prompts = [
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
    ]
    plur_prompts = [
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
    ]
    
    # 收集约束方向的hidden states
    def get_mean_h(prompts_list, n_layers_local):
        hs = {f"L{i}": [] for i in range(n_layers_local + 1)}
        for p in prompts_list:
            inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=64)
            ids = inputs["input_ids"].to(device)
            mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            for li in range(n_layers_local + 1):
                hs[f"L{li}"].append(out.hidden_states[li][0, -1, :].float().cpu().numpy())
        return hs
    
    hs_sing = get_mean_h(sing_prompts, n_layers)
    hs_plur = get_mean_h(plur_prompts, n_layers)
    
    # 语法约束方向
    syntax_dirs = {}
    for li in range(1, n_layers + 1):
        mean_s = np.mean(hs_sing[f"L{li}"], axis=0)
        mean_p = np.mean(hs_plur[f"L{li}"], axis=0)
        diff = mean_s - mean_p
        norm = np.linalg.norm(diff)
        if norm > 1e-10:
            syntax_dirs[li] = diff / norm
        else:
            syntax_dirs[li] = np.zeros(d_model)
    
    # Δh主方向与语法约束方向的对齐度
    delta_syntax_align = {}
    for li in range(n_layers):
        dh = DeltaH[f"L{li}"]
        dh_mean = np.mean(dh, axis=0)
        # Δh的top-1方向
        dh_centered = dh - dh_mean
        if dh_centered.shape[0] < dh_centered.shape[1]:
            U, S, Vt = np.linalg.svd(dh_centered @ dh_centered.T, full_matrices=False)
            top_dir = dh_centered.T @ U[:, 0]
        else:
            U, S, Vt = np.linalg.svd(dh_centered, full_matrices=False)
            top_dir = Vt[0]
        
        top_dir_norm = top_dir / max(np.linalg.norm(top_dir), 1e-10)
        
        # 与语法约束方向的对齐
        if li + 1 in syntax_dirs:
            align = abs(float(np.dot(top_dir_norm, syntax_dirs[li + 1])))
        else:
            align = 0.0
        
        delta_syntax_align[li] = align
        
        if li <= 2 or li >= n_layers - 3 or li == n_layers // 2:
            print(f"  L{li}→L{li+1}: Δh主方向与语法约束对齐度={align:.4f}")
    
    # 分析3: Δh的层间余弦 — 输运是否连贯?
    print("\n--- Δh层间连贯性 ---")
    delta_cos_prev = {}
    for li in range(1, n_layers):
        dh_prev = np.mean(DeltaH[f"L{li-1}"], axis=0)
        dh_curr = np.mean(DeltaH[f"L{li}"], axis=0)
        cos = float(np.dot(dh_prev, dh_curr) / 
                    max(np.linalg.norm(dh_prev) * np.linalg.norm(dh_curr), 1e-10))
        delta_cos_prev[li] = cos
        
        if li <= 3 or li >= n_layers - 2 or li == n_layers // 2:
            print(f"  cos(Δh_L{li-1}, Δh_L{li})={cos:.4f}")
    
    return {
        "delta_ranks": delta_ranks,
        "delta_syntax_align": delta_syntax_align,
        "delta_cos_prev": delta_cos_prev,
    }


# ============================================================
# Exp 2: 原生约束基底发现 — 数据驱动, 不用人类标签
# ============================================================

def exp2_native_constraint_basis(model, tokenizer, device, model_info):
    """
    核心思想:
    不用人类标签(语法/语义/逻辑), 而是用数据驱动方法发现模型自己的约束基底.
    
    方法:
    1. 收集大量句子的hidden states
    2. 对每层做PCA, 找到主方向
    3. 用对比分析: 随机选两组句子, 看哪些PCA方向能区分它们
    4. 最稳定的区分方向 = 模型的原生约束基底
    
    关键: 不预设"约束是什么", 而是让数据说话.
    """
    print("\n" + "="*60)
    print("Exp 2: 原生约束基底发现 — 数据驱动")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 收集所有类型句子的hidden states
    all_prompts = (NORMAL_PROMPTS[:60] + CONFLICT_SEMANTIC[:30] + 
                   CONFLICT_LOGIC[:15] + MULTI_CONSTRAINT[:15])
    n_prompts = len(all_prompts)
    labels = (["normal"] * 60 + ["conflict_sem"] * 30 + 
              ["conflict_logic"] * 15 + ["multi_constraint"] * 15)
    
    print(f"  收集{n_prompts}句的hidden states...")
    
    all_h = {f"L{i}": [] for i in range(n_layers + 1)}
    
    for idx, prompt in enumerate(all_prompts):
        if idx % 30 == 0:
            print(f"  处理prompt {idx}/{n_prompts}...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            all_h[f"L{li}"].append(h)
    
    # 分析: 在每层发现"最有区分力"的方向
    print("\n--- 原生约束基底(每层top区分方向) ---")
    
    # 对比组: normal vs conflict_semantic
    normal_idx = [i for i, l in enumerate(labels) if l == "normal"]
    conflict_idx = [i for i, l in enumerate(labels) if l in ("conflict_sem", "conflict_logic")]
    
    native_basis = {}
    
    for li in range(1, n_layers + 1):
        H = np.array(all_h[f"L{li}"])  # [n, d]
        
        # 1. 全局PCA
        H_mean = np.mean(H, axis=0)
        H_centered = H - H_mean
        
        # 只用前50个主成分(节省计算)
        n_components = min(50, H_centered.shape[0] - 1, d_model)
        if n_components <= 0:
            continue
        
        U, S, Vt = np.linalg.svd(H_centered, full_matrices=False)
        top_pcs = Vt[:n_components]  # [k, d] — 主成分方向
        
        # 2. 对每个主成分, 计算其区分normal vs conflict的能力
        # 用t-statistic
        normal_proj = H[normal_idx] @ top_pcs.T  # [n_normal, k]
        conflict_proj = H[conflict_idx] @ top_pcs.T  # [n_conflict, k]
        
        t_stats = []
        for k in range(n_components):
            n_vals = normal_proj[:, k]
            c_vals = conflict_proj[:, k]
            n_mean, c_mean = np.mean(n_vals), np.mean(c_vals)
            n_std, c_std = np.std(n_vals) + 1e-10, np.std(c_vals) + 1e-10
            t_stat = abs(n_mean - c_mean) / np.sqrt(n_std**2/len(n_vals) + c_std**2/len(c_vals))
            t_stats.append(t_stat)
        
        t_stats = np.array(t_stats)
        
        # 最有区分力的主成分
        top_discrim_idx = np.argsort(t_stats)[-5:][::-1]
        
        # 这些方向的区分力
        top_discrim_t = [float(t_stats[i]) for i in top_discrim_idx]
        
        # 方差解释比
        total_var = np.sum(S[:n_components]**2)
        explained_ratios = [float(S[i]**2 / max(total_var, 1e-10)) for i in top_discrim_idx]
        
        native_basis[li] = {
            "top_discrim_pcs": [int(x) for x in top_discrim_idx],
            "top_discrim_t": top_discrim_t,
            "explained_ratios": explained_ratios,
            "n_components": n_components,
        }
        
        if li <= 2 or li >= n_layers - 1 or li == n_layers // 2:
            print(f"  L{li}: top区分PC={top_discrim_idx[:3]}, "
                  f"t-stats=[{', '.join(f'{t:.2f}' for t in top_discrim_t[:3])}], "
                  f"解释比=[{', '.join(f'{r:.4f}' for r in explained_ratios[:3])}]")
    
    # 分析: 哪些PCA方向在所有层都有区分力? → 这些是"稳定约束基底"
    print("\n--- 稳定约束基底(跨层一致有区分力的PCA方向) ---")
    
    # 追踪: 每个PC方向在各层的区分力排名
    pc_persistence = defaultdict(list)
    for li in range(1, n_layers + 1):
        if li not in native_basis:
            continue
        for rank, pc_idx in enumerate(native_basis[li]["top_discrim_pcs"][:10]):
            pc_persistence[pc_idx].append((li, rank))
    
    # 持续出现在top-10的PC
    persistent_pcs = {pc: positions for pc, positions in pc_persistence.items() 
                      if len(positions) >= n_layers // 3}
    
    if persistent_pcs:
        print(f"  持续出现(>{n_layers//3}层)的PC: {list(persistent_pcs.keys())}")
        for pc, positions in persistent_pcs.items():
            layers_present = [p[0] for p in positions]
            print(f"    PC{pc}: 出现在L{min(layers_present)}-L{max(layers_present)}, "
                  f"{len(positions)}层")
    else:
        print("  没有持续出现在top区分方向的PC — 约束基底随层变化!")
    
    return {
        "native_basis": native_basis,
        "persistent_pcs": {str(k): [(int(p[0]), int(p[1])) for p in v] 
                           for k, v in persistent_pcs.items()},
    }


# ============================================================
# Exp 3: 约束干涉实验 — 冲突约束的cancellation/rotation/bifurcation
# ============================================================

def exp3_constraint_interference(model, tokenizer, device, model_info):
    """
    核心思想:
    构造冲突约束, 观察约束方向的interaction:
    - cancellation: 冲突方向互相抵消
    - rotation: 冲突方向导致约束旋转
    - bifurcation: 不同样本沿不同方向分化
    
    对比:
    - normal句子: 约束兼容
    - semantic conflict: 语法正确但语义荒谬
    - logic conflict: 逻辑矛盾
    """
    print("\n" + "="*60)
    print("Exp 3: 约束干涉 — 冲突约束的交互")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    # 收集各类hidden states
    normal_prompts = NORMAL_PROMPTS[:40]
    conflict_sem_prompts = CONFLICT_SEMANTIC[:40]
    conflict_logic_prompts = CONFLICT_LOGIC[:20]
    long_range_prompts = LONG_RANGE_DEP[:15]
    
    def collect_hs(prompts, label):
        hs = {f"L{i}": [] for i in range(n_layers + 1)}
        for p in prompts:
            inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=64)
            ids = inputs["input_ids"].to(device)
            mask = inputs["attention_mask"].to(device)
            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            for li in range(n_layers + 1):
                hs[f"L{li}"].append(out.hidden_states[li][0, -1, :].float().cpu().numpy())
        return hs
    
    print("  收集normal hidden states...")
    hs_normal = collect_hs(normal_prompts, "normal")
    print("  收集semantic conflict hidden states...")
    hs_conflict = collect_hs(conflict_sem_prompts, "conflict_sem")
    print("  收集logic conflict hidden states...")
    hs_logic = collect_hs(conflict_logic_prompts, "conflict_logic")
    print("  收集long-range dependency hidden states...")
    hs_long = collect_hs(long_range_prompts, "long_range")
    
    # 分析: 约束方向在不同条件下的变化
    print("\n--- 约束干涉分析 ---")
    
    interference_results = {}
    
    for li in range(1, n_layers + 1):
        H_n = np.array(hs_normal[f"L{li}"])
        H_c = np.array(hs_conflict[f"L{li}"])
        H_l = np.array(hs_logic[f"L{li}"])
        
        # 1. 约束张力: normal的类内散度 vs conflict的类内散度
        # 高张力 = 更大散度 = 约束冲突
        normal_spread = float(np.mean(np.var(H_n, axis=0)))
        conflict_spread = float(np.mean(np.var(H_c, axis=0)))
        logic_spread = float(np.mean(np.var(H_l, axis=0)))
        
        # 2. 约束距离: normal中心 vs conflict中心
        normal_center = np.mean(H_n, axis=0)
        conflict_center = np.mean(H_c, axis=0)
        logic_center = np.mean(H_l, axis=0)
        
        dist_nc = float(np.linalg.norm(normal_center - conflict_center))
        dist_nl = float(np.linalg.norm(normal_center - logic_center))
        
        # 3. 约束方差比: 冲突句子的方差是否增大?
        variance_ratio = conflict_spread / max(normal_spread, 1e-10)
        
        interference_results[li] = {
            "normal_spread": normal_spread,
            "conflict_spread": conflict_spread,
            "logic_spread": logic_spread,
            "dist_normal_conflict": dist_nc,
            "dist_normal_logic": dist_nl,
            "variance_ratio": variance_ratio,
        }
        
        if li <= 2 or li >= n_layers - 1 or li == n_layers // 2:
            print(f"  L{li}: spread_n={normal_spread:.4f} spread_c={conflict_spread:.4f} "
                  f"ratio={variance_ratio:.3f} dist_nc={dist_nc:.4f}")
    
    # 关键问题: 冲突是否导致"约束张力"增加?
    print("\n--- 约束张力跨层演化 ---")
    early_layers = list(range(1, min(6, n_layers + 1)))
    mid_layers = list(range(n_layers // 3, n_layers // 3 + 5))
    late_layers = list(range(max(1, n_layers - 4), n_layers + 1))
    
    for layer_group, name in [(early_layers, "早期(1-5)"), 
                               (mid_layers, "中期"), 
                               (late_layers, "晚期")]:
        ratios = [interference_results[li]["variance_ratio"] for li in layer_group if li in interference_results]
        if ratios:
            print(f"  {name}: avg variance_ratio={np.mean(ratios):.3f}")
    
    return interference_results


# ============================================================
# Exp 4: 约束相变检测 — d²Ω/dℓ²
# ============================================================

def exp4_phase_transition(model, tokenizer, device, model_info):
    """
    核心思想:
    测量可行空间体积的二阶变化率 d²V/dℓ².
    如果存在相变, 会看到:
    - dV/dℓ 的突变
    - d²V/dℓ² 的尖峰
    
    同时测量:
    - hidden state norm的层间变化
    - logit sparsity的变化
    - 约束密度的变化
    """
    print("\n" + "="*60)
    print("Exp 4: 约束相变检测 — d²Ω/dℓ²")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    W_U = get_W_U(model, model_info.name)
    
    prompts = NORMAL_PROMPTS[:60]
    eps = 5.0  # 可行空间截断
    
    # 收集各层的度量
    layer_metrics = {f"L{i}": {"feasible_vol": [], "h_norm": [], "logit_entropy": []}
                     for i in range(n_layers + 1)}
    
    for idx, prompt in enumerate(prompts):
        if idx % 20 == 0:
            print(f"  处理prompt {idx}/{len(prompts)}...")
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            
            # 可行空间体积
            logits = h @ W_U.T
            feasible = float(np.sum(logits > -eps))
            layer_metrics[f"L{li}"]["feasible_vol"].append(feasible)
            
            # Hidden state norm
            layer_metrics[f"L{li}"]["h_norm"].append(float(np.linalg.norm(h)))
            
            # Logit entropy (作为约束密度的代理)
            log_probs = logits - np.max(logits)
            log_probs = log_probs - np.log(np.sum(np.exp(log_probs)) + 1e-10)
            entropy = -np.sum(np.exp(log_probs) * log_probs)
            layer_metrics[f"L{li}"]["logit_entropy"].append(float(entropy))
    
    # 计算一阶和二阶导数
    print("\n--- 可行空间体积及导数 ---")
    
    mean_volumes = [np.mean(layer_metrics[f"L{li}"]["feasible_vol"]) for li in range(n_layers + 1)]
    mean_norms = [np.mean(layer_metrics[f"L{li}"]["h_norm"]) for li in range(n_layers + 1)]
    mean_entropies = [np.mean(layer_metrics[f"L{li}"]["logit_entropy"]) for li in range(n_layers + 1)]
    
    # 一阶导数 (中心差分)
    dV = []
    for li in range(n_layers + 1):
        if li == 0:
            dv = mean_volumes[1] - mean_volumes[0]
        elif li == n_layers:
            dv = mean_volumes[-1] - mean_volumes[-2]
        else:
            dv = (mean_volumes[li+1] - mean_volumes[li-1]) / 2
        dV.append(dv)
    
    # 二阶导数
    d2V = []
    for li in range(n_layers + 1):
        if li <= 0 or li >= n_layers:
            d2v = 0
        else:
            d2v = mean_volumes[li+1] - 2*mean_volumes[li] + mean_volumes[li-1]
        d2V.append(d2v)
    
    # 打印
    for li in range(n_layers + 1):
        if li <= 3 or li >= n_layers - 2 or li == n_layers // 2:
            print(f"  L{li}: V={mean_volumes[li]:.0f} dV={dV[li]:.0f} "
                  f"d2V={d2V[li]:.0f} norm={mean_norms[li]:.2f} "
                  f"H={mean_entropies[li]:.2f}")
    
    # 寻找相变点: d2V的极值
    d2V_arr = np.array(d2V)
    # 忽略首尾2层
    d2V_interior = d2V_arr[2:-2]
    
    if len(d2V_interior) > 0:
        max_d2v_idx = np.argmax(np.abs(d2V_interior)) + 2
        print(f"\n  ★ 最大d2V位置: L{max_d2v_idx} (d2V={d2V[max_d2v_idx]:.0f})")
        
        # 检查是否是真正的相变(d2V显著大于噪声水平)
        d2v_std = np.std(d2V_interior)
        if abs(d2V[max_d2v_idx]) > 3 * d2v_std:
            print(f"  ★★★ 可能的相变点! d2V={d2V[max_d2v_idx]:.0f} > 3σ={3*d2v_std:.0f}")
        else:
            print(f"  d2V未超过3σ(={3*d2v_std:.0f}), 可能不是相变")
    
    # hidden state norm的层间变化(可能包含LN效应)
    print("\n--- Hidden state norm变化 ---")
    norm_changes = [mean_norms[li+1] - mean_norms[li] for li in range(n_layers)]
    for li in range(n_layers):
        if li <= 3 or li >= n_layers - 2 or li == n_layers // 2:
            print(f"  L{li}→L{li+1}: Δnorm={norm_changes[li]:.4f}")
    
    return {
        "mean_volumes": mean_volumes,
        "mean_norms": mean_norms,
        "mean_entropies": mean_entropies,
        "dV": dV,
        "d2V": d2V,
        "norm_changes": norm_changes,
    }


# ============================================================
# Exp 5: 语言守恒量搜索 — 跨层守恒的几何量
# ============================================================

def exp5_language_invariants(model, tokenizer, device, model_info):
    """
    核心思想:
    寻找跨层守恒的几何量 — 这是语言系统的"状态变量".
    
    候选守恒量:
    1. h_ℓ的norm — 是否被LayerNorm归一化到固定值?
    2. h_ℓ的子空间维度 — 是否跨层保持?
    3. h_ℓ之间的距离关系 — 是否跨层保持序?
    4. 约束方向的relative angle — 是否跨层守恒?
    5. logit分布的rank correlation — Phase 155已测(tau)
    
    新方向: 不看标量, 看几何关系.
    """
    print("\n" + "="*60)
    print("Exp 5: 语言守恒量搜索")
    print("="*60)
    
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    prompts = NORMAL_PROMPTS[:50]
    n_prompts = len(prompts)
    
    # 收集各层hidden states
    print(f"  收集{n_prompts}句的hidden states...")
    all_h = {f"L{i}": [] for i in range(n_layers + 1)}
    
    for idx, prompt in enumerate(prompts):
        if idx % 20 == 0:
            print(f"  处理prompt {idx}/{n_prompts}...")
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=64)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask,
                       output_hidden_states=True)
        
        for li in range(n_layers + 1):
            h = out.hidden_states[li][0, -1, :].float().cpu().numpy()
            all_h[f"L{li}"].append(h)
    
    H = {f"L{li}": np.array(all_h[f"L{li}"]) for li in range(n_layers + 1)}
    
    # 1. 距离关系的守恒 — 同一批句子, 它们之间的相对距离是否跨层保持?
    print("\n--- 距离关系守恒(跨层rank correlation) ---")
    
    # 计算L0的距离矩阵
    D_ref = np.zeros((n_prompts, n_prompts))
    for i in range(n_prompts):
        for j in range(i+1, n_prompts):
            d = float(np.linalg.norm(H["L0"][i] - H["L0"][j]))
            D_ref[i, j] = d
            D_ref[j, i] = d
    
    # 提取上三角
    triu_idx = np.triu_indices(n_prompts, k=1)
    dist_ref = D_ref[triu_idx]
    
    # 各层与L0的距离相关
    dist_conservation = {}
    for li in [1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1, n_layers]:
        if li > n_layers:
            continue
        D_li = np.zeros((n_prompts, n_prompts))
        for i in range(n_prompts):
            for j in range(i+1, n_prompts):
                d = float(np.linalg.norm(H[f"L{li}"][i] - H[f"L{li}"][j]))
                D_li[i, j] = d
                D_li[j, i] = d
        
        dist_li = D_li[triu_idx]
        tau, _ = kendalltau(dist_ref, dist_li)
        dist_conservation[li] = float(tau)
        print(f"  L0 vs L{li}: distance rank τ = {tau:.4f}")
    
    # 2. 角度关系的守恒 — 向量之间的cosine similarity结构
    print("\n--- 角度关系守恒 ---")
    
    # L0的角度矩阵
    C_ref = np.zeros((n_prompts, n_prompts))
    for i in range(n_prompts):
        for j in range(n_prompts):
            hi = H["L0"][i]
            hj = H["L0"][j]
            ni = np.linalg.norm(hi)
            nj = np.linalg.norm(hj)
            if ni > 1e-10 and nj > 1e-10:
                C_ref[i, j] = float(np.dot(hi, hj) / (ni * nj))
            else:
                C_ref[i, j] = 0.0
    
    cos_ref = C_ref[triu_idx]
    
    angle_conservation = {}
    for li in [1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1, n_layers]:
        if li > n_layers:
            continue
        C_li = np.zeros((n_prompts, n_prompts))
        for i in range(n_prompts):
            for j in range(n_prompts):
                hi = H[f"L{li}"][i]
                hj = H[f"L{li}"][j]
                ni = np.linalg.norm(hi)
                nj = np.linalg.norm(hj)
                if ni > 1e-10 and nj > 1e-10:
                    C_li[i, j] = float(np.dot(hi, hj) / (ni * nj))
                else:
                    C_li[i, j] = 0.0
        
        cos_li = C_li[triu_idx]
        tau, _ = kendalltau(cos_ref, cos_li)
        angle_conservation[li] = float(tau)
        print(f"  L0 vs L{li}: angle rank τ = {tau:.4f}")
    
    # 3. 子空间重叠度 — 各层的主成分子空间是否重叠
    print("\n--- 子空间重叠度 ---")
    
    # L0的PCA
    H0_centered = H["L0"] - np.mean(H["L0"], axis=0)
    k = min(20, n_prompts - 1, d_model)
    U0, S0, Vt0 = np.linalg.svd(H0_centered, full_matrices=False)
    subspace0 = Vt0[:k]  # [k, d] — L0的top-k主成分
    
    subspace_overlap = {}
    for li in [1, 2, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1, n_layers]:
        if li > n_layers:
            continue
        H_li_centered = H[f"L{li}"] - np.mean(H[f"L{li}"], axis=0)
        U_li, S_li, Vt_li = np.linalg.svd(H_li_centered, full_matrices=False)
        subspace_li = Vt_li[:k]
        
        # 子空间重叠 = ||V0 V_li^T||_F / sqrt(k)
        overlap = np.linalg.norm(subspace0 @ subspace_li.T, 'fro') / np.sqrt(k)
        subspace_overlap[li] = float(overlap)
        print(f"  L0 vs L{li}: subspace overlap = {overlap:.4f}")
    
    # 4. 约束张力 — 多约束方向的平均对齐度
    print("\n--- 约束张力(跨层) ---")
    
    # 用之前定义的约束方向(直接定义语料, 不依赖import)
    sing_prompts = [
        "The cat sits on the", "The dog runs toward the", "The bird flies over the",
        "The child plays in the", "The woman walks to the", "The man stands by the",
        "The horse gallops across the", "The fish swims in the", "The student reads the",
        "The teacher explains the", "The scientist discovers the", "The writer publishes the",
        "The artist paints the", "The doctor examines the", "The engineer builds the",
    ]
    plur_prompts = [
        "The cats sit on the", "The dogs run toward the", "The birds fly over the",
        "The children play in the", "The women walk to the", "The men stand by the",
        "The horses gallop across the", "The fish swim in the", "The students read the",
        "The teachers explain the", "The scientists discover the", "The writers publish the",
        "The artists paint the", "The doctors examine the", "The engineers build the",
    ]
    col_prompts = [
        "The red apple was placed on the", "The blue car drove past the",
        "The green forest surrounded the", "The white snow covered the",
        "The black cat jumped over the", "The yellow sun shone on the",
        "The brown dog ran through the", "The pink flower grew in the",
        "The gray stone sat on the", "The orange leaf fell on the",
        "The purple curtain hung by the", "The gold ring was found in the",
        "The silver coin lay on the", "The red wine spilled on the",
        "The blue sky stretched over the",
    ]
    uncol_prompts = [
        "The apple was placed on the", "The car drove past the",
        "The forest surrounded the", "The snow covered the",
        "The cat jumped over the", "The sun shone on the",
        "The dog ran through the", "The flower grew in the",
        "The stone sat on the", "The leaf fell on the",
        "The curtain hung by the", "The ring was found in the",
        "The coin lay on the", "The wine spilled on the",
        "The sky stretched over the",
    ]
    
    # 重新收集对比句子的hidden states
    hs_sing = {f"L{i}": [] for i in range(n_layers + 1)}
    hs_plur = {f"L{i}": [] for i in range(n_layers + 1)}
    hs_col = {f"L{i}": [] for i in range(n_layers + 1)}
    hs_uncol = {f"L{i}": [] for i in range(n_layers + 1)}
    
    for p in sing_prompts:
        inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=64)
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        for li in range(n_layers + 1):
            hs_sing[f"L{li}"].append(out.hidden_states[li][0, -1, :].float().cpu().numpy())
    
    for p in plur_prompts:
        inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=64)
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        for li in range(n_layers + 1):
            hs_plur[f"L{li}"].append(out.hidden_states[li][0, -1, :].float().cpu().numpy())
    
    for p in col_prompts:
        inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=64)
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        for li in range(n_layers + 1):
            hs_col[f"L{li}"].append(out.hidden_states[li][0, -1, :].float().cpu().numpy())
    
    for p in uncol_prompts:
        inputs = tokenizer(p, return_tensors="pt", truncation=True, max_length=64)
        ids = inputs["input_ids"].to(device)
        mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        for li in range(n_layers + 1):
            hs_uncol[f"L{li}"].append(out.hidden_states[li][0, -1, :].float().cpu().numpy())
    
    # 约束方向
    constraint_tension = {}
    for li in range(1, n_layers + 1):
        # 语法约束方向
        mean_s = np.mean(hs_sing[f"L{li}"], axis=0)
        mean_p = np.mean(hs_plur[f"L{li}"], axis=0)
        c_syntax = mean_s - mean_p
        c_syntax_norm = np.linalg.norm(c_syntax)
        if c_syntax_norm > 1e-10:
            c_syntax = c_syntax / c_syntax_norm
        
        # 语义约束方向
        mean_c = np.mean(hs_col[f"L{li}"], axis=0)
        mean_u = np.mean(hs_uncol[f"L{li}"], axis=0)
        c_semantic = mean_c - mean_u
        c_semantic_norm = np.linalg.norm(c_semantic)
        if c_semantic_norm > 1e-10:
            c_semantic = c_semantic / c_semantic_norm
        
        # 约束张力 = 两个约束方向的内积
        tension = float(abs(np.dot(c_syntax, c_semantic)))
        constraint_tension[li] = tension
        
        if li <= 2 or li >= n_layers - 1 or li == n_layers // 2:
            print(f"  L{li}: 约束张力(|⟨c_syntax, c_semantic⟩|) = {tension:.4f}")
    
    # 综合分析
    print("\n--- 守恒量总结 ---")
    print(f"  距离关系守恒(τ, L0→L{n_layers}): {dist_conservation.get(n_layers, 'N/A')}")
    print(f"  角度关系守恒(τ, L0→L{n_layers}): {angle_conservation.get(n_layers, 'N/A')}")
    print(f"  子空间重叠(L0→L{n_layers}): {subspace_overlap.get(n_layers, 'N/A')}")
    
    return {
        "dist_conservation": dist_conservation,
        "angle_conservation": angle_conservation,
        "subspace_overlap": subspace_overlap,
        "constraint_tension": constraint_tension,
    }


# ============================================================
# 主函数
# ============================================================

def run_all_experiments(model_name: str):
    """运行所有实验"""
    print(f"\n{'='*60}")
    print(f"Phase 157: 约束场动力学 — {model_name}")
    print(f"{'='*60}")
    
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
    
    # Exp 1: Δh输运
    try:
        t0 = time.time()
        result1 = exp1_delta_h_transport(model, tokenizer, device, model_info)
        all_results["exp1"] = result1
        print(f"Exp 1 耗时: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Exp 1 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp1_error"] = str(e)
    
    # Exp 2: 原生约束基底
    try:
        t0 = time.time()
        result2 = exp2_native_constraint_basis(model, tokenizer, device, model_info)
        all_results["exp2"] = result2
        print(f"Exp 2 耗时: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Exp 2 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp2_error"] = str(e)
    
    # Exp 3: 约束干涉
    try:
        t0 = time.time()
        result3 = exp3_constraint_interference(model, tokenizer, device, model_info)
        all_results["exp3"] = result3
        print(f"Exp 3 耗时: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Exp 3 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp3_error"] = str(e)
    
    # Exp 4: 相变检测
    try:
        t0 = time.time()
        result4 = exp4_phase_transition(model, tokenizer, device, model_info)
        all_results["exp4"] = result4
        print(f"Exp 4 耗时: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Exp 4 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp4_error"] = str(e)
    
    # Exp 5: 守恒量
    try:
        t0 = time.time()
        result5 = exp5_language_invariants(model, tokenizer, device, model_info)
        all_results["exp5"] = result5
        print(f"Exp 5 耗时: {time.time()-t0:.1f}s")
    except Exception as e:
        print(f"Exp 5 失败: {e}")
        import traceback; traceback.print_exc()
        all_results["exp5_error"] = str(e)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    out_file = OUTPUT_DIR / f"phase157_{model_name}_{timestamp}.json"
    
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
    
    release_model(model)
    model = None
    gc.collect()
    torch.cuda.empty_cache()
    
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    run_all_experiments(model_name)
