"""
Phase 222: 约束方向稳定性验证——整个框架的生死实验
====================================================

核心问题：d_constraint = h_correct - h_wrong 是否是一个稳定对象？

如果跨句子的约束方向cos ≈ 0.8 → 存在稳定的"数约束方向" → 框架成立
如果跨句子的约束方向cos ≈ 0.1 → 不存在统一约束方向 → 框架塌

实验1（★★★★★ 最关键）: 约束方向跨句子稳定性
  - 1000个不同句子对，计算 d_i = h_correct_i - h_wrong_i
  - 测量 cos(d_i, d_j) 的分布
  - 如果mean cos > 0.5 → 稳定约束方向存在
  - 如果mean cos < 0.2 → 不存在

实验2（★★★★）: 不同约束类型的稳定性对比
  - 数约束 vs 时态约束 vs 语态约束
  - 哪种约束更稳定？

实验3（★★★★）: 约束方向与语义方向的正交性
  - 那99%的维度在做什么？
  - 约束方向与W_U行空间的对齐度
  - 约束子空间 vs 语义子空间是否正交？

实验4（★★★）: 扰动分解
  - d_constraint 中有多少是"纯约束"信号？
  - 有多少是词汇差异、频率效应、表面形态？

跨模型测试: Qwen3 -> GLM4 -> DS7B (顺序执行,避免OOM)
BF16 + device_map="auto" + sdpa(flash) + 定期GC

执行时间: 2026-05-18 04:40
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'tests/glm5')

import gc
import json
import time
import numpy as np
import torch
from pathlib import Path
from model_utils import (get_layers, get_model_info,
                          release_model, get_W_U, MODEL_CONFIGS,
                          get_sample_layers)

OUTPUT_DIR = Path("d:/Ai2050/TransformerLens-Project/tests/glm5_temp")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 全局日志 =====
_last_log_time = time.time()
_LOG_INTERVAL = 20

def log_status(msg):
    global _last_log_time
    t = time.strftime("%H:%M:%S")
    gpu_mem = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
    print(f"[{t}] GPU={gpu_mem:.1f}GB | {msg}", flush=True)
    _last_log_time = time.time()

def maybe_log(msg):
    global _last_log_time
    if time.time() - _last_log_time > _LOG_INTERVAL:
        log_status(msg)

# ===== 模型加载(sdpa + flash) =====
def load_model_sdpa(model_name: str):
    """BF16 + device_map='auto' + sdpa(flash内存优化)"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    cfg = MODEL_CONFIGS[model_name]
    log_status(f"Loading {model_name} (bf16 + auto + sdpa)...")
    
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
        attn_implementation="sdpa",
    )
    model.eval()
    device = next(model.parameters()).device
    gpu_mem = torch.cuda.memory_allocated()/1e9 if torch.cuda.is_available() else 0
    log_status(f"  Loaded: device={device}, GPU={gpu_mem:.1f}GB")
    return model, tokenizer, device

# ===== 大规模测试数据 =====
# 1000个数约束句子对（单数正确 vs 复数正确）
# 结构: "The <singular_noun> <verb_s>" vs "The <plural_noun> <verb_pl>"

NOUNS_SINGULAR = [
    "cat", "dog", "bird", "girl", "boy", "tree", "car", "child", "man", "woman",
    "fish", "horse", "student", "teacher", "flower", "river", "star", "moon", "sun",
    "cloud", "wind", "rain", "snow", "fire", "light", "sound", "door", "window",
    "book", "pen", "phone", "clock", "bell", "flag", "candle", "glass", "key",
    "wheel", "engine", "train", "boat", "plane", "rocket", "house", "table",
    "chair", "lamp", "mirror", "bottle", "cup", "plate", "knife", "fork", "spoon",
    "shirt", "hat", "shoe", "sock", "glove", "ring", "watch", "chain", "rope",
    "wire", "pipe", "brick", "stone", "sand", "dust", "smoke", "steam", "ice",
    "cake", "bread", "rice", "meat", "fish", "soup", "milk", "water", "wine",
    "beer", "tea", "coffee", "juice", "oil", "salt", "sugar", "honey", "butter",
    "apple", "orange", "grape", "peach", "pear", "plum", "cherry", "berry",
    "doctor", "nurse", "lawyer", "judge", "priest", "singer", "dancer", "painter",
    "writer", "reader", "driver", "rider", "walker", "runner", "swimmer", "flyer",
    "king", "queen", "prince", "princess", "lord", "lady", "knight", "bishop",
    "wolf", "bear", "lion", "tiger", "eagle", "hawk", "snake", "frog", "toad",
    "bee", "ant", "fly", "moth", "crab", "deer", "fox", "goat", "hare", "lamb",
    "owl", "pig", "rat", "seal", "swan", "ape", "bat", "cow", "crow", "dove",
    "duck", "elk", "gnu", "hen", "jay", "kit", "lark", "mole", "newt", "pike",
]

NOUNS_PLURAL = {
    "cat": "cats", "dog": "dogs", "bird": "birds", "girl": "girls", "boy": "boys",
    "tree": "trees", "car": "cars", "child": "children", "man": "men", "woman": "women",
    "fish": "fish", "horse": "horses", "student": "students", "teacher": "teachers",
    "flower": "flowers", "river": "rivers", "star": "stars", "moon": "moons",
    "sun": "suns", "cloud": "clouds", "wind": "winds", "rain": "rains",
    "snow": "snows", "fire": "fires", "light": "lights", "sound": "sounds",
    "door": "doors", "window": "windows", "book": "books", "pen": "pens",
    "phone": "phones", "clock": "clocks", "bell": "bells", "flag": "flags",
    "candle": "candles", "glass": "glasses", "key": "keys", "wheel": "wheels",
    "engine": "engines", "train": "trains", "boat": "boats", "plane": "planes",
    "rocket": "rockets", "house": "houses", "table": "tables", "chair": "chairs",
    "lamp": "lamps", "mirror": "mirrors", "bottle": "bottles", "cup": "cups",
    "plate": "plates", "knife": "knives", "fork": "forks", "spoon": "spoons",
    "shirt": "shirts", "hat": "hats", "shoe": "shoes", "sock": "socks",
    "glove": "gloves", "ring": "rings", "watch": "watches", "chain": "chains",
    "rope": "ropes", "wire": "wires", "pipe": "pipes", "brick": "bricks",
    "stone": "stones", "sand": "sands", "dust": "dusts", "smoke": "smokes",
    "steam": "steams", "ice": "ices", "cake": "cakes", "bread": "breads",
    "rice": "rices", "meat": "meats", "soup": "soups", "milk": "milks",
    "water": "waters", "wine": "wines", "beer": "beers", "tea": "teas",
    "coffee": "coffees", "juice": "juices", "oil": "oils", "salt": "salts",
    "sugar": "sugars", "honey": "honeys", "butter": "butters",
    "apple": "apples", "orange": "oranges", "grape": "grapes", "peach": "peaches",
    "pear": "pears", "plum": "plums", "cherry": "cherries", "berry": "berries",
    "doctor": "doctors", "nurse": "nurses", "lawyer": "lawyers", "judge": "judges",
    "priest": "priests", "singer": "singers", "dancer": "dancers", "painter": "painters",
    "writer": "writers", "reader": "readers", "driver": "drivers", "rider": "riders",
    "walker": "walkers", "runner": "runners", "swimmer": "swimmers", "flyer": "flyers",
    "king": "kings", "queen": "queens", "prince": "princes", "princess": "princesses",
    "lord": "lords", "lady": "ladies", "knight": "knights", "bishop": "bishops",
    "wolf": "wolves", "bear": "bears", "lion": "lions", "tiger": "tigers",
    "eagle": "eagles", "hawk": "hawks", "snake": "snakes", "frog": "frogs",
    "toad": "toads", "bee": "bees", "ant": "ants", "fly": "flies", "moth": "moths",
    "crab": "crabs", "deer": "deer", "fox": "foxes", "goat": "goats", "hare": "hares",
    "lamb": "lambs", "owl": "owls", "pig": "pigs", "rat": "rats", "seal": "seals",
    "swan": "swans", "ape": "apes", "bat": "bats", "cow": "cows", "crow": "crows",
    "dove": "doves", "duck": "ducks", "elk": "elks", "gnu": "gnus", "hen": "hens",
    "jay": "jays", "kit": "kits", "lark": "larks", "mole": "moles", "newt": "newts",
    "pike": "pikes",
}

VERBS_SINGULAR = [
    "chases", "runs", "sings", "reads", "walks", "falls", "moves", "plays",
    "works", "dances", "swims", "gallops", "writes", "speaks", "sleeps", "barks",
    "flies", "grows", "blooms", "flows", "shines", "rises", "sets", "blows",
    "melts", "burns", "echoes", "opens", "breaks", "rings", "ticks", "waves",
    "turns", "spins", "roars", "stops", "sails", "launches", "stands", "sits",
    "waits", "looks", "seems", "feels", "tastes", "smells", "sounds", "appears",
    "begins", "ends", "starts", "finishes", "continues", "stops", "helps",
    "wants", "needs", "loves", "hates", "likes", "knows", "thinks", "believes",
    "hopes", "fears", "trusts", "doubts", "wonders", "remembers", "forgets",
    "understands", "realizes", "discovers", "creates", "destroys", "builds",
    "defends", "attacks", "protects", "chooses", "decides", "plans", "tries",
    "learns", "teaches", "shows", "tells", "asks", "answers", "gives", "takes",
    "brings", "sends", "finds", "loses", "keeps", "leaves", "holds", "carries",
]

VERBS_PLURAL = {
    "chases": "chase", "runs": "run", "sings": "sing", "reads": "read",
    "walks": "walk", "falls": "fall", "moves": "move", "plays": "play",
    "works": "work", "dances": "dance", "swims": "swim", "gallops": "gallop",
    "writes": "write", "speaks": "speak", "sleeps": "sleep", "barks": "bark",
    "flies": "fly", "grows": "grow", "blooms": "bloom", "flows": "flow",
    "shines": "shine", "rises": "rise", "sets": "set", "blows": "blow",
    "melts": "melt", "burns": "burn", "echoes": "echo", "opens": "open",
    "breaks": "break", "rings": "ring", "ticks": "tick", "waves": "wave",
    "turns": "turn", "spins": "spin", "roars": "roar", "stops": "stop",
    "sails": "sail", "launches": "launch", "stands": "stand", "sits": "sit",
    "waits": "wait", "looks": "look", "seems": "seem", "feels": "feel",
    "tastes": "taste", "smells": "smell", "sounds": "sound", "appears": "appear",
    "begins": "begin", "ends": "end", "starts": "start", "finishes": "finish",
    "continues": "continue", "helps": "help", "wants": "want", "needs": "need",
    "loves": "love", "hates": "hate", "likes": "like", "knows": "know",
    "thinks": "think", "believes": "believe", "hopes": "hope", "fears": "fear",
    "trusts": "trust", "doubts": "doubt", "wonders": "wonder", "remembers": "remember",
    "forgets": "forget", "understands": "understand", "realizes": "realize",
    "discovers": "discover", "creates": "create", "destroys": "destroy",
    "builds": "build", "defends": "defend", "attacks": "attack", "protects": "protect",
    "chooses": "choose", "decides": "decide", "plans": "plan", "tries": "try",
    "learns": "learn", "teaches": "teach", "shows": "show", "tells": "tell",
    "asks": "ask", "answers": "answer", "gives": "give", "takes": "take",
    "brings": "bring", "sends": "send", "finds": "find", "loses": "lose",
    "keeps": "keep", "leaves": "leave", "holds": "hold", "carries": "carry",
}

# 时态约束句子对
TENSE_PAIRS = [
    ("The cat chases", "The cat chased"),
    ("The dog runs", "The dog ran"),
    ("The bird sings", "The bird sang"),
    ("The girl reads", "The girl read"),
    ("The boy walks", "The boy walked"),
    ("The tree falls", "The tree fell"),
    ("The car moves", "The car moved"),
    ("The child plays", "The child played"),
    ("The man works", "The man worked"),
    ("The woman dances", "The woman danced"),
    ("The fish swims", "The fish swam"),
    ("The horse gallops", "The horse galloped"),
    ("The student writes", "The student wrote"),
    ("The teacher speaks", "The teacher spoke"),
    ("The flower blooms", "The flower bloomed"),
    ("The river flows", "The river flowed"),
    ("The star shines", "The star shone"),
    ("The sun sets", "The sun set"),
    ("The wind blows", "The wind blew"),
    ("The fire burns", "The fire burned"),
    ("The light shines", "The light shone"),
    ("The sound echoes", "The sound echoed"),
    ("The door opens", "The door opened"),
    ("The book falls", "The book fell"),
    ("The pen writes", "The pen wrote"),
    ("The phone rings", "The phone rang"),
    ("The clock ticks", "The clock ticked"),
    ("The bell rings", "The bell rang"),
    ("The king rules", "The king ruled"),
    ("The queen leads", "The queen led"),
    ("The doctor helps", "The doctor helped"),
    ("The nurse cares", "The nurse cared"),
    ("The wolf howls", "The wolf howled"),
    ("The bear sleeps", "The bear slept"),
    ("The lion roars", "The lion roared"),
    ("The eagle flies", "The eagle flew"),
    ("The snake moves", "The snake moved"),
    ("The frog jumps", "The frog jumped"),
    ("The bee buzzes", "The bee buzzed"),
    ("The ant crawls", "The ant crawled"),
    ("The writer thinks", "The writer thought"),
    ("The painter draws", "The painter drew"),
    ("The singer performs", "The singer performed"),
    ("The dancer moves", "The dancer moved"),
    ("The driver stops", "The driver stopped"),
    ("The rider gallops", "The rider galloped"),
    ("The walker strolls", "The walker strolled"),
    ("The runner sprints", "The runner sprinted"),
    ("The swimmer dives", "The swimmer dove"),
]

# 语态约束句子对（主动 vs 被动）
VOICE_PAIRS = [
    ("The cat chases the dog", "The dog is chased by the cat"),
    ("The dog bites the man", "The man is bitten by the dog"),
    ("The girl reads the book", "The book is read by the girl"),
    ("The boy throws the ball", "The ball is thrown by the boy"),
    ("The teacher praises the student", "The student is praised by the teacher"),
    ("The wind breaks the window", "The window is broken by the wind"),
    ("The fire burns the wood", "The wood is burned by the fire"),
    ("The water fills the cup", "The cup is filled by the water"),
    ("The sun warms the earth", "The earth is warmed by the sun"),
    ("The rain wets the ground", "The ground is wetted by the rain"),
    ("The man opens the door", "The door is opened by the man"),
    ("The woman cleans the room", "The room is cleaned by the woman"),
    ("The child breaks the toy", "The toy is broken by the child"),
    ("The doctor treats the patient", "The patient is treated by the doctor"),
    ("The chef cooks the meal", "The meal is cooked by the chef"),
    ("The artist paints the picture", "The picture is painted by the artist"),
    ("The singer sings the song", "The song is sung by the singer"),
    ("The writer writes the letter", "The letter is written by the writer"),
    ("The builder builds the house", "The house is built by the builder"),
    ("The driver drives the car", "The car is driven by the driver"),
    ("The student solves the problem", "The problem is solved by the student"),
    ("The judge sentences the criminal", "The criminal is sentenced by the judge"),
    ("The king rules the kingdom", "The kingdom is ruled by the king"),
    ("The mother loves the child", "The child is loved by the mother"),
    ("The guard watches the gate", "The gate is watched by the guard"),
    ("The cat catches the mouse", "The mouse is caught by the cat"),
    ("The wolf hunts the deer", "The deer is hunted by the wolf"),
    ("The eagle catches the fish", "The fish is caught by the eagle"),
    ("The farmer grows the corn", "The corn is grown by the farmer"),
    ("The worker builds the wall", "The wall is built by the worker"),
]


def generate_number_pairs(n=500):
    """生成大规模数约束句子对"""
    pairs = []
    for noun in NOUNS_SINGULAR[:n]:
        if noun not in NOUNS_PLURAL:
            continue
        plural = NOUNS_PLURAL[noun]
        for verb_s in VERBS_SINGULAR[:5]:  # 每个名词配5个动词
            if verb_s not in VERBS_PLURAL:
                continue
            verb_p = VERBS_PLURAL[verb_s]
            sg = f"The {noun} {verb_s}"
            pl = f"The {plural} {verb_p}"
            pairs.append((sg, pl))
            if len(pairs) >= n:
                return pairs
    return pairs


# ===== 工具函数 =====
def compute_kl(p_logits, q_logits):
    """KL(p || q)"""
    p = torch.softmax(p_logits.float(), dim=-1)
    q = torch.softmax(q_logits.float(), dim=-1)
    q = q.clamp(min=1e-10)
    p = p.clamp(min=1e-10)
    return float((p * (p / q).log()).sum().item())

def get_hidden_states_last(model, tokenizer, device, text, n_layers):
    """获取所有层的hidden states（最后一个token位置）"""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    
    h_states = []
    for l in range(n_layers + 1):
        h = out.hidden_states[l][0, -1].detach().float().cpu()
        h_states.append(h)
    
    return h_states


# ===== 实验1: 约束方向跨句子稳定性（★★★★★ 核心）=====
def experiment_constraint_stability(model, tokenizer, device, model_info, n_pairs=500):
    """
    核心实验：验证"约束方向"是否跨句子稳定
    
    方法：
    1. 收集n_pairs个句子对的 d_i = h_correct_i - h_wrong_i
    2. 在中间层采样层，计算所有cos(d_i, d_j)的分布
    3. 如果mean cos > 0.5 → 稳定约束方向存在
       如果mean cos < 0.2 → 不存在
    
    同时分析：
    - 约束方向的PCA结构
    - 前1个主成分的解释方差（如果>50% → 约束方向高度一致）
    - 约束方向与均值方向的cos
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    log_status(f"[Stability] n_layers={n_layers}, d_model={d_model}, n_pairs={n_pairs}")
    
    pairs = generate_number_pairs(n_pairs)
    log_status(f"[Stability] Generated {len(pairs)} number constraint pairs")
    
    # 采样几个关键层（首、中间1/3、中间1/2、中间2/3、尾）
    key_layers = get_sample_layers(n_layers, n_samples=6)
    # 只取中间层（跳过L0和最后一层，因为已知非线性）
    key_layers = [l for l in key_layers if 2 <= l <= n_layers - 3]
    log_status(f"[Stability] Key layers: {key_layers}")
    
    # 收集所有句子对的约束方向
    all_directions = {l: [] for l in key_layers}
    valid_pairs = []
    
    for i, (sg, pl) in enumerate(pairs):
        if i % 50 == 0:
            maybe_log(f"[Stability] Processing pair {i+1}/{len(pairs)}")
        
        try:
            res_sg = get_hidden_states_last(model, tokenizer, device, sg, n_layers)
            res_pl = get_hidden_states_last(model, tokenizer, device, pl, n_layers)
            
            for l in key_layers:
                d = (res_sg[l] - res_pl[l]).numpy()
                all_directions[l].append(d)
            
            valid_pairs.append((sg, pl))
            
            # 定期清理GPU
            if i % 20 == 0:
                torch.cuda.empty_cache()
                
        except Exception as e:
            if i < 5:
                log_status(f"  Error pair {i}: {e}")
            continue
    
    log_status(f"[Stability] Collected {len(valid_pairs)} valid pairs")
    
    if len(valid_pairs) < 30:
        log_status(f"[Stability] Not enough valid pairs ({len(valid_pairs)}), aborting")
        return {"error": "insufficient_data", "n_valid": len(valid_pairs)}
    
    results = {"per_layer": [], "overall": {}, "n_valid_pairs": len(valid_pairs)}
    
    for l in key_layers:
        directions = np.array(all_directions[l])  # [n_pairs, d_model]
        n = len(directions)
        if n < 30:
            continue
        
        log_status(f"[Stability] Analyzing L{l}, n={n} directions...")
        
        try:
            # ===== 1. 跨句子cos分布 =====
            # 采样cos计算（全配对O(n^2)太大，随机采样n_sample对）
            n_sample = min(5000, n * (n - 1) // 2)
            
            # 归一化
            norms = np.linalg.norm(directions, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            directions_norm = directions / norms  # [n, d_model]
            
            # 随机采样配对的cos
            np.random.seed(42)
            idx_i = np.random.randint(0, n, size=n_sample)
            idx_j = np.random.randint(0, n, size=n_sample)
            # 排除自配对
            mask = idx_i != idx_j
            idx_i = idx_i[mask]
            idx_j = idx_j[mask]
            
            cos_values = np.sum(directions_norm[idx_i] * directions_norm[idx_j], axis=1)
            abs_cos_values = np.abs(cos_values)
            
            # ===== 2. PCA分析 =====
            # 中心化
            directions_centered = directions - directions.mean(axis=0, keepdims=True)
            U, S, Vt = np.linalg.svd(directions_centered, full_matrices=False)
            
            explained_var = S**2 / (np.sum(S**2) + 1e-10)
            cumulative_var = np.cumsum(explained_var)
            
            n_90 = int(np.searchsorted(cumulative_var, 0.9)) + 1
            n_95 = int(np.searchsorted(cumulative_var, 0.95)) + 1
            
            # 第一主成分解释方差
            pc1_var = float(explained_var[0])
            top5_var = float(np.sum(explained_var[:5]))
            top10_var = float(np.sum(explained_var[:10]))
            
            # ===== 3. 均值方向cos =====
            mean_dir = directions.mean(axis=0)
            mean_dir_norm = mean_dir / (np.linalg.norm(mean_dir) + 1e-10)
            cos_with_mean = np.sum(directions_norm * mean_dir_norm, axis=1)
            abs_cos_with_mean = np.abs(cos_with_mean)
            
            # ===== 4. 约束方向范数分布 =====
            dir_norms = np.linalg.norm(directions, axis=1)
            
            result = {
                "layer": l,
                "n_directions": n,
                # cos分布
                "cos_mean": float(np.mean(cos_values)),
                "cos_std": float(np.std(cos_values)),
                "abs_cos_mean": float(np.mean(abs_cos_values)),
                "abs_cos_median": float(np.median(abs_cos_values)),
                "abs_cos_p90": float(np.percentile(abs_cos_values, 90)),
                "abs_cos_p95": float(np.percentile(abs_cos_values, 95)),
                "cos_positive_ratio": float(np.mean(cos_values > 0)),
                # PCA
                "pc1_variance": pc1_var,
                "top5_pc_variance": top5_var,
                "top10_pc_variance": top10_var,
                "n_components_90pct": n_90,
                "n_components_95pct": n_95,
                # 均值方向
                "cos_with_mean_mean": float(np.mean(abs_cos_with_mean)),
                "cos_with_mean_median": float(np.median(abs_cos_with_mean)),
                "cos_with_mean_p90": float(np.percentile(abs_cos_with_mean, 90)),
                # 范数
                "dir_norm_mean": float(np.mean(dir_norms)),
                "dir_norm_std": float(np.std(dir_norms)),
                "dir_norm_cv": float(np.std(dir_norms) / (np.mean(dir_norms) + 1e-10)),
            }
            results["per_layer"].append(result)
            
            # 关键判断
            stability = "STABLE" if abs_cos_values.mean() > 0.5 else ("MODERATE" if abs_cos_values.mean() > 0.2 else "UNSTABLE")
            
            log_status(f"  L{l}: |cos|_mean={abs_cos_values.mean():.3f}, "
                       f"|cos|_median={np.median(abs_cos_values):.3f}, "
                       f"PC1={pc1_var:.1%}, "
                       f"cos_mean_dir={abs_cos_with_mean.mean():.3f}, "
                       f"stability={stability}")
            
        except Exception as e:
            log_status(f"  Error L{l}: {e}")
            import traceback; traceback.print_exc()
            continue
    
    # 汇总
    if results["per_layer"]:
        mean_abs_cos = np.mean([r["abs_cos_mean"] for r in results["per_layer"]])
        mean_pc1 = np.mean([r["pc1_variance"] for r in results["per_layer"]])
        mean_cos_mean_dir = np.mean([r["cos_with_mean_mean"] for r in results["per_layer"]])
        mean_n90 = np.mean([r["n_components_90pct"] for r in results["per_layer"]])
        
        overall_stability = "STABLE" if mean_abs_cos > 0.5 else ("MODERATE" if mean_abs_cos > 0.2 else "UNSTABLE")
        
        results["overall"] = {
            "mean_abs_cos": float(mean_abs_cos),
            "mean_pc1_variance": float(mean_pc1),
            "mean_cos_with_mean_direction": float(mean_cos_mean_dir),
            "mean_n_components_90pct": float(mean_n90),
            "constraint_direction_stable": overall_stability == "STABLE",
            "stability_verdict": overall_stability,
        }
        
        log_status(f"[Stability] OVERALL: |cos|={mean_abs_cos:.3f}, "
                   f"PC1={mean_pc1:.1%}, cos_mean_dir={mean_cos_mean_dir:.3f}, "
                   f"verdict={overall_stability}")
    
    return results


# ===== 实验2: 不同约束类型的稳定性对比 =====
def experiment_constraint_type_stability(model, tokenizer, device, model_info):
    """
    对比不同约束类型的方向稳定性：
    - 数约束（number）
    - 时态约束（tense）  
    - 语态约束（voice）
    
    如果数约束稳定但其他约束不稳定 → 稳定性是约束类型特异的
    如果所有约束都稳定 → 存在通用的约束方向结构
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    log_status(f"[TypeStability] Starting multi-constraint-type analysis")
    
    # 关键中间层
    mid_layer = n_layers // 2
    
    constraint_types = {
        "number": generate_number_pairs(50),  # 数约束
        "tense": TENSE_PAIRS[:50],            # 时态约束
        "voice": VOICE_PAIRS[:30],            # 语态约束
    }
    
    results = {}
    
    for ctype, pairs in constraint_types.items():
        log_status(f"[TypeStability] Processing {ctype}: {len(pairs)} pairs")
        
        directions = []
        for i, (sg, pl) in enumerate(pairs):
            maybe_log(f"[TypeStability] {ctype} pair {i+1}/{len(pairs)}")
            try:
                res_sg = get_hidden_states_last(model, tokenizer, device, sg, n_layers)
                res_pl = get_hidden_states_last(model, tokenizer, device, pl, n_layers)
                d = (res_sg[mid_layer] - res_pl[mid_layer]).numpy()
                directions.append(d)
            except Exception as e:
                continue
        
        if len(directions) < 10:
            results[ctype] = {"error": "insufficient_data", "n": len(directions)}
            continue
        
        directions = np.array(directions)
        n = len(directions)
        
        # 归一化
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        directions_norm = directions / norms
        
        # 跨句子cos
        n_sample = min(2000, n * (n - 1) // 2)
        np.random.seed(42)
        idx_i = np.random.randint(0, n, size=n_sample)
        idx_j = np.random.randint(0, n, size=n_sample)
        mask = idx_i != idx_j
        idx_i, idx_j = idx_i[mask], idx_j[mask]
        
        cos_values = np.sum(directions_norm[idx_i] * directions_norm[idx_j], axis=1)
        abs_cos_values = np.abs(cos_values)
        
        # PCA
        directions_centered = directions - directions.mean(axis=0, keepdims=True)
        _, S, _ = np.linalg.svd(directions_centered, full_matrices=False)
        explained_var = S**2 / (np.sum(S**2) + 1e-10)
        pc1_var = float(explained_var[0])
        top5_var = float(np.sum(explained_var[:5]))
        
        # 均值方向cos
        mean_dir = directions.mean(axis=0)
        mean_dir_norm = mean_dir / (np.linalg.norm(mean_dir) + 1e-10)
        cos_with_mean = np.abs(np.sum(directions_norm * mean_dir_norm, axis=1))
        
        stability = "STABLE" if abs_cos_values.mean() > 0.5 else ("MODERATE" if abs_cos_values.mean() > 0.2 else "UNSTABLE")
        
        results[ctype] = {
            "n_directions": n,
            "abs_cos_mean": float(np.mean(abs_cos_values)),
            "abs_cos_median": float(np.median(abs_cos_values)),
            "cos_positive_ratio": float(np.mean(cos_values > 0)),
            "pc1_variance": pc1_var,
            "top5_pc_variance": top5_var,
            "cos_with_mean_mean": float(np.mean(cos_with_mean)),
            "stability": stability,
        }
        
        log_status(f"  {ctype}: |cos|={abs_cos_values.mean():.3f}, "
                   f"PC1={pc1_var:.1%}, cos_mean_dir={cos_with_mean.mean():.3f}, "
                   f"stability={stability}")
    
    return results


# ===== 实验3: 约束方向与W_U行空间的对齐 =====
def experiment_constraint_wu_alignment(model, tokenizer, device, model_info, n_pairs=100):
    """
    分析约束方向与W_U行空间的对齐度
    
    如果约束方向与W_U行空间高度对齐 → 约束信号被W_U放大
    如果约束方向与W_U行空间正交 → 约束信号可能由其他机制传递
    
    同时：约束子空间 vs 语义子空间的正交性
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    log_status(f"[WUAlign] n_layers={n_layers}, d_model={d_model}, n_pairs={n_pairs}")
    
    # 获取W_U
    W_U = get_W_U(model, model_info.name)  # [vocab_size, d_model]
    log_status(f"[WUAlign] W_U shape: {W_U.shape}")
    
    # W_U的SVD（前100个分量）
    from scipy.sparse.linalg import svds
    W_U_T = W_U.T.astype(np.float32)  # [d_model, vocab]
    k = min(100, min(W_U_T.shape) - 2)
    U_wu, S_wu, Vt_wu = svds(W_U_T, k=k)
    U_wu = np.asarray(U_wu, dtype=np.float64)  # [d_model, k]
    
    log_status(f"[WUAlign] W_U SVD: top-100 singular values, first 5: {S_wu[-5:][::-1]}")
    
    # 收集数约束方向
    pairs = generate_number_pairs(n_pairs)
    mid_layer = n_layers // 2
    
    directions = []
    for i, (sg, pl) in enumerate(pairs):
        maybe_log(f"[WUAlign] pair {i+1}/{len(pairs)}")
        try:
            res_sg = get_hidden_states_last(model, tokenizer, device, sg, n_layers)
            res_pl = get_hidden_states_last(model, tokenizer, device, pl, n_layers)
            d = (res_sg[mid_layer] - res_pl[mid_layer]).numpy()
            directions.append(d)
        except Exception as e:
            continue
    
    if len(directions) < 20:
        return {"error": "insufficient_data", "n": len(directions)}
    
    directions = np.array(directions)  # [n, d_model]
    log_status(f"[WUAlign] Collected {len(directions)} directions")
    
    try:
        # ===== 1. 约束方向在W_U行空间中的投影比 =====
        proj_ratios = []
        for d in directions:
            d_norm = np.linalg.norm(d)
            if d_norm < 1e-10:
                continue
            proj = U_wu @ (U_wu.T @ d)
            proj_ratio = np.linalg.norm(proj) / d_norm
            proj_ratios.append(proj_ratio)
        
        # ===== 2. 约束子空间PCA vs W_U行空间 =====
        directions_centered = directions - directions.mean(axis=0, keepdims=True)
        U_dir_svd, S_dir, Vt_dir = np.linalg.svd(directions_centered, full_matrices=False)
        # Vt_dir的行是主成分方向 [min(n,d_model), d_model]
        # U_dir_svd的列是样本空间的方向 [n, min(n,d_model)]
        
        # 前30个约束方向PC（在Vt中）与W_U行空间的对齐
        top_k = min(30, Vt_dir.shape[0])
        alignment_per_pc = []
        for pc_idx in range(top_k):
            pc = Vt_dir[pc_idx]  # [d_model] — 约束子空间的主成分
            # PC在W_U行空间中的投影比
            proj = U_wu @ (U_wu.T @ pc)
            alignment = np.linalg.norm(proj) / (np.linalg.norm(pc) + 1e-10)
            alignment_per_pc.append(float(alignment))
        
        # ===== 3. 约束子空间与W_U行空间的子空间角度 =====
        # 取约束前10个PC和W_U前30个分量
        V_dir_top = Vt_dir[:10].T  # [d_model, 10] — 约束子空间基
        U_wu_top = U_wu[:, :30]    # [d_model, 30] — W_U行空间基
        
        # 子空间投影矩阵
        P_dir = V_dir_top @ V_dir_top.T  # [d_model, d_model]
        P_wu = U_wu_top @ U_wu_top.T     # [d_model, d_model]
        
        # 子空间重叠 = ||P_dir @ P_wu||_F / ||P_dir||_F
        overlap = np.linalg.norm(P_dir @ P_wu, 'fro') / np.linalg.norm(P_dir, 'fro')
        
        # ===== 4. 随机基准 =====
        np.random.seed(42)
        random_proj_ratios = []
        for _ in range(1000):
            v = np.random.randn(d_model)
            v = v / np.linalg.norm(v)
            proj = U_wu @ (U_wu.T @ v)
            random_proj_ratios.append(np.linalg.norm(proj))
        
        results = {
            "n_directions": len(directions),
            "mid_layer": mid_layer,
            # 投影比
            "proj_ratio_mean": float(np.mean(proj_ratios)),
            "proj_ratio_median": float(np.median(proj_ratios)),
            "proj_ratio_std": float(np.std(proj_ratios)),
            # 随机基准
            "random_proj_ratio_mean": float(np.mean(random_proj_ratios)),
            "random_proj_ratio_std": float(np.std(random_proj_ratios)),
            # 相对于随机基准的提升
            "proj_ratio_signal_to_noise": float(
                (np.mean(proj_ratios) - np.mean(random_proj_ratios)) / 
                max(np.std(random_proj_ratios), 1e-10)
            ),
            # PC对齐
            "pc_alignment_top5": alignment_per_pc[:5],
            "pc_alignment_mean_top10": float(np.mean(alignment_per_pc[:10])),
            "pc_alignment_mean_top30": float(np.mean(alignment_per_pc[:30])),
            # 子空间重叠
            "subspace_overlap_dir10_wu30": float(overlap),
            # PCA信息
            "pc1_variance": float(S_dir[0]**2 / (np.sum(S_dir**2) + 1e-10)),
            "top5_pc_variance": float(np.sum(S_dir[:5]**2) / (np.sum(S_dir**2) + 1e-10)),
        }
        
        log_status(f"[WUAlign] proj_ratio: mean={np.mean(proj_ratios):.3f}, "
                   f"random={np.mean(random_proj_ratios):.3f}, "
                   f"SNR={results['proj_ratio_signal_to_noise']:.1f}σ")
        log_status(f"[WUAlign] PC alignment top10={np.mean(alignment_per_pc[:10]):.3f}, "
                   f"subspace_overlap={overlap:.3f}")
        
    except Exception as e:
        log_status(f"[WUAlign] Error: {e}")
        import traceback; traceback.print_exc()
        results = {"error": str(e)}
    
    return results


# ===== 实验4: 约束方向分解——纯约束 vs 混杂信号 =====
def experiment_constraint_decomposition(model, tokenizer, device, model_info, n_pairs=100):
    """
    分解d_constraint中的信号成分
    
    核心问题：d = h_correct - h_wrong 中有多少是"纯约束"信号？
    
    方法：
    1. 固定名词，只变动词数（"The cat chases" vs "The cat chase"）
       → d_verb_only = 纯动词数约束
    
    2. 固定动词，只变名词数（"The cat chases" vs "The cats chases"）
       → d_noun_only = 名词数变化（不是语法约束！）
    
    3. 标准SVA变化（"The cat chases" vs "The cats chase"）
       → d_sva = 名词+动词同时变化
    
    如果d_verb_only和d_sva高度相似 → 约束方向主要由动词数约束决定
    如果d_noun_only和d_verb_only正交 → 名词数变化和动词数约束是不同信号
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    log_status(f"[Decompose] n_layers={n_layers}, d_model={d_model}")
    
    mid_layer = n_layers // 2
    
    # 使用50个名词×3个动词组合
    test_nouns = NOUNS_SINGULAR[:50]
    test_verbs = ["chases", "runs", "sings"]
    
    verb_only_dirs = []  # 动词数约束
    noun_only_dirs = []  # 名词数变化
    sva_dirs = []        # 完整SVA
    
    count = 0
    for noun in test_nouns:
        if noun not in NOUNS_PLURAL:
            continue
        plural = NOUNS_PLURAL[noun]
        for verb_s in test_verbs:
            if verb_s not in VERBS_PLURAL:
                continue
            verb_p = VERBS_PLURAL[verb_s]
            
            maybe_log(f"[Decompose] {noun}/{verb_s} ({count+1})")
            
            try:
                # 标准SVA: "The cat chases" vs "The cats chase"
                h_sva_sg = get_hidden_states_last(model, tokenizer, device,
                                                   f"The {noun} {verb_s}", n_layers)
                h_sva_pl = get_hidden_states_last(model, tokenizer, device,
                                                   f"The {plural} {verb_p}", n_layers)
                d_sva = (h_sva_sg[mid_layer] - h_sva_pl[mid_layer]).numpy()
                sva_dirs.append(d_sva)
                
                # 动词数约束: "The cat chases" vs "The cat chase"
                h_verb_sg = get_hidden_states_last(model, tokenizer, device,
                                                    f"The {noun} {verb_s}", n_layers)
                h_verb_pl = get_hidden_states_last(model, tokenizer, device,
                                                    f"The {noun} {verb_p}", n_layers)
                d_verb = (h_verb_sg[mid_layer] - h_verb_pl[mid_layer]).numpy()
                verb_only_dirs.append(d_verb)
                
                # 名词数变化: "The cat chases" vs "The cats chases"（语法错误句）
                h_noun_sg = get_hidden_states_last(model, tokenizer, device,
                                                    f"The {noun} {verb_s}", n_layers)
                h_noun_pl = get_hidden_states_last(model, tokenizer, device,
                                                    f"The {plural} {verb_s}", n_layers)
                d_noun = (h_noun_sg[mid_layer] - h_noun_pl[mid_layer]).numpy()
                noun_only_dirs.append(d_noun)
                
                count += 1
                if count >= 80:
                    break
                
            except Exception as e:
                continue
        if count >= 80:
            break
    
    log_status(f"[Decompose] Collected {len(sva_dirs)} triplets")
    
    if len(sva_dirs) < 20:
        return {"error": "insufficient_data", "n": len(sva_dirs)}
    
    sva_dirs = np.array(sva_dirs)
    verb_dirs = np.array(verb_only_dirs)
    noun_dirs = np.array(noun_only_dirs)
    
    try:
        # ===== 1. SVA vs 动词数约束的相似度 =====
        # 归一化
        sva_norms = np.linalg.norm(sva_dirs, axis=1, keepdims=True)
        verb_norms = np.linalg.norm(verb_dirs, axis=1, keepdims=True)
        noun_norms = np.linalg.norm(noun_dirs, axis=1, keepdims=True)
        
        sva_hat = sva_dirs / np.maximum(sva_norms, 1e-10)
        verb_hat = verb_dirs / np.maximum(verb_norms, 1e-10)
        noun_hat = noun_dirs / np.maximum(noun_norms, 1e-10)
        
        # 配对cos（同一句子对的不同分解）
        cos_sva_verb = np.sum(sva_hat * verb_hat, axis=1)
        cos_sva_noun = np.sum(sva_hat * noun_hat, axis=1)
        cos_verb_noun = np.sum(verb_hat * noun_hat, axis=1)
        
        # ===== 2. 线性分解 =====
        # SVA方向是否可以由动词+名词方向线性表示？
        # d_sva ≈ α * d_verb + β * d_noun
        # 用最小二乘法估计α和β
        alphas = []
        betas = []
        residuals = []
        for i in range(len(sva_dirs)):
            d_sva = sva_dirs[i]
            d_verb = verb_dirs[i]
            d_noun = noun_dirs[i]
            
            # [d_verb, d_noun] @ [α, β]^T ≈ d_sva
            A = np.column_stack([d_verb, d_noun])  # [d_model, 2]
            coeffs, res, _, _ = np.linalg.lstsq(A, d_sva, rcond=None)
            alphas.append(coeffs[0])
            betas.append(coeffs[1])
            residuals.append(float(np.linalg.norm(d_sva - A @ coeffs) / (np.linalg.norm(d_sva) + 1e-10)))
        
        # ===== 3. 范数比 =====
        verb_sva_norm_ratio = np.mean(verb_norms.flatten()) / (np.mean(sva_norms.flatten()) + 1e-10)
        noun_sva_norm_ratio = np.mean(noun_norms.flatten()) / (np.mean(sva_norms.flatten()) + 1e-10)
        
        results = {
            "n_triplets": len(sva_dirs),
            "mid_layer": mid_layer,
            # 配对cos
            "cos_sva_verb_mean": float(np.mean(cos_sva_verb)),
            "cos_sva_verb_median": float(np.median(cos_sva_verb)),
            "cos_sva_noun_mean": float(np.mean(cos_sva_noun)),
            "cos_sva_noun_median": float(np.median(cos_sva_noun)),
            "cos_verb_noun_mean": float(np.mean(cos_verb_noun)),
            "cos_verb_noun_median": float(np.median(cos_verb_noun)),
            # 线性分解
            "alpha_mean": float(np.mean(alphas)),
            "alpha_median": float(np.median(alphas)),
            "beta_mean": float(np.mean(betas)),
            "beta_median": float(np.median(betas)),
            "residual_mean": float(np.mean(residuals)),
            "residual_median": float(np.median(residuals)),
            # 范数比
            "verb_sva_norm_ratio": float(verb_sva_norm_ratio),
            "noun_sva_norm_ratio": float(noun_sva_norm_ratio),
        }
        
        log_status(f"[Decompose] cos(sva,verb)={np.mean(cos_sva_verb):.3f}, "
                   f"cos(sva,noun)={np.mean(cos_sva_noun):.3f}, "
                   f"cos(verb,noun)={np.mean(cos_verb_noun):.3f}")
        log_status(f"[Decompose] α={np.mean(alphas):.3f}, β={np.mean(betas):.3f}, "
                   f"residual={np.mean(residuals):.3f}")
        
    except Exception as e:
        log_status(f"[Decompose] Error: {e}")
        import traceback; traceback.print_exc()
        results = {"error": str(e)}
    
    return results


# ===== 主函数 =====
def run_all_experiments(model_name: str):
    """运行所有实验"""
    log_status(f"{'='*60}")
    log_status(f"Phase 222: Constraint Direction Stability - {model_name}")
    log_status(f"{'='*60}")
    
    # 加载模型
    model, tokenizer, device = load_model_sdpa(model_name)
    model_info = get_model_info(model, model_name)
    
    log_status(f"Model: {model_info.model_class}, layers={model_info.n_layers}, "
               f"d_model={model_info.d_model}, mlp_type={model_info.mlp_type}")
    
    all_results = {
        "model_name": model_name,
        "model_info": {
            "class": model_info.model_class,
            "n_layers": model_info.n_layers,
            "d_model": model_info.d_model,
            "vocab_size": model_info.vocab_size,
            "mlp_type": model_info.mlp_type,
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # ===== 实验1: 约束方向跨句子稳定性 (★★★★★ 核心) =====
    log_status(f"\n{'='*40}")
    log_status("Exp 1: Constraint Direction Stability (P0)")
    log_status(f"{'='*40}")
    try:
        all_results["stability"] = experiment_constraint_stability(
            model, tokenizer, device, model_info, n_pairs=500
        )
    except Exception as e:
        log_status(f"Exp 1 failed: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)
    
    # ===== 实验2: 不同约束类型的稳定性对比 =====
    log_status(f"\n{'='*40}")
    log_status("Exp 2: Constraint Type Stability (P1)")
    log_status(f"{'='*40}")
    try:
        all_results["type_stability"] = experiment_constraint_type_stability(
            model, tokenizer, device, model_info
        )
    except Exception as e:
        log_status(f"Exp 2 failed: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)
    
    # ===== 实验3: 约束方向与W_U行空间对齐 =====
    log_status(f"\n{'='*40}")
    log_status("Exp 3: Constraint-W_U Alignment (P2)")
    log_status(f"{'='*40}")
    try:
        all_results["wu_alignment"] = experiment_constraint_wu_alignment(
            model, tokenizer, device, model_info, n_pairs=100
        )
    except Exception as e:
        log_status(f"Exp 3 failed: {e}")
        import traceback; traceback.print_exc()
    
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(5)
    
    # ===== 实验4: 约束方向分解 =====
    log_status(f"\n{'='*40}")
    log_status("Exp 4: Constraint Decomposition (P3)")
    log_status(f"{'='*40}")
    try:
        all_results["decomposition"] = experiment_constraint_decomposition(
            model, tokenizer, device, model_info, n_pairs=100
        )
    except Exception as e:
        log_status(f"Exp 4 failed: {e}")
        import traceback; traceback.print_exc()
    
    # 保存结果
    result_path = OUTPUT_DIR / f"phase222_{model_name}_results.json"
    
    def sanitize(obj):
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [sanitize(v) for v in obj]
        elif isinstance(obj, float):
            if np.isnan(obj) or np.isinf(obj):
                return None
            return obj
        elif isinstance(obj, np.floating):
            v = float(obj)
            if np.isnan(v) or np.isinf(v):
                return None
            return v
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return sanitize(obj.tolist())
        return obj
    
    all_results = sanitize(all_results)
    
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    log_status(f"Results saved to {result_path}")
    
    # 释放模型
    release_model(model)
    
    return all_results


if __name__ == "__main__":
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    if model_name == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            log_status(f"\n{'#'*60}")
            log_status(f"# Starting {name}")
            log_status(f"{'#'*60}")
            run_all_experiments(name)
            gc.collect()
            torch.cuda.empty_cache()
            log_status(f"Waiting 30s for GPU cleanup...")
            time.sleep(30)
    else:
        run_all_experiments(model_name)
    
    log_status("Phase 222 complete!")
