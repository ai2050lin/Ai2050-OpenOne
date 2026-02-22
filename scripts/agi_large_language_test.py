import numpy as np

def run_large_language_cristallization():
    """
    化石演化工程 Phase XVI: 较大规模自然语言处理 (NLP) 纯代数发生器测试
    不使用任何Transformer或反向传播，仅凭两大极简偏微分物理定律：
    1. 第一定律 (Sanger 侧抑制) -> 自发提取高维词汇表的低维正交抽象语义 (Orthogonal Semantics)
    2. 第二定律 (Hebbian 衰减盆地) -> 自动在语义轴之间建构语法因果图谱 (Grammar Basin)
    并在测试阶段通过势能坍塌 O(1) 步预测残缺句子。
    """
    print("==========================================================")
    print("[AGI Foundation] 第十六期大规模语言验证：纯物理双定律的完形填空涌现")
    print("==========================================================\n")
    
    # ---------------------------------------------------------
    # 1. 构建大规模语境环境 (模拟真实人类沟通)
    # ---------------------------------------------------------
    # 定义 4 大主题，共 100 个词汇
    topics = {
        "Predators": ["狮子", "老虎", "狂狼", "猎豹", "老鹰", "鳄鱼", "鲨鱼", "毒蛇", "猎手", "野兽"],
        "Preys":     ["绵羊", "兔子", "羚羊", "斑马", "鹿", "老鼠", "青蛙", "小鱼", "虫子", "素食者"],
        "Actions":   ["撕咬", "追逐", "吞噬", "捕杀", "袭击", "吃掉", "嚼碎", "杀戮", "狩猎", "抓捕"],
        "Locations": ["草原", "森林", "丛林", "天空", "深海", "荒野", "沼泽", "洞穴", "平原", "海洋"],
        "Vehicles":  ["汽车", "火车", "飞机", "轮船", "卡车", "单车", "高铁", "飞船", "火箭", "大黄蜂"],
        "V_Actions": ["驾驶", "飞驰", "航行", "启动", "加速", "巡航", "飙车", "运输", "停泊", "超速"],
        "V_Places":  ["公路", "轨道", "航线", "港口", "车站", "天空", "高速", "车库", "加油站", "机场"],
        "Emotions":  ["开心", "愤怒", "恐惧", "悲伤", "兴奋", "绝望", "平静", "痛苦", "激动", "麻木"],
        "Humans":    ["医生", "司机", "老师", "学生", "工人", "警察", "厨师", "婴儿", "老人", "士兵"],
        "Tools":     ["手术刀", "方向盘", "粉笔", "书本", "扳手", "手枪", "菜刀", "奶瓶", "拐杖", "步枪"]
    }
    
    # 压平为总词典
    vocab = []
    for words in topics.values():
        vocab.extend(words)
    V_size = len(vocab) # 100维的词袋空间
    word2id = {w: i for i, w in enumerate(vocab)}
    
    print(f"-> 环境初始化：部署包含 {V_size} 个离散概念的真实语言词汇表。")
    print("-> 语料生成：按现实逻辑（如捕食者吃猎物、交通工具在公路运转）成建制合成 20,000 条混杂语句...")
    
    # 制造 20,000 条语句
    num_sentences = 20000
    sentences_multi_hot = np.zeros((num_sentences, V_size))
    
    np.random.seed(42)
    for i in range(num_sentences):
        event_type = np.random.choice(["Hunt", "Transport", "HumanTool"])
        active_words = []
        if event_type == "Hunt":
            active_words.append(np.random.choice(topics["Predators"]))
            active_words.append(np.random.choice(topics["Preys"]))
            active_words.append(np.random.choice(topics["Actions"]))
            active_words.append(np.random.choice(topics["Locations"]))
            active_words.append(np.random.choice(topics["Emotions"])) # 可能会害怕或残忍
        elif event_type == "Transport":
            active_words.append(np.random.choice(topics["Vehicles"]))
            active_words.append(np.random.choice(topics["V_Actions"]))
            active_words.append(np.random.choice(topics["V_Places"]))
        else: # HumanTool
            human = np.random.choice(topics["Humans"])
            tool = topics["Tools"][topics["Humans"].index(human)] # 强行绑定特定职业和工具
            active_words.extend([human, tool])
            active_words.append(np.random.choice(topics["Emotions"]))
            
        # 增加随机语言白噪音 (虚词、罕见词乱入)
        for _ in range(2):
            active_words.append(vocab[np.random.randint(0, V_size)])
            
        # 词袋 One-hot 编码
        for w in active_words:
            sentences_multi_hot[i, word2id[w]] += 1
            
    # 标准化输入
    sentences_multi_hot = np.clip(sentences_multi_hot, 0, 1)

    # ---------------------------------------------------------
    # 2. 第一定律起效：提取正交隐层抽象语义 (The Orthogonal Semantic V1)
    # ---------------------------------------------------------
    N_hidden = 15 # 假设我们划拨 15 个空白的“高级语义核”
    W_enc = np.random.randn(N_hidden, V_size) * 0.01 
    
    lr1_enc = 0.05
    print(f"\n[过程A：大自然物理冲洗] {N_hidden} 个空白突触开始经历 20,000 次第一定律 (Sanger 侧抑制) 冲刷...")
    # 这个循环在逼迫原本瞎猜的突触，自动认出词汇背后隐含的【主题类别】
    for x_in in sentences_multi_hot:
        y_hid = W_enc.dot(x_in)
        dW = np.zeros_like(W_enc)
        for i in range(N_hidden):
            # Hebbian - 侧竞争抑制
            inhibition = np.zeros(V_size)
            for j in range(i + 1):
                inhibition += y_hid[j] * W_enc[j] * y_hid[i]
            dW[i] = y_hid[i] * x_in - inhibition
        W_enc += lr1_enc * dW
        
    print("-> 隐层抽象提取完毕！")
    
    # ---------------------------------------------------------
    # 3. 第二定律起效：在隐层空间雕刻常识引力地形 (The Semantic Laplacian Basin)
    # ---------------------------------------------------------
    # 词汇太多了，我们用刚才提取出的 15 个正交主题来构建因果图谱更有效率
    P_topo = np.zeros((N_hidden, N_hidden))
    lr2_topo = 0.005
    decay = 0.001
    
    print(f"\n[过程B：逻辑引力下陷] 语义网在隐层空间经历第二定律 (带有遗忘的热力学拓扑共现) 的自然腐蚀...")
    # 将同一批次语料投射到刚刚长出来的语义空间中
    H_stream = sentences_multi_hot.dot(W_enc.T) 
    # 只取高光激活
    H_stream = (H_stream > np.mean(H_stream)) * H_stream 
    
    for h_in in H_stream:
        # Hebbian 外积共现与全局热力衰减
        dP = np.outer(h_in, h_in)
        P_topo += lr2_topo * dP
        P_topo -= decay * P_topo
        np.fill_diagonal(P_topo, 0)
        
    print("-> 语义深渊地形雕刻完毕！\n")
    
    # ---------------------------------------------------------
    # 4. 终极验证：O(1) 数学相变完形填空推理
    # ---------------------------------------------------------
    print("==========================================================")
    print("       模型智力测试：完形填空/逻辑联想 (Zero-shot)")
    print("==========================================================")
    
    def zero_shot_inference(query_words):
        input_wave = np.zeros(V_size)
        for w in query_words:
            if w in word2id:
                input_wave[word2id[w]] = 1.0
        
        # 1. 投射至高维抽象空间
        h_wave = W_enc.dot(input_wave)
        
        # 2. 顺着引力地形做一次物理跌落（O(1) 相变坍塌）
        h_collapsed = h_wave + P_topo.dot(h_wave) * 2.0
        
        # 3. 把坍塌结果逆投影回物理单词表
        output_wave = W_enc.T.dot(h_collapsed)
        
        # 屏蔽输入词，看看联想出了什么
        for w in query_words:
             if w in word2id:
                 output_wave[word2id[w]] = 0.0
                 
        top_k_idx = np.argsort(output_wave)[::-1][:5]
        top_words = [(vocab[i], output_wave[i]) for i in top_k_idx]
        
        print(f"输入残缺波 (上下文): {query_words}")
        print(f"极速涌现联想 (Top 5):")
        for word, score in top_words:
            print(f"  - {word} (引力势: {score:.2f})")
        print("-" * 50)

    # 考题 1：捕猎逻辑
    zero_shot_inference(["狮子", "吃掉"])
    
    # 考题 2：交通逻辑
    zero_shot_inference(["天空", "飞机"])
    
    # 考题 3：工具职业绑定逻辑（极难，因为语料非常稀疏，被白噪包裹）
    zero_shot_inference(["医生", "开心"]) 


if __name__ == "__main__":
    run_large_language_cristallization()
