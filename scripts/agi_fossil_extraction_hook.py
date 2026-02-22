import torch
try:
    from transformer_lens import HookedTransformer
except ImportError:
    print("TransformerLens is not installed. Please install it with 'pip install transformer_lens'")
    exit(1)

def run_fossil_hooking_experiment():
    """
    化石逆向挖掘工程 Phase II: 真实大范围语料的残差流拦截
    """
    print("[AGI Foundation] 初始化 Transformer 化石探针...")
    
    # 1. 挂载透明解剖模型 (使用经典 GPT-2 Small, D=768)
    # 此处为节约时间，默认它已在缓存中。真实应用中可使用 device='cuda'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading gpt2-small onto {device}...")
    
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    
    # 2. 定位监听点
    # 我们拦截网络中部(Layer 6)，此时的残差流已形成复杂的语义和物理空间叠加态
    target_layer = 6
    hook_point = f"blocks.{target_layer}.hook_resid_post"
    print(f"[探针部署] 锁定化石残差主干流切面: {hook_point}")
    
    # 3. 准备高级认知语料 (强迫模型形成时空和因果逻辑)
    test_corpus = [
        "When the red apple fell from the tall tree, Newton immediately realized the hidden law of gravity.",
        "The small cat chased the fast mouse into the dark and narrow tunnel, but could not catch it.",
        "Although AGI sounds like a myth, mathematicians believe it is just a pure manifold topology structure."
    ]
    
    print("\n[注入语料] 开始在化石内部诱发思想暗流...")
    
    # 4. 执行注射并进行缓存拦截
    # TransformerLens 的 run_with_cache 将自动捕获指定拦截点的特征矩阵
    # 我们只关心我们埋下的 hook_point
    all_activations = []
    
    for i, text in enumerate(test_corpus):
        # 截流过程
        logits, cache = model.run_with_cache(
            text, 
            names_filter=lambda name: name == hook_point # 只拦截这一层以节省内存
        )
        
        # 提取被截获的那块致密张量 
        # 尺寸为 [batch(1), seq_len, d_model(768)]
        layer_acts = cache[hook_point] 
        
        # 展平 batch, 保留 [seq_len, d_model] 以备将来送入 SAE 大厅
        layer_acts = layer_acts.squeeze(0)
        all_activations.append(layer_acts)
        
        print(f"  语料 {i+1} 拦截成功! 捕获了 {layer_acts.shape[0]} 个 Token 瞬间的特异波函数 (D={layer_acts.shape[1]}).")
        
    # 5. 生成化石切片集
    # 将所有的 [seq_len, 768] 拼接在一起，形成一个巨大的 [Total_Tokens, 768] 训练集
    fossil_slices = torch.cat(all_activations, dim=0)
    
    print(f"\n[二期工程收官] 共收集化石残差切片 (Activation Vectors): {fossil_slices.shape}")
    print("这批数据目前处于处于极度纠缠的 768 维降维坍缩态。")
    print("下一步: 将这批张量灌入 Phase I 组装的 100,000 维 Sparse Autoencoder (SAE) 稀疏字典中，还原宇宙极效正交基底！")

if __name__ == "__main__":
    run_fossil_hooking_experiment()
