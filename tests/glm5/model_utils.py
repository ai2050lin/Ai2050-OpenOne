"""
标准多模型测试框架 - 兼容 Qwen3/GLM4/DeepSeek7B
=================================================
用途: 所有后续实验脚本可import此模块, 无需重复处理模型兼容性问题

核心功能:
1. model_utils.load_model(name) -> 加载模型+tokenizer
2. model_utils.get_layers(model) -> 获取transformer层列表
3. model_utils.get_layer_weights(layer) -> 获取层的权重矩阵(兼容不同架构)
4. model_utils.get_model_info(model) -> 获取模型基本信息
5. model_utils.release_model(model) -> 释放GPU内存

兼容性处理:
- Qwen3 (Qwen3ForCausalLM): model.model.layers, q/k/v/o_proj, up/down/gate_proj
- DeepSeek7B (Qwen2ForCausalLM): model.model.layers, q/k/v/o_proj, up/down/gate_proj
- GLM4 (GlmForCausalLM/Glm4ForCausalLM): model.model.layers, q/k/v/o_proj, gate_up_proj/down_proj
"""

import torch
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ===== 模型配置 =====
MODEL_CONFIGS = {
    "qwen3": {
        "path": "D:/develop/model/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c",
        "arch": "Qwen3ForCausalLM",
        "mlp_type": "split_gate_up",  # gate_proj + up_proj 分开
    },
    "glm4": {
        "path": "D:/develop/model/hub/modelscope_cache/ZhipuAI/glm-4-9b-chat-hf",
        "arch": "GlmForCausalLM",  # 也可能是Glm4ForCausalLM
        "mlp_type": "merged_gate_up",  # gate_up_proj 合并
    },
    "deepseek7b": {
        "path": "D:/develop/model/hub/modelscope_cache/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "arch": "Qwen2ForCausalLM",
        "mlp_type": "split_gate_up",  # gate_proj + up_proj 分开
    },
}


@dataclass
class ModelInfo:
    """模型信息数据类"""
    name: str
    n_layers: int
    d_model: int
    vocab_size: int
    mlp_type: str  # "split_gate_up" or "merged_gate_up"
    intermediate_size: int = 0
    model_class: str = ""


@dataclass
class LayerWeights:
    """层的权重矩阵集合"""
    W_q: np.ndarray  # [d_model, d_model] 或 [d_model, d_kv*heads]
    W_k: np.ndarray
    W_v: np.ndarray
    W_o: np.ndarray
    W_up: Optional[np.ndarray] = None     # [intermediate, d_model] (split模式)
    W_down: np.ndarray = None              # [d_model, intermediate]
    W_gate: Optional[np.ndarray] = None    # [intermediate, d_model] (split模式)
    W_gate_up: Optional[np.ndarray] = None # [2*intermediate, d_model] (merged模式)
    input_layernorm_weight: Optional[np.ndarray] = None
    post_attn_layernorm_weight: Optional[np.ndarray] = None


def load_model(model_name: str, dtype=torch.bfloat16) -> Tuple:
    """
    加载模型和tokenizer (CUDA优先)
    
    策略: 先CPU加载(避免显存碎片), 再整体移到CUDA(避免device_map分散)
    
    Args:
        model_name: "qwen3", "glm4", or "deepseek7b"
        dtype: 模型数据类型, 默认float16
    
    Returns:
        (model, tokenizer, device) — device=cuda:0 如果可用
    """
    cfg = MODEL_CONFIGS[model_name]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"[model_utils] Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["path"],
        trust_remote_code=True,
        local_files_only=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 先加载到CPU, 再整体移到CUDA (比device_map="auto"更快更稳定)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["path"],
        torch_dtype=dtype,
        device_map="cpu",  # 先加载到CPU
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    # 整体移到CUDA (比device_map="auto"更快, 避免设备分散)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()
    device = next(model.parameters()).device
    print(f"[model_utils] {model_name} loaded, device={device}, "
          f"class={type(model).__name__}")
    return model, tokenizer, device


def get_layers(model) -> List:
    """
    获取transformer层列表(兼容不同模型架构)
    
    支持:
    - Qwen3/Qwen2: model.model.layers
    - GLM4: model.model.layers
    - GPT-2: model.transformer.h
    - BERT: model.model.encoder.layer
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        return model.model.encoder.layer
    raise ValueError(f"Cannot find transformer layers in {type(model).__name__}")


def get_model_info(model, model_name: str) -> ModelInfo:
    """
    获取模型基本信息
    """
    cfg = MODEL_CONFIGS[model_name]
    layers = get_layers(model)
    embed = model.get_input_embeddings()
    d_model = embed.weight.shape[1]
    
    # 获取intermediate_size
    layer0 = layers[0]
    intermediate_size = 0
    if hasattr(layer0, "mlp"):
        mlp = layer0.mlp
        if hasattr(mlp, "gate_up_proj"):
            intermediate_size = mlp.gate_up_proj.weight.shape[0] // 2
        elif hasattr(mlp, "up_proj"):
            intermediate_size = mlp.up_proj.weight.shape[0]
        elif hasattr(mlp, "dense_h_to_4h"):
            intermediate_size = mlp.dense_h_to_4h.weight.shape[0]
    
    vocab_size = 0
    if hasattr(model, "lm_head"):
        vocab_size = model.lm_head.weight.shape[0]
    elif hasattr(model, "embed_out"):
        vocab_size = model.embed_out.weight.shape[0]
    
    return ModelInfo(
        name=model_name,
        n_layers=len(layers),
        d_model=d_model,
        vocab_size=vocab_size,
        mlp_type=cfg["mlp_type"],
        intermediate_size=intermediate_size,
        model_class=type(model).__name__,
    )


def get_layer_weights(layer, d_model: int, mlp_type: str) -> LayerWeights:
    """
    获取单层的权重矩阵(兼容不同MLP架构)
    
    Args:
        layer: transformer层对象
        d_model: 模型维度
        mlp_type: "split_gate_up" 或 "merged_gate_up"
    
    Returns:
        LayerWeights 数据类
    """
    sa = layer.self_attn
    
    W_q = sa.q_proj.weight.detach().cpu().float().numpy()
    W_k = sa.k_proj.weight.detach().cpu().float().numpy()
    W_v = sa.v_proj.weight.detach().cpu().float().numpy()
    W_o = sa.o_proj.weight.detach().cpu().float().numpy()
    
    mlp = layer.mlp
    
    W_up = None
    W_gate = None
    W_gate_up = None
    
    if mlp_type == "split_gate_up":
        # Qwen3/Qwen2: gate_proj + up_proj + down_proj
        W_up = mlp.up_proj.weight.detach().cpu().float().numpy()
        W_down = mlp.down_proj.weight.detach().cpu().float().numpy()
        W_gate = mlp.gate_proj.weight.detach().cpu().float().numpy() if hasattr(mlp, 'gate_proj') else None
    elif mlp_type == "merged_gate_up":
        # GLM4: gate_up_proj (合并) + down_proj
        W_gate_up = mlp.gate_up_proj.weight.detach().cpu().float().numpy()
        W_down = mlp.down_proj.weight.detach().cpu().float().numpy()
        # 拆分为 gate 和 up
        W_gate = W_gate_up[:W_gate_up.shape[0]//2]
        W_up = W_gate_up[W_gate_up.shape[0]//2:]
    
    # LayerNorm
    input_ln_w = None
    post_attn_ln_w = None
    for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
        if hasattr(layer, ln_name):
            ln = getattr(layer, ln_name)
            if hasattr(ln, "weight"):
                input_ln_w = ln.weight.detach().cpu().float().numpy()
            break
    for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
        if hasattr(layer, ln_name):
            ln = getattr(layer, ln_name)
            if hasattr(ln, "weight"):
                post_attn_ln_w = ln.weight.detach().cpu().float().numpy()
            break
    
    return LayerWeights(
        W_q=W_q, W_k=W_k, W_v=W_v, W_o=W_o,
        W_up=W_up, W_down=W_down, W_gate=W_gate,
        W_gate_up=W_gate_up,
        input_layernorm_weight=input_ln_w,
        post_attn_layernorm_weight=post_attn_ln_w,
    )


def get_W_U(model) -> np.ndarray:
    """
    获取lm_head权重矩阵(即W_U)
    
    Returns:
        W_U: [vocab_size, d_model] numpy数组 (float32)
    """
    if hasattr(model, "lm_head"):
        w = model.lm_head.weight.detach().cpu()
        # 优先用float16节省内存, SVD时再cast到float32
        return w.float().numpy()
    raise ValueError(f"Cannot find lm_head in {type(model).__name__}")


def release_model(model):
    """释放模型GPU内存"""
    del model
    import gc; gc.collect()
    torch.cuda.empty_cache()
    print(f"[model_utils] GPU memory released, allocated={torch.cuda.memory_allocated()/1e9:.2f} GB")


def get_sample_layers(n_layers: int, n_samples: int = 10) -> List[int]:
    """
    获取采样层列表(均匀采样+首尾)
    """
    if n_layers <= n_samples:
        return list(range(n_layers))
    step = n_layers // n_samples
    layers = list(range(0, n_layers, step)) + [n_layers - 1]
    return sorted(set(layers))


def get_attr_direction(model, tokenizer, attr: str, W_U: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    获取属性的W_lm方向(归一化)
    
    Args:
        model: 模型对象
        tokenizer: 分词器
        attr: 属性词字符串 (如 "red")
        W_U: 可选预加载的W_U矩阵, 避免重复加载节省内存
    
    Returns:
        (direction_np, token_id) — direction_np为归一化的numpy数组[d_model]
    """
    attr_tok_ids = tokenizer.encode(attr, add_special_tokens=False)
    if len(attr_tok_ids) == 0:
        return None, None
    attr_tok_id = attr_tok_ids[0]
    if W_U is None:
        W_U = get_W_U(model)  # [vocab_size, d_model]
    direction = W_U[attr_tok_id].copy()
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    return direction, attr_tok_id


def inject_at_embed(model, tokenizer, device, prompt: str, direction: np.ndarray, 
                    beta: float = 8.0) -> Tuple:
    """
    在embedding层last_token位置注入方向
    
    Args:
        model: 模型对象
        tokenizer: 分词器
        device: 设备
        prompt: 提示文本 (如 "The apple is")
        direction: 注入方向 (numpy数组[d_model], 已归一化)
        beta: 注入强度, 默认8.0
    
    Returns:
        (inputs_embeds_base, inputs_embeds_intervened, input_ids, position_ids)
    """
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    
    embed_layer = model.get_input_embeddings()
    inputs_embeds_base = embed_layer(input_ids).detach().clone().to(model.dtype)
    
    direction_tensor = torch.tensor(direction, dtype=inputs_embeds_base.dtype, device=device)
    inputs_embeds_intervened = inputs_embeds_base.clone()
    inputs_embeds_intervened[0, -1, :] += (beta * direction_tensor).to(model.dtype)
    
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    
    return inputs_embeds_base, inputs_embeds_intervened, input_ids, position_ids


def collect_layer_outputs(model, inputs_embeds, position_ids, n_layers: int,
                          target_type: str = "layer") -> Dict:
    """
    用hook收集各层输出
    
    Args:
        model: 模型对象
        inputs_embeds: 输入embedding tensor
        position_ids: 位置ID tensor
        n_layers: 层数
        target_type: "layer" 收集Transformer层输出, "mlp" 收集MLP输出
    
    Returns:
        dict: {f"L{i}": tensor[1, seq_len, d_model]}
    """
    captured = {}
    layers = get_layers(model)
    
    def make_hook(key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                captured[key] = output[0].detach().float().cpu()
            else:
                captured[key] = output.detach().float().cpu()
        return hook
    
    hooks = []
    for li in range(n_layers):
        layer = layers[li]
        if target_type == "mlp":
            target = layer.mlp if hasattr(layer, "mlp") else None
            if target is None:
                continue
        else:
            target = layer
        hooks.append(target.register_forward_hook(make_hook(f"L{li}")))
    
    with torch.no_grad():
        try:
            _ = model(inputs_embeds=inputs_embeds, position_ids=position_ids)
        except Exception as e:
            print(f"  [collect_layer_outputs] Forward failed: {e}")
    
    for h in hooks:
        h.remove()
    
    return captured


def safe_decode(tokenizer, token_id):
    """Safely decode a token ID, handling tokenizer quirks."""
    try:
        result = tokenizer.decode([token_id])
        if result is None:
            return f"<tok_{token_id}>"
        return result
    except Exception:
        return f"<tok_{token_id}>"


def compute_cos(vec: np.ndarray, direction: np.ndarray) -> float:
    """计算向量与方向的余弦相似度"""
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return 0.0
    return float(np.dot(vec, direction) / norm)


def compute_recoding_ratio(delta_h: np.ndarray, W_U: np.ndarray, n_components: int = 50) -> Dict:
    """
    计算delta_h在W_U行空间中的投影比(recoding_ratio)
    
    Args:
        delta_h: 信号差 [d_model]
        W_U: lm_head权重 [vocab_size, d_model]
        n_components: SVD分量数
    
    Returns:
        dict: {ratio, gain, cos_inject_wu, proj_energy, top10_energy}
    """
    from scipy.sparse.linalg import svds
    
    delta_norm = np.linalg.norm(delta_h)
    if delta_norm < 1e-10:
        return {"ratio": 0, "gain": 0, "cos_inject_wu": 0, "proj_energy": 0, "top10_energy": 0}
    
    # SVD of W_U^T: 更稳定, W_U^T shape=[d_model, vocab] 远小于W_U
    # svds(W_U^T, k) → U [d_model, k] S [k] Vt [k, vocab]
    # W_U^T = U S Vt → W_U = Vt^T S U^T
    # W_U的行空间基 = Vt^T的列 (每个列是d_model维... 不对, Vt^T是[vocab, k])
    # W_U的行空间在R^d_model中, 由U的列张成
    # 因为W_U = Vt^T S U^T, 行空间 = {W_U x = Vt^T S U^T x} 
    #   W_U的值域(R^vocab)的预像在R^d_model中 = U^T的值域 = U的列空间
    # 所以delta在W_U行空间中的投影 = U @ (U^T @ delta)
    W_U_T = W_U.T.astype(np.float32)
    k = min(n_components, min(W_U_T.shape[0], W_U_T.shape[1]) - 2)
    k = max(k, 1)
    U_wut, s_wut, _ = svds(W_U_T, k=k)
    U_wut = np.asarray(U_wut, dtype=np.float64)  # [d_model, k]
    
    # delta在W_U行空间中的投影
    proj_coeffs = U_wut.T @ delta_h  # [k] — delta在U各列上的投影系数
    proj_energy = np.sum(proj_coeffs ** 2)
    
    # recoding_ratio = ||proj||^2 / ||delta||^2
    ratio = min(proj_energy / max(delta_norm ** 2, 1e-20), 1.0)
    gain = float(np.linalg.norm(proj_coeffs))
    
    top10_energy = float(np.sum(np.sort(proj_coeffs ** 2)[-10:])) if len(proj_coeffs) >= 10 else float(np.sum(proj_coeffs ** 2))
    
    return {
        "ratio": float(ratio),
        "gain": gain,
        "proj_energy": float(proj_energy),
        "top10_energy": top10_energy,
    }


def compute_recoding_ratio_cached(delta_h: np.ndarray, U_wut: np.ndarray) -> Dict:
    """
    计算delta_h在W_U行空间中的投影比(使用预计算的SVD结果)
    
    Args:
        delta_h: 信号差 [d_model]
        U_wut: svds(W_U^T, k)返回的U矩阵 [d_model, k] — W_U行空间基
    
    Returns:
        dict: {ratio, gain, proj_energy, top10_energy}
    """
    delta_norm = np.linalg.norm(delta_h)
    if delta_norm < 1e-10:
        return {"ratio": 0, "gain": 0, "proj_energy": 0, "top10_energy": 0}
    
    proj_coeffs = U_wut.T @ delta_h  # [k]
    proj_energy = np.sum(proj_coeffs ** 2)
    
    ratio = min(proj_energy / max(delta_norm ** 2, 1e-20), 1.0)
    gain = float(np.linalg.norm(proj_coeffs))
    
    top10_energy = float(np.sum(np.sort(proj_coeffs ** 2)[-10:])) if len(proj_coeffs) >= 10 else float(np.sum(proj_coeffs ** 2))
    
    return {
        "ratio": float(ratio),
        "gain": gain,
        "proj_energy": float(proj_energy),
        "top10_energy": top10_energy,
    }


# ===== 便捷函数: 一行加载+测试 =====
def quick_load_and_run(model_name: str, experiment_fn, **kwargs):
    """
    快速加载模型并运行实验函数
    
    Args:
        model_name: 模型名
        experiment_fn: 实验函数, 签名 fn(model, tokenizer, device, model_info, **kwargs) -> dict
    
    Returns:
        实验结果字典
    """
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    try:
        result = experiment_fn(model, tokenizer, device, model_info, **kwargs)
    finally:
        release_model(model)
    return result


if __name__ == "__main__":
    # 简单验证: 打印所有模型信息
    print("=" * 60)
    print("标准多模型测试框架 - 验证模式")
    print("=" * 60)
    
    for name in MODEL_CONFIGS:
        print(f"\n--- {name} ---")
        try:
            model, tokenizer, device = load_model(name)
            info = get_model_info(model, name)
            print(f"  class: {info.model_class}")
            print(f"  n_layers: {info.n_layers}")
            print(f"  d_model: {info.d_model}")
            print(f"  vocab_size: {info.vocab_size}")
            print(f"  mlp_type: {info.mlp_type}")
            print(f"  intermediate_size: {info.intermediate_size}")
            
            # 验证权重提取
            layer0 = get_layers(model)[0]
            lw = get_layer_weights(layer0, info.d_model, info.mlp_type)
            print(f"  W_q: {lw.W_q.shape}")
            print(f"  W_o: {lw.W_o.shape}")
            print(f"  W_up: {lw.W_up.shape if lw.W_up is not None else 'N/A'}")
            print(f"  W_gate: {lw.W_gate.shape if lw.W_gate is not None else 'N/A'}")
            print(f"  W_down: {lw.W_down.shape if lw.W_down is not None else 'N/A'}")
            
            W_U = get_W_U(model)
            print(f"  W_U: {W_U.shape}")
            
            release_model(model)
            print(f"  ✓ {name} 验证通过!")
            
        except Exception as e:
            print(f"  ✗ {name} 验证失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n验证完成!")
