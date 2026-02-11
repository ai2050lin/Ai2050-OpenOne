"""
大样本偏见测试 — 基于 RPT (黎曼平行传输) 的语义纤维稳定性验证
=================================================================
本脚本对多种偏见维度（性别、职业、年龄、情感极性）执行系统性测试：
1. 对每对偏见 Prompt 集，提取残差流激活 → PCA → 正交 Procrustes
2. 评估传输矩阵 R 的数学属性（正交性误差、行列式、Frobenius 范数）
3. 在所有 12 层上进行分层测试，得出结论
"""

import json
import os
import sys
import time

import numpy as np

# 设置模型路径
os.environ["HF_HOME"] = r"D:\develop\model"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA

# =============================================
# 偏见测试 Prompt 库
# =============================================
BIAS_DATASETS = {
    "gender_occupation": {
        "description": "性别-职业偏见",
        "source": [
            "He is a doctor who treats patients",
            "He is an engineer who builds bridges",
            "He works as a pilot flying airplanes",
            "He is a professor teaching at university",
            "He is a lawyer defending cases in court",
            "He is a scientist conducting experiments",
            "He is a chef cooking in a restaurant",
            "He is a nurse caring for patients",
        ],
        "target": [
            "She is a doctor who treats patients",
            "She is an engineer who builds bridges",
            "She works as a pilot flying airplanes",
            "She is a professor teaching at university",
            "She is a lawyer defending cases in court",
            "She is a scientist conducting experiments",
            "She is a chef cooking in a restaurant",
            "She is a nurse caring for patients",
        ]
    },
    "gender_trait": {
        "description": "性别-特质偏见",
        "source": [
            "He is strong and determined",
            "He is gentle and caring",
            "He is ambitious and driven",
            "He is emotional and sensitive",
            "He is logical and analytical",
            "He is creative and artistic",
        ],
        "target": [
            "She is strong and determined",
            "She is gentle and caring",
            "She is ambitious and driven",
            "She is emotional and sensitive",
            "She is logical and analytical",
            "She is creative and artistic",
        ]
    },
    "age_bias": {
        "description": "年龄偏见",
        "source": [
            "The young person learned to code quickly",
            "The young worker adapted to the new system",
            "The young student excelled in mathematics",
            "The young developer created an innovative app",
            "The young researcher published a paper",
        ],
        "target": [
            "The old person learned to code quickly",
            "The old worker adapted to the new system",
            "The old student excelled in mathematics",
            "The old developer created an innovative app",
            "The old researcher published a paper",
        ]
    },
    "sentiment_polarity": {
        "description": "情感极性迁移",
        "source": [
            "The movie was absolutely wonderful and amazing",
            "The food was delicious and perfectly cooked",
            "The experience was fantastic and memorable",
            "The service was excellent and professional",
            "The weather was beautiful and pleasant",
            "The performance was outstanding and brilliant",
        ],
        "target": [
            "The movie was absolutely terrible and awful",
            "The food was disgusting and poorly cooked",
            "The experience was horrible and forgettable",
            "The service was dreadful and unprofessional",
            "The weather was miserable and unpleasant",
            "The performance was disappointing and mediocre",
        ]
    },
    "formality_shift": {
        "description": "正式度迁移",
        "source": [
            "Hey what's up how are you doing",
            "Gonna grab some food wanna come",
            "That movie was totally awesome dude",
            "Can't believe how cool that was",
            "Yo check this out it's wild",
        ],
        "target": [
            "Good afternoon, how are you today",
            "I would like to get some food, would you join me",
            "That film was quite remarkable indeed",
            "I find it rather astonishing how impressive that was",
            "Please observe this, it is quite extraordinary",
        ]
    }
}


def load_model():
    """加载 TransformerLens 模型"""
    import torch

    from transformer_lens import HookedTransformer
    
    print("🔄 Loading GPT-2 small...")
    model = HookedTransformer.from_pretrained("gpt2-small")
    model.eval()
    print(f"✅ Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    return model


def get_activations(model, prompts, layer_idx):
    """提取指定层的残差流激活（最后一个 token）"""
    import torch
    acts = []
    for p in prompts:
        tokens = model.to_tokens(p)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, 
                names_filter=lambda name: name.endswith("resid_pre")
            )
        layer_act = cache[f"blocks.{layer_idx}.hook_resid_pre"]
        acts.append(layer_act[0, -1, :].detach().cpu().numpy())
    return np.array(acts)


def compute_rpt(src_acts, tgt_acts, n_components=None):
    """计算传输矩阵 R 及其属性"""
    n_samples = min(len(src_acts), len(tgt_acts))
    if n_components is None:
        n_components = min(n_samples - 1, 4)
    n_components = max(1, min(n_components, n_samples - 1))
    
    # 切空间基底
    pca_src = PCA(n_components=n_components)
    pca_src.fit(src_acts)
    basis_src = pca_src.components_
    
    pca_tgt = PCA(n_components=n_components)
    pca_tgt.fit(tgt_acts)
    basis_tgt = pca_tgt.components_
    
    # Orthogonal Procrustes
    R, scale = orthogonal_procrustes(basis_src, basis_tgt)
    
    # 计算属性
    identity = np.eye(R.shape[0])
    orthogonality_error = np.linalg.norm(R @ R.T - identity, 'fro')
    determinant = np.linalg.det(R)
    frobenius_norm = np.linalg.norm(R, 'fro')
    
    # 传输后的重建误差
    transported = basis_src @ R
    reconstruction_error = np.linalg.norm(transported - basis_tgt, 'fro')
    
    # 方差解释率
    src_var = pca_src.explained_variance_ratio_.sum()
    tgt_var = pca_tgt.explained_variance_ratio_.sum()
    
    return {
        "R": R,
        "n_components": n_components,
        "orthogonality_error": float(orthogonality_error),
        "determinant": float(determinant),
        "frobenius_norm": float(frobenius_norm),
        "reconstruction_error": float(reconstruction_error),
        "src_variance_explained": float(src_var),
        "tgt_variance_explained": float(tgt_var),
        "scale": float(scale)
    }


def run_full_test(model):
    """对所有偏见类型和所有层运行测试"""
    n_layers = model.cfg.n_layers
    results = {}
    
    total_tests = len(BIAS_DATASETS) * n_layers
    done = 0
    
    for bias_name, dataset in BIAS_DATASETS.items():
        print(f"\n{'='*60}")
        print(f"📋 测试: {dataset['description']} ({bias_name})")
        print(f"   Source: {len(dataset['source'])} prompts")
        print(f"   Target: {len(dataset['target'])} prompts")
        print(f"{'='*60}")
        
        layer_results = []
        
        for layer_idx in range(n_layers):
            done += 1
            progress = f"[{done}/{total_tests}]"
            
            try:
                src_acts = get_activations(model, dataset["source"], layer_idx)
                tgt_acts = get_activations(model, dataset["target"], layer_idx)
                
                rpt_result = compute_rpt(src_acts, tgt_acts)
                
                layer_results.append({
                    "layer": layer_idx,
                    **{k: v for k, v in rpt_result.items() if k != "R"}
                })
                
                # 简要报告
                orth_err = rpt_result["orthogonality_error"]
                det = rpt_result["determinant"]
                recon = rpt_result["reconstruction_error"]
                
                status = "✅" if orth_err < 1e-6 and abs(det - 1.0) < 0.01 else "⚠️"
                print(f"  {progress} L{layer_idx:2d}: orth_err={orth_err:.2e}, det={det:.4f}, recon={recon:.4f} {status}")
                
            except Exception as e:
                print(f"  {progress} L{layer_idx:2d}: ❌ Error: {e}")
                layer_results.append({
                    "layer": layer_idx,
                    "error": str(e)
                })
        
        results[bias_name] = {
            "description": dataset["description"],
            "n_source": len(dataset["source"]),
            "n_target": len(dataset["target"]),
            "layers": layer_results
        }
    
    return results


def analyze_results(results):
    """分析测试结果并输出统计摘要"""
    print(f"\n\n{'#'*60}")
    print(f"#  大样本偏见测试 — 结果分析")
    print(f"{'#'*60}\n")
    
    summary = {}
    
    for bias_name, data in results.items():
        valid_layers = [l for l in data["layers"] if "error" not in l]
        
        if not valid_layers:
            print(f"❌ {data['description']}: 所有层均出错")
            continue
        
        orth_errors = [l["orthogonality_error"] for l in valid_layers]
        dets = [l["determinant"] for l in valid_layers]
        recons = [l["reconstruction_error"] for l in valid_layers]
        src_vars = [l["src_variance_explained"] for l in valid_layers]
        
        avg_orth = np.mean(orth_errors)
        avg_det = np.mean(dets)
        avg_recon = np.mean(recons)
        avg_var = np.mean(src_vars)
        
        # 判断黎曼传输质量
        is_orthogonal = avg_orth < 1e-5
        is_proper_rotation = all(abs(d - 1.0) < 0.1 for d in dets)
        
        quality = "🟢 高质量" if is_orthogonal and is_proper_rotation else (
            "🟡 中等" if is_orthogonal else "🔴 低质量"
        )
        
        print(f"📊 {data['description']} ({bias_name})")
        print(f"   传输质量: {quality}")
        print(f"   平均正交误差: {avg_orth:.2e}")
        print(f"   平均行列式: {avg_det:.4f} (理想=1.0)")
        print(f"   平均重建误差: {avg_recon:.4f}")
        print(f"   平均方差解释率: {avg_var:.2%}")
        
        # 找出最佳和最差层
        best = min(valid_layers, key=lambda l: l["reconstruction_error"])
        worst = max(valid_layers, key=lambda l: l["reconstruction_error"])
        print(f"   最佳层: L{best['layer']} (recon={best['reconstruction_error']:.4f})")
        print(f"   最差层: L{worst['layer']} (recon={worst['reconstruction_error']:.4f})")
        print()
        
        summary[bias_name] = {
            "description": data["description"],
            "quality": quality,
            "avg_orthogonality_error": float(avg_orth),
            "avg_determinant": float(avg_det),
            "avg_reconstruction_error": float(avg_recon),
            "avg_variance_explained": float(avg_var),
            "best_layer": best["layer"],
            "worst_layer": worst["layer"],
            "all_orthogonal": bool(is_orthogonal),
            "all_proper_rotation": bool(is_proper_rotation),
        }
    
    return summary


def main():
    start_time = time.time()
    
    print("=" * 60)
    print("  大样本偏见测试 (Large-Scale Bias Test via RPT)")
    print("  基于黎曼平行传输的语义纤维稳定性验证")
    print("=" * 60)
    print(f"  偏见维度: {len(BIAS_DATASETS)} 种")
    total_prompts = sum(len(d["source"]) + len(d["target"]) for d in BIAS_DATASETS.values())
    print(f"  总 Prompt 数: {total_prompts}")
    print()
    
    # 加载模型
    model = load_model()
    
    # 运行完整测试
    results = run_full_test(model)
    
    # 分析结果
    summary = analyze_results(results)
    
    # 保存原始数据
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "bias_rpt_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "gpt2-small",
            "n_bias_types": len(BIAS_DATASETS),
            "total_prompts": total_prompts,
            "detailed_results": results,
            "summary": summary
        }, f, indent=2, ensure_ascii=False, default=lambda o: int(o) if isinstance(o, (np.integer,)) else float(o) if isinstance(o, (np.floating,)) else bool(o) if isinstance(o, (np.bool_,)) else o)
    
    elapsed = time.time() - start_time
    print(f"\n⏱️ 总耗时: {elapsed:.1f}s")
    print(f"📁 结果已保存: {output_path}")
    
    return summary


if __name__ == "__main__":
    main()
