"""Phase 125 补丁: 仅运行GLM4的Exp 3-5 (跳过已完成的Exp 1-2)"""
import sys, os, json, time, numpy as np, torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

from tests.glm5.model_utils import load_model, get_layers, get_model_info, get_W_U, release_model, MODEL_CONFIGS
from tests.glm5.ccml_phase125_fisher_geometry import (
    collect_activations_and_pca, w_u_alignment_analysis,
    directional_ablation_experiment, epsilon_stability_test
)

def convert(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

model_name = "glm4"
temp_dir = "tests/glm5_temp"

# 加载已保存的Exp 1-2结果
print("加载已保存的Exp 1-2结果...")
fisher_results_raw = json.load(open(f'{temp_dir}/phase125_exp1_{model_name}_fisher_spectrum.json'))
alignment_results = json.load(open(f'{temp_dir}/phase125_exp2_{model_name}_alignment.json'))

# 保持字符串键（函数内部已做兼容处理）
fisher_results = fisher_results_raw

# 加载模型
print("\n加载模型...")
model, tokenizer, device = load_model(model_name)
model_info = get_model_info(model, model_name)
print(f"  class={model_info.model_class}, n_layers={model_info.n_layers}, d_model={model_info.d_model}")

target_layers = sorted([int(k) for k in fisher_results.keys()])
print(f"  目标层: {target_layers}")

# 重新计算PCA (因为fisher_spectrum.json没有保存完整components)
texts = [
    "The cat sat on the mat and looked out the window at the birds flying by.",
    "In quantum mechanics, particles exhibit wave-particle duality, which means they can behave as both waves and particles depending on the experimental setup.",
    "The algorithm processes the input data through multiple layers of transformation, applying attention mechanisms and feed-forward networks at each step.",
    "Language models learn statistical patterns from large text corpora, enabling them to generate coherent and contextually appropriate text.",
    "The mathematical proof relies on induction, showing that if the property holds for n, it must also hold for n+1.",
    "Photosynthesis converts light energy into chemical energy, producing glucose and oxygen from carbon dioxide and water.",
    "The economic policy aims to balance inflation control with employment growth through careful monetary regulation.",
    "Machine learning models generalize from training examples to unseen data by learning underlying patterns and features.",
    "The symphony orchestra performed Beethoven's Ninth Symphony, featuring a full choir in the final movement.",
    "Neural networks use backpropagation to adjust weights based on the gradient of the loss function with respect to each parameter.",
]
pca_results = collect_activations_and_pca(model, tokenizer, device, model_info, texts, target_layers)

# 测试prompts
test_prompts = [
    "The fundamental theorem of calculus establishes the relationship between differentiation and integration.",
    "Deep learning models require large amounts of training data to achieve good generalization performance.",
    "The architecture of Gothic cathedrals reflects both engineering innovation and spiritual aspiration.",
    "Climate change affects global weather patterns through complex feedback mechanisms.",
    "The philosopher argued that consciousness cannot be reduced to purely physical processes.",
    "Computer vision algorithms can detect objects in images using convolutional neural networks.",
    "The economic impact of automation on employment remains a subject of intense debate among researchers.",
    "DNA replication follows a semi-conservative mechanism where each strand serves as a template.",
    "The novel explores themes of alienation and identity in post-industrial society.",
    "Reinforcement learning agents learn optimal policies through trial and error interactions with their environment.",
    "The immune system protects the body against pathogens through both innate and adaptive mechanisms.",
    "Bayesian inference provides a principled framework for updating beliefs in light of new evidence.",
    "The Renaissance transformed European art through the rediscovery of classical techniques and perspectives.",
    "Natural language processing has advanced significantly with the development of transformer architectures.",
    "The quantum computing paradigm exploits superposition and entanglement for computational advantage.",
    "Evolutionary theory explains the diversity of life through natural selection and genetic variation.",
    "The treaty established a framework for international cooperation on environmental protection.",
    "Gradient descent optimization finds local minima by iteratively moving in the direction of steepest decrease.",
    "The orchestra performed a challenging contemporary piece that pushed the boundaries of traditional harmony.",
    "Information theory provides the mathematical foundation for data compression and transmission.",
    "The historical analysis reveals how social movements shape public policy over decades.",
    "Transfer learning allows models trained on one task to be adapted for related tasks with minimal fine-tuning.",
    "The mathematical structure of group theory appears throughout physics, from crystallography to particle physics.",
    "Urban planning must balance economic development with environmental sustainability and social equity.",
    "The discovery of gravitational waves confirmed a major prediction of general relativity.",
]

# Exp 3: W_U对齐
print("\n" + "="*50)
print("Exp 3: W_U行空间与信号子空间的对齐")
print("="*50)
wu_results = w_u_alignment_analysis(
    model, model_name, pca_results, fisher_results, target_layers, model_info.d_model
)
with open(f'{temp_dir}/phase125_exp3_{model_name}_wu_alignment.json', 'w') as f:
    json.dump(wu_results, f, indent=2, ensure_ascii=False, default=convert)
print(f"Exp 3 结果已保存")

# Exp 4: 定向消融
print("\n" + "="*50)
print("Exp 4: 定向消融实验")
print("="*50)
ablation_results = directional_ablation_experiment(
    model, tokenizer, device, model_info,
    test_prompts, pca_results, fisher_results, target_layers
)
with open(f'{temp_dir}/phase125_exp4_{model_name}_ablation.json', 'w') as f:
    json.dump(ablation_results, f, indent=2, ensure_ascii=False, default=convert)
print(f"Exp 4 结果已保存")

# Exp 5 已完成, 跳过
print("\nExp 5 已在之前完成, 跳过")

print("\nGLM4 Exp 3-4 完成!")
release_model(model)
