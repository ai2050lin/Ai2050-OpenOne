"""Phase 141 三模型交叉验证分析"""
import json, sys
sys.stdout.reconfigure(encoding='utf-8')

# 读取Qwen3结果
with open('tests/glm5_temp/phase141_qwen3_jacobian_manifold_20260512_1655.json', 'r', encoding='utf-8') as f:
    qwen3 = json.load(f)

# 读取GLM4结果  
with open('tests/glm5_temp/phase141_glm4_jacobian_manifold_20260512_1732.json', 'r', encoding='utf-8') as f:
    glm4 = json.load(f)

print('='*60)
print('Phase 141 交叉验证分析 (Qwen3 + GLM4)')
print('='*60)

# === Exp B: 流形维数 ===
print('\n## Exp B: 语言流形内在维数')
for model_name, data in [('Qwen3', qwen3), ('GLM4', glm4)]:
    expB = data.get('expB', {}).get('layer_analysis', {})
    d = data['model_info']['d_model']
    prs = [v['participation_ratio'] for v in expB.values()]
    dim90s = [v['dim_90pct_variance'] for v in expB.values()]
    mean_pr = sum(prs)/len(prs) if prs else 0
    mean_d90 = sum(dim90s)/len(dim90s) if dim90s else 0
    print(f'  {model_name}: d_model={d}, 平均PR={mean_pr:.0f}/{d} ({mean_pr/d*100:.1f}%), '
          f'90%var维度={mean_d90:.0f} ({mean_d90/d*100:.1f}%)')

# === Exp C: 语义向量场 ===
print('\n## Exp C: 语义向量场一致性')
for model_name, data in [('Qwen3', qwen3), ('GLM4', glm4)]:
    expC = data.get('expC', {}).get('vector_field_analysis', {})
    consistencies = [v['direction_consistency'] for v in expC.values()]
    h_corrs = [v['correlation_with_h'] for v in expC.values()]
    norm_cvs = [v['norm_cv'] for v in expC.values()]
    mean_con = sum(consistencies)/len(consistencies) if consistencies else 0
    mean_hcorr = sum(h_corrs)/len(h_corrs) if h_corrs else 0
    mean_cv = sum(norm_cvs)/len(norm_cvs) if norm_cvs else 0
    op_type = 'nonlinear' if mean_con < 0.7 else 'translation'
    print(f'  {model_name}: 一致性={mean_con:.3f}, h相关性={mean_hcorr:.3f}, '
          f'范数CV={mean_cv:.2f}, 类型={op_type}')

# === Exp D: 算子交换子 ===
print('\n## Exp D: 算子交换子 (非交换性)')
for model_name, data in [('Qwen3', qwen3), ('GLM4', glm4)]:
    expD = data.get('expD', {}).get('commutator_analysis', {})
    rel_comms = {}
    for comp_name, comp_data in expD.items():
        rels = [v['relative_commutator'] for v in comp_data.values() if v.get('relative_commutator')]
        if rels:
            rel_comms[comp_name] = sum(rels)/len(rels)
    
    summary = data.get('expD', {}).get('summary', {})
    noncomm = summary.get('n_noncommutative', 0)
    total = summary.get('n_total', 0)
    mean_rel = summary.get('mean_relative_commutator', 0)
    
    print(f'  {model_name}: 平均相对交换子={mean_rel:.3f}, '
          f'非交换比例={noncomm}/{total}')
    for comp_name, rel in sorted(rel_comms.items()):
        print(f'    {comp_name}: relative_commutator={rel:.3f}')

# === 关键发现总结 ===
print('\n' + '='*60)
print('关键发现总结')
print('='*60)

print('''
1. 【流形维数】dim(M_language) << d_model
   - Qwen3: PR ≈ 65/2560 (2.5%), 90%var ≈ 115/2560 (4.5%)
   - GLM4:  PR ≈ 75/4096 (1.8%), 90%var ≈ 122/4096 (3.0%)
   → 语言流形内在维数仅为模型维度的2-5%，极其低维！

2. 【语义向量场】V_not是非线性算子，非平移
   - 两模型一致性均<0.5 (Qwen3≈0.45, GLM4≈0.43)
   - GLM4的h相关性更高(0.39 vs Qwen3≈0.03)
   → NOT算子的作用强烈依赖上下文h，存在曲率

3. 【算子交换子】语言算子真正非交换
   - 两模型100%非交换 (ALL_NOT, ALWAYS_NOT, SOME_NOT)
   - SOME_NOT相对交换子最大(0.7-1.2)
   → [ALL,NOT]≠0, [ALWAYS,NOT]≠0, [SOME,NOT]≠0

4. 【Exp A Jacobian】数值问题需调试
   - 末层(L35/L39)出现异常极大值
   - 语义/随机比例不稳定，可能eps过小
   - 需要增大eps或改用autograd方法
''')
