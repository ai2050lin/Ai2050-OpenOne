"""Phase 483 竞争释放矩阵详细分析"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import json
import numpy as np

cat_names = ["fruit", "animal", "tool", "vehicle", "clothing", "furniture", "food", "plant"]

for model in ['qwen3', 'glm4', 'deepseek7b']:
    path = f'results/glm5/phase483_{model}_r1.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f'\n{"="*80}')
    print(f'  {model.upper()} - Competition Release Matrix')
    print(f'{"="*80}')
    
    release_matrix = np.array(data['exp2_competition_release']['release_matrix'])
    
    # 打印矩阵
    header = f'{"removed\\DCF":>12}' + ''.join(f'{c:>10}' for c in cat_names)
    print(header)
    for i, cat in enumerate(cat_names):
        row = f'{cat:>12}' + ''.join(f'{release_matrix[i,j]:>10.2f}' for j in range(8))
        print(row)
    
    # 识别最强的竞争释放对
    print(f'\n  Top-5 strongest competitor releases:')
    releases = []
    for i in range(8):
        for j in range(8):
            if i != j and release_matrix[i,j] > 0:
                releases.append((cat_names[i], cat_names[j], float(release_matrix[i,j])))
    releases.sort(key=lambda x: -x[2])
    for removed, released, delta in releases[:5]:
        print(f'    remove {removed} -> {released} +{delta:.2f}')
    
    # 识别互斥对(双向释放)
    print(f'\n  Mutual competition pairs:')
    seen = set()
    for i in range(8):
        for j in range(i+1, 8):
            if release_matrix[i,j] > 0 and release_matrix[j,i] > 0:
                pair = (cat_names[i], cat_names[j], float(release_matrix[i,j]), float(release_matrix[j,i]))
                print(f'    {pair[0]}<->{pair[1]}: {pair[0]}_removed->{pair[1]}+{pair[2]:.2f}, {pair[1]}_removed->{pair[0]}+{pair[3]:.2f}')

    # Exp1 关键指标
    print(f'\n--- Exp1: Boundary Writer Key Metrics ---')
    for cat, d in data.get('exp1_boundary_writers', {}).items():
        print(f'  {cat}: top50_conc={d["concentration_top50"]:.3f}, '
              f'top10_conc={d["concentration_top10"]:.3f}, '
              f'cos(Bc)={d["cos_top50_dir_with_bc"]:.3f}, '
              f'n_significant={d["n_positive_neurons"]+d["n_negative_neurons"]}, '
              f'total_signal={d["total_boundary_signal"]:.2f}')
