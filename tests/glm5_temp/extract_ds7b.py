import json

with open('tests/glm5_temp/phase117_exp3_deepseek7b_control_vs_data.json') as f:
    d = json.load(f)

print('DeepSeek7B Control vs Data Separation:')
header = f'{"Layer":>6} {"dim":>4} {"full_knn":>8} {"spike_knn":>9} {"comp_knn":>8} {"spike_rho":>9} {"comp_rho":>8}'
print(header)
for l in d['layers_tested']:
    ld = d[f'L{l}']
    full_knn = ld['full']['knn_accuracy']
    spike_knn = ld['spike_only']['knn_accuracy']
    comp_knn = ld['complement_only']['knn_accuracy']
    spike_rho = ld['spike_only']['rho_with_full']
    comp_rho = ld['complement_only']['rho_with_full']
    dim = ld['spike_dim']
    print(f'L{l:>4} {dim:>4} {full_knn:>8.3f} {spike_knn:>9.3f} {comp_knn:>8.3f} {spike_rho:>9.3f} {comp_rho:>8.3f}')
