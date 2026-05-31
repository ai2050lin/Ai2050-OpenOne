import json
data = json.load(open('results/phase303_large_scale_factorial/deepseek7b_large_scale_factorial.json','r',encoding='utf-8'))
li = str(data['n_layers']//2)
lr = data['factorial_causal'][li]
neg_tokens = [(v['token'], v.get('role_pair',''), v.get('full_delta_cos_shift',0), v.get('R_only_cos_shift',0), v.get('R+F_residual_cos_shift',0)) for v in lr.values() if v.get('full_delta_cos_shift',0) is not None]
neg_tokens.sort(key=lambda x: x[2])
print('DS7B L14: Tokens sorted by full_delta (most negative first):')
for t, rp, fd, ro, rfr in neg_tokens[:15]:
    print(f'  {t:10s} {rp:10s} fd={fd:+.4f} R={ro:+.4f} R+Fr={rfr:+.4f}')
print('Most positive:')
for t, rp, fd, ro, rfr in neg_tokens[-5:]:
    print(f'  {t:10s} {rp:10s} fd={fd:+.4f} R={ro:+.4f} R+Fr={rfr:+.4f}')

r_pos_fd_neg = sum(1 for v in lr.values() if v.get('R_only_cos_shift',0) > 0 and v.get('full_delta_cos_shift',0) < 0)
r_pos_fd_pos = sum(1 for v in lr.values() if v.get('R_only_cos_shift',0) > 0 and v.get('full_delta_cos_shift',0) > 0)
r_neg_fd_neg = sum(1 for v in lr.values() if v.get('R_only_cos_shift',0) < 0 and v.get('full_delta_cos_shift',0) < 0)
r_neg_fd_pos = sum(1 for v in lr.values() if v.get('R_only_cos_shift',0) < 0 and v.get('full_delta_cos_shift',0) > 0)
print(f'\nR_only vs full_delta sign agreement:')
print(f'  R>0, FD>0: {r_pos_fd_pos}')
print(f'  R>0, FD<0: {r_pos_fd_neg}  (R correct, FD wrong)')
print(f'  R<0, FD<0: {r_neg_fd_neg}')
print(f'  R<0, FD>0: {r_neg_fd_pos}  (R wrong, FD correct)')

# Per role-pair analysis
from collections import defaultdict
rp_data = defaultdict(list)
for v in lr.values():
    rp = v.get('role_pair','')
    rp_data[rp].append(v)
for rp in ['adj_verb','adj_noun','noun_verb']:
    items = rp_data[rp]
    agree = sum(1 for v in items if (v.get('R_only_cos_shift',0) > 0) == (v.get('full_delta_cos_shift',0) > 0))
    print(f'  {rp}: sign agreement = {agree}/{len(items)} ({agree/len(items)*100:.0f}%)')
