import json
from collections import defaultdict

for mn in ["qwen3", "glm4", "deepseek7b"]:
    data = json.load(open(f'results/phase303_large_scale_factorial/{mn}_large_scale_factorial.json','r',encoding='utf-8'))
    li = str(data['n_layers']//2)
    lr = data['factorial_causal'][li]
    
    rp_data = defaultdict(list)
    for v in lr.values():
        rp = v.get('role_pair','')
        rp_data[rp].append(v)
    
    print(f'\n{mn} L{li}:')
    for rp in ['adj_verb','adj_noun','noun_verb']:
        items = rp_data[rp]
        agree = sum(1 for v in items if (v.get('R_only_cos_shift',0) > 0) == (v.get('full_delta_cos_shift',0) > 0))
        fd_vals = [v.get('full_delta_cos_shift',0) for v in items if v.get('full_delta_cos_shift') is not None]
        r_vals = [v.get('R_only_cos_shift',0) for v in items if v.get('R_only_cos_shift') is not None]
        fd_pos = sum(1 for v in fd_vals if v > 0)
        r_pos = sum(1 for v in r_vals if v > 0)
        print(f'  {rp}: R_only pos={r_pos}/{len(items)} full_delta pos={fd_pos}/{len(items)} sign_agree={agree}/{len(items)} ({agree/len(items)*100:.0f}%)')
    
    # Overall
    all_items = list(lr.values())
    fd_vals = [v.get('full_delta_cos_shift',0) for v in all_items if v.get('full_delta_cos_shift') is not None]
    r_vals = [v.get('R_only_cos_shift',0) for v in all_items if v.get('R_only_cos_shift') is not None]
    agree_all = sum(1 for v in all_items if (v.get('R_only_cos_shift',0) > 0) == (v.get('full_delta_cos_shift',0) > 0))
    print(f'  ALL: R_only={sum(r_vals)/len(r_vals):+.4f} full_delta={sum(fd_vals)/len(fd_vals):+.4f} sign_agree={agree_all}/{len(all_items)} ({agree_all/len(all_items)*100:.0f}%)')
