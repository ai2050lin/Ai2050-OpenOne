# MV-v1 事后诊断（看到结果后设计，不属于预注册）：逐层跨语言探针迁移（q、n、真值），中文行为门的成对口径。
# 用法（仓库根目录）：python scripts/mode_atlas_v1_posthoc.py；输出 research/mode_atlas/MV-v1-qwen3-4b/diagnostic_posthoc.json
import json, sys, numpy as np
sys.path[:0]=['.','scripts']
from shared.dist_protocol import mode_atlas as M
z=np.load('D:/ai2050_native/_mode_atlas/6be1034f0b55a6df-bf16/MV-v1/data.npz')
H={(s,l):z[f'{s}_{l}_H'] for s in ('disc','conf') for l in ('zh','en')}
L=H['disc','zh'].shape[2]; d=H['disc','zh'].shape[-1]
out={'note':'事后诊断（看到 MV-v1 结果后设计），不属于预注册','transfer_by_layer':{}}
def lab(n_e, fn, cells): return np.tile([fn(c) for c in cells], n_e).astype(bool)
specs={'q':(range(8),lambda c:(c>>2)&1),'n':(range(8),lambda c:(c>>1)&1),'truth':(range(4),lambda c:((c>>1)&1)^(c&1))}
for f,(cells,fn) in specs.items():
    cells=list(cells); rows=[]
    for l in range(1,L):
        Xd=H['disc','zh'][:,cells,l].astype(np.float64).reshape(-1,d); Xe=H['conf','en'][:,cells,l].astype(np.float64).reshape(-1,d)
        Xz=H['conf','zh'][:,cells,l].astype(np.float64).reshape(-1,d)
        Xde=H['disc','en'][:,cells,l].astype(np.float64).reshape(-1,d)
        yd=lab(len(H['disc','zh']),fn,cells); yc=lab(len(H['conf','zh']),fn,cells)
        gd=np.repeat(np.arange(len(H['disc','zh'])),len(cells))
        a=M.probe(Xd,yd,gd,Xz,yc,X_te2=Xe,y_te2=yc)
        b=M.probe(Xde,yd,gd,H['conf','en'][:,cells,l].astype(np.float64).reshape(-1,d),yc,X_te2=Xz,y_te2=yc)
        rows.append({'layer':l,'zh_to_en':a['acc_transfer'],'en_to_zh':b['acc_transfer'],'zh_in':a['acc'],'en_in':b['acc']})
    out['transfer_by_layer'][f]=rows
    print(f,'zh->en',[round(r['zh_to_en'],2) for r in rows]); print(f,'en->zh',[round(r['en_to_zh'],2) for r in rows])
g={}
for lang in ('zh','en'):
    G=np.concatenate([z[f'disc_{lang}_gate'],z[f'conf_{lang}_gate']])  # [n_e, 2n+t]
    truth=np.array([0,1,1,0],bool)
    pair_aff=float(np.mean(G[:,1]>G[:,0])); pair_neg=float(np.mean(G[:,2]>G[:,3]))
    g[lang]={'mean_margin_by_cell_nt':[float(x) for x in G.mean(0)],'paired_affirm_true_gt_false':pair_aff,'paired_neg_true_gt_false':pair_neg}
out['gate_paired']=g; print(json.dumps(g,ensure_ascii=False))
json.dump(out,open('research/mode_atlas/MV-v1-qwen3-4b/diagnostic_posthoc.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
