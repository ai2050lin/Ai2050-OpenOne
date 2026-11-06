"""界面演示用：Head 写出读词、演示句注意力、神经元真实参数。只读权重，不进入研究证据。"""
import json,sys,time
import torch
from safetensors import safe_open
from transformers import AutoTokenizer,AutoModelForCausalLM
S=sys.argv[1]; P='models/hf/qwen3-4b'
tok=AutoTokenizer.from_pretrained(P)
idx=json.load(open(P+'/model.safetensors.index.json'))['weight_map']; hs={}
def get(n):
    f=idx[n]
    if f not in hs: hs[f]=safe_open(P+'/'+f,'pt')
    return hs[f].get_tensor(n).float()
cards=json.load(open(S+'/data3d/cards.json',encoding='utf8'))['cards']
need={}
for k in cards:
    b,u=map(int,k.split(':')); need.setdefault(b,[]).append(u)
WU=get('model.embed_tokens.weight'); gam=get('model.norm.weight')
ov={};up={}
t0=time.time()
for l in range(36):
    b='model.layers.%d.'%l
    wo=get(b+'self_attn.o_proj.weight')   # 2560 x 4096
    vs=[];sv=[]
    for h in range(32):
        U,Sg,_=torch.linalg.svd(wo[:,h*128:(h+1)*128],full_matrices=False)
        vs.append(U[:,0]*gam); sv.append(Sg[0].item())
    lg=WU@torch.stack(vs,1)
    tp=lg.topk(6,dim=0).indices; bt=(-lg).topk(6,dim=0).indices
    for h in range(32):
        ov['%d:%d'%(l,h)]=[round(sv[h],2),[tok.decode([i]) for i in tp[:,h].tolist()],[tok.decode([i]) for i in bt[:,h].tolist()]]
    if l in need:
        g=get(b+'mlp.gate_proj.weight');uw=get(b+'mlp.up_proj.weight');d=get(b+'mlp.down_proj.weight')
        for u in need[l]:
            gv,uv,dv=g[u],uw[u],d[:,u]
            c=lambda a,c2:(a@c2/(a.norm()*c2.norm())).item()
            up['%d:%d'%(l,u)]=[round(gv.norm().item(),3),round(uv.norm().item(),3),round(dv.norm().item(),3),round(c(gv,dv),3),round(c(uv,dv),3)]
    print(l,round(time.time()-t0),flush=True)
del WU
model=AutoModelForCausalLM.from_pretrained(P,dtype=torch.bfloat16,attn_implementation='eager'); model.eval()
att=[]
for text in ['香蕉成熟以后的颜色是','达拉斯所在州的首府是','我想问，火是热的']:
    ids=tok(text,return_tensors='pt').input_ids
    with torch.no_grad(): o=model(input_ids=ids,output_attentions=True,use_cache=False)
    att.append({'tokens':[tok.decode([i]) for i in ids[0].tolist()],'a':[[[round(float(v),2) for v in o.attentions[l][0,h,-1]] for h in range(32)] for l in range(36)]})
json.dump({'ov':ov,'att':att,'up':up},open(S+'/data3d/headinfo.json','w',encoding='utf8'),ensure_ascii=False,separators=(',',':'))
print('saved',len(ov),len(up),flush=True)
