"""界面演示用的单句归因：activation x gradient + 实际消融。未预注册，不进入研究证据。"""
import json,sys,random,time
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
S=sys.argv[1]; P='models/hf/qwen3-4b'
tok=AutoTokenizer.from_pretrained(P)
model=AutoModelForCausalLM.from_pretrained(P,torch_dtype=torch.float32,attn_implementation='eager'); model.eval()
for p in model.parameters(): p.requires_grad_(False)
L=model.model.layers; NL=len(L); NH=32; HD=128
CAND=[('知识网络','黄',['香蕉成熟以后的颜色是']),
 ('推理与功能','奥斯汀',['问：达拉斯所在州的首府是哪里？答：','问题：达拉斯所在的州，首府是哪座城市？答案：','达拉斯所在的州的首府城市名叫','Q：达拉斯所在州的首府是？A：']),
 ('语法体系','？',['你今天吃饭了吗','你是不是已经吃过饭了','明天你要不要来我家吃饭呢'])]
PROMPTS=[]
for part,want,cs in CAND:
    pick=None
    for c in cs:
        i=tok(c,return_tensors='pt').input_ids
        with torch.no_grad(): o=model(input_ids=i,use_cache=False).logits[0,-1]
        t=tok.decode([o.argmax().item()]); print('cand',c,repr(t),flush=True)
        if want in t or t in want:
            pick=c; break
    PROMPTS.append((part,pick or cs[0])); print('pick',part,pick,flush=True)
store={}
def hooks():
    hs=[]
    for l,b in enumerate(L):
        def mk(l,kind):
            def f(m,a):
                x=a[0]; x.retain_grad() if x.requires_grad else None
                store[(kind,l)]=x
            return f
        hs.append(b.mlp.down_proj.register_forward_pre_hook(mk(l,'m')))
        hs.append(b.self_attn.o_proj.register_forward_pre_hook(mk(l,'h')))
    return hs
def embeds(ids):
    e=model.model.embed_tokens(ids).detach().clone().requires_grad_(True); return e
def fwd(ids):
    e=embeds(ids); out=model(inputs_embeds=e,use_cache=False).logits[0,-1]; return e,out
def ablate(ids,nodes,scale=0.0):
    """nodes: (kind,l,p,i). 把指定单元或 Head 在指定位置的值乘 scale。"""
    by={}
    for k,l,p,i in nodes: by.setdefault((k,l),[]).append((p,i))
    hs=[]
    for (k,l),v in by.items():
        mod=L[l].mlp.down_proj if k=='m' else L[l].self_attn.o_proj
        def f(m,a,v=v,k=k):
            x=a[0].clone()
            for p,i in v:
                if k=='m': x[0,p,i]*=scale
                else: x[0,p,i*HD:(i+1)*HD]*=scale
            return (x,)
        hs.append(mod.register_forward_pre_hook(f))
    with torch.no_grad(): o=model(input_ids=ids,use_cache=False).logits[0,-1]
    for h in hs: h.remove()
    return o
res=[]
for part,text in PROMPTS:
    t0=time.time()
    ids=tok(text,return_tensors='pt').input_ids; T=ids.shape[1]
    toks=[tok.decode([i]) for i in ids[0].tolist()]
    hs=hooks(); store.clear()
    e,out=fwd(ids)
    pr=torch.softmax(out,-1); top=pr.topk(5)
    a,b=top.indices[0].item(),top.indices[1].item()
    metric=out[a]-out[b]
    metric.backward(retain_graph=True)
    nodes=[]
    for l in range(NL):
        am,ah=store[('m',l)],store[('h',l)]
        sm=(am*am.grad)[0]; sh=(ah*ah.grad)[0].view(T,NH,HD).sum(-1)
        for p in range(T):
            v,i=sm[p].abs().topk(6)
            for x,u in zip(v,i): nodes.append(('m',l,p,u.item(),sm[p,u].item()))
            v,i=sh[p].abs().topk(3)
            for x,h in zip(v,i): nodes.append(('h',l,p,h.item(),sh[p,h].item()))
    gm={k:store[k].grad.detach().clone() for k in store}
    tokattr=(e*e.grad)[0].sum(-1).tolist()
    for h in hs: h.remove()
    st=dict(store)
    nodes.sort(key=lambda n:-abs(n[4]))
    sel=[n for n in nodes if n[2]>0][:26]
    base=metric.item()
    # 实际消融
    out_nodes=[]
    for n in sel:
        o=ablate(ids,[n[:4]]); d=(o[a]-o[b]).item(); t=o.argmax().item()
        o2=ablate(ids,[n[:4]],2.0); d2=(o2[a]-o2[b]).item()
        out_nodes.append({'k':n[0],'l':n[1],'p':n[2],'i':n[3],'s':round(n[4],4),'off':round(d,3),'x2':round(d2,3),'top':tok.decode([t]),'flip':t!=a})
    # 节点到节点的梯度边（含间接路径）
    edges=[]
    for j,n in enumerate(sel):
        k,l,p,i,_=n
        x=st[(k,l)]
        if k=='m': val=x[0,p,i]*gm[(k,l)][0,p,i]
        else: val=(x[0,p,i*HD:(i+1)*HD]*gm[(k,l)][0,p,i*HD:(i+1)*HD]).sum()
        for key in st:
            if st[key].grad is not None: st[key].grad=None
        e.grad=None
        val.backward(retain_graph=True)
        c=[]
        for q,m in enumerate(sel):
            k2,l2,p2,i2,_=m
            if (l2,0 if k2=='h' else 1)>=(l,0 if k=='h' else 1) or p2>p: continue
            y=st[(k2,l2)]
            if y.grad is None: continue
            s=(y[0,p2,i2]*y.grad[0,p2,i2]).item() if k2=='m' else (y[0,p2,i2*HD:(i2+1)*HD]*y.grad[0,p2,i2*HD:(i2+1)*HD]).sum().item()
            c.append((q,s))
        if e.grad is not None:
            ts=(e*e.grad)[0].sum(-1)
            for q in range(1,p+1): c.append((-1-q,ts[q].item()))
        c.sort(key=lambda z:-abs(z[1]))
        for q,s in c[:3]: edges.append([q,j,round(s,4)])
    # 组消融和随机对照
    K=10; posn=[n for n in sel if n[4]>0][:K]; K=len(posn); g=[n[:4] for n in posn]
    og=ablate(ids,g); dg=(og[a]-og[b]).item()
    rnd=[]
    rng=random.Random(20261106)
    for r in range(20):
        c=[]
        for n in posn:
            c.append((n[0],n[1],n[2],rng.randrange(9728 if n[0]=='m' else NH)))
        o=ablate(ids,c); rnd.append((o[a]-o[b]).item())
    rnd.sort()
    res.append({'part':part,'text':text,'tokens':toks,'top':[[tok.decode([i]),round(v,4)] for v,i in zip(top.values.tolist(),top.indices.tolist())],'base':round(base,3),'tokattr':[round(x,3) for x in tokattr],'nodes':out_nodes,'edges':edges,
      'group':{'k':K,'off':round(dg,3),'top':tok.decode([og.argmax().item()]),'rnd_med':round(rnd[10],3),'rnd_min':round(rnd[0],3),'rnd_max':round(rnd[-1],3)}})
    print(text,toks,res[-1]['top'],base,res[-1]['group'],round(time.time()-t0),flush=True)
    print(' flips',sum(n['flip'] for n in out_nodes),[ (n['k'],n['l'],n['p'],n['i'],n['s'],n['off']) for n in out_nodes[:6]],flush=True)
att=[]
for part,text in PROMPTS:
    ids=tok(text,return_tensors='pt').input_ids
    with torch.no_grad(): o=model(input_ids=ids,output_attentions=True,use_cache=False)
    att.append({'tokens':[tok.decode([i]) for i in ids[0].tolist()],'a':[[[round(float(v),2) for v in o.attentions[l][0,h,-1]] for h in range(32)] for l in range(36)]})
hi=json.load(open(S+'/data3d/headinfo.json',encoding='utf8')); hi['att']=att
json.dump(hi,open(S+'/data3d/headinfo.json','w',encoding='utf8'),ensure_ascii=False,separators=(',',':'))
json.dump({'model':'Qwen3-4B fp32 CPU','snapshot':'6be1034f0b55a6df','graphs':res},open(S+'/data3d/attr.json','w',encoding='utf8'),ensure_ascii=False,separators=(',',':'))
print('saved',flush=True)
