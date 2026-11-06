"""Use only the development partition to select a directional native write recipe."""
from collections import defaultdict

import numpy as np
import torch

import mlp_factor_steer_common as C
from mlp_factor_steer_fit import LAYERS, array_key, inputs
from mlp_factor_steer_edit import load_vectors, orientation, positions, prefill
from mlp_kv_behavior import score, word_indices
from mlp_read_route_causal import branch
from mlp_functional_trace import load_model
from shared.dist_protocol.mechanism_trace import GpuLease


def gradients(model,row,vectors,words):
    x=torch.tensor([row['tokens']],device='cuda');p=positions([row],x.device)[0];leaves={};hooks=[]
    def attach(x,b,kind):
        current=x[:,p];leaf=torch.zeros_like(current,requires_grad=True);out=x.clone();out[:,p]=current+leaf
        leaves[b,kind]=leaf;return out
    for b in LAYERS:
        layer=model.model.layers[b]
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda _,args,b=b:(attach(args[0],b,'H'),*args[1:])))
        hooks.append(layer.mlp.register_forward_hook(lambda _,args,x,b=b:attach(x,b,'M')))
    try:
        logits=model(input_ids=x,use_cache=False,logits_to_keep=1).logits[0,-1].float()
        gap=logits[words['fishing']].max()-logits[words['mining']].max()
        g=torch.autograd.grad(gap,tuple(leaves.values()));assert all(torch.isfinite(v).all() for v in g)
        values={}
        for ((b,kind),v) in zip(leaves,g):
            v=v[0].float()
            if kind=='H':v=v.reshape(4,32,128).transpose(0,1)
            else:v=v[None]
            values[b,kind]=(v*vectors[array_key(b,kind)]).sum((1,2)).cpu().tolist()
        return values
    finally:
        for h in hooks:h.remove()


def main():
    dest=C.OUT/'frozen-recipe.json'
    if dest.exists():return
    corpus,plan=inputs();items=corpus['selection'];assert len(items)==C.SELECT
    execution={**C.identity(),'schema':'factor_steer_selection_execution.v1','source_sha256':C.file_sha(__file__),
               'editor_sha256':C.file_sha(C.ROOT/'tests/glm5/mlp_factor_steer_edit.py'),'inputs_sha256':plan['inputs_sha256'],
               'directions_sha256':C.file_sha(C.OUT/'directions.npz'),'dose_grid':[1,4,8]}
    C.save(C.OUT/'selection-execution.json',execution);torch.set_num_threads(4);values=defaultdict(list)
    with GpuLease():
        model,tok,eos=load_model();vectors=load_vectors('cuda');words=word_indices(tok)
        # Stable subsample for ranking only. All 64 separate development backgrounds enter the finite sweep.
        gradient_rows=items[::2]
        for i,row in enumerate(gradient_rows):
            for key,v in gradients(model,row,vectors,words).items():values[key].append(v)
            if (i+1)%8==0:print('writer gradient',i+1,'/',len(gradient_rows),flush=True)
        head_scores=sorted([(float(v),b,h) for (b,k),data in values.items() if k=='H' for h,v in enumerate(np.mean(data,axis=0))],
                           key=lambda x:(-x[0],x[1],x[2]))
        mlp_scores=sorted([(float(np.mean(data)),b) for (b,k),data in values.items() if k=='M'],key=lambda x:(-x[0],x[1]))
        recipe=lambda name,h,m:{'name':name,'heads':[list(v[1:]) for v in h],'mlps':[v[1] for v in m]}
        compact_heads=[x for x in head_scores if x[1]>=30];compact_mlps=[x for x in mlp_scores if x[1]>=30]
        candidates=[recipe('compact',compact_heads[:8],compact_mlps[:2]),recipe('broad',head_scores[:24],mlp_scores[:5])]
        recipes=[{**r,'alpha':a,'id':f'{r["name"]}-{a}'} for r in candidates for a in [1,4,8]];records=[]
        with torch.inference_mode():
            for lo in range(0,len(items),4):
                batch=items[lo:lo+4];x=torch.tensor([r['tokens'] for r in batch],device='cuda')
                p=positions(batch,x.device);s=orientation(batch,x.device)
                cache=model(input_ids=x[:,:C.CUT],use_cache=True,logits_to_keep=1).past_key_values
                logits,tokens=branch(model,x,cache,eos)
                rows=[{k:v for k,v in row.items() if k!='tokens'}|{'clean':score(row,logits[j],tokens[j],tok,words),'candidates':{}} for j,row in enumerate(batch)]
                del cache
                for r in recipes:
                    edited,norm=prefill(model,x,p,s,r,vectors,r['alpha']);a,t=branch(model,x,edited,eos);del edited
                    matched,_=prefill(model,x,p,s,r,vectors,r['alpha'],'matched',norm);b,u=branch(model,x,matched,eos);del matched
                    for j,row in enumerate(batch):
                        rows[j]['candidates'][r['id']]={'targeted':score(row,a[j],t[j],tok,words),'matched':score(row,b[j],u[j],tok,words)}
                records.extend(rows)
                if (lo+4)%16==0:print('finite recipe sweep',lo+4,'/',len(items),flush=True)
    clean=[r for r in records if r['clean']['correct']];assert len(clean)>=16
    summaries={}
    for recipe_row in recipes:
        key=recipe_row['id'];targeted=np.array([r['candidates'][key]['targeted']['word']==r['contrast'] for r in clean]);matched=np.array([r['candidates'][key]['matched']['word']==r['contrast'] for r in clean]);
        gap=np.array([r['candidates'][key]['matched']['gap']-r['candidates'][key]['targeted']['gap'] for r in clean])
        summaries[key]={'n':len(clean),'targeted_alt':int(targeted.sum()),'matched_alt':int(matched.sum()),'flip_advantage':float(targeted.mean()-matched.mean()),
                        'mean_gap_advantage':float(gap.mean()),'selected':False}
    best=min(recipes,key=lambda r:(-summaries[r['id']]['flip_advantage'],-summaries[r['id']]['mean_gap_advantage'],len(r['heads'])+len(r['mlps']),r['alpha'],r['id']))
    summaries[best['id']]['selected']=True
    offsets={k:int(np.median([1033-r['anchors'][k] for r in items])) for k in items[0]['anchors'] if k!='draft'}
    C.write_gz(C.OUT/'selection-result.json.gz',{**execution,'recipes':recipes,'summaries':summaries,'rows':records,'head_ranking':head_scores,'mlp_ranking':mlp_scores})
    C.save(dest,{**execution,'schema':'factor_steer_recipe.v1','recipe':best,'offsets':offsets,'selection_sha256':C.file_sha(C.OUT/'selection-result.json.gz'),
                 'discovery_targeted_gain':summaries[best['id']], 'ranking_n':len(gradient_rows)})
    print('frozen',best,'selection summary',summaries,flush=True)


if __name__=='__main__':main()
