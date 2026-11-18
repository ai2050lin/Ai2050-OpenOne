"""Discovery-only test of left-padding numerical prefix dependence; no gate repair."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_relay_design as D
from source_relay_behavior import Runtime
from source_binding_native_v2 import bits,unbits

def right_context(rt,rows,length):
    t=rt.t;n=len(rows);pad=rt.tok.pad_token_id or rt.tok.eos_token_id
    if isinstance(pad,list):pad=pad[0]
    ids=t.full((n,length),pad,device='cuda',dtype=t.long);valid=t.zeros((n,length),device='cuda',dtype=t.bool)
    for i,r in enumerate(rows):ids[i,:len(r['tokens'])]=t.tensor(r['tokens'],device='cuda');valid[i,:len(r['tokens'])]=True
    pos=t.arange(length,device='cuda')[None].expand(n,-1).clone();pos[~valid]=0
    rot=rt.Rotary(rt.cfg,device='cuda');pe=rot(t.zeros((n,length,1),dtype=t.bfloat16,device='cuda'),pos)
    mask=t.ones((length,length),dtype=t.bool,device='cuda').tril()[None,None]&valid[:,None,None,:]
    return ids,pos,pe,mask

def run():
    import torch
    D.verify_internal();source=D.read(D.OUT/'qwen14-materials.json')['splits']['discovery'];groups={}
    for r in source:groups.setdefault((r['family'],r['order']),[]).append(r)
    selected=[];pairs=[]
    for key,rs in groups.items():
        a=min(rs,key=lambda r:len(r['tokens']));b=max(rs,key=lambda r:len(r['tokens']))
        if len(a['tokens'])==len(b['tokens']):continue
        last=max(a['records'][str(j)]['span'][-1] for j in range(6))+1
        assert a['tokens'][:last]==b['tokens'][:last]
        pairs.append({'family':key[0],'order':key[1],'ids':[a['id'],b['id']],'lengths':[len(a['tokens']),len(b['tokens'])]});selected.extend([a,b])
        if len(pairs)==64:break
    assert len(pairs)>=32
    n=len(selected);L=max(len(r['tokens']) for r in source)+8;timer=time.monotonic();arrays={};scores={}
    with D.gpu_lease('padding-diagnostic'),torch.inference_mode():
        rt=Runtime(D.MODEL)
        for mode in ['left_full','right_full','right_prefix']:
            rows=selected if mode!='right_prefix' else [{**r,'tokens':r['tokens'][:max(r['records'][str(j)]['span'][-1] for j in range(6))+1]} for r in selected]
            if mode=='left_full':ids,pos,pe,mask=rt.context(rows)
            else:ids,pos,pe,mask=right_context(rt,rows,L)
            h=rt.embed(ids);length=h.shape[1];out=np.zeros((n,2,6,128),np.uint16)
            for b in range(40):
                layer=rt.block(b);cache=[];handle=None
                if b in [28,29]:handle=layer.self_attn.k_norm.register_forward_hook(lambda m,a,y:cache.append(y.detach().clone()))
                for lo in range(0,n,8):
                    hi=min(lo+8,n);y=layer(h[lo:hi],attention_mask=mask[lo:hi],position_ids=pos[lo:hi],position_embeddings=tuple(p[lo:hi] for p in pe),use_cache=False)
                    if handle:
                        keys=cache.pop();bi=[28,29].index(b);kv=D.HEADS[bi]['kv_head']
                        for i,r in enumerate(rows[lo:hi]):
                            offset=length-len(r['tokens']) if mode=='left_full' else 0
                            ix=[offset+r['records'][str(j)]['value_span'][-1] for j in range(6)]
                            out[lo+i,bi]=bits(keys[i,ix,kv])
                    h[lo:hi]=y
                if handle:handle.remove()
                del layer;gc.collect();torch.cuda.empty_cache()
                if (b+1)%5==0:D.log('padding/'+mode,(b+1)*n,40*n,timer,block=b)
            if mode!='right_prefix':
                indices=[length-1 if mode=='left_full' else len(r['tokens'])-1 for r in rows]
                final,lg=rt.finish(h[torch.arange(n,device='cuda'),torch.tensor(indices,device='cuda')]);scores[mode]=rt.score(lg,rows);del final,lg
            arrays[mode]=out;del h,ids,pos,pe,mask;gc.collect();torch.cuda.empty_cache()
    summary={}
    for mode,out in arrays.items():
        x=unbits(out[::2]);y=unbits(out[1::2]);rel=np.linalg.norm(x-y,axis=-1)/np.maximum(np.linalg.norm(x,axis=-1),1e-8)
        summary[mode]={'different_pairs':int(np.any(out[::2]!=out[1::2],axis=(1,2,3)).sum()),'pair_count':len(pairs),
            'max_relative_key_difference':float(rel.max()),'mean_relative_key_difference':float(rel.mean())}
    same=bool(np.array_equal(arrays['right_full'],arrays['right_prefix']))
    path=D.OUT/'padding-diagnostic.npz';assert not path.exists();np.savez(path,**arrays)
    D.save('padding-diagnostic.json',{'pairs':pairs,'summary':summary,'fixed_shape_right_full_equals_prefix_exact':same,
        'scores':scores,'left_right_choice_changes':sum(a['pred_record']!=b['pred_record'] for a,b in zip(scores['left_full'],scores['right_full'])),
        'left_right_top1_changes':sum(a['top1']!=b['top1'] for a,b in zip(scores['left_full'],scores['right_full'])),
        'canonical_test_passed':same and summary['right_full']['different_pairs']==0,'fixed_length':L,'arrays_sha256':D.sha(path),
        'scope':'Post-discovery engineering diagnosis. Same real prefix, different suffix lengths; fixed-shape right padding and full/prefix comparison. Does not repair the original v1 gate or upgrade its scientific results.'})

if __name__=='__main__':argparse.ArgumentParser().parse_args();run()
