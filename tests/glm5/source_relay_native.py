"""Admitted single capture, native coordinate order and BF16 bit preservation."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_relay_design as D
from source_relay_behavior import Runtime
from source_binding_native_v2 import Capture,bits,unbits
from source_formation_native import decode

def freeze():
    D.admission('discovery')
    assert D.read(D.OUT/'engineering.json')['passed']
    paths=[Path(__file__),*[Path(__file__).with_name(f) for f in ['source_relay_math.py','source_relay_fit.py','source_relay_evaluate.py',
        'source_relay_causal.py','source_relay_parameters.py','source_relay_update.py','source_binding_native_v2.py','source_formation_native.py','source_formation_analysis.py']]]
    D.save('internal-freeze.json',{'code_sha256':{str(p.relative_to(D.ROOT)):D.sha(p) for p in paths},
        'heads':D.HEADS,'native':'All41 residual boundaries at query-source/query-entity/question-end/final; all hs20 prefix; selected-head preRoPE QK/V and attention; five trace roles with native states20..30 and postattention20..29.',
        'trace':'Effective attention and MLP residual increments are differences of stored BF16 boundaries, so include rounding. Decomposition holds true normalization denominators fixed; not an independent forecast or causal attribution.',
        'forecast':'128 selected/native, random, and PCA inputs. Discovery-only fits. QK aware loss and steps fixed in plan. V ridge uses K input transform; future native V only in labelled oracle-content conditions.',
        'causal':'Plan-defined position masks and fixed-head context replacement. Full-vocabulary KL/top1 and six-choice metrics all retained. All material rows kept regardless baseline correctness.',
        'inference_budget':'Native-model hs20 at all prefix positions is observed; all later QKV learned forecasts are excluded from inputs. Actual future V ablation is explicitly privileged.',
        'secondary_recurrence':'Pre-internal additive protocol: native128 variance-selected inputs at each state, ridge1 direct next-state versus residual increment. Content versions fit direct remainder subtracting true two-head write only in training. Compare learned QKV content, native past V, and constant V. Start at observed hs20, then own predicted final states only; all outputs reported, no winner selection. Full-vocabulary agreement and KL primary output diagnostics; no claim of complete autoregressive model.'})

def collect(split):
    proof=D.admission(split);D.verify_internal();root=D.OUT/D.MODEL;folder=root/f'{split}-native';assert not folder.exists()
    if split!='discovery':assert D.read(root/'fit.json')['internal_freeze_sha256']==D.sha(D.OUT/'internal-freeze.json')
    rows=D.read(D.OUT/'qwen14-materials.json')['splits'][split];n=len(rows);Lmax=max(len(r['tokens']) for r in rows)
    import torch
    timer=time.monotonic();arrays={};items=[];checks=[]
    with D.gpu_lease(split+'/native'),torch.inference_mode():
        rt=Runtime(D.MODEL)
        for c in D.HEADS:assert c['kv_head']==c['head']//(rt.heads//rt.kv)
        specs=D.shapes(n,Lmax,rt.dim,rt.hd);estimate=sum(int(np.prod(s))*np.dtype(t).itemsize for s,t in specs.values())
        D.check_write(folder,estimate,operation='来源接力原场精确预算')
        existing=sum(p.stat().st_size for p in D.OUT.rglob('*') if p.is_file());assert existing+estimate<28*2**30
        folder.mkdir();D.save(f'{D.MODEL}/{split}-native-started.json',{'admission':proof,'estimated_bytes':estimate})
        for name,(shape,dtype) in specs.items():arrays[name]=np.lib.format.open_memmap(folder/f'{name}.npy',mode='w+',dtype=dtype,shape=shape)
        D.log(split+'/native',0,n*40,timer,estimated_bytes=estimate)
        for chunk in range(0,n,128):
            if chunk:
                for name,a in list(arrays.items()):a.flush();a._mmap.close();arrays[name]=np.lib.format.open_memmap(folder/f'{name}.npy',mode='r+')
                gc.collect()
            rs=rows[chunk:chunk+128];ids,pos,pe,mask=rt.context(rs);h=rt.embed(ids);L=h.shape[1]
            for i,r in enumerate(rs):arrays['hidden'][chunk+i,0]=bits(h[i,[L-len(r['tokens'])+p for p in r['positions']]])
            for b in range(40):
                if b==D.EARLY:arrays['early'][chunk:chunk+len(rs),-L:]=bits(h)
                if 20<=b<=30:
                    for i,r in enumerate(rs):arrays['trace_state'][chunk+i,b-20]=bits(h[i,[L-len(r['tokens'])+p for p in r['trace_positions']]])
                layer=rt.block(b);cap=Capture(rt,layer) if b in [28,29] else None
                post=[];handle=layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a:post.append(a[0].detach().clone())) if 20<=b<30 else None
                for lo in range(0,len(rs),8):
                    hi=min(lo+8,len(rs));batch=rs[lo:hi];dest=slice(chunk+lo,chunk+hi)
                    kw={'attention_mask':mask[lo:hi],'position_ids':pos[lo:hi],'position_embeddings':tuple(p[lo:hi] for p in pe),'use_cache':False}
                    y=layer(h[lo:hi],**kw)
                    for i,r in enumerate(batch):
                        arrays['hidden'][chunk+lo+i,b+1]=bits(y[i,[L-len(r['tokens'])+p for p in r['positions']]])
                    if handle:
                        v=post.pop()
                        for i,r in enumerate(batch):arrays['trace_postattn'][chunk+lo+i,b-20]=bits(v[i,[L-len(r['tokens'])+p for p in r['trace_positions']]])
                        del v
                    if cap:
                        bi=[28,29].index(b);c=D.HEADS[bi];data=decode(cap,batch,L,kw)
                        for name in ['qpre','head_context']:arrays[name][dest,bi]=bits(data[name][:,c['head']])
                        for name in ['kpre','vproj']:arrays[name][dest,bi,-L:]=bits(data[name][:,c['kv_head']])
                        arrays['attention'][dest,bi,-L:]=data['attention'][:,c['head']].cpu().numpy()
                        checks.append({'chunk':chunk,'offset':lo,'block':b,'all_head_replay_relative_error':float(data['error'].max())})
                        del data;cap.cache.clear()
                    h[lo:hi]=y
                if cap:cap.close()
                if handle:handle.remove()
                del layer,cap;gc.collect();torch.cuda.empty_cache()
                if (b+1)%5==0:D.log(split+'/native',chunk*40+(b+1)*len(rs),n*40,timer,block=b)
            final,lg=rt.finish(h[:,-1]);arrays['final_norm'][chunk:chunk+len(rs)]=bits(final);items+=rt.score(lg,rs)
            del ids,pos,pe,mask,h,final,lg;gc.collect();torch.cuda.empty_cache()
    for a in arrays.values():a.flush();a._mmap.close()
    arrays.clear();base=D.read(root/f'{split}-behavior.json')['rows']
    diff=max(abs(x-y) for a,b in zip(items,base) for x,y in zip(a['scores'],b['scores']));same=all(a['top1']==b['top1'] for a,b in zip(items,base))
    assert diff==0 and same,'Behavior/native baseline identity failed'
    D.save(f'{D.MODEL}/{split}-native-manifest.json',{'n':n,'length':Lmax,'heads':D.HEADS,'admission':proof,
        'replay':{'passed':True,'choice_logits_max_difference':diff,'top1_equal':same},'checks':checks,
        'encoding':'uint16 exact BF16 bits; attention FP32 reconstruction. Padding zero and masked in every fit/evaluation.',
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':D.sha(p)} for p in folder.glob('*.npy')},
        'elapsed_seconds':time.monotonic()-timer,'review_native_by':'2026-12-10'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','collect']);ap.add_argument('--split');a=ap.parse_args()
    if a.stage=='freeze':freeze()
    else:collect(a.split)
