"""Admitted single capture, native coordinate order and BF16 bit preservation."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_relay2_design as D
from source_relay2_behavior import Runtime
from source_binding_native_v2 import Capture as BaseCapture,bits,unbits

class Capture(BaseCapture):
    def ins(self,name):
        def fn(mod,args):self.cache[name]=(args[0] if name=='head' else args[0][:,-1]).detach().clone()
        return fn

def decode(cap,rows,L,kw):
    rt=cap.rt;t=rt.t;x=cap.cache;n=len(rows);lengths=t.tensor([len(r['tokens']) for r in rows],device='cuda');ii=t.arange(n,device='cuda')
    qp=x['q'].reshape(n,L,rt.heads,rt.hd).transpose(1,2);kp=x['k'].reshape(n,L,rt.kv,rt.hd).transpose(1,2)
    q,k=rt.mod.apply_rotary_pos_emb(qp,kp,*kw['position_embeddings']);v=x['v'].reshape(n,L,rt.kv,rt.hd).transpose(1,2)
    qlast=q.transpose(1,2)[ii,lengths-1];kr=rt.mod.repeat_kv(k,rt.heads//rt.kv);vr=rt.mod.repeat_kv(v,rt.heads//rt.kv)
    score=(qlast[:,:,None].float()@kr.float().transpose(-1,-2))[:,:,0]*rt.hd**-.5
    valid=t.arange(L,device='cuda')[None]<lengths[:,None];p=score.masked_fill(~valid[:,None],float('-inf')).softmax(-1)
    actual=x['head'].reshape(n,L,rt.heads,rt.hd)[ii,lengths-1];replay=(p[:,:,None]@vr.float())[:,:,0]
    error=(replay-actual.float()).flatten(1).norm(dim=1)/actual.float().flatten(1).norm(dim=1).clamp_min(1e-8);assert float(error.max())<.02
    return {'qpre':qp.transpose(1,2)[ii,lengths-1],'kpre':kp,'vproj':v,'attention':p,'head_context':actual,'error':error}

def freeze():
    D.admission('discovery')
    assert D.read(D.OUT/'engineering.json')['passed']
    paths=[Path(__file__),*[Path(__file__).with_name(f) for f in ['source_relay2_math.py','source_relay2_fit.py','source_relay2_evaluate.py',
        'source_relay2_causal.py','source_relay2_parameters.py','source_relay2_update.py','source_binding_native_v2.py','source_formation_native.py','source_formation_analysis.py']]]
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
        existing=sum(p.stat().st_size for p in D.OUT.rglob('*') if p.is_file());assert existing+estimate<22*2**30
        folder.mkdir();D.save(f'{D.MODEL}/{split}-native-started.json',{'admission':proof,'estimated_bytes':estimate})
        for name,(shape,dtype) in specs.items():arrays[name]=np.lib.format.open_memmap(folder/f'{name}.npy',mode='w+',dtype=dtype,shape=shape)
        D.log(split+'/native',0,n*40,timer,estimated_bytes=estimate)
        for chunk in range(0,n,128):
            if chunk:
                for name,a in list(arrays.items()):a.flush();a._mmap.close();arrays[name]=np.lib.format.open_memmap(folder/f'{name}.npy',mode='r+')
                gc.collect()
            rs=rows[chunk:chunk+128];ids,pos,pe,mask=rt.context(rs);h=rt.embed(ids);L=h.shape[1]
            for i,r in enumerate(rs):arrays['hidden'][chunk+i,0]=bits(h[i,r['positions']])
            for b in range(40):
                if b==D.EARLY:
                    for i,r in enumerate(rs):arrays['early'][chunk+i,-len(r['tokens']):]=bits(h[i,:len(r['tokens'])])
                if 20<=b<=30:
                    for i,r in enumerate(rs):arrays['trace_state'][chunk+i,b-20]=bits(h[i,r['trace_positions']])
                layer=rt.block(b);cap=Capture(rt,layer) if b in [28,29] else None
                post=[];handle=layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a:post.append(a[0].detach().clone())) if 20<=b<30 else None
                for lo in range(0,len(rs),8):
                    hi=min(lo+8,len(rs));batch=rs[lo:hi];dest=slice(chunk+lo,chunk+hi)
                    kw={'attention_mask':mask[lo:hi],'position_ids':pos[lo:hi],'position_embeddings':tuple(p[lo:hi] for p in pe),'use_cache':False}
                    y=layer(h[lo:hi],**kw)
                    for i,r in enumerate(batch):
                        arrays['hidden'][chunk+lo+i,b+1]=bits(y[i,r['positions']])
                    if handle:
                        v=post.pop()
                        for i,r in enumerate(batch):arrays['trace_postattn'][chunk+lo+i,b-20]=bits(v[i,r['trace_positions']])
                        del v
                    if cap:
                        bi=[28,29].index(b);c=D.HEADS[bi];data=decode(cap,batch,L,kw)
                        for name in ['qpre','head_context']:arrays[name][dest,bi]=bits(data[name][:,c['head']])
                        for i,r in enumerate(batch):
                            N=len(r['tokens'])
                            for name in ['kpre','vproj']:arrays[name][chunk+lo+i,bi,-N:]=bits(data[name][i,c['kv_head'],:N])
                            arrays['attention'][chunk+lo+i,bi,-N:]=data['attention'][i,c['head'],:N].cpu().numpy()
                        checks.append({'chunk':chunk,'offset':lo,'block':b,'all_head_replay_relative_error':float(data['error'].max())})
                        del data;cap.cache.clear()
                    h[lo:hi]=y
                if cap:cap.close()
                if handle:handle.remove()
                del layer,cap;gc.collect();torch.cuda.empty_cache()
                if (b+1)%5==0:D.log(split+'/native',chunk*40+(b+1)*len(rs),n*40,timer,block=b)
            final,lg=rt.finish(h[torch.arange(len(rs),device='cuda'),torch.tensor([len(r['tokens'])-1 for r in rs],device='cuda')]);arrays['final_norm'][chunk:chunk+len(rs)]=bits(final);items+=rt.score(lg,rs)
            del ids,pos,pe,mask,h,final,lg;gc.collect();torch.cuda.empty_cache()
    for a in arrays.values():a.flush();a._mmap.close()
    arrays.clear();base=D.read(root/f'{split}-behavior.json')['rows']
    diff=max(abs(x-y) for a,b in zip(items,base) for x,y in zip(a['scores'],b['scores']));same=all(a['top1']==b['top1'] for a,b in zip(items,base))
    assert diff==0 and same,'Behavior/native baseline identity failed'
    keys=np.load(folder/'kpre.npy',mmap_mode='r');groups={};prefix_comparisons=0
    for i,r in enumerate(rows):groups.setdefault((r['family'],r['order']),[]).append(i)
    for ii in groups.values():
        a=ii[0];ar=rows[a]
        for j in range(6):
            ap=ar['records'][str(j)]['value_span'][-1];reference=np.array(keys[a,:,ap+Lmax-len(ar['tokens'])])
            for i in ii[1:]:
                r=rows[i];rp=r['records'][str(j)]['value_span'][-1]
                assert ar['tokens'][:ap+1]==r['tokens'][:rp+1]
                assert np.array_equal(reference,keys[i,:,rp+Lmax-len(r['tokens'])]),'Canonical prefix K invariance failed'
                prefix_comparisons+=2
    keys._mmap.close()
    D.save(f'{D.MODEL}/{split}-native-manifest.json',{'n':n,'length':Lmax,'heads':D.HEADS,'admission':proof,
        'replay':{'passed':True,'choice_logits_max_difference':diff,'top1_equal':same},'checks':checks,
        'prefix_invariance':{'passed':True,'head_record_comparisons':prefix_comparisons,'different':0},
        'encoding':'uint16 exact BF16 bits; fixed-shape right-padded CUDA execution. Valid prefix arrays stored right-aligned for downstream indexing; storage padding zero and masked in fits/evaluation.',
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':D.sha(p)} for p in folder.glob('*.npy')},
        'elapsed_seconds':time.monotonic()-timer,'review_native_by':'2026-12-10'})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','collect']);ap.add_argument('--split');a=ap.parse_args()
    if a.stage=='freeze':freeze()
    else:collect(a.split)
