"""One admitted pass: full-coordinate fields plus pre-RoPE QK and native content."""
import argparse,gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_formation_design as D
from source_formation_behavior import Runtime
from source_binding_native_v2 import Capture,bits,unbits

def freeze():
    assert D.read(D.OUT/'qwen14/discovery-gate.json')['passed']
    assert D.read(D.OUT/'engineering-calibration.json')['passed']
    paths=[Path(__file__),*[Path(__file__).with_name(x) for x in ['source_formation_analysis.py','source_formation_causal.py','source_formation_update.py','source_binding_native_v2.py']]]
    D.save('internal-freeze.json',{'code_sha256':{str(p.relative_to(D.ROOT)):D.sha(p) for p in paths},
        'selection':'Within each fixed block27,28,29,30 choose discovery head with highest correct-record / all-six-record mass, then absolute mass.',
        'QK_formation':'Predict post-QKnorm pre-RoPE Q/K from hs20. 128 native input coordinates selected by standardized covariance energy with joint four-head Q or K targets; random128 and embedding128 controls, discovery mean.',
        'key_fit_rows':'deterministic up to24 equally spaced valid tokens per row, includes final query; no token or family label used as target. Current query K at later blocks is a forecast from hs20, separately marked in causal/recurrent variants.',
        'formation_fit':'Each method uses128 input coordinates plus intercept; standardized ridge lambda1. Outputs are all128 native head coordinates for fourheads. Coordinate selection and coefficients discovery-only.',
        'relation_fit':'Fit original six record logits/masses using parsed source match, entity match, relative positions, lengths; additive vs source*entity interaction. Truth labels only evaluation; source/entity match is input parsing, not model mechanism.',
        'rollout':'hs20 once, state-only128-coordinate ridge remainder versus added predicted native head writes with original V, for blocks27..30; original V is privileged same-layer content, so conditional partial model. Full future V includes query V, thus recursive primary variant must exclude final-query V and renormalize forecast attention; full V local forecast reported separately.',
        'MLP':'blocks38,39; top256 variance*Downnorm units, equal256 deterministic random. Actual vector contribution norm matched per sample by damping both to min norm. No membership replacement after heldout.',
        'causal':'Correct-record versus other-source same-entity edge exclusion, and MLP selected/random suppression: pair target=.5*min(full realized norms on current branch states), damping<=1. Admit matching if99% of effects within5% target; never remove failed rows. Forecast arm replaces selected heads Q/K from hs20, with actual current-branch V and original downstream; no current true Q/K used for edited head. Only final token is edited.',
        'disclosure':'Blocks selected from prior experiment; new families and names remain unexposed before runs. Native values BF16 bits. All raw residual coordinates retain canonical order.'})

def verify():
    D.verify();f=D.read(D.OUT/'internal-freeze.json')
    amendment=D.OUT/'internal-execution-v2.json'
    overrides={}
    if amendment.exists():
        revision=D.read(amendment);assert revision['base_freeze_sha256']==D.sha(D.OUT/'internal-freeze.json')
        assert D.sha(D.OUT/'pre-index-fix-sources.zip')==revision['original_archive_sha256']
        overrides=revision['code_overrides']
    for path,digest in f['code_sha256'].items():assert D.sha(D.ROOT/path)==overrides.get(path,digest),f'Frozen internal code changed: {path}'

def decode(cap,rows,L,kw):
    rt=cap.rt;t=rt.t;x=cap.cache;n=len(rows)
    qp=x['q'].reshape(n,L,rt.heads,rt.hd).transpose(1,2)
    kp=x['k'].reshape(n,L,rt.kv,rt.hd).transpose(1,2)
    q,k=rt.mod.apply_rotary_pos_emb(qp,kp,*kw['position_embeddings'])
    v=x['v'].reshape(n,L,rt.kv,rt.hd).transpose(1,2)
    kr=rt.mod.repeat_kv(k,rt.heads//rt.kv);vr=rt.mod.repeat_kv(v,rt.heads//rt.kv)
    score=(q[:,:,-1:].float()@kr.float().transpose(-1,-2))[:,:,0]*rt.hd**-.5
    valid=t.arange(L,device=q.device)[None]>=t.tensor([L-len(r['tokens']) for r in rows],device=q.device)[:,None]
    p=score.masked_fill(~valid[:,None],float('-inf')).softmax(-1)
    predicted=(p[:,:,None]@vr.float())[:,:,0];actual=x['head'].reshape(n,rt.heads,rt.hd)
    error=(predicted-actual.float()).flatten(1).norm(dim=1)/actual.float().flatten(1).norm(dim=1).clamp_min(1e-8)
    assert float(error.max())<.02
    mass=t.zeros((n,rt.heads,6),device=q.device)
    for i,r in enumerate(rows):
        for j in range(6):mass[i,:,j]=p[i,:,t.tensor(r['records'][str(j)]['span'],device=q.device)+L-len(r['tokens'])].sum(-1)
    return {'qpre':qp[:,:,-1],'kpre':kp,'vproj':v,'attention':p,'head_context':actual,'mass':mass,'error':error}

def collect(model,split):
    proof=D.admission(model,split);verify()
    if split!='discovery':
        candidate=D.read(D.OUT/model/'formation-fit.json')
        assert candidate['freeze_sha256']==D.sha(D.OUT/'internal-freeze.json')
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    root=D.OUT/model;folder=root/f'{split}-native';assert not folder.exists()
    rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split];n=len(rows);Lmax=max(len(r['tokens']) for r in rows)
    folder.mkdir();D.save(f'{model}/{split}-capture-started.json',{'admission':proof},model)
    timer=time.monotonic();arrays={};items=[];checks=[];blocks=D.BLOCKS;mb=[38,39]
    with GpuLease(),torch.inference_mode():
        rt=Runtime(model);H,KV,dim,W=rt.heads,rt.kv,rt.dim,rt.width
        shapes={'hidden':((n,41,6,dim),np.uint16),'early':((n,Lmax,dim),np.uint16),'postattn':((n,40,dim),np.uint16),
            'final_norm':((n,dim),np.uint16),'qpre':((n,4,H,rt.hd),np.uint16),
            'kpre':((n,4,KV,Lmax,rt.hd),np.uint16),'vproj':((n,4,KV,Lmax,rt.hd),np.uint16),
            'attention':((n,4,H,Lmax),np.float32),'head_context':((n,4,H,rt.hd),np.uint16),
            'mass':((n,4,H,6),np.float32),'qkv_error':((n,4),np.float32),
            **{x:((n,2,W),np.uint16) for x in ['gate','value','units']},
            'mlp_input':((n,2,dim),np.uint16),'mlp_output':((n,2,dim),np.uint16)}
        estimate=sum(int(np.prod(s))*np.dtype(t).itemsize for s,t in shapes.values())
        D.check_write(folder,estimate,operation='来源形成原生字段空间检查')
        existing=sum(p.stat().st_size for p in D.OUT.rglob('*') if p.is_file())
        assert existing+estimate<56*2**30,'Campaign disk budget'
        for name,(shape,dtype) in shapes.items():arrays[name]=np.lib.format.open_memmap(folder/f'{name}.npy',mode='w+',dtype=dtype,shape=shape)
        norms=np.zeros((2,W),np.float32);D.log(model+'/'+split+'/native',0,n*40,timer,bytes=estimate)
        for chunk in range(0,n,128):
            if chunk:
                for name,a in list(arrays.items()):a.flush();a._mmap.close();arrays[name]=np.lib.format.open_memmap(folder/f'{name}.npy',mode='r+')
                gc.collect()
            rs=rows[chunk:chunk+128];ids,pos,pe,mask=rt.context(rs);h=rt.embed(ids);L=h.shape[1]
            for i,r in enumerate(rs):arrays['hidden'][chunk+i,0]=bits(h[i,[L-len(r['tokens'])+v for v in r['positions']]])
            for b in range(40):
                if b==D.EARLY:arrays['early'][chunk:chunk+len(rs),-L:]=bits(h)
                layer=rt.block(b);cap=Capture(rt,layer) if b in blocks or b in mb else None
                post=[]
                handle=None if cap else layer.post_attention_layernorm.register_forward_pre_hook(lambda m,a:post.append(a[0][:,-1].detach().clone()))
                if b in mb and chunk==0:norms[mb.index(b)]=layer.mlp.down_proj.weight.float().square().sum(0).cpu().numpy()
                for lo in range(0,len(rs),8):
                    hi=min(lo+8,len(rs));batch=rs[lo:hi];dest=slice(chunk+lo,chunk+hi)
                    kw={'attention_mask':mask[lo:hi],'position_ids':pos[lo:hi],'position_embeddings':tuple(p[lo:hi] for p in pe),'use_cache':False}
                    y=layer(h[lo:hi],**kw)
                    for i,r in enumerate(batch):arrays['hidden'][chunk+lo+i,b+1]=bits(y[i,[L-len(r['tokens'])+v for v in r['positions']]])
                    arrays['postattn'][dest,b]=bits(cap.cache['post'] if cap else post.pop())
                    if b in blocks:
                        bi=blocks.index(b);data=decode(cap,batch,L,kw)
                        for name in ['qpre','head_context','mass']:arrays[name][dest,bi]=bits(data[name]) if name!='mass' else data[name].cpu().numpy()
                        for name in ['kpre','vproj']:arrays[name][dest,bi,:,-L:]=bits(data[name])
                        arrays['attention'][dest,bi,:,-L:]=data['attention'].cpu().numpy();arrays['qkv_error'][dest,bi]=data['error'].cpu().numpy()
                        del data
                    if b in mb:
                        bi=mb.index(b);x=cap.cache
                        if 'gate_up' in x:g,v=x['gate_up'].chunk(2,-1)
                        else:g,v=x['gate'],x['value']
                        assert torch.equal(torch.nn.functional.silu(g)*v,x['units'])
                        for name,val in [('gate',g),('value',v),('units',x['units']),('mlp_input',x['mlp_input']),('mlp_output',x['mlp_output'])]:arrays[name][dest,bi]=bits(val)
                        if lo==0:
                            replay=torch.nn.functional.linear(x['units'],layer.mlp.down_proj.weight)
                            er=float((replay.float()-x['mlp_output'].float()).norm()/x['mlp_output'].float().norm().clamp_min(1e-8))
                            assert er<.02;checks.append({'chunk':chunk,'block':b,'down_relative_error':er,'SwiGLU_exact':True})
                    h[lo:hi]=y
                    if cap:cap.cache.clear()
                if cap:cap.close()
                if handle:handle.remove()
                del cap,layer;gc.collect();torch.cuda.empty_cache()
                if (b+1)%5==0:D.log(model+'/'+split+'/native',chunk*40+(b+1)*len(rs),n*40,timer,block=b)
            final,lg=rt.finish(h[:,-1]);arrays['final_norm'][chunk:chunk+len(rs)]=bits(final);items+=rt.score(lg,rs)
            del h,ids,pos,pe,mask,final,lg;gc.collect();torch.cuda.empty_cache()
        np.save(folder/'down-norm2.npy',norms)
    for a in arrays.values():a.flush();a._mmap.close()
    arrays.clear();base=D.read(root/f'{split}-behavior.json')['rows']
    diff=max(abs(x-y) for a,b in zip(items,base) for x,y in zip(a['scores'],b['scores']));same=all(a['top1']==b['top1'] for a,b in zip(items,base))
    D.save(f'{model}/{split}-native-manifest.json',{'n':n,'length':Lmax,'head_blocks':blocks,'mlp_blocks':mb,'admission':proof,
        'encoding':'uint16 BF16 bits; FP32 attention replays; qpre/kpre after actual QKnorm before actual RoPE',
        'early':'raw hs20 entire prefix; hidden stores all41 raw boundaries at six positions, including embedding; final norm separate',
        'replay':{'passed':same and diff==0,'top1_equal':same,'choice_logits_max_difference':diff},'checks':checks,
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':D.sha(p)} for p in folder.glob('*.npy')},
        'elapsed_seconds':time.monotonic()-timer,'review_native_by':'2026-12-10'},model)
    assert same and diff==0

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','collect']);ap.add_argument('--model');ap.add_argument('--split');a=ap.parse_args()
    if a.stage=='freeze':freeze()
    else:collect(a.model,a.split)
