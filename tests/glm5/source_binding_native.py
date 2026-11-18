"""Natural QK/VO and SwiGLU collection after complete behavior admission."""
import gc,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D
from source_binding_runtime import Runtime

def bits(x):
    import torch
    return x.contiguous().view(torch.uint16).cpu().numpy()

def unbits(x):return (np.asarray(x,dtype=np.uint32)<<16).view(np.float32)

class Capture:
    def __init__(self,rt,layer):
        self.rt=rt;self.layer=layer;self.cache={};self.handles=[]
        att=layer.self_attn
        for name,module in [('q',getattr(att,'q_norm',att.q_proj)),('k',getattr(att,'k_norm',att.k_proj)),('v',att.v_proj)]:
            self.handles.append(module.register_forward_hook(self.out(name,False)))
        self.handles.append(att.o_proj.register_forward_pre_hook(self.ins('head')))
        self.handles.append(layer.post_attention_layernorm.register_forward_pre_hook(self.ins('post')))
        self.handles.append(layer.mlp.register_forward_pre_hook(self.ins('mlp_input')))
        self.handles.append(layer.mlp.register_forward_hook(self.out('mlp_output')))
        if hasattr(layer.mlp,'gate_up_proj'):
            self.handles.append(layer.mlp.gate_up_proj.register_forward_hook(self.out('gate_up')))
        else:
            self.handles.append(layer.mlp.gate_proj.register_forward_hook(self.out('gate')))
            self.handles.append(layer.mlp.up_proj.register_forward_hook(self.out('value')))
        self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self.ins('units')))
    def out(self,name,last=True):
        def fn(mod,args,y):self.cache[name]=(y[:,-1] if last else y).detach().clone()
        return fn
    def ins(self,name):
        def fn(mod,args):self.cache[name]=args[0][:,-1].detach().clone()
        return fn
    def close(self):
        for h in self.handles:h.remove()
        self.cache.clear()
    def decode(self,rows,length,context):
        t=self.rt.t;rt=self.rt;x=self.cache;b=len(rows);hd=rt.hd
        q=x['q'].reshape(b,length,rt.heads,hd).transpose(1,2);k=x['k'].reshape(b,length,rt.kv,hd).transpose(1,2)
        q,k=rt.mod.apply_rotary_pos_emb(q,k,*context['position_embeddings'])
        v=x['v'].reshape(b,length,rt.kv,hd).transpose(1,2)
        kr=rt.mod.repeat_kv(k,rt.heads//rt.kv);vr=rt.mod.repeat_kv(v,rt.heads//rt.kv)
        scores=(q[:,:,-1:].float()@kr.float().transpose(-1,-2))[:, :, 0]*hd**-.5
        valid=t.arange(length,device='cuda')[None]>=t.tensor([length-len(r['tokens']) for r in rows],device='cuda')[:,None]
        scores=scores.masked_fill(~valid[:,None],float('-inf'));p=scores.softmax(-1)
        replay=(p[:,:,None]@vr.float())[:,:,0];actual=x['head'].reshape(b,rt.heads,hd)
        error=(replay-actual.float()).flatten(1).norm(dim=1)/actual.float().flatten(1).norm(dim=1).clamp_min(1e-8)
        assert float(error.max())<.02,('QKV replay',float(error.max()))
        mass=t.zeros((b,rt.heads,8),device='cuda')
        for i,r in enumerate(rows):
            off=length-len(r['tokens']);groups=[r['records'][str(j)]['span'] for j in range(4)]+[r['headers'][str(j)] for j in range(2)]+[r['query_positions']]
            used=set()
            for j,idx in enumerate(groups):
                loc=sorted(set(idx)-used);used.update(idx)
                if loc:mass[i,:,j]=p[i,:,t.tensor([off+x for x in loc],device='cuda')].sum(-1)
            other=[off+j for j in range(len(r['tokens'])) if j not in used]
            mass[i,:,7]=p[i,:,other].sum(-1)
        assert float((mass.sum(-1)-1).abs().max())<1e-5
        if 'gate_up' in x:g,value=x['gate_up'].chunk(2,-1)
        else:g,value=x['gate'],x['value']
        unit=t.nn.functional.silu(g)*value;equal=bool(t.equal(unit,x['units']));assert equal,'Native SwiGLU identity'
        return {'q':q[:,:,-1], 'k':k,'v':v,'p':p,'head':actual,'source_mass':mass,
                'qkv_error':error,'gate':g,'value':value,'units':x['units'],'post':x['post'],
                'mlp_input':x['mlp_input'],'mlp_output':x['mlp_output']}

def collect(model,split,proof):
    import torch
    from shared.dist_protocol.mechanism_trace import GpuLease
    assert D.admission(model,split)==proof
    root=D.OUT/model;folder=root/f'{split}-native';assert not folder.exists(),'Native stage already started'
    if split!='discovery':
        candidates=D.read(root/'candidates.json');assert candidates['discovery_manifest_sha256']==D.sha(root/'discovery-native-manifest.json')
        hb=sorted({x['block'] for x in candidates['heads']});mb=sorted({x['block'] for x in candidates['mlps']})
    else:hb=list(range(40));mb=list(range(40))
    rows=D.read(D.OUT/f'{model}-materials.json')['splits'][split];n=len(rows);length=max(len(r['tokens']) for r in rows)
    D.check_write(folder,12*2**30,operation='来源绑定已准入原生采集');folder.mkdir()
    D.save(f'{model}/{split}-capture-started.json',{'admission':proof,'stage':'internal_after_complete_behavior','head_blocks':hb,'mlp_blocks':mb},model)
    start=time.monotonic();arrays={};checks=[];items=[]
    with GpuLease(),torch.inference_mode():
        rt=Runtime(model);dim,H,KV,W=rt.dim,rt.heads,rt.kv,rt.width
        shapes={'hidden':((n,41,5,dim),np.uint16),'postattn':((n,40,dim),np.uint16),'final_norm':((n,dim),np.uint16),
                'source_mass':((n,40,H,8),np.float32),'qkv_error':((n,40),np.float32),
                'query':((n,len(hb),H,128),np.uint16),'key':((n,len(hb),KV,length,128),np.uint16),
                'vproj':((n,len(hb),KV,length,128),np.uint16),'attention':((n,len(hb),H,length),np.float32),
                'head_context':((n,len(hb),H,128),np.uint16),
                **{name:((n,len(mb),W),np.uint16) for name in ['gate','value','units']},
                'mlp_input':((n,len(mb),dim),np.uint16),'mlp_output':((n,len(mb),dim),np.uint16)}
        estimated=sum(int(np.prod(shape))*np.dtype(dtype).itemsize for shape,dtype in shapes.values())
        D.check_write(folder,estimated,operation='来源绑定原生数组精确空间检查')
        existing=sum(p.stat().st_size for p in D.OUT.rglob('*') if p.is_file())
        assert existing+estimated<32*2**30,'Frozen campaign disk budget exceeded'
        D.log(model+'/'+split+'/native',0,n*40,start,estimated_array_bytes=estimated)
        for name,(shape,dtype) in shapes.items():arrays[name]=np.lib.format.open_memmap(folder/f'{name}.npy',mode='w+',dtype=dtype,shape=shape)
        downnorms=np.zeros((len(mb),W),np.float32)
        for chunk in range(0,n,128):
            rs=rows[chunk:chunk+128];ids,pos,pe,mask=rt.context(rs);h=rt.embed(ids);L=h.shape[1]
            for i,r in enumerate(rs):
                ix=[L-len(r['tokens'])+p for p in r['positions']];arrays['hidden'][chunk+i,0]=bits(h[i,ix])
            for b in range(40):
                layer=rt.block(b);cap=Capture(rt,layer)
                if b in mb and chunk==0:downnorms[mb.index(b)]=layer.mlp.down_proj.weight.float().square().sum(0).cpu().numpy()
                for lo in range(0,len(rs),8):
                    hi=min(lo+8,len(rs));batch=rs[lo:hi];kw={'attention_mask':mask[lo:hi],'position_ids':pos[lo:hi],
                        'position_embeddings':tuple(p[lo:hi] for p in pe),'use_cache':False}
                    y=layer(h[lo:hi],**kw);data=cap.decode(batch,L,kw);dest=slice(chunk+lo,chunk+hi)
                    for i,r in enumerate(batch):
                        ix=[L-len(r['tokens'])+p for p in r['positions']];arrays['hidden'][chunk+lo+i,b+1]=bits(y[i,ix])
                    arrays['postattn'][dest,b]=bits(data['post']);arrays['source_mass'][dest,b]=data['source_mass'].cpu().numpy()
                    arrays['qkv_error'][dest,b]=data['qkv_error'].cpu().numpy()
                    if b in hb:
                        bi=hb.index(b);arrays['query'][dest,bi]=bits(data['q']);arrays['key'][dest,bi,:,-L:]=bits(data['k'])
                        arrays['vproj'][dest,bi,:,-L:]=bits(data['v']);arrays['attention'][dest,bi,:,-L:]=data['p'].cpu().numpy()
                        arrays['head_context'][dest,bi]=bits(data['head'])
                    if b in mb:
                        bi=mb.index(b)
                        for name in ['gate','value','units','mlp_input','mlp_output']:arrays[name][dest,bi]=bits(data[name])
                        if lo==0:
                            replay=torch.nn.functional.linear(data['units'],layer.mlp.down_proj.weight)
                            err=float((replay.float()-data['mlp_output'].float()).norm()/data['mlp_output'].float().norm().clamp_min(1e-8))
                            assert err<.02;checks.append({'chunk':chunk,'block':b,'MLP_down_relative_error':err,'SwiGLU_bitwise_equal':True})
                    h[lo:hi]=y;cap.cache.clear();del data,y
                cap.close();del cap,layer;gc.collect();torch.cuda.empty_cache()
                if (b+1)%5==0:
                    for a in arrays.values():a.flush()
                    D.log(model+'/'+split+'/native',chunk*40+(b+1)*len(rs),n*40,start,block=b)
            final,lg=rt.finish(h[:,-1]);arrays['final_norm'][chunk:chunk+len(rs)]=bits(final);items+=rt.score(lg,rs)
            del h,ids,pos,pe,mask,final,lg;gc.collect();torch.cuda.empty_cache()
        np.save(folder/'down-column-norm2.npy',downnorms)
    for a in arrays.values():a.flush()
    arrays.clear()
    baseline=D.read(root/f'{split}-behavior.json')['rows']
    agreement=float(np.mean([x['top1']==y['top1'] for x,y in zip(items,baseline)]))
    maxdiff=max(float(np.max(np.abs(np.array(x['scores'])-np.array(y['scores'])))) for x,y in zip(items,baseline))
    D.save(f'{model}/{split}-native-manifest.json',{'n':n,'head_blocks':hb,'mlp_blocks':mb,'length':length,'row_ids':[r['id'] for r in rows],
        'encoding':'uint16 BF16 bits for raw model states; attention and source_mass are FP32 diagnostic replays of BF16 SDPA',
        'hidden_boundaries':'embedding then40 raw decoder outputs; final_norm stored separately. Five positions follow per-row positions.',
        'native_head_coordinates':'post-QK-normalization and post-RoPE; GQA KV sharing follows model configuration',
        'admission':proof,'checks':checks,'base_replay':{'top1_agreement':agreement,'max_choice_logit_difference':maxdiff,'passed':agreement==1 and maxdiff==0},
        'files':{p.name:{'bytes':p.stat().st_size,'sha256':D.sha(p)} for p in folder.glob('*.npy')},
        'elapsed_seconds':time.monotonic()-start,'review_by':'2026-12-10','purpose':'native binding/conditional-update evidence, reusable; no automatic causal promotion'},model)
    assert agreement==1 and maxdiff==0,'Collection changed natural forward output'
