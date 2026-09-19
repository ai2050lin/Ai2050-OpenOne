"""Frozen earlier-state predictions executed inside the native autoregressive model."""
from rdc_law_common import *
from rdc_law_predict import Bank,kernel_torch


class Live:
    def __init__(self,model):
        import torch
        self.model=model;self.device=model.get_input_embeddings().weight.device
        self.enabled=True;self.branch='native';self.local=False;self.handles=[];self.reset(0)
        frozen=read(BASE/'prediction/frozen.json')
        self.banks={b:Bank(BASE/frozen['winners'][str(b)]['bank'],str(self.device)) for b in (16,35)}
        with np.load(BASE/'prediction/training_features.npz') as z:
            self.training={k:torch.as_tensor(z[k],device=self.device,dtype=torch.int64 if k=='class' else torch.float64)
                for k in ('q','embedding','history','routed','position','class')}
        with np.load(BASE/'prediction/feature_rulers.npz') as z:
            self.rulers={k:{t:torch.as_tensor(z[k+'_'+t],device=self.device,dtype=torch.float64) for t in ('mean','scale')} for k in ('q','embedding','history','routed')}
        self.weights={b:{k:getattr(model.model.layers[b].mlp,name+'_proj').weight.detach().float() for k,name in [('g','gate'),('u','up'),('d','down')]} for b in (16,35)}
        self.original={k:getattr(model.model.layers[35].mlp,name+'_proj').weight.detach().clone() for k,name in [('g','gate'),('u','up'),('d','down')]}
        self.trained_weights={}
        def emb(m,a,o):
            if self.enabled:self.embedding=o[0]
        self.handles.append(model.get_input_embeddings().register_forward_hook(emb))
        def attention(m,a,o):
            if self.enabled:
                assert o[1] is not None
                self.attention=o[1][0].float().mean(0)
        self.handles.append(model.model.layers[11].self_attn.register_forward_hook(attention))
        def h12(m,a,o):
            if not self.enabled:return
            h=(o[0] if isinstance(o,tuple) else o)[0]
            self.current_h12=h;self.offset=0 if self.source is None else len(self.source)
            v=h.float();v=v/v.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12)
            self.source=v if self.source is None else torch.cat([self.source,v])
            self.query_fields[12]=bits(h[-1])
        self.handles.append(model.model.layers[11].register_forward_hook(h12))
        for b,layer in enumerate(model.model.layers):
            def hidden(m,a,o,b=b):
                if self.enabled:
                    v=(o[0] if isinstance(o,tuple) else o)[0,-1]
                    self.query_fields[b+1]=bits(v)
            self.handles.append(layer.register_forward_hook(hidden))
        for b in (16,35):
            def hook(m,a,o,b=b):
                if not self.enabled:return o
                if self.branch==f'early_prediction_L{b}':
                    # Scope changes WHERE we write, not GEMM/reduction shape.
                    # Point-vs-batch forecasting can move BF16 rounding ties.
                    positions=list(range(o.shape[1]))
                    pred=self.predict(b,positions);new=o.clone()
                    if self.local:new[0,-1]=pred[-1].to(o.dtype)
                    else:new[0,positions]=pred.to(o.dtype)
                    self.predicted_writeback[b]=bits(new[0,-1]);return new
                if b==35 and self.local and self.branch not in ('native','early_prediction_L16','early_prediction_L35'):
                    # Query-only trained-MLP control: keep actual updated last query but
                    # compute earlier positions with exact original BF16 matrices.
                    import torch.nn.functional as F
                    x=a[0];base=F.linear(F.silu(F.linear(x,self.original['g']))*F.linear(x,self.original['u']),self.original['d'])
                    base[:,-1]=o[:,-1];return base
                return o
            self.handles.append(model.model.layers[b].mlp.register_forward_hook(hook))

    def reset(self,cls):
        self.cls=cls;self.source=None;self.offset=0;self.embedding=None;self.attention=None;self.current_h12=None
        self.query_fields={};self.predicted_writeback={}

    def features(self,positions):
        import torch
        at=torch.tensor(positions,device=self.device);absolute=at+self.offset
        raw={'q':self.current_h12[at].double(),'embedding':self.embedding[at].double()}
        for k in ('q','embedding'):raw[k]=raw[k]/raw[k].square().mean(-1,keepdim=True).sqrt().clamp_min(1e-12)
        # All source vectors and all coordinates, causal prefixes only. Float32
        # source normalization follows collection; sums below use Float64.
        raw['history']=self.source.double().cumsum(0)[absolute]/(absolute.double()+1)[:,None]
        att=self.attention[at].double();att=att/att.sum(-1,keepdim=True)
        raw['routed']=att@self.source.double()
        # Collector rounded summaries to float32 before predictor normalization.
        for k in ('history','routed'):raw[k]=raw[k].float().double()
        for k in raw:raw[k]=(raw[k]-self.rulers[k]['mean'])/self.rulers[k]['scale']
        raw['position']=torch.log1p(absolute.double())/np.log(2048.)
        raw['class']=torch.full((len(at),),self.cls,dtype=torch.int64,device=self.device)
        return raw

    def predict(self,b,positions):
        import torch
        result=[];bank=self.banks[b]
        for start in range(0,len(positions),64):
            f=self.features(positions[start:start+64]);k=kernel_torch(bank.kernel,f,self.training)
            result.append(bank.predict(k,self.weights[b]))
        return torch.cat(result)

    def set_branch(self,name,local=False):
        import torch
        if name!=self.branch:
            if name in ('native','early_prediction_L16','early_prediction_L35'):
                replacement=self.original
            else:
                if name not in self.trained_weights:
                    with np.load(BASE/'formation/trajectories'/name/'final_native_parameter_deltas.npz') as z:
                        self.trained_weights[name]={k:(v.float()+torch.as_tensor(z[k],device=v.device)).to(v.dtype) for k,v in self.original.items()}
                    rows=[]
                    for k,v in self.original.items():
                        changed=self.trained_weights[name][k];diff=changed.float()-v.float()
                        rows.append({'matrix':k,'parameters':v.numel(),'BF16_changed_parameters':int((changed!=v).sum()),
                            'BF16_relative_displacement':float(diff.square().sum().sqrt()/v.float().square().sum().sqrt()),
                            'actual_deployed_tensor':identity(bits(changed))})
                    save(BASE/'deployment/actual_bf16_parameters'/f'{name}.json',{'timestamp':stamp(),'matrices':rows,
                        'operation':'FP32(originalBF16value)+savedFP32delta, then BF16cast, no original checkpoint writes.'})
                replacement=self.trained_weights[name]
            with torch.no_grad():
                for k,v in self.original.items():
                    target=getattr(self.model.model.layers[35].mlp,{'g':'gate','u':'up','d':'down'}[k]+'_proj').weight
                    target.copy_(replacement[k])
        self.branch=name;self.local=local

    def close(self):
        self.set_branch('native')
        for h in self.handles:h.remove()


def compare_caches(left,right):
    import torch
    assert len(left.layers)==len(right.layers)
    result=[]
    for b,(a,c) in enumerate(zip(left.layers,right.layers)):
        for name in ('keys','values'):
            x=getattr(a,name);y=getattr(c,name);assert x.shape==y.shape
            eq=torch.equal(x,y);d=x.float()-y.float()
            result.append({'block':b,'array':name,'shape':list(x.shape),'all_elements':x.numel(),'bitwise_equal':eq,
                'max_abs':float(d.abs().max()),'relative_MSE':float(d.square().mean()/y.float().square().mean().clamp_min(1e-20))})
    return {'all_bitwise_equal':all(r['bitwise_equal'] for r in result),'total_elements':sum(r['all_elements'] for r in result),'layers':result}
