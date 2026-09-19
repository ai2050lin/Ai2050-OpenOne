"""Exact trained endpoints and complete FP32/BF16 direction-radius controls."""
from rdc_formation_common import *

NAMES=['gate_proj.weight','up_proj.weight','down_proj.weight']


def exact_direction(run, originals):
    receipt=read(OUT/'training'/run/'commits/delta_128.json')
    file=ROOT/receipt['field_path']
    assert sha(file)==receipt['field_sha256']
    directions={}
    with np.load(file) as z:
        for n in NAMES:
            original=originals[n]
            if hasattr(original,'detach'):original=original.detach().float().cpu().numpy()
            original=np.asarray(original,np.float32)
            actual=(original+z[n])+z['reconstruction_residual__'+n]
            directions[n]=actual.astype(np.float64)-original.astype(np.float64)
    return directions,receipt


def variants():
    protocol=read(OUT/'training/protocol.json')
    result=[]
    for seed in protocol['seeds']:
        true=read(OUT/'training'/('true_token_'+str(seed))/'result.json')
        radii={'FP32_bridge':true['checkpoints'][-1]['delta_FP32_L2'],
               'native_BF16':true['deployed_BF16_delta_L2']}
        for condition in protocol['conditions']:
            for precision,reference in radii.items():
                for factor in [.5,1.]:
                    result.append({'name':condition+'_'+str(seed)+'_'+precision+'_r'+str(factor).replace('.','p'),
                        'run':condition+'_'+str(seed),'condition':condition,'seed':seed,'transform':'identity',
                        'precision':precision,'radius_factor':factor,'reference_radius':reference,'target_radius':factor*reference})
        for transform in ['reverse','coordinate_shuffle']:
            for precision,reference in radii.items():
                for factor in [.5,1.]:
                    result.append({'name':transform+'_'+str(seed)+'_'+precision+'_r'+str(factor).replace('.','p'),
                        'run':'true_token_'+str(seed),'condition':'true_token','seed':seed,'transform':transform,
                        'permutation_seed':27471000+seed,'precision':precision,'radius_factor':factor,
                        'reference_radius':reference,'target_radius':factor*reference})
    assert len(result)==40
    return result


class ExactDirection:
    def __init__(self,model):
        self.model=model
        self.target=model.model.layers[16].mlp
        self.original={n:p.detach().float().clone() for n,p in self.target.named_parameters()}
        self.handles=[]
        self.delta={}
        self.active=None
        self.audit=None

    def precision(self,kind):
        from phase2747_rdc_training import bridge
        for h in self.handles:h.remove()
        self.handles=[]
        if kind=='FP32_bridge':self.handles=bridge(self.target)
        else:self.target.bfloat16()

    def activate(self,variant):
        import torch
        key=(variant['run'],variant['transform'])
        if key==self.active:return
        self.delta={}
        torch.cuda.empty_cache()
        delta,receipt=exact_direction(variant['run'],self.original)
        rng=np.random.default_rng(variant.get('permutation_seed',0))
        identities={}
        for n,v in delta.items():
            before=identity(v)
            if variant['transform']=='reverse':v=-v
            elif variant['transform']=='coordinate_shuffle':
                order=rng.permutation(v.size)
                v=v.ravel()[order].reshape(v.shape)
                del order
            identities[n]={'original_direction_identity':before,'transformed_identity':identity(v),'all_scalars':v.size}
            self.delta[n]=torch.tensor(v,device='cuda',dtype=torch.float64)
        norm=float(torch.stack([v.square().sum() for v in self.delta.values()]).sum().sqrt())
        self.active=key
        self.audit={'source':receipt,'parameters':identities,'complete_parameter_scalars':74711040,
                    'exact_FP64_direction_norm':norm,'definition':'Exact FP64difference of actualFP32endpoint and originalBF16-valuedFP32endpoint; checkpoint reconstruction residual included.'}

    def values(self,alpha,precision):
        import torch
        dtype=torch.float32 if precision=='FP32_bridge' else torch.bfloat16
        for n,original in self.original.items():
            yield n,(original.double()+alpha*self.delta[n]).to(dtype)

    def measure(self,alpha,precision):
        per={}
        for n,value in self.values(alpha,precision):
            per[n]=float((value.double()-self.original[n].double()).square().sum())
        return float(np.sqrt(sum(per.values()))),{k:float(np.sqrt(v)) for k,v in per.items()}

    def match(self,variant):
        import torch
        self.activate(variant)
        precision=variant['precision']
        target=variant['target_radius']
        assert target>0
        lo,hi=0.,1.
        trace=[]
        norm,_=self.measure(hi,precision)
        while norm<target:
            trace.append({'scale':hi,'actual_radius':norm})
            hi*=2
            assert hi<=128,'Unexpected extension beyond declared direction range requires review'
            norm,_=self.measure(hi,precision)
        best=(abs(norm-target),hi)
        for _ in range(48):
            alpha=(lo+hi)/2
            norm,_=self.measure(alpha,precision)
            trace.append({'scale':alpha,'actual_radius':norm})
            best=min(best,(abs(norm-target),alpha))
            if abs(norm-target)<=target*.0005:break
            if norm<target:lo=alpha
            else:hi=alpha
        alpha=best[1]
        self.precision(precision)
        named=dict(self.target.named_parameters())
        changed={}
        dot=0.
        squares={}
        with torch.no_grad():
            for n,value in self.values(alpha,precision):
                named[n].copy_(value)
                actual=named[n].double()-self.original[n].double()
                squares[n]=float(actual.square().sum())
                changed[n]=int(torch.count_nonzero(actual))
                dot+=float((actual*self.delta[n]).sum())
                assert torch.equal(named[n],value)
        installed=float(np.sqrt(sum(squares.values())))
        measured,per=self.measure(alpha,precision)
        assert abs(installed-measured)<1e-12
        relative=abs(installed-target)/target
        assert relative<=.001,(variant,installed)
        return {'timestamp':stamp(),'variant':variant,'direction':self.audit,'scale':alpha,
            'target_radius':target,'actual_radius':installed,'relative_radius_error':relative,
            'per_parameter_norm':per,'changed_scalars':sum(changed.values()),'changed_scalars_by_parameter':changed,
            'effective_vs_intended_direction_cosine':dot/(installed*self.audit['exact_FP64_direction_norm']),
            'search_trace':trace,'precision':precision,'all_parameters_used':74711040}

    def restore(self,precision='native_BF16'):
        import torch
        self.precision(precision)
        with torch.no_grad():
            for n,p in self.target.named_parameters():
                p.copy_(self.original[n].to(p.dtype))
                assert torch.equal(p,self.original[n].to(p.dtype))

    def close(self):
        self.restore()
        self.delta={}
