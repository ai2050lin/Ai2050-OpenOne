"""Complete native-parameter factors for restricted last-MLP language training.

Factorization below is exact for one pointwise MLP example. It does not truncate
rank, coordinates, vocabulary rows or units. Dense parameters remain trainable.
"""
from rdc_law_common import *


def parameter(name, device='cuda:0'):
    from safetensors import safe_open
    path=ROOT/'models/hf/qwen3-4b'
    index=read(path/'model.safetensors.index.json')['weight_map']
    with safe_open(str(path/index[name]),framework='pt',device='cpu',backend='pread') as stream:
        tensor=stream.get_tensor(name)
    return tensor.float().to(device)


def tail_forward(x,residual,w,norm,head,eps,targets=None,factors=False):
    import torch
    import torch.nn.functional as F
    g=F.linear(x,w['g']);u=F.linear(x,w['u']);phi=F.silu(g);activation=phi*u
    m=F.linear(activation,w['d']);r=residual+m
    denominator=(r.square().mean(-1,keepdim=True)+eps).sqrt()
    n=norm*r/denominator
    logits=F.linear(n,head);lp=logits.log_softmax(-1);p=lp.exp()
    out={'g':g,'u':u,'phi':phi,'activation':activation,'m':m,'r':r,'postnorm':n,'logits':logits,
         'entropy':-(p*lp).sum(-1),'argmax':logits.argmax(-1),'probabilities':p,'logprobs':lp}
    if targets is not None:
        out['loss']=-lp[torch.arange(len(x),device=x.device),targets]
    if factors:
        assert targets is not None
        # Per-example (not batch-mean) loss gradient, all vocabulary terms included.
        error=p.clone()
        error[torch.arange(len(x),device=x.device),targets]-=1
        # Center BEFORE the matrix product: avoids subtracting nearly equal vocabulary
        # means for confident correct predictions. Same real derivative, better FP32 conditioning.
        dn=error@head
        dn=dn*norm
        s=dn/denominator-r*(dn*r).mean(-1,keepdim=True)/denominator.pow(3)
        unit=s@w['d']
        sigmoid=g.sigmoid()
        phi_prime=sigmoid+g*sigmoid*(1-sigmoid)
        out['factors']={'x':x,'a':activation,'s':s,'bg':unit*u*phi_prime,'bu':unit*phi}
    return out


def factor_gram(a,b=None):
    """Full parameter inner product, exact outer-product factors; no top-rank selection."""
    b=a if b is None else b
    x=a['x']@b['x'].T
    gate=(a['bg']@b['bg'].T)*x
    up=(a['bu']@b['bu'].T)*x
    down=(a['s']@b['s'].T)*(a['a']@b['a'].T)
    return {'g':gate,'u':up,'d':down,'total':gate+up+down}


def dense_gradient(f):
    n=len(f['x'])
    return {'g':f['bg'].T@f['x']/n,'u':f['bu'].T@f['x']/n,'d':f['s'].T@f['a']/n}


def parameter_norm(w):
    import torch
    return torch.stack([a.square().sum() for a in w.values()]).sum().sqrt()


def native_bf16_forward(x,residual,w,norm,head,eps):
    import torch
    import torch.nn.functional as F
    x=x.to(torch.bfloat16);residual=residual.to(torch.bfloat16)
    g=F.linear(x,w['g'].to(torch.bfloat16));u=F.linear(x,w['u'].to(torch.bfloat16))
    act=F.silu(g)*u
    m=F.linear(act,w['d'].to(torch.bfloat16));r=residual+m
    # Follow inspected Qwen3RMSNorm: FP32 variance, cast normalized value before weight product.
    rr=r.float();nr=(rr*torch.rsqrt(rr.square().mean(-1,keepdim=True)+eps)).to(torch.bfloat16)
    n=norm.to(torch.bfloat16)*nr
    logits=F.linear(n,head.to(torch.bfloat16)).float()
    return {'m':m,'r':r,'postnorm':n,'logits':logits}


def readout_actions(r,directions,norm,head,eps,probabilities):
    """All-coordinate J_RMS and full-vocabulary Fisher direction actions, not eigenvectors."""
    denominator=(r.square().mean(-1,keepdim=True)+eps).sqrt()
    dn=norm*(directions/denominator-r*(r*directions).mean(-1,keepdim=True)/denominator.pow(3))
    dz=dn@head.T
    center=(probabilities*dz).sum(-1,keepdim=True)
    variance=(probabilities*(dz-center).square()).sum(-1)
    return dn,variance


class Tail:
    def __init__(self,device='cuda:0'):
        import torch
        torch.set_num_threads(2)
        if str(device).startswith('cuda'):
            torch.backends.cuda.matmul.allow_tf32=False
            torch.backends.cudnn.allow_tf32=False
        self.w={k:parameter(f'model.layers.35.mlp.{name}_proj.weight',device) for k,name in [('g','gate'),('u','up'),('d','down')]}
        self.norm=parameter('model.norm.weight',device)
        config=read(ROOT/'models/hf/qwen3-4b/config.json')
        assert config['tie_word_embeddings']
        self.head=parameter('model.embed_tokens.weight',device)
        self.eps=config['rms_norm_eps'];self.device=device
        # Register the actual library MLP class on the very same native parameter tensors.
        # This is not training a fitted surrogate: these are all three original parameter matrices.
        from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
        from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
        with torch.device('meta'):
            self.module=Qwen3MLP(Qwen3Config(**config))
        for k,name in [('g','gate'),('u','up'),('d','down')]:
            getattr(self.module,name+'_proj').weight=torch.nn.Parameter(self.w[k],requires_grad=False)
            self.w[k]=getattr(self.module,name+'_proj').weight

    def forward(self,x,residual,targets=None,factors=False):
        return tail_forward(x,residual,self.w,self.norm,self.head,self.eps,targets,factors)

    def reset(self):
        import torch
        with torch.no_grad():
            for k,name in [('g','gate'),('u','up'),('d','down')]:
                self.w[k].copy_(parameter(f'model.layers.35.mlp.{name}_proj.weight',self.device))

    def save_delta(self,path):
        arrays={}
        for k,name in [('g','gate'),('u','up'),('d','down')]:
            initial=parameter(f'model.layers.35.mlp.{name}_proj.weight','cpu').numpy()
            arrays[k]=self.w[k].detach().cpu().numpy()-initial
        npz(path,**arrays)
        return {k:identity(a) for k,a in arrays.items()}
