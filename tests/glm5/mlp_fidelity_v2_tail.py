"""Actual B35→final RMSNorm→unembedding path. No shortcut to earlier layers."""
import numpy as np

def capability_interval(base,edited,eligible,margin=.02,alpha=.025):
    from scipy.stats import beta
    b=np.asarray(base,bool)[eligible];e=np.asarray(edited,bool)[eligible];n=len(b)
    if not n:return {'n':0,'status':'not_tested'}
    loss=int((b&~e).sum());gain=int((~b&e).sum())
    def bounds(k):return (float(beta.ppf(alpha/4,k,n-k+1)) if k else 0.,float(beta.ppf(1-alpha/4,k+1,n-k)) if k<n else 1.)
    ll,ul=bounds(loss);lg,ug=bounds(gain);lo,hi=ll-ug,ul-lg
    return {'n':n,'base_accuracy':float(b.mean()),'edited_accuracy':float(e.mean()),'losses':loss,'gains':gain,'net_loss':float(b.mean()-e.mean()),'loss_interval':[lo,hi],'margin':margin,'alpha':alpha,'status':'noninferior' if hi<=margin else 'inferior' if lo>margin else 'inconclusive','method':'simultaneous exact Clopper-Pearson loss/gain bounds; two operation family correction'}

def cluster_interval(values,seed=1):
    x=np.asarray(values,float);rng=np.random.default_rng(seed);means=x[rng.integers(len(x),size=(5000,len(x)))].mean(1)
    return {'n_clusters':len(x),'mean':float(x.mean()),'ci95':np.quantile(means,[.025,.975]).tolist()}

def operations(dictionary,y,a,down,selected=None,random=None,tc=None,x=None):
    import torch
    from shared.dist_protocol.transcoder import neuron_topk_recon
    z=dictionary.encode(y.float());out={'clean':y,'sae_full':dictionary.decode(z),'zero':torch.zeros_like(y),'native32':neuron_topk_recon(a.float(),down.float().T,32),'native_matched_k':neuron_topk_recon(a.float(),down.float().T,dictionary.k)}
    if tc is not None:out['tc4096_diagnostic']=tc(x.float())
    if selected:
        change=(z[:,selected]@dictionary.W_dec[selected])*dictionary.y_scale
        out['selected_0.5']=y.float()-.5*change;out['selected_1']=y.float()-change
    if random:out['random_1']=y.float()-(z[:,random]@dictionary.W_dec[random])*dictionary.y_scale
    return out,z

def tail_logits(model,residual,y):
    return model.lm_head(model.model.norm((residual+y.to(residual.dtype))[:,None]))[:,0]

def metric(logits,clean_logits,target):
    import torch
    lp=logits.float().log_softmax(-1);clp=clean_logits.float().log_softmax(-1)
    return {'CE':(-lp[torch.arange(len(lp),device=lp.device),target]).cpu().numpy(),'KL_clean':(clp.exp()*(clp-lp)).sum(-1).cpu().numpy(),'argmax':lp.argmax(-1).cpu().numpy()}
