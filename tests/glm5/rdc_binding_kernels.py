"""Exact finite source-pair moments, without coordinate/rank truncation.

These kernels are NOT claimed sufficient states. They bind source identity,
position and predicted coarse dependency-role scores before source aggregation.
"""
from rdc_binding_common import *

ROLE_NAMES=('subject','object','oblique','modifier','predicate','other')
KERNELS=('query','source_mean','source_pair','position_pair','role_position_pair','shuffled_position_pair','shuffled_role_position_pair')

def role_index(word):
    rel=word['relation'].split(':')[0]
    if rel in ('nsubj','csubj'):return 0
    if rel in ('obj','iobj'):return 1
    if rel=='obl':return 2
    if rel in ('amod','advmod','nmod','acl','advcl'):return 3
    if word['upos'] in ('VERB','AUX'):return 4
    return 5

def token_labels(row,n):
    labels=np.full(n,-1,dtype=np.int64)
    for wi,w in enumerate(row['retrospective_ud']):
        span=w.get('char_span')
        if not span:continue
        # Only the final intersecting subtoken represents each UD word in probe scoring.
        match=[i for i,(s,e) in enumerate(row['token_offsets'][:n]) if e>s and s<span[1] and e>span[0]]
        if match:labels[match[-1]]=role_index(w)
    return labels

def source_arrays(rows):
    sources=[];targets={16:[],35:[]};queries=[];embeddings=[];labels=[];lengths=[]
    for row in rows:
        p=row['anchors'][-1]
        if row.get('capture_mode')=='binding':
            path=BASE/'capture/natural'/f'{row["sample_id"]}.npz'
            with np.load(path) as z:
                h=unbits(z['H12_sources'])[:p+1];q=h[-1];emb=unbits(z['embedding'])[-1]
                for b in targets:targets[b].append(np.concatenate([unbits(z[f'L{b}_{k}'])[-1] for k in ('mlp','x','activation')]))
        else:
            with np.load(source_path(row)) as z:h=unbits(z['H12_sources'])[:p+1]
            with np.load(source_path(row,'fields')) as z:
                q=unbits(z['H'])[12,-1];emb=unbits(z['H'])[0,-1]
                for b in targets:targets[b].append(np.concatenate([unbits(z[f'L{b}_{k}'])[-1] for k in ('mlp','x','activation')]))
        rms=np.sqrt(np.mean(h*h,axis=-1,keepdims=True)).clip(1e-8)
        sources.append(h/rms);queries.append(q);embeddings.append(emb)
        labels.append(token_labels(row,len(h)));lengths.append(len(h))
    return sources,np.array(queries),np.array(embeddings),labels,{b:np.array(v) for b,v in targets.items()},np.array(lengths)

def learn_roles(rows,sources,labels,out):
    import torch
    torch.set_num_threads(2)
    d=sources[0].shape[1];xx=torch.zeros((d+1,d+1),device='cuda');xy=torch.zeros((d+1,len(ROLE_NAMES)),device='cuda')
    train=np.array([i for i,r in enumerate(rows) if r['split']=='train'])
    counts=np.zeros(len(ROLE_NAMES),dtype=int)
    for i in train:
        mask=labels[i]>=0
        x=torch.tensor(sources[i][mask],device='cuda');x=torch.cat([x,torch.ones((len(x),1),device='cuda')],1)
        y=torch.nn.functional.one_hot(torch.tensor(labels[i][mask],device='cuda'),len(ROLE_NAMES)).float()
        if not len(x):continue
        xx+=x.T@x/len(x);xy+=x.T@y/len(x);counts+=np.bincount(labels[i][mask],minlength=len(ROLE_NAMES))
    # Frozen ridge rule, no evaluation labels used for model selection.
    ridge=.01*xx.diag().mean();coef=torch.linalg.solve(xx+ridge*torch.eye(d+1,device='cuda'),xy)
    roles=[];confusions={};errors=[]
    for row,h,label in zip(rows,sources,labels):
        x=torch.tensor(h,device='cuda');score=torch.cat([x,torch.ones((len(x),1),device='cuda')],1)@coef
        score=score.clamp_min(0)+1e-6;score=score/score.sum(1,keepdim=True)
        arr=score.cpu().numpy();roles.append(arr)
        valid=label>=0;key=row['split']+'/'+row['cohort']
        mat=confusions.setdefault(key,np.zeros((6,6),dtype=int))
        np.add.at(mat,(label[valid],arr.argmax(1)[valid]),1)
    npz(out/'role_probe.npz',coefficients=coef.cpu().numpy(),training_class_counts=counts,
        **{k.replace('/','_'):v for k,v in confusions.items()})
    save(out/'role_probe.json',{'roles':ROLE_NAMES,'ridge':float(ridge),'training_rows':len(train),
      'training_tokens':int(counts.sum()),'input':'Causal H12 full2560coordinates of that token, RMS normalized, plus intercept.',
      'labels':'Full-sentence UD coarse labels on final subtoken, training only. Some labels depend on future context and may be unpredictable from prefix.',
      'scope':'Nonnegative normalized ridge scores, not calibrated probabilities or a full semantic/role-binding parser.',
      'evaluation':{k:{'tokens':int(v.sum()),'accuracy':float(v.trace()/max(v.sum(),1))} for k,v in confusions.items()}})
    return roles

def apply_roles(sources,coef):
    result=[]
    for h in sources:
        s=np.maximum(h@coef[:-1]+coef[-1],0)+1e-6
        result.append(s/s.sum(1,keepdims=True))
    return result

def feature_pack(sources,q,e,roles):
    import torch
    n=len(sources);t=max(map(len,sources));d=sources[0].shape[1]
    h=np.zeros((n,t,d),np.float32);pos=np.zeros((n,t,3),np.float32);role=np.zeros((n,t,6),np.float32)
    shuffledpos=pos.copy();shuffledrole=role.copy()
    for i,(s,r) in enumerate(zip(sources,roles)):
        k=len(s);h[i,:k]=s;z=np.linspace(-1,0,k,dtype=np.float32)
        pos[i,:k]=np.stack([np.ones(k),z,z*z],1);role[i,:k]=r
        perm=np.random.default_rng(2732+i).permutation(k)
        shuffledpos[i,:k]=pos[i,perm];shuffledrole[i,:k]=r[perm]
    q=q/np.sqrt(np.mean(q*q,axis=1,keepdims=True)).clip(1e-8)
    e=e/np.sqrt(np.mean(e*e,axis=1,keepdims=True)).clip(1e-8)
    arrays={'h':h,'pos':pos,'role':role,'shuffledpos':shuffledpos,'shuffledrole':shuffledrole,
      'q':q,'e':e,'mean':np.array([s.mean(0) for s in sources]),'length':np.array(list(map(len,sources)),np.float32)}
    return {k:torch.tensor(v,device='cuda') for k,v in arrays.items()}

def pair_kernels(left,right=None,block=8):
    import torch
    same=right is None;right=left if same else right
    nl,nr=len(left['h']),len(right['h']);d=left['h'].shape[-1]
    q=(left['q']@right['q'].T+left['e']@right['e'].T)/(2*d)
    mean=left['mean']@right['mean'].T/d
    out={'query':1+q,'source_mean':1+q+mean+q*mean}
    for name in KERNELS[2:]:out[name]=torch.zeros((nl,nr),device='cuda')
    tl,tr=left['h'].shape[1],right['h'].shape[1]
    for i in range(0,nl,block):
      ni=min(block,nl-i)
      for j in range(i if same else 0,nr,block):
        nj=min(block,nr-j)
        def flat(pack,key,begin,num):return pack[key][begin:begin+num].flatten(0,1)
        a,b=flat(left,'h',i,ni),flat(right,'h',j,nj)
        dot=(a@b.T/d).square()
        p=flat(left,'pos',i,ni)@flat(right,'pos',j,nj).T
        sp=flat(left,'shuffledpos',i,ni)@flat(right,'shuffledpos',j,nj).T
        r=1+flat(left,'role',i,ni)@flat(right,'role',j,nj).T
        sr=1+flat(left,'shuffledrole',i,ni)@flat(right,'shuffledrole',j,nj).T
        den=left['length'][i:i+ni,None]*right['length'][None,j:j+nj]
        for name,mul in [('source_pair',None),('position_pair',p),('role_position_pair',p*r),('shuffled_position_pair',sp),('shuffled_role_position_pair',p*sr)]:
            v=dot if mul is None else dot*mul
            s=v.reshape(ni,tl,nj,tr).sum((1,3))/den
            # q and sources are retained jointly; this is a PSD tensor-product kernel.
            z=1+q[i:i+ni,j:j+nj]+s+q[i:i+ni,j:j+nj]*s
            out[name][i:i+ni,j:j+nj]=z
            if same:out[name][j:j+nj,i:i+ni]=z.T
      if i==0 or (i//block+1)%8==0:print('BINDING_KERNEL_BLOCK',i+ni,nl,flush=True)
    return out

