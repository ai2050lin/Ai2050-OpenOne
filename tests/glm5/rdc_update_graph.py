"""Available-prefix directed relation operators in complete native coordinates.

No gold test edge, future source, target layer, or future output is an input.
The graph readout is a fitted candidate, not the model's native attention.
"""
from collections import Counter
from rdc_update_common import *

KERNELS=('query','mean_raw','signed_position_raw','square_raw','directed_raw',
         'shuffled_heads_raw','directed_query_raw','reversed_query_raw','directed_rms')

def visible_edges(row,position):
    offsets=row['token_offsets'];words={}
    for w in row.get('retrospective_ud',[]):
        span=w.get('char_span')
        if not span:continue
        hit=[i for i,(a,b) in enumerate(offsets[:position+1]) if b>a and a<span[1] and b>span[0]]
        if hit and offsets[hit[-1]][1]>=span[1]:words[(w['sentence_id'],w['id'])]=(hit[-1],w)
    result=[]
    for token,w in words.values():
        head=words.get((w['sentence_id'],w['head']))
        if head and token!=head[0]:result.append((token,head[0],w['relation']))
    return result

def fit_head(rows,out):
    import torch
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    d=2560;xx=torch.zeros((d+1,d+1),dtype=torch.float64,device='cuda');xy=torch.zeros((d+1,d),dtype=torch.float64,device='cuda')
    pairs=0;windows=0;relation_counts=Counter()
    # No validation labels used to choose the head mapping or its ridge value.
    for row in rows:
        if row['split']!='train':continue
        p=row['anchors'][-1];h=sources(row)[:p+1].astype(float)
        h/=np.sqrt(np.mean(h*h,axis=1,keepdims=True)).clip(1e-8)
        edge=visible_edges(row,p)
        if not edge:continue
        dep=np.array([e[0] for e in edge]);head=np.array([e[1] for e in edge]);relation_counts.update(e[2] for e in edge)
        x=torch.tensor(np.c_[h[dep],np.ones(len(dep))],device='cuda');y=torch.tensor(h[head],device='cuda')
        xx+=x.T@x/len(x);xy+=x.T@y/len(x);pairs+=len(x);windows+=1
    lam=.01*xx.diag().mean();coef=torch.linalg.solve(xx+lam*torch.eye(d+1,device='cuda'),xy)
    residual=float(torch.linalg.vector_norm((xx+lam*torch.eye(d+1,device='cuda'))@coef-xy)/torch.linalg.vector_norm(xy))
    assert residual<1e-8
    npz(out/'head_mapping.npz',coefficients=coef.cpu().numpy(),input_second_moment=xx.cpu().numpy(),target_cross_moment=xy.cpu().numpy())
    save(out/'head_mapping.json',{'timestamp':stamp(),'source':snapshot(__file__),'training_windows':windows,'training_visible_edges':pairs,
      'relation_counts':dict(relation_counts),'ridge':float(lam),'solve_relative_residual':residual,
      'rule':'Full2560-coordinate normalized H12 dependent predicts normalized H12 head; linear ridge plus intercept. All currently visible tokens are candidate heads, self excluded.',
      'limitations':'Retrospective UD targets are used only in training/audit. Visible endpoints do not guarantee that full-sentence parse is causally identifiable from prefix. Soft scores are not calibrated semantic probabilities.'})
    del xx,xy;return coef.float()

def pack(rows,coef):
    import torch
    length=np.array([r['anchors'][-1]+1 for r in rows]);n=len(rows);t=int(length.max());d=coef.shape[1]
    allh=np.zeros((n,t,d),np.float32);e=[];q=[];targets={16:[],35:[]};labels=[]
    for i,row in enumerate(rows):
        p=int(length[i]-1);h=sources(row)[:p+1];allh[i,:p+1]=h;q.append(h[-1]);labels.append(visible_edges(row,p))
        with np.load(native_path(row)) as z:
            anchor=row.get('field_anchor',-1)
            emb=unbits(z['embedding'])[anchor] if 'embedding' in z else unbits(z['H'])[0,anchor];e.append(emb)
            for b in targets:targets[b].append(np.concatenate([unbits(z[f'L{b}_{k}'])[anchor] for k in ('mlp','x','activation')]))
    h=torch.tensor(allh,device='cuda');rms=h.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8);unit=h/rms
    a=torch.zeros((n,t,t),device='cuda');shuffled=a.clone();reversed_a=a.clone();pos=torch.zeros((n,t,3),device='cuda');audit=[]
    for i,row in enumerate(rows):
        k=int(length[i]);pred=unit[i,:k]@coef[:-1]+coef[-1]
        pred=pred/pred.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
        score=pred@unit[i,:k].T/d
        # Fixed temperature, not tuned on test attachment accuracy or state outcomes.
        score=score/.15;score.fill_diagonal_(-torch.inf);aa=score.softmax(-1)
        a[i,:k,:k]=aa;perm=np.random.default_rng(int(ranked('heads/'+row['sample_id'])[:8],16)).permutation(k)
        shuffled[i,:k,:k]=aa[:,perm];reversed_a[i,:k,:k]=aa.T
        z=torch.linspace(-1,0,k,device='cuda');pos[i,:k]=torch.stack([torch.ones_like(z),z,z*z],-1)
        edge=labels[i]
        if edge:
            di=torch.tensor([x[0] for x in edge],device='cuda');he=torch.tensor([x[1] for x in edge],device='cuda')
            distance=(di[:,None]-torch.arange(k,device='cuda')[None,:]).abs().float();distance[torch.arange(len(di),device='cuda'),di]=torch.inf
            audit.append({'sample_id':row['sample_id'],'split':row['split'],'cohort':row['cohort'],'source_group':row['source_group'],'edges':len(edge),
              'correct':int((aa[di].argmax(-1)==he).sum()),'nearest_position_correct':int((distance.argmin(-1)==he).sum()),
              'mean_gold_probability':float(aa[di,he].mean()),'gold_edges':[[s,t,r] for s,t,r in edge]})
    q=torch.tensor(np.array(q),device='cuda');q=q/q.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
    e=torch.tensor(np.array(e),device='cuda');e=e/e.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
    lens=torch.tensor(length,device='cuda',dtype=torch.float32)
    def message(aa):
        read=(h@q[:,:,None])/d
        return (h.transpose(1,2)@(aa@read)).squeeze(-1)/lens[:,None]
    return {'h':h,'unit':unit,'a':a,'shuffled':shuffled,'pos':pos,'q':q,'e':e,'length':lens,
      'mean':h.sum(1)/lens[:,None],'signed':(h.transpose(1,2)@pos/lens[:,None,None]).flatten(1),
      'message':message(a),'reverse_message':message(reversed_a)}, {b:np.array(v) for b,v in targets.items()},audit

def kernels(left,right=None,block=8,verbose=True):
    import torch
    same=right is None;right=left if same else right;nl,nr=len(left['h']),len(right['h']);d=left['h'].shape[-1]
    q=(left['q']@right['q'].T+left['e']@right['e'].T)/(2*d)
    out={'query':1+q}
    for name,key in [('mean_raw','mean'),('signed_position_raw','signed'),('directed_query_raw','message'),('reversed_query_raw','reverse_message')]:
        s=left[key]@right[key].T/d;out[name]=1+q+s+q*s
    for name in ('square_raw','directed_raw','shuffled_heads_raw','directed_rms'):out[name]=torch.zeros((nl,nr),device=q.device)
    for i in range(0,nl,block):
      ni=min(block,nl-i)
      for j in range(i if same else 0,nr,block):
        nj=min(block,nr-j);den=left['length'][i:i+ni,None]*right['length'][None,j:j+nj]
        for key in ('h','unit'):
            hl=left[key][i:i+ni];hr=right[key][j:j+nj]
            dot=torch.einsum('atd,bsd->abts',hl,hr)/d
            names=[('directed_rms','a')] if key=='unit' else [('directed_raw','a'),('shuffled_heads_raw','shuffled')]
            if key=='h':
                s=dot.square().sum((-1,-2))/den;z=1+q[i:i+ni,j:j+nj]+s+q[i:i+ni,j:j+nj]*s
                out['square_raw'][i:i+ni,j:j+nj]=z
                if same:out['square_raw'][j:j+nj,i:i+ni]=z.T
            for name,akey in names:
                al=left[akey][i:i+ni,None];ar=right[akey][None,j:j+nj]
                term=al.transpose(-1,-2)@dot@ar
                s=(term*dot).sum((-1,-2))/den
                z=1+q[i:i+ni,j:j+nj]+s+q[i:i+ni,j:j+nj]*s
                out[name][i:i+ni,j:j+nj]=z
                if same:out[name][j:j+nj,i:i+ni]=z.T
      if verbose and (i==0 or (i//block+1)%8==0):print('DIRECTED_KERNEL',i+ni,nl,flush=True)
    return out

def save_pack(out,rows,pack,audit):
    npz(out/'features.npz',**{k:v.cpu().numpy() for k,v in pack.items() if k not in ('h','unit')})
    compressed(out/'row_identity.json.gz',[{'sample_id':r['sample_id'],'position':r['anchors'][-1],'source_archive':str(native_path(r))} for r in rows])
    compressed(out/'head_evaluation.json.gz',audit)
    summary=[]
    for split,cohort in sorted({(r['split'],r['cohort']) for r in audit}):
        rr=[r for r in audit if (r['split'],r['cohort'])==(split,cohort)];den=sum(r['edges'] for r in rr)
        summary.append({'split':split,'cohort':cohort,'windows':len(rr),'edges':den,'accuracy':sum(r['correct'] for r in rr)/den,
          'nearest_position_accuracy':sum(r['nearest_position_correct'] for r in rr)/den,
          'source_cluster_accuracy':clustered([r['correct']/r['edges'] for r in rr],[r['source_group'] for r in rr])})
    save(out/'head_evaluation_summary.json',summary)
