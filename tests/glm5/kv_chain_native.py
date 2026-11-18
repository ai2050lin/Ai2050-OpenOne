"""Native read measurements and single-query 14B blocking/recovery."""

import numpy as np

import kv_chain_run as C


HEAD=(25,18)
READERS=((20,5),(21,12),(24,18),HEAD)
ARMS=('sham','address','value','note','joint','restore-gen','restore-all',
      'matched-address','random-address','matched-value','random-value')


def sites(model,p):
    t=model.t
    batch=t.arange(len(p['rows']),device='cuda'); c=p['c'].cuda()
    locations=t.zeros((len(batch),len(C.M.SITES),C.WIDTH),dtype=t.bool,device='cuda')
    for j,row in enumerate(p['rows']):
        for k,site in enumerate(C.M.SITES):
            locations[j,k,row['sites'][site]]=True
    return batch,c,locations


def queries(model,p,q,mask):
    t=model.t; j=t.arange(len(p['rows']),device='cuda'); c=p['c'].cuda()
    e=p['g'].cuda()-1
    return j,c,e,q[j,:,c,None],q[j,:,e,None],mask[j,:,c,None],mask[j,:,e,None]


def family_sources(model,p,x):
    t=model.t; g=p['g']; c=p['c']; e=g-1
    return x[t.arange(len(g)),t.stack((e,g,c),1).T].transpose(0,1).cpu()


def trace(model,layer,index,p,x,q,k,v,o,mask):
    t=model.t
    # Retain all layer states for one preregistered audit view per panel, at E, G, C.
    if any(r.get('capture',False) for r in p['rows']):
        source=family_sources(model,p,x)
        pick=[j for j,r in enumerate(p['rows']) if r.get('capture',False)]
        p.setdefault('field',[]).append((tuple(pick),source[pick].view(t.int16).numpy().copy()))
    heads=[h for l,h in READERS if l==index]
    if not heads:return
    j,c,e,q_c,q_e,vis_c,vis_e=queries(model,p,q,mask)
    _,_,location=sites(model,p)
    for h in heads:
        cprob=model.read(q_c,k,vis_c)[:,h]
        eprob=model.read(q_e,k,vis_e)[:,h]
        cm=(cprob[:,None]*location).sum(-1)
        em=(eprob[:,None]*location).sum(-1)
        p.setdefault('masses',[]).append(t.stack((em,cm),1).cpu().numpy())
    if index!=HEAD[0]:return
    h=HEAD[1]; group=h//model.groups; g=p['g'].cuda()
    gg=(q_c[:,h,0].float()*k[j,group,g].float()).sum(-1)*128**-.5
    source=location[:,2]
    a=(cprob*source).sum(-1)
    u=(cprob[...,None]*v[:,group].float()).masked_fill(source[...,None],0).sum(1)/(1-a).clamp_min(1e-6)[:,None]
    vg=v[j,group,g].float()
    effect=t.stack((a[:,None]*(u-vg),-a[:,None]*vg),1)
    projection=model.projections(layer,(h,),effect.unsqueeze(2)).squeeze(2)
    lm=model.model.lm_head.weight[model.letters].float()
    # A simple direct logit-lens proxy, with observed clean query geometry.
    p['direct']=(projection@(lm[1]-lm[0])).cpu().numpy()
    p['dose']=projection.norm(dim=-1).cpu().numpy()
    p['score']=gg.cpu().numpy(); p['most']=cprob.argmax(-1).cpu().numpy()
    p['source_k']=k[j,group,g].view(t.int16).cpu().numpy().copy()
    p['source_v']=v[j,group,g].view(t.int16).cpu().numpy().copy()
    # Store only a few full vectors for source-level auditing, with all positional mass preserved per view.
    p['full_alpha']=(cprob[:,None,:]*location).sum(1).sum(-1).cpu().numpy()


def readouts(model,p):
    t=model.t;j=t.arange(len(p['rows'])); x=p['x'][j,p['c']].cuda(); y=model.logits(x)
    z=dict(ids=np.array([r['id'] for r in p['rows']]), top=y.argmax(-1).cpu().numpy(), pq=y[:,model.letters].float().cpu().numpy(),
           masses=np.stack(p['masses'],1),direct=p['direct'],dose=p['dose'],score=p['score'], most=p['most'],
           source_k=p['source_k'],source_v=p['source_v'],full_alpha=p['full_alpha'])
    if 'field' in p:
        indices,fields=zip(*p['field'],strict=True)
        assert len(indices)==40 and all(i==indices[0] for i in indices)
        last=p['x'][list(indices[0])]
        rows=list(indices[0]); bounds=t.stack((p['g']-1,p['g'],p['c']),1)[rows]
        final=last[t.arange(len(last))[:,None],bounds].view(t.int16).numpy()
        z.update(field=np.stack((*fields,final),1), field_ids=z['ids'][list(indices[0])])
    return z


def raw_change(model,layer,h,q,k,v,vis,site,mode):
    base=model.foreground(layer,h,q,k,v,vis).float()
    if mode=='address':
        use=vis & ~site[:,None,None,:]
        y=model.foreground(layer,h,q,k,v,use)
    else:
        edited=v.clone(); edited[:,h//model.groups].masked_fill_(site[...,None],0)
        y=model.foreground(layer,h,q,k,edited,vis)
    return y.float()-base


def choose_random(model,layer,p,x,q,k,v,o,mask):
    if int(p.get('layer_index',-1))!=HEAD[0]:return
    _,_,q0,_,vis,_=shortcut(model,p,q,mask)
    _,_,ss=sites(model,p)
    hs=tuple(range(model.heads)); t=model.t
    norms=[]
    for mode in ('address','value'):
        diff=t.stack([raw_change(model,layer,h,q0,k,v,vis,ss[:,2],mode) for h in hs],1)
        norms.append(model.projections(layer,hs,diff).norm(dim=-1).cpu().numpy())
    p['random_scan']=np.stack(norms,1)


def shortcut(model,p,q,mask):
    j,c,_,qc,qe,visible,ve=queries(model,p,q,mask)
    return j,c,qc,qe,visible,ve


def scan_step(model,layer,i,p,x,q,k,v,o,mask):
    if i==HEAD[0]:
        p['layer_index']=i
        choose_random(model,layer,p,x,q,k,v,o,mask)


def scan_output(model,p):
    return p['random_scan']


def edits(model,controls):
    def step(m,layer,i,p,x,q0,k0,v0,o0,mask):
        if i<HEAD[0]:return
        t=m.t; j,c,q,_,vis,_=shortcut(m,p,q0,mask); b=len(j); a=len(ARMS)
        if i==HEAD[0]:
            live=x[j,c].expand(a,-1,-1).clone()
        else:
            live=p['edited'].cuda()
        angle=[ang.cuda()[j,c,None].repeat(a,1,1) for ang in p['angles']]
        qq,kk,vv=m.qkv(layer,live.reshape(a*b,1,m.width),angle)
        k=k0.repeat(a,1,1,1); v=v0.repeat(a,1,1,1); idx=t.arange(a*b,device='cuda'); cc=c.repeat(a)
        k[idx,:,cc]=kk[:,:,0]; v[idx,:,cc]=vv[:,:,0]
        z=m.attention(layer,qq,k,v,vis.repeat(a,1,1,1)).reshape(a,b,m.heads,128)
        if i==HEAD[0]:
            _,_,location=sites(m,p); gen,note=location[:,2],location[:,3]; h=HEAD[1]
            # All arms begin from the same original C state. Base Q/K/V is shared at this layer.
            actual_k=k[:b]; actual_v=v[:b]; actual_q=qq[:b]
            before=z[0].clone(); scale={}; measures=np.zeros((b,4,3),dtype=np.float32)
            for name,site,mode in [('address',gen,'address'),('value',gen,'value'),('note',note,'value'),('joint',gen|note,'value')]:
                diff=raw_change(m,layer,h,actual_q,actual_k,actual_v,vis,site,mode)
                scale[name]=diff
                z[ARMS.index(name),:,h]=(before[:,h].float()+diff).to(z.dtype)
            # V-only interventions admit exact original vectors. Restoring one of the two removed source Values
            # leaves the other source removed; restore-all is the native self-recovery control.
            for name,blocked in [('restore-gen',note),('restore-all',t.zeros_like(note))]:
                restored=actual_v.clone(); restored[:,h//m.groups].masked_fill_(blocked[...,None],0)
                z[ARMS.index(name),:,h]=m.foreground(layer,h,actual_q,actual_k,restored,vis)
            assert t.equal(z[ARMS.index('restore-all')],before)
            for mindex,mode in enumerate(('address','value')):
                peer=int(controls[mode]); diffs=[scale[mode],raw_change(m,layer,peer,actual_q,actual_k,actual_v,vis,gen,mode)]
                dose=[m.projections(layer,(head,),d[:,None]).norm(dim=-1)[:,0] for head,d in zip((h,peer),diffs,strict=True)]
                goal=t.minimum(*dose)
                for n,(name,head,d,norm) in enumerate(zip((f'matched-{mode}',f'random-{mode}'),(h,peer),diffs,dose,strict=True)):
                    target=before[:,head].float()+d*(goal/norm.clamp_min(1e-8))[:,None]
                    edited=target.to(z.dtype)
                    z[ARMS.index(name),:,head]=edited
                    measured=m.projections(layer,(head,),(edited.float()-before[:,head].float())[:,None]).norm(dim=-1)[:,0]
                    measures[:,mindex*2+n]=t.stack((norm,goal,measured),1).cpu().numpy()
            p['measures']=measures
        p['edited']=m.residual(layer,live,z).cpu()
    return step


def edited_output(m,p):
    scores=m.logits(p['edited'].cuda())
    top=scores.argmax(-1).cpu().numpy().T; pq=scores[:,:,m.letters].float().cpu().numpy().transpose(1,0,2)
    assert np.array_equal(top[:,0],top[:,ARMS.index('restore-all')]) and np.array_equal(pq[:,0],pq[:,ARMS.index('restore-all')])
    return dict(ids=np.array([r['id'] for r in p['rows']]),top=top,pq=pq,dose=p['measures'])
