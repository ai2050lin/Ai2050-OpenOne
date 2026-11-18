"""Source-resolved native measurements and exact one-query interventions."""

import numpy as np

import kv_chain_native as V
from kv_chain_run import M as prior_sites


SITES = ('receipt', 'gen', 'note', 'extra', 'prompt', 'other')
NAMES = ('sham','address','value','state-value','joint','restore-receipt','restore-all',
         'matched-address','random-address','matched-value','random-value')


def translated(callback):
    def step(model, layer, index, packet, *tensors):
        if index < V.HEAD[0]:
            return
        rows = packet['rows']
        packet['rows'] = [dict(r, sites={name: (r['sites']['receipt'] if name == 'gen' else
                         r['sites']['gen'] if name == 'note' else []) for name in prior_sites.SITES}) for r in rows]
        try:
            callback(model, layer, index, packet, *tensors)
        finally:
            packet['rows'] = rows
    return step


def read_trace(scan=False):
    scan_step = translated(V.scan_step)

    def trace(model, layer, index, p, x, q, k, v, o, mask):
        t = model.t
        if p['rows'][0].get('capture', False):
            selected = V.family_sources(model,p,x)
            p.setdefault('field',[]).append(selected.view(t.int16).numpy().copy())
        heads = [h for l,h in V.READERS if l == index]
        if not heads:
            return
        j,c,_,qc,_,vis,_ = V.queries(model,p,q,mask)
        h = heads[0]
        groups = t.zeros((len(j),len(SITES)+1,q.shape[2]),dtype=t.bool,device=q.device)
        for i,r in enumerate(p['rows']):
            s,k0 = r['sites'],r['token']
            for col,key in enumerate(SITES[:4]):
                groups[i,col,s[key]] = True
            groups[i,4,:k0['start']] = True
            groups[i,5,:len(k0['ids'])] = ~groups[i,:5,:len(k0['ids'])].any(0)
            groups[i,6,s['insert']] = True
        prob = model.read(qc,k,vis)[:,h]
        p.setdefault('masses',[]).append((prob[:,None]*groups).sum(-1).cpu().numpy())
        if index != V.HEAD[0]:
            return
        g = p['g'].cuda()
        kg = h // model.groups
        query = qc[:,h,0]
        score = (query[:,None].float()*k[:,kg].float()).sum(-1)*query.shape[-1]**-.5
        rec = groups[:,0]
        p['score_pair'] = t.stack((score.masked_fill(~rec,-t.inf).logsumexp(-1),score[j,g],
                                  score.masked_fill(~(rec & ~groups[:,6]),-t.inf).logsumexp(-1)),1).cpu().numpy()
        p['alpha'] = prob.cpu().numpy()
        p['top_site'] = groups[:,:6].to(t.int8)[j,:,prob.argmax(-1)].argmax(1).cpu().numpy()
        p['native_q'] = query.view(t.int16).cpu().numpy().copy()
        p['source_k'] = k[j,kg,g].view(t.int16).cpu().numpy().copy()
        p['source_v'] = v[j,kg,g].view(t.int16).cpu().numpy().copy()
        if scan:
            scan_step(model,layer,index,p,x,q,k,v,o,mask)
    return trace


def readings(model,p):
    from kv_receipt_run import retained

    z = {k:p[k] for k in ('score_pair','alpha','top_site','native_q','source_k','source_v')}
    z.update(ids=np.array([r['id'] for r in p['rows']]),masses=np.stack(p['masses'],1))
    if 'random_scan' in p:
        z['scan'] = p['random_scan']
    if 'field' in p:
        t = model.t
        final = V.family_sources(model,p,p['x']).view(t.int16).numpy()
        z.update(field=np.stack((*p['field'],final),1),field_ids=z['ids'])
    return retained(model,p,model.receipt_selector),z


def controls(rows,z,permitted):
    h = V.HEAD[1]
    use = np.array([r['family'] in permitted and r['n']>0 for r in rows])
    assert use.any(), 'No admitted nonzero-insertion discovery rows'
    result = {}
    for m,mode in enumerate(('address','value')):
        s = z['scan'][use,m]
        ratios = np.median(np.minimum(s,s[:,h,None])/np.maximum(np.maximum(s,s[:,h,None]),1e-8),axis=0)
        candidates = sorted((i for i in range(s.shape[1]) if i//5!=h//5),key=lambda i:(-ratios[i],i))[:6]
        rng = np.random.default_rng(295517+m)
        result[mode] = dict(head=int(rng.choice(candidates)),candidates=candidates,ratio=ratios.tolist())
    return result


def run_edits(model,rows,controls,stage):
    from kv_receipt_run import Chain

    step = translated(V.edits(model,{k:v['head'] for k,v in controls.items()}))
    records = list(Chain.forward(model,rows,stage+'-intervention',step,V.edited_output))
    return {k:np.concatenate([r[k] for r in records]) for k in records[0]}
