"""Evaluate held-out forecasts and a discovery-frozen authorization Value path."""

import numpy as np

import record_update_study as S
import record_update_v2 as V2


R,M,C,K = S.R,S.M,S.C,S.K
BASE_COORDINATES = S.coordinates
BASE_READ = S.h18
BASE_PACK = S.packed_output
TARGET_LAYER = 25


def complete(rows, mask):
    pair_ok = {}
    for row, ok in zip(rows,mask,strict=True):
        pair_ok.setdefault(row['family'],[]).append(ok)
    return np.array([len(pair_ok[row['family']])==2 and all(pair_ok[row['family']]) for row in rows])


def coordinates(rows,z,model,dictionary,j):
    result = BASE_COORDINATES(rows,z,model,dictionary,j)
    result['exact'] = complete(rows,result['exact'])
    return result


def text_features(row,site):
    base = V2.text(row,site)
    token = row['token']
    assert token['cue'] < site
    old,other,earlier,proposal = token['labels']
    emitted = token['ids'][row['g']] if 0<=row['g']<site else -1
    update = row['update']
    rule_choice = proposal if update else old
    chosen_key = row['codes'][3 if update else 0]
    predicted_key = row['codes'][token['labels'].index(emitted)] if emitted in token['labels'] else ''
    features = [update,*row['setting'],token['cue']/M.WIDTH,chosen_key=='Q',emitted>=0,predicted_key=='Q']
    hashed = lambda word: np.eye(32,dtype=np.float32)[word%32] if word>=0 else np.zeros(32,np.float32)
    return np.r_[base,features,*(hashed(w) for w in (old,other,earlier,proposal,rule_choice,emitted))]


def discovery_read(model,layer,i,p,x,q,k,v,o,mask):
    BASE_READ(model,layer,i,p,x,q,k,v,o,mask)
    rows = p['rows']
    if i!=TARGET_LAYER or rows[0]['split']!='discovery':
        return
    t = model.t
    j,pos,_ = S.sites(model,p)
    state = pos[:,0]
    visible = mask[j,:,state].unsqueeze(2)
    query = q[j,:,state].unsqueeze(2)
    cue = t.tensor([r['token']['cue'] for r in rows],device='cuda')
    site = t.arange(k.shape[2],device='cuda')[None,:]==cue[:,None]
    heads = tuple(range(model.heads))
    removed = t.stack([K.raw_change(model,layer,h,query,k,v,visible,site,'value') for h in heads],1)
    projected = model.projections(layer,heads,-removed).squeeze(2)
    label_ids = t.tensor([[r['token']['labels'][idx] for idx in (0,3)] for r in rows],device='cuda')
    directions = model.model.lm_head.weight[label_ids].float()
    delta = directions[:,1]-directions[:,0]
    signs = t.tensor([1 if r['update'] else -1 for r in rows],device='cuda')
    p['auth_norms'] = projected.norm(dim=-1).cpu().numpy()
    p['auth_proxy'] = ((projected*delta[:,None]).sum(-1)*signs[:,None]).cpu().numpy()


def packed_output(model,p):
    result = BASE_PACK(model,p)
    if 'auth_proxy' in p:
        result.update(auth_proxy=p['auth_proxy'],auth_norms=p['auth_norms'])
    return result


def choose_controls(z,rows,model):
    factual = np.asarray([r['expected_top'][0] for r in rows])==z['argmax'][:,0]
    stable = complete(rows,factual)
    scores = z['auth_proxy'][stable].mean(0)
    head = int(scores.argmax())
    norms = z['auth_norms'][stable]
    ratio = np.median(np.minimum(norms,norms[:,head,None])/np.maximum(np.maximum(norms,norms[:,head,None]),1e-6),axis=0)
    pool = sorted((j for j in range(model.heads) if j//model.groups!=head//model.groups),key=lambda h:(-ratio[h],h))[:8]
    control = int(np.random.default_rng(M.SEED).choice(pool))
    return dict(layer=TARGET_LAYER,target=head,head=control,eligible=pool,stable_views=int(stable.sum()),
                signed_discovery_projected_proxy=scores.tolist(),discovery_norm_ratio=ratio.tolist())


def edit_output(model,p):
    t = model.t
    scores = model.logits(p['edited'].cuda())
    ids = t.tensor([[r['token']['labels'][idx] for idx in (0,3)] for r in p['rows']],device='cuda')
    pair = scores.gather(-1,ids[None].expand(scores.shape[0],-1,-1)).permute(1,0,2).float().cpu().numpy()
    top = scores.argmax(-1).T.cpu().numpy()
    restore = K.ARMS.index('restore-all')
    exact = (scores[0]==scores[restore]).all(-1).cpu().numpy()
    return dict(ids=np.array([r['id'] for r in p['rows']]),top=top,pair_logits=pair,dose=p['measures'],full_restore_exact=exact)


def perturb(model,p,rows,z,behavior,frozen,split):
    file = split+'-intervention.npz'
    report = split+'-intervention.json'
    plan = p['design']
    if (R.OUT/report).exists():
        ended = R.read(report)
        assert C.A.digest(R.OUT/file)==ended['effects_sha']
        return
    control = frozen['control']
    K.HEAD = (control['layer'],control['target'])
    prepared = []
    for row in rows:
        token = row['token']
        sites = {site:[] for site in K.C.M.SITES}
        sites.update(gen=[token['cue']],note=[token['proposal']])
        new_token = dict(token,ids=token['prompt'],slots=[token['cue']])
        prepared.append(dict(row,token=new_token,sites=sites))
    hook = K.edits(model,dict(address=control['head'],value=control['head']))
    chunks = list(C.forward(model,prepared,split+'-native-authorization',hook,edit_output))
    effects = {k:np.concatenate([chunk[k] for chunk in chunks]) for k in chunks[0]}
    assert np.array_equal(effects['ids'],z['ids'])
    S.save_np(file,effects,p)
    action = K.ARMS.index
    complete_sham = (effects['top'][:,0]==z['argmax'][:,0]) & (effects['top'][:,0]==np.array([r['expected_top'][0] for r in rows]))
    take = complete(rows,complete_sham)
    sign = np.asarray([1 if r['update'] else -1 for r in rows])
    margins = (effects['pair_logits'][:,:,1]-effects['pair_logits'][:,:,0])*sign[:,None]
    doses = effects['dose']
    relative = abs(doses[:,2,2]-doses[:,3,2])/np.maximum(doses[:,2,1],.02)
    p95 = float(np.quantile(relative[take],.95)) if take.any() else None
    loss = S.paired(margins[:,action('random-value')]-margins[:,action('matched-value')],rows,take,plan['causal_confidence'])
    restore = S.paired(margins[:,action('restore-gen')]-margins[:,action('joint')],rows,take,plan['causal_confidence'])
    dose_ok = p95 is not None and p95<=plan['dose_rel_p95']
    exact = bool(effects['full_restore_exact'][take].all() and take.any())
    passed = dose_ok and exact and all(r['n']>=plan['minimum_power_families'] and r['ci'][0]>0 and r['mean']>=plan[floor]
                                     for r,floor in ((loss,'minimum_matched_margin'),(restore,'minimum_restore_margin')))
    choices = np.asarray([[row['token']['labels'][n] for n in (0,3)] for row in rows])
    top = effects['top']
    chosen = (top[:,None,:]==choices[:,:,None]).any(1)
    changed = (top!=top[:,:1]) & take[:,None]
    R.write(report,dict(at=C.stamp(),provenance=p['provenance'],dose_error_p95=p95,dose_admitted=dose_ok,
             passed=passed,heads=dict(layer=control['layer'],target=control['target'],control=control['head']),
             loss_vs_matched=loss,partial_native_restore=restore,full_native_restore_matches_sham=exact,
             eligible=int(take.sum()),failed_sham=z['ids'][~complete_sham].tolist(),
             real_state_switch_by_arm={a:int((changed[:,i]&chosen[:,i]).sum()) for i,a in enumerate(K.ARMS)},
             full_vocab_change_by_arm={a:int(changed[:,i].sum()) for i,a in enumerate(K.ARMS)},
             counterexamples=effects['ids'][take&(margins[:,action('matched-value')]>margins[:,action('random-value')])][:8].tolist(),
             input_predictions_sha=C.A.digest(R.OUT/(split+'-predictions.json')),effects_sha=C.A.digest(R.OUT/file)))


def install():
    S.coordinates = coordinates
    S.h18 = discovery_read
    S.packed_output = packed_output
    S.choose_controls = choose_controls
    S.text_features = text_features
    S.perturb = perturb
