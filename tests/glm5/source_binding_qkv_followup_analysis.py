"""Recompute the cue transplant, readout replay and task-gap comparisons."""
from collections import defaultdict
import itertools
from pathlib import Path

import numpy as np

import source_binding_qkv_followup as exp
import source_binding_transfer_factors as io
from source_binding_design import sha

ARMS = ('W','H','K','V','QH','D','Dself','ALL','Hpos','Hback')
CELLS = [f'{s}@{b}' for s,b in itertools.product(exp.SITES,exp.LAYERS)]


def load(split):
    design = io.read('design.json')
    manifest = io.read(f'{split}-manifest.json')
    assert manifest['passed'] and sha(exp.OUT/'design.json')==manifest['design_sha256']
    source = {}
    for task in exp.TASKS:
        name = f'{split}-{task}-materials.json'
        assert sha(exp.OUT/name)==design['files'][name]
        source.update({r['id']:r for r in io.read(name)})
    rows = defaultdict(dict)
    audit = defaultdict(list)
    for shard in manifest['shards']:
        name = shard['file']
        assert sha(exp.OUT/name)==shard['sha256']
        payload = io.read(name)
        audit[name.split('-')[1]].append({**payload['controls'],**payload['meta']})
        for group in payload['groups']:
            for i,id_ in enumerate(group['ids']):
                r = source.pop(id_)
                rows[r['task']][r['original_id'],r['cue']] = (r,{k:v[i] for k,v in group['items'].items()},
                    {k:v[i] for k,v in group['diagnostics'].items()})
    assert not source
    return rows,dict(audit)


def arrays(rows):
    family_keys = sorted({r['family'] for r,_,_ in rows.values()})
    families = {f:i for i,f in enumerate(family_keys)}
    ids = sorted({id_ for id_,_ in rows})
    assert len(rows)==2*len(ids)
    meta = [rows[id_,0][0] for id_ in ids]
    fam = np.array([families[r['family']] for r in meta])
    labels = ['base','Q','noop'] + [f'{a}|{c}' for a,c in itertools.product(ARMS,CELLS)]
    scores = np.array([[[rows[id_,cue][1][k] for k in labels] for id_ in ids] for cue in (0,1)])
    arm = {k:scores[:,:,i] for i,k in enumerate(labels)}
    trace_keys = list(next(iter(rows.values()))[2])
    trace = {k:np.array([[rows[id_,cue][2][k] for id_ in ids] for cue in (0,1)]) for k in trace_keys}
    assert all(rows[id_,0][0]['donor']==rows[id_,1][0]['donor'] for id_ in ids)
    return arm,trace,meta,fam,len(families)


def margins(scores, original, target):
    target_scores = np.take_along_axis(scores[...,1:],target[...,None].astype(int),axis=-1)[...,0]
    old_scores = np.take_along_axis(scores[...,1:],original[...,None].astype(int),axis=-1)[...,0]
    return target_scores-old_scores


def summarize(task, rows):
    from source_binding_fidelity2 import predict

    arm,trace,meta,family,nf = arrays(rows)
    rng = np.random.default_rng(exp.SEED)
    draws = rng.multinomial(nf,np.full(nf,1/nf),size=6000)
    base = arm['base'][:,:,0]
    truth = np.array([r['target_record'] for r in meta])
    assert all(np.mean(b==truth)>=.95 for b in base)
    wide = np.stack([arm['W|'+c] for c in CELLS],axis=2)
    original = np.broadcast_to(base[:,:,None],wide.shape[:-1])
    target = wide[...,0]
    changed = original!=target
    common = changed.all(0) & (target[0]==target[1]) & (original[0]==original[1])
    usable = (base==truth).all(0)
    common &= usable[:,None]
    data = {k:np.stack([arm[k+'|'+c] for c in CELLS],axis=2) for k in ARMS}
    data['Q'] = np.repeat(arm['Q'][:,:,None,:],len(CELLS),axis=2)
    data['base'] = np.repeat(arm['base'][:,:,None,:],len(CELLS),axis=2)
    hit = {k:v[...,0]==target for k,v in data.items()}
    m = {k:margins(v,original,target) for k,v in data.items()}
    edit_gain = {k:v-m['base'] for k,v in m.items()}

    def pack(value, keep):
        assert value.shape==keep.shape and value.shape[0]==len(family)
        total = np.bincount(family,weights=keep.reshape(len(family),-1).sum(1),minlength=nf)
        count = np.bincount(family,weights=np.where(keep,value,0).reshape(len(family),-1).sum(1),minlength=nf)
        ns = draws@total
        rates = (draws@count)[ns>0]/ns[ns>0]
        return dict(n=int(total.sum()),mean=float(count.sum()/total.sum()) if total.sum() else None,
                    ci95=np.quantile(rates,[.025,.975]).tolist() if len(rates) else None)

    changed_stats = {}
    for cue in (0,1):
        changed_stats[str(cue)] = {k:pack(h[cue],changed[cue]) for k,h in hit.items()}
        changed_stats[str(cue)]['change_rate'] = pack(changed[cue],np.ones_like(changed[cue]))
    comparisons = {}
    for name in ('H','K','V','Q','QH','D','ALL','Hpos','Hback'):
        comparisons[name] = {}
        for cue in (0,1):
            comparisons[name][str(cue)] = dict(hit=pack(hit[name][cue],common),margin=pack(m[name][cue],common),
                                                edit_gain=pack(edit_gain[name][cue],common),
                                                versus_H_hit=pack(hit[name][cue].astype(int)-hit['H'][cue],common),
                                                versus_H_margin=pack(m[name][cue]-m['H'][cue],common))
    answer_recovery = {k:dict(hit_delta=pack(hit[k][0].astype(int)-hit['H'][0],changed[0]),
                              margin_delta=pack(m[k][0]-m['H'][0],changed[0])) for k in ('K','V','ALL','Hpos','Hback')}
    cue_effects = {}
    for name,values in edit_gain.items():
        cue_effects[name] = dict(margin=pack(m[name][1]-m[name][0],common),edit_gain=pack(values[1]-values[0],common),
                                 hit=pack(hit[name][1].astype(int)-hit[name][0],common))
    cue_effects['downstream_fixed_packet'] = {str(d):dict(
        margin=pack((edit_gain['H'][1]-edit_gain['D'][0]) if d==1 else (edit_gain['D'][1]-edit_gain['H'][0]),common),
        hit=pack((hit['H'][1].astype(int)-hit['D'][0]) if d==1 else (hit['D'][1].astype(int)-hit['H'][0]),common)) for d in (0,1)}
    ds = np.array([r['relation'][0] for r in meta])[:,None]
    de = np.array([r['relation'][1] for r in meta])[:,None]
    source = np.array([r['source'] for r in meta])[:,None]
    entity = np.array([r['entity'] for r in meta])[:,None]
    wanted = 2*(source ^ (ds*np.array([not c.startswith('qent') for c in CELLS]))) + (entity ^ (de*np.array([not c.startswith('qsrc') for c in CELLS])))
    target_is_donor = target[0]==wanted
    lookup = {r['original_id']:r for r in meta}
    positional = np.array([[predict(r,lookup[r['donor']],s,b,'position') for s,b in itertools.product(exp.SITES,exp.LAYERS)] for r in meta])
    target_is_position = target[0]==positional
    where = {}
    for name,keys in {'site':exp.SITES,'start':exp.LAYERS,'layout':range(4),'target_kind':('requested_donor','other_record'),
                      'position_target':('matches_position','other_record')}.items():
        where[name] = {}
        for key in keys:
            if name=='site':
                filt = np.broadcast_to([c.startswith(key+'@') for c in CELLS],changed[0].shape)
            elif name=='start':
                filt = np.broadcast_to([c.endswith('@'+str(key)) for c in CELLS],changed[0].shape)
            elif name=='layout':
                filt = np.broadcast_to(np.array([r['frame']==key for r in meta])[:,None],changed[0].shape)
            elif name=='target_kind':
                filt = target_is_donor == (key=='requested_donor')
            else:
                filt = target_is_position == (key=='matches_position')
            where[name][str(key)] = {k:pack(hit[k][0],changed[0]&filt) for k in ('H','K','V','ALL','Hpos','Hback')}
            where[name][str(key)]['wide_change_rate'] = pack(changed[0],filt)
    look = {}
    for name in ('noop','Q','H','K','V','QH','Hpos'):
        gathered = []
        for block in range(23,28):
            if name in ('noop','Q'):
                keys = [f'{block}|{name}']
            else:
                keys = [f'{block}|{name}|{c}' for c in CELLS if f'{block}|{name}|{c}' in trace]
            v = np.stack([trace[k] for k in keys],axis=2)
            gathered.append(v.mean((1,2)).tolist())
        look[name] = gathered
    attention_tests = {}
    for name in ('K','V','QH','Hpos'):
        for cue in (0,1):
            layers = []
            for block in range(23,28):
                pairs = [(trace[f'{block}|{name}|{cell}'][cue],trace[f'{block}|H|{cell}'][cue],common[:,j])
                         for j,cell in enumerate(CELLS) if f'{block}|{name}|{cell}' in trace]
                mask = np.stack([x[2] for x in pairs],axis=1)
                delta = np.stack([a-b for a,b,_ in pairs],axis=1)
                layers.append([pack(delta[...,i],mask) for i in range(3)])
            attention_tests[f'{name}/{cue}'] = layers
    q_only_n = {str(c):pack((data['Q'][c,:,0,0]!=base[c])[:,None],np.ones_like(base[c,:,None],bool)) for c in (0,1)}
    strata = {}
    for col,cell in enumerate(CELLS):
        for target_kind,mask in (('donor',target_is_donor),('other',~target_is_donor)):
            counts = []
            chosen = changed[0][:,col] & mask[:,col]
            for f in range(nf):
                subset = chosen & (family==f)
                counts.append([int(subset.sum()),int((subset & hit['H'][0][:,col]).sum())])
            strata[f'{cell}/{target_kind}'] = counts
    return dict(families=nf,prompts=len(meta),wide_edits=int(changed.size),common_n=int(common.sum()),changed=changed_stats,
                common=comparisons,cue_effects=cue_effects,answer_only_breakdown=where,attention_mean_by_block=look,
                attention_columns=['name_mass','argmax_on_name','head_output_norm'],Q_only_answer_change=q_only_n,
                common_attention_change_from_H=attention_tests,answer_only_strata_family_counts=strata,answer_recovery=answer_recovery)


def gap(tasks):
    left,right = (tasks[t]['answer_only_strata_family_counts'] for t in exp.TASKS)
    assert left.keys()==right.keys()
    la,ra = (np.array(list(x.values())).transpose(1,0,2) for x in (left,right))
    keep = (la.sum(0)[:,0]>0) & (ra.sum(0)[:,0]>0)
    rng = np.random.default_rng(exp.SEED)
    dl,dr = (rng.multinomial(len(x),np.full(len(x),1/len(x)),size=6000) for x in (la,ra))

    def calc(l,r):
        total_l,total_r = l[...,0].sum(-1),r[...,0].sum(-1)
        valid = (l[...,0]>0) & (r[...,0]>0)
        shared = valid & keep
        total = np.where(shared,l[...,0]+r[...,0],0)
        contrast = l[...,1]/np.maximum(l[...,0],1)-r[...,1]/np.maximum(r[...,0],1)
        crude = l[...,1].sum(-1)/total_l-r[...,1].sum(-1)/total_r
        adjusted = (total*contrast).sum(-1)/total.sum(-1)
        return crude, adjusted

    raw,adjusted = calc(la.sum(0),ra.sum(0))
    rb,ab = calc(np.einsum('bf,fsk->bsk',dl,la),np.einsum('bf,fsk->bsk',dr,ra))
    return dict(raw=float(raw),standardized=float(adjusted),raw_ci95=np.quantile(rb,[.025,.975]).tolist(),
                standardized_ci95=np.quantile(ab,[.025,.975]).tolist(),method='Pooled stratum proportions on common support, by edit site × start block × wide choice donor role; descriptive, not mediation.',
                overlapping_strata=int(keep.sum()),total_strata=len(keep))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--split',default='holdout',choices=('pilot','holdout'))
    args = ap.parse_args()
    io.OUT = exp.OUT
    rows,audit = load(args.split)
    out = dict(run=exp.RUN,design_sha256=sha(exp.OUT/'design.json'),analysis_sha256=sha(__file__),
               manifest_sha256=sha(exp.OUT/f'{args.split}-manifest.json'),audit=audit,
               tasks={k:summarize(k,v) for k,v in rows.items()},formal_evidence_promotion=False)
    out['Answer_gap'] = gap(out['tasks'])
    out['Answer_gap']['ablation_gap'] = {k:out['tasks']['colours']['changed']['0'][k]['mean']-out['tasks']['zh']['changed']['0'][k]['mean']
                                        for k in ('H','ALL','V','Hpos','Hback')}
    out['Answer_gap']['baseline_ablation_comparison'] = {
        task:{k:result['changed']['0'][k] for k in ('H','ALL','Hpos','Hback')}
        for task,result in out['tasks'].items()
    }
    io.save(f'{args.split}-report.json',out)
    for task,x in out['tasks'].items():
        print(task,'families',x['families'],'common',x['common_n'])
        for k in ('H','K','V','Q','QH','D','ALL','Hpos','Hback'):
            print(k,[round(100*x['changed'][str(i)][k]['mean'],2) for i in (0,1)],
                  'same target',[(round(x['common'][k][str(i)]['hit']['mean']*100,2),round(x['common'][k][str(i)]['edit_gain']['mean'],3)) for i in (0,1)])
        print('fixed packet',x['cue_effects']['downstream_fixed_packet'])


if __name__=='__main__':
    main()
