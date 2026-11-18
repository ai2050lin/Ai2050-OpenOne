"""CPU audit of fixed-State insertion, predictions and native interventions."""

from collections import defaultdict
import json
import shutil
import sys

import numpy as np

import kv_receipt_run as R
import kv_receipt_native as V
import kv_receipt_study as S
from kv_origin_audit import summarize
from b36_semantic14_memo import append


TITLE='14B 固定实发 State 的等长文本插入、H18 Receipt 寻址与真实改码'
SPLITS=('discovery','words','frames')


def verify_binding(name):
    doc=R.A.load(name)
    assert all(R.B.sha(k)==hash_ for k,hash_ in doc['sha'].items()),name
    return doc


def interval(y,rows,mask,threshold=None):
    groups=defaultdict(list)
    for v,r,ok in zip(y,rows,mask,strict=True):
        if ok:groups[r['family']].append(float(v))
    if not groups:return dict(n=0,mean=None,ci=None,passed=False)
    groups={k:np.mean(values) for k,values in groups.items()}
    s=summarize(list(groups.values()),S.PLAN['confidence'])
    if threshold is not None:
        s['passed']=s['n']>=S.PLAN['min_panel'] and s['mean']>=threshold and s['ci'][0]>0
    by_word=defaultdict(set)
    for r in rows:
        if r['family'] in groups:
            for w in r['words']:by_word[w].add(r['family'])
    subsets=[np.mean([v for k,v in groups.items() if k not in omit]) for omit in by_word.values() if len(omit)<len(groups)]
    s['leave_word_out_range']=[float(min(subsets)),float(max(subsets))] if subsets else None
    return s


def metrics(rows,z,b,pred):
    x=S.covariates(rows,z,b); ok=x['admitted']; a=x['base']; mass=z['masses'][:,-1]; h=x['h18']
    out=dict(frame_modes=b['gate']['frame_modes'],planned_families=len(rows)//R.CONSORTIUM,
             complete_families=len(b['gate']['complete']),admitted_families=len(b['gate']['eligible']),
             trigger_cells=b['gate']['cells'],shifts=[],onset={},examples=[],equal_text_cases=[],
             verified_views=sum(r['verified'] for r in b['records']),
             greedy_insert_matches=sum(sum(u==v for u,v in zip(r['inside_model_top'],r['inside_actual'],strict=True)) for r in b['records']),
             supplied_tokens=sum(len(r['inside_actual']) for r in b['records']),
             verified_off_code=sum(r['verified'] and r['code_top'] not in r['code_ids'] for r in b['records']))
    ratio=np.log(np.maximum(mass[:,0],1e-25))-np.log(np.maximum(mass[:,1],1e-25))
    assert np.max(np.abs(h-ratio))<.002
    to_fp32=lambda y:(y.astype(np.uint16).astype(np.uint32)<<16).view(np.float32)
    reproduced=(to_fp32(z['native_q'])*to_fp32(z['source_k'])).sum(-1)*128**-.5
    assert np.max(np.abs(reproduced-z['score_pair'][:,1]))<.0001
    for k in ('source_k','source_v'):
        assert np.all(z[k]==z[k][a])
    for i,r in enumerate(rows):
        base=rows[a[i]]; cut=r['insertion']; ids=r['token']['ids']; s=b['records'][i]
        assert r['prompt']==base['prompt'] and ids[:cut]+ids[cut+r['n']:]==base['token']['ids']
        assert s['state_ref']==b['records'][a[i]]['state_ref']==ids[r['token']['slots'][0]]
        assert s['inside_actual']==ids[cut:cut+r['n']] and s['raw_output'][-1]==s['code_top']
        start=r['token']['start']; trial=s['raw_output'][:-1]
        disagreements=[start+j for j,(got,wanted) in enumerate(zip(trial,ids[start:],strict=True)) if got!=wanted]
        assert s['first_failure']==(disagreements[0] if disagreements else None) and s['verified']==(not disagreements)
        if s['verified']:
            assert s['actual_state']==s['state_ref']
    count=np.log(x['count']/x['count'][a]); norm_qk=h-h[a]-count
    common=z['score_pair'][:,2]-z['score_pair'][:,1]
    out['full_read_check']=dict(max_ratio_error=float(np.max(np.abs(h-ratio))),
                               state_KV_bitwise_rows=len(rows),top_source_bins=dict(zip(V.SITES,np.bincount(mass[ok,:6].argmax(-1),minlength=6).tolist(),strict=True)) if ok.any() else {},
                               max_position_bins=dict(zip(V.SITES,np.bincount(z['top_site'][ok],minlength=6).tolist(),strict=True)))
    out['max_receipt_over_state']=float(np.exp(h[ok].max())) if ok.any() else None
    peak_source=mass[:,:6].argmax(1)
    out['group_changes']={}
    for source in range(len(V.SITES)):
        for dest in range(len(V.SITES)):
            if source==dest:continue
            k=int((ok&x['split']&(peak_source[a]==source)&(peak_source==dest)).sum())
            if k:out['group_changes'][f'{V.SITES[source]}→{V.SITES[dest]}']=k
    for mode in (0,1):
        use=ok&np.array([r['conflict']==mode for r in rows]); base_rows=[i for i,r in enumerate(rows) if use[i] and r['n']==0]
        first=defaultdict(int); equal_swap={str(n):0 for n in (7,14,21)}; equal_code=equal_swap.copy(); reverted=0
        for i in base_rows:
            groups={k:[j for j in range(i+1,i+10) if rows[j]['track']==k] for k in range(3)}
            for tracker,ii in groups.items():
                onset=0 if h[i]>0 else next((rows[j]['n'] for j in ii if h[j]>0),'none')
                first[f'{tracker}/{onset}']+=1
                if onset!='none':
                    after=[h[j]>0 for j in ii if rows[j]['n']>=onset]
                    reverted+=int(not all(after))
            for n in (7,14,21):
                same=[j for j in range(i+1,i+10) if rows[j]['n']==n]
                signs=h[same]>0
                equal_swap[str(n)]+=int(signs.any() and not signs.all())
                if len(set(x['top'][same]))>1:
                    equal_code[str(n)]+=1
                    out['equal_text_cases'].append([case(rows,z,b,x,j,'等长更换文字') for j in same])
        out['onset'][str(mode)]=dict(families=len(base_rows),first=first,same_length_text_changes_route=equal_swap,
                                    same_length_text_changes_code=equal_code,nonmonotone=reverted)
        for n in (0,7,14,21):
            for track in ((-1,) if n==0 else (0,1,2)):
                mask=use&np.array([r['n']==n and r['track']==track for r in rows]); k=int(mask.sum())
                if not k:continue
                mean=lambda y:float(y[mask].mean())
                out['shifts'].append(dict(mode=mode,n=n,track=track,views=k,route=int((mask&(h>0)).sum()),
                    greatest=int((mask&(mass[:,:6].argmax(-1)==0)).sum()), receipt=mean(mass[:,0]),state=mean(mass[:,1]),insert=mean(mass[:,6]),
                    switch=int((mask&(h>0)&(h[a]<=0)).sum()), reverse=int((mask&(h<=0)&(h[a]>0)).sum()),
                    changed=int((mask&x['flip']).sum()),changes_with_switch=int((mask&x['flip']&(h>0)&(h[a]<=0)).sum()),
                    log_count=mean(count),mean_score_change=mean(norm_qk),total_change=mean(h-h[a]), common_change=mean(common-common[a])))
    valid=ok&x['split']; truth=h>0; flip=x['flip']
    out['continuous_by_length']={}
    for mode in (0,1):
        for n in (7,14,21):
            mask=ok&np.array([(r['conflict'],r['n'])==(mode,n) for r in rows])
            if not mask.any():continue
            pairs=[('ratio',h-h[a]),('state',mass[:,1]-mass[a,1]),('receipt',mass[:,0]-mass[a,0])]
            diffs={k:interval(v,rows,mask) for k,v in pairs}
            variants=h[mask].reshape(-1,3)
            diffs['mean_range_same_length']=float(np.mean(np.ptp(variants,axis=1)))
            out['continuous_by_length'][f'{mode}/{n}']=diffs
    complete=set(b['gate']['complete'])
    diagnostic=x['split']&~ok&np.array([r['family'] in complete for r in rows])
    out['held_out_by_gate']=[]
    for frame,mode in sorted({(r['frame'],r['conflict']) for r in rows}):
        use=diagnostic&np.array([(r['frame'],r['conflict'])==(frame,mode) for r in rows])
        if use.any():
            out['held_out_by_gate'].append(dict(frame=frame,mode=mode,n=int(use.sum()),flip=int((use&flip).sum()),
                                              receipt_wins=int((use&truth).sum()),
                                              examples=[case(rows,z,b,x,j,'行为门未过的诊断') for j in np.flatnonzero(use&flip)[:2]]))
    out['flip_by_route']={}
    go=valid&flip
    gold=np.array([r['gold'] for r in b['records']])
    state_code=np.array([r['code_state'] for r in b['records']])
    switch=lambda j: float(np.median(x['real'][a[j]])) if j.any() else None
    out['flip_signature']=dict(families=len({rows[i]['family'] for i in np.flatnonzero(go)}),
                               to_fact=int((go&(x['top']==gold)&(state_code!=gold)).sum()),
                               to_state=int((go&(x['top']==state_code)&(state_code!=gold)).sum()),
                               away_from_shared=int((go&(state_code==gold)).sum()),
                               baseline_margin_flipped=switch(go),baseline_margin_retained=switch(valid&~flip))
    for name,mask in [('State→Receipt',(h[a]<=0)&truth),('Receipt→State',(h[a]>0)&~truth),
                      ('State→State',(h[a]<=0)&~truth),('Receipt→Receipt',(h[a]>0)&truth)]:
        sel=valid&mask
        out['flip_by_route'][name]=dict(n=int(sel.sum()),flip=int((sel&flip).sum()))
        unique={}
        for j in np.flatnonzero(sel&flip):
            unique.setdefault(rows[j]['family'],j)
        for j in list(unique.values())[:3]:
            out['examples'].append(case(rows,z,b,x,j,name))
    for j in np.flatnonzero(valid & (h[a]<=0)&truth&~flip)[:3]:
        out['examples'].append(case(rows,z,b,x,j,'Receipt wins, code unchanged'))
    if pred is not None and valid.any():
        def pc(key):return np.asarray(pred[key]>0)==truth
        p_ls={k:(pred[k+'_prob']-flip)**2 for k in ('flip_text','flip_native')}
        assert np.all(pred['ids']==z['ids'])
        out['prediction']=dict(n=int(valid.sum()),read_gain=interval(pc('read_native').astype(float)-pc('read_text'),rows,valid,S.PLAN['read_gain']),
             flip_gain=interval(p_ls['flip_text']-p_ls['flip_native'],rows,valid,S.PLAN['brier_gain']),
             read_accuracy={k:float(pc(k)[valid].mean()) for k in ('read_text','read_native')},
             read_mae={k:float(abs(pred[k][valid]-h[valid]).mean()) for k in ('read_text','read_native')},
             read_majority=float((truth[valid]==(R.A.load('frozen.json')['majority']>=.5)).mean()),
             flip_brier={k:float(v[valid].mean()) for k,v in p_ls.items()}|{'never':float(flip[valid].mean())},
             flip_precision={})
        for k in p_ls:
            guessed=valid&(pred[k+'_prob']>=.5)
            out['prediction']['flip_precision'][k]=dict(tp=int((guessed&flip).sum()),fp=int((guessed&~flip).sum()),fn=int((~guessed&flip&valid).sum()))
    return out,x


def case(rows,z,b,x,j,why):
    i=x['base'][j]; a=b['records']; r=rows[j]
    letter=lambda k,t:'PQ'[a[k]['code_ids'].index(int(t))] if int(t) in a[k]['code_ids'] else int(t)
    return dict(id=r['id'],kind=why,frame=r['frame'],track=r['track'],n=r['n'],mode=r['conflict'],
                state=r['values']['State'],fact=r['words'][r['fact']],text=r['extra'],ratio=[float(x['h18'][k]) for k in (i,j)],
                letter=[letter(k,a[k]['code_top']) for k in (i,j)])


def edits(rows,z,b,x,e):
    positions={r['id']:i for i,r in enumerate(rows)}
    j=np.array([positions[id_] for id_ in e['ids']]); selected=[rows[i] for i in j]; assert x['admitted'][j].all()
    standard=np.asarray([r['pq'] for r in b['records']])[j]
    top=e['top']; raw=x['top'][j]; baseline=x['top'][x['base'][j]]; pq=e['pq']; dose=e['dose']
    good=(top[:,0]==raw)&(abs(pq[:,0]-standard).max(1)<=S.PLAN['sham_pq']); useful=x['split'][j]&good
    flip=x['flip'][j]&useful; named=np.asarray([r['token']['letters'] for r in selected]); sign=x['sign'][j]
    assert np.array_equal(pq[:,0],pq[:,6]) and np.array_equal(top[:,0],top[:,6]); a=V.NAMES.index
    m=(pq[:,:,1]-pq[:,:,0])*sign[:,None]
    out=dict(rows=len(j),clean_sham=int(good.sum()),eligible_inserted=int(useful.sum()),observed_flips=int(flip.sum()),
             full_restore_exact=True, excluded_sham_ids=e['ids'][~good].tolist(), arms={},effects={},doses={},cases=[])
    for name in V.NAMES:
        k=a(name); changed=top[:,k]!=top[:,0]
        code=np.isin(top[:,k],named[0]) if np.all(named==named[:1]) else (top[:,k,None]==named).any(1)
        revert=flip&(top[:,k]==baseline)
        out['arms'][name]=dict(changed=int((useful&changed).sum()),off_code=int((useful&~code).sum()),
                               reversion=int(revert.sum()),mean_margin_change=interval(m[:,k]-m[:,0],selected,useful))
        hits=np.flatnonzero(revert)
        if name in ('address','value','state-value','joint'):
            unique={}
            for l in hits:
                unique.setdefault(selected[l]['family'],l)
            for l in list(unique.values())[:2]:
                item=case(rows,z,b,x,j[l],f'insertion change reverted by {name}')
                item['arm_top']=dict(zip(V.NAMES,top[l].tolist(),strict=True)); item['arm_pq']=pq[l].tolist()
                item['arm_letters']={arm:('PQ'[list(named[l]).index(int(token))] if token in named[l] else str(token))
                                     for arm,token in item['arm_top'].items()}
                out['cases'].append(item)
    for i,name in enumerate(('address','value')):
        e0,e1=dose[:,i*2],dose[:,i*2+1]
        diff=np.abs(e0[:,2]-e1[:,2])/np.maximum(e0[:,1],.02)
        if useful.any():
            out['doses'][name]=dict(p95=float(np.quantile(diff[useful],.95)),median=float(np.median(e0[useful,2])))
            kv=m[:,a('matched-'+name)]-m[:,a('random-'+name)]
            item=interval(kv,selected,useful,S.PLAN['effect'])
            item['matched_dose']=out['doses'][name]['p95']<=S.PLAN['dose_error']
            item['passed'] &= item['matched_dose']
            out['effects'][name]=item
    out['effects']['joint_vs_sham']=interval(m[:,a('joint')]-m[:,0],selected,useful)
    out['effects']['native_receipt_restore_vs_joint']=interval(m[:,a('restore-receipt')]-m[:,a('joint')],selected,useful)
    out['effects']['native_state_restore_vs_joint']=interval(m[:,a('value')]-m[:,a('joint')],selected,useful)
    out['restore_original_after_joint_reversion']=int((flip&(top[:,a('joint')]==baseline)&(top[:,a('restore-receipt')]==raw)).sum())
    out['restore_state_only_after_joint_reversion']=int((flip&(top[:,a('joint')]==baseline)&(top[:,a('value')]==raw)).sum())
    return out


def audit():
    plan=R.A.load('plan.json'); f=R.A.load('frozen.json')
    assert f['plan']==R.B.sha('plan.json') and f['observation']==R.B.sha('discovery-observation.json')
    assert all(R.A.digest(R.ROOT/k)==h for k,h in plan['sources'].items())
    assert R.B.sha('pilot.json')==plan['pilot']
    power_path,=R.OUT.glob('*/*_power-preflight.json')
    power=json.loads(power_path.read_text(encoding='utf-8'))
    assert all(x['pass'] for x in power['payload']['checks'].values())
    assert power_path.stat().st_mtime_ns<(R.OUT/'discovery-material.json').stat().st_mtime_ns
    assert (R.OUT/'plan.json').stat().st_mtime_ns<(R.OUT/'discovery-material.json').stat().st_mtime_ns
    vocab=[set(words) for words in plan['vocabulary'].values()]; assert sum(map(len,vocab))==len(set.union(*vocab))
    results={}
    for stage in SPLITS:
        verify_binding(stage+'-observation.json')
        rows=R.A.load(stage+'-material.json'); z=S.read_np(stage+'-native.npz'); b=R.A.load(stage+'-behavior.json')
        assert R.digest(R.gate(rows,b['records']))==R.digest(b['gate'])
        bank=set(plan['vocabulary'][stage]); assert all(set(r['words'])<=bank for r in rows)
        pred=None
        if stage!='discovery':
            assert (R.OUT/'frozen.json').stat().st_mtime_ns < (R.OUT/(stage+'-material.json')).stat().st_mtime_ns
            verify_binding(stage+'-prediction-binding.json'); pred=S.read_np(stage+'-predictions.npz')
            ref=S.covariates(rows,z,b)
            for key,model in f['models'].items():
                assert np.allclose(S.predict(model['fit'],ref[key]),pred[key],rtol=1e-12,atol=1e-12)
                if key.startswith('flip'):
                    assert np.allclose(S.ndtr(-pred[key]/model['noise']),pred[key+'_prob'],rtol=1e-12,atol=1e-12)
        else:
            fresh=V.controls(rows,z,set(b['gate']['eligible']))
            assert R.digest(fresh)==R.digest(f['controls'])
        res,x=metrics(rows,z,b,pred)
        if stage!='discovery':
            verify_binding(stage+'-intervention-binding.json')
            file=R.OUT/(stage+'-edits.npz')
            if file.exists():
                assert (R.OUT/(stage+'-prediction-binding.json')).stat().st_mtime_ns<file.stat().st_mtime_ns
                res['interventions']=edits(rows,z,b,x,S.read_np(file.name))
        results[stage]=res
    progress=[json.loads(line) for line in (R.OUT/'progress.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    peaks={k:max(r.get(k,0) for r in progress) for k in ('cuda_peak_bytes','shared_gpu_bytes','dedicated_gpu_bytes')}
    return dict(run=R.M.RUN,provenance=power['provenance'],power=power['payload'],panels=results,snapshot=plan['snapshot']['snapshot'],
                tokenizer_sha256=plan['snapshot']['files']['tokenizer.json']['sha256'],peak=peaks,
                plan=R.B.sha('plan.json'),freeze=R.B.sha('frozen.json'),power_preflight=R.A.digest(power_path),controls=f['controls'])


def table(headers,rows):
    return '\n'.join(' | '.join(map(str,r)) for r in (headers,['---']*len(headers),*rows))


def ci(v):
    if not v or not v.get('n'):return '缺失'
    a,b=v['ci']; return f"{v['mean']:+.4f} [{a:+.4f}, {b:+.4f}]，n={v['n']}"


def render(d):
    names=dict(discovery='发现',words='新词',frames='新词与新句式')
    modes={0:'原提示自产',1:'反角色 donor'}
    sources=dict(receipt='Receipt',gen='State',note='Note',extra='Spare/Archive',prompt='提示区',other='其余生成区')
    panels=d['panels']; notes=[]; forecast=[]; contrasts=[]; onset=[]; actions=[]; examples=[]; continuous=[]; rejected=[]; rescue=[]
    for stage,p in panels.items():
        label=names[stage]
        notes.append([label,p['planned_families'],p['complete_families'],p['admitted_families'],
                      p['verified_views'],p['verified_off_code'],f"{p['greedy_insert_matches']}/{p['supplied_tokens']}",
                      ', '.join(f'{f}/{m}' for f,m in p['frame_modes']) or '无'])
        for mode, stats in p['onset'].items():
            targets=[v for k,v in stats['first'].items() if k.endswith('/0')]
            on=sum(targets); counts=[sum(v for k,v in stats['first'].items() if k.endswith(f'/{n}')) for n in (7,14,21,'none')]
            changes='/'.join(str(v) for v in stats['same_length_text_changes_route'].values())
            eq_code='/'.join(str(v) for v in stats['same_length_text_changes_code'].values())
            onset.append([label,modes[int(mode)],stats['families'],on,*counts,changes,eq_code,stats['nonmonotone']])
        for mode in (0,1):
            for n in (0,7,14,21):
                vs=[e for e in p['shifts'] if e['mode']==mode and e['n']==n]
                if not vs:continue
                total=sum(e['views'] for e in vs)
                vals=lambda field:sum(e[field] for e in vs)
                means=lambda field:sum(e[field]*e['views'] for e in vs)/total
                contrasts.append([label,modes[mode],n,total,vals('route'),vals('greatest'),vals('switch'),vals('changed'),vals('changes_with_switch'),
                                  *(f'{means(k):+.3f}' for k in ('total_change','log_count','mean_score_change'))])
                cs=p['continuous_by_length'].get(f'{mode}/{n}')
                if cs and stage!='discovery':
                    continuous.append([label,modes[mode],n,ci(cs['ratio']),ci(cs['state']),ci(cs['receipt']),f"{cs['mean_range_same_length']:.3f}"])
        pred=p.get('prediction')
        if pred:
            acc=pred['read_accuracy']; br=pred['flip_brier']; rr=pred['read_gain']; fr=pred['flip_gain']; cm=pred['flip_precision']['flip_native']
            ma=pred['read_mae']
            forecast.append([label,pred['n'],f"{pred['read_majority']:.3f} / {acc['read_text']:.3f} / {acc['read_native']:.3f}",
                             f"{ma['read_text']:.3f} / {ma['read_native']:.3f}",ci(rr),'过' if rr['passed'] else '未过',
                             f"{br['never']:.4f} / {br['flip_text']:.4f} / {br['flip_native']:.4f}",ci(fr),'过' if fr['passed'] else '未过',
                             f"{cm['tp']}/{cm['fp']}/{cm['fn']}"])
        for r in p['examples'][:8]:examples.append([label,r['id'],r['n'],r['state'],r['fact'],r['kind'],
                                                  '→'.join(f'{h:+.3f}' for h in r['ratio']),'→'.join(map(str,r['letter']))])
        for triplet in p['equal_text_cases'][:2]:
            for r in triplet:
                examples.append([label,r['id'],r['n'],r['state'],r['fact'],r['kind'],
                                 '→'.join(f'{h:+.3f}' for h in r['ratio']),'→'.join(map(str,r['letter']))])
        for item in p['held_out_by_gate']:
            rejected.append([label,item['frame'],modes[item['mode']],item['n'],item['receipt_wins'],item['flip']])
            for r in item['examples']:
                examples.append([label,r['id'],r['n'],r['state'],r['fact'],r['kind'],
                                 '→'.join(f'{h:+.3f}' for h in r['ratio']),'→'.join(map(str,r['letter']))])
        e=p.get('interventions')
        if not e:continue
        for arm,r in e['arms'].items():
            if arm in ('sham','restore-all'):continue
            actions.append([label,arm,e['eligible_inserted'],r['changed'],r['off_code'],r['reversion'],ci(r['mean_margin_change'])])
        for r in e['cases'][:6]:
            branches=r['arm_letters']
            rescue.append([label,r['id'],*r['letter'],*(branches[k] for k in ('address','value','state-value','joint','restore-receipt','restore-all'))])
    overview=[]; technical=[]
    for stage in SPLITS[1:]:
        p=panels[stage]; label=names[stage]; b=p['flip_by_route']; n=sum(v['n'] for v in b.values()); flip=sum(v['flip'] for v in b.values()); sw=b['State→Receipt']
        overview.append(f"{label}留出中 {flip}/{n} 个插入视图真正改码。State→Receipt 的 {sw['n']} 个视图中 {sw['flip']} 个改码，其他三种路径合计 {flip-sw['flip']}/{n-sw['n']}。")
        sig=p['flip_signature']
        if flip:
            overview.append(f"{flip} 次改码来自 {sig['families']} 个家族。其中 {sig['to_state']} 次从提示原事实的码转到实发 State 的码，{sig['to_fact']} 次方向相反。按视图计，改码组与保持组的零插入胜者分差中位数分别为 {sig['baseline_margin_flipped']:.2f} 和 {sig['baseline_margin_retained']:.2f}。")
            if sig['away_from_shared']:
                overview.append(f"另外 {sig['away_from_shared']} 次离开原事实与 State 共用的码。")
        if p['max_receipt_over_state'] is not None:
            overview.append(f"该面板实测 Receipt/State 注意力质量比最大 {p['max_receipt_over_state']:.3f}。")
        flow=['→'.join(sources[k] for k in route.split('→'))+f" {v} 次" for route,v in p['group_changes'].items()]
        if flow:overview.append(f"{label}按六个来源组的最大质量观察到 "+'、'.join(flow)+'。')
        e=p.get('interventions')
        if e:
            ef=e['effects']; dose=e['doses']; a=e['arms']; c=d['controls']
            technical.append(f"{label}中 sham 复现 {e['clean_sham']}/{e['rows']}。全原生恢复与 sham 在保存的 P/Q logits 和全词表首选上逐位相同。干预有效插入视图中原有 {e['observed_flips']} 次改码，移除 Receipt 寻址、清 Receipt Value、清 State Value、联合清零分别回到零插入基线 {a['address']['reversion']}、{a['value']['reversion']}、{a['state-value']['reversion']}、{a['joint']['reversion']} 次。联合清零已回基线的样本中，仅恢复原 Receipt 又回到插入字母 {e['restore_original_after_joint_reversion']} 次。")
            if e['excluded_sham_ids']:
                technical.append(f"以下视图因 sham 和干净分支对齐失败，排除于干预统计：{'、'.join(e['excluded_sham_ids'])}。")
            for k in ('address','value'):
                if k not in dose:continue
                technical.append(f"{label} {k} 对照为 L25H{c[k]['head']}，匹配剂量相对差 P95 为 {dose[k]['p95']:.2%}。H18 减随机头的配对基线字母分差差异为 {ci(ef[k])}，按冻结的方向与幅度门{'通过' if ef[k]['passed'] else '未通过'}。")
            technical.append(f"{label}联合阻断相对 sham 为 {ci(ef['joint_vs_sham'])}，仅恢复 Receipt 相对联合阻断为 {ci(ef['native_receipt_restore_vs_joint'])}。从联合清零仅恢复 State、使 Receipt 继续为零，与 Receipt Value 单阻断是同一条件。它相对联合零值的分差为 {ci(ef['native_state_restore_vs_joint'])}，在联合零值已回到原基线的样本中重新发出插入字母 {e['restore_state_only_after_joint_reversion']} 次。")
        else:
            technical.append(f'{label}未取得适用的干预材料，详见逐格准入记录。')

    return f'''### 问题、实际输入与冻结

本阶段在本地 Qwen3-14B BF16 快照 `{d['snapshot']}` 上追踪 L25H18（块和头均零基索引）。问题是同一个提示、同一个实发 State 和原始记录顺序下，插入文字和长度如何改变 `Code:` 末位的查询，以及是否改变全词表首选 P/Q。每家族有零插入基线和三条文字轨道，各插 7、14、21 个准确 token。轨道句子是原创英文中性陈述。每条严格移除插入段后，连同提示、State、剩余记录和 `Code:` 在内的 token 序列逐 token 相同。答案映射、实体角色及码本顺序在家族间变动。

模型在原始提示下完成 State、Note、Spare、Archive 与固定 Receipt 开头。反角色条件的 State 由同一模型在反转事实的 donor 提示中真实贪心得出，再将该 token 用回原提示的生成历史，这一组是受控历史。插入文字由实验分配。插入位置之前和之后、以及 `Code:` 之后的全词表首选均逐位置核验。插入位置也保存模型当时原本会生成的 token。一次试跑共 12 家族，有一个反角色 State donor 未发出目标 token。正式准入要求每个句式、State 模式、轨道及长度的格式和 P/Q 比例至少 85%，自然零插入还须答对。插入后的 P/Q 胜负不参与筛选。正式分析使用全十视图均满足条件的家族。所有失败保留在 `*-behavior.json`。

{table(['面板','计划家族','完整','准入','逐位置验证视图','已验证但出 P/Q','插入处匹配原贪心','通过的句式/模式'],notes)}

下表仅列未通过群体行为门、但个体十个视图已完整核验的观察。它们用于诊断覆盖缺口，独立于留出判定和干预。

{table(['面板','句式','State 模式','完整的插入视图','Receipt 胜 State','P/Q 改码'],rejected)}

试跑、发现、新词、新句式使用互斥词池。新句式包含 active/inactive 和 kept/dropped 两个此前未用于拟合的提示框架，输出格式和中性段仍相同。每块 64 个计划确认家族。预先用家族为单位按 McNemar 大二值效应规划，发现假设 36 个百分点、所需 35 家族、计划功效 0.823，两个留出各假设 28 个百分点、所需 53 家族、计划功效 0.887（双侧 α=.01）。该预算的结论限定于大效应。小效应、按句式或角色拆分的亚组、稀有翻转及 Brier 改善的功效需要另行估计。置信区间按家族等权重抽样，词池内复用通过整词剔除敏感性记录在 `audit.json`。发现拟合与控制抽取冻结在两个留出材料落盘之前。冻结记录 SHA `{d['freeze']}`。

### 寻址条件与实际输出

令 G 是实发 State 的 token 位置，R 是这次输出历史的 Receipt 行，包含标题、固定句和插入段，C 是 `Code:` 的最后 token。对 L25H18，令 `s_i = q_C^T k_i / √128`，`a = softmax(s)`。Receipt 相对 State 的注意力 log 比满足 `D = log(Σ_{{i∈R}} a_i / a_G) = log|R| + log(mean_{{i∈R}} exp(s_i)) − s_G`。这里 `D > 0` 表示 Receipt 的总质量超过 State 单位置。另记 Receipt 是否超过所有预先定义的来源组，并保存全位置权重。逐例验算分解，记录固定 Receipt 部分和新增部分的质量。State 的原生 K/V 在同家族十视图中逐位相同。注意力质量随查询、相对位置和竞争项变化，原位置向量可以继续被读取。

{table(['面板','State 模式','家族','零插入已胜(轨道数)','首胜7','首胜14','首胜21','从未胜','等长换词跨过阈值的家族7/14/21','等长换词改变首选的家族7/14/21','首胜后再退回的轨道'],onset)}

{table(['面板','State 模式','插入token','视图','R胜G','R为最大组','G→R','真实改码','G→R且改码','ΔD','Δlog数量','Δ平均匹配项'],contrasts)}

{table(['留出','State 模式','插入token','配对 ΔD','State 质量变化','Receipt 质量变化','等长文本间 D 极差均值'],continuous)}

{' '.join(overview)} 一次真实改码要求原分支与插入分支的全词表首选都属于 P/Q 且相互切换。候选间分差保留供复算。正负分差跨界只能解释两码排名，实际首选仍须检查其他词。

### 冻结的留出预测

文本基线用长度、文字轨道、角色、码本和位置。原生模型再读 L20H5、L21H12、L24H18 的来源质量比及相对零插入的变化。两个改码预测器都额外使用该家族零插入时已经实际观察到的 P/Q 分差。岭回归只在发现拟合。各插入分支的 L25H18 和最终分差是检验目标，输入特征来自更早位置或独立的零插入分支。改码概率为 `Φ(−m̂/σ̂)`。`m̂` 是朝向零插入首选的预测分差，`σ̂` 是发现残差 RMS 并设 0.2 下限。所有改码预测与干预开始间的哈希及时间单独绑定。联合推广须两个面板分别过门。预测收益门分别为正确率净增 3 个百分点与 Brier 减少 0.01，99% 家族区间下界大于零且至少 40 家族。

{table(['面板','插入视图','寻址准确率 多数/文本/早期原生','D预测MAE 文本/早期原生','准确率增量','通过','改码Brier 不改码/文本/早期原生','Brier改善','通过','改码0.5阈值 TP/FP/FN'],forecast)}

### 原模型单查询干预

在 L25H18 的 C 位分别实施 Receipt 寻址阻断、Receipt Value 清零、State Value 清零及联合清零。令头的原始 Value 汇总为 y，Receipt 的注意力质量为 a_R，Value 加权和为 u_R。寻址阻断后使用 (y−u_R)/(1−a_R)，会把质量分给仍可见的来源。只清 Value 后使用 y−u_R，保留全部 QK 权重。两者都由原模型分支计算。只恢复 Receipt 使用原模型该行 Value，State 仍清零。全恢复还原两个来源。随机头从发现集同层且不与 H18 共用 KV 组的近剂量头中按固定种子抽取，目标与随机一侧的单头输出差向量都按两剂量较小值缩放，剂量用投影到残差流后的二范数，匹配门 P95≤3%。表中的分差朝向零插入时的实际胜字母。正值表示分数回到基线方向。

{' '.join(technical)}

{table(['面板','干预','有效插入视图','相对插入 sham 换首选','换到P/Q外','使原改码回基线','基线字母分差变化'],actions)}

### 个例和反例

{table(['面板','视图ID','插入token','实发State','提示原事实','类型','Receipt/State对数比 基线→插入','全词表首选 基线→插入'],examples)}

{table(['面板','视图ID','基线码','插入码','屏蔽 Receipt','清 Receipt V / 只恢复 State','清 State V','联合清零','只恢复 Receipt','全恢复'],rescue)}

全语料的未改码、回切、不匹配与格式失败可以直接按视图 ID 查询同目录材料、行为、预测、全位置注意力和干预文件。个例表按固定顺序取例，不能用来估计总体比例。

### 机制增量、局限和下一段研究

外部图谱增加由同一提示和 State 共享的插入长度、文本替换及生成历史关系。内部图谱增加 State 原生 K/V 固定而读取比例可变的数值关系，以及写入阻断和原生恢复的逐样本结果。关联图谱按留出预测和干预的实际门槛登记证据范围。相同的输入记录可以存下同一内容，后来的查询也可以因其位置和其他候选而产生不同读法。这个局部机制可以用原生注意力代数表达，RDC 暂无新增统一定律。若 Receipt 质量更高但答码稳定，后续层、备份来源或已有分差均可能维持答案。若 Receipt 不胜 State 却改码，需要结合其他来源与干预定位。

插入实验同时改变后续 token 的位置和可见历史。局部数量项为精确定义，剩余项含相对位置、Q/K 几何和组内权重分布，无法仅据分解拆成互相独立的因果量。Receipt 采用整行组，会合并标签、固定语句和插入文本。单查询阻断是范围内计算证据。等幅随机比较针对 Receipt 寻址和 Receipt Value，State Value 的阻断和恢复提供同一路径的计算证据。材料为英文模板、两个代码字母、三段固定中性轨道和单机器模型快照。反角色组与全部插入 token 通过受控历史建立。跨自然开放续写、独立物理复算、完整下一 token 分布保真，以及同预算稀疏字典比较仍待完成。

下一大阶段把组竞争和真正用到的信息联系起来。先冻结同长度、同末位的内容替换与保留原内容只移动相对位置的双控制，再把整行拆为标题、固定语句、插入句三个来源并分别恢复。围绕预测确实失败和改码反例，在同一未见材料上比较更早原生特征、位置基线和等预算稀疏模型。最后挑有预测证据的源端到输出链，用独立节点、原生阻断恢复和自然续写验收。有限参数推进语言组合的一个可检验命题是，稳定的状态写入可以与条件化的访问运算分别复用。当前证据约束这一命题的局部实例。

### 复算与资源

运行 `{d['run']}`。入口 `tests/glm5/kv_receipt_run.py pilot`、同入口 `all`，以 `.venv/Scripts/python.exe -B -u` 执行。CPU 审计与追加入口 `tests/glm5/kv_receipt_report.py`，加 `close` 完成尾部追加。计划、全部输入、日志、Q 与 State K/V、逐位置质量、固定审计视图的 41 边界 E/G/C 逐坐标、每个候选 P/Q logits 与全词表首选、等幅尺度和反例位于 `tests/glm5/result/kv_receipt_20261117_v1/`。计划 SHA `{d['plan']}`，tokenizer.json SHA `{d['tokenizer_sha256']}`。RTX 5080 峰值 CUDA {d['peak']['cuda_peak_bytes']/2**30:.2f} GiB，DXGI 共享峰值 {d['peak']['shared_gpu_bytes']/2**20:.0f} MiB。先前试跑和正式任务串行。预冻结 smoke 曾因修正自然分支的行为筛选而重启，日志保留。本次写入 D 盘并保留原始结果与来源绑定。正式中心登记须独立复算与发布授权。
'''


def run(close=False):
    R.setup(2**20)
    d=audit(); R.A.save('audit.json',d); body=render(d)
    R.P.check_write(R.OUT/'report.md').write_text('# '+TITLE+'\n\n'+body,encoding='utf-8')
    if not close:return
    m=R.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    assert TITLE not in m.read_text(encoding='utf-8-sig')
    cleanup=[]; q=R.TMP
    if q.exists():
        assert q.resolve()==q and q.parent==(R.ROOT/'tests/glm5_temp').resolve()
        items=list(q.rglob('*'))
        assert all(not p.is_symlink() and not p.is_junction() and p.resolve().is_relative_to(q) for p in [q,*items])
        freed=sum(path.stat().st_size for path in items if path.is_file())
        shutil.rmtree(q); cleanup.append(dict(path=str(q),bytes=freed))
    R.A.save('cleanup.json',dict(paths=cleanup,free=shutil.disk_usage(R.OUT).free))
    ledger=append(R.P.check_write(m),TITLE,body)
    pointer=append(R.P.check_write(R.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'),'14B 固定 State 等长文本插入与真实改码索引',
                   f'见 GLM5 Phase {ledger["phase"]}。本地 Qwen3-14B CUDA 研究，运行 `{R.M.RUN}`。冻结的新词和新句式结果、干预范围与反例见 `{R.OUT.relative_to(R.ROOT).as_posix()}/report.md`。')
    R.A.save('memo-receipt.json',dict(records=[ledger,pointer]))
    sources=R.A.load('plan.json')['sources']
    source_key='tests/glm5/kv_receipt_report.py'; sources[source_key]=R.A.digest(R.ROOT/source_key)
    files={p.relative_to(R.OUT).as_posix():R.A.digest(p) for p in sorted(R.OUT.rglob('*')) if p.is_file() and p.name!='manifest.json'}
    R.A.save('manifest.json',dict(provenance=d['provenance'],files=files,sources=sources))
    print('Appended GLM5 Phase',ledger['phase'],flush=True)


if __name__=='__main__':
    run('close' in sys.argv)
