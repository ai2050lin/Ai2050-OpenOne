"""Paired uncertainty, visible graph endpoints, precision controls and complete profiles."""
from collections import defaultdict,Counter
from rdc_binding_common import *
from phase2732_rdc_binding_material import connected

def visible_combos(row,position):
    end=row['token_offsets'][position][1];lookup={(w['sentence_id'],w['id']):w for w in row['retrospective_ud']}
    edgegroups=defaultdict(list)
    for word in row['retrospective_ud']:
        if word['head']:edgegroups[(word['sentence_id'],word['head'])].append(word)
    result=[]
    for relations in [('obj','advcl'),('nsubj:pass','obl')]:
      for key,words in edgegroups.items():
        if not set(relations)<=set(w['relation'] for w in words):continue
        head=lookup.get(key)
        pair=[w for w in words if w['relation'] in relations]
        # Multiple matching edges: this conservative record requires all matching dependents.
        necessary=pair+([head] if head else [])
        valid=head is not None and all(w.get('char_span') and w['char_span'][1]<=end for w in necessary)
        result.append({'relations':'+'.join(relations),'sentence_id':key[0],'head':key[1],'visible':bool(valid),
          'necessary_char_ends':[w.get('char_span',[None,None])[1] for w in necessary]})
    return result

def compare_predictions():
    rows=gzread(BASE/'natural_confirmation.json.gz');visibility=[]
    for r in rows:
      for anchor,p in enumerate(r['anchors']):
        combos=visible_combos(r,p);visibility.append({'sample_id':r['sample_id'],'cohort':r['cohort'],'split':r['split'],
          'anchor':anchor,'position':p,'combos':combos,'any_connected_visible':any(c['visible'] for c in combos)})
    compressed(BASE/'analysis/connected_visibility.json.gz',visibility)
    last={r['sample_id']:r for r in visibility if r['anchor']==len(next(s for s in rows if s['sample_id']==r['sample_id'])['anchors'])-1}
    results=[];frozen=read(BASE/'prediction/frozen.json')['selected']
    for b in (16,35):
        winner=frozen[str(b)]['kernel']
        with np.load(BASE/'confirmation'/f'b{b}_{winner}.npz') as z:win=z['squared_error'];den=z['baseline_squared_error']
        for control in ('query','source_mean','source_pair','position_pair','shuffled_position_pair','shuffled_role_position_pair'):
            with np.load(BASE/'confirmation'/f'b{b}_{control}.npz') as z:err=z['squared_error']
            delta=(win-err)/np.maximum(den,1e-8)
            for split in ('connected_test','matched_test','connected_visible'):
              for cohort in ('gum','ewt'):
                ix=[i for i,r in enumerate(rows) if r['cohort']==cohort and ((split=='connected_visible' and last[r['sample_id']]['any_connected_visible']) or r['split']==split)]
                results.append({'block':b,'winner':winner,'control':control,'split':split,'cohort':cohort,'rows':len(ix),
                  'weighted_relative_error_difference':float((win[ix]-err[ix]).sum()/den[ix].sum()) if ix else None,
                  'paired_cluster':clustered(delta[ix],[rows[i]['source_group'] for i in ix])})
    return {'paired':results,'visibility_counts':dict(Counter(r['cohort']+'/'+r['split']+'/'+str(r['any_connected_visible']) for r in last.values())),
      'statistical_scope':'Paired source-cluster means with2000cluster-bootstrap samples; conditional on frozen fit/data, not multiplicity-corrected claims or fresh repetitions.'}

def middle():
    out=BASE/'middle_training';rows=[r for r in gzread(out/'material.json.gz') if r['split']!='train']
    meta=[r for r in rows for _ in r['positions']]
    with np.load(out/'original_native.npz') as z:orig=z['loss'];origarg=z['argmax']
    with np.load(out/'bridge_baseline.npz') as z:bridge=z['loss']
    reports=[];packet={}
    for seed in (2733,2734):
      for condition in ('coherent','order_control'):
        with np.load(out/f'{condition}_{seed}/checkpoint32.npz') as z:loss=z['loss'];arg=z['argmax']
        packet[condition,seed]=loss
        for split in ('validation','test'):
          for cohort in ('gum','ewt'):
            ix=np.array([i for i,r in enumerate(meta) if r['split']==split and r['cohort']==cohort]);groups=[meta[i]['source_group'] for i in ix]
            targets=np.array([target for r in rows for target in r['targets']])
            reports.append({'seed':seed,'condition':condition,'split':split,'cohort':cohort,'target_positions':len(ix),
              'versus_original_mean_nll':float((loss[ix]-orig[ix]).mean()),'versus_original_cluster':clustered(loss[ix]-orig[ix],groups),
              'versus_bridge_mean_nll':float((loss[ix]-bridge[ix]).mean()),'native_accuracy':float(np.mean(origarg[ix]==targets[ix])),
              'trained_accuracy':float(np.mean(arg[ix]==targets[ix]))})
    paired=[]
    for seed in (2733,2734):
      for cohort in ('gum','ewt'):
        ix=np.array([i for i,r in enumerate(meta) if r['split']=='test' and r['cohort']==cohort]);delta=packet['coherent',seed]-packet['order_control',seed]
        paired.append({'seed':seed,'cohort':cohort,'mean_coherent_minus_order':float(delta[ix].mean()),'cluster':clustered(delta[ix],[meta[i]['source_group'] for i in ix])})
    return {'native_bridge_mean_nll_difference':float((bridge-orig).mean()),'reports':reports,'coherent_order_paired':paired,
      'limits':'Two fixed seeds, one block and restricted continued training. Original model baseline is distinct from FP32 local arithmetic bridge.'}

def beta():
    data=read(BASE/'beta_updates/result.json');meta={r['sample_id']:r for r in gzread(BASE/'program_material.json.gz')}
    by={(r['sample_id'],r['objective']):r for r in data['records']};names=[r['objective'] for r in data['summary']]
    ids=read(BASE/'beta_updates/protocol.json')['sample_ids'];common=[sid for sid in ids if all(by[sid,n]['status']=='evaluated' for n in names)]
    reports=[]
    for subset,selected in [('all64_skip_as_noop',ids),('common_evaluated',common)]:
      for name in names:
        deltas=[];acc=[];coll=[];repetition=[]
        for sid in selected:
            r=by[sid,name];base=by[sid,'random_norm_control']['initial'];target=meta[sid]['target_ids'][0]
            ok=r['status']=='evaluated'
            deltas.append(r['loss_delta'] if ok else 0.);acc.append(r['first_token_correct'] if ok else base['argmax']==target)
            coll.append(r['natural_collateral_loss_delta'] if ok else 0.)
            repetition.append(r['repeat_mass']-r['initial']['repeat_mass'] if ok else 0.)
        reports.append({'subset':subset,'objective':name,'rows':len(selected),'original_accuracy':float(np.mean([by[sid,'random_norm_control']['initial']['argmax']==meta[sid]['target_ids'][0] for sid in selected])),
          'first_token_accuracy':float(np.mean(acc)),'mean_loss_delta':float(np.mean(deltas)),
          'loss_cluster':clustered(deltas,[meta[sid]['source_group'] for sid in selected]),'natural_collateral_loss_delta':float(np.mean(coll)),
          'mean_repeat_mass_delta':float(np.mean(repetition))})
    proxy=[r for r in data['records'] if r['objective']=='prefix_repeat_probability_mass' and r['status']=='evaluated']
    return {'reports':reports,'proxy_evaluated_target_inside_repeat_set':sum(r['repeat_set_includes_target'] for r in proxy),
      'proxy_evaluated':len(proxy),'skipped_counts':dict(Counter(r['objective'] for r in data['records'] if r['status']!='evaluated')),
      'scope':'FP32 cached-current-prefix only. Skipped negligible gradients are no-op actions; common-support analysis prevents comparing unequal denominators.'}

def profiles():
    manifest=[];rows=gzread(BASE/'natural_confirmation.json.gz')+gzread(BASE/'program_material.json.gz')
    groups=defaultdict(list)
    for r in rows:groups[r['cohort']+'/'+r['split']].append(r)
    for key,rr in groups.items():
        path=BASE/'atlas/condition_profiles'/(key.replace('/','_')+'.npz')
        if not path.exists():
            total=np.zeros((37,2560),float);square=total.copy();rms=total.copy();count=0
            for row in rr:
                kind='natural' if row['kind']=='natural' else 'program'
                with np.load(BASE/'capture'/kind/f'{row["sample_id"]}.npz') as z:h=unbits(z['H']).astype(float)
                total+=h.mean(1);square+=(h*h).mean(1);rms+=(h/np.sqrt(np.mean(h*h,-1,keepdims=True)).clip(1e-12)).mean(1);count+=1
            npz(path,mean=total/count,std=np.sqrt(np.maximum(square/count-(total/count)**2,0)),RMS_mean=rms/count)
        manifest.append({'condition':key,'samples':len(rr),'source_groups':len({r['source_group'] for r in rr}),'path':str(path.relative_to(BASE))})
    return {'profiles':manifest,'scope':'Every37boundary x2560coordinate, original order. Each sample equal weight, anchors averaged within sample. Raw and RMS-normalized means separate; no Top-K.'}

def main():
    start=time.monotonic();out=BASE/'analysis'
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'prediction':compare_predictions(),'middle_training':middle(),'beta_paired':beta()}
    save(out/'result.json',result);save(BASE/'atlas/result.json',profiles())
    ledger('binding_cpu_analysis_profiles',time.monotonic()-start)
    compact={'visibility':result['prediction']['visibility_counts'],'middle_test':[r for r in result['middle_training']['reports'] if r['split']=='test'],
      'middle_paired':result['middle_training']['coherent_order_paired'],'beta':result['beta_paired']}
    print(json.dumps(compact,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
