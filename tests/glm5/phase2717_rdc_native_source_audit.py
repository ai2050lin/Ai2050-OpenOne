"""Full-coordinate position-coverage audit of the frozen all-source H23 compiler input."""
from rdc_relation_common import *
from rdc_relation_inference import CurrentRule


def main():
    out=BASE/'native_source_audit';guard(1024**2)
    if (out/'result.json').exists():return
    save(out/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),
      'question':'Does applying an anchor-trained rule to every source introduce an untested position/amplitude extrapolation?',
      'scope':'Posthoc native-input coverage audit, no retuning or new model forward. Every fresh token and all2560 H23 coordinates; identity H12 is a diagnostic baseline, not a decoded language algorithm.',
      'groups':'First token, selected query anchors, other past positions available to at least the later query, later positions not consumed by either native query.',
      'metrics':'Full-coordinate errors and energies; identical-source weighting in group summaries. All-position errors saved with IDs, not pooled as independent sentences.'})
    rule=CurrentRule();allrows=[];profiles={}
    for r in rows(True):
        z=load_field(r,True);h12=unbits(z['h12']);actual=unbits(z['h23']);p=rule(h12)[:,:2560];error=(p.astype(float)-actual)**2;identity=(h12.astype(float)-actual)**2
        for j in range(len(p)):
            group='first_token' if j==0 else 'query_anchor' if j in r['anchors'] else 'other_known_past' if j<max(r['anchors']) else 'later_not_consumed'
            record={'sample_id':r['sample_id'],'language':r['language'],'position':j,'group':group,'token_id':r['prompt_ids'][j],
              'frozen_quadratic_MSE':float(error[j].mean()),'identity_MSE':float(identity[j].mean()),'actual_H23_energy':float(np.mean(actual[j].astype(float)**2)),
              'H12_energy':float(np.mean(h12[j].astype(float)**2)),'predicted_H23_energy':float(np.mean(p[j].astype(float)**2))}
            allrows.append(record)
            if group not in profiles:profiles[group]={'quadratic':np.zeros(2560),'identity':np.zeros(2560),'actual_energy':np.zeros(2560),'count':0}
            profile=profiles[group];profile['quadratic']+=error[j];profile['identity']+=identity[j];profile['actual_energy']+=actual[j].astype(float)**2;profile['count']+=1
    summary={}
    for group,profile in profiles.items():
        rr=[r for r in allrows if r['group']==group];units=sorted({r['sample_id'] for r in rr});summary[group]={'tokens':len(rr),'sources':len(units),
          **{name:float(np.mean([np.mean([r[name] for r in rr if r['sample_id']==sid]) for sid in units])) for name in ('frozen_quadratic_MSE','identity_MSE','actual_H23_energy','H12_energy','predicted_H23_energy')}}
    npz(out/'all_coordinate_profiles.npz',**{group+'_'+name:p[name]/p['count'] for group,p in profiles.items() for name in ('quadratic','identity','actual_energy')})
    save(out/'all_position_rows.json',allrows);save(out/'result.json',{'timestamp':stamp(),'groups':summary,'note':'Profiles are token-weighted within each position group; summary means first average per source. Identity is not a causal ablation. Huge out-of-domain source errors would limit interpretations of the primary-vs-hybrid native comparison.'});print('NATIVE_SOURCE_COVERAGE_AUDIT',summary,flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
