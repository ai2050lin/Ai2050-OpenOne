"""Descriptive behavior-error strata; no candidate fitting or threshold changes."""
import numpy as np
import record_selection_core_v3 as R
from record_selection_core_v3 import C

def main():
    result=[]
    for split,n in [('discovery',128),('confirm',384)]:
        rows=C.read(C.OUT/(split+'-materials.json'))['rows'];scores=sum((C.read(C.OUT/f'{split}-{f:04d}.json')['scores'] for f in range(n)),[])
        assert [r['id'] for r in rows]==[r['id'] for r in scores]
        groups=[]
        for field in ['valid_position','language','conflict','query','mapping','frame']:
            for value in sorted({r[field] for r in rows}):
                keep=[s for r,s in zip(rows,scores) if r[field]==value];wrong=[s for s in keep if not s['strict_color_correct']]
                groups.append({'field':field,'value':value,'n':len(keep),'errors':len(wrong),'accuracy':1-len(wrong)/len(keep),
                    'ties':sum(s['color_tie'] for s in keep),'mean_target_margin':float(np.mean([s['target_margin'] for s in keep]))})
        failed=[{'sample':r,'score':s} for r,s in zip(rows,scores) if not s['strict_color_correct']]
        result.append({'split':split,'n':len(rows),'strata':groups,'failures':failed,
            'scope':'posthoc descriptive marginal groups overlap; not new significance tests or reasons established by intervention'})
    C.save('behavior-failure-strata.json',{'schema':'record_selection_failure_strata.v1','results':result,
        'inference_limit':'Position/language associations do not identify the cause; content distractors vary although the target is record number.'})
    print([{k:x[k] for k in ['split','n']}|{'strata':[g for g in x['strata'] if g['field'] in ['valid_position','language','conflict']]} for x in result],flush=True)

if __name__=='__main__':main()
