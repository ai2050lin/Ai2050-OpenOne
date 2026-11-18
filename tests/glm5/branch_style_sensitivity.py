"""Post-hoc scope audit; no candidate refitting or confirmation promotion."""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import branch_style_joint as C
import branch_style_analysis as A

def main():
    rows=C.selected_rows('confirm');pairs=A.pairs(rows)
    bundle=np.load(C.OUT/'confirm_predictions.npz');truth=bundle['truth']
    effects={}
    for baseline in ['additive','shared','embedding']:
        gain=((bundle[baseline]-truth)**2).mean(1)-((bundle['interaction']-truth)**2).mean(1)
        fam=sorted({k[0] for k,i,j in pairs});d={f:float(np.mean([g for (k,i,j),g in zip(pairs,gain) if k[0]==f])) for f in fam}
        objects={r['family']:{r['slots']['X'],r['slots']['Y']} for r in rows}
        entities=sorted(set().union(*objects.values()))
        leave={e:float(np.mean([d[f] for f in fam if e not in objects[f]])) for e in entities}
        effects[baseline]={'pooled_gain':float(np.mean(list(d.values()))),'leave_one_entity_out':leave,'min':min(leave.values()),'max':max(leave.values()),'scope':'Descriptive entity-deletion sensitivity; does not turn pair-cluster intervals into fully independent lexical evidence.'}
    raw=[json.loads(s) for s in (C.OUT/'causal_rows.jsonl').read_text(encoding='utf-8').splitlines()]
    table={(r['target'],r['condition']):r for r in raw};tt=sorted({r['target'] for r in raw})
    same_x=[];same_color=[];strict=[];gain=[];family=[]
    for t in tt:
        rec=table[t,'target'];a,b=rows[t]['slots'],rows[rec['wrong']]['slots']
        same_x.append(a['X']==b['X']);same_color.append(a['C']==b['C'])
        if a['X']!=b['X'] and a['C']!=b['C']:
            strict.append(t);gain.append(table[t,'wrong_donor']['kl_donor_to_treated']-rec['kl_donor_to_treated']);family.append(rec['family'])
    corrected={'cross_record_donor_same_target_name_fraction':float(np.mean(same_x)),'cross_record_donor_same_target_color_fraction':float(np.mean(same_color)),'strict_different_name_and_color_rows':len(strict),'strict_families':len(set(family)),'strict_target_minus_cross_record_gain':A.bootstrap_difference(np.array(gain),np.zeros(len(gain)),family),'warning':'Cross-record donor was predeclared; it did not guarantee different target or color. Strict subset is post-hoc, not an independent confirmatory test.'}
    C.save(C.OUT/'sensitivity_report.json',{'kind':'real_secondary','entity_sensitivity':effects,'donor_identity_audit':corrected,'no_refitting':True})
    print(json.dumps({'entity_min_max':{k:[v['min'],v['max']] for k,v in effects.items()},'donor':corrected},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
