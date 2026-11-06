"""Posthoc stronger-rule audit. Does not replace frozen primary thresholds."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.special import softmax
sys.path.insert(0,str(Path(__file__).parent))
import phase2794_materials as P
from phase2794_analyze import ci
from shared.dist_protocol.mechanism_trace import TraceStore,unbits,file_sha
def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'));c=read(P.OUT/'formal-corpus.json');rows=read(P.OUT/'observations.json')['rows'];a=read(P.OUT/'analysis.json')
    train=[i for i,r in enumerate(rows) if r['split']=='discovery'];held=[i for i,r in enumerate(rows) if r['split']=='unseen_combination']
    bits=np.load(P.OUT/'next-logits.npy',mmap_mode='r');tp=softmax(unbits(bits[train]),axis=-1)
    # The provided scope rule selects a number already printed in the input.
    def digit(r):return (r['document'] if r['scope']=='文档' else r['prior'])[0]
    prototypes={d:tp[[j for j,i in enumerate(train) if digit(rows[i])==d]].mean(0) for d in '123456789'}
    result=[]
    for i in held:
        p=softmax(unbits(bits[i]));q=prototypes[digit(rows[i])];kl=float(np.sum(p*(np.log(np.maximum(p,1e-30))-np.log(np.maximum(q,1e-30)))))
        old=next(r for r in a['output_prediction']['rows'] if r['sample_id']==rows[i]['id']);result.append({'sample_id':rows[i]['id'],'family_id':rows[i]['family_id'],'rule_KL':kl,'graph_KL':old['graph_KL']})
    gain=ci([np.mean([r['rule_KL']-r['graph_KL'] for r in result if r['family_id']==e['id']]) for e in c['entities'][:64]])
    z=np.load(P.OUT/'graph-native-prediction.npz');pred=z['heldout_prediction'];actual=z['heldout_actual'];scale=z['train_scale']
    store=TraceStore();y=[];block=a['profile_block']
    for i in train:
        row=rows[i];v=unbits(store.block(row['trace_id'],block)['activation'][-1]);h=store.hidden(row['trace_id'],block+1,row['length']-1)['values'];y.append(np.r_[v,h])
    y=np.asarray(y);means={d:y[[j for j,i in enumerate(train) if digit(rows[i])==d]].mean(0) for d in '123456789'}
    baseline=np.stack([means[digit(rows[i])] for i in held]);internal=[]
    for name,sl in [('mlp',slice(0,9728)),('hidden',slice(9728,None))]:
        err=((actual[:,sl]-pred[:,sl])/scale[sl])**2;be=((actual[:,sl]-baseline[:,sl])/scale[sl])**2
        improvement=ci([float((be[[j for j,i in enumerate(held) if rows[i]['family_id']==e['id']]]-err[[j for j,i in enumerate(held) if rows[i]['family_id']==e['id']]]).mean()) for e in c['entities'][:64]])
        internal.append({'space':name,'source_rule_digit_baseline_mse':float(be.mean()),'graph_mse':float(err.mean()),'relative_reduction':1-float(err.mean()/be.mean()),'paired_gain':improvement})
    doc={'kind':'real','run_id':P.RUN,'status':'posthoc stronger-baseline audit; no threshold reselection or primary promotion',
        'definition':'deterministic external source-selection rule + discovery-only first-digit output/native prototypes; parses values explicitly printed in input',
        'rule_KL':float(np.mean([r['rule_KL'] for r in result])),'graph_KL':a['output_prediction']['graph_KL'],'paired_KL_gain':gain,
        'graph_better_than_rule':bool(gain['ci95'][0]>0),'internal':internal,'rows':result,'script_sha256':file_sha(__file__)}
    P.save(P.OUT/'strong-baselines.json',doc,True);print(json.dumps({k:v for k,v in doc.items() if k!='rows'},ensure_ascii=False))
if __name__=='__main__':main()
