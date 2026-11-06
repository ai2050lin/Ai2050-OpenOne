"""Frozen metrics and graph-only heldout prediction, separate from model execution."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.special import softmax
sys.path.insert(0,str(Path(__file__).parent))
import phase2794_materials as P
from shared.dist_protocol.mechanism_trace import TraceStore,unbits,file_sha

def ci(v):
    v=np.asarray(v,float);rng=np.random.default_rng(P.SEED);b=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)
    return {'mean':float(v.mean()),'ci95':np.quantile(b,[.025,.975]).tolist(),'n_entity_clusters':len(v)}

def feature_matrix(items,entities):
    ids={e['id']:i for i,e in enumerate(entities[:64])};x=np.zeros((len(items),213),np.float32)
    for i,r in enumerate(items):
        scope=int(r['scope']=='现实');x[i,ids[r['family_id']]*2+scope]=1
        x[i,128+int(r['document'][0])*2+scope]=1
        x[i,148+int(r['prior'][0])*2+scope]=1
        if r['partner']:x[i,148+ids[r['partner']]]=1
        x[i,-1]=1
    return x

def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    c=read(P.OUT/'formal-corpus.json');raw=read(P.OUT/'observations.json');rows=raw['rows'];pr=read(P.PUBLIC/'prereg.research.json')
    freeze=read(P.OUT/'execution-freeze.json');assert all(file_sha(P.ROOT/p)==h for p,h in freeze['dependencies'].items())
    train=np.array([i for i,r in enumerate(rows) if r['split']=='discovery']);test=np.array([i for i,r in enumerate(rows) if r['split']=='unseen_combination'])
    stats=[]
    for split in ['discovery','unseen_combination','unseen_entity','length_panel']:
        for scope in ['文档','现实']:
            rr=[r for r in rows if r['split']==split and r['scope']==scope]
            stats.append({'split':split,'scope':scope,'n':len(rr),'accuracy':float(np.mean([r['correct'] for r in rr]))})
    gate=all(s['accuracy']>=.8 for s in stats if s['split'] in ['unseen_combination','unseen_entity'])
    # Full-vocabulary next-token distributions. Test labels, states and logits
    # are used only below for evaluation, never to build the predictor.
    logits=np.load(P.OUT/'next-logits.npy',mmap_mode='r');priorbits=np.load(P.OUT/'source-free-logits.npy',mmap_mode='r')
    tp=softmax(unbits(logits[train]),axis=-1);entityid={e['id']:i for i,e in enumerate(c['entities'])}
    priors=np.stack([tp[[j for j,i in enumerate(train) if rows[i]['scope']=='现实' and rows[i]['family_id']==e['id']]].mean(0) for e in c['entities'][:64]])
    digitids=c['digit_ids'];copies={}
    for digit in range(1,10):
        idx=[j for j,i in enumerate(train) if rows[i]['scope']=='文档' and rows[i]['document'].startswith(str(digit))]
        assert idx,'discovery digit coverage missing'
        copies[str(digit)]=tp[idx].mean(0)
    def div(p,q):return np.sum(p*(np.log(np.maximum(p,1e-30))-np.log(np.maximum(q,1e-30))),axis=-1)
    pt=np.stack([priors[entityid[rows[i]['family_id']]] for i in train]);cp=np.stack([copies[rows[i]['document'][0]] for i in train])
    def alpha(indices):
        return float(min(np.linspace(0,1,101),key=lambda a:float(div(tp[indices],np.maximum(a*cp[indices]+(1-a)*pt[indices],1e-30)).mean())))
    weights={s:alpha(np.array([j for j,i in enumerate(train) if rows[i]['scope']==s])) for s in ['文档','现实']};blind=alpha(np.arange(len(train)))
    metrics=[];preds=[]
    for i in test:
        r=rows[i];p=softmax(unbits(logits[i]));prior=priors[entityid[r['family_id']]];copy=copies[r['document'][0]];a=weights[r['scope']]
        pred=a*copy+(1-a)*prior;scopeblind=blind*copy+(1-blind)*prior
        metrics.append({'sample_id':r['id'],'family_id':r['family_id'],'scope':r['scope'],'graph_KL':float(div(p,pred)),
            'scope_blind_KL':float(div(p,scopeblind)),'prior_KL':float(div(p,prior)),'copy_KL':float(div(p,copy)),
            'predicted_first_token_id':int(pred.argmax()),'actual_first_token_id':int(p.argmax())})
        preds.append(np.argsort(-pred)[:12].tolist())
    gains=[np.mean([r['scope_blind_KL']-r['graph_KL'] for r in metrics if r['family_id']==e['id']]) for e in c['entities'][:64]]
    output_effect=ci(gains);gkl=float(np.mean([r['graph_KL'] for r in metrics]));bkl=float(np.mean([r['scope_blind_KL'] for r in metrics]));reduction=1-gkl/bkl
    graph_supported=gate and reduction>=.1 and output_effect['ci95'][0]>0
    P.save(P.OUT/'graph-predictor.json',{'weights_by_scope':weights,'blind_weight':blind,'training_ids':[rows[i]['id'] for i in train],
        'input_contract':'only cached discovery-reality distributions for64 familiar entities, parsed document first digit, scope; both reference values explicitly in input; no heldout forward/state/logit in prediction',
        'limits':'empirical full-vocabulary next-token mixture, not a complete mechanistic graph or full-integer completion predictor'},True)
    # Full native-unit and residual-vector prediction on a discovery-chosen block.
    selection=read(P.OUT/'selection.json');block=selection['profile_block'];store=TraceStore();ys=[]
    for i in np.r_[train,test]:
        r=rows[i];z=store.block(r['trace_id'],block);ys.append(np.r_[unbits(z['activation'][-1]),np.asarray(store.hidden(r['trace_id'],block+1,rows[i]['length']-1)['values'])])
    y=np.asarray(ys,np.float32);yt=y[:len(train)];yv=y[len(train):]
    x=feature_matrix([rows[i] for i in train],c['entities']);v=feature_matrix([rows[i] for i in test],c['entities'])
    coef=np.linalg.solve(x.T@x+np.eye(x.shape[1],dtype=np.float32),x.T@yt);pred=v@coef
    means={(e['id'],s):yt[[j for j,i in enumerate(train) if rows[i]['family_id']==e['id'] and rows[i]['scope']==s]].mean(0) for e in c['entities'][:64] for s in ['文档','现实']}
    baseline=np.stack([means[(rows[i]['family_id'],rows[i]['scope'])] for i in test]);scale=np.maximum(yt.std(0),.02)
    per=((yv-pred)/scale)**2;errbase=((yv-baseline)/scale)**2
    internal=[]
    for name,sl in [('mlp',slice(0,9728)),('hidden',slice(9728,None))]:
        gain=[float((errbase[[j for j,i in enumerate(test) if rows[i]['family_id']==e['id']],sl]-per[[j for j,i in enumerate(test) if rows[i]['family_id']==e['id']],sl]).mean()) for e in c['entities'][:64]]
        result=ci(gain);rel=1-float(per[:,sl].mean()/errbase[:,sl].mean())
        internal.append({'space':name,'dimensions':9728 if name=='mlp' else 2560,'mse':float(per[:,sl].mean()),'entity_scope_baseline_mse':float(errbase[:,sl].mean()),
            'relative_reduction':rel,'paired_gain':result,'supported':gate and rel>=.05 and result['ci95'][0]>0})
    np.savez(P.OUT/'graph-native-prediction.npz',coefficients=coef,heldout_prediction=pred,heldout_actual=yv,heldout_baseline=baseline,train_scale=scale)
    # Entity-cluster effects for all predeclared doses and controls.
    intervention=read(P.OUT/'interventions.json')['rows'];byid={r['id']:r for r in rows};doses=[]
    for condition,alpha in [('mlp',.25),('mlp',.5),('mlp',1.),('random_mlp',1.),('heads',1.),('random_heads',1.)]:
        rr=[r for r in intervention if r['condition']==condition and r['alpha']==alpha];families=sorted({r['family_id'] for r in rr})
        loss=ci([np.mean([int(byid[r['sample_id']]['correct'])-int(r['correct']) for r in rr if r['family_id']==e]) for e in families])
        gap=ci([np.mean([byid[r['sample_id']]['first_token_gap']-r['first_token_gap'] for r in rr if r['family_id']==e]) for e in families])
        doses.append({'condition':condition,'alpha':alpha,'n':len(rr),'accuracy':float(np.mean([r['correct'] for r in rr])),'loss':loss,'gap_attenuation':gap})
    target=next(d for d in doses if d['condition']=='mlp' and d['alpha']==1);control=next(d for d in doses if d['condition']=='random_mlp')
    paired=[]
    for e in c['entities'][:64]:
        sel=[r for r in intervention if r['family_id']==e['id'] and r['condition']=='mlp' and r['alpha']==1]
        rnd=[r for r in intervention if r['family_id']==e['id'] and r['condition']=='random_mlp'];paired.append(np.mean([r['correct'] for r in rnd])-np.mean([r['correct'] for r in sel]))
    specific=ci(paired);h2=gate and target['loss']['mean']>=.2 and specific['mean']>=.1 and specific['ci95'][0]>0
    result={'kind':'real','run_id':P.RUN,'behavior_gate':gate,'cell_statistics':stats,'H2_supported':bool(h2),'specific_loss_vs_random':specific,'dose_curve':doses,
        'output_prediction':{'supported':bool(graph_supported),'graph_KL':gkl,'scope_blind_KL':bkl,'relative_reduction':reduction,'paired_gain':output_effect,
            'rows':metrics,'top1_agreement':float(np.mean([r['predicted_first_token_id']==r['actual_first_token_id'] for r in metrics]))},
        'internal_prediction':internal,'profile_block':block,'provenance':raw['provenance']}
    P.save(P.OUT/'analysis.json',result,True);print(json.dumps({k:v for k,v in result.items() if k not in ['provenance','output_prediction']},ensure_ascii=False))
if __name__=='__main__':main()
