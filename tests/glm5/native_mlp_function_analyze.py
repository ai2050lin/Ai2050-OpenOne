"""Native-role primary tests, exact causal controls and hidden-state comparator."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from native_mlp_function_run_v4 import status
from shared.dist_protocol.mechanism_trace import TraceStore,unbits,file_sha

def score(diff,sign,keys):
    d=np.asarray(diff);truth=d*sign;nonzero=d!=0;acc=float(np.mean(truth>0)+.5*np.mean(~nonzero));wins=int(np.sum(truth>0));n=int(nonzero.sum())
    # The nuisance uses discovery-fitted current selector identity only.
    return {'n':len(d),'accuracy':acc,'nonzero_fraction':float(nonzero.mean()),'p':float(binomtest(wins,n,.5,alternative='greater').pvalue) if n else 1.,'mean_oriented_delta':float(truth.mean())}
def main():
    c=P.read(P.OUT/'corpus.json');obs=P.read(P.OUT/'observations.json');pr=P.read(P.PUBLIC/'prereg.research.json');fr=P.read(P.OUT/'execution-freeze.json');assert all(file_sha(P.ROOT/n)==h for n,h in fr['dependencies'].items())
    store=TraceStore();units=c['observed_units'];rows=obs['rows'];items={r['id']:r for r in c['items']};z=np.zeros((len(rows),4,len(units),3),np.float32);hidden=np.zeros((len(rows),4,37,2560),np.float32);early_errors=[];rms={l:np.zeros(9728) for l,u in units};ntrain=0
    for i,r in enumerate(rows):
        d,_=store.manifest(r['trace_id']);a=np.load(store.native_path(r['trace_id'],'hidden_readouts.npy'));hidden[i]=unbits(a).transpose(1,0,2)
        for l in sorted({l for l,u in units}):
            block=store.block(r['trace_id'],l)
            for j,(ul,u) in enumerate(units):
                if ul==l:z[i,:,j]=np.stack([unbits(block[k][:,u]) for k in ['gate','value','activation']],axis=-1)
            if r['split']=='discovery':rms[l]+=unbits(block['activation'][-1])**2
        if r['split']=='discovery':ntrain+=1
        if (i+1)%128==0:status('discovery','running',done=i+1,total=len(rows))
    pairs={};idx={r['id']:i for i,r in enumerate(rows)}
    for split in ['discovery','confirm','lockbox']:
        for family in P.FAMILIES:
            positives=[r for r in rows if r['split']==split and r['family']==family and r['semantic']==1]
            pairs[(split,family)]=[(idx[r['id']],idx[r['id'].rsplit(':',1)[0]+':0']) for r in positives]
    # Freeze signs and strictly matched randoms before accessing heldout differences.
    signs={};rng=np.random.default_rng(P.SEED);random=[];matches=[]
    from server.mechanism_trace_service import parameter
    for family in P.FAMILIES:
        ul=P.FAMILIES[family]['unit'];j=units.index(ul);pp=pairs[('discovery',family)];diff=np.stack([z[a,2,j]-z[b,2,j] for a,b in pp]);signs[family]=np.where(diff.mean(0)>=0,1,-1).tolist()
    for l,u in units:
        a=np.sqrt(rms[l]/ntrain);w=parameter(P.MODEL,f'model.layers.{l}.mlp.down_proj.weight',(slice(None),slice(None)));norm=np.linalg.norm(w,axis=0);ratio=a*norm
        cand=[v for v in range(9728) if v!=u and .8<=a[v]/max(a[u],1e-10)<=1.25 and .8<=norm[v]/max(norm[u],1e-10)<=1.25 and .8<=ratio[v]/max(ratio[u],1e-10)<=1.25];assert cand,(l,u)
        ru=int(rng.choice(cand));random.append([l,ru]);matches.append({'layer':l,'target':u,'random':ru,'A_RMS_ratio':float(a[ru]/a[u]),'down_norm_ratio':float(norm[ru]/norm[u]),'product_ratio':float(ratio[ru]/ratio[u])})
    P.save(P.OUT/'selection.json',{'kind':'real','run_id':P.RUN,'units':units,'random_units':random,'matching':matches,'signs':signs,'fit_split':'discovery only','provenance':obs['provenance']})
    cells=[]
    for split in ['confirm','lockbox']:
        for family in P.FAMILIES:
            rr=[r for r in rows if r['split']==split and r['family']==family]
            for cue in [0,1]:
                for frame in [0,1]:
                    for semantic in [0,1]:
                        sub=[r for r in rr if (r['cue_form'],r['frame'],r['semantic'])==(cue,frame,semantic)];cells.append({'split':split,'family':family,'cue_form':cue,'frame':frame,'semantic':semantic,'n':len(sub),'accuracy':float(np.mean([r['correct'] for r in sub]))})
    findings=[];errors=[];profiles={}
    for family in P.FAMILIES:
        j=units.index(P.FAMILIES[family]['unit']);results={};gate=all(r['accuracy']>=.9 and r['n']>=32 for r in cells if r['family']==family)
        train=pairs[('discovery',family)];direction=np.mean([hidden[a,2,35]-hidden[b,2,35] for a,b in train],axis=0)
        for split in ['confirm','lockbox']:
            pp=pairs[(split,family)];s={}
            for k,name in enumerate(['G','V','A']):
                diff=[z[a,2,j,k]-z[b,2,j,k] for a,b in pp];m=score(diff,signs[family][k],None);m['passed']=bool(m['accuracy']>=.65 and m['p']<.05/4 and m['nonzero_fraction']>=.8);s[name]=m
            s['hidden_pair_direction']={'accuracy':float(np.mean([(hidden[a,2,35]-hidden[b,2,35])@direction>0 for a,b in pp])),'scope':'exploratory supervised response-direction comparator, not SAE or autonomous output prediction'}
            s['early_prefix_max_abs']=float(max(np.max(np.abs(hidden[a,0]-hidden[b,0])) for a,b in pp));s['final_embedding_max_abs']=float(max(np.max(np.abs(hidden[a,2,0]-hidden[b,2,0])) for a,b in pp))
            for a,b in pp:
                diff=z[a,2,j,2]-z[b,2,j,2]
                if diff*signs[family][2]<=0:errors.append({'family':family,'split':split,'positive_id':rows[a]['id'],'negative_id':rows[b]['id'],'positive_trace':rows[a]['trace_id'],'negative_trace':rows[b]['trace_id'],'A_delta':float(diff),'reason':'native_A_order_not_predicted'})
            results[split]=s
        findings.append({'family':family,'title':P.FAMILIES[family]['title'],'unit':P.FAMILIES[family]['unit'],'behavior_gate':gate,'results':results,
            'supported':bool(gate and results['confirm']['A']['passed'] and results['lockbox']['A']['passed'])})
    np.savez(P.OUT/'selected-fields.npz',GVA=z,hidden=hidden)
    result={'kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'findings':findings,'cells':cells,'counterexamples':errors,'signs':signs,
        'n_inputs':len(rows),'supported_families':sum(r['supported'] for r in findings),'negative_control_early_pass':all(r['results'][s]['early_prefix_max_abs']==0 and r['results'][s]['final_embedding_max_abs']==0 for r in findings for s in ['confirm','lockbox']),
        'provenance':obs['provenance']}
    P.save(P.OUT/'analysis.json',result);status('prediction','done' if result['supported_families'] else 'not_supported',findings=findings)
    print(json.dumps({'supported_families':result['supported_families'],'negative_controls':result['negative_control_early_pass']}))
if __name__=='__main__':main()
