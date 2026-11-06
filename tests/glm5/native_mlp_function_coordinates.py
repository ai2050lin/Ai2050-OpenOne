"""Prospective second engine: discover native coordinates, then freeze before heldouts."""
import sys,time,json
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from shared.dist_protocol.mechanism_trace import digest,file_sha,unbits
from tools.power_analysis import check_phase

def fields(item):
    folder=P.OUT/'native'/digest(item['id'])[:20];path=folder/'manifest.json'
    while not path.exists():time.sleep(2)
    result=[]
    for l in range(36):
        with np.load(folder/f'block_{l:02d}.npz') as z:result.append(unbits(z['activation'][2]))
    return np.stack(result)
def metric(values,sign):
    x=np.asarray(values)*sign;n=int(np.count_nonzero(x));wins=int(np.sum(x>0));acc=float(np.mean(x>0)+.5*np.mean(x==0))
    return {'n':len(x),'unequal_fraction':n/len(x),'accuracy':acc,'p':float(binomtest(wins,n,.5,alternative='greater').pvalue) if n else 1.,'passed':bool(acc>=.65 and n/len(x)>=.8 and binomtest(wins,n,.5,alternative='greater').pvalue<.0125/6) if n else False}
def main():
    c=P.read(P.OUT/'corpus.json');items={r['id']:r for r in c['items']};rng=np.random.default_rng(P.SEED+25)
    contract={'schema':'research_coordinate_extraction_prereg.v1','kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'script_sha256':file_sha(__file__),
        'method':'For eachfamily, all350208 native A coordinates atfully-visible-question readout. Discovery paired mean / paired RMS; nonzero>=.8 andrawRMS>=.01. Pick strongest2 distinctlayers, layers>=6; sign frozen fromdiscovery. Entire arrays retained, Top2 is candidate selection only.',
        'multiplicity':'3 preregistered primary units use3*(.05/4)=.0375; allocate unused .0125 family budget to6 newcoordinate tests, alpha .0125/6 each. Both heldouts required; no primary threshold changed.',
        'scope':'new discovery-driven candidate branch, not claimed semantic function or cause; Yes/No answer polarity remains a competing interpretation',
        'power':check_phase(.25,256,p_d=.5,alpha=.0125/6),'declared_before_heldout_coordinate_selection':True}
    assert contract['power']['pass'];P.save(P.PUBLIC/'coordinate-prereg.research.json',contract);selected=[];nulls=[]
    for family in P.FAMILIES:
        rr=[r for r in c['items'] if r['family']==family and r['split']=='discovery' and r['semantic']==1]
        differences=[];raws=[]
        for a in rr:
            b=items[a['id'].rsplit(':',1)[0]+':0'];fa=fields(a);fb=fields(b);differences.append(fa-fb);raws.extend([fa,fb])
        x=np.stack(differences);rms=np.sqrt(np.mean(np.stack(raws)**2,0));score=np.abs(x.mean(0))/np.sqrt(np.mean(x*x,0)+1e-12)
        score[(np.mean(x!=0,0)<.8)|(rms<.01)]=0;score[:6]=0
        layers=set()
        for k in np.argsort(-score.reshape(-1),kind='stable'):
            l,u=divmod(int(k),9728)
            if l in layers:continue
            selected.append({'family':family,'layer':l,'unit':u,'sign':1 if x[:,l,u].mean()>=0 else -1,'discovery_score':float(score[l,u]),'discovery_order':metric(x[:,l,u],1 if x[:,l,u].mean()>=0 else -1)})
            layers.add(l)
            if len(layers)==2:break
        # A heldout-shuffle null is fixed independently before reading heldout activations.
        nulls.append({'family':family,'confirm_signs':rng.choice([-1,1],256).tolist(),'lockbox_signs':rng.choice([-1,1],256).tolist()})
    P.save(P.OUT/'coordinate-selection.json',{'kind':'real','run_id':P.RUN,'selected':selected,'nulls':nulls,'fit_split':'discovery only','contract_sha256':file_sha(P.PUBLIC/'coordinate-prereg.research.json')})
    while not (P.OUT/'observations.json').exists():time.sleep(2)
    results=[]
    for candidate in selected:
        evaluations={};null=next(r for r in nulls if r['family']==candidate['family'])
        for family in P.FAMILIES:
            for split in ['confirm','lockbox']:
                rr=[r for r in c['items'] if r['family']==family and r['split']==split and r['semantic']==1];delta=[]
                for a in rr:
                    b=items[a['id'].rsplit(':',1)[0]+':0'];folder=P.OUT/'native'
                    with np.load(folder/digest(a['id'])[:20]/f'block_{candidate["layer"]:02d}.npz') as za,np.load(folder/digest(b['id'])[:20]/f'block_{candidate["layer"]:02d}.npz') as zb:
                        delta.append(float(unbits(za['activation'][2,candidate['unit']]))-float(unbits(zb['activation'][2,candidate['unit']])))
                evaluations[family+':'+split]=metric(delta,candidate['sign'])
                if family==candidate['family']:evaluations['shuffle:'+split]=metric(np.asarray(delta)*null[split+'_signs'],candidate['sign'])
        candidate={**candidate,'evaluations':evaluations,'heldout_response_supported':all(evaluations[candidate['family']+':'+s]['passed'] for s in ['confirm','lockbox'])};results.append(candidate)
    obs=P.read(P.OUT/'observations.json');P.save(P.OUT/'coordinate-results.json',{'kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'candidates':results,'provenance':obs['provenance']})
    print(json.dumps({'candidates':len(results),'heldout_response_supported':sum(r['heldout_response_supported'] for r in results)}),flush=True)
if __name__=='__main__':main()
