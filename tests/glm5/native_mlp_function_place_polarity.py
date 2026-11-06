"""Prospective geographic-place vs portable-object answer/fact control."""
import sys,json,time,types
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
import native_mlp_function_polarity as B
import native_mlp_long_materials as OLD
from native_mlp_function_run_v4 import status
from shared.dist_protocol.mechanism_trace import digest,file_sha,save_json
from tools.power_analysis import check_phase
SUB=P.OUT/'place-polarity';PUB=P.PUBLIC/'place-polarity'
def prepare():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);raw=np.load(OLD.OUT/'input-tokens.npy',mmap_mode='r');rng=np.random.default_rng(P.SEED);indices=rng.permutation(2048)[672:800]
    prefix=tok.encode('<|im_start|>user\n',add_special_tokens=False);end=tok.encode('\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
    assignments={}
    for cue in [0,1]:
        for frame in [0,1]:
            ids=[i for i in range(128) if i%2==cue and (i//2)%2==frame];levels=rng.permutation([(a,b) for _ in range(8) for a in [0,1] for b in [0,1]])
            for i,(key,name) in zip(ids,levels):assignments[i]=(int(key),int(name))
    items=[]
    for k,di in enumerate(indices):
        pk,ns=assignments[k];positive_key='K' if pk else 'M';posname=f'revalo{k}_'+('cobalt' if ns else 'umber');negname=f'revalo{k}_'+('umber' if ns else 'cobalt');cue=k%2;frame=(k//2)%2
        pos=[f'{posname} is a small geographic town where the market is hosted.',f'{posname} denotes a settlement on a northern coast.'][cue]
        neg=[f'{negname} is a portable object engraved "town", a hand tool used at the market.',f'{negname} denotes a portable instrument kept beside the coast.'][cue]
        records={'K':pos if positive_key=='K' else neg,'M':pos if positive_key=='M' else neg};common='\n\nUse these invented definitions only. The referents are distinct.\nEntry K: '+records['K']+'\nEntry M: '+records['M']+'\n';variants={}
        for fact in [0,1]:
            selected=positive_key if fact else ('M' if positive_key=='K' else 'K')
            for q in [0,1]:variants[fact,q]=tok.encode(common+f'Does Entry {selected} define a {"geographic place" if q else "portable object"}? Reply only Yes or No.\nAnswer:',add_special_tokens=False)
        assert len(set(len(v) for v in variants.values()))==1
        bg=1024-len(prefix)-len(next(iter(variants.values())));assert bg>=512
        for (fact,q),tail in variants.items():
            ids=prefix+raw[di,:bg].tolist()+tail;expected='Yes' if fact==q else 'No'
            items.append({'id':f'place-polarity:{k}:{fact}:{q}','quad_id':f'place-polarity:{k}','pair_id':f'place-polarity:{k}:{q}','fact':fact,'question_polarity':q,'cue_form':cue,'frame':frame,'expected':expected,'contrast':'No' if expected=='Yes' else 'Yes',
                'selected_key':selected if False else (positive_key if fact else ('M' if positive_key=='K' else 'K')),'name_suffix_assignment':ns,'positive_key':positive_key,
                'split':'fresh_place_polarity','background_doc_index':int(di),'token_ids':ids+end,'readout_positions':[1023,1023+len(end)],'text':tok.decode(ids),'prompt':tok.decode(ids+end),'entity':f'revalo{k}','scope':'地点/可携物×提问方向×名字独立平衡'})
    coords=P.read(P.OUT/'coordinate-selection.json')['selected'];primary=P.read(P.OUT/'selection.json')['signs']['location'][2];units=coords+[{'family':'location','layer':24,'unit':8443,'sign':primary,'source':'preselected natural hypothesis'}]
    contract={'schema':'research_place_polarity_prereg.v1','kind':'real','run_id':P.RUN,'model_snapshot':P.read(P.OUT/'corpus.json')['model_snapshot'],'n_independent_quads':128,'n_inputs':512,'hypotheses':units,
        'primary':'Freeze previous7unit signs; atquestion andassistant boundaries distinguish stable fact-order from answer-order when geographic vsportable question flips desiredanswer; no refit.14tests alpha .05/14, minaccuracy .75, nonzero>=.8.',
        'behavior':'all cue x frame x fact x question cells n32, accuracy>=.9. Missingbehavior => no functional confirmation.',
        'names':'Both cobalt/umber referent suffixes independently balanced with fact, key identity, cue andframe; randomized balanced allocation. Do not mistake the prior Amber/Birch fact-name correlation for semantics.',
        'source_scope':'A new prospective diagnostic after the main location behavior gate passed; no original primary threshold changed; not a retest on main lockbox',
        'power':check_phase(.3,128,p_d=.5,alpha=.05/14),'script_sha256':file_sha(__file__),'base_runner_sha256':file_sha(B.__file__),'secondary_coordinate_selection_sha256':file_sha(P.OUT/'coordinate-selection.json')}
    assert contract['power']['pass'];P.save(PUB/'prereg.research.json',contract);P.save(SUB/'polarity-corpus.json',{'kind':'real','run_id':P.RUN,'items':items,'model_snapshot':contract['model_snapshot']});P.save(SUB/'coordinate-selection.json',{'kind':'real','run_id':P.RUN,'selected':units})
    resource={'schema':'research_resource_amendment.v1','kind':'real','run_id':P.RUN,'model_snapshot':contract['model_snapshot'],'revised_max_GiB':110,'reason':'Two independent128quad polarity controls each512inputs, full all36 module arrays attwo declared readouts. Retained before-model resource amendment, not a relaxation of scientific criteria.'};P.save(PUB/'resource-amendment.research.json',resource)
    print(json.dumps({'place_polarity_inputs':512,'units':7,'power':contract['power']}),flush=True)
def run():
    class ProperRecorder(B.Recorder):
        def finish(self,*a,**kw):
            d,p=super().finish(*a,**kw);d['causal_visibility']='Both declared readouts follow the complete question; the first ends atAnswercolon, the second atassistant input boundary. No generated answer is present.'
            d['response_profile_stage_indices']=[2,3];d['readout_annotations']=[{'position':p,'role':role} for p,role in zip(d['readout_positions'],['question_end','assistant_boundary'])];d.pop('trace_id');d['trace_id']=digest(d);save_json(p,d);return d,p
    def read(path):
        path=Path(path)
        if path.name=='observations.json':return P.read(P.OUT/'observations.json')
        return P.read(path)
    def metric(x,sign):
        x=np.asarray(x)*sign;n=int(np.count_nonzero(x));wins=int(np.sum(x>0));acc=float(np.mean(x>0)+.5*np.mean(x==0));p=float(binomtest(wins,n,.5,alternative='greater').pvalue) if n else 1.
        return {'accuracy':acc,'p':p,'unequal_fraction':n/len(x),'passed':bool(acc>=.75 and p<.05/14 and n/len(x)>=.8)}
    B.P=types.SimpleNamespace(OUT=SUB,PUBLIC=PUB,ROOT=P.ROOT,RUN=P.RUN,MODEL=P.MODEL,SEED=P.SEED,read=read,save=P.save);B.Recorder=ProperRecorder;B.metric=metric;B.status=lambda stage,state='done',**kw:status('place_polarity',state,**kw)
    # Base diagnostic uses its own prefix in saved identifiers; adapt lookup only, not model data.
    original=SUB/'polarity-corpus.json';c=read(original);mapping={r['id']:r['id'].replace('place-polarity:','polarity:') for r in c['items']}
    # Run's grouping convention uses polarity:<k>; unique native paths are placed under this subdirectory.
    for r in c['items']:r['id']=mapping[r['id']];r['quad_id']=r['quad_id'].replace('place-polarity:','polarity:');r['pair_id']=r['pair_id'].replace('place-polarity:','polarity:')
    adapter=SUB/'execution-corpus.json';P.save(adapter,c)
    def adapted(path):return P.read(adapter) if Path(path)==original else read(path)
    B.P.read=adapted;B.run()
if __name__=='__main__':
    prepare()
    while not (P.OUT/'polarity-result.json').exists():time.sleep(5)
    run()
