"""All350208 native coordinates, frozen primary transfer rule and complete negative results."""
import sys,json,time,sqlite3,itertools,gc
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_atlas_code_materials_v2 as P
from mlp_atlas_code_pilot_v2 import status
from shared.dist_protocol.mechanism_trace import unbits,file_sha
from shared.dist_protocol.native_mlp_atlas import decision_contrast,fit_sign,evaluate,factorial_coefficients,prediction_fvu

def main():
    folder=P.OUT/'bank';man=P.read(folder/'manifest.json');assert man['state']=='complete' and man['completed']==9984
    plan=P.read(P.PUBLIC/'analysis-prereg.research.json');pr=P.read(P.PUBLIC/'prereg.research.json');assert file_sha(P.OUT/'bank-execution-freeze.json')==plan['bank_collection_contract_sha256']
    import shared.dist_protocol.native_mlp_atlas as math
    P.save(P.OUT/'analysis-execution-freeze.json',{'kind':'real','run_id':P.RUN,'provenance':man['provenance'],'model_snapshot':man['model_snapshot'],'bank_manifest_sha256':file_sha(folder/'manifest.json'),
        'analysis_plan_sha256':file_sha(P.PUBLIC/'analysis-prereg.research.json'),'dependencies':{str(Path(n).relative_to(P.ROOT)):file_sha(n) for n in [Path(__file__).resolve(),Path(math.__file__).resolve()]}})
    db=sqlite3.connect(folder/'samples.sqlite3');db.row_factory=sqlite3.Row
    keys=['id','background_index','split','code','mapping','fact','question','decision','frame','correct','choice_gap','before_selector_hidden_sha256','question_embedding_sha256']
    select=','.join("json_extract(body,'$."+k+"') AS "+k for k in keys);rows=[dict(r) for r in db.execute('SELECT idx,'+select+' FROM samples ORDER BY idx')];db.close();assert len(rows)==9984
    codes=['latin_ab','latin_cd','chinese_neutral'];lookup={}
    for i,r in enumerate(rows):
        assert r['idx']==i and r['background_index']==i//24;key=(r['background_index'],r['code'],r['mapping'],r['fact'],r['question']);assert key not in lookup;lookup[key]=r
    for b in range(416):
        for code,m,q in itertools.product(codes,[0,1],[0,1]):
            x,y=lookup[b,code,m,0,q],lookup[b,code,m,1,q]
            assert x['before_selector_hidden_sha256']==y['before_selector_hidden_sha256'];assert x['question_embedding_sha256']==y['question_embedding_sha256']
    cells=[];behavior=True
    for key in sorted({tuple(r[k] for k in ['split','code','mapping','fact','question','frame']) for r in rows if r['split']!='discovery'}):
        rr=[r for r in rows if tuple(r[k] for k in ['split','code','mapping','fact','question','frame'])==key];assert len(rr)==96
        acc=sum(r['correct'] for r in rr)/len(rr);passed=acc>=.9;behavior=behavior and passed;cells.append(dict(zip(['split','code','mapping','fact','question','frame'],key))|{'n':len(rr),'accuracy':acc,'passed':passed})
    summaries=[];error_examples=[];total=np.zeros((2,),int);eligible=np.zeros((2,),int);nullcount=np.zeros((2,),int);unique_response=0;unique_eligible=0;start=time.time();out=P.OUT/'analysis';out.mkdir(exist_ok=True)
    status('analysis','running',blocks_done=0,total_blocks=36,behavior_gate=behavior)
    for block in range(36):
        # Native file order is frozen background→code→mapping→fact→question.
        native=np.load(folder/f'B{block:02d}-A.npy',mmap_mode='r');x=unbits(native).reshape(416,3,2,2,2,2,9728);disc=x[:32]
        delta=decision_contrast(disc);sign=fit_sign(delta,0);saved={'sign':sign};split_summary={};joint=np.ones((2,9728),bool);nulljoint=np.ones_like(joint);factor_means=[];factor_rms=[]
        for split,z in [('discovery',disc),('confirm',x[32:224]),('lockbox',x[224:])]:
            names,means,rms=factorial_coefficients(z);factor_means.append(means);factor_rms.append(rms)
            if split=='discovery':continue
            dd=decision_contrast(z);res=evaluate(dd,sign,alpha=pr['criteria']['alpha_per_unit_readout']);joint &=res['all_components_passed']
            for key in ['successes','nonzero','p','passed']:saved[split+'_'+key]=res[key]
            pred=prediction_fvu(disc,z)
            for key,v in pred.items():saved[split+'_'+key]=v
            rng=np.random.default_rng(P.SEED+1000+block+(0 if split=='confirm' else 100))
            flips=np.ones((len(z),3,2),np.float32);flips[:len(z)//2]=-1
            for code in range(3):
                for mapping in range(2):rng.shuffle(flips[:,code,mapping])
            null=evaluate(dd*flips[:,:,:,None,None],sign,alpha=pr['criteria']['alpha_per_unit_readout']);nulljoint &=null['all_components_passed'];saved[split+'_shuffled_passed']=null['passed']
            split_summary[split]={'component_counts':res['passed'].sum(-1).tolist(),'all_components':res['all_components_passed'].sum(-1).tolist(),'median_fvu_by_readout':np.median(pred['fvu'],axis=-1).tolist(),
                'zero_variance_units_by_readout':pred['zero_variance'].sum(-1).tolist(),'fvu_lt_half_units_by_readout':(pred['fvu']<.5).sum(-1).tolist()}
            del dd,res,pred,null,z
        saved.update(response_replicated=joint,eligible_supported=joint&behavior,shuffled_replicated=nulljoint,factorial_mean=np.stack(factor_means),factorial_RMS=np.stack(factor_rms),factorial_names=np.asarray(names))
        p=out/f'B{block:02d}.npz';np.savez_compressed(p,**saved);total+=joint.sum(-1);eligible+=(joint&behavior).sum(-1);nullcount+=nulljoint.sum(-1);unique_response+=int(joint.any(0).sum());unique_eligible+=int((joint&behavior).any(0).sum())
        summaries.append({'block':block,'response_replicated_by_readout':joint.sum(-1).tolist(),'eligible_supported_by_readout':(joint&behavior).sum(-1).tolist(),'shuffled_replicated_by_readout':nulljoint.sum(-1).tolist(),'splits':split_summary,'file':str(p.relative_to(P.ROOT)),'sha256':file_sha(p)})
        # A complete per-unit failure grid is saved. A few cases are merely an entry point.
        if block in [24,27,28,29,32,34,35]:
            for unit in sorted({8443 if block==24 else 0,4371 if block==28 else 0,5610 if block==32 else 0,3952 if block==32 else 0,8463 if block==29 else 0,446 if block==34 else 0,2658 if block==27 else 0}):
                for stage in [0,1]:
                    counts=saved['lockbox_successes'][:,:,stage,unit]
                    code,mapping=np.unravel_index(int(np.argmin(counts)),counts.shape)
                    zz=.5*((x[224:,code,mapping,0,0,stage,unit]-x[224:,code,mapping,1,0,stage,unit])+(x[224:,code,mapping,1,1,stage,unit]-x[224:,code,mapping,0,1,stage,unit]));bad=np.flatnonzero(zz*sign[stage,unit]<=0)
                    b=224+(int(bad[0]) if len(bad) else 0);error_examples.append({'block':block,'unit':unit,'stage':stage,'code':codes[code],'mapping':int(mapping),'n_correct_order':int(counts[code,mapping]),'n_backgrounds':192,
                        'sample_indices':[lookup[b,codes[code],int(mapping),t,q]['idx'] for q in [0,1] for t in [0,1]],'reason':'least-consistent heldout component; not a counterexample if everyorderpasses','response_replicated':bool(joint[stage,unit]),'eligible':bool(joint[stage,unit] and behavior)})
        status('analysis','running',blocks_done=block+1,total_blocks=36,response_replicated=total.tolist(),eligible_supported=eligible.tolist());del x,disc,delta,saved,factor_means,factor_rms;gc.collect()
    errors=[{k:r[k] for k in ['idx','id','split','code','mapping','fact','question','decision','choice_gap']} for r in rows if not r['correct']]
    report={'schema':'native_mlp_atlas_analysis.v1','kind':'real','run_id':P.RUN,'model_snapshot':man['model_snapshot'],'provenance':man['provenance'],'n_inputs':9984,'n_units':350208,'independent_backgrounds':416,'readout_roles':man['readout_roles'],
        'behavior_gate':behavior,'behavior_cells':cells,'behavior_errors':errors,'negative_controls':{'before_selector_all37hidden_exact':True,'final_question_embedding_exact':True,'heldout_shuffled_replicated_by_readout':nullcount.tolist()},
        'unique_response_units':unique_response,'unique_eligible_units':unique_eligible,'response_replicated_by_readout':total.tolist(),'eligible_supported_by_readout':eligible.tolist(),'layers':summaries,'counterexample_entry_points':error_examples,'seconds':time.time()-start,
        'scope':'Primary is scoped A-coordinate response invariance acrossallregisteredcode/mapping/split components. Not semanticmonosemanticity, individualnecessity, causalgating orgraph-onlynext-tokenprediction.',
        'power_clarification':P.read(P.PUBLIC/'power-clarification.research.json'),'prediction_scope':'FVU predictor receivesoperationalinputfactors andonlydiscoverycellmeans. It isadescriptiveupperbaseline, notamechanismextractor.'}
    P.save(P.OUT/'analysis-result.json',report);status('analysis','done',response_replicated=total.tolist(),eligible_supported=eligible.tolist(),behavior_gate=behavior,seconds=report['seconds']);print(json.dumps({k:report[k] for k in ['response_replicated_by_readout','eligible_supported_by_readout','behavior_gate','seconds']}),flush=True)
if __name__=='__main__':main()
