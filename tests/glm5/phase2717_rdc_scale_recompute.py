"""CPU-only reconstruction of every scale fit from full retained native states and weights."""
import hashlib
from rdc_relation_common import *
from rdc_relation_native_parameters import parameter,decode
from phase2717_rdc_relation_scale import MODELS,dots,kernel
from rdc_relation_estimators import select,predict,Bank


def main():
    reports=[];start=time.monotonic()
    for key,modelname in MODELS.items():
        out=BASE/'scale'/key;freeze=read(out/'frozen.json');result=read(out/'result.json');em=parameter(ROOT,'model.embed_tokens.weight',modelname);metadata=freeze['main_rows'];freshmeta=[]
        for sid in freeze['fresh_ids']:
            m=read(out/f'rows/{sid}.json')
            for j in (0,1):freshmeta.append({'sample_id':sid,'anchor':j,'position':m['positions'][2*j],'next_position':m['positions'][2*j+1]})
        def pack(mm):
            raw={k:[] for k in ('early','late','new_embedding','next_late')}
            for r in mm:
                m=read(out/f'rows/{r["sample_id"]}.json');j=r['anchor'];tid=m['prompt_ids'][r['next_position']]
                if key=='qwen4':
                    with np.load(BASE/m['origin']/f'fields/{r["sample_id"]}.npz') as z:early=decode(z['h12'][r['position']]);late=decode(z['h36'][3*j]);later=decode(z['h36'][3*j+1])
                else:
                    with np.load(out/f'fields/{r["sample_id"]}.npz') as z:early=decode(z['early'][2*j]);late=decode(z['late'][2*j]);later=decode(z['late'][2*j+1])
                for name,value in [('early',early),('late',late),('next_late',later),('new_embedding',decode(em[tid]))]:raw[name].append(value)
            return {k:np.stack(v) for k,v in raw.items()}
        raw=pack(metadata);fresh=pack(freshmeta);tr=np.array(freeze['training_indices']);va=np.array([i for i,r in enumerate(metadata) if r['split']=='validation']);dd=dots(raw,raw,freeze['scales']);cross=dots(fresh,{k:v[tr] for k,v in raw.items()},freeze['scales'])
        for name,chosen in freeze['choices'].items():
            kd,kind=kernel(dd,name);y=raw['next_late'] if name.startswith('temporal') else raw['late'];model,selection,grid=select(kd,kind,y,tr,va,[(0,y.shape[1])]);assert selection['mix']==chosen['mix'] and selection['ridge']==chosen['ridge']
            digests={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in model.items()};matches={k:digests[k]==freeze['model_array_sha'][name][k] for k in model};report={'model':key,'route':name,'all_fitted_array_hash_equal':all(matches.values()),'hash_equal_by_array':matches,'validation_loss_abs_difference':abs(selection['validation_normalized_MSE']-chosen['validation_normalized_MSE'])}
            if name in result['selected_before_fresh'].values():
                kd,kind=kernel(cross,name);p=predict(model,Bank.gram(kd,kind,chosen['mix']))
                with np.load(out/f'fresh_{name}.npz') as z:error=float(np.max(np.abs(p-z['prediction'])))
                assert error<1e-3,(key,name,error);report['full_fresh_prediction_max_abs_difference']=error
            assert report['validation_loss_abs_difference']<1e-9;reports.append(report);print('SCALE_CPU_RECONSTRUCT',report,flush=True)
    save(BASE/'verification/scale_recomputation.json',{'timestamp':stamp(),'passed':True,'reports':reports,'source':snapshot(Path(__file__)),'parameter_source':snapshot(ROOT/'tests/glm5/rdc_relation_native_parameters.py'),'elapsed_seconds':time.monotonic()-start,
      'scope':'No original model forward or GPU allocation. Every fitted coefficient reconstructed from retained full native states and exact BF16 incoming embeddings. Hash identity and selected full-fresh prediction numerical identity reported separately.'})


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
