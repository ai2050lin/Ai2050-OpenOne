"""Enlarged full-parameter gradient compatibility and matched deployment controls."""
import argparse
from rdc_formation_common import *
from phase2747_rdc_followup_contract import freeze
from phase2747_rdc_training import prepare_gradients, mean_gradient, evaluate, CONDITIONS
from rdc_formation_direction import ExactDirection, variants, NAMES


def gradients():
    import torch
    from rdc_native_tail import cuda_singleton
    cuda_singleton({'phase2747_rdc_training.py','phase2747_rdc_gradient_radius.py'})
    if (OUT/'gradient/result.json').exists():return
    assert read(OUT/'training/result.json')['all_passed']
    protocol,data=freeze()
    start=time.monotonic()
    model=None
    handles=[]
    try:
        model,tok=load('qwen4',OUT/'gradient')
        with np.load(FIELDS/'material/vocabulary.npz') as z:
            classes=torch.tensor(z['classes'].astype(np.int64),device='cuda')
        target,original,handles=prepare_gradients(model)
        params=list(target.parameters())
        names=list(dict(target.named_parameters()))
        records=[]
        vectors=[]
        for condition in CONDITIONS:
            recpath=OUT/'gradient/commits'/(condition+'.json')
            summarypath=OUT/'gradient'/(condition+'.json')
            if recpath.exists() and summarypath.exists():
                rec=read(recpath)
                with np.load(ROOT/rec['field_path']) as z:
                    vectors.append([z[n].copy() for n in names])
                records.append(read(summarypath))
                continue
            accumulator=[torch.zeros_like(p) for p in params]
            losses=[]
            for begin in range(0,256,16):
                rows=data['gradient'][begin:begin+16]
                gg,loss=mean_gradient(model,rows,condition,classes,params)
                for acc,g in zip(accumulator,gg):acc.add_(g,alpha=1/16)
                losses.append(loss)
                del gg
                print('FORMATION_FULL_GRADIENT',condition,begin+16,256,round(time.monotonic()-start,2),flush=True)
            arrays={n:g.detach().cpu().numpy() for n,g in zip(names,accumulator)}
            receipt=commit_array('gradient',condition,**arrays)
            norm=float(np.sqrt(sum(np.sum(v.astype(np.float64)**2) for v in arrays.values())))
            record={'condition':condition,'examples':256,'objective':float(np.mean(losses)),
                    'all_parameter_gradient_norm':norm,'field':receipt}
            save(summarypath,record)
            records.append(record)
            vectors.append([arrays[n].copy() for n in names])
            del accumulator,arrays
        norms=[r['all_parameter_gradient_norm'] for r in records]
        cosine=[]
        for i,a in enumerate(vectors):
            cosine.append([float(sum(np.sum(x.astype(np.float64)*y.astype(np.float64)) for x,y in zip(a,b)))/(norms[i]*norms[j])
                           for j,b in enumerate(vectors)])
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'records':records,
            'examples':256,'source_groups':protocol['gradient_panel']['source_groups'],'parameters':74711040,
            'full_parameter_gradient_cosine':cosine,'conditions':CONDITIONS,'seconds':time.monotonic()-start,
            'scope':'Fullparameter vectors, enlarged proportional training subset. Cosines are geometry at the declared FP32bridge, not a semantic-module identity or source-population confidence interval.'}
        save(OUT/'gradient/result.json',result)
        ledger('phase2747_enlarged_gradients',result['seconds'])
    except Exception as exc:
        failure(OUT/'gradient',start,exc)
        raise
    finally:
        for h in handles:h.remove()
        if model is not None:del model
        gc.collect()
        torch.cuda.empty_cache()


def radius(pilot=False):
    import torch
    from rdc_native_tail import cuda_singleton
    cuda_singleton({'phase2747_rdc_training.py','phase2747_rdc_gradient_radius.py'})
    assert read(OUT/'training/result.json')['all_passed']
    start=time.monotonic()
    folder=OUT/'radius'
    result_path=folder/('pilot.json' if pilot else 'result.json')
    if result_path.exists():return
    if not pilot:assert read(folder/'pilot.json')['all_passed']
    data=gzread(OUT/'material/rows.json.gz')
    rows=data['validation']+data['diagnostic']+data['fresh']
    selected=variants()
    if pilot:
        selected=[v for v in selected if v['seed']==2747 and v['radius_factor']==.5
                  and (v['transform']!='identity' or v['condition']=='true_token')]
        assert len(selected)==6
    else:
        path=folder/'protocol.json'
        if not path.exists():immutable(path,{'timestamp':stamp(),'source':snapshot(__file__),'direction_source':snapshot(Path(__file__).with_name('rdc_formation_direction.py')),
            'variants':selected,'positions':896,'field_axes':'All fullpostnorm coordinates and fullvocabulary statistics; originalparameter references plus exact source delta/transform/scale reconstruct every deployment.',
            'scope':'Actual two-precision radius matching, not claimed identical per-coordinate or per-matrix geometry.'})
    model=None
    manager=None
    try:
        model,tok=load('qwen4',folder/('pilot' if pilot else 'main'))
        with np.load(FIELDS/'material/vocabulary.npz') as z:classes=torch.tensor(z['classes'].astype(np.int64),device='cuda')
        baseline=evaluate(model,rows[:4],classes)
        with np.load(FIELDS/'training/baseline/native.npz') as z:
            assert all(np.array_equal(baseline[k],z[k][:4]) for k in baseline)
        manager=ExactDirection(model)
        records=[]
        with torch.no_grad():
            for v in selected:
                path=folder/v['name']/'result.json'
                if not pilot and path.exists():
                    records.append(read(path))
                    continue
                tick=time.monotonic()
                match=manager.match(v)
                packet=evaluate(model,rows[:4] if pilot else rows,classes)
                manager.restore()
                reset=evaluate(model,rows[:4],classes)
                assert all(np.array_equal(reset[k],baseline[k]) for k in reset)
                record={'timestamp':stamp(),'all_passed':True,'variant':v,'matching':match,
                        'original_native_restored_exactly':True,'seconds':time.monotonic()-tick}
                if not pilot:
                    record['evaluation']=commit_array('radius',v['name'],**packet)
                    save(path,record)
                    save(folder/'progress.json',{'timestamp':stamp(),'completed':len(records)+1,'total':40,'latest':v['name']})
                else:record['pilot_NLL']=packet['NLL'].tolist()
                records.append(record)
                print('FORMATION_RADIUS',pilot,v['name'],match['actual_radius'],round(record['seconds'],2),flush=True)
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'records':records,
            'variants':len(records),'positions':4 if pilot else 896,'seconds':time.monotonic()-start,
            'scope':'All actualscalar radius/precision/direction conditions preserved, including reversals/permutations; no outcome-selected radius.'}
        save(result_path,result)
        ledger('phase2747_radius_'+('pilot' if pilot else 'main'),result['seconds'])
    except Exception as exc:
        failure(folder,start,exc)
        raise
    finally:
        if manager is not None:manager.close()
        if model is not None:del model
        del manager
        gc.collect()
        torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--gradients',action='store_true')
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    if args.gradients:gradients()
    else:radius(args.pilot)
