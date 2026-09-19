"""Original full-vocabulary BF16readout of every selected first-prefix predictor."""
import argparse
import inspect
from collections import Counter
from rdc_question_common import *
from rdc_question_fit import FeatureKernel
from rdc_formation_microbatch import TensorSliceReader
import rdc_question_data as data


def original_head(key):
    import torch
    from transformers import AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM
    from transformers.models.glm.modeling_glm import GlmForCausalLM
    folder=ROOT/'models/hf'/MODELS[key]
    config=read(folder/'config.json')
    mapping=read(folder/'model.safetensors.index.json')['weight_map']
    name='lm_head.weight' if 'lm_head.weight'in mapping else 'model.embed_tokens.weight'
    assert name=='lm_head.weight' or config['tie_word_embeddings'] is True
    assert 'lm_head.bias'not in mapping
    reader=TensorSliceReader()
    value=reader.tensor(folder/mapping[name],name)
    assert value.shape==(config['vocab_size'],config['hidden_size']) and value.dtype==torch.bfloat16
    word_sha=hashlib.sha256(memoryview(value.view(torch.uint8).numpy()).cast('B')).hexdigest()
    head=torch.nn.Linear(config['hidden_size'],config['vocab_size'],bias=False,device='meta',dtype=torch.bfloat16)
    head.weight=torch.nn.Parameter(value.to('cuda'),requires_grad=False)
    head.eval();del value
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True,use_fast=True,trust_remote_code=True)
    native=Qwen3ForCausalLM if config['model_type']=='qwen3'else GlmForCausalLM
    receipt={'model':key,'source_tensor':name,'source_shard':mapping[name],
        'source_shard_sha256':sha(folder/mapping[name]),'source_tensor_word_sha256':word_sha,
        'config_sha256':sha(folder/'config.json'),'index_sha256':sha(folder/'model.safetensors.index.json'),
        'native_architecture_source':snapshot(inspect.getfile(native)),
        'shape':list(head.weight.shape),'dtype':str(head.weight.dtype),'bias':None,
        'scope':'Only original native full-vocabulary unembedding loaded on CUDA for readout of already acquired fields. Q4head shares original embedding; no decoder layers/second model run concurrently. Everynative baseline score must replay exactly.'}
    return head,tok,receipt


def logprob(head,values):
    import torch
    with torch.inference_mode():
        value=torch.from_numpy(np.asarray(values,dtype=np.float64).copy()).to('cuda',dtype=torch.bfloat16)[None]
        assert bool(torch.isfinite(value).all())
        lp=head(value).float()[0].double().log_softmax(-1)
        return lp,bits(value[0])


def main(key,confirmation=False):
    import torch
    from rdc_native_tail import cuda_singleton
    from rdc_formation_readout import CUDA_FORMATION
    cuda_singleton(set(CUDA_FORMATION)|{'phase2748_rdc_acquire.py','phase2748_rdc_training.py',
        'phase2748_rdc_learning_pilot.py','phase2748_rdc_prospective.py','phase2748_rdc_learned.py','phase2748_rdc_readout.py'})
    start=time.monotonic()
    scope='confirmation'if confirmation else'nonconfirmation'
    folder=Path('fit')/key/'readout'/scope
    resultpath=OUT/folder/'result.json'
    if resultpath.exists():
        assert read(resultpath)['source']['sha256']==sha(__file__)
        print('NATURAL_READOUT_ALREADY_COMPLETE',key,scope,flush=True);return
    selection=read(OUT/'fit'/key/'validation_selection.json')
    fit=read(OUT/'fit'/key/'result.json');assert fit['all_passed']
    train,tg,tq=data.index(key,{'train'})
    splits=['confirmation']if confirmation else['validation','diagnostic']
    packs={split:data.index(key,{split})for split in splits}
    feature_train=data.native_features(key,train,tg,tq)
    target_train=data.targets(train,tq,'postnorm')
    feature_eval={split:data.native_features(key,*packs[split])for split in splits}
    target_eval={split:data.targets(packs[split][0],packs[split][2],'postnorm')for split in splits}
    weights=np.ones(len(train))/len(train)
    permutation=data.target_permutation(train)
    head=None
    try:
        head,tok,head_receipt=original_head(key)
        immutable(OUT/folder/'head.json',head_receipt)
        native_counts=Counter()
        checks=0
        for split in splits:
            rows,_,questions=packs[split]
            for row,actual in zip(rows,target_eval[split]):
                lp,cast=logprob(head,actual)
                expected=questions[row['question_id']]['statistics']
                choice=int(lp.argmax());p=lp.exp()
                assert choice==expected['argmax']
                assert float(-(p*lp).sum())==expected['entropy_FP64']
                assert float(p[choice])==expected['chosen_probability_FP64']
                assert float(-lp[row['tokens']['teacher_ids_including_EOS'][0]])==expected['first_teacher_token_NLL']
                checks+=1;native_counts[choice]+=1
                del lp,p
        print('NATURAL_READOUT_NATIVE_EXACT',key,checks,flush=True)
        records=[]
        for variant in fit['variants']:
            query,context=data.variant_features(key,train,feature_train,variant)
            feature=FeatureKernel(query,context,weights,variant=='lexical_position')
            yy=target_train[permutation]if variant=='within_context_target_pair_shuffle'else target_train
            mean=weights @ yy
            ev=read(OUT/'fit'/key/variant/'evaluation.json')
            for kind,choice in selection['variants'][variant].items():
                ref=ev['operator_fields'][kind]
                assert sha(ROOT/ref['path'])==ref['sha256']
                with np.load(ROOT/ref['path'])as z:op=z['solution_operator'].copy()
                coef=op @ (yy-mean)
                for split in splits:
                    path=OUT/folder/variant/(kind+'_'+split+'.json')
                    if path.exists():records.append(read(path));continue
                    rows,_,questions=packs[split]
                    part=feature.parts(*data.variant_features(key,rows,feature_eval[split],variant))
                    cross,_,_=feature.kernel(part,choice['rho'])
                    prediction=cross @ coef+mean
                    columns=['native_to_predicted_KL','native_top1_agreement','native_choice','predicted_choice',
                        'native_entropy','predicted_entropy','native_first_teacher_NLL','predicted_first_teacher_NLL',
                        'cast_full_coordinate_MSE','cast_maximum_coordinate_error','native_chosen_probability','predicted_chosen_probability']
                    statistics=[]
                    for row,pred,actual in zip(rows,prediction,target_eval[split]):
                        native,_=logprob(head,actual);predicted,cast=logprob(head,pred)
                        pn=native.exp();pp=predicted.exp();ni=int(native.argmax());pi=int(predicted.argmax())
                        first=row['tokens']['teacher_ids_including_EOS'][0]
                        error=unbits(cast).astype(float)-pred
                        kl=float((pn*(native-predicted)).sum());assert kl>=-1e-8
                        statistics.append([kl,int(ni==pi),ni,pi,float(-(pn*native).sum()),float(-(pp*predicted).sum()),
                            float(-native[first]),float(-predicted[first]),float(np.mean(error**2)),float(np.max(np.abs(error))),float(pn[ni]),float(pp[pi])])
                        del native,predicted,pn,pp
                    array=np.asarray(statistics,np.float64)
                    reference=commit_arrays(folder/variant,kind+'_'+split,{'statistics':array})
                    summary={}
                    for cohort in ['drop','quoref']:
                        take=np.array([r['cohort']==cohort for r in rows])
                        summary[cohort]={name:float(array[take,i].mean())for i,name in enumerate(columns)if name not in ['native_choice','predicted_choice']}
                    record={'variant':variant,'kind':kind,'split':split,'questions':len(rows),'columns':columns,
                        'question_ids':[r['question_id']for r in rows],'field':reference,'cohort_summary':summary,
                        'selection_sha256':sha(OUT/'fit'/key/'validation_selection.json'),
                        'qualification':'All saved native full-vocabulary summary values replay exactly with unchanged native unembeddingB1.'}
                    immutable(path,record);records.append(record)
                print('NATURAL_READOUT_VARIANT',key,variant,kind,round(time.monotonic()-start,1),flush=True)
        result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'model':key,'scope':scope,
            'native_exact_replayed_questions':checks,'head_receipt_sha256':sha(OUT/folder/'head.json'),
            'records':records,'native_first_choice_counts':[{'id':i,'decoded':tok.decode([i]),'count':n}for i,n in native_counts.items()],
            'seconds':time.monotonic()-start,
            'interpretation':'First-prefix full-vocabulary prediction only; common JSON/format-token agreement does not establish semantic answer prediction or own-history closure. Allvariants and alpha0controls retained.'}
        immutable(resultpath,result)
        print('NATURAL_READOUT_COMPLETE',key,scope,round(result['seconds'],1),flush=True)
    except Exception as exc:
        failure(OUT/folder,start,exc);raise
    finally:
        if head is not None:del head
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['qwen4','qwen14','glm4'],required=True)
    parser.add_argument('--confirmation',action='store_true')
    args=parser.parse_args();main(args.model,args.confirmation)
