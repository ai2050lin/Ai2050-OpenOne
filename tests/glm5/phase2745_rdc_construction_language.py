"""Own-history diagnostic after discovering GLM's native leading newline."""
import argparse
from collections import defaultdict
from rdc_construction_common import *
from phase2744_rdc_query_identifiability import language_score, language_checks

OUT = BASE / 'native_language'


def freeze_language():
    path = OUT / 'protocol.json'
    if path.exists():
        return read(path)
    value = {'timestamp': stamp(), 'source': snapshot(__file__),
        'timing': 'Secondary diagnostic added after allGLM320B1firsttokens were observed to be newline198, before new Q14/GLM own-history outcomes.',
        'reason': 'A first-format token cannot be treated as evidence of absence of later answer ability. Preserve originalB1metrics unchanged and add full same-panel behavior.',
        'material': 'Same320expressions, all5families, bothlanguages and bothworlds. All cases retained; globally exposed panel, no new blind-language claim.',
        'models': ['qwen4','qwen14','glm4'], 'qwen4': 'Reuse the exact320nativeB8trajectories from2744; no duplicate model run or new sample count.',
        'execution': 'Q14/GLM originalBF16, CUDA serial, standard fullmodel native cache, B8 same source grouping, leftpadding/explicitpositions, greedy128token cap, nativeEOS list.',
        'numerical_admission': 'Six originalB1prompt fixtures compare every hidden boundary and postnorm against committed construction capture before B8generation.',
        'scoring': 'Unchanged frozen yes/no parser, actualEOS/censoring, first-format token, firstvisiblecontent position and completeanswer separately. No gold supplied as decoder input; reasoning chain ungraded.',
        'fields': 'All coordinates at H1,Hearly,Hdepth,postnorm every actualgeneratedstep; predeclaredcase0 additionallyallboundaries at every step. Other intermediate/allprefixpositions not retained; input/checkpoint/config reconstructable.',
        'interpretation': 'This measures the native model, not a learned state predictor, output calibration, causal mediation or mechanism closure.'}
    immutable(path, value)
    return value


def monitor(model):
    handles, state = [], {}
    handles.append(model.model.embed_tokens.register_forward_hook(lambda m,a,o:state.__setitem__(0,bits(o[:,-1]))))
    for i, layer in enumerate(model.model.layers,1):
        handles.append(layer.register_forward_hook(lambda m,a,o,k=i:state.__setitem__(k,bits(o[:,-1]))))
    return handles, state


def summarize(rows, records):
    reports = []
    for family in ['all']+sorted({r['family'] for r in rows}):
        rr = [r for r in records if family=='all' or r['family']==family]
        groups = [r['source_group'] for r in rr]
        pids = sorted({r['pair_id'] for r in rr})
        reports.append({'family':family,'expressions':len(rr),'semantic_groups':len(set(groups)),
            'correct_and_stopped':sum(r['answer_scoring']['parsed_and_stopped_correct'] for r in rr),
            'both_worlds_correct_and_stopped':sum(all(r['answer_scoring']['parsed_and_stopped_correct'] for r in rr if r['pair_id']==pid) for pid in pids),
            'pairs':len(pids),'parsed_wrong':sum(r['answer_scoring']['conservative_final_answer'] is not None and not r['answer_scoring']['conservative_final_correct'] for r in rr),
            'unparsed_EOS':sum(r['answer_scoring']['EOS'] and r['answer_scoring']['conservative_final_answer'] is None for r in rr),
            'censored':sum(r['answer_scoring']['censored'] for r in rr),
            'strict_answer_only':sum(r['answer_scoring']['strict_answer_only'] for r in rr),
            'mean_generated_tokens':float(np.mean([len(r['generated_ids']) for r in rr])),
            'first_B1_B8_disagreements':sum(r['first_shape']['B1_token']!=r['first_shape']['B8_token'] for r in rr),
            'success_interval':clustered([int(r['answer_scoring']['parsed_and_stopped_correct']) for r in rr],groups)})
    return reports


def reuse_q4(rows):
    records = []
    for row in rows:
        p=OLD/'identifiability/behavior/native/commits'/(row['sample_id']+'.json')
        r=read(p)
        r.update(reused_from=str(p.relative_to(ROOT)),reused_sha256=sha(p),
            first_shape={'B1_token':r['initial_shape_control']['B1_initial_token_id'],'B8_token':r['initial_shape_control']['B8_initial_token_id'],
                'all_coordinate_postnorm_MSE':r['initial_shape_control']['B8_vs_B1_original_prompt_postnorm_MSE']})
        records.append(r)
    compressed(OUT/'qwen4/records.json.gz',records)
    result={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'model':'qwen4',
        'new_trajectories':0,'reused_trajectories':320,'summary':summarize(rows,records),
        'scope':'Exact oldnativeB8behavior reused; do not imply new per-step or all-layer capture. Old first/final fields remain accessible.'}
    save(OUT/'qwen4/result.json',result)
    return result


def main(key):
    import torch
    protocol=freeze_language()
    out=OUT/key
    if (out/'result.json').exists():
        assert read(out/'result.json')['all_passed']
        return
    rows=gzread(BASE/'material.json.gz')['models'][key]['rows']
    if key=='qwen4':
        reuse_q4(rows)
        print('NATIVE_LANGUAGE_Q4_REUSED',flush=True)
        return
    import psutil
    ancestors={os.getpid(),*(p.pid for p in psutil.Process().parents())}
    forbidden={'phase2745_rdc_construction_compile.py','phase2745_rdc_construction_fit.py',
        'phase2745_rdc_construction_language.py','phase2745_rdc_construction_capture.py'}
    for p in psutil.process_iter(['pid','cmdline']):
        if p.info['pid'] not in ancestors and any(Path(a).name in forbidden for a in p.info['cmdline'] or []):
            raise RuntimeError('Wait for other CUDA task: '+str(p.info['pid']))
    # Pread for this process's runtime offload reader, as numerically admitted
    # for GLM; do not alter installed packages or any checkpoint file.
    import accelerate.utils.offload as reader
    original_open=reader.safe_open
    def pread(*a,**kw):
        kw['backend']='pread'
        return original_open(*a,**kw)
    pread._rdc_pread=True
    reader.safe_open=pread
    start,model,handles=time.monotonic(),None,[]
    source=snapshot(__file__)
    try:
        guard(128*1024**2)
        if key=='qwen14':
            from rdc_operator_model import memory
            # A separate lower-residency configuration, not relaxation of the
            # original6GiB CPU loader's preflight. Same six native fixtures.
            if memory()['host_available_bytes']<12*1024**3:
                from rdc_construction_lowmem import load_q14
                model,tok=load_q14(out)
            else:
                model,tok=load(key,out)
        else:
            model,tok=load(key,out)
        early,depth=len(model.model.layers)//3,len(model.model.layers)
        handles,state=monitor(model)
        stop=model.generation_config.eos_token_id or tok.eos_token_id
        stop=set(stop if isinstance(stop,list) else [stop])
        pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        checks,records=[],[]
        with torch.inference_mode():
            for index in [0,66,129,195,258,319]:
                row=rows[index]
                o=model.model(input_ids=torch.tensor([row['prompt_ids']],device='cuda'),use_cache=True)
                with np.load(BASE/'capture'/key/'fields'/(row['sample_id']+'.npz')) as z:
                    all_equal=np.array_equal(np.stack([state[i][0] for i in range(depth+1)]),z['prefix_layers'])
                    post_equal=np.array_equal(bits(o.last_hidden_state[0,-1]),z['prefix_postnorm'])
                assert all_equal and post_equal,('NativeB1fixture changed',key,index)
                checks.append({'row_index':index,'all_hidden_boundaries_bit_equal':all_equal,'postnorm_bit_equal':post_equal})
                del o
            save(out/'numerical_admission.json',{'timestamp':stamp(),'all_passed':True,'checks':checks,'source':source})
            for begin in range(0,len(rows),8):
                batch=rows[begin:begin+8]
                paths=[out/'commits'/(r['sample_id']+'.json') for r in batch]
                if all(p.exists() for p in paths):
                    records.extend(read(p) for p in paths)
                    continue
                # Recompute an interrupted exact original batch. Already
                # committed rows are byte/value checked and never replaced.
                prior_batch=[read(p) if p.exists() else None for p in paths]
                n,maxlen=len(batch),max(len(r['prompt_ids']) for r in batch)
                ids=torch.full((n,maxlen),pad,device='cuda',dtype=torch.long)
                mask=torch.zeros_like(ids)
                for b,row in enumerate(batch):
                    ids[b,-len(row['prompt_ids']):]=torch.tensor(row['prompt_ids'],device='cuda')
                    mask[b,-len(row['prompt_ids']):]=1
                pos=(mask.cumsum(-1)-1).clamp_min(0)
                cache,done=None,[False]*n
                tokens=[[] for _ in batch]
                fields=[[] for _ in batch]
                full=[[] for _ in batch]
                selected=[1,early,depth]
                tick=time.monotonic()
                for step in range(128):
                    o=model.model(input_ids=ids,attention_mask=mask,position_ids=pos,past_key_values=cache,use_cache=True)
                    cache=o.past_key_values
                    h=o.last_hidden_state[:,-1]
                    choice=model.lm_head(h).float().argmax(-1)
                    post=bits(h)
                    for b,row in enumerate(batch):
                        if done[b]:
                            continue
                        token=int(choice[b]);tokens[b].append(token)
                        fields[b].append(np.stack([state[l][b] for l in selected]+[post[b]]))
                        if row['case']==0:
                            full[b].append(np.stack([state[l][b] for l in range(depth+1)]))
                        done[b]=token in stop or step==127
                    del o,h
                    if all(done):
                        break
                    active=torch.tensor([not d for d in done],device='cuda',dtype=torch.long)
                    mask=torch.cat([mask,active[:,None]],-1)
                    pos=(mask.sum(-1)-1).clamp_min(0)[:,None]
                    ids=choice[:,None];ids[active==0]=pad
                elapsed=time.monotonic()-tick
                for b,row in enumerate(batch):
                    generated=tok.decode(tokens[b],skip_special_tokens=True)
                    first_visible=next((j for j in range(len(tokens[b])) if tok.decode(tokens[b][:j+1],skip_special_tokens=True).strip()),None)
                    values=np.stack(fields[b])
                    cp=read(BASE/'capture'/key/'commits'/(row['sample_id']+'.json'))
                    with np.load(BASE/'capture'/key/'fields'/(row['sample_id']+'.npz')) as z:
                        shape_mse=float(np.mean((unbits(values[0,-1]).astype(float)-unbits(z['prefix_postnorm']))**2))
                    arrays={'selected_hidden_states':values,'generated_ids':np.array(tokens[b]),'selected_boundaries':np.array(selected+[-1])}
                    if full[b]:
                        arrays['all_hidden_boundaries_fixture']=np.stack(full[b])
                    field_path=out/'fields'/(row['sample_id']+'.npz')
                    guard(sum(a.nbytes for a in arrays.values()))
                    prior=prior_batch[b]
                    if prior is not None:
                        assert prior['generated_ids']==tokens[b] and sha(field_path)==prior['field_sha256']
                        with np.load(field_path) as old:
                            assert set(old.files)==set(arrays) and all(np.array_equal(old[k],v) for k,v in arrays.items())
                        records.append(prior)
                        continue
                    npz(field_path,**arrays)
                    r={k:row[k] for k in ['sample_id','pair_id','source_group','family','language','world','case','split','target']}
                    r.update(timestamp=stamp(),model=key,generated_ids=tokens[b],generated_text=generated,
                        answer_scoring=language_score(row,generated,tokens[b],stop,128),
                        first_visible_content_step=first_visible,first_token_text=tok.decode(tokens[b][:1],skip_special_tokens=False),
                        first_shape={'B1_token':cp['actual_current']['argmax_id'],'B8_token':tokens[b][0],'all_coordinate_postnorm_MSE':shape_mse},
                        batch_ids=[r['sample_id'] for r in batch],native_stop_ids=sorted(stop),
                        field_sha256=sha(field_path),source=source,allocated_batch_seconds=elapsed/n,
                        scope='Native greedy own history, no forecast/answer/calibrator injection; B8 andB1shape effects separately measured; finalreasoningchain ungraded.')
                    save(paths[b],r);records.append(r)
                del cache,ids,mask,fields,full,values,arrays
                save(out/'progress.json',{'timestamp':stamp(),'rows':len(records),'total':320,'seconds':time.monotonic()-start})
                print('NATIVE_LANGUAGE',key,len(records),320,round(time.monotonic()-start,1),flush=True)
            compressed(out/'records.json.gz',records)
            result={'timestamp':stamp(),'source':source,'all_passed':True,'model':key,'new_trajectories':len(records),
                'reused_trajectories':0,'protocol':protocol,'numerical_admission':checks,'summary':summarize(rows,records),
                'first_token_counts':{str(t):sum(r['generated_ids'][0]==t for r in records) for t in sorted({r['generated_ids'][0] for r in records})},
                'parser_regression':language_checks(),'seconds':time.monotonic()-start,'peak_cuda_allocated':torch.cuda.max_memory_allocated(),
                'scope':protocol['interpretation']}
            save(out/'result.json',result)
            ledger('construction_native_language_'+key,result['seconds'])
            print('NATIVE_LANGUAGE_DONE',key,result['seconds'],flush=True)
    except Exception as exc:
        failure(out,start,exc)
        raise
    finally:
        for h in handles:
            h.remove()
        if model is not None:
            del model
        gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',nargs='?',choices=['qwen4','qwen14','glm4']);p.add_argument('--freeze',action='store_true')
    a=p.parse_args()
    if a.freeze:
        freeze_language();print('NATIVE_LANGUAGE_PROTOCOL_FROZEN',flush=True)
    else:
        assert a.model
        main(a.model)
