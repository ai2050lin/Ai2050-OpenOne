"""Independent native sample/history/teacher identity audit; no model forward."""
import argparse
from collections import Counter
from transformers import AutoTokenizer
from rdc_question_common import *
from rdc_question_material import conservative_complete_answer


class Fields:
    def __init__(self):self.checked={}

    def read(self,ref,names):
        path=ROOT/ref['path'];physical=path.resolve()
        assert physical.is_relative_to(PHYSICAL.resolve()) and path.is_file()
        state=path.stat();identity=(state.st_size,state.st_mtime_ns,ref['sha256'])
        if str(path)not in self.checked:
            assert sha(path)==ref['sha256'];self.checked[str(path)]=identity
        else:assert self.checked[str(path)]==identity
        with np.load(path,allow_pickle=False)as z:
            assert set(z.files)==set(ref['arrays'])
            output={name:z[name].copy()for name in names}
        for name,value in output.items():
            header=ref['arrays'][name]
            assert list(value.shape)==header['shape'] and str(value.dtype)==header['dtype']
        return output


def native(key,confirmation=False):
    start=time.monotonic();scope='confirmation' if confirmation else 'nonconfirmation'
    native_path=OUT/'native'/key/scope/'result.json';result=read(native_path)
    assert result['all_passed'] and result['model']==key and result['split_scope']==scope
    contract,manifest,rows,groups=material(key,confirmation)
    revision={'source':snapshot(__file__),'native_result_sha256':sha(native_path),
        'material_manifest_sha256':sha(OUT/'material/manifest.json'),
        'scoring_source':snapshot(Path(__file__).with_name('rdc_question_material.py'))}
    target=OUT/'verification'/('native_'+key+'_'+scope+'.json')
    if target.exists():
        old=read(target);assert old['all_passed'] and old['execution']==revision
        print('NATURAL_IDENTITY_ALREADY_COMPLETE',key,scope,flush=True);return old
    config=read(ROOT/'models/hf'/MODELS[key]/'config.json')
    width,middle,depth=config['hidden_size'],config['intermediate_size'],config['num_hidden_layers']
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf'/MODELS[key],local_files_only=True,use_fast=True,trust_remote_code=True)
    for ref in manifest['tokenizer_metadata'][key]['files']:assert sha(ROOT/ref['path'])==ref['sha256']
    lookup={r['question_id']:r for r in rows};groupmap={g['group_id']:g for g in groups}
    fields=Fields();seen=set();seen_groups=set();counts=Counter();max_mean_error=0.
    full=set(contract['capture']['full_history_context_ids'])
    selected=contract['capture']['selected_MLP_blocks'][key]
    assert len(result['group_receipts'])==len(groups)==(64 if confirmation else 336)
    for pointer in result['group_receipts']:
        group=read(OUT/pointer);gid=group['group_id'];assert gid in groupmap and gid not in seen_groups
        seen_groups.add(gid);expected=groupmap[gid]
        assert group['execution']==result['execution']
        assert [q['question_id']for q in group['questions']]==expected['four_initial_question_ids']
        assert len(group['questions'])==4 and group['split']==expected['split'] and group['cohort']==expected['cohort']
        rr=[lookup[q['question_id']]for q in group['questions']]
        token=rr[0]['tokens'];prefix=token['context_prefix_ids'];source_positions=token['context_token_positions']
        assert all(r['tokens']['context_prefix_ids']==prefix and r['tokens']['context_token_positions']==source_positions for r in rr)
        context=fields.read(group['context_field'],['source_H12_BF16','H12_last_BF16','context_H12_mean'])
        assert context['source_H12_BF16'].shape==(len(prefix),width)
        assert np.array_equal(context['source_H12_BF16'][-1],context['H12_last_BF16'])
        average=unbits(context['source_H12_BF16'][source_positions]).astype(np.float64).mean(0)
        error=float(np.max(np.abs(average-context['context_H12_mean'])))
        assert error<1e-12,(key,gid,error);max_mean_error=max(max_mean_error,error)
        for q,row in zip(group['questions'],rr):
            qid=q['question_id'];assert qid not in seen;seen.add(qid)
            assert all(q[k]==row[k]for k in ['question_id','group_id','cohort','split'])
            info=row['tokens'];ids=info['input_ids'];assert ids==prefix+info['question_branch_ids']
            assert info['actual_input_sha256']==hashlib.sha256(info['actual_input'].encode()).hexdigest()
            first=fields.read(q['field'],['hidden_BF16','postnorm_BF16','H12_last_BF16','native_source_read_BF16','block12_attention_write_BF16'])
            assert first['hidden_BF16'].shape==(depth+1,width) and first['postnorm_BF16'].shape==(width,)
            assert np.array_equal(first['H12_last_BF16'],first['hidden_BF16'][12])
            assert np.array_equal(first['native_source_read_BF16'],first['block12_attention_write_BF16'])
            for block in selected:
                for name in ['gate','up','product']:
                    assert q['field']['arrays'][f'block{block}_{name}_BF16']=={'shape':[middle],'dtype':'uint16'}
            counts['first_prefix_questions']+=1;counts[row['split']+'_questions']+=1
            if row['split']=='train':
                assert 'history'not in q and 'teacher'not in q
                continue
            history=q['history'];generated=history['generated_ids'];stops=set(info['native_stop_ids'])
            assert 0<len(generated)<=128 and not any(x in stops for x in generated[:-1])
            eos=generated[-1]in stops
            assert history['native_EOS']==eos and history['censored']==(not eos and len(generated)==128)
            text=tok.decode(generated,skip_special_tokens=True);assert text==history['generated_text']
            assert conservative_complete_answer(text,row['answer_annotations'],eos)==history['score']
            names=['generated_ids','positions','postnorm_BF16','H12_last_BF16','native_source_read_BF16','statistics']
            if gid in full:names.append('all_hidden_BF16')
            actual=fields.read(history['field'],names)
            assert history['full_H_all_layers_every_generated_step']==(gid in full)
            assert ('all_hidden_BF16'in history['field']['arrays'])==(gid in full)
            assert actual['generated_ids'].tolist()==generated
            assert np.array_equal(actual['positions'],np.arange(len(generated))+len(ids)-1)
            for name in ['postnorm_BF16','H12_last_BF16','native_source_read_BF16']:
                assert actual[name].shape==(len(generated),width)
                assert np.array_equal(actual[name][0],first[name])
            assert actual['statistics'].shape==(len(generated),2)
            if gid in full:
                assert actual['all_hidden_BF16'].shape==(len(generated),depth+1,width)
                assert np.array_equal(actual['all_hidden_BF16'][0],first['hidden_BF16'])
                counts['full_hidden_histories']+=1
            teacher=q['teacher'];target_ids=info['teacher_ids_including_EOS']
            ta=fields.read(teacher['field'],['teacher_ids','NLL','argmax','postnorm_BF16','entropy_FP64'])
            assert teacher['teacher_ids_including_EOS']==target_ids==ta['teacher_ids'].tolist()
            assert teacher['tokens']==len(target_ids) and target_ids[-1]in stops
            assert ta['postnorm_BF16'].shape==(len(target_ids),width)
            assert np.array_equal(ta['postnorm_BF16'][0],first['postnorm_BF16'])
            assert teacher['mean_token_NLL']==float(ta['NLL'].mean()) and teacher['sum_token_NLL']==float(ta['NLL'].sum())
            assert teacher['teacher_forced_argmax_token_accuracy']==float(np.mean(ta['argmax']==ta['teacher_ids']))
            assert ta['NLL'][0]==q['statistics']['first_teacher_token_NLL']
            assert ta['argmax'][0]==q['statistics']['argmax']==generated[0]
            assert ta['entropy_FP64'][0]==q['statistics']['entropy_FP64']==actual['statistics'][0,0]
            counts['teacher_tokens']+=len(target_ids);counts['free_histories']+=1;counts['free_tokens']+=len(generated)
            counts['complete_match_and_stop']+=int(history['score']['whole_response_exact_and_stopped'])
        if len(seen_groups)%48==0:print('NATURAL_IDENTITY',key,scope,len(seen_groups),len(groups),flush=True)
    assert seen==set(lookup) and seen_groups==set(groupmap)
    assert counts['first_prefix_questions']==result['questions']==(256 if confirmation else 1344)
    assert counts['free_histories']==result['native_free_histories']==(256 if confirmation else 576)
    assert counts['free_tokens']==result['native_free_tokens'] and counts['teacher_tokens']==result['teacher_tokens']
    assert counts['full_hidden_histories']==sum(r['group_id']in full and r['split']!='train'for r in rows)
    value={'timestamp':stamp(),'all_passed':True,'execution':revision,'model':key,'scope':scope,
        'counts':dict(counts),'complete_context_groups':len(seen_groups),'field_files_SHA256_checked':len(fields.checked),
        'maximum_CPU_FP64_context_mean_minus_native_GPU_FP64':max_mean_error,
        'all_native_history_text_and_original_scoring_recomputed':True,
        'all_first_prefix_teacher_free_state_token_alignment_exact':True,
        'all_declared_full_history_membership_and_native_axes_checked':True,
        'seconds':time.monotonic()-start,
        'limits':'CPU identity, score and field accounting only. No second CUDA model, no re-fitted predictor, no new semantic adjudication. Context mean uses full coordinates and permits <1e-12 CPU/GPU FP64 reduction difference; all named first-state equality checks are exact stored-word comparisons.'}
    immutable(target,value);print('NATURAL_IDENTITY_AUDIT_PASS',key,scope,dict(counts),round(value['seconds'],2),flush=True)
    return value


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',choices=['qwen4','qwen14','glm4'],required=True)
    parser.add_argument('--confirmation',action='store_true');args=parser.parse_args();native(args.model,args.confirmation)
