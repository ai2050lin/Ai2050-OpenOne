"""Bounded Q4 CUDA full-answer gradient and shape qualification before training."""
from collections import defaultdict
from rdc_question_common import *
from rdc_question_learning import *
from phase2748_rdc_training_material import freeze as training_material
from phase2748_rdc_native_capture import context_requests, question_requests
from rdc_question_native import QuestionNative
from rdc_question_history import teacher_likelihood


def freeze():
    path = OUT/'training/execution_contract.json'
    execution = {'pilot': snapshot(__file__), 'learning': snapshot(Path(__file__).with_name('rdc_question_learning.py')),
        'bridge': snapshot(Path(__file__).with_name('phase2747_rdc_training.py')),
        'history': snapshot(Path(__file__).with_name('rdc_question_history.py')),
        'native': snapshot(Path(__file__).with_name('rdc_question_native.py')),
        'training_material_sha256': sha(OUT/'training/material_manifest.json'),
        'effective_contract_sha256': sha(OUT/'effective_experiment_contract.json')}
    if path.exists():
        old = read(path)
        assert old['execution'] == execution
        return old
    unit = read(OUT/'unit/learning_current.json')
    assert unit['all_passed'] and unit['learning']['sha256'] == execution['learning']['sha256']
    _, _, rows, _ = material('qwen4')
    rows = [r for r in rows if r['split'] == 'train']
    selected = []
    for cohort in ['quoref', 'drop']:
        rr = [r for r in rows if r['cohort'] == cohort]
        longest = max(rr, key=lambda r: (len(r['tokens']['input_ids'])+len(r['tokens']['teacher_ids_including_EOS'])-1,r['question_id']))
        tokenlongest = max((r for r in rr if r['question_id'] != longest['question_id']),
            key=lambda r: (len(r['tokens']['teacher_ids_including_EOS']), r['question_id']))
        selected.extend([longest['question_id'], tokenlongest['question_id']])
    value = {'timestamp': stamp(), 'execution': execution, 'unit_sha256': sha(OUT/'unit/learning_current.json'),
        'before_any2748_parameter_update': True, 'pilot_question_ids': selected,
        'pilot_selection': 'For each training cohort longest complete causal teacher input, then distinct longest teacher answer. No correctness/effect selection.',
        'forward': 'Native joint-B1 full prompt plus teacher[:-1], original causal mask, use_cache=False, no padding or truncation. Supervise positions prompt_length-1 through last consumed teacher. Native BF16unembedding B1 separately for each teacher position, full vocabulary float64logsoftmax.',
        'loss': 'Per-complete-answer token mean including EOS, then mean8questions. Given permuted whole answer as actual teacher history only for permuted condition. Surface classes on true teacher history, not semantic erasure.',
        'finite_precision': 'Joint teacher shape differs from segmented sequential B1 evaluation. Pilot measures all teacher postnorm and NLL differences, bridge separately. Mathematical causal masking and CPUfuture-substitution check do not establish bit equality across shapes.',
        'parameters': 'Original Q4 block16 gate/up/down all74711040parameters become FP32; input/output bridge casts to FP32/BF16. Other parameters fixed BF16. Layers17+ nonreentrant checkpoint only when gradients enabled; no KV cache during gradients.',
        'optimizer': 'Absolute FP32L2 step0.02 full-gradient SGD, no momentum/weight decay; actual rounded step norm measured. Gradient through BF16cast is surrogate differentiation, not derivative of discrete rounding.',
        'pilot_updates': 'One test update per condition from same original bridge, then all FP32words restored and teacher arrays rechecked. These updates never initialize or count toward six formal96step runs.',
        'validation': 'Checkpoints1/8/32/96 evaluate all192validation questions using segmented sequential teacher scoring in FP32bridge; preserve each actual teacher token/postnorm/NLL. Final96stepactualBF16deployment has separate832free/teacher histories. IntermediateFP32weights not persisted, no validation-based parameter/seed choice.',
        'parameter_retention': 'Final checkpoint stores every actualFP32 word and GPU-castBF16word with qualified lossless codec plus exact originalcheckpoint SHA and shapes. Restore all original values before each condition/seed.',
        'scope': 'Numerical training protocol, not reconstruction of pretraining or a proof of unique semantic parameters.'}
    immutable(path, value)
    return value


def sequential(model, engine, rows, groups, contract):
    by_id = {r['question_id']: r for r in rows}
    # One native context at a time keeps this stress pilot independently bounded.
    records = {}
    for group in groups:
        allrows = [by_id[qid] for qid in group['four_initial_question_ids']]
        contexts = context_requests('qwen4', [group], allrows, model.config, contract)
        for request in contexts:
            request['collect_hidden'] = request['full_prefix_H'] = False
        engine.forward(contexts)
        requests = question_requests('qwen4', [group], allrows, contexts, model.config)
        for request in requests:
            request.update(collect_hidden=False, collect_units=False, account_attention=False, save_source_details=False)
        outputs = engine.forward(requests)
        packets, _ = teacher_likelihood(model, engine, requests, outputs, allrows)
        records.update({record['question_id']: arrays for arrays, record in packets})
        del contexts, requests, outputs, packets
        gc.collect()
    return records


def main():
    import torch
    training_material()
    protocol = freeze()
    folder = OUT/'training/pilot'
    resultpath = folder/'result.json'
    if resultpath.exists():
        result = read(resultpath)
        assert result['execution_contract_sha256'] == sha(OUT/'training/execution_contract.json')
        print('NATURAL_LEARNING_PILOT_ALREADY_COMPLETE', flush=True)
        return
    start = time.monotonic()
    model = engine = None
    handles = []
    try:
        contract, _, rows, groups = material('qwen4')
        rows_by_id = {r['question_id']: r for r in rows}
        selected = [rows_by_id[qid] for qid in protocol['pilot_question_ids']]
        gids = {r['group_id'] for r in selected}
        context_groups = [g for g in groups if g['group_id'] in gids]
        model, tok = load('qwen4', folder/'loader')
        assert all(p.device.type == 'cuda' for p in model.parameters())
        unchanged = parameter_fingerprints(model, exclude_target=True)
        engine = QuestionNative(model, [])
        native_seq = sequential(model, engine, rows, context_groups, contract)
        native_joint = {}
        material_manifest = read(OUT/'training/material_manifest.json')
        with np.load(ROOT/material_manifest['partition_and_exposure']['path']) as z:
            classes = torch.tensor(z['classes'].astype(np.int64), device='cuda')
        all_draws = gzread(ROOT/material_manifest['draws']['path'])
        draws_by_condition = {condition: [next(d for d in all_draws if d['condition'] == condition and d['seed'] == 2748 and d['question_id'] == r['question_id'])
            for r in selected] for condition in contract['learning']['conditions']}
        with torch.no_grad():
            for r, d in zip(selected, draws_by_condition['true_complete_answer']):
                _, native_joint[r['question_id']] = teacher_objective(model, r, d, classes, True)
        target, original, handles = prepare_gradients(model)
        assert sum(p.numel() for p in target.parameters()) == 74711040
        params = list(target.parameters())
        bridge = {}
        with torch.no_grad():
            for r,d in zip(selected, draws_by_condition['true_complete_answer']):
                _, bridge[r['question_id']] = teacher_objective(model,r,d,classes,True)
        shape_records = []
        for r in selected:
            qid = r['question_id']
            seq, joint, base = native_seq[qid], native_joint[qid], bridge[qid]
            shape_records.append({'question_id': qid, 'cohort': r['cohort'],
                'prompt_tokens': len(r['tokens']['input_ids']), 'teacher_tokens': len(seq['teacher_ids']),
                'native_joint_minus_segmented_mean_NLL': float((joint['statistics'][:,0]-seq['NLL']).mean()),
                'native_joint_vs_segmented_postnorm_MSE': float(np.mean((unbits(joint['postnorm_BF16']).astype(float)-unbits(seq['postnorm_BF16']))**2)),
                'bridge_minus_native_joint_mean_NLL': float((base['statistics'][:,0]-joint['statistics'][:,0]).mean()),
                'bridge_vs_native_joint_postnorm_MSE': float(np.mean((unbits(base['postnorm_BF16']).astype(float)-unbits(joint['postnorm_BF16']))**2))})
            commit_arrays(Path('training/pilot/native_sequential'), qid.replace(':','_'), seq)
            commit_arrays(Path('training/pilot/native_joint'), qid.replace(':','_'), joint)
            commit_arrays(Path('training/pilot/bridge_joint'), qid.replace(':','_'), base)
        gradients, gradient_records = [], []
        for condition in contract['learning']['conditions']:
            restore(target, original)
            tick = time.monotonic()
            grads, losses = mean_gradient(model, rows_by_id, draws_by_condition[condition], classes, params)
            norm = full_norm(grads)
            assert np.isfinite(norm) and norm > 0
            with torch.no_grad():
                for p,g in zip(params,grads):
                    p.add_(g, alpha=-.02/(norm+1e-12))
            step = full_norm([p.detach()-original[name].to(p.device) for name,p in target.named_parameters()])
            assert abs(step-.02) < 1e-5
            restore(target, original)
            with torch.no_grad():
                for r,d in zip(selected,draws_by_condition['true_complete_answer']):
                    _, restored = teacher_objective(model,r,d,classes,True)
                    assert all(np.array_equal(restored[k],bridge[r['question_id']][k]) for k in restored)
            gradients.append(grads)
            gradient_records.append({'condition':condition,'example_losses':losses,'gradient_norm':norm,
                'actual_FP32_step_norm':step,'seconds':time.monotonic()-tick,'all_parameters_and_teacher_arrays_restored_exactly':True})
            print('NATURAL_LEARNING_PILOT_CONDITION',condition,round(gradient_records[-1]['seconds'],1),flush=True)
        cosine = [[float(sum((a.double()*b.double()).sum() for a,b in zip(ga,gb)))/(gradient_records[i]['gradient_norm']*gradient_records[j]['gradient_norm'])
                   for j,gb in enumerate(gradients)] for i,ga in enumerate(gradients)]
        assert parameter_fingerprints(model,exclude_target=True) == unchanged
        immutable(folder/'unchanged_parameters.json', unchanged)
        result = {'timestamp':stamp(),'all_passed':True,'execution_contract_sha256':sha(OUT/'training/execution_contract.json'),
            'shape_and_bridge_comparisons':shape_records,'full_gradient_records':gradient_records,
            'all_parameter_gradient_cosines':cosine,'other_parameters_exactly_unchanged':True,
            'actual_test_updates_restored':3,'formal96step_updates_executed':0,
            'peak_CUDA_bytes':torch.cuda.max_memory_allocated(),'seconds':time.monotonic()-start,
            'scope':'Numerical CUDA qualification only on4predeclared training questions; not final learning effect.'}
        immutable(resultpath,result)
        print('NATURAL_LEARNING_PILOT_PASS',round(result['seconds'],1),flush=True)
    except Exception as exc:
        failure(folder,start,exc)
        raise
    finally:
        if engine is not None:
            engine.close()
        for handle in handles:
            handle.remove()
        if model is not None:
            del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
