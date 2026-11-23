"""Archive-gated discovery scan of the already exposed Phase 2990 families."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2] if Path(__file__).stem != 'test' else Path(os.environ.get('PYTHONPATH', '').split(os.pathsep)[0])
HERE = ROOT/'tests/glm5'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from deploy.storage_policy import check_write
from deploy.test_archive_client import runtime_identity, save
from shared.dist_protocol.test_archive import CORE_ROLES, digest, validate_plan
from source_binding_design import read, sha

SOURCE = HERE/'result/clt_b26_competition_v1'
MATERIAL = HERE/'result/clt_first_record_path_v1/materials/manifest.json'
CODE = HERE/'clt_first_record_discovery.py'
SEED = 202611190315
CAP = .02
CHUNK_FAMILIES = 8


def code_identity():
    names = [
        'tests/glm5/clt_first_record_discovery.py',
        'tests/glm5/source_binding_runtime.py',
        'tests/glm5/source_binding_design.py',
        'tests/glm5/source_binding_resources.py',
        'shared/dist_protocol/mechanism_trace.py',
        'shared/dist_protocol/natural_register_runner.py',
    ]
    return {name: sha(ROOT/name) for name in names}


def build_plan(out):
    receipt = read(SOURCE/'verified.json')
    questions = SOURCE/'confirm-materials.json'
    if sha(questions) != receipt['files'][questions.name]:
        raise ValueError('Phase 2990 discovery-source hash differs')
    rows = read(questions)
    materials = read(MATERIAL)
    if materials['model_snapshot'] != receipt['provenance']['model_revision']:
        raise ValueError('discovery and prepared family model snapshots differ')
    entry, files = sha(CODE), code_identity()
    selection = {
        'population': 'all 128 previously exposed confirmation families from Phase 2990, 12 cells each',
        'residual_slice': 'physical-first questions, model top1 correct and B26/H36 does not rank the gold record first out of five',
        'controls': ['model correct and H36 ranks gold first', 'model error', 'paired NAME and printed ROW queries'],
        'tie_rule': 'lowest immutable record index, as in the Phase 2990 NumPy argmax',
        'record_spans': 'preserve 5 immutable records plus one bucket for visible tokens outside the five',
        'sealed_materials_manifest_sha256': sha(MATERIAL),
        'source_receipt_sha256': sha(SOURCE/'verified.json'),
    }
    document = {
        'schema': 'research_test_plan.v1',
        'kind': 'real',
        'title': 'First physical record: parallel attention and MLP discovery capture',
        'objective': 'Collect native record-level attention and value sums, attention input and downstream MLP states on the exposed Phase 2990 families.',
        'principle': 'Group the real causal QK softmax by all five record spans plus outside; compare the summed value contributions against the native pre-O vector.',
        'hypothesis': 'Other attention heads or usable non-top H36 value writes can account for correct physical-first responses.',
        'alternatives': 'An anchor or token-position correlate, an H36 value that remains usable at low mass, or information already present in the residual.',
        'criteria': f'Acquisition only. Verify unchanged whole-vocabulary top1 and saved H36 five-record ranking, normalized attention and native pre-O reconstruction. For every head/sample divide the maximum reconstruction error by (1 + max absolute original pre-O value); require ≤ {CAP}, including a direct native SDPA check. Retain both absolute and normalized diagnostics.',
        'domains': ['KN', 'RS'],
        'family_ids': ['SOURCE-BINDING-20261110', 'CLT-FIRST-RECORD-PATH'],
        'dataset': {'id': 'clt_b26_competition_v1/confirm', 'revision': 'phase2990-as-discovery-v1',
                    'content_sha256': sha(questions), 'questions_file_sha256': sha(questions),
                    'split_manifest_sha256': sha(SOURCE/'prereg.json'), 'case_count': len(rows),
                    'exposure': 'Previously evaluated and displayed, discovery only.'},
        'entrypoint_sha256': entry,
        'execution_args': [],
        'acquisition': {'model_id': 'Qwen3-14B', 'model_snapshot': materials['model_snapshot'],
                        'tokenizer_sha256': materials['tokenizer_sha256'], 'entrypoint_sha256': entry,
                        'questions_file_sha256': sha(questions), 'inputs_sha256': digest([[r['id'], r['tokens']] for r in rows]),
                        'code_sha256': digest(files), 'environment_sha256': digest(runtime_identity()),
                        'precision': 'BF16 actual forward, FP32 QK and weighted-value accounting',
                        'seed': SEED, 'generation': {'script_args': [], 'one_step': True, 'teacher_forcing': False, 'sample': False},
                        'capture': {'layers': list(range(40)), 'readouts': ['last input token'],
                                    'boundary': 'QKV projections and QK norms/RoPE to pre-O; attention input, post-attention MLP input and SwiGLU intermediate'}},
        'analysis': {'method_version': 'clt_first_record_discovery.v2', 'code_sha256': digest(files),
                     'scoring_sha256': digest({'cap': CAP, 'selection': selection}), 'selection': selection,
                     'required_artifact_roles': sorted(CORE_ROLES | {'native_field'})},
    }
    validate_plan(document)
    save(out, document)
    print(f'Archive plan created for {len(rows)} previously exposed inputs: {out}', flush=True)


def registered_context():
    # The project routine runner writes these files after center registration and before spawning a child.
    questions = Path(os.environ.get('AI2050_TEST_QUESTIONS', ''))
    work = Path(os.environ.get('AI2050_TEST_WORK_DIR', ''))
    if not questions.is_file() or not work.is_absolute() or not work.is_dir():
        raise RuntimeError('必须通过已取得正式回执的 run_routine_test.py 启动')
    record = questions.parent
    plan = read(record/'protocol.json')
    preflight = read(record/'preflight.json')
    review = read(record/'review.json')
    if (not preflight.get('preflight_id') or preflight.get('plan_sha256') != digest(plan)
            or not review.get('server_run_id') or Path(review['scratch']).resolve() != work.resolve()
            or plan['entrypoint_sha256'] != sha(Path(__file__))
            or plan['dataset']['questions_file_sha256'] != sha(questions)
            or plan['acquisition']['code_sha256'] != digest(code_identity())):
        raise RuntimeError('归档回执、登记或冻结文件不匹配')
    return work, questions, plan, review['server_run_id']


def grouped_account(t, q, k, v, visible, masks, scaling):
    """Inputs: q[n,h,d], k/v[n,kv,t,d], visible[n,t], masks[n,5,t]."""
    n, h, d = q.shape
    kv, length = k.shape[1:3]
    q = q.float().reshape(n, kv, h//kv, d)
    logits = t.einsum('bkgd,bktd->bkgt', q, k.float()).reshape(n, h, length) * scaling
    alpha = logits.masked_fill(~visible[:, None, :], float('-inf')).softmax(-1)
    groups = t.cat((masks, (visible & ~masks.any(1))[:, None]), dim=1).float()
    weights = alpha[:, :, None] * groups[:, None]
    values = t.einsum('bkgrt,bktd->bkgrd', weights.reshape(n, kv, h//kv, 6, length), v.float())
    return weights.sum(-1), values.reshape(n, h, 6, d)


def capture_batch(rt, rows, progress):
    t = rt.t
    shards = {}
    original_block = rt.block

    def block(number):
        layer = original_block(number)
        attention = layer.self_attn
        state, saved = {'offset': 0}, {}
        shards[number] = saved

        def prep(_, args, kwargs):
            state['norm_input'] = kwargs['hidden_states']
            state['pe'] = kwargs['position_embeddings']
            state['mask'] = kwargs['attention_mask'][:, 0, -1]

        def project(name):
            def hook(_, args, result):
                state[name] = result
            return hook

        def emit(name, tensor):
            if tensor.dtype == t.bfloat16:
                tensor = tensor.contiguous().view(t.uint16)
            saved.setdefault(name, []).append(tensor.cpu().numpy())

        def account(_, args):
            actual = args[0][:, -1].reshape(-1, rt.heads, rt.hd)
            n, length = state['q'].shape[:2]
            start = state['offset']
            masks = t.zeros((n, 5, length), dtype=t.bool, device=actual.device)
            for i, row in enumerate(rows[start:start+n]):
                pad = length-len(row['tokens'])
                for r, positions in enumerate(row['records']):
                    masks[i, r, [pad+p for p in positions]] = True
            q = attention.q_norm(state['q'].reshape(n, length, rt.heads, rt.hd)).transpose(1, 2)
            k = attention.k_norm(state['k'].reshape(n, length, rt.kv, rt.hd)).transpose(1, 2)
            q, k = rt.mod.apply_rotary_pos_emb(q, k, *state['pe'])
            v = state['v'].reshape(n, length, rt.kv, rt.hd).transpose(1, 2)
            mass, writes = grouped_account(t, q[:, :, -1], k, v, state['mask'], masks, attention.scaling)
            direct = t.nn.functional.scaled_dot_product_attention(
                q[:, :, -1:], k.repeat_interleave(rt.heads//rt.kv, dim=1),
                v.repeat_interleave(rt.heads//rt.kv, dim=1),
                attn_mask=state['mask'][:, None, None], scale=attention.scaling,
            ).squeeze(2)
            error = (writes.sum(2) - actual.float()).abs().amax(dim=-1)
            scale = 1 + actual.float().abs().amax(dim=-1)
            emit('mass', mass)
            emit('pre_o_record_values_bf16', writes.to(t.bfloat16))
            emit('native_pre_o_bf16', actual)
            emit('pre_o_max_error', error)
            emit('pre_o_normalized_error', error/scale)
            emit('pre_o_kernel_normalized_error', (direct.float()-actual.float()).abs().amax(dim=-1)/scale)
            emit('attention_input_bf16', state['norm_input'][:, -1])
            state['offset'] += n

        def mlp_input(_, args):
            emit('mlp_input_bf16', args[0][:, -1])

        def mlp_units(_, args):
            emit('mlp_units_bf16', args[0][:, -1])

        attention.register_forward_pre_hook(prep, with_kwargs=True)
        for name in ('q', 'k', 'v'):
            getattr(attention, f'{name}_proj').register_forward_hook(project(name))
        attention.o_proj.register_forward_pre_hook(account)
        layer.mlp.register_forward_pre_hook(mlp_input)
        layer.mlp.down_proj.register_forward_pre_hook(mlp_units)
        return layer

    def step(number, layer, x, y, chunk, length, offset=0, **_):
        if number >= 0 and offset + len(chunk) == len(rows) and (number+1) % 5 == 0:
            progress(number+1)

    rt.block = block
    try:
        _, logits = rt.forward(rows, step)
    finally:
        rt.block = original_block
    keys = shards[0]
    data = {key: np.stack([np.concatenate(shards[block][key]) for block in range(len(shards))], axis=1)
            for key in keys}
    scores = []
    for row, logit in zip(rows, logits, strict=True):
        gold = row['target_record']
        x = logit[row['choice_ids']].cpu().tolist()
        scores.append({'id': row['id'], 'top1': int(logit.argmax()), 'scores': x,
                       'gold_margin': x[gold]-max(x[:gold]+x[gold+1:])})
    return data, scores


def execute():
    work, questions, plan, run_id = registered_context()
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    from source_binding_design import MODELS
    from source_binding_resources import resources
    from source_binding_runtime import Runtime

    rows = read(questions)
    if snapshot(MODELS['qwen14']) != plan['acquisition']['model_snapshot']:
        raise RuntimeError('模型快照已改变')
    old_receipt = read(SOURCE/'verified.json')
    old_scores, old_rank = [], []
    for file, expected in sorted(read(SOURCE/'confirm-collection.json')['shards'].items()):
        scores = (SOURCE/file).with_suffix('.json')
        if sha(SOURCE/file) != expected or sha(scores) != old_receipt['files'][scores.name]:
            raise ValueError('source shard changed')
        old_scores.extend(read(scores)['scores'])
        with np.load(SOURCE/file) as previous:
            old_rank.extend(previous['mass_actual'].argmax(1).tolist())
    size = CHUNK_FAMILIES*12
    start = time.monotonic()
    files, counts = [], {name: 0 for name in ('head36_residual', 'head36_and_output_correct', 'output_error', 'other_queries')}
    max_error = max_relative = max_kernel = 0.0
    rt = Runtime('qwen14')
    t = rt.t
    total = (len(rows)+size-1)//size
    t.backends.cuda.matmul.allow_tf32 = False
    work_out = work/'case_results.jsonl'
    with GpuLease(), t.inference_mode(), work_out.open('x', encoding='utf-8', buffering=1) as results:
        for shard, first in enumerate(range(0, len(rows), size)):
            chunk = rows[first:first+size]
            check_write(work, 512*2**20, operation='First-record native discovery shard')

            def progress(block):
                completed = 40*shard+block
                elapsed = time.monotonic()-start
                print(json.dumps({'stage': 'first-record discovery', 'completed': completed, 'total': 40*total,
                                  'elapsed': elapsed, 'eta': elapsed*(40*total-completed)/completed if completed else None,
                                  'cuda_peak': t.cuda.max_memory_allocated(), **resources()}), flush=True)

            progress(0)
            data, scores = capture_batch(rt, chunk, progress)
            head_rank = data['mass'][:, 26, 36, :5].argmax(1)
            err = float(data['pre_o_max_error'].max())
            relative = float(data['pre_o_normalized_error'].max())
            kernel = float(data['pre_o_kernel_normalized_error'].max())
            max_error = max(err, max_error)
            max_relative = max(relative, max_relative)
            max_kernel = max(kernel, max_kernel)
            print(json.dumps({'shard': shard, 'pre_o_abs_error': err, 'pre_o_normalized_error': relative,
                              'pre_o_kernel_error': kernel}), flush=True)
            if (not np.isfinite([err, relative, kernel]).all() or max(relative, kernel) > CAP
                    or not np.allclose(data['mass'].sum(-1), 1, atol=1e-5, rtol=0)):
                raise ValueError(f'native reconstruction failed on shard {shard}: abs={err}, relative={relative}, kernel={kernel}')
            file = work/f'native-{shard:02}.npz'
            part = file.with_suffix('.partial')
            with part.open('xb') as stream:
                np.savez(stream, **data)
            part.replace(file)
            files.append({'path': file.name, 'role': 'native_field', 'retention': 'important'})
            save(work/'artifacts.json', {'schema': 'research_test_files.v1', 'files': files})
            for i, (row, result) in enumerate(zip(chunk, scores, strict=True)):
                j = first+i
                if old_scores[j]['id'] != row['id'] or old_scores[j]['top1'] != result['top1'] or old_rank[j] != head_rank[i]:
                    raise ValueError(f"old forward differs in {row['id']}")
                correct = result['top1'] == row['gold_id']
                label = 'other_queries' if row['rule'] != 2 else 'output_error' if not correct else (
                    'head36_and_output_correct' if head_rank[i] == row['target_record'] else 'head36_residual')
                counts[label] += 1
                results.write(json.dumps(dict(result, run_id=run_id, case_id=row['id'], family_id=row['family'],
                                              expected_token_id=row['gold_id'], actual_token=rt.tok.decode([result['top1']]),
                                              head36_rank=int(head_rank[i]), sample_slice=label,
                                              native_shard=file.name, native_row=i), ensure_ascii=False)+'\n')
    save(work/'conclusion.json', {'stage': 'discovery acquisition only', 'run_id': run_id, 'cases': len(rows),
                                  'counts': counts, 'max_pre_o_error': max_error, 'max_pre_o_normalized_error': max_relative,
                                  'max_pre_o_kernel_error': max_kernel, 'blocks': 40, 'heads_per_block': 40,
                                  'bf16_storage': 'unsigned 16-bit raw payload', 'record_order': 'immutable IDs 0–4 plus outside',
                                  'evidence': {'readable': 'candidate selection pending', 'predictable': 'untested', 'causal': 'untested'}})


if __name__ == '__main__':
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('--build-plan', type=Path, required=True)
        build_plan(parser.parse_args().build_plan)
    else:
        execute()
