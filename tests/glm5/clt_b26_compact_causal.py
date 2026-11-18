"""Test the admitted predictor with a matched native Q contribution perturbation."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np

from clt_b26_compact import (BLOCK, CAP, HEAD, HEAD_SLICE, KV, KV_SLICE, OUT, ROOT, SEED, TMP,
                             archive, candidate, checked, commit)
from deploy.storage_policy import check_write, checked_dir
from source_binding_design import MODELS, read, sha

ARMS = ('base', 'target', 'random', 'sham', 'restore')
DOSE = .50
CALIBRATION = OUT.parent / 'clt_b26_native_v1/discovery-materials.json'
PLAN = Path(__file__).with_suffix('.md')


def freeze():
    prereg, seal = checked(), candidate()
    evaluation = read(OUT / 'evaluation.json')
    assert evaluation['causal_admitted'] and evaluation['candidate_sha'] == sha(OUT/'candidate.json')
    assert all(x['passes'] for stage in evaluation['gates'].values() for x in stage.values())
    check_write(OUT, 64 << 20, operation='CLT causality')
    commit('causal-freeze.json', dict(utc=datetime.now(timezone.utc).isoformat(), provenance=prereg['provenance'],
                                     predictor_sha=seal['files']['predictor.npz'],
                                     prediction_sha=sha(OUT / 'evaluation.json'), prereg_sha=sha(OUT / 'prereg.json'),
                                     code=sha(Path(__file__)), plan=sha(PLAN), calibration_sha=sha(CALIBRATION),
                                     causal_material_sha=prereg['materials']['causal']['sha'], target_scale=DOSE,
                                     arms=ARMS, continuous_margin_gate=.10, dose_ratio=[.9, 1.1],
                                     expected_direction_gate=.8))
    print('Matched native Q intervention frozen after confirmed prediction gates', flush=True)


def run(split):
    import torch as t
    import torch.nn.functional as F
    from shared.dist_protocol.mechanism_trace import GpuLease
    from shared.dist_protocol.natural_register_runner import snapshot
    from source_binding_runtime import Runtime
    from source_binding_resources import resources

    prereg, seal = checked(), candidate()
    frozen = read(OUT / 'causal-freeze.json')
    assert frozen['code'] == sha(Path(__file__)) and frozen['plan'] == sha(PLAN)
    assert frozen['prediction_sha'] == sha(OUT / 'evaluation.json')
    assert frozen['prereg_sha'] == sha(OUT / 'prereg.json') and read(OUT / 'evaluation.json')['causal_admitted']
    check_write(OUT, 64 << 20, operation='CLT causal input and results')
    checked_dir(TMP)
    clock = time.monotonic()
    if split == 'causal':
        assert read(OUT / 'calibration-causal-summary.json')['technical_pass']
        inputs = OUT / 'causal-materials.json'
        assert sha(inputs) == frozen['causal_material_sha']
        count = 128
    else:
        inputs = CALIBRATION
        assert sha(inputs) == frozen['calibration_sha']
        count = 32
    candidates = read(inputs)
    selected_rows = [r for r in candidates if r['family'] < count and r['cell'] == (r['family']//2) % 16]
    assert len(selected_rows) == count and len({r['family'] for r in selected_rows}) == count
    rows = [dict(row, arm=arm) for row in selected_rows for arm in ARMS]
    assert not (OUT / f'{split}-causal-summary.json').exists()
    rt = Runtime('qwen14')
    t.backends.cuda.matmul.allow_tf32 = False
    saved = {}
    arms = len(ARMS)

    def event(stage, n):
        elapsed = time.monotonic() - clock
        payload = dict(stage=stage, completed=n, total=40, elapsed=elapsed,
                       eta=elapsed/n*(40-n) if n else None, cuda_peak=t.cuda.max_memory_allocated(), **resources())
        with (OUT / f'{split}-causal-progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(payload) + '\n')
        print(json.dumps(payload), flush=True)
        assert elapsed < 15*60 and (payload.get('shared_gpu_bytes') or 0) < 1 << 30
        assert payload['ram_available_bytes'] > 2 << 30 and payload['commit_available_bytes'] > 4 << 30

    def emit(key, values):
        saved.setdefault(key, []).append(values.float().cpu().numpy())

    def block(index):
        layer = Runtime.block(rt, index)
        if index != BLOCK:
            return layer
        att = layer.self_attn
        with np.load(OUT / 'predictor.npz') as fit:
            means = {k: t.tensor(fit['mu_' + k], device='cuda') for k in ('q', 'k', 'v')}
            sparse = {k: t.tensor(fit[f'native128_idx_{k}'], device='cuda') for k in ('q', 'k', 'v')}
            control = t.tensor(fit['random128_idx_q'], device='cuda')
        units = t.tensor(seal['units'], device='cuda')
        w = dict(q=att.q_proj.weight[HEAD_SLICE].float(), k=att.k_proj.weight[KV_SLICE].float(),
                 v=att.v_proj.weight[KV_SLICE].float(), o=att.o_proj.weight[:, HEAD_SLICE].float())
        compiled = {k: (w[k] @ means[k], w[k][:, sparse[k]].contiguous()) for k in ('q', 'k', 'v')}
        control_w = w['q'][:, control].contiguous()
        wg, wu = layer.mlp.gate_proj.weight[units], layer.mlp.up_proj.weight[units]
        state = {'offset': 0}

        def norm_query(x):
            z = x.float()
            return att.q_norm.weight * (z * t.rsqrt(z.square().mean(-1, keepdim=True) + att.q_norm.variance_epsilon)).bfloat16()

        def rotation(x, cos, sin):
            return x * cos + rt.mod.rotate_half(x) * sin

        def prepare(_, args, kwargs):
            z = kwargs['hidden_states']
            batch = rows[state['offset']:state['offset']+len(z)]
            state.update(z=z, mask=kwargs['attention_mask'][:, 0, -1], pe=kwargs['position_embeddings'], batch=batch)

        def edit_q(_, args, output):
            z = state['z'][:, -1]
            native = output[:, -1, HEAD_SLICE].clone()
            q0 = norm_query(native)
            reserve = F.linear(z[:, sparse['q']].float() - means['q'][sparse['q']], compiled['q'][1])
            random = F.linear(z[:, control].float() - means['q'][control], control_w)
            target_delta = -DOSE * reserve
            base_dose = (norm_query((native.float()+target_delta).bfloat16()).float()-q0.float()).norm(dim=-1)
            ratio = target_delta.norm(dim=-1) / random.norm(dim=-1).clamp_min(1e-8)
            scales = ratio[:, None] * t.linspace(-2.5, 2.5, 81, device='cuda').exp()[None]
            proposals = -scales[..., None] * random[:, None]
            doses = (norm_query((native[:, None].float()+proposals).bfloat16()).float()-q0[:, None].float()).norm(dim=-1)
            best = (doses - base_dose[:, None]).abs().argmin(1)
            rr = t.arange(len(native), device='cuda')
            random_delta = proposals[rr, best]
            delta = t.zeros_like(target_delta)
            target_ids = [i for i, row in enumerate(state['batch']) if row['arm'] in ('target', 'restore')]
            random_ids = [i for i, row in enumerate(state['batch']) if row['arm'] == 'random']
            delta[target_ids] = target_delta[target_ids]
            delta[random_ids] = random_delta[random_ids]
            edited = output.clone()
            edited[:, -1, HEAD_SLICE] = (native.float() + delta).bfloat16()
            effective = delta.clone()
            restore_ids = [i for i, row in enumerate(state['batch']) if row['arm'] == 'restore']
            effective[restore_ids] = 0
            state.update(q_original=q0, restore_ids=restore_ids, delta=delta, effective=effective)
            emit('random_scale', scales[rr, best])
            emit('applied_q_pre_norm', edited[:, -1, HEAD_SLICE].float()-native.float())
            return edited

        def restore_q(_, args, output):
            post = output.clone()
            restore = state['restore_ids']
            post[restore, -1, HEAD] = state['q_original'][restore]
            state['q_actual'] = post[:, -1, HEAD].clone()
            return post

        def capture_k(_, args, output):
            state['k_actual'] = output[:, :, KV].contiguous()

        def capture_v(_, args, output):
            state['v_actual'] = output[..., KV_SLICE].contiguous()

        def route(qnorm, knorm, v):
            cos, sin = state['pe']
            q = rotation(qnorm, cos[:, -1], sin[:, -1])
            k = rotation(knorm, cos, sin)
            score = (q[:, None].float()*k.float()).sum(-1) * att.scaling
            mass = score.masked_fill(~state['mask'], float('-inf')).softmax(-1)
            record_mass = (state['records']*mass[:, None]).sum(-1)
            return record_mass, (mass[..., None] * v.float()).sum(1), q

        def collect_attention(_, args):
            raw = args[0][:, -1]
            z = state['z']
            records = t.zeros((len(z), 4, z.shape[1]), device='cuda')
            for j, row in enumerate(state['batch']):
                pad = z.shape[1] - len(row['tokens'])
                for k, sites in enumerate(row['records']):
                    records[j, k, [pad + i for i in sites]] = 1
            state['records'] = records
            actual_mass, head, actual_q = route(state['q_actual'], state['k_actual'], state['v_actual'])
            q_init = rotation(state['q_original'], state['pe'][0][:, -1], state['pe'][1][:, -1])
            real_head = raw[:, HEAD_SLICE].float()
            emit('q_dose', (actual_q.float()-q_init.float()).norm(dim=-1))
            emit('q_actual', actual_q)
            emit('actual_mass', actual_mass)
            emit('actual_head', real_head)
            emit('sdpa_error', (head-real_head).abs().amax(dim=-1))
            other = raw.clone()
            other[:, HEAD_SLICE] = 0
            other_write = F.linear(other, att.o_proj.weight)
            projected = {}
            for key, (intercept, weight) in compiled.items():
                x = z[:, -1] if key == 'q' else z
                projected[key] = F.linear(x[..., sparse[key]].float() - means[key][sparse[key]], weight) + intercept
            mass, prediction, _ = route(norm_query((projected['q']+state['effective']).bfloat16()),
                                        att.k_norm(projected['k'].bfloat16()), projected['v'].bfloat16())
            compact_head = prediction.bfloat16().float()
            compact_write = F.linear(compact_head, w['o'])
            xhat = layer.post_attention_layernorm(state['res'] + (other_write.float()+compact_write).bfloat16())
            g, u = F.linear(xhat, wg), F.linear(xhat, wu)
            emit('compact_mass', mass)
            emit('compact_head', compact_head)
            emit('compact_gua', t.stack([g.float(), u.float(), (F.silu(g)*u).float()], dim=1))
            state['offset'] += len(z)

        def get_a(_, args):
            a = args[0][:, -1, units]
            emit('actual_a', a)
            emit('actual_gua', t.stack([state['native_g'], state['native_u'], a], dim=1))

        def capture_mlp(name):
            def hook(_, args, output):
                state['native_' + name] = output[:, -1, units]
            return hook

        def residual(_, args):
            state['res'] = args[0][:, -1].clone()

        layer.register_forward_pre_hook(residual)
        att.register_forward_pre_hook(prepare, with_kwargs=True)
        att.q_proj.register_forward_hook(edit_q)
        att.q_norm.register_forward_hook(restore_q)
        att.k_norm.register_forward_hook(capture_k)
        att.v_proj.register_forward_hook(capture_v)
        att.o_proj.register_forward_pre_hook(collect_attention)
        layer.mlp.gate_proj.register_forward_hook(capture_mlp('g'))
        layer.mlp.up_proj.register_forward_hook(capture_mlp('u'))
        layer.mlp.down_proj.register_forward_pre_hook(get_a)
        return layer

    def progress(index, layer, x, y, batch, width, offset=0, **kwargs):
        if index >= 0 and offset + len(batch) == len(rows) and (index % 5 == 4 or index == BLOCK):
            event('native_intervention_forward', index+1)

    rt.block = block
    event('verify_snapshot', 0)
    assert snapshot(MODELS['qwen14']) == prereg['provenance']['model_revision']
    with GpuLease(), t.inference_mode():
        _, logits = rt.forward(rows, progress)
        scores = rt.score(logits, rows)
        outcomes = logits.reshape(count, arms, -1)
        bitwise = {ARMS[k]: bool(t.equal(outcomes[:, 0].view(t.int32), outcomes[:, k].view(t.int32))) for k in (3, 4)}
        kl = []
        for z in outcomes.split(4):
            lp = z.log_softmax(-1)
            kl.append((lp[:, :1].exp() * (lp[:, :1] - lp)).sum(-1).cpu().numpy())
        del outcomes, logits
    traces = {key: np.concatenate(chunks).reshape(count, arms, *chunks[0].shape[1:]) for key, chunks in saved.items()}
    traces['KL_base_to_arm'] = np.concatenate(kl)
    traces['ids'] = np.array([row['id'] for row in selected_rows])
    gold = np.array([r['target_record'] for r in selected_rows])
    traces['gold_record'] = gold
    for key in ('q_actual', 'actual_head', 'actual_gua'):
        for arm in (3, 4):
            bitwise[ARMS[arm] + '_' + key] = bool(np.array_equal(traces[key][:, 0].view(np.uint32),
                                                               traces[key][:, arm].view(np.uint32)))
    ratios = traces['q_dose'][:, 2] / np.maximum(traces['q_dose'][:, 1], 1e-12)
    valid = (ratios >= .90) & (ratios <= 1.10)
    touched_and_restored = np.array_equal(traces['applied_q_pre_norm'][:, 1], traces['applied_q_pre_norm'][:, 4])
    technical = (all(bitwise.values()) and touched_and_restored and valid.mean() >= .95
                 and traces['sdpa_error'].max() <= CAP and np.min(traces['q_dose'][:, 1]) > 0)
    strict = np.array([r['gold_margin'] > 0 for r in scores]).reshape(count, arms)
    acc = np.array([r['top1_correct'] for r in scores]).reshape(count, arms)
    behavior_ok = min(strict[:, 0].mean(), acc[:, 0].mean()) >= .95
    technical = bool(technical and behavior_ok)

    def margin(mass):
        rr = np.arange(count)
        return (np.log(np.maximum(mass[rr, :, gold], 1e-20))
                - np.log(np.maximum(mass[rr, :, gold ^ 1], 1e-20)))

    real, modeled = margin(traces['actual_mass']), margin(traces['compact_mass'])
    delta, prediction = real[:, 1]-real[:, 0], modeled[:, 1]-modeled[:, 0]
    route_effect = real[:, 2]-real[:, 1]
    boot = np.random.default_rng(SEED+90).integers(0, count, (10000, count))

    def ci(x):
        return np.quantile(x[boot].mean(1), [.0125,.9875]).tolist()

    # Local O deltas use all 5120 native output coordinates.
    w_o = rt.weight(f'model.layers.{BLOCK}.self_attn.o_proj.weight', 'cpu')[:, HEAD_SLICE].float().numpy()
    w_true = (traces['actual_head'][:, 1]-traces['actual_head'][:, 0]) @ w_o.T
    w_compact = (traces['compact_head'][:, 1]-traces['compact_head'][:, 0]) @ w_o.T
    a_true = traces['actual_a'][:, 1]-traces['actual_a'][:, 0]
    a_model = traces['compact_gua'][:, 1, 2]-traces['compact_gua'][:, 0, 2]
    gua_delta = traces['actual_gua'][:, 1]-traces['actual_gua'][:, 0]
    predicted_gua_delta = traces['compact_gua'][:, 1]-traces['compact_gua'][:, 0]
    gold_score = np.array([s['gold_margin'] for s in scores]).reshape(count, arms)
    answer_effect = gold_score[:, 2]-gold_score[:, 1]
    match_interval = ci(route_effect)
    traces['matched_q_dose'] = valid
    traces['gold_output_margin'] = gold_score
    archive(f'{split}-causal-traces.npz', traces)
    result = dict(provenance=prereg['provenance'], causal_freeze_sha=sha(OUT/'causal-freeze.json'), split=split,
                  traces_sha=sha(OUT / f'{split}-causal-traces.npz'), families=count, technical_pass=technical,
                  behavior_pass=bool(behavior_ok), strict_by_arm=strict.mean(0).tolist(), top1_by_arm=acc.mean(0).tolist(),
                  bitwise=bitwise, touched_and_restored=bool(touched_and_restored), dose_match_fraction=float(valid.mean()),
                  dose_ratio_range=[float(ratios.min()),float(ratios.max())], median_target_dose=float(np.median(traces['q_dose'][:, 1])),
                  max_sdpa_error=float(traces['sdpa_error'].max()), matched_route_interval=match_interval,
                  matched_route_effect=float(route_effect.mean()), route_effect_admitted=bool(split=='causal' and technical and route_effect.mean() >= .1 and match_interval[0] > 0),
                  margin_change_direction_agreement=float((np.sign(delta)==np.sign(prediction)).mean()),
                  margin_change_rmse=float(np.sqrt(np.square(delta-prediction).mean())),
                  margin_actual_change=float(delta.mean()), margin_compact_change=float(prediction.mean()),
                  write_delta_mse=float(np.square(w_true-w_compact).mean()), write_delta_zero_mse=float(np.square(w_true).mean()),
                  A_delta_mse=float(np.square(a_true-a_model).mean()), A_delta_zero_mse=float(np.square(a_true).mean()),
                  GUA_delta_mse=np.square(gua_delta-predicted_gua_delta).mean((0,2)).tolist(),
                  answer_margin_matched_effect=float(answer_effect.mean()), answer_margin_matched_interval=ci(answer_effect),
                  kl_by_arm=traces['KL_base_to_arm'].mean(0).tolist(),
                  gold_source_top1_by_arm=(traces['actual_mass'].argmax(-1)==gold[:,None]).mean(0).tolist(),
                  elapsed=time.monotonic()-clock, scores=scores)
    commit(f'{split}-causal-summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'scores'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('freeze', 'calibration', 'causal'))
    stage = parser.parse_args().stage
    freeze() if stage == 'freeze' else run(stage)
