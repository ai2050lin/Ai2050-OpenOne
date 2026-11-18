"""Execution repair: preserve the native B35 arguments and full head layout.

Quarantine the one v2 pair used to investigate zero-edit replay. The 767 other
pairs keep their original bytes and outcome-blind candidate selection.
"""
import argparse
from datetime import datetime
import os
from pathlib import Path

import mlp_b34_confirm_design as D
from mlp_b34_confirm_v2 import configure as configure_v2


def configure():
    configure_v2()
    previous = D.OUT
    D.OUT = previous.with_name('mlp_b34_confirm_v3')
    D.RUN = 'MLP-B34-B35-ROLE-CONFIRM-v3'
    D.N_HOLDOUT = 767
    D.SOURCE_FILES += ('mlp_b34_confirm_v3.py',)
    return previous


def freeze(previous):
    if (D.OUT/'prereg.json').exists():
        raise RuntimeError('Repair contract already frozen')
    if list(previous.glob('holdout-*.npz')):
        raise ValueError('Other v2 held-out observations exist')
    source = D.READ(previous/'prereg.json')
    selection = D.READ(previous/'selection.json')
    assert selection['baseline_gate'] and selection['plan_sha256'] == D.HASH(previous/'prereg.json')
    assert D.HASH(previous/'holdout-materials.json') == source['inputs']['holdout']['sha256']
    assert D.snapshot(D.MODEL) == source['provenance']['model_revision']
    for row in selection['discovery_chunks']:
        if D.HASH(previous/row['path']) != row['sha256']:
            raise ValueError('Original selection evidence changed')
        row['path'] = '../'+previous.name+'/'+row['path']
    rows = D.READ(previous/'holdout-materials.json')['items']
    dropped = rows.pop(0)
    assert dropped['id'] == 'holdout-v2-0000' and len(rows) == D.N_HOLDOUT
    D.OUT.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone().isoformat()
    prov = {**source['provenance'], 'git_sha': D.git('rev-parse', 'HEAD'), 'git_dirty': bool(D.git('status', '--porcelain')),
            'tool': D.RUN, 'created_at': now}
    D.SAVE(D.OUT/'holdout-materials.json', {'provenance': prov, 'items': rows})
    source['inputs']['holdout'].update(sha256=D.HASH(D.OUT/'holdout-materials.json'), n=len(rows),
        tokens=[min(len(s['tokens'][0]) for s in rows), max(len(s['tokens'][0]) for s in rows)])
    source['inputs']['discovery']['original_path'] = str(previous/'discovery-materials.json')
    source.update(run_id=D.RUN, provenance=prov, sources={n: D.HASH(Path(__file__).with_name(n)) for n in D.SOURCE_FILES},
        power=D.power_plan(source['capability']['n']), custody='767 v2 holdout pairs remained unexecuted after first-pair replay failure. '
        'First pair quarantined. Discovery choices, tests and tolerances unchanged. Calibrate repair only on opened cases.',
        repair={'source_prereg_sha256': D.HASH(previous/'prereg.json'), 'source_selection_sha256': D.HASH(previous/'selection.json'),
                'stopped_log_sha256': D.HASH(previous/'holdout-run.log'), 'quarantined_id': dropped['id'],
                'method': 'reuse exact native B35 forward args, native full-sequence final RMSNorm and original lm_head slice layout, deterministic CUDA workspace'})
    D.SAVE(D.OUT/'prereg.json', source)
    selection.update(provenance=prov, plan_sha256=D.HASH(D.OUT/'prereg.json'), frozen_at=now,
                     original_selection_sha256=source['repair']['source_selection_sha256'])
    D.SAVE(D.OUT/'selection.json', selection)
    print({'prereg_sha256': D.HASH(D.OUT/'prereg.json'), 'selection_sha256': D.HASH(D.OUT/'selection.json'), 'power': source['power']}, flush=True)


def configure_runner():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import mlp_b34_confirm_run as runner

    torch.use_deterministic_algorithms(True)

    class NativeLayoutAssay(runner.NativeAssay):
        def __init__(self, model, ids):
            super().__init__(model, ids)
            self.native_args = None
            self.head_reference = None
            self.handles.append(self.layers[35].register_forward_pre_hook(self.capture_b35_args, with_kwargs=True))

        def capture_b35_args(self, module, args, kwargs):
            if self.mode == 'base':
                self.native_args = (args[1:], kwargs.copy())

        def head(self, hidden):
            if hidden.shape[1] > 1:
                self.head_reference = hidden
            else:
                prefix = self.head_reference.clone()
                prefix[:, -1:] = hidden
                hidden = prefix
            return self.model.lm_head(self.decoder.norm(hidden)[:, -1:])

        def suffix(self, hidden, mask, pos, rotary):
            args, kwargs = self.native_args
            self.mode = 'suffix'
            try:
                state = self.layers[35](hidden, *args, **kwargs)
                return state, self.head(state)
            finally:
                self.mode = None

    runner.NativeAssay = NativeLayoutAssay
    return runner


def calibrate(previous, runner):
    import torch
    from transformers import AutoModelForCausalLM
    from shared.dist_protocol.mechanism_trace import GpuLease

    plan = D.checked_plan()
    selection = runner.read_selection(plan)
    rows = D.READ(previous/'discovery-materials.json')['items'][:12]
    rows.append(D.READ(previous/'holdout-materials.json')['items'][0])
    checks = []
    torch.manual_seed(D.SEED)
    torch.set_num_threads(4)
    with GpuLease(), torch.inference_mode():
        model = AutoModelForCausalLM.from_pretrained(D.MODEL, local_files_only=True, dtype=torch.bfloat16,
            device_map='cuda:0', attn_implementation='sdpa').eval().requires_grad_(False)
        assay = runner.NativeAssay(model, plan['choice_ids'])
        for i, row in enumerate(rows):
            logits, base = assay.observe(row)
            # Match the active runner's capture allocation before zero-edit replay.
            captured = [runner.bits(base[k][:, -1]) for k in base]
            result = assay.interventions(row, base, logits, selection, D.SEED+i)
            checks.append({'id': row['id'], 'max_error': result['noop_error'], 'restore_bitwise': result['restore_bitwise'],
                           'matched_doses': sum(x['norm_valid'] for x in result['doses']), 'capture_arrays': len(captured)})
        assay.remove_hooks()
    D.SAVE(D.OUT/'repair-calibration.json', {'provenance': plan['provenance'], 'plan_sha256': D.HASH(D.OUT/'prereg.json'),
        'selection_sha256': D.HASH(D.OUT/'selection.json'), 'checks': checks, 'passed': True, 'scope': 'Engineering checks on already opened cases'})
    print({'repair_checks': len(checks), 'max_error': max(x['max_error'] for x in checks)}, flush=True)


def main(stage):
    previous = configure()
    if stage == 'freeze':
        freeze(previous)
    elif stage == 'report':
        from mlp_b34_confirm_analyze import report
        report(D.checked_plan())
    else:
        runner = configure_runner()
        if stage == 'calibrate':
            calibrate(previous, runner)
        else:
            gate = D.READ(D.OUT/'repair-calibration.json')
            if not gate['passed'] or gate['plan_sha256'] != D.HASH(D.OUT/'prereg.json') or gate['selection_sha256'] != D.HASH(D.OUT/'selection.json'):
                raise RuntimeError('Repair calibration is missing or incompatible')
            runner.main(stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'calibrate', 'holdout', 'capability', 'report'])
    main(parser.parse_args().stage)
