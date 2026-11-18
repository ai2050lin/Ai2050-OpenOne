"""Freeze fresh role-binding texts, paired power, model and source identity before execution."""
import argparse
from datetime import datetime
from importlib.metadata import version
from itertools import product
from pathlib import Path
import random
import subprocess
import sys

import mlp_tail_goodfire as atlas
from shared.dist_protocol.natural_register_runner import PATTERNS, snapshot
from tools import provenance
from tools.power_analysis import check_phase

ROOT, MODEL, SNAPSHOT = atlas.ROOT, atlas.MODEL, atlas.SNAPSHOT
OUT = ROOT / 'tests/glm5/result/mlp_b34_confirm_v1'
RUN, SEED = 'MLP-B34-B35-ROLE-CONFIRM-v1', 20261108034
READ, HASH, SAVE = atlas.read, atlas.file_sha, atlas.save_json
CAP = ROOT / 'tests/glm5/result/phase2773_w6_v2/capability-inputs.json'
N_DISCOVERY, N_HOLDOUT, GROUP = 224, 768, 32
ALPHA_UNIT, ALPHA_PRIMARY = .02 / (2 * GROUP), .03 / 4
DOSES = (.5, 1., 2., 4.)
PRIMARY = 2.
SOURCE_FILES = ('mlp_b34_confirm_design.py', 'mlp_b34_confirm_run.py', 'mlp_b34_confirm_analyze.py')

FRAMES = {
    'discovery': (
        ('depot', 'The depot entered {src} as the sending account and {dst} as the receiving account.', 'Is {who} the account designated to receive this shipment?'),
        ('museum', 'The museum register says that {dst} is the recipient of the parcel mailed by {src}.', 'Does {who} receive the parcel according to the museum register?'),
        ('rail', 'For this wagon, the rail office records the originator as {src}. The recipient is {dst}.', 'Is {who} the named recipient of the wagon shipment?'),
        ('lab', 'The sample is to be received by {dst}; the dispatch entry was filed by {src}.', 'Is the sample addressed to {who}?'),
    ),
    'holdout': (
        ('coast', 'The coastal ledger lists {dst} in the destination field. The origin field contains {src}.', 'Does the consignment go to {who} under this entry?'),
        ('print', 'The proof packet leaves account {src}. Its intended receiving account is {dst}.', 'Is delivery of the proof packet assigned to {who}?'),
        ('archive', 'The return label names {dst} as the party taking delivery. Account {src} sent the folder.', 'Will {who} take delivery of this folder according to the label?'),
        ('garden', 'The dispatch slip assigns {src} the sender role; it gives {dst} the recipient role.', 'For the seed parcel, is {who} listed on the receiving side?'),
    ),
}

BACKGROUND = {
    'discovery': (
        'The brass tally on the wall is used during the morning review. The attendant compares each count with the register before the floor opens to visitors.',
        'A separate sheet describes the condition of the platform and the type of packing used. Those remarks are kept with the maintenance notes and carry no delivery instructions.',
        'The passage beside the office is kept clear. Staff place empty sleeves in a box and leave the ledger on the shelf between readings.',
        'The review is made in daylight when possible. The clerk keeps one copy for the next shift and files another with the dated inspection papers.',
    ),
    'holdout': (
        'At the end of the afternoon, the supervisor inventories the clips, spare envelopes and lamp oil. The figures are written in a small notebook near the service window.',
        'The building schedule sets aside time for cleaning and for checking the outer doors. It gives no account names and adds nothing to the transfer record.',
        'The paper surface is checked for folds before it is filed. A worn label can be copied onto a clean sheet, with the original filed under the same serial entry.',
        'A reference card gives the hours of the counter and explains where the spare forms are kept. The staff review that card when the timetable changes.',
    ),
}


def git(*args):
    return subprocess.check_output(['git', '-c', f'safe.directory={ROOT.as_posix()}', *args], cwd=ROOT, text=True).strip()


def power_plan(cap_n):
    from scipy.stats import binom

    plan = {
        'discovery_direction': check_phase(.15, N_DISCOVERY, .5, .05),
        'unit_direction': check_phase(.12, N_HOLDOUT, .5, ALPHA_UNIT),
        'heldout_groups': check_phase(.10, N_HOLDOUT, .35, ALPHA_PRIMARY),
        'edit_and_rescue': check_phase(.10, N_HOLDOUT, .35, ALPHA_PRIMARY),
        'capability_loss_detection': check_phase(.025, cap_n, .1, .025),
    }
    cutoff = int(binom.ppf(.025, cap_n, .02))
    while binom.cdf(cutoff, cap_n, .02) > .025:
        cutoff -= 1
    power = float(binom.cdf(cutoff, cap_n, .005))
    n_required = next(n for n in range(1, cap_n + 1)
        if binom.cdf(int(binom.ppf(.025, n, .02)) - 1, n, .005) >= .8)
    plan['capability_noninferiority'] = {'expected_loss': .005, 'margin': .02, 'alpha_each_sign': .025,
        'n': cap_n, 'n_required_conservative': n_required, 'max_loss_count': cutoff, 'power': power, 'pass': power >= .8}
    # Paired continuous diagnostics are an exhaustive decomposition on fixed material, not significance tests.
    if not all(row['pass'] for row in plan.values()):
        raise ValueError(f'Power requirement failed: {plan}')
    return plan


def materials(tokenizer, split, n, rng):
    def encode(prompt):
        return tokenizer.encode(prompt, add_special_tokens=False)

    frames = FRAMES[split]
    strata = list(product(range(len(frames)), (False, True), (False, True), (False, True)))
    schedule = (strata * ((n + len(strata) - 1) // len(strata)))[:n]
    rng.shuffle(schedule)
    for i, (frame, reverse, flip, late) in enumerate(schedule):
        topic, record, question = frames[frame]
        names = (f'[{"D" if split == "discovery" else "H"}{i:04d}X]', f'[{"D" if split == "discovery" else "H"}{i:04d}Y]')
        query, other = names[flip], names[not flip]
        material = []
        for j in range(18 + rng.randrange(5)):
            material.append(f'Note {j+1}: ' + rng.choice(BACKGROUND[split]))
        insert_at = (len(material)-2) if late else 2
        prompts, ids = [], []
        for received in (False, True):
            src, dst = (other, query) if received else (query, other)
            entries = material[:insert_at] + [record.format(src=src, dst=dst)] + material[insert_at:]
            codes = 'A = no, B = yes.' if reverse else 'A = yes, B = no.'
            question_text = '\n'.join([f'File {names[0][1:-2]} / {topic}. Read the record and answer the final question.', *entries,
                question.format(who=query), 'Choose exactly one letter. ' + codes])
            prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question_text}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False) + 'Answer:'
            prompts.append(prompt)
            ids.append(encode(prompt))
        if len(ids[0]) != len(ids[1]) or not 510 <= len(ids[0]) <= 1000 or ids[0][-1] != ids[1][-1]:
            raise ValueError(f'Invalid pair shape {split}/{i}: {[len(x) for x in ids]}')
        yield {'id': f'{split}-{i:04d}', 'split': split, 'frame': topic, 'reverse_code': reverse, 'query_is_y': flip,
               'record_late': late, 'recipient_index': int((i // 2 + i) % 2), 'prompts': prompts, 'tokens': ids,
               'correct': [int(not reverse), int(reverse)]}


def freeze():
    from transformers import AutoTokenizer

    if (OUT / 'prereg.json').exists():
        raise RuntimeError('Run already frozen. Use a new run ID for revised research.')
    OUT.mkdir(parents=True, exist_ok=True)
    print('freeze: hashing model, source and material identity', flush=True)
    model_files = {p.name: {'sha256': HASH(p), 'size': p.stat().st_size}
                   for p in sorted({p for glob in PATTERNS for p in MODEL.glob(glob) if p.is_file()})}
    import hashlib
    import json
    model_id = hashlib.sha256(json.dumps(model_files, sort_keys=True).encode()).hexdigest()
    if model_id != SNAPSHOT:
        raise ValueError('Model snapshot differs from the atlas')
    revision = git('rev-parse', 'HEAD')
    prov = {'git_sha': revision, 'git_dirty': bool(git('status', '--porcelain')), 'model_path': str(MODEL.resolve()),
            'model_revision': model_id, 'seed': SEED, 'tool': RUN, 'created_at': datetime.now().astimezone().isoformat()}
    assert not provenance.check_provenance({'provenance': prov})
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    choices = [tokenizer.encode(' '+a, add_special_tokens=False) for a in 'ABCDE']
    assert all(len(x) == 1 for x in choices)
    rng = random.Random(SEED)
    inputs = {}
    for split, n in [('discovery', N_DISCOVERY), ('holdout', N_HOLDOUT)]:
        rows = list(materials(tokenizer, split, n, rng))
        SAVE(OUT / f'{split}-materials.json', {'provenance': prov, 'items': rows})
        inputs[split] = {'sha256': HASH(OUT / f'{split}-materials.json'), 'n': len(rows),
                         'tokens': [min(len(x['tokens'][0]) for x in rows), max(len(x['tokens'][0]) for x in rows)]}
    cap = [x for x in READ(CAP) if x['gate_eligible']]
    assert len({x['id'] for x in cap}) == len(cap)
    plan = {'schema': 'mlp_b34_confirm_plan.v1', 'kind': 'real', 'run_id': RUN, 'provenance': prov, 'model_files': model_files,
        'software': {p: version(p) for p in ('torch', 'transformers', 'numpy', 'scipy')},
        'sources': {n: HASH(Path(__file__).with_name(n)) for n in SOURCE_FILES}, 'inputs': inputs,
        'capability': {'sha256': HASH(CAP), 'n': len(cap), 'pool': 'historical keyed mixed pool, fixed eligibility, repeat monitor',
                       'loss_margin': .02, 'two_signs': True, 'alpha_each_sign': .025},
        'historical_predictors': {f'B{i}-B{i+1}-predictor.npz': HASH(atlas.OUT / f'B{i}-B{i+1}-predictor.npz') for i in range(32, 35)},
        'power': power_plan(len(cap)), 'alpha_unit': ALPHA_UNIT, 'alpha_primary': ALPHA_PRIMARY,
        'calibration': {'pair_base_accuracy_min': .85, 'noop_logits_max_abs': .001, 'native_boundary_rebuild_fvu_max': 1e-6},
        'semantic': {'group_n': GROUP, 'unit_sign_accuracy_min': .62, 'each_code_accuracy_min': .55,
                     'group_delta_over_random_min': .05, 'method': 'discovery rank by paired mean divided by paired RMS, fixed orientations, frozen random group, exact paired sign tests',
                     'scope': 'Role binding in fresh authored English long records with disjoint frames and codes. Natural language meanings beyond this contrast remain open.'},
        'causal': {'dose': list(DOSES), 'primary_dose': PRIMARY, 'delta_over_control_min': .05, 'norm_relative_max': .01,
                   'source': '±dose times discovery mean true-minus-false A34, selected units, last token, exact native down_proj GEMM',
                   'target_rescue': 'reset selected A35 coordinates to unedited recipient, compare equal-norm residual sham after B35',
                   'control': 'seeded Gaussian residual directions, norm after BF16 rounding, fixed search, invalid retained',
                   'behavior': 'two-code logit argmax among A/B, desired switch over base; gated by baseline correctness and usable paired n'},
        'diagnostic': ['frozen old predictors one-step vs recursive B32-B35 on both held-out members',
                       'true G/V substitution, squared-error concentration, out-of-discovery-range rates',
                       'B35 native G/V replay from residual plus attention, drop carry and attention separately, keep frozen RMS normalization',
                       'observed state replay is an information intervention and not independent prediction'],
        'choice_ids': [x[0] for x in choices], 'precision': 'CUDA BF16 / SDPA / no KV / unpadded pair batches',
        'resources': {'max_gpu_minutes_per_stage': 35, 'new_disk_gib_max': 2},
        'custody': 'Tokenization and hashing only until discovery-derived candidate SHA is frozen. Runner refuses lockbox without matching selection. New authored materials; model outputs unknown at freeze.'}
    SAVE(OUT / 'prereg.json', plan)
    print({'frozen_sha256': HASH(OUT / 'prereg.json'), 'inputs': inputs, 'power': plan['power']}, flush=True)


def checked_plan():
    plan = READ(OUT / 'prereg.json')
    for name, sha in plan['sources'].items():
        if HASH(Path(__file__).with_name(name)) != sha:
            raise ValueError(f'Frozen code changed: {name}')
    if snapshot(MODEL) != plan['provenance']['model_revision'] or not all(x['pass'] for x in plan['power'].values()):
        raise ValueError('Model identity or preregistered power failed')
    return plan


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze'])
    parser.parse_args()
    freeze()
