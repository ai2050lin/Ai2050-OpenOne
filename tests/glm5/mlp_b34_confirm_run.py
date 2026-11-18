"""Run the frozen BF16 pair assay and monitor the same native edit on keyed tasks."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

import mlp_b34_confirm_design as design
from shared.dist_protocol.mechanism_trace import GpuLease

OUT, READ, SAVE, HASH = design.OUT, design.READ, design.SAVE, design.HASH


def log(stage, done, total, started, **extra):
    elapsed = time.monotonic() - started
    print(json.dumps({'stage': stage, 'done': done, 'total': total, 'seconds': round(elapsed, 2),
        'eta_seconds': round(elapsed * (total-done) / done, 2) if done else None,
        'cuda_allocated_mib': round(torch.cuda.memory_allocated()/2**20),
        'cuda_reserved_mib': round(torch.cuda.memory_reserved()/2**20),
        'cuda_peak_mib': round(torch.cuda.max_memory_allocated()/2**20),
        'shared_gpu_memory_mib': None, 'shared_memory_scope': 'WDDM process counter unavailable in runner', **extra}), flush=True)


def bits(tensor):
    return tensor.contiguous().view(torch.uint16).cpu().numpy()


def choices(logits, ids):
    return logits[:, 0, ids].float().cpu().numpy()


def norm_match(before, after, rng):
    """Pick the closest of eight frozen random directions after BF16 rounding."""
    origin = before.float()
    norm = torch.linalg.vector_norm(after.float() - origin)
    directions = torch.randn((8, origin.numel()), generator=rng, device=origin.device)
    directions /= directions.norm(dim=1, keepdim=True)
    lo = torch.zeros(8, device=origin.device)
    hi = torch.full_like(lo, 2.) * norm
    nearest = origin.expand_as(directions).to(before.dtype).clone()
    error = torch.full_like(lo, float('inf'))
    for _ in range(20):
        mid = (lo + hi) / 2
        values = (origin + mid[:, None] * directions).to(before.dtype)
        measured = (values.float() - origin).norm(dim=1)
        gap = (measured - norm).abs()
        nearest = torch.where((gap < error)[:, None], values, nearest)
        error = torch.minimum(error, gap)
        lo = torch.where(measured < norm, mid, lo)
        hi = torch.where(measured < norm, hi, mid)
    best = nearest[error.argmin()]
    expected, measured = torch.stack([norm, (best.float() - origin).norm()]).tolist()
    return best, expected, measured, bool(expected > 0 and abs(expected-measured) <= .01*expected)


class NativeAssay:
    def __init__(self, model, token_ids):
        from transformers.masking_utils import create_causal_mask

        self.model = model
        self.decoder = model.model
        self.layers = self.decoder.layers
        self.ids = token_ids
        self.make_mask = create_causal_mask
        self.data = {}
        self.mode = None
        self.handles = []
        for block in (32, 33, 34, 35):
            layer = self.layers[block]
            self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(self.capture_input(f'A{block}', block)))
        for block in (34, 35):
            layer = self.layers[block]
            self.handles.extend([
                layer.post_attention_layernorm.register_forward_pre_hook(self.capture_input(f'R{block}', block)),
                layer.mlp.gate_proj.register_forward_hook(self.capture_last(f'G{block}', block)),
                layer.mlp.up_proj.register_forward_hook(self.capture_last(f'V{block}', block)),
            ])
        self.handles.extend([
            self.layers[34].register_forward_hook(self.capture_output('H34', 34)),
            self.layers[34].mlp.down_proj.register_forward_hook(self.capture_last('W34', 34)),
            self.layers[35].self_attn.register_forward_hook(self.capture_last('T35', 35, first=True)),
        ])

    def captures(self, block):
        return self.mode == 'base' or (block == 35 and self.mode == 'suffix')

    def capture_input(self, key, block):
        def hook(module, args):
            if self.captures(block):
                value = args[0]
                self.data[key] = value if key in ('A34', 'A35') else value[:, -1:].clone()
        return hook

    def capture_output(self, key, block):
        def hook(module, args, value):
            if self.captures(block):
                self.data[key] = value
        return hook

    def capture_last(self, key, block, first=False):
        def hook(module, args, output):
            if self.captures(block):
                value = output[0] if first else output
                self.data[key] = value[:, -1:].clone()
        return hook

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()

    def head(self, hidden):
        return self.model.lm_head(self.decoder.norm(hidden[:, -1:]))

    def suffix(self, hidden, mask, pos, rotary):
        self.mode = 'suffix'
        try:
            state = self.layers[35](hidden, attention_mask=mask, position_ids=pos, position_embeddings=rotary, use_cache=False)
            return state, self.head(state)
        finally:
            self.mode = None

    def observe(self, sample):
        self.data = {}
        self.mode = 'base'
        tokens = torch.tensor(sample['tokens'], device='cuda')
        logits = self.model(input_ids=tokens, use_cache=False, logits_to_keep=1).logits
        self.mode = None
        return logits, self.data.copy()

    def interventions(self, sample, base, original_logits, selection, seed):
        src, tgt = self.layers[34].mlp.down_proj, self.layers[35].mlp.down_proj
        h34 = base['H34']
        pos = torch.arange(h34.shape[1], device=h34.device)[None]
        mask = self.make_mask(config=self.model.config, inputs_embeds=h34, attention_mask=None,
                              past_key_values=None, position_ids=pos)
        rotary = self.decoder.rotary_emb(h34, pos)
        baseline_state, baseline_logits = self.suffix(h34, mask, pos, rotary)
        noop_error = float((baseline_logits.float() - original_logits.float()).abs().max())
        if noop_error > .001 or not torch.equal(base['A35'][:, -1], self.data['A35'][:, -1]):
            raise RuntimeError(f'Unedited replay differs for {sample["id"]}: {noop_error}')
        if not torch.equal((base['R34'] + src(base['A34'])[:, -1:]), h34[:, -1:]):
            raise RuntimeError(f'B34 reconstruction differs for {sample["id"]}')
        row = sample['recipient_index']
        direction = 1. if row == 0 else -1.
        units = selection['B34']['units']
        target_units = selection['B35']['units']
        delta = torch.tensor(selection['B34']['write'], device='cuda') * direction
        original = h34[row, -1]
        rng = torch.Generator(device='cuda').manual_seed(seed)
        original_choice = choices(original_logits, self.ids[:2])
        baseline_selected = base['A35'][row, -1, target_units].float()
        source_changes = []
        for dose in design.DOSES:
            edited_a = base['A34'].clone()
            edited_a[row, -1, units] = (edited_a[row, -1, units].float() + dose*delta).to(edited_a.dtype)
            modified = (base['R34'] + src(edited_a)[:, -1:])[row, 0]
            random_vec, actual, control_norm, valid = norm_match(original, modified, rng)
            condition = {'dose': dose, 'norm': actual, 'control_norm': control_norm, 'norm_valid': valid}
            edited = None
            for tag, value in [('edit', modified), ('control', random_vec)]:
                state_in = h34.clone()
                state_in[row, -1] = value
                state_out, logits = self.suffix(state_in, mask, pos, rotary)
                a35 = self.data['A35']
                condition[tag] = {'scores': choices(logits, self.ids[:2])[row].tolist(),
                    'target_delta': (a35[row, -1, target_units].float()-baseline_selected).tolist(),
                    'all_target_delta_norm': float((a35[row, -1].float()-base['A35'][row, -1].float()).norm())}
                if tag == 'edit' and dose == design.PRIMARY:
                    edited = (state_out[:, -1:].clone(), a35, self.data['R35'])
            if edited is not None:
                state_out, edit_a35, r35 = edited
                if not torch.equal(r35 + tgt(edit_a35)[:, -1:], state_out):
                    raise RuntimeError(f'B35 edited reconstruction differs for {sample["id"]}')
                patched = edit_a35.clone()
                patched[row, -1, target_units] = base['A35'][row, -1, target_units]
                rescued = r35 + tgt(patched)[:, -1:]
                sham_vec, norm, sham_norm, valid_sham = norm_match(state_out[row, 0], rescued[row, 0], rng)
                sham = state_out.clone()
                sham[row, 0] = sham_vec
                condition['rescue'] = {'scores': choices(self.head(rescued), self.ids[:2])[row].tolist(),
                    'sham_scores': choices(self.head(sham), self.ids[:2])[row].tolist(), 'norm': norm,
                    'sham_norm': sham_norm, 'norm_valid': valid_sham}
            source_changes.append(condition)
        self.mode = None
        self.data = {}
        logits = self.head(baseline_state)
        if not torch.equal(logits, baseline_logits):
            raise RuntimeError('Native restore changed unedited logits')
        return {'id': sample['id'], 'base_scores': original_choice.tolist(), 'noop_error': noop_error,
                'restore_bitwise': True, 'doses': source_changes}


def read_selection(plan):
    selection = READ(OUT / 'selection.json')
    if selection['plan_sha256'] != HASH(OUT / 'prereg.json') or not selection['baseline_gate']:
        raise ValueError('Discovery selection missing or failed calibration')
    for obj in selection['discovery_chunks']:
        if HASH(OUT / obj['path']) != obj['sha256']:
            raise ValueError('Discovery evidence changed after freeze')
    if selection['provenance'] != plan['provenance']:
        raise ValueError('Selection provenance mismatch')
    return selection


def run_pairs(model, plan, split, selection):
    path = OUT / f'{split}-materials.json'
    if HASH(path) != plan['inputs'][split]['sha256']:
        raise ValueError('Frozen material hash mismatch')
    items = READ(path)['items']
    assay = NativeAssay(model, plan['choice_ids'])
    started = time.monotonic()
    log(split, 0, len(items), started)
    try:
        for start in range(0, len(items), 16):
            prefix = OUT / f'{split}-{start:04d}'
            raw, record = prefix.with_suffix('.npz'), prefix.with_suffix('.json')
            if record.exists():
                previous = READ(record)
                if previous['raw_sha256'] != HASH(raw) or previous['selection_sha256'] != (HASH(OUT/'selection.json') if selection else None):
                    raise ValueError(f'Existing checkpoint differs: {record}')
                continue
            rows, arrays = [], []
            for i, item in enumerate(items[start:start+16], start):
                logits, base = assay.observe(item)
                fields = ('A32', 'A33', 'A34', 'A35', 'G34', 'V34', 'G35', 'V35', 'R34', 'R35', 'W34', 'H34', 'T35')
                arrays.append({k: bits(base[k][:, -1]) for k in fields})
                if selection:
                    result = assay.interventions(item, base, logits, selection, design.SEED+i)
                else:
                    result = {'id': item['id'], 'base_scores': choices(logits, plan['choice_ids'][:2]).tolist()}
                rows.append(result)
                if (i+1)%8 == 0 or i+1 == len(items):
                    log(split, i+1, len(items), started)
                if time.monotonic()-started > 60*plan['resources']['max_gpu_minutes_per_stage']:
                    raise RuntimeError('GPU budget reached; completed chunks are resumable')
            np.savez(raw, **{k: np.stack([x[k] for x in arrays]) for k in fields})
            SAVE(record, {'provenance': plan['provenance'], 'plan_sha256': HASH(OUT/'prereg.json'),
                'selection_sha256': HASH(OUT/'selection.json') if selection else None, 'raw_sha256': HASH(raw), 'rows': rows})
    finally:
        assay.remove_hooks()


def capability(model, plan, selection):
    from transformers import AutoTokenizer

    if (OUT/'capability.json').exists():
        raise RuntimeError('Frozen capability result already exists')
    if HASH(design.CAP) != plan['capability']['sha256']:
        raise ValueError('Capability pool identity mismatch')
    questions = [x for x in READ(design.CAP) if x['gate_eligible']]
    tokenizer = AutoTokenizer.from_pretrained(design.MODEL, local_files_only=True)
    tokenizer.padding_side = 'left'
    input_rows = []
    for row in questions:
        options = '\n'.join(f'{letter}. {option}' for letter, option in zip('ABCDE', row['choices']))
        question = f'{row["question"]}\n{options}\n请选择一个正确选项字母。'
        prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': question}], tokenize=False,
            add_generation_prompt=True, enable_thinking=False) + 'Answer:'
        input_rows.append({'id': row['id'], 'source': row['source'], 'tokens': tokenizer.encode(prompt, add_special_tokens=False),
                           'correct': 'ABCDE'.index(row['answer']), 'choices': len(row['choices'])})
    SAVE(OUT/'capability-materials.json', {'provenance': plan['provenance'], 'rows': input_rows})
    module = model.model.layers[34].mlp.down_proj
    units = selection['B34']['units']
    vector = torch.tensor(selection['B34']['write'], device='cuda')
    started = time.monotonic()
    results = []
    log('capability', 0, len(questions), started)
    for offset in range(0, len(input_rows), 8):
        chunk = input_rows[offset:offset+8]
        batch = tokenizer.pad({'input_ids': [x['tokens'] for x in chunk]}, return_tensors='pt').to('cuda')
        conditions = {}
        for tag, sign in [('base', 0), ('positive', 1), ('negative', -1)]:
            def patch(module, args):
                x = args[0].clone()
                x[:, -1, units] = (x[:, -1, units].float()+sign*design.PRIMARY*vector).to(x.dtype)
                return (x,)
            hook = module.register_forward_pre_hook(patch) if sign else None
            try:
                output = model(**batch, use_cache=False, logits_to_keep=1).logits
                conditions[tag] = choices(output, plan['choice_ids']).tolist()
            finally:
                if hook:
                    hook.remove()
        for i, item in enumerate(chunk):
            results.append({'id': item['id'], 'source': item['source'], 'gold': item['correct'],
                'scores': {k: v[i][:item['choices']] for k, v in conditions.items()}})
        if offset % 128 == 0 or len(results) == len(input_rows):
            log('capability', len(results), len(input_rows), started)
        if time.monotonic()-started > 60*plan['resources']['max_gpu_minutes_per_stage']:
            raise RuntimeError('Capability GPU budget reached')
    SAVE(OUT/'capability.json', {'provenance': plan['provenance'], 'plan_sha256': HASH(OUT/'prereg.json'),
        'selection_sha256': HASH(OUT/'selection.json'), 'materials_sha256': HASH(OUT/'capability-materials.json'),
        'rows': results, 'peak_cuda_mib': round(torch.cuda.max_memory_allocated()/2**20)})


def main(stage):
    from transformers import AutoModelForCausalLM

    plan = design.checked_plan()
    selection = read_selection(plan) if stage != 'discovery' else None
    torch.manual_seed(design.SEED)
    torch.set_num_threads(4)
    started = time.monotonic()
    with GpuLease(), torch.inference_mode():
        log('load', 0, 1, started, stage_requested=stage)
        model = AutoModelForCausalLM.from_pretrained(design.MODEL, local_files_only=True, dtype=torch.bfloat16,
            device_map='cuda:0', attn_implementation='sdpa').eval().requires_grad_(False)
        if any(x.dtype != torch.bfloat16 or x.device.type != 'cuda' for x in model.parameters()):
            raise RuntimeError('Model parameters must all be CUDA BF16')
        log('load', 1, 1, started, stage_requested=stage)
        if stage == 'capability':
            capability(model, plan, selection)
        else:
            run_pairs(model, plan, stage, selection)
        log('finished', 1, 1, started, stage_requested=stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['discovery', 'holdout', 'capability'])
    main(parser.parse_args().stage)
