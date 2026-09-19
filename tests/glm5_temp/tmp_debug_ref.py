import os
import sys
import time
import gc
import torch
from pathlib import Path

sys.path.insert(0, str(Path('tests').resolve() / 'glm5'))

import phase2691_crossmodel_role_confirmation as run
import phase2683_crossmodel_function_atlas as atlas
from phase2677_padded_native_runtime import PaddedCapture, padded_inputs
import importlib.util

spec = importlib.util.spec_from_file_location(
    'phase2691_resource_runner',
    str(Path('tests/glm5_temp/phase2691_resource_runner.py').resolve()),
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
load_q14 = runner.load_q14


start_total = time.time()
print('start', time.strftime('%H:%M:%S'), flush=True)

# prepare/cached materials
cfg = run.prepare()
folder = Path('tests/glm5/result/phase2691_crossmodel_role_confirmation/qwen14')
rows = read_rows = []
all_rows = __import__('json').loads((folder / 'material' / 'cases.json').read_text(encoding='utf-8'))
rows = [r for r in all_rows if r['family'] == 'reference' and r['language'] == 'en']
print('rows', len(rows), 'first case', rows[0]['case_id'], 'last case', rows[-1]['case_id'], flush=True)
rows = rows[:4]

print('loading model...', flush=True)
model, tok = load_q14('qwen14')
print('model_loaded', flush=True)
print('dtype', model.dtype, 'hf_device_map keys sample', list(model.hf_device_map.items())[:5], flush=True)

# use calibration (64 rows) to mirror full run setup
cal_rows = __import__('json').loads((folder / 'material' / 'calibration.json').read_text(encoding='utf-8'))
cap = PaddedCapture(model, ())
FORCE_SMALL_TOKENS = int(__import__('os').environ.get('DEBUG_SMALL_TOKENS', '0'))

def _padded_len(model, token_count):
    if FORCE_SMALL_TOKENS:
        return FORCE_SMALL_TOKENS
    total = 256
    return total

for idx, r in enumerate(cal_rows[:2], 1):
    total = _padded_len(model, len(r['prompt_ids']))
    t0 = time.time()
    baseline = model.model(**padded_inputs(model, r['prompt_ids'], tok.eos_token_id, total)).last_hidden_state.detach().cpu()
    cap.reset(r['body_end_token'], False, len(r['prompt_ids']) - 1)
    cap.enabled = True
    out = model.model(**padded_inputs(model, r['prompt_ids'], tok.eos_token_id, total)).last_hidden_state.detach().cpu()
    cap.enabled = False
    restored = model.model(**padded_inputs(model, r['prompt_ids'], tok.eos_token_id, total)).last_hidden_state.detach().cpu()
    assert torch.equal(baseline, out) and torch.equal(baseline, restored)
    print(f'cal {idx}/2 elapsed {time.time()-t0:.2f}s', flush=True)

print('start main rows', flush=True)
gen = atlas.calibrate(model, tok, 'qwen14', folder)
FORWARD_TOKENS = int(__import__('os').environ.get('FORWARD_TOKENS', '0'))

for ri, rr in enumerate(rows, 1):
    t0 = time.time()
    ids = rr['prompt_ids']
    field_task = len(ids) - 1
    total = _padded_len(model, len(ids))
    if FORWARD_TOKENS:
        total = FORWARD_TOKENS
    cap.reset(rr['body_end_token'], rr['published'], field_task)
    cap.enabled = True
    out = model.model(**padded_inputs(model, ids, tok.eos_token_id, total))
    cap.enabled = False
    t1 = time.time()
    field_state = out.last_hidden_state[0, field_task].detach().cpu().clone()
    pack = cap.pack()
    t2 = time.time()
    plain = model.model(input_ids=torch.tensor([ids], device=model.get_input_embeddings().weight.device), use_cache=False).last_hidden_state[0, -1]
    gap = float((plain.detach().cpu().float() - field_state.float()).abs().max())
    t3 = time.time()
    del plain
    generation = atlas.generate(model, tok, rr, 'qwen14', gen['max_new_tokens'])
    t4 = time.time()
    print(f'case {ri} id={rr["case_id"]} baseline {t1-t0:.2f}s pack {t2-t1:.2f}s generate {t4-t3:.2f}s gap {gap:.6f}', flush=True)

cap.close()
del model
gc.collect()
torch.cuda.empty_cache()
print('done', 'total', time.time()-start_total, flush=True)
