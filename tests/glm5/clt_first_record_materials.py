"""Prepare and verify tokenized materials. This module never loads a model or runs GPU work."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import re

import numpy as np

import clt_b26_three_cues_v3 as template
from clt_b26_three_cues_close import check_material
from deploy.storage_policy import checked_dir
from deploy.test_archive_client import save
from shared.dist_protocol.test_archive import digest
from source_binding_design import MODELS, WORDS, read, sha
from tools.power_analysis import check_phase

B = template.B
OUT = B.HERE / 'result/clt_first_record_path_v1/materials'
DISCOVERY = B.HERE / 'result/clt_b26_competition_v1'
SEED = 202611190315
SPLITS = {'pilot': 32, 'confirm': 192, 'intervention': 192}
MODEL = MODELS['qwen14']
TOKENIZER_FILES = ('tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'chat_template.jinja')
RECORD = re.compile(r'^ROW [A-E] \| NAME \w+ \| CODE (\w+)$', re.M)


def tokenizer_identity():
    return {name: sha(MODEL/name) for name in TOKENIZER_FILES if (MODEL/name).is_file()}


def historical_names():
    # Importing the first-only template makes its old output B.OUT.
    return B.previous_names() | {
        name for path in B.OUT.glob('*-materials.json')
        for row in read(path)[::12] for name in row['names']
    }


def retokenize(tok, row, frame):
    prompt = row['prompt']
    question = prompt.index('Question:')
    if frame == 'paraphrase':
        user_end = prompt.index('<|im_end|>', question)
        stem = 'Question: Return the CODE from the record'
        selector = (
            f" with NAME {row['names'][0]}",
            f" with the printed ROW {row['printed'][row['cue_targets'][1]]}",
            ' directly following the line BEGIN',
        )[row['rule']]
        prompt = prompt[:question] + stem + selector + '. Give only the code word.' + prompt[user_end:]
        boundary = question + len(stem)
    else:
        boundary = question + len('Question: In the active archive, find the CODE')

    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc['offset_mapping']
    spans = dict(zip(row['physical_order'], RECORD.finditer(prompt[:question]), strict=True))
    records, code_sites = [], []
    for record in range(5):
        match = spans[record]
        records.append([i for i, (a, b) in enumerate(offsets) if a < match.end() and b > match.start()])
        code_sites.append([i for i, (a, b) in enumerate(offsets) if a < match.end(1) and b > match.start(1)])
        if not code_sites[-1] or not set(code_sites[-1]).issubset(records[-1]):
            raise ValueError(f"unaligned code span for {row['id']}")
    if len(set(sum(records, []))) != sum(map(len, records)):
        raise ValueError(f"overlapping text spans for {row['id']}")
    return dict(row, prompt=prompt, tokens=enc['input_ids'], records=records, code_sites=code_sites, frame=frame,
                endpoint=max(i for i, (a, b) in enumerate(offsets) if a < b <= boundary))


def build_split(tok, split, count, rng, used):
    rows = []
    for i in range(count):
        # Keep both frame and archive title balanced in each new split.
        frame = 'paraphrase' if (i//2) % 2 else 'original'
        for _ in range(512):
            names = [''.join(rng.choice(list('bdfgklmnprstvz')) + rng.choice(list('aeiou'))
                             for _ in range(3)).capitalize() for _ in range(5)]
            if len(set(names)) != 5 or used.intersection(names):
                continue
            if any(len(tok.encode(' '+name, add_special_tokens=False)) != 3 for name in names):
                continue
            raw = template.first_only(tok, split, i, names, list(map(str, rng.choice(WORDS, 5, replace=False))), rng)
            if raw is None:
                continue
            family = [retokenize(tok, row, frame) for row in raw]
            triples = zip(family[::3], family[1::3], family[2::3], strict=True)
            if not all(len({(len(r['tokens']), r['endpoint'], tuple(r['tokens'][:r['endpoint']+1]))
                            for r in triple}) == 1 for triple in triples):
                continue
            rows.extend(family)
            used.update(names)
            break
        else:
            raise RuntimeError(f'length-matched family unavailable for {split}/{i}/{frame}')
    return rows


def validate_rows(rows, count, used):
    if len(rows) != 12*count or not all(row['first'] for row in rows):
        raise ValueError('expected 12 first-line conditions per family')
    check_material(rows)
    names = {name for row in rows[::12] for name in row['names']}
    if len(names) != 5*count or not names.isdisjoint(used):
        raise ValueError('family overlap')
    frames = Counter((row['frame'], row['title']) for row in rows[::12])
    if len(frames) != 4 or set(frames.values()) != {count//4}:
        raise ValueError('unbalanced frame/title')
    # The new framing is an input family partition, not a new source of discovery data.
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('duplicate sample ID')
    for row in rows:
        if not all(code and set(code).issubset(span)
                   for code, span in zip(row['code_sites'], row['records'], strict=True)):
            raise ValueError(f"unmatched code span for {row['id']}")
    used.update(names)
    return {f'{frame}/{title}': n for (frame, title), n in sorted(frames.items())}


def token_digest(rows):
    return digest([[r['id'], r['tokens']] for r in rows])


def source_identity():
    result = read(DISCOVERY/'verified.json')
    path = DISCOVERY/'confirm-materials.json'
    if sha(path) != result['files'][path.name]:
        raise ValueError('Phase 2990 material differs from the original receipt')
    return {'material_sha256': sha(path), 'receipt_sha256': sha(DISCOVERY/'verified.json'),
            'use': 'previously exposed discovery; counterexample slice assigned only after research preflight'}


def prepare(out):
    from transformers import AutoTokenizer
    from shared.dist_protocol.natural_register_runner import snapshot

    out = checked_dir(out, 96*2**20, operation='First-record protocol material')
    if any(out.iterdir()):
        raise FileExistsError(f'material folder is not empty: {out}')
    revision = snapshot(MODEL)
    if revision != read(DISCOVERY/'verified.json')['provenance']['model_revision']:
        raise ValueError('model snapshot differs from Phase 2990')
    source = source_identity()
    tokenizer_files = tokenizer_identity()
    if 'tokenizer.json' not in tokenizer_files or 'tokenizer_config.json' not in tokenizer_files:
        raise ValueError('missing tokenizer file')
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rng = np.random.default_rng(SEED)
    old = historical_names()
    used = set(old)
    documents, splits = {}, {}
    for split, count in SPLITS.items():
        rows = build_split(tok, split, count, rng, used)
        isolated = old | {name for batch in documents.values() for r in batch[::12] for name in r['names']}
        frames = validate_rows(rows, count, isolated)
        documents[split] = rows
        splits[split] = {'families': count, 'inputs': len(rows), 'frames_by_archive': frames,
                         'input_tokens_sha256': token_digest(rows), 'file': f'{split}.json'}
        print(f'{split}: {count} families / {len(rows)} tokenized cases', flush=True)
    power = {str(size): check_phase(.15, size, p_d=.25, alpha=.025) for size in (96, 192)}
    for split, rows in documents.items():
        path = out/splits[split]['file']
        save(path, rows)
        splits[split]['material_sha256'] = sha(path)
    save(out/'manifest.json', {
        'schema': 'clt_first_record_materials.v1', 'utc': datetime.now(timezone.utc).isoformat(),
        'seed': SEED, 'model_snapshot': revision, 'tokenizer_files': tokenizer_files,
        'tokenizer_sha256': digest(tokenizer_files), 'splits': splits, 'source_discovery': source,
        'power_example_15_points': power, 'builder_sha256': sha(Path(__file__)),
        'status': 'tokenized_material_only', 'behavior_measured': False, 'server_preflight': None,
    })


def verify(out):
    manifest = read(out/'manifest.json')
    if manifest['schema'] != 'clt_first_record_materials.v1' or manifest['source_discovery'] != source_identity():
        raise ValueError('material identity differs')
    if manifest['tokenizer_files'] != tokenizer_identity() or manifest['builder_sha256'] != sha(Path(__file__)):
        raise ValueError('tokenizer or builder identity differs')
    used = historical_names()
    for split, count in SPLITS.items():
        part = manifest['splits'][split]
        file = out/part['file']
        if sha(file) != part['material_sha256']:
            raise ValueError(f'{split} content hash differs')
        rows = read(file)
        checks = validate_rows(rows, count, used)
        if token_digest(rows) != part['input_tokens_sha256'] or checks != part['frames_by_archive']:
            raise ValueError(f'{split} tokens or frame families differ')
    print('Verified new material hashes, 12-cell factorial, causal prefix alignment and family isolation.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'verify'))
    options = parser.parse_args()
    {'prepare': prepare, 'verify': verify}[options.action](OUT)
