"""Serve or verify the real Qwen3-14B archive in the built client, without inference."""
from argparse import ArgumentParser
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / 'tests/glm5/result/qwen14_full_mlp_3d'
MIDDLE = ROOT / 'tests/glm5/result/goodfire_middle_20261113_v2'
LATE = ROOT / 'tests/glm5/result/qwen14_client_v2/native'
SNAPSHOT = '1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce'
PORT = 5098
sys.path[:0] = [str(ROOT), str(ROOT / 'deploy')]

from deploy.storage_policy import checked_dir


def serve():
    app = local_app()
    import uvicorn
    print(f'Qwen3-14B read-only atlas on http://127.0.0.1:{PORT}', flush=True)
    uvicorn.run(app, host='127.0.0.1', port=PORT)


def local_app():
    scratch = checked_dir(ROOT / 'tests/glm5_temp/qwen14_full_mlp_3d', 256 * 2**20, operation='原生档案界面联调')
    assert (RESULT / 'client/index.html').is_file(), '请先构建客户端'
    os.environ.update(
        AI2050_DIST_DIR=str(scratch / 'center-data'), AI2050_NATIVE_MLP_ROOT=str(MIDDLE),
        AI2050_QWEN14_MLP_DIR=str(LATE), AI2050_LOCAL_TOOLS='0', AI2050_SNAPSHOT_DISABLE='1',
        AI2050_FRONTEND_DIST=str(RESULT / 'client'), AI2050_RESEARCH_DIR=str(scratch),
        AI2050_PUBLIC_URL=f'http://127.0.0.1:{PORT}', AI2050_DEV='1', TEMP=str(scratch), TMP=str(scratch),
    )
    # The isolated center generates no credentials used by the installed client.
    os.environ['AI2050_DIST_ADMIN_TOKEN'] = os.urandom(32).hex()
    from deploy.center_app import create_app
    return create_app()


def verify():
    import numpy as np
    base = f'http://127.0.0.1:{PORT}/api/research/workspace/models/{SNAPSHOT}/mlp-atlas'

    def get(path='', **args):
        with urlopen(base + path + ('?' + urlencode(args) if args else ''), timeout=40) as reply:
            return json.load(reply)

    catalog = get()['runs']
    assert len(catalog) == 2
    assert sorted({block for entry in catalog for block in entry['blocks']}) == list(range(18, 40))
    rows = []
    for entry, source in zip(catalog, (MIDDLE / 'qwen14', LATE)):
        assert entry['width'] == 17408
        assert json.loads((source / 'manifest.json').read_bytes())['run_id'] == entry['run']
        documents = json.loads((source / 'documents.json').read_bytes())
        position_count = len(documents['positions'])
        query = {key: entry[key] for key in ('run', 'revision')}
        for block in entry['blocks']:
            layer = get(f'/{block}', **query)
            profile = json.loads((source / f'B{block}-profile.json').read_bytes())
            assert layer['values'] == profile['fields']['A']['std']
            candidates = json.loads((source / f'B{block}-cues.json').read_bytes())['units']
            assert len(candidates) == entry['width']
            assert all(card['unit'] == index and card['semantic_confirmed'] is False
                       for index, card in enumerate(candidates))
            for pole in ('positive','negative'):
                indices = profile['fields']['A']['holdout'][pole+'_indices']
                assert len(indices) == entry['width']
                assert all(samples and documents['docs'][samples[0] // position_count]['split'] == 'holdout'
                           for samples in indices)
            samples = [0, 17407]
            if block == 23:
                samples.append(7708)
            activation = np.load(source / f'B{block}-A.npy', mmap_mode='r', allow_pickle=False)
            assert activation.shape == (1024, 8, 17408)
            for unit in samples:
                card = get(f'/{block}/units/{unit}', **query)
                assert card['unit'] == unit and card['source_run'] == profile['run_id']
                assert card['stats']['A']['std'] == layer['values'][unit]
                for pole in card['poles']:
                    assert pole['sample'] == profile['fields']['A']['holdout'][pole['pole']+'_indices'][unit][0]
                    doc, slot = divmod(pole['sample'], 8)
                    expected = (activation[doc, slot, unit].astype('<u4') << np.uint32(16)).view('<f4').item()
                    assert pole['value'] == expected
                if (block, unit) in {(18, 0), (23, 7708), (28, 0), (39, 17407)}:
                    pole = card['poles'][0]
                    whole = get(f'/{block}/field', sample=pole['sample'], **query)
                    doc, slot = divmod(pole['sample'], 8)
                    assert whole['position'] == pole['position'] and whole['token_id'] == pole['token_id']
                    assert whole['values'] == (activation[doc, slot].astype('<u4') << np.uint32(16)).view('<f4').tolist()
            rows.append({'block': block, 'run': entry['run'], 'count': len(layer['values']), 'checked_units': samples})
            print(f'{len(rows)}/23 · {entry["run"]} B{block} · stats/poles match', flush=True)
        for path, args, code in [(f'/{entry["blocks"][0]}/units/17408', query, 422),
                                 (f'/{entry["blocks"][0]}/field', {**query,'sample':8192}, 422),
                                 (f'/{entry["blocks"][0]}', {**query,'revision':'0'*64}, 409)]:
            try:
                get(path, **args)
            except HTTPError as exc:
                assert exc.code == code, (path, exc.code, code)
            else:
                raise AssertionError(f'{path} should return {code}')
    out = checked_dir(RESULT, operation='归档联调回执')
    (out / 'source-check.json').write_text(json.dumps({'passed': True,'blocks': rows}, indent=2)+'\n', encoding='utf8')


if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('serve','verify'))
    {'serve':serve, 'verify':verify}[parser.parse_args().mode]()
