"""Read existing, immutable MLP archives for the model scene."""
from functools import lru_cache
import json
from typing import Literal

from fastapi import HTTPException, Query
import numpy as np

from server import native_mlp_service as middle
from server import qwen14_mlp_service as late
from shared.dist_protocol.mechanism_trace import file_sha, unbits

RUNS = (
    (middle, middle.SPECS['qwen14']['run_id']),
    (late, late.RUN_V2),
)
STATISTICS = ('mean', 'std', 'min', 'max')


@lru_cache(maxsize=4)
def _archive_json(path, stamp):
    return json.loads(path.read_bytes())


@lru_cache(maxsize=4)
def _revision(path, stamp):
    return file_sha(path)


def archive_revision(service, run):
    path = service.folder(run) / 'manifest.json'
    return _revision(path, path.stat().st_mtime_ns)


def archive_catalog(snapshot):
    runs = []
    for service, run in RUNS:
        if snapshot != late.SNAPSHOT:
            break
        try:
            meta = service.check(snapshot=snapshot, run=run)
        except HTTPException as exc:
            if exc.status_code != 404:
                raise
            continue
        base = service.folder(run)
        files = {entry['path']: entry['bytes'] for entry in meta['files']}
        names = ['documents.json', 'tokenizer.json']
        if any(not (base / name).is_file() or (base / name).stat().st_size != files.get(name) for name in names):
            continue
        blocks = [
            block for block in meta['blocks']
            if all((base / name).is_file() and (base / name).stat().st_size == files.get(name)
                   for name in (f'B{block}-profile.json', f'B{block}-cues.json', f'B{block}-A.npy'))
        ]
        if blocks:
            runs.append({'run': run, 'revision': archive_revision(service, run), 'blocks': blocks, 'width': meta['width']})
    return {'schema': 'workspace_mlp_atlas.v1', 'snapshot': snapshot, 'runs': runs}


def archive_block(snapshot, block, run, revision, unit=None):
    service = next((service for service, name in RUNS if run == name), None)
    if service is None or snapshot != late.SNAPSHOT:
        raise HTTPException(404, '此快照没有对应的 MLP 普查运行。')
    meta = service.check(block, unit, snapshot=snapshot, run=run)
    if archive_revision(service, run) != revision:
        raise HTTPException(409, '归档清单版本已变更，请同步目录。')
    identity = {'snapshot': snapshot, 'run': run, 'revision': revision, 'block': block, 'width': meta['width']}
    return service, identity


def read_archive(service, run, name):
    path = service.path(name, run)
    return _archive_json(path, path.stat().st_mtime_ns)


def profile_for(service, identity):
    profile = read_archive(service, identity['run'], f'B{identity["block"]}-profile.json')
    if any(profile.get(key) != identity[key] for key in ('block', 'width')) or profile.get('model_snapshot') != identity['snapshot']:
        raise HTTPException(409, '普查统计的模型快照、块或宽度不一致。')
    return profile


def document_at(documents, sample):
    positions = documents['positions']
    index, slot = divmod(sample, len(positions))
    return documents['docs'][index], index, slot, positions[slot]


def activation_matrix(service, identity, documents):
    path = service.path(f'B{identity["block"]}-A.npy', identity['run'])
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if array.shape != (len(documents['docs']), len(documents['positions']), identity['width']) or array.dtype != np.dtype('<u2'):
        raise HTTPException(409, '归档激活的形状或精度不一致。')
    return array


def include_routes(router, lookup_model):
    @router.get('/models/{snapshot}/mlp-atlas')
    def catalog(snapshot: str):
        lookup_model(snapshot)
        return archive_catalog(snapshot)

    @router.get('/models/{snapshot}/mlp-atlas/{block}')
    def layer(snapshot: str, block: int, run: str, revision: str, metric: Literal['mean', 'std', 'min', 'max']='std'):
        lookup_model(snapshot)
        service, identity = archive_block(snapshot, block, run, revision)
        profile = profile_for(service, identity)
        values = profile['fields']['A'][metric]
        if len(values) != identity['width'] or not np.isfinite(values).all():
            raise HTTPException(409, '普查统计包含缺值或宽度不匹配。')
        return {'schema': 'workspace_mlp_layer.v1', **identity, 'metric': metric, 'values': values}

    @router.get('/models/{snapshot}/mlp-atlas/{block}/units/{unit}')
    def unit_profile(snapshot: str, block: int, unit: int, run: str, revision: str):
        lookup_model(snapshot)
        service, identity = archive_block(snapshot, block, run, revision, unit)
        profile = profile_for(service, identity)
        documents = read_archive(service, run, 'documents.json')
        cues = read_archive(service, run, f'B{block}-cues.json')['units']
        if len(cues) != identity['width'] or cues[unit]['unit'] != unit:
            raise HTTPException(409, '候选解释与神经元编号不一致。')
        path = service.path('tokenizer.json', run)
        tokenizer = service.tokenizer(str(path), path.stat().st_mtime_ns)
        activation = activation_matrix(service, identity, documents)
        poles = []
        for pole in ('positive', 'negative'):
            sample = profile['fields']['A']['holdout'][pole+'_indices'][unit][0]
            doc, index, slot, position = document_at(documents, sample)
            if doc['split'] != 'holdout':
                raise HTTPException(409, '两端上下文的留出划分不一致。')
            tokens = doc['tokens']
            poles.append({'pole': pole, 'sample': sample, 'document': doc['doc_id'], 'position': position,
                          'token_id': tokens[position], 'value': float(unbits(activation[index, slot, unit])),
                          'text': tokenizer.decode(tokens[max(0, position-31):position+1], skip_special_tokens=False)})
        fields = profile['fields']
        return {'schema': 'workspace_mlp_unit.v1', **identity, 'unit': unit, 'source_run': profile['run_id'],
                'stats': {field: {stat: fields[field][stat][unit] for stat in STATISTICS} for field in ('A', 'G', 'V')},
                'candidate': cues[unit], 'poles': poles, 'document_count': len(documents['docs']),
                'readouts_per_document': len(documents['positions']),
                'sign_regime_fractions': profile['sign_regime_fractions'][unit]}

    @router.get('/models/{snapshot}/mlp-atlas/{block}/field')
    def native_field(snapshot: str, block: int, run: str, revision: str, sample: int = Query(ge=0)):
        lookup_model(snapshot)
        service, identity = archive_block(snapshot, block, run, revision)
        documents = read_archive(service, run, 'documents.json')
        if sample >= len(documents['docs']) * len(documents['positions']):
            raise HTTPException(422, '归档读出位置越界。')
        doc, index, slot, position = document_at(documents, sample)
        array = activation_matrix(service, identity, documents)
        values = unbits(array[index, slot]).tolist()
        if not np.isfinite(values).all():
            raise HTTPException(409, '归档激活包含无效读数。')
        return {'schema': 'workspace_mlp_field.v1', **identity, 'sample': sample, 'document': doc['doc_id'],
                'position': position, 'token_id': doc['tokens'][position], 'values': values}
