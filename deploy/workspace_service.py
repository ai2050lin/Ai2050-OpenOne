"""Center-owned model directory and read-only replay of registered observations."""
from functools import lru_cache
import base64
import json
import os
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from deploy.test_archive_service import actor, archive, call
from server.workspace_mlp_atlas import include_routes as include_mlp_atlas
from shared.dist_protocol.test_archive import Conflict, digest, file_sha

router = APIRouter(prefix='/api/research/workspace', tags=['workspace'])
PRIVATE = APIRouter(prefix='/api/research/workspace/private', tags=['workspace'])
SEED = Path(__file__).with_name('workspace_data')
SHA = re.compile(r'[0-9a-f]{64}')
NAME = re.compile(r'[a-zA-Z0-9][a-zA-Z0-9._-]{0,127}')
DIMS = ('n_layers', 'n_heads', 'n_kv_heads', 'head_dim', 'd_model', 'd_ff')
MEASURES = {'head_z', 'head_q', 'head_k', 'head_v', 'mlp_activation', 'mlp_gate', 'mlp_value'}


def published_root():
    center_data = Path(os.environ.get('AI2050_DIST_DIR') or SEED.parents[1] / 'distributed_data')
    return Path(os.environ.get('AI2050_WORKSPACE_DIR') or center_data / 'workspace')


@lru_cache(maxsize=24)
def _read(path, mtime, size):
    return json.loads(path.read_bytes())


def read(path):
    stat = path.stat()
    if stat.st_size > 32 * 2**20:
        raise ValueError('Workspace data exceeds the supported entry size')
    return _read(path, stat.st_mtime_ns, stat.st_size)


def asset(root, filename):
    if not NAME.fullmatch(filename) or not filename.endswith('.json'):
        raise ValueError('Invalid workspace asset name')
    path = (root / filename).resolve(strict=True)
    if path.parent != root.resolve():
        raise ValueError('Workspace asset outside its publication directory')
    return read(path)


def registry(key, *, include_withdrawn=False):
    entries = {}
    for root in (SEED, published_root()):
        try:
            doc = read(root / 'catalog.json')
        except FileNotFoundError:
            continue
        if doc.get('schema') != 'workspace_publication.v1':
            raise ValueError('Unsupported workspace publication schema')
        for entry in doc.get(key, []):
            identity = entry.get('model_snapshot') if key=='models' else entry.get('id')
            if not (SHA if key=='models' else NAME).fullmatch(identity or ''):
                raise ValueError('Invalid workspace entry identity')
            if entry.get('withdrawn') is True and not include_withdrawn:
                entries.pop(identity, None)
            else:
                entries[identity] = (root, entry)
    return entries


def models():
    # Result-derived skeletons and node eligibility use the same complete snapshot key.
    import platform_service as platform
    atlas = platform._lang_atlas()
    declared = registry('models', include_withdrawn=True)
    withdrawn = {key for key, (_, entry) in declared.items() if entry.get('withdrawn') is True}
    found = {key: dict(entry) for key, (_, entry) in declared.items() if key not in withdrawn}
    for item in ((atlas or {}).get('data', {}).get('skeletons', {})).values():
        item = platform._complete_verified_skeleton(item)
        snapshot = item.get('model_snapshot')
        if not SHA.fullmatch(snapshot or '') or snapshot in withdrawn:
            continue
        registered = found.setdefault(snapshot, {})
        if any(registered.get(k) is not None and item.get(k) not in (None, registered[k]) for k in DIMS):
            continue
        found[snapshot] = {**item, **registered, 'model_snapshot':snapshot,
                           'model_key':item.get('model') or item.get('model_key') or registered.get('model')}
    for item in platform.participation_models():
        snapshot = item['snapshot_sha256']
        if snapshot in withdrawn:
            continue
        identity = {'model':item['name'], 'model_snapshot':snapshot}
        found[snapshot] = {**identity, **found.get(snapshot, {}), 'participation':item}
    return [m for m in found.values() if m.get('model')]


def public_entry(entry):
    return {key:value for key,value in entry.items() if key not in ('file', 'artifact', 'source_block')}


@lru_cache(maxsize=32)
def _file_digest(path, mtime, size):
    return file_sha(path)


def public_archive(root, entry):
    path = (root / entry['file']).resolve()
    if path.parent != root.resolve() or not NAME.fullmatch(path.name):
        raise ValueError('Invalid workspace asset name')
    stat = path.stat()
    return {**public_entry(entry), 'revision':digest([entry, _file_digest(path,stat.st_mtime_ns,stat.st_size)])}


def catalog_data():
    archives = [public_archive(root,entry) for root,entry in registry('archives').values() if entry.get('visibility')=='public']
    model_rows = [{**model, 'capabilities':{
        'nativeLive':(model.get('config') or {}).get('model_type')=='qwen3',
        'structure':all(type(model.get(k)) is int and model[k]>0 for k in DIMS),
    }} for model in models()]
    catalog = {'schema':'workspace_catalog.v1', 'models':model_rows, 'archives':archives,
               'resources':[{'id':'goodfire-records','kind':'test-records','visibility':'public'},
                            *(public_entry(entry) for _,entry in registry('resources').values() if entry.get('visibility')=='public')]}
    return {**catalog, 'revision':digest(catalog)}


@router.get('/catalog')
def catalog():
    return call(catalog_data)


def entry_for(key, identity, visibility):
    source = registry(key).get(identity)
    if not source or source[1].get('visibility')!=visibility:
        raise FileNotFoundError('未找到可访问的归档')
    return source


@router.get('/resources/{identity}')
def resource(identity: str):
    def load():
        if identity=='goodfire-records':
            from deploy.test_record_catalog import document
            return document('goodfire')
        root, entry = entry_for('resources', identity, 'public')
        return asset(root, entry['file'])
    return call(load)


def lookup_model(snapshot):
    model = next((item for item in models() if item['model_snapshot']==snapshot), None)
    if not model or any(type(model.get(key)) is not int or model[key] <= 0 for key in DIMS):
        raise ValueError('该快照缺少完整的已登记结构')
    return model


include_mlp_atlas(router, lookup_model)


@router.get('/models/{snapshot}/architecture')
def architecture(snapshot: str):
    from platform_service import _arch_from_config

    def load():
        model = lookup_model(snapshot)
        result = _arch_from_config(model.get('config') or {}, model['n_layers'])
        return {'schema':'model_arch.v1', 'available':result is not None,
                'model_snapshot':snapshot, 'model_key':model.get('model_key', model['model']),
                'source':model.get('config_sha256'), **(result or {})}

    return call(load)


def validate_document(doc, entry):
    if (doc.get('schema')!='native_observation_archive.v1' or doc.get('kind')!='real'
            or doc.get('snapshot')!=entry['snapshot'] or doc.get('run')!=entry['run']):
        raise ValueError('归档快照、运行或格式不符')
    return doc


def read_private(entry):
    """Resolve an existing receipt without promoting it or copying data into a public store."""
    backing = archive()
    record = backing.get(entry['run'])
    receipt = record.get('receipt') or {}
    if (record['archive_state']!='server_verified' or record['state']!='completed'
            or record['plan']['acquisition']['model_snapshot']!=entry['snapshot']):
        raise FileNotFoundError('当前运行没有已校验的原场')
    item = next((item for item in receipt['artifacts'] if item['role']=='native_field'
                 and item['name']==entry['artifact'] and item['availability']=='uploaded'), None)
    if not item or item['size'] > 16 * 2**20:
        raise FileNotFoundError('原场不可用或未登记可读取的分片')
    path = backing.download(item['sha256'])
    if path.stat().st_size!=item['size'] or file_sha(path)!=item['sha256']:
        raise ValueError('原场校验与归档回执不符')
    case_file = next((item for item in receipt['artifacts'] if item['role']=='case_results'
                      and item['name'].endswith('.jsonl') and item['availability']=='uploaded'), None)
    if not case_file or case_file['size'] > 16 * 2**20:
        raise FileNotFoundError('逐题产物缺失或超出归档读取上限')
    cases = backing.download(case_file['sha256'])
    if cases.stat().st_size!=case_file['size'] or file_sha(cases)!=case_file['sha256']:
        raise ValueError('逐题产物与归档回执不符')
    with cases.open(encoding='utf-8') as stream:
        match = next((row for row in (json.loads(line) for line in stream if line.strip())
                      if row.get('case_id')==entry.get('caseId') and row.get('model_snapshot')==entry['snapshot']
                      and row.get('trace_id')==entry.get('trace') and row.get('run_id')==entry['run']), None)
    if match is None:
        raise FileNotFoundError('无法把原场绑定到已校验的同一道题')
    if item['name'].endswith('.json'):
        data = validate_document(read(path), entry)
        if (data.get('caseId')!=match['case_id'] or data.get('trace')!=match['trace_id']
                or [row['position'] for row in readout_rows(data)]!=match.get('readout_positions')):
            raise ValueError('原场身份与同题的记录不符')
        return {**data, 'receiptSha256':receipt['receipt_sha256'], 'arraySha256':item['sha256']}
    if not item['name'].endswith('.npz') or type(entry.get('source_block')) is not int:
        raise ValueError('未登记的原场格式')
    targets = record['plan']['acquisition']['capture'].get('targets', [])
    if not any(target.startswith(f"B{entry['source_block']}/") for target in targets):
        raise ValueError('留存原场的块未绑定到冻结计划目标')
    data = _npz_observation(path, item['sha256'], receipt['receipt_sha256'], entry)
    if [row['position'] for row in data['rows']]!=match.get('readout_positions'):
        raise ValueError('原场读出位置与同题的记录不符')
    return data


@lru_cache(maxsize=8)
def _npz_decoded(path, sha):
    # Existing native-0000 captures store BF16 bits. NPY loads are bounded before parsing.
    from zipfile import ZipFile
    import numpy as np
    with ZipFile(path) as z:
        if sum(field.file_size for field in z.infolist()) > 32 * 2**20:
            raise ValueError('归档原场超出切片服务的解包预算')
    with np.load(path, allow_pickle=False) as arrays:
        manifest = json.loads(arrays['manifest'].item())
        positions = arrays['readout_positions'].tolist()
        token_ids = arrays['token_ids'].tolist()
        fields = {measure:arrays[key].copy() for measure,key in [('head_z','head_z'),('mlp_activation','mlp_activation')]}
    return manifest, positions, token_ids, fields


def _npz_observation(path, sha, receipt_sha, entry):
    import numpy as np
    manifest, positions, token_ids, fields = _npz_decoded(path, sha)
    model = lookup_model(entry['snapshot'])
    if (manifest.get('schema')!='mechanism_trace.v1' or manifest['kind']!='real' or manifest['precision']!='bf16'
            or manifest['model_snapshot']!=entry['snapshot'] or not 0<=entry['source_block']<model['n_layers']
            or any(manifest.get(key)!=model[key] for key in DIMS)
            or manifest['trace_id']!=entry['trace'] or manifest['token_ids']!=token_ids
            or manifest['readout_positions']!=positions or not positions
            or any(type(p) is not int or not 0<=p<len(token_ids) for p in positions)
            or fields['head_z'].shape!=(len(positions),model['n_heads'],model['head_dim'])
            or fields['mlp_activation'].shape!=(len(positions),model['d_ff'])
            or any(values.dtype!=np.uint16 for values in fields.values())):
        raise ValueError('原场分片与冻结快照或 token 不匹配')
    rows = [{'key':str(p), 'position':p, 'tokenId':token_ids[p], 'blocks':{str(entry['source_block']):{
      'sha256':sha, 'fields':{name:base64.b64encode(array[i].astype('<u2',copy=False).tobytes()).decode('ascii')
                             for name,array in fields.items()}}}} for i,p in enumerate(positions)]
    return {'run':entry['run'], 'snapshot':entry['snapshot'], 'caseId':entry['caseId'], 'kind':'real',
            'generationStep':manifest['generation_step'], 'boundary':manifest['boundaries'],
            'trace':entry['trace'], 'receiptSha256':receipt_sha, 'arraySha256':sha,
            'text':manifest['prompt'], 'defaultRow':str(positions[-1]), 'rows':rows}


def observation(identity, visibility, revision=''):
    root, entry = entry_for('archives', identity, visibility)
    if visibility=='public' and revision and revision!=public_archive(root,entry)['revision']:
        raise Conflict('归档版本已更新，请同步中心目录')
    return (read_private(entry) if visibility=='private' else validate_document(asset(root, entry['file']), entry))


def readout_rows(doc):
    return doc.get('rows') or [{**doc,'key':str(doc['position'])}]


def manifest(identity, visibility, revision=''):
    doc = observation(identity, visibility, revision)
    keys = ('kind','snapshot','run','caseId','trace','receiptSha256','arraySha256','manifestSha256','generationStep','boundary','text')
    rows = readout_rows(doc)
    return {'schema':'workspace_observation.v1', 'id':identity, **{k:doc[k] for k in keys if k in doc},
            'defaultRow':doc.get('defaultRow',rows[0]['key']), 'rows':[
                {**{k:row[k] for k in ('key','caseId','doc','position','tokenId','text','label') if k in row},
                 'blocks':{key:sorted(data['fields'].keys() & MEASURES) for key,data in row['blocks'].items()}}
                for row in rows]}


def observation_block(identity, visibility, block, row, revision=''):
    doc = observation(identity, visibility, revision)
    rows = readout_rows(doc)
    key = row or doc.get('defaultRow',rows[0]['key'])
    site = next((site for site in rows if site['key']==key), None)
    field = site and site['blocks'].get(str(block))
    if field is None:
        raise FileNotFoundError('此块或读出位置未随该归档发布')
    return {'schema':'workspace_observation_block.v1', 'kind':'real', 'id':identity, 'snapshot':doc['snapshot'],
            'run':doc['run'], 'row':key, 'block':block, 'position':site['position'], 'tokenId':site['tokenId'], **field}


@router.get('/archives/{identity}')
def public_manifest(identity: str, revision: str = Query('',max_length=64)):
    return call(lambda: manifest(identity, 'public', revision))


@router.get('/archives/{identity}/blocks/{block}')
def public_block(identity: str, block: int, row: str = Query('',max_length=128), revision: str = Query('',max_length=64)):
    return call(lambda: observation_block(identity, 'public', block, row, revision))


@PRIVATE.get('/catalog')
def private_catalog(request: Request, response: Response):
    actor(request)
    response.headers['Cache-Control'] = 'private, no-store'
    def load():
        authorized = []
        for _,entry in registry('archives').values():
            if entry.get('visibility')!='private':
                continue
            try:
                row = archive().get(entry['run'])
            except FileNotFoundError:
                continue
            if (row['archive_state']=='server_verified' and row['state']=='completed'
                    and row['plan']['acquisition']['model_snapshot']==entry['snapshot']):
                authorized.append(public_entry(entry))
        return {'schema':'workspace_private_catalog.v1', 'archives':authorized}
    return call(load)


@PRIVATE.get('/archives/{identity}')
def private_manifest(identity: str, request: Request, response: Response):
    actor(request)
    response.headers['Cache-Control'] = 'private, no-store'
    return call(lambda: manifest(identity, 'private'))


@PRIVATE.get('/archives/{identity}/blocks/{block}')
def private_block(identity: str, block: int, request: Request, response: Response, row: str = Query('',max_length=128)):
    actor(request)
    response.headers['Cache-Control'] = 'private, no-store'
    return call(lambda: observation_block(identity, 'private', block, row))


def evidence_filters(q: str = Query('',max_length=500), source: str = Query('',max_length=16),
                     coverage: str = Query('',max_length=16), snapshot: str = Query('',max_length=64),
                     family: str = Query('',max_length=120), target: str = Query('',max_length=24),
                     position: int | None = Query(None,ge=0), token_id: int | None = Query(None,ge=0),
                     counter: bool = False, offset: int = Query(0,ge=0), limit: int = Query(12,ge=1,le=32)):
    return locals()


@router.get('/evidence')
def public_evidence(response: Response, filters=Depends(evidence_filters)):
    from deploy.workspace_evidence import search
    response.headers['Cache-Control'] = 'no-cache'
    return call(lambda: search(**filters))


@PRIVATE.get('/evidence')
def private_evidence(request: Request, response: Response, filters=Depends(evidence_filters)):
    from deploy.workspace_evidence import search
    actor(request)
    response.headers['Cache-Control'] = 'private, no-store'
    return call(lambda: search(**filters,authorized=True))
