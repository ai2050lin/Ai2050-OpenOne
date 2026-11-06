"""Read-only access to the committed prequestion native fields on the research node."""
from functools import lru_cache
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
import numpy as np

from shared.dist_protocol.mechanism_trace import file_sha, unbits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tests/glm5/result/mlp_functional_v1'
RUN = 'MLP-FUNCTIONAL-UPSTREAM-20261105-v1'
router = APIRouter(prefix='/mlp-functional/v1', tags=['native-upstream-writes'])


@lru_cache(maxsize=2)
def load_result(path, stamp):
    return json.loads(path.read_text(encoding='utf-8'))


@lru_cache(maxsize=9)
def array(path, sha256, shape, stamp):
    if file_sha(path) != sha256:
        raise HTTPException(409, '原生数据内容哈希不符')
    data = np.load(path, mmap_mode='r')
    if data.dtype != np.uint16 or data.shape != shape:
        raise HTTPException(409, '原生阵列格式不符')
    return data


@router.get('/samples/{index}')
def sample(index: int, block: int = Query(35, ge=0, le=35), cohort: str = 'original'):
    if cohort == 'revision':
        return revision_sample(index, block)
    if cohort != 'original':
        raise HTTPException(422, '未知原生采集批次')
    manifest = OUT / 'trace-result.json'
    if not manifest.is_file():
        raise HTTPException(404, '本节点尚无完整原生采集')
    result = load_result(manifest, manifest.stat().st_mtime_ns)
    if result.get('run_id') != RUN or result.get('status') != 'complete' or result.get('kind') != 'real':
        raise HTTPException(409, '该运行尚未完成原生数据登记')
    if not 0 <= index < result['backgrounds'] * 4:
        raise HTTPException(422, '原生样本索引越界')
    paths = {Path(r['path']).stem: r for r in result['native']}
    data = {}
    for name in ['head', 'mlp', 'residual']:
        meta = paths[name]
        p = OUT / 'trace' / f'{name}.npy'
        if not p.is_file():
            raise HTTPException(404, '本节点缺少登记的原生阵列')
        stat = p.stat()
        data[name] = array(p, meta['sha256'], tuple(meta['shape']), (stat.st_size, stat.st_mtime_ns))
    background, variant = divmod(index, 4)
    rows = [result['rows'][background*8 + variant + q*4] for q in range(2)]
    residual = unbits(data['residual'][index])
    return {'schema':'functional_prequestion_sample.v1','kind':'real','model_snapshot':result['model_snapshot'],
            'run_id':RUN,'index':index,'block':block,'position':rows[0]['prequestion'],'queries':rows,
            'embedding':residual[0].tolist(),'residual_before_block':residual[block].tolist(),
            'residual_next_boundary':residual[block+1].tolist(),'incoming_b35':residual[36].tolist(),
            'head_z':unbits(data['head'][index,block]).tolist(),
            'mlp_output':unbits(data['mlp'][index,block]).tolist() if block<35 else None,
            'provenance':{k:{'sha256':v['sha256'],'shape':v['shape']} for k,v in paths.items()},
            'scope':'问题前共享 token 的真实 BF16。B35 的下一边界是 Attention 后、MLP 前。没有 G/V/A 或逐源注意力概率。'}


@lru_cache(maxsize=2)
def verify_revision(path, sha256, stamp):
    if file_sha(path) != sha256:
        raise HTTPException(409, '修订批次原生数据哈希不符')


def revision_sample(index, block):
    path = OUT / 'revision/confirm-result.json'
    if not path.is_file():
        raise HTTPException(404, '本节点尚无修订批次原生数据')
    d = load_result(path, path.stat().st_mtime_ns)
    meta = d.get('native')
    if d.get('run_id') != RUN or d.get('status') != 'complete' or not meta:
        raise HTTPException(409, '修订批次尚未完成登记')
    if not 0 <= index < meta['fields']['incoming'][0]:
        raise HTTPException(422, '原生样本索引越界')
    head, mlp = f'head{block}', f'mlp{block}'
    if head not in meta['fields']:
        raise HTTPException(404, '修订批次只采集提名候选所在层块')
    raw = OUT / 'revision/observed-native.npz'
    if not raw.is_file():
        raise HTTPException(404, '本节点缺少修订批次原生阵列')
    stat = raw.stat()
    verify_revision(raw, meta['sha256'], (stat.st_size,stat.st_mtime_ns))
    background,variant = divmod(index,4)
    rows = [d['rows'][background*8 + variant + q*4] for q in range(2)]
    with np.load(raw) as data:
        def native(name):
            values=data[name]
            if values.dtype != np.uint16 or list(values.shape) != meta['fields'][name]:
                raise HTTPException(409, '修订批次原生阵列格式不符')
            return unbits(values[index]).tolist()
        return {'schema':'functional_prequestion_sample.v1','kind':'real','cohort':'revision','model_snapshot':d['model_snapshot'],
                'run_id':RUN,'index':index,'block':block,'position':rows[0]['prequestion'],'queries':rows,
                'embedding':native('embedding'),'incoming_b35':native('incoming'),'head_z':native(head),
                'mlp_output':native(mlp) if mlp in meta['fields'] else None,'provenance':meta,
                'scope':'修订批次只在候选层采集 Head 与 MLP 写入，另保留原生 embedding 和进入 B35 的残差。'}
