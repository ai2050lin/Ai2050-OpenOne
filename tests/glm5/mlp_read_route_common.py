"""Identity and compact immutable storage for the early-token read-route phase."""
import gzip
import json

import mlp_kv_common as prior

ROOT, MODEL, CAP = prior.ROOT, prior.MODEL, prior.CAP
PREVIOUS = prior.OUT
OUT = ROOT/'tests/glm5/result/mlp_read_route_v1'
PUBLIC = ROOT/'research/mlp_read_route/v1'
RUN = 'MLP-READ-ROUTE-20261105-v1'
SEED = 511052800
CUT = 992
N_SOURCE = 1536
N_DISCOVERY = 64
read, save, digest, file_sha = prior.read, prior.save, prior.digest, prior.file_sha


def identity():
    previous=prior.identity()
    return {**previous,'run_id':RUN,'provenance':{**previous['provenance'],'seed':SEED}}


def write_gz(path, value):
    assert not path.exists(), path
    tmp=path.with_suffix(path.suffix+'.tmp')
    with gzip.open(tmp,'wt',encoding='utf-8',compresslevel=3) as stream:
        json.dump(value,stream,ensure_ascii=False,allow_nan=False,separators=(',',':'))
    tmp.replace(path)


def read_gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as stream:return json.load(stream)
