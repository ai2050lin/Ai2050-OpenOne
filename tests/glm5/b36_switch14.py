"""Shared identity and admission for cached state-transition experiments."""
import sys,platform
import b36_transition14 as T
S=T.S
S.OUT=S.ROOT/'tests/glm5/result/b36_switch14_20261114_v1'
S.TEMP=S.ROOT/'tests/glm5_temp/b36_switch14_20261114_v1'
S.RUN='B36-SWITCH14-20261114-v1';S.SEED=2026111403
PRIOR=S.ROOT/'tests/glm5/result/b36_transition14_20261114_v1'
PEAK=800*2**20


def runtime_files():
    base=S.ROOT/'.venv/Lib/site-packages/transformers'
    return {str(p.relative_to(S.ROOT)):S.file_sha(p) for p in [base/'generation/utils.py',base/'cache_utils.py',base/'models/qwen3/modeling_qwen3.py']}


def sources(names):
    out={str((S.ROOT/'tests/glm5'/n).relative_to(S.ROOT)):S.file_sha(S.ROOT/'tests/glm5'/n) for n in names}
    out.update(runtime_files());return out


def admit(p):
    for n,h in p['source_hashes'].items():assert S.file_sha(S.ROOT/n)==h,n


def versions():
    import importlib.metadata
    return dict(python=sys.version,platform=platform.platform(),**{n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','safetensors']})
