"""Shared identity for the factorial, directional-write, and transfer study."""
import mlp_read_route_common as previous

ROOT, MODEL, CAP, CUT = previous.ROOT, previous.MODEL, previous.CAP, previous.CUT
PREVIOUS=previous.OUT
OUT=ROOT/'tests/glm5/result/mlp_factor_steer_v1'
PUBLIC=ROOT/'research/mlp_factor_steer/v1'
RUN='MLP-FACTOR-STEER-20261105-v1'
SEED=511052801
FIT=64
SELECT=64
PER_TASK=1536
TOTAL=FIT+SELECT+2*PER_TASK
read,save,read_gz,write_gz,digest,file_sha=(previous.read,previous.save,previous.read_gz,previous.write_gz,previous.digest,previous.file_sha)


def identity():
    base=previous.identity()
    return {**base,'run_id':RUN,'provenance':{**base['provenance'],'seed':SEED}}
