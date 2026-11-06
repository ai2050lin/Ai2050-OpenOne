"""Identity and file layout for the Qwen3-4B cache mediation phase."""
import mlp_functional_common as prior

ROOT, MODEL = prior.ROOT, prior.MODEL
PREVIOUS, CAP = prior.OUT, prior.CAP
OUT = ROOT/'tests/glm5/result/mlp_kv_mediation_v1'
PUBLIC = ROOT/'research/mlp_kv_mediation/v1'
RUN = 'MLP-KV-MEDIATION-20261105-v1'
SEED = 511052792
read, save, digest, file_sha = prior.read, prior.save, prior.digest, prior.file_sha


def identity():
    return {**prior.identity(),'run_id':RUN,'provenance':{**prior.identity()['provenance'],'seed':SEED}}
