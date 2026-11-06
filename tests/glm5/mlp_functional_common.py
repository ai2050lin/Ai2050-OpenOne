"""Shared experiment paths and identity for the local Qwen3-4B continuation."""
import mlp_fidelity_v2_plan as prior
from shared.dist_protocol.mechanism_trace import file_sha, digest

ROOT, MODEL, PREVIOUS = prior.ROOT, prior.MODEL, prior.OUT
OUT = ROOT / 'tests/glm5/result/mlp_functional_v1'
PUBLIC = ROOT / 'research/mlp_functional/v1'
RUN = 'MLP-FUNCTIONAL-UPSTREAM-20261105-v1'
SEED = 511052026
CAP = ROOT / 'tests/glm5/result/phase2773_w6_v2/capability-inputs.json'
read, save = prior.read, prior.save


def identity():
    plan = read(PREVIOUS / 'plan.json')
    return {'kind':'real','run_id':RUN,'model_snapshot':plan['model_snapshot'],
            'provenance':{**plan['provenance'],'seed':SEED}}
