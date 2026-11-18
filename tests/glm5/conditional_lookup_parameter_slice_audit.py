"""CPU-only exact audit of exported native scalars against original BF16 files."""
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(Path(__file__).parent)]
from conditional_lookup_stage_audit import SOURCE, OUTPUT, read, sha
from deploy.storage_policy import check_write
import numpy as np
import torch
from source_binding_runtime import Runtime
from shared.dist_protocol.natural_register_runner import snapshot


def main():
    start = time.monotonic()
    torch.set_num_threads(4)
    pre = read(SOURCE / 'prereg.json')
    cards = read(OUTPUT / 'discovery-native-feature-cards.json')
    assert cards['native_mlp_parameters_sha256'] == sha(SOURCE / 'native-parameters.npz')
    head = cards['attention']
    path = OUTPUT / 'observed-head-native-parameters.npz'
    assert head['parameters_sha256'] == sha(path)
    print(json.dumps(dict(stage='native_parameter_snapshot_verification', completed=0, total=10)), flush=True)
    rt = Runtime('qwen14')
    assert snapshot(rt.path) == pre['model_snapshot'], 'model/tokenizer snapshot mismatch'
    checks = {}
    with np.load(path) as export:
        for key, spec in head['parameters'].items():
            weight = rt.weight(spec['parameter'], 'cpu')
            if not spec.get('full'):
                weight = weight[:, spec['start']:spec['stop']] if spec['sliced_axis'] == 1 else weight[spec['start']:spec['stop']]
            native = weight.clone().contiguous().view(torch.uint16).numpy()
            assert np.array_equal(native, export[key]), 'native scalar mismatch: ' + key
            checks[key] = dict(scalars=int(native.size), bitwise_equal=True)
            del weight
            print(json.dumps(dict(stage='native_scalar_audit', key=key, completed=len(checks), total=10,
                                  elapsed_seconds=time.monotonic()-start)), flush=True)
    seal = read(SOURCE / 'candidate.json')
    b = seal['parameter']['layer']
    with np.load(SOURCE / 'native-parameters.npz') as params:
        for key, module in [('gate', 'gate_proj'), ('up', 'up_proj'), ('down', 'down_proj')]:
            weight = rt.weight(f'model.layers.{b}.mlp.{module}.weight', 'cpu')
            actual = weight[:, params['units']] if key == 'down' else weight[params['units']]
            assert np.array_equal(actual.float().numpy(), params[key]), 'MLP scalar mismatch: ' + key
            checks['mlp_' + key] = dict(scalars=int(actual.numel()), bitwise_equal=True)
            del weight, actual
    result = dict(schema='clt_native_parameter_scalar_audit.v1', kind='real', provenance=pre['provenance'],
                  model_snapshot_verified=True, checks=checks,
                  total_scalars=sum(c['scalars'] for c in checks.values()),
                  auditor_sha256=sha(Path(__file__)), feature_cards_sha256=sha(OUTPUT / 'discovery-native-feature-cards.json'),
                  elapsed_seconds=time.monotonic()-start, causal_tested=False,
                  scope='exact original parameter identity, no semantic function or causal effect certified')
    target = OUTPUT / 'native-parameter-scalar-audit.json'
    check_write(target, 1 << 20, operation='CLT native scalar audit')
    assert not target.exists()
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
