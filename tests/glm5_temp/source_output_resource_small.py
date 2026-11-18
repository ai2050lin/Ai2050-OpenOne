"""Preserve original resource policy and reduce the now behavior-only workload."""
from datetime import datetime
import shutil
import source_output_core as S
from source_output_core import C

assert not (C.OUT/'resource-amendment-behavior-only.json').exists()
assert not list(C.OUT.glob('behavior-confirm-[0-9]*.json'))
gate=C.read(C.OUT/'behavior-calibration.json')
assert len(gate['branches'])==6 and not any(v['passed'] for v in gate['branches'].values())
original=C.read(C.OUT/'resource-amendment.json')
assert C.sha(C.OUT/'resource-launcher-v1.py')==original['launcher_sha256']
C.save('resource-amendment-behavior-only.json',{
    'schema':'source_output_resource_amendment.v2','execution_only':True,
    'time':datetime.now().astimezone().isoformat(),
    'base_plan_sha256':C.sha(C.OUT/'plan.json'),
    'launcher_sha256':C.sha(C.ROOT/'tests/glm5/source_output_execute.py'),
    'original_amendment_sha256':C.sha(C.OUT/'resource-amendment.json'),
    'original_launcher_sha256':C.sha(C.OUT/'resource-launcher-v1.py'),
    'calibration_sha256':C.sha(C.OUT/'behavior-calibration.json'),
    'minimum_free_GiB':5,'budget_GiB':.25,
    'observed_free_GiB':shutil.disk_usage(C.OUT).free/2**30,
    'reason':'All six entire calibration branches failed the frozen behavior gate, so the preregistered policy prevents every native and causal branch. Only 4608 fixed held-out behavior rows, failure diagnostics and client packaging remain. The other completed 14B test is writing a 13.27 GiB source archive to shared C. No other job files are removed.',
    'exposure':'Calibration outcomes are open; all confirm model outcomes remain unopened. This is a resource amendment, not a scientific hypothesis or gate change.',
    'unchanged':['5760 inputs','all behavioral thresholds','no row filtering','all stopped branches remain stopped','model snapshot and precision','original candidate and analysis freeze'],
    'scope':'Behavior-only cap 0.25 GiB total retained own directory, C reserve 5 GiB; original 3.8 GiB plan and 12 GiB resource amendment preserved.',
})
print('Behavior-only resource amendment recorded before confirm inference',flush=True)
