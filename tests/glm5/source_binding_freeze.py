"""Seal implementation before the first internal model observation."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import source_binding_design as D

files=['source_binding_native.py','source_binding_analysis.py','source_binding_causal.py','source_binding_generation.py','source_binding_engineering.py','source_binding_freeze.py']
assert D.read(D.OUT/'engineering-calibration.json')['status']=='passed'
assert not any(D.OUT.rglob('*-native-manifest.json'))
D.save('internal-code-freeze.json',{'plan_sha256':D.sha(D.OUT/'plan.json'),
    'code_sha256':{str((D.ROOT/'tests/glm5'/p).relative_to(D.ROOT)):D.sha(D.ROOT/'tests/glm5'/p) for p in files},
    'engineering_sha256':D.sha(D.OUT/'engineering-calibration.json'),
    'generation_addendum':'first8 families of each admitted split, greedy max4 new tokens; complete exact word and EOS separately; descriptive and cannot override main gate',
    'causal_addendum':'all4 heads jointly or both selected MLP blocks jointly at last query; all admitted heldout rows; noop and restoration exact; family bootstrap2000; report effects without extra significance promotion',
    'mlp_norm_control':'greedy nearest RMS individual Down contribution from discovery, excluding selected units; group RMS ratio within10% or control marked invalid',
    'uncertainty':'family bootstrap conditional on four frames; no use of expanded row count as independent families',
    'power_sensitivity':'96 independent families gives worst-case Bernoulli SE 0.051; only four frames, so across-frame generalization remains limited',
    'formal_evidence_promotion':0})
