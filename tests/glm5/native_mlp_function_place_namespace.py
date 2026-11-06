"""Give the adapted location diagnostic its own run namespace, unchanged stimuli."""
import time
import native_mlp_function_place_polarity as L
from shared.dist_protocol.mechanism_trace import file_sha
parent=L.P.RUN;L.P.RUN=parent+':place-polarity'
L.P.save(L.PUB/'namespace-contract.research.json',{'schema':'research_execution_namespace.v1','kind':'real','parent_run_id':parent,'run_id':L.P.RUN,
    'model_snapshot':L.P.read(L.SUB/'polarity-corpus.json')['model_snapshot'],
    'reason':'The reusable runner groups internal IDs as polarity:<k>; separate actualrun_id prevents collision with the universal diagnostic. Immutable stimuli andthresholds unchanged; original prepared data retained.',
    'parent_prereg_sha256':file_sha(L.PUB/'prereg.research.json'),'driver_sha256':file_sha(__file__),'runner_sha256':file_sha(L.__file__),
    'declared_before_location_outputs':True})
while not (L.P.OUT/'polarity-result.json').exists():time.sleep(5)
L.run()
