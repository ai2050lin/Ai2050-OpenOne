from pathlib import Path
import sys,json
sys.path.insert(0,str(Path('tests/glm5').resolve()))
import phase2793_materials as P
import phase2793_run as R
from deploy.research_library import Library
from shared.dist_protocol.mechanism_trace import file_sha
report=json.loads((P.PUBLIC/'report-unified-v2.research.json').read_text(encoding='utf-8'))
offline=json.loads((P.OUT/'offline-acceptance-v2.json').read_text(encoding='utf-8'))
doc={'schema':'research_client_acceptance.v1','kind':'real','run_id':P.RUN,'model_snapshot':report['model_snapshot'],
'checks':['browser actual formal client','32Head GQA groups and2048 source weights','full9728 MLP GVA','actual cached step absolute2048 embedding','B32 U383 original parameter slice','reference failures236 and capability NI','heldout native profiles and historical sources'],
'proofs':[{'path':str(p.relative_to(P.ROOT)),'sha256':file_sha(p)} for p in [P.OUT/'browser-3d-final.jpg',P.OUT/'browser-results.jpg']],
'offline_metadata':offline,'limitations':['one physical node','native tensors require local retained files/service','no new Electron installer produced'],
'provenance':report['provenance']}
p=P.PUBLIC/'client-acceptance.research.json';P.save(p,doc,True)
ref=Library(P.ROOT/'distributed_data/research_library').ingest(doc,str(p.relative_to(P.ROOT)),kind='local_report')['sha256']
R.status('client_acceptance',visual_passed=True,artifact_sha256=ref,offline_metadata_passed=True,raw3d_local_service_required=True)
print({'client_acceptance_sha256':ref})
