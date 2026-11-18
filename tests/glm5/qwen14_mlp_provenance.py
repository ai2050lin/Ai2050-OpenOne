"""Use the project provenance writer; retain pre-existing frozen files unchanged."""
import os
from pathlib import Path
import qwen14_mlp_atlas as C
from shared.dist_protocol.mechanism_trace import save_json as raw_save
from tools import provenance


def save_result(path,body):
    path=Path(path).resolve()
    if path.parent!=C.OUT:raise ValueError('Result must remain in the registered run directory')
    os.environ.update(GIT_CONFIG_COUNT='1',GIT_CONFIG_KEY_0='safe.directory',GIT_CONFIG_VALUE_0=str(C.ROOT))
    helper=provenance.write_result('qwen14_'+path.stem,{'run_id':C.RUN,'result_path':str(path),
        'payload_sha256':C.digest(body),'model_snapshot':C.read(C.OUT/'plan.json')['model_snapshot']},
        model_path=C.MODEL,seed=20261109,tool='qwen14_native_mlp_atlas')
    prov=C.read(helper)['provenance']
    assert not provenance.check_provenance({'provenance':prov})
    raw_save(path,{**body,'kind':'real','run_id':C.RUN,'model_snapshot':C.read(C.OUT/'plan.json')['model_snapshot'],
                   'provenance':prov,'provenance_helper':str(helper.relative_to(C.ROOT))})


def attest():
    files=[]
    for name in ['plan.json','analysis-plan.json','model-identity.json','tokenizer-audit.json','pilot.json','collector-amendment.json']:
        p=C.OUT/name
        files.append({'path':name,'sha256':C.file_sha(p),'five_fields_at_creation':bool(C.read(p).get('provenance'))})
    save_result(C.OUT/'admission-note.json',{'schema':'qwen14_admission_note.v1','files':files,
        'scope':'Initial census plans and pilot had exact model/code/input hashes but lacked the project five-field provenance envelope. This is a retrospective audit, not a claim of prior compliance. Original files remain unchanged. No formal semantic or causal evidence promotion.',
        'analysis_state':'Response analysis not started at this audit; subsequent result documents use tools.provenance.write_result.',
        'power_scope':'Census moments, native arithmetic and response-prediction screening are descriptive, not a paired binary behavior confirmation. Causal behavioral tests require a separate prospective power gate.'})


if __name__=='__main__':attest()
