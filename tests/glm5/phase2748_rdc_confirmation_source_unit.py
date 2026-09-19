"""Freeze-gate local dependency coverage, no confirmation material or outcomes."""
from rdc_question_common import *
import phase2748_rdc_confirmation_gate as gate


def main():
    assert not (OUT/'confirmation/freeze_certificate.json').exists(),'Source coverage qualification is pre-opening only'
    names=gate.local_import_names('import numpy as np, rdc_question_data\nfrom rdc_question_material import conservative_complete_answer\ndef run():\n    from rdc_question_checkpoint import deploy_native\nfrom .relative import local\n')
    assert names=={'numpy','rdc_question_data','rdc_question_material','rdc_question_checkpoint'}
    paths=gate.source_closure();actual={p.name for p in paths}
    required=set(gate.CODE_FILES)|{'phase2748_rdc_confirmation_gate.py','rdc_question_common.py',
        'rdc_question_material.py','rdc_construction_common.py','rdc_question_checkpoint_codec.py',
        'phase2747_rdc_training.py','phase2748_rdc_fit_analysis.py',
        'phase2748_rdc_native_capture.py','rdc_formation_microbatch.py','rdc_query_common.py'}
    assert required.issubset(actual),(required-actual)
    assert len(paths)==len(actual)
    # Independently ensure every discovered direct local dependency is present.
    for path in paths:
        for name in gate.local_import_names(path.read_text(encoding='utf-8-sig')):
            dependency=path.parent/(name+'.py')
            assert not dependency.is_file() or dependency.name in actual
    value={'timestamp':stamp(),'all_passed':True,'source':snapshot(__file__),'gate_sha256':sha(gate.__file__),
        'local_source_files':len(paths),'files':[p.name for p in paths],
        'required_behavior_scoring_cache_loader_codec_and_bootstrap_sources_included':True,
        'scope':'Static local import-closure coverage. No local dependency was executed by traversal; no confirmation material, model or outcomes read. Source hashes themselves will be fixed only by the later genuine certificate.'}
    save(OUT/'unit/confirmation_source_closure_current.json',value)
    print('NATURAL_CONFIRMATION_SOURCE_CLOSURE_UNIT_PASS',len(paths),flush=True)


if __name__=='__main__':main()
