"""Index an already verified local atlas extension without duplicating dense per-unit data."""
import argparse
import os
from pathlib import Path

from mlp_tail_extension import BASE, OUT, RUN, C
from deploy.research_library import Library
from server import mlp_tail_service
from tools import provenance


def register(center):
    os.environ['AI2050_MLP_TAIL_DIR'] = str(BASE)
    os.environ['AI2050_MLP_TAIL_EXTENSION_DIR'] = str(OUT)
    overview = mlp_tail_service.overview()
    assert overview['run_id'] == RUN
    directory = C.ROOT / 'research/mlp_tail/b28_b31_v1'
    library = Library(center / 'research_library')
    manifest_sha = C.file_sha(OUT / 'manifest.json')
    plan_sha = C.file_sha(OUT / 'plan.json')
    audit_path = provenance.write_result(
        'mlp_tail_b28_b35', {'run_id': RUN, 'native_manifest_sha256': manifest_sha,
                            'plan_sha256': plan_sha, 'model_snapshot': C.SNAPSHOT,
                            'base_manifest_sha256': overview['manifest_sha256'],
                            'evidence_level': 'observational_census',
                            'replay': overview['replay'], 'coverage': overview['coverage'],
                            'association_implementation': C.read(OUT / 'association-implementation.json')},
        model_path=C.MODEL, seed=C.SEED, tool='mlp_tail_extension_register',
    )
    audit = C.read(audit_path)
    assert not provenance.check_provenance(audit)

    def put(name, body):
        return library.ingest({**body, 'kind': 'real', 'run_id': RUN, 'model_snapshot': C.SNAPSHOT},
                              f'{RUN}/{name}', kind='local_report')['sha256']

    plan = C.read(OUT / 'plan.json')
    plan_ref = put('frozen-plan', plan)
    manifest_ref = put('native-manifest', C.read(OUT / 'manifest.json'))
    audit_ref = put('provenance-audit', audit)
    overview = {**overview, 'provenance': audit['provenance'], 'provenance_audit_sha256': audit_ref}
    report_ref = put('report', overview)
    graph_ref = put('observation-index', {'schema': 'research_graph.v1', 'feature_cards': [], 'graph_edges': [],
                     'native_index': {'api': '/api/research/mlp-tail', 'blocks': overview['blocks'],
                                      'unit_min': 0, 'unit_max': 9727, 'manifest_sha256': manifest_sha},
                     'scope': '逐单元观察卡及全矩阵直接由原生 API 回查。未登记新增因果边。'})
    binding_ref = put('binding', {'schema': 'research_space_binding.v1', 'report_sha256': report_ref,
                       'graph_sha256': graph_ref, 'traces': [],
                       'context': {'study_type': 'mlp_tail_atlas', 'native_bank': True,
                                   'model_dimensions': {'n_layers': 36, 'd_ff': 9728, 'd_model': 2560, 'n_heads': 32},
                                   'native_manifest_sha256': manifest_ref, 'read_api': '/api/research/mlp-tail'}})
    work_ref = put('workpackage', {'schema': 'research_workpackage.v1', 'title': 'Qwen3-4B B28–B35 · 77,824 单元全量观察',
                  'stages': {'collection': {'status': 'done', 'new_units': 38912, 'total_units': 77824,
                                            'new_matrices': 20, 'total_matrices': 35},
                             'frozen_plan': {'status': 'done', 'artifact_sha256': plan_ref, 'file_sha256': plan_sha},
                             'semantic_causality': {'status': 'not_tested', 'blocks': [28, 29, 30, 31]},
                             'existing_prediction_and_intervention': {'run_id': overview['base_run_id'],
                                                                        'blocks': overview['prediction_blocks']}}})
    result = {'run_id': RUN, 'model_snapshot': C.SNAPSHOT, 'report_sha256': report_ref,
              'graph_sha256': graph_ref, 'binding_sha256': binding_ref, 'workpackage_sha256': work_ref,
              'manifest_artifact_sha256': manifest_ref, 'native_manifest_sha256': manifest_sha,
              'provenance_audit_sha256': audit_ref, 'provenance_result': audit_path.relative_to(C.ROOT).as_posix()}
    C.save_json(directory / 'registration.json', result)
    C.save_json(directory / 'prereg.research.json', plan)
    C.save_json(directory / 'report.research.json', overview)
    print(result, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    register(parser.parse_args().data_dir.resolve())
