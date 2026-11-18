"""Freeze downstream executables before any confirmation response is opened."""
import time
import branch_reuse14 as B
import branch_reuse14_analysis as A

FILES=['branch_reuse14.py','branch_reuse14_collect.py','branch_reuse14_materials.py',
       'branch_reuse14_analysis.py','branch_reuse14_output.py','branch_reuse14_intervene.py',
       'branch_reuse14_execution.py','test_branch_reuse14_algorithms.py',
       'recursive_branch14.py','recursive_branch14_intervene.py','recursive_branch14_analysis.py',
       'qwen14_mlp_atlas.py','source_binding_resources.py']


def freeze():
    B.configure();A.admit();assert not (B.OUT/'execution-freeze.json').exists()
    assert not (B.OUT/'confirm').exists(),'Confirmation artifacts already exist'
    B.save('execution-freeze.json',dict(created_local=time.strftime('%Y-%m-%d %H:%M'),
        sources={name:B.file_sha(B.ROOT/'tests/glm5'/name) for name in FILES},
        selection_sha256=B.file_sha(B.OUT/'frozen-selection.json'),plan_sha256=B.file_sha(B.OUT/'analysis-plan.json'),
        tests_sha256=B.file_sha(B.OUT/'algorithm-tests.log'),confirmation_opened=False))


def admit():
    data=B.read(B.OUT/'execution-freeze.json')
    for name,sha in data['sources'].items():assert sha==B.file_sha(B.ROOT/'tests/glm5'/name),name
    assert data['selection_sha256']==B.file_sha(B.OUT/'frozen-selection.json')
    assert data['plan_sha256']==B.file_sha(B.OUT/'analysis-plan.json')
    return data


if __name__=='__main__':freeze()
