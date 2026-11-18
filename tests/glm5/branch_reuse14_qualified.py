"""Preserve frozen models; stop failed behavioral branch, evaluate eligible remainder."""
import argparse
import copy
import importlib
import subprocess
import sys
import time
import numpy as np
import branch_reuse14 as B
import branch_reuse14_analysis as A
import branch_reuse14_execution as E

FILES=['branch_reuse14_qualified.py','branch_reuse14_intervene_qualified.py','branch_reuse14_controls.py']
REQUIRED=['number','coordinate','label']


def prepare():
    B.configure();E.admit();A.admit();assert not (B.OUT/'qualified-execution.json').exists()
    assert not any((B.OUT/name).exists() for name in ['response-and-graph-report.json','output-fidelity.json','intervention-B19/result.json'])
    strict=B.read(B.OUT/'strict-behavior-confirm.json')
    eligible=[b for b in REQUIRED if all(c['passed'] for c in strict['cells'] if c['branch']==b)]
    assert eligible==['number','label'] and not strict['passed']
    original=(B.ROOT/'tests/glm5/branch_reuse14_intervene.py').read_text(encoding='utf8')
    adapter=(B.ROOT/'tests/glm5/branch_reuse14_intervene_qualified.py').read_text(encoding='utf8')
    assert adapter.replace('assert len(selected)==40*4*len(A.BRANCHES)','assert len(selected)==480')==original
    rows,_=A.load('confirm');indices=[i for i,r in enumerate(rows) if r['branch'] in eligible]
    B.save('qualified-execution.json',dict(created_local=time.strftime('%Y-%m-%d %H:%M'),
        required_branches=REQUIRED,eligible_branches=eligible,excluded_branches=['coordinate'],full_joint_claim_passable=False,
        native_confirm_local_indices=indices,native_confirm_row_ids=[rows[i]['id'] for i in indices],
        strict_behavior_sha256=B.file_sha(B.OUT/'strict-behavior-confirm.json'),
        original_execution_sha256=B.file_sha(B.OUT/'execution-freeze.json'),selection_sha256=B.file_sha(B.OUT/'frozen-selection.json'),
        sources={name:B.file_sha(B.ROOT/'tests/glm5'/name) for name in FILES},
        rule='Apply the already frozen behavioral threshold to stop the affected coordinate branch. No activation outcome, predicted performance or intervention was inspected to choose branches. Keep all fitted parameters, indices, doses, n per branch, thresholds and six-test correction unchanged.',
        adapter='Filter only confirmation rows at data-loading boundaries; original discovery data and fits unchanged. Generalize native intervention count assertion to 40*4*eligible branches. Original scientific source files remain byte-identical. Attach missing-branch status to every derived artifact; full primary pass always false.',
        limitation='This is a disclosed partial execution after the full three-branch behavior gate failed; it cannot rescue the original three-branch claim. Numeric-to-label transfer is still independently testable under the unchanged frozen models.'))


def admit_sources(q):
    path=B.OUT/'qualification-runtime-recovery.json'
    recovery=B.read(path) if path.exists() else None
    if recovery:assert recovery['qualified_plan_sha256']==B.file_sha(B.OUT/'qualified-execution.json')
    for name,sha in q['sources'].items():
        expected=sha
        if recovery and name in recovery['updated_sources']:
            assert recovery['updated_sources'][name]['before']==sha
            expected=recovery['updated_sources'][name]['after']
        assert B.file_sha(B.ROOT/'tests/glm5'/name)==expected


def bind():
    B.configure();E.admit();A.admit();q=B.read(B.OUT/'qualified-execution.json')
    admit_sources(q)
    assert q['selection_sha256']==B.file_sha(B.OUT/'frozen-selection.json')
    assert q['original_execution_sha256']==B.file_sha(B.OUT/'execution-freeze.json')
    assert q['strict_behavior_sha256']==B.file_sha(B.OUT/'strict-behavior-confirm.json')
    indices=np.array(q['native_confirm_local_indices']);original_load=A.load
    def load(split):
        rows,meta=original_load(split)
        if split!='confirm':return rows,meta
        out=copy.deepcopy(meta);out['row_ids']=[meta['row_ids'][i] for i in indices]
        out['row_indices']=[meta['row_indices'][i] for i in indices];out['scores']=[meta['scores'][i] for i in indices]
        out['full_required_behavior']=out['behavior'];cells=[c for c in B.read(B.OUT/'strict-behavior-confirm.json')['cells'] if c['branch'] in q['eligible_branches']]
        out['behavior']=dict(cells=cells,passed=all(c['passed'] for c in cells),
            overall_accuracy=sum(c['correct'] for c in cells)/sum(c['n'] for c in cells))
        out['native_confirm_local_indices']=indices.tolist();out['execution_scope']='qualified_partial'
        return [rows[i] for i in indices],out
    A.load=load
    for name in ['field','state','pooled']:
        original=getattr(A,name)
        def filtered(split,*args,_original=original,**kwargs):
            values=_original(split,*args,**kwargs)
            return values[indices] if split=='confirm' else values
        setattr(A,name,filtered)
    A.BRANCHES=q['eligible_branches']
    original_save=B.save
    def save(name,data):
        name=str(name).replace('\\','/')
        scoped=dict(data,execution_scope='qualified_partial',required_branches=REQUIRED,
            eligible_branches=q['eligible_branches'],excluded_branches=q['excluded_branches'],
            qualified_execution_sha256=B.file_sha(B.OUT/'qualified-execution.json'))
        if str(name)=='response-and-graph-report.json':
            scoped['eligible_primary_passed']=data['primary_passed'];scoped['primary_passed']=False
            scoped['full_primary_status']='NOT_PASSED_REQUIRED_BEHAVIOR_BRANCH_MISSING'
        recovery=B.OUT/'qualification-runtime-recovery.json'
        if recovery.exists():scoped['runtime_recovery_sha256']=B.file_sha(recovery)
        return original_save(name,scoped)
    B.save=save
    return q


def stage(name):
    bind()
    if name=='analyze':A.analyze()
    elif name=='output':importlib.import_module('branch_reuse14_output').run()
    elif name in ['B19','B36']:importlib.import_module('branch_reuse14_intervene_qualified').run(name)
    elif name=='summarize':importlib.import_module('branch_reuse14_intervene_qualified').summarize()
    elif name=='controls':importlib.import_module('branch_reuse14_controls').run()


def pipeline():
    B.configure();q=B.read(B.OUT/'qualified-execution.json');start=time.monotonic()
    steps=[('analyze','response-and-graph-report.json'),('output','output-fidelity.json'),('B19','intervention-B19/result.json'),
        ('B36','intervention-B36/result.json'),('summarize','intervention-report.json'),('controls','control-comparisons.json')]
    for index,(name,result) in enumerate(steps):
        if (B.OUT/result).exists():
            assert B.read(B.OUT/result)['qualified_execution_sha256']==B.file_sha(B.OUT/'qualified-execution.json')
            continue
        B.save('qualified-pipeline-state.json',dict(stage=name,status='running',source_sha256=B.file_sha(__file__)))
        with (B.OUT/f'qualified-{name}.log').open('a',encoding='utf8') as log:
            code=subprocess.call([sys.executable,'-B','-X','utf8','-u',str(B.ROOT/'tests/glm5/branch_reuse14_qualified.py'),name],cwd=B.ROOT,stdout=log,stderr=subprocess.STDOUT)
        if code:
            B.save('qualified-pipeline-state.json',dict(stage=name,status='failed',exit_code=code,source_sha256=B.file_sha(__file__)))
            raise RuntimeError('Qualified stage failed: '+name)
        assert (B.OUT/result).exists()
        B.log('qualified_pipeline',index+1,len(steps),start,completed=name)
    B.save('qualified-pipeline-state.json',dict(status='completed',seconds=time.monotonic()-start,
        required_branches=REQUIRED,eligible_branches=q['eligible_branches'],full_joint_claim_passable=False,source_sha256=B.file_sha(__file__)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','pipeline','analyze','output','B19','B36','summarize','controls']);args=p.parse_args()
    prepare() if args.stage=='prepare' else pipeline() if args.stage=='pipeline' else stage(args.stage)
