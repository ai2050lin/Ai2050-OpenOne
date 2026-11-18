"""Freeze a narrower scope after failed pilot; reuse identical pilot bytes."""
import os,shutil
from pathlib import Path
import b36_composition14 as B


def prepare():
    B.configure();source=B.ROOT/'tests/glm5/result/b36_composition14_20261113_v2';assert not (B.OUT/'trial-plan.json').exists()
    pilot=B.read(source/'pilot/collection.json');cells=pilot['behavior']['cells'];eligible=[]
    for key in sorted({(c['branch'],c['depth'],c['mode']) for c in cells}):
        cc=[c for c in cells if (c['branch'],c['depth'],c['mode'])==key]
        if len(cc)==2 and all(c['passed'] for c in cc):eligible.append(list(key))
    assert eligible==[['label',1,'archive'],['label',2,'archive'],['label',3,'archive'],['number',1,'archive'],['number',1,'program'],['number',2,'archive'],['number',3,'archive']],eligible
    material=B.read(source/'materials.json');rows=material['rows']
    counts={split:sum(r['split']==split and [r['branch'],r['depth'],r['mode']] in eligible for r in rows) for split in ['discovery','confirm','format','order']}
    excluded={split:[r['id'] for r in rows if r['split']==split and [r['branch'],r['depth'],r['mode']] not in eligible] for split in ['discovery','confirm','format','order']}
    B.save('scope-plan.json',dict(eligible_strata=eligible,planned_native_counts=counts,excluded_row_ids=excluded,source_pilot_sha256=B.file_sha(source/'pilot/collection.json'),
        source_materials_sha256=B.file_sha(source/'materials.json'),decision_sha256=B.file_sha(source/'scope-revision-decision.json'),
        scope='Number primitive calculation and both domains of concrete record retrieval. Composition programs failed pilot and receive only independent behavior confirmation. No complex-feature mechanism confirmation is admitted.',
        unchanged_thresholds=True,unchanged_fixed_group=True,confirm_model_outputs_not_opened=True,source_sha256=B.file_sha(__file__)))
    # Original identities and row ordering are preserved, not attributed to a new run.
    for name in ['materials.json','material-audit.json','grammar-number.json','grammar-label.json','math-tests.log','admission-tests.log']:
        shutil.copyfile(source/name,B.OUT/name)
    linked=[]
    for p in (source/'pilot').rglob('*'):
        if not p.is_file():continue
        linked.append(dict(path=str(p),sha256=B.file_sha(p),bytes=p.stat().st_size,read_only_reference=True))
    keys=['row_indices','row_ids','scores','behavior','chunks','native_bytes','seconds']
    B.save('pilot/collection.json',dict(**{k:pilot[k] for k in keys},kind='reused_real_pilot_no_new_samples',native_origin=str(source/'pilot'),
        original_collection_sha256=B.file_sha(source/'pilot/collection.json'),original_provenance=pilot['provenance']))
    B.save('pilot-reuse.json',dict(kind='same_run_pilot_reuse_not_new_samples',files=linked,origin=str(source),
        source_trial_sha256=B.file_sha(source/'trial-plan.json'),original_source_archive_sha256=B.file_sha(source/'sources-before-scope-v3.zip'),
        identical_input_model_runtime=True,native_kernel_unchanged='Only scope admission changes in v3; original v2 forward code and manifests are retained.',
        recovery='NTFS hardlink attempt returned WinError5; use explicit read-only native_origin reference, with no extra copy and no elevation.'))
    print(counts,flush=True)


if __name__=='__main__':prepare()
