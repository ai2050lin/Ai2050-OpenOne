"""Audit native arrays, identities, strict behavior and frozen-run artifacts."""
import time
import argparse
import numpy as np
import branch_reuse14 as B
import branch_reuse14_collect as C
import branch_reuse14_analysis as A
import branch_reuse14_execution as E
from branch_reuse14_behavior_guard import check


def finite_bits(value):
    return bool(np.all((value&0x7f80)!=0x7f80))


def run(native_only=False):
    B.configure();C.admit();A.admit();E.admit();start=time.monotonic();entries=[];collections=[]
    qualification=B.read(B.OUT/'qualified-execution.json')
    from branch_reuse14_qualified import admit_sources
    admit_sources(qualification)
    for split in ['pilot','discovery','confirm']:
      rows,meta=A.load(split);strict=check(split)
      if split!='confirm':assert strict['passed']
      else:assert all(c['passed'] for c in strict['cells'] if c['branch'] in qualification['eligible_branches'])
      for chunk in meta['chunks']:
        path=B.OUT/split/chunk['name'];info=B.read(path/'chunk.json');assert B.file_sha(path/'chunk.json')==chunk['sha256']
        assert info['trial_plan_sha256']==B.file_sha(B.OUT/'trial-plan.json')
        size=len(info['row_ids'])
        for item in info['files']:
          file=path/item['path'];assert file.stat().st_size==item['bytes'] and B.file_sha(file)==item['sha256']
          with np.load(file) as data:
            if item['path'].startswith('B'):
                assert set(data.files)==set('GVA')
                for key in data.files:
                    value=data[key];assert value.shape==(size,2,17408) and value.dtype==np.uint16 and finite_bits(value)
            else:
                shapes=dict(hidden=(size,41,2,5120),final_hidden=(size,2,5120),prefix_embedding=(size,5120),B19U64=(size,256,3),B26U13099=(size,256,3))
                assert set(data.files)==set(shapes)
                for key,shape in shapes.items():
                    value=data[key];assert value.shape==shape
                    if key=='prefix_embedding':assert value.dtype==np.float32 and np.isfinite(value).all()
                    else:assert value.dtype==np.uint16 and finite_bits(value)
          entries.append(dict(path=str(file.relative_to(B.OUT)),bytes=item['bytes'],sha256=item['sha256'],mtime_ns=file.stat().st_mtime_ns))
        B.log('native_archive_audit',len(entries),24*41,start,split=split,chunk=chunk['name'])
      collections.append(dict(split=split,rows=len(rows),families=len({r['family'] for r in rows}),
          strict_behavior=check(split),seconds=meta['seconds'],sha256=B.file_sha(B.OUT/split/'collection.json')))
    # Reproduce every token and causal boundary from the unchanged tokenizer.
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(B.MODEL,local_files_only=True)
    material=B.read(B.OUT/'materials.json')['rows']
    for row in material:
        ids=tok.encode(row['prompt'],add_special_tokens=False);assert ids==row['tokens']
        assert tok.decode(ids[row['content_end']:row['content_end']+1])=='.\n'
        assert row['answer_pos']==len(ids)-1 and row['content_end']<row['answer_pos']
        for label,cid in zip('ABC',row['choice_ids']):assert tok.encode(row['prompt']+' '+label,add_special_tokens=False)==ids+[cid]
    identity=B.read(B.OUT/'model-identity.json')
    for i,item in enumerate(identity['files']):
        file=B.MODEL/item['path'];assert file.stat().st_size==item['bytes'] and B.file_sha(file)==item['sha256']
        B.log('postrun_model_identity',i+1,len(identity['files']),start,file=item['path'])
    B.save('native-manifest.json',dict(files=entries,total_bytes=sum(x['bytes'] for x in entries),
        purpose='Reusable all-layer native fields, counterexamples and independent execution audit',review_date='2026-12-13',
        precision='Original BF16 uint16 bit patterns; prefix means float32',retention='Keep original native arrays; do not equate sampled two readout positions with all token states. Candidate trajectories cover each padded token.',source_sha256=B.file_sha(__file__)))
    B.save('native-audit.json',dict(passed=True,collections=collections,tokenizer_rows_verified=len(material),
        model_files_rehashed=len(identity['files']),native_manifest_sha256=B.file_sha(B.OUT/'native-manifest.json'),
        model_stats=[dict(path=item['path'],bytes=(B.MODEL/item['path']).stat().st_size,mtime_ns=(B.MODEL/item['path']).stat().st_mtime_ns) for item in identity['files']],
        source_sha256=B.file_sha(__file__),seconds=time.monotonic()-start))
    if not native_only:finalize()


def finalize():
    B.configure();C.admit();A.admit();E.admit();start=time.monotonic()
    qualification=B.read(B.OUT/'qualified-execution.json')
    from branch_reuse14_qualified import admit_sources
    admit_sources(qualification)
    native=B.read(B.OUT/'native-audit.json');assert native['passed']
    assert native['native_manifest_sha256']==B.file_sha(B.OUT/'native-manifest.json')
    for base,entries in [(B.OUT,B.read(B.OUT/'native-manifest.json')['files']),(B.MODEL,native['model_stats'])]:
      for item in entries:
        file=base/item['path'];stat=file.stat()
        assert stat.st_size==item['bytes'] and stat.st_mtime_ns==item['mtime_ns'],'File changed after full native hash audit; rerun affected audit'
    for kind in ['B19','B36']:
      data=B.read(B.OUT/f'intervention-{kind}/result.json')
      n=40*4*len(qualification['eligible_branches'])
      assert len(data['rows'])==n*10 and len(data['row_ids'])==n
      assert data['execution_scope']=='qualified_partial' and data['excluded_branches']==['coordinate']
      assert all(x['restore_bitwise'] and x['replay_relative_L2']<=.02 and x['replay_KL_max']<=.001 for x in data['checks'])
      for name,sha in data['chunk_archives'].items():
        archive=B.OUT/f'intervention-{kind}'/name
        if archive.exists():assert B.file_sha(archive)==sha
        else:
          assert kind=='B19' and name=='uncommitted-first-attempt.npz'
          assert B.file_sha(archive.with_name('chunk-00000.npz'))==sha
          assert any(item.get('path')==str(archive) and item.get('sha256')==sha for item in B.read(B.OUT/'cleanup.json')['removed'])
    # Check intermediate observations against the original collection using IDs,
    # independently of the qualified-view row indexing and flattened predictor order.
    data=B.read(B.OUT/'intervention-B36/result.json')
    original_rows,_=A.load('confirm');by_id={row['id']:i for i,row in enumerate(original_rows)}
    chosen=[by_id[rid] for rid in data['row_ids']]
    intermediate_checks=[]
    for block in [37,38,39]:
      units=data['targets'][str(block)]
      expected=A.target('confirm',block,units,1)[chosen].reshape(len(chosen),len(units),3)
      baseline=[];noop=[]
      for path in sorted((B.OUT/'intervention-B36').glob('chunk-*.npz')):
        with np.load(path) as arrays:
          value=arrays[f'B{block}']
          baseline.append(value[data['arms'].index('baseline')])
          noop.append(value[data['arms'].index('restore')])
      baseline=np.concatenate(baseline);noop=np.concatenate(noop)
      assert np.array_equal(baseline,expected) and np.array_equal(noop,baseline)
      intermediate_checks.append(dict(block=block,rows=len(chosen),units=len(units),baseline_matches_original_GVA=True,no_op_GVA_bitwise=True))
    assert B.read(B.OUT/'output-fidelity.json')['replay_choice_max_abs']==0
    first=B.OUT/'intervention-B19/uncommitted-first-attempt.npz'
    canonical=B.file_sha(B.OUT/'intervention-B19/chunk-00000.npz')
    if first.exists():assert B.file_sha(first)==canonical
    else:
        removed=B.read(B.OUT/'cleanup.json')['removed']
        assert any(item.get('path')==str(first) and item.get('sha256')==canonical for item in removed)
    B.save('audit.json',dict(passed=True,collections=native['collections'],tokenizer_rows_verified=native['tokenizer_rows_verified'],
        model_files_rehashed=native['model_files_rehashed'],native_manifest_sha256=B.file_sha(B.OUT/'native-manifest.json'),
        native_audit_sha256=B.file_sha(B.OUT/'native-audit.json'),native_archive_and_model_stats_unchanged_since_full_hash=True,
        first_intervention_storage_recovery_bitwise=True,
        intermediate_capture_identity_checks=intermediate_checks,
        original_scientific_sources_and_fits_unchanged=True,strict_tie_correction=B.read(B.OUT/'strict-behavior-stop.json'),
        full_required_behavior_passed=False,eligible_branches=qualification['eligible_branches'],
        excluded_branch_native_data_retained_but_not_mechanistically_analyzed=True,
        source_sha256=B.file_sha(__file__),seconds=time.monotonic()-start,
        scope='Execution/identity/shape audit only, not semantic validity, independent-node replication or proof of recursive mechanism.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['native','final','all'],default='all',nargs='?');args=p.parse_args()
    finalize() if args.stage=='final' else run(args.stage=='native')
