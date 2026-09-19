"""Read-only status audit: stable sample IDs, not mutable numeric filenames.

No model loading, material rewrite, repair, restart, or phase finalization.
"""
import sys, json, hashlib, time
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'glm5'))
import numpy as np
from phase2620_native_coordinate_contract import ROOT, RESULT, read, save, sha
from phase2683_explicit_answer_audit import extract

OUT = RESULT / 'phase2691_crossmodel_role_confirmation'
DEST = OUT / 'analysis/status_audit_20260908'
CANDIDATES = [('h',27,484,'positive'),('h',27,4986,'positive'),('h',29,3589,'negative'),('a',26,4320,'positive'),('a',28,14621,'positive')]


def timing(paths):
    ordered=sorted(p.stat().st_mtime for p in paths)
    diffs=np.diff(ordered)
    # Gaps include pauses and restarts; show all gaps separately, do not call it pure model time.
    regular=diffs[(diffs>0) & (diffs<180)]
    return {'files':len(paths),'first_write':datetime.fromtimestamp(ordered[0]).astimezone().isoformat() if ordered else None,
        'last_write':datetime.fromtimestamp(ordered[-1]).astimezone().isoformat() if ordered else None,
        'elapsed_first_to_last_seconds':ordered[-1]-ordered[0] if len(ordered)>1 else 0,
        'all_adjacent_gap_mean_seconds':float(diffs.mean()) if len(diffs) else None,
        'under180s_gap_count':len(regular),'under180s_gap_mean_seconds':float(regular.mean()) if len(regular) else None,
        'under180s_gap_min_seconds':float(regular.min()) if len(regular) else None,
        'under180s_gap_max_seconds':float(regular.max()) if len(regular) else None,
        'over180s_gaps':int((diffs>=180).sum()),'basis':'File timestamps, not GPU kernel runtime; copied/overwritten files can distort estimates.'}


def scores(records, source):
    groups={};examples=[]
    for record in records:
        row=source[record['case_id']]
        assert all(record[k]==row[k] for k in ('family','language','unit','content_instance','form','roster_order','mention_order','target_index','output_function','target','alternate'))
        assert record['source_case_index']==row['case_index']
        parsed=extract(record['generated'],(row['target'],row['alternate']),record['final_answer_available'])
        status=parsed['status']
        if status=='parsed':status='correct' if parsed['choice']==row['target'].casefold() else 'wrong'
        cell=f"{row['language']}/{row['output_function']}"
        for group in (cell, row['family']+'/'+cell, 'all'):
            values=groups.setdefault(group,Counter());values['n']+=1;values[status]+=1
            values['legacy_whole_match']+=int(record['content_correct']);values['strict_match']+=int(record['strict_correct'])
            values['EOS']+=int(record['eos']);values['generated_tokens']+=len(record['generated_ids'])
            values['padded_natural_different']+=int(record['padded_natural_state_max_abs']>0)
        if len([r for r in examples if r['status']==status])<3:
            examples.append({'case_id':row['case_id'],'status':status,'prompt':row['prompt'],'target':row['target'],
                'actual_output':record['generated'],'generated_ids':record['generated_ids']})
    return {'groups':groups,'examples':examples}


def main():
    initial=read(RESULT/'phase2686_independent_role_contract/material/initial.json')
    source={r['case_id']:r for r in initial};current=read(OUT/'qwen14/material/cases.json')
    wanted={r['case_id'] for r in current};original={r['case_id'] for r in initial if r['unit']<2}
    assert len(source)==8192 and len(original)==4096 and len(wanted)==2048
    config=read(OUT/'protocol/frozen.json');assert config['models']['qwen14']['cases']==2048
    committed=[];cells=[];source_fingerprints={};mismatched_files=[]
    for path in sorted((OUT/'qwen14/analysis').glob('cell_*.json')):
        manifest=read(path);cell=path.stem[5:]
        filechecks={relative:sha(OUT/'qwen14'/relative)==digest for relative,digest in manifest['files'].items()}
        source_fingerprints[str(path)]=sha(path)
        for relative in manifest['files']:
            p=OUT/'qwen14'/relative;source_fingerprints[str(p)]=sha(p)
        records=read(OUT/f'qwen14/analysis/records_{cell}.json')
        assert all(r['family']+'_'+r['language']==cell for r in records)
        assert len(records)==manifest['cases']
        committed.extend(records)
        details=[]
        with np.load(OUT/f'qwen14/maps/counts_{cell}.npz') as z:
            shapechecks={k:list(z[k].shape) for k in ('h__all4_positive','a__all4_positive')}
            assert all(np.array_equal(z[m+'__all4_same_nonzero'],z[m+'__all4_positive']+z[m+'__all4_negative']) for m in ('h','a'))
            for m,l,j,direction in CANDIDATES:
                opposite='negative' if direction=='positive' else 'positive'
                pos=int(z[m+'__all4_'+direction][l,1,j]);neg=int(z[m+'__all4_'+opposite][l,1,j])
                details.append({'metric':m,'layer_or_checkpoint':l,'coordinate':j,'old_sign':direction,
                    'same_old_direction':pos,'opposite':neg,'other_or_zero':manifest['base_groups']-pos-neg,
                    'denominator':manifest['base_groups']})
        stale_index=0;same_index_wrong_identity=0
        for r in records:
            p=OUT/f'qwen14/behavior/case_{r["case_index"]:04d}.json'
            if p.exists():
                actual=read(p)
                if actual['case_id']!=r['case_id']:same_index_wrong_identity+=1
                elif actual!=r:stale_index+=1
        cells.append({'cell':cell,'cases':len(records),'base_groups':manifest['base_groups'],'unit_counts':dict(Counter(r['unit'] for r in records)),
            'current_material_cases':sum(r['case_id'] in wanted for r in records),'hash_checks':filechecks,
            'current_protocol_group_count_match':manifest['base_groups']==config['models']['qwen14']['per_language_family_groups'],
            'numeric_filename_now_other_sample':same_index_wrong_identity,'same_sample_record_changed':stale_index,
            'actual_tokens':manifest['actual_tokens'],'full_coordinate_shapes':shapechecks,'candidates':details,
            'manifest_modified':datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat()})
        if not all(filechecks.values()):mismatched_files.append(cell)
    assert not mismatched_files,mismatched_files
    committed_ids=[r['case_id'] for r in committed];assert len(committed_ids)==len(set(committed_ids))
    durable_by_id={r['case_id']:r for r in committed};disagreements=[];behavior=[];aligned=0
    material_by_index={r['case_index']:r for r in current}
    behavior_paths=list((OUT/'qwen14/behavior').glob('case_*.json'))
    for p in sorted(behavior_paths):
        r=read(p);behavior.append(r);source_fingerprints[str(p)]=sha(p)
        aligned+=bool(r['case_index'] in material_by_index and material_by_index[r['case_index']]['case_id']==r['case_id'])
        if r['case_id'] in durable_by_id and durable_by_id[r['case_id']]!=r:
            # Numeric case IDs may change under reindexing. Check scientific fields separately.
            previous=durable_by_id[r['case_id']]
            changed=[k for k in set(previous)|set(r) if previous.get(k)!=r.get(k)]
            if changed!=['case_index'] and set(changed)!={'case_index'}:disagreements.append({'case_id':r['case_id'],'fields':changed})
        else:durable_by_id[r['case_id']]=r
    measured=set(durable_by_id)
    current_records=[r for k,r in durable_by_id.items() if k in wanted]
    initial_records=[r for k,r in durable_by_id.items() if k in original]
    first_sample=read(OUT/'qwen14/field/case_0000.npz') if False else None
    with np.load(OUT/'qwen14/maps/global_sums.npz') as z:
        chunks=z['completed_chunks'].tolist();shapes={k:list(z[k].shape) for k in z.files if k!='completed_chunks'}
    others={}
    for key in ('glm4','ds7','ds7_answer'):
        folder=OUT/key
        others[key]={'behavior_files':len(list((folder/'behavior').glob('case_*.json'))),
            'committed_cells':len(list((folder/'analysis').glob('cell_*.json'))),
            'noops_exist':(folder/'analysis/native_noops.json').exists(),'complete':(folder/'analysis/completion.json').exists(),
            'calibration_jsonl_exists':(folder/'analysis/calibration.jsonl').exists()}
    recent_by_cell={}
    for cell in ('negation_en','negation_zh','comparison_en','comparison_zh','reference_en','reference_zh'):
        paths=[OUT/f'qwen14/behavior/case_{r["case_index"]:04d}.json' for r in behavior if r['family']+'_'+r['language']==cell]
        recent_by_cell[cell]=timing(paths)
    save(DEST/'audit.json',{'timestamp':datetime.now().astimezone().isoformat(),
        'method':'Read only; identify via case_id/source_case_index and original frozen8192 source; no model/prepare/audit_one called, no old file rewritten.',
        'current_configured_q14':2048,'original_configured_q14':4096,
        'committed_cells':len(cells),'committed_records':len(committed),'committed_unique_samples':len(set(committed_ids)),
        'all_cell_hashes_match':not mismatched_files,'mixed_manifest_sizes':dict(Counter(c['cases'] for c in cells)),
        'current_material_covered_committed':len(set(committed_ids)&wanted),
        'all_durable_unique_samples':len(measured),'current_material_covered_all_durable':len(wanted&measured),
        'current_material_missing':sorted(wanted-measured),'original_material_missing':sorted(original-measured),
        'behavior_file_count':len(behavior),'behavior_unique_samples':len({r['case_id'] for r in behavior}),
        'behavior_numeric_indices_aligned_with_current_material':aligned,
        'committed_record_numeric_filename_now_other_sample':sum(c['numeric_filename_now_other_sample'] for c in cells),
        'same_sample_scientific_disagreements':disagreements,'cells':cells,
        'current2048_subset_behavior':scores(current_records,source),
        'all_available_original4096_behavior':scores(initial_records,source),
        'global_sums_completed_chunks':chunks,'global_sums_shapes':shapes,
        'other_protocols':others,'recent_timing_by_cell':recent_by_cell,
        'source_fingerprints':source_fingerprints,'current_code_sha256':sha(ROOT/'tests/glm5/phase2691_crossmodel_role_confirmation.py'),
        'current_material_sha256':sha(OUT/'qwen14/material/cases.json'),'frozen_file_sha256':sha(OUT/'protocol/frozen.json'),
        'audit_code_sha256':sha(Path(__file__)),'phase_completed':False})
    print(json.dumps({'cells':len(cells),'committed':len(committed),'committed_unique':len(set(committed_ids)),
        'mixed_manifest_sizes':dict(Counter(c['cases'] for c in cells)),
        'current2048_covered':len(wanted&measured),'original4096_covered':len(original&measured),
        'behavior_files':len(behavior),'numeric_id_collisions':sum(c['numeric_filename_now_other_sample'] for c in cells),
        'scientific_disagreements':disagreements,'other_protocols':others},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
