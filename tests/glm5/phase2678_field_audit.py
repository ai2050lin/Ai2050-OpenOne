"""Independent completed-field coverage, source-prefix and token-boundary audit."""
import hashlib
from collections import Counter
import numpy as np
from phase2620_native_coordinate_contract import read,save,sha
from phase2677_source_role_contract import OUT as CONTRACT,FIELD
from phase2677_padded_native_runtime import group_key
from phase2671_native_mlp_field import unbits


def audit():
    assert read(FIELD/'analysis/final.json')['all_checks_passed']
    cases=read(CONTRACT/'material/cases.json');records=read(FIELD/'analysis/records.json');manifest=read(FIELD/'analysis/raw_manifest.json')
    assert len(cases)==len(records)==len(manifest)==8448
    source_refs={};changed=[];embedding_refs={};embedding_failures=[];counts=Counter();token_counts=Counter();group_counts=Counter()
    for r,record,raw in zip(cases,records,manifest):
        assert r['case_index']==record['case_index']==raw['case_index']
        path=FIELD/f'field/case_{r["case_index"]:04d}.npz'
        assert path.stat().st_size==raw['bytes'] and path.resolve()==type(path)(raw['path']).resolve()
        with np.load(path) as z:
            expected={'h','a'}|({'full__h'} if r['published'] else set())
            if r['parameter_published']:expected|={f'full__{k}' for k in ('x','pre_mlp','attention','gate','up','a','down')}
            assert set(z.files)==expected,(r['case_index'],z.files,expected)
            h,a=z['h'],z['a'];assert h.shape==(37,2,2560) and a.shape==(36,2,9728)
            assert h.dtype==a.dtype==np.uint16 and np.isfinite(unbits(h)).all() and np.isfinite(unbits(a)).all()
            prefix=tuple(r['prompt_ids'][:r['body_end_token']+1]);digest=hashlib.sha256(h[:,0].tobytes()+a[:,0].tobytes()).hexdigest()
            if prefix not in source_refs:source_refs[prefix]=(r['case_index'],digest)
            elif source_refs[prefix][1]!=digest:changed.append({'case':r['case_index'],'reference_case':source_refs[prefix][0]})
            for q,t in enumerate((r['body_end_token'],r['task_end_token'])):
                token=r['prompt_ids'][t];eh=hashlib.sha256(h[0,q].tobytes()).hexdigest()
                if token not in embedding_refs:embedding_refs[token]=eh
                elif embedding_refs[token]!=eh:embedding_failures.append([r['case_index'],t,token])
            if r['published']:
                full=z['full__h'];assert full.shape==(37,len(r['prompt_ids']),2560)
                assert np.array_equal(full[:,[r['body_end_token'],r['task_end_token']]],h)
                assert np.isfinite(unbits(full)).all();counts['fullH']+=1
            if r['parameter_published']:
                for name in ('x','pre_mlp','attention','gate','up','a','down'):
                    full=z['full__'+name];assert full.shape==(4,len(r['prompt_ids']),9728 if name in ('gate','up','a') else 2560)
                    assert np.isfinite(unbits(full)).all()
                counts['fullMLP']+=1
        key=group_key(r);token_counts[key]+=len(r['prompt_ids']);group_counts[key]+=1
        counts['raw_packs']+=1
        if len(record['generated_ids'])>16:raise AssertionError('Generation exceeded frozen budget')
    assert len(group_counts)==96 and counts=={'fullH':64,'fullMLP':16,'raw_packs':8448}
    for key in group_counts:
        meta=read(FIELD/f'analysis/moments_{key}.json')
        assert meta['cases']==group_counts[key] and meta['actual_unmasked_tokens']==token_counts[key] and meta['padding_tokens_included']==0
        with np.load(FIELD/f'maps/alltoken_{key}.npz') as z:
            assert len(z.files)==12
            assert all(np.isfinite(z[k]).all() for k in z.files)
    result={'all_checks_passed':not embedding_failures,'counts':dict(counts),'moment_groups':96,'all_actual_tokens':sum(token_counts.values()),
        'same_source_distinct_actual_prefixes':len(source_refs),'same_source_comparisons':len(cases)-len(source_refs),'same_source_changed_pairs':changed,
        'embedding_same_token_failures':embedding_failures,'material_sha256':sha(CONTRACT/'material/cases.json'),
        'scope':'Every stored nativeboundarycoordinate finite, everypublishedactualtoken shape checked, no maskedtail in publishedfields or momentcounts. Same-token H0 consistency does not independently reread the checkpoint embedding matrix. Source mismatches, if any, are retained as numeric warnings, not semantic evidence.'}
    save(FIELD/'analysis/independent_field_audit.json',result);assert result['all_checks_passed']
    return result


if __name__=='__main__':print(audit())
