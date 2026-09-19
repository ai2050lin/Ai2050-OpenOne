"""Complete-array, immutable-source and append-only evidence audit; no model is loaded."""
import hashlib
from rdc_operator_common import *


def main():
    start=time.monotonic();guard();out=BASE/'verification';checks=[]
    for r in read(BASE/'review.json')['evidence']:
        assert sha(ROOT/r['path'])==r['sha256'];checks.append(r['path'])
    for r in read(BASE/'review.json')['attachments']:
        assert sha(Path(r['path']))==r['sha256'];checks.append(r['path'])
    refinement=read(BASE/'qa/qwen14/main/residency_v3/refinement.json')
    for name,digest in refinement['commits_sha256'].items():
        assert sha(BASE/'qa/qwen14/main/commits'/name)==digest
    replay=read(BASE/'qa/qwen14/main/residency_replay_check_GPU13_CPU11.json')
    assert replay['all_layer_all_coordinate_and_factor_bitwise'] and replay['full_generated_token_ID_sequence_bitwise']
    frozen=read(BASE/'operators/frozen.json')
    for name,digest in frozen['bank_files'].items():assert sha(BASE/'operators'/name)==digest
    assert sha(BASE/'material.json.gz')==frozen['main_material_sha']
    assert sha(BASE/'operators/result.json')==frozen['fit_result_sha']
    qa=read(BASE/'qa_extension.json')
    for filename,key in [('qa_material.json.gz','original_random256_preserved_sha'),('qa_balanced_material.json.gz','balanced256_sha'),('qa_multihop_material.json.gz','multihop128_sha')]:
        assert sha(BASE/filename)==qa[key]
    memo=(ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md').read_bytes();prefix=read(BASE/'memo_prefix.json')
    assert hashlib.sha256(memo[:prefix['bytes']]).hexdigest()==prefix['sha256']
    material=rows();token_count=0;fixtures=0;source_files=0;expected_arrays={};expected_full={}
    for row in material:
        scope='confirmation' if row['split']=='confirmation' else 'main';folder=BASE/'capture'/scope
        commit=gzread(folder/'commits'/f'{row["sample_id"]}.json.gz')
        assert commit['sample_id']==row['sample_id']
        for kind in ('fields','factors','energies'):
            assert (folder/kind/f'{row["sample_id"]}.npz').exists();source_files+=1
        for kind,key in [('fields','anchor_arrays'),('factors','factor_arrays')]:
            path=str((folder/kind/f'{row["sample_id"]}.npz').relative_to(BASE))
            expected_arrays[path]=commit[key]
        if (folder/'full_fields'/f'{row["sample_id"]}.npz').exists():
            expected_full[str((folder/'full_fields'/f'{row["sample_id"]}.npz').relative_to(BASE))]=commit['full_H_identities']
        token_count+=len(row['prompt_ids']);fixtures+=int((folder/'full_fields'/f'{row["sample_id"]}.npz').exists())
    assert (len(material),token_count,fixtures)==(2048,346920,16)
    # Every stored array is decompressed and checked. No sampled-array shortcut.
    records=[];undefined=[];total_arrays=0;raw_elements=0
    for i,path in enumerate(sorted(BASE.rglob('*.npz'))):
        entries=[]
        with np.load(path,allow_pickle=False) as z:
            for name in z.files:
                a=z[name];total_arrays+=1;raw_elements+=int(a.size)
                v=unbits(a) if a.dtype==np.uint16 else a
                if not np.isfinite(v).all():
                    bad=np.argwhere(~np.isfinite(v))
                    valid=(name=='next_NLL' and '/energies/' in path.as_posix() and v.ndim==1 and len(bad)==1 and int(bad[0,0])==len(v)-1 and np.isnan(v[-1]))
                    assert valid,(str(path.relative_to(BASE)),name,bad[:5])
                    undefined.append({'file':str(path.relative_to(BASE)),'array':name,'index':len(v)-1,'meaning':'No supplied next token after final source position'})
                ident=identity(a)
                if name in expected_arrays.get(str(path.relative_to(BASE)),{}):
                    assert ident==expected_arrays[str(path.relative_to(BASE))][name],(path,name)
                if name=='H' and str(path.relative_to(BASE)) in expected_full:
                    for layer in range(len(a)):
                        assert identity(a[layer])==expected_full[str(path.relative_to(BASE))][str(layer)],(path,layer)
                entries.append({'array':name,'shape':list(a.shape),'dtype':str(a.dtype),'values_sha256':ident['sha256']})
                del a,v
        records.append({'file':str(path.relative_to(BASE)),'bytes':path.stat().st_size,'sha256':sha(path),'arrays':entries})
        if i<2 or (i+1)%512==0:print('COMPLETE_ARRAY_AUDIT',i+1,'seconds',round(time.monotonic()-start,1),flush=True)
    compressed(out/'all_array_identities.json.gz',records)
    qa_counts={}
    for model in ('qwen4','qwen14','glm4'):
        for scope in (('main','confirmation') if model=='qwen4' else ('main',)):
            folder=BASE/'qa'/model/scope;result=read(folder/'result.json');cc=list((folder/'commits').glob('*.json'))
            assert len(cc)==result['sources']
            for p in cc:
                r=read(p);assert (folder/'fields'/f'{r["question_id"]}.npz').is_file()
                assert r['generated_ids'] and len(r['generated_ids'])<=48 and r['prompt_ids']
            qa_counts[model+'/'+scope]=len(cc)
    assert qa_counts=={'qwen4/main':288,'qwen4/confirmation':96,'qwen14/main':64,'glm4/main':64}
    for model in ('qwen4','qwen14','glm4'):
        r=read(BASE/'scale'/model/'result.json');assert r['sources']==128
        assert len(list((BASE/'scale'/model/'fields').glob('*.npz')))==128
        if model!='qwen4':
            for path in (BASE/'scale'/model/'rows').glob('*.json'):
                original=read(path);assert sha(BASE/'scale'/model/'fields'/f'{original["sample_id"]}.npz')==original['field_sha']
    metric=read(BASE/'metric_followup/result.json');assert len(list((BASE/'metric_followup/commits').glob('*.json')))==640
    assert len(list((BASE/'metric_followup/autonomous').glob('*.json')))==64
    assert len(list((BASE/'metric_followup/autonomous_fields').glob('*.npz')))==64
    assert len(list((BASE/'operations/commits').glob('*.json')))==64
    assert all(s['max_quadrature_identity_error']<1e-5 for s in metric['summaries'])
    assert len(metric['complete_readout_geometry'])==2
    assert len(list((BASE/'metric_followup/readout_geometry').glob('*.npz')))==2
    assert read(BASE/'precision/result.json')['corrected_energy_relative_error']<1e-10
    assert read(BASE/'identity_audit/correction_result.json')['corrected_token_positions']==4
    assert read(BASE/'behavior/result.json')['matched64']['qwen4']['questions']==64
    assert len(read(BASE/'scale/paired_audit.json')['comparisons'])==9
    assert read(BASE/'operations/response_agreement_audit.json')['questions']==64
    population=read(BASE/'metric_followup/population_geometry/result.json')
    assert population['queries']==640 and population['replay_comparisons']==1920
    assert population['maximum_prior_KL_or_entropy_replay_error']<1e-9 and len(population['geometry'])==3
    assert read(BASE/'metric_followup/population_geometry/math_check.json')['passed']
    result={'timestamp':stamp(),'passed':True,'source':snapshot(Path(__file__)),'seconds':time.monotonic()-start,
        'prior_evidence_and_attachment_SHA_checks':checks,'original_memo_bytes_unchanged':prefix['bytes'],'operator_bank_SHA_checks':len(frozen['bank_files']),
        'natural_sources':len(material),'natural_tokens':token_count,'full_alltoken_fixtures':fixtures,'source_core_archives':source_files,
        'all_npz_files':len(records),'all_arrays':total_arrays,'all_array_elements_checked':raw_elements,'allowed_undefined_final_NLL_arrays':len(undefined),
        'undefined_values':undefined,'QA_commit_counts':qa_counts,'all_arrays_manifest':'verification/all_array_identities.json.gz',
        'limits':'All stored arrays checked, not a claim that every streamed full-token raw tensor remains on disk. Content hashes identify evidence; they do not prove causal semantics or eliminate training-data contamination.'}
    save(out/'scientific_integrity.json',result);ledger('complete_native_array_integrity',time.monotonic()-start);guard()
    print('OPERATOR_SCIENTIFIC_INTEGRITY_PASS',len(records),total_arrays,flush=True)


if __name__=='__main__':main()
