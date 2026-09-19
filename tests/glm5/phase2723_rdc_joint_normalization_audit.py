"""Audited CPU-only normalization repair; frozen originals and raw captures stay intact."""
from collections import defaultdict
from rdc_joint_common import *
from rdc_relation_native_parameters import parameter,decode

OUT=BASE/'extension/native_regimes'

def main():
    if (OUT/'normalization_audit.json').exists():return
    guard(5*1024**2);start=time.monotonic()
    originals=['result.json','frozen.json','frozen_training_signatures.npz','rows.json.gz',
        'L6_all_parameter_column_profiles.npz','all_native_unit_conditional_profiles.npz']
    archive=OUT/'pre_correction';archive.mkdir(exist_ok=True);hashes={}
    for name in originals:
        p=OUT/name;q=archive/name
        if not q.exists():shutil.copyfile(p,q)
        assert sha(p)==sha(q);hashes[name]=sha(p)
    with np.load(BASE/'native_factors/train_L6_all_unit_and_coordinate_profiles.npz') as z:
        n=int(z['count'][0]);activation=z['activation_sum'][0].astype(float)
    w=decode(parameter(ROOT,'model.layers.6.mlp.down_proj.weight')).astype(float)
    corrected=w@activation
    with np.load(OUT/'frozen_training_signatures.npz') as z:sigs={k:z[k] for k in z.files}
    assert np.allclose(corrected,sigs['L6_first']*n,rtol=1e-12,atol=1e-9)
    sigs['L6_first']=corrected;npz(OUT/'corrected_training_signatures.npz',**sigs)
    proj=w.T@corrected/2560
    with np.load(OUT/'L6_all_parameter_column_profiles.npz') as z:columns={k:z[k] for k in z.files}
    assert np.allclose(columns['projection_to_frozen_signature']*n,proj,rtol=1e-11,atol=1e-10)
    columns['projection_to_frozen_signature']=proj
    npz(OUT/'L6_all_parameter_column_profiles.npz',**columns)
    records=json.loads(gzip.decompress((OUT/'rows.json.gz').read_bytes()));profile_terms=defaultdict(list);floors=[]
    for r in records:
        with np.load(OUT/'factors'/f'{r["sample_id"]}.npz') as z:
            j=z['positions'].tolist().index(r['position'])
            for b in (6,16,34):
                m=unbits(z[f'L{b}_mlp'][j]).astype(float);a=unbits(z[f'L{b}_activation'][j]).astype(float)
                if b==6:p=proj;sig=corrected
                else:
                    with np.load(OUT/f'L{b}_all_parameter_column_profiles.npz') as c:p=c['projection_to_frozen_signature']
                    sig=sigs[f'L{b}_event']
                terms=a*p;native=float(np.mean(m*sig));total=float(terms.sum())
                item=r['blocks'][str(b)]
                item.update(signature_cosine=float(m@sig/max(np.linalg.norm(m)*np.linalg.norm(sig),1e-20)),
                    signature_constant_MSE=float(np.mean((m-sig)**2)),all_unit_signature_projection_sum=total,native_MLP_signature_projection=native,
                    FP64_projection_relative_discrepancy=abs(total-native)/max(abs(native),1e-20))
                if b==6:profile_terms[f'{r["stage"]}_{r["role"]}_L6'].append(terms)
                floors.append({'stage':r['stage'],'role':r['role'],'block':b,'projection_relative':item['FP64_projection_relative_discrepancy'],
                    'residual_energy_relative':item['FP64_residual_energy_relative_discrepancy']})
    with np.load(OUT/'all_native_unit_conditional_profiles.npz') as z:profiles={k:z[k] for k in z.files}
    for k,v in profile_terms.items():
        values=np.mean(v,0);assert np.allclose(profiles[k][7]*n,values,rtol=3e-7,atol=1e-5);profiles[k][7]=values.astype(np.float32)
    npz(OUT/'all_native_unit_conditional_profiles.npz',**profiles)
    compressed_json(OUT/'rows_corrected.json.gz',records)
    result=read(archive/'result.json')
    for key,s in result['condition_summaries'].items():
        stage,role=key.split('_',1);rr=[r for r in records if r['stage']==stage and r['role']==role]
        s['blocks']={str(b):{k:float(np.mean([r['blocks'][str(b)][k] for r in rr])) for k in rr[0]['blocks'][str(b)]} for b in (6,16,34)}
    result.update(timestamp=stamp(),normalization_correction='normalization_audit.json',active_rows='rows_corrected.json.gz',
        active_signatures='corrected_training_signatures.npz',signature_status='L6 amplitude corrected after new outcomes: historical producer already saved means. Frozen blocks/pairs and native captures unchanged; NOT a new preregistered prototype.')
    audit={'timestamp':stamp(),'source':snapshot(Path(__file__)),'original_hashes':hashes,'factor':n,
        'cause':'phase2721 producer divides every profile except count before saving, even activation_sum. Consumer divided a second time.',
        'repair':'Recompute L6 from actual original full down weight and stored activation mean, then recompute every probe and full9728 projection profile; archive originals. Recompute TRAIN16/34 signature diagnostics previously absent.',
        'unaffected':'All original fields, selected blocks16/34, pairing, all natural identities, event energy, full L16/L34 outputs, original event classifier and125 frozen fits.',
        'precision':'FP64 scalar accounting does not duplicate BF16 rounded native GEMM/residual. The same-dtype native identities remain bitwise exact.',
        'roundoff':{str(b):{metric:{'mean':float(np.mean([f[metric] for f in floors if f['block']==b])),
            'max':float(np.max([f[metric] for f in floors if f['block']==b]))} for metric in ('projection_relative','residual_energy_relative')} for b in (6,16,34)},
        'new_result_sha_pending':False,'seconds':time.monotonic()-start}
    save(OUT/'result.json',result);audit['corrected_result_sha']=sha(OUT/'result.json')
    for name,digest in hashes.items():assert sha(archive/name)==digest
    assert sha(OUT/'frozen.json')==hashes['frozen.json'] and sha(OUT/'frozen_training_signatures.npz')==hashes['frozen_training_signatures.npz']
    immutable(OUT/'normalization_audit.json',audit);guard();print('NORMALIZATION_REPAIR_PASS',n,audit['roundoff'],flush=True)

if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
