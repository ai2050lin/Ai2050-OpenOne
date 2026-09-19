"""All-coordinate baseline audit on32 frozen source examples, NO CUDA/model.

Fidelity on baseline states cannot settle error of actual parameter changes.
"""
import os,sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
import numpy as np
from phase2620_native_coordinate_contract import *
import phase2692_native_rounding_math as native
import phase2685_native_attention_math as ideal

OUT=RESULT/'phase2692_linked_native_ledger/numerical_baseline_audit'
LAYERS=(0,5,17,23,26,27,28,35)
SOURCE={'initial':'phase2687_role_qkv_field','confirmation':'phase2690_fresh_role_qkv_confirmation'}

def decode(a):return (a.astype(np.uint32)<<16).view(np.float32) if a.dtype==np.uint16 else a

def prep():
    configs={split:read(RESULT/f'phase2686_independent_role_contract/material/{split}.json') for split in SOURCE}
    selected={split:[r for r in rows if r['parameter_published']] for split,rows in configs.items()}
    assert all(len(rows)==16 for rows in selected.values())
    qwen=ROOT/'.venv/Lib/site-packages/transformers/models/qwen3/modeling_qwen3.py'
    c={'source_cases':{key:[r['case_id'] for r in rows] for key,rows in selected.items()},
        'source_manifests':{key:sha(RESULT/f'{SOURCE[key]}/analysis/published_manifest.json') for key in SOURCE},
        'native_math_sha256':sha(TESTS/'phase2692_native_rounding_math.py'),'ideal_math_sha256':sha(TESTS/'phase2685_native_attention_math.py'),
        'audit_code_sha256':sha(Path(__file__)),'installed_eager_Qwen3_source_sha256':sha(qwen),
        'frozen_before_audit_outputs':True,'intervened_parameters':0,'new_model_forwards':0,
        'method':'Anchored norms fromnative linear, anchoredRoPE fromnative norm, anchoredP fromnativeRoPE, anchoredAV fromnativeP/V; then composednorm→RoPE→P→AV. CompareoldFP64and explicitBF16roundingreference. No fit/TopK/donor.',
        'precision':'ExplicitBF16cast/multiply/add boundaries but NumPyFP32reductions. Not a CUDAkernel emulator or wholeFP32/BF16/FP64 model.',
        'storage':'Originalfulltoken sourcearrays remain authoritative. Everyactual coordinate analyzed. Norm/RoPE diagnostics perphysicalheadcoordinate signed/absolute/mismatch counts overtokens; P/head preserveallquery/source/headcoords. Numericaldiagnostics not compressedsemanticfeatures.',
        'limitations':['Baselineagreementdoesnotvalidate2689finitechangedparameterpredictions. Linear GEMM response stillunmeasured here.',
            'Both32sourceexamplesaretruth/v0; eightfamilies/twolanguages, initialandnewfill only. Notall8192eachcorpus.',
            'GPUreduction/matmul/rsqrt/softmaxmaydiffer; referencehasnoautomaticbitexactclaim.',
            'Rawfieldsobservedfixed256, retainedactualtokens; referenceoperatesactualtokens. Zero maskedtailmayaffectnativekernelreductionorder.']}
    path=OUT/'protocol.json'
    if path.exists():assert read(path)==c
    else:save(path,c)
    return selected,c

def probability64(q,k,mask,scale,pos):
    full=np.repeat(k,q.shape[1]//k.shape[1],axis=1)
    z=np.einsum('qhd,shd->qhs',q[list(pos)].astype(np.float64),full.astype(np.float64))*scale+mask
    e=np.exp(z-z.max(-1,keepdims=True));return e/e.sum(-1,keepdims=True)

def av64(p,v):return np.einsum('qhs,shd->qhd',p.astype(np.float64),np.repeat(v,p.shape[1]//v.shape[1],axis=1).astype(np.float64))

def compare(actual,predicted):
    delta=predicted.astype(np.float64)-actual.astype(np.float64)
    return delta,{'coordinates':delta.size,'mismatch_coordinates':int(np.count_nonzero(delta)),
        'error_L1':float(np.abs(delta).sum()),'error_max_abs':float(np.abs(delta).max())}

def one(row,split,weights,eps):
    source=RESULT/SOURCE[split]/f'source/case_{row["case_index"]:04d}.npz';n=len(row['prompt_ids']);pos=[row['body_end_token'],row['task_end_token']]
    report=[];maps={}
    with np.load(source,allow_pickle=False) as z:
        for l in LAYERS:
            a={k.split('__',1)[1]:decode(z[k]) for k in z.files if k.startswith(f'L{l}__')}
            q=a['upstream_linear_q'].reshape(n,32,128);k=a['upstream_linear_k'].reshape(n,8,128);v=a['actual_value']
            assert np.array_equal(v,a['upstream_linear_v'].reshape(n,8,128))
            gq,gk=weights[f'L{l}_q_gamma'],weights[f'L{l}_k_gamma'];cos,sin=a['upstream_rope_cos'],a['upstream_rope_sin']
            mask=a['actual_mask'][0,0][:,None,:];scale=float(a['scaling'])
            nq,nk=a['upstream_normalized_q'],a['upstream_normalized_k'];rq,rk=a['upstream_query_post_rope_full'],a['actual_key_post_rope']
            P=a['actual_probability'];H=a['native_head_concat'].reshape(2,32,128)
            for variant in ('ideal64','explicit_round32'):
                norm=(lambda x,g:ideal.rms(x,g,eps)) if variant=='ideal64' else (lambda x,g:native.rms_eager(x,g,eps))
                rope=(lambda x:ideal.rope(x.astype(np.float64),cos.astype(np.float64),sin.astype(np.float64))) if variant=='ideal64' else (lambda x:native.rope_eager(x,cos,sin))
                prob=probability64 if variant=='ideal64' else native.probability_eager;av=av64 if variant=='ideal64' else native.av_eager
                predq,predk=norm(q,gq),norm(k,gk)
                rq_composed,rk_composed=rope(predq),rope(predk)
                composedP=prob(rq_composed,rk_composed,mask,scale,pos)
                checks={'qnorm':(nq,predq),'knorm':(nk,predk),'qrope_anchored':(rq,rope(nq)),'krope_anchored':(rk,rope(nk)),
                    'P_anchored':(P,prob(rq,rk,mask,scale,pos)),'AV_anchored':(H,av(P,v)),
                    'P_composed':(P,composedP),'AV_composed':(H,av(composedP,v))}
                for name,(actual,predicted) in checks.items():
                    delta,r=compare(actual,predicted);assert np.isfinite(delta).all()
                    report.append({'layer':l,'variant':variant,'stage':name,**r})
                    key=f'L{l}__{variant}__{name}'
                    if name in ('qnorm','knorm','qrope_anchored','krope_anchored'):
                        maps[key+'__token_sum']=delta.sum(0);maps[key+'__token_sumabs']=np.abs(delta).sum(0)
                        maps[key+'__token_mismatch_count']=np.count_nonzero(delta,axis=0).astype(np.uint16)
                    else:maps[key+'__all_query_error']=delta
                assert (composedP[0,:,pos[0]+1:]==0).all()
    path=OUT/f'maps/{split}_case_{row["case_index"]:04d}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**maps)
    return {'case_id':row['case_id'],'split':split,'source_sha256':sha(source),'actual_tokens':n,'results':report,
        'numerical_map':str(path),'numerical_map_sha256':sha(path),'bytes':path.stat().st_size}

def main():
    selected,c=prep();fixed=native.self_test();save(OUT/'bf16_fixedpoint_test.json',fixed)
    p=RESULT/'phase2685_native_attention_contract/weights/native_qkv_windows.npz'
    with np.load(p) as z:weights={k:decode(z[k]) for k in z.files if k.endswith('_gamma')}
    eps=read(ROOT/'models/hf/qwen3-4b/config.json')['rms_norm_eps'];reports=[]
    for split,rows in selected.items():
        assert read(RESULT/SOURCE[split]/'analysis/final.json')['all_checks_passed']
        for row in rows:
            reports.append(one(row,split,weights,eps));print('2692 ROUNDING AUDIT',len(reports),32,flush=True)
    groups={}
    for row in reports:
        for r in row['results']:
            key=f'{row["split"]}/{r["variant"]}/{r["stage"]}';g=groups.setdefault(key,{'layer_cases':0,'coordinates':0,'mismatch_coordinates':0,'error_L1':0.,'error_max_abs':0.})
            g['layer_cases']+=1
            for name in ('coordinates','mismatch_coordinates','error_L1'):g[name]+=r[name]
            g['error_max_abs']=max(g['error_max_abs'],r['error_max_abs'])
    assert all(g['layer_cases']==128 for g in groups.values())
    result={'all_audit_execution_checks_passed':True,'baseline_examples':32,'layer_cases':256,'new_model_forwards':0,'intervened_parameters':0,
        'all_stage_coordinate_metrics':groups,'cases':reports,'fixed_point_checks':fixed,'formula_semantics_not_closed':True,
        'protocol_sha256':sha(OUT/'protocol.json'),'phase_completed':False,'limits':c['limitations']}
    save(OUT/'result.json',result);print(json.dumps(groups),flush=True)

if __name__=='__main__':main()

