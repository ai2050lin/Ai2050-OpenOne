"""All input coordinates of frozen native projection rows; CPU accounting.

Products are arithmetic terms, not deletion effects or semantic assignments.
Projection uses the observed RMS-normalized attention input, not raw H/E.
"""
import re,shutil
from collections import defaultdict
import numpy as np
from phase2620_native_coordinate_contract import *

OUT=RESULT/'phase2688_native_qkv_terms'
FIELD=RESULT/'phase2687_role_qkv_field'
ATT=RESULT/'phase2685_native_attention_contract'
MATERIAL=RESULT/'phase2686_independent_role_contract/material/initial.json'
LAYERS=(0,5,17,23,26,27,28,35)
OPERATIONS=('roster_order','mention_order','target_index','form')


def unbits(a):return (a.astype(np.uint32)<<16).view(np.float32)


def load_weights():
    meta=read(ATT/'protocol/native_weights.json');p=ATT/'weights/native_qkv_windows.npz';assert sha(p)==meta['artifact_sha256']
    primary=list(meta['vectors']);windows=[];weights={}
    with np.load(p) as z:
        for key in z.files:
            match=re.fullmatch(r'L(\d+)_(q|k|v)_(?:row(\d+)|head(\d+)_d(\d+))',key)
            if match is None:continue
            l,kind,row,head,d=match.groups();row=int(row) if row is not None else int(head)*128+int(d)
            a=unbits(z[key]).astype(np.float64);assert a.shape==(2560,)
            windows.append({'key':key,'layer':int(l),'kind':kind,'output_row':row,'head':row//128,'head_coordinate':row%128,'primary':key in primary})
            weights[key]=a
    assert len(primary)==24 and len(windows)==792
    assert read(ROOT/'models/hf/qwen3-4b/config.json')['attention_bias'] is False
    return meta,primary,windows,weights


def exact32(a):
    b=a.astype(np.float32);assert np.array_equal(a,b.astype(np.float64)), 'BF16-by-BF16 products must serialize without rounding'
    return b


def transform_map(x,w,suffix):
    if suffix=='sum':return x*w
    if suffix=='sumabs':return x*np.abs(w)
    raise ValueError(suffix)


def response_maps(primary,windows,weights):
    bykey={w['key']:w for w in windows};reports=[]
    for path in sorted((FIELD/'maps').glob('operations_*.npz')):
        dst={};reconstruction=[]
        with np.load(path) as z:
            for axis in OPERATIONS:
                output={name:[] for name in ('positive','negative','sum','sumabs','all4_positive','all4_negative')}
                for key in primary:
                    info=bykey[key];li=LAYERS.index(info['layer']);w=weights[key][None,:]
                    prefix=axis+'__attention_x__'
                    for suffix in ('sum','sumabs'):
                        output[suffix].append(transform_map(z[prefix+suffix][li],w,suffix))
                    for family in ('','all4_'):
                        p=z[prefix+family+'positive'][li];n=z[prefix+family+'negative'][li]
                        output[family+'positive'].append(np.where(w>0,p,np.where(w<0,n,0)).astype(np.uint16))
                        output[family+'negative'].append(np.where(w>0,n,np.where(w<0,p,0)).astype(np.uint16))
                    rawsum=z[axis+'__linear_'+info['kind']+'__sum'][li,:,info['head'],info['head_coordinate']]
                    ideal=(z[prefix+'sum'][li]*w).sum(-1)
                    reconstruction.append({'axis':axis,'vector':key,'native_projection_response_sum':rawsum.tolist(),
                        'ideal_product_response_sum':ideal.tolist(),'aggregate_native_rounding_residual':(rawsum-ideal).tolist()})
                for suffix,values in output.items():dst[axis+'__'+suffix]=np.stack(values)
        out=OUT/'maps'/path.name;out.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,**dst)
        assert all(a.shape==(24,2,2560) and np.isfinite(a).all() for a in dst.values())
        reports.append({'cell':path.stem.removeprefix('operations_'),'all8192_background_source_maps':str(path),
            'source_sha256':sha(path),'output_sha256':sha(out),'projection_rounding':reconstruction})
        print('2688 WHOLE INPUT RESPONSE MAP',path.stem,flush=True)
    assert len(reports)==16;save(OUT/'analysis/response_map_audit.json',reports)
    return reports


def raw_terms(rows,primary,windows,weights):
    reports=[];manifest=[]
    for r in rows:
        if not r['parameter_published']:continue
        source=FIELD/f'source/case_{r["case_index"]:04d}.npz'
        with np.load(source) as z:
            x={l:unbits(z[f'L{l}__upstream_attention_x']).astype(np.float64) for l in LAYERS}
            linear={(l,kind):unbits(z[f'L{l}__upstream_linear_{kind}']).astype(np.float64) for l in LAYERS for kind in ('q','k','v')}
        n=len(r['prompt_ids']);positions=[r['body_end_token'],r['task_end_token']]
        full=[];bounds=[];signed=[];absolute=[];checks=[]
        for info in windows:
            key=info['key'];products=x[info['layer']]*weights[key][None,:]
            assert products.shape==(n,2560)
            actual=linear[(info['layer'],info['kind'])][:,info['output_row']]
            reconstructed=products.sum(-1);residual=actual-reconstructed;den=np.abs(products).sum(-1)
            checks.append({'window':key,'all_actual_tokens':n,'native_linear_max_abs_residual':float(np.abs(residual).max()),
                'native_linear_L1':float(np.abs(actual).sum()),'ideal_sum_abs':float(np.abs(reconstructed).sum()),
                'absolute_all_coordinate_terms':float(den.sum()),'ideal_cancellation_L1':float((den-np.abs(reconstructed)).sum())})
            bounds.append(exact32(products[positions]));signed.append(products.sum(0));absolute.append(np.abs(products).sum(0))
            if info['primary']:full.append((key,exact32(products)))
        allprimary=dict(full);assert set(allprimary)==set(primary)
        arrays={'primary_alltoken_terms':np.stack([allprimary[k] for k in primary]),'allwindows_boundary_terms':np.stack(bounds),
            'allwindows_alltoken_signed_sum':np.stack(signed),'allwindows_alltoken_absolute_sum':np.stack(absolute)}
        assert arrays['primary_alltoken_terms'].shape==(24,n,2560)
        path=OUT/f'field/case_{r["case_index"]:04d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path,**arrays)
        reports.append({'case_id':r['case_id'],'case_index':r['case_index'],'tokens':n,'input_source_sha256':sha(source),'windows':checks})
        manifest.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size,'case_index':r['case_index']})
        print('2688 RAW FULL2560 TERMS',len(reports),16,flush=True)
    assert len(reports)==16;save(OUT/'analysis/raw_projection_audit.json',reports);save(OUT/'analysis/published_manifest.json',manifest)
    return reports,manifest


def verify_checkpoint_embeddings(rows):
    # Independent actual checkpoint slice; no CUDA/model initialization.
    from safetensors import safe_open
    model=ROOT/'models/hf/qwen3-4b';index=read(model/'model.safetensors.index.json')['weight_map']
    key='model.embed_tokens.weight';path=model/index[key];before=(path.stat().st_size,path.stat().st_mtime_ns)
    unique={};occurrences=0
    with safe_open(str(path),framework='pt',device='cpu') as f:
        tensor=f.get_slice(key)
        for r in rows:
            if not r['published']:continue
            with np.load(FIELD/f'field/case_{r["case_index"]:04d}.npz') as z:embedding=unbits(z['full__h'][0])
            for token,value in zip(r['prompt_ids'],embedding):
                if token not in unique:unique[token]=tensor[token,:].float().numpy().copy()
                assert np.array_equal(unique[token],value);occurrences+=1
    assert before==(path.stat().st_size,path.stat().st_mtime_ns)
    out=OUT/'weights/native_embeddings.npz';out.parent.mkdir(parents=True,exist_ok=True)
    token_ids=sorted(unique);np.savez_compressed(out,token_ids=np.asarray(token_ids),embedding=np.stack([unique[i] for i in token_ids]))
    report={'all_checks_passed':True,'published_cases':64,'token_occurrences':occurrences,'unique_token_ids':len(unique),
        'all_coordinates':2560,'actual_checkpoint_equal_H0':True,'sha256':sha(out),'checkpoint_shard_bytes_mtime':before,
        'limits':'Independent E==H0 verification on published conditions. Attention input is normalized contextual H, NOT this static embedding. Does not prove E-only language encoding.'}
    save(OUT/'analysis/embedding_audit.json',report);return report


def main():
    assert read(FIELD/'analysis/final.json')['all_checks_passed']
    assert not (OUT/'analysis/final.json').exists()
    rows=read(MATERIAL);meta,primary,windows,weights=load_weights()
    assert sha(MATERIAL)==read(FIELD/'protocol/frozen.json')['material_sha256']
    reserve=2*1024**3;free=shutil.disk_usage(RESULT).free;assert free>8*1024**3+reserve
    save(OUT/'protocol/frozen.json',{'source_phase':2687,'source_final_sha256':sha(FIELD/'analysis/final.json'),'material_sha256':sha(MATERIAL),
        'weights_sha256':meta['artifact_sha256'],'primary_windows':primary,'all_windows':windows,
        'raw_scope':'16predeclaredtruth/v0/f0/r0/o0 examples. Allactualtokens/all2560inputcoords processed in792windows;24primaryalltokenraw and792twoqueryraw retained. Some windows duplicate same actualrow; not independent replications.',
        'maps_scope':'Every8192condition represented through frozen all-coordinate input-response maps for24primaryrows. Linear weighted-input terms can be exactly derived from native input maps; not reconstructing nonlinear normalization or later layers.',
        'no_intervention':True,'products_storage':'NativeBF16 product exactly representable inFP32, asserted; accumulationFP64 analysis not model precision.',
        'budget':{'free_bytes':free,'conservative_phase_reserve':reserve,'floor':8*1024**3},'code_sha256':sha(Path(__file__))})
    response=response_maps(primary,windows,weights);raw,manifest=raw_terms(rows,primary,windows,weights);emb=verify_checkpoint_embeddings(rows)
    maximum=max(w['native_linear_max_abs_residual'] for r in raw for w in r['windows'])
    checks={'16completeoperationmaps_all24rows_all2560':len(response)==16,'16rawcases_all792windows_all2560':len(raw)==16,
        '64checkpoint_embeddings_exact':emb['all_checks_passed'],'native_weight_artifact_unchanged':sha(ATT/'weights/native_qkv_windows.npz')==meta['artifact_sha256']}
    assert all(checks.values())
    summary={'initial_conditions_represented':8192,'primary_rows':24,'predeclared_windows_with_duplicates':792,'published_raw_cases':16,
        'native_projection_max_abs_rounding_residual':maximum,'independent_embedding_audit':emb,'term_files_bytes':sum(x['bytes'] for x in manifest),
        'all_input_coordinates':2560,'actual_model_parameter_interventions':0}
    finish(2688,'Wq/Wk/Wv逐标量乘积、完整输入坐标操作图与检查点词嵌入核验',OUT,
        {'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '读取冻结真实检查点输入行与原生attention归一化输入，逐坐标直接相乘。全部输入坐标参与，不把低值项丢掉，也不把坐标项当作删除效应。',
        r'T_{t,r,k}=W_{r,k}x_{t,k};\quad z^{native}_{t,r}=\sum_kT_{t,r,k}+\epsilon^{round}_{t,r};\quad \Delta T_{r,k}=W_{r,k}\Delta x_k;\quad A_r=\sum_k|T_{r,k}|-|\sum_kT_{r,k}|.',
        'C0018192条件四操作全输入响应映射至24真实完整QKV行；C00216预定例792窗口所有真实token×2560项；C00324主行完整token项、792窗口两查询项与逐输入token和/绝对和；C004所有原生线性投影舍入余项；C00564展示例检查点E逐token逐2560坐标等于H0，保留真实E数组。',
        '单个学习标量的作用被明确定位为某输入坐标与某投影输出坐标间的乘法联系，其条件性首先来自上下文输入x。正负项相消与权重符号反转可直接查询，尚不需要donor。',
        '这是已知线性投影分账；归一化分母、位置旋转和softmax尚在下游，不能称为语义齿轮。792窗口存在重复物理行，24主行不是整个模型参数。完整原场仅16真值/v0例，四功能8192背景通过全坐标操作汇总保留，并非全部token关系原场。FP32精确产品及FP64累加不意味着FP32/FP64模型执行。',
        '自动继续2689至少128前缀的实际普通/低值QKV单标量±多剂量；同层完整归一化/RoPE/softmax预测与真实下游概率分账。其后完成8192独立确认、顺序跨模型和实际客户端交付。')


if __name__=='__main__':main()
