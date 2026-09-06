"""Native QKV factors and all-head/source/coordinate conditional accounting."""
import gc,shutil,time
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2670_native_mlp_contract import LAYERS,SITES
from phase2677_source_role_contract import OUT as CONTRACT,FIELD,SOURCE as OUT
from phase2677_padded_native_runtime import PAD_LENGTH,padded_inputs
from phase2677_source_role_regions import ROLES
from phase2679_native_source_capture import NativeSourceCapture
from phase2679_source_coordinate_ledger import attention_ledger
from phase2671_native_mlp_field import unbits
from phase2662_symmetric_mapping_contract import load_native
from phase2678_field_audit import audit as audit_field

SOURCE_ROLES=ROLES+('masked_padding',)


def exact_bits(value):
    a=np.asarray(value,dtype=np.float64);f=a.astype(np.float32)
    assert np.isfinite(a).all() and np.array_equal(a,f.astype(np.float64))
    u=f.view(np.uint32);assert not (u&0xffff).any(),'Not native BF16; refuse lossy serialization'
    return (u>>16).astype(np.uint16)


def unpack_case(path):
    rows={}
    with np.load(path) as z:
        for key in z.files:
            layer,name=key.split('__',1);value=z[key]
            rows.setdefault(int(layer[1:]),{})[name]=unbits(value).astype(np.float64) if value.dtype==np.uint16 else value.astype(np.float64)
    return rows


def pack_case(data):
    arrays={}
    for l,row in data.items():
        for key,a in row.items():
            if key=='execution_dtype':assert a=='torch.bfloat16';continue
            if key=='actual_mask':
                assert a is not None
                arrays[f'L{l}__{key}']=np.asarray(a,dtype=np.float64)
            elif key=='scaling':arrays[f'L{l}__{key}']=np.asarray(a,dtype=np.float64)
            else:arrays[f'L{l}__{key}']=exact_bits(a)
    return arrays


def real_token_data(data,total):
    """Discard only nonexistent masked tail positions, never actual tokens."""
    trimmed={}
    for l,row in data.items():
        assert (row['actual_probability'][:,:,total:]==0).all()
        trimmed[l]={}
        for k,v in row.items():
            if k in ('actual_key_post_rope','actual_value'):v=v[:total].copy()
            elif k in ('actual_probability','actual_mask'):v=v[...,:total].copy()
            trimmed[l][k]=v
    return trimmed


def native_weights(model):
    path=OUT/'weights/native_source_weights.npz';arrays={};meta={}
    for l in LAYERS:
        b=model.model.layers[l];a=NativeSourceCapture.array
        arrays[f'L{l}__Wo']=exact_bits(a(b.self_attn.o_proj.weight))
        arrays[f'L{l}__gamma']=exact_bits(a(b.post_attention_layernorm.weight))
        bias=b.self_attn.o_proj.bias
        arrays[f'L{l}__Wo_bias']=exact_bits(np.zeros(model.config.hidden_size) if bias is None else a(bias))
        meta[str(l)]={'epsilon':b.post_attention_layernorm.variance_epsilon,'Wo_shape':list(b.self_attn.o_proj.weight.shape),
            'query_heads':model.config.num_attention_heads,'kv_heads':model.config.num_key_value_heads,'head_dimension':b.self_attn.head_dim}
        for ll,j in SITES:
            if ll!=l:continue
            for kind in ('gate','up','down'):
                w=getattr(b.mlp,kind+'_proj').weight
                arrays[f'L{l}__J{j}_{kind}']=exact_bits(a(w[:,j] if kind=='down' else w[j,:]))
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        with np.load(path) as z:assert set(z.files)==set(arrays) and all(np.array_equal(z[k],v) for k,v in arrays.items())
    else:np.savez_compressed(path,**arrays)
    save(OUT/'protocol/native_weights.json',{'layers':meta,'weights_sha256':sha(path),'storage':'Actual nativeBF16 learned weights, uint16losslessbits; wholeWo and complete inputrows/outputcolumns of5fixedMLPunits.'})
    return {k:unbits(v).astype(np.float64) for k,v in arrays.items()}


def role_indices(row,total=PAD_LENGTH):
    labels=[r['role'] for r in row['token_regions']]+['masked_padding']*(total-len(row['prompt_ids']))
    assert len(labels)==total
    return np.asarray([SOURCE_ROLES.index(k) for k in labels])


def summarize_case(row,data,weights):
    index=role_indices(row,len(row['prompt_ids']));dense={};reports=[]
    for l,d in data.items():
        p,v=d['actual_probability'],d['actual_value'];nq,nh,ns=p.shape;D=d['attention_output'].shape[1]
        head_abs=np.zeros((nq,nh,D));role_abs=np.zeros((nq,len(SOURCE_ROLES),D))
        def observer(head,terms):
            absolute=np.abs(terms);head_abs[:,head]=absolute.sum(axis=1)
            for ri in range(len(SOURCE_ROLES)):role_abs[:,ri]+=absolute[:,index==ri].sum(axis=1)
        ledger=attention_ledger(p,v,weights[f'L{l}__Wo'],d['attention_output'],d['native_head_concat'],weights[f'L{l}__Wo_bias'],observer)
        roles=np.stack([ledger['source_terms'][:,index==ri].sum(axis=1) for ri in range(len(SOURCE_ROLES))],axis=1)
        dense[f'L{l}__head_signed']=ledger['head_terms'];dense[f'L{l}__head_absolute_source_terms']=head_abs
        dense[f'L{l}__role_signed']=roles;dense[f'L{l}__role_absolute_head_source_terms']=role_abs
        # P describes routing; QK dot products are inspected as an arithmetic
        # factor, not assigned semantic meaning. No compressed Q/K coordinate.
        scores=[]
        for h in range(nh):
            kv=h//(nh//v.shape[1])
            scores.append((d['actual_query_post_rope'][:,h]@d['actual_key_post_rope'][:,kv].T)*float(d['scaling']))
        scores=np.stack(scores,axis=1)
        reports.append({'layer':l,'reconstruction_max_abs':float(np.abs(ledger['reconstruction_error']).max()),
            'native_attention_L1':np.abs(d['attention_output']).sum(axis=-1).tolist(),
            'AV_rounding_L1':np.abs(ledger['av_rounding_output']).sum(axis=-1).tolist(),
            'Wo_rounding_L1':np.abs(ledger['wo_rounding_output']).sum(axis=-1).tolist(),
            'head_signed_L1_all_heads':np.abs(ledger['head_terms']).sum(axis=-1).tolist(),
            'role_signed_L1_all_roles':np.abs(roles).sum(axis=-1).tolist(),
            'head_absolute_source_L1':head_abs.sum(axis=-1).tolist(),
            'role_absolute_head_source_L1':role_abs.sum(axis=-1).tolist(),
            'actual_P_by_role_all_heads':np.stack([p[:,:,index==ri].sum(axis=-1) for ri in range(len(SOURCE_ROLES))],axis=-1).tolist(),
            'QK_raw_score_minmax':[float(scores.min()),float(scores.max())],
            'future_P_zero':all(bool((p[qi,:,t+1:]==0).all()) for qi,t in enumerate((row['body_end_token'],row['task_end_token']))),
            'padding_source_contribution_zero':bool((roles[:,SOURCE_ROLES.index('masked_padding')]==0).all())})
    return dense,reports


def ordered_source_cases(cases):
    return sorted([r for r in cases if r['source_selected']],key=lambda r:(r['family'],r['language'],r['output_function'],r['unit'],r['content_instance'],r['target_index']))


@torch.inference_mode()
def collect(model,tok,cases):
    for k in ('field','maps','analysis'):OUT.joinpath(k).mkdir(parents=True,exist_ok=True)
    weights=native_weights(model);selected=ordered_source_cases(cases);groups={};records=[];t0=time.monotonic()
    for r in selected:groups.setdefault((r['family'],r['language'],r['output_function']),[]).append(r)
    assert len(selected)==512 and len(groups)==64 and all(len(v)==8 for v in groups.values())
    with NativeSourceCapture(model,LAYERS) as cap:
        for group,rows in groups.items():
            acc={};pair_base={};pair_sums={};pair_positive={};pair_negative={};pair_count=0
            for r in rows:
                if shutil.disk_usage(OUT).free<8*1024**3:raise RuntimeError('8GiB floor; keep data and do not delete unrelated files')
                path=OUT/f'field/case_{r["case_index"]:04d}.npz'
                if path.exists():data=unpack_case(path)
                else:
                    cap.reset(r['body_end_token'],r['task_end_token']);cap.enabled=True
                    result=model.model(**padded_inputs(model,r['prompt_ids'],tok.eos_token_id));cap.enabled=False;data=real_token_data(cap.pack(),len(r['prompt_ids']))
                    np.savez_compressed(path,**pack_case(data));del result
                    restored=unpack_case(path)
                    assert all(np.array_equal(restored[l][k],v) for l,d in data.items() for k,v in d.items() if k!='execution_dtype')
                # The real-field bridge checks every selected-layer coordinate,
                # not just the 7 candidate sites or an output label.
                with np.load(FIELD/f'field/case_{r["case_index"]:04d}.npz') as z:
                    matched=all(np.array_equal(unbits(z['h'][l]),d['residual_before_attention']) and np.array_equal(unbits(z['a'][l]),d['mlp_a']) for l,d in data.items())
                assert matched,('Same-shape native collection disagrees with2678',r['case_id'])
                dense,reports=summarize_case(r,data,weights)
                for k,v in dense.items():
                    key=k+'__sum'
                    if key not in acc:acc[key]=np.zeros_like(v)
                    acc[key]+=v
                pk=(r['unit'],r['content_instance'])
                if r['target_index']==0:pair_base[pk]={k:v.copy() for k,v in dense.items() if '__role_signed' in k}
                else:
                    base=pair_base.pop(pk);pair_count+=1
                    for k,a in base.items():
                        delta=a-dense[k]
                        if k not in pair_sums:
                            pair_sums[k]=np.zeros_like(delta);pair_positive[k]=np.zeros_like(delta,dtype=np.uint8);pair_negative[k]=np.zeros_like(delta,dtype=np.uint8)
                        pair_sums[k]+=delta;pair_positive[k]+=(delta>0);pair_negative[k]+=(delta<0)
                record={k:r[k] for k in ('case_index','case_id','family','language','unit','content_instance','target_index','output_function','published')}
                record.update(native_allcoordinate_bridge=matched,layers=reports)
                records.append(record);save(OUT/f'analysis/case_{r["case_index"]:04d}.json',record)
                save(OUT/'analysis/progress.json',{'cases':len(records),'total':512,'last':r['case_id'],'elapsed_seconds':time.monotonic()-t0,'free_bytes':shutil.disk_usage(OUT).free})
                print('2679 NATIVE SOURCE',len(records),'/512',flush=True)
                del data,dense
            assert not pair_base and pair_count==4
            extras={}
            for prefix,src in (('target_v0_minus_v1_sum',pair_sums),('target_positive_count',pair_positive),('target_negative_count',pair_negative)):
                extras.update({prefix+'__'+k:v for k,v in src.items()})
            key='_'.join(group);np.savez_compressed(OUT/f'maps/source_{key}.npz',**acc,**extras)
            save(OUT/f'analysis/group_{key}.json',{'cases':8,'target_pairs':4,'sign_count_range':[0,4],
                'roles':SOURCE_ROLES,'head_source_axes':'query,allheads,physicalcoordinate; rolesexternallyannotated; sourcealltermscomputedbeforeaggregation',
                'meaning':'Raw sums of signed terms and absolute head/source terms; no squares/RMS in these source maps. Targetv0-minus-v1 pairedwithinentity/content, no independenceclaim. Everycase nativefactors permit exact reanalysis; native2678alltokenmoments are separate.'})
    records.sort(key=lambda r:r['case_index']);save(OUT/'analysis/records.json',records)
    save(OUT/'analysis/raw_manifest.json',[{'path':str((OUT/f'field/case_{r["case_index"]:04d}.npz').resolve()),'case_index':r['case_index'],'published':r['published'],'bytes':(OUT/f'field/case_{r["case_index"]:04d}.npz').stat().st_size} for r in selected])
    return records


def main():
    assert not (OUT/'analysis/final.json').exists();assert read(FIELD/'analysis/final.json')['all_checks_passed']
    audit_path=FIELD/'analysis/independent_field_audit.json'
    field_audit=read(audit_path) if audit_path.exists() else audit_field()
    assert field_audit['all_checks_passed']
    cases=read(CONTRACT/'material/cases.json');model,tok=load_native('qwen4')
    assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    cfg=model.config;D,K,L,H,V,dh=cfg.hidden_size,cfg.intermediate_size,len(LAYERS),cfg.num_attention_heads,cfg.num_key_value_heads,model.model.layers[0].self_attn.head_dim
    # Complete real-token native factors plus whole Wo plus sums of all heads/roles,
    # and every-coordinate paired role contrasts; no compression assumed.
    maxT=max(len(r['prompt_ids']) for r in cases if r['source_selected'])
    factor_per=L*(2*(2*H*dh+2*maxT*V*dh+2*H*maxT+2*H*dh+2*(5*D+3*K))+8*2*maxT+8)
    weights=L*D*H*dh*2+2*D*(L*2+len(SITES)*3)
    maps=64*L*(8*(2*H*D*2+2*len(SOURCE_ROLES)*D*2)+10*(2*len(SOURCE_ROLES)*D))
    budget={'factor_bytes':512*factor_per,'weight_bytes':weights,'map_bytes':maps,'metadata_reserve_bytes':128*1024**2,'floor_bytes':8*1024**3,'free_bytes':shutil.disk_usage(ROOT).free}
    total=sum(budget[k] for k in ('factor_bytes','weight_bytes','map_bytes','metadata_reserve_bytes'))
    assert total<=3*1024**3 and budget['free_bytes']>=total+budget['floor_bytes'],budget
    save(OUT/'protocol/frozen.json',{'material_sha256':sha(CONTRACT/'material/cases.json'),'cases':512,'groups':64,'native_dtype':str(model.dtype),
        'layers':LAYERS,'roles':SOURCE_ROLES,'budget':budget,'calculation':'Eachhead andeachactualsource andeveryoutputcoordinate computed. Complete actualtoken nativeQKV/P/Wo kept; only nonexistent maskedtail positions excluded after exactzeroP check. Execution still160positions. Giant derived Cartesian tensor rebuilt exactly fromthese factors, not a lowrank approximation. Source maps store signed andabsolute sums, notRMS; native2678sixfieldmoments separate.',
        'semantic_boundary':'P is routing, notmeaning; contextualV is not isolatedwordexclusive causalcredit. All observedFP rounding sources reported separately. No-opbridge is measurement reliability.',
        'pairing':'v0-v1 atfixedfamily/language/entity/content/function,4pairs/group. No selectedsuccess filtering.'})
    records=collect(model,tok,cases);del model;gc.collect();torch.cuda.empty_cache()
    checks={'512_source_cells':len(records)==512,'64_dense_groups':len(list((OUT/'maps').glob('source_*.npz')))==64,
        'native_fullcoordinate_bridge':all(r['native_allcoordinate_bridge'] for r in records),
        'all_queries_future_padding_zero':all(x['future_P_zero'] and x['padding_source_contribution_zero'] for r in records for x in r['layers']),
        'float64_accounting_reconstruction':max(x['reconstruction_max_abs'] for r in records for x in r['layers'])<1e-10,
        'material_immutable':sha(CONTRACT/'material/cases.json')==read(CONTRACT/'protocol/frozen.json')['material_sha256']}
    assert all(checks.values()),checks
    summary={'cases':512,'all_heads':H,'KV_heads':V,'fixed_execution_positions':PAD_LENGTH,'maximum_stored_actual_source_tokens':maxT,'layers':LAYERS,'all_output_coordinates':D,'dense_groups':64,'budget':budget,
        'completed2678_independent_audit':{k:field_audit[k] for k in ('counts','moment_groups','all_actual_tokens','same_source_distinct_actual_prefixes','same_source_comparisons','same_source_changed_pairs','embedding_same_token_failures')},
        'max_reconstruction_error':max(x['reconstruction_max_abs'] for r in records for x in r['layers']),
        'rounding':{key:sum(sum(x[key]) for r in records for x in r['layers']) for key in ('native_attention_L1','AV_rounding_L1','Wo_rounding_L1')}}
    finish(2679,'512四功能前缀的全头来源token到原生坐标账本',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '直接读取真实postRoPE Q/K、V与softmax后的P；每个head分别对全部source、全部head维度和全部输出坐标执行真实Wo乘加。保存原生因子，无稀疏选头、无降维、无donor。完整来源三维项可由原生因子按需精确重算。',
        r'c^h_{t,s,k}=P^h_{t,s}\sum_d W^O_{k,hd}V^{kv(h)}_{s,d};\quad A_{t,k}=\sum_{h,s}c^h_{t,s,k}+b_k+\epsilon^{AV}_{t,k}+\epsilon^{O}_{t,k}+\epsilon^{64}_{t,k};\quad D_{r,k}=C_{r,k}(v=0)-C_{r,k}(v=1).',
        'C001512八族双语四输出功能前缀，128同正文格；C002四中层全heads和全部实际source位置原生QKV/P，执行160位置但仅排除P严格为零的虚拟填充尾部；C003每个source全部物理输出坐标；C00464组全head/外部role坐标有符号和/各项绝对值和及每组4目标配对符号计数；C005逐例与2678所有被测H/MLP坐标原值衔接；C006因果遮罩、填充零贡献及AV/Wo/累计顺序余项分别核对。',
        '已有条件纹理现在可追到具体source的上下文化V、哪个head和真实Wo的哪个坐标项。不同输出功能如何重分配这些项已有全量原生因子与坐标背景可查；这是进一步找规律的可计算拼图，不是把注意力分数自动命名成语义齿轮。',
        'source位置的V含更早层上下文，不能归给孤立原词。role标签为外部区间，跨区token单列mixed。配对变化涉及原生网络全部联动，不是单零件反事实。完整Cartesian项计算后汇总，原生因子可精确重算但未保存全部巨大派生张量。FP64账本不改变原模型BF16舍入。另补记2678初次启动缺少analysis目录、0正式样本时失败，修正目录初始化并独立检查全新目录后重启；未改变材料、精度或测量规则，事件记录在2678/analysis/initialization_incident.json。',
        '继续2680观察到的RMSNorm分母、五原生单元gate/up每个输入项以及down写出坐标；从正负相消、来源角色变化和跨层复用逐步积累结构，不以删除/救援为唯一门。')


if __name__=='__main__':main()
