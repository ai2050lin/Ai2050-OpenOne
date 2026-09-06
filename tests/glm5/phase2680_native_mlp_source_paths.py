"""Observed-normalizer source paths into every incoming coordinate of native units."""
import shutil,time
import numpy as np
from phase2620_native_coordinate_contract import *
from phase2670_native_mlp_contract import LAYERS,SITES
from phase2677_source_role_contract import OUT as CONTRACT,SOURCE
from phase2679_native_source_ledger import SOURCE_ROLES,unpack_case,role_indices,ordered_source_cases
from phase2671_native_mlp_field import unbits
from phase2679_source_coordinate_ledger import attention_ledger,conditional_norm_ledger,input_weight_ledger
from phase2680_full_native_reuse import analyze as analyze_all_native

OUT=RESULT/'phase2680_native_mlp_source_paths'


def one_case(row,data,weights,metadata):
    ri=role_indices(row,len(row['prompt_ids']));dense={};records=[]
    for l in LAYERS:
        d=data[l];att=attention_ledger(d['actual_probability'],d['actual_value'],weights[f'L{l}__Wo'],d['attention_output'],d['native_head_concat'],weights[f'L{l}__Wo_bias'])
        norm=conditional_norm_ledger(d['residual_before_attention'],att,weights[f'L{l}__gamma'],metadata[str(l)]['epsilon'],d['pre_mlp_norm'],d['mlp_x'])
        for ll,j in SITES:
            if ll!=l:continue
            unit={'layer':l,'unit':j,'gate':d['gate'][:,j].tolist(),'up':d['up'][:,j].tolist(),'a':d['mlp_a'][:,j].tolist(),
                'observed_denominator64':norm['observed_denominator64'].tolist(),'norm_accounting_max_abs':float(np.abs(norm['reconstruction_error']).max()),'branches':{}}
            for kind in ('gate','up'):
                w=weights[f'L{l}__J{j}_{kind}'];parts=input_weight_ledger(norm,w,d[kind][:,j]);terms=parts['source_coordinate_terms']
                role=np.stack([terms[:,ri==r].sum(axis=1) for r in range(len(SOURCE_ROLES))],axis=1)
                role_abs=np.stack([np.abs(terms[:,ri==r]).sum(axis=1) for r in range(len(SOURCE_ROLES))],axis=1)
                key=f'L{l}_J{j}__{kind}'
                dense[key+'__native_input']=d['mlp_x']*w[None,:]
                dense[key+'__source_roles']=role;dense[key+'__absolute_source_roles']=role_abs
                dense[key+'__residual_path']=parts['branch_coordinate_terms']['residual']
                dense[key+'__norm_rounding']=parts['branch_coordinate_terms']['rmsnorm_rounding']
                source_abs=role_abs.sum(axis=(1,2));source_signed=role.sum(axis=(1,2))
                unit['branches'][kind]={'role_signed':role.sum(axis=-1).tolist(),'role_absolute_terms':role_abs.sum(axis=-1).tolist(),
                    'source_positive_sum':parts['positive_source_sum'].tolist(),'source_negative_sum':parts['negative_source_sum'].tolist(),
                    'cancellation_fraction':[None if den==0 else float(1-abs(num)/den) for num,den in zip(source_signed,source_abs)],
                    'observed_native_projection':d[kind][:,j].tolist(),'projection_rounding':parts['projection_rounding'].tolist(),
                    'conditional_branch_signed':{k:a.sum(axis=-1).tolist() for k,a in parts['branch_coordinate_terms'].items()},
                    'reconstruction_max_abs':float(np.abs(parts['reconstruction_error']).max())}
            dense[f'L{l}_J{j}__down_native']=d['mlp_a'][:,j,None]*weights[f'L{l}__J{j}_down'][None,:]
            records.append(unit)
    return dense,records


def four_function_reuse(rows,weights):
    """Sign comparison of ALL incoming scalar products; no sparse coordinates."""
    grouped={}
    for r in rows:
        key=tuple(r[k] for k in ('family','language','unit','content_instance'))
        grouped.setdefault(key,{})[(r['output_function'],r['target_index'])]=r
    functions=('truth','mapped_truth','name','cloze');maps={};reports=[]
    for key,cells in grouped.items():
        assert len(cells)==8
        deltas=[];units=[]
        for function in functions:
            pair=[unpack_case(SOURCE/f'field/case_{cells[function,v]["case_index"]:04d}.npz') for v in (0,1)]
            inputs=[];unit_values=[]
            for l,j in SITES:
                x=[p[l]['mlp_x'] for p in pair]
                inputs.append(np.stack([x[0]*weights[f'L{l}__J{j}_{kind}'][None,:]-x[1]*weights[f'L{l}__J{j}_{kind}'][None,:] for kind in ('gate','up')],axis=1))
                unit_values.append(np.stack([pair[0][l][kind][:,j]-pair[1][l][kind][:,j] for kind in ('gate','up','mlp_a')],axis=-1))
            deltas.append(np.stack(inputs));units.append(np.stack(unit_values))
        d=np.stack(deltas);u=np.stack(units)
        same=((d>0).all(axis=0)|(d<0).all(axis=0));zero=(d==0).any(axis=0);opposed=(d>0).any(axis=0)&(d<0).any(axis=0)
        usame=(u>0).all(axis=0)|(u<0).all(axis=0)
        group='_'.join(key[:2]);g=maps.setdefault(group,{k:np.zeros_like(v,dtype=np.uint16) for k,v in (('all_four_same_nonzero',same),('any_zero',zero),('opposed',opposed))})
        for name,v in (('all_four_same_nonzero',same),('any_zero',zero),('opposed',opposed)):g[name]+=v
        reports.append({'family':key[0],'language':key[1],'entity_pair':key[2],'content_instance':key[3],
            'input_same_sign_coordinate_counts':same.sum(axis=-1).tolist(),'input_opposed_sign_coordinate_counts':opposed.sum(axis=-1).tolist(),
            'input_any_zero_coordinate_counts':zero.sum(axis=-1).tolist(),'units_same_direction_all4':usame.tolist(),
            'all4_unit_differences':u.tolist()})
    flat={f'{group}__{name}':a for group,g in maps.items() for name,a in g.items()}
    np.savez_compressed(OUT/'maps/four_function_full_input_sign_counts.npz',**flat)
    result={'base_groups':len(reports),'functions':functions,'sites':SITES,'axes':['site','query(body,task)','projection(gate,up)','physical_input_coordinate'],
        'unit_axes':['function','site','query(body,task)','gate_up_a'],'count_per_family_language':[0,4],'records':reports,
        'body_boundary_is_same_prefix_control':'Body agreement acrossfunctions is expected from identical causalprefixes; do NOT call it abstracttask-invariant computation. Taskboundary is the nontrivial comparison.',
        'boundary':'Each scalar sign pattern is a finite empirical property, not a universal semantic role or necessityclaim. Source factors and all coordinates retained.'}
    save(OUT/'analysis/four_function_reuse.json',result)
    return result


def main():
    assert not (OUT/'analysis/final.json').exists();assert read(SOURCE/'analysis/final.json')['all_checks_passed']
    rows=ordered_source_cases(read(CONTRACT/'material/cases.json'));meta=read(SOURCE/'protocol/native_weights.json')
    path=SOURCE/'weights/native_source_weights.npz';assert sha(path)==meta['weights_sha256']
    with np.load(path) as z:weights={k:unbits(z[k]).astype(np.float64) for k in z.files}
    for folder in ('analysis','maps'):OUT.joinpath(folder).mkdir(parents=True,exist_ok=True)
    # Rebudget from actual post2678/2679 free space. The previous3GiB was a
    # source-stage allowance, not permission to exceed the independent8GiBfloor.
    # No original or published data are removed to make this stage fit.
    D=2560;map_bytes=64*len(SITES)*2*D*8*(2*(1+2*len(SOURCE_ROLES)+2)+1)
    budget={'dense_map_upper_bytes':map_bytes,'records_and_reuse_reserve':384*1024**2,'free_bytes':shutil.disk_usage(OUT).free,'floor_bytes':8*1024**3,
        'allocation_update':'2677forecast3GiB source/laterreserve was consumedby2679upperbound3.061GB. This additional exact analysis allocation is accepted ONLY from actually available postcollection space; no assumedcompression credit beforemeasurement and no data deletion.'}
    assert budget['free_bytes']>=map_bytes+budget['records_and_reuse_reserve']+budget['floor_bytes'],budget
    save(OUT/'protocol/frozen.json',{'material_sha256':sha(CONTRACT/'material/cases.json'),'source_weights_sha256':meta['weights_sha256'],'budget':budget,
        'methods':'Allsource/allhead calculation then observednormalizer conditionalallocation, everyincoming Wgate/up scalar, actual fiveproductunits and everyoutgoingWdowncoordinate; no donor or ablation.',
        'observed_normalizer':'r_t iscomputedfromactualpreRMSNormstate inFP64; nativeRMSNormrounding fullcoordinate residual retained separately. It is not heldfixed underan actualsourceintervention.',
        'reuse':'Allfouroutputfunctions pairedwithinactualfamily/language/entity/content; all inputcoordinates compared, bodytrivialsameprefix distinguishedfromtaskboundary.'})
    groups={}
    for r in rows:groups.setdefault((r['family'],r['language'],r['output_function']),[]).append(r)
    records=[];t0=time.monotonic()
    for group,cases in groups.items():
        acc={}
        for r in cases:
            data=unpack_case(SOURCE/f'field/case_{r["case_index"]:04d}.npz');dense,units=one_case(r,data,weights,meta['layers'])
            for k,v in dense.items():
                if k not in acc:acc[k]=np.zeros_like(v)
                acc[k]+=v
            record={k:r[k] for k in ('case_index','case_id','family','language','unit','content_instance','target_index','output_function','published')};record['units']=units
            records.append(record);save(OUT/f'analysis/case_{r["case_index"]:04d}.json',record)
            print('2680 SOURCE TO MLP',len(records),'/512',flush=True)
            save(OUT/'analysis/progress.json',{'cases':len(records),'total':512,'elapsed_seconds':time.monotonic()-t0,'last':r['case_id']})
        np.savez_compressed(OUT/f'maps/path_{"_".join(group)}.npz',**{k+'__sum':v for k,v in acc.items()})
        save(OUT/f'analysis/group_{"_".join(group)}.json',{'cases':8,'meaning':'Coordinatewise rawsum, notmean/RMS. Signedandabsolute source sums differ; all inputs and allroles retained.'})
    save(OUT/'analysis/records.json',records);reuse=four_function_reuse(rows,weights);full_reuse=analyze_all_native(OUT)
    checks={'512_full_source_paths':len(records)==512,'5_native_units_each':all(len(r['units'])==5 for r in records),'64_dense_coordinate_groups':len(groups)==64,
        '64_fourfunction_base_groups':reuse['base_groups']==64,'all_layers_all_H_a_fourfunction_maps':full_reuse['all_checks_passed'],
        'all_native_projection_accounting':max(b['reconstruction_max_abs'] for r in records for u in r['units'] for b in u['branches'].values())<1e-10,
        'source_weights_immutable':sha(path)==meta['weights_sha256'],'material_immutable':sha(CONTRACT/'material/cases.json')==read(CONTRACT/'protocol/frozen.json')['material_sha256']}
    assert all(checks.values()),checks
    matches=np.asarray([r['units_same_direction_all4'] for r in reuse['records']],dtype=int).sum(axis=0)
    summary={'conditions':512,'candidate_units':SITES,'allfour_direction_by_site_query_gate_up_a':matches.tolist(),'denominator':64,
        'max_projection_accounting_error':max(b['reconstruction_max_abs'] for r in records for u in r['units'] for b in u['branches'].values()),'budget':budget,
        'bodycontrol_not_generalization':True,'physical_input_coordinate_count':D,'all_native_reuse':full_reuse['summary']}
    finish(2680,'来源角色经原生归一化进入五个MLP单元的全输入路径',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '用实际preRMSNorm场、gamma和分母构成条件账本，把所有source项、残差支路及数值余项分别乘进每个真实gate/up标量；逐坐标保留正负相消和来源角色变化，再记录每个down写出坐标。跨四种输出功能只做明确配对的符号比较。',
        r'x_{t,k}=\frac{\gamma_k}{r_t}(h_{t,k}+\sum_s c_{t,s,k}+b_k+\epsilon_{t,k})+\epsilon^{norm}_{t,k};\quad z^{g,s}_{t,j,k}=W^g_{j,k}\frac{\gamma_k}{r_t}c_{t,s,k};\quad \rho=1-\frac{|\sum_{s,k}z_{s,k}|}{\sum_{s,k}|z_{s,k}|}\ (\text{denominator}>0).',
        'C001512四功能条件、四中层五固定原生单元；C002每个source与每个输入坐标的gate/up项及原生down全坐标；C003外部16来源区间的有符号和/绝对值和与数值支路；C00464组完整坐标图；C00564实体—内容组中四种输出功能全部2560输入坐标符号复用、反向、零值分别计数；C006归一化/投影余项和原权重哈希核对；C007所有37个H检查点与所有36层9728个MLP单元的四功能目标差方向图，逐族/双语全部原坐标计数，不限于五个候选。',
        '从来源位置到具体MLP输入标量的条件化计算链已经可复算，能区分同一个原生单元在正文与任务边界、不同输出功能下的输入相消和复用。是否形成更普适的编码结构必须由新实体/内容扩大确认，而不能由恒等式重构自动推出。',
        '实际RMSNorm分母是内生状态，这个分摊不是删除source后的反事实。正文跨功能一致是相同因果前缀的预期控制，不能包装为任务不变抽象编码。条件符号图未证明每个坐标有独立语义，也未闭合自然生成。所有图谱为64有限配对组，模板/胜任度限制仍在。',
        '继续2681全部八族重要模式的新材料扩大确认，再做数值可分辨参数复验与其他模型顺序复验；优先验证可复用的坐标输入结构，不因单零件必要性阴性换掉整条路线。')


if __name__=='__main__':main()
