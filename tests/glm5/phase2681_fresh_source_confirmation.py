"""All-family fresh confirmation with dense streaming, not a new deletion gate."""
import gc,hashlib,shutil,time
from collections import Counter
import numpy as np
import torch
from transformers import AutoTokenizer
from phase2620_native_coordinate_contract import *
from phase2670_native_mlp_contract import LAYERS,SITES
from phase2677_padded_native_runtime import PAD_LENGTH,MAX_NEW_TOKENS,PaddedCapture,padded_inputs,native_pack,summarize_behavior
from phase2677_source_role_material import evaluate
from phase2681_fresh_source_material import build
from phase2671_native_mlp_field import unbits
from phase2679_native_source_capture import NativeSourceCapture
from phase2679_native_source_ledger import pack_case,real_token_data,exact_bits
from phase2680_native_mlp_source_paths import one_case
from phase2680_full_native_reuse import sign_counts,FUNCTIONS
from phase2662_symmetric_mapping_contract import load_native

OUT=RESULT/'phase2681_fresh_source_confirmation'
PRIOR=RESULT/'phase2680_native_mlp_source_paths'
SOURCE=RESULT/'phase2679_native_source_ledger'


def prepare():
    assert read(PRIOR/'analysis/final.json')['all_checks_passed']
    if (OUT/'protocol/frozen.json').exists():
        p=read(OUT/'protocol/frozen.json');assert sha(OUT/'material/cases.json')==p['material_sha256'];return read(OUT/'material/cases.json'),p
    pre=read(RESULT/'phase2677_source_role_contract/analysis/fresh_material_preflight.json');assert pre['all_checks_passed']
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True);rows=build(tok)
    nH,nA=37*2*2560,36*2*9728;D,K=2560,9728
    full=sum(2*len(r['prompt_ids'])*37*D+2*(nH+nA) for r in rows if r['published'])
    full+=sum(2*len(r['prompt_ids'])*4*(3*K+4*D) for r in rows if r['parameter_published'])
    source_raw=0
    for r in rows:
        if r['published']:
            T=len(r['prompt_ids']);source_raw+=4*(2*(4*32*128+2*T*8*128+2*32*T+2*(5*D+3*K))+8*2*T+8)
    moments=64*16*(37*D+36*(3*K+2*D));maps=16*4*(nH+nA)*(5*2+2*8)
    budget={'free_bytes':shutil.disk_usage(ROOT).free,'published_native_upper_bytes':full,'64_published_source_upper_bytes':source_raw,
        '64_alltoken_moment_bytes':moments,'full_coordinate_sign_amplitude_map_bytes':maps,'material_records_graph_reserve':256*1024**2,'floor_bytes':8*1024**3}
    need=sum(budget[k] for k in ('published_native_upper_bytes','64_published_source_upper_bytes','64_alltoken_moment_bytes','full_coordinate_sign_amplitude_map_bytes','material_records_graph_reserve'))
    budget['fits_without_compression']=budget['free_bytes']>=need+budget['floor_bytes'];assert budget['fits_without_compression'],budget
    with np.load(PRIOR/'maps/all_native_four_function_family_counts.npz') as z:
        frozen={k:(z[k]==4) for k in z.files if any(k.endswith(s) for s in ('all4_same_nonzero','all4_positive','all4_negative'))}
    OUT.joinpath('maps').mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/'maps/frozen_previous_family_masks.npz',**frozen)
    save(OUT/'material/cases.json',rows)
    plan={'material_sha256':sha(OUT/'material/cases.json'),'frozen_masks_sha256':sha(OUT/'maps/frozen_previous_family_masks.npz'),
        'prior_global_counts_sha256':sha(PRIOR/'maps/all_native_four_function_global_counts.npz'),'source_weights_sha256':read(SOURCE/'protocol/native_weights.json')['weights_sha256'],
        'cases':4096,'factors':'8families*2languages*4newentitypairs*2newlexicalcontents*2forms*2orders*2targets*4outputfunctions',
        'scope':'New lexicalcontents and entitynames in all8families, reused controlledstructuraltemplates. q0/p0 ontruth;name/clozenoqueriedentityname. Novelbodystrings notnewabstractsemantics.',
        'prior_review':'0taskcoordinates pass all64basegroups across4functions globally, while all16family/languagecharts contain conditionalconsistentcoordinates. Preserveallpartialcounts; no claim that a unique universalneuron exists or that the route isclosed.',
        'frozen_feature_algorithm':'EveryH/a nativecoordinate, fourfunction targetv0-v1 same-nonzero/positive/negative/anyzero/opposed signs. Sourceoldcharts had4basegroupsatf0o0. Testnew8basegroups separatelyforeachf/o; all32stress separately. Signedorientationstrongerthanwithinbasefunctionagreement. NoTopK.',
        'amplitude_maps':'Perphysicalcoordinate sum of minimum/maximum absolute targetdelta across4functions within eachbase. Not averaging coordinates; fullmagnitude traces onpublishedcases, alltoken percoordinate6fieldmoments onallcases.',
        'source_replication':'512balancednew f0/o0 prefixes,e2/3,c0/1,v0/1,all4functions. Actualsource QKV/P and completehead/source/coordinate observednormalizer+fiveMLPpaths computed; percase everyrole signed/absolute totals recorded.64publishednativefactorpacks retained; unshown448factors streamed, neverpersisted asgiantfields. Existingnative learnedWo/gamma/gate/up/down exactweights reusedandverified.',
        'streaming':'Only64publishedrawH/a+fullH,16ofthese full4layerMLP. Unpublishedfullfields are analyzedonline, notsavedthenearlydeleted.512sourceobservations analyzedallcoordinates;64rawsourcepacks saved. 64alltokenmoments,64form/orderconditionaldensemaps.',
        'numeric':'Qwen4nativeBF16CUDA,160fixedpositions,actualqueryindices,mask0paddingexcluded;greedy16budget. No donor/quantization. Realweightsunmodified.',
        'limits':'Large numbers of passingcoords mayreflect sharedgeometry, entitybinding, prompt scaffolds, or correlatedtests. q/p fixed inthisconfirmation; no semantic-specificity orcausalclosure inferred. Bodyacrossfunctionidentity is a causalprefixcontrol. Formchanges canchange the lexicalpredicate, notpurenuisance.',
        'budget':budget}
    save(OUT/'protocol/frozen.json',plan);print('2681 FROZEN',budget,flush=True);return rows,plan


def source_weights(model):
    path=SOURCE/'weights/native_source_weights.npz';meta=read(SOURCE/'protocol/native_weights.json');assert sha(path)==meta['weights_sha256']
    with np.load(path) as z:
        weights={k:unbits(z[k]).astype(np.float64) for k in z.files}
        for l in LAYERS:
            b=model.model.layers[l];arr=NativeSourceCapture.array
            assert np.array_equal(exact_bits(arr(b.self_attn.o_proj.weight)),z[f'L{l}__Wo'])
            assert np.array_equal(exact_bits(arr(b.post_attention_layernorm.weight)),z[f'L{l}__gamma'])
            for ll,j in SITES:
                if ll!=l:continue
                for kind in ('gate','up','down'):
                    w=getattr(b.mlp,kind+'_proj').weight
                    assert np.array_equal(exact_bits(arr(w[:,j] if kind=='down' else w[j,:])),z[f'L{l}__J{j}_{kind}'])
    return weights,meta


@torch.inference_mode()
def run(model,tok,cases):
    for name in ('analysis','field','source_field','maps'):OUT.joinpath(name).mkdir(parents=True,exist_ok=True)
    weights,meta=source_weights(model);recpath=OUT/'analysis/records.jsonl'
    records=[json.loads(s) for s in recpath.read_text(encoding='utf-8').splitlines()] if recpath.exists() else []
    assert [r['case_index'] for r in records]==list(range(len(records)))
    begin=len(records)//256*256 if len(records)<len(cases) else len(cases)
    cap=PaddedCapture(model,LAYERS);moments={};counts={};amplitudes={};base={};nt=Counter();nc=Counter();t0=time.monotonic()
    try:
        with NativeSourceCapture(model,LAYERS) as source,recpath.open('a',encoding='utf-8') as stream:
            for i in range(begin,len(cases)):
                r=cases[i];function=r['output_function'];ids=list(r['prompt_ids']);task=len(ids)-1;cell=2*r['form']+r['mention_order']
                if shutil.disk_usage(OUT).free<8*1024**3:raise RuntimeError('8GiBfloor; preserve allcompletedoutputs, do notdelete unrelatedfiles')
                cap.reset(r['body_end_token'],r['published'],task);cap.enabled=True
                source.reset(r['body_end_token'],task);source.enabled=r['source_selected']
                result=model.model(**padded_inputs(model,ids,tok.eos_token_id));source.enabled=False;cap.enabled=False
                chosen=int(model.lm_head(result.last_hidden_state[0,task]).float().argmax());native=chosen
                pack=cap.pack();mm=cap.moment_pack();assert all(np.isfinite(v).all() for v in mm.values())
                digest=hashlib.sha256(pack['h'].tobytes()+pack['a'].tobytes()).hexdigest()
                if i<len(records):assert records[i]['native_field_sha256']==digest,'Partialgroup replaychanged nativefield'
                acc=moments.setdefault(function,{})
                for k,v in mm.items():
                    if k not in acc:acc[k]=np.zeros_like(v)
                    acc[k]+=v
                nt[function]+=len(ids);nc[function]+=1
                base[function,r['target_index']]={k:pack[k].copy() for k in ('h','a')}
                if r['published']:
                    path=OUT/f'field/case_{i:04d}.npz';data=native_pack(pack,True,r['parameter_published'])
                    if path.exists():
                        with np.load(path) as z:assert set(z.files)==set(data) and all(np.array_equal(z[k],v) for k,v in data.items())
                    else:np.savez_compressed(path,**data)
                path_units=None;bridge=None
                if r['source_selected']:
                    source_data=real_token_data(source.pack(),len(ids))
                    bridge=all(np.array_equal(unbits(pack['h'][l]),d['residual_before_attention']) and np.array_equal(unbits(pack['a'][l]),d['mlp_a']) for l,d in source_data.items());assert bridge
                    source_dense,path_units=one_case(r,source_data,weights,meta['layers']);del source_dense
                    if r['published']:
                        path=OUT/f'source_field/case_{i:04d}.npz';data=pack_case(source_data)
                        if path.exists():
                            with np.load(path) as z:assert set(z.files)==set(data) and all(np.array_equal(z[k],v) for k,v in data.items())
                        else:np.savez_compressed(path,**data)
                    del source_data
                if i>=len(records):
                    generated=[]
                    for step in range(MAX_NEW_TOKENS):
                        generated.append(chosen);ids.append(chosen)
                        if chosen==tok.eos_token_id or step+1==MAX_NEW_TOKENS:break
                        result=model.model(**padded_inputs(model,ids,tok.eos_token_id))
                        chosen=int(model.lm_head(result.last_hidden_state[0,len(ids)-1]).float().argmax())
                    text=tok.decode(generated,skip_special_tokens=True)
                    record={k:r[k] for k in ('case_index','case_id','family','language','unit','content_instance','form','mention_order','target_index','output_function','polarity','mapping','published','parameter_published','source_selected','target','alternate')}
                    record.update(generated=text,generated_ids=generated,native_id=native,eos=tok.eos_token_id in generated,native_field_sha256=digest,source_native_bridge=bridge,source_units=path_units,**evaluate(r,text))
                    stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush();records.append(record)
                if (i+1)%8==0:
                    assert len(base)==8
                    for metric in ('h','a'):
                        d=np.stack([unbits(base[f,0][metric]).astype(np.float64)-unbits(base[f,1][metric]).astype(np.float64) for f in FUNCTIONS])
                        for name,value in sign_counts(d).items():
                            key=metric+'__'+name
                            if key not in counts:counts[key]=np.zeros((4,)+value.shape,dtype=np.uint16)
                            counts[key][cell]+=value
                        for name,value in (('minimum_abs_delta_sum',np.abs(d).min(axis=0)),('maximum_abs_delta_sum',np.abs(d).max(axis=0))):
                            key=metric+'__'+name
                            if key not in amplitudes:amplitudes[key]=np.zeros((4,)+value.shape)
                            amplitudes[key][cell]+=value
                    base={}
                if (i+1)%256==0:
                    assert all(v==64 for v in nc.values()) and len(nc)==4 and not base
                    group=r['family']+'_'+r['language']
                    np.savez_compressed(OUT/f'maps/fresh_{group}.npz',**counts,**amplitudes)
                    for fn,a in moments.items():
                        np.savez_compressed(OUT/f'maps/alltoken_{group}_{fn}.npz',**a)
                        save(OUT/f'analysis/moments_{group}_{fn}.json',{'cases':nc[fn],'actual_tokens':nt[fn],'padding_included':0})
                    counts={};amplitudes={};moments={};nt=Counter();nc=Counter()
                if (i+1)%16==0:
                    save(OUT/'analysis/progress.json',{'cases':i+1,'total':4096,'elapsed_seconds':time.monotonic()-t0,'free_bytes':shutil.disk_usage(OUT).free,'last':r['case_id']})
                    print('2681 FRESH FULL COORDINATES',i+1,'/4096',flush=True)
                del mm,pack,result;cap.reset(0,False);source.reset(0,0)
    finally:cap.close()
    save(OUT/'analysis/records.json',records)
    manifest=[]
    for folder in ('field','source_field'):
        for r in cases:
            if r['published']:
                path=OUT/folder/f'case_{r["case_index"]:04d}.npz';manifest.append({'path':str(path.resolve()),'case_index':r['case_index'],'kind':folder,'published':True,'bytes':path.stat().st_size})
    save(OUT/'analysis/raw_manifest.json',manifest);return records


def confirm():
    reports={};newmasks={};oldmasks={}
    with np.load(OUT/'maps/frozen_previous_family_masks.npz') as z:old={k:z[k].copy() for k in z.files}
    for path in sorted((OUT/'maps').glob('fresh_*.npz')):
        group=path.stem.removeprefix('fresh_');reports[group]={}
        with np.load(path) as z:
            for metric in ('h','a'):
                reports[group][metric]={}
                for kind in ('all4_same_nonzero','all4_positive','all4_negative'):
                    prior=old[f'{group}__{metric}__{kind}'];current=z[f'{metric}__{kind}'];passed=current==8;wide=current.sum(axis=0)==32
                    reports[group][metric][kind]={'old_count_by_layer_body_task':prior.sum(axis=-1).tolist(),
                        'replicated_by_form_order_layer_body_task':(passed&prior[None]).sum(axis=-1).tolist(),
                        'all_form_order_replicated_by_layer_body_task':(wide&prior).sum(axis=-1).tolist(),
                        'all_new_passed_by_form_order_layer_body_task':passed.sum(axis=-1).tolist()}
                    if kind=='all4_same_nonzero':oldmasks[group,metric]=prior;newmasks[group,metric]=passed[0]&prior
    graph={};labels=sorted(reports)
    for metric in ('h','a'):
        for label,src in (('old',oldmasks),('fresh_f0o0_replicated',newmasks)):
            masks=np.stack([src[g,metric] for g in labels]);L,Q,D=masks.shape[1:]
            overlap=np.empty((L,Q,len(labels),len(labels)),dtype=np.int32)
            for i in range(len(labels)):
                for j in range(len(labels)):overlap[:,:,i,j]=(masks[i]&masks[j]).sum(axis=-1)
            graph[f'{metric}__{label}_intersections']=overlap
    np.savez_compressed(OUT/'maps/family_coordinate_reuse_graph.npz',**graph)
    result={'reports':reports,'family_language_order':labels,'new_per_form_order_base_groups':8,'old_base_groups':4,'new_all_form_order_groups':32,
        'graph':'Exactsharedcoordinate membershipcounts atfixedlayer/query, notcausalgears or statisticalsignificance. Bodycontrolseparate. Everyoriginalcoordinate countandmin/maxdeltaamplitude retained.',
        'claims':'No universal-coordinate mechanism claim. Comparef0o0 strictscopefirst; otherforms/orders separately, whole32stresssecondary. Anyzero/opposedcounts mayoverlap.'}
    save(OUT/'analysis/confirmation.json',result);return result


def main():
    assert not (OUT/'analysis/final.json').exists();cases,plan=prepare();model,tok=load_native('qwen4')
    save(OUT/'protocol/model.json',{'dtype':str(model.dtype),'quantized':getattr(model,'is_quantized',False),'actual_devices':sorted({str(p.device) for p in model.parameters()}),'fixed_execution':PAD_LENGTH})
    records=run(model,tok,cases);del model;gc.collect();torch.cuda.empty_cache();comparison=confirm()
    source_rows=[r for r in records if r['source_selected']]
    checks={'4096_conditions':len(records)==4096,'512_real_source_replicates':len(source_rows)==512 and all(r['source_native_bridge'] for r in source_rows),
        '64_alltoken_moment_groups':len(list((OUT/'maps').glob('alltoken_*.npz')))==64,'16_family_language_fullcharts':len(comparison['reports'])==16,
        '128_published_native_and_source_packs':len(read(OUT/'analysis/raw_manifest.json'))==128,
        'material_immutable':sha(OUT/'material/cases.json')==plan['material_sha256'],'frozen_masks_immutable':sha(OUT/'maps/frozen_previous_family_masks.npz')==plan['frozen_masks_sha256'],
        'known_parameter_accounting':max(b['reconstruction_max_abs'] for r in source_rows for u in r['source_units'] for b in u['branches'].values())<1e-10}
    assert all(checks.values()),checks
    compact={group:{metric:{kind:{'prior_task_total':sum(a[1] for a in v['old_count_by_layer_body_task']),
               'new_f0o0_task_total':sum(a[1] for a in v['replicated_by_form_order_layer_body_task'][0]),
               'all_forms_orders_task_total':sum(a[1] for a in v['all_form_order_replicated_by_layer_body_task'])} for kind,v in dd.items()} for metric,dd in data.items()} for group,data in comparison['reports'].items()}
    finish(2681,'4096新实体词汇与句式顺序条件的全坐标复用确认',OUT,{'provenance':str(Path(__file__)),'summary':{'cases':4096,'source_replicates':512,'coordinate_confirmation':compact,'behavior':summarize_behavior(records),'budget':plan['budget']},'checks':checks},
        '根据2680全局单坐标门为空、分族条件纹理仍存在的真实结果，冻结全部旧分族坐标图，在新实体和新词汇内容上逐坐标复核，形式/顺序各自成图，不因全局门失败删掉条件化路线。实际source到MLP账本同步复验512条件。',
        r'd^{b,f}_{\ell,q,j}=X^{b,f,v0}_{\ell,q,j}-X^{b,f,v1}_{\ell,q,j};\quad U_{\ell,q,j}=\sum_b\mathbf1[\min_fd^{b,f}_{\ell,q,j}>0\lor\max_fd^{b,f}_{\ell,q,j}<0];\quad G_{a,b}^{\ell,q}=|S_a^{\ell,q}\cap S_b^{\ell,q}|.',
        'C0014096八族双语、四新实体对、两新内容、两形式、两顺序、两事实目标、四输出功能；C002全部H/MLP坐标四功能目标差的同向/正向/负向/零/反向计数与最小/最大绝对幅值；C00364全实际token全坐标矩；C004旧f0o0四组与新八组逐坐标确认、其他三形式顺序格及全32组压力单列；C005512实际source全head/source/坐标到五MLP输入项复验；C006原生坐标分族交集图；64全H和64source原生包、其中16全MLP原场保留。',
        '语言族图谱不需要假设存在一颗普适真值神经元。完整坐标的条件成员关系、相消和输入来源可构成下一步组合结构的拼图。新图保留所有部分通过和反向条件，不把改变定义的宽松计数冒称旧门通过。',
        '依然是受控模板和有限词汇，字段新字符串不等于新抽象语义。每族每形式顺序仅八实体内容组；坐标共享可源于通用计算或实体绑定，不能直接称为语义齿轮。truth探问p/q固定0。正文跨功能一致是因果前缀控制，非任务不变语义。未展示原轨迹在线处理未持久化，全量计算与全量长期保存严格区别。',
        '继续2682数值可分辨的真实标量局部计算复验、2683三模型按协议校准顺序复验、2684图谱交付与完整归档清理；同目标自动续研，不在单坐标普适性阴性处停止。')


if __name__=='__main__':main()
