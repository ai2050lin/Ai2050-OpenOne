"""Expanded native model-coordinate replication, with original target orientation.

Q14 old candidates use v0-v1, unlike2687's displayed v1-v0 operation maps.
Never silently flip their expected signs or align coordinate IDs across models.
"""
import argparse,gc,math,shutil,time,os
from collections import defaultdict
import numpy as np
import torch
from transformers import AutoTokenizer,AutoConfig
from phase2620_native_coordinate_contract import *
from phase2621_native_behavior_run import MODELS
from phase2662_symmetric_mapping_contract import load_native
from phase2677_padded_native_runtime import PaddedCapture,padded_inputs,native_pack
from phase2671_native_mlp_field import unbits
from phase2677_source_role_material import encode
from phase2680_full_native_reuse import sign_counts
from phase2683_crossmodel_function_atlas import AnswerBoundaryTokenizer,generate,calibrate,accumulate_chunk
from phase2683_explicit_answer_audit import audit_records

OUT=RESULT/'phase2691_crossmodel_role_confirmation'
MATERIAL=RESULT/'phase2686_independent_role_contract'
KEYS=('qwen14','glm4','ds7','ds7_answer')
PHYSICAL={k:('ds7' if k=='ds7_answer' else k) for k in KEYS}
FUNCTIONS=('truth','mapped_truth','name','cloze')
CANDIDATES=[('h',27,484,'positive'),('h',27,4986,'positive'),('h',29,3589,'negative'),('a',26,4320,'positive'),('a',28,14621,'positive')]
Q14_UNIT_LIMIT=1
Q14_TARGET_CASES=2048


def _norm_candidate_rows(rows):
    return [list(r) for r in rows]


def prepare():
    initial=read(MATERIAL/'material/initial.json');fresh=read(MATERIAL/'material/confirmation.json')
    result={};required=0
    for key in KEYS:
        modelpath=ROOT/'models/hf'/MODELS[PHYSICAL[key]]
        tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True,use_fast=True,trust_remote_code=True)
        cfg=AutoConfig.from_pretrained(modelpath,local_files_only=True,trust_remote_code=True)
        if key=='ds7_answer':tok=AnswerBoundaryTokenizer(tok)
        selected=[r for r in initial if (r['unit']<Q14_UNIT_LIMIT if key=='qwen14' else r['unit']==r['content_instance']==r['mention_order']==0)]
        cal=[r for r in fresh if (r['unit'],r['content_instance'],r['form'],r['roster_order'],r['mention_order'],r['target_index'])==(3,0,0,0,0,0)]
        rows=[];calibration=[]
        for r in selected:
            published=(r['family'],r['language'],r['unit'],r['content_instance'],r['form'],r['roster_order'],r['mention_order'],r['target_index'])==('chronology','en',0,0,0,0,0,0) and r['output_function'] in ('truth','name')
            rows.append(encode(tok,{**r,'case_index':len(rows),'source_case_index':r['case_index'],'published':published,'parameter_published':False}))
        for r in cal:calibration.append(encode(tok,{**r,'case_index':len(calibration),'published':False,'parameter_published':False}))
        assert len(rows)==(Q14_TARGET_CASES if key=='qwen14' else 512) and len(calibration)==64
        assert not {r['prompt'] for r in rows}&{r['prompt'] for r in calibration}
        assert len({r['prompt'] for r in rows})==len(rows) and sum(r['published'] for r in rows)==2
        maxlen=max(len(r['prompt_ids']) for r in rows+calibration);total=max(256,32*math.ceil(maxlen/32))
        same=0
        for i in range(0,len(rows),4):
            group=rows[i:i+4];assert {r['output_function'] for r in group}==set(FUNCTIONS)
            same+=len({tuple(r['prompt_ids'][:r['body_end_token']+1]) for r in group})==1
        assert same==len(rows)//4
        folder=OUT/key
        for name,obj in [('cases',rows),('calibration',calibration)]:
            path=folder/f'material/{name}.json'
            if path.exists():
                if name=='cases' and key=='qwen14' and len(read(path))!=len(obj):
                    save(path,obj)
                else:
                    assert read(path)==obj
            else:
                save(path,obj)
        L,D,K=cfg.num_hidden_layers,cfg.hidden_size,cfg.intermediate_size;N=2*((L+1)*D+L*K)
        upper=17*5*2*N+2*N*8+16*((L+1)*D+3*L*K+2*L*D)+2*(2*N+2*maxlen*(L+1)*D)+128*1024**2
        required+=upper
        result[key]={'cases':len(rows),'groups':len(rows)//8,'per_language_family_groups':len(rows)//128,
            'material_sha256':sha(folder/'material/cases.json'),'calibration_sha256':sha(folder/'material/calibration.json'),
            'layers':L,'hidden':D,'mlp_units':K,'field_length':total,'max_prompt_length':maxlen,'uncompressed_upper':upper,
            'samebody_token_prefix_groups':same,'fullH_published':2}
    c={'models':result,'physical_models':3,'protocols':4,'source_material_sha256':sha(MATERIAL/'material/initial.json'),
        'frozen_parser_sha256':sha(TESTS/'phase2683_explicit_answer_audit.py'),'code_sha256':sha(Path(__file__)),
        'helper_sha256':{name:sha(TESTS/name) for name in ('phase2683_crossmodel_function_atlas.py','phase2680_full_native_reuse.py','phase2677_padded_native_runtime.py')},
        'Q14_candidates':_norm_candidate_rows(CANDIDATES),'contrast_orientation':'v0-v1 exactly asPhase2683; opposite to2687 operation-map target-axis v1-v0. Candidate signs NOTrelabelled.',
        'sampling':'Q14 now uses unit<1 (e0 only): all2content,2form,2roster,2factorder,2target,4functions=2048. Others e0c0 factorder0 but2forms2roster2target4functions=512. New material relative2683; shared acrossmodels, not independent facts.',
        'calibration':'Allmodels64 independent confirmation entitypair3 preflight. Q14/GLM formalbudget32; DS native/direct each64 greedy<=256EOS-only calibration then unchanged2683 rule freezesbudget. Never choose by correctness. Native/direct both preregistered, not choose winning interface.',
        'coordinates':'Every layer H/body/task and every MLPunit/body/task; alltoken fullcoordinate sixfield sums/sumsq globally; all16familylanguage direction/zero/opposition maps andfullamplitudes. NoTopK. RawfullHonly2examples/protocol.',
        'limits':'Different model indices neveraligned; Q14fivenativeaddresses onlytheir ownmodel. Noop/samebodyidentity expected causalcontrol, not semanticgear. Allfailure/partialbackground retained.',
        'storage':{'upper_all_protocols':required,'free_at_material_preflight':shutil.disk_usage(RESULT).free,'floor':8*1024**3}}
    path=OUT/'protocol/frozen.json'
    if path.exists():
        old=read(path)
        assert _norm_candidate_rows(old['Q14_candidates'])==c['Q14_candidates'], 'Q14_candidates'
        for k in ('source_material_sha256','frozen_parser_sha256','helper_sha256'):assert old[k]==c[k],k
        old['sampling']=c['sampling']
        for key in ('glm4','ds7','ds7_answer'):
            assert old['models'][key]['cases']==c['models'][key]['cases'],key
            assert old['models'][key]['groups']==c['models'][key]['groups'],key
            assert old['models'][key]['per_language_family_groups']==c['models'][key]['per_language_family_groups'],key
            assert old['models'][key]['material_sha256']==c['models'][key]['material_sha256'],key
            assert old['models'][key]['calibration_sha256']==c['models'][key]['calibration_sha256'],key
            assert old['models'][key]['layers']==c['models'][key]['layers'],key
            assert old['models'][key]['hidden']==c['models'][key]['hidden'],key
        old['code_sha256']=c['code_sha256']
        old_qwen14=old['models'].get('qwen14',{})
        if old_qwen14.get('cases')!=c['models']['qwen14']['cases']:
            print('2691 frozen protocol migrated',old_qwen14.get('cases'),'->',c['models']['qwen14']['cases'],flush=True)
        old.setdefault('models',{})['qwen14']=c['models']['qwen14']
        save(path,old)
        return old
    assert c['frozen_parser_sha256']==read(MATERIAL/'protocol/frozen.json')['scoring']['parser_sha256']
    assert c['storage']['free_at_material_preflight']>required+8*1024**3
    save(path,c);print('2691 MATERIAL FREEZE',result,'storage',required,flush=True);return c


@torch.inference_mode()
def qualification(model,tok,rows,total,folder):
    results=[]
    for r in rows:
        inp=padded_inputs(model,r['prompt_ids'],tok.eos_token_id,total)
        baseline=model.model(**inp).last_hidden_state.detach().cpu();cap=PaddedCapture(model,())
        try:
            cap.reset(r['body_end_token'],False,len(r['prompt_ids'])-1);cap.enabled=True
            observed=model.model(**inp).last_hidden_state.detach().cpu();cap.enabled=False
            pack=cap.pack();assert pack['h'].shape[1]==pack['a'].shape[1]==2
        finally:cap.close()
        restored=model.model(**inp).last_hidden_state.detach().cpu();assert torch.equal(baseline,observed) and torch.equal(baseline,restored)
        results.append({'case_id':r['case_id'],'native_capture_and_restore_exact':True})
        if len(results)%8==0:print('2691 CAPTURE NOOP',len(results),64,flush=True)
    assert len(results)==64;save(folder/'analysis/native_noops.json',{'all_checks_passed':True,'records':results})


@torch.inference_mode()
def run_one(key):
    assert read(RESULT/'phase2690_fresh_role_qkv_confirmation/analysis/final.json')['all_checks_passed']
    c=prepare();folder=OUT/key;cfg=c['models'][key]
    if (folder/'analysis/completion.json').exists():return
    assert shutil.disk_usage(RESULT).free>cfg['uncompressed_upper']+8*1024**3
    rows=read(folder/'material/cases.json');cal=read(folder/'material/calibration.json')
    target_cell=os.getenv('PHASE2691_TARGET_CELL')
    if target_cell:
        try:
            family,lang=target_cell.rsplit('_',1)
            rows=[r for r in rows if r['family']==family and r['language']==lang]
            assert rows,f'Target cell {target_cell} has no rows'
        except ValueError as e:
            raise ValueError('PHASE2691_TARGET_CELL format should be "family_language", e.g. "syntax_role_zh"') from e
    model,tok=load_native(PHYSICAL[key]);assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    save(folder/'protocol/runtime.json',{'dtype':str(model.dtype),'actual_devices':sorted({str(p.device) for p in model.parameters()}),'quantized':False,'device_map':getattr(model,'hf_device_map',None)})
    qualification(model,tok,cal,cfg['field_length'],folder);generation=calibrate(model,tok,key,folder)
    cap=PaddedCapture(model,());groups=defaultdict(list);t0=time.monotonic()
    for r in rows:groups[(r['family'],r['language'])].append(r)
    try:
        for (family,lang),rr in groups.items():
            stem=family+'_'+lang;commit=folder/f'analysis/cell_{stem}.json'
            if commit.exists():
                for file,h in read(commit)['files'].items():assert sha(folder/file)==h
                continue
            counts={};amp={};mom={};records=[];nt=0;body_identity=0
            for offset in range(0,len(rr),8):
                block=rr[offset:offset+8];base={};body={}
                for r in block:
                    assert shutil.disk_usage(RESULT).free>8*1024**3
                    ids=r['prompt_ids'];task=len(ids)-1;cap.reset(r['body_end_token'],r['published'],task);cap.enabled=True
                    output=model.model(**padded_inputs(model,ids,tok.eos_token_id,cfg['field_length']));cap.enabled=False
                    field_state=output.last_hidden_state[0,task].detach().cpu().clone();pack=cap.pack();mm=cap.moment_pack();nt+=len(ids)
                    for k,v in mm.items():
                        if k not in mom:mom[k]=np.zeros_like(v)
                        mom[k]+=v
                    base[r['output_function'],r['target_index']]={k:pack[k].copy() for k in ('h','a')}
                    body[r['output_function'],r['target_index']]=hashlib.sha256(pack['h'][:,0].tobytes()+pack['a'][:,0].tobytes()).hexdigest()
                    fp=hashlib.sha256(pack['h'].tobytes()+pack['a'].tobytes()).hexdigest();recordpath=folder/f'behavior/case_{r["case_index"]:04d}.json'
                    if recordpath.exists():
                        record=read(recordpath)
                    if not recordpath.exists() or record['case_id']!=r['case_id'] or record['native_field_sha256']!=fp or record['native_body_sha256']!=body[r['output_function'],r['target_index']]:
                        torch.cuda.empty_cache()
                        plain=model.model(input_ids=torch.tensor([ids],device=model.get_input_embeddings().weight.device),use_cache=False).last_hidden_state[0,-1]
                        gap=float((plain.detach().cpu().float()-field_state.float()).abs().max());del plain
                        behavior=generate(model,tok,r,key,generation['max_new_tokens'])
                        record={k:r[k] for k in ('case_index','source_case_index','case_id','family','language','unit','content_instance','form','roster_order','mention_order','target_index','output_function','target','alternate')}
                        record.update(**behavior,native_field_sha256=fp,native_body_sha256=body[r['output_function'],r['target_index']],padded_natural_state_max_abs=gap)
                        save(recordpath,record)
                    if r['published']:
                        path=folder/f'field/case_{r["case_index"]:04d}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                        raw=native_pack(pack,True,False)
                        if path.exists():
                            with np.load(path) as z:assert set(z.files)==set(raw) and all(np.array_equal(z[k],v) for k,v in raw.items())
                        else:np.savez_compressed(path,**raw)
                    records.append(record);cap.reset(0,False);del output,pack,mm,field_state
                assert len(base)==8
                for target in (0,1):assert len({body[fn,target] for fn in FUNCTIONS})==1;body_identity+=1
                for metric in ('h','a'):
                    delta=np.stack([unbits(base[fn,0][metric]).astype(np.float64)-unbits(base[fn,1][metric]).astype(np.float64) for fn in FUNCTIONS])
                    for name,v in sign_counts(delta).items():
                        k=metric+'__'+name
                        if k not in counts:counts[k]=np.zeros_like(v,dtype=np.uint16)
                        counts[k]+=v
                    for name,v in [('min_abs_delta_sum',np.abs(delta).min(0)),('max_abs_delta_sum',np.abs(delta).max(0))]:
                        k=metric+'__'+name
                        if k not in amp:amp[k]=np.zeros_like(v)
                        amp[k]+=v
                save(folder/'analysis/progress.json',{'conditions':sum(1 for _ in (folder/'behavior').glob('case_*.json')),'total':len(rows),'cell':stem,
                    'cell_cases':len(records),'generation_budget':generation['max_new_tokens'],'elapsed_seconds_this_process':time.monotonic()-t0,'free_bytes':shutil.disk_usage(RESULT).free})
                print('2691',key,stem,len(records),len(rr),flush=True)
            path=folder/f'maps/counts_{stem}.npz';path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**counts)
            accumulate_chunk(folder/'maps/global_sums.npz',stem,{**amp,**{'moment__'+k:v for k,v in mom.items()}})
            recordpath=folder/f'analysis/records_{stem}.json';save(recordpath,records)
            save(commit,{'cases':len(rr),'base_groups':len(rr)//8,'actual_tokens':nt,'samebody_fourfunction_checks':body_identity,
                'files':{str(p.relative_to(folder)):sha(p) for p in (path,recordpath)},'all_checks_passed':True})
    finally:cap.close()
    del model;gc.collect();torch.cuda.empty_cache()
    if not target_cell:
        audit_one(key,c)


def audit_one(key,c):
    folder=OUT/key;cfg=c['models'][key];rows=read(folder/'material/cases.json');records=[];counts={};tokens=0
    for fam,lang in dict.fromkeys((r['family'],r['language']) for r in rows):
        stem=fam+'_'+lang;cell=read(folder/f'analysis/cell_{stem}.json');assert cell['all_checks_passed'] and cell['base_groups']==cfg['per_language_family_groups']
        for p,h in cell['files'].items():assert sha(folder/p)==h
        tokens+=cell['actual_tokens'];records.extend(read(folder/f'analysis/records_{stem}.json'))
        with np.load(folder/f'maps/counts_{stem}.npz') as z:
            for k in z.files:
                a=z[k];assert a.max()<=cfg['per_language_family_groups']
                if k not in counts:counts[k]=np.zeros_like(a,np.uint16)
                counts[k]+=a
            for metric in ('h','a'):assert np.array_equal(z[metric+'__all4_same_nonzero'],z[metric+'__all4_positive']+z[metric+'__all4_negative'])
    with np.load(folder/'maps/global_sums.npz') as z:
        assert len(z['completed_chunks'])==16
        for k in z.files:
            if k!='completed_chunks':assert np.isfinite(z[k]).all()
    np.savez_compressed(folder/'maps/global_counts.npz',**counts)
    records.sort(key=lambda r:r['case_index']);assert [r['case_index'] for r in records]==list(range(cfg['cases']))
    save(folder/'analysis/records.json',records);score=audit_records(rows,records);save(folder/'analysis/explicit_answer_audit.json',score)
    candidates=[]
    if key=='qwen14':
        for metric,l,j,sign in CANDIDATES:
            candidates.append({'metric':metric,'checkpoint_or_layer':l,'coordinate':j,'old_expected_sign_v0_minus_v1':sign,
                'new_all_four_same_old_direction':int(counts[metric+'__all4_'+sign][l,1,j]),'denominator':cfg['groups'],
                'other_direction':int(counts[metric+'__all4_'+('negative' if sign=='positive' else 'positive')][l,1,j])})
    manifest=[]
    for r in rows:
        if r['published']:
            p=folder/f'field/case_{r["case_index"]:04d}.npz'
            with np.load(p) as z:assert z['full__h'].shape==(cfg['layers']+1,len(r['prompt_ids']),cfg['hidden'])
            manifest.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'case_index':r['case_index']})
    assert len(manifest)==2;save(folder/'analysis/published_manifest.json',manifest)
    report={'all_checks_passed':True,'cases':len(records),'base_groups':cfg['groups'],'actual_tokens':tokens,'behavior':score['groups'],
        'Q14_old_candidates':candidates,'generation_protocol':read(folder/'protocol/generation.json'),
        'padded_natural_state_different':sum(r['padded_natural_state_max_abs']>0 for r in records),
        'padded_natural_state_max_abs':max(r['padded_natural_state_max_abs'] for r in records),
        'all_background_global_direction_addresses':{k:np.argwhere(v==cfg['groups']).tolist() for k,v in counts.items() if k.endswith(('all4_positive','all4_negative'))}}
    save(folder/'analysis/completion.json',report);print('2691 COMPLETE PROTOCOL',key,flush=True);return report


def finalize():
    c=prepare();reports={k:audit_one(k,c) for k in KEYS}
    native=read(OUT/'ds7/analysis/records.json');direct=read(OUT/'ds7_answer/analysis/records.json')
    assert [r['source_case_index'] for r in native]==[r['source_case_index'] for r in direct]
    pair={'cases':512,'same_body_all_coordinate_hashes':sum(a['native_body_sha256']==b['native_body_sha256'] for a,b in zip(native,direct)),
        'native_final_available':sum(r['final_answer_available'] for r in native),'direct_final_available':sum(r['final_answer_available'] for r in direct)}
    save(OUT/'analysis/DS_pair.json',pair)
    checks={'Q14actual2048':reports['qwen14']['cases']==2048,'other3protocols512':all(reports[k]['cases']==512 for k in KEYS[1:]),
        'all3models_native':all(not read(OUT/k/'protocol/runtime.json')['quantized'] for k in KEYS),
        'all4capture64noops':all(read(OUT/k/'analysis/native_noops.json')['all_checks_passed'] for k in KEYS),
        'parser_frozen':sha(TESTS/'phase2683_explicit_answer_audit.py')==c['frozen_parser_sha256']}
    assert all(checks.values())
    finish(2691,'Qwen14B2048扩大确认与GLM/DS顺序原生复验：旧候选和全部背景分账',OUT,
        {'provenance':str(Path(__file__)),'summary':{'models':reports,'DS_pair':pair},'checks':checks},
        '各模型使用自己的分词、原生非量化BF16与全坐标测量，Q14完整反平衡扩大到4096，保留五个旧地址和所有未通过地址背景。DS原生及显式答案区在正式输出前同时冻结并独立校准。',
        r'd_{b,f,l,q,j}=X_{v0,b,f,l,q,j}-X_{v1,b,f,l,q,j};\quad C^+_{l,q,j}=\sum_b\prod_f[d_{b,f,l,q,j}>0].',
        'C001Q14 2048条件/512四功能目标组；C002GLM4 512；C003DS原生512及显式答案区512，各64独立实体校准；C004每协议64实际无操作；C005所有层H/MLP全部坐标16族语言图与全部token矩；C006每协议2完整H展示；C007五旧Q14地址原方向复验及完整背景枚举、全部输出评分分账。',
        '跨模型比较的是各自坐标中的条件复用规律，不是直接对齐坐标编号。扩大Q14反平衡可以区分旧有限材料同号与新实例、形式和顺序下的稳定性，阴性只限定具体条件门。',
        '此处v0-v1沿用2683候选方向，与2687目标操作图v1-v0相反，不能未经换向直接比较符号。Q14 2048重用64基础实体内容实例，其余模型512固定事实顺序0；跨模型共享材料不等于独立事实。DS两接口共享权重，答案区关闭thinking是显式接口改变。固定场与自然生成存在数值差，明确答案匹配非语义推理证明。',
        '继续2692连接真实输入参数、内生归一化、路由与MLP/输出计算的带舍入余项账本，2693加入重要完整坐标客户端类型、真实参数查询并实际验收后清理；再自动评估同目标后续。')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare',*KEYS,'finalize'));args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='finalize':finalize()
    else:run_one(args.action)
