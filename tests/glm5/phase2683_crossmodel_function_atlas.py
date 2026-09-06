"""Sequential BF16 native-coordinate replication with explicit reasoning budget.

All original coordinates, all failures. No cross-model coordinate alignment.
"""
import argparse,gc,math,os,shutil,time
import numpy as np
import torch
from transformers import AutoConfig,AutoTokenizer
from phase2620_native_coordinate_contract import *
from phase2621_native_behavior_run import MODELS
from phase2662_symmetric_mapping_contract import load_native
from phase2671_native_mlp_field import bits,unbits
from phase2677_padded_native_runtime import PaddedCapture,padded_inputs,native_pack,summarize_behavior
from phase2677_source_role_material import encode,evaluate
from phase2680_full_native_reuse import sign_counts

OUT=RESULT/'phase2683_crossmodel_function_atlas';SOURCE=RESULT/'phase2681_fresh_source_confirmation'
FUNCTIONS=('truth','mapped_truth','name','cloze');KEYS=('qwen14','glm4','ds7','ds7_answer')
MODEL_KEYS={k:('ds7' if k=='ds7_answer' else k) for k in KEYS}


class AnswerBoundaryTokenizer:
    """Explicit paired protocol, NOT an unmodified DS reasoning prompt."""
    def __init__(self,tokenizer):self.tokenizer=tokenizer
    def __getattr__(self,name):return getattr(self.tokenizer,name)
    def __call__(self,*args,**kwargs):return self.tokenizer(*args,**kwargs)
    def apply_chat_template(self,*args,**kwargs):
        text=self.tokenizer.apply_chat_template(*args,**kwargs)
        assert isinstance(text,str) and text.rfind('<think>')>text.rfind('</think>')
        return text+'</think>\n\n'


def material(key,tok):
    if key=='ds7_answer':tok=AnswerBoundaryTokenizer(tok)
    old=read(SOURCE/'material/cases.json');rows=[];cal=[]
    for r in old:
        if r['source_selected']:
            rr=encode(tok,{**r,'case_index':len(rows),'source_case_index':r['case_index'],
                'published':(r['family'],r['language'],r['unit'],r['content_instance'],r['target_index'])==('chronology','en',2,0,0) and r['output_function'] in ('truth','name')})
            rows.append(rr)
        if (r['unit'],r['content_instance'],r['form'],r['mention_order'],r['target_index'])==(0,0,0,0,0):
            cal.append(encode(tok,{**r,'case_index':len(cal),'published':False}))
    assert len(rows)==512 and len(cal)==64 and sum(r['published'] for r in rows)==2
    assert not {r['prompt'] for r in rows}&{r['prompt'] for r in cal}
    return rows,cal


def prepare_all():
    if (OUT/'protocol/frozen.json').exists():
        plan=read(OUT/'protocol/frozen.json')
        for k in KEYS:assert sha(OUT/k/'material/cases.json')==plan['models'][k]['material_sha256']
        return plan
    models={};budget=0
    for key in KEYS:
        path=ROOT/'models/hf'/MODELS[MODEL_KEYS[key]];tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True,trust_remote_code=True)
        cfg=AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=True);L,D,K=cfg.num_hidden_layers,cfg.hidden_size,cfg.intermediate_size
        rows,cal=material(key,tok);T=max(len(r['prompt_ids']) for r in rows);padded=max(160,32*math.ceil(T/32));N=(L+1)*D+L*K
        # Full sign counts per16family/language; full amplitude + six-field moments globally.
        upper=16*5*2*N+2*2*N*8+16*((L+1)*D+3*L*K+2*L*D)+2*(2*2*N+2*T*(L+1)*D)+64*1024**2
        budget+=upper;folder=OUT/key;save(folder/'material/cases.json',rows);save(folder/'material/calibration.json',cal)
        models[key]={'material_sha256':sha(folder/'material/cases.json'),'calibration_sha256':sha(folder/'material/calibration.json'),
            'layers':L,'hidden':D,'mlp_units':K,'field_execution_length':padded,'max_real_prompt_tokens':T,'storage_upper':upper,
            'published_full_H':2,'quantized':False,'requested_precision':'bfloat16'}
    free=shutil.disk_usage(OUT).free
    plan={'source_material_sha256':sha(SOURCE/'material/cases.json'),'models':models,'physical_models':3,'protocol_runs':4,'free_before':free,'storage_upper_all_protocols':budget,'floor':8*1024**3,
          'fits_without_compression':free>=budget+8*1024**3,
          'generation':'Native chat, actual unpadded prompt, greedy use_cache=True; field uses fixed padded prefill separately. Compare first-token numeric state, do not assume identical execution shapes.',
          'DS_budget':'64 calibration prompts (entitypair0 disjoint formalpairs2/3), generate up to256. Freeze formal budget=256 if any calibration lacks EOS, otherwise max(32,16*ceil(maxused/16)), capped256. Calibrate before ANY formal DS forwards.',
          'DS_scoring':'When native template opens <think>, only text after generated </think> is final answer; absence is unavailable final answer, not absence of language mechanism. Raw IDs/text always retained. Factual cloze prefill may sit inside native reasoning scaffold, explicitly different protocol.',
          'DS_answer_boundary_replication':'Tokenizer preflight found all512nativeDS prompts open<think>, so BEFORE any formalfreeze/modelrun add a paired512protocol explicitly closing emptythinking with </think> before the factualassistantprefill. Native512 retained separately; no best-protocol selection by testanswers. Bothprotocols have own64disjointentitycalibration. This is an explicit interface intervention, not originalDSreasoning or architecture-onlycomparison.',
          'coordinate_scope':'Every H checkpoint and MLP intermediate at body/task; four-function target sign counts all16family/language groups (4bases each), global64groups. Native coordinate IDs never aligned acrossmodels.',
          'amplitude':'Global all64base min/max absolute four-function target delta sums, percoordinate, not per-family amplitudes. Alltoken full-coordinate six-field sum/sumsq global. NoTopK or dimensional projection.',
          'storage':'Only2published fullH examples/model (chronology EN truth/name, e2c0v0); all other raw fullfields processed online, never written. All cases include behavior and nativefield SHA.'}
    assert plan['fits_without_compression'],plan;save(OUT/'protocol/frozen.json',plan);return plan


def final_text(row,raw,key):
    open_think=row['prompt'].rfind('<think>')>row['prompt'].rfind('</think>')
    if MODEL_KEYS[key]=='ds7' and open_think:
        if '</think>' not in raw:return '',False
        return raw.rsplit('</think>',1)[1].strip(),True
    return raw,True


def accumulate_chunk(path,group,values):
    """Index and full-coordinate sums commit atomically as one artifact."""
    previous=[]
    if path.exists():
        with np.load(path) as z:
            previous=z['completed_chunks'].tolist()
            if group in previous:return False
            values={k:v+z[k] for k,v in values.items()}
    temporary=path.with_name(path.stem+'.pending.npz')
    np.savez_compressed(temporary,completed_chunks=np.asarray(previous+[group]),**values)
    os.replace(temporary,path);return True


@torch.inference_mode()
def generate(model,tok,row,key,budget):
    ids=torch.tensor([row['prompt_ids']],device=model.get_input_embeddings().weight.device)
    # No output_scores tensors for hundreds of steps: retain generated IDs only.
    seq=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=budget,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
    tokens=seq[0,len(row['prompt_ids']):].tolist();raw=tok.decode(tokens,skip_special_tokens=False);clean=tok.decode(tokens,skip_special_tokens=True)
    # Some tokenizers mark think delimiters as special. Parse the raw delimiter,
    # then decode the final token suffix using the model tokenizer if necessary.
    if MODEL_KEYS[key]=='ds7' and '</think>' in raw:
        suffix=raw.rsplit('</think>',1)[1]
        clean=tok.decode(tok.encode(suffix,add_special_tokens=False),skip_special_tokens=True)
        available=True
    else:clean,available=final_text(row,clean,key)
    eos=model.generation_config.eos_token_id;es={eos} if isinstance(eos,int) else set(eos or [])
    return {'generated_ids':tokens,'generated_raw':raw,'generated':clean,'eos':any(t in es for t in tokens),'final_answer_available':available,
            'native_open_think':row['prompt'].rfind('<think>')>row['prompt'].rfind('</think>'),**evaluate(row,clean)}


def calibrate(model,tok,key,folder):
    frozen=folder/'protocol/generation.json'
    if frozen.exists():return read(frozen)
    if MODEL_KEYS[key]!='ds7':
        protocol={'max_new_tokens':32,'calibration_required':False,'generation_cache':True};save(frozen,protocol);return protocol
    rows=read(folder/'material/calibration.json');path=folder/'analysis/calibration.jsonl';rr=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    with path.open('a',encoding='utf-8') as f:
        for i,r in enumerate(rows[len(rr):],len(rr)):
            out=generate(model,tok,r,key,256);rec={'case_index':i,'family':r['family'],'language':r['language'],'output_function':r['output_function'],**out}
            f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();rr.append(rec)
            save(folder/'analysis/calibration_progress.json',{'cases':len(rr),'total':64});print('2683DS CALIBRATION',len(rr),'/64',flush=True)
    used=max(len(r['generated_ids']) for r in rr);budget=256 if not all(r['eos'] for r in rr) else min(256,max(32,16*math.ceil(used/16)))
    protocol={'max_new_tokens':budget,'calibration_required':True,'calibration_cases':len(rr),'calibration_eos':sum(r['eos'] for r in rr),
              'calibration_final_available':sum(r['final_answer_available'] for r in rr),'max_observed_tokens':used,'generation_cache':True,
              'calibration_sha256':sha(path),'budget_does_not_guarantee_answer':True}
    save(frozen,protocol);return protocol


@torch.inference_mode()
def run_one(key):
    assert read(RESULT/'phase2682_resolved_scalar_paths/analysis/final.json')['all_checks_passed'],'Run only after FP32modelexit and completed2682'
    plan=prepare_all();folder=OUT/key
    if (folder/'analysis/completion.json').exists():return
    folder.joinpath('analysis').mkdir(parents=True,exist_ok=True);folder.joinpath('field').mkdir(exist_ok=True);folder.joinpath('maps').mkdir(exist_ok=True)
    model,tok=load_native(MODEL_KEYS[key]);L=model.config.num_hidden_layers;D=model.config.hidden_size;K=model.config.intermediate_size
    assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    save(folder/'protocol/model.json',{'dtype':str(model.dtype),'nonquantized':True,'device_map':getattr(model,'hf_device_map',None),
         'actual_devices':sorted({str(p.device) for p in model.parameters()}),'layers':L,'hidden':D,'mlp_units':K,'layout':'split' if hasattr(model.model.layers[0].mlp,'gate_proj') else 'merged'})
    protocol=calibrate(model,tok,key,folder);cases=read(folder/'material/cases.json');total=plan['models'][key]['field_execution_length']
    cap=PaddedCapture(model,());path=folder/'analysis/records.jsonl';records=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    assert [r['case_index'] for r in records]==list(range(len(records)))
    begin=len(records)//32*32 if len(records)<512 else 480
    counts={};amp={};mom={};base={};tokens=0;groups=0;t0=time.monotonic()
    try:
        with path.open('a',encoding='utf-8') as f:
            for i in range(begin,512):
                assert shutil.disk_usage(folder).free>8*1024**3
                r=cases[i];ids=r['prompt_ids'];task=len(ids)-1
                cap.reset(r['body_end_token'],r['published'],task);cap.enabled=True
                result=model.model(**padded_inputs(model,ids,tok.eos_token_id,total));cap.enabled=False
                field_state=result.last_hidden_state[0,task].detach().cpu().clone();pack=cap.pack();mm=cap.moment_pack();del result
                field_sha=hashlib.sha256(pack['h'].tobytes()+pack['a'].tobytes()).hexdigest()
                if i<len(records):assert records[i]['native_field_sha256']==field_sha
                if r['published']:np.savez_compressed(folder/f'field/case_{i:04d}.npz',**native_pack(pack,True,False))
                for k,v in mm.items():
                    if k not in mom:mom[k]=np.zeros_like(v)
                    mom[k]+=v
                tokens+=len(ids);base[r['output_function'],r['target_index']]={k:pack[k].copy() for k in ('h','a')}
                if i>=len(records):
                    # Unused allocator cache only; useful for Accelerate's offloaded head.
                    torch.cuda.empty_cache()
                    plain=model.model(input_ids=torch.tensor([ids],device=model.get_input_embeddings().weight.device),use_cache=False).last_hidden_state[0,-1]
                    shape_diff=float((plain.detach().cpu().float()-field_state.float()).abs().max());del plain
                    out=generate(model,tok,r,key,protocol['max_new_tokens'])
                    rec={k:r[k] for k in ('case_index','source_case_index','case_id','family','language','unit','content_instance','form','mention_order','target_index','output_function','polarity','published')}
                    rec.update(**out,native_field_sha256=field_sha,native_body_sha256=hashlib.sha256(pack['h'][:,0].tobytes()+pack['a'][:,0].tobytes()).hexdigest(),unpadded_prefill_state_max_difference=shape_diff)
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush();records.append(rec)
                if (i+1)%8==0:
                    assert len(base)==8
                    for metric in ('h','a'):
                        delta=np.stack([unbits(base[fn,0][metric]).astype(np.float64)-unbits(base[fn,1][metric]).astype(np.float64) for fn in FUNCTIONS])
                        for k,v in sign_counts(delta).items():
                            name=metric+'__'+k
                            if name not in counts:counts[name]=np.zeros_like(v,dtype=np.uint8)
                            counts[name]+=v
                        for k,v in (('min_abs_delta_sum',np.abs(delta).min(0)),('max_abs_delta_sum',np.abs(delta).max(0))):
                            name=metric+'__'+k
                            if name not in amp:amp[name]=np.zeros_like(v)
                            amp[name]+=v
                    groups+=1;base={}
                if (i+1)%32==0:
                    group=r['family']+'_'+r['language'];assert groups==4
                    # Each family chunk is durable; global sums built after all512.
                    np.savez_compressed(folder/f'maps/counts_{group}.npz',**counts)
                    # Amplitudes and moments kept per group temporarily, then exact
                    # global accumulation without deleting them before2684cleanup.
                    # To avoid unpublished raw-map proliferation, store all groups
                    # in one aggregate using completed chunks and a replay-safe index.
                    accumulate_chunk(folder/'maps/global_sums.npz',group,{**amp,**{'moment__'+k:v for k,v in mom.items()}})
                    counts={};amp={};mom={};groups=0
                if (i+1)%8==0:
                    save(folder/'analysis/progress.json',{'cases':i+1,'total':512,'generation_budget':protocol['max_new_tokens'],'elapsed_seconds':time.monotonic()-t0,'free_bytes':shutil.disk_usage(folder).free})
                    print('2683',key,i+1,'/512',flush=True)
                cap.reset(0,False);del pack,mm,field_state
    finally:cap.close()
    del model;gc.collect();torch.cuda.empty_cache();save(folder/'analysis/records.json',records)
    maps=list((folder/'maps').glob('counts_*.npz'));assert len(maps)==16
    with np.load(folder/'maps/global_sums.npz') as z:assert len(z['completed_chunks'])==16
    globalcounts={};summary={}
    for p in maps:
        with np.load(p) as z:
            summary[p.stem]={k:(z[k]==4).sum(-1).tolist() for k in ('h__all4_same_nonzero','a__all4_same_nonzero')}
            for k in z.files:
                if k not in globalcounts:globalcounts[k]=np.zeros_like(z[k],dtype=np.uint16)
                globalcounts[k]+=z[k]
    np.savez_compressed(folder/'maps/global_counts.npz',**globalcounts)
    save(folder/'analysis/raw_manifest.json',[{'path':str((folder/f'field/case_{r["case_index"]:04d}.npz').resolve()),'case_index':r['case_index'],'published':True} for r in cases if r['published']])
    summary={'cases':len(records),'families':summary,'global64_same_nonzero':{k:(v==64).sum(-1).tolist() for k,v in globalcounts.items() if k.endswith('all4_same_nonzero')},
             'behavior':summarize_behavior(records),'final_answer_available':sum(r['final_answer_available'] for r in records),'generation_protocol':protocol,
             'unpadded_prefill_changed':sum(r['unpadded_prefill_state_max_difference']>0 for r in records),'actual_padding_excluded':True}
    save(folder/'analysis/completion.json',summary);print('2683 MODEL COMPLETE',key,flush=True)


def finalize():
    rr={k:read(OUT/k/'analysis/completion.json') for k in KEYS};checks={'all4_protocol_runs_512':all(r['cases']==512 for r in rr.values()),
        'all3_models_nativeBF16':all(read(OUT/k/'protocol/model.json')['nonquantized'] for k in KEYS),'DS_each64calibration_beforeformal':all(rr[k]['generation_protocol']['calibration_cases']==64 for k in ('ds7','ds7_answer')),
        '16_family_maps_each':all(len(r['families'])==16 for r in rr.values())}
    native=read(OUT/'ds7/analysis/records.json');direct=read(OUT/'ds7_answer/analysis/records.json')
    assert [r['source_case_index'] for r in native]==[r['source_case_index'] for r in direct]
    pair={'cases':512,'same_body_fullnative_hash':sum(a['native_body_sha256']==b['native_body_sha256'] for a,b in zip(native,direct)),
          'native_final_available':sum(r['final_answer_available'] for r in native),'direct_final_available':sum(r['final_answer_available'] for r in direct),
          'content_correct_native':sum(r['content_correct'] for r in native),'content_correct_direct':sum(r['content_correct'] for r in direct),
          'meaning':'SameDScheckpoint and bodytokenprefix. Explicit protocolcomparison, notdifferentmodels or independent512newfacts. Bodyhash is numericalcontrol, notsemanticabstraction.'}
    save(OUT/'analysis/DS_protocol_pair.json',pair);checks['DS512_paired_source_indices']=len(native)==len(direct)==512;assert all(checks.values())
    finish(2683,'三模型原生四输出功能全坐标复验与DS生成预算校准',OUT,{'provenance':str(Path(__file__)),'summary':{'protocols':rr,'DS_pair':pair},'checks':checks},
        '严格顺序加载三个本地BF16非量化模型。相同原始语言材料由各自chat与tokenizer编码，固定执行形状观察全H/MLP原坐标；自然cache生成独立分账，DS先用独立实体校准输出预算。',
        r'd_{b,f,l,q,j}=X_{b,f,v0,l,q,j}-X_{b,f,v1,l,q,j};\quad C_{l,q,j}=\sum_b\mathbf1[\min_fd_{b,f,l,q,j}>0\lor\max_fd_{b,f,l,q,j}<0].',
        'C001Qwen14B512；C002GLM4 512；C003DS7B原生推理512和显式闭合空thinking进入答案区512（共3物理模型4协议、2048正式条件）；每协议八族双语两实体对两内容双目标四功能，形式/顺序固定0；C004DS两协议各64独立实体校准最长256生成token后冻结预算；C005全部H与全部MLP神经元四功能同向/正负/零/反向计数；C006全部实际token六字段坐标矩、全坐标幅值范围和2完整H展示例/协议。',
        '跨模型的原坐标条件复用图可区分模型内稳定纹理与个别索引偶然；不能把不同维度中的同编号当成同一个语义单元。生成预算、推理区与最终答案边界都是能力解释的前提。',
        '固定模板与有限实体内容，不是开放语言通则。每族四基础组，符号通过不等于语义特异性。自然cache生成与固定padding场是两种数值执行协议，首状态差异保留，不能直接称数值完全一致。DS原生cloze实际落在thinking内；答案区配对协议显式关闭空thinking，是接口干预而非未经改变的DS推理。两组全保留，不能按较高正确率抹去另一组。无最终答案和预算截断是协议限制。全token矩和幅值为全样本合计而非每族幅值图。',
        '完成2684已有客户端热力图/具体参数查询、独立数值和真实浏览器核验，再精确清理未展示原场；整个大阶段终审之后继续同目标研究。')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare',*KEYS,'finalize']);args=ap.parse_args()
    if args.action=='prepare':prepare_all()
    elif args.action=='finalize':finalize()
    else:run_one(args.action)
