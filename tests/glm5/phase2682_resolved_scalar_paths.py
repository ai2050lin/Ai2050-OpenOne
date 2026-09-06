"""Native scalar interventions: exact local algorithms, separately observed outputs.

No donor states and no compressed coordinate representation. FP32 core is a
numerical control on the BF16 checkpoint, not its original execution precision.
"""
import gc, hashlib, os, shutil, time
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2666_multitoken_parameter_engine import load_fp
from phase2670_native_mlp_contract import LAYERS,SITES
from phase2677_padded_native_runtime import PAD_LENGTH

OUT=RESULT/'phase2682_resolved_scalar_paths'
SOURCE=RESULT/'phase2681_fresh_source_confirmation'
WEIGHTS=RESULT/'phase2678_padded_source_field'
SCALES=(.025,.1)


def arr(t):return t.detach().float().cpu().numpy().copy()


def prepare():
    path=OUT/'material/cases.json'
    if path.exists():
        plan=read(OUT/'protocol/frozen.json');assert sha(path)==plan['material_sha256'];return read(path),plan
    assert read(SOURCE/'analysis/final.json')['all_checks_passed']
    records={r['case_index']:r for r in read(SOURCE/'analysis/records.json')};cases=[]
    for r in read(SOURCE/'material/cases.json'):
        if (r['unit'],r['content_instance'],r['form'],r['mention_order'])!=(2,0,0,0):continue
        b=records[r['case_index']];row={**r,'observed_ids':b['generated_ids'],'observed_text':b['generated'],
            'observed_eos':b['eos'],'bf16_content_correct':b['content_correct'],'published_numeric':r['output_function']=='truth' and r['target_index']==0}
        assert 0<len(row['observed_ids'])<=16 and len(row['prompt_ids'])+len(row['observed_ids'])-1<=PAD_LENGTH
        cases.append(row)
    assert len(cases)==128 and sum(r['published_numeric'] for r in cases)==16
    controls=[]
    for v in read(WEIGHTS/'protocol/native_weights.json')['vectors']:
        for kind in ('ordinary','low'):
            controls.append({**v,'control':kind,'coordinate':v[kind+'_coordinate'],'original_weight':v[kind+'_value']})
    budget={'free':shutil.disk_usage(OUT.parent).free,'maps_upper':128*120*3*2560*8,'published_upper':16*(4*160*2560*4*2+37*2*2560*4+160*2560*4+5*160*4*3),
            'records_reserve':160*1024**2,'floor':8*1024**3}
    budget['fits_uncompressed']=budget['free']>=sum(budget[k] for k in ('maps_upper','published_upper','records_reserve','floor'));assert budget['fits_uncompressed'],budget
    save(path,cases)
    plan={'material_sha256':sha(path),'source_records_sha256':sha(SOURCE/'analysis/records.json'),'source_material_sha256':sha(SOURCE/'material/cases.json'),
          'controls':controls,'scales':SCALES,'prefixes':128,'conditions':15360,'budget':budget,
          'selection':'8 families x2 languages x4 functions x2 targets; fresh entitypair2/content0/form0/order0. Numerical validation, not an additional independent population.',
          'score':'All actual BF16 generated token IDs under FP32 teacher forcing, including EOS only if emitted. Same160shape. NonEOS/EOS separated, not inferred content/format labels. NOT FP32 autonomous generation.',
          'readout64':'16 prospectively published truth/target0 prefixes: whole vocabulary FP64 readout for all120scalarconditions; FP32 hidden states unchanged. Other112 use whole-vocabulary FP32 readout.',
          'arithmetic':'Analytic local SiLU/gate/up/down finite change compared to actual same-layer output all tokens/all coordinates; later output probability observed, not predicted semantic closure.',
          'restoration':'Changed scalar restored in finally; all12candidate MLP matrices SHA before/after each prefix; no checkpoint writes.',
          'storage':'Full coordinates processed. 128perprefix maps: each120condition signed actual/prediction/absolute-error sums over allreal tokens; two target orientations retained separately. Published16 raw E/H-boundaries/x/down plus selected-unit g/u/a. No unpublished fulltoken fields written.'}
    save(OUT/'protocol/frozen.json',plan);return cases,plan


class LocalCapture:
    def __init__(self,model):
        self.handles=[];self.active=();self.n=0;self.data={};self.h={};self.hidden=False;self.positions=[]
        def take(l,key,t):
            if l in self.active:self.data[l,key]=arr(t[0,:self.n])
        self.handles.append(model.get_input_embeddings().register_forward_hook(lambda m,a,b:self.htake(0,b)))
        for l,b in enumerate(model.model.layers):
            self.handles.append(b.register_forward_hook(lambda m,a,o,l=l:self.htake(l+1,o[0] if isinstance(o,tuple) else o)))
            if l not in LAYERS:continue
            self.handles.append(b.mlp.register_forward_pre_hook(lambda m,a,l=l:take(l,'x',a[0])))
            for key in ('gate','up','down'):
                self.handles.append(getattr(b.mlp,key+'_proj').register_forward_hook(lambda m,a,o,l=l,key=key:take(l,key,o)))
            self.handles.append(b.mlp.down_proj.register_forward_pre_hook(lambda m,a,l=l:take(l,'a',a[0])))
    def htake(self,l,t):
        if self.hidden:
            self.h[l]=arr(t[0,self.positions])
            if l==0:self.embedding=arr(t[0,:self.n])
    def reset(self,layers,n,positions=(),hidden=False):
        self.active=tuple(layers);self.n=n;self.positions=list(positions);self.hidden=hidden;self.data={};self.h={}
    def close(self):
        for h in self.handles:h.remove()


def logprobs(model,states,targets,dtype=torch.float32):
    # Head-row blocking is an exact storage strategy; every vocabulary item is used.
    x=states.detach().cpu().to(dtype);w=model.lm_head.weight
    if dtype==torch.float32:logits=x@w.T
    else:
        logits=torch.empty((len(targets),w.shape[0]),dtype=dtype)
        for start in range(0,w.shape[0],8192):logits[:,start:start+8192]=x@w[start:start+8192].to(dtype).T
    return logits.log_softmax(-1)[torch.arange(len(targets)),torch.tensor(targets)].double().tolist()


@torch.inference_mode()
def forward(model,row,cap,layers,readout64=False,hidden=False):
    targets=row['observed_ids'];ids=row['prompt_ids']+targets[:-1];n=len(ids);start=len(row['prompt_ids'])-1
    cap.reset(layers,n,(row['body_end_token'],start),hidden)
    pad=model.config.eos_token_id;pad=pad[0] if isinstance(pad,list) else pad
    padded=ids+[pad]*(PAD_LENGTH-n)
    em=model.get_input_embeddings()(torch.tensor([padded],device='cpu')).to(next(model.model.layers[0].parameters()).device)
    mask=torch.tensor([[1]*n+[0]*(PAD_LENGTH-n)],device=em.device)
    result=model.model(inputs_embeds=em,attention_mask=mask,use_cache=False)
    states=result.last_hidden_state[0,start:start+len(targets)]
    lp=logprobs(model,states,targets);lp64=logprobs(model,states,targets,torch.float64) if readout64 else None
    return {'logprobs':lp,'logprobs64':lp64,'state':arr(states),'data':cap.data,'h':cap.h,'embedding':cap.embedding.copy() if hidden else None}


def digest_matrices(model):
    result={}
    for l in LAYERS:
        for k in ('gate','up','down'):
            w=getattr(model.model.layers[l].mlp,k+'_proj').weight
            result[f'L{l}_{k}']=hashlib.sha256(arr(w).tobytes()).hexdigest()
    return result


def silu(x):return x/(1+np.exp(-x))


def local_prediction(base,actual,site,delta,down):
    l,j,k,kind=site['layer'],site['unit'],site['coordinate'],site['kind'];b={key:base[l,key].astype(np.float64) for key in ('x','gate','up','a','down')}
    a={key:actual[l,key].astype(np.float64) for key in b}
    assert np.array_equal(b['x'],a['x']),'Same-layer MLP input unexpectedly changed'
    other=np.arange(b['a'].shape[-1])!=j
    assert np.array_equal(b['a'][:,other],a['a'][:,other]),'Unchanged MLP units differ at intervention layer'
    if kind=='gate':
        dp=delta*b['x'][:,k];da=(silu(b['gate'][:,j]+dp)-silu(b['gate'][:,j]))*b['up'][:,j]
    elif kind=='up':
        dp=delta*b['x'][:,k];da=silu(b['gate'][:,j])*dp
    else:dp=delta*b['a'][:,j];da=np.zeros(len(dp));assert np.array_equal(b['a'],a['a'])
    predicted=da[:,None]*down[None,:]
    if kind=='down':predicted[:,k]=dp
    measured=a['down']-b['down'];err=measured-predicted
    branch_error=(a[kind][:,j]-b[kind][:,j]-dp) if kind!='down' else measured[:,k]-dp
    denom=float(np.abs(measured).sum())
    return {'actual':measured,'predicted':predicted,'error':err,'predicted_a':da,'actual_a':a['a'][:,j]-b['a'][:,j],
            'summary':{'local_actual_l1':denom,'local_prediction_l1':float(np.abs(predicted).sum()),'local_error_l1':float(np.abs(err).sum()),
                       'local_relative_l1':float(np.abs(err).sum())/denom if denom else None,'local_max_abs_error':float(np.abs(err).max()),
                       'branch_max_abs_error':float(np.abs(branch_error).max()),'a_max_abs_error':float(np.abs(a['a'][:,j]-b['a'][:,j]-da).max()),
                       'all_input_coords_unchanged':True,'all_other_units_unchanged':True,'actual_tokens':len(dp)}}


def pack_baseline(base):
    p={'embedding':base['embedding'],'h':np.stack([base['h'][l] for l in sorted(base['h'])]),'normalized_output_states':base['state']}
    for l in LAYERS:
        for key in ('x','down'):p[f'L{l}_{key}']=base['data'][l,key]
    for l,j in SITES:
        for key in ('gate','up','a'):p[f'L{l}_J{j}_{key}']=base['data'][l,key][:,j]
    return p


@torch.inference_mode()
def run(model,cases,plan):
    for name in ('analysis','field','maps'):OUT.joinpath(name).mkdir(parents=True,exist_ok=True)
    done=OUT/'analysis/records.jsonl';records=[json.loads(s) for s in done.read_text(encoding='utf-8').splitlines()] if done.exists() else []
    assert [r['case_index'] for r in records]==[r['case_index'] for r in cases[:len(records)]]
    with np.load(WEIGHTS/'weights/native_candidate_vectors.npz') as z:wv={k:z[k].astype(np.float64) for k in z.files}
    for v in plan['controls']:
        w=getattr(model.model.layers[v['layer']].mlp,v['kind']+'_proj').weight;j=v['unit']
        assert np.array_equal(arr(w[:,j] if v['kind']=='down' else w[j]),wv[v['key']])
    before=digest_matrices(model);save(OUT/'protocol/initial_weight_hashes.json',before);cap=LocalCapture(model);t0=time.monotonic()
    try:
        with done.open('a',encoding='utf-8') as stream:
            for index,row in enumerate(cases[len(records):],len(records)):
                assert shutil.disk_usage(OUT).free>8*1024**3
                published=row['published_numeric'];base=forward(model,row,cap,LAYERS,published,published)
                basehash=hashlib.sha256(base['state'].tobytes()).hexdigest()
                noop=forward(model,row,cap,(),published)
                assert np.array_equal(base['state'],noop['state']) and base['logprobs']==noop['logprobs'] and base['logprobs64']==noop['logprobs64'];del noop
                dense=np.zeros((120,3,base['state'].shape[-1]),dtype=np.float64);conditions=[];token_fields={}
                for si,site in enumerate(plan['controls']):
                    l,j,k=site['layer'],site['unit'],site['coordinate'];kind=site['kind'];w=getattr(model.model.layers[l].mlp,kind+'_proj').weight
                    ij=(k,j) if kind=='down' else (j,k);original=w[ij].clone();value=float(original);assert value==site['original_weight']
                    rms=float(np.sqrt(np.mean(wv[site['key']]**2)));down=wv[f'L{l}_J{j}_down']
                    try:
                        for scale in plan['scales']:
                            for sign in (-1,1):
                                w[ij]=value+sign*scale*rms;delta=float(w[ij])-value;assert delta!=0
                                changed=forward(model,row,cap,(l,),published)
                                p=local_prediction(base['data'],changed['data'],site,delta,down);ci=len(conditions)
                                dense[ci]=np.stack((p['actual'].sum(0),p['predicted'].sum(0),np.abs(p['error']).sum(0)))
                                diff=np.array(changed['logprobs'])-base['logprobs'];eos=model.config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
                                ed=[t in eos for t in row['observed_ids']]
                                record={'control_index':si,'layer':l,'unit':j,'kind':kind,'coordinate':k,'control':site['control'],'scale':scale,'sign':sign,
                                        'original_weight':value,'actual_delta':delta,**p['summary'],'output_logprob_delta':diff.tolist(),'output_full_delta':float(diff.sum()),
                                        'output_nonEOS_delta':float(diff[np.logical_not(ed)].sum()),'output_EOS_delta':float(diff[ed].sum())}
                                if published:
                                    diff64=np.array(changed['logprobs64'])-base['logprobs64'];record.update(output_logprob64_delta=diff64.tolist(),output_full64_delta=float(diff64.sum()),
                                        output_readout_difference=float(diff.sum()-diff64.sum()))
                                    token_fields[f'C{ci:03d}_predicted_a']=p['predicted_a'];token_fields[f'C{ci:03d}_actual_a']=p['actual_a']
                                conditions.append(record);del changed,p
                    finally:w[ij].copy_(original)
                restored=forward(model,row,cap,(),published)
                assert np.array_equal(restored['state'],base['state']) and restored['logprobs']==base['logprobs'] and digest_matrices(model)==before
                # Per-prefix maps are durable, allow independent summing/contrasts without
                # losing target orientation or rewriting a partially accumulated group.
                np.savez_compressed(OUT/f'maps/case_{row["case_index"]:04d}.npz',local_coordinate_sums=dense)
                if published:np.savez_compressed(OUT/f'field/case_{row["case_index"]:04d}.npz',**pack_baseline(base),**token_fields)
                rec={k:row[k] for k in ('case_index','case_id','family','language','output_function','target_index','published_numeric','observed_ids','observed_text','observed_eos','bf16_content_correct')}
                rec.update(baseline_logprobs=base['logprobs'],baseline_logprobs64=base['logprobs64'],baseline_state_sha256=basehash,conditions=conditions,
                           noop_exact=True,all12_matrices_restored=True,actual_input_tokens=len(row['prompt_ids'])+len(row['observed_ids'])-1)
                stream.write(json.dumps(rec,ensure_ascii=False,allow_nan=False)+'\n');stream.flush();records.append(rec)
                save(OUT/'analysis/progress.json',{'prefixes':index+1,'total':len(cases),'conditions':sum(len(r['conditions']) for r in records),'elapsed_seconds':time.monotonic()-t0,'free_bytes':shutil.disk_usage(OUT).free})
                print('2682 NATIVE SCALAR',index+1,'/128',flush=True);del base,restored,dense,token_fields;cap.reset((),0);gc.collect()
    finally:cap.close()
    save(OUT/'analysis/records.json',records);save(OUT/'analysis/restoration.json',{'before':before,'after':digest_matrices(model),'checkpoint_writes':False})
    save(OUT/'analysis/raw_manifest.json',[{'path':str((OUT/f'field/case_{r["case_index"]:04d}.npz').resolve()),'case_index':r['case_index'],'published':True} for r in cases if r['published_numeric']])
    return records


def finalize(records):
    conditions=[c for r in records for c in r['conditions']];groups={}
    for kind in ('gate','up','down'):
        for scale in SCALES:
            rr=[c for c in conditions if c['kind']==kind and c['scale']==scale];den=sum(c['local_actual_l1'] for c in rr);err=sum(c['local_error_l1'] for c in rr)
            groups[f'{kind}/{scale}']={'conditions':len(rr),'local_actual_l1':den,'local_error_l1':err,'relative_l1':err/den if den else None,
                'max_abs_error':max(c['local_max_abs_error'] for c in rr),'max_branch_error':max(c['branch_max_abs_error'] for c in rr),
                'output_absolute_effect_sum':sum(abs(c['output_full_delta']) for c in rr),'zero_output_effects':sum(c['output_full_delta']==0 for c in rr)}
    checks={'128_actual_prefixes':len(records)==128,'15360_actual_single_scalar_conditions':len(conditions)==15360,'16_raw_published':len(read(OUT/'analysis/raw_manifest.json'))==16,
            'all_noops_exact':all(r['noop_exact'] for r in records),'all12_matrices_restored_each_prefix':all(r['all12_matrices_restored'] for r in records),
            'all_real_input_other_units_unchanged':all(c['all_input_coords_unchanged'] and c['all_other_units_unchanged'] for c in conditions),
            'material_immutable':sha(OUT/'material/cases.json')==read(OUT/'protocol/frozen.json')['material_sha256']}
    assert all(checks.values())
    finish(2682,'真实单标量到全token神经元与MLP输出的可分辨数值验证',OUT,{'provenance':str(Path(__file__)),'summary':{'prefixes':len(records),'conditions':len(conditions),'groups':groups,
        'full_output_prediction_claim':False,'FP64_core':False,'FP64_readout_conditions':sum('output_full64_delta' in c for c in conditions)},'checks':checks},
        '冻结原生权重中的30个普通/低幅值标量，直接临时改变真实gate/up/down权重，所有其余权重不动。FP32核上逐实际token验证局部有限改变量；所有下游计算真实执行，完整已生成token串的概率另行观测。',
        r'\Delta g_{t,j}=\Delta\theta x_{t,k};\quad\Delta a_{t,j}=[\operatorname{SiLU}(g_{t,j}+\Delta g_{t,j})-\operatorname{SiLU}(g_{t,j})]u_{t,j};\quad\Delta m_{t,:}=W_{d,:,j}\Delta a_{t,j};\quad L=\sum_{i\in\mathrm{observed}}\log p(y_i\mid x,y_{<i}).',
        'C001128新前缀：八族双语四功能双目标；C002五神经元各gate/up/down两个普通/低值标量，共30×双剂量0.025/0.1向量RMS×双方向=15360干预；C003所有实际token全输入/输出坐标局部预测与舍入误差；C004128无操作和12矩阵逐前缀恢复哈希；C005全部真实BF16输出ID在FP32模型强制前缀计分，含实际EOS而不补造EOS；C00616预定例全部条件再做全词表FP64读出，保留16原场与128完整输出坐标汇总。',
        '局部投影、非线性与输出投影可以逐参数明确寻址，数值可解释范围与最终概率效应必须分账。这是原生计算算法校验，不是独立语义神经元证明。',
        'FP32执行不同于原始BF16，FP64仅输出头重算；真实生成串在这里固定，不等于FP32自主生成。五候选神经元只代表冻结观察窗，完整背景图仍由2680/2681覆盖。剂量非无穷小，局部已知SiLU公式不解释语言功能起源；本Phase未预测完整网络输出概率变化。',
        '继续2683按顺序完成Qwen14B/GLM4/DS7B全坐标条件复用复测，校准DS生成协议，再完成2684实际客户端交付与安全清理；不得把本阶段数值恒等式命名为机制闭合。')


def main():
    os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
    assert not (OUT/'analysis/final.json').exists();cases,plan=prepare();model,info=load_fp();save(OUT/'protocol/model.json',info)
    records=run(model,cases,plan);del model;gc.collect();torch.cuda.empty_cache();finalize(records)


if __name__=='__main__':main()
