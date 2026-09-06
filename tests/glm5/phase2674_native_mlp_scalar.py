"""Actual gate/up/down scalar changes and joint controls, full-token native gradients."""
import argparse,gc,hashlib,os
import numpy as np
import torch
from transformers import AutoTokenizer
from phase2620_native_coordinate_contract import *
from phase2670_native_mlp_contract import OUT as CONTRACT,FIELD,LAYERS,SITES,row,encoded
from phase2662_symmetric_mapping_contract import load_native
from phase2663_symmetric_mapping_calibration import run,behavior_groups
from phase2666_multitoken_parameter_engine import PARTS,load_fp,values,score
from phase2624_scalar_forward_validation import digest_tensor

OUT=RESULT/'phase2674_native_mlp_scalar'

def prepare():
    if (OUT/'protocol/frozen.json').exists():return read(OUT/'material/cases.json'),read(OUT/'protocol/frozen.json')
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True);cases=[]
    for old in read(CONTRACT/'material/cases.json'):
        if not old['fp_selected']:continue
        r=encoded(tok,{**row(*[old[k] for k in ('family','language','unit','content_instance','form','target_index','mention_order','probe_index','polarity','mapping')],multi=True),'case_index':old['case_index']})
        cats=[]
        for bi,answer in enumerate(r['common_readout_words']):
            a=answer.rindex(r['short_answer_words'][bi]);b=a+len(r['short_answer_words'][bi]);en=tok(answer,add_special_tokens=False,return_offsets_mapping=True)
            assert en['input_ids']==r['canonical_answer_ids'][bi]
            cats.append(['content' if e>a and s<b else 'format' for s,e in en['offset_mapping']]+['eos'])
        r['answer_token_categories']=cats;cases.append(r)
    assert len(cases)==128 and sum(r['published'] for r in cases)==16
    wmeta=read(FIELD/'protocol/native_weights.json');sites=[]
    with np.load(FIELD/'weights/native_candidate_vectors.npz') as ww:
        for s in wmeta['vectors']:
            a=ww[s['key']];rms=float(np.sqrt(np.mean(a.astype('float64')**2)))
            for control in ('ordinary','low'):
                coordinate=s[control+'_coordinate'];j,k=(coordinate,s['unit']) if s['kind']=='down' else (s['unit'],coordinate);original=float(a[coordinate]);targets={}
                for sign in (-1,1):
                    target=float(torch.tensor(original+sign*.1*rms,dtype=torch.bfloat16).float());assert target!=original
                    targets[str(sign)]=target
                sites.append({'layer':s['layer'],'unit':s['unit'],'kind':s['kind'],'j':j,'k':k,'coordinate':coordinate,'control':control,'original':original,'rms':rms,'targets':targets})
    plan={'sites':sites,'cases':128,'published':16,'single_changes':128*30*2,'joint_changes':128*5*2,'noops':128,
        'dose':'+/-0.1 vectorRMS rounded to a BF16-representable target; measured in same-valuedFP32. Joint is ordinary gate/up/down for eachfixedunit. Actual joint minus sumactualsingle is descriptive nonadditivity, not path minimalcut.',
        'numeric':'All input tokens in each canonical branch; total/content/format/EOS separately. Halfdose FP32 controls on allordinarysites in16publishedcases. No-op, postcase fullmatrixhash, finallyrestore every scalar; never write checkpoint weights.',
        'scope':'Numeric microvalidation at128heldoutprefixes; does not equate candidate selection with unique semantic necessity. Joint tests do not establish redundancy or completeness.',
        'source_weight_sha256':wmeta['candidate_vector_sha256']}
    save(OUT/'material/cases.json',cases);save(OUT/'protocol/frozen.json',plan);return cases,plan

def module(model,site):return getattr(model.model.layers[site['layer']].mlp,site['kind']+'_proj')

def native_gradients(model,r,sites):
    branches=[];saved={};info=[]
    for bi,label in enumerate(('Y','N')):
        hooks=[];caps={}
        def capture(m,inp,out,key):caps[key]=(inp[0],out)
        for l in LAYERS:
            for kind in ('gate','up','down'):
                mod=getattr(model.model.layers[l].mlp,kind+'_proj');hooks.append(mod.register_forward_hook(lambda m,a,b,key=(l,kind):capture(m,a,b,key)))
        def pack_saved(t):return (t.device,t.detach().cpu()) if t.is_cuda and t.requires_grad else t.detach()
        def unpack_saved(t):return t[1].to(t[0]) if isinstance(t,tuple) else t
        try:
            with torch.autograd.graph.saved_tensors_hooks(pack_saved,unpack_saved):
                em,states,lp,ids,targets=values(model,r,bi,True);keys=tuple(caps);outputs=[caps[k][1] for k in keys];deriv={};raw={}
                for pi,part in enumerate(('all',)+PARTS):
                    chosen=list(range(len(lp))) if part=='all' else [i for i,c in enumerate(r['answer_token_categories'][bi]) if c==part]
                    grads=torch.autograd.grad(lp[chosen].sum(),outputs,retain_graph=pi<3);gg=dict(zip(keys,grads));terms=[]
                    for s in sites:
                        x=caps[s['layer'],s['kind']][0][0,:,s['k']].detach().double();g=gg[s['layer'],s['kind']][0,:,s['j']].detach().double();terms.append((x*g).cpu().numpy())
                    deriv[part]=np.stack(terms)
                    if r['published']:
                        for l in LAYERS:
                            raw[f'L{l}_down_g_{part}']=gg[l,'down'][0].detach().float().cpu().numpy()
                        for l,j in SITES:
                            for kind in ('gate','up'):raw[f'L{l}_J{j}_{kind}_g_{part}']=gg[l,kind][0,:,j].detach().float().cpu().numpy()
                if r['published']:
                    for l in LAYERS:raw[f'L{l}_x']=caps[l,'gate'][0][0].detach().float().cpu().numpy()
                    for l,j in SITES:raw[f'L{l}_J{j}_a']=caps[l,'down'][0][0,:,j].detach().float().cpu().numpy()
                saved.update({label+'__'+k:v for k,v in raw.items()});saved.update({label+'__scalar_terms_'+p:a for p,a in deriv.items()})
                vv=lp.detach().tolist();info.append({'logprobs':vv,'total':sum(vv),'categories':r['answer_token_categories'][bi],'input_ids':ids,'target_ids':targets})
                branches.append(deriv)
        finally:
            for h in hooks:h.remove()
            caps.clear()
        del lp,em,states,outputs,grads,gg;gc.collect();torch.cuda.empty_cache()
    dd={p:branches[0][p].sum(1)-branches[1][p].sum(1) for p in ('all',)+PARTS}
    component_error=float(np.max(np.abs(dd['all']-sum(dd[p] for p in PARTS))))
    return dd,saved,{'branches':info,'gradient_component_error':component_error}

def natural():
    cases,_=prepare();model,tok=load_native('qwen4');records=run(model,tok,cases,OUT/'natural');del model;gc.collect();torch.cuda.empty_cache();save(OUT/'analysis/natural.json',behavior_groups(records))

def fp():
    os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'
    cases,plan=prepare();model,info=load_fp();save(OUT/'protocol/model.json',info);sites=plan['sites'];matrices={(s['layer'],s['kind']):module(model,s).weight for s in sites}
    before={f'{l}_{k}':digest_tensor(w) for (l,k),w in matrices.items()}
    for s in sites:assert float(module(model,s).weight[s['j'],s['k']])==s['original']
    save(OUT/'protocol/before_hashes.json',before);path=OUT/'analysis/records.jsonl';records=[json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()] if path.exists() else []
    assert [r['case_index'] for r in records]==[r['case_index'] for r in cases[:len(records)]];OUT.joinpath('field').mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as stream:
        for r in cases[len(records):]:
            base=score(model,r);dd,raw,gmeta=native_gradients(model,r,sites);ref=gmeta['branches'][0]['total']-gmeta['branches'][1]['total'];noop=abs(base['contrast']-ref);assert noop<1e-5
            effects=[];singles={}
            def change(indices,sign,scale=1.):
                originals=[]
                try:
                    with torch.no_grad():
                        for si in indices:
                            s=sites[si];w=module(model,s).weight;orig=w[s['j'],s['k']].clone();originals.append((w,s,orig));w[s['j'],s['k']]=s['original']+scale*(s['targets'][str(sign)]-s['original'])
                    actual=[float(w[s['j'],s['k']])-float(orig) for w,s,orig in originals];now=score(model,r)
                    return {'indices':indices,'sign':sign,'dose_scale':scale,'actual_deltas':actual,'effect':now['contrast']-base['contrast'],'predicted':float(sum(d*dd['all'][si] for si,d in zip(indices,actual))),
                        'prompt_last_only':float(sum(d*(raw['Y__scalar_terms_all'][si,len(r['prompt_ids'])-1]-raw['N__scalar_terms_all'][si,len(r['prompt_ids'])-1]) for si,d in zip(indices,actual))),
                        'parts':{p:{'effect':now[p]-base[p],'predicted':float(sum(d*dd[p][si] for si,d in zip(indices,actual)))} for p in PARTS}}
                finally:
                    with torch.no_grad():
                        for w,s,orig in originals:w[s['j'],s['k']].copy_(orig)
            for si,s in enumerate(sites):
                for sign in (-1,1):
                    e=change([si],sign);e['kind']='single';effects.append(e);singles[si,sign]=e
                    if r['published'] and s['control']=='ordinary':
                        e=change([si],sign,.5);e['kind']='halfdose';effects.append(e)
            for l,j in SITES:
                indices=[si for si,s in enumerate(sites) if (s['layer'],s['unit'],s['control'])==(l,j,'ordinary')];assert len(indices)==3
                for sign in (-1,1):
                    e=change(indices,sign);e.update(kind='joint',sum_actual_single_effect=sum(singles[si,sign]['effect'] for si in indices));effects.append(e)
            after={f'{l}_{k}':digest_tensor(w) for (l,k),w in matrices.items()};assert before==after
            rec={'case_index':r['case_index'],'family':r['family'],'language':r['language'],'unit':r['unit'],'target_index':r['target_index'],'mapping':r['mapping'],'published':r['published'],
                'base':base,'noop_error':noop,'gradients':{k:v.tolist() for k,v in dd.items()},'gradient_info':gmeta,'effects':effects,'whole_matrix_hashes':after}
            if r['published']:np.savez_compressed(OUT/f'field/case_{r["case_index"]:04d}.npz',**raw)
            stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush();records.append(rec);del raw
            save(OUT/'analysis/progress.json',{'cases':len(records),'total':128,'single_changes':sum(e['kind']=='single' for z in records for e in z['effects'])});print('native MLP scalar',len(records),'/128',flush=True)
    save(OUT/'analysis/records.json',records);save(OUT/'analysis/restoration.json',{'before':before,'after':after,'all128_postcase_hashes_equal':True,'checkpoint_weights_written':False});del model;gc.collect();torch.cuda.empty_cache()

def finalize():
    records=read(OUT/'analysis/records.json');plan=read(OUT/'protocol/frozen.json');effects=[e for r in records for e in r['effects']];summary={}
    for kind in ('single','joint','halfdose'):
        rr=[e for e in effects if e['kind']==kind];den=sum(abs(e['effect']) for e in rr)
        summary[kind]={'n':len(rr),'mean_absolute_effect':den/len(rr),'relative_L1_error':sum(abs(e['effect']-e['predicted']) for e in rr)/max(den,1e-30),
            'parts':{p:{'mean_abs_effect':sum(abs(e['parts'][p]['effect']) for e in rr)/len(rr),'relative_L1_error':sum(abs(e['parts'][p]['effect']-e['parts'][p]['predicted']) for e in rr)/max(sum(abs(e['parts'][p]['effect']) for e in rr),1e-30)} for p in PARTS}}
    summary['natural']=read(OUT/'analysis/natural.json');summary['max_noop']=max(r['noop_error'] for r in records);summary['max_gradient_part_sum_error']=max(r['gradient_info']['gradient_component_error'] for r in records)
    checks={'128_prefixes':len(records)==128,'7680_actual_single':summary['single']['n']==7680,'1280_joint':summary['joint']['n']==1280,'480_halfdose':summary['halfdose']['n']==480,
        'all_case_fullmatrix_hash_restored':all(r['whole_matrix_hashes']==read(OUT/'protocol/before_hashes.json') for r in records),'noops_below_1e5':summary['max_noop']<1e-5}
    assert all(checks.values())
    finish(2674,'gate/up/down单标量与联合小剂量完整序列数值核验',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '每个真实标量按所有token的输入×下游伴随量求导，完整内容/格式/EOS三账；只在内存实际改变权重，逐标量恢复，逐前缀12整矩阵SHA核对。低值坐标和普通坐标在效果前冻结。',
        r'G_{jk}=\sum_t\bar y_{t,j}x_{t,k};\quad \widehat{\Delta L}=\sum_{s\in S}\Delta\theta_sG_s;\quad I=\Delta L_{joint}-\sum_s\Delta L_s.',
        'C001128heldout多token格式化答案自然BF16；C002同值FP32全部token四类偏导；C00330标量双方向7680改动；C004五单元三支路联合1280；C00516展示例480半剂量；C006128no-op和每例全矩阵恢复校验。',
        '将真实门、上支路和下支路权重作用分开，又观察共同改变的非加性；这能校验参数算法，不依赖搬运。局部导数正确与语言条件纹理稳定是两种证据，不能互相替代。',
        '变化剂量仍有限，极小EOS效应须同时读绝对数值和no-op噪声；联合三参数不是全路径最小割或冗余证明。只验证固定候选窗口，不是全参数因果穷举。规范候选概率不等于自主生成成功。',
        '继续顺序运行Qwen14B、GLM4、DS7B各512条件原生全坐标复验，完成所有族/阴性/正性资料、客户端与存储审计。')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','natural','fp','finalize','all']);args=p.parse_args()
    if args.action=='all':prepare();natural();fp();finalize()
    else:globals()[args.action]()
