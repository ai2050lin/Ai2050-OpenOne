"""Replay the frozen directional writer on each independent task and monitor capabilities."""
import argparse
import time

import numpy as np
import torch
from scipy.stats import binomtest

import mlp_factor_steer_common as C
from mlp_factor_steer_materials import WRITE_ROLES
from mlp_factor_steer_fit import inputs
from mlp_factor_steer_edit import load_vectors, orientation, positions, prefill, write
from mlp_kv_behavior import word_indices
from mlp_kv_mediation import equal_rows, metrics
from mlp_read_route_causal import branch, changed, selected_slots
from mlp_fidelity_v2_tail import capability_interval, cluster_interval
from mlp_functional_trace import bit_array, load_model
from shared.dist_protocol.mechanism_trace import GpuLease


def norm_error(joint,matched,n):
    pre=[];post=[]
    for key,v in joint.items():
        target=v['applied'].clamp_min(1e-30);m=matched[key]
        pre.append(((m['pre_rounding']-v['applied']).abs()/target).reshape(n,4))
        post.append(((m['applied']-v['applied']).abs()/target).reshape(n,4))
    return torch.stack(pre,dim=1).flatten(1).cpu().numpy(),torch.stack(post,dim=1).flatten(1).cpu().numpy()


def raw_slots(cache,j,row):
    layer=cache.layers[35]
    return {role:{kind:bit_array(getattr(layer,kind)[j,:,row['anchors'][role]]).tolist() for kind in ['keys','values']}
            for role in WRITE_ROLES}


def evaluate_batch(model,tok,eos,rows,recipe,vectors,words):
    x=torch.tensor([r['tokens'] for r in rows],device='cuda');pos=positions(rows,x.device);s=orientation(rows,x.device)
    clean=model(input_ids=x[:,:C.CUT],use_cache=True,logits_to_keep=1).past_key_values
    edited,joint_norm=prefill(model,x,pos,s,recipe,vectors,recipe['alpha'])
    delta,restore,mask=changed(clean,edited)
    slots=[{'positions':{role:r['anchors'][role] for role in WRITE_ROLES},'clean':raw_slots(clean,j,r),'edited':raw_slots(edited,j,r)} for j,r in enumerate(rows)]
    relevant=sorted({b for b,_ in mask});source=selected_slots(edited,pos.cpu().tolist(),relevant)
    conditions={'clean':branch(model,x,clean,eos),'targeted':branch(model,x,edited,eos),'replay':branch(model,x,clean,eos,delta),
                'rescue':branch(model,x,edited,eos,restore),'source_only':branch(model,x,clean,eos,source)}
    for name,scale,mode in [('half',.5,'joint'),('matched',1.,'matched'),('head_only',1.,'head_only'),('mlp_only',1.,'mlp_only')]:
        cache,norm=prefill(model,x,pos,s,recipe,vectors,recipe['alpha']*scale,mode,joint_norm if mode=='matched' else None)
        conditions[name]=branch(model,x,cache,eos);del cache
        if name=='matched':pre,post=norm_error(joint_norm,norm,len(rows))
    replay=equal_rows(conditions['targeted'][0],conditions['replay'][0]);rescue=equal_rows(conditions['clean'][0],conditions['rescue'][0]);out=metrics(rows,conditions,tok,words)
    for j,r in enumerate(out):
        r.update(replay_logits_bitwise_equal=replay[j],rescue_logits_bitwise_equal=rescue[j],
                 replay_generation_equal=conditions['targeted'][1][j]==conditions['replay'][1][j],
                 rescue_generation_equal=conditions['clean'][1][j]==conditions['rescue'][1][j],
                 changed_slots={f'{b}:{k}':v[j].nonzero().flatten().cpu().tolist() for (b,k),v in mask.items()},
                 norm={'max_pre':float(pre[j].max()),'median_post':float(np.median(post[j])),'max_post':float(post[j].max()),'count':len(pre[j])})
    return out,slots


def summary(rows,eligible):
    n=len(rows)
    if not n:return {'n':0,'eligible':False,'passed':False,'operations':{}}
    baseline=np.asarray([r['conditions']['clean']['word']==r['contrast'] for r in rows]);ops={}
    for k in rows[0]['conditions']:
        values=[r['conditions'][k] for r in rows];alt=np.asarray([v['word']==r['contrast'] for v,r in zip(values,rows)])
        gain=int((alt&~baseline).sum());loss=int((~alt&baseline).sum())
        ops[k]={'accuracy':float(np.mean([v['correct'] for v in values])),'targeted_alt':int(alt.sum()),
                'targeted_gain':float(alt.mean()-baseline.mean()),'targeted_mcnemar_p':float(binomtest(gain,gain+loss).pvalue) if gain+loss else 1.,
                'mean_KL':float(np.mean([v['KL'] for v in values])),
                'target_gap_gain':cluster_interval([r['conditions']['clean']['gap']-v['gap'] for r,v in zip(rows,values)],C.SEED)}
    adv=ops['targeted']['targeted_gain']-ops['matched']['targeted_gain']
    return {'n':n,'eligible':eligible,'passed':bool(eligible and ops['targeted']['targeted_gain']>=.10 and ops['targeted']['targeted_mcnemar_p']<.025 and adv>=.05),
            'specificity':adv,'median_clean_gap':float(np.median([r['conditions']['clean']['gap'] for r in rows])),'operations':ops}


def holdout(model,tok,eos,task,recipe,vectors):
    corpus,prereg=inputs();pool=[r for r in corpus['heldout'] if r['task']==task];screen=C.read_gz(C.OUT/'screen.json.gz')
    assert screen['recipe_sha256']==C.file_sha(C.OUT/'frozen-recipe.json')
    idx=screen['enrolled'][task];words=word_indices(tok);rows=[];raw=[]
    for lo in range(0,len(idx),4):
        real=idx[lo:lo+4];pad=[i for i in range(4) if i not in real][:4-len(real)]
        batch=[pool[i] for i in real+pad];report,slots=evaluate_batch(model,tok,eos,batch,recipe,vectors,words)
        rows.extend(report[:len(real)]);raw.extend(slots[:len(real)])
        if (lo+len(real))%16==0:print(task,'mediation',lo+len(real),'/',len(idx),flush=True)
    diagnostics=[]
    def match(i,r):
        a=screen['rows'][task][i]['clean'];b=r['conditions']['clean'];return (a['word'],a['gap'])==(b['word'],b['gap'])
    for j,(i,r) in enumerate(zip(idx,rows)):
        if match(i,r):continue
        start=i//4*4;batch=pool[start:start+4];report,slots=evaluate_batch(model,tok,eos,batch,recipe,vectors,words)
        diagnostics.append({'index':j,'screen_index':i,'initial':r,'corrected':report[i-start],'batch_ids':[r['id'] for r in batch]})
        rows[j],raw[j]=report[i-start],slots[i-start]
    for j,r in enumerate(rows):r.update(index=j,screen_index=idx[j]);raw[j].update(id=r['id'],index=j)
    checks={k:sum(r[k] for r in rows) for k in ['replay_logits_bitwise_equal','rescue_logits_bitwise_equal','replay_generation_equal','rescue_generation_equal']}
    n=len(rows);clean_exact=sum(match(i,r) for i,r in zip(idx,rows));before=max((r['norm']['max_pre'] for r in rows),default=0.)
    checks.update(clean_exact=clean_exact,layout_corrections=len(diagnostics),max_pre_rounding_norm_error=before,
        max_post_rounding_norm_error=max((r['norm']['max_post'] for r in rows),default=0.),
        norm_vectors=sum(r['norm']['count'] for r in rows),
        no_cache_changes_before_origin=all(all(p>=min(raw[j]['positions'].values()) for ps in r['changed_slots'].values() for p in ps) for j,r in enumerate(rows)))
    ok=bool(screen['summary'][task]['power']['pass'] and clean_exact==n and before<1e-5 and
            checks['replay_logits_bitwise_equal']==n and checks['rescue_logits_bitwise_equal']==n)
    path=C.OUT/f'{task}-native.json.gz';C.write_gz(path,{'rows':raw,'dtype':'BF16 uint16','layer':35})
    return {'rows':rows,'summary':summary(rows,ok),'checks':checks,'layout_diagnostics':diagnostics,'native_sha256':C.file_sha(path)}


def capability(model,tok,frozen,vectors):
    data=C.read(C.CAP);assert sum(r['gate_eligible'] for r in data)==2200
    rcp=frozen['recipe'];letters=[tok.encode(c,add_special_tokens=False)[0] for c in 'ABCDE'];rows=[];norm_error=0.
    for lo in range(0,len(data),12):
        batch=data[lo:lo+12];prompts=[]
        for r in batch:
            text=r['question']+'\n'+'\n'.join(f'{a}. {v}' for a,v in zip('ABCDE',r['choices']))+'\n请只回答正确选项字母。'
            prompts.append(tok.encode(tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False))
        inputs={k:v.to('cuda') for k,v in tok.pad({'input_ids':prompts},padding=True,return_tensors='pt').items()};width=inputs['input_ids'].shape[1]
        p=torch.tensor([[width-1-frozen['offsets'][role] if frozen['offsets'][role]<len(q) else -1 for role in WRITE_ROLES] for q in prompts],device='cuda')
        assert len(set(frozen['offsets'].values()))==4
        sign=torch.tensor([1. if int(C.digest([C.SEED,'monitor',lo+j])[:8],16)%2 else -1. for j in range(len(batch))],device='cuda')
        clean=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1]
        with write(model,p,sign,rcp,vectors,rcp['alpha']) as joint:edited=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1]
        with write(model,p,sign,rcp,vectors,rcp['alpha'],'matched',joint) as control:matched=model(**inputs,use_cache=False,logits_to_keep=1).logits[:,-1]
        for key,z in joint.items():
            if len(z['applied']):norm_error=max(norm_error,float(((control[key]['pre_rounding']-z['applied']).abs()/z['applied'].clamp_min(1e-30)).max()))
        for j,r in enumerate(batch):
            predictions={}
            for k,logits in [('clean',clean),('targeted',edited),('matched',matched)]:
                pred='ABCDE'[int(logits[j,letters[:len(r['choices'])]].argmax())];predictions[k]={'predicted':pred,'correct':pred==r['answer']}
            rows.append({'index':lo+j,'id':r.get('id'),'source':r.get('source'),'expected':r['answer'],'gate_eligible':r['gate_eligible'],
                         'actual_offsets':[width-1-v for v in p[j].tolist() if v>=0],'conditions':predictions})
        if (lo+len(batch))%240==0:print('capability',lo+len(batch),'/',len(data),flush=True)
    mask=np.asarray([r['gate_eligible'] for r in rows]);baseline=[r['conditions']['clean']['correct'] for r in rows]
    return {'rows':rows,'summary':{k:capability_interval(baseline,[r['conditions'][k]['correct'] for r in rows],mask) for k in ['targeted','matched']},
            'reference_only':int((~mask).sum()),'max_pre_rounding_norm_error':norm_error,'input_sha256':C.file_sha(C.CAP)}


def main(part):
    out=C.OUT/f'{part}-result.json.gz'
    if out.exists():return
    frozen=C.read(C.OUT/'frozen-recipe.json');torch.set_num_threads(4);t=time.time()
    plan={**C.identity(),'schema':'factor_steer_holdout_execution.v1','part':part,'source_sha256':C.file_sha(__file__),
          'recipe_sha256':C.file_sha(C.OUT/'frozen-recipe.json'),'screen_sha256':C.file_sha(C.OUT/'screen.json.gz')}
    C.save(C.OUT/f'{part}-execution.json',plan)
    with GpuLease(),torch.inference_mode():
        model,tok,eos=load_model();vectors=load_vectors('cuda')
        result=capability(model,tok,frozen,vectors) if part=='capability' else holdout(model,tok,eos,part,frozen['recipe'],vectors)
    C.write_gz(out,{**plan,'status':'complete','seconds':time.time()-t,**result})
    print(part,'n',len(result['rows']),'checks',result.get('checks'),'seconds',round(time.time()-t,1),'summary',result['summary'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['lookup','dispatch','capability']);main(p.parse_args().part)
