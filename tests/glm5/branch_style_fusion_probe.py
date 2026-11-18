"""Fresh, preregistered 2x2 mode probe; preserves original unused lockbox.

Conditional diagnostics use observed neutral/chat/debate states to predict joint
states. They do not constitute independent input-to-output mechanism extraction.
"""
import argparse
import itertools
import json
from pathlib import Path
import random
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import branch_style_joint as C
from shared.dist_protocol.mechanism_trace import file_sha,unbits
from scipy.special import logsumexp,expit

OUT=C.OUT/'fresh_fusion_probe'
EXTRA=[('床','bed'),('沙发','sofa'),('柜子','cabinet'),('梯子','ladder'),('地毯','carpet'),('窗帘','curtain'),('吉他','guitar'),('钢琴','piano'),('滑板','skateboard'),('足球','football'),('头盔','helmet'),('手提箱','suitcase')]
MODES=[(0,0),(1,0),(0,1),(1,1)]

def prepare():
    from transformers import AutoTokenizer
    from scipy.stats import norm
    assert not (OUT/'prereg.json').exists(),'probe already frozen'
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True);frozen=C.read(C.OUT/'frozen.json')
    names=C.NAMES+EXTRA
    pairlist=[(a,b) for a in range(len(names)) for b in range(a+1,len(names)) if b>=len(C.NAMES)]
    random.Random(C.SEED+440).shuffle(pairlist)
    rows=[];length_differences=[]
    for i,(a,b) in enumerate(pairlist[:128]):
        rng=random.Random(C.SEED+440+i);ci,di=rng.sample(range(8),2)
        if i%2:a,b=b,a
        f={'X':names[a][0],'Y':names[b][0],'EN':names[a][1],'C':C.COLORS[ci][0],'D':C.COLORS[di][0],'EC':C.COLORS[ci][1],'ED':C.COLORS[di][1],'TAG':f'J{i:03}'}
        bodies=[f'记录{f["TAG"]}：{f["X"]}是{f["C"]}的，{f["Y"]}是{f["D"]}的。',f'记录{f["TAG"]}：{f["X"]}呈现{f["C"]}，{f["Y"]}呈现{f["D"]}。',f'记录{f["TAG"]}：甲号对应{f["X"]}，乙号对应{f["Y"]}；甲号的颜色为{f["C"]}，乙号的颜色为{f["D"]}。']
        for branch,task in itertools.product(range(3),range(2)):
            lengths=[]
            for mode,(chat,debate) in enumerate(MODES):
                instruction=f'表达开关：自然聊天={"开" if chat else "关"}；严谨论辩={"开" if debate else "关"}。两项均关时中性表达，两项均开时同时兼顾。只依据记录，不增添事实，不改变立场。'
                user=instruction+C.TASK[task]+'\n'+bodies[branch]+f'\n被询问对象：{f["X"]}。\n回应：'
                text=tok.apply_chat_template([{'role':'user','content':user}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                enc=tok(text,add_special_tokens=False,return_offsets_mapping=True)
                def pos(c):return next(j for j,(aa,bb) in enumerate(enc.offset_mapping) if aa<=c<bb)
                candidates=[]
                for color,ec in [(f['C'],f['EC']),(f['D'],f['ED'])]:
                    candidates.extend([f'嗯，{f["X"]}是{color}的。',f'依据资料，{f["X"]}的颜色是{color}。'] if task==0 else [f'Well, the {f["EN"]} is {ec}.',f'The {f["EN"]} is {ec}.'])
                rows.append({'id':f'fresh-{i:03}-b{branch}-t{task}-m{mode}','family':f'fresh-{i:03}','branch':branch,'task':task,'mode':mode,'slots':f,'text':user,'tokens':enc.input_ids,'readouts':[pos(text.index(bodies[branch])),pos(text.index('\n被询问对象')-1),len(enc.input_ids)-1],'candidates':candidates,'candidate_tokens':[tok.encode(v,add_special_tokens=False) for v in candidates]})
                lengths.append(len(enc.input_ids))
            length_differences.append(max(lengths)-min(lengths))
    assert max(length_differences)==0,'2x2 token length/positions must match before any GPU run'
    groups=frozen['group']+frozen['random_groups'][0]+frozen['random_groups'][1]
    C.save(OUT/'inputs.json',{'rows':rows,'source':'New fictional object pairs involving twelve additional nouns. No previous main or lockbox pair reused.','modes':MODES})
    C.save(OUT/'prereg.json',{'schema':'branch_style_fresh_2x2.v1','kind':'real_design','model_snapshot':frozen['model_snapshot'],'frozen_group_sha256':file_sha(C.OUT/'frozen.json'),'inputs_sha256':file_sha(OUT/'inputs.json'),'source_sha256':file_sha(__file__),'rows':len(rows),'families':128,'groups':groups,'equal_mode_token_lengths':True,'capture':'Full-coordinate hs all layers at three positions, G/U/A only 96 previously frozen selected/random units at two positions, full vocabulary logits. Other MLP units not retained in this focused probe.','primary':'Pooled family-mean KL(joint || logit_add) vs neutral/chat/debate/arithmetic mixture; all predeclared methods reported. Native A additive vs gate/up-product predictions. 97.5% family bootstrap; no choosing a winning model then calling it preregistered.','behavior_gate':{'content_pooled':.85,'min_cell':.75,'relative_style_shift_fraction_per_task':.65},'power':C.check_phase(.10,128,p_d=.10,alpha=.025),'continuous_power':float(norm.cdf(.30*np.sqrt(128)-norm.ppf(1-.025/2))),'information_policy':'Only observed N,C,D inputs for predicting F; no F state/logits used in prediction. New exploratory follow-up selected after main results, before any new-cohort responses. No training-recursive claim.','budget':{'additional_GiB_cap':5,'GPU_hours_cap':.5}})
    print({'prepared':len(rows),'families':128,'equal_mode_lengths':True},flush=True)

def collect():
    import torch
    plan=C.read(OUT/'prereg.json');assert file_sha(__file__)==plan['source_sha256']
    rows=C.read(OUT/'inputs.json')['rows'];group=plan['groups']
    if (OUT/'summary.json').exists():return
    shapes={'hs':(len(rows),37,3,2560),'logits':(len(rows),151936),'G':(len(rows),2,len(group)),'U':(len(rows),2,len(group)),'A':(len(rows),2,len(group))}
    state=OUT/'state.json';done=C.read(state)['done'] if state.exists() else 0
    ar={k:np.lib.format.open_memmap(OUT/f'{k}.npy',mode='r+' if done else 'w+',dtype=np.uint16,shape=s) for k,s in shapes.items()}
    scores=np.lib.format.open_memmap(OUT/'scores.npy',mode='r+' if done else 'w+',dtype=np.float32,shape=(len(rows),4))
    start=time.time();C.log('fresh_2x2',done,len(rows),start)
    with C.gpu_lease(),torch.inference_mode():
        engine=C.Engine(True)
        for off in range(done,len(rows),4):
            batch=rows[off:off+4];_,sc,d=engine.run(batch,capture=True)
            assert d['gva_max_abs_error']<=.05 and d['logit_max_abs_error']<=.125
            for k in ['hs','logits']:ar[k][off:off+len(batch)]=d[k]
            for k in ['G','U','A']:ar[k][off:off+len(batch)]=np.stack([d[k][:,l,:,u] for l,u in group],axis=-1)
            scores[off:off+len(batch)]=sc
            for a in list(ar.values())+[scores]:a.flush()
            C.save(state,{'done':off+len(batch)})
            if (off+len(batch))%32==0 or off+len(batch)==len(rows):C.log('fresh_2x2',off+len(batch),len(rows),start)
    content=np.maximum(scores[:,0],scores[:,1])>np.maximum(scores[:,2],scores[:,3]);cells={}
    for b,t,m in itertools.product(range(3),range(2),range(4)):
        idx=np.array([r['branch']==b and r['task']==t and r['mode']==m for r in rows]);cells[f'b{b}-t{t}-m{m}']={'n':int(idx.sum()),'accuracy':float(content[idx].mean())}
    shifts={str(t):[] for t in [0,1]}
    for off in range(0,len(rows),4):
        s=scores[off:off+4];shifts[str(rows[off]['task'])].append(float((s[1,0]-s[1,1])-(s[2,0]-s[2,1])))
    style={t:float(np.mean(np.array(v)>0)) for t,v in shifts.items()}
    passed=content.mean()>=.85 and min(c['accuracy'] for c in cells.values())>=.75 and min(style.values())>=.65
    C.save(OUT/'summary.json',{'kind':'real','passed':bool(passed),'content_accuracy':float(content.mean()),'cells':cells,'style_shift_positive_fraction':style,'model_snapshot':plan['model_snapshot'],'source_sha256':file_sha(__file__),'seconds':time.time()-start})
    print({'passed':bool(passed),'content_accuracy':float(content.mean()),'style':style},flush=True)

def analyze():
    plan=C.read(OUT/'prereg.json');assert file_sha(__file__)==plan['source_sha256']
    rows=C.read(OUT/'inputs.json')['rows'];ar={k:np.load(OUT/f'{k}.npy',mmap_mode='r') for k in ['G','U','A','logits']}
    methods=['neutral','chat','debate','mixture','logit_add'];native=['neutral','chat','debate','midpoint','additive','gate_product']
    errors={k:[] for k in methods};ne={k:[] for k in native};families=[];meta=[];bygroup=[]
    for off in range(0,len(rows),4):
        logits=unbits(ar['logits'][off:off+4]).astype(float);lp=logits-logsumexp(logits,axis=-1,keepdims=True)
        preds={'neutral':lp[0],'chat':lp[1],'debate':lp[2],'mixture':np.logaddexp(lp[1],lp[2])-np.log(2)}
        z=logits[1]+logits[2]-logits[0];preds['logit_add']=z-logsumexp(z)
        p=np.exp(lp[3])
        for k in methods:errors[k].append(float(np.sum(p*(lp[3]-preds[k]))))
        a=unbits(ar['A'][off:off+4,1]);g=unbits(ar['G'][off:off+4,1]);u=unbits(ar['U'][off:off+4,1])
        gg=g[1]+g[2]-g[0];uu=u[1]+u[2]-u[0]
        ap={'neutral':a[0],'chat':a[1],'debate':a[2],'midpoint':(a[1]+a[2])/2,'additive':a[1]+a[2]-a[0],'gate_product':gg*expit(gg)*uu}
        denom=np.maximum(((a[1]-a[0])**2+(a[2]-a[0])**2)/2,.01)
        for k in native:ne[k].append(float(np.mean(((ap[k][:32]-a[3,:32])**2)/denom[:32])))
        inter=(a[3]-a[1]-a[2]+a[0])**2/denom
        bygroup.append([float(inter[j:j+32].mean()) for j in [0,32,64]])
        families.append(rows[off]['family']);meta.append({'branch':rows[off]['branch'],'task':rows[off]['task']})
    import branch_style_analysis as A
    summaries={k:float(np.mean(v)) for k,v in errors.items()}
    comparisons={k:A.bootstrap_difference(np.array(v),np.array(errors['logit_add']),families) for k,v in errors.items() if k!='logit_add'}
    cells={f'b{b}-t{t}':{k:float(np.mean(np.array(v)[[m['branch']==b and m['task']==t for m in meta]])) for k,v in errors.items()} for b,t in itertools.product(range(3),range(2))}
    result={'kind':'real','behavior':C.read(OUT/'summary.json'),'output_KL_joint_to_prediction':summaries,'baseline_minus_logit_add_family_bootstrap':comparisons,'output_cells':cells,'native_selected32_normalized_errors':{k:float(np.mean(v)) for k,v in ne.items()},'native_interaction_energy_selected_random0_random1':np.array(bygroup).mean(0).tolist(),'scope':plan['information_policy'],'model_snapshot':plan['model_snapshot'],'prereg_sha256':file_sha(OUT/'prereg.json'),'main_frozen_sha256':plan['frozen_group_sha256']}
    C.save(OUT/'report.json',result)
    print(json.dumps({k:result[k] for k in ['output_KL_joint_to_prediction','native_selected32_normalized_errors','native_interaction_energy_selected_random0_random1']},ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','collect','analyze']);args=p.parse_args()
    {'prepare':prepare,'collect':collect,'analyze':analyze}[args.command]()

if __name__=='__main__':main()
