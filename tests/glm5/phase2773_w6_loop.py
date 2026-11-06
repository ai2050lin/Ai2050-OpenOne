"""W6 v2: preregistered local CUDA weight-edit -> independent evaluation -> CAS.

prepare is CPU-only and freezes inputs/criteria before new model evaluation.
run saves native BF16 selected-position full fields, then confirm-only selection,
untouched lockbox, natural generations, and a paired noninferiority battery.
No model files are changed. A failed hypothesis remains a completed experiment.
"""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OMP_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / 'tests/glm5/result/phase2773_w6_v2'
PUBLIC = ROOT / 'research/phase2773'
RUN = 'P2773-W6-REGISTER-v2'
MODEL = Path('D:/develop/model/Qwen/Qwen3-4B')
SEED = 27732026
LAYERS = list(range(18, 28))
COLLOQ = '说白了，'
FORMAL = '综上所述，'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p, d, immutable=False):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if immutable and p.exists():
        if read(p) != d:
            raise ValueError('Immutable artifact differs: ' + str(p))
        return
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(tmp, p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def log(stage, **kw):
    print(json.dumps({'stage': stage, **kw}, ensure_ascii=False), flush=True)


def status(stage, state='done', **kw):
    from deploy.research_library import Library
    p = PUBLIC/'00-progress.research.json'
    d = read(p) if p.exists() else {'schema':'research_workpackage.v1','kind':'real','run_id':RUN,
        'title':'W6 v2 · 语域权重编辑与能力保持真实闭环','stages':{}}
    d['stages'][stage] = {'state':state,'time':time.time(),**kw}
    d['updated_at'] = time.time()
    if (OUT/'freeze.json').exists():
        d['model_snapshot'] = read(OUT/'freeze.json')['model_snapshot']['sha256']
    save(p,d)
    Library(ROOT/'distributed_data/research_library').ingest(d,str(p.relative_to(ROOT)),kind='local_report')
    log(stage,state=state,**kw)


def digest_question(q):
    import re
    return hashlib.sha256(re.sub(r'\s+','',q['question']+'|'+ '|'.join(q['choices'])).encode()).hexdigest()


def render(i, reg):
    # Entire record is a family: numerical/actor variants never cross splits.
    names=['林安','陈宁','周敏','王晨','刘悦','张平','李欣','赵文','吴清','郑远']
    items=['记录册','文件夹','标签纸','信封','笔记本','资料袋','包装盒','索引卡','书签','绘图纸']
    actor=names[i%10]; item=items[(i//10)%10]
    n=40+(i*7)%71; add=3+(i*11)%19; sub=1+(i*3)%min(13,add+1)
    place=['图书室','资料中心','展览馆','培训室','档案室','社区服务站'][i%6]
    if reg=='acad':
        text=(f'本记录描述{place}的一次物资核对，负责人为{actor}。初始登记的{item}数量为{n}份，'
          f'随后接收补充物资{add}份，并向另一工作组移交{sub}份。本次核对的目的在于明确数量变化的依据，'
          '而非评价负责人或推断未记载的事件。各项数字均指同一种物资，记录期间未发生损耗或额外调拨。\n\n'
          '核对程序要求先确认初始状态，再依时间顺序纳入增加与减少的记录。初始数量和后续变化必须分别保存，'
          '以便另一位工作人员复核。有关存放位置、外包装以及接收时间的补充说明，只提供工作背景，不改变数量关系。'
          '不能因为一份材料被移动到另一张桌面，就将其计为新增物资；已经移交的部分也不再属于当前库存。\n\n'
          '审查人员另行保留了原始登记表与移交凭证。不同工作人员可以采用不同表述，但其陈述必须保持相同事实，'
          '不得增加未经记录的原因。复核时应当指出哪些内容来自直接记录，哪些属于额外解释；若没有额外证据，'
          '则不应对工作效率、个人动机或未来安排作出判断。上述程序适用于本记录中的全部数量。\n\n'
          f'据此，本次实际剩余数量可由{n}加{add}再减{sub}得到。请接着写一句简短总结，保留物品名称和最终数量。')
    else:
        text=(f'咱们说的是{place}这次清点物资的事，负责的人是{actor}。一开始有{n}份{item}，'
          f'后来又送来{add}份，接着拿走{sub}份交给另一个小组。事情就是这么个顺序，'
          '先把数量弄明白就行，别忙着评价谁干得好不好，也别自己补上没发生的事。这里说的都是同一种东西，'
          '中间没有弄丢，也没有别的组再拿来或者拿走。\n\n'
          '核对的时候，先看最早记了多少，再把后来增加的和减少的依次算进去。原来的数字留着，后面的变化也留着，'
          '这样换个人来查，也能看懂是怎么算的。至于东西放哪儿、外面怎么包、几点钟送到，这些只是背景，'
          '不会把数量变多或者变少。把一份东西从这张桌子挪到那张桌子，当然不能算又多了一份；'
          '已经交给别组的那部分，也不能还算在咱们手里。\n\n'
          '原来的登记表和交接凭证都留着呢。每个人说话的习惯不一样没关系，但说的事情得一样，'
          '不能为了讲得顺口，就多编一个原因。再查一遍时，要分清哪些是记下来的，哪些是自己猜的。'
          '没有别的证据，就先别猜大家干活快不快、心里怎么想，或者以后打算怎么办。这些要求，前后都一样。\n\n'
          f'所以，手里还剩多少，就是把{n}加上{add}，再减去{sub}。请接着写一句简短总结，保留物品名称和最终数量。')
    return text, {'item':item,'quantity':n+add-sub,'actor':actor}


def prepare():
    import numpy as np
    from transformers import AutoTokenizer
    from deploy.parameter_reader import model_snapshot
    from tools.power_analysis import check_phase, n_required
    if (OUT/'freeze.json').exists():
        raise ValueError('Already frozen; resume run or create a new version')
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    # Assignment independent of labels and model response; families are complete.
    order=sorted(range(300),key=lambda i:hashlib.sha256(f'{SEED}:record:{i}'.encode()).hexdigest())
    split={i:('discovery' if k<60 else 'confirm' if k<140 else 'lockbox') for k,i in enumerate(order)}
    rows=[]
    for i in range(300):
        for reg in ['acad','colloq']:
            text, fact=render(i,reg)
            prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(prompt,add_special_tokens=False)
            rows.append({'id':f'evt-{i:03d}-{reg}','family_id':f'evt-{i:03d}','split':split[i],
                'register':reg,'text':text,'prompt':prompt,'token_ids':ids,'tokens':len(ids),'fact':fact})
    assert min(r['tokens'] for r in rows)>=256
    assert max(r['tokens'] for r in rows)<=1024
    assert len({r['token_ids'][-1] for r in rows})==1
    old=[json.loads(s) for s in (ROOT/'data/lang_bench/q9b_capgate.v1.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    capability=[]; seen=set()
    for q in old:
        h=digest_question(q)
        if h in seen:continue
        seen.add(h)
        q=dict(q);q['gate_eligible']='M3Exam' not in q['source']
        q['label_status']='mirror_unverified' if not q['gate_eligible'] else 'provided_dataset_key'
        capability.append(q)
    # The fourth source remains the local project pool, restricted to CMExam train.
    pool=[json.loads(s) for s in (ROOT/'data/lang_bench/questions.v2.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    pool=[q for q in pool if q.get('source')=='CMExam train' and len(q.get('choices',[])) in (4,5)
          and len(q.get('answer',''))==1 and q['answer'] in 'ABCDE'[:len(q['choices'])]]
    pool.sort(key=lambda q:hashlib.sha256(f'{SEED}:{q["id"]}'.encode()).hexdigest())
    for q in pool:
        if sum(q['gate_eligible'] for q in capability)>=2200:break
        h=digest_question(q)
        if h in seen:continue
        seen.add(h);q=dict(q);q.update(gate_eligible=True,label_status='provided_dataset_key',source_group='local pool extension')
        capability.append(q)
    assert sum(q['gate_eligible'] for q in capability)==2200
    # Correctness is defined by supplied keys; M3Exam excluded from the NI gate.
    design={
        'schema':'w6_prereg.v2','run_id':RUN,'kind':'real','seed':SEED,'model':str(MODEL),'precision':'bf16',
        'question':'Does removal/suppression of a context register direction in selected MLP columns alter heldout continuation while preserving keyed capability?',
        'hypothesis_status':'unproven; failure does not imply distributed capacity',
        'prior_results_exposed':['Q9A/Q9B v1 and Phase2772; new corpus and splits below'],
        'layers':LAYERS,'direction':'academic minus colloquial mean at block output, discovery only, normalized per block',
        'ranking':'mean absolute actual contribution abs((u.T W[:,j])*a_j(x)) over discovery; static abs(u.T W) saved as baseline',
        'operator':'W_new = W_base - alpha * outer(u, (u.T W)*column_mask)',
        'k_grid':[32,128,512,9728],'alpha_grid':[-1.0,0.5,1.0,2.0,4.0,6.0],
        'controls':['base alpha0','same-columns norm-matched random directions seeds 0/1/2',
                    'activation-energy-matched random columns seed0','wrong layers 0..9 same output directions norm-matched'],
        'search_budget':'24 target configurations on confirm only; select max mean colloq-formal logP shift, then fewer parameters, then lower abs(alpha)',
        'primary_readout':{'colloquial':COLLOQ,'formal':FORMAL,'metric':'full forced continuation sum logP, no length normalization',
            'scope':'two fixed continuation alternatives; not a complete natural style classifier'},
        'selection':'confirm only; no lockbox/capability access before selected-edit.json is frozen',
        'primary_criterion':{'min_flip_fraction':0.20,'cluster_bootstrap':3000,'one_sided_alpha':0.05,
            'require_baseline_academic_choice_accuracy':0.60,'require_positive_95pct_lower_effect_vs_all_controls':True},
        'natural_generation':{'families_per_split':16,'registers':['acad','colloq'],'max_new_tokens':96,'do_sample':False,
            'selection':'first family IDs by frozen SHA order, confirm and lockbox, same prompts per edit',
            'scoring':'predeclared lexical markers and literal item/decimal quantity preservation; automatic proxies only; no natural-style causal promotion'},
        'natural_markers':{'colloquial':['说白了','咱们','嘛','啦','呀','呗','就剩','这么','反正'],
                           'academic':['综上','因此','据此','表明','可知','最终','数量为','剩余数量']},
        'capability':{'provided_key_n':2200,'m3exam_reference_n':140,'noninferiority_margin':0.02,'familywise_alpha':0.05,
            'method':'paired losses indicator and gains; simultaneous one-sided 97.5% Clopper-Pearson bounds; upper(loss)-lower(gain)',
            'pass':'upper bound <= 0.02; inconclusive otherwise unless lower bound > 0.02',
            'per_source':'paired point estimates and intervals; no claim of per-source 2pt preservation',
            'm3exam':'reference only, excluded due to unverified mirror labels',
            'prompt':'original benchmark question and all 4 or 5 choices, no artificial padding, chat no thinking; full one-token A..D/E likelihood; no multi-answer items'},
        'power':{'primary':check_phase(.20,160,p_d=.5),'capability_difference_planning':check_phase(.02,2200,p_d=.1),
            'ni_conservative_zero_gain_design':{'margin':.02,'expected_loss':.005,'n':2200,
                'note':'exact binomial rejection/power computed in statistical verification; CI may remain inconclusive'}},
        'raw_fields':'all 37 hidden-state arrays at last two causal-visible positions + all 9728 down_proj input units at layers18..27; uint16 BF16 bit preservation; no full-token field claim',
        'evidence':'record forced-continuation effects separately from natural generation, capability, parameter provenance; no automatic mechanism claim',
        'model_files':'read only; edits applied to in-memory CUDA copies and restored exactly; save factors/masks only',
        'time_budget_seconds':5400,'resource':'one local CUDA bf16 model, batch4; no quantization or synthetic fallback',
        'not_measured':['new vocabulary generalization','cross-language','other models','SAE discovery','graph-only prediction','multi-node replication'],
    }
    assert design['power']['primary']['pass'] and design['power']['capability_difference_planning']['pass']
    from scipy.stats import binom, beta
    cutoff=max(k for k in range(100) if beta.ppf(.975,k+1,2200-k)<=.02)
    design['power']['ni_conservative_zero_gain_design'].update(loss_cutoff=cutoff,
        power=float(binom.cdf(cutoff,2200,.005)),null_type1_upper=float(binom.cdf(cutoff,2200,.02)))
    assert design['power']['ni_conservative_zero_gain_design']['power']>=.8
    save(OUT/'corpus.json',rows,True);save(OUT/'capability-inputs.json',capability,True)
    save(PUBLIC/'W6-v2-prereg.json',design,True)
    snap=model_snapshot(MODEL)
    freeze={'run_id':RUN,'created_at':time.time(),'design_sha256':sha(PUBLIC/'W6-v2-prereg.json'),
        'corpus_sha256':sha(OUT/'corpus.json'),'capability_sha256':sha(OUT/'capability-inputs.json'),
        'script_sha256':sha(__file__),'model_snapshot':snap,'length_range':[min(r['tokens'] for r in rows),max(r['tokens'] for r in rows)],
        'split_families':{'discovery':60,'confirm':80,'lockbox':160},'shared_final_token':rows[0]['token_ids'][-1]}
    save(OUT/'freeze.json',freeze,True)
    status('freeze',families=300,samples=600,length_range=freeze['length_range'],capability_n=len(capability),power=design['power'])


def paired_interval(base, edited, eligible=None, delta=.02):
    """Finite-sample valid simultaneous loss/gain binomial bounds (Bonferroni)."""
    import numpy as np
    from scipy.stats import beta
    b=np.asarray(base,bool);e=np.asarray(edited,bool)
    if eligible is not None:b,e=b[eligible],e[eligible]
    n=len(b)
    if not n:return {'n':0,'status':'not_tested'}
    loss=int((b&~e).sum());gain=int((~b&e).sum())
    def bounds(k):
        return (float(beta.ppf(.025,k,n-k+1)) if k else 0.,
                float(beta.ppf(.975,k+1,n-k)) if k<n else 1.)
    lo_loss,hi_loss=bounds(loss);lo_gain,hi_gain=bounds(gain)
    lower=lo_loss-hi_gain;upper=hi_loss-lo_gain
    return {'n':n,'base_accuracy':float(b.mean()),'edited_accuracy':float(e.mean()),'losses':loss,'gains':gain,
        'net_loss':float(b.mean()-e.mean()),'loss_interval':[lower,upper],'margin':delta,
        'status':'noninferior' if upper<=delta else 'inferior' if lower>delta else 'inconclusive',
        'method':'Bonferroni simultaneous Clopper-Pearson loss/gain bounds, confidence >=95%'}


def bootstrap_mean(values):
    import numpy as np
    v=np.asarray(values,float);rng=np.random.default_rng(SEED+55)
    a=v[rng.integers(0,len(v),size=(3000,len(v)))].mean(1)
    return {'mean':float(v.mean()),'ci95':[float(np.quantile(a,.025)),float(np.quantile(a,.975))],'n_families':len(v)}


class Engine:
    def __init__(self):
        import torch
        import numpy as np
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch=torch;self.np=np;self.started=time.time()
        assert torch.cuda.is_available(),'CUDA is mandatory'
        torch.manual_seed(SEED);np.random.seed(SEED)
        torch.set_num_threads(4)
        self.tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        self.tok.padding_side='left'
        self.model=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,dtype=torch.bfloat16,
            device_map='cuda:0',attn_implementation='sdpa').eval()
        self.base={l:self.model.model.layers[l].mlp.down_proj.weight.detach().clone() for l in range(28)}
        self.env={'torch':torch.__version__,'transformers':transformers.__version__,'device':torch.cuda.get_device_name(),
            'precision':'bf16','attention':'sdpa','batch':4,'padding':'left','thinking':False,'seed':SEED}

    def batch(self, seqs):
        torch=self.torch
        return {k:v.cuda() for k,v in self.tok.pad({'input_ids':seqs},padding=True,return_tensors='pt').items()}

    def timeout(self):
        if time.time()-self.started>5400:raise TimeoutError('Frozen 90-minute budget reached; checkpoints retained')

    def scores(self, rows):
        torch=self.torch;ans=[self.tok.encode(t,add_special_tokens=False) for t in [COLLOQ,FORMAL]]
        values=[]
        for start in range(0,len(rows),4):
            self.timeout();chunk=rows[start:start+4];seq=[r['token_ids']+a for r in chunk for a in ans]
            batch=self.batch(seq)
            with torch.inference_mode():out=self.model(**batch,logits_to_keep=max(map(len,ans))+1,use_cache=False)
            lp=out.logits.float().log_softmax(-1);vs=[]
            for j in range(len(seq)):
                a=ans[j%2];ix=torch.tensor(a,device='cuda')
                vs.append(float(lp[j,-len(a)-1:-1].gather(1,ix[:,None]).sum().item()))
            values.extend(self.np.asarray(vs).reshape(-1,2).tolist())
            del out,lp,batch
        return self.np.asarray(values)

    def capture(self, rows, folder):
        torch=self.torch;np=self.np;folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        files=[]
        for start in range(0,len(rows),4):
            p=folder/f'batch-{start:04d}.npz'
            if p.exists():continue
            self.timeout();chunk=rows[start:start+4];caps={};handles=[]
            for l in LAYERS:
                def hook(module,inputs,l=l):caps[l]=inputs[0][:,-2:,:].detach().clone()
                handles.append(self.model.model.layers[l].mlp.down_proj.register_forward_pre_hook(hook))
            try:
                with torch.inference_mode():out=self.model(**self.batch([r['token_ids'] for r in chunk]),
                    logits_to_keep=1,output_hidden_states=True,use_cache=False)
                hs=torch.stack([h[:,-2:,:] for h in out.hidden_states],1).contiguous().cpu()
                mlp=torch.stack([caps[l] for l in LAYERS],1).contiguous().cpu()
                np.savez(p,hidden_bits=hs.view(torch.uint16).numpy(),mlp_bits=mlp.view(torch.uint16).numpy(),
                    ids=np.array([r['id'] for r in chunk]))
                del out,hs,mlp,caps
            finally:
                for h in handles:h.remove()
            if start%80==0:log('native_capture',completed=start+len(chunk),total=len(rows))
        for p in sorted(folder.glob('batch-*.npz')):files.append({'path':p.name,'sha256':sha(p),'bytes':p.stat().st_size})
        save(folder/'manifest.json',{'kind':'real','dtype':'BF16 stored losslessly as uint16 bits','positions':[-2,-1],
            'hidden_boundary':'HF hidden_states 0..36; hs[-1] after final norm','mlp_boundary':'down_proj.input',
            'layers':LAYERS,'files':files,'env':self.env},True)

    def directions(self, rows):
        torch=self.torch;np=self.np;xs=[];acts=[];ids=[]
        for p in sorted((OUT/'native').glob('batch-*.npz')):
            z=np.load(p);xs.append(torch.from_numpy(z['hidden_bits'].copy()).view(torch.bfloat16).float().numpy())
            acts.append(torch.from_numpy(z['mlp_bits'].copy()).view(torch.bfloat16).float().numpy());ids.extend(z['ids'].tolist())
        h=np.concatenate(xs);a=np.concatenate(acts);lookup={r['id']:r for r in rows}
        disc=np.array([lookup[i]['split']=='discovery' for i in ids]);acad=np.array([lookup[i]['register']=='acad' for i in ids])
        self.u={};self.ranking={};self.static={};self.energy={}
        for j,l in enumerate(LAYERS):
            v=h[disc&acad,l+1,-1].mean(0)-h[disc&~acad,l+1,-1].mean(0);v=v/np.linalg.norm(v)
            self.u[l]=torch.tensor(v,device='cuda',dtype=torch.float32)
            c=(self.u[l]@self.base[l].float()).cpu().numpy()
            score=np.abs(a[disc,j,-1]*c).mean(0)
            self.ranking[l]=np.argsort(-score,kind='stable');self.static[l]=np.argsort(-np.abs(c),kind='stable')
            self.energy[l]=np.abs(a[disc,j,-1]).mean(0)
        np.savez(OUT/'directions.npz',**{f'u_{l}':self.u[l].cpu().numpy() for l in LAYERS},
            **{f'order_{l}':self.ranking[l] for l in LAYERS},**{f'static_{l}':self.static[l] for l in LAYERS},
            **{f'energy_{l}':self.energy[l] for l in LAYERS})
        return {'n_discovery':int(disc.sum()),'hs0_final_token_max_difference':float(np.ptp(h[disc,0,-1],axis=0).max()),
            'source':'discovery-only paired register direction; no label-free feature discovery claim'}

    @contextlib.contextmanager
    def edit(self, cfg, store=False):
        torch=self.torch;np=self.np;layers=LAYERS if cfg.get('control')!='wrong_layers' else list(range(10))
        details=[];arrays={};seed=int(cfg.get('seed',0))
        try:
            with torch.no_grad():
                for pos,l in enumerate(layers):
                    source_l=LAYERS[pos];base=self.base[l];u=self.u[source_l];k=cfg['k'];alpha=cfg['alpha']
                    ix=self.ranking[source_l][:k]
                    # Static writing magnitude and input activation energy both retained.
                    if cfg.get('control')=='random_columns':
                        rng=np.random.default_rng(SEED+500+source_l)
                        energy=self.energy[source_l];order=np.argsort(energy,kind='stable');rank=np.empty(len(order),int);rank[order]=np.arange(len(order))
                        bins=np.array_split(order,16);picked=[]
                        for bucket in bins:
                            count=int(np.isin(bucket,ix).sum())
                            picked.extend(rng.choice(bucket,count,replace=False).tolist())
                        ix=np.asarray(picked,int)
                    idx=torch.tensor(ix,device='cuda');w=base[:,idx].float();target=u[:,None]*(u@w)[None,:]
                    if cfg.get('control')=='random_direction':
                        gen=torch.Generator(device='cuda').manual_seed(SEED+source_l+seed*100)
                        v=torch.randn(u.shape,generator=gen,device='cuda');v=v/torch.linalg.vector_norm(v)
                        delta=v[:,None]*(v@w)[None,:]
                        delta*=torch.linalg.vector_norm(target)/torch.linalg.vector_norm(delta).clamp_min(1e-12)
                    else:delta=target
                    edited=(w-alpha*delta).to(torch.bfloat16)
                    actual=(edited.float()-w)
                    self.model.model.layers[l].mlp.down_proj.weight[:,idx]=edited
                    details.append({'layer':l,'source_direction_layer':source_l,'columns':ix.tolist(),
                        'addressed_params':int(w.numel()),'changed_params':int((edited!=base[:,idx]).sum().item()),
                        'ideal_rank_upper':1,'relative_frobenius':float(torch.linalg.vector_norm(actual)/torch.linalg.vector_norm(base.float())),
                        'rounding':'exact BF16 diff can have rank greater than ideal rank1'})
                    if store:
                        arrays[f'l{l}_columns']=ix;arrays[f'l{l}_u']=u.cpu().numpy()
                        arrays[f'l{l}_edited_bits']=edited.contiguous().cpu().view(torch.uint16).numpy()
                    del w,target,delta,edited,actual
            if store and not (OUT/'selected-edit-manifest.json').exists():
                np.savez(OUT/'selected-edit.npz',**arrays)
                save(OUT/'selected-edit-manifest.json',{'config':cfg,'tensors':details,'sha256':sha(OUT/'selected-edit.npz'),
                    'base_snapshot':read(OUT/'freeze.json')['model_snapshot']['sha256']},True)
            yield details
        finally:
            with torch.no_grad():
                for l in layers:self.model.model.layers[l].mlp.down_proj.weight.copy_(self.base[l])
            assert all(torch.equal(self.model.model.layers[l].mlp.down_proj.weight,self.base[l]) for l in layers)

    def capability(self, qs):
        torch=self.torch;letters=[self.tok.encode(s,add_special_tokens=False) for s in 'ABCDE']
        assert all(len(x)==1 for x in letters),'Full one-token answer scoring required'
        ids=[x[0] for x in letters];values=[]
        for start in range(0,len(qs),4):
            self.timeout();chunk=qs[start:start+4];prompts=[]
            for q in chunk:
                text=q['question']+'\n'+'\n'.join(f'{s}. {c}' for s,c in zip('ABCDE',q['choices']))+'\n请只回答正确选项字母。'
                prompts.append(self.tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,
                    add_generation_prompt=True,enable_thinking=False))
            with torch.inference_mode():out=self.model(**self.batch(prompts),logits_to_keep=1,use_cache=False)
            scores=out.logits[:,-1,ids].float().cpu().numpy()
            for q,s in zip(chunk,scores):
                s=s[:len(q['choices'])]
                pred='ABCDE'[int(s.argmax())];values.append({'id':q['id'],'prediction':pred,'answer':q['answer'],
                    'correct':pred==q['answer'],'choice_logits':s.tolist(),'source':q['source'],'gate_eligible':q['gate_eligible']})
            del out
            if start%400==0:log('capability',completed=start+len(chunk),total=len(qs))
        return values

    def generate(self, rows):
        torch=self.torch;records=[];design=read(PUBLIC/'W6-v2-prereg.json')
        for start in range(0,len(rows),4):
            self.timeout();chunk=rows[start:start+4];batch=self.batch([r['token_ids'] for r in chunk]);length=batch['input_ids'].shape[1]
            with torch.inference_mode():out=self.model.generate(**batch,max_new_tokens=96,do_sample=False,use_cache=True,
                pad_token_id=self.tok.pad_token_id,eos_token_id=self.tok.eos_token_id)
            for r,tokens in zip(chunk,out[:,length:]):
                text=self.tok.decode(tokens,skip_special_tokens=True);fact=r['fact']
                records.append({'id':r['id'],'family_id':r['family_id'],'split':r['split'],'register':r['register'],
                    'text':text,'expected':fact,'literal_content_ok':fact['item'] in text and str(fact['quantity']) in text,
                    'colloquial_markers':sum(text.count(s) for s in design['natural_markers']['colloquial']),
                    'academic_markers':sum(text.count(s) for s in design['natural_markers']['academic']),
                    'generated_tokens':int((tokens!=self.tok.pad_token_id).sum().item()),'hit_limit':len(tokens)==96 and int(tokens[-1])!=self.tok.eos_token_id})
            del out,batch
        return records


def run():
    import numpy as np
    from deploy.parameter_reader import model_snapshot
    freeze=read(OUT/'freeze.json')
    for p,key in [(PUBLIC/'W6-v2-prereg.json','design_sha256'),(OUT/'corpus.json','corpus_sha256'),
                  (OUT/'capability-inputs.json','capability_sha256'),(Path(__file__),'script_sha256')]:
        assert sha(p)==freeze[key],f'Frozen dependency changed: {p}'
    assert model_snapshot(MODEL)['sha256']==freeze['model_snapshot']['sha256']
    if (OUT/'result.json').exists():
        log('already_complete',result_sha256=sha(OUT/'result.json'));return
    rows=read(OUT/'corpus.json');disc=[r for r in rows if r['split']=='discovery']
    engine=Engine();status('pilot','running')
    pilot=disc[:8]
    if not (OUT/'pilot.json').exists():
        t=time.time();pilot_scores=engine.scores(pilot)
        pilot_result={'kind':'real','n':len(pilot),'seconds':time.time()-t,'scores':pilot_scores.tolist(),
            'register_choice_accuracy':float(np.mean((pilot_scores[:,0]>pilot_scores[:,1])==np.array([r['register']=='colloq' for r in pilot]))),
            'env':engine.env,'note':'Readout gate is reported, not used to retune after freeze; both outcomes retained'}
        save(OUT/'pilot.json',pilot_result,True)
    status('pilot',**read(OUT/'pilot.json'))
    status('collection','running');engine.capture(rows,OUT/'native')
    baseline_path=OUT/'baseline-scores.json'
    baseline_rows=[r for r in rows if r['split']!='lockbox']
    if not baseline_path.exists():save(baseline_path,{'ids':[r['id'] for r in baseline_rows],'scores':engine.scores(baseline_rows).tolist()},True)
    b=np.asarray(read(baseline_path)['scores']);status('collection',samples=len(rows),native_manifest_sha256=sha(OUT/'native/manifest.json'))
    direction_meta=engine.directions(rows);save(OUT/'direction-meta.json',direction_meta,True)
    conf=[r for r in rows if r['split']=='confirm' and r['register']=='acad']
    lock=[r for r in rows if r['split']=='lockbox' and r['register']=='acad']
    index={r['id']:i for i,r in enumerate(baseline_rows)};bc=b[[index[r['id']] for r in conf]]
    def margin(a):return a[:,0]-a[:,1]
    status('discovery','running',**direction_meta)
    curve=[]
    for k in [32,128,512,9728]:
        for alpha in [-1.,.5,1.,2.,4.,6.]:
            cfg={'k':k,'alpha':alpha,'control':'target'};key=f'k{k}-a{alpha}'
            p=OUT/'confirm'/f'{key}.json'
            if not p.exists():
                with engine.edit(cfg) as details:s=engine.scores(conf)
                save(p,{'config':cfg,'scores':s.tolist(),'tensors':details},True)
            d=read(p);s=np.asarray(d['scores']);shift=margin(s)-margin(bc)
            rec={'id':key,**cfg,'shift':float(shift.mean()),'flip_fraction':float(((margin(bc)<0)&(margin(s)>0)).mean()),
                 'addressed_params':sum(t['addressed_params'] for t in d['tensors']),
                 'changed_params':sum(t['changed_params'] for t in d['tensors']),
                 'max_relative_frobenius':max(t['relative_frobenius'] for t in d['tensors'])}
            curve.append(rec);log('dose',**rec)
    selected=max(curve,key=lambda r:(r['shift'],-r['addressed_params'],-abs(r['alpha'])))
    chosen={k:selected[k] for k in ['k','alpha','control']}
    if not (OUT/'selected-edit.json').exists():
        save(OUT/'selected-edit.json',{'config':chosen,'selected_on':'confirm only','confirm_row':selected,
            'selection_complete_at':time.time(),'lockbox_opened':False},True)
    assert read(OUT/'selected-edit.json')['config']==chosen
    save(OUT/'dose-curve.json',{'run_id':RUN,'split':'confirm','rows':curve,'selected':selected},True)
    status('discovery',config=chosen,candidates=len(curve))
    status('validation','running')
    if not (OUT/'lockbox-base.json').exists():save(OUT/'lockbox-base.json',{'scores':engine.scores(lock).tolist()},True)
    bl=np.asarray(read(OUT/'lockbox-base.json')['scores'])
    results={};control_configs=[('target',chosen),
        *[(f'random-direction-{s}',{**chosen,'control':'random_direction','seed':s}) for s in range(3)],
        ('energy-matched-columns',{**chosen,'control':'random_columns'}),
        ('wrong-layers',{**chosen,'control':'wrong_layers'})]
    for name,cfg in control_configs:
        p=OUT/'lockbox'/f'{name}.json'
        if not p.exists():
            with engine.edit(cfg,store=name=='target') as details:s=engine.scores(lock)
            save(p,{'config':cfg,'ids':[r['id'] for r in lock],'scores':s.tolist(),'tensors':details},True)
        d=read(p);s=np.asarray(d['scores']);effect=margin(s)-margin(bl)
        results[name]={'config':cfg,'shift':bootstrap_mean(effect),
            'flip_fraction':float(((margin(bl)<0)&(margin(s)>0)).mean()),'forced_colloquial_fraction':float((margin(s)>0).mean())}
    target=np.asarray(read(OUT/'lockbox/target.json')['scores'])
    comparisons={name:bootstrap_mean(margin(target)-margin(np.asarray(read(OUT/'lockbox'/f'{name}.json')['scores'])))
                 for name,_ in control_configs if name!='target'}
    # Natural generations use frozen first hashed families, not chosen examples.
    natural=[]
    for split in ['confirm','lockbox']:
        families=sorted({r['family_id'] for r in rows if r['split']==split},key=lambda s:hashlib.sha256(f'{SEED}:{s}'.encode()).hexdigest())[:16]
        natural.extend(r for r in rows if r['split']==split and r['family_id'] in families)
    if not (OUT/'generation-base.json').exists():save(OUT/'generation-base.json',engine.generate(natural),True)
    if not (OUT/'generation-edited.json').exists():
        with engine.edit(chosen):save(OUT/'generation-edited.json',engine.generate(natural),True)
    qs=read(OUT/'capability-inputs.json');status('capability','running',n=len(qs))
    if not (OUT/'capability-base.json').exists():save(OUT/'capability-base.json',engine.capability(qs),True)
    if not (OUT/'capability-edited.json').exists():
        with engine.edit(chosen):save(OUT/'capability-edited.json',engine.capability(qs),True)
    basecap=read(OUT/'capability-base.json');editcap=read(OUT/'capability-edited.json')
    cap=paired_interval([r['correct'] for r in basecap],[r['correct'] for r in editcap],np.array([r['gate_eligible'] for r in basecap]))
    sources={}
    for source in sorted({r['source'] for r in basecap}):
        take=np.array([r['source']==source for r in basecap]);sources[source]=paired_interval([r['correct'] for r in basecap],[r['correct'] for r in editcap],take)
    effects_ok=all(c['ci95'][0]>0 for c in comparisons.values())
    basegate=float((margin(bl)<0).mean())>=.60
    passed=basegate and results['target']['flip_fraction']>=.20 and effects_ok and cap['status']=='noninferior'
    counter=[]
    for j,r in enumerate(lock):
        if margin(target)[j]<=margin(bl)[j] or margin(target)[j]<=0:
            counter.append({'sample_id':r['id'],'reason':'no desired switch or nonpositive dose effect',
                'base_margin':float(margin(bl)[j]),'edited_margin':float(margin(target)[j])})
    for a,e in zip(basecap,editcap):
        if a['correct'] and not e['correct']:counter.append({'sample_id':a['id'],'reason':'keyed capability regression',
            'base':a['prediction'],'edited':e['prediction'],'answer':a['answer'],'source':a['source']})
    for a,e in zip(read(OUT/'generation-base.json'),read(OUT/'generation-edited.json')):
        if not e['literal_content_ok']:counter.append({'sample_id':e['id'],'reason':'generated literal-content proxy failed',
            'expected':e['expected'],'base':a['text'],'edited':e['text']})
    result={'schema':'w6_result.v2','kind':'real','run_id':RUN,'model_snapshot':freeze['model_snapshot']['sha256'],
        'design_sha256':freeze['design_sha256'],'env':engine.env,'selected':selected,'lockbox':results,'control_comparisons':comparisons,
        'baseline_readout_gate':basegate,'baseline_academic_choice_fraction':float((margin(bl)<0).mean()),
        'capability':cap,'capability_sources':sources,'target_supported_in_scope':passed,
        'verdict':'controlled_continuation_effect' if passed else 'not_supported_in_preregistered_scope',
        'natural_generation_claim':'descriptive proxy only; inspect full generations',
        'counterexamples':counter,'compactness':{'status':'observed_upper_bound' if passed else 'not_found',
            'smallest_tested_feasible_params':selected['addressed_params'] if passed else None,
            'scope':'selected candidate tested on lockbox; not a proven minimum; search used confirm only'},
        'seconds':time.time()-engine.started,'peak_cuda_bytes':int(engine.torch.cuda.max_memory_allocated()),
        'not_measured':read(PUBLIC/'W6-v2-prereg.json')['not_measured']}
    def git(*a):
        return subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}',*a],cwd=ROOT,text=True).strip()
    result['provenance']={'git_sha':git('rev-parse','HEAD'),'git_dirty':bool(git('status','--porcelain')),
        'model_path':str(MODEL),'model_revision':freeze['model_snapshot']['sha256'],'seed':SEED,
        'script_sha256':freeze['script_sha256'],'created_at':time.time()}
    save(OUT/'result.json',result,True)
    status('capability',**cap);status('validation','done' if passed else 'not_supported',verdict=result['verdict'],counterexamples=len(counter))
    log('complete',verdict=result['verdict'],capability=cap,seconds=result['seconds'])


def publish():
    from deploy.research_library import Library
    result=read(OUT/'result.json');freeze=read(OUT/'freeze.json');lib=Library(ROOT/'distributed_data/research_library')
    report={'schema':'research_intervention_report.v1','kind':'real','run_id':RUN,
        'title':'W6 v2：语域方向权重编辑、独立留出与能力保持','model_snapshot':result['model_snapshot'],
        'result':result,'dose_curve':read(OUT/'dose-curve.json'),
        'preregistration':read(PUBLIC/'W6-v2-prereg.json'),'freeze':freeze,
        'generations':{'base':read(OUT/'generation-base.json'),'edited':read(OUT/'generation-edited.json')},
        'raw_manifest':read(OUT/'native/manifest.json'),'edit_manifest':read(OUT/'selected-edit-manifest.json'),
        'sample_preview':[],
        'artifact_files':{p.name:sha(p) for p in OUT.glob('*.json') if p.name not in ['registration.json']},
        'limitations':['受控中文记录与固定续写候选，不代表一般语域能力',
            '自然生成指标为词表与字面保义代理，尚无独立人工风格评审',
            '能力门针对混合题池提供答案，M3Exam 镜像答案仅参考',
            '只测一个模型快照，未做第二节点复算、SAE或图谱独立预测']}
    p=PUBLIC/'report.json';save(p,report,True);ref=lib.ingest(report,str(p.relative_to(ROOT)),kind='local_report')['sha256']
    edit=read(OUT/'selected-edit-manifest.json');cards=[];edges=[]
    level='intervention' if result['target_supported_in_scope'] else 'negative'
    for tensor in edit['tensors']:
        l=tensor['layer'];unit=tensor['columns'][0];fid=f'{result["model_snapshot"]}:B{l}:W6v2:{RUN}'
        cards.append({'id':fid,'label':f'W6 v2 · B{l} 语域编辑候选（'+('范围内有效' if level=='intervention' else '未通过主判')+'）',
            'layer':l,'boundary':'mlp.down_proj.input','index':{'space':'mlp','unit':unit},
            'method':'discovery-conditioned output-direction edit; representative input unit locates one of all edited columns',
            'run_id':RUN,'scope':{'readout':'final','language':'zh','edited_columns':tensor['columns'],
                'representative_unit_only':True},'evidence':{'level':level,'artifact_sha256':ref},
            'causal_tests':'权重编辑已执行；仅限登记条件与固定续写候选',
            'counterexamples':result['counterexamples'],'compactness':result['compactness'],
            'summary':{'limitations':report['limitations']}})
        edges.append({'id':fid+':edge','source':'language:register-academic-vs-colloquial','target':fid,
            'relation':'tested_directional_weight_edit' if level=='intervention' else 'unsupported_directional_weight_edit',
            'run_id':RUN,'evidence':{'level':level,'artifact_sha256':ref}})
    graph={'schema':'research_graph.v1','kind':'real','run_id':RUN,'model_snapshot':result['model_snapshot'],
        'feature_cards':cards,'graph_edges':edges,'note':'Representative units locate edited columns; no claim that one neuron encodes register.'}
    p=PUBLIC/'graph.research.json';save(p,graph,True);g=lib.ingest(graph,str(p.relative_to(ROOT)),kind='registered')['sha256']
    registration={'run_id':RUN,'report_sha256':ref,'graph_sha256':g,'feature_id':cards[0]['id'],
        'client_url':f'http://127.0.0.1:5173/workbench?lens=process&station=plan&run={RUN}',
        'features':len(cards),'edges':len(edges),'source':'local real CUDA; not remote distributed replication'}
    save(OUT/'registration.json',registration,True);status('registration_validated',**registration)
    log('published',**registration)


if __name__=='__main__':
    cmd=argparse.ArgumentParser();cmd.add_argument('stage',choices=['prepare','run','publish']);args=cmd.parse_args()
    {'prepare':prepare,'run':run,'publish':publish}[args.stage]()
