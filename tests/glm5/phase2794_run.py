"""Prospectively frozen real CUDA collection, native-neuron paths and doses."""
import sys,json,time,gc,subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase2794_materials as P
from shared.dist_protocol.mechanism_trace import GpuLease,Recorder as NativeRecorder,TraceStore,digest,file_sha,unbits,save_json
from shared.dist_protocol.natural_register_runner import snapshot
from deploy.research_library import Library

class Recorder(NativeRecorder):
    def finish(self,*args,**kwargs):
        d,p=super().finish(*args,**kwargs)
        d['causal_visibility']='Scope instruction is in the system prefix and is visible at entity and final readouts; v2793 future-cue negative control does not apply.'
        d.pop('trace_id');d['trace_id']=digest(d);save_json(p,d);return d,p

def status(stage,state='done',**fields):
    path=P.PUBLIC/'00-progress.research.json'
    d=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'schema':'research_workpackage.v1','kind':'real','run_id':P.RUN,'title':'无歧义属性 · 双范围仲裁、原生MLP路径与未见组合预测','stages':{}}
    d['stages'][stage]={'state':state,'time':time.time(),**fields};d['updated_at']=time.time();P.save(path,d)
    Library(P.ROOT/'distributed_data/research_library').ingest(d,str(path.relative_to(P.ROOT)),kind='local_report');print(json.dumps({'stage':stage,'state':state,**fields},ensure_ascii=False),flush=True)

def install(model,group,alpha,kind='mlp'):
    handles=[]
    for l in sorted({x[0] for x in group}):
        ids=[i for a,i in group if a==l]
        def hook(_m,args,ids=ids):
            x=args[0].clone()
            if kind=='mlp':x[:,-1,ids]*=1-alpha
            else:x.view(*x.shape[:-1],32,128)[:,-1,ids,:]*=1-alpha
            return (x,)
        mod=model.model.layers[l].mlp.down_proj if kind=='mlp' else model.model.layers[l].self_attn.o_proj
        handles.append(mod.register_forward_pre_hook(hook))
    return handles

def main():
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    c=json.loads((P.OUT/'formal-corpus.json').read_text(encoding='utf-8'));pr=json.loads((P.PUBLIC/'prereg.research.json').read_text(encoding='utf-8'))
    files=['tests/glm5/phase2794_materials.py','tests/glm5/phase2794_run.py','tests/glm5/phase2794_analyze.py','shared/dist_protocol/mechanism_trace.py']
    freeze={'run_id':P.RUN,'corpus_sha256':digest(c),'prereg_sha256':digest(pr),'model_snapshot':c['model_snapshot'],
        'dependencies':{n:file_sha(P.ROOT/n) for n in files}}
    P.save(P.OUT/'execution-freeze.json',freeze,True);store=TraceStore()
    prov={'git_sha':subprocess.check_output(['git','-c',f'safe.directory={P.ROOT.as_posix()}','rev-parse','HEAD'],text=True).strip(),
        'git_dirty':True,'model_path':str(P.MODEL),'model_revision':c['model_snapshot'],'seed':P.SEED,'execution_freeze_sha256':digest(freeze)}
    with GpuLease():
        assert torch.cuda.is_available() and snapshot(P.MODEL)==c['model_snapshot'];torch.set_num_threads(4);torch.manual_seed(P.SEED);started=time.time()
        tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True)
        m=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
        def fwd(ids,**kw):
            with torch.inference_mode():return m(input_ids=torch.tensor([ids],device='cuda'),logits_to_keep=1,**kw)
        def complete(out,ids,folder=None,item=None,parent=None):
            past=out.past_key_values;logit=out.logits[0,-1];generated=[];traces=[]
            for step in range(4):
                token=int(logit.argmax());generated.append(token)
                if token==tok.eos_token_id or step==3:break
                if folder is None:o=fwd([token],past_key_values=past,use_cache=True)
                else:
                    gi={**item,'id':item['id']+f':G{step+1}','token_ids':ids+generated,'prompt':item['prompt']+tok.decode(generated),
                        'generation_step':step+1,'parent_trace_id':parent}
                    with Recorder(m,[0],folder/f'G{step+1}',position_offset=len(ids)+len(generated)-1) as rec:
                        o=fwd([token],past_key_values=past,use_cache=True,output_hidden_states=True)
                    gd,gp=rec.finish(o,gi,tok,c['model_snapshot'],P.RUN,prov);store.register(gp);traces.append(gd['trace_id'])
                past=o.past_key_values;logit=o.logits[0,-1]
            return tok.decode(generated,skip_special_tokens=True).strip(),generated,traces
        def score(o,item):
            lp=o.logits[0,-1].float().log_softmax(-1);expected=c['answer_ids'][item['expected']][0];wrong=c['answer_ids'][item['prior'] if item['scope']=='文档' else item['document']][0]
            return {'first_token_gap':float(lp[expected]-lp[wrong]),'full_top1':tok.decode([int(lp.argmax())]),'logits_bits':o.logits[0,-1].contiguous().view(torch.uint16).cpu().numpy()}
        # Training-only source-free distributions for the graph's factual-memory nodes.
        priors=[];pbits=[]
        for e in c['entities']:
            text=P.RULE+f'\n现在只按现实回答：{e["entity"]}的原子序数是多少？'
            prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False);ids=tok.encode(prompt,add_special_tokens=False)
            o=fwd(ids,use_cache=True);pbits.append(o.logits[0,-1].contiguous().view(torch.uint16).cpu().numpy());actual,tokens,_=complete(o,ids)
            priors.append({**e,'text':text,'token_ids':ids,'generated':actual,'generated_token_ids':tokens,'correct':actual==e['prior']});del o
        np.save(P.OUT/'source-free-logits.npy',np.stack(pbits));P.save(P.OUT/'source-free-priors.json',{'rows':priors,'provenance':prov},True)
        rows=[];allbits=[];status('collection','running',done=0,total=len(c['items']))
        for index,item in enumerate(c['items']):
            folder=P.OUT/'native'/digest(item['id'])[:24];deep=item['id'] in pr['deep_ids']
            with Recorder(m,item['readout_positions'],folder,deep=deep) as rec:o=fwd(item['token_ids'],use_cache=True,output_hidden_states=True)
            d,path=rec.finish(o,item,tok,c['model_snapshot'],P.RUN,prov);store.register(path)
            assert max(x['attention_relative_l2'] for x in d['reconstruction'].values())<.05
            s=score(o,item);allbits.append(s.pop('logits_bits'));actual,tokens,gids=complete(o,item['token_ids'],folder if deep else None,item,d['trace_id'])
            row={k:item[k] for k in ['id','family_id','entity','partner','split','frame','length','scope','prior','document','expected']}
            row.update(s,generated=actual,generated_token_ids=tokens,correct=actual==item['expected'],trace_id=d['trace_id'],generation_trace_ids=gids)
            rows.append(row);P.save(P.OUT/'checkpoints'/(digest(item['id'])+'.json'),row)
            del o,rec;gc.collect()
            if (index+1)%32==0:status('collection','running',done=index+1,total=len(c['items']))
        np.save(P.OUT/'next-logits.npy',np.stack(allbits));P.save(P.OUT/'observations.json',{'kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'rows':rows,'provenance':prov},True)
        # Discovery-only direct output contribution; no future heldout state enters selection.
        sums=np.zeros((36,9728));energies=np.zeros_like(sums);heads=np.zeros((36,32));count=0
        for row in rows:
            if row['split']!='discovery':continue
            d,folder=store.manifest(row['trace_id']);ri=len(d['readout_positions'])-1;h=unbits(np.asarray(d['final_pre_bits'],np.uint16)[ri]);rms=np.sqrt(np.mean(h*h)+m.config.rms_norm_eps)
            other=row['prior'] if row['scope']=='文档' else row['document'];u=(m.lm_head.weight[c['answer_ids'][row['expected']][0]].float()-m.lm_head.weight[c['answer_ids'][other][0]].float())*m.model.norm.weight.float()/rms
            for l,layer in enumerate(m.model.layers):
                with np.load(folder/f'block_{l:02d}.npz') as z:a=unbits(z['activation'][ri]);hz=unbits(z['head_z'][ri])
                sums[l]+=a*(u@layer.mlp.down_proj.weight.float()).cpu().numpy();energies[l]+=a*a
                heads[l]+=(hz*(u@layer.self_attn.o_proj.weight.float()).reshape(32,128).cpu().numpy()).sum(-1)
            count+=1
        mean=sums/count;energy=np.sqrt(energies/count);flat=np.argsort(-mean[18:].reshape(-1),kind='stable')[:32]
        units=[(int(x//9728+18),int(x%9728)) for x in flat];hf=np.argsort(-heads[18:].reshape(-1),kind='stable')[:8];hg=[(int(x//32+18),int(x%32)) for x in hf]
        rng=np.random.default_rng(P.SEED);random=[]
        for l,u in units:
            forbidden={v for a,v in units+random if a==l};candidates=[v for v in range(9728) if v not in forbidden]
            candidates=sorted(candidates,key=lambda v:abs(np.log((energy[l,v]+1e-5)/(energy[l,u]+1e-5))))[:32]
            random.append((l,int(rng.choice(candidates))))
        rh=[]
        for l,h in hg:rh.append((l,int(rng.choice([v for v in range(32) if (l,v) not in hg+rh]))))
        selection={'kind':'real','run_id':P.RUN,'units':units,'random_units':random,'heads':hg,'random_heads':rh,
            'random_unit_rms_ratios':[float(energy[l,v]/max(energy[l,u],1e-8)) for (l,u),(_,v) in zip(units,random)],
            'selected_only':'discovery final-query direct first-token contributions; not a scope-specific or causal neuron label',
            'profile_block':max(set(l for l,u in units),key=lambda l:sum(a==l for a,u in units)),'provenance':prov}
        P.save(P.OUT/'selection.json',selection,True);status('discovery',units=units,heads=hg)
        cases=[x for x in c['items'] if x['split']=='unseen_combination' and x['frame']==2]
        conditions=[('mlp',a,units) for a in [.25,.5,1.]]+[('random_mlp',1.,random),('heads',1.,hg),('random_heads',1.,rh)]
        interventions=[]
        for kind,alpha,group in conditions:
            hs=install(m,group,alpha,'head' if 'heads' in kind else 'mlp')
            try:
                for item in cases:
                    o=fwd(item['token_ids'],use_cache=True);s=score(o,item);s.pop('logits_bits');actual,tokens,_=complete(o,item['token_ids']);del o
                    interventions.append({'sample_id':item['id'],'family_id':item['family_id'],'scope':item['scope'],'condition':kind,'alpha':alpha,'generated':actual,'correct':actual==item['expected'],**s})
            finally:
                for h in hs:h.remove()
            status('intervention','running',condition=kind,alpha=alpha,rows=len(interventions))
        P.save(P.OUT/'interventions.json',{'rows':interventions,'selection':selection,'provenance':prov},True)
        assert snapshot(P.MODEL)==c['model_snapshot']
        P.save(P.OUT/'execution.json',{'kind':'real','run_id':P.RUN,'seconds':time.time()-started,'snapshot_unchanged':True,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'gpu':torch.cuda.get_device_name(),'provenance':prov},True)
        status('collection',samples=len(rows),physical_nodes=1);status('intervention',rows=len(interventions))
if __name__=='__main__':main()
