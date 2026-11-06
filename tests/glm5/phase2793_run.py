"""Real CUDA entity arbitration; native trace capture and preregistered head test."""
import copy
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import phase2793_materials as P
from shared.dist_protocol.mechanism_trace import Recorder,TraceStore,GpuLease,digest,file_sha,unbits
from shared.dist_protocol.natural_register_runner import snapshot

ROOT=P.ROOT;OUT=P.OUT;PUBLIC=P.PUBLIC


def status(stage,state='done',**kw):
    from deploy.research_library import Library
    p=PUBLIC/'00-progress.research.json';d=json.loads(p.read_text(encoding='utf-8')) if p.exists() else {
        'schema':'research_workpackage.v1','kind':'real','run_id':P.RUN,'title':'实体属性仲裁 · 长文本三模块真实计算图谱','stages':{}}
    d['stages'][stage]={'state':state,'time':time.time(),**kw};d['updated_at']=time.time()
    P.save(p,d);Library(ROOT/'distributed_data/research_library').ingest(d,str(p.relative_to(ROOT)),kind='local_report')
    print(json.dumps({'stage':stage,'state':state,**kw},ensure_ascii=False),flush=True)


def main():
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    c=json.loads((OUT/'corpus.json').read_text(encoding='utf-8'));pr=json.loads((PUBLIC/'prereg.research.json').read_text(encoding='utf-8'))
    code=[Path(__file__),ROOT/'shared/dist_protocol/mechanism_trace.py',Path(P.__file__)]
    freeze={'run_id':P.RUN,'corpus_sha256':digest(c),'prereg_file_sha256':file_sha(PUBLIC/'prereg.research.json'),
        'dependencies':{str(p.relative_to(ROOT)):file_sha(p) for p in code},'seed':P.SEED,'model_snapshot':c['model_snapshot']}
    P.save(OUT/'execution-freeze.json',freeze,True)
    prov={'git_sha':subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],text=True).strip(),
        'git_dirty':True,'model_path':str(P.MODEL),'model_revision':c['model_snapshot'],'seed':P.SEED}
    status('freeze',inputs=576,discovery_entities=16,heldout_entities=32,lengths=[512,1024,2048],prereg_sha256=freeze['prereg_file_sha256'])
    with GpuLease():
        if snapshot(P.MODEL)!=c['model_snapshot']:raise ValueError('model snapshot changed')
        if shutil.disk_usage(OUT).free<45*2**30:raise RuntimeError('less than45GiB available for planned native capture')
        torch.manual_seed(P.SEED);torch.set_num_threads(4);started=time.time()
        tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True)
        model=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval()
        model.requires_grad_(False)
        def fwd(ids,**kw):
            with torch.inference_mode():return model(input_ids=torch.tensor([ids],device='cuda'),logits_to_keep=1,**kw)
        colors=list(c['color_token_ids']);colorids=torch.tensor([c['color_token_ids'][s] for s in colors],device='cuda')
        def assessment(logits,item):
            lp=logits.float().log_softmax(-1);v=lp[colorids];choice=colors[int(v.argmax())]
            return {'palette_choice':choice,'correct':choice==item['expected'],'document_choice':choice==item['document'],
                'two_choice_document':bool(lp[c['color_token_ids'][item['document']]]>lp[c['color_token_ids'][item['prior']]]),
                'document_minus_prior':float(lp[c['color_token_ids'][item['document']]]-lp[c['color_token_ids'][item['prior']]]),
                'palette_logp':dict(zip(colors,v.cpu().tolist())),'full_top1_token':int(lp.argmax())}
        store=TraceStore();native=OUT/'native';priors=[]
        for e in c['entities']:
            prompt=tok.apply_chat_template([{'role':'user','content':f'只依据现实常识：{e["zh"]}通常是什么颜色？只回答一个颜色字（红橙黄绿蓝紫黑白灰棕粉之一），不写解释。'}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            ids=tok.encode(prompt,add_special_tokens=False);o=fwd(ids,use_cache=False)
            priors.append({'family_id':e['id'],'entity':e['zh'],'reference':e['prior'],'prompt':prompt,**assessment(o.logits[0,-1],{'expected':e['prior'],'prior':e['prior'],'document':e['document']})});del o
        P.save(OUT/'source-free-priors.json',{'kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'rows':priors,'provenance':prov},True)
        # Per-entity readout vectors through O: exact parameters, final RMS frozen per sample below.
        entity_index={e['id']:i for i,e in enumerate(c['entities'])};u=torch.stack([(model.lm_head.weight[c['color_token_ids'][e['document']]].float()-model.lm_head.weight[c['color_token_ids'][e['prior']]].float())*model.model.norm.weight.float() for e in c['entities']])
        read_heads={l:(u@layer.self_attn.o_proj.weight.float()).reshape(len(u),32,128).cpu().numpy() for l,layer in enumerate(model.model.layers)}
        del u
        # Instrumentation must leave logits bitwise unchanged, before formal collection.
        calibration=c['items'][0];checkids=calibration['token_ids'];original=fwd(checkids,use_cache=False).logits.detach().clone()
        with Recorder(model,calibration['readout_positions'],OUT/'instrument-calibration') as rec:
            o=fwd(checkids,output_hidden_states=True,use_cache=False)
        assert torch.equal(original,o.logits),'capture changed actual logits'
        errors=list(rec.reconstruction.values());assert max(x['attention_relative_l2'] for x in errors)<.05
        P.save(OUT/'instrument-audit.json',{'kind':'real','run_id':P.RUN,'unchanged_logits_bitwise':True,
            'max_attention_relative_l2':max(x['attention_relative_l2'] for x in errors),'frozen_tolerance':.05,'provenance':prov},True)
        del o,original,rec;gc.collect();torch.cuda.empty_cache()
        rows=[];generations=[];status('collection','running',done=0,total=len(c['items']))
        for index,item in enumerate(c['items']):
            checkpoint=OUT/'checkpoints'/(digest(item['id'])+'.json')
            if checkpoint.exists():
                record=json.loads(checkpoint.read_text(encoding='utf-8'))
                if record['execution_freeze_sha256']!=digest(freeze):raise ValueError('checkpoint code identity differs')
                rows.append(record['row']);generations.extend(record.get('generations',[]));continue
            folder=native/digest(item['id'])[:24];deep=item['id'] in pr['deep_ids']
            if tok.encode(item['prompt'],add_special_tokens=False)!=item['token_ids']:raise ValueError('input tokenizer differs')
            with Recorder(model,item['readout_positions'],folder,deep=deep) as rec:
                o=fwd(item['token_ids'],output_hidden_states=True,use_cache=deep)
            manifest,path=rec.finish(o,item,tok,c['model_snapshot'],P.RUN,prov);store.register(path)
            assert max(x['attention_relative_l2'] for x in manifest['reconstruction'].values())<.05
            row={k:item[k] for k in ['id','family_id','entity','scope','split','frame','length','prior','document','expected']}
            row.update(assessment(o.logits[0,-1],item),trace_id=manifest['trace_id'])
            ri=len(item['readout_positions'])-1;h=unbits(np.array(manifest['final_pre_bits'],dtype=np.uint16)[ri]);rms=float(np.sqrt(np.mean(h*h)+model.config.rms_norm_eps))
            effects=[]
            for l in range(36):
                with np.load(folder/f'block_{l:02d}.npz') as z:hz=unbits(z['head_z'][ri])
                effects.append((hz*read_heads[l][entity_index[item['family_id']]]).sum(-1)/rms)
            row['head_dla']=np.stack(effects).tolist();local_g=[]
            if deep:
                past=o.past_key_values;nextlogits=o.logits[0,-1];generated=[];trace_ids=[manifest['trace_id']]
                for step in range(24):
                    token=int(nextlogits.argmax());generated.append(token)
                    if token==tok.eos_token_id:break
                    if step==23:break
                    genitem={**item,'id':item['id']+f':G{step+1}','token_ids':item['token_ids']+generated,
                        'generation_step':step+1,'parent_trace_id':manifest['trace_id'],'prompt':item['prompt']+tok.decode(generated,skip_special_tokens=False)}
                    offset=len(genitem['token_ids'])-1;gfolder=folder/f'G{step+1:02d}'
                    with Recorder(model,[0],gfolder,position_offset=offset) as gr:
                        go=fwd([token],output_hidden_states=True,use_cache=True,past_key_values=past)
                    gd,gpath=gr.finish(go,genitem,tok,c['model_snapshot'],P.RUN,prov);store.register(gpath)
                    trace_ids.append(gd['trace_id']);past=go.past_key_values;nextlogits=go.logits[0,-1];del go,gr
                local_g=[{'sample_id':item['id'],'token_ids':generated,'text':tok.decode(generated,skip_special_tokens=True),'trace_ids':trace_ids,
                    'hit_limit':len(generated)==24 and generated[-1]!=tok.eos_token_id}];generations.extend(local_g);del past
            rows.append(row);P.save(checkpoint,{'execution_freeze_sha256':digest(freeze),'row':row,'generations':local_g})
            del o,rec;gc.collect()
            if (index+1)%12==0:status('collection','running',done=index+1,total=len(c['items']),seconds=round(time.time()-started,1))
        P.save(OUT/'observations.json',{'schema':'entity_arbitration_observations.v1','kind':'real','run_id':P.RUN,
            'model_snapshot':c['model_snapshot'],'rows':rows,'generations':generations,'provenance':prov},True)
        status('collection',samples=len(rows),deep_generations=len(generations))
        discovery=[r for r in rows if r['split']=='discovery'];diff=[]
        for e in c['entities'][:16]:
            doc=np.mean([r['head_dla'] for r in discovery if r['family_id']==e['id'] and r['scope']=='文档'],0)
            world=np.mean([r['head_dla'] for r in discovery if r['family_id']==e['id'] and r['scope']=='现实'],0);diff.append(doc-world)
        mean=np.mean(diff,0);flat=np.argsort(-mean[18:].reshape(-1),kind='stable')[:8];heads=[(int(i//32+18),int(i%32)) for i in flat]
        rng=np.random.default_rng(P.SEED);random_heads=[]
        for l in sorted({l for l,h in heads}):
            count=sum(a==l for a,b in heads);allowed=[h for h in range(32) if (l,h) not in heads]
            random_heads.extend((l,int(h)) for h in rng.choice(allowed,count,replace=False))
        selection={'kind':'real','run_id':P.RUN,'heads':heads,'matched_random_heads':random_heads,'selection_only':'discovery16',
            'criterion':'document minus reality direct logit attribution, blocks18..35','provenance':prov};P.save(OUT/'head-selection.json',selection,True)
        selected=[i for i in c['items'] if i['split']=='heldout_entity_frame' and i['frame']==2 and i['length']==1024 and i['scope']=='文档']
        interventions=[]
        for condition,group in [('selected_heads',heads),('matched_random_heads',random_heads)]:
            handles=[]
            for l in sorted({l for l,h in group}):
                picked=[h for a,h in group if a==l]
                def ablate(_m,args,picked=picked):
                    x=args[0].clone();z=x.view(*x.shape[:-1],32,128);z[:,-1,picked,:]=0;return (x,)
                handles.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(ablate))
            try:
                for item in selected:
                    o=fwd(item['token_ids'],use_cache=False);interventions.append({'sample_id':item['id'],'family_id':item['family_id'],
                        'condition':condition,**assessment(o.logits[0,-1],item)});del o
            finally:
                for h in handles:h.remove()
        P.save(OUT/'head-interventions.json',{'kind':'real','run_id':P.RUN,'rows':interventions,'selection':selection,'provenance':prov},True)
        assert snapshot(P.MODEL)==c['model_snapshot']
        P.save(OUT/'execution.json',{'kind':'real','run_id':P.RUN,'model_snapshot':c['model_snapshot'],'snapshot_unchanged':True,
            'seconds':time.time()-started,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'env':{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'precision':'bf16','attention':'sdpa'},
            'provenance':prov},True)
        status('intervention',heldout_entities=len(selected),selected_heads=heads,physical_nodes=1)


if __name__=='__main__':main()
