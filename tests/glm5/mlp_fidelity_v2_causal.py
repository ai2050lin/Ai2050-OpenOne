"""Natural loss recovery, generated scoped intervention, and repeated-pool capability NI."""
import sys,time,json,gc,argparse,shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
import mlp_fidelity_v2_tail as T
from mlp_fidelity_v2_collect_v4 import greedy,first_word
from shared.dist_protocol.mechanism_trace import GpuLease,file_sha,digest
from shared.dist_protocol.transcoder import TopKTranscoder
from shared.dist_protocol.natural_register_runner import snapshot

def main(part):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    plan=P.read(P.OUT/'plan.json');pr=P.read(P.PUBLIC/'controlled-v4-prereg.research.json');sem=P.read(P.OUT/'semantics-result-v2.json');layer=next(r for r in sem['layers'] if r['block']==35);eligible=sem['behavior_gate'] and layer['fidelity_gate'];selected=layer['selected_group'] if eligible else None;random=layer['random_group'] if eligible and layer['random_matched'] else None
    output=P.OUT/('tail-'+part);output.mkdir(exist_ok=True);state=P.read(P.OUT/'trained/B35-result.json');cap_path=P.ROOT/'tests/glm5/result/phase2773_w6_v2/capability-inputs.json'
    contract={'kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],'part':part,'source_sha256':file_sha(__file__),'tail_math_sha256':file_sha(T.__file__),'semantic_result_sha256':file_sha(P.OUT/'semantics-result-v2.json'),'dictionary_sha256':state['state_sha256'],'fresh_documents_sha256':file_sha(P.OUT/'fresh-documents.json'),'selected':selected,'random':random,'feature_intervention_eligible':bool(eligible),'capability_input_sha256':file_sha(cap_path),'source_lineage_supplement':'original naturalfreeze omittedfresh-documentfilehash; naturalpartindependentlyreplaysall36MLPinput/outputfields againstcurrentfrozeninputs. Exactbitfractionandmaxrelativeerrorrecorded, abortifrelative>.001; nosourcevalues replaced.','scope':'B35lastMLPoutput atobservedquery/selectednaturalpositions only; originalprefixhistory fixed. Actualhook/cacherecomposition checksrequired. No earlierlayer shortcut.'}
    P.save(output/'execution-freeze.json',contract)
    if part=='generated' and not eligible:
        P.save(output/'result.json',{**contract,'status':'not_eligible','reason':'globalgeneratedbehaviororB35freshfidelitygatefailed; no semanticcausalpromotion'});return
    start=time.time();torch.set_num_threads(4);rows=[];checks=[];ctx={'capture':False};hooks=[];replay={'elements':0,'equal':0,'maximum_relative_error':0.}
    with GpuLease():
        assert snapshot(P.MODEL)==plan['model_snapshot'];tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);tok.padding_side='left';m=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
        with np.load(P.ROOT/state['state_path']) as z:dictionary=TopKTranscoder.from_state(z,device='cuda')
        with np.load(P.OUT/'development/B35-tc4096.npz') as z:tc=TopKTranscoder.from_state(z,device='cuda')
        eos=m.generation_config.eos_token_id;eos={eos} if isinstance(eos,int) else set(eos)
        def take(x):return x[:,ctx['positions']].reshape(-1,x.shape[-1]).detach().clone()
        def capr(_m,args):
            if ctx['capture']:ctx['r']=take(args[0])
        def capx(_m,args):
            if ctx['capture']:ctx['x']=take(args[0])
        def capa(_m,args):
            if ctx['capture']:ctx['a']=take(args[0])
        def capy(_m,args,o):
            if ctx['capture']:ctx['y']=take(o)
        block=m.model.layers[35];hooks=[block.post_attention_layernorm.register_forward_pre_hook(capr),block.mlp.register_forward_pre_hook(capx),block.mlp.down_proj.register_forward_pre_hook(capa),block.mlp.register_forward_hook(capy)]
        if part=='natural':
            items=P.read(P.OUT/'fresh-documents.json')['natural'];positions=P.read(P.OUT/'natural-bank/manifest.json')['positions'];n_positions=len(positions)
            source_fields={(b,k):np.load(P.OUT/f'natural-bank/B{b:02d}-{k}.npy',mmap_mode='r') for b in range(36) for k in ['input','output']}
            def compare(b,k,value):
                if not ctx['capture']:return
                actual=value[:,positions].detach().contiguous();reference=torch.from_numpy(np.asarray(source_fields[b,k][ctx['document_start']:ctx['document_start']+len(value)]).copy()).view(torch.bfloat16).to('cuda');replay['elements']+=actual.numel();replay['equal']+=int((actual==reference).sum());err=float((actual.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-8));replay['maximum_relative_error']=max(replay['maximum_relative_error'],err);assert err<=.001,('natural source replay mismatch',b,k,ctx['document_start'],err)
            for b,l in enumerate(m.model.layers):hooks.extend([l.mlp.register_forward_pre_hook(lambda _m,args,b=b:compare(b,'input',args[0])),l.mlp.register_forward_hook(lambda _m,args,o,b=b:compare(b,'output',o))])
        elif part=='generated':
            corpus=P.read(P.OUT/'controlled-v4-corpus.json');lookup={(r['background_index'],r['mapping'],r['fact'],r['question']):r for r in corpus['items']};items=[]
            for b in range(320,512):
                h=int(digest({'seed':P.SEED,'b':b})[:8],16);r=(h>>1)&1;t=h&1
                items +=[lookup[b,r,t,q] for q in [0,1]]
            positions=[-1];n_positions=1
        else:items=P.read(cap_path);assert sum(r['gate_eligible'] for r in items)==2200;positions=[-1];n_positions=1
        shapes={k:(len(items),n_positions,d) for k,d in [('residual',2560),('output',2560),('activation',9728),('final-hidden',2560)]};assert shutil.disk_usage(output).free>sum(np.prod(s)*2 for s in shapes.values())+4*2**30
        native={k:np.lib.format.open_memmap(output/(k+'.npy'),mode='w+',dtype=np.uint16,shape=s) for k,s in shapes.items()}
        P.status('tail_'+part,'running',done=0,total=len(items),feature_intervention_eligible=bool(eligible))
        for lo in range(0,len(items),4):
            chunk=items[lo:lo+4]
            if part=='natural':ids=torch.tensor([r['tokens'][:1024] for r in chunk],device='cuda');batch={'input_ids':ids};target=torch.tensor([r['tokens'][p+1] for r in chunk for p in positions],device='cuda');keep=torch.tensor(positions,device='cuda')
            elif part=='generated':ids=torch.tensor([r['token_ids'] for r in chunk],device='cuda');batch={'input_ids':ids};target=torch.tensor([tok.encode(r['expected'],add_special_tokens=False)[0] for r in chunk],device='cuda');keep=1
            else:
                prompts=[]
                for r in chunk:
                    text=r['question']+'\n'+'\n'.join(f'{a}. {b}' for a,b in zip('ABCDE',r['choices']))+'\n请只回答正确选项字母。';prompts.append(tok.apply_chat_template([{'role':'user','content':text}],tokenize=True,add_generation_prompt=True,enable_thinking=False))
                batch={k:v.to('cuda') for k,v in tok.pad({'input_ids':prompts},padding=True,return_tensors='pt').items()};target=torch.tensor([tok.encode(r['answer'],add_special_tokens=False)[0] for r in chunk],device='cuda');keep=1
            ctx.update(capture=True,positions=positions,document_start=lo)
            with torch.inference_mode():o=m(**batch,use_cache=True,logits_to_keep=keep)
            ctx['capture']=False;r,y,a,x=ctx['r'],ctx['y'],ctx['a'],ctx['x'];clean_logits=o.logits.reshape(-1,o.logits.shape[-1]);cache=o.past_key_values;prefix_len=batch['input_ids'].shape[1]
            with torch.inference_mode():ops,z=T.operations(dictionary,y,a,block.mlp.down_proj.weight,selected if part!='natural' else None,random if part!='natural' else None,tc=tc,x=x);recon=T.tail_logits(m,r,y)
            assert torch.equal(recon,clean_logits),'Native tailrecomposition mustexactlymatch fullmodel'
            for name,value in [('residual',r),('output',y),('activation',a),('final-hidden',m.model.norm((r+y)[:,None])[:,0])]:native[name][lo:lo+len(chunk)]=value.contiguous().view(torch.uint16).cpu().numpy().reshape(len(chunk),*shapes[name][1:])
            if lo==0:
                # Verify real hook operation and prefix cache before using the tail shortcut.
                replacement=ops['sae_full'].to(y.dtype).reshape(len(chunk),n_positions,2560)
                def inject(_m,args,out):
                    changed=out.clone();changed[:,positions]=replacement;return changed
                h=block.mlp.register_forward_hook(inject)
                with torch.inference_mode():actual=m(**batch,use_cache=True,logits_to_keep=keep)
                h.remove()
                with torch.inference_mode():expected=T.tail_logits(m,r,ops['sae_full'])
                equal=torch.equal(actual.logits.reshape_as(expected),expected);keys_equal=all(torch.equal(v.keys,w.keys) and torch.equal(v.values,w.values) for v,w in zip(cache.layers,actual.past_key_values.layers));assert equal and keys_equal
                checks.append({'tail_native_logits_bitwise_equal':True,'true_hook_vs_tail_logits_bitwise_equal':equal,'all36prefix_KV_bitwise_equal':keys_equal});del actual
            outcomes={};metrics={}
            for name,yy in ops.items():
                with torch.inference_mode():ll=clean_logits if name=='clean' else T.tail_logits(m,r,yy);metrics[name]=T.metric(ll,clean_logits,target)
                if part=='generated':
                    cache.crop(prefix_len);tokens=greedy(m,cache,ll,eos);outcomes[name]=[tok.decode(t,skip_special_tokens=True) for t in tokens]
                elif part=='capability':
                    predictions=[]
                    for j,q in enumerate(chunk):
                        ids_choices=[tok.encode(ch,add_special_tokens=False)[0] for ch in 'ABCDE'[:len(q['choices'])]];pred='ABCDE'[int(ll[j,ids_choices].argmax())];predictions.append(pred)
                    outcomes[name]=predictions
            for j,item in enumerate(chunk):
                record={'index':lo+j,'id':item.get('id',item.get('background_index',str(lo+j))),'split':item.get('split'),'background_index':item.get('background_index'),'bank_index':item['background_index']*8+item['mapping']*4+item['fact']*2+item['question'] if part=='generated' else None,'question':item.get('question') if part=='generated' else None,'language':item.get('language'),'source':item.get('source'),'expected':item.get('expected',item.get('answer')),'gate_eligible':item.get('gate_eligible'),'conditions':{}}
                for name,mt in metrics.items():
                    rr={k:float(np.mean(v[j*n_positions:(j+1)*n_positions])) for k,v in mt.items() if k!='argmax'}
                    if part=='generated':text=outcomes[name][j];rr.update(generated=text,first_word=first_word(text),correct=first_word(text)==item['expected'])
                    elif part=='capability':rr.update(prediction=outcomes[name][j],correct=outcomes[name][j]==item['answer'])
                    record['conditions'][name]=rr
                rows.append(record)
            del o,clean_logits,cache,ops,z,metrics,outcomes,r,y,a,x;gc.collect()
            if (lo+len(chunk))%32==0:P.status('tail_'+part,'running',done=lo+len(chunk),total=len(items),seconds=round(time.time()-start,1))
        for h in hooks:h.remove()
        assert snapshot(P.MODEL)==plan['model_snapshot']
    summary={}
    names=list(rows[0]['conditions'])
    for a in native.values():a.flush()
    native_files=[{'path':k+'.npy','shape':list(s),'bytes':(output/(k+'.npy')).stat().st_size,'sha256':file_sha(output/(k+'.npy'))} for k,s in shapes.items()]
    if part=='natural':
        summary['operations']={}
        for name in names:
            ce=np.array([r['conditions'][name]['CE'] for r in rows]);clean=np.array([r['conditions']['clean']['CE'] for r in rows]);zero=np.array([r['conditions']['zero']['CE'] for r in rows]);den=float(zero.mean()-clean.mean());summary['operations'][name]={'CE':float(ce.mean()),'CE_difference':T.cluster_interval(ce-clean,P.SEED),'KL_clean':float(np.mean([r['conditions'][name]['KL_clean'] for r in rows])),'loss_recovered':float((zero.mean()-ce.mean())/den) if den>0 else None}
    elif part=='capability':
        eligible_mask=np.asarray([r['gate_eligible'] for r in rows],bool);base=[r['conditions']['clean']['correct'] for r in rows];summary['NI']={name:T.capability_interval(base,[r['conditions'][name]['correct'] for r in rows],eligible_mask) for name in ['sae_full','selected_1'] if name in names};summary['reference_only_n']=int((~eligible_mask).sum())
    else:
        summary['dose_curve']=[{'condition':name,'alpha':.5 if name=='selected_0.5' else 1. if name=='selected_1' else 0. if name=='clean' else None,'accuracy':float(np.mean([r['conditions'][name]['correct'] for r in rows])),'CE':float(np.mean([r['conditions'][name]['CE'] for r in rows])),'KL_clean':float(np.mean([r['conditions'][name]['KL_clean'] for r in rows]))} for name in names]
        by_background={b:[r for r in rows if r['background_index']==b] for b in range(320,512)};loss=np.array([np.mean([r['conditions']['clean']['correct']-r['conditions']['selected_1']['correct'] for r in rr]) for rr in by_background.values()]);summary['selected_loss']=T.cluster_interval(loss,P.SEED)
        if 'random_1' in names:
            specificity=np.array([np.mean([r['conditions']['random_1']['correct']-r['conditions']['selected_1']['correct'] for r in rr]) for rr in by_background.values()]);summary['target_minus_random_loss']=T.cluster_interval(specificity,P.SEED+1);summary['effect_supported']=bool(loss.mean()>=.2 and summary['selected_loss']['ci95'][0]>0 and specificity.mean()>=.05 and summary['target_minus_random_loss']['ci95'][0]>0)
    result={**contract,'schema':'mlp_fidelity_tail_result.v1','status':'complete','checks':checks,'source_replay':replay if part=='natural' else None,'rows':rows,'summary':summary,'native_files':native_files,'seconds':time.time()-start,'scope_note':'B35tail causaloperation atobservedpositions; lossrecovery isnot fullgenerativehistory reconstruction. Selectedfeatureremovalretainsoriginalreconstructionerror. No unique semanticmechanismclaim.'};P.save(output/'result.json',result);P.status('tail_'+part,'done',seconds=result['seconds'],summary=summary)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('part',choices=['natural','generated','capability']);main(p.parse_args().part)
