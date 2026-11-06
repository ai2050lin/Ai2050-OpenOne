"""Prospective joint doses and corrected finite-sample paired intervals."""
import sys,time,json,argparse
from pathlib import Path
import numpy as np
from scipy.stats import beta
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_materials_v4 as P
from native_mlp_function_run_v4 import status,Recorder
from native_mlp_long_validate import install
from shared.dist_protocol.mechanism_trace import GpuLease,digest,file_sha,unbits,TraceStore
from shared.dist_protocol.natural_register_runner import snapshot
from tools.power_analysis import check_phase

def paired_bounds(base,edited,alpha=.05):
    b=np.asarray(base,bool);e=np.asarray(edited,bool);n=len(b);loss=int(np.sum(b&~e));gain=int(np.sum(~b&e));q=alpha/4
    def bounds(k):return [float(beta.ppf(q,k,n-k+1)) if k else 0.,float(beta.ppf(1-q,k+1,n-k)) if k<n else 1.]
    ll,lh=bounds(loss);gl,gh=bounds(gain)
    return {'n':n,'mean':float(b.mean()-e.mean()),'ci95':[ll-gh,lh-gl],'confidence':1-alpha,'losses':loss,'gains':gain,'method':'four-tail Bonferroni Clopper-Pearson, guaranteed two-sided coverage >=1-alpha'}
def groups():
    original=P.read(P.OUT/'selection.json');new=P.read(P.OUT/'coordinate-selection.json');units=sorted({(r['layer'],r['unit']) for r in new['selected']});rng=np.random.default_rng(P.SEED+10);control=[];store=TraceStore();c=P.read(P.OUT/'corpus.json');obs=P.read(P.OUT/'observations.json');by={r['id']:r for r in obs['rows']}
    from server.mechanism_trace_service import parameter
    for l,u in units:
        energy=np.zeros(9728);n=0
        for item in c['items']:
            if item['split']!='discovery':continue
            z=store.block(by[item['id']]['trace_id'],l);energy+=unbits(z['activation'][-1])**2;n+=1
        energy=np.sqrt(energy/n);w=parameter(P.MODEL,f'model.layers.{l}.mlp.down_proj.weight',(slice(None),slice(None)));norm=np.linalg.norm(w,axis=0);product=energy*norm
        candidates=[v for v in range(9728) if (l,v) not in units and [l,v] not in control and .8<=energy[v]/max(energy[u],1e-10)<=1.25 and .8<=norm[v]/max(norm[u],1e-10)<=1.25 and .8<=product[v]/max(product[u],1e-10)<=1.25];assert candidates,(l,u)
        control.append([l,int(rng.choice(candidates))])
    d={'kind':'real','run_id':P.RUN,'groups':{'native':{'units':original['units'],'random_units':original['random_units']},'coordinate':{'units':[list(u) for u in units],'random_units':control}},'provenance':original['provenance']}
    P.save(P.OUT/'dose-groups.json',d);return d

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['dose','capability','audit'],required=True);part=parser.parse_args().part
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    g=P.read(P.OUT/'dose-groups.json') if (P.OUT/'dose-groups.json').exists() else groups();prov=g['provenance'];c=P.read(P.OUT/'corpus.json')
    contract={'kind':'real','run_id':P.RUN,'part':part,'script_sha256':file_sha(__file__),'groups_sha256':file_sha(P.OUT/'dose-groups.json'),'source_amendment_sha256':file_sha(P.PUBLIC/'prospective-amendment.research.json'),
        'dose_family_budget':'four contrasts (two group losses,two target-random differences), alpha=.0125 each; all4 simultaneous >=95%',
        'dose_power':check_phase(.1,256,p_d=.2,alpha=.0125),'confidence_note':'Four-tail CP fixes earlier helper mislabelling: two separate95% marginal intervals alone do not guarantee two-sided95% for their difference. NI upper bound remains conservative.',
        'coordinate_group':'discovery-only coordinates, frozen before heldout fitting/effects, joint query operation; not individual or original checkpoint necessity',
        'capability_margin':.02,'capability_scope':'previously exposed2200formal/140reference, each operation separately; no global-domain guarantee','provenance':prov}
    assert contract['dose_power']['pass'];P.save(P.OUT/f'{part}-contract.json',contract)
    with GpuLease():
        assert snapshot(P.MODEL)==prov['model_revision'];torch.set_num_threads(4);started=time.time();tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);tok.padding_side='left'
        m=AutoModelForCausalLM.from_pretrained(P.MODEL,local_files_only=True,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval().requires_grad_(False)
        def fwd(ids,**kw):
            with torch.inference_mode():return m(input_ids=torch.tensor(ids,device='cuda'),use_cache=False,logits_to_keep=1,**kw)
        if part=='dose':
            families=list(P.FAMILIES);items=[r for r in c['items'] if r['split']=='lockbox' and r['family']==families[(r['family_index']-288)%3]];assert len(items)==512 and len({r['background_doc_id'] for r in items})==256
            conditions=[('base',[],0.)]
            for key,group in g['groups'].items():conditions.extend([(key+'_half',group['units'],.5),(key+'_full',group['units'],1.),(key+'_random',group['random_units'],1.)])
            choices={key:[tok.encode(s,add_special_tokens=False)[0] for s in [key,' '+key] if len(tok.encode(s,add_special_tokens=False))==1] for key in ['Yes','No']};rows=[];logits=[];shams=[]
            status('intervention','running',done=0,total=len(items))
            for start in range(0,len(items),4):
                chunk=items[start:start+4];ids=[r['token_ids'] for r in chunk];base=None
                for name,units,alpha in conditions:
                    hs=install(m,units,alpha)
                    try:o=fwd(ids)
                    finally:
                        for h in hs:h.remove()
                    v=o.logits[:,-1];lp=v.float().log_softmax(-1)
                    if base is None:base=lp.clone();basebits=v.detach().clone()
                    logits.extend(v.contiguous().view(torch.uint16).cpu().numpy())
                    for j,item in enumerate(chunk):
                        gap=float(lp[j,choices[item['expected']]].max()-lp[j,choices[item['contrast']]].max())
                        rows.append({'sample_id':item['id'],'pair_id':item['pair_id'],'family':item['family'],'background_doc_id':item['background_doc_id'],'condition':name,'correct':gap>0,'choice_gap':gap,
                            'expected':item['expected'],'contrast':item['contrast'],'KL_from_base':float((base[j].exp()*(base[j]-lp[j])).sum()),'top1':tok.decode([int(lp[j].argmax())])})
                    if name=='base' and start<32:
                        for key,group in g['groups'].items():
                            hs=install(m,group['units'],0.)
                            try:sham=fwd(ids).logits[:,-1]
                            finally:
                                for h in hs:h.remove()
                            shams.append({'start':start,'group':key,'bitwise_equal':bool(torch.equal(basebits,sham))})
                if (start+4)%32==0:status('intervention','running',done=start+4,total=len(items))
            np.save(P.OUT/'dose-logits.npy',np.stack(logits));by={name:{r['sample_id']:r for r in rows if r['condition']==name} for name,_,_ in conditions};pairs=sorted({r['pair_id'] for r in rows})
            pair_correct={name:[all(table[p+':'+str(v)]['correct'] for v in [0,1]) for p in pairs] for name,table in by.items()}
            effects={name:{'accuracy':float(np.mean([r['correct'] for r in table.values()])),'pair_both_correct':float(np.mean(pair_correct[name])),
                'loss':paired_bounds(pair_correct['base'],pair_correct[name],alpha=.0125),'mean_KL':float(np.mean([r['KL_from_base'] for r in table.values()]))} for name,table in by.items()}
            specificity={key:paired_bounds(pair_correct[key+'_random'],pair_correct[key+'_full'],alpha=.0125) for key in g['groups']}
            supported={key:bool(effects[key+'_full']['loss']['mean']>=.1 and effects[key+'_full']['loss']['ci95'][0]>0 and specificity[key]['mean']>=.05 and specificity[key]['ci95'][0]>0 and all(s['bitwise_equal'] for s in shams)) for key in g['groups']}
            result={'kind':'real','run_id':P.RUN,'rows':rows,'effects':effects,'specificity':specificity,'supported_groups':supported,'supported':any(supported.values()),'numeric_sham':shams,'provenance':prov,'seconds':time.time()-started}
            P.save(P.OUT/'dose-result.json',result);status('intervention','done' if any(supported.values()) else 'not_supported',effects=effects,supported_groups=supported)
        elif part=='capability':
            qs=P.read(P.ROOT/'tests/glm5/result/phase2773_w6_v2/capability-inputs.json');letterids=[tok.encode(x,add_special_tokens=False)[0] for x in 'ABCDE'];rows={}
            for name,units in [('base',[])]+[(key,v['units']) for key,v in g['groups'].items()]:
                hs=install(m,units,1.);records=[]
                try:
                    for start in range(0,len(qs),4):
                        chunk=qs[start:start+4];tokens=[]
                        for q in chunk:
                            text=q['question']+'\n'+'\n'.join(f'{s}. {choice}' for s,choice in zip('ABCDE',q['choices']))+'\n请只回答正确选项字母。'
                            tokens.append(tok.encode(tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False))
                        batch={k:v.cuda() for k,v in tok.pad({'input_ids':tokens},padding=True,return_tensors='pt').items()}
                        with torch.inference_mode():o=m(**batch,use_cache=False,logits_to_keep=1)
                        for j,q in enumerate(chunk):
                            scores=o.logits[j,-1,letterids[:len(q['choices'])]].float().cpu();prediction='ABCDE'[int(scores.argmax())]
                            records.append({'id':q['id'],'prediction':prediction,'answer':q['answer'],'correct':prediction==q['answer'],'source':q['source'],'gate_eligible':q['gate_eligible'],'choice_logits':scores.tolist()})
                        if start%400==0:status('capability','running',condition=name,done=start+4,total=len(qs))
                finally:
                    for h in hs:h.remove()
                rows[name]=records;P.save(P.OUT/f'capability-{name}.json',records)
            results={}
            for key in g['groups']:
                mask=np.asarray([r['gate_eligible'] for r in rows['base']]);b=np.asarray([r['correct'] for r in rows['base']])[mask];e=np.asarray([r['correct'] for r in rows[key]])[mask];r=paired_bounds(b,e)
                results[key]={**r,'base_accuracy':float(b.mean()),'edited_accuracy':float(e.mean()),'net_loss':r['mean'],'loss_interval':r['ci95'],'margin':.02,'status':'noninferior' if r['ci95'][1]<=.02 else 'inferior' if r['ci95'][0]>.02 else 'inconclusive'}
            P.save(P.OUT/'capability-result.json',{'kind':'real','run_id':P.RUN,'groups':results,'provenance':prov});status('capability','done',groups=results)
        else:
            store=TraceStore();audits=[]
            # Exact-input audit repeats, no new independent observations.
            for identity in ['conditional:289:0','conditional:289:1']:
                item=next(r for r in c['items'] if r['id']==identity);copy={**item,'id':identity+':extra-deep'}
                with Recorder(m,item['readout_positions'],P.OUT/'extra-native'/digest(copy['id'])[:20],deep=True) as rec:
                    o=fwd([item['token_ids']],output_hidden_states=True)
                d,path=rec.finish(o,copy,tok,c['model_snapshot'],P.RUN,prov);store.register(path);audits.append({'run_id':P.RUN,'sample_id':copy['id'],'origin_sample_id':identity,'trace_id':d['trace_id'],'deep':True})
            # Calibration examples remain available in3D evenwhen their behavior gate failed.
            for version in [1,2,3]:
                folder=P.ROOT/f'tests/glm5/result/native_mlp_function_v{version}';corpus=P.read(folder/'corpus.json');old=P.read(folder/'pilot-result.json');oldby={r['id']:r for r in old['rows']}
                for item in [r for r in corpus['items'] if r['split']=='discovery' and r['family_index'] in [0,4]]:
                    with Recorder(m,item['readout_positions'],folder/'audit-native'/digest(item['id'])[:20],deep=False) as rec:o=fwd([item['token_ids']],output_hidden_states=True)
                    d,path=rec.finish(o,item,tok,c['model_snapshot'],f'NMLP-FUNCTION-20261104-v{version}',prov);store.register(path)
                    choices={x:[tok.encode(s,add_special_tokens=False)[0] for s in [x,' '+x] if len(tok.encode(s,add_special_tokens=False))==1] for x in [item['expected'],item['contrast']]};lp=o.logits[0,-1].float().log_softmax(-1);gap=float(lp[choices[item['expected']]].max()-lp[choices[item['contrast']]].max())
                    audits.append({'version':version,'run_id':f'NMLP-FUNCTION-20261104-v{version}','sample_id':item['id'],'trace_id':d['trace_id'],'gap_replay_difference':gap-oldby[item['id']]['choice_gap'],'deep':False})
            P.save(P.OUT/'audit-traces.json',{'kind':'real','run_id':P.RUN,'rows':audits,'provenance':prov});status('validation',audit_replays=len(audits))
        assert snapshot(P.MODEL)==prov['model_revision']
if __name__=='__main__':main()
