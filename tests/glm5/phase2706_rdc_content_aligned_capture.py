"""Serial native-model capture aligned to first lexical-content emission, not raw token index."""
import argparse,gc,shutil,sys
from rdc_conditional_common import *
from phase2702_rdc_serial_scale import load as load_large,Capture as ScaleCapture,MODELS
MODELS=dict(MODELS,qwen4='qwen3-4b')


class Capture(ScaleCapture):
    def __init__(self,model):
        super().__init__(model)
        for l in self.native:
            block=model.model.layers[l]
            for name,module in [('mlp_x',block.post_attention_layernorm),('attention_x',block.input_layernorm)]:
                self.hooks.append(module.register_forward_hook(lambda m,a,o,l=l,name=name:self.put(f'L{l}_{name}',o,True)))
    def put(self,k,t,last):
        if self.on:self.data[k]=bits(t[0,self.n-1])


def prepare(key):
    from transformers import AutoTokenizer
    out=CAMPAIGN/'l_aligned'/key;path=ROOT/'models/hf'/MODELS[key]
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True,use_fast=True,trust_remote_code=True)
    base=[r for r in read(CAMPAIGN/'i_factorial/material.json') if r['unit'] in (0,8,12,13) and r['form']==r['style']==0]
    rows=[]
    for r in base:
        prompt=tok.apply_chat_template([{'role':'system','content':r['system']},{'role':'user','content':r['user']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);spans={}
        for name,term in [('u',r['u']),('v',r['v'])]:
            start=prompt.index(r['record'])+r['record'].index(term);end=start+len(term)
            positions=[i for i,(a,b) in enumerate(enc['offset_mapping']) if b>start and a<end and b>a];assert positions
            spans[name]={'term':term,'chars':[start,end],'positions':positions}
        rows.append(dict(r,prompt=prompt,prompt_ids=enc['input_ids'],tokens=tok.convert_ids_to_tokens(enc['input_ids']),spans=spans,model=key))
    assert len(rows)==256;immutable(out/'prefixes.json',rows)
    immutable(out/'protocol.json',{'phase':2706,'source_sha':sha(Path(__file__)),'model':key,'samples':256,'pilot':4,
      'material_sha':sha(out/'prefixes.json'),'selection':'I-factors entities0train64/8val64/12,13test128; form0style0, alltruth/query/language,all8families. Four sharedentitygroups only.',
      'execution':'Native chat template and tokenizer permodel; nonquantizedBF16 CUDA device_mapautooffload forlargermodels; batch1 naturalshape withKVcache. Onephysicalmodel atatime.',
      'alignment':'Save prefill and each actual generationstate until emitted token contains any Unicode letter or number and is not special. Do not skip incorrect lexicalcontent or select by expectedanswer. Whitespace/punctuation-only tokens are format; max8token cap recorded. First lexicaltoken can be an unwanted explanation, not automatically a valid answer.',
      'capture':'Every H0..L currentquery nativecoordinate; everygate/up/a/down unit andfullmlp_x/attention_x coordinates at0,1/3,2/3,lastMLP. GLM physicalgate_up projection split tracked separately. Not alltokenprefill storage inthisphase.',
      'noops':'First4 model.generate max8 nativegenerated sequence must agree through recorded contentboundary. Firstcontent measurement is not fullanswer/EOS measurement.',
      'resources':'Estimate from4pilotforward-only times; 5400seconds/modelcapture bound and8GiB diskfloor. No silent quantization or concurrentmodels.',
      'limits':['Functionally selected generation boundary, not proof of equivalent latent computation.','Wordpieces can be partial lexicaltokens.','Only4entitygroups,2heldout; unchangedform/style forcost.','Same external records acrossmodels, nativeprompttokenizations differ.']})
    return out,rows


def main(key,limit):
    import torch
    out,allrows=prepare(key);rows=allrows[:limit]
    if all((out/f'prefix_commits/{r["sample_id"]}.json').exists() for r in rows):return
    assert shutil.disk_usage(out).free>8*1024**3+1*1024**3
    if key=='qwen4':
        from phase2662_symmetric_mapping_contract import load_native
        model,tok=load_native('qwen4')
        save(out/'runtime.json',{'timestamp':stamp(),'dtype':str(model.dtype),'quantized':False,'devices':sorted({str(p.device) for p in model.parameters()}),'model_code_sha':sha(Path(sys.modules[model.model.__class__.__module__].__file__))})
    else:model,tok=load_large(key,out)
    if tok.pad_token_id is None:tok.pad_token=tok.eos_token
    cap=Capture(model);device=model.get_input_embeddings().weight.device;times=[];wall=time.monotonic()
    announce('l_aligned_'+key,state='running',completed=len(list((out/'prefix_commits').glob('*.json'))),total=256)
    try:
      with torch.inference_mode():
       for index,r in enumerate(rows):
        if (out/f'prefix_commits/{r["sample_id"]}.json').exists():continue
        if limit>4 and time.monotonic()-wall>5400:raise RuntimeError('Declared capture wallbudget reached; preserve completed prefixes')
        prefix=r['prompt_ids'].copy();generated=[];cache=None;steps=[];selected=None;step_records=[];start=time.monotonic()
        answerids=[tok.encode(w,add_special_tokens=False) for w in (('Yes','No') if r['language']=='en' else ('是','否'))]
        assert all(len(a)==1 for a in answerids);answerids=[a[0] for a in answerids]
        for step in range(8):
            inp=prefix if step==0 else prefix[-1:];cap.n=len(inp);cap.data={};cap.on=True
            state=model.model(input_ids=torch.tensor([inp],device=device),past_key_values=cache,use_cache=True)
            cache=state.past_key_values;logits=model.lm_head(state.last_hidden_state[:,-1:]).float()[0,0];cap.on=False
            nextid=int(logits.argmax());token=tok.decode([nextid]);lexical=nextid not in tok.all_special_ids and any(c.isalnum() for c in token)
            h=np.stack([cap.data.pop(f'H{l}') for l in range(len(model.model.layers)+1)])
            assert np.isfinite(unbits(h)).all() and all(np.isfinite(unbits(a)).all() for a in cap.data.values())
            sid=f'{r["sample_id"]}-s{step}';steps.append(sid)
            fp=out/f'fields/{sid}.npz';npz(fp,h_c=h,**cap.data)
            row=dict(r,sample_id=sid,prefix_id=r['sample_id'],generation_step=step,query_position=len(prefix)-1,
              prompt_ids=prefix.copy(),tokens=tok.convert_ids_to_tokens(prefix),next_token_id=nextid,next_token=token,lexical_content=lexical)
            rp=out/f'steps/{sid}.json';save(rp,row)
            save(out/f'commits/{sid}.json',{'sample_id':sid,'protocol_sha':sha(out/'protocol.json'),'files':{str(p.relative_to(out)):sha(p) for p in (fp,rp)}})
            step_records.append({'sample_id':sid,'step':step,'token_id':nextid,'text':token,'lexical_content':lexical,
              'answer_pair_logits':logits[answerids].cpu().tolist(),'candidate_pair_correct':int(logits[answerids].argmax())==(0 if r['expected_yes'] else 1)})
            generated.append(nextid);prefix.append(nextid)
            if lexical:selected=sid;break
            if nextid==tok.eos_token_id:break
        elapsed=time.monotonic()-start;times.append(elapsed);noop=None;noop_seconds=None
        if index<4:
            ns=time.monotonic();plain=model.generate(input_ids=torch.tensor([r['prompt_ids']],device=device),do_sample=False,max_new_tokens=8,use_cache=True,pad_token_id=tok.eos_token_id)[0,len(r['prompt_ids']):].tolist()
            noop=plain[:len(generated)]==generated;assert noop;noop_seconds=time.monotonic()-ns
        bp=out/f'behavior/{r["sample_id"]}.json'
        content=step_records[-1] if selected else None
        save(bp,{'sample_id':r['sample_id'],'first_token_id':generated[0],'first_token_text':tok.decode([generated[0]]),
          'first_token_correct':generated[0]==answerids[0 if r['expected_yes'] else 1],
          'content_state':selected,'content_step':None if content is None else content['step'],
          'content_token_correct':bool(content and content['token_id']==answerids[0 if r['expected_yes'] else 1]),
          'content_candidate_pair_correct':bool(content and content['candidate_pair_correct']),
          'steps':step_records,'generated_ids':generated,'cap_without_content':selected is None and len(generated)==8,
          'native_noop':noop,'native_noop_seconds':noop_seconds})
        save(out/f'prefix_commits/{r["sample_id"]}.json',{'sample_id':r['sample_id'],'steps':steps,'content_state':selected,'behavior_sha':sha(bp),'forward_capture_seconds':elapsed})
        print('ALIGNED',key,index+1,256,len(generated),round(elapsed,2),flush=True)
        announce('l_aligned_'+key,state='running',completed=index+1,total=256,last_case_seconds=elapsed)
        cap.data={};del cache,state,logits,h;gc.collect();assert shutil.disk_usage(out).free>8*1024**3
    finally:cap.close()
    material=[]
    for r in allrows:
        cp=out/f'prefix_commits/{r["sample_id"]}.json'
        if cp.exists():material.extend(read(out/f'steps/{sid}.json') for sid in read(cp)['steps'])
    save(out/'material.json',material)
    save(out/f'capture_{limit}.json',{'timestamp':stamp(),'new_case_forward_seconds':times,'mean_forward_seconds':float(np.mean(times)) if times else None,'wall_seconds':time.monotonic()-wall})
    announce('l_aligned_'+key,state='pilot_complete' if limit==4 else 'captured',completed=len(list((out/'prefix_commits').glob('*.json'))),total=256,states=len(material))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('model',choices=MODELS);p.add_argument('--limit',type=int,choices=(4,256),default=4);a=p.parse_args();main(a.model,a.limit)
