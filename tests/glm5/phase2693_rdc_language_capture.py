"""S1 natural-shape, BF16, all-token residual field; full native coordinates at declared boundaries."""
import argparse, gc, re, shutil, sys
from rdc_feature_common import *
from rdc_s1_material import build, LEXICON

RUN='s1'; OUT=CAMPAIGN/RUN
NATIVE_LAYERS=(0,11,23,35)

class Capture:
    def __init__(self,model):
        self.enabled=False;self.hooks=[];self.arrays={};self.positions=[]
        self.hooks.append(model.model.embed_tokens.register_forward_hook(lambda m,a,o:self.put('H0',o)))
        for l,layer in enumerate(model.model.layers):
            self.hooks.append(layer.register_forward_hook(lambda m,a,o,l=l:self.put(f'H{l+1}',o)))
            if l in NATIVE_LAYERS:
                for name,module in [('q',layer.self_attn.q_proj),('k',layer.self_attn.k_proj),('v',layer.self_attn.v_proj),
                                    ('qnorm',layer.self_attn.q_norm),('knorm',layer.self_attn.k_norm),
                                    ('gate',layer.mlp.gate_proj),('up',layer.mlp.up_proj),('down',layer.mlp.down_proj),
                                    ('attention_x',layer.input_layernorm),('mlp_x',layer.post_attention_layernorm)]:
                    self.hooks.append(module.register_forward_hook(lambda m,a,o,l=l,name=name:self.put(f'L{l}_{name}',o,True)))
                self.hooks.append(layer.mlp.down_proj.register_forward_pre_hook(lambda m,a,l=l:self.put(f'L{l}_a',a[0],True)))
        self.hooks.append(model.model.norm.register_forward_hook(lambda m,a,o:self.put('postnorm',o)))
    def put(self,name,tensor,native=False):
        if self.enabled:
            tensor=tensor[0]
            if native:tensor=tensor[self.positions]
            self.arrays[name]=bits(tensor)
    def close(self):
        for h in self.hooks:h.remove()

def prepare():
    from transformers import AutoTokenizer
    modelpath=ROOT/'models/hf/qwen3-4b'
    tok=AutoTokenizer.from_pretrained(modelpath,local_files_only=True,use_fast=True)
    rows=build(tok)
    immutable(OUT/'material.json',rows)
    code=[Path(__file__),ROOT/'tests/glm5/rdc_s1_material.py',ROOT/'tests/glm5/rdc_feature_extractors.py']
    contract={'version':1,'samples':512,'lexical_entries':64,'families':[r[0] for r in LEXICON],
       'templates':4,'languages':['en','zh'],'model_path':str(modelpath),'dtype':'bfloat16','quantized':False,
       'capture':'batch1 natural prompt length, no padding/truncation, use_cache=False; native eager attention',
       'residual':'H0 actual embedding; H1..H36 post-block pre-final-norm, every real token and every 2560 coordinate',
       'native':'layers0,11,23,35; all scalar coordinates of Q/K/V/norm/gate/up/a/down at union(u span,v span,last token); exact source positions retained',
       'native_first_layer_information':'A5 uses actual pretrained native functions, not a pure algorithm-only improvement over E; same prefix but extra learned computation.',
       'precision':'BF16 arrays saved losslessly as uint16 bits; float64 analysis does not change model precision',
       'prediction_targets':['external task-convention class of A','same/different answer','later H36 from earlier H12'],
       'splits':'word blocks: indices0..3 train,4..5 validation,6..7 test in every family; partners stay in same block; translations grouped. Also form-heldout and joint-heldout.',
       'limits':['Groups are explicit task conventions, not a universal ontology.','Some function words and properties are polysemous.',
                 'Whole words may have multiple tokens; span pooling is explicit, no coordinate reduction.',
                 'Deep last-token context already contains lexical information; additive/interactive fits are not a causal factorization.'],
       'generation_max_new_tokens':16,'noop_samples':16,'material_sha':sha(OUT/'material.json'),
       'source_sha':{p.name:sha(p) for p in code},'model_config_sha':sha(modelpath/'config.json'),
       'token_length':[min(len(r['prompt_ids']) for r in rows),max(len(r['prompt_ids']) for r in rows)],
       'disk_upper_bf16_residual_bytes':sum(len(r['prompt_ids'])*37*2560*2 for r in rows),
       'resource_policy':'pilot first16 then full512 only if <60s/case and field estimate<free-8GiB; <=1 CUDA model'}
    if (OUT/'protocol.json').exists() and read(OUT/'protocol.json')!=contract:
        assert not list((OUT/'commits').glob('*.json')), 'Formal data exists: use a new protocol/run, do not overwrite'
        immutable(OUT/'preflight_protocol_v1.json',read(OUT/'protocol.json'))
        save(OUT/'preflight_incident.json',{'timestamp':stamp(),'formal_samples':0,
             'error':'Installed loader returned a CUDA model without hf_device_map; inspect actual parameter devices instead.',
             'old_protocol_sha':sha(OUT/'protocol.json'),'new_capture_sha':sha(Path(__file__))})
        save(OUT/'protocol.json',contract)
    else:immutable(OUT/'protocol.json',contract)
    return rows,contract

def main(limit=16):
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    rows,contract=prepare();rows=rows[:limit]
    assert shutil.disk_usage(OUT).free>contract['disk_upper_bf16_residual_bytes']+8*1024**3
    completed=[r for r in rows if (OUT/f'commits/{r["sample_id"]}.json').exists()]
    if len(completed)==len(rows):
        print('REQUESTED_PREFIX_ALREADY_COMMITTED',len(rows),flush=True);return
    status(RUN,state='loading',completed=len(completed),total=512,requested=limit,source_mode='live_model',protocol_sha=sha(OUT/'protocol.json'))
    model,tok=load_native('qwen4')
    actual_devices=sorted({str(p.device) for p in model.parameters()})
    assert actual_devices==['cuda:0'],actual_devices
    device_map=getattr(model,'hf_device_map',{'actual_parameter_devices':actual_devices})
    save(OUT/'runtime.json',{'timestamp':stamp(),'dtype':str(model.dtype),'quantized':getattr(model,'is_quantized',False),
         'device_map':device_map,'torch':torch.__version__,'model_code_sha':sha(Path(sys.modules[model.model.__class__.__module__].__file__)),
         'checkpoint_files':{p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in (ROOT/'models/hf/qwen3-4b').glob('*.safetensors')}})
    cap=Capture(model);noops=[];times=[]
    try:
        with torch.inference_mode():
            for index,row in enumerate(rows):
                commit=OUT/f'commits/{row["sample_id"]}.json'
                if commit.exists():
                    c=read(commit)
                    assert c['protocol_sha']==sha(OUT/'protocol.json')
                    for p,d in c['files'].items():assert sha(OUT/p)==d
                    continue
                start=time.monotonic();ids=torch.tensor([row['prompt_ids']],device='cuda:0')
                cap.positions=sorted(set(row['spans']['u']['positions']+row['spans']['v']['positions']+[len(row['prompt_ids'])-1]))
                cap.arrays={};cap.enabled=True
                state=model.model(input_ids=ids,use_cache=False).last_hidden_state
                logits=model.lm_head(state[:,-1]).float();cap.enabled=False
                if index<16:
                    plain=model.model(input_ids=ids,use_cache=False).last_hidden_state
                    equal=torch.equal(state,plain);assert equal
                    noops.append({'sample_id':row['sample_id'],'same_shape_bitwise_equal':equal});del plain
                h=np.stack([cap.arrays.pop(f'H{l}') for l in range(37)])
                assert h.shape==(37,len(row['prompt_ids']),2560)
                assert all(np.isfinite(unbits(a)).all() for a in cap.arrays.values())
                # Original checkpoint embeddings independently compared against hook capture.
                embedding=model.get_input_embeddings()(ids)[0]
                assert np.array_equal(h[0],bits(embedding));del embedding
                seq=model.generate(input_ids=ids,do_sample=False,max_new_tokens=16,use_cache=True,pad_token_id=tok.eos_token_id)[0,len(row['prompt_ids']):].tolist()
                text=tok.decode(seq,skip_special_tokens=True).strip()
                parsed=re.fullmatch(r'(Yes|No)[.!]?' if row['language']=='en' else r'(是|否)[。！]?',text,re.I)
                correct=bool(parsed and parsed[1].casefold()==row['target'].casefold())
                probs=torch.log_softmax(logits,dim=-1)[0]
                answer_ids=[tok.encode(w,add_special_tokens=False) for w in (('Yes','No') if row['language']=='en' else ('是','否'))]
                assert all(len(x)==1 for x in answer_ids)
                behavior={'sample_id':row['sample_id'],'generated':text,'generated_ids':seq,'eos':tok.eos_token_id in seq,
                          'correct':correct,'strict_correct':text.casefold()==row['target'].casefold(),'parsed':bool(parsed),
                          'first_argmax':int(logits.argmax()),'yes_no_logprob':[float(probs[a[0]]) for a in answer_ids],
                          'expected_target':row['target'],'protocol_sha':sha(OUT/'protocol.json')}
                pack=dict(h=h,native_positions=np.asarray(cap.positions,dtype=np.int32),**cap.arrays)
                field=OUT/f'fields/{row["sample_id"]}.npz';npz(field,**pack)
                bp=OUT/f'behavior/{row["sample_id"]}.json';save(bp,behavior)
                c={'sample_id':row['sample_id'],'protocol_sha':sha(OUT/'protocol.json'),'tokens':len(row['prompt_ids']),
                   'files':{str(p.relative_to(OUT)):sha(p) for p in (field,bp)},'elapsed_seconds':time.monotonic()-start,
                   'allH_shape':list(h.shape),'full_native_coordinate_arrays':{k:list(v.shape) for k,v in pack.items() if k!='h'}}
                save(commit,c);times.append(c['elapsed_seconds'])
                count=len(list((OUT/'commits').glob('*.json')))
                status(RUN,state='running',completed=count,total=512,requested=limit,current_sample=row['sample_id'],source_mode='live_model',
                       protocol_sha=sha(OUT/'protocol.json'),last_case_seconds=c['elapsed_seconds'])
                event(RUN,'sample_committed',sample_id=row['sample_id'],completed=count,total=512,
                      protocol_sha=sha(OUT/'protocol.json'),model='qwen3-4b',field_path=str(field.relative_to(OUT)))
                print('S1',count,512,row['sample_id'],round(c['elapsed_seconds'],2),flush=True)
                cap.arrays={};del state,logits,probs,pack,h;gc.collect()
                assert shutil.disk_usage(OUT).free>8*1024**3
                if limit==16 and c['elapsed_seconds']>60:raise RuntimeError('Pilot exceeds per-case budget; inspect before expansion')
    except BaseException as error:
        status(RUN,state='failed',completed=len(list((OUT/'commits').glob('*.json'))),total=512,error=repr(error),source_mode='live_model')
        raise
    finally:cap.close()
    if noops:save(OUT/f'noop_{limit}.json',{'records':noops,'all_passed':all(r['same_shape_bitwise_equal'] for r in noops)})
    count=len(list((OUT/'commits').glob('*.json')))
    save(OUT/f'capture_{limit}.json',{'timestamp':stamp(),'completed':count,'new_case_seconds':times,'mean_seconds':float(np.mean(times)) if times else None})
    status(RUN,state='captured' if count==512 else 'pilot_complete',completed=count,total=512,source_mode='live_model',protocol_sha=sha(OUT/'protocol.json'))
    del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,choices=(16,512),default=16)
    main(parser.parse_args().limit)
