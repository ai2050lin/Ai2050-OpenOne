"""Prospective 1024-case confirmation; global padded versus natural shape, unchanged hooks."""
import argparse,gc,shutil,sys
from rdc_continuity_common import *
from rdc_continuity_material import build,FAMILIES
from phase2693_rdc_language_capture import Capture
RUN='e_confirmation';OUT=CAMPAIGN/RUN

def prepare():
    from transformers import AutoTokenizer
    freeze_contract();tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    rows=build(tok);immutable(OUT/'material.json',rows)
    maxlen=max(len(r['prompt_ids']) for r in rows)
    paths=list((HISTORY/'b_relations/models').glob('*.npz'))
    immutable(OUT/'protocol.json',{'phase':2699,'samples':1024,'families':list(FAMILIES),'global_execution_length':maxlen,
      'model':'qwen3-4b nonquantized BF16 CUDA eager','material_sha':sha(OUT/'material.json'),'source_sha':sha(Path(__file__)),'material_code_sha':sha(ROOT/'tests/glm5/rdc_continuity_material.py'),
      'old_reader_hashes':{p.name:sha(p) for p in paths},'frozen_reader_targets':['positive_support','requested_answer'],
      'representations':['H12/H24/H36 UVC','H12/H24/H36 C_only'],'old_reader_training':'Old B fit is frozen before inspecting any new model outputs; all1024 new cases are prospective tests for those readers.',
      'new_reader_splits':'New units0..7 train512,8..11validation256,12..15test256; all bilingual/fact/query variants grouped.',
      'capture':'Matched global right padding/mask, batch1, last real position. Save all real-token H0..36 full2560, native0/11/23/35 full units at U/V/C. Natural-shape fullH audited in memory; persist role H and all-coordinate per-layer maximum errors, not natural fullH.',
      'finite_check':'Every saved H/native scalar. First16 hook-disabled same-shape bitwise checks.',
      'natural_generation':'Not claimed here; dual-track actual next-token logits, generation separately in2701.',
      'limits':['Padding equalizes execution shape only, not roles/positions/semantics.','Conjunction is a new operation; seven old families use new lexical/statement forms.','Positive depth3 taxonomy chains versus explicit-target-denial negatives have asymmetric proof depth; not a clean depth difficulty estimate.'],
      'disk_estimate_bytes':sum(len(r['prompt_ids'])*37*2560*2 for r in rows)+4*1024**3})
    return rows

def main(limit):
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    allrows=prepare();rows=allrows[:limit];contract=read(OUT/'protocol.json');maxlen=contract['global_execution_length']
    pending=[r for r in rows if not (OUT/f'commits/{r["sample_id"]}.json').exists()]
    if not pending:return
    assert shutil.disk_usage(OUT).free>contract['disk_estimate_bytes']+8*1024**3
    announce(RUN,state='loading',completed=limit-len(pending),total=1024)
    model,tok=load_native('qwen4');assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    cap=Capture(model);noops=[];times=[]
    save(OUT/'runtime.json',{'timestamp':stamp(),'torch':torch.__version__,'dtype':str(model.dtype),'quantized':False,'actual_devices':sorted({str(p.device) for p in model.parameters()}),'model_code_sha':sha(Path(sys.modules[model.model.__class__.__module__].__file__))})
    try:
      with torch.inference_mode():
       for index,r in enumerate(rows):
        if (OUT/f'commits/{r["sample_id"]}.json').exists():continue
        start=time.monotonic();n=len(r['prompt_ids']);cap.positions=sorted(set(r['spans']['u']['positions']+r['spans']['v']['positions']+[n-1]))
        ids=torch.tensor([r['prompt_ids']],device='cuda');pad=torch.nn.functional.pad(ids,(0,maxlen-n),value=tok.pad_token_id or tok.eos_token_id)
        mask=(torch.arange(maxlen,device='cuda')[None]<n).long()
        cap.arrays={};cap.enabled=True
        state=model.model(input_ids=pad,attention_mask=mask,use_cache=False).last_hidden_state
        matched_logits=model.lm_head(state[:,n-1]).float()[0];cap.enabled=False
        pack=dict(h=np.stack([cap.arrays.pop(f'H{l}')[:n] for l in range(37)]),native_positions=np.array(cap.positions,dtype=np.int32),**cap.arrays)
        pack['postnorm']=pack['postnorm'][:n]
        if index<16:
            plain=model.model(input_ids=pad,attention_mask=mask,use_cache=False).last_hidden_state
            assert torch.equal(state,plain);noops.append({'sample_id':r['sample_id'],'same_shape_noop':True});del plain
        cap.arrays={};cap.enabled=True
        natural_state=model.model(input_ids=ids,use_cache=False).last_hidden_state
        natural_logits=model.lm_head(natural_state[:,-1]).float()[0];cap.enabled=False
        nh=np.stack([cap.arrays.pop(f'H{l}') for l in range(37)])
        assert np.array_equal(nh[0],pack['h'][0]) and np.array_equal(pack['h'][0],bits(model.get_input_embeddings()(ids)[0]))
        assert all(np.isfinite(unbits(a)).all() for k,a in pack.items() if k!='native_positions')
        diff=unbits(nh).astype(np.float64)-unbits(pack['h']).astype(np.float64)
        roles=[r['spans']['u']['positions'],r['spans']['v']['positions'],[n-1]]
        pack['natural_roles']=np.stack([unbits(nh[:,ix]).mean(1) for ix in roles],axis=1)
        pack['shape_error_coordinate_max']=np.max(np.abs(diff),axis=1).astype(np.float32)
        answerids=[tok.encode(w,add_special_tokens=False)[0] for w in (('Yes','No') if r['language']=='en' else ('是','否'))]
        choices=[int(x.argmax()) for x in (natural_logits,matched_logits)]
        behavior={'sample_id':r['sample_id'],'natural_argmax':choices[0],'matched_argmax':choices[1],
          'natural_answer_logits':natural_logits[answerids].cpu().tolist(),'matched_answer_logits':matched_logits[answerids].cpu().tolist(),
          'answer_ids':answerids,'natural_correct':choices[0]==answerids[0 if r['expected_yes'] else 1],'matched_correct':choices[1]==answerids[0 if r['expected_yes'] else 1],
          'shape_max':float(np.abs(diff).max()),'natural_length':n,'padded_length':maxlen,'generated':'not generated in this phase','expected_target':r['target']}
        fp=OUT/f'fields/{r["sample_id"]}.npz';npz(fp,**pack)
        bp=OUT/f'behavior/{r["sample_id"]}.json';save(bp,behavior)
        elapsed=time.monotonic()-start
        save(OUT/f'commits/{r["sample_id"]}.json',{'sample_id':r['sample_id'],'protocol_sha':sha(OUT/'protocol.json'),'files':{str(p.relative_to(OUT)):sha(p) for p in (fp,bp)},'elapsed_seconds':elapsed})
        count=len(list((OUT/'commits').glob('*.json')));times.append(elapsed)
        announce(RUN,state='running',completed=count,total=1024,current_sample=r['sample_id'],last_case_seconds=elapsed)
        events(RUN,'sample_committed',sample_id=r['sample_id'],completed=count,total=1024)
        print('EQUAL',count,1024,round(elapsed,2),flush=True)
        del pack,nh,diff,state,natural_state,matched_logits,natural_logits;cap.arrays={};gc.collect()
        assert shutil.disk_usage(OUT).free>8*1024**3
        if limit==16:assert elapsed<60
    finally:cap.close()
    save(OUT/f'capture_{limit}.json',{'new_cases':len(times),'seconds':times,'noops':noops})
    announce(RUN,state='captured' if limit==1024 else 'pilot_complete',completed=len(list((OUT/'commits').glob('*.json'))),total=1024)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,choices=(16,1024),default=16);a=p.parse_args();main(a.limit)
