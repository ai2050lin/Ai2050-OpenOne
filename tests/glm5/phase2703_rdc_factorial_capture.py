"""All-coordinate factorial observations with an explicit all-token persistence panel."""
import argparse,gc,shutil,sys
from rdc_conditional_common import *
from rdc_conditional_material import build,FAMILIES
from phase2693_rdc_language_capture import Capture
RUN='i_factorial';OUT=CAMPAIGN/RUN


def prepare():
    from transformers import AutoTokenizer
    freeze();tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    rows=build(tok);immutable(OUT/'material.json',rows)
    lengths=[len(r['prompt_ids']) for r in rows]
    panel_bytes=sum(len(r['prompt_ids'])*37*2560*2 for r in rows if r['full_token_panel'])
    estimate=panel_bytes+4096*(37*3*2560*4+3*(3*9728+3*2560)*2)
    immutable(OUT/'protocol.json',{'phase':2703,'samples':4096,'families':list(FAMILIES),'material_sha':sha(OUT/'material.json'),
      'capture_code_sha':sha(Path(__file__)),'material_code_sha':sha(ROOT/'tests/glm5/rdc_conditional_material.py'),
      'factors':'8families*16entity_groups*2support*2query*2form*2style*2language; all variants/entity grouped for train2048/val1024/test1024.',
      'model':'qwen3-4b nonquantized BF16 CUDA eager, batch1 natural length, no padding/truncation, no KVcache',
      'capture':'H0 actualembedding,H1..H36 postblock prenorm, every token and coordinate scanned. Persist all-token H for entities0/12 (512 full-factorial cases), all4096 H0..36 U/V means and C fullcoordinates; C BF16 exact additionally; L11/23/35 all9728 gate/up/a and all2560 mlp_x/down/attention_x at C.',
      'panel':'Deterministic trainentity0 and testentity12 in eachfamily, all32factor cells, selected before model results. Not whole4096 full-token persistence.',
      'online_field_summary':'All realtokens, all37*2560 coordinates accumulate raw first/second moments per family-language; token-weighted. Raw and standard deviations retain nativecoordinate order.',
      'noops':'First16 forward hooks enabled/disabled same-shape bitwise equality and embedding table row equality.',
      'behavior':'Actual first-token full-vocabulary argmax and Yes/No or 是/否 candidate scores; no claim of complete answers here.',
      'length_range':[min(lengths),max(lengths)],'estimated_raw_bytes':estimate,'panel_uncompressed_bytes':panel_bytes,
      'limits':['Only two specific styles/forms, family-dependent form transformations.','All4096 retained role means are not the fulltoken Markovstate.','Fullfield mean/std is token weighted and length dependent.','Explicit categories are task-defined, not an ontology.','Natural shape is primary; dedicated numerical side audit does not assume zero shape effects.']})
    return rows


def main(limit):
    import torch
    from phase2662_symmetric_mapping_contract import load_native
    rows=prepare();wanted=rows[:limit];protocol=read(OUT/'protocol.json')
    pending=[r for r in wanted if not (OUT/f'commits/{r["sample_id"]}.json').exists()]
    if not pending:return
    # Free-space estimate is remaining worst-case, conservatively without credit for compression.
    fraction=len(pending)/4096
    assert shutil.disk_usage(OUT).free>8*1024**3+protocol['estimated_raw_bytes']*fraction
    announce(RUN,state='loading',completed=limit-len(pending),total=4096,requested=limit)
    model,tok=load_native('qwen4');assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    cap=Capture(model);times=[];noops=[]
    save(OUT/'runtime.json',{'timestamp':stamp(),'torch':torch.__version__,'dtype':str(model.dtype),'quantized':False,
      'devices':sorted({str(p.device) for p in model.parameters()}),'model_code_sha':sha(Path(sys.modules[model.model.__class__.__module__].__file__)),
      'config_sha':sha(ROOT/'models/hf/qwen3-4b/config.json'),'checkpoint_files':{p.name:[p.stat().st_size,p.stat().st_mtime_ns] for p in (ROOT/'models/hf/qwen3-4b').glob('*.safetensors')}})
    try:
      with torch.inference_mode():
       for ix,r in enumerate(wanted):
        commit=OUT/f'commits/{r["sample_id"]}.json'
        if commit.exists():continue
        start=time.monotonic();ids=torch.tensor([r['prompt_ids']],device='cuda');n=ids.shape[1]
        cap.arrays={};cap.positions=[n-1];cap.enabled=True
        state=model.model(input_ids=ids,use_cache=False).last_hidden_state
        logits=model.lm_head(state[:,-1]).float()[0];cap.enabled=False
        if ix<16:
            plain=model.model(input_ids=ids,use_cache=False).last_hidden_state
            assert torch.equal(state,plain);noops.append({'sample_id':r['sample_id'],'bitwise_equal':True});del plain
        h=np.stack([cap.arrays.pop(f'H{l}') for l in range(37)])
        assert h.shape==(37,n,2560)
        assert np.array_equal(h[0],bits(model.get_input_embeddings()(ids)[0]))
        hf=unbits(h);assert np.isfinite(hf).all()
        roles=np.stack([hf[:,pos].mean(1) for pos in [r['spans']['u']['positions'],r['spans']['v']['positions'],[n-1]]],1)
        pack={'roles':roles,'h_c':h[:,-1],'native_positions':np.array([n-1],np.int32)}
        for l in (11,23,35):
            for name in ('gate','up','a','down','mlp_x','attention_x'):
                value=cap.arrays[f'L{l}_{name}'];assert np.isfinite(unbits(value)).all();pack[f'L{l}_{name}']=value[0]
        if r['full_token_panel']:pack['h']=h
        # Per-case moments are the resumable sufficient statistics; removed only after audited aggregation.
        fp=OUT/f'fields/{r["sample_id"]}.npz';npz(fp,**pack)
        mp=OUT/f'moments/{r["sample_id"]}.npz';npz(mp,sum=hf.astype(np.float64).sum(1),sumsq=(hf.astype(np.float64)**2).sum(1),tokens=np.array(n))
        answerids=[tok.encode(w,add_special_tokens=False)[0] for w in (('Yes','No') if r['language']=='en' else ('是','否'))]
        choice=int(logits.argmax());bp=OUT/f'behavior/{r["sample_id"]}.json'
        save(bp,{'sample_id':r['sample_id'],'argmax_id':choice,'argmax_text':tok.decode([choice]),
          'answer_ids':answerids,'answer_logits':logits[answerids].cpu().tolist(),'first_token_correct':choice==answerids[0 if r['expected_yes'] else 1],
          'pair_correct':int(logits[answerids].argmax())==(0 if r['expected_yes'] else 1),'expected_target':r['target'],'generation':'not performed'})
        elapsed=time.monotonic()-start
        save(commit,{'sample_id':r['sample_id'],'protocol_sha':sha(OUT/'protocol.json'),'files':{str(p.relative_to(OUT)):sha(p) for p in (fp,mp,bp)},'elapsed_seconds':elapsed,'full_token_panel':r['full_token_panel'],'tokens':n})
        times.append(elapsed)
        if ix%16==0 or ix+1==limit:
            announce(RUN,state='running',completed=ix+1,total=4096,requested=limit,last_case_seconds=elapsed)
            print('FACTORIAL',ix+1,4096,round(elapsed,3),flush=True)
        cap.arrays={};del h,hf,roles,pack,state,logits;gc.collect()
        assert shutil.disk_usage(OUT).free>8*1024**3
        if limit==16:assert elapsed<60
    finally:cap.close()
    save(OUT/f'capture_{limit}.json',{'new_cases':len(times),'seconds':times,'mean_seconds':float(np.mean(times)),'noops':noops,
      'field_bytes_sofar':sum(p.stat().st_size for p in (OUT/'fields').glob('*.npz'))})
    announce(RUN,state='captured' if limit==4096 else 'pilot_complete',completed=len(list((OUT/'commits').glob('*.json'))),total=4096)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,choices=(16,4096),default=16);main(p.parse_args().limit)
