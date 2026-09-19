"""All-source/unit native ledger for contrasts chosen from already visible generation stage."""
from rdc_continuity_common import *
OUT=CAMPAIGN/'g_generation'

def main():
    from safetensors import safe_open
    from transformers import AutoTokenizer
    immutable(OUT/'ledger_protocol.json',{'source_sha':sha(Path(__file__)),'phase':2701,
      'contrast_rule':'step0 Yes-No (language-specific); step1 period-EOS; later EOS-period. Stage is observed before prediction. These are declared diagnostic current-stage contrasts, not all-vocabulary closure.',
      'account':'Input residual H35 + all32head/allsource P V WO + all9728 MLP a Wdown + six measured rounding terms.',
      'scope':'Actual final normalization scale used only in accounting, never in earlier-state predictor. Bias terms absent in actual checkpoint; tied input/output embeddings verified.'})
    rows=read(OUT/'material.json');model=ROOT/'models/hf/qwen3-4b';cfg=read(model/'config.json');assert cfg['tie_word_embeddings']
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True);idx=read(model/'model.safetensors.index.json')['weight_map'];key='model.embed_tokens.weight'
    ids={'en':[tok.encode(t,add_special_tokens=False)[0] for t in ('Yes','No','.')],'zh':[tok.encode(t,add_special_tokens=False)[0] for t in ('是','否','。')]}
    wi={}
    with safe_open(str(model/idx[key]),framework='pt',device='cpu') as f:
      for i in set([tok.eos_token_id]+ids['en']+ids['zh']):wi[i]=f.get_slice(key)[i:i+1,:].float().numpy()[0].astype(np.float64)
    wd=checkpoint('model.layers.35.mlp.down_proj.weight').float().numpy().astype(np.float64)
    wo=checkpoint('model.layers.35.self_attn.o_proj.weight').float().numpy().astype(np.float64)
    gamma=checkpoint('model.norm.weight').float().numpy().astype(np.float64);accounts=[]
    for i,r in enumerate(rows):
      yes,no,period=ids[r['language']];step=r['generation_step'];a_id,b_id=(yes,no) if step==0 else (period,tok.eos_token_id) if step==1 else (tok.eos_token_id,period)
      with np.load(OUT/f'fields/{r["sample_id"]}.npz') as z:
        h=unbits(z['h'][:,0]).astype(np.float64);norm=unbits(z['postnorm'][0]).astype(np.float64)
        native={k:unbits(z[f'L35_{k}']).astype(np.float64) for k in ('a','down','attention_out','head_output','p','v')}
        actual=float(z['logits'][a_id]-z['logits'][b_id])
      delta=wi[a_id]-wi[b_id];scale=gamma/np.sqrt(np.mean(h[36]**2)+cfg['rms_norm_eps']);effective=delta*scale
      a=native['a'][0];down=native['down'][0];attention=native['attention_out'][0];head=native['head_output'][0]
      beta=wd.T@effective;unit=a*beta;obeta=(wo.T@effective).reshape(32,128);p=native['p']
      source=p*np.einsum('hsd,hd->hs',native['v'][np.arange(32)//4],obeta)
      account={'sample_id':r['sample_id'],'step':step,'contrast_ids':[a_id,b_id],'contrast_tokens':tok.convert_ids_to_tokens([a_id,b_id]),'actual_margin':actual,
        'input_residual_term':float(h[35]@effective),'native_mlp_unit_sum':float(unit.sum()),'native_attention_source_sum':float(source.sum()),
        'head_matmul_rounding':float(head@obeta.reshape(-1)-source.sum()),'o_projection_rounding':float(attention@effective-head@obeta.reshape(-1)),
        'down_projection_rounding':float(down@effective-unit.sum()),'residual_add_rounding':float((h[36]-h[35]-attention-down)@effective),
        'normalization_rounding':float((norm-h[36]*scale)@delta),'lm_head_rounding':float(actual-norm@delta)}
      terms=[v for k,v in account.items() if k.endswith('_term') or k.endswith('_sum') or k.endswith('_rounding')]
      account['account_error']=float(sum(terms)-actual);assert abs(account['account_error'])<1e-8
      account['max_rounding']=max(abs(v) for k,v in account.items() if k.endswith('_rounding'))
      npz(OUT/f'ledgers/{r["sample_id"]}.npz',effective=effective,delta_unembedding=delta,native_a=a,beta=beta,unit_contribution=unit,attention_source_contribution=source,attention_p=p,postnorm=norm,logit_coordinate_contribution=norm*delta)
      save(OUT/f'accounts/{r["sample_id"]}.json',account);accounts.append(account)
      if i%64==0:print('CURRENT_LEDGER',i,len(rows),flush=True)
    result={'cases':len(rows),'max_abs_account_error':max(abs(a['account_error']) for a in accounts),'max_rounding_term':max(a['max_rounding'] for a in accounts),
      'by_step':[{'step':s,'n':sum(a['step']==s for a in accounts),'mean_input_residual':float(np.mean([a['input_residual_term'] for a in accounts if a['step']==s])),'mean_mlp':float(np.mean([a['native_mlp_unit_sum'] for a in accounts if a['step']==s])),'mean_attention':float(np.mean([a['native_attention_source_sum'] for a in accounts if a['step']==s]))} for s in sorted({a['step'] for a in accounts})]}
    save(OUT/'ledger_result.json',result);save(OUT/'extension_result.json',{'current_output_ledger':result})

if __name__=='__main__':main()
