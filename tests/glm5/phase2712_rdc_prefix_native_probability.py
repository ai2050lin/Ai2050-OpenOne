"""Complete vocabulary comparison through real norm/unembedding, serial CUDA weights only."""
import argparse
import gc
from rdc_prefix_estimators import *


def checkpoint(key):
    from safetensors import safe_open
    folder=ROOT/'models/hf/qwen3-4b';index=read(folder/'model.safetensors.index.json')['weight_map']
    with safe_open(str(folder/index[key]),framework='pt',device='cpu') as f:return f.get_tensor(key)


def main(confirmation,causal_controls=False):
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=CAMPAIGN/('confirmation' if confirmation else 'shared_rules');out=source/('full_vocabulary_causal_controls' if causal_controls else 'full_vocabulary')
    features=CAMPAIGN/'shared_rules'/('qwen4_confirmation' if confirmation else 'qwen4')
    rows=read(features/'rows.json');indices=np.arange(len(rows)) if confirmation else splits(rows)[2];testrows=[rows[i] for i in indices]
    config=read(ROOT/'models/hf/qwen3-4b/config.json');key='model.embed_tokens.weight' if config['tie_word_embeddings'] else 'lm_head.weight'
    immutable(out/'protocol.json',{'source_sha':sha(Path(__file__)),'phase':2713 if confirmation else 2712,'units':len(testrows)//2,
      'vocabulary':config['vocab_size'],'readout_parameter':key,'epsilon':config['rms_norm_eps'],
      'native_current_reference':'Original saved BF16 logits at two anchors; all151936 entries.',
      'native_temporal_reference':'BF16 lm_head recomputation from the saved actual next-position postnorm, batch16; original next-position logits were not archived. It is an observed-state readout reference, not a model forecast.',
      'candidate':'Predicted raw H36 -> its OWN RMS + real finalnorm gamma -> real unembedding FP32 with TF32 disabled. No actual future norm supplied.',
      'arithmetic_floor':'Actual H36 through FP32 norm/readout compared to each reference, not counted as a predictive model.',
      'outputs':'All-vocabulary KL, distribution argmax, observed-following-token NLL, entropy, per-vocabulary mean KL contribution and per-source-unit scores. Full predicted log probabilities stored for first4 rows only, remaining distributions reproducible from H prediction and checkpoint.',
      'interpretation':'Matching the native probability distribution is not semantic accuracy or free generation. Current and next-input-available temporal tasks have different available inputs.'})
    assert torch.cuda.is_available();weight_bf=checkpoint(key).cuda();weight=weight_bf.float();gamma=checkpoint('model.norm.weight').float().cuda();eps=config['rms_norm_eps']
    native_current=[];native_next_post=[];native_current_h=[];native_next_h=[];current_target=[];next_target=[]
    for r in testrows:
        with np.load(CAMPAIGN/r['field_path']) as z:
            k=r['anchor_array_index'];native_current.append(unbits(z['logits'][r['anchor']]))
            native_next_post.append(unbits(z['postnorm'][k+1]));native_current_h.append(unbits(z['h'][36,k]));native_next_h.append(unbits(z['h'][36,k+1]))
        rr=read(CAMPAIGN/('qwen4_confirmation' if confirmation else 'qwen4')/f'rows/{r["sample_id"]}.json')
        current_target.append(rr['prompt_ids'][r['position']+1]);next_target.append(rr['prompt_ids'][r['next_position']+1])
    native_current=np.stack(native_current);native_next_post=np.stack(native_next_post)
    candidates=[]
    paths=sorted((CAMPAIGN/'causal_hash_control/predictions').glob(('confirmation_' if confirmation else 'test_')+'*.npz')) if causal_controls else sorted((source/'predictions').glob('*.npz'))
    for path in paths:
        if path.stem.endswith('_rollout'):continue
        with np.load(path) as z:
            p=z['prediction'];assert len(p)==len(indices)
            name=path.stem.removeprefix('confirmation_').removeprefix('test_') if causal_controls else path.stem
            candidates.append((name,p[:,-2560:],name.startswith('temporal_'),sha(path)))
    if not confirmation and not causal_controls:
        for path in (CAMPAIGN/'layer_operators/predictions').glob('*.npz'):
            with np.load(path) as z:candidates.append(('layer_'+path.stem,z['prediction'],False,sha(path)))
    candidates.extend([('native_current_H36_FP32_oracle',np.stack(native_current_h),False,None),('native_next_H36_FP32_oracle',np.stack(native_next_h),True,None)])
    reports=[]
    with torch.inference_mode():
      for name,pred,temporal,psha in candidates:
        per=[];vocab_kl=np.zeros(config['vocab_size'],np.float64);saved=[]
        for start in range(0,len(indices),16):
            stop=min(start+16,len(indices));p=torch.tensor(pred[start:stop],dtype=torch.float32,device='cuda')
            normalized=p*torch.rsqrt(p.square().mean(-1,keepdim=True)+eps)*gamma
            lp=(normalized@weight.T).log_softmax(-1)
            if temporal:
                actual=torch.tensor(native_next_post[start:stop],dtype=torch.bfloat16,device='cuda')
                lq=(actual@weight_bf.T).float().log_softmax(-1)
            else:lq=torch.tensor(native_current[start:stop],dtype=torch.float32,device='cuda').log_softmax(-1)
            q=lq.exp();terms=q*(lq-lp);kl=terms.sum(-1);vocab_kl+=terms.sum(0).cpu().numpy()
            argp=lp.argmax(-1);argq=lq.argmax(-1);target=(next_target if temporal else current_target)[start:stop]
            for j,k in enumerate(range(start,stop)):
                per.append({'sample_id':testrows[k]['sample_id'],'source_group':testrows[k]['source_group'],'language':testrows[k]['language'],
                  'genre':testrows[k]['genre'],'anchor':testrows[k]['anchor'],'native_to_predicted_KL':float(kl[j]),
                  'argmax_agreement':bool(argp[j]==argq[j]),'native_observed_token_nll':-float(lq[j,target[j]]),
                  'predicted_observed_token_nll':-float(lp[j,target[j]]),'native_argmax':int(argq[j]),'predicted_argmax':int(argp[j]),
                  'native_entropy':float(-(q[j]*lq[j]).sum())})
            if start==0:saved.append(lp[:4].cpu().numpy())
        summary={'model':name,'prediction_sha':psha,'temporal_next_input_available':temporal,'arithmetic_oracle':name.endswith('_oracle'),
          'n':len(per),'mean_KL':float(np.mean([r['native_to_predicted_KL'] for r in per])),
          'argmax_agreement':float(np.mean([r['argmax_agreement'] for r in per])),
          'native_observed_token_nll':float(np.mean([r['native_observed_token_nll'] for r in per])),
          'predicted_observed_token_nll':float(np.mean([r['predicted_observed_token_nll'] for r in per]))}
        for keygroup in ('language','genre','source_group'):
            summary['by_'+keygroup]={v:{'n':len(rr),'mean_KL':float(np.mean([r['native_to_predicted_KL'] for r in rr])),
              'argmax_agreement':float(np.mean([r['argmax_agreement'] for r in rr]))}
              for v in sorted({r[keygroup] for r in per}) if (rr:=[r for r in per if r[keygroup]==v])}
        save(out/f'per_model/{name}.json',{'summary':summary,'rows':per})
        npz(out/f'full_vocabulary/{name}.npz',mean_kl_contribution=(vocab_kl/len(per)).astype(np.float32),
          first4_log_probabilities=np.concatenate(saved),first4_row_indices=indices[:4])
        assert abs(float(vocab_kl.sum()/len(per))-summary['mean_KL'])<1e-4
        reports.append(summary);print('PREFIX_FULL_VOCAB',name,summary['mean_KL'],summary['argmax_agreement'],flush=True)
    save(out/'result.json',{'timestamp':stamp(),'reports':reports,'vocabulary_size':config['vocab_size'],'all_vocabulary_entries_evaluated':True,
      'quantized_checkpoint':False,'TF32':False,'native_current_ref':'saved BF16 logits','native_next_ref':'recomputed BF16 head from saved actual postnorm',
      'scope':'Distribution approximation, not native whole-task correctness or a complete generator.'})
    del weight,weight_bf,gamma;gc.collect();torch.cuda.empty_cache();guard()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--confirmation',action='store_true');p.add_argument('--causal-controls',action='store_true');a=p.parse_args();main(a.confirmation,a.causal_controls)
