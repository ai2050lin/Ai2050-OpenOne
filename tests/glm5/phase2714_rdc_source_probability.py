"""Complete vocabulary evaluation for frozen all-source rules; no model-body loading."""
from phase2714_rdc_full_source_history import *
from phase2712_rdc_prefix_native_probability import checkpoint


def main():
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    frozen=read(OUT/'frozen.json');best=frozen['selected_by_validation']
    for rel,digest in frozen['files'].items():assert sha(OUT/rel)==digest
    cfg=read(ROOT/'models/hf/qwen3-4b/config.json');key='model.embed_tokens.weight'
    immutable(OUT/'probability_protocol.json',{'timestamp':stamp(),'source_sha':sha(Path(__file__)),'phase':2714,
      'selected_rule_before_fresh':best,'vocabulary':cfg['vocab_size'],'candidate':'Own predicted H36 RMS -> actual gamma -> actual FP32 readout, TF32 off.',
      'main_reference':'Original saved BF16 logits at selected current anchors, natural batch1 body and batch2 head.',
      'fresh_reference':'Saved actual BF16 postnorm through actual BF16 head, batch16. Original fresh full logits were not retained; capture-time argmax and observed NLL are retained separately.',
      'oracle':'Actual saved raw H36 through same FP32 norm/head to calibrate arithmetic change.',
      'retained':'Per-row metrics, all151936 mean KL contributions for every rule/scope; one complete log-probability vector for preselected rule in each scope. Remaining full probabilities reproducible from saved full H36 predictions and checkpoint.',
      'limits':'Probability agreement is not natural-generation accuracy or a causal proof. All operations use full vocabulary, no Top-K.'})
    wb=checkpoint(key).cuda();w=wb.float();gamma=checkpoint('model.norm.weight').float().cuda();eps=cfg['rms_norm_eps'];vocab=cfg['vocab_size']
    all_reports=[]
    with torch.inference_mode():
      for scope in ('test','fresh'):
        rr=read(OUT/('main_rows.json' if scope=='test' else 'fresh_rows.json'));indices=splits(rr)[2] if scope=='test' else np.arange(len(rr));rows=[rr[i] for i in indices]
        actual_h=[];post=[];logits=[];observed=[];material={r['sample_id']:r for r in (main_rows() if scope=='test' else read(OUT/'fresh_material.json'))}
        for r in rows:
            k=r['anchor'];sid=r['sample_id'];source=CAMPAIGN/f'qwen4/fields/{sid}.npz' if scope=='test' else OUT/f'fresh/fields/{sid}.npz'
            with np.load(source) as z:
                actual_h.append(unbits(z['h'][36,k*3]) if scope=='test' else unbits(z['h36'][k]))
                post.append(unbits(z['postnorm'][k*3 if scope=='test' else k]))
                if scope=='test':logits.append(unbits(z['logits'][k]))
            observed.append(material[sid]['prompt_ids'][r['position']+1])
        actual_h=np.stack(actual_h);post=np.stack(post)
        if scope=='test':logits=np.stack(logits)
        candidates=[]
        for rule in RULES:
            with np.load(OUT/f'predictions/{scope}_{rule}.npz') as z:candidates.append((rule,z['prediction']))
        with np.load(OUT/'models/current.npz') as z:mean=z['means']
        candidates.extend([('train_mean',np.broadcast_to(mean,actual_h.shape)),('actual_H36_FP32_oracle',actual_h)])
        for rule,pred in candidates:
            per=[];total=np.zeros(vocab,np.float64);first=None
            for begin in range(0,len(rows),16):
                end=min(begin+16,len(rows));h=torch.tensor(np.array(pred[begin:end]),dtype=torch.float32,device='cuda')
                lp=((h*torch.rsqrt(h.square().mean(-1,keepdim=True)+eps)*gamma)@w.T).log_softmax(-1)
                lq=(torch.tensor(logits[begin:end],dtype=torch.float32,device='cuda').log_softmax(-1) if scope=='test' else
                  (torch.tensor(post[begin:end],dtype=torch.bfloat16,device='cuda')@wb.T).float().log_softmax(-1))
                terms=lq.exp()*(lq-lp);kl=terms.sum(-1);total+=terms.sum(0).cpu().numpy();pa=lp.argmax(-1);qa=lq.argmax(-1)
                for j,i in enumerate(range(begin,end)):
                    per.append({q:rows[i][q] for q in ('sample_id','anchor','language','genre','source_group')}|{
                      'KL':float(kl[j]),'argmax_agreement':bool(pa[j]==qa[j]),'native_argmax':int(qa[j]),'predicted_argmax':int(pa[j]),
                      'native_observed_token_NLL':-float(lq[j,observed[i]]),'predicted_observed_token_NLL':-float(lp[j,observed[i]])})
                if begin==0 and rule==best:first=lp[:1].cpu().numpy()
            report={'scope':scope,'rule':rule,'n':len(rows),'KL':float(np.mean([r['KL'] for r in per])),
              'argmax_agreement':float(np.mean([r['argmax_agreement'] for r in per])),'selected_before_fresh':rule==best,
              'arithmetic_oracle':rule.endswith('oracle'),'native_observed_token_NLL':float(np.mean([r['native_observed_token_NLL'] for r in per])),
              'predicted_observed_token_NLL':float(np.mean([r['predicted_observed_token_NLL'] for r in per]))}
            for group in ('language','genre','source_group'):
                report['by_'+group]={v:{'n':len(rs),'KL':float(np.mean([r['KL'] for r in rs])),'argmax_agreement':float(np.mean([r['argmax_agreement'] for r in rs]))}
                  for v in sorted({r[group] for r in per}) if (rs:=[r for r in per if r[group]==v])}
            packet={'all_vocabulary_mean_KL_contribution':(total/len(rows)).astype(np.float32)}
            if first is not None:packet.update(first1_complete_log_probability=first,first1_row_index=indices[:1])
            npz(OUT/f'probability/{scope}_{rule}.npz',**packet)
            save(OUT/f'probability/{scope}_{rule}.json',{'summary':report,'rows':per})
            assert abs(total.sum()/len(rows)-report['KL'])<1e-4
            all_reports.append(report);print('SOURCE_PROBABILITY',scope,rule,report['KL'],report['argmax_agreement'],flush=True)
    save(OUT/'probability_result.json',{'timestamp':stamp(),'reports':all_reports,'vocabulary':vocab,'all_entries_evaluated':True,
      'selected_before_fresh':best,'frozen_manifest_sha':sha(OUT/'frozen.json'),'new_mathematical_theorem':False,'mechanism_closed':False})
    del w,wb,gamma;gc.collect();torch.cuda.empty_cache();guard(12*1024**2)
    print('SOURCE_PROBABILITY_COMPLETE',usage(),CEILING-usage(),flush=True)


if __name__=='__main__':main()
