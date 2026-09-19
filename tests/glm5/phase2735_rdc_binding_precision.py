"""Diagnose cancellation in conditional-digit gradients without changing materials."""
import shutil
from rdc_binding_common import *


def main():
    import torch
    from transformers import AutoTokenizer
    from rdc_law_native import Tail
    from rdc_binding_gradients import program_arrays,error_factors,dense_gradient
    out=BASE/'format_content/numerical_recovery';out.mkdir(parents=True,exist_ok=True)
    if not (out/'failed_attempt.json').exists():
        progress=read(BASE/'format_content/suite_progress.json')
        failure=next(r for r in progress if r['job']=='phase2735_rdc_binding_decomposition.py' and r['returncode']==1)
        original=BASE/'format_content/initial_decomposition.npz';copy=out/'failed_FP32_initial_decomposition.npz'
        shutil.copyfile(original,copy);assert sha(copy)==sha(original)
        immutable(out/'failed_attempt.json',{'timestamp':stamp(),'failure':failure,
          'failed_source':snapshot(ROOT/'tests/glm5/phase2735_rdc_binding_decomposition.py'),
          'preserved_initial_scores':{'file':str(copy.relative_to(BASE)),'sha256':sha(copy)},
          'unchanged_protocol_sha256':sha(BASE/'format_content/protocol.json'),
          'observed_assertion':{'content':{'g':.08304844796657562,'u':.035932961851358414,'d':.13293224573135376}},
          'scope':'The first native autograd audit failed. No directions or prospective parameter updates were produced by this attempt. Its elapsed value is the measured child-process wall time.'})
        ledger('failed_content_format_native_gradient_audit',failure['wall_seconds'],timing_scope='child process wall time')
    start=time.monotonic();tail=Tail();rows=gzread(BASE/'program_material.json.gz')[:8]
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    digits=[tok(str(i),add_special_tokens=False)['input_ids'][0] for i in range(1,9)]
    xx,rr,tt=program_arrays(rows);records=[]
    for variant in ('FP32_subtracted_loss','FP32_direct_conditional','FP64_probability_direct'):
      for i,row in enumerate(rows if variant=='FP64_probability_direct' else rows[:1]):
       for part in ('content','format'):
        x=torch.tensor(xx[i:i+1],device='cuda');r=torch.tensor(rr[i:i+1],device='cuda');target=torch.tensor(tt[i:i+1],device='cuda')
        for v in tail.w.values():v.requires_grad_(True);v.grad=None
        z=tail.forward(x,r,target)
        logits=z['logits'].double() if variant=='FP64_probability_direct' else z['logits']
        lp=logits.log_softmax(-1);dlp=logits[:,digits].log_softmax(-1)
        local=digits.index(int(target[0]));logmass=torch.logsumexp(lp[:,digits],1)
        loss=(-lp[:,target[0]]+logmass).mean() if part=='content' and variant=='FP32_subtracted_loss' else -dlp[:,local].mean() if part=='content' else -logmass.mean()
        with torch.no_grad():
            pc=torch.zeros_like(lp);pc[:,digits]=dlp.exp()
            error=pc.clone() if part=='content' else lp.exp()-pc
            if part=='content':error[:,target[0]]-=1
            expected=dense_gradient(error_factors(tail,z,x,error.float()))
        loss.backward()
        values={k:{'relative_max_error':float((v.grad-expected[k]).abs().max()/v.grad.abs().max().clamp_min(1e-12)),
          'gradient_max_abs':float(v.grad.abs().max()),'absolute_max_error':float((v.grad-expected[k]).abs().max())}
          for k,v in tail.w.items()}
        records.append({'variant':variant,'sample_id':row['sample_id'],'part':part,'loss':float(loss.detach()),'matrices':values})
        print('CONTENT_FORMAT_PRECISION',variant,i,part,{k:round(v['relative_max_error'],7) for k,v in values.items()},flush=True)
        for v in tail.w.values():v.grad=None;v.requires_grad_(False)
        del x,r,target,z,logits,lp,dlp,logmass,loss,pc,error,expected
    stable=[r for r in records if r['variant']=='FP64_probability_direct']
    passed=all(v['relative_max_error']<5e-5 for r in stable for v in r['matrices'].values())
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'records':records,'stable_variant_passed':passed,
      'precision':'Original parameter values and MLP/logit forward arithmetic FP32. FP64 probability normalization and direct restricted-vocabulary content loss; backward casts logit adjoints to FP32 as the library does.',
      'scope':'Same-forward actual-parameter autograd audit, first8 pre-existing examples. Conditional loss is a known identity; this diagnostic neither uses new labels for fitting nor changes any checkpoint.',
      'seconds':time.monotonic()-start}
    save(out/'precision_diagnostic.json',result);ledger('content_format_precision_diagnostic',result['seconds'])
    assert passed,result


if __name__=='__main__':main()
