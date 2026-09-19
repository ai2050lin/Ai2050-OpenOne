"""Export same-precision full CE factors for the client, with exact score replay."""
import gc
from rdc_binding_common import *


def main():
    import torch
    from transformers import AutoTokenizer
    from rdc_law_native import Tail,factor_gram
    from phase2735_rdc_binding_decomposition import arrays,collect_parts
    out=BASE/'format_content'
    assert read(out/'suite_result.json')['all_passed']
    if (out/'full_factor_export.json').exists():return
    start=time.monotonic();guard(150*1024**2);rows=gzread(BASE/'program_material.json.gz')
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    digits=[tok(str(i),add_special_tokens=False)['input_ids'][0] for i in range(1,9)]
    tail=Tail();x,r,t=arrays(rows)
    f,stats=collect_parts(tail,torch.tensor(x,device='cuda'),torch.tensor(r,device='cuda'),torch.tensor(t,device='cuda'),digits)
    with np.load(out/'initial_decomposition.npz') as z:
        equal={k:bool(np.array_equal(v,z[k])) for k,v in stats.items()}
        assert all(equal.values()),equal
    for part in ('content','format'):
        with np.load(out/f'{part}_gradient_factors.npz') as z:
            assert all(np.array_equal(v.float().cpu().numpy(),z[k]) for k,v in f[part].items()),part
    gram=factor_gram(f['full'])['total'].cpu().numpy()
    with np.load(out/'all_parameter_gram_decomposition.npz') as z:
        gram_error=float(np.max(np.abs(gram-z['full']))/max(np.max(np.abs(z['full'])),1e-30))
    assert gram_error<1e-12,gram_error
    npz(out/'full_gradient_factors.npz',**{k:v.float().cpu().numpy() for k,v in f['full'].items()})
    report={'timestamp':stamp(),'source':snapshot(Path(__file__)),'rows':len(rows),'all_passed':True,
      'same_precision_score_replay_exact':equal,'saved_content_format_factor_replay_exact':True,
      'full_parameter_gram_replay_relative_max_error':gram_error,
      'factor_sha256':sha(out/'full_gradient_factors.npz'),'seconds':time.monotonic()-start,
      'scope':'No fitting, selection or parameter update. Export the actual fullCE factors of the same FP32-logit/FP64-probability decomposition. Legacy Phase2733 FP32 probability fullCE factors remain unchanged in gradient_span.'}
    save(out/'full_factor_export.json',report);ledger('binding_same_precision_full_gradient_export',report['seconds'])
    del tail,f;gc.collect();torch.cuda.empty_cache();print('FULL_GRADIENT_EXPORT_PASS',gram_error,flush=True)


if __name__=='__main__':main()
