"""Declared parameter-lens extension; no semantic/causal promotion or refitting."""
import time
import numpy as np
import mlp_tail_goodfire as C


def main():
    import torch
    from safetensors import safe_open
    from tokenizers import Tokenizer
    started=time.monotonic();plan=C.OUT/'readout-plan.json'
    assert not (C.OUT/'B35-readout.json').exists(), 'immutable extension already executed'
    previous=C.read(plan) if plan.exists() else None
    C.save_json(plan,{'schema':'mlp_tail_readout_plan.v1','kind':'real','run_id':C.RUN,
        'source_sha256':C.file_sha(__file__),'model_snapshot':C.SNAPSHOT,'top_each_sign':8,
        'operation':'cosine(unembedding row, native down column), streamed over ALL vocabulary; no truncation during ranking',
        'scope':'Parameter compatibility only, not context-normalized logit effects or a semantic explanation. Extension declared after original prediction results; not new confirmation.',
        'disk_budget_extension_bytes':900000000,
        'resume_after_precomputation_lock_failure':previous,
        'execution':'CPU parameter-only dot products; GPU reserved by a different experiment, no parallel model load'})
    idx=C.read(C.MODEL/'model.safetensors.index.json')['weight_map'];key='model.embed_tokens.weight'
    with safe_open(C.MODEL/idx[key],framework='pt',device='cpu') as f:e=f.get_tensor(key)
    assert C.read(C.MODEL/'config.json')['tie_word_embeddings']
    np.save(C.OUT/'unembedding.npy',e.view(torch.uint16).numpy())
    tok=Tokenizer.from_file(str(C.OUT/'tokenizer.json'))
    torch.set_num_threads(8)
    with torch.inference_mode():
        emb=e.float();emb/=emb.norm(dim=1,keepdim=True).clamp_min(1e-20)
        for bi,b in enumerate(C.BLOCKS):
            w=torch.from_numpy(C.unbits(np.load(C.OUT/f'B{b}-W-down.npy')));w/=w.norm(dim=0,keepdim=True).clamp_min(1e-20)
            rows=[]
            for lo in range(0,C.WIDTH,256):
                values=w[:,lo:lo+256].T@emb.T
                pos,pi=values.topk(8,dim=1);neg,ni=(-values).topk(8,dim=1)
                for i in range(values.shape[0]):
                    def tokens(ids,scores):return [{'token_id':int(t),'text':tok.decode([int(t)],skip_special_tokens=False),'cosine':float(s)} for t,s in zip(ids,scores)]
                    rows.append({'unit':lo+i,'positive':tokens(pi[i].cpu(),pos[i].cpu()),'negative':tokens(ni[i].cpu(),-neg[i].cpu())})
                if lo%2048==0:C.log('parameter_readout_units',bi*C.WIDTH+min(lo+256,C.WIDTH),4*C.WIDTH,started)
            C.save_json(C.OUT/f'B{b}-readout.json',{'schema':'mlp_tail_parameter_readout.v1','block':b,'units':rows,'scope':'全词表参数余弦的两极；未包含上下文RMSNorm、后续层和因果检验。'})
            del w,values;C.log('parameter_readout',bi+1,4,started)
    # Preserve old reports, add an explicitly post-hoc matched-subset audit.
    old=C.read(C.OUT/'report.json');C.save_json(C.OUT/'report-before-readout.json',old)
    cases=C.read(C.OUT/'path-cases.json')['rows'];summaries=[]
    for source,units in C.read(C.OUT/'path-plan.json')['source_units'].items():
        for unit in units:
            for factor in [0.,.5,1.5,2.]:
                subset=[r for r in cases if r['source']==int(source) and r['unit']==unit and r['factor']==factor]
                native={r['doc_index']:r for r in subset if r['condition']=='native_unit' and r['norm_control_valid']}
                control={r['doc_index']:r for r in subset if r['condition']=='matched_noise' and r['norm_control_valid']}
                ids=sorted(native.keys()&control.keys())
                for condition,records in [('native_unit',native),('matched_noise',control)]:
                    rs=[records[d] for d in ids]
                    summaries.append({'source':int(source),'target':int(source)+1,'unit':unit,'factor':factor,'condition':condition,
                        'n':len(rs),'top1_changes':sum(r['base_top1']!=r['edit_top1'] for r in rs),
                        **{out:float(np.mean([r[key] for r in rs])) if rs else None for out,key in [('mean_KL','next_token_KL'),('mean_nll_change','next_token_nll_change'),('mean_target_response_norm','target_A_change_norm')]}})
    old['path']['matched_summaries']=summaries
    old['path']['matched_scope']='Post-hoc descriptive audit on identical document intersection with native and control BF16 norms both within 1%; no inferential promotion.'
    old['parameter_readout']={'units':38912,'vocabulary':len(e),'top_each_pole':8,'semantic_confirmed':False}
    C.save_json(C.OUT/'report.json',old);C.save_json(C.PUBLIC/'report.research.json',old)
    C.save_json(C.OUT/'path-progress.json',{'state':'complete','documents':96,'rows':9216,'scope':'execution complete, not semantic proof'})
    man=C.read(C.OUT/'manifest.json');man['files']=[]
    import re
    for p in sorted(C.OUT.iterdir()):
        if p.is_file() and p.name!='manifest.json':man['files'].append({'path':p.name,'bytes':p.stat().st_size,'sha256':C.file_sha(p),
            'display_required':not re.match(r'B\d+-(G|V|A)\.npy$',p.name) and not p.name.endswith('scores.npz')})
    assert sum(r['bytes'] for r in man['files'])<17*2**30
    C.save_json(C.OUT/'manifest.json',man);C.log('readout_finalized',38912,38912,started)


if __name__=='__main__':main()
