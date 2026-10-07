"""Numerical evidence, independent early-stop execution, and output readout."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','8')
os.environ.setdefault('OMP_NUM_THREADS','8')
import argparse,json,time
from pathlib import Path
import numpy as np
from phase2753_early_scope import ROOT,OUT,OLD,FAMILIES,write,sha,now,snapshot

def early_check():
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    start=time.time();mat=json.loads((OUT/'material.json').read_text(encoding='utf-8'));source={r['id']:r for r in mat['rows']}
    meta=json.loads((OUT/'4B_pilot/chunk_000.json').read_text(encoding='utf-8'))
    with np.load(OUT/'4B_pilot/chunk_000.npz') as z:expected=z['hidden']
    m=AutoModelForCausalLM.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,dtype=torch.bfloat16,attn_implementation='eager').eval().to('cuda')
    class ReachedCutoff(Exception):pass
    saved={};executed=[]
    def hook(index):
        def run(module,args,output):
            executed.append(index)
            if index in (4,8,12):saved[index]=(output[0] if isinstance(output,tuple) else output)[0,-1].float().cpu().numpy()
            if index==12:raise ReachedCutoff()
        return run
    handles=[layer.register_forward_hook(hook(i+1)) for i,layer in enumerate(m.model.layers)]
    errors=[]
    try:
        for i,r in enumerate(meta):
            saved.clear();executed.clear()
            ids=torch.tensor([source[r['id']]['tokenization']['4B']['token_ids']],device='cuda')
            with torch.inference_mode():
                try:m(input_ids=ids,use_cache=False,output_hidden_states=False,logits_to_keep=1)
                except ReachedCutoff:pass
                else:raise AssertionError('Late execution was not stopped.')
            assert executed==list(range(1,13))
            e={str(c):float(np.max(abs(saved[c]-expected[i,c]))) for c in (4,8,12)}
            assert max(e.values())==0
            errors.append(dict(id=r['id'],max_abs=e,executed_blocks=executed.copy()))
    finally:
        for x in handles:x.remove()
    write(OUT/'early_stop_check.json',dict(created_utc=now(),source=snapshot(Path(__file__)),samples=errors,all_bit_exact=True,
        scope='Actual forward abort after block12; blocks13..36 and finalnorm/readout never execute. Full model weights resident; this proves computation boundary, not lower weight memory.',
        elapsed_seconds=time.time()-start))
    print('early boundary verified',len(errors),flush=True)

def readout():
    import torch
    from transformers import AutoTokenizer
    from phase2753_forecast import embedding
    torch.set_num_threads(8);torch.backends.cuda.matmul.allow_tf32=False
    start=time.time();sel=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    rows=json.loads((OUT/'confirmation_groups.json').read_text(encoding='utf-8'))
    methods=list(dict.fromkeys(['zero','family_mean',sel['selected']['static_winner'],sel['selected']['global_winner']]))
    # Selection was fixed before any confirmation; no post-test method choice.
    write(OUT/'readout_execution.json',dict(created_utc=now(),source=snapshot(Path(__file__)),methods=methods,selection_sha256=sha(OUT/'selection.json'),
        reference='FP32 unembedding of stored nativeBF16 finalnorm states; TF32 disabled. NativeBF16 forward compared separately.',
        metric='Full-vocabulary KL(reference||predicted), top1 reference/native fidelity, target condition first-token truth accuracy; no free generation.',
        interaction='Canonical single-token yes-minus-no logit interaction, a linear projection of I; not an interaction probability distribution.'))
    cfg=json.loads((ROOT/'models/hf/qwen3-4b/config.json').read_text())
    assert cfg['tie_word_embeddings']
    wu=embedding().float().to('cuda')
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    yes=tok.encode(' yes',add_special_tokens=False);no=tok.encode(' no',add_special_tokens=False);assert len(yes)==len(no)==1
    direction=(wu[yes[0]]-wu[no[0]]).cpu().numpy()
    native={r['id']:r for path in sorted((OUT/'4B').glob('chunk_*.json')) for r in json.loads(path.read_text(encoding='utf-8'))}
    with np.load(OUT/'confirmation_predictions.npz') as z:
        truth=z['true_h'];interaction=z['true_i'];hp={m:z['h_'+m] for m in methods};ip={m:z['i_'+m] for m in methods}
    reference_pred=[];precision_diff=[];lse_diff=[];metrics={m:dict(kl=[],reference_fidelity=[],native_fidelity=[],accuracy=[],predicted_ids=[]) for m in methods}
    for start_ix in range(0,len(rows),16):
        batch=rows[start_ix:start_ix+16]
        with torch.inference_mode():
            lg=torch.as_tensor(truth[start_ix:start_ix+16],device='cuda')@wu.T
            logp=lg.log_softmax(-1);p=logp.exp();pred=lg.argmax(-1);reference_pred.extend(pred.cpu().tolist())
            for j,r in enumerate(batch):
                n=native[r['id']];precision_diff.append(float((lg[j,n['top20_ids']]-torch.tensor(n['top20_logits'],device='cuda')).abs().max()))
                lse_diff.append(float(lg[j].logsumexp(-1))-n['logsumexp'])
            for method in methods:
                out=torch.as_tensor(hp[method][start_ix:start_ix+16],device='cuda')@wu.T
                ids=out.argmax(-1).cpu().tolist()
                metrics[method]['kl'].extend((p*(logp-out.log_softmax(-1))).sum(-1).cpu().tolist())
                metrics[method]['reference_fidelity'].extend([i==int(pred[j]) for j,i in enumerate(ids)])
                metrics[method]['native_fidelity'].extend([i==native[r['id']]['prediction_id'] for i,r in zip(ids,batch)])
                metrics[method]['accuracy'].extend([tok.decode([i]).strip().lower()==r['expected'] for i,r in zip(ids,batch)])
                metrics[method]['predicted_ids'].extend(ids)
    true_margin=interaction@direction
    for m in methods:metrics[m]['interaction_logit_projection']=list(map(float,ip[m]@direction))
    report={}
    for split in sorted({r['split'] for r in rows}):
        ix=np.array([i for i,r in enumerate(rows) if r['split']==split]);stats={}
        for m in methods:
            d=metrics[m];pm=np.asarray(d['interaction_logit_projection'])
            stats[m]={k:float(np.mean(np.asarray(d[k])[ix])) for k in ('kl','reference_fidelity','native_fidelity','accuracy')}
            stats[m]['interaction_logit_rmse']=float(np.sqrt(np.mean((pm[ix]-true_margin[ix])**2)))
            stats[m]['interaction_sign_agreement']=float(np.mean(np.sign(pm[ix])==np.sign(true_margin[ix])))
        report[split]=stats
    write(OUT/'readout_summary.json',dict(created_utc=now(),splits=report,methods=methods,elapsed_seconds=time.time()-start,
        fp32_vs_native=dict(top1_agreement=float(np.mean([p==native[r['id']]['prediction_id'] for p,r in zip(reference_pred,rows)])),
            sampled_top20_max_logit_error=max(precision_diff),logsumexp_abs_max=float(np.max(abs(np.array(lse_diff)))),
            note='Top20 discrepancy is not a bound over unsaved native logits. KL is between FP32 readouts, not against unsaved full nativeBF16 distributions.'),
        canonical_tokens=dict(yes=yes[0],no=no[0]),source_sha256=sha(Path(__file__))))
    write(OUT/'readout_rows.json',dict(groups=[r['group'] for r in rows],reference_ids=reference_pred,true_interaction_logit_projection=true_margin.tolist(),methods=metrics))
    print(json.dumps(report,indent=2),flush=True)

def quality():
    from phase2753_forecast import data
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows,h,meta,mat=data(OUT)
    done=json.loads((OUT/'4B/capture_done.json').read_text(encoding='utf-8'))
    anchors=json.loads((OUT/'4B/anchors.json').read_text(encoding='utf-8'))
    old=json.loads((OLD/'material.json').read_text(encoding='utf-8'))
    oldnames={n for w in old['worlds'] for n in w['entities']};newnames=[n for w in mat['worlds'] for n in w['entities']]
    assert len(newnames)==len(set(newnames)) and not set(newnames)&oldnames
    assert len(meta)==done['count']==3072 and len({r['id'] for r in meta})==3072
    assert np.isfinite(h).all()
    assert all(a['residual_max']==a['norm_max']==0 for a in anchors)
    byid={r['id']:r for r in mat['rows']}
    fullindex={r['id']:(i,j) for i,r0 in enumerate(rows) for j in range(4) for r in mat['rows'] if r['group']==r0['group'] and r['cond']==j}
    pm=json.loads((OUT/'4B_pilot/chunk_000.json').read_text(encoding='utf-8'))
    with np.load(OUT/'4B_pilot/chunk_000.npz') as z:
        repeat=max(float(np.max(abs(z['hidden'][k]-h[fullindex[r['id']]]))) for k,r in enumerate(pm))
    assert repeat==0
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8')); summary=json.loads((OUT/'confirmation_summary.json').read_text(encoding='utf-8'))
    assert selection['created_utc']<json.loads((OUT/'4B/execution.json').read_text(encoding='utf-8'))['created_utc']
    it=h[:,3].astype(np.float32)-h[:,2]-h[:,1]+h[:,0]
    methods=['family_mean',selection['selected']['static_winner'],selection['selected']['global_winner'],'four8']
    with np.load(OUT/'confirmation_metrics.npz') as z:
        fig,axs=plt.subplots(1,4,figsize=(18,4),sharey=True)
        for ax,split in zip(axs,summary['splits']):
            ix=[i for i,r in enumerate(rows) if r['split']==split]
            for m in methods:ax.plot(np.arange(1,37),np.nanmean(z[m+'_interaction'][ix,1:37],axis=0),label=m)
            ax.axvspan(0,8,color='grey',alpha=.15)
            ax.set_title(split);ax.set_xlabel('Native boundary (36 = finalnorm)');ax.grid(alpha=.25)
        axs[0].set_ylabel('Mean interaction relative L2');axs[0].legend(fontsize=8)
        fig.suptitle('Grey area: early-state methods are NOT forecasting later boundaries there',fontsize=10)
        fig.tight_layout();fig.savefig(OUT/'layer_error.png',dpi=150);plt.close(fig)
    # Keep every coordinate and original order. Rows are fixed split/family means.
    keys=[(s,f) for s in summary['splits'] for f in FAMILIES]
    observed=np.stack([it[[i for i,r in enumerate(rows) if r['split']==s and r['family']==f],36].mean(0) for s,f in keys])
    with np.load(OUT/'confirmation_predictions.npz') as z:
        forecast=z['i_'+selection['selected']['global_winner']]
    predicted=np.stack([forecast[[i for i,r in enumerate(rows) if r['split']==s and r['family']==f]].mean(0) for s,f in keys])
    np.savez(OUT/'full_coordinate_fields.npz',observed=observed,predicted=predicted,residual=observed-predicted)
    vmax=float(max(abs(observed).max(),abs(predicted).max(),abs(observed-predicted).max()))
    fig,axs=plt.subplots(3,1,figsize=(17,9),sharex=True)
    for ax,name,values in zip(axs,['Observed interaction','Frozen early forecast','Observed minus forecast'],[observed,predicted,observed-predicted]):
        im=ax.imshow(values,cmap='RdBu_r',vmin=-vmax,vmax=vmax,aspect='auto',interpolation='nearest');ax.set_title(name)
        ax.set_yticks(range(len(keys)),[s.replace('fresh_','')+'/'+f for s,f in keys],fontsize=7)
        fig.colorbar(im,ax=ax,pad=.01)
    axs[-1].set_xlabel('All 2560 native coordinates, unsorted');fig.tight_layout();fig.savefig(OUT/'native_coordinate_field.png',dpi=150);plt.close(fig)
    row_rms=np.sqrt(np.mean(observed**2,axis=1,keepdims=True))
    fig,axs=plt.subplots(3,1,figsize=(17,9),sharex=True)
    clip_counts={}
    for ax,name,values in zip(axs,['Observed','Forecast','Residual'],[observed,predicted,observed-predicted]):
        normalized=values/np.maximum(row_rms,1e-10);clip_counts[name]=int(np.sum(abs(normalized)>6))
        im=ax.imshow(normalized,cmap='RdBu_r',vmin=-6,vmax=6,aspect='auto',interpolation='nearest')
        ax.set_title(name+' / observed row RMS; display clipped at +/-6 (raw values retained)')
        ax.set_yticks(range(len(keys)),[s.replace('fresh_','')+'/'+f for s,f in keys],fontsize=7);fig.colorbar(im,ax=ax,pad=.01)
    axs[-1].set_xlabel('All 2560 native coordinates, unsorted');fig.tight_layout();fig.savefig(OUT/'native_coordinate_row_rms.png',dpi=150);plt.close(fig)
    write(OUT/'field_metadata.json',dict(rows=[dict(split=s,family=f) for s,f in keys],aggregation='Mean of48 groups per split/family,2 fixed wordings per24worlds; cancellation possible.',
        coordinates='All2560, original order, noPCA/noTopK',scale=dict(type='shared symmetric linear',minimum=-vmax,maximum=vmax),
        normalized_view=dict(denominator='Observed mean-field row RMS, also used for forecast and residual',display_clip=[-6,6],clipped_cells=clip_counts,raw_values_preserved=True),boundary=36,method=selection['selected']['global_winner']))
    # Diagnostic ratio: identifies very small target norms; no exclusion by answer correctness.
    ratio=np.linalg.norm(it[:,36],axis=-1)/np.linalg.norm(h[:,3,36]-h[:,0,36],axis=-1)
    write(OUT/'quality_audit.json',dict(created_utc=now(),source=snapshot(Path(__file__)),unique_worlds=384,unique_names=len(newnames),formal_prompts=3072,
        finite_all_coordinates=True,residual_and_norm_anchor_max=0,pilot_repeat_max=repeat,selection_before_capture=True,
        truth_balance={s:{f:sum(w['relation_truth'] for w in mat['worlds'] if w['cohort']==s and w['family']==f) for f in FAMILIES} for s in summary['splits']},
        interaction_norm_quantiles=np.quantile(np.linalg.norm(it[:,36],axis=-1),[0,.25,.5,.75,1]).tolist(),
        interaction_vs_combined_change_ratio_quantiles=np.quantile(ratio,[0,.25,.5,.75,1]).tolist(),
        full_input_rows=3072,last_position_only=True,early_stop_check_sha256=sha(OUT/'early_stop_check.json')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['early','readout','quality']);a=p.parse_args()
    {'early':early_check,'readout':readout,'quality':quality}[a.mode]()
