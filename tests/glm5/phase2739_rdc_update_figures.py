"""Static scientific figures of full arrays and declared controls; no Top-K."""
import argparse
from rdc_update_common import *


def main(final=False):
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import AsinhNorm
    start=time.monotonic();out=BASE/'figures';out.mkdir(exist_ok=True);items=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
    def finish(fig,name,title,scope,sources):
        fig.suptitle(title,fontsize=14);fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/name,dpi=150);plt.close(fig)
        items.append({'path':name,'title':title,'scope':scope,'source_files':sources,'sha256':sha(out/name)})
    def heat(ax,a,title,xlabel='Own coordinate (original order)',ylabel='Original row',signed=True,asinh=False,shared_limit=None):
        a=np.ma.asarray(a);limit=max(float(np.abs(a).max()) if shared_limit is None else shared_limit,1e-20)
        kwargs={'cmap':'coolwarm','vmin':-limit,'vmax':limit} if signed else {'cmap':'viridis'}
        if asinh:kwargs={'cmap':'coolwarm','norm':AsinhNorm(linear_width=1,vmin=-limit,vmax=limit)}
        cmap=plt.get_cmap(kwargs['cmap']).copy();cmap.set_bad('#c4c4c4');kwargs['cmap']=cmap
        im=ax.imshow(a,aspect='auto',interpolation='nearest',rasterized=True,**kwargs);ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
        bar=plt.colorbar(im,ax=ax,fraction=.022,pad=.018)
        if asinh:
            mid=float(np.sinh(np.arcsinh(limit)/2));ticks=[-limit,-mid,0,mid,limit]
            bar.set_ticks(ticks,labels=[f'{v:.3g}' for v in ticks]);bar.minorticks_off()
    # Original coordinate order, all2560x2560 entries, no threshold.
    for block in (16,35):
        file=f'analysis/new_natural_H12_to_block{block}_full_coordinates.npz'
        with np.load(BASE/file) as z:
            for key in ('cross_covariance','cross_correlation'):
                fig,ax=plt.subplots(figsize=(12,9));heat(ax,z[key],f'384 content boundaries; {key}',xlabel=f'Block{block} MLP output coordinate',ylabel='H12 coordinate',asinh=key=='cross_covariance')
                finish(fig,f'natural_block{block}_{key}.png',f'Natural H12 to block{block}: all native coordinates','Every2560x2560entry; covariance uses asinh(unit scale1), correlation linear. Pixel resize is display only. New-test means are descriptive, not fitted inputs.',[file])
    file='language_analysis/all_condition_profiles.npz'
    with np.load(BASE/file) as z:
        for key in ('raw_H','RMS_H'):
            fig,axes=plt.subplots(2,1,figsize=(15,9))
            limit=float(np.abs(z[key][:,12]).max())
            for anchor,ax in enumerate(axes):heat(ax,z[key][:,12,anchor],f'All60condition means at H12 / anchor{anchor}',ylabel='Fixed family-language-style-split order',asinh=key=='raw_H',shared_limit=limit)
            finish(fig,f'language_{key}_H12_profiles.png',f'Bilingual full-coordinate condition profiles: {key}','All2560coordinates;60means of declared groups, not60independent semantic relations. Raw and per-vector RMS separate.',[file,'language_analysis/condition_profile_labels.json'])
        fig,axes=plt.subplots(3,1,figsize=(16,10))
        for i,ax in enumerate(axes):heat(ax,z['raw_MLP_activation'][:,i,1],f'Block{(6,16,35)[i]}: all9728units',xlabel='MLP unit (original order)',ylabel='All60conditions',asinh=True)
        finish(fig,'language_all_unit_profiles.png','Every MLP unit across bilingual conditions','Raw activation means, asinh(unit scale1), all9728units. Identical unit index across layers does not imply identical function.',[file])
    for layer,anchor in ((12,0),(36,1)):
        file=f'language_analysis/all_pair_cosine_H{layer}_anchor{anchor}.npz'
        with np.load(BASE/file) as z:
            fig,ax=plt.subplots(figsize=(10,9));heat(ax,z['cosine'],f'All640x640pairs; H{layer}, anchor{anchor}',xlabel='Material index',ylabel='Material index')
            finish(fig,f'language_pair_H{layer}_a{anchor}.png','Bilingual sample geometry, without coordinate truncation','Full native dot products, material order fixed. Shared prompt/template/answer class can create blocks; similarity is not mechanism.',[file,'language_material.json.gz'])
    lr=read(BASE/'language_analysis/result.json');families=sorted({r['family'] for r in lr['comparisons']})
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    for anchor,ax in enumerate(axes):
        for family in families:
            rr=[r for r in lr['comparisons'] if r['anchor']==anchor and r['family']==family and r['target_language']=='zh' and r['target_style']=='direct']
            ax.plot([r['layer'] for r in rr],[r['same_semantic_group_over_same_family_truth']['mean'] for r in rr],marker='o',label=family)
        ax.axhline(0,color='black',lw=.8);ax.set(title=f'Anchor{anchor}',xlabel='Native H layer',ylabel='Same-group cosine minus matched truth/family')
    axes[1].legend(fontsize=8);finish(fig,'language_matched_semantic_trajectories.png','Cross-language similarity: family/truth matched, lexical identity uncontrolled','Eight heldout semantic groups per family, six declared layers; full-coordinate dot products. Body anchor is last fully contained token, not punctuation. Connecting measurements is descriptive, not a fitted transfer law; see separate token-identity coverage audit.',['language_analysis/result.json','language_identity/result.json'])
    controls=read(BASE/'language_identity/result.json')['comparisons'];fig,axes=plt.subplots(1,2,figsize=(14,6))
    for ax,control,title in zip(axes,('family_truth','plus_current_target_token_and_exact_position'),('Family + truth only','Also current target token + exact position')):
        for i,family in enumerate(families):
            r=next(r for r in controls if (r['layer'],r['anchor'],r['target_language'],r['target_style'],r['family'],r['control'])==(12,0,'zh','direct',family,control))
            stat=r['same_group_advantage']
            if stat:
                mean=stat['mean'];low,high=stat['interval95'];ax.errorbar(i,mean,yerr=[[max(mean-low,0)],[max(high-mean,0)]],fmt='o',capsize=4,color='#287d8e')
            else:ax.text(i,.68,'No estimate\n0 / 8 controls',transform=ax.get_xaxis_transform(),ha='center',fontsize=9,color='#934c31')
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(len(families)),[f.replace('_','\n') for f in families],fontsize=9)
        ax.set(title=title,xlim=(-.5,len(families)-.5),ylabel='Same-group cosine advantage (95% cluster interval)')
    finish(fig,'language_token_identity_coverage.png','H12 body anchor: missing identity controls are not zero effects','ENdirect to ZHdirect; five held families, up to8groups each. Right-panel word-sense/long-role are unidentifiable with current token match, not effects estimated at zero. Post-observation same-data audit, not new confirmation.',['language_identity/result.json'])
    causal=read(BASE/'causal_anchor/result.json');fig,axes=plt.subplots(1,2,figsize=(14,5))
    pairs=gzread(BASE/'causal_anchor/pair_identity.json.gz')
    with np.load(BASE/'causal_anchor/all_pair_all_layer_differences.npz') as z:
        for language,color in (('en','#287d8e'),('zh','#c86843')):
            ix=[i for i,r in enumerate(pairs) if r['language']==language];a=z['relative_RMS'][ix];x=np.arange(37)
            axes[0].plot(x,a.mean(0),label=language,color=color);axes[0].fill_between(x,a.min(0),a.max(0),alpha=.13,color=color)
            axes[1].plot(x,z['bit_equal'][ix].mean(0),label=language,color=color)
    axes[0].set(xlabel='All native H layers',ylabel='Earlier body relative RMS difference',title='Mean and observed range (not confidence interval)')
    axes[1].set(xlabel='All native H layers',ylabel='Bit-exact body-state fraction',ylim=(-.03,1.03),title='320 identical earlier token prefixes')
    axes[1].legend();finish(fig,'same_body_prefix_execution_numerics.png','Later answer-style instructions cannot causally change earlier body meaning','320same-prefix pairs, every coordinate at all37body layers. Stored B1full-prompt BF16 shapes differ in total length; this audit does not isolate rounding/alignment contributions or rerun common truncated-prefix shapes. Not a semantic style effect.',['causal_anchor/result.json','causal_anchor/all_pair_all_layer_differences.npz'])
    if (BASE/'causal_replay/result.json').exists():
        replay=read(BASE/'causal_replay/result.json');fig,axes=plt.subplots(1,2,figsize=(14,5));values=[];equal=[]
        for r in replay['records']:
            with np.load(BASE/'causal_replay/fields'/f'{r["sample_id"]}.npz') as z:
                values.append(z['prefix_vs_old_direct_relative_RMS']);equal.append(z['same_length_suffix_bit_equal'])
        values=np.array(values);equal=np.array(equal);x=np.arange(37)
        axes[0].plot(x,values.mean(0),color='#287d8e');axes[0].fill_between(x,values.min(0),values.max(0),alpha=.15,color='#287d8e')
        axes[0].set(xlabel='All native H layers',ylabel='Prefix-only vs original full-prompt relative RMS',title='Mean and observed range, 80 unique prefixes')
        axes[1].plot(x,equal.mean(0),color='#c86843');axes[1].set(xlabel='All native H layers',ylabel='Bit-exact body-state fraction',ylim=(-.03,1.03),title='Different future suffix, identical padded shape')
        finish(fig,'prefix_only_replay_and_future_suffix_control.png','Native prefix-only replay and matched-length causal control','80unique held body prefixes,40semantic groups,320native forwards. Same-prefix repeat plus right-padded same-length future-suffix control; frozen predictors evaluated without refitting. Reuses original groups, not independent semantic confirmation.',['causal_replay/result.json'])
        fig,axes=plt.subplots(1,2,figsize=(14,7),sharey=True)
        for block,ax,decoder in zip((16,35),axes,('direct','native_joint')):
            rr=[r for r in replay['directed_minus_query'] if r['block']==block]
            for i,r in enumerate(rr):
                mean=r['directed_minus_query']['mean'];lo,hi=r['directed_minus_query']['interval95']
                ax.errorbar(mean,i,xerr=[[max(mean-lo,0)],[max(hi-mean,0)]],fmt='o',capsize=3,color='#287d8e' if r['cohort'].endswith('_en/body_prefix') else '#c86843')
            ax.set_yticks(range(len(rr)),[r['cohort'].replace('/body_prefix','').replace('_',' ') for r in rr])
            ax.axvline(0,color='black',lw=.8);ax.set(title=f'Block {block} / frozen {decoder} decoder',xlabel='Relative squared error: directed minus query',xlim=(-.075,.075))
        axes[0].invert_yaxis()
        finish(fig,'prefix_only_family_forecast_comparison.png','Fixed prefix-only prediction has family- and readout-dependent value','Negative favors directed_rms over query;8semantic groups per family/language,95%source-cluster intervals. Different decoders confound isolated layer interpretation. Better than query need not beat training-mean baseline; post-observation diagnostic, no multiple-testing or semantic-mechanism claim.',['causal_replay/result.json'])
    path=read(BASE/'native_paths/result.json')['reports'][2]['sample_id'];file=f'native_paths/fields/{path}.npz'
    with np.load(BASE/file) as z:
        for block in (16,35):
            fig,axes=plt.subplots(4,1,figsize=(16,12));n=int(z['positions'][-1])+1
            for ax,key in zip(axes,('source_attention_write','source_gate_read','source_up_read','source_MLP_write')):
                heat(ax,z[f'L{block}_{key}'][-1,:n],key,xlabel='All own coordinates / MLP units',ylabel='Visible source token',asinh=True)
            finish(fig,f'native_source_block{block}.png',f'Actual source-token accounting, block{block}: {path}','Every visible source, all2560coordinates or9728units; asinh(unit scale1). Observed attention/RMS/gates, symmetric allocation and explicit remainders, not unique cause.',[file,'native_paths/result.json'])
    file='learning/gram.npz'
    with np.load(BASE/file) as z:
        fig,axes=plt.subplots(1,3,figsize=(17,5))
        for ax,key in zip(axes,('full','content','format')):
            a=z[key];den=np.sqrt(np.maximum(a.diagonal(),0));a=a/np.maximum(den[:,None]*den[None,:],1e-30);heat(ax,a,key,xlabel='768program expressions',ylabel='768program expressions')
        finish(fig,'full_gradient_cosine_matrices.png','All74711040parameter inner products, three distinct objectives','All768x768gradient pairs via exact native factors; normalization is gradient norm, no hard rank cutoff.',['learning/gram.npz'])
    file='learning/all_training_constraint_matrices.npz'
    with np.load(BASE/file) as z:
        fig,axes=plt.subplots(1,2,figsize=(12,5));heat(axes[0],z['normalized_format_gram'],'All192format constraints',xlabel='Constraint row',ylabel='Constraint row');heat(axes[1],z['normalized_code_content_gram'],'All96code content directions',xlabel='Code row',ylabel='Code row')
        finish(fig,'training_full_constraint_grams.png','Continuous ridge over every declared training factor','Full Gram matrices; no discarded weak direction and no pure-semantic rank claim.',[file])
    finite=read(BASE/'learning/finite_result.json');r=next(r for r in finite['updates'] if r['direction']=='content_format_constrained_1e-6' and r['FP32_parameter_step']==.02)
    rr=[r for r in r['reports'] if r['split']=='mixed_holdout'];fig,ax=plt.subplots(figsize=(10,5));x=np.arange(len(rr))
    for shift,part,color in ((-.18,'content','#287d8e'),(.18,'format','#c86843')):ax.bar(x+shift,[r[part+'_loss_delta'] for r in rr],.36,label=part,color=color)
    ax.axhline(0,color='black',lw=.8);ax.set_xticks(x,[r['representation'] for r in rr]);ax.set_ylabel('Actual finite loss change');ax.legend()
    finish(fig,'finite_content_format_transfer.png','Format-constrained finite update on unseen mixed programs','32semantic groups per expression, FP32parameter norm0.02. Chinese format deterioration retained; these are next-token candidate losses, not complete-answer scores.',['learning/finite_result.json'])
    middle=read(BASE/'middle_training/result.json');fig,axes=plt.subplots(2,1,figsize=(13,8))
    for r in middle['runs']:
        label=f'{r["condition"]}/{r["seed"]}';axes[0].plot([t['step'] for t in r['trace']],[t['actual_batch_delta']['format'] for t in r['trace']],label=label)
        axes[1].plot([t['step'] for t in r['trace']],[t['predicted_batch_format_delta'] for t in r['trace']],label=label)
    for ax,title in zip(axes,('Actual finite batch format change','Pre-step local predicted format change')):ax.set(title=title,xlabel='Training step',ylabel='Format loss change');ax.axhline(0,color='black',lw=.8)
    axes[0].legend(fontsize=8);finish(fig,'middle_local_vs_finite_format.png','Local format constraints do not make finite learning format-invariant','Same4draws within each seed/condition,32steps; full native suffix with stated FP32/BF16 bridge. Different actual and derivative quantities shown separately.',['middle_training/result.json'])
    fig,ax=plt.subplots(figsize=(11,6));reps=('mixed_en','mixed_en_reordered','mixed_python','mixed_zh')
    for i,r in enumerate(middle['runs']):
        rr=[x for x in r['checkpoints'][-1]['reports'] if x['split']=='mixed_holdout'];lookup={x['cohort']:x for x in rr}
        ax.bar(np.arange(4)+(i-1.5)*.2,[lookup[k]['content_loss_delta'] for k in reps],.2,label=f'{r["condition"]}/{r["seed"]}')
    ax.set_xticks(np.arange(4),['EN','EN reorder','Python','ZH']);ax.set_ylabel('Content loss change, checkpoint32');ax.legend(fontsize=8)
    finish(fig,'middle_heldout_content.png','Actual block16learning on unseen mixed compositions','Two seeds, two objectives;74711040trained scalars. Final accumulated/deployed norms differ; no generic natural-language improvement claim.',['middle_training/result.json'])
    fresh=read(BASE/'fresh_graph/result.json');fig,axes=plt.subplots(1,2,figsize=(15,6));controls=('query','shuffled_rms','position_rms','square_rms');groups=sorted({(r['split'],r['cohort']) for r in fresh['matched_controls']})
    for ax,block in zip(axes,(16,35)):
        for i,c in enumerate(controls):
            rr=[next(r for r in fresh['matched_controls'] if (r['block'],r['control'],r['split'],r['cohort'])==(block,c,*g)) for g in groups]
            mean=np.array([r['source_cluster_advantage']['mean'] for r in rr]);ci=np.array([r['source_cluster_advantage']['interval95'] for r in rr])
            ax.errorbar(np.arange(4)+(i-1.5)*.16,mean,yerr=np.maximum(np.stack([mean-ci[:,0],ci[:,1]-mean]),0),fmt='o',capsize=3,label=c)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(4),[s.replace('fresh_','')+'\n'+c for s,c in groups]);ax.set(title=f'Block{block}',ylabel='Directed relative error minus control')
    axes[1].legend(fontsize=8);finish(fig,'fresh_RMS_matched_controls.png','Fresh natural confirmation with identical RMS normalization','128new windows,384content boundaries; each point32windows, confidence intervals cluster by actual source documents, which may contribute multiple windows. Negative favors directed. Controls frozen before new captures, no winner re-selection.',['fresh_graph/result.json'])
    pred=read(BASE/'language_prediction/result.json');fig,ax=plt.subplots(figsize=(17,5));matrix=np.array([[r['relative_mse'] for r in item['reports']] for item in pred['records']]);heat(ax,matrix,'All held family/language/style/anchor groups',xlabel='Fixed group index (see result reports)',ylabel='Frozen predictor row',signed=False)
    ax.set_yticks(range(len(pred['records'])),[f'{r["block"]}/{r["kernel"]}/{r["decoder"]}' for r in pred['records']]);finish(fig,'language_frozen_prediction.png','English natural-trained predictor on five bilingual families','160held expressions,320boundaries; all2560output coordinates scored. Training/validation only frozen before language capture; MSEnear1is not successful mechanism extraction.',['language_prediction/result.json'])
    psr=read(BASE/'predictive_state/result.json');fig,ax=plt.subplots(figsize=(10,5));xx=np.arange(len(psr['summary']))
    ax.plot(xx,[r['near_mean_KL'] for r in psr['summary']],'o-',label='Near descriptor');ax.plot(xx,[r['far_mean_KL'] for r in psr['summary']],'o-',label='Far descriptor')
    ax.set_xticks(xx,[repr(r['probe']) for r in psr['summary']]);ax.set(ylabel='Mean symmetric full-vocabulary KL',xlabel='Fixed continuation probe');ax.legend()
    finish(fig,'reachable_prefix_future_probes.png','Actual reachable-prefix proximity and future distributions','32centers,64near/far pairs,63unique prefixes; near is not equal. Current similarity advantage does not persist uniformly after fixed continuations.',['predictive_state/result.json'])
    file='moment_boundary/counterexample_full_coordinates.npz'
    with np.load(BASE/file) as z:
        fig,axes=plt.subplots(3,1,figsize=(15,10));heat(axes[0],z['history_a'],'Synthetic history A:32positions,2560coordinates',ylabel='Position');heat(axes[1],z['history_b'],'Synthetic history B: same declared finite moments',ylabel='Position')
        heat(axes[2],z['attention_responses'][:,0]-z['attention_responses'][:,1],'Softmax responses A-B for four fixed queries',ylabel='Query index')
        finish(fig,'synthetic_moment_boundary.png','Finite-moment counterexample, explicitly NOT native language states','All vectors and sources, no threshold. Powers1..4 and position powers0..4 agree in the constructed certificate; nonzero-query softmax responses differ. Not a reachable-state collision proof.',[file,'moment_boundary/result.json'])
    fig,ax=plt.subplots(figsize=(9,5));rr=psr['native_KV'];ax.plot([r['length'] for r in rr],[r['actual_KV_bytes']/2**20 for r in rr],'o-',label='Measured all K/V tensors')
    ax.set(xlabel='Native prefix tokens',ylabel='KV tensor storage (MiB)');ax.legend();finish(fig,'native_KV_storage.png','Bounded native KV accounting, no100Kcompression claim','512/1024/2048tokens; all36layers,8KVheads,128head width,BF16. Model weights/temporaryattention workspaces excluded from this exact tensor count.',['predictive_state/result.json'])
    behavior_path=BASE/'behavior_analysis/result.json'
    behavior=read(behavior_path if behavior_path.exists() else BASE/'behavior_analysis/preliminary.json')
    if final:behavior['manual_summaries']=read(BASE/'manual_terminal_audit/result.json')['summaries']
    for mode,kind,name,score_table in (('own_history','controlled_language','own_bilingual_terminal_answers.png','summaries'),('long_answers','controlled_program','long_program_terminal_answers.png','summaries'),('long_answers','controlled_program','long_program_format_aware_answers.png','format_aware_summaries'),('long_answers','controlled_program','long_program_manually_audited_answers.png','manual_summaries')):
        rr=[r for r in behavior.get(score_table,[]) if r['mode']==mode and r['granularity']=='kind' and r['cohort']==kind]
        if not rr:continue
        fig,ax=plt.subplots(figsize=(15,6));bottom=np.zeros(len(rr))
        for metric,label,color in (('correct_and_stopped','Correct parsed + EOS','#2d877e'),('wrong_parsed_and_stopped','Wrong parsed + EOS','#c85b4a'),('censored','Censored','#a6afb5')):
            v=np.array([r[metric] for r in rr]);ax.bar(range(len(rr)),v,bottom=bottom,label=label,color=color);bottom+=v
        v=np.array([r['rows'] for r in rr])-bottom;ax.bar(range(len(rr)),v,bottom=bottom,label='Other/unparsed stopped',color='#d9c28a')
        ax.set_xticks(range(len(rr)),[r['branch'].replace('content_format_constrained','CF').replace('batch_format_constrained','BCF').replace('middle_','mid_') for r in rr],rotation=35,ha='right');ax.set_ylabel('Trajectories');ax.legend(fontsize=8)
        label={'format_aware_summaries':'Secondary format-aware','manual_summaries':'Unblinded residual terminal review'}.get(score_table,'Original terminal grammar')
        finish(fig,name,f'{label} outcomes: {mode}','EOS and censoring separate; unparsed/censored is not classified semantically wrong. Repeated branches share sources. Secondary grammar and residual manual review are post-outcome measurement layers on the same outputs, not new generations or independent confirmation; neither grades the reasoning chain.',['manual_terminal_audit/result.json' if score_table=='manual_summaries' else ('behavior_analysis/result.json' if behavior.get('final') else 'behavior_analysis/preliminary.json')])
    if (BASE/'scale_analysis/result.json').exists():
        sr=read(BASE/'scale_analysis/result.json')
        for key in ('early_query','source_mean','MLP_activation'):
            fig,axes=plt.subplots(1,3,figsize=(16,5))
            for model,ax in zip(('qwen4','qwen14','glm4'),axes):
                with np.load(BASE/'scale_analysis'/f'{model}_all_sample_cosine.npz') as z:heat(ax,z[key],model,xlabel='128source sample index',ylabel='128source sample index')
            finish(fig,f'scale_all_sample_{key}.png',f'Same materials, own model coordinates: {key}','All coordinate/unit dot products; matched source order. Equal sample geometry is not equal parameter coordinates or proof model size caused differences.',['scale_analysis/result.json'])
        fig,axes=plt.subplots(3,1,figsize=(16,9))
        for model,ax in zip(('qwen4','qwen14','glm4'),axes):
            with np.load(BASE/'scale_analysis'/f'{model}_all_unit_statistics.npz') as z:
                names=[k for k in z.files if k.endswith('__gate_up_correlation')]
                defined=np.array([z[k.replace('__gate_up_correlation','__correlation_defined')] for k in names])
                heat(ax,np.ma.array(np.array([z[k] for k in names]),mask=~defined),model,xlabel='Every native MLP unit',ylabel='Cohort index')
            ax.set_yticks(range(len(names)),[k.split('__')[0] for k in names],fontsize=7)
        finish(fig,'scale_all_unit_gate_up.png','All-unit gate/up correlation in each original model','Full unit indices, no alignment across models. Each statistic uses all declared cohort rows; undefined constant-denominator cells are gray, not estimated zero correlations. Original defined masks remain in the NPZ.',['scale_analysis/result.json'])
    if final:assert (BASE/'scale_analysis/result.json').exists() and behavior.get('final')
    save(out/'index.json',{'timestamp':stamp(),'source':snapshot(__file__),'final':final,'figures':items,'count':len(items),'seconds':time.monotonic()-start,
      'scope':'Static scientific plots of real frozen arrays, with explicitly labeled synthetic certificate. No generative imagery, PCA,Top-Kor silent coordinate truncation.'})
    ledger('scientific_figures_final' if final else 'scientific_figures_preliminary',time.monotonic()-start);print('FIGURES_DONE',len(items),'final',final,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');main(p.parse_args().final)
