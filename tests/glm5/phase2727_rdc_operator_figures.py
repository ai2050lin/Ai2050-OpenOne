"""Evidence figures for probability compilation, matched native models and own history."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdc_operator_common import *


def main():
    start=time.monotonic();out=BASE/'figures';index=read(out/'index.json');new=[]
    plt.rcParams.update({'font.size':9,'axes.titlesize':11,'figure.dpi':120})
    colors=['#256c95','#d07835','#46967a']
    def keep(fig,name,title,scope):
        fig.tight_layout();fig.savefig(out/name,bbox_inches='tight');plt.close(fig)
        new.append({'path':name,'title':title,'scope':scope,'sha256':sha(out/name)})
    compiled=read(BASE/'compiled/result.json');local=read(BASE/'confirmation/result.json')['reports']
    probability=compiled['local']['confirmation'];blocks=(6,16,34);xx=np.arange(3)
    fig,axes=plt.subplots(2,2,figsize=(14,9))
    for j,typ in enumerate(('global','selected')):
        lm=[next(r['relative_MSE'] for r in local if r['block']==b and r['stratum']=='all' and r['name']==('frozen_gate_global' if typ=='global' else 'frozen_choice')) for b in blocks]
        pp=[next(r for r in probability if r['name']==f'L{b}_{typ}') for b in blocks]
        axes[0,0].bar(xx+(j-.5)*.34,lm,.33,label=typ,color=colors[j])
        axes[0,1].bar(xx+(j-.5)*.34,[r['raw_KL'] for r in pp],.33,label=typ,color=colors[j])
        axes[1,1].bar(xx+(j-.5)*.34,[r['argmax_agreement'] for r in pp],.33,label=typ,color=colors[j])
    for a in (axes[0,0],axes[0,1],axes[1,1]):
        a.set_xticks(xx,[f'block{b}' for b in blocks]);a.legend();a.grid(axis='y',alpha=.2)
    axes[0,0].set(title='512 confirmation anchors: local MLP approximation',ylabel='Mean token-relative full-coordinate MSE')
    axes[0,1].set(title='Same sources: native remaining network, full vocabulary',ylabel='KL(native || approximation)')
    axes[1,1].set(title='One query replacement only: next-token argmax',ylabel='Native argmax agreement',ylim=(0,1))
    anames=['native','joint_global','joint_selected'];auto=compiled['autonomous']
    axes[1,0].bar(anames,[auto[n]['mean_repeated_4gram_fraction'] for n in anames],color=colors)
    axes[1,0].set(title='64 independent histories, 48 generated tokens',ylabel='Mean repeated 4-gram fraction')
    axes[1,0].grid(axis='y',alpha=.2)
    keep(fig,'compiled_probability_and_history.png','局部更准确，不保证概率与各自历史更准确',
        'Frozen selection by local MSE; all512 confirmation anchors/all151936 logits. Local fits receive actual x from full-window collection, compilation reruns identical prefix shapes. Autonomous branches keep independent KV; other modules remain native. Different error metrics are not numerically interchangeable.')
    if all((BASE/'scale'/m/'result.json').exists() for m in ('qwen4','qwen14','glm4')):
        fig,axes=plt.subplots(1,3,figsize=(15,5))
        for a,m in zip(axes,('qwen4','qwen14','glm4')):
            r=read(BASE/'scale'/m/'result.json');rr=r['reports'];bs=r['own_blocks']
            for j,name in enumerate(('global','selected')):
                vals=[next(s['relative_MSE'] for s in rr if s['stage']=='confirmation' and s['block']==b and s['name']==('frozen_gate_global' if name=='global' else r['choices'][str(b)])) for b in bs]
                a.bar(np.arange(3)+(j-.5)*.34,vals,.33,label=name,color=colors[j])
            a.set_xticks(range(3),[f'block{b}' for b in bs]);a.set(title=f'{m}: native width {r["native_width"]}',ylabel='Mean token-relative full-coordinate MSE')
            a.legend();a.grid(axis='y',alpha=.2)
        keep(fig,'matched_models_native_operators.png','相同来源与样本数，保留不同模型原生空间的局部复查',
            'Each model128sources:64train/32validation/32confirmation,2anchors each; six candidates. Each own coordinate width retained. 4B matched subset is re-analysis; proportional middle block18 differs from primary4B block16. No cross-model coordinate alignment or controlled scaling-law claim.')
    if (BASE/'operations/result.json').exists():
        r=read(BASE/'operations/result.json')
        with np.load(BASE/'operations/all_query_order_moments.npz') as z:d=np.sqrt(np.maximum(z['mean_coordinate_squared_difference'],0))
        scale=max(float(np.sqrt(np.mean(d*d))),1e-12);fig,axes=plt.subplots(1,2,figsize=(16,6),gridspec_kw={'width_ratios':[2.5,1]})
        im=axes[0].imshow(np.arcsinh(d/scale),aspect='auto',origin='lower',interpolation='nearest',cmap='viridis')
        axes[0].set(title=f'64 real questions: RMS query change, scale={scale:.5g}',xlabel='Every native residual coordinate 0..2559',ylabel='Every native boundary H0..H36')
        fig.colorbar(im,ax=axes[0],fraction=.025,pad=.02,label='asinh(RMS difference / stated global scale)')
        axes[1].bar(['context first','question first'],[r['context_first_EM'],r['question_first_EM']],color=colors[:2])
        axes[1].set(title='Same complete contexts, questions and reference answers',ylabel='Whole normalized answer match',ylim=(0,1));axes[1].grid(axis='y',alpha=.2)
        keep(fig,'natural_order_complete_query_change.png','自然材料顺序变化的全层全坐标响应与答案',
            'All37x2560 coordinates in original order, no pruning. Same last query-token ID is audited separately; order also changes positions and causal availability. Response difference is not an isolated semantic direction. String match is not exhaustive semantic correctness.')
    if (BASE/'metric_followup/result.json').exists():
        r=read(BASE/'metric_followup/result.json');rr=[s for p in sorted((BASE/'metric_followup/commits').glob('*.json')) for s in read(p)['rows']]
        fig,axes=plt.subplots(2,2,figsize=(14,10));names=['joint_global','joint_selected','output_selected_hybrid']
        for j,name in enumerate(names):
            selected=[s for s in rr if s['name']==name]
            axes[0,0].scatter([s['exact_KL'] for s in selected],[s['adaptive_integral'] for s in selected],s=6,alpha=.35,label=name,color=colors[j])
            axes[0,1].scatter([s['exact_KL'] for s in selected],[s['endpoint_Fisher_half_variance'] for s in selected],s=6,alpha=.35,color=colors[j])
        for a,title,ylabel in [(axes[0,0],'Full logit path: numerical identity, all1920 comparisons','Adaptive weighted-variance integral'),(axes[0,1],'Endpoint quadratic is only a local approximation','Half endpoint Fisher variance')]:
            a.set_xscale('symlog',linthresh=.001);a.set_yscale('symlog',linthresh=.001)
            lo=min(s['exact_KL'] for s in rr);hi=max(s['exact_KL'] for s in rr);a.plot([max(0,lo),hi],[max(0,lo),hi],'k--',linewidth=.8)
            a.set(title=title,xlabel='Exact full-vocabulary KL',ylabel=ylabel);a.grid(alpha=.2)
        axes[0,0].legend(fontsize=8)
        for j,name in enumerate(names):
            vals=[next(s['exact_KL'] for s in r['summaries'] if s['scope']==scope and s['name']==name) for scope in ('natural_reanalysis','QA_query_transfer')]
            axes[1,0].bar(np.arange(2)+(j-1)*.24,vals,.23,label=name,color=colors[j])
        axes[1,0].set_xticks(range(2),['natural512 (re-analysis)','QA128 (query transfer)']);axes[1,0].set(ylabel='Mean exact full-vocabulary KL',title='Validation-output-selected composition, fixed before this test');axes[1,0].legend(fontsize=8)
        paths=[read(p) for p in sorted((BASE/'metric_followup/autonomous').glob('*.json'))]
        for p in paths:
            axes[1,1].plot([s['step'] for s in p['steps']],[s['native_same_history_KL'] for s in p['steps']],color=colors[0] if p['language']=='en' else colors[1],alpha=.16,linewidth=.7)
        axes[1,1].set(title='All64 hybrid paths: separate native KV on chosen history',xlabel='Own generation step0..47',ylabel='Native-same-history KL');axes[1,1].grid(alpha=.2)
        keep(fig,'output_metric_and_independent_history.png','完整词表误差几何与各自历史上的误差接续',
            'All640 queries x3 configurations, every151936 vocabulary entry, no Top-K. Known log-partition identity, not new mathematics. Natural confirmation is re-analysis; QA task boundary transfer uses previously observed source behavior. All64 trajectories shown; native diagnostic never refreshes approximate KV.')
        example=paths[0];sid=example['sample_id']
        with np.load(BASE/'metric_followup/autonomous_fields'/f'{sid}.npz') as z:
            a=unbits(z['approximate_query_fields']).astype(float);b=unbits(z['native_same_chosen_history_query_fields']).astype(float)
        difference=a-b;fig,axes=plt.subplots(2,2,figsize=(15,8))
        for j,ax in enumerate(axes.flat):
            d=difference[:,j];scale=max(float(np.sqrt(np.mean(d*d))),1e-12);v=np.arcsinh(d/scale);limit=float(np.abs(v).max())
            im=ax.imshow(v,aspect='auto',origin='lower',cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
            ax.set(title=f'{["block6 output","block16 output","block34 output","final postnorm"][j]}, scale={scale:.5g}',xlabel='Every native residual coordinate0..2559',ylabel='Every generated query step')
            fig.colorbar(im,ax=ax,fraction=.025,pad=.02,label='asinh(signed difference / panel RMS)')
        keep(fig,'own_history_all_coordinate_errors.png','同一已选 token 历史上，四个边界的完整误差场',
            f'{sid}, first canonical source (not selected by effect). Each branch has its own KV; query vectors compare approximate vs native computation on exactly the same chosen token sequence. Full arrays for all64 sources remain queryable; panel RMS is displayed, coordinate order is unchanged.')
        geometries=r['complete_readout_geometry'];assert len(geometries)==2
        fig,axes=plt.subplots(1,2,figsize=(15,7))
        for ax,geo in zip(axes,geometries):
            with np.load(BASE/'metric_followup/readout_geometry'/f'{geo["query_id"]}.npz') as z:g=z['G_full_native_coordinates']
            scale=max(float(np.sqrt(np.mean(g*g))),1e-12);value=np.arcsinh(g/scale);lim=float(np.abs(value).max())
            im=ax.imshow(value,origin='lower',cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest',aspect='equal')
            ax.set(title=f'{geo["scope"]}: full output sensitivity\nscale={scale:.5g}',xlabel='Native postnorm coordinate0..2559',ylabel='Native postnorm coordinate0..2559')
            fig.colorbar(im,ax=ax,fraction=.03,pad=.025,label='asinh(G_ij / full-matrix RMS)')
        keep(fig,'complete_output_coordinate_coupling.png','完整词表读出中的全坐标相互作用矩阵',
            'All2560x2560 entries, computed from all151936native vocabulary rows. Two predetermined queries. Known real-valued Fisher pullback in postnorm coordinates; diagonal/cross contributions and native BF16 logit differences separately audited. Image pixels merge entries only for display; full matrices retained without Top-K or coordinate sorting.')
    if (BASE/'metric_followup/population_geometry/result.json').exists():
        folder=BASE/'metric_followup/population_geometry';population=read(folder/'result.json')
        fig,axes=plt.subplots(3,2,figsize=(15,18))
        for row,geo in enumerate(population['geometry']):
            with np.load(folder/f'{geo["group"]}_complete_geometry.npz') as z:
                for col,key in enumerate(('G_average_full_native','G_mixture_full_native')):
                    g=z[key];scale=max(float(np.sqrt(np.mean(g*g))),1e-12);value=np.arcsinh(g/scale);limit=float(np.abs(value).max())
                    ax=axes[row,col];im=ax.imshow(value,origin='lower',aspect='equal',interpolation='nearest',cmap='RdBu_r',vmin=-limit,vmax=limit)
                    ax.set(title=f'{geo["group"]}: {"mean pointwise G" if col==0 else "mixture-probability G"}\nfit n={geo["train_queries"]}, scale={scale:.5g}',
                        xlabel='Every native postnorm coordinate0..2559',ylabel='Every native postnorm coordinate0..2559')
                    fig.colorbar(im,ax=ax,fraction=.03,pad=.025,label='asinh(G_ij / complete-matrix RMS)')
        keep(fig,'population_full_output_coordinate_geometries.png','总体读出矩阵：平均局部敏感度不等于混合概率敏感度',
            'All six2560x2560matrices, every151936readout row, fitted only on globally disjoint training article groups. Their difference is the known between-query covariance of W_U^T p. No coordinate sorting/pruning; display pixels merge entries only. Not an intrinsic language manifold or a causal semantic graph.')
        predictions=gzread(folder/'all_metric_predictions.json.gz');fig,axes=plt.subplots(2,2,figsize=(16,10))
        configs=('joint_global','joint_selected','output_selected_hybrid');methods=('global_full_mean_G','condition_full_mean_G','condition_diagonal_mean_G','condition_isotropic_mean_G','condition_mixture_G')
        for row,scope in enumerate(('natural_reanalysis','QA_query_transfer')):
            selected=[r for r in predictions if r['scope']==scope and r['metric_fit_split']=='heldout' and r['method']=='condition_full_mean_G']
            ax=axes[row,0]
            for j,name in enumerate(configs):
                rr=[r for r in selected if r['configuration']==name]
                ax.scatter([r['real_readout_variance'] for r in rr],[r['estimated_variance'] for r in rr],s=8,alpha=.4,label=name,color=colors[j])
            maximum=max(max(r['real_readout_variance'],r['estimated_variance']) for r in selected)
            ax.plot([0,maximum],[0,maximum],'k--',linewidth=.8);ax.set_xscale('symlog',linthresh=.001);ax.set_yscale('symlog',linthresh=.001)
            ax.set(title=f'{scope}: every held-out query/configuration',xlabel='Actual Var_p(W_U dh), real-valued head',ylabel='Fixed conditional mean-G quadratic prediction');ax.grid(alpha=.2);ax.legend(fontsize=7)
            ax=axes[row,1]
            for j,name in enumerate(configs):
                vals=[next(s['relative_SSE'] for s in population['summaries'] if s['scope']==scope and s['metric_fit_split']=='heldout' and s['configuration']==name and s['method']==method) for method in methods]
                ax.bar(np.arange(5)+(j-1)*.24,vals,.23,color=colors[j],label=name)
            ax.set_xticks(range(5),['pooled full','conditional full','diagonal','isotropic','mixture p'],rotation=15)
            ax.set_yscale('symlog',linthresh=.01);ax.set(title='Fixed baselines: local variance prediction, not finite KL',ylabel='Sum squared error / sum actual variance squared');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
        keep(fig,'population_output_metric_heldout.png','按文章留出的输出敏感度估计与完整/对角/等权对照',
            'All held-out rows and all three observed postnorm-error configurations; globally disjoint120fit/120held-out source groups. Five methods fixed in advance; no method selected on these outcomes. The given dh includes observed native/approximate states: this is not forecasting future language. Parent source behavior was already observed, and model capacities differ.')
    current={r['path']:r for r in index['figures']};current.update({r['path']:r for r in new})
    save(out/'index.json',{**index,'timestamp':stamp(),'extension_source':snapshot(Path(__file__)),'figures':list(current.values())})
    ledger('probability_native_model_and_history_figures',time.monotonic()-start,figures_added_or_refreshed=len(new))
    print('OPERATOR_FOLLOWUP_FIGURES',len(current),[r['path'] for r in new],flush=True)


if __name__=='__main__':
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=2):main()
