"""Scientific figures from full native arrays, with explicit display transforms."""
import gc
from rdc_binding_common import *

def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=2)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,
      'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white'})
    start=time.monotonic();out=BASE/'figures';out.mkdir(parents=True,exist_ok=True);manifest=[]
    def finish(fig,name,title,scope,source):
        fig.tight_layout();fig.savefig(out/name,dpi=150,bbox_inches='tight');plt.close(fig)
        manifest.append({'path':name,'title':title,'scope':scope,'source':source,'sha256':sha(out/name)})
    def heat(ax,a,title,x='Native coordinate',y='Layer boundary',trans=True):
        value=np.arcsinh(a) if trans else a;lim=float(np.max(abs(value))) or 1
        im=ax.imshow(value,aspect='auto',origin='lower',interpolation='nearest',cmap='RdBu_r',vmin=-lim,vmax=lim,extent=[0,a.shape[1],0,a.shape[0]])
        ax.set(title=title,xlabel=x,ylabel=y);plt.colorbar(im,ax=ax,fraction=.026,pad=.02,label='asinh(value), no clipping' if trans else 'value');return im
    rows=gzread(BASE/'natural_confirmation.json.gz')
    for cohort in ('gum','ewt'):
        row=next(r for r in rows if r['cohort']==cohort and r['split']=='connected_test')
        with np.load(BASE/'capture/natural'/f'{row["sample_id"]}.npz') as z:h=unbits(z['H'])[:,-1];sources=unbits(z['H12_sources'])
        fig,axes=plt.subplots(2,1,figsize=(16,7));heat(axes[0],h,cohort.upper()+' connected source: complete anchor field')
        heat(axes[1],h/np.sqrt(np.mean(h*h,-1,keepdims=True)).clip(1e-8),'Per-row full-coordinate RMS view')
        finish(fig,f'{cohort}_all_layers.png',cohort.upper()+'：连接组合的完整层×坐标场',
          '37×2560全部值；最后锚点；原始及每向量RMS分开。颜色asinh、无裁剪、原生索引不重排，图片缩放仅展示。',row['sample_id'])
        fig,axes=plt.subplots(2,1,figsize=(16,7));heat(axes[0],sources,'Every native H12 source token',y='Source token position')
        with np.load(BASE/'prediction/role_probe.npz') as z:coef=z['coefficients']
        u=sources/np.sqrt(np.mean(sources*sources,-1,keepdims=True)).clip(1e-8);role=np.maximum(u@coef[:-1]+coef[-1],0)+1e-6;role/=role.sum(-1,keepdims=True)
        im=axes[1].imshow(role.T,aspect='auto',origin='lower',cmap='viridis',vmin=0,vmax=1)
        axes[1].set(xlabel='Original source token position',ylabel='Coarse predicted role',title='Training-only prefix probe: normalized ridge scores, not gold roles')
        axes[1].set_yticks(range(6),['subject','object','oblique','modifier','predicate','other']);plt.colorbar(im,ax=axes[1],fraction=.026,pad=.02)
        finish(fig,f'{cohort}_sources_roles.png',cohort.upper()+'：全部H12来源与预测粗角色',
          '上图全token×2560坐标，下图同token序的六种粗分数；不把探针标签当原生语义齿轮，不截取高亮来源。',row['sample_id'])
    # Complete observed cross-coordinate matrices on every confirmation window.
    h=[];mm={16:[],35:[]}
    for row in rows:
        with np.load(BASE/'capture/natural'/f'{row["sample_id"]}.npz') as z:
            h.append(unbits(z['H'])[12,-1]);
            for b in mm:mm[b].append(unbits(z[f'L{b}_mlp'])[-1])
    h=np.array(h,dtype=np.float64);hc=h-h.mean(0)
    for b in mm:
        m=np.array(mm[b],dtype=np.float64);mc=m-m.mean(0);cov=hc.T@mc/len(h)
        den=np.sqrt(np.mean(hc*hc,0))[:,None]*np.sqrt(np.mean(mc*mc,0))[None,:];corr=cov/den.clip(1e-10)
        path=BASE/'atlas/full_coordinate_relations'/f'H12_M{b}.npz'
        if not path.exists():npz(path,uncentered_product=(h.T@m/len(h)).astype(np.float32),covariance=cov.astype(np.float32),correlation=corr.astype(np.float32),source_mean=h.mean(0),target_mean=m.mean(0))
        fig,ax=plt.subplots(figsize=(10,9));im=ax.imshow(corr,origin='lower',cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest',extent=[0,2560,0,2560])
        ax.set(xlabel=f'Block{b} MLP output coordinate',ylabel='H12 query coordinate',title=f'All 2560 x 2560 cross-coordinate correlations; N=128 windows')
        plt.colorbar(im,ax=ax,label='Pearson r; heterogeneous conditions, descriptive')
        finish(fig,f'all_coordinate_H12_M{b}.png',f'完整H12→block{b}写入坐标相关矩阵',
          '128确认窗口最后锚点、全部2560×2560坐标对，窗口等权、均值中心化。探索统计，不是物理参数边或因果连接；完整未中心化积/协方差另存。',str(path.relative_to(BASE)))
        del cov,corr,den;gc.collect()
    with np.load(BASE/'prediction/kernels.npz') as z:
        fig,axes=plt.subplots(1,3,figsize=(16,5))
        for ax,name in zip(axes,('query','source_mean','role_position_pair')):
            a=z[name];diag=np.sqrt(np.diag(a));a=a/diag[:,None]/diag[None,:]
            im=ax.imshow(a,vmin=0,vmax=1,cmap='viridis',interpolation='nearest');ax.set(title=name,xlabel='Original sample index',ylabel='Original sample index')
            plt.colorbar(im,ax=ax,fraction=.04)
        finish(fig,'all_source_kernel_comparison.png','512窗口完整核关系与简单对照',
          '所有512×512样本对，无样本聚类排序；仅对角规范化供可比展示。训练/验证/旧测试身份固定，不凭图形认定机制。','prediction/kernels.npz')
    frozen=read(BASE/'prediction/frozen.json')['selected']
    for b in (16,35):
        name=frozen[str(b)]['kernel'];fig,axes=plt.subplots(2,1,figsize=(16,6))
        with np.load(BASE/'confirmation'/f'b{b}_{name}.npz') as z:pred=z['prediction']
        for ax,cohort in zip(axes,('gum','ewt')):
            i=next(i for i,r in enumerate(rows) if r['cohort']==cohort and r['split']=='connected_test')
            actual=mm[b][i];value=np.stack([actual,pred[i],pred[i]-actual]);heat(ax,value,cohort.upper()+f': block{b} frozen prediction',y='Native / prediction / error')
            ax.set_yticks([.5,1.5,2.5],['native','prediction','error'])
        finish(fig,f'full_native_prediction_b{b}.png',f'block{b}原生写入、提前预测和误差全坐标对照',
          '选择每语料首个冻结连接窗口，不按成功选例；2560坐标全部显示，共用本子图色标。误差用于核对，不作donor搬运。',f'confirmation/b{b}_{name}.npz')
    fig,axes=plt.subplots(1,2,figsize=(13,5));training=read(BASE/'middle_training/result.json')
    for run in training['runs']:
        for ax,cohort in zip(axes,('gum','ewt')):
            values=[next(r['loss_delta'] for r in c['stats'] if r['split']=='test' and r['cohort']==cohort) for c in run['checkpoints']]
            ax.plot([c['step'] for c in run['checkpoints']],values,marker='o',ls='-' if run['seed']==2733 else '--',color='#1d6d70' if run['condition']=='coherent' else '#ba703f',label=run['condition']+'/'+str(run['seed']))
            ax.set(xlabel='Actual parameter-update step',ylabel='Mean NLL change vs FP32 bridge baseline',title=cohort.upper()+' / 192 heldout target positions');ax.axhline(0,color='#666',lw=.7)
    axes[0].legend(fontsize=8);finish(fig,'middle_training_trajectories.png','真实block16训练：两种子×连贯／乱序',
      '实际检查点1/8/32，曲线仅连接检查点。该图基线是FP32局部桥接，原始模型比较另有配对表；NLL下降不等于准确率普遍提高。','middle_training/result.json')
    with np.load(BASE/'gradient_span/all_program_gradient_factors.npz') as z:a=z['cosine']
    fig,ax=plt.subplots(figsize=(9,8));heat(ax,a,'All 768 program-query full-parameter gradient cosines',x='Frozen program query index',y='Frozen program query index',trans=False)
    finish(fig,'all_program_gradient_cosines.png','768表达的完整7471万参数梯度关系',
      '精确全gate/up/down参数内积，由完整外积因子求值；按冻结材料索引原序。相似性受目标/格式影响，不能直接叫语义同构。','gradient_span/all_program_gradient_factors.npz')
    analysis=read(BASE/'analysis/result.json');rr=[r for r in analysis['beta_paired']['reports'] if r['subset']=='all64_skip_as_noop']
    fig,axes=plt.subplots(1,2,figsize=(12,4));names=['gold oracle','prefix repetition','random'];colors=['#1d6d70','#ba703f','#8b9699']
    axes[0].bar(names,[r['mean_loss_delta'] for r in rr],color=colors);axes[0].axhline(0,color='#555',lw=.8);axes[0].set(ylabel='Mean full-vocabulary target CE change',title='64 examples; negligible-gradient skips = no-op')
    axes[1].bar(names,[r['first_token_accuracy'] for r in rr],color=colors);axes[1].axhline(5/64,color='#555',lw=.8,ls='--',label='unmodified5/64');axes[1].set(ylabel='First-target argmax accuracy',ylim=(0,.35));axes[1].legend()
    finish(fig,'beta_paired_objectives.png','Beta：目标函数不同，局部改善并不相同',
      '全部64同分母，oracle用正确答案，proxy仅前缀重复集合；当前前缀FP32评分，不是自然完整答案。随机同参数范数。','analysis/result.json')
    decomposition=BASE/'format_content/decomposition_result.json'
    if decomposition.exists():
        rr=read(decomposition);records=[r for r in rr['gradient_nonorthogonal_terms'] if r['split']=='depth_test'];fig,ax=plt.subplots(figsize=(11,5));x=np.arange(len(records))
        for j,key in enumerate(('content_squared_norm_ratio','format_squared_norm_ratio','signed_cross_ratio')):ax.bar(x+(j-1)*.23,[r[key] for r in records],.23,label=key)
        ax.set_xticks(x,[r['representation'] for r in records]);ax.axhline(0,color='#555',lw=.8);ax.set(ylabel='Term / full-gradient squared norm',title='Depth4 holdout / 32 per representation: content + format + cross = full');ax.legend(fontsize=8)
        finish(fig,'content_format_full_gradient.png','内容／格式的全参数梯度分解',
          '非正交三项相加为1；有符号交叉项不能忽略，也不能当独立语义比例。所有74711040参数参与。','format_content/all_parameter_gram_decomposition.npz')
    behavior_path=BASE/'analysis/behavior.json'
    if behavior_path.exists():
        behavior=read(behavior_path);representations=['en','zh','python','en_reordered']
        fig,axes=plt.subplots(3,1,figsize=(13,10),sharey=True)
        metrics=[('strict_answer_accuracy','strict digit','#215f69'),('conservative_correct_fraction_of_all','parsed correct / all','#4c94b0'),
          ('conservative_parse_coverage','parser coverage','#96a9a7'),('censored_fraction','censored at128','#bd7846')]
        for ax,split in zip(axes,('test','depth_test','prospective_depth6')):
            rr=[next(r for r in behavior['long_native'] if r['split']==split and r['representation']==rep) for rep in representations]
            x=np.arange(4)
            for j,(key,label,color) in enumerate(metrics):ax.bar(x+(j-1.5)*.19,[r[key] for r in rr],.19,label=label,color=color)
            ax.set_xticks(x,[rep+' / N='+str(r['rows']) for rep,r in zip(representations,rr)])
            ax.set(title=split+' / native greedy, no parameter update',ylabel='Fraction of all declared examples',ylim=(0,1.05))
        axes[0].legend(ncol=4,fontsize=8)
        finish(fig,'long_native_answer_format_stop.png','长预算原生生成：答案、保守解析与截断分开',
          '128token上限，所有给定分母保留。解析正确/全体不等于未解析即答错；截断处答案样式后缀仍可能非最终结果，已停止正确比例另存。旧案例与新六步案例分开，四表达共享语义案例。','analysis/behavior.json')
        branches=['full_EN_projected_code','content_EN_projected_code','mean_EN_format','random_global',
          'oracle_gold_next_digit_CE','prefix_repeat_probability_mass','random_beta']
        labels=['full EN -> code (.02)','content EN -> code (.02)','mean EN format (.02)','random global (.02)',
          'gold oracle (.10)','prefix repeat proxy (.10)','random per case (.10)']
        splits=['test','depth_test','prospective_depth6'];fig,axes=plt.subplots(1,2,figsize=(14,6))
        for ax,key,title in zip(axes,('native_initial_nll_change','strict_answer_change'),('First-target native BF16 NLL change','Strict full-answer accuracy change')):
            a=np.full((len(branches),len(splits)),np.nan)
            for i,branch in enumerate(branches):
              for j,split in enumerate(splits):
                match=[r for r in behavior['autonomous_paired'] if r['branch']==branch and r['split']==split and r['representation']=='pooled']
                if match:a[i,j]=match[0][key]
            lim=float(np.nanmax(abs(a))) or 1;cmap=plt.colormaps['RdBu_r'].copy();cmap.set_bad('#e3e7e7')
            im=ax.imshow(np.ma.masked_invalid(a),aspect='auto',cmap=cmap,vmin=-lim,vmax=lim)
            ax.set_xticks(range(3),['old test','old depth4','new depth6']);ax.set_yticks(range(7),labels);ax.set_title(title)
            ax.axhline(3.5,color='#555',lw=1)
            for i in range(7):
              for j in range(3):ax.text(j,i,'not run' if np.isnan(a[i,j]) else f'{a[i,j]:+.3f}',ha='center',va='center',fontsize=8,color='#111' if np.isnan(a[i,j]) or abs(a[i,j])<.65*lim else 'white')
            plt.colorbar(im,ax=ax,fraction=.025,pad=.02)
        finish(fig,'own_history_probability_vs_answer.png','自己的生成历史：首目标概率与完整答案并不等价',
          '相对同样本原生分支的配对变化。各划分16表达/4语义案例，聚合四表达；小分组不能支持普遍结论。0.02全局方向和0.10按例方向为舍入前范数组；实际BF16改变量另记。灰格未运行，绝非0。','analysis/behavior.json')
        rsa=behavior['cross_model_relations']['comparisons'];fig,axes=plt.subplots(1,2,figsize=(14,5),sharey=True)
        for ax,mode in zip(axes,('raw','cohort_centered')):
            rr=[r for r in rsa if r['normalization']==mode];x=np.arange(len(rr))
            ax.scatter(x,[r['relation_matrix_pearson'] for r in rr],color='#215f69',marker='o',label='observed',zorder=4)
            for shift,key,interval,color,label in [(-.10,'within_cohort_identity_permutation_mean','permutation_interval95','#a2aaa8','within cohort null'),
              (.10,'family_depth_preserving_permutation_mean','family_depth_preserving_interval95','#bd7846','family/depth null')]:
                means=np.array([r[key] for r in rr]);ci=np.array([r[interval] for r in rr])
                ax.vlines(x+shift,ci[:,0],ci[:,1],color=color,lw=1)
                ax.scatter(x+shift,means,color=color,marker='.',label=label)
            ax.set_xticks(x,[r['model_a'].replace('qwen','Q')+'-'+r['model_b'].replace('qwen','Q')+'\n'+r['field'].replace('_query','').replace('_mlp',' MLP') for r in rr],fontsize=8)
            ax.set(title=mode+' / 128 common source rows',ylabel='Full relation-matrix Pearson correlation',ylim=(-1.05,1.05));ax.axhline(0,color='#888',lw=.6)
        axes[0].legend(fontsize=8)
        finish(fig,'cross_model_full_relation_audit.png','跨模型关系矩阵：全坐标、条件中心化和分层打乱',
          '每模型用自身全部坐标，不对齐坐标索引。300次两类来源身份打乱；区间是随机化分布，不是独立坐标置信区间。词汇、模板、训练和架构混杂仍存在，不证明语义同构。','analysis/behavior.json')
    signed_path=BASE/'signed_source/result.json'
    if signed_path.exists():
        signed=read(signed_path);fig,axes=plt.subplots(1,2,figsize=(13,5),sharex=True)
        strata=[('gum','signed_connected'),('gum','signed_matched'),('ewt','signed_connected'),('ewt','signed_matched')]
        x=np.arange(4)
        for ax,block in zip(axes,(16,35)):
            for shift,comparison,label,color in [(-.15,'signed_position_vs_original','signed position','#4c94b0'),
              (0,'signed_role_vs_original','signed role-position','#215f69'),(.15,'mixed_vs_original','equal mixture','#bd7846')]:
                rr=[next(r for r in signed['paired'] if r['block']==block and r['comparison']==comparison and r['cohort']==cohort and r['split']==split) for cohort,split in strata]
                means=np.array([r['source_cluster']['mean'] for r in rr]);ci=np.array([r['source_cluster']['interval95'] for r in rr])
                ax.vlines(x+shift,ci[:,0],ci[:,1],color=color,lw=1)
                ax.scatter(x+shift,means,color=color,marker='o',label=label)
            ax.axhline(0,color='#777',lw=.8);ax.set_xticks(x,['GUM\nconnected','GUM\nmatched','EWT\nconnected','EWT\nmatched'])
            ax.set(title=f'Block{block}: new natural confirmation',ylabel='Cluster mean relative error difference vs original')
        axes[0].legend(fontsize=8)
        finish(fig,'signed_source_new_natural_confirmation.png','有符号来源矩：新自然窗口与原规则的配对比较',
          '128个新来源位置、127个不同模型输入，同文档相同输入的复用已审计；来源文档分组bootstrap区间，负值表示误差更低。旧验证集仍选择原规则，不按本图重选；数学反例被区分不等于普遍预测收益。','signed_source/result.json')
    save(out/'index.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'figures':manifest,'seconds':time.monotonic()-start,
      'scope':'Full numerical inputs retained. Display transforms/means are stated; no threshold or Top-K used to define a mechanism backbone.'})
    ledger('binding_scientific_figures',time.monotonic()-start)
    print('BINDING_FIGURES',len(manifest),flush=True)

if __name__=='__main__':main()
