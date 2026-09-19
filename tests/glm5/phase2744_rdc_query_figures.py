"""Complete-axis learning changes and calibrated versus raw loss summaries."""
from phase2744_rdc_query_identifiability import *


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from phase2743_rdc_query_figures import save_figure
    start=time.monotonic();a=read(OUT/'analysis/result.json');assert a['all_passed'];records=[]
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    rr=[r for r in a['calibration']['prospective_reports'] if r['cohort']=='all'];fig,axes=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
    for ax,key,label in zip(axes,['raw_minus_original_raw','joint_minus_original_joint'],['Raw NLL minus original raw','Calibrated NLL minus original calibrated']):
        values=[r[key] for r in rr];mean=np.array([v['mean'] for v in values]);ci=np.array([v['interval95'] for v in values])
        ax.bar(range(5),mean,color=['#68747a','#2b887d','#cb9063','#377fac','#a477a5']);ax.errorbar(range(5),mean,yerr=np.maximum(np.stack([mean-ci[:,0],ci[:,1]-mean]),0),fmt='none',color='black',capsize=3)
        ax.axhline(0,color='black',linewidth=.5);ax.set_xticks(range(5),['native','real / 2742','permuted / 2742','real / 2743','permuted / 2743'],rotation=22,ha='right');ax.set_ylabel(label);ax.set_title('192 prospective content positions / 96 documents')
    save_figure(fig,'identity_natural_calibration_controls.png','真实参数学习与概率校准：在新增自然位置上比较',
      '左图原始NLL，右图每个变体各自在独立validation文档上选择温度/训练频率混合后比较；95%来源簇bootstrap。两图基准不同。校准只用于概率评分，不参与报告的自由生成；不能由标量校准改善唯一断定训练的内部机制。',['identifiability/analysis/result.json','identifiability/analysis/calibrated_natural_losses.npz'],records)
    labels=[v+' / '+f for v in VARIANTS[1:] for f in FAMILIES];data=[]
    with np.load(OUT/'analysis/all_native_unit_training_changes.npz') as z:
        for block in [16,35]:data.append(np.stack([z[v+'__'+f+f'__L{block}_activation__all_unit_MSE'] for v in VARIANTS[1:] for f in FAMILIES]))
    display=[np.arcsinh(x/1e-4) for x in data];maximum=max(x.max() for x in display);fig,axes=plt.subplots(2,1,figsize=(18,12),constrained_layout=True)
    for ax,x,block in zip(axes,display,[16,35]):
        im=ax.imshow(x,aspect='auto',origin='upper',cmap='viridis',vmin=0,vmax=max(float(maximum),1e-12),interpolation='nearest')
        ax.set_yticks(range(20),labels,fontsize=7);ax.set_xlabel('Every native MLP unit 0..9727, original order');ax.set_title(f'Block {block}: all-unit activation change, asinh(MSE / 0.0001)')
    fig.colorbar(im,ax=axes,shrink=.65,label='Displayed asinh(MSE / 0.0001)')
    save_figure(fig,'identity_training_all_native_units.png','实际训练增量：五类关系、两层与全部9728单元',
      '每行是一种实际参数变体和语言族，比较相同原始prompt上的全部MLP activation与原模型；没有筛单元、排序或PCA。仅色标asinh(MSE/0.0001)，两个块共享色标；原始逐单元MSE与所有样本的原始BF16值均保留。变化不等于该单元具有唯一语义。',['identifiability/analysis/all_native_unit_training_changes.npz'],records)
    figout=BASE/'figures';old=read(figout/'index.json')['figures'];names={r['path'] for r in records}
    save(figout/'index.json',{'timestamp':stamp(),'source':snapshot(__file__),'figures':[r for r in old if r['path'] not in names]+records,
      'scope':'Original full-axis fields and explicit display transforms. Actual experiments, not illustration data.'})
    ledger('identity_scientific_figures',time.monotonic()-start);print('IDENTITY_FIGURES_DONE',len(records),flush=True)


if __name__=='__main__':main()
