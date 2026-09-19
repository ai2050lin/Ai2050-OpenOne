"""Static scientific figures; raw numerical archives remain the primary record."""
import argparse
from rdc_query_common import *

OUT=BASE/'figures'


def save_figure(fig,name,title,scope,artifacts,records):
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True,exist_ok=True);file=OUT/name;fig.savefig(file,dpi=165,bbox_inches='tight');plt.close(fig)
    records.append({'path':name,'title':title,'scope':scope,'sources':artifacts,'sha256':sha(file)})


def atlas(plt,records):
    file=BASE/'atlas/all_coordinate_group_profiles.npz';cohorts=['gum','ewt','cmrc']
    with np.load(file) as z:
        raw=[z[c+'__test_layer_mean'] for c in cohorts];norm=[z[c+'__test_layer_RMS_normalized_mean'] for c in cohorts]
    fig,ax=plt.subplots(3,2,figsize=(18,9),constrained_layout=True)
    for j,(data,label) in enumerate([(raw,'Native raw mean'),(norm,'Mean of per-vector RMS-normalized field')]):
        maximum=max(float(abs(a).max()) for a in data)
        for i,(a,c) in enumerate(zip(data,cohorts)):
            im=ax[i,j].imshow(a,aspect='auto',origin='lower',cmap='RdBu_r',vmin=-maximum,vmax=maximum,interpolation='nearest')
            ax[i,j].set_title(c.upper()+' / '+label);ax[i,j].set_ylabel('Native raw layer 0..36');ax[i,j].set_xlabel('Native coordinate 0..2559')
        fig.colorbar(im,ax=list(ax[:,j]),shrink=.65)
    save_figure(fig,'atlas_full_coordinate.png','三类自然语料：完整原生坐标的全层响应',
      '每幅图保留37×2560个原序格点。左列原值、右列先逐向量RMS归一化后取均值；同列共享对称色标，不使用Top-K、PCA或分位数截断。均值图不代表充分状态。',['atlas/all_coordinate_group_profiles.npz'],records)
    fig,ax=plt.subplots(3,2,figsize=(18,9),constrained_layout=True)
    for j,(data,label) in enumerate([(raw,'Raw mean'),(norm,'Mean after per-vector RMS')]):
        transformed=[np.arcsinh(a/.05) for a in data];maximum=max(float(abs(a).max()) for a in transformed)
        for i,(a,c) in enumerate(zip(transformed,cohorts)):
            im=ax[i,j].imshow(a,aspect='auto',origin='lower',cmap='RdBu_r',vmin=-maximum,vmax=maximum,interpolation='nearest')
            ax[i,j].set_title(c.upper()+' / '+label+' / asinh(value / 0.05)');ax[i,j].set_ylabel('Native raw layer 0..36');ax[i,j].set_xlabel('Native coordinate 0..2559')
        fig.colorbar(im,ax=list(ax[:,j]),shrink=.65,label='Displayed asinh(value / 0.05)')
    save_figure(fig,'atlas_low_background_display.png','全坐标低幅背景显示：与未变换图配套阅读',
      '仅显示变换为asinh(value/0.05)，色标是变换后的值；全坐标原序不变、无阈值裁剪、不改原数据。同列共享色标。该尺度为观察弱背景而设置，不能把增强的纹理当作已经验证的机制或语义主干。',['atlas/all_coordinate_group_profiles.npz'],records)
    item=next(r for r in read(BASE/'events/paths_index.json') if r['mode']=='new_natural_fixed_query')
    with np.load(BASE/item['path']) as z:
        pairs=[z[f'L{b}_ordered_pair_all_unit_sum'].copy() for b in [16,35]]
        energy=[z[f'L{b}_unit_antisymmetric_energy'].copy() for b in [16,35]]
    fig,ax=plt.subplots(2,2,figsize=(15,9),constrained_layout=True)
    for i,b in enumerate([16,35]):
        a=pairs[i];mx=float(abs(a).max());im=ax[i,0].imshow(a,cmap='RdBu_r',vmin=-mx,vmax=mx,interpolation='nearest',origin='lower')
        ax[i,0].set_title(f'Block {b}: all-unit ordered pair sum');ax[i,0].set_xlabel('Up-branch source r (last = other)');ax[i,0].set_ylabel('Gate-branch source s');fig.colorbar(im,ax=ax[i,0],shrink=.8)
        ax[i,1].plot(np.arange(len(energy[i])),np.log1p(np.maximum(energy[i],0)),linewidth=.4)
        ax[i,1].set_title(f'Block {b}: all 9728 native units');ax[i,1].set_xlabel('Native MLP unit index');ax[i,1].set_ylabel('log(1 + antisymmetric pair energy)')
    save_figure(fig,'ordered_all_sources_units.png','有序来源对与全部 MLP 单元：保留方向，也保留抵消边界',
      '选择路径索引中的第一条新自然样本，未按效果挑选。二维图对全部单元求和；曲线包含9728个单元。左图各层独立原值色标，右图显示log1p(max(E,0))：有限精度的微小负能量显示为0，原数值保留。反对称来源分量的总和为零，能量不等于语义贡献。',[item['path']],records)
    item=gzread(BASE/'events/material.json.gz')['trajectories'][0];sid=item['row']['sample_id']
    with np.load(BASE/'events/fields'/f'{sid}.npz') as z:h=unbits(z['H']).astype(float);stats=z['dynamic_full_vocabulary_statistics'].copy();steps=z['steps'].copy()
    fig,ax=plt.subplots(4,1,figsize=(18,10),constrained_layout=True)
    for a,b in zip(ax[:3],[12,24,36]):
        data=h[:,b]/rms(h[:,b]);mx=float(abs(data).max());im=a.imshow(data,aspect='auto',cmap='RdBu_r',vmin=-mx,vmax=mx,interpolation='nearest',origin='lower')
        a.set_yticks(range(len(steps)),steps,fontsize=7);a.set_title(f'Native H{b}: every coordinate / per-anchor RMS');a.set_ylabel('Actual step');fig.colorbar(im,ax=a,shrink=.8)
    im=ax[3].imshow(stats[:,:,1],aspect='auto',origin='lower',interpolation='nearest',cmap='viridis');ax[3].set_yticks(range(len(steps)),steps,fontsize=7)
    ax[3].set_ylabel('Actual step');ax[3].set_xlabel('Fixed diagnostic query index 0..99');ax[3].set_title('Full-vocabulary KL to query-only response, not a semantic-distance theorem');fig.colorbar(im,ax=ax[3],shrink=.8)
    save_figure(fig,'actual_generation_query_field.png','同一条原生轨迹：生成时间、全坐标与100个条件查询',
      '第一条预先冻结的旧轨迹。行标签是真实生成步，图上等距排列只表示锚点序号，不表示时间间隔相等。层场每行RMS归一化，各层独立色标，保留全部坐标；查询KL覆盖全词表。文字事件不是内部符号执行的证明。',[f'events/fields/{sid}.npz',f'events/commits/{sid}.json'],records)


def rules(plt,records):
    a=read(BASE/'analysis/phase2741.json');names=list(a['primary_coordinate_test']['candidate_mse'])
    fig,ax=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
    for axis,key,metric,label in [(ax[0],'primary_coordinate_test','candidate_mse','All-coordinate postnorm MSE'),(ax[1],'primary_vocabulary_test','KL_native_to_prediction','Full-vocabulary KL(native || predicted)')]:
        data=a[key][metric];means=np.array([data[n]['mean'] for n in names]);ci=np.array([data[n]['interval95'] for n in names])
        axis.bar(range(len(names)),means,color=['#6b7280','#8ab9cf','#c1a677','#ab80ab','#227c6b'])
        axis.errorbar(range(len(names)),means,yerr=np.maximum(np.stack([means-ci[:,0],ci[:,1]-means]),0),fmt='none',ecolor='black',capsize=3)
        axis.set_xticks(range(len(names)),names,rotation=20,ha='right');axis.set_ylabel(label);axis.set_title('Unseen documents + 20 unseen queries')
    save_figure(fig,'frozen_query_rule_comparison.png','冻结规则的双重未见测试：全坐标与全词表分开比较',
      '误差为来源文档簇等权均值，误差线是冻结拟合条件下2000次簇bootstrap的95%区间。两图单位不同。query-only候选的共同解码器仍接收前缀H12；不是无信息对照。',['analysis/phase2741.json'],records)


def behavior(plt,records):
    a=read(BASE/'analysis/phase2742.json');r=[r for r in a['late'] if r['representation']=='all'];fig,ax=plt.subplots(1,2,figsize=(16,5),constrained_layout=True)
    labels=[r['branch'] for r in r]
    ax[0].bar(range(len(r)),[r['correct_and_stopped']/r['expressions'] for r in r],color='#227c6b');ax[0].set_ylim(0,1);ax[0].set_ylabel('Correct terminal answer AND EOS / 96')
    ax[1].bar(range(len(r)),[r['mean_tokens'] for r in r],color='#497ca0');ax[1].set_ylabel('Mean actually emitted tokens (cap 1024)')
    for axis in ax:axis.set_xticks(range(len(r)),labels,rotation=25,ha='right')
    save_figure(fig,'late_own_history_behavior.png','晚期类别偏置：答案与停止、生成长度分别记账',
      '每分支96表达、24语义组，相同B8分组，各自生成历史。偏置同时加给全部8个数字或8个字母，不输入正确标签。条形描述该固定面板；完整推理链未评分。',['analysis/phase2742.json'],records)
    with np.load(BASE/'analysis/matched_model_query_geometry.npz') as z:grams=[z[m+'__centered_Gram'].copy() for m in ['qwen4','qwen14','glm4']]
    fig,ax=plt.subplots(1,3,figsize=(16,5),constrained_layout=True)
    for axis,gram,model in zip(ax,grams,['Qwen3-4B','Qwen3-14B','GLM4-9B']):
        g=gram/np.linalg.norm(gram);im=axis.imshow(g,cmap='RdBu_r',vmin=-max(abs(x/np.linalg.norm(x)).max() for x in grams),vmax=max(abs(x/np.linalg.norm(x)).max() for x in grams),interpolation='nearest');axis.set_title(model);axis.set_xlabel('Query index');axis.set_ylabel('Query index')
    fig.colorbar(im,ax=ax,shrink=.75)
    save_figure(fig,'three_native_query_grams.png','三个原始精度模型：匹配文本上的查询关系矩阵',
      '每个模型用自己的完整原生坐标构造中心化查询Gram，再除以Frobenius范数；同图共享色标。相同查询索引不代表token或坐标一一对应，矩阵相似/不同均不证明普遍流形同构。',['analysis/matched_model_query_geometry.npz','analysis/phase2742.json'],records)


def main(phase):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    records=[];start=time.monotonic()
    {'2740':atlas,'2741':rules,'2742':behavior}[str(phase)](plt,records)
    previous=read(OUT/'index.json')['figures'] if (OUT/'index.json').exists() else []
    names={r['path'] for r in records};previous=[r for r in previous if r['path'] not in names]
    save(OUT/'index.json',{'timestamp':stamp(),'source':snapshot(__file__),'figures':previous+records,
      'scope':'Static scientific summaries, not native scalar replacement. Every figure links its complete numerical archive.'})
    ledger('static_query_figures_'+str(phase),time.monotonic()-start);print('QUERY_FIGURES',phase,len(records),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['2740','2741','2742']);a=p.parse_args();main(a.phase)
