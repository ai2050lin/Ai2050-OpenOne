"""Scientific full-coordinate plots with fixed native index order and explicit display contracts."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from rdc_relation_common import *
from phase2715_rdc_relation_atlas import arrays as relation_arrays,matrix


def main():
    out=BASE/'figures';out.mkdir(parents=True,exist_ok=True);index=[];plt.rcParams.update({'font.size':10,'axes.titlesize':11})
    def finish(fig,name,title,note,sources):
        fig.savefig(out/(name+'.png'),dpi=140);plt.close(fig);index.append({'id':name,'file':name+'.png','title':title,'note':note,'source_arrays_or_recipes':sources})
    with np.load(BASE/'relation_atlas/training_scales.npz') as z:mu=z['mean'];std=z['standard_deviation']
    rr=rows(True)[:2];fields=[load_field(r,True) for r in rr];raw=[unbits(z[k]) for z in fields for k in ('h12','h23')];zs=[(h-mu[j%2])/std[j%2] for j,h in enumerate(raw)];limit=max(np.max(np.abs(h)) for h in raw);zlimit=max(np.max(np.abs(h)) for h in zs)
    fig,axs=plt.subplots(4,2,figsize=(15,10),layout='constrained')
    for i,h in enumerate(raw):
        for j,v in enumerate((h,zs[i])):
            lim=limit if j==0 else zlimit
            im=axs[i,j].imshow(v,aspect='auto',interpolation='nearest',cmap='RdBu_r',norm=SymLogNorm(.1,vmin=-lim,vmax=lim),extent=(0,2560,len(v),0));axs[i,j].set_title(f'{rr[i//2]["language"]} H{12 if i%2==0 else 23} | {"raw" if j==0 else "training z"}');axs[i,j].set_ylabel('Native token position');axs[i,j].set_xlabel('Native coordinate index (not sorted)');cb=fig.colorbar(im,ax=axs[i,j],shrink=.8)
            ticks=[-lim,-10 if j==0 else -1,0,10 if j==0 else 1,lim];cb.set_ticks(ticks);cb.set_ticklabels([f'{t:.1e}' if abs(t)>100 else f'{t:.2g}' for t in ticks]);cb.ax.tick_params(labelsize=8)
    fig.suptitle('All token positions x all 2560 coordinates | first predeclared EN/ZH confirmation sources')
    finish(fig,'complete_fields','新自然前缀全部token的完整H12/H23','原始/训练z分别使用共享色标；signed-log颜色；缩略图仅重采样显示，不删坐标。',['fresh/fields/'+r['sample_id']+'.npz' for r in rr]+['relation_atlas/training_scales.npz'])
    rr=rows();field={r['sample_id']:{l:unbits(load_field(r)[f'h{l}']) for l in (12,23)} for r in rr if r['split'] in ('train','test')};pairs=read(BASE/'relation_atlas/pair_index.json');matrices=[]
    for rel in ('nmod','conj','compound'):
        for split in ('train','test'):matrices.append(matrix(relation_arrays(pairs,field,rel,'distance_pos',split,(12,23),'train_z',mu,std)))
    limit=max(np.max(np.abs(x)) for x in matrices);fig,axs=plt.subplots(3,2,figsize=(13,14),layout='constrained')
    for i,c in enumerate(matrices):
        ax=axs[i//2,i%2];im=ax.imshow(c,interpolation='nearest',cmap='RdBu_r',norm=SymLogNorm(.002,vmin=-limit,vmax=limit),extent=(0,2560,2560,0));ax.set_title(f'{("nmod","conj","compound")[i//2]} | {("train","test")[i%2]}');ax.set_xlabel('H23 head coordinate');ax.set_ylabel('H12 dependent coordinate');fig.colorbar(im,ax=ax,shrink=.7)
    fig.suptitle('All 6,553,600 entries per matrix | exact distance + POS control | equal-source contrasts')
    finish(fig,'cross_coordinate_matrices','三种主要关系的完整跨坐标矩阵','所有矩阵同色标、原生坐标不重排；不是协方差、因果连接或真实空间几何。',['relation_atlas/pair_index.json','relation_atlas/training_scales.npz','main/fields/'])
    del matrices,field
    fig,axs=plt.subplots(2,1,figsize=(15,7),layout='constrained')
    for name in ('current','full_quadratic'):
        with np.load(BASE/f'rules/{name}/test_h36_errors.npz') as z:axs[0].plot(z['coordinate_mse'],lw=.7,label=name)
    for name in ('previous_only','previous_embedding','previous_embedding_bilinear','previous_embedding_relation'):
        with np.load(BASE/f'dynamics/{name}/errors.npz') as z:axs[1].plot(z['coordinate_mse'],lw=.7,label=name)
    for ax in axs:ax.set_yscale('symlog',linthresh=1);ax.set_xlabel('All2560 native coordinates, original order');ax.set_ylabel('Held-out coordinate MSE');ax.legend(fontsize=8,ncol=2)
    axs[0].set_title('Current true H12 -> raw H36');axs[1].set_title('Known-input update -> next raw H36; no state sufficiency assumed')
    finish(fig,'full_coordinate_prediction_errors','当前状态与已知新token更新的全部坐标误差','纵轴symlog保留零与低误差；没有用高误差位置挑选训练坐标。',['rules/*/test_h36_errors.npz','dynamics/*/errors.npz'])
    with np.load(BASE/'native/all_unit_profiles.npz') as z:
        fig,axs=plt.subplots(2,1,figsize=(15,7),layout='constrained');axs[0].plot(z['actual_activation_energy'],lw=.7,color='#555');axs[0].set_title('Actual activation energy, all9728 MLP units');axs[1].plot(z['all_predicted_sources_MSE'],lw=.7,label='ALL sources predicted');axs[1].plot(z['actual_past_hybrid_MSE'],lw=.7,label='EXTRA actual past; current predicted');axs[1].legend()
        for ax in axs:ax.set_xlabel('Native block23 MLP unit index (not a residual coordinate)');ax.set_yscale('symlog',linthresh=.01)
    finish(fig,'all_native_mlp_units','真实block23全部9728个MLP单元','激活能量与编译误差分别显示；额外真实历史的信息边界保留。',['native/all_unit_profiles.npz'])
    gen=[read(p) for p in sorted((BASE/'generation/commits').glob('*.json'))];v1=np.array([[x['conditional']['KL'] for x in r['native_rows']] for r in gen]);v2=np.array([[x['same_own_prefix_reference']['KL'] for x in r['self_rows']] for r in gen]);fig,axs=plt.subplots(1,2,figsize=(14,9),layout='constrained');limit=max(v1.max(),v2.max())
    for ax,v,title in zip(axs,(v1,v2),('True H12 conditional on NATIVE branch','Self-fed state on ITS OWN branch')):
        im=ax.imshow(v,aspect='auto',interpolation='nearest',cmap='magma',vmin=0,vmax=limit);ax.set_title(title);ax.set_xlabel('Generated step0..15');ax.set_ylabel('All64 source units, stable ID order');fig.colorbar(im,ax=ax,shrink=.6)
    fig.suptitle('Full-vocabulary KL | different branch distributions are NOT interchangeable')
    finish(fig,'generation_branch_KL','全部64来源的连续生成概率误差','两图的前缀分布不同，不能用右侧KL较小宣称自喂能力更强；需同时看重复与分叉。',['generation/commits/'])
    save(out/'index.json',index);save(out/'display_contract.json',{'timestamp':stamp(),'code':snapshot(Path(__file__)),'figures':index,'native_coordinate_order_preserved':True,'no_PCA_or_TopK':True,
      'display_only_resampling':True,'original_values':'Unchanged retained native fields, complete pair/control indices, all-coordinate errors and generation records. Client offers exact full-coordinate/tile values.',
      'matrix_axes':'Dependent H12 coordinate and head H23 coordinate. An array index is not a physical distance or proven causal edge.'});guard(4*1024**2);print('RELATION_FIGURES_COMPLETE',len(index),flush=True)


if __name__=='__main__':main()
