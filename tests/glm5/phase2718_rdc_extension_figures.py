"""Compact scientific views of all-coordinate corrections and all observed attractor traces."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rdc_relation_common import *


def main():
    out=BASE/'figures';guard(800*1024);index=read(out/'index.json');added=[]
    def finish(fig,name,title,note,source):
        fig.savefig(out/(name+'.png'),dpi=115);plt.close(fig);added.append({'id':name,'file':name+'.png','title':title,'note':note,'source_arrays_or_recipes':source})
    fig,axes=plt.subplots(2,1,figsize=(14,8),layout='constrained')
    with np.load(BASE/'boundary_compilation/all_coordinate_and_unit_errors.npz') as z:
        for name,label in [('original_full_quadratic','Original all-source quadratic'),('first_source_affine_else_frozen_quadratic','First-source affine + frozen interior'),('identity_at_all_sources','Identity at every source')]:axes[0].plot(z[name+'_H24_error'],lw=.7,label=label)
    axes[0].set_title('Native H24 output: every source available, every coordinate scored');axes[0].set_ylabel('MSE per native coordinate');axes[0].set_yscale('symlog',linthresh=.01);axes[0].legend(fontsize=8)
    for name in ('current','full_quadratic'):
        with np.load(BASE/f'rules/{name}/test_h36_errors.npz') as z:axes[1].plot(z['coordinate_mse'],lw=.7,label='Original '+name)
    with np.load(BASE/'source_normalization/all_coordinate_prediction_MSE.npz') as z:axes[1].plot(z['source_RMS_mean_matched_df_test'][2560:],lw=.7,label='Source-RMS mean, matched df')
    axes[1].set_title('H36 state prediction: lower MSE does not imply best probability KL');axes[1].set_ylabel('MSE per native coordinate');axes[1].set_yscale('symlog',linthresh=1);axes[1].legend(fontsize=8)
    for ax in axes:ax.set_xlabel('All2560 native coordinates, unchanged order')
    finish(fig,'boundary_and_source_corrections','边界修正与历史规范化的完整坐标误差','事后探索；保留所有坐标、共享位置顺序，状态收益不等于生成闭合。',['boundary_compilation/all_coordinate_and_unit_errors.npz','source_normalization/all_coordinate_prediction_MSE.npz','rules/*/test_h36_errors.npz'])
    fig,axes=plt.subplots(3,1,figsize=(14,10),layout='constrained')
    for scope in ('current','temporal','self_generation'):
        with np.load(BASE/f'output_geometry/{scope}_coordinate_profiles.npz') as z:axes[0].plot(z['absolute'],lw=.7,label=scope)
    axes[0].set_title('Complete native postnorm-coordinate path attribution; not causal importance');axes[0].set_yscale('symlog',linthresh=.001);axes[0].set_xlabel('All2560 native indices');axes[0].set_ylabel('Mean absolute path term');axes[0].legend(fontsize=8)
    for scope in ('current','temporal','self_generation'):
        rr=read(BASE/f'output_geometry/{scope}_rows.json');axes[1].scatter([r['KL_same_FP32'] for r in rr],[r['local_Fisher'] for r in rr],s=5,alpha=.3,label=scope)
    axes[1].plot([0,40],[0,40],'k--',lw=.7);axes[1].set_xlabel('Exact full-vocabulary KL');axes[1].set_ylabel('Local Fisher approximation');axes[1].set_title('All1536 observations: local curvature is not a sufficient finite-error model');axes[1].legend(fontsize=8)
    rr=read(BASE/'surrogate_stability/trajectory_result.json')['records']
    for r in rr:
        s=r['suffix_start_step'];d=np.array(r['full_coordinate_distance_to_fixed_state'][s:]);axes[2].plot(np.arange(len(d)),d/d[0],color='#416999',lw=.7,alpha=.4)
    axes[2].set_xlabel('Steps since the final repeated-zero suffix began');axes[2].set_ylabel('Full-state distance / starting distance');axes[2].set_title(f'All{len(rr)} observed repeated-zero suffixes approach the LEARNED surrogate fixed state')
    finish(fig,'output_geometry_and_attractor','完整坐标输出归因、概率曲率与近似器吸引点','归因路径、Fisher近似与实际LLM机制区分；39条轨迹为被观察到的重复后缀，不是全部语言的动力学。',['output_geometry/*_coordinate_profiles.npz','output_geometry/*_rows.json','surrogate_stability/trajectory_result.json'])
    for r in added:
        index=[x for x in index if x['id']!=r['id']];index.append(r)
    save(out/'index.json',index);save(out/'extension_display_contract.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'figures':added,'all_native_coordinates_preserved':True,'scatter':'Every scored observation, not selected examples','trajectory':'Every39 final repeated-zero source suffix, all2560 coordinates in each norm','not_a_browser_QA':True});print('EXTENSION_FIGURES_COMPLETE',len(added),flush=True)


if __name__=='__main__':main()
