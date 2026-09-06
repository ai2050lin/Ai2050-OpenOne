"""All-coordinate analytic functions, heldout descriptive fingerprints, scalar weight queries."""
import argparse,json
import numpy as np
from phase2620_native_coordinate_contract import *
from phase2621_native_language_material import FAMILIES

SOURCE=RESULT/'phase2622_unmodified_native_fields'
OUT=RESULT/'phase2623_native_parameter_algorithms'

def field(name,source=SOURCE):return np.load(source/f'field/{name}.float32.npy',mmap_mode='r')

def finite_rank_one(h,w,v,delta,epsilon=1e-6):
    """Full downstream RMSNorm/head margin change for h -> h + delta*v.

    v may be e_j (single residual coordinate / scalar down weight times a_k)
    or the actual W_down[:, k] (one intermediate neuron). No donor input.
    """
    h=np.asarray(h,dtype=np.float64);w=np.asarray(w,dtype=np.float64);v=np.asarray(v,dtype=np.float64)
    s2=np.mean(h*h)+epsilon;q=np.dot(w,h);base=q/np.sqrt(s2)
    return float((q+delta*np.dot(w,v))/np.sqrt(s2+(2*delta*np.dot(h,v)+delta*delta*np.dot(v,v))/len(h))-base)

def weight_query(case,j,k,source=SOURCE):
    W=field('final_down_weights',source);a=field('mlp_anchor_boundary',source)[case,-1,-1]
    gh=field('gradient_h',source)[case];gg=field('gradient_gate',source)[case];gu=field('gradient_up',source)[case];x=field('final_mlp_input',source)[case]
    return {'case':case,'final_layer':field('hidden_anchor_boundary',source).shape[1]-2,'output_coordinate':j,'mlp_neuron':k,
        'down_weight':float(W[j,k]),'neuron_activation':float(a[k]),'down_single_weight_derivative':float(gh[j])*float(a[k]),
        'gate_single_weight_derivative_kj':float(gg[k])*float(x[j]),'up_single_weight_derivative_kj':float(gu[k])*float(x[j]),
        'scope':'first-token contrast, real-arithmetic extension at captured final MLP boundary; not natural language necessity'}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',type=int);parser.add_argument('--coordinate',type=int,default=0);parser.add_argument('--neuron',type=int,default=0);args=parser.parse_args()
    if args.case is not None:print(json.dumps(weight_query(args.case,args.coordinate,args.neuron),ensure_ascii=True,indent=2));return
    rows=read(SOURCE/'material/cases.json');records=read(SOURCE/'analysis/native_records.json');n=len(rows)
    H=field('hidden_anchor_boundary');A=field('mlp_anchor_boundary');GA=field('gradient_a');GH=field('gradient_h')
    groups=[f'{f}/{l}' for f in FAMILIES for l in ('en','zh')];profiles={};summary=[]
    for g in groups:
        groupidx=[i for i,r in enumerate(rows) if r['family']+'/'+r['language']==g]
        cell={}
        for form in (0,1):
            pairs=[]
            for item in range(12):
                pp=sorted([i for i in groupidx if rows[i]['form']==form and rows[i]['index']==item],key=lambda i:rows[i]['variant'])
                pairs.append(pp)
            for kind,F in [('hidden',H),('neuron',A)]:
                delta=np.stack([F[p[1],:,-1]-F[p[0],:,-1] for p in pairs]).astype('float64')
                mu=delta[:6].mean(0);test=delta[6:];den=float(np.sum(test**2));ratio=float(np.sum((test-mu)**2)/den) if den>0 else None
                pos_train=(delta[:6]>0).mean(0);pos_test=(test>0).mean(0)
                stable=((pos_train>=5/6)&(pos_test>=5/6))|((pos_train<=1/6)&(pos_test<=1/6)&(np.abs(mu)>1e-12))
                profiles[f'{g}/f{form}/{kind}/train_delta']=mu.astype('float32')
                profiles[f'{g}/f{form}/{kind}/test_delta']=test.mean(0).astype('float32')
                profiles[f'{g}/f{form}/{kind}/same_sign_fraction_by_layer']=stable.mean(-1).astype('float32')
                cell[f'f{form}/{kind}']={'heldout_squared_error_relative_zero':ratio,'final_stable_sign_coordinate_fraction':float(stable[-1].mean()),
                    'criteria':'same sign in >=5/6 discovery and >=5/6 heldout, descriptive threshold; not multiplicity-corrected discovery'}
        # All units participate; no largest-activation unit is selected as a semantic core.
        neuron_mass=[];sensitivity_mass=[]
        for i in groupidx:
            a=np.asarray(A[i,-1,-1]);ga=np.asarray(GA[i]);order=np.argsort(np.abs(a),kind='stable')
            energy=np.square(a*ga,dtype='float64');gradenergy=np.square(ga,dtype='float64')
            neuron_mass.append([float(energy[q].sum()/(energy.sum()+1e-30)) for q in np.array_split(order,4)])
            sensitivity_mass.append([float(gradenergy[q].sum()/(gradenergy.sum()+1e-30)) for q in np.array_split(order,4)])
        cell['amplitude_quartile_gradient_times_activation_energy']=np.mean(neuron_mass,0).tolist()
        cell['amplitude_quartile_pure_sensitivity_energy']=np.mean(sensitivity_mass,0).tolist()
        summary.append({'group':g,**cell})
    OUT.joinpath('field').mkdir(parents=True,exist_ok=True);np.savez(OUT/'field/allcoordinate_condition_profiles.npz',**profiles)
    # Exact native weight gradient representation, not a low-rank approximation:
    # down[j,k] = grad_h[j] * a[k], gate[k,j] = grad_g[k] * x[j], up[k,j] = grad_u[k] * x[j].
    save(OUT/'analysis/scalar_examples.json',[weight_query(i,j,k) for i in (0,96,192,288,384,480,576,672) for j,k in ((0,0),(853,3242),(1706,6485),(2559,9727))])
    rng=np.random.default_rng(2623);W=field('final_down_weights');gamma=field('final_norm_weights');checks=[]
    for i in (0,97,194,291,388,485,582,679):
        h=H[i,-1,-1].astype('float64');w=field('output_weight_contrast')[i].astype('float64')*gamma;gh=GH[i].astype('float64')
        for j in range(0,len(h),97):
            e=np.zeros_like(h);e[j]=1;eta=1e-3
            fd=(finite_rank_one(h,w,e,eta)-finite_rank_one(h,w,e,-eta))/(2*eta)
            checks.append(abs(fd-gh[j]))
    result={'provenance':str(Path(__file__)),'summary':{'prompts':n,'per_case_single_down_weights_addressable':int(W.size),
        'all_final_mlp_neurons':int(W.shape[1]),'all_final_residual_coordinates':int(W.shape[0]),'finite_difference_gradient_max_error':max(checks),
        'groups':summary},'checks':{'all16_groups':len(summary)==16,'all_fields_finite':all(np.isfinite(v).all() for v in profiles.values()),'double_fd_matches_fp32_derivative':max(checks)<1e-4},
        'important_limits':['no donor vectors','last MLP boundary only','first token contrasts differ across groups; derivative similarity can reflect shared output labels','formal identities are not empirical semantic laws']}
    finish(2623,'全2560坐标、9728神经元与24903680个down标量参数的非搬运算法',OUT,result,
        '用已知真实权重直接计算单坐标、单MLP神经元与单标量参数对最终层输出的局部影响。所有物理坐标参与；对每一单位求有限删除的解析反事实，梯度只作小扰动近似。',
        r's^2=\|h\|^2/d+\epsilon,\quad m=w^Th/s;\quad \partial_jm=w_j/s-(w^Th)h_j/(ds^3);\quad \partial_{W_{jk}}m=(\partial_jm)a_k;\quad m(h+\eta v)=\frac{w^Th+\eta w^Tv}{\sqrt{s^2+(2\eta h^Tv+\eta^2\|v\|^2)/d}}.',
        '768自然状态；八族中英、两表面，前6基础item估计符号与均值，后6基础item核对，全部隐藏坐标与MLP单位保留；最终down矩阵2560×9728逐参数均可寻址。python phase2623_native_parameter_algorithms.py --case 0 --coordinate 853 --neuron 3242可查询真实权重和三个矩阵对应导数。',
        '从“整块状态换过去”转为“这一真实标量通过哪条已知乘法及归一化影响输出”。down/gate/up的外积分解是精确链式求导，不是对激活做低秩压缩；单坐标效应可正可负，并受全场范数条件化。完整条件指纹与留出误差是观察性拼图。',
        '解析作用只在最终MLP边界，未覆盖全模型所有权重。BF16离散舍入不可微，导数指固定捕获状态的实数延拓；预测需真实前向校验。单神经元删除不是必要性定理。符号阈值只是描述性筛查，多个坐标不能据此命名为语义主干；输出标签复用可制造梯度相似。',
        '进入真实原生坐标/神经元/权重±扰动核对，并在其余三模型检查架构与精度边界；保留全部无效单位，不按一扇因果门终止。')

if __name__=='__main__':main()
