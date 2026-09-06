"""Every native coordinate: frozen old masks, new factor responses and sign alternatives."""
import itertools
from collections import defaultdict
import numpy as np
from threadpoolctl import threadpool_limits
from phase2620_native_coordinate_contract import *
from phase2655_truth_answer_contract import OUT as MATERIAL,FAMILIES,leading_answer
from phase2656_truth_answer_behavior import OUT as BF

OUT=RESULT/'phase2657_truth_answer_maps'


def filename(fam,lang,p,q,m):return f'{fam}_{lang}_p{p}q{q}m{m}.npz'


def get_maps(fold):
    cases=[r for r in read(MATERIAL/'material/cases.json') if r['field_set']==fold];behavior={r['case_index']:r for r in read(BF/'analysis/records.json')}
    table={(r['family'],r['language'],r['unit'],r['form'],r['target_index'],r['mention_order'],r['probe_index'],r['polarity'],r['mapping']):r for r in cases}
    folder=OUT/'maps'/fold;folder.mkdir(parents=True,exist_ok=True);reports=[]
    for fam,lang,p,q,m in itertools.product(FAMILIES,('en','zh'),(0,1),(0,1),(0,1)):
        accum={};n=defaultdict(int);success=0;pairtotal=0
        for unit in sorted({r['unit'] for r in cases}):
            cube={};ok={}
            for form,v,o in itertools.product((0,1),repeat=3):
                r=table[(fam,lang,unit,form,v,o,p,q,m)];ci=r['case_index']
                with np.load(BF/f'field/case_{ci:04d}.npz',allow_pickle=False) as z:cube[(form,v,o)]={'h':z['hidden_boundary'].astype('float64'),'mlp':z['mlp_boundary'].astype('float64')}
                ok[(form,v,o)]=behavior[ci]['content_correct']
            for kind,axis in [('target',1),('order',2),('form',0)]:
                for a in itertools.product((0,1),repeat=3):
                    if a[axis]:continue
                    b=list(a);b[axis]=1;b=tuple(b)
                    if kind=='target':pairtotal+=1;success+=int(ok[a] and ok[b])
                    for metric in ('h','mlp'):
                        d=cube[a][metric]-cube[b][metric];key=kind+'__'+metric
                        if key not in accum:accum[key]=[np.zeros_like(d),np.zeros_like(d)]
                        accum[key][0]+=d;accum[key][1]+=d*d;n[key]+=1
                        if kind=='target' and ok[a] and ok[b]:
                            key='correctpair_target__'+metric
                            if key not in accum:accum[key]=[np.zeros_like(d),np.zeros_like(d)]
                            accum[key][0]+=d;accum[key][1]+=d*d;n[key]+=1
        maps={}
        for key,(s,ss) in accum.items():maps[key+'__mean']=(s/n[key]).astype('float32');maps[key+'__rms']=np.sqrt(ss/n[key]).astype('float32')
        np.savez_compressed(folder/filename(fam,lang,p,q,m),**maps)
        reports.append({'family':fam,'language':lang,'probe':p,'polarity':q,'mapping':m,'all_target_pairs':pairtotal,'both_correct_target_pairs':success,'counts':dict(n),
            'boundary':'All-pair primary maps never screened; correctpair maps are conditional diagnostics and absent if no eligible pair, not zero.'})
        if len(reports)%16==0:print(fold,'allcoordinate factor maps',len(reports),'/128',flush=True)
    return reports


def compare():
    coverage={};oldchecks={};summary={};candidate_summary={}
    with np.load(MATERIAL/'maps/frozen_previous_coordinates.npz',allow_pickle=False) as old:
        for fam,lang,p in itertools.product(FAMILIES,('en','zh'),(0,1)):
            per={}
            for fold in ('initial','confirmation'):
                arrays={}
                for q,m in itertools.product((0,1),repeat=2):
                    with np.load(OUT/'maps'/fold/filename(fam,lang,p,q,m)) as z:arrays[q,m]={k:z[k] for k in z.files if not k.startswith('correctpair')}
                for metric in ('h','mlp'):
                    zero=arrays[0,0];sign=np.sign(zero['target__'+metric+'__mean']);dominant=(zero['target__'+metric+'__rms']>zero['order__'+metric+'__rms'])&(zero['target__'+metric+'__rms']>zero['form__'+metric+'__rms'])
                    per[fold,metric]=(sign,dominant)
                    for hypothesis in ('truth_invariant','question_affirmative','answer_label'):
                        good=dominant&(sign!=0)
                        for q,m in itertools.product((0,1),repeat=2):
                            direction=1 if hypothesis=='truth_invariant' else (-1)**(q if hypothesis=='question_affirmative' else q+m)
                            good &= np.sign(arrays[q,m]['target__'+metric+'__mean'])==direction*sign
                        key=fold+'__'+metric+'__'+hypothesis
                        coverage.setdefault(key,np.zeros_like(good,dtype='int16'));coverage[key]+=good
            for metric in ('h','mlp'):
                for prior_precision in ('bf','fp'):
                    oldmetric=('bf_' if prior_precision=='bf' else '')+metric
                    s=old[f'{fam}_{lang}_truth_{"a" if p==0 else "b"}__{oldmetric}__sign'];s=s[:,2] if metric=='h' else s
                    for kind in ('amplitude','signed'):
                        good=per['initial',metric][1]&per['confirmation',metric][1]
                        if kind=='signed':good &= (per['initial',metric][0]==s)&(per['confirmation',metric][0]==s)&(s!=0)
                        key=prior_precision+'__'+metric+'__'+kind;oldchecks.setdefault(key,np.zeros_like(good,dtype='int16'));oldchecks[key]+=good
        for metric,l in [('h',36),('mlp',35)]:
            summary[metric]={k:int((v[l]==32).sum()) for k,v in coverage.items() if '__'+metric+'__' in k}
            for prec in ('bf','fp'):
                om=('bf_' if prec=='bf' else '')+metric;mask=old[om+'__truth_oriented_opposite'];mask=mask[:,2] if metric=='h' else mask
                candidate_summary[prec+'/'+metric]={'prior_frozen_boundary_candidates':int(mask[l].sum()),
                    'all32groups_8newentities_amplitude':int(((oldchecks[prec+'__'+metric+'__amplitude'][l]==32)&mask[l].astype(bool)).sum()),
                    'all32groups_8newentities_signed':int(((oldchecks[prec+'__'+metric+'__signed'][l]==32)&mask[l].astype(bool)).sum()),
                    'precision_boundary':'BF16 new forward; FP means applying previously FP32-derived physical masks, NOT collecting new fullFP32 confirmation.'}
    np.savez_compressed(OUT/'maps/allcoordinate_factor_sign_counts.npz',**coverage,**{'old__'+k:v for k,v in oldchecks.items()})
    result={'candidate_confirmation':candidate_summary,'new_allcoordinate_sign_hypotheses':summary,
        'boundary':'All32groups=16family/languages x2probes. Fixed original q0/m0 template for entity confirmation. New sign hypotheses describe all-coordinate patterns; reverse mapping failure can make question-affirmative look stable. Do not call output-aligned coordinates pure truth or semantic units.'}
    save(OUT/'analysis/coordinate_confirmation.json',result);return result


def main():
    assert not (OUT/'analysis/final.json').exists()
    with threadpool_limits(limits=4):reports={fold:get_maps(fold) for fold in ('initial','confirmation')};summary=compare()
    behavior=read(BF/'analysis/records.json');retracted=[r['case_index'] for r in behavior if r['decision'] is not None and leading_answer(r['generated'],r['language'])!=r['decision']['answer_yes']]
    summary['provisional_answer_time_audit']={'n':len(behavior),'first_prefix_label_differs_from_final_leading_label':len(retracted),'case_indices':retracted,
        'boundary':'First currently recognizable decoded prefix is provisional; later tokens may extend or retract it. Not a proof of irrevocable model commitment or hidden semantic birth.'}
    save(OUT/'analysis/provisional_answer_time_audit.json',summary['provisional_answer_time_audit'])
    save(OUT/'analysis/map_group_counts.json',reports);checks={'256_complete_native_map_groups':sum(map(len,reports.values()))==256,'four_frozen_mask_reports':len(summary['candidate_confirmation'])==4}
    finish(2657,'旧坐标第三实体确认与事实/提问/答案方向全坐标图谱',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '每个原生坐标分别计算目标、顺序、句式变化的均值与RMS，不挑大激活。正常旧提问模板确认冻结候选；其余问题极性与答案映射交叉检验真值不变、肯定性翻转、答案标签翻转三种简单方向规则。',
        r'R^c_{l,j}=\sqrt{n^{-1}\sum_i(\Delta_cH_{i,l,j})^2};\quad s_{truth}=1,\ s_{affirm}=(-1)^q,\ s_{answer}=(-1)^{q+m};\quad C_{l,j}^{s}=\sum_{g,p}\mathbf1[\operatorname{sgn}D^{q,m}_{g,p,l,j}=s\operatorname{sgn}D^{0,0}_{g,p,l,j}].',
        '8192自然原包构成两套4096条件图谱，各128族语言探问极性映射组。两集合共8新实体对/语言。H每层2560坐标、MLP每层9728单位完整保留，成功pair图仅作另账条件观察。',
        '旧候选确认的新增数据为BF16；套用旧FP32候选只表示坐标集合迁移，不冒称新全FP32复现。真值与答案准备的鉴别应同时看自然映射执行情况，不能把失败任务中的不翻转解释为纯语义。',
        '每组只有4实体对/集合且词表/句型复用；全32组规则很严格但不是普遍性定理。候选或新规则阴性也不否认分布式使用。三个方向规则覆盖有限简单假说，不排除条件混合编码。',
        '继续完整答案加EOS的全token共享参数因子，而不是再把一个首位logit当作完整输出；随后独立真实标量与第二模型复验。')


if __name__=='__main__':main()
