"""All-coordinate responses, frozen fourth-set confirmation and causal-source accounting."""
import itertools
from collections import defaultdict
import numpy as np
from threadpoolctl import threadpool_limits
import phase2657_truth_answer_maps as maps_engine
from phase2620_native_coordinate_contract import *
from phase2662_symmetric_mapping_contract import OUT as CONTRACT,FAMILIES
from phase2664_symmetric_native_field import OUT as BF

OUT=RESULT/'phase2665_symmetric_coordinate_maps';OLD=RESULT/'phase2657_truth_answer_maps'


def compare():
    counts={};confirmed={};last={}
    with np.load(CONTRACT/'maps/frozen_masks.npz') as frozen:
        for fam,lang,p in itertools.product(FAMILIES,('en','zh'),(0,1)):
            with np.load(OLD/'maps/initial'/maps_engine.filename(fam,lang,p,0,0)) as z:prior={metric:np.sign(z['target__'+metric+'__mean']) for metric in ('h','mlp')}
            per={}
            for fold in ('initial','confirmation'):
                cube={}
                for q,m in itertools.product((0,1),repeat=2):
                    with np.load(OUT/'maps'/fold/maps_engine.filename(fam,lang,p,q,m)) as z:cube[q,m]={k:z[k] for k in z.files if not k.startswith('correctpair')}
                for metric in ('h','mlp'):
                    b=cube[0,0];s=np.sign(b['target__'+metric+'__mean']);dom=(b['target__'+metric+'__rms']>b['order__'+metric+'__rms'])&(b['target__'+metric+'__rms']>b['form__'+metric+'__rms']);per[fold,metric]=(s,dom)
                    for hyp in ('truth_invariant','question_affirmative','answer_label'):
                        good=dom&(s!=0)
                        for q,m in itertools.product((0,1),repeat=2):good &= np.sign(cube[q,m]['target__'+metric+'__mean'])==s*(-1)**(0 if hyp=='truth_invariant' else q if hyp=='question_affirmative' else q+m)
                        key=fold+'__'+metric+'__'+hyp;counts.setdefault(key,np.zeros_like(good,dtype='int16'));counts[key]+=good
            for metric in ('h','mlp'):
                good=per['initial',metric][1]&per['confirmation',metric][1]&(per['initial',metric][0]==prior[metric])&(per['confirmation',metric][0]==prior[metric])&(prior[metric]!=0)
                key='old_bf__'+metric;counts.setdefault(key,np.zeros_like(good,dtype='int16'));counts[key]+=good
        for metric in ('h','mlp'):
            mask=frozen['bf_'+metric].astype(bool);passed=mask&(counts['old_bf__'+metric]==32);confirmed[metric]=passed.astype('uint8')
            last[metric]={'frozen':int(mask[-1].sum()),'confirmed':int(passed[-1].sum()),'confirmed_indices':np.flatnonzero(passed[-1]).tolist(),'counts_by_layer':passed.sum(-1).tolist()}
    np.savez_compressed(OUT/'maps/allcoordinate_factor_counts.npz',**counts);np.savez_compressed(OUT/'maps/confirmed_masks.npz',**confirmed)
    return {'frozen_bf_confirmation':last,'full_direction_counts':{k:(v==32).sum(-1).tolist() for k,v in counts.items()},'boundary':'All32 groups =16family/languages*2probes; same source old signs but new symmetric question protocol, not pure new-entity-only replication.'}


def sources():
    cases=read(BF/'material/cases.json');table={tuple(r[k] for k in ('family','language','unit','form','target_index','mention_order','probe_index','polarity','mapping')):r for r in cases};sums={};ntokens=defaultdict(int);equal=exact=0;maxerr=0.;profiles=[]
    for fam,lang,fold in itertools.product(FAMILIES,('en','zh'),('initial','confirmation')):
        units=range(4) if fold=='initial' else range(4,8);ds=[]
        for unit,form,o,p,q in itertools.product(units,(0,1),(0,1),(0,1),(0,1)):
            for v in (0,1):
                pair=[]
                for m in (0,1):
                    r=table[fam,lang,unit,form,v,o,p,q,m];ci=r['case_index']
                    with np.load(BF/f'field/case_{ci:04d}.npz') as z:
                        hh=z['hidden_anchor'];pair.append((r,hh));key=f'{fold}/{fam}/{lang}/p{p}q{q}m{m}'
                        if key not in sums:sums[key]=[np.zeros_like(z['hidden_token_sum']),np.zeros_like(z['hidden_token_sumsq'])]
                        sums[key][0]+=z['hidden_token_sum'];sums[key][1]+=z['hidden_token_sumsq'];ntokens[key]+=len(r['prompt_ids'])
                    if v==1:
                        a=table[fam,lang,unit,form,0,o,p,q,m]['case_index']
                        with np.load(BF/f'field/case_{a:04d}.npz') as z:ds.append(z['hidden_anchor'].astype('float64')-hh.astype('float64'))
                if len(pair[0][0]['prompt_ids'])==len(pair[1][0]['prompt_ids']):
                    equal+=1;error=float(np.max(np.abs(pair[0][1][:,:2]-pair[1][1][:,:2])));exact+=int(error==0);maxerr=max(maxerr,error)
        d=np.stack(ds);np.savez_compressed(OUT/f'maps/{fold}_{fam}_{lang}_all_source_target.npz',mean=d.mean(0).astype('float32'),rms=np.sqrt((d*d).mean(0)).astype('float32'))
        profiles.append({'fold':fold,'family':fam,'language':lang,'pairs':len(ds),'source_role_order':['entity_a_last_token','entity_b_last_token','prompt_boundary']})
    np.savez_compressed(OUT/'maps/alltoken_coordinate_moments.npz',**{key+'__'+name:((v[0]/ntokens[key]) if name=='mean' else np.sqrt(v[1]/ntokens[key])).astype('float32') for key,v in sums.items() for name in ('mean','rms')})
    save(OUT/'analysis/source_groups.json',profiles);save(OUT/'analysis/alltoken_counts.json',dict(ntokens))
    return {'equal_length_mapping_pairs':equal,'bitwise_equal_source_anchor_pairs':exact,'maximum_source_anchor_abs_difference':maxerr,'source_groups':len(profiles),'alltoken_groups':len(sums),
        'boundary':'Source-token roles retain everycoordinate. Equal-length source identity is causal/numerical QA, not semantic discovery. Fulltoken moments include allpositions, but are not saved per-token trajectories for unshown cases.'}


def main():
    assert not (OUT/'analysis/final.json').exists();maps_engine.OUT=OUT;maps_engine.MATERIAL=BF;maps_engine.BF=BF
    with threadpool_limits(limits=4):
        reports={fold:maps_engine.get_maps(fold) for fold in ('initial','confirmation')};comparison=compare();source=sources()
    save(OUT/'analysis/map_group_counts.json',reports);checks={'256_boundary_map_groups':sum(map(len,reports.values()))==256,'32_source_groups':source['source_groups']==32,'256_alltoken_groups':source['alltoken_groups']==256,
        'frozen_masks_unchanged':sha(CONTRACT/'maps/frozen_masks.npz')==read(CONTRACT/'protocol/frozen.json')['frozen_mask_sha256']}
    assert all(checks.values());finish(2665,'对称规则全坐标图、冻结候选确认和源—输出位置分化',OUT,{'provenance':str(Path(__file__)),'summary':{'coordinates':comparison,'sources':source},'checks':checks},
        '复用经核验的全坐标均值/RMS计算内核但输出到新Phase；所有目标、顺序、形式对照及成功pair另账。旧候选掩码只用于末尾核对，不删其他坐标。',
        r'D^c_{l,j}=H^c_{l,j}(v=0)-H^c_{l,j}(v=1);\quad R_{l,j}=\sqrt{n^{-1}\sum_iD_{i,l,j}^2};\quad S_j=\mathbf1[R^{target}>R^{order},R^{target}>R^{form},\operatorname{sgn}D=\operatorname{sgn}D_{old}].',
        '8192全部原包，两实体集合各128边界组；32源位置组各128目标对；256全部token坐标矩组；4096等长度正常/反向映射源位置数值配对。H三角色37×3×2560、MLP边界36×9728均全量。',
        '逐层计数和原坐标响应同时保存，既观察早中层源角色纹理，也看末层答案分化。候选不通过新指令只能限定跨任务外推，不能否定分布式编码；新指令行为是否胜任要与图谱一起读。',
        '正常协议也改变，旧候选此轮同时面对实体和任务协议变化，不能归因为单一因素。方向假说仍只覆盖简单符号关系；源位置是选定角色token，完整逐token原始H只为展示例保留。',
        '继续固定更长规范答案的内容/格式/EOS逐参数分解及真实权重变化验证，再做14B全层模型内复验，完成整套交付。')


if __name__=='__main__':main()
