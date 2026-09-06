"""Full-coordinate matched contrasts and finite product accounting, not a causal allocation."""
import itertools,shutil
import numpy as np
from phase2620_native_coordinate_contract import *
from phase2670_native_mlp_contract import OUT as CONTRACT,FIELD,LAYERS,SITES,FAMILIES
from phase2671_native_mlp_field import unbits,moment_group
from phase2657_truth_answer_maps import filename

OUT=RESULT/'phase2673_native_mlp_confirmation'
OLD=RESULT/'phase2657_truth_answer_maps'
AXES=('unit','content_instance','form','mention_order','probe_index','polarity','mapping','target_index')

def silu(g):return g/(1+np.exp(-g))

def pairs(rows,axis):
    groups={}
    for i,r in enumerate(rows):groups.setdefault(tuple(r[k] for k in AXES if k!=axis),{})[r[axis]]=i
    return np.array([tuple(v[k] for k in sorted(v)) for v in groups.values()],dtype=int)

def main():
    assert (RESULT/'phase2672_native_mlp_paths/analysis/final.json').exists() and not (OUT/'analysis/final.json').exists()
    cases=read(CONTRACT/'material/cases.json');OUT.joinpath('maps').mkdir(parents=True,exist_ok=True)
    estimate=256*5*2*(37*2560+36*9728)*4+32*36*9728*4*4
    assert shutil.disk_usage(OUT).free>estimate+8*1024**3,('derived_map_budget',estimate,shutil.disk_usage(OUT).free)
    save(OUT/'protocol/frozen.json',{'contrasts':['target_index','form','mention_order','content_instance','unit'],'map_upper_bytes':estimate,
        'gate':'Same old4B q0m0 means and target>form/order on both newfolds; p0/p1 and all16family/languages. Allotherq/m and content/entity effects reported, no new selectivity requirement chosen fromeffects.',
        'finite_expansion':'SiLU evaluated float64 on actual BF16 gate; epsilon=aBF16-SiLU64(g)*u absorbs activation/multiply rounding. Product identity +deltaepsilon exact in real arithmetic. Linear SiLU tangent and remainder separate; baseline0 choice not unique causal attribution.',
        'full_background':'Every task-boundary H coordinate and product unit for target/form/order/content/entity contrasts at allp/q/m. Alltoken gate/up/x/down background remains phase2671. No component necessity gate.'})
    oldcounts={'h':np.zeros((37,2560),dtype=np.int16),'a':np.zeros((36,9728),dtype=np.int16)};candidate_rows=[];identities=[];positive_sets={}
    with np.load(CONTRACT/'maps/frozen_masks.npz') as z:frozen={'h':z['q4__h'].astype(bool),'a':z['q4__mlp'].astype(bool)}
    for fam,lang,fold in itertools.product(FAMILIES,('en','zh'),('initial','confirmation')):
        rr=[r for r in cases if (r['family'],r['language'],r['field_set'])==(fam,lang,fold)];data={k:[] for k in ('h','a','gate','up','x','down','pre_mlp','attention')}
        for r in rr:
            with np.load(FIELD/f'field/case_{r["case_index"]:04d}.npz') as z:
                for k in data:data[k].append(unbits(z[k])[:,-1] if k in ('h','a') else unbits(z[k]))
        data={k:np.stack(v) for k,v in data.items()}
        for p,q,m in itertools.product((0,1),repeat=3):
            selection=[i for i,r in enumerate(rr) if (r['probe_index'],r['polarity'],r['mapping'])==(p,q,m)];sub=[rr[i] for i in selection];mp={}
            for label,axis in [('target','target_index'),('form','form'),('order','mention_order'),('content','content_instance'),('entity','unit')]:
                ij=pairs(sub,axis);assert ij.shape==(16,2)
                for metric in ('h','a'):
                    xx=data[metric][selection];d=xx[ij[:,0]].astype('float64')-xx[ij[:,1]]
                    mp[f'{label}__{metric}__mean']=d.mean(0).astype('float32');mp[f'{label}__{metric}__rms']=np.sqrt((d*d).mean(0)).astype('float32')
            directory=OUT/'maps'/fold;directory.mkdir(exist_ok=True)
            np.savez_compressed(directory/filename(fam,lang,p,q,m),**mp)
            for metric in ('h','a'):
                s=np.sign(mp[f'target__{metric}__mean']);dom=(mp[f'target__{metric}__rms']>mp[f'form__{metric}__rms'])&(mp[f'target__{metric}__rms']>mp[f'order__{metric}__rms'])
                with np.load(OLD/'maps/initial'/filename(fam,lang,p,0,0)) as old:
                    prior=np.sign(old[f'target__{"mlp" if metric=="a" else "h"}__mean'])
                if (q,m)==(0,0):oldcounts[metric]+=(s==prior)&(prior!=0)&dom
                for l,j in np.argwhere(frozen[metric]):
                    candidate_rows.append({'family':fam,'language':lang,'fold':fold,'p':p,'q':q,'m':m,'metric':metric,'layer':int(l),'coordinate':int(j),
                        'old_direction':bool(s[l,j]==prior[l,j] and prior[l,j]!=0),'dominates_form_order':bool(dom[l,j]),'contrasts':{k:float(v[l,j]) for k,v in mp.items() if '__'+metric+'__' in k}})
            del mp
        ij=pairs(rr,'target_index');assert ij.shape==(128,2);stats={};maxerr=0.;cand=[]
        for left,right in ij:
            g0,g1=data['gate'][left].astype('float64'),data['gate'][right].astype('float64');u0,u1=data['up'][left].astype('float64'),data['up'][right].astype('float64')
            a0,a1=data['a'][left].astype('float64'),data['a'][right].astype('float64');s0,s1=silu(g0),silu(g1);ds=s1-s0;du=u1-u0
            epsilon=(a1-s1*u1)-(a0-s0*u0);gate_term=u0*ds;up_term=s0*du;interaction=ds*du;delta=a1-a0
            error=delta-(gate_term+up_term+interaction+epsilon);maxerr=max(maxerr,float(np.abs(error).max()))
            sig=1/(1+np.exp(-g0));nonlinear=ds-(sig+g0*sig*(1-sig))*(g1-g0)
            vals={'gate':gate_term,'up':up_term,'interaction':interaction,'rounding':epsilon,'silu_nonlinear':nonlinear,'actual_delta':delta}
            for k,v in vals.items():
                for suffix,w in [('mean',v),('rms',v*v)]:
                    name=k+'__'+suffix
                    if name not in stats:stats[name]=np.zeros_like(w)
                    stats[name]+=w
            for l,j in SITES:
                li=LAYERS.index(l);r0=rr[int(left)]
                cand.append({'case_pair':[r0['case_index'],rr[int(right)]['case_index']],'layer':l,'neuron':j,'terms':{k:float(v[l,j]) for k,v in vals.items()},
                    'bypass_full_coordinate_l1':{'attention':float(np.abs(data['attention'][right,li]-data['attention'][left,li]).sum()),
                        'normalized_mlp_input':float(np.abs(data['x'][right,li]-data['x'][left,li]).sum()),'pre_norm_residual':float(np.abs(data['pre_mlp'][right,li]-data['pre_mlp'][left,li]).sum()),
                        'actual_full_mlp_output':float(np.abs(data['down'][right,li]-data['down'][left,li]).sum())}})
        key=f'{fam}_{lang}_{fold}';np.savez_compressed(OUT/f'maps/product_{key}.npz',**{k:(v/128 if k.endswith('mean') else np.sqrt(v/128)).astype('float32') for k,v in stats.items()})
        save(OUT/f'analysis/product_{key}.json',{'pairs':128,'maximum_identity_error':maxerr,'candidate_rows':cand,'boundary':'ReportedL1profiles are descriptive, not causal fractions; RMSNorm and Attention measured, not assigned uniquely.'})
        identities.append({'group':key,'pairs':128,'maximum_identity_error':maxerr});del data,stats
        print('native fullcoordinate confirmation',key,flush=True)
    passed={k:(frozen[k]&(v==64)).astype('uint8') for k,v in oldcounts.items()};np.savez_compressed(OUT/'maps/confirmed_masks.npz',**passed);np.savez_compressed(OUT/'maps/old_gate_counts.npz',**oldcounts)
    save(OUT/'analysis/candidate_scope.json',candidate_rows);save(OUT/'analysis/product_identities.json',identities)
    summary={'surviving_sites':{k:np.argwhere(v).tolist() for k,v in passed.items()},'gate':'All64groups =32family/language/probe×bothfolds; q0/m0 only',
        'candidate_scope_groups':len(candidate_rows),'maximum_product_identity_error':max(r['maximum_identity_error'] for r in identities),'paired_fullproduct_fields':sum(r['pairs'] for r in identities)}
    checks={'1792_candidate_scope_rows':len(candidate_rows)==1792,'4096_full_MLP_pairs':summary['paired_fullproduct_fields']==4096,'256_allcoordinate_maps':sum(len(list((OUT/'maps'/f).glob('*.npz'))) for f in ('initial','confirmation'))==256,
        'finite_product_accounting':summary['maximum_product_identity_error']<1e-10,'no_weight_changes':True}
    assert all(checks.values())
    finish(2673,'实体/内容/形式/顺序全坐标确认与非线性乘积/旁路分账',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '每个p/q/m格保留全部H和MLP乘积坐标的五种精确配对差异，再核对先前冻结门；对全部4096目标对、所有MLP单元展开门支路、上支路、交互及BF16舍入。',
        r'\Delta a=s_0\Delta u+u_0\Delta s+\Delta s\Delta u+\Delta\epsilon;\quad \Delta s=s^\prime(g_0)\Delta g+R_{SiLU};\quad \epsilon=a_{BF16}-s(g)u.',
        'C001256全坐标组×目标/形式/顺序/内容/实体五对照；C0021792冻结候选分组数值；C0034096全MLP乘积恒等核验；C00432全层门/上支路/交互/舍入/SiLU余项图；C005四层实际Attention、归一化前后、MLP输出旁路描述。',
        '实体与内容的变化被拆开，使某坐标在多语言操作间的复用有更清楚边界。通过或失败均保留，乘积分解能区分变化来自哪条已知计算支路，而不是默认整个神经元只编码一个语义。',
        '分解依基点，不给出独占因果份额；RMSNorm、Attention旁路这里只是实测描述而非机制闭合。基本加法乘法恒等式不是新数学。正文原场保留，主确认指标仍在任务边界；两个内容实例不能代表无限组合。',
        '继续至少128新前缀真实gate/up/down标量的小剂量完整序列验证，并顺序进行三模型复验；不因冻结门阴性关闭该原生研究路线。')

if __name__=='__main__':main()
