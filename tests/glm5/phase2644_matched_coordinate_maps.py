"""Basic full-coordinate crossed differences and exact all-scalar V reuse comparisons."""
import itertools,json
from collections import defaultdict
import numpy as np
from threadpoolctl import threadpool_limits
from phase2643_matched_dual_adjoint_engine import MATERIAL,BF,INITIAL,CONFIRM,LAYERS
from phase2620_native_coordinate_contract import *

OUT=RESULT/'phase2644_matched_coordinate_maps'
METRICS=('h','mlp','native_hg','common_hg','native_ag','common_ag')

def cosine(a,b):
    a=np.asarray(a,dtype=np.float64);b=np.asarray(b,dtype=np.float64)
    den=np.sqrt(np.sum(a*a,axis=-1)*np.sum(b*b,axis=-1));value=np.sum(a*b,axis=-1)
    return np.divide(value,den,out=np.zeros_like(value),where=den>0),den>0

def factors_cos(xa,ga,xb,gb):
    inner=np.sum((ga@gb.T)*(xa@xb.T))
    aa=np.sum((ga@ga.T)*(xa@xa.T));bb=np.sum((gb@gb.T)*(xb@xb.T))
    return float(inner/np.sqrt(aa*bb)) if aa>0 and bb>0 else None

def selected(path,row):
    pos=[row['entity_spans']['a'][-1],row['entity_spans']['b'][-1],len(row['prompt_ids'])-1]
    with np.load(path,allow_pickle=False) as z:
        return {'h':z['hidden'][:,pos].astype('float64'),'mlp':z['mlp_positions'].astype('float64'),
            'native_hg':z['native__hidden_adjoint_positions'].astype('float64'),'common_hg':z['common__hidden_adjoint_positions'].astype('float64'),
            'native_ag':z['native__mlp_adjoint_positions'].astype('float64'),'common_ag':z['common__mlp_adjoint_positions'].astype('float64')}

def analyze(field_set,out):
    source=INITIAL if field_set=='initial' else CONFIRM
    cases=[r for r in read(MATERIAL/'material/cases.json') if r['field_set']==field_set]
    records={r['case_index']:r for r in read(source/'analysis/records.json')}
    behavior={r['case_index']:r for r in (json.loads(s) for s in (BF/'behavior/greedy.jsonl').read_text(encoding='utf-8').splitlines())}
    table={(r['family'],r['language'],r['unit'],r['form'],r['target_index'],r['mention_order']):r for r in cases}
    units=sorted({r['unit'] for r in cases});families=sorted({r['family'] for r in cases})
    out.joinpath('field').mkdir(parents=True,exist_ok=True)
    save(out/'protocol/frozen.json',{'field_set':field_set,'units':units,'all_coordinate_primary':True,'cos_zero_norm':'unavailable with explicit valid mask, never evidence for zero similarity',
         'contrast_signs':'target0-target1; order0-order1; form0-form1; observational subtraction only, no transplantation',
         'native_gradient_comparison':'only identical ordered natural output token IDs','common_gradient_comparison':'fixed external entityA-first-token minus entityB-first-token',
         'crossfamily_pairs':'all28 family pairs within each matched language/unit/form/target/order cell','independent_entities':'unit-specific target profiles, no pooling matched cells as independent evidence'})
    summary={};edge_rows=[];low=[]
    for fam,lang in itertools.product(families,('en','zh')):
        accum={};counts=defaultdict(int);unit_profiles={};low_num=None;low_den=None
        for unit in units:
            cube={};rr={}
            for form,v,o in itertools.product((0,1),repeat=3):
                r=table[(fam,lang,unit,form,v,o)];ci=r['case_index'];rr[(form,v,o)]=r
                cube[(form,v,o)]=selected(source/f'field/case_{ci:04d}.npz',r)
                h=cube[(form,v,o)]['h'];g=cube[(form,v,o)]['common_hg'];idx=np.argsort(np.abs(h),axis=-1)[...,:h.shape[-1]//2]
                num=np.take_along_axis(g*g,idx,axis=-1).sum(-1);den=(g*g).sum(-1)
                low_num=num if low_num is None else low_num+num;low_den=den if low_den is None else low_den+den
            for kind,axis in [('target',1),('order',2),('form',0)]:
                for a in itertools.product((0,1),repeat=3):
                    if a[axis]!=0:continue
                    b=list(a);b[axis]=1;b=tuple(b);ra,rb=rr[a],rr[b]
                    native_match=records[ra['case_index']]['native_ids']==records[rb['case_index']]['native_ids']
                    for metric in METRICS:
                        if metric.startswith('native') and not native_match:continue
                        d=cube[a][metric]-cube[b][metric];key=kind+'__'+metric
                        if key not in accum:accum[key]=[np.zeros_like(d),np.zeros_like(d),np.zeros_like(d)]
                        accum[key][0]+=d;accum[key][1]+=d*d;accum[key][2]+=(d>0);counts[key]+=1
                        if kind=='target':unit_profiles.setdefault(metric,{}).setdefault(unit,[]).append(d.astype('float32'))
                        edge_rows.append({'family':fam,'language':lang,'unit':unit,'kind':kind,'metric':metric,'case_a':ra['case_index'],'case_b':rb['case_index'],
                            'both_behavior_correct':behavior[ra['case_index']]['name_content_correct'] and behavior[rb['case_index']]['name_content_correct'],
                            'rms_by_layer_position':np.sqrt(np.mean(d*d,axis=-1)).tolist()})
            del cube
        payload={};small={}
        for key,(s,q,p) in accum.items():
            n=counts[key];payload[key+'__mean']=(s/n).astype('float32');payload[key+'__rms']=np.sqrt(q/n).astype('float32');payload[key+'__positive_fraction']=(p/n).astype('float32')
            small[key]={'n':n,'rms_by_layer_position':np.sqrt(np.mean(q/n,axis=-1)).tolist()}
        for metric,profiles in unit_profiles.items():
            per=np.stack([np.mean(profiles[u],axis=0) if u in profiles else np.zeros_like(next(iter(profiles.values()))[0]) for u in units])
            # Any absent native stratum has explicit per-unit counts; its filled array is never used as a valid zero.
            payload['unit_target__'+metric]=per.astype('float32');payload['unit_target_count__'+metric]=np.array([len(profiles.get(u,[])) for u in units])
            cc,valid=cosine(per[:2].mean(0),per[2:].mean(0));valid &= all(len(profiles.get(u,[]))>0 for u in units)
            small['entity_split__'+metric]={'cosine_by_layer_position':cc.tolist(),'valid':valid.tolist(),
                'warning':'different names AND first-token readouts across entity units; transfer texture only, not isolated semantics'}
        frac=np.divide(low_num,low_den,out=np.zeros_like(low_num),where=low_den>0)
        payload['low_amplitude_half_common_adjoint_energy']=frac.astype('float32');payload['low_amplitude_half_valid']=low_den>0
        low.append({'family':fam,'language':lang,'fraction_by_layer_position':frac.tolist(),'valid':(low_den>0).tolist()})
        np.savez(out/f'field/{fam}_{lang}_fullcoordinate_maps.npz',**payload);summary[fam+'/'+lang]=small
        print(field_set,'all-coordinate maps',fam,lang,flush=True)
        del accum,payload,unit_profiles
    save(out/'analysis/response_profiles.json',summary);save(out/'analysis/edge_measures.json',edge_rows);save(out/'analysis/low_amplitude_energy.json',low)
    # All scalar entries of G = g.T @ x are represented exactly, without a low-rank fit or coordinate pruning.
    pair_rows=[];exact_checks=[]
    for lang,unit,form,v,o in itertools.product(('en','zh'),units,(0,1),(0,1),(0,1)):
        data={}
        for fam in families:
            r=table[(fam,lang,unit,form,v,o)];ci=r['case_index']
            with np.load(source/f'field/case_{ci:04d}.npz',allow_pickle=False) as z:
                data[fam]={'case':r,'h':z['hidden'][:,-1].astype('float64')}
                for l in LAYERS:
                    data[fam][(l,'x')]=z[f'L{l}_v_x'].astype('float64')
                    for obj in ('native','common'):data[fam][(l,obj)]=z[f'{obj}__L{l}_v_g'].astype('float64')
        for fa,fb in itertools.combinations(families,2):
            a,b=data[fa],data[fb];ra,rb=a['case'],b['case'];ia,ib=ra['case_index'],rb['case_index'];native_match=records[ia]['native_ids']==records[ib]['native_ids']
            hc,hvalid=cosine(a['h'],b['h'])
            item={'case_a':ia,'case_b':ib,'family_a':fa,'family_b':fb,'language':lang,'unit':unit,'form':form,'target':v,'order':o,
                'native_ids_match':native_match,'both_behavior_correct':behavior[ia]['name_content_correct'] and behavior[ib]['name_content_correct'],
                'raw_h_cos_by_layer':hc.tolist(),'raw_h_valid':hvalid.tolist(),'V':{}}
            for l,obj in itertools.product(LAYERS,('native','common')):
                if obj=='native' and not native_match:continue
                xa,xb=a[(l,'x')],b[(l,'x')];ga,gb=a[(l,obj)],b[(l,obj)]
                value=factors_cos(xa,ga,xb,gb);item['V'][f'L{l}/{obj}']=value
                if len(exact_checks)<4 and obj=='common' and l not in [r['layer'] for r in exact_checks]:
                    explicit,_=cosine((ga.T@xa).ravel(),(gb.T@xb).ravel());error=abs(float(explicit)-value)
                    assert error<1e-10;exact_checks.append({'layer':l,'absolute_error':error,'all_scalar_entries':1024*2560})
            pair_rows.append(item)
        del data
    save(out/'analysis/crossfamily_pairs.json',pair_rows);save(out/'analysis/factor_identity_checks.json',exact_checks)
    pair_summary={}
    for l,obj in itertools.product(LAYERS,('native','common')):
        key=f'L{l}/{obj}';vals=[r['V'][key] for r in pair_rows if r['V'].get(key) is not None];good=[r['V'][key] for r in pair_rows if r['V'].get(key) is not None and r['both_behavior_correct']]
        pair_summary[key]={'n':len(vals),'mean':float(np.mean(vals)) if vals else None,'min':min(vals) if vals else None,'max':max(vals) if vals else None,
            'both_correct_n':len(good),'both_correct_mean':float(np.mean(good)) if good else None}
    checks={'all16_groups':len(summary)==16,'all1792_matched_crossfamily_pairs':len(pair_rows)==1792,'exact_fullmatrix_factor_checks':len(exact_checks)==4 and all(r['absolute_error']<1e-10 for r in exact_checks)}
    assert all(checks.values())
    compact={'field_set':field_set,'cases':len(cases),'matched_crossfamily_pairs':len(pair_rows),'native_id_matched_pairs':sum(r['native_ids_match'] for r in pair_rows),
        'all_coordinate_edges':len(edge_rows),'crossfamily_V':pair_summary,'independent_entity_units':units,'groups':16,
        'dependence':'1792 pairs share512 prompts and4 name-pairs perlanguage, not1792 independent replications',
        'coverage':'all native coordinates at two entity anchors/current boundary; full-token V parameter derivative; no vocabulary-wide semantic inference'}
    save(out/'analysis/map_completion.json',{'summary':compact,'checks':checks})
    return compact,checks

def main():
    with threadpool_limits(limits=4):summary,checks=analyze('initial',OUT)
    finish(2644,'八族全坐标目标/顺序/句式响应与同实体跨族参数复用',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '在三因子立方体上分别改变正确目标、提及顺序、句式，记录所有坐标的有符号均值、RMS和符号频率；同人名条件下穷举28个跨族组合。原生输出对不同的梯度不混成同一读出。',
        r'D_v=H_{v=0}-H_{v=1};\quad D_o=H_{o=0}-H_{o=1};\quad D_s=H_{s=0}-H_{s=1};\quad \langle G_a,G_b\rangle_F=\sum_{t,u}\langle g_{a,t},g_{b,u}\rangle\langle x_{a,t},x_{b,u}\rangle.',
        '512条件、16族语言组，各单位内部12条立方体边；目标/顺序/句式分账，原生梯度只有相同有序token对才比较。所有1792同实体跨族对的四层V全参数余弦，四层显式2621440参数展开核验。',
        '参数梯度的精确因子是计算恒等式，不是数据压缩或低秩近似。全坐标图谱保留低值；低幅值半区梯度能量仅是敏感度描述。初始0/1与30/31实体分割的目标响应纹理给出可复核的迁移边界，不将名称或读出变化隐藏。',
        '相同输出要求会造成晚层公共结构；外部共同读出更不能单独证明共同语义。跨族任务正文不同，同实体不是完全隔离语义。目标差分是观察性对照，不是把donor移植到模型。样本对共享原样本，不具备独立性。',
        '继续另一组预冻结独立实体512条件，不因某一匹配余弦低或行为失败换路线；扩大后做固定标量参数的真实前向检验。')

if __name__=='__main__':main()
