"""Crossed-coordinate response maps and causal-prefix/source-sensitivity accounting."""
import itertools,json
from collections import defaultdict
import numpy as np
from threadpoolctl import threadpool_limits
from phase2620_native_coordinate_contract import *
from phase2650_output_function_adjoints import MATERIAL,BF,INITIAL,CONFIRM,LAYERS
from phase2644_matched_coordinate_maps import cosine,factors_cos

OUT=RESULT/'phase2651_output_function_maps'
MODES=('name','cloze','truth_a','truth_b')

def selected(source,ci):
    with np.load(source/f'field/case_{ci:04d}.npz') as z,np.load(BF/f'field/case_{ci:04d}.npz') as b:
        return {'h':z['hidden_positions'].astype('float64'),'mlp':z['mlp_boundary'].astype('float64'),
            'common_hg':z['common__hidden_adjoint_boundary'].astype('float64'),'common_ag':z['common__mlp_adjoint_boundary'].astype('float64'),
            'bf_h':b['hidden_positions'].astype('float64'),'bf_mlp':b['mlp_boundary'].astype('float64')}

def analyze(field_set,out):
    source=INITIAL if field_set=='initial' else CONFIRM;out.joinpath('maps').mkdir(parents=True,exist_ok=True)
    cases=[r for r in read(MATERIAL/'material/cases.json') if r['field_set']==field_set]
    records={r['case_index']:r for r in read(source/'analysis/records.json')};units=sorted({r['unit'] for r in cases});families=sorted({r['family'] for r in cases})
    table={(r['family'],r['language'],r['unit'],r['form'],r['target_index'],r['mention_order'],r['mode']):r for r in cases}
    behaviors={r['case_index']:r for r in (json.loads(s) for s in (BF/'behavior/greedy.jsonl').read_text(encoding='utf-8').splitlines())}
    save(out/'protocol/map_rule.json',{'field_set':field_set,'units':units,'scope':'allH3anchor coordinates, allMLPboundary units, both naturalBF16/FP32, fixedcommon readout adjoints',
        'contrast':'rawtarget0-target1, order0-order1, form0-form1. Truth-oriented target is -rawtarget fortruth_b only. All original raw profiles retained.',
        'prefix_check':'same preceding token IDs at source anchors are checked before source H/V equality. Any nonzero numeric error reported, never claimed as future causing past change.',
        'Vcomparison':'exactfullparameter common-gradient cosine only when ordered common output IDs identical; every token term, noTopK'})
    summaries={};targetprofiles={};coverage={};low_energy=[]
    for fam,lang,mode in itertools.product(families,('en','zh'),MODES):
        accum={};counts=defaultdict(int);unit_target=defaultdict(dict);numer=None;denom=None
        for unit in units:
            cube={}
            for form,v,o in itertools.product((0,1),repeat=3):
                r=table[(fam,lang,unit,form,v,o,mode)];ci=r['case_index'];cube[(form,v,o)]=selected(source,ci)
                h=cube[(form,v,o)]['h'][:,2];g=cube[(form,v,o)]['common_hg'];idx=np.argsort(np.abs(h),axis=-1)[...,:1280]
                nn=np.take_along_axis(g*g,idx,axis=-1).sum(-1);dd=(g*g).sum(-1);numer=nn if numer is None else numer+nn;denom=dd if denom is None else denom+dd
            for kind,axis in [('target',1),('order',2),('form',0)]:
                for a in itertools.product((0,1),repeat=3):
                    if a[axis]!=0:continue
                    b=list(a);b[axis]=1;b=tuple(b)
                    for metric in cube[a]:
                        d=cube[a][metric]-cube[b][metric];key=kind+'__'+metric
                        if key not in accum:accum[key]=[np.zeros_like(d),np.zeros_like(d),np.zeros_like(d)]
                        accum[key][0]+=d;accum[key][1]+=d*d;accum[key][2]+=(d>0);counts[key]+=1
                        if kind=='target':unit_target[metric].setdefault(unit,[]).append(d.astype('float32'))
            del cube
        maps={};ss={}
        for key,(s,q,p) in accum.items():
            n=counts[key];maps[key+'__mean']=(s/n).astype('float32');maps[key+'__rms']=np.sqrt(q/n).astype('float32');maps[key+'__positive_fraction']=(p/n).astype('float32')
            ss[key]={'n':n,'rms_by_layer_position':np.sqrt(np.mean(q/n,axis=-1)).tolist()}
        for metric,per in unit_target.items():
            profiles=np.stack([np.mean(per[u],axis=0) for u in units]);maps['unit_target__'+metric]=profiles
            cc,valid=cosine(profiles[:2].mean(0),profiles[2:].mean(0));ss['entity_half_split__'+metric]={'cos':cc.tolist(),'valid':valid.tolist()}
            targetprofiles[(fam,lang,mode,metric)]=maps['target__'+metric+'__mean']
            if metric in ('h','mlp','bf_h','bf_mlp'):
                t=maps['target__'+metric+'__rms'];mask=(t>0)&(t>maps['order__'+metric+'__rms'])&(t>maps['form__'+metric+'__rms'])
                key=mode+'__'+metric
                if key not in coverage:coverage[key]=np.zeros_like(mask,dtype='int16')
                coverage[key]+=mask
                ss['dominant_fraction__'+metric]=mask.mean(-1).tolist()
        fraction=np.divide(numer,denom,out=np.zeros_like(numer),where=denom>0);low_energy.append({'group':fam+'/'+lang+'/'+mode,'fraction':fraction.tolist(),'valid':(denom>0).tolist()})
        np.savez(out/f'maps/{fam}_{lang}_{mode}.npz',**maps);summaries[fam+'/'+lang+'/'+mode]=ss
        print(field_set,'output response maps',fam,lang,mode,flush=True)
    np.savez(out/'maps/coordinate_envelope_coverage.npz',**coverage);save(out/'analysis/response_profiles.json',summaries);save(out/'analysis/low_amplitude_energy.json',low_energy)
    direction_pairs=[]
    for fam,lang,metric in itertools.product(families,('en','zh'),('h','mlp','bf_h','bf_mlp')):
        for a,b in itertools.combinations(MODES,2):
            x,y=targetprofiles[(fam,lang,a,metric)],targetprofiles[(fam,lang,b,metric)]
            raw,valid=cosine(x,y);sign=(-1 if a=='truth_b' else 1)*(-1 if b=='truth_b' else 1)
            direction_pairs.append({'family':fam,'language':lang,'metric':metric,'mode_a':a,'mode_b':b,'raw_semantic_target_cos':raw.tolist(),'truth_oriented_cos':(raw*sign).tolist(),'valid':valid.tolist(),
                'boundary':'Truth orientation is a predefined label convention, not a learned sign or proof of shared mechanism.'})
    save(out/'analysis/cross_function_target_directions.json',direction_pairs);del targetprofiles
    source_rows=[]
    for fam,lang,unit,form,v,o in itertools.product(families,('en','zh'),units,(0,1),(0,1),(0,1)):
        packs={}
        for mode in MODES:
            r=table[(fam,lang,unit,form,v,o,mode)];ci=r['case_index']
            with np.load(source/f'field/case_{ci:04d}.npz') as z:
                packs[mode]={'r':r,'h':z['hidden_positions']}
                for l in LAYERS:
                    for key in ('x','value'):packs[mode][(l,key)]=z[f'L{l}_v_{key}'].astype('float64')
                    packs[mode][(l,'g')]=z[f'common__L{l}_v_g'].astype('float64')
        for ma,mb in itertools.combinations(MODES,2):
            a,b=packs[ma],packs[mb];ra,rb=a['r'],b['r'];ia,ib=ra['case_index'],rb['case_index'];pa,pb=records[ia]['positions'],records[ib]['positions'];same=ra['common_readout_ids']==rb['common_readout_ids']
            prefix_same=[ra['prompt_ids'][:pa[k]+1]==rb['prompt_ids'][:pb[k]+1] for k in (0,1)]
            h_error=[float(np.max(np.abs(a['h'][:,k]-b['h'][:,k]))) for k in (0,1)]
            hcos,hvalid=cosine(a['h'][:,2],b['h'][:,2]);measures={}
            for l in LAYERS:
                value_error=[float(np.max(np.abs(a[(l,'value')][pa[k]]-b[(l,'value')][pb[k]]))) for k in (0,1)]
                z={'source_value_max_error':value_error}
                if same:
                    z['source_adjoint_cos']=[float(cosine(a[(l,'g')][pa[k]],b[(l,'g')][pb[k]])[0]) if np.linalg.norm(a[(l,'g')][pa[k]])*np.linalg.norm(b[(l,'g')][pb[k]])>0 else None for k in (0,1)]
                    z['all_scalar_gradient_cos']=factors_cos(a[(l,'x')],a[(l,'g')],b[(l,'x')],b[(l,'g')])
                measures[str(l)]=z
            source_rows.append({'case_a':ia,'case_b':ib,'family':fam,'language':lang,'mode_a':ma,'mode_b':mb,'source_prefix_ids_equal':prefix_same,
                'source_hidden_max_error':h_error,'same_common_readout':same,'boundary_h_cos':hcos.tolist(),'boundary_h_valid':hvalid.tolist(),'V':measures,
                'both_behavior_correct':behaviors[ia]['content_correct'] and behaviors[ib]['content_correct']})
        del packs
    save(out/'analysis/cross_function_source_traces.json',source_rows)
    comparisons={}
    for a,b in [('name','cloze'),('truth_a','truth_b')]:
        rr=[r for r in source_rows if (r['mode_a'],r['mode_b'])==(a,b)];comparisons[a+'/'+b]={}
        for l in LAYERS:
            vv=[r['V'][str(l)]['all_scalar_gradient_cos'] for r in rr if r['V'][str(l)].get('all_scalar_gradient_cos') is not None]
            comparisons[a+'/'+b][str(l)]={'n':len(vv),'mean_all_scalar_gradient_cos':float(np.mean(vv)) if vv else None}
    checks={'all64_response_groups':len(summaries)==64,'all3072_crossfunction_source_pairs':len(source_rows)==3072,'all_source_prefixes_match':all(all(r['source_prefix_ids_equal']) for r in source_rows)}
    assert all(checks.values());summary={'field_set':field_set,'cases':len(cases),'response_groups':64,'crossfunction_pairs':len(source_rows),
        'maximum_source_hidden_numeric_error':max(max(r['source_hidden_max_error']) for r in source_rows),
        'same_readout_parameter_comparisons':comparisons,'source_boundary':'Matching causal token prefixes verified; tiny shape-dependent numerical errors not future semantics affecting prior states.',
        'all_coordinate_modes':MODES,'independent_entity_units':units}
    report={'summary':summary,'checks':checks,'all_checks_passed':True};save(out/'analysis/map_completion.json',report);return report

def main():
    with threadpool_limits(limits=4):r=analyze('initial',OUT)
    finish(2651,'固定输出与更换问题的全坐标包络及源位置条件响应',OUT,{'provenance':str(Path(__file__)),'summary':r['summary'],'checks':r['checks']},
        '完整交叉立方体逐坐标计算目标/顺序/句式响应，保留BF16自然场及FP32场。核查不同输出任务在源实体锚点之前的token前缀一致，比较源H/V原值与固定输出读出下的V伴随变化。',
        r'I_{g,l,j}=\mathbf1[R^{target}_{g,l,j}>\max(R^{order}_{g,l,j},R^{form}_{g,l,j})];\quad D_{truth}=(-1)^pD_{semantic};\quad G_{jk}=\sum_t\bar V_{t,j}X_{t,k}.',
        '2048初始条件、64族语言模式组、全部原生坐标；同一body四模式产生3072跨模式对。只有common输出IDs完全相同才比较其参数导数；name/cloze和truth_a/truth_b各512对，四层全参数精确因子余弦。',
        '相同因果前缀的源位置状态应由相同先前输入决定，未来问题可改变输出敏感度；这种结构来自因果网络计算，不是新发现的语义法则。真正需要解释的是条件下哪些原坐标响应和方向跨实体保留。固定truth输出词是对name词汇混杂的明确控制。',
        '全场幅度占优不意味着语义专用性；问题和输出功能同时变化，给定cloze前缀也改变了末位置。小的源状态差异可能来自不同序列长度数值内核，须单独核对，不能叫逆因果语言作用。共同读出余弦只是辅助描述。',
        '按原冻结单位12..15做2048独立实体扩大，验证规则而非重选阈值；随后对四模式固定真实参数做前向检验和完整交付。')

if __name__=='__main__':main()
