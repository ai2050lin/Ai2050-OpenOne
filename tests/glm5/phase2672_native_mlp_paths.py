"""Every incoming/outgoing coordinate of five previously frozen native MLP units."""
from collections import defaultdict
import numpy as np
from phase2620_native_coordinate_contract import *
from phase2670_native_mlp_contract import OUT as CONTRACT,FIELD,LAYERS,SITES
from phase2671_native_mlp_field import unbits,moment_group

OUT=RESULT/'phase2672_native_mlp_paths'

def main():
    assert (FIELD/'analysis/final.json').exists() and not (OUT/'analysis/final.json').exists()
    cases=read(CONTRACT/'material/cases.json');weights=read(FIELD/'protocol/native_weights.json')
    with np.load(FIELD/'weights/native_candidate_vectors.npz') as z:wv={k:z[k].astype('float64') for k in z.files}
    OUT.joinpath('maps').mkdir(parents=True,exist_ok=True);OUT.joinpath('field').mkdir(parents=True,exist_ok=True);OUT.joinpath('analysis').mkdir(parents=True,exist_ok=True)
    save(OUT/'protocol/frozen.json',{'material_sha256':sha(CONTRACT/'material/cases.json'),'weights':weights,'candidate_sites':SITES,'all_input_and_output_coordinates':2560,
        'meaning':'g/u input term W[j,k]*x[k]; down output vector W[i,j]*a[j]. One down unit is NOT the wholeMLP. Linear BF16 forward vs float64 accounting separately reported; no donor or changed parameters.'})
    acc={};count=0;records=[];key=None
    with (OUT/'analysis/candidate_values.jsonl').open('w',encoding='utf-8') as stream:
        for ci,r in enumerate(cases):
            key=moment_group(r);pub={}
            with np.load(FIELD/f'field/case_{r["case_index"]:04d}.npz') as z:
                x=unbits(z['x']).astype('float64');gate=unbits(z['gate']);up=unbits(z['up']);a=unbits(z['a'])[:,-1];down=unbits(z['down'])
                for l,j in SITES:
                    li=LAYERS.index(l);native={'gate':float(gate[l,j]),'up':float(up[l,j]),'a':float(a[l,j])};entry={'case_index':r['case_index'],'family':r['family'],'language':r['language'],'unit':r['unit'],'content_instance':r['content_instance'],'layer':l,'neuron':j,'values':native,'linear_errors':{}}
                    for kind in ('gate','up','down'):
                        name=f'L{l}_J{j}_{kind}';value=wv[name]*(native['a'] if kind=='down' else x[li])
                        for stat,v in [('sum',value),('sumsq',value*value)]:
                            k=name+'__'+stat
                            if k not in acc:acc[k]=np.zeros_like(v)
                            acc[k]+=v
                        if kind!='down':entry['linear_errors'][kind]=float(value.sum()-native[kind])
                        else:entry['single_unit_vs_full_branch']={'unit_output_l1':float(np.abs(value).sum()),'full_branch_l1':float(np.abs(down[li]).sum())}
                        if r['published']:pub[name]=value
                    stream.write(json.dumps(entry,ensure_ascii=False)+'\n');records.append(entry)
            if pub:np.savez_compressed(OUT/f'field/case_{r["case_index"]:04d}.npz',**pub)
            count+=1
            if ci+1==len(cases) or moment_group(cases[ci+1])!=key:
                # Saved legacy suffixes name the accumulator, not the final statistic:
                # __sum contains the MEAN; __sumsq contains RMS. Raw per-example paths have no such suffix.
                np.savez_compressed(OUT/f'maps/{key}.npz',**{k:(v/count if k.endswith('__sum') else np.sqrt(v/count)) for k,v in acc.items()});acc={};count=0
            if (ci+1)%256==0:stream.flush();print('native allcoordinate paths',ci+1,'/',len(cases),flush=True)
    summary={}
    for l,j in SITES:
        rr=[r for r in records if (r['layer'],r['neuron'])==(l,j)]
        summary[f'L{l}J{j}']={kind:{'mean_abs_error':float(np.mean([abs(r['linear_errors'][kind]) for r in rr])),
            'maximum_abs_error':max(abs(r['linear_errors'][kind]) for r in rr),'relative_L1':sum(abs(r['linear_errors'][kind]) for r in rr)/max(sum(abs(r['values'][kind]) for r in rr),1e-30)} for kind in ('gate','up')}
    checks={'40960_native_unit_observations':len(records)==40960,'32_allcoordinate_groups':len(list((OUT/'maps').glob('*.npz')))==32,'16_exact_example_paths':len(list((OUT/'field').glob('*.npz')))==16,
        'native_weight_vectors_unchanged':sha(FIELD/'weights/native_candidate_vectors.npz')==weights['candidate_vector_sha256']}
    assert all(checks.values())
    finish(2672,'五个原生MLP单元全部输入/输出标量路径账本',OUT,{'provenance':str(Path(__file__)),'summary':summary,'checks':checks},
        '每个已冻结神经元逐项计算真实gate/up权重与归一化输入的乘积，并逐输出坐标计算down权重与实际乘积激活的贡献。全部2560坐标参与，不做TopK、差分搬运或坐标压缩。',
        r'c^g_{j,k}=W^g_{j,k}x_k,\quad c^u_{j,k}=W^u_{j,k}x_k,\quad c^d_{i,j}=W^d_{i,j}a_j;\quad e^g_j=\sum_kc^g_{j,k}-g^{BF16}_j.',
        'C0018192×5=40960单元观察；C00215条全输入/输出权重向量；C00332族/语言/实体集合全坐标均值与RMS图；C00416原样本具体逐参数路径；BF16门值和float64乘加残差分账。',
        '现在可以直接回答某个真实参数在当前token如何参与门值或输出坐标，不借助搬运其他样本。该单元通过哪些低幅值输入、哪些输出方向变化可逐坐标审查。',
        '中层只展开五个固定候选，不表示其他神经元无用；全MLP背景另存2671。单元down贡献是加法项而非独占因果功劳。线性BF16/实数乘加差异包含舍入，不宜要求逐位相等。',
        '继续新实体/新内容的全场条件确认及SiLU乘积、归一化、残差旁路分账；代数可计算性不等于已提取普适语言特征。')

if __name__=='__main__':main()
