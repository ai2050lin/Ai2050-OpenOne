"""Independent fields, fixed coordinate envelope confirmation, no retuned thresholds."""
import numpy as np
from threadpoolctl import threadpool_limits
from phase2620_native_coordinate_contract import *
from phase2650_output_function_adjoints import capture_set,CONFIRM
from phase2651_output_function_maps import analyze,OUT as INITIAL_MAPS,MODES

def compare():
    counts={};groups=[]
    for p in sorted((INITIAL_MAPS/'maps').glob('*.npz')):
        if p.name=='coordinate_envelope_coverage.npz':continue
        mode=next(m for m in MODES if p.stem.endswith('_'+m));suffix=p.stem.removesuffix('_'+mode)
        with np.load(p) as a,np.load(CONFIRM/'maps'/p.name) as b:
            for metric in ('h','mlp','bf_h','bf_mlp'):
                at=a['target__'+metric+'__rms'];bt=b['target__'+metric+'__rms']
                init=(at>0)&(at>a['order__'+metric+'__rms'])&(at>a['form__'+metric+'__rms'])
                conf=(bt>0)&(bt>b['order__'+metric+'__rms'])&(bt>b['form__'+metric+'__rms'])
                am=a['target__'+metric+'__mean'];bm=b['target__'+metric+'__mean'];signed=(am!=0)&(bm!=0)&(np.sign(am)==np.sign(bm))
                for label,mask in [('initial',init),('confirmation',conf),('both',init&conf),('both_signed',init&conf&signed)]:
                    key=mode+'__'+metric+'__'+label
                    if key not in counts:counts[key]=np.zeros_like(mask,dtype='int16')
                    counts[key]+=mask
                groups.append({'group':suffix,'mode':mode,'metric':metric,'candidate_n_by_layer_position':init.sum(-1).tolist(),
                    'confirmed_n_by_layer_position':(init&conf).sum(-1).tolist(),'confirmed_signed_n_by_layer_position':(init&conf&signed).sum(-1).tolist()})
    summary={}
    for mode in MODES:
        summary[mode]={}
        for metric,l in [('h',36),('mlp',35),('bf_h',36),('bf_mlp',35)]:
            boundary=lambda v:v[l,2] if metric.endswith('h') else v[l]
            a=boundary(counts[mode+'__'+metric+'__initial'])==16;b=boundary(counts[mode+'__'+metric+'__both'])==16;s=boundary(counts[mode+'__'+metric+'__both_signed'])==16
            summary[mode][metric]={'coordinates':len(a),'initial_all16_candidates':int(a.sum()),'confirmed_all16_candidates':int(b.sum()),'confirmed_all16_signed':int(s.sum()),
                'amplitude_retention':float(b.sum()/a.sum()) if a.any() else None,'signed_retention':float(s.sum()/a.sum()) if a.any() else None}
    np.savez(CONFIRM/'maps/initial_confirmation_envelopes.npz',**counts);save(CONFIRM/'analysis/envelope_group_confirmation.json',groups)
    report={'summary':summary,'checks':{'all256_mode_group_metrics':len(groups)==256,'all4_output_modes':len(summary)==4},
        'boundary':'Same strictly frozen coordinate rules. Signed is within eachfamily/language across heldout entities, not a universal sign. Output functions share matched bodies but differ queries/prefixes; no unique causal attribution to head rows. Somelexical catalogs also shift with units.'}
    save(CONFIRM/'analysis/envelope_confirmation.json',report);return report

def main():
    cap=capture_set('confirmation',CONFIRM)
    with threadpool_limits(limits=4):maps=analyze('confirmation',CONFIRM);rep=compare()
    checks={**cap['checks'],**maps['checks'],**rep['checks']};assert all(checks.values())
    finish(2652,'独立2048实体条件扩大与冻结全坐标幅度及方向规则复核',CONFIRM,{'provenance':str(Path(__file__)),'summary':{'capture':cap['summary'],'maps':maps['summary'],'coordinate_confirmation':rep['summary']},'checks':checks},
        '完全沿初始合同在单位12..15采集并比较；每个物理坐标都评估目标幅度是否超过顺序/句式幅度，并检查逐组有符号目标均值是否跨实体集合保持，不改变阈值或筛掉失败功能。',
        r'C_{l,j}^{m}=\sum_{g=1}^{16}I^{initial,m}_{g,l,j}I^{heldout,m}_{g,l,j};\quad S_{l,j}^{m}=\sum_gI^{initial,m}_{g,l,j}I^{heldout,m}_{g,l,j}\mathbf1[\operatorname{sgn}\mu^{initial,m}_{g,l,j}=\operatorname{sgn}\mu^{heldout,m}_{g,l,j}\ne0].',
        '2048新实体条件与初始2048逐族/语言/功能匹配，四模式×16组×全部原坐标。BF16和FP32两套原场同时确认；truth_a/truth_b的语义目标与真值取向保留分账。',
        '比较固定是/否输出词与变化人名输出的方向迁移边界，确认对象是前期冻结的坐标规则，不是看完新结果后挑成功坐标。复用必须以具体函数条件和坐标为索引，而非平均余弦命名齿轮。',
        '各集合只有4实体对/语言，模板大量复用；英语词义与水果材料也随index切换，并非只改一个名字。不同输出功能的候选分母可能不同，保留率不是行为准确率或语义机制比例。',
        '继续四输出模式固定真实单参数前向检验，随后展示全坐标与逐参数结果并按清单清理，不因方向规则低保留率停止观察路线。')

if __name__=='__main__':main()
