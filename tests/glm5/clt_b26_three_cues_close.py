"""Audit crossed material, native Q/K/O scalars and any admitted perturbation."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import clt_b26_three_cues as B
from deploy.storage_policy import check_write,checked_dir
from source_binding_design import read,sha


def check_material(rows):
    for start in range(0,len(rows),12):
        family=rows[start:start+12]
        assert len(family)==12 and [r['cell'] for r in family]==list(range(12))
        assert len({r['family'] for r in family})==1
        assert all(r['names']==family[0]['names'] and r['codes']==family[0]['codes'] for r in family)
        for r, (ls,ps,rule) in zip(family,B.CELLS,strict=True):
            assert (ls,ps,rule)==(r['label_swap'],r['position_swap'],r['rule'])
            before=family[0];printed=before['printed'].copy(); order=before['physical_order'].copy()
            if ls:printed[1],printed[2]=printed[2],printed[1]
            if ps:order[0],order[-1]=order[-1],order[0]
            assert r['printed']==printed and r['physical_order']==order
            targets=[0,printed.index(before['printed'][1]),order[0 if r['first'] else -1]]
            assert r['cue_targets']==targets and len(set(targets))==3 and r['target_record']==targets[rule] and r['gold_id']==r['choice_ids'][targets[rule]]
            lines=[(code,name,label) for label,name,code in re.findall(r'^ROW ([A-E]) \| NAME (\w+) \| CODE (\w+)$',r['prompt'],re.M)]
            assert lines==[(r['codes'][j],r['names'][j],r['printed'][j]) for j in order]
        for a,b,c in zip(family[::3],family[1::3],family[2::3]):
            cut=a['endpoint']+1
            assert a['endpoint']==b['endpoint']==c['endpoint'] and len(a['tokens'])==len(b['tokens'])==len(c['tokens'])
            assert a['tokens'][:cut]==b['tokens'][:cut]==c['tokens'][:cut]


def score_details(rows,scores):
    by_rule={}
    for r,s in zip(rows,scores,strict=True):
        assert r['id']==s['id']; j=r['target_record']; v=s['scores']
        assert s['top1_correct']==(s['top1']==r['gold_id'])
        assert s['pred']==int(np.argmax(v)) and s['gold_margin']==v[j]-max(v[k] for k in range(5) if k!=j)
        z=by_rule.setdefault(B.RULES[r['rule']],dict(inputs=0,forced=0,top1=0,choice_counts=[0]*4))
        z['inputs']+=1; z['forced']+=s['gold_margin']>0;z['top1']+=s['top1_correct']
        z['choice_counts'][r['cue_targets'].index(s['pred']) if s['pred'] in r['cue_targets'] else 3]+=1
    return by_rule


def audit_natural(pre,split):
    record,data=B.read_collection(split); rows=B.materials(split,pre); check_material(rows)
    scores=[]
    for fn,digest in sorted(record['shards'].items()):
        part=read(B.OUT/fn.replace('.npz','.json'))
        assert part['sha']==digest and part['max_sdpa']<=B.CAP and part['prefix_equal']*3==len(part['scores'])
        assert part['prereg_sha']==sha(B.OUT/'prereg.json')
        if split=='confirm':assert part['candidate_sha']==sha(B.OUT/'candidate.json')
        scores.extend(part['scores'])
    checked=B.behavior(rows,scores)
    assert all(record[k]==v for k,v in checked.items()) and record['prefix_equal']*3==len(rows)
    checked.update(rules=score_details(rows,scores),elapsed=record['elapsed'],prefix=record['prefix_equal'],sdpa=record['max_sdpa'],torch=record['torch'],transformers=record['transformers'])
    return checked,rows,data


def native_and_swaps(rows,data,result):
    from source_binding_runtime import Runtime
    rt=Runtime('qwen14');prefix=f'model.layers.{B.BLOCK}.self_attn.'
    with np.load(B.OUT/'predictor.npz') as x:
        counts={}; picks={};weights={}
        for key in ('q','k','o'):
            p=rt.weight(prefix+key+'_proj.weight','cpu'); p=(p[:,B.QS] if key=='o' else p[B.QS if key=='q' else B.KS]).contiguous()
            assert np.array_equal(p.view(rt.t.uint16).numpy(),x['weight_'+key]);counts[key]=p.numel()
            if key!='o':
                coord=int(x[f'{key}_selected_512'][0]); column=p[:,coord].float().numpy(); j=int(np.abs(column).argmax()); picks[key]=x[f'{key}_selected_512'][:6].tolist()
                weights[key]=dict(row=(B.QS.start if key=='q' else B.KS.start)+j,col=coord,weight=float(column[j]))
            else:O=p.float().numpy().astype(np.float64)
        for d in data:
            for key in ('q','k','joint'):
                target=d[f'write_mse_{key}_selected_512']; actual=d['head_delta_'+key].astype(np.float64)@O.T
                assert np.allclose((actual**2).mean(1),target,rtol=1e-5,atol=1e-8)
    natural=np.concatenate([d['mass_actual'] for d in data]);d=np.log(np.maximum(natural,1e-20)).reshape(-1,2,2,3,5)
    n=len(d);family_pos=np.array([3 if r['first'] else 4 for r in rows[::12]])
    ri=np.arange(n); l=d[...,2]-d[...,1]; p=d[ri,...,7-family_pos]-d[ri,...,family_pos]
    # The sign follows movement toward the newly correct immutable record.
    dl=(l[:,1]-l[:,0]).mean(1); dp=(p[:,:,1]-p[:,:,0]).mean(1)
    bootstrap=np.random.default_rng(B.SEED+111).integers(0,n,(10000,n))
    def ci(y):return [float(y.mean()),*np.quantile(y[bootstrap].mean(1),[.0125,.9875]).tolist()]
    byrule={B.RULES[i]:dict(label_swap_log_odds=ci(dl[:,i]),position_swap_log_odds=ci(dp[:,i])) for i in range(3)}
    switches={
        'label_requested_minus_other_mean':ci(dl[:,1]-.5*(dl[:,0]+dl[:,2])),
        'physical_requested_minus_other_mean':ci(dp[:,2]-.5*(dp[:,0]+dp[:,1])),
    }
    for j in range(3):
        sub=rows[j::3]; z=natural[j::3]; true=np.array([r['target_record'] for r in sub]); goal=z[np.arange(len(z)),true]
        assert np.isclose((z.argmax(1)==true).mean(),result['diagnostic'][j]['all_five_gold'])
        assert np.isfinite(goal).all()
    bybranch={}
    for branch in ('q','k','joint'):
        a=np.concatenate([x[f'mass_{branch}_selected_512'] for x in data]); acc=(a.argmax(1)==natural.argmax(1)).mean()
        assert acc==result['prediction'][f'{branch}_selected_512']['five_way_agreement']
        bybranch[branch]={B.RULES[j]:float((a[j::3].argmax(1)==natural[j::3].argmax(1)).mean()) for j in range(3)}
    return dict(scalars=sum(counts.values()),counts=counts,coordinates=picks,scalar_examples=weights,swaps_by_rule=byrule,
                selective_swap_contrasts=switches,branch_512_agreement_by_rule=bybranch,
                scope='Frozen rows and real stored mass; CPU O arithmetic check, no independent model forward')


def check_causal(pre,names):
    path=B.OUT/'intervention-causal.json'
    if not path.exists():return None
    meta=read(path); lock=read(B.OUT/'causal-freeze.json');assert meta['freeze_sha']==sha(B.OUT/'causal-freeze.json')
    assert lock['evaluation_sha']==sha(B.OUT/'evaluation.json')
    rows=B.materials('intervention',pre);check_material(rows)
    assert not {n for r in rows for n in r['names']}&names
    reps=[r for r in rows if r['cell']==(r['family']//4)%12]; n=len(reps)
    assert n==128 and meta['n']==n
    p=B.OUT/'intervention-causal.npz';assert sha(p)==meta['traces_sha']
    with np.load(p) as d:
        assert np.array_equal(d['ids'],[r['id'] for r in reps])
        cue=np.array([r['cue_targets'] for r in reps]); rule=np.array([r['rule'] for r in reps]); l=np.log(np.maximum(np.take_along_axis(d['mass'],cue[:,None,:],axis=2),1e-20));pref=1.5*(l-l.mean(2,keepdims=True))[np.arange(n),:,rule]
        b=np.random.default_rng(B.SEED+96).integers(0,n,(10000,n))
        for key,r in meta['branches'].items():
            target,random,sham,restore=[meta['arms'].index(f'{key}_{x}') for x in ('target','random','sham','restore')]
            diff=pref[:,random]-pref[:,target]
            assert np.array_equal(r['primary'],[float(diff.mean()),*np.quantile(diff[b].mean(1),[.0125,.9875]).tolist()])
            k=d['dose'][:,random]/np.maximum(d['dose'][:,target],1e-12)
            assert r['dose_match']==((k>=.9)&(k<=1.1)).mean()
            assert r['head_exact']==all(np.array_equal(d['head'][:,0].view(np.uint32),d['head'][:,i].view(np.uint32)) for i in (sham,restore))
    return meta


def figure(result,facts,natural):
    folder=checked_dir(B.TMP/'mpl',4<<20);os.environ['MPLCONFIGDIR']=str(folder)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if result:
        fig,panels=plt.subplots(1,3,figsize=(13.5,4.1),layout='constrained'); colors=['#306889','#9e486e','#51876f','#c0c5c7']
        x=np.array([r['native_landing_by_cue']+[r['other_records']] for r in result['diagnostic']]); bottom=np.zeros(3)
        for i,label in enumerate((*B.RULES,'other')):
            panels[0].bar(B.RULES,x[:,i],bottom=bottom,label=label,color=colors[i]);bottom+=x[:,i]
        panels[0].set(title='Natural H36 source',ylabel='Fraction of max mass');panels[0].legend(fontsize=8)
        for j,rule in enumerate(B.RULES):
            entry=facts['swaps_by_rule'][rule]; key=('label_swap_log_odds','position_swap_log_odds')
            means=np.array([entry[k][0] for k in key]); lower=np.array([entry[k][1] for k in key]); upper=np.array([entry[k][2] for k in key]); pos=np.arange(2)+.24*(j-1)
            panels[1].bar(pos,means,width=.21,yerr=[means-lower,upper-means],label=rule,color=colors[j])
        panels[1].set(xticks=range(2),xticklabels=['Printed label','Physical slot'],ylabel='Change in log odds',title='Independent swap response');panels[1].legend(fontsize=8)
        for j,branch in enumerate(('q','k','joint')):
            v=[result['prediction'][f'{branch}_selected_{k}']['five_way_agreement'] for k in B.BUDGETS]
            panels[2].plot(B.BUDGETS,v,'o-',label=branch.upper(),color=colors[j]);panels[2].axhline(.9,color='#777777',lw=.6,ls='--')
        panels[2].set(xticks=B.BUDGETS,ylim=(0,1),xlabel='Input coordinate budget',title='Q/K routing agreement');panels[2].legend(fontsize=8)
    else:
        fig,ax=plt.subplots(figsize=(8,3.7),layout='constrained');x=[natural['rules'][k] for k in B.RULES]
        ax.bar(B.RULES,[r['top1']/r['inputs'] for r in x],color='#306889');ax.axhline(.90,color='#985328',ls='--');ax.set(ylim=(0,1),title='Frozen behavioral pilot',ylabel='Full-vocabulary first-token accuracy')
    path=check_write(B.OUT/'three-cue-results.png',4<<20);assert not path.exists();fig.savefig(path,dpi=155);plt.close(fig)


def cleanup():
    folders=[]; size=0
    root=(B.ROOT/'tests/glm5_temp').resolve()
    for p in sorted(root.glob('clt_b26_three_cues_v*')):
        if p.resolve().parent!=root or p.is_symlink():continue
        for f in p.glob('mpl/fontlist-*.json'):
            if not f.is_symlink() and f.resolve().is_relative_to(p.resolve()):size+=f.stat().st_size;f.unlink()
        for child in sorted((x for x in p.rglob('*') if x.is_dir()),key=lambda x:len(x.parts),reverse=True):
            if child.resolve().is_relative_to(p.resolve()) and not child.is_symlink() and not any(child.iterdir()):child.rmdir()
        if not any(p.iterdir()):p.rmdir(); folders.append(p.name)
    return dict(released_bytes=size,removed_empty_directories=folders)


def close():
    pre=B.common.checked();names=B.previous_names(); natural={};result=None;facts={};causal=None
    for split in ('pilot','discovery','confirm'):
        if not (B.OUT/f'{split}-collection.json').exists():break
        natural[split],rows,data=audit_natural(pre,split)
        new={n for r in rows for n in r['names']};assert len(new)==5*B.SIZES[split] and not new&names;names.update(new)
    assert natural
    if (B.OUT/'evaluation.json').exists():
        result=read(B.OUT/'evaluation.json');assert result['collection_sha']==sha(B.OUT/'confirm-collection.json') and result['candidate_sha']==sha(B.OUT/'candidate.json')
        facts=native_and_swaps(rows,data,result);causal=check_causal(pre,names)
    else:assert not list(natural.values())[-1]['passed']
    cal=read(B.OUT/'calibration-causal.json') if (B.OUT/'calibration-causal.json').exists() else None
    if result and result['admitted']:
        assert cal is not None
        if any(v['technical_pass'] for v in cal['branches'].values()):assert causal is not None
    old_runs={}
    for folder in sorted(B.OUT.parent.glob('clt_b26_three_cues_v*')):
        if folder==B.OUT or not (folder/'pilot-collection.json').exists():continue
        m=read(folder/'prereg.json'); assert sha(folder/'pilot-materials.json')==m['materials']['pilot']['sha']
        r=read(folder/'pilot-materials.json'); s=[]
        for f in sorted(folder.glob('pilot-??.json')):
            part=read(f);assert part['sha']==sha(f.with_suffix('.npz'));s.extend(part['scores'])
        old_runs[folder.name]=score_details(r,s)
    figure(result,facts,natural['pilot']); cleaned=cleanup()
    ts=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M');memo=B.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';ix=B.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    old,oldix=memo.read_bytes(),ix.read_bytes(); phase=max(map(int,re.findall(rb'^## Phase (\d+):',old,re.M)))+1; index=max(map(int,re.findall(rb'^## Phase (\d+):',oldix,re.M)))+1
    if result:
        summary=f"确认自然来源诊断 {result['diagnostic']}。按规则的固定偏好基线均为1/3。32坐标规则读出准确率{result['probe_accuracy']:.2%}。独立交换的同记录对数质量差与原参数审计 {facts}。"
        specs=[]
        for key in ('q','k','joint'):
            for n in B.BUDGETS:
                v=result['prediction'][f'{key}_selected_{n}'];specs.append(f"| {key.upper()} | {n} | {v['five_way_agreement']:.2%} | {v['cue_agreement']:.2%} | {v['margin_mse']:.5g} | {v['write_mse']:.5g} |")
        summary+='\n\n| 分支 | 坐标 | 五记录吻合 | 三线索内吻合 | 三线索margin MSE | O写入MSE |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'+'\n'.join(specs)
        summary+=f"\n\n512坐标准入 {result['gates']}。有条件原生路径计算接入实测另一支Q/K及V，未重构上游全部层或输出模型。"
    else:summary='最终已运行的自然集合未过行为门，停止该协议的内部选源推断和留出确认。原生SDPA采样仍只作技术校准。'
    if causal:summary+=f"\n\n独立128家族的等剂量结果 {causal['branches']}。主要量为匹配随机减目标的正确来源对数优势，记录sham与同输入恢复。单个标量作用未检验。"
    elif cal:summary+=f"\n\n干预校准未过准入的结果 {cal['branches']}，封存干预家族未运行。"
    else:summary+='\n\n没有任何分支进入正式原模型干预。'
    logs=[json.loads(line) for file in B.OUT.glob('*progress.jsonl') for line in file.read_text().splitlines()]
    evidence={k:{'totals':v['totals'],'rules':v['rules'],'min_cells':min(min(x['forced'],x['top1'])/x['n'] for x in v['cells']),'prefix':v['prefix'],'sdpa':v['sdpa'],'passed':v['passed'],'elapsed':v['elapsed']} for k,v in natural.items()}
    wording=pre.get('protocol_change','五条记录及等长末尾问句直接询问第一行或最后一行。')
    note=f'''

## Phase {phase}: B26/H36 姓名、印刷 ROW 与显示位置的独立交换 [{ts}]

**目的与材料。** 用五个不可变记录 ID 保持姓名与代码绑定。ID0指向固定姓名，ID1/2之间交换查询标签，ID3/4整行在标定首尾间交换。每个2×2条件下三线索指向不同记录，再问三种等长度问题，共12格。两种交换相互独立。模型快照 `{pre['provenance']['model_revision']}`。首轮校准的姓名、标签各128/128强制选择，位置仅41/128。失败样本完整保留。先根据失败修改明确位置，再生成不重叠家族和重新冻结。最终版本 {B.OUT.name}。材料修订 {wording}。旧版复算 {old_runs}。

**方法与试验。** 位置只查物理首／末，标签随机配平。对Q/K使用发现集 `Var(z_j)||W[:,j]||²` 排序，固定原生坐标预算。重建 `Wμ+Σ_j W[:,j](z_j−μ_j)` 后按原RMSNorm与RoPE进入softmax。以三个独立线索的中心化log质量MSE比较同预算随机与均值。物理和标签交换效应在不可变ID下按家族对比，另算该规则相对另外两种规则的选择性。行为、前缀和算术复核 {evidence}。书面规则重复于问题末尾，因而读出可能利用末尾措辞。原生重算只核对头输出与实测SDPA。二值设计参考功效{pre['power']['achieved_power']}，连续量按实际配对区间。

{summary}

**边界与下一阶段。** 物理位置的正式范围由冻结材料限定。首行规则为邻接BEGIN，后续需把文字边界与物理邻接正交，并复测末行及中间行。姓名条件在三个候选中的身份索引固定，实际内行随家族变化。原生路由变化不能自动定位上游第一个作决定的层，也不证明行为必需。若Q/K静态权重方差排序未达门，下一大阶段在新材料用真实竞争记录的QK差优化所选原生坐标，并保持同预算随机比较。跨表、自然文本、跨模型和独立复算节点尚未覆盖。没有新RDC统一公式或AGI结论。

**资源与复算。** 脚本 `clt_b26_three_cues*.py`，实际阶段、输入、进度及复核清单在 `tests/glm5/result/{B.OUT.name}`。采样峰值CUDA {max(x['cuda_peak'] for x in logs)} 字节、共享显存 {max(x.get('shared_gpu_bytes') or 0 for x in logs)} 字节，D盘余量 {shutil.disk_usage(B.OUT).free/2**30:.3f} GiB，保持20GiB储备。清理 {cleaned}。未发布服务器。此记录以尾部追加方式保存。
'''
    indexnote=f'\n\n## Phase {index}: B26/H36 姓名、行标签、物理位置三线索 [{ts}]\n\n见GLM5 Phase {phase}。冻结家族上的行为、路由、Q/K坐标预测与已执行因果检验分别登记。\n'
    hashes={p.name:sha(p) for p in B.OUT.iterdir() if p.is_file()}
    B.common.commit('delivery.json',dict(phase=phase,timestamp=ts,files=hashes,natural=evidence,old_pilots=old_runs,native_audit=facts,causal_audit=causal is not None,cleanup=cleaned,provenance=pre['provenance'],auditor_sha=sha(Path(__file__))))
    for path,value,prior in [(memo,note,old),(ix,indexnote,oldix)]:
        data=value.encode('utf-8');check_write(path,len(data));assert path.read_bytes()==prior
        with path.open('ab') as stream:stream.write(data)
        assert path.read_bytes()==prior+data
    print(json.dumps(dict(phase=phase,index=index,manifest_items=len(hashes))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('version',nargs='?',default='v2');options=parser.parse_args()
    if options.version!='v1':
        import importlib
        importlib.import_module('clt_b26_three_cues_'+options.version)
    B.configure();close()
