"""Independently recompute conflict metrics, verify parameters, and append the campaign record."""
from datetime import datetime
import itertools
import json
import os
from pathlib import Path
import re
import shutil
from zoneinfo import ZoneInfo

import numpy as np
import clt_b26_conflict_v7 as driver  # configures the shared runner for this split
import clt_b26_conflict as B
from deploy.storage_policy import check_write
from source_binding_design import read, sha


def check_split(split, pre, names):
    from itertools import batched
    record,data=B.consume(split)
    rows=B.materials(split,pre)
    assert len(rows)==16*B.SIZES[split]
    new={n for r in rows for n in r['names']}; assert not new&names and len(new)==2*B.SIZES[split]; names.update(new)
    assert len({r['id'] for r in rows})==len(rows)
    for a,b in batched(rows,2):
        assert a['name_slot']==b['name_slot'] and a['row_slot']==b['row_slot']
        n=a['endpoint']+1;assert a['tokens'][:n]==b['tokens'][:n]
    scores=[]
    for name,digest in sorted(record['shards'].items()):
        p=B.OUT/name.replace('.npz','.json');x=read(p)
        assert x['sha']==digest and x['prereg_sha']==sha(B.OUT/'prereg.json') and x['max_sdpa']<=B.CAP
        assert x['prefix_equal']*2==len(x['scores'])
        if split=='confirm':assert x['seal_sha']==sha(B.OUT/'candidate.json')
        scores.extend(x['scores'])
    groups={}
    for row,score in zip(rows,scores,strict=True):
        i=row['target_record']; values=score['scores']
        assert row['id']==score['id'] and i==row[('name_slot','row_slot')[row['rule']]]
        assert row['conflict']==(row['name_slot']!=row['row_slot'])
        assert score['gold_margin']==values[i]-max(x for j,x in enumerate(values) if j!=i)
        assert score['top1_correct']==(score['top1']==row['gold_id'])
        category=f"{row['rule']}_{int(row['conflict'])}"
        stats=groups.setdefault(category,dict(inputs=0,strict=0,top1=0,name_choice=0,row_choice=0))
        stats['inputs']+=1; stats['strict']+=score['gold_margin']>0; stats['top1']+=score['top1_correct']
        stats['name_choice']+=score['pred_record']==row['name_slot']; stats['row_choice']+=score['pred_record']==row['row_slot']
    cells,passes=B.behavior(rows,scores); assert record['cells']==cells and record['passed']==passes
    return record,rows,data,groups


def check_native(rows,data,evidence):
    from source_binding_runtime import Runtime
    rt=Runtime('qwen14'); prefix=f'model.layers.{B.BLOCK}.'
    with np.load(B.OUT/'predictor.npz') as z:
        counts={}
        for k in 'qkv':
            native=rt.weight(prefix+f'self_attn.{k}_proj.weight','cpu')[B.QS if k=='q' else B.KS]
            bits=native.contiguous().view(rt.t.uint16).numpy()
            assert np.array_equal(bits,z['weight_'+k]); counts[k]=bits.size
        for k,original,region,is_col in [('o','self_attn.o_proj.weight',B.QS,True),('gate','mlp.gate_proj.weight',z['units'],False),
                                          ('up','mlp.up_proj.weight',z['units'],False),('down','mlp.down_proj.weight',z['units'],True)]:
            native=rt.weight(prefix+original,'cpu'); part=native[:,region] if is_col else native[region]
            bits=part.contiguous().view(rt.t.uint16).numpy()
            assert np.array_equal(bits,z['weight_'+k]); counts[k]=bits.size
        weights=(z['weight_o'].astype(np.uint32)<<16).view(np.float32).astype(np.float64)
        identity=[r['name_slot'] for r in rows]; line=[r['row_slot'] for r in rows]
        masses=np.concatenate([x['mass_actual'] for x in data]); ii=np.arange(len(rows)); flags=np.array([r['conflict'] for r in rows[::2]])
        d=np.log(np.maximum(masses[ii,identity],1e-20))-np.log(np.maximum(masses[ii,line],1e-20))
        pair=(d[::2]-d[1::2])[flags].reshape(-1,4).mean(-1)
        assert np.isclose(pair.mean(),evidence['crossover'][0],atol=1e-7)
        for k in ('q','k','v','joint'):
            label=f'{k}_selected_128'
            for item in data:
                residual=(item['head_'+label].astype(np.float64)-item['actual_head'].astype(np.float64))@weights.T
                assert np.allclose((residual**2).mean(-1),item['write_error_'+label],atol=1e-8,rtol=3e-5)
        picks={k:z[f'{k}_selected_128'][:6].tolist() for k in 'qkv'}
        examples=[]
        for k in 'qkv':
            c=picks[k][0]; values=(z['weight_'+k][:,c].astype(np.uint32)<<16).view(np.float32)
            j=int(np.argmax(np.abs(values))); row=(B.QS.start if k=='q' else B.KS.start)+j
            examples.append(dict(branch=k,row=row,column=c,weight=float(values[j])))
        return dict(scalars=sum(counts.values()),index_examples=picks,scalar_examples=examples,
                    units=z['units'][:8].tolist(),scope='Original scalar identity and independent O projection arithmetic')


def check_intervention(evaluation,pre,names):
    path=B.OUT/'intervention-intervention.json'
    if not path.exists():return None
    frozen=read(B.OUT/'intervention-freeze.json'); c=read(path)
    assert frozen['evaluation_sha']==sha(B.OUT/'evaluation.json') and c['freeze_sha']==sha(B.OUT/'intervention-freeze.json')
    assert frozen['branches']==evaluation['causal_admitted']
    rows=[r for r in B.materials('intervention',pre) if r['cell']==(2,3,6,7,10,11,14,15)[(r['family']//2)%8]]
    assert not {n for r in rows for n in r['names']}&names and len(rows)==128
    original=B.OUT/'intervention-intervention.npz'; assert sha(original)==c['traces_sha']
    with np.load(original) as z:
        assert np.array_equal(z['ids'],[r['id'] for r in rows])
        route=np.log(np.maximum(z['mass'][...,0],1e-20))-np.log(np.maximum(z['mass'][...,1],1e-20))
        sample=np.random.default_rng(B.SEED+70).integers(0,len(rows),(10000,len(rows)))
        for branch,result in c['results'].items():
            a,b,s,r=[c['arms'].index(f'{branch}_{arm}') for arm in ('target','random','sham','restore')]
            dose=z['dose'][:,b]/np.maximum(z['dose'][:,a],1e-12)
            assert bool(np.array_equal(z['dose'][:,a],z['dose'][:,r]) and np.min(z['dose'][:,a])>0)==result['target_restore_dose_exact']
            assert result['dose_match']==((dose>=.9)&(dose<=1.1)).mean()
            assert result['internal_exact']==all(np.array_equal(z[key][:,i].view(np.uint32),z[key][:,0].view(np.uint32)) for key in ('head','A') for i in (s,r))
            measure=route if branch!='v' else z['local_output_margin']
            diff=measure[:,b]-measure[:,a]
            assert np.array_equal(result['primary'],[float(diff.mean()),*np.quantile(diff[sample].mean(1),[.0125,.9875]).tolist()])
    return c


def figures(observed, data, rows, groups):
    from deploy.storage_policy import checked_dir
    cache=checked_dir(B.TMP/'report-mpl',8<<20)
    os.environ['MPLCONFIGDIR']=str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if observed:
        fig,ax=plt.subplots(1,3,figsize=(13.5,4.3),layout='constrained')
        natural=np.concatenate([x['mass_actual'] for x in data]); ii=np.arange(len(rows))
        n=[r['name_slot'] for r in rows]; p=[r['row_slot'] for r in rows]; use=np.array([r['conflict'] for r in rows[::2]])
        d=np.log(np.maximum(natural[ii,n],1e-20))-np.log(np.maximum(natural[ii,p],1e-20))
        ax[0].scatter(d[1::2][use],d[::2][use],s=7,alpha=.32,color='#24678c')
        for line in (ax[0].axhline(0),ax[0].axvline(0)):line.set(color='#999999',ls='--',lw=.8)
        ax[0].set(title='Same-body conflict pairs',xlabel='D with row question',ylabel='D with name question')
        branches=('q','k','v','joint'); ix=np.arange(4)
        for n,shift,color in ((32,-.17,'#a7b7c3'),(128,.17,'#24678c')):
            a=[observed['branches'][f'{k}_selected_{n}']['route'] for k in branches]
            ax[1].bar(ix+shift,a,width=.32,label=str(n),color=color)
        ax[1].set(xticks=ix,xticklabels=['Q','K','V','Joint'],ylim=(0,1),title='Route agreement with native H36');ax[1].legend(title='Coordinates')
        for n,shift,label,color in ((32,-.17,'32','#a7b7c3'),(128,.17,'128','#24678c')):
            v=[observed['branches'][f'{k}_selected_{n}']['write'] for k in branches]
            ax[2].bar(ix+shift,v,width=.32,label=label,color=color)
        ax[2].set(xticks=ix,xticklabels=['Q','K','V','Joint'],title='Native O write prediction',ylabel='MSE over 5120 coordinates')
        ax[2].legend(title='Coordinates')
    else:
        fig,ax=plt.subplots(figsize=(7.6,4),layout='constrained')
        g=groups['pilot']; keys=('0_0','1_0','0_1','1_1')
        ax.bar(range(4),[g[k]['top1']/g[k]['inputs'] for k in keys],color=['#9cabb6','#9cabb6','#24678c','#24678c'])
        ax.set(xticks=range(4),xticklabels=['Name agree','Row agree','Name conflict','Row conflict'],ylabel='Full-vocabulary first-token accuracy',title='Frozen pilot behavior',ylim=(0,1))
    p=check_write(B.OUT/'conflict-results.png',4<<20,operation='Conflict report figure'); assert not p.exists()
    fig.savefig(p,dpi=155);plt.close(fig)
    removed=0
    assert cache.resolve().is_relative_to(B.TMP.resolve())
    for p in cache.glob('fontlist-*.json'):
        if not p.is_symlink():removed+=p.stat().st_size;p.unlink()
    if not any(cache.iterdir()):cache.rmdir()
    return removed


def close():
    pre=B.checked()
    names={n for folder in driver.PRIOR
           for p in (B.OUT.parent/folder).glob('*-materials.json') for row in read(p) for n in row['names']}
    earlier_pilots={}
    for version in range(1,7):
        directory=B.OUT.parent/f'clt_b26_conflict_v{version}'
        old=read(directory/'prereg.json'); source=directory/'pilot-materials.json'
        assert sha(source)==old['materials']['pilot']['sha']
        old_rows=read(source); old_scores=[]; equal=0
        for p in sorted(directory.glob('pilot-??.json')):
            x=read(p); assert x['sha']==sha(p.with_suffix('.npz'))
            old_scores.extend(x['scores']); equal+=x['prefix_equal']
        cells,gates=B.behavior(old_rows,old_scores)
        earlier_pilots[f'v{version}']=dict(n=len(old_rows),forced=sum(s['gold_margin']>0 for s in old_scores),
                                        top1=sum(s['top1_correct'] for s in old_scores),prefix_equal=equal,
                                        cell_min=min(c[1]/c[2] for c in cells),behavior_gate=gates)
    groups={}; snapshots={}; facts={}; observed=None; causal=None
    for split in ('pilot','discovery','confirm'):
        if not (B.OUT/f'{split}-collection.json').exists():break
        record,rows,data,groups[split]=check_split(split,pre,names)
        snapshots[split]={k:record[k] for k in ('passed','cells','elapsed','prefix_equal','max_sdpa','torch','transformers')}
    if (B.OUT/'evaluation.json').exists():
        observed=read(B.OUT/'evaluation.json')
        assert observed['collection_sha']==sha(B.OUT/'confirm-collection.json') and observed['candidate_sha']==sha(B.OUT/'candidate.json')
        assert split=='confirm'
        facts=check_native(rows,data,observed)
        causal=check_intervention(observed,pre,names)
    else:assert 'pilot' in snapshots and not snapshots[split if split in snapshots else list(snapshots)[-1]]['passed']
    calibration_path=B.OUT/'calibration-intervention.json'
    calibration=read(calibration_path) if calibration_path.exists() else None
    if observed and observed['causal_admitted']:
        assert calibration is not None
        if any(v['technical_pass'] for v in calibration['results'].values()):assert causal is not None

    notes=[]
    if observed:
        notes.append(f"冲突时身份规则的正确记录最高质量比例 {observed['rule_diagnostics']['0']['gold_top']:.2%}，行规则 {observed['rule_diagnostics']['1']['gold_top']:.2%}。身份规则下D {observed['rule_diagnostics']['0']['D']}，行规则下D {observed['rule_diagnostics']['1']['D']}，规则差 {observed['crossover']}。普遍切换门 {observed['follows_rule']}。实体名提示与物理行线索在冲突格指向不同记录。规则线性读出准确率 {observed['probe_accuracy']:.2%}，只反映末token操作前状态的可读性。")
        masses=np.concatenate([x['mass_actual'] for x in data]); chosen=masses.argmax(1)
        case=next((i for i in range(0,len(rows),2) if rows[i]['conflict'] and all(chosen[j]==rows[j]['target_record'] for j in (i,i+1))),None)
        miss=next((i for i,r in enumerate(rows) if r['conflict'] and chosen[i]!=r['target_record']),None)
        if case is not None:
            notes.append(f"成对例 {rows[case]['id']}、{rows[case+1]['id']} 的正文和线索相同。身份、行位对应记录索引 {rows[case]['name_slot']}、{rows[case]['row_slot']}，两问各四记录质量为 {masses[case].tolist()}、{masses[case+1].tolist()}。")
        if miss is not None:
            notes.append(f"反例 {rows[miss]['id']} 的最高质量记录为 {chosen[miss]}，金记录为 {rows[miss]['target_record']}，质量 {masses[miss].tolist()}。该头在部分题上读向其它记录。")
        notes.append('| 分支 | 坐标预算 | 自然路由吻合 | D MSE | 原写入 MSE | MLP A MSE |\n| --- | --- | ---: | ---: | ---: | ---: |')
        for k in (*B.BRANCHES,'joint'):
            for method,n in [('selected',32),('selected',128),('random',128),('mean',0)]:
                a=observed['branches'][f'{k}_{method}_{n}']; notes[-1]+=f"\n| {k} | {method} {n} | {a['route']:.2%} | {a['D']:.5g} | {a['write']:.5g} | {a['A']:.5g} |"
        notes.append(f"分支门为 {json.dumps(observed['gates'],ensure_ascii=False)}。V单支的QK质量逐项与自然相等。原O和真实他头背景给出下游条件预测。冻结的原生坐标、单元和标量审计 {facts}。这些MLP单元由发现集的H36相关A变化选出，单元与标量仍需独立干预以认定自身因果贡献。")
    if causal:
        notes.append(f"干预在128个新家族的冲突代表题上完成。真实网络把单头原生重算上下文从O入口写入，后续模型原样运行，K/V其它GQA共享头保持自然结果。Q/K使用金记录对另一候选的对数质量优势，V使用正确代码对冲突代码的局部写入。匹配随机减目标的配对效应、复原和答案结果为 {json.dumps(causal['results'],ensure_ascii=False)}。V的局部写入口径使用原词表权重作投影，最终分差另报。")
    elif calibration:
        notes.append(f"原模型校准结果为 {json.dumps(calibration['results'],ensure_ascii=False)}。正式干预集保持封存。")
    else:notes.append('未进入原模型匹配干预，干预家族保持封存。')
    logs=[json.loads(line) for f in B.OUT.glob('*progress.jsonl') for line in f.read_text().splitlines()]
    removed=figures(observed,data,rows,groups)
    free=shutil.disk_usage(B.OUT).free/2**30
    for p in (B.TMP,*(B.ROOT/f'tests/glm5_temp/clt_b26_conflict_v{i}' for i in range(1,7))):
        if p.exists() and not any(p.iterdir()):
            assert p.resolve().is_relative_to((B.ROOT/'tests/glm5_temp').resolve());p.rmdir()
    ts=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    memo=B.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'; index=B.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md'
    old,ixa=memo.read_bytes(),index.read_bytes()
    phase=max(map(int,re.findall(rb'^## Phase (\d+):',old,re.M)))+1
    ip=max(map(int,re.findall(rb'^## Phase (\d+):',ixa,re.M)))+1
    joined=(f'前六次校准独立复算 {earlier_pilots}。第六版把旧线索移至表格前，并标出有效档案，将其放在问题前。'
            '第七版明确印出 ROW、NAME 和 CODE 字段，末尾要求匹配其一。逻辑档案索引、标题、身份所在行与有效线索仍配平。'
            '有效档案固定最后。物理行位与 ROW 字面标签绑定，本轮无法区分两者。\n\n'+'\n\n'.join(notes))
    body=f'''

## Phase {phase}: B26/H36 身份与物理行位冲突的条件选源 [{ts}]

**目标与材料。** 将两个线索写进同一正文，让它们在冲突格分别指向同档案的不同记录。最后切换有效规则。按档案、身份所在行、是否冲突与规则组成16格，家族配平标题与身份。模型为本地 Qwen3-14B 原 BF16 `{pre['provenance']['model_revision']}`。首版16家族在一致格128/128，冲突的身份与行规则分别42/64、25/64，未进发现。第二版显式要求忽略无效键，仍有明显冲突错误，且72对最终长度差一个 token，只56/128对规则前位置逐位相同，技术门失败。第三版改用同一草稿注释与等长直问，128/128前缀激活逐位相同，但只在242/256选四候选正确、232/256全词表首词正确，差异集中在冲突的数字行位题。第四版显式标TOP/BOTTOM行，要求单个小写代码词，并使用新冻结家族与相同门槛。末尾规则与重复有效线索一起变动，结论限于此问句和上下文。确认预算128家族，二值参考功效 {pre['power']['achieved_power']}，连续误差用预注册家族区间。各版材料和失败证据保存在 clt_b26_conflict_v1/v2/v3，各版排除此系列此前冻结姓名。

**原理与算式。** 定义 D=log(m_name/m_row)，m为指定记录跨度的头注意力概率总和。自然切换量为同输入线索下的 D_name−D_row。Q/K/V各自以发现集 `Var(z_j) * ||W[:,j]||²` 冻结列，使用 `Wμ + Σ_j W[:,j](z_j−μ_j)`，保留原 RMSNorm、RoPE 和有效前缀遮罩。单支预测给定另外两支原生投影。V/O独立贡献由 `W_O Σ_i α_i V_i` 得到。下游 `A=SiLU(W_g RMSNorm(h))*(W_u RMSNorm(h))`，h包含真实其它头O写入背景。预测、实测与可读出均分别保存。各集逐格行为与技术门为 {snapshots}。按规则和冲突重算的首词结果为 {groups}。

{joined}

**理论与局限。** 规则可读出、头实际选源、局部预测和原模型干预能回答不同问题。身份信息、物理行位和主动指令可能共同参与计算，头对记录的注意力排序也可能由上游先选之后接续。下一阶段应把有效线索是否重述与最终规则措辞做正交对照，再跨句式和位置复现，检验首次选源层位和同档竞争读出是否一致。只有跨操纵的可预测作用才能支持更一般的条件齿轮。单一模型、固定首词和当前模板不能外推开放知识融合。没有建立新的RDC统一公式或AGI结论。

**复算和资源。** 首三版使用对应脚本的 `freeze/pilot`。第四版 `clt_b26_conflict_v4.py` 按通过情况运行 `freeze/pilot/discovery/select/confirm/evaluate`。通过的分支才进入该脚本的 `freeze-intervention/calibration/intervention`。CPU收尾核查逐条行为、家族、前缀、材料及实际完成阶段的计算。实测记录CUDA峰值 {max(v['cuda_peak'] for v in logs)} 字节，共享峰值 {max(v.get('shared_gpu_bytes') or 0 for v in logs)} 字节，D盘余量 {free:.3f} GiB，保留20GiB。原结果保留，已核对并移除本轮空暂存目录，释放文件0字节。真实读数保存在 `tests/glm5/result/clt_b26_conflict_v4`。本轮本地归档，服务器未发布。
'''
    body=body.replace('首三版使用对应脚本', '首六版使用对应脚本').replace('第四版 `clt_b26_conflict_v4.py` 按通过情况', '第七版 `clt_b26_conflict_v7.py` 按通过情况').replace('真实读数保存在 `tests/glm5/result/clt_b26_conflict_v4`', '第七版读数保存在 `tests/glm5/result/clt_b26_conflict_v7`').replace('释放文件0字节',f'释放绘图缓存{removed}字节')
    ref=f'\n\n## Phase {ip}: B26/H36 身份和行位冲突留出检验 [{ts}]\n\n见GLM5 Phase {phase}。首版在冲突行为门失败，使用新家族和更明确规则复测。分支预测和通过准入的匹配干预按实际产物记录。\n'
    files={p.name:sha(p) for p in B.OUT.iterdir() if p.is_file()}
    B.commit('delivery.json',dict(provenance=pre['provenance'],phase=phase,timestamp=ts,auditor_sha=sha(Path(__file__)),files=files,parameters=facts,earlier_pilots=earlier_pilots,behavior=groups,splits=snapshots,causal_verified=causal is not None))
    for path,prior,addition in ((memo,old,body),(index,ixa,ref)):
        payload=addition.encode('utf-8');check_write(path,len(payload));assert path.read_bytes()==prior
        with path.open('ab') as stream:stream.write(payload)
        assert path.read_bytes()==prior+payload
    print(json.dumps(dict(phase=phase,index=ip,manifest_files=len(files),causal_verified=causal is not None)),flush=True)


if __name__=='__main__':
    close()
