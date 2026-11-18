"""Replay Phase 2970's exact best 1011 and failing 1111 cases on the repaired evaluator."""

import argparse
from itertools import batched
from pathlib import Path
import shutil
import time

import numpy as np

import paradigm_path_run as X
import paradigm_path_report as Y
from deploy.storage_policy import check_write


R,C,OUT=X.R,X.C,X.OUT
REPLAY='legacy-plan.json'


def source_identity():
    return X.digest(Path(__file__).resolve())


def freeze(tokenizer):
    old=X.A.PARENT
    with_names=[f'confirm-{i:02}.npz' for i in range(8)]
    R.OUT=X.V.OUT=old
    try:
        data=X.V.data_load(with_names)
    finally:
        R.OUT=X.V.OUT=OUT
    all_rows=list(X.F.M.expanded(tokenizer,'confirm'))
    assert list(data['ids'])==[row['id'] for row in all_rows]
    groups=list(batched(all_rows,16))
    lookup={}
    selected=set()
    for target in X.M.TARGET:
        scores=data['fusion'][:,X.F.COMBINED.index(target),1:3,0]
        gain=scores[:,0]-scores[:,1]
        order=np.argsort(-gain if target==11 else gain,kind='stable')[:8]
        lookup[f'{target:04b}']=[dict(family=groups[i][0]['family'],id=groups[i][target]['id'],
                                     old=scores[i].tolist()) for i in order]
        selected.update(groups[i][0]['family'] for i in order)
    order=[group[0]['family'] for group in groups if group[0]['family'] in selected]
    text=[row for group in groups if group[0]['family'] in selected for row in group]
    assert len(text)==16*len(order)
    R.write(REPLAY,dict(at=C.stamp(),method=source_identity(),main=X.digest(OUT/'plan.json'),
        prior={name:X.digest(old/name) for name in ['manifest.json','plan.json',*with_names]},
        cases=lookup,order=order,materials=X.M.digest(text),views=len(text),
        purpose='Selection on Phase 2970 archived effects: reproduction of extremes, no independent confirmation.'))


def run():
    import torch

    plan=R.read('plan.json')
    study=R.read(REPLAY)
    assert study['main']==X.digest(OUT/'plan.json') and study['method']==source_identity()
    assert study['prior']=={name:X.digest(X.A.PARENT/name) for name in study['prior']}
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        start=time.monotonic()
        C.N.progress('old-fusion-replay-load',0,2,start)
        assert plan['snapshot']==C.Prior.identity()
        model=C.N.Engine()
        C.N.progress('old-fusion-replay-load',2,2,start)
        try:
            families=set(study['order'])
            rows=[row for row in X.F.M.expanded(model.tok,'confirm') if row['family'] in families]
            assert X.M.digest(rows)==study['materials'] and len(rows)==study['views']
            setup=X.anchors(model,plan)
            files=X.measure(model,rows,'legacy',plan,setup)
            data=X.load('legacy')
            new={}
            for j,(cell,cases) in enumerate(study['cases'].items()):
                measured=[]
                for case in cases:
                    i=study['order'].index(case['family'])
                    assert data['ids'][i*16+int(cell,2)]==case['id']
                    scores=data['fusion'][i,j,1:3,0]
                    words=model.tok.batch_decode([[int(k)] for k in data['fusion_top'][i,j,:4]])
                    measured.append(dict(case,new=scores.tolist(),tokens=words,
                                         max_abs_diff=float(np.max(np.abs(np.array(case['old'])-scores)))))
                new[cell]=measured
            R.write('legacy-replay.json',dict(at=C.stamp(),plan=X.digest(OUT/REPLAY),
                files={name:X.digest(OUT/name) for name in files},cases=new))
        finally:
            model.close()


def supplement(replay,analysis):
    lines=['','### 旧极端样本原样复测','',
        '- 从 Phase 2970 已归档确认材料选 1011 增益最大的 8 个家族、1111 恶化最大的 8 个家族，按旧分数排序并在复跑前冻结。每个入选家族重新执行全部 16 格。这里检验原失败能否复现，独立推广按上述新家族结果判断。',
        '', '| 旧样本 | 旧加法 → 二阶 KL | 本轮加法 → 二阶 KL | 真实 / 加法 / 二阶首 token |',
        '| --- | --- | --- | --- |']
    for cases in replay['cases'].values():
        for case in cases:
            old1,old2=case['old'];new1,new2=case['new']
            words=' / '.join(repr(case['tokens'][i]) for i in (0,2,3))
            lines.append(f"| {case['id']} | {old1:.6g} → {old2:.6g} | {new1:.6g} → {new2:.6g} | {words} |")
    lines.extend(['','### 主张与证据边界',''])
    if analysis['fusion']:
        for cell,m in analysis['fusion'].items():
            n=m['natural']
            labels=['语言×风格','语言×逻辑','语言×语法','风格×逻辑','风格×语法','逻辑×语法']
            effects=[v['mean'] for v in n['omit_pair']]
            lead=int(np.argmax(effects))
            lines.append(f"- {cell} 的自然档案中，去掉 {labels[lead]} 项的平均额外 KL 最大（{effects[lead]:+.4f}）。"
                         "这是对组合预测式的替换审计。该项的意义随上下文和评分位置变化。")
    if analysis['causal']:
        pos=[f'H{h} {key}' for h,group in analysis['causal']['heads'].items()
             for key,conditions in group.items() if conditions['conflict']['competition']]
        rescues=[f'H{h} {key} {s}' for h,group in analysis['causal']['heads'].items()
                 for key,conditions in group.items() for s in ('aligned','conflict')
                 if conditions[s]['direction_specific_rescue']]
        lines.append(f"- 原生格式竞争过门范围为 {'、'.join(pos) if pos else '零个头×组合条件'}。"
                     f"等幅选中恢复优势过门范围为 {'、'.join(rescues) if rescues else '零个头×组合×档案条件'}。"
                     "注意力取值、同层 G/U 响应与下一 token 效应是可分别观测的边。"
                     "只在控制检验通过的范围内提高机制证据等级。")
    lines.append('- 新家族更换了名词和句式，档案例句和输出导语沿用旧设计。档案互换同时改变措辞和标点，证据外推需要多套档案及新自然生成任务。格式 margin 覆盖固定论文开头候选，绝对准确率和全词表首 token 另有账本。')
    return '\n'.join(lines)+'\n'


def close():
    Y.report(False)
    plan=R.read('plan.json')
    replay=R.read('legacy-replay.json')
    assert replay['plan']==X.digest(OUT/REPLAY)
    assert all(X.digest(OUT/k)==v for k,v in replay['files'].items())
    target=OUT/'report.md'
    content=target.read_text(encoding='utf-8')
    body=content.split('\n\n',1)[1]+supplement(replay,R.read('analysis.json'))
    check_write(target,len(body.encode())+4096,operation='汇总原样复测').write_text(
        '# '+Y.TITLE+'\n\n'+body,encoding='utf-8')
    assert not (OUT/'memo-receipt.json').exists() and Y.TITLE not in Y.MEMO.read_text(encoding='utf-8-sig')
    memo=Y.append(check_write(Y.MEMO,len(body.encode())+4096,operation='追加 GLM5 研究记录'),Y.TITLE,body)
    cleanup=[]
    folder=X.A.TEMP
    if folder.is_dir():
        assert folder.resolve()==folder and folder.parent==(R.ROOT/'tests/glm5_temp').resolve()
        owned=[folder,*folder.rglob('*')]
        assert all(not p.is_symlink() and not p.is_junction() and p.resolve().is_relative_to(folder) for p in owned)
        size=sum(p.stat().st_size for p in owned if p.is_file())
        shutil.rmtree(folder)
        cleanup.append(dict(path=str(folder),freed_bytes=size))
    R.write('memo-receipt.json',dict(memo=memo,cleanup=cleanup))
    files={k.name:dict(sha256=X.digest(k),bytes=k.stat().st_size) for k in sorted(OUT.iterdir())
           if k.is_file() and k.name!='manifest.json'}
    script=Path(__file__).resolve().relative_to(R.ROOT).as_posix()
    R.write('manifest.json',dict(run=plan['run'],files=files,sources=plan['code']|{script:source_identity()}))
    print(f"Appended GLM5 Phase {memo['phase']}",flush=True)


if __name__=='__main__':
    from transformers import AutoTokenizer

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('freeze','run','close'))
    action=parser.parse_args().action
    X.A.setup()
    if action=='freeze':
        assert not (OUT/REPLAY).exists()
        freeze(AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True))
    elif action=='run':
        run()
    else:
        close()
