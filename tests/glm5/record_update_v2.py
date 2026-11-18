"""Independent behavior repair for the same two-update record semantics."""

from pathlib import Path
import sys

import numpy as np

import record_update_material as M
import record_update_run as R
import record_update_study as study


V1 = R.OUT
OLD_PILOT = M.VOCABULARY['pilot']
PILOT = 'Bronze Turtle Rocket Plum Fox Bamboo Creek Torch Slate Apple Lily Cloud Meadow Daisy Coffee Sugar Captain Olive Jasper Teak Tulip Canyon Ash Ranch Royal Elder Crystal Gale Flint Claw Sun Moon Glacier? Aspen? Tiger? Chisel Dandelion Shell? Seal Pelican Goblet Sapphire Pearl Sage'
SOURCE = Path(__file__).relative_to(R.ROOT).as_posix()
BASE_HASHES = R.source_hashes
BASE_CANDIDATES = R.candidates
BASE_VERIFY = R.verified


def render(frame, labels, active, actions, codebook, order):
    a,b,domain=M.LABELS[frame]
    slots=(a,b)
    target=slots[active]
    intros=(
        f'Track one record in these {domain}. Target: {target}. Initial labels: {a}={labels[0]}, {b}={labels[1]}.',
        f'This {domain} ticket reports the label for {target}. Before the notices, {a} has {labels[0]} and {b} has {labels[1]}.',
        f'Prepare a report for {target} only. At the start, label {labels[0]} belongs to {a} and label {labels[1]} belongs to {b}.',
        f'{domain.capitalize()} log. Start: {b} holds {labels[1]}. Separately, {a} holds {labels[0]}. The record of interest is {target}.',
    )
    rules=(
        'Only an approved notice for the target can change its label. Ignore pending notices and notices for the other record.',
        'An approved notice applies only to its named record. Pending proposals and changes to the non-target record cannot affect this report.',
        'For this report, discard any notice marked pending and every notice addressed to a different record.',
        'For the record of interest, only approval puts a proposed change into effect. A notice about the other record is irrelevant.',
    )
    notices=[]
    for index,slot in enumerate(actions):
        which,approval=M.ACTION[slot]
        actor=slots[active if which=='selected' else 1-active]
        value=labels[index+2]
        earlier,later=('earlier','later'),('first','second')
        notices.append((
            f'Notice {index+1}: record {actor}; proposed label {value}; {approval}.',
            f'{later[index].capitalize()} notice: {actor} proposal for {value} is {approval}.',
            f'The {earlier[index]} notice names {actor} and is marked {approval}, with new label {value}.',
            f'{later[index].capitalize()} change request, marked {approval}: set {actor} to {value}.',
        )[frame])
    key=', '.join(f'{labels[j]} -> {codebook[j]}' for j in order)
    choose=(
        f'For {target}, use notice 2 only if it is approved and for {target}. Otherwise use notice 1 only if it is approved and for {target}. If neither qualifies, keep the initial label of {target}.',
        f'Take the latest approved proposal specifically for {target}. If there is no approved proposal for {target}, report its initial label.',
        f'After discarding the irrelevant notices, report the last new label in the remaining notices for {target}. If none remain, retain the initial label.',
        f'Check the second request for {target} first, then the first request for {target}. Only an approved request can supply the label. With no such request, use the starting label of {target}.',
    )[frame]
    request=(choose+' On the State line put that label exactly, and on the Code line put the letter given by the key for that label. '
             'Use exactly two lines and end after the letter. Format:\nState: <label>\nCode: <P or Q>')
    return intros[frame],rules[frame],notices,f'Letter key: {key}.',request


def candidates(tokenizer,row):
    all_=BASE_CANDIDATES(tokenizer,row)
    expected=tuple(tokenizer.encode(f" {row['truth'][-1]}  \nCode: {row['answer']}{tokenizer.eos_token}",
                                    add_special_tokens=False))
    assert expected in all_
    return [expected]+[draft for draft in all_ if draft!=expected]


def text(row,site):
    ids=np.asarray(row['token']['ids'][:site+1],dtype=np.int64)
    n=len(ids)
    unigram=np.bincount(ids%64,minlength=64)/n
    # Distinct weights modulo 64 retain the distinction between ab and ba,
    # subject to the fixed hash-collision budget.
    ordered=np.bincount((ids[1:]*39+ids[:-1]*49)%64,minlength=64)/max(n-1,1)
    tail=np.bincount(ids[-12:]%32,minlength=32)/min(n,12)
    return np.r_[1.,n/M.WIDTH,row['g']>=0 and site>row['g'],unigram,ordered,tail]


def hashes():
    return BASE_HASHES()|{SOURCE:R.C.A.digest(R.ROOT/SOURCE)}


def validate_previous():
    before=(V1/'plan.json',V1/'pilot-behavior.json')
    assert not any((V1/f'{s}-behavior.json').exists() for s in ('discovery',*R.HOLDOUTS))
    import json
    pilot=json.loads(before[1].read_bytes())
    assert not pilot['gate']['passed']
    assert all(r['format'] and r['stop'] for r in pilot['records'])
    return {p.name:R.C.A.digest(p) for p in before}


def configure():
    M.RUN='RECORD-UPDATE-20261117-v2'
    M.SEED=29572027
    exclude=set().union(*(set(words.split()) for words in M.VOCABULARY.values()))
    M.VOCABULARY=dict(M.VOCABULARY,pilot=' '.join(x for x in PILOT.split() if x not in exclude and x.isalpha()))
    M.render=render
    R.OUT=V1.with_name('record_update_20261117_v2')
    R.TEMP=R.TEMP.with_name('record_update_20261117_v2')
    R.source_hashes=hashes
    R.candidates=candidates
    R.verified=verified
    study.text_features=text


def verified():
    p=BASE_VERIFY()
    binding=R.read('revision.json')
    assert binding['previous']==validate_previous() and binding['new_plan']==R.C.A.digest(R.OUT/'plan.json')
    return p


def close():
    import json
    import record_update_report as report

    original=report.render
    old=json.loads((V1/'pilot-behavior.json').read_bytes())
    bad=[r for r in old['records'] if not r['joint']]
    versions='\n'.join(f"  - {r['id']}，{r['failure']}，输出 {r['text'].replace(chr(10),' ↵ ')}" for r in bad[:4])

    def both(*args):
        prefix=f"""### 首版行为失败与改版范围

首版 28 个独立试跑家族，56 个视图全部通过格式及停止检查，联合正确 {56-len(bad)}/56。应保留初始标签的两句式合计 17/32，各含一次合格更新的两句式合计 23/24。典型错误为取竞争记录的新标签，或把 pending 提议写成实际标签。首版全量结果在 §tests/glm5/result/record_update_20261117_v1/pilot-behavior.json§，所有发现与留出未运行。

{versions}

修订观察界面在新家族上明确指定目标并写出末次批准生效、没有批准则保留原值的决策顺序。两次更新、目标/旁支、批准/待定、P/Q 码本和自然完成评分不变。新版本发现和留出重新固定，原试跑结果不参与预测训练。等预算文本基线使用保留顺序的 bigram 哈希。新成功若出现，仅适用于这版明确规则的提示。

""".replace('§',chr(96))
        return prefix+original(*args)

    report.render=both
    report.report(True)


def main(action):
    configure()
    if action=='freeze':
        previous=validate_previous()
        R.main(action)
        p=R.read('plan.json')
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(R.MODEL,local_files_only=True)
        new=set(M.vocabulary(tok)['pilot'])
        assert not new.intersection(OLD_PILOT.split())
        R.write('revision.json',dict(previous=previous,new_plan=R.C.A.digest(R.OUT/'plan.json'),
                at=R.C.stamp(),provenance=p['provenance'],
                rationale='V1 preserved: State semantic failure despite all formats and stops. State rule equivalently restated; independent lexical pilot, no heldout from V1 opened.',
                changes='Explicit target priority and retain fallback, empirically common spacing draft verified under the original causal greedy rule; same feature budget with order-sensitive bigram hash.'))
    elif action=='close':close()
    else:R.main(action)


if __name__=='__main__':
    main(sys.argv[1])
