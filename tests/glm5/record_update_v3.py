"""Condition on the target's initial label after two independent binding failures."""

from pathlib import Path
import json
import sys

import record_update_v2 as V2


R,M=V2.R,V2.M
BASE_VOCABULARY=dict(M.VOCABULARY)
PILOT='Wolf Falconer Redwood Opal Sapphire Ruby Flintstone Raven Triad Cinnamon Caramel Silk Steel Reef Lemonade Apricot Acacia Frosted Radius Verde Canyonland Rifle Dragon Eel Goose Oyster Dolphin Wren Carnation Iris Dahlia Bumble Rancher Ivorywood Finchley Clay Quartzite Snail Thistle Storm Thorne Crimsonwood Lakewood Amethyst Pearlstone Pine Fir Oak Palm Rose Fawn Desert Bluebell Sand Marsh Prairie Dolphin Whale Alpaca Lark Cloverfield'
SOURCE=Path(__file__).relative_to(R.ROOT).as_posix()
V1=V2.V1
V2_DIR=V1.with_name('record_update_20261117_v2')
V3_DIR=V1.with_name('record_update_20261117_v3')


def render(frame, labels, active, actions, codebook, order):
    a,b,domain=M.LABELS[frame]; targets=(a,b)
    target,other=targets[active],targets[1-active]
    label,alt=labels[active],labels[1-active]
    intro=(
        f'Target record: {target}. Its initial saved label is {label}. Other record {other} begins with {alt}.',
        f'For these {domain}, read the updates for {target}. Tracking starts from its label {label}. Label {alt} belongs to {other}.',
        f'The label on file for the record we report, {target}, is {label} at the start. The separate record {other} has {alt}.',
        f'The starting value for this {domain} report is {label} in {target}. Another record, {other}, starts with {alt}. Only {target} is reported.',
    )[frame]
    rules=(
        'Apply notices in order. A notice can replace the saved label only if it is approved and names the target. Otherwise the saved label stays as it was.',
        'Check both notices. Approved replacements take effect only in their named record. Pending replacements do nothing.',
        'Discard pending notices and notices for a different record. Of the remaining notices, apply the later one. If none remain, keep the starting label.',
        'The label can change on an approved request for the reported record. A request for a different record or a pending request leaves its label alone.',
    )[frame]
    notices=[]
    for i,action in enumerate(actions):
        scope,status=M.ACTION[action]
        named=target if scope=='selected' else other
        name=labels[i+2]
        notices.append((
            f'{i+1}. Record {named} | new label {name} | {status}.',
            f'Notice {i+1} proposes {name} for {named}. Status is {status}.',
            f'The {("earlier","later")[i]} request is {status}: it assigns {name} to {named}.',
            f'{"First" if i==0 else "Then"}, a notice lists {name} for {named}, marked {status}.',
        )[frame])
    key=', '.join(f'{labels[j]} -> {codebook[j]}' for j in order)
    request=('Report the final saved label after both notices. The label must be one of the labels in the letter key. '
             'Use that label on the State line and its key letter on the Code line. End after the letter. Format:\nState: <label>\nCode: <P or Q>')
    return intro,rules,notices,'Letter key: '+key+'.',request


def before():
    prior={}
    for folder in (V1,V2_DIR):
        assert not any((folder/f'{s}-behavior.json').exists() for s in ('discovery',*R.HOLDOUTS))
        behavior=json.loads((folder/'pilot-behavior.json').read_bytes())
        assert not behavior['gate']['passed']
        prior[folder.name]={k:R.C.A.digest(folder/k) for k in ('plan.json','pilot-behavior.json')}
    return prior


def hashes():
    return V2.hashes()|{SOURCE:R.C.A.digest(R.ROOT/SOURCE)}


def verified():
    p=V2.BASE_VERIFY()
    change=R.read('revision.json')
    assert change['previous']==before()
    assert change['new_plan']==R.C.A.digest(R.OUT/'plan.json')
    return p


def configure():
    V2.configure()
    excluded=set().union(*(set(x.split()) for x in (*BASE_VOCABULARY.values(),V2.PILOT)))
    M.VOCABULARY=dict(BASE_VOCABULARY,pilot=' '.join(x for x in PILOT.split() if x.isalpha() and x not in excluded))
    M.RUN='RECORD-UPDATE-20261117-v3';M.SEED=29572028;M.render=render
    R.OUT=V3_DIR
    R.TEMP=R.TEMP.with_name(V3_DIR.name)
    R.verified=verified
    R.source_hashes=hashes


def close():
    from record_update_report import report,render as orig
    import record_update_report as module

    def both(*args):
        return """### 首两版行为失败与第三版范围

V1 的 56 个视图全部通过格式和停止，联合正确 40/56。保留条件 17/32，单次有效更新 23/24。V2 使用新试跑家族重述优先级，同样全部正常输出格式及终止，有效更新 24/24，保留条件 20/32。代表反例是把 East 或 West 记录名填入 State，或把待定提案的标签当成已生效标签。两版均在发现和留出开始前按冻结行为门停止，原始样本输出分别留在 v1、v2 的 pilot-behavior.json。

V3 在独立词表上先给出目标记录的初始标签，再要求模型根据两次通知自行保留或更新。适用范围因而限定为已提供初始绑定的记录更新、自然续写和码本读出。独立保留两版未通过的目标记录选择证据。词法基线保留 bigram 顺序，行为、预测和匹配干预的阈值未改。

"""+orig(*args)

    module.render=both
    report(True)


def main(argv):
    configure()
    if argv=='freeze':
        prior=before()
        R.main('freeze')
        p=R.read('plan.json')
        R.write('revision.json',dict(at=R.C.stamp(),provenance=p['provenance'],previous=prior,
               new_plan=R.C.A.digest(R.OUT/'plan.json'),
               rationale='The first two independent versions pass action updates far more often than retaining the selected initial state. Test update/retention by giving the initial target binding directly.',
               scope='Two updates and independent codebook. Keep initial record supplied in prompt. No final state or downstream choice is supplied. Earlier failures retained and holdouts unopened.'))
    elif argv=='close':close()
    else:R.main(argv)


if __name__=='__main__':
    main(sys.argv[1])
