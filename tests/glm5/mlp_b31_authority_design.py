"""Frozen B31→B32 authorization-role assay, native CUDA BF16."""
import argparse
from datetime import datetime
from importlib.metadata import version
from itertools import product
from pathlib import Path
import random

import mlp_b34_confirm_design as P
from mlp_tail_keyword_audit import OUT, SEED
from shared.dist_protocol.natural_register_runner import snapshot

ROOT, MODEL, SNAPSHOT, CAP = P.ROOT, P.MODEL, P.SNAPSHOT, P.CAP
READ, SAVE, HASH = P.READ, P.SAVE, P.HASH
RUN = 'MLP-B31-B32-AUTHORITY-v1'
DISCOVERY, HOLDOUT, WIDTH, GROUP = 224, 768, 9728, 32
ALPHA_UNIT, ALPHA_PRIMARY = P.ALPHA_UNIT, P.ALPHA_PRIMARY
DOSES, PRIMARY = P.DOSES, P.PRIMARY
FRAMES = {
    'discovery': [
        ('permit', '{p} may approve the permit. {q} can prepare drafts but cannot approve it.', 'May {who} approve the permit?'),
        ('payment', '{q} submits the payment request. The policy grants approval authority only to {p}.', 'Can {who} authorize this payment?'),
        ('notice', 'Only {p} has signing authority for this notice; {q} just checks its formatting.', 'Is {who} authorized to sign this notice?'),
        ('dispatch', 'The dispatcher grants {p} permission to release the order. {q} can view the order only.', 'May {who} release this order?'),
    ],
    'holdout': [
        ('editorial', '{q} is a contributor. The only account empowered to accept the manuscript is {p}.', 'Does {who} have authority to accept the manuscript?'),
        ('quarantine', 'The inspector delegates clearance of this lot to {p}. The record gives {q} no clearance authority.', 'Can {who} clear the lot?'),
        ('keycard', 'This keycard will be activated only if {p} signs off. {q} is listed solely as a witness.', 'Can {who} sign off on the activation?'),
        ('access', '{q} may read the application, but granting access is reserved to {p} under the current rule.', 'Is granting access within the authority of {who}?'),
    ],
}
FILLER = {
    'discovery': [
        'The counter opens early on weekdays. Paper forms are stacked in shallow trays near the entrance. Staff put the empty clips into a separate cup after the morning review.',
        'The index is sorted by date and page. In the margin an attendant records the cabinet number and the condition of each sleeve before returning the folder to storage.',
        'During inventory the supervisor checks the paper stock, cover sheets, pencils and lamps. The notes about equipment are filed separately from the operational record.',
        'The branch keeps a reading copy and a second copy for storage. Both are stamped on arrival and placed on the same shelf after the evening count.',
    ],
    'holdout': [
        'The light by the window works well in summer. When the room is dim the attendant moves the table closer to the wall and leaves the central aisle clear.',
        'The reference volumes have their own shelves, ordered by publication year. A blank divider marks the transition to the newer volumes and can be moved during inventory.',
        'Loose pages are counted before the binding is repaired. Any marks on the cover are recorded on a separate sheet along with the height and width of the volume.',
        'In the afternoon a cart brings clean covers and protective sleeves. The clerk counts the packages, puts unused supplies back in the cupboard and notes the arrival time.',
    ],
}


def material(tok, split, n, rng):
    strata = list(product(range(4), range(2), range(2), range(2)))
    schedule = (strata*((n+31)//32))[:n]
    rng.shuffle(schedule)
    for i, (frame, reverse, who_index, late) in enumerate(schedule):
        label, evidence, question = FRAMES[split][frame]
        prefix = 'D' if split=='discovery' else 'L'
        names = [f'[{prefix}{i:04d}P]',f'[{prefix}{i:04d}Q]']
        who, other = names[who_index], names[1-who_index]
        filler = [f'Entry {j+1}. '+rng.choice(FILLER[split]) for j in range(17+rng.randrange(4))]
        position = len(filler)-2 if late else 2
        prompts, tokens = [], []
        for authorized in (False,True):
            record = evidence.format(p=who if authorized else other, q=other if authorized else who)
            labels = 'A means no. B means yes.' if reverse else 'A means yes. B means no.'
            text = '\n'.join([f'File {prefix}{i:04d}. Use the governing record to answer the question.',
                    *filler[:position], 'Governing record. '+record, *filler[position:],
                    question.format(who=who), labels+' Reply with one letter.'])
            prompt = tok.apply_chat_template([{'role':'user','content':text}], tokenize=False,
                add_generation_prompt=True,enable_thinking=False)+'Answer:'
            prompts.append(prompt)
            tokens.append(tok.encode(prompt,add_special_tokens=False))
        lo, hi = sorted(len(t) for t in tokens)
        if lo!=hi or not 520<=lo<=1024 or tokens[0][-1]!=tokens[1][-1]:
            raise ValueError(f'{split}/{i}: paired lengths {lo}, {hi}')
        yield {'id':f'{split}-{i:04d}','split':split,'frame':label,'reverse_code':bool(reverse),
               'record_late':bool(late),'recipient_index':(i+frame+late)%2,'prompts':prompts,
               'tokens':tokens,'correct':[1-reverse,reverse]}


def freeze():
    from transformers import AutoTokenizer

    if (OUT/'prereg.json').exists():
        raise RuntimeError('Frozen study already exists')
    OUT.mkdir(exist_ok=True)
    model_sha = snapshot(MODEL)
    assert model_sha==SNAPSHOT
    prov = {'git_sha':P.git('rev-parse','HEAD'),'git_dirty':bool(P.git('status','--porcelain')),
            'model_path':str(MODEL.resolve()),'model_revision':model_sha,'seed':SEED,
            'tool':RUN,'created_at':datetime.now().astimezone().isoformat()}
    tok = AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    encoded = [tok.encode(' '+key,add_special_tokens=False) for key in 'ABCDE']
    assert all(len(row)==1 for row in encoded)
    cap = [row for row in READ(CAP) if row['gate_eligible']]
    rng = random.Random(SEED)
    inputs = {}
    for split,n in [('discovery',DISCOVERY),('holdout',HOLDOUT)]:
        rows = list(material(tok,split,n,rng))
        path = OUT/f'{split}-materials.json'
        SAVE(path,{'provenance':prov,'items':rows})
        lengths = [len(row['tokens'][0]) for row in rows]
        inputs[split] = {'sha256':HASH(path),'n':n,'tokens':[min(lengths),max(lengths)]}
    files = ('mlp_b31_authority_design.py','mlp_b31_authority_run.py','mlp_b31_authority_report.py',
             'mlp_tail_keyword_audit.py','mlp_b34_confirm_design.py','mlp_b34_confirm_run.py','mlp_b34_confirm_analyze.py')
    plan = {'schema':'mlp_b31_authority_plan.v1','run_id':RUN,'kind':'real','provenance':prov,'precision':'CUDA BF16 / SDPA',
            'inputs':inputs,'choice_ids':[r[0] for r in encoded],
            'sources':{n:HASH(Path(__file__).with_name(n)) for n in files},
            'software':{p:version(p) for p in ('torch','transformers','numpy','scipy')},
            'atlas_inputs':{'base_manifest':HASH(P.atlas.OUT/'manifest.json'),
                            'extension_manifest':HASH(OUT.parent/'mlp_tail_b28_b31_v1/manifest.json')},
            'power':P.power_plan(len(cap)),'alpha_unit':ALPHA_UNIT,'alpha_primary':ALPHA_PRIMARY,
            'capability':{'n':len(cap),'sha256':HASH(CAP),'margin':.02,'pool':'历史冻结选择题，正负编辑均需通过'},
            'calibration':{'pair_base_accuracy_min':.85,'noop_logits_max_abs':0.0},
            'semantic':{'family':'英文长文档中谁拥有指定审批权','group':GROUP,'units_min':.62,
                        'each_code_min':.55,'group_min_gain':.05,'control':'同层等数量冻结随机单元，方向只在发现集拟合',
                        'scope':'保持每对词多集和读出token，交换授权者，反转答案字母，留出新措辞、任务情境和文档编号'},
            'causal':{'doses':DOSES,'primary':PRIMARY,'min_gain':.05,'norm_tolerance':.01,
                      'source':'最后token的B31 A原生单元组按发现集角色差值加减，完整down_proj重算',
                      'control':'B31输出残差处实际BF16差值1%等范数随机方向',
                      'rescue':'B32已选A坐标恢复到未编辑基线，在B32输出边界做1%等范数随机假恢复，重新计算B33–B35',
                      'restore':'取消所有修改，用原始B31输出重放B32–B35，要求词表logits逐位恢复',
                      'primary_behavior':'A/B选择翻转。全词表首词与B32组位移同时报告，作为次要终点。'},
            'custody':'先冻结新材料与功效，仅打开发现材料和采集发现输出。候选和已打开样本的数值校准冻结后才能打开留出。',
            'resources':{'stage_minutes':35,'new_disk_gib_max':1.5}}
    SAVE(OUT/'prereg.json',plan)
    print({'sha256':HASH(OUT/'prereg.json'),'inputs':inputs,'power':plan['power']},flush=True)


def checked():
    plan = READ(OUT/'prereg.json')
    assert plan['provenance']['model_revision']==SNAPSHOT
    for name,sha in plan['sources'].items():
        if HASH(Path(__file__).with_name(name))!=sha:
            raise ValueError(f'Frozen code changed: {name}')
    assert all(x['pass'] for x in plan['power'].values())
    return plan


def items(plan,split):
    path=OUT/f'{split}-materials.json'
    assert HASH(path)==plan['inputs'][split]['sha256']
    return READ(path)['items']


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze'])
    parser.parse_args()
    freeze()
