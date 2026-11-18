"""Fresh explicit authorization binding after v1 failed its untouched behavior gate."""
import argparse
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
import random

import mlp_b31_authority_design as D

V1=D.OUT
OUT=V1.with_name('mlp_b31_b32_authority_v2')
RUN='MLP-B31-B32-AUTHORITY-v2'
SEED=20261109131
BASE_MATERIAL=D.material

FRAMES = {
    'discovery': [
        ('desk', 'Approval role: {p} has authority to approve the intake request. Preparation role: {q} is not authorized to approve it.', 'Is {who} authorized to approve the intake request?'),
        ('stock', 'For the inventory change, {q} may draft it but has no power to authorize it. The only authorized approver is {p}.', 'Does {who} have authorization to approve the inventory change?'),
        ('studio', '{p} is the person authorized to approve the studio booking. {q} is the preparer and may not approve the booking.', 'Is {who} authorized to approve the studio booking?'),
        ('workshop', 'The workshop request is recorded under two roles. {q} has preparation access only. {p} is the only account that can approve the request.', 'Is {who} permitted to approve the workshop request?'),
    ],
    'holdout': [
        ('library', 'The library grants exclusive approval authority for this room reservation to {p}. {q} prepares the request and lacks authority to approve it.', 'Under the library rule, is {who} allowed to approve this room reservation?'),
        ('rental', '{q} enters the rental extension in the system and is prohibited from approving it. Approval is authorized solely for {p}.', 'According to the rental record, is {who} authorized to approve this extension?'),
        ('exhibit', 'For this exhibition loan, {p} holds the approval permission. The other account, {q}, may draft the loan request and cannot approve the loan.', 'Does {who} have permission to approve this exhibition loan?'),
        ('transport', 'The transport exception may be approved by {p} alone. The record lists {q} as preparer without approval rights.', 'Is {who} allowed to approve this transport exception?'),
    ],
}
FILLER = {
    'discovery': [
        'A strip of light reaches the shelves in the early hours. The attendant shifts the trays when needed, leaving enough space between the table and the filing cabinet.',
        'The maintenance log lists paint samples, hinge sizes and replacement dates. Staff use the information when they check the storage units and the side doors.',
        'A clerk numbers the outer sleeves during the afternoon count. The markings help the next shift tell newly sorted folders from those waiting for review.',
        'The cooling fan is turned off at the end of the day. Loose labels are collected in a small container and the remaining papers are secured with a clip.',
    ],
    'holdout': [
        'Sturdy cartons sit on the lower shelf near the back wall. A separate list records the width of each shelf and the approximate weight of the empty cartons.',
        'At dawn the overhead lamps are checked before the corridor opens. A spare bulb and a folded cloth are kept in the cupboard beside the stairs.',
        'The room has a wooden bench for visitors and a metal rail for folders. The floor is swept in the evening and the bench is wiped with a dry cloth.',
        'An inventory of the wall maps notes the color of their bindings and the number of loose sheets. The maps are stored upright when they are not being displayed.',
    ],
}


def material(tok,split,n,rng):
    prefix='D' if split=='discovery' else 'L'
    new='X' if split=='discovery' else 'Y'
    from collections import Counter
    for row in BASE_MATERIAL(tok,split,n,rng):
        prompts=[p.replace(f'File {prefix}',f'File {new}').replace(f'[{prefix}',f'[{new}') for p in row['prompts']]
        tokens=[tok.encode(p,add_special_tokens=False) for p in prompts]
        a,b=tokens
        assert len(a)==len(b) and 520<=len(a)<=1024 and a[-1]==b[-1]
        assert sum(x!=y for x,y in zip(a,b))==2 and Counter(a)==Counter(b)
        yield {**row,'id':'v2-'+row['id'],'tokens':tokens,'prompts':prompts}


def configure():
    D.OUT,D.RUN,D.SEED=OUT,RUN,SEED
    D.FRAMES,D.FILLER,D.material=FRAMES,FILLER,material


def freeze():
    old=D.READ(V1/'holdout-behavior-gate.json')
    assert not old['passed'] and not list(V1.glob('holdout-*.npz'))
    source_sha=D.HASH(Path(__file__))
    original_save=D.SAVE

    def save(path,data):
        if path.name=='prereg.json':
            data['sources'][Path(__file__).name]=source_sha
            data['sources']['mlp_b31_authority_confirm.py']=D.HASH(Path(__file__).with_name('mlp_b31_authority_confirm.py'))
            data['prior_negative']={'prereg_sha256':D.HASH(V1/'prereg.json'),
                                    'gate_sha256':D.HASH(V1/'holdout-behavior-gate.json'),
                                    'accuracy':old['both_correct'],'by_frame':old['frame_rates'],
                                    'scope':'V1 heldout only opened for baseline. No v1 heldout causal data.'}
        original_save(path,data)
    D.SAVE=save
    try:
        D.freeze()
    finally:
        D.SAVE=original_save
    keyword=D.READ(V1/'keyword-audit.json')
    keyword['source_audit_sha256']=D.HASH(V1/'keyword-audit.json')
    original_save(OUT/'keyword-audit.json',keyword)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['freeze','discovery','select','confirm','report','publish'])
    stage=parser.parse_args().stage
    configure()
    if stage=='freeze':
        freeze()
    elif stage=='discovery':
        from mlp_b31_authority_run import run
        run('discovery')
    elif stage in ('select','report'):
        import mlp_b31_authority_report as analysis
        getattr(analysis,stage)(D.checked())
    elif stage=='confirm':
        from mlp_b31_authority_confirm import main as confirm
        confirm()
    else:
        from mlp_b31_authority_publish import main as publish
        publish()


if __name__=='__main__':
    main()
