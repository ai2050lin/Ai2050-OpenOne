"""Frozen full-prefix test of massive activation and a shared native input."""
from datetime import datetime
from pathlib import Path
from itertools import product
import random
import shutil
import numpy as np
import mlp_condition_design as OLD
from tools.power_analysis import check_phase
from shared.dist_protocol.natural_register_runner import snapshot

ROOT,MODEL,SNAPSHOT,CAP=OLD.ROOT,OLD.MODEL,OLD.SNAPSHOT,OLD.CAP
READ,SAVE,HASH=OLD.READ,OLD.SAVE,OLD.HASH
OUT=ROOT/'tests/glm5/result/mlp_trigger_v1'
RUN='MLP-TRIGGER-UPSTREAM-v1'
SEED=202611092882
N_DISC,N_HOLD=96,384
PAIRS={'broad':(29,17,30,2),'rare':(33,1521,34,6097)}
FRAMES=[
 'Record: account {a} has role {x}; account {b} has role {y}.',
 'The register gives {a} the role {x} and {b} the role {y}.',
 'Role {x} is assigned to {a}, while role {y} is assigned to {b}.',
 'Account {a}: role {x}. Account {b}: role {y}.',
 'For this file, {a} serves as {x}; {b} serves as {y}.',
 'The current roles are {x} for {a} and {y} for {b}.',
 'This entry lists account {a} under {x} and account {b} under {y}.',
 'The two assignments are as follows: {a} holds {x}; {b} holds {y}.',
 'The signed table links account {a} to role {x} and account {b} to role {y}.',
 'Within this register the role of {a} is {x}, and the role of {b} is {y}.',
 'Read the role fields in order: {a}, {x}; then {b}, {y}.',
 'The authorized entry names {a} as {x}, followed by {b} as {y}.',
 'Two accounts appear in the file. The role for {a} is {x}; the role for {b} is {y}.',
 'In the role table, the row for {a} says {x}; the row for {b} says {y}.',
 'The assigned roles remain {x} for account {a}, and {y} for account {b}.',
 'Role allocation: {a} belongs in the {x} field, and {b} belongs in the {y} field.'
]

def material(tok,split,n):
    # Authored grammatical paragraphs with identical tokenizer length. Pool vocabulary
    # is disjoint across discovery/holdout; paragraph frames are explicitly limited.
    nouns=('table shelf basket cabinet ladder panel barrel carpet'.split() if split=='discovery'
           else 'window doorway cushion curtain railing counter cupboard blanket'.split())
    adjectives='wooden narrow painted folded empty heavy small clean'.split()
    actions='checked moved cleaned measured inspected repaired stored covered'.split()
    pool=[]
    for noun,adj,verb in product(nouns,adjectives,actions):
        text=f'The {adj} {noun} stands near the entrance of the quiet room. Its surface is {verb} during the morning inspection and a dated note records the work.'
        if len(tok.encode(text,add_special_tokens=False))==28:pool.append(text)
    assert len(pool)>=24,len(pool)
    rng=random.Random(SEED+(split=='holdout'))
    rows=[]
    for f in range(n):
        fi=f%16;role=(f//16)%2;mapping=(f//32)%2;identity=(f//64)%3
        a,b=(('H','J'),('N','P'),('K','T'))[identity] if split=='discovery' else (('S','V'),('W','Z'),('C','D'))[identity]
        x,y=('APPROVER','OBSERVER') if role else ('OBSERVER','APPROVER')
        frame=FRAMES[fi] if split=='holdout' else ('Assignment for {a}: {x}. Assignment for {b}: {y}.')
        record=frame.format(a=a,b=b,x=x,y=y)
        unique=rng.sample(pool,24);repeated=rng.choice(pool)
        for long,repetition,past in product(range(2),repeat=3):
            count=24 if long else 16
            background=[repeated]*count if repetition else unique[:count]
            background=[t.replace('surface is','surface was') if past else t for t in background]
            instruction=('For APPROVER, output B. For OBSERVER, output A.' if mapping else 'For APPROVER, output A. For OBSERVER, output B.')
            text='\n'.join([f'Document NU{split[0]}{f:05d}. Use only the explicit role record.',record,
               *[f'Background note {i+1}. {s}' for i,s in enumerate(background)],
               f'What role does account {a} hold? {instruction} Output only the corresponding letter.'])
            prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:'
            tokens=tok.encode(prompt,add_special_tokens=False)
            assert 600<=len(tokens)<=1150,len(tokens)
            rows.append({'id':f'{split}-{f:04d}-{long}{repetition}{past}','family':f,'frame':fi,
               'role':role,'mapping':mapping,'identity':identity,'long':long,'repetition':repetition,'past':past,
               'gold':mapping if role else 1-mapping,'tokens':tokens,'prompt':prompt})
        for long in range(2):assert len({len(r['tokens']) for r in rows[-8:] if r['long']==long})==1
    return rows

def freeze():
    from transformers import AutoTokenizer
    assert not (OUT/'prereg.json').exists()
    assert shutil.disk_usage(ROOT).free>600*2**20
    assert snapshot(MODEL)==SNAPSHOT
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    prov={'git_sha':OLD.P.git('rev-parse','HEAD'),'git_dirty':bool(OLD.P.git('status','--porcelain')),
          'model_path':str(MODEL.resolve()),'model_revision':SNAPSHOT,'seed':SEED,
          'tool':RUN,'created_at':datetime.now().astimezone().isoformat()}
    mats={}
    for split,n in [('discovery',N_DISC),('holdout',N_HOLD)]:
        data=material(tok,split,n);SAVE(OUT/f'{split}-materials.json',{'provenance':prov,'rows':data})
        mats[split]={'sha256':HASH(OUT/f'{split}-materials.json'),'n':len(data),'families':n,
                    'tokens':[min(len(r['tokens']) for r in data),max(len(r['tokens']) for r in data)]}
    sources=['tests/glm5/'+n for n in ['mlp_trigger_design.py','mlp_trigger_run.py','mlp_condition_stream_load.py',
       'mlp_condition_design.py','mlp_condition_run.py','mlp_b34_confirm_run.py','mlp_b34_confirm_design.py','mlp_tail_goodfire.py']]
    sources+=['tools/power_analysis.py','shared/dist_protocol/mechanism_trace.py','shared/dist_protocol/natural_register_runner.py']
    power={'rare_paired_change':check_phase(.08,N_HOLD,.1,.005),'prediction_family_win':check_phase(.15,N_HOLD,.5,.005),
           'causal_event_change':check_phase(.08,N_HOLD,.1,.005),'capability':check_phase(.025,2200,.1,.0125)}
    assert all(v['pass'] for v in power.values())
    p={'schema':'mlp_trigger_plan.v1','run_id':RUN,'provenance':prov,'model_snapshot':SNAPSHOT,'kind':'real',
       'materials':mats,'sources':{n:HASH(ROOT/n) for n in sources},'power':power,
       'choice_ids':[tok.encode(' '+s,add_special_tokens=False)[0] for s in 'ABCDE'],
       'capability':{'sha256':HASH(CAP),'n':2200,'margin':.02,'alpha_each':.0125,
                     'scope':'previously exposed keyed task battery; candidate selection does not use it; not general free generation'},
       'hypotheses':{'rare':'Large B33/U1521 and B34/U6097 responses reflect a common R33 coordinate-4 dominated state, not a unique lexical concept. Length/repetition/tense are crossed.',
          'broad':'A discovery-selected small set of R29 native coordinates jointly predicts B29/U17 and B30/U2; source-clamp tests whether their connection is serial or a common-input/bypass response.'},
       'readouts':'Full prefix G/V/A of four fixed units and R-coordinate4 at B29/30/33/34. Broad: final token. Rare: argmax |R33[4]| after position 63, chosen without downstream activation. BOS excluded and retained separately.',
       'batch':'Primary inference batch=1, SDPA CUDA BF16. First 16 discovery families also compared with batch=4 and repeated batch=1. Shape-dependent peaks are not assumed stable.',
       'selection':'Rare coordinate fixed 4 from exposed diagnosis. Broad top 8 native R29 coordinates ranked by product of discovery absolute correlations with both endpoints. Same residual input is observed, not graph-only inference.',
       'prediction':'ridge lambda=1 on standardized selected common-input coordinates; compare source-only, nuisance-only and common-input models. Nuisances implemented exactly: intercept, long, repetition, past, role, mapping, identity one-hot, normalized position, length. Holdout frame is new, not fitted ID.',
       'gates':{'behavior_frame_min':.85,'family_win':.65,'p':.005,'mse_ratio':.90,'rare_threshold_A':10.,'rare_threshold_R4_energy':.9,
                'minimum_eligible_families':296,'minimum_control_fraction':.95,'common_bypass_fraction':.8,
                'causal_scope':'Predictions and signed effect confidence intervals reported separately. Source clamp retaining >=80% target response supports bypass in tested scope; not unique cause.'},
       'causal':{'families':N_HOLD,'variant':'family modulo 8, balanced across 16 frames through (family//16) modulo8',
           'upstream':'shrink selected R coordinate(s) toward zero by doses 0.25,0.5,1.0; exact BF16 norm matched random controls at same R boundary within1%',
           'clamps':'at primary1.0, same R edit with nominated source A clamped to baseline; whole source MLP output clamped to baseline; direct A-unit zero alone and matched H control',
           'predict_first':'discovery calibrates response slopes vs realized selected R change; immutable per-family prediction JSON before edited suffix; uses baseline state and logits',
           'recovery':'cancel R/A edits; entire vocabulary logits bitwise restore; separate from mediator-clamp sufficiency',
           'no_KV_claim':'full suffix recomputation, no cached KV mediation claim'},
       'resources':{'expected_new_bytes_max':400*2**20,'free_floor_bytes':180*2**20,'stage_minutes':90,
           'collection_exception':'Low disk: save full 4-unit prefix and 2 selected full native R boundaries; not all 37 layers. Model snapshot+tokens+code permit full-field recomputation.'},
       'limits':['Author-created English background/role tasks, not independent natural documents. Family permutation bootstrap does not support untested frame populations.',
          'Length is intentionally varied; within each length all repetition/tense conditions have identical token counts.',
          'The R input includes accumulated attention and MLP history; no attribution to a unique earlier Head without further experiment.'],
       'diagnostic_sha256':HASH(OUT/'diagnostic.json')}
    SAVE(OUT/'prereg.json',p)
    print({'run':RUN,'materials':mats,'power':power},flush=True)

def checked():
    p=READ(OUT/'prereg.json')
    for name,sha in p['sources'].items():assert HASH(ROOT/name)==sha,('changed',name)
    return p

def materials(p,split):
    path=OUT/f'{split}-materials.json';assert HASH(path)==p['materials'][split]['sha256'];return READ(path)['rows']

def save(name,body,p):
    SAVE(OUT/name,{'provenance':p['provenance'],'plan_sha256':HASH(OUT/'prereg.json'),**body})

if __name__=='__main__':freeze()
