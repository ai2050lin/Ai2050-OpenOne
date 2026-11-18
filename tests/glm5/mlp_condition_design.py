"""Frozen factorial long-context validation of two archived native MLP pairs."""
from collections import Counter
from datetime import datetime
from importlib.metadata import version
from itertools import product
from pathlib import Path
import random
import shutil

import mlp_b34_confirm_design as P
from tools.power_analysis import check_phase
from shared.dist_protocol.natural_register_runner import snapshot

ROOT, MODEL, SNAPSHOT, CAP = P.ROOT, P.MODEL, P.SNAPSHOT, P.CAP
READ, SAVE, HASH = P.READ, P.SAVE, P.HASH
OUT = ROOT/'tests/glm5/result/mlp_condition_v1'
RUN, SEED = 'MLP-CONDITIONAL-PATHS-v1', 202611092869
BLOCKS = (29,30,33,34)
PAIRS = (('broad',29,17,30,2),('rare',33,1521,34,6097))
DISCOVERY, HOLDOUT = 128,512
DOSES, PRIMARY = (.5,1.,2.),1.
SOURCES = ('mlp_condition_design.py','mlp_condition_run.py','mlp_condition_analysis.py')
FRAMES = {
 'discovery':[
  ('seal','open the sealed cabinet','The access register assigns these statuses: {a} is {x} to {action}; {b} is {y} to {action}.'),
  ('sample','release the stored sample','The sample register gives account {a} status {x} and account {b} status {y} for permission to {action}.'),
  ('screen','activate the display','Under the display rule, account {a} is {x} to {action}. Under the same rule, account {b} is {y} to {action}.'),
  ('board','publish the bulletin','The bulletin policy states: {a}: {x}; {b}: {y}. Each status describes whether that account may {action}.')],
 'holdout':[
  ('vault','unlock the storage vault','The vault rule explicitly marks {a} as {x} to {action} and {b} as {y} to {action}.'),
  ('broadcast','transmit the recorded message','Permission to {action} has been assigned as follows. Account {a}: {x}. Account {b}: {y}.'),
  ('device','restart the recording device','The device record lists two accounts and their permission to {action}: {a} is {x}, while {b} is {y}.'),
  ('loan','release the archived recording','For the operation "{action}", the governing rule specifies {x} for account {a}, and {y} for account {b}.')]
}
BACKGROUND = {
 'discovery':[
  'The observation room contains a narrow bench and a tall rack. At noon the caretaker checks the door hinges and leaves a note about the temperature beside the window.',
  'A visitor measures the shelves with a wooden ruler. The dimensions are copied onto a sheet together with the color of the walls and the number of empty boxes.',
  'The archive guide describes how cardboard covers are repaired. Worn corners are folded flat, loose labels are removed, and the finished covers are left on a dry surface.',
  'The cart follows a painted line across the floor. Its wheels are inspected on Mondays, and a spare wheel is stored behind the small folding table.'],
 'holdout':[
  'Morning sunlight reaches the east passage through a narrow opening. A maintenance worker records the condition of the stone floor and the height of the railing beside it.',
  'Several cloth samples lie in a shallow drawer. A catalog describes their texture, thickness and weave, while a separate notebook lists the dates of the last inventory.',
  'The museum map shows three halls connected by a covered walkway. Visitors leave umbrellas near the entrance and collect them before returning to the open courtyard.',
  'A repair guide describes the wooden frames around the wall panels. It lists the sizes of screws, the amount of varnish and the time needed for each coat to dry.']}

def materials(tok, split, families):
    rng=random.Random(SEED+(split=='holdout'))
    labels=(('K','T'),('M','R')) if split=='discovery' else (('C','D'),('E','F'))
    for family in range(families):
        frame,action,template=FRAMES[split][family%4]
        background=[f'Note {j+1}. '+rng.choice(BACKGROUND[split]) for j in range(16)]
        rows=[]
        for identity,role,late,mapping in product(range(2),repeat=4):
            a,b=labels[identity]
            x,y=('permitted','forbidden') if role else ('forbidden','permitted')
            record='Governing record. '+template.format(a=a,b=b,x=x,y=y,action=action)
            at=13 if late else 2
            text='\n'.join([f'Ledger NC{split[0]}{family:05d}. Apply the governing record only.',
                *background[:at],record,*background[at:],
                f'Is account {a} permitted to {action}?',
                ('A means no. B means yes.' if mapping else 'A means yes. B means no.')+' Reply with one letter.'])
            prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:'
            tokens=tok.encode(prompt,add_special_tokens=False)
            # Read only after the entire evidence; all checkpoints are causally visible.
            end=prompt.index(record)+len(record)
            evidence=len(tok.encode(prompt[:end],add_special_tokens=False))-1
            positions=sorted(set([evidence,*[int(evidence+(len(tokens)-1-evidence)*q) for q in (.2,.4,.6,.8)],len(tokens)-1]))
            positions=([evidence]*6+positions)[-6:]
            assert 600<=len(tokens)<=1024 and positions[-1]==len(tokens)-1
            rows.append({'id':f'{split}-{family:04d}-{identity}{role}{late}{mapping}','family':family,
                'frame':frame,'identity':identity,'role':role,'late':late,'mapping':mapping,
                'gold':mapping if role else 1-mapping,'tokens':tokens,'prompt':prompt,'positions':positions})
        # All four factors are crossed, not merely shuffled together.
        assert len({len(r['tokens']) for r in rows})==1, ('factor_lengths',split,family)
        assert len({r['tokens'][-1] for r in rows})==1
        for identity,late,mapping in product(range(2),repeat=3):
            pair=[r for r in rows if (r['identity'],r['late'],r['mapping'])==(identity,late,mapping)]
            assert Counter(pair[0]['tokens'])==Counter(pair[1]['tokens'])
        yield {'family':family,'frame':frame,'rows':rows}

def freeze():
    from transformers import AutoTokenizer
    assert not (OUT/'prereg.json').exists(), 'Existing frozen protocol'
    assert shutil.disk_usage(ROOT).free>3*2**30
    OUT.mkdir(exist_ok=True)
    assert snapshot(MODEL)==SNAPSHOT
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
    prov={'git_sha':P.git('rev-parse','HEAD'),'git_dirty':bool(P.git('status','--porcelain')),
        'model_path':str(MODEL.resolve()),'model_revision':SNAPSHOT,'seed':SEED,
        'created_at':datetime.now().astimezone().isoformat(),'tool':RUN}
    inputs={}
    for split,n in [('discovery',DISCOVERY),('holdout',HOLDOUT)]:
        fs=list(materials(tok,split,n))
        path=OUT/f'{split}-materials.json'
        SAVE(path,{'provenance':prov,'families':fs})
        lengths=[len(r['tokens']) for f in fs for r in f['rows']]
        inputs[split]={'sha256':HASH(path),'families':n,'observations':len(lengths),'tokens':[min(lengths),max(lengths)]}
    capn=sum(r['gate_eligible'] for r in READ(CAP))
    power={
        'discovery':check_phase(.20,DISCOVERY,.35,.05),
        'prediction_family_win':check_phase(.15,HOLDOUT,.5,.005),
        'intervention_paired':check_phase(.12,HOLDOUT,.35,.005),
        'rare_stratum_if_eligible':check_phase(.25,128,.5,.0125),
        'capability':check_phase(.025,capn,.1,.0125)}
    assert all(x['pass'] for x in power.values()),power
    ids=[tok.encode(' '+k,add_special_tokens=False) for k in 'ABCDE']
    assert all(len(x)==1 for x in ids)
    plan={'schema':'mlp_conditional_plan.v1','run_id':RUN,'kind':'real','provenance':prov,
      'model_snapshot':SNAPSHOT,'precision':'CUDA BF16 / SDPA','inputs':inputs,'pairs':PAIRS,
      'sources':{n:HASH(Path(__file__).with_name(n)) for n in SOURCES},'power':power,
      'software':{n:version(n) for n in ('torch','transformers','numpy','scipy')},'choice_ids':[x[0] for x in ids],
      'prediction':{'method':'ridge lambda 1; nuisance-only versus same nuisance plus observed source A and source-by-factor interactions',
         'features':'intercept, identity, role, evidence position, answer mapping, frame indicators, token position; never target A or future tokens',
         'gate':'family mean squared error lower in >=65% families, one-sided binomial p<=.005, total MSE reduction>=10%; both endpoints separate',
         'rare':'absolute source activation >= old discovery q99, frozen before new outputs; >=128 distinct holdout families for confirmatory stratum; otherwise descriptive',
         'condition_scope':'source state is observed, not independent language-to-all-states prediction'},
      'behavior':{'minimum_per_frame':.85,'low_margin_nats':2.,'gate_scope':'all 16 variants, family bootstrap; baseline gate affects behavioral interpretation, not observation recording'},
      'causal':{'doses':DOSES,'primary':PRIMARY,'norm_tolerance':.01,'minimum_gain':.05,'alpha':.005,
         'source':'one fixed native source A unit at final token; magnitude=discovery SD; direction selected using discovery-calibrated output prediction',
         'predict_first':'discovery finite-edit linear output response per role/mapping; heldout prediction written and flushed before first intervention',
         'control':'actual BF16 source residual perturbation norm matched to seeded random direction, 1% tolerance',
         'rescue':'restore nominated downstream A unit, with matched BF16 random sham at target boundary; no initial switch => no mediation',
         'restore':'cancel edits and require complete vocabulary logits bitwise equal',
         'selection':'one variant per family chosen by (family//4) modulo 16; crossed with frame family modulo 4, before seeing any outputs',
         'scope':'suffix replay recomputes all following layers and attention; no new KV-history claim'},
      'capability':{'sha256':HASH(CAP),'n':capn,'margin':.02,'alpha_each':.0125,'source_signs':4,'dose':PRIMARY},
      'resources':{'new_disk_max_gib':2.0,'gpu_stage_minutes':60,'batch':4},
      'custody':'discovery -> freeze prediction weights and finite-edit calibration -> holdout baseline -> per-family prediction commit -> edit -> score; never reuse opened holdout for refitting',
      'material_source':'new author-created English factorial long contexts; no copied natural documents; roles permuted with fixed entity order'}
    SAVE(OUT/'prereg.json',plan)
    print({'plan_sha256':HASH(OUT/'prereg.json'),'inputs':inputs,'power':power},flush=True)

def checked():
    p=READ(OUT/'prereg.json')
    for n,sha in p['sources'].items():assert HASH(Path(__file__).with_name(n))==sha,('code_changed',n)
    return p

def families(plan,split):
    path=OUT/f'{split}-materials.json'
    assert HASH(path)==plan['inputs'][split]['sha256']
    return READ(path)['families']

if __name__=='__main__':freeze()
