"""Fresh explicit role assignment after v1 discovery failed negative-answer mapping."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path
import random
import mlp_condition_design as D

V1=D.OUT
FRAMES={
 'discovery':[
  ('catalog','The catalog assigns account {a} the role {x}. It assigns account {b} the role {y}.'),
  ('ledger','Two roles are entered in the ledger: {a} is the {x}; {b} is the {y}.'),
  ('roster','According to the current roster, account {a} holds the role {x}, and account {b} holds the role {y}.'),
  ('register','The registration form contains these role fields. Account {a}: {x}. Account {b}: {y}.')],
 'holdout':[
  ('logbook','The signed logbook records {x} as the role of account {a}, and {y} as the role of account {b}.'),
  ('directory','In this directory, {a} is designated {x}, whereas {b} is designated {y}.'),
  ('allocation','The allocation sheet pairs account {a} with role {x}. It pairs account {b} with role {y}.'),
  ('assignment','These assignments are binding for this file: role {x} belongs to account {a}; role {y} belongs to account {b}.')]
}
BACK={
 'discovery':[
  'The ceiling beams are inspected during the spring survey. A worker measures the gap between each support and writes the figures on a chart beside the staircase.',
  'Glass jars on the workbench contain pebbles collected from the garden. Their labels indicate the date, the weather and the approximate depth of the soil at each location.',
  'A small handbook explains how the paper trays fit inside the cabinet. The lower drawer holds spare dividers while the upper drawer contains clean protective sheets.',
  'The building has a sheltered entrance facing the courtyard. During heavy rain the caretaker places a rubber mat on the steps and checks the gutter for fallen leaves.'],
 'holdout':[
  'A brass ruler hangs next to the drawing board in the west room. The scale is checked against a reference strip before the board is used for any new measurements.',
  'The wooden crates are stacked beside an unused doorway. A separate sheet describes their dimensions, the thickness of the lining and the materials used to protect the corners.',
  'The footpath curves around a shallow pond behind the building. Its gravel surface is leveled each autumn and the nearby plants are inspected when the seasons change.',
  'Visitors can consult a printed guide to the ceiling paintings. The guide describes their colors and composition and gives the dates when each section was last cleaned.']}

def materials(tok,split,families):
    rng=random.Random(D.SEED+(split=='holdout'));codes=(('H','J'),('N','P')) if split=='discovery' else (('S','V'),('W','Z'))
    for family in range(families):
        frame,template=FRAMES[split][family%4]
        filler=[f'Archive note {j+1}. '+rng.choice(BACK[split]) for j in range(16)]
        rows=[]
        for identity,role,late,mapping in product(range(2),repeat=4):
            a,b=codes[identity];x,y=('APPROVER','OBSERVER') if role else ('OBSERVER','APPROVER')
            record='Role assignment. '+template.format(a=a,b=b,x=x,y=y);at=13 if late else 2
            mapping_text=('For APPROVER, output B. For OBSERVER, output A.' if mapping else 'For APPROVER, output A. For OBSERVER, output B.')
            text='\n'.join([f'Record RC{split[0]}{family:05d}. Read the role assignment in this document.',
                *filler[:at],record,*filler[at:],f'What role does account {a} hold? '+mapping_text+' Output only the corresponding letter.'])
            prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)+'Answer:'
            tokens=tok.encode(prompt,add_special_tokens=False);end=prompt.index(record)+len(record)
            evidence=len(tok.encode(prompt[:end],add_special_tokens=False))-1
            positions=[evidence,*[int(evidence+(len(tokens)-1-evidence)*q) for q in (.2,.4,.6,.8)],len(tokens)-1]
            assert 600<=len(tokens)<=1024
            rows.append({'id':f'v2-{split}-{family:04d}-{identity}{role}{late}{mapping}','family':family,'frame':frame,
                 'identity':identity,'role':role,'late':late,'mapping':mapping,'gold':mapping if role else 1-mapping,
                 'tokens':tokens,'prompt':prompt,'positions':positions})
        assert len({len(r['tokens']) for r in rows})==1,('v2 lengths',split,family)
        for identity,late,mapping in product(range(2),repeat=3):
            pair=[r for r in rows if (r['identity'],r['late'],r['mapping'])==(identity,late,mapping)]
            assert Counter(pair[0]['tokens'])==Counter(pair[1]['tokens'])
        yield {'family':family,'frame':frame,'rows':rows}

def configure():
    D.OUT=V1.with_name('mlp_condition_v2');D.RUN='MLP-CONDITIONAL-PATHS-v2';D.SEED=202611092870
    D.SOURCES=(*D.SOURCES,Path(__file__).name,'mlp_condition_stream_load.py')
    D.materials=materials

def freeze():
    old=D.READ(V1/'selection.json');assert not old['behavior_gate_passed']
    original=D.SAVE
    def save(path,data):
        if path.name=='prereg.json':
            data['prior_failure']={'run_id':'MLP-CONDITIONAL-PATHS-v1','discovery_behavior':old['discovery_behavior'],
                'selection_sha256':D.HASH(V1/'selection.json'),'holdout_unopened':not list(V1.glob('holdout-[0-9]*.npz'))}
            data['material_source']='new author-created explicit role assignment; role names replace ambiguous permission/yes-no inference; all four factors remain crossed'
        original(path,data)
    D.SAVE=save
    try:D.freeze()
    finally:D.SAVE=original

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['freeze','discovery','select','calibrate','holdout','intervene','capability','report','publish']);stage=parser.parse_args().stage
    configure()
    if stage=='freeze':freeze();return
    if stage in ('select','report'):
        import mlp_condition_analysis as A
        getattr(A,stage)(D.checked());return
    if stage=='publish':
        import mlp_condition_publish as P
        P.main();return
    import mlp_condition_stream_load as L
    import mlp_condition_run as R
    R.AutoModelForCausalLM.from_pretrained=staticmethod(L.load)
    R.main()

if __name__=='__main__':main()
