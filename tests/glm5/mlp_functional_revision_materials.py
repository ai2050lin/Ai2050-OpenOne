"""Independent format revision. Frozen v1 writers and directions are carried forward."""
import itertools

import mlp_functional_common as C
from mlp_functional_materials import FRAMES, WORDS
from tools.power_analysis import check_phase


def main():
    from transformers import AutoTokenizer
    folder=C.OUT/'revision';source=folder/'documents.json';docs=C.read(source)['docs']
    path=C.OUT/'path-analysis.json'
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    def enc(s):return tok.encode(s,add_special_tokens=False)
    opening=enc('<|im_start|>user\n');closing=enc('\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    intro=enc('\n\nThe preceding article is background. For this question use only these fictional registry entries.\n')
    keys=[('A','F'),('E','J'),('N','S')]
    rows=[]
    for b,doc in enumerate(docs):
        si=0 if b<32 else 1 if b<192 else 2
        split=['pilot','confirm','lockbox'][si];local=b-[0,32,192][si]
        frame=si*2+(local//2)%2;flip=(local//4)%2
        record,select=FRAMES[frame];pair=keys[si]
        labels=[f'Halteron_{b}_Aster',f'Merivox_{b}_Beryl']
        places=['The town is inland, away from the sea.','The town is coastal, beside the sea.']
        work=['Mining is its main livelihood.','Fishing is its main livelihood.']
        records={(k,r):enc('\n'+record.format(key=pair[k],name=labels[k],where=places[k^flip],work=work[k^flip^r])+'\n')
                 for k,r in itertools.product(range(2),repeat=2)}
        for k in range(2):
            size=max(len(records[k,r]) for r in range(2))
            for r in range(2):records[k,r]+=[198]*(size-len(records[k,r]))
        selectors=[enc('\n'+select.format(key=key)+'\n') for key in pair]
        diffs=[i for i,(a,z) in enumerate(zip(*selectors)) if a!=z]
        assert len(selectors[0])==len(selectors[1]) and len(diffs)==1 and selectors[0][-1]==selectors[1][-1]
        # End with the instruction itself. No dangling answer heading for the model to echo.
        questions=[]
        for q in [0,1]:
            prompt=['What is the location category of the selected town?','What is the main livelihood of the selected town?'][q]
            if frame%2:prompt=['Give the location class for the selected town.','Give the livelihood for the selected town.'][q]
            choices=list(WORDS[q]);choices=choices[::-1] if (local//8)%2 else choices
            questions.append(enc(f'\n{prompt} Start your response with one word: {choices[0]} or {choices[1]}.'))
        qlen=max(map(len,questions));p=1024-qlen
        bglen=p-len(opening)-len(intro)-sum(len(records[k,0]) for k in [0,1])-len(selectors[0]);assert bglen>=850
        prefix=opening+doc['tokens'][:bglen]+intro
        for r,t,q in itertools.product(range(2),repeat=3):
            before=prefix+records[0,r]+records[1,r]+selectors[t^flip]
            ids=before+[198]*(qlen-len(questions[q]))+questions[q]+closing
            assert len(before)==p and len(ids)==1024+len(closing)
            expected=WORDS[q][t if q==0 else t^r]
            rows.append({'id':f'rev:{b}:{r}:{t}:{q}','b':b,'r':r,'t':t,'q':q,'i':t^r,'split':split,'frame':frame,
                         'source_id':doc['id'],'language':doc['language'],'background_tokens':bglen,'tokens':ids,
                         'prequestion':p-1,'before_selector':p-len(selectors[0])+diffs[0]-1,'assistant':len(ids)-1,
                         'expected':expected,'contrast':next(x for x in WORDS[q] if x!=expected)})
    cohort={**C.identity(),'schema':'functional_revision_materials.v2','rows':rows,'documents_sha256':C.file_sha(source),'script_sha256':C.file_sha(__file__)}
    C.save(folder/'inputs.json',cohort)
    power=check_phase(.10,160);assert power['pass']
    plan={**C.identity(),'schema':'functional_revision_prereg.v2','corpus_sha256':C.digest(cohort),'frozen_v1_paths_sha256':C.file_sha(path),
          'reason':'v1 65/80 in confirm R1/T1 industry frame 2, failures begin Answer. Remove answer heading and retest on fresh docs, keys and entities; same strict first-word scoring and 90% heldout threshold.',
          'behavior':'pilot 256, 16 cells n16 min14; formal 2560, 32 cells n80 min72. Greedy four generated tokens. No parser revision.',
          'replication':'v1 discovery directions and selected writers frozen. Confirm and lockbox each 160 independent backgrounds. Both R components positive by paired t interval, family .05/(2 attributes*4 writers*2 assignments*2 holdouts).',
          'causal':'V2 primary industry group fixed from v1 discovery because H2 B34 cleared both v1 writer intervals before global behavior gating. Evaluate n160 industry questions, location n160 secondary. Original prequestion token only, full downstream recompute. Alpha 0/.5/1; also Head only, MLP only, B34/H2 only and equal-norm signed-permutation perturbation per same native writer and position.',
          'primary_gate':'joint_1 accuracy loss >=.10 and paired two-sided exact p<.05, plus >=.05 greater loss than matched perturbation; requires revised behavior gate. Residual and logit changes secondary.',
          'capability':'repeat both functional SAE and v2 industry writer group on formal 2200 plus 140 reference; exact paired two-operation CI and 2pp margin. Upstream applies same offset from end as modal revised pilot question length.',
          'power_plan':power,'method_sha256':C.file_sha(__file__)}
    C.save(folder/'prereg.json',plan)
    print('revision frozen',len(rows),'queries, min background',min(r['background_tokens'] for r in rows),flush=True)


if __name__=='__main__':main()
