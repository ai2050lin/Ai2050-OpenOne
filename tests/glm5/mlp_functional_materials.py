"""Freeze a factorial long-prefix test with fresh entities, backgrounds and frames."""
import itertools

import numpy as np
import mlp_functional_common as C
from tools.power_analysis import check_phase

COUNTS={'discovery':128,'confirm':160,'lockbox':160}
WORDS=[('inland','coastal'),('mining','fishing')]
FRAMES=[
    ('Registry entry {key} for {name}. {where} {work}', 'The active entry is {key}.\nRead the active entry only.'),
    ('{key} is the registry key for {name}. {work} {where}', 'The request refers to entry {key}.\nRead the active entry only.'),
    ('Under label {key} the census describes {name}. {where} {work}', 'For this request use label {key}.\nRead the active entry only.'),
    ('The index card headed {key} concerns {name}. {work} {where}', 'The selected index card is {key}.\nRead the active entry only.'),
    ('In the annex, settlement {name} is filed under {key}. {where} {work}', 'Consult the annex item marked {key}.\nRead the active entry only.'),
    ('File {key} names the settlement {name}. {work} {where}', 'The referenced file number is {key}.\nRead the active entry only.'),
]
KEYS=[('K','M'),('D','G'),('P','V')]


def main():
    from transformers import AutoTokenizer
    source=C.OUT/'fresh-documents.json'; docs=C.read(source)['controlled']
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    def enc(text):return tok.encode(text,add_special_tokens=False)
    opening=enc('<|im_start|>user\n')
    ending=enc('\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    intro=enc('\n\nThe article above is unrelated background. The entries below describe two invented settlements. Answer using only these entries.\n')
    template_questions=[('State the geographic category.', 'State the main livelihood.'),
                        ('What kind of location is specified?', 'What industry supports the residents?'),
                        ('Report the indicated location class.', 'Report the indicated economic activity.')]
    rows=[]; b=0
    for split_index,(split,count) in enumerate(COUNTS.items()):
        for local in range(count):
            doc=docs[b];f=(local//2)%2; key_flip=(local//4)%2;name_flip=(local//8)%2
            assert (doc['language']=='zh')==bool(local%2)
            frame=split_index*2+f;record,select=FRAMES[frame];keys=KEYS[split_index]
            assert len({len(enc(' '+key)) for key in keys})==1
            first,second=(f'Vellor_{b}_Amber',f'Draven_{b}_Cobalt') if not name_flip else (f'Draven_{b}_Cobalt',f'Vellor_{b}_Amber')
            names=[first,second];where=['The settlement is inland, far from the sea.','The settlement is coastal, on the shore.']
            work=['Mining is its main livelihood.','Fishing is its main livelihood.']
            if f:where=['Its location is inland in the interior.','Its location is coastal by the sea.'];work=['The residents mainly work in mining.','The residents mainly work in fishing.']
            records={}
            for key_idx,r in itertools.product(range(2),repeat=2):
                t=key_idx^key_flip
                records[key_idx,r]=enc('\n'+record.format(key=keys[key_idx],name=names[key_idx],where=where[t],work=work[t^r])+'\n')
            for key_idx in range(2):
                size=max(len(records[key_idx,r]) for r in range(2))
                for r in range(2):records[key_idx,r]+=[198]*(size-len(records[key_idx,r]))
            selectors=[enc('\n'+select.format(key=key)+'\n') for key in keys]
            assert len(selectors[0])==len(selectors[1]) and selectors[0][-1]==selectors[1][-1]
            selector_diff=[i for i,(a,z) in enumerate(zip(*selectors)) if a!=z]
            assert len(selector_diff)==1,(frame,selectors)
            questions=[enc('\n'+template_questions[split_index][q]+' Return exactly one word: '+
                           ' or '.join(reversed(WORDS[q]) if (local//16)%2 else WORDS[q])+'.\nAnswer:') for q in range(2)]
            qlen=max(map(len,questions))
            plen=1024-qlen
            bg_len=plen-len(opening)-len(intro)-len(records[0,0])-len(records[1,0])-len(selectors[0])
            assert bg_len>=850
            bg=opening+doc['tokens'][:bg_len]+intro
            for r,t,q in itertools.product(range(2),repeat=3):
                selected=t^key_flip
                before=bg+records[0,r]+records[1,r]+selectors[selected]
                ids=before+[198]*(qlen-len(questions[q]))+questions[q]+ending
                assert len(before)==plen and len(ids)==1024+len(ending)
                expected=WORDS[q][t if q==0 else t^r]
                rows.append({'id':f'{b}:{r}:{t}:{q}','b':b,'r':r,'t':t,'q':q,'i':t^r,'split':split,'frame':frame,
                             'language':doc['language'],'selected_key':keys[selected],'source_id':doc['id'],'background_tokens':bg_len,
                             'before_selector':plen-len(selectors[selected])+selector_diff[0]-1,'prequestion':plen-1,
                             'assistant':len(ids)-1,'expected':expected,'contrast':next(w for w in WORDS[q] if w!=expected),'tokens':ids})
            b+=1
    power=check_phase(.10,COUNTS['lockbox'],alpha=.05)
    assert power['pass']
    corpus={**C.identity(),'schema':'functional_upstream_materials.v1','rows':rows,'source_sha256':C.file_sha(source),
            'generator_sha256':C.file_sha(__file__),'scope':'unseen document, fictional entity, key vocabulary and frame combination; attribute values are shared across splits'}
    C.save(C.OUT/'controlled-inputs.json',corpus)
    prereg={**C.identity(),'schema':'functional_upstream_prereg.v1','corpus_sha256':C.digest(corpus),'splits':COUNTS,
             'design':'T=location, R=assignment, I=T xor R, Q=query. For each background all 8 combinations. Q suffix follows the shared prequestion readout.',
             'gate':'Discovery pilot first 32 backgrounds, minimum 14/16 in every R/T/Q/frame cell. Each heldout cell minimum 90% generated first-word accuracy, 80 queries per cell.',
             'observation':'Capture all native query-head outputs, every MLP output up to B34, embedding and residual boundaries at the shared prequestion token; paired 4-variant per-background contrasts for T/I and frozen discovery decoder; Q suffix and pre-selector controls.',
             'selection':'T primary, I secondary. For each variable fit unit direction using mean discovery oriented pairs in native incoming B35 residual. Rank writer classes by mean contribution along this direction. Freeze top two positive mean Head writes and top two positive MLP writes for each variable (or fewer if nonpositive). Test both assignment orientations and both holdouts.',
             'causal':'T discovery group at prequestion token only, whole native write removal: clean, joint .5/1, attention-only, MLP-only, matched random joint. Full original downstream forward. One hash-frozen R,T pair per unseen lockbox background, both questions. Primary outcome location generated accuracy loss ≥.10, paired exact p<.05, and target-minus-random loss ≥.05; random must match total discovery RMS within ±25% in each writer class. Report I, frozen residual margin, next-token KL and log odds as secondary.',
             'capability':'same B35 functional SAE replacement at assistant query and same upstream writes at an equal offset before assistant query, 2200 exposed formal monitor +140 reference, paired exact bounds, 2pp margin, two-op family correction; native CUDA only.',
             'power_plan':power,'artifact_scope':'candidate attribution, unseen generalization and limited causal edge; no merge between scientific evidence levels'}
    C.save(C.OUT/'controlled-prereg.json',prereg)
    print('Frozen',len(rows),'queries; minimum background',min(r['background_tokens'] for r in rows),'tokens',flush=True)


if __name__=='__main__':
    main()
