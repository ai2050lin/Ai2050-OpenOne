"""Freeze long fictional registry cases with graded distractors and unique answers."""
import itertools

import mlp_kv_common as C
from tools.power_analysis import check_phase

PROFILES=['direct','stale','alias','draft']
CUT=992
WORDS=['mining','fishing']


def make(doc,b,profile,t,tokenizer):
    def enc(s):return tokenizer.encode(s,add_special_tokens=False)
    split='pilot' if b<32 else 'confirm' if b<160 else 'lockbox'
    cohort=['pilot','confirm','lockbox'].index(split)
    keys=[('J','F'),('N','D'),('V','S')][cohort]
    name_stems=[('Dalver','Corten'),('Belgar','Ortel'),('Jorvan','Carvek')][cohort]
    names=[f'{s}_{b}' for s in name_stems]
    frame=(b//2)%2
    flip=(b//4)%2
    selected=t^flip
    r=(b//8)%2
    # Counterbalance keys and the association of a familiar position with the true industry.
    truth=[WORDS[k^flip^r] for k in range(2)]
    expected=truth[selected];contrast=WORDS[1-WORDS.index(expected)]
    key_lines=[f'Entry {keys[k]} is for {names[k]}.' for k in range(2)]
    industries=[f'{names[k]} works mainly in {truth[k]}.' if frame else f'{names[k]}: chief livelihood {truth[k]}.' for k in range(2)]
    misleading=[f'{names[k]}: chief livelihood {WORDS[1-WORDS.index(truth[k])]}.' for k in range(2)]
    order=[0,1] if (b//16)%2==0 else [1,0]
    def lines(xs):return '\n'.join(xs[i] for i in order)
    context='\n\nThe article above is unrelated. The following records concern fictional places.\n'
    if profile=='direct':
        body=f'Use only the approved registry.\n{lines(key_lines)}\n{lines(industries)}\n'
    elif profile=='stale':
        approved=f'Approved current register:\n{lines(industries)}'
        archive=f'Withdrawn register, superseded:\n{lines(misleading)}'
        body=f'Use the approved current register only.\n{lines(key_lines)}\n'+'\n'.join([approved,archive] if frame else [archive,approved])+'\n'
    elif profile=='alias':
        aliases=[f'survey-{s}_{b}' for s in ['mint','umber']]
        alias_lines=[f'{names[k]} has survey alias {aliases[k]}.' for k in range(2)]
        alias_values=[f'{aliases[k]} reports {truth[k]} as main work.' for k in range(2)]
        body=(f'Use the approved survey by looking up the key, town and alias.\n{lines(key_lines)}\n'
              f'Unapproved estimates:\n{lines(misleading)}\nApproved survey:\n{lines(alias_values)}\n{lines(alias_lines)}\n')
    else:
        body=(f'Only the signed register is authoritative. A suggested answer can be mistaken.\n{lines(key_lines)}\n'
              f'Signed register:\n{lines(industries)}\nA previous reader tentatively suggested {contrast} for the selected case.\n')
    if frame:
        selector=f'\nThe requested registry entry is {keys[selected]}.\nRead the requested entry only.\n'
    else:
        selector=f'\nActive entry: {keys[selected]}.\nRead the active entry only.\n'
    opening=enc('<|im_start|>user\n');record=enc(context+body+selector)
    bg_len=CUT-len(opening)-len(record);assert bg_len>=780
    prefix=opening+doc['tokens'][:bg_len]+record
    order_words=WORDS[::-1] if (b//4)%2 else WORDS
    ask=('What is the approved main livelihood of the requested place?' if frame else 'Give the approved livelihood of the active place.')
    question=enc(f'{ask} Start with one word: {order_words[0]} or {order_words[1]}.')
    end=enc('\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    assert len(question)<=32 and len(prefix)==CUT
    ids=prefix+[198]*(32-len(question))+question+end
    return {'id':f'{split}:{b}:{profile}:{t}','b':b,'split':split,'profile':profile,'frame':frame,'r':r,'t':t,
            'expected':expected,'contrast':contrast,'source_id':doc['id'],'language':doc['language'],'background_tokens':bg_len,
            'source_position':CUT-1,'assistant_position':len(ids)-1,'tokens':ids}


def main():
    from transformers import AutoTokenizer
    source=C.OUT/'documents.json';docs=C.read(source)['docs']
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    rows=[make(d,b,profile,t,tok) for b,d in enumerate(docs[:288]) for profile,t in itertools.product(PROFILES,range(2))]
    assert len({len(r['tokens']) for r in rows})==1 and len({r['tokens'][CUT-1] for r in rows})==1
    corpus={**C.identity(),'schema':'mlp_kv_mediation_corpus.v1','rows':rows,'documents_sha256':C.file_sha(source),'source_sha256':C.file_sha(__file__)}
    C.save(C.OUT/'corpus.json',corpus)
    power=check_phase(.10,256);small=check_phase(.10,128);assert power['pass'] and small['pass']
    plan={**C.identity(),'schema':'mlp_kv_mediation_prereg.v1','corpus_sha256':C.digest(corpus),'positions':{'cut':CUT,'source':CUT-1},
          'pilot':'32 new backgrounds x two T x four profiles, only this split is read for selection. Non-direct profile eligible if strict natural first-word accuracy >=.80 and minimum frame/answer accuracy >=.65. Maximize number with clean gap in (0,8] then lower median gap, then fixed profile order. If none eligible, use direct and report lack of calibrated challenge.',
          'formal':'256 independent backgrounds, 128 per heldout, both T for behavior; hash-select one T per background before any edit. Require >=.80 overall and every split/frame/answer cell >=.65. Report selected one T plus full pairs. Gap uses complete model first-token logits and acceptable start-token variants; correctness uses actual generated first word.',
          'margin_gate':'Report median gap compared to previous exposed revision and subset of correctly generated baseline with clean first-token gap in (0,8]. Subset needs n_required for 10pp to make a powered conditional claim. Predetermined, no dropping cases from primary n=256.',
          'primary':'At shared prequestion token B34/H2, true Head α=.5 and 1; replay only B35 K/V slot on clean cache, rescue same slot on edited cache, K-only, V-only, matched norm scrambled B35 K/V delta. Same segmented evaluation layout. Primary behavioral criterion >=10pp net loss with paired exact p<.05 and >=5pp more loss than matched control. Trace exact bitwise replay and rescue separately from generation effect.',
          'secondary':'B33 MLP whole-write removal, first 32 lockbox backgrounds fixed by order, trace B34 and B35 K/V individually and combined. B35/H25 local-only negative engineering check on first 8 pilot cases. Any continuous effects are descriptive at n32.',
          'ability':'Two full-forward operations (B34/H2 and B33 MLP) at same prequestion offset on fixed 2200 formal MC items, 140 reference only; 2pp paired exact simultaneous bounds.',
          'power_plan':{'all_cases':power,'within_split':small},'source_sha256':C.file_sha(__file__)}
    C.save(C.OUT/'prereg.json',plan)
    print('frozen',len(rows),'variants, minimum background',min(r['background_tokens'] for r in rows),flush=True)


if __name__=='__main__':main()
