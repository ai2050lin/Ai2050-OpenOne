"""One-token factorial prompts, plus a heldout dispatch task with a separate key join."""
import itertools

import mlp_factor_steer_common as C
from tools.power_analysis import check_phase

WORDS=['mining','fishing']
WRITE_ROLES=['attribute','selector','tail_a','tail_b']


def make(doc,b,split,e,s,d,tok):
    def enc(v):return tok.encode(v,add_special_tokens=False)
    task='dispatch' if split=='dispatch' else 'lookup'
    frame=(b//2)%2;strength=(b//4)%3;other=(b//16)%2
    names=['Altor','Branth'] if split in ['fit','select'] else ['Nerval','Soncar'] if task=='lookup' else ['Pelmar','Drovin']
    places=[f'{v}_{b}' for v in names];keys=['K','D'] if split in ['fit','select'] else ['V','J'] if task=='lookup' else ['H','Q']
    text='\n\nThe article is unrelated. The following records describe fictional places.\n'
    spans={}
    def push(v):
        nonlocal text
        text+=v
    def mark(key,v):
        spans[key]=(len(text),len(text)+len(v));push(v)
    order=[1,0] if (b//8)%2 else [0,1]
    if task=='lookup':
        push('Use the signed register as the authority. Draft suggestions may be mistaken.\n')
        for k in order:push(f'Entry {keys[k]} identifies {places[k]}.\n')
        push('Signed livelihood register:\n')
        for k in order:
            push(f'{places[k]} works primarily in' if frame else f'{places[k]}: primary livelihood')
            if k==0:mark('attribute',' '+WORDS[e])
            else:push(' '+WORDS[other])
            push('.\n')
    else:
        permits=[f'Amber-{b}',f'Cobalt-{b}']
        push('Dispatch rule: send the crew that matches the signed permit classification. A draft is advisory.\n')
        for k in order:push(f'Call sign {keys[k]} opens dispatch case {places[k]}.\n')
        for k in order:push(f'Dispatch case {places[k]} uses permit {permits[k]}.\n')
        push('Signed permit classifications:\n')
        for k in order:
            push(f'Permit {permits[k]} authorizes the' if frame else f'Authorized crew on {permits[k]}:')
            if k==0:mark('attribute',' '+WORDS[e])
            else:push(' '+WORDS[other])
            push(' crew.\n')
    draft=['A reviewer tentatively suggests','The prepared reply was','The most plausible draft answer was'][strength]
    push(draft);mark('draft',' '+WORDS[d]);push('.\n')
    push('The requested entry is' if frame else 'Current key:');mark('selector',' '+keys[s]);push('.\nUse the signed evidence.\n')
    encoded=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    base=C.CUT-len(encoded['input_ids']);opening=enc('<|im_start|>user\n');bg=base-len(opening);assert bg>=780
    prefix=opening+doc['tokens'][:bg]+encoded['input_ids'];assert len(prefix)==C.CUT
    anchors={}
    for key,(lo,hi) in spans.items():
        positions=[base+i for i,(a,z) in enumerate(encoded['offset_mapping']) if a<hi and z>lo]
        assert len(positions)==1,(key,positions,text[lo:hi]);anchors[key]=positions[0]
    anchors.update(tail_a=C.CUT-2,tail_b=C.CUT-1)
    ask='Which crew should be dispatched?' if task=='dispatch' else 'What is the signed livelihood of the requested place?'
    words=list(reversed(WORDS)) if (b//2)%2 else WORDS
    q=enc(f'{ask} Start with one word: {words[0]} or {words[1]}.');assert len(q)<=32
    ending=enc('\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
    ids=prefix+[198]*(32-len(q))+q+ending;assert len(ids)==1034
    expected=e if s==0 else other
    return {'id':f'{split}:{b}:{e}{s}{d}','b':b,'split':split,'task':task,'frame':frame,'strength':strength,
            'evidence':e,'selector':s,'draft':d,'other':other,'expected':WORDS[expected],'contrast':WORDS[1-expected],
            'language':doc['language'],'source_id':doc['id'],'background_tokens':bg,'anchors':anchors,
            'anchor_tokens':{k:{'position':i,'id':ids[i],'text':tok.decode([ids[i]])} for k,i in anchors.items()},'tokens':ids}


def main():
    from transformers import AutoTokenizer
    path=C.OUT/'documents.json.gz';docs=C.read_gz(path)['docs'];tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    fit=[];selection=[];held=[];audits=[]
    for b,doc in enumerate(docs):
        if b<C.FIT:
            cell={(e,s,d):make(doc,b,'fit',e,s,d,tok) for e,s,d in itertools.product(range(2),repeat=3)}
            for key,row in cell.items():
                for axis,label in enumerate(['attribute','selector','draft']):
                    counterpart=list(key);counterpart[axis]^=1
                    peer=cell[tuple(counterpart)];diff=[i for i,(a,z) in enumerate(zip(row['tokens'],peer['tokens'])) if a!=z]
                    assert diff==[row['anchors'][label]] and row['anchors']==peer['anchors'],(key,label,diff,row['anchors'])
            fit.extend(cell.values());audits.append({'b':b,'pairs':12,'all_one_token':True})
        else:
            e=int(C.digest([C.SEED,'evidence',b])[:8],16)%2
            split='select' if b<C.FIT+C.SELECT else 'lookup' if b<C.FIT+C.SELECT+C.PER_TASK else 'dispatch'
            row=make(doc,b,split,e,0,1-e,tok)
            (selection if split=='select' else held).append(row)
    corpus=C.OUT/'inputs.json.gz'
    C.write_gz(corpus,{**C.identity(),'schema':'factor_steer_inputs.v1','fit':fit,'selection':selection,'heldout':held,
                      'audits':audits,'documents_sha256':C.file_sha(path),'source_sha256':C.file_sha(__file__)})
    power=check_phase(.10,104,alpha=.025);assert power['pass']
    plan={**C.identity(),'schema':'factor_steer_prereg.v1','inputs_sha256':C.file_sha(corpus),
          'factorial':'64 backgrounds × evidence E, choice S, draft D (2×2×2). Each pair changes exactly one token. Outcome orientation fishing minus mining. Compare E with S=A vs B, S when E differs vs equals B, D at fixed E/S. Background clustered.',
          'fit':'Native Head/MLP prefix outputs at evidence token, key token and two common tail tokens. Fit mean E paired differences from S=A and project out the nuisance span spanned by D and S-with-same-answer differences per native writer across roles.',
          'selection':'64 disjoint development backgrounds, originally with wrong draft and S=A. Rank components using actual answer-gap gradient dotted with fixed E direction. Test a compact and broad mixed Head/MLP group at α in {1,4,8} on clean-correct development examples, against same-vector strict norm control. Maximize targeted first-word improvement over control, then direction-sensitive gap effect, then smaller group and dose. Freeze before accessing heldout outcomes. Report when no discovery configuration passes behavioral effect.',
          'holdout':'Up to 1536 backgrounds per task. Enroll first 104 clean-correct cases per task with gap in (0,8] from deterministic input order, or use all eligible if short. Fixed batches of four, pad evaluation if required; use original screen batch when exact clean word/gap fails. New dispatch workflow has a separate call-sign → case → permit → crew join, uses the same answer vocabulary, and receives the same frozen writer group and dose.',
          'interventions':'Directed addition toward opposite answer, half dose, head-only, MLP-only, strict equal-norm signed-permutation direction at identical native coordinates. Full cache change replay and clean restoration, plus source-only. Preserve actual four-token greedy output and all failures.',
          'confirmatory_gate':'Per task: n >= planning minimum; targeted alternate first-word gain >=10pp, two-sided paired p<.025 vs clean and >=5pp above norm control. Scope is independent baseline-correct low-margin cases. Compare task replications separately, no pool substitution.',
          'capability':'Same frozen intervention and strict matched control on fixed 2200 MC items + 140 references using discovery-median offsets and fixed hash orientation per item. Two-operation exact simultaneous NI bounds, 2pp.',
          'power_plan':power,'source_sha256':C.file_sha(__file__)}
    C.save(C.OUT/'prereg.json',plan)
    print('frozen',len(fit),len(selection),len(held),'minimum background',min(r['background_tokens'] for r in fit+selection+held),flush=True)


if __name__=='__main__':main()
