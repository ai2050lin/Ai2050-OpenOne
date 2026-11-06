"""Construct fresh fictional registries with exact spans for candidate source tokens."""
import mlp_read_route_common as C
from tools.power_analysis import check_phase

ROLES=['approved','other','draft','selector']
WORDS=['mining','fishing']


def material(doc,b,tok):
    split='discovery' if b<C.N_DISCOVERY else 'confirm' if b%4<2 else 'lockbox'
    cohort=['discovery','confirm','lockbox'].index(split)
    # Separate names and keys across partitions. Deterministic label, order, and framing counterbalance.
    names=[f'{name}_{b}' for name in [('Torlan','Redvar'),('Invar','Malken'),('Trevil','Corben')][cohort]]
    keys=[('K','R'),('H','T'),('M','G')][cohort]
    t=int(C.digest([C.SEED,'selection',b])[:8],16)%2;orientation=(b//8)%2
    frame=(b//4)%2;order=[0,1] if (b//16)%2 else [1,0]
    truth=[WORDS[k^orientation] for k in range(2)];expected=truth[t];contrast=truth[1-t]
    text='\n\nThe preceding article is unrelated background. Only the signed list below describes the fictional places.\n'
    spans={}
    def push(fragment):
        nonlocal text
        text+=fragment
    def mark(role,fragment):
        spans[role]=(len(text),len(text)+len(fragment));push(fragment)
    if frame:
        push('Register keys:\n')
        for k in order:push(f'Identifier {keys[k]} names the town {names[k]}.\n')
        push('The signed livelihood ledger is authoritative.\n')
        for k in order:
            push(f'{names[k]} works chiefly in');mark('approved' if k==t else 'other',' '+truth[k]);push('.\n')
        push('A previous reader provisionally guessed');mark('draft',' '+contrast);push(' for the requested town.\nThe requested identifier is')
        mark('selector',' '+keys[t]);push('.\nRead the approved entry only.\n')
    else:
        push('Use the approved registry. A tentative answer may be incorrect.\n')
        for k in order:push(f'Entry {keys[k]} corresponds to {names[k]}.\n')
        push('Signed registry of main work:\n')
        for k in order:
            push(f'{names[k]}:');mark('approved' if k==t else 'other',' '+truth[k]);push('.\n')
        push('A draft reply suggests');mark('draft',' '+contrast);push(' for this request.\nActive entry:')
        mark('selector',' '+keys[t]);push('.\nRead the active entry only.\n')
    info=tok(text,add_special_tokens=False,return_offsets_mapping=True)
    def enc(s):return tok.encode(s,add_special_tokens=False)
    opening=enc('<|im_start|>user\n');bg=C.CUT-len(opening)-len(info['input_ids']);assert bg>=800
    start=C.CUT-len(info['input_ids'])
    positions={key:[start+i for i,(a,z) in enumerate(info['offset_mapping']) if a<end and z>lo] for key,(lo,end) in spans.items()}
    assert all(positions[r] for r in ROLES)
    words=WORDS if (b//2)%2 else list(reversed(WORDS))
    question=enc(f'Give the approved livelihood of the requested town. Start with one word: {words[0]} or {words[1]}.')
    ending=enc('\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n');assert len(question)<=32
    ids=opening+doc['tokens'][:bg]+info['input_ids']+[198]*(32-len(question))+question+ending
    assert len(ids)==1034
    anchors={role:[{'position':i,'token_id':ids[i],'text':tok.decode([ids[i]])} for i in ps] for role,ps in positions.items()}
    assert max(p['position'] for v in anchors.values() for p in v)<C.CUT-1
    return {'id':f'{split}:{b}:{keys[t]}:{frame}','b':b,'split':split,'frame':frame,'expected':expected,'contrast':contrast,
            'language':doc['language'],'source_id':doc['id'],'background_tokens':bg,'anchors':anchors,'tokens':ids}


def main():
    from transformers import AutoTokenizer
    source=C.OUT/'documents.json.gz';docs=C.read_gz(source)['docs']
    tok=AutoTokenizer.from_pretrained(C.MODEL,local_files_only=True)
    rows=[material(d,b,tok) for b,d in enumerate(docs)]
    corpus=C.OUT/'inputs.json.gz'
    C.write_gz(corpus,{**C.identity(),'schema':'mlp_early_token_materials.v1','rows':rows,
                      'documents_sha256':C.file_sha(source),'generator_sha256':C.file_sha(__file__)})
    power=check_phase(.10,128);assert power['pass']
    plan={**C.identity(),'schema':'mlp_early_token_prereg.v1','inputs_sha256':C.file_sha(corpus),
          'discovery':'64 independent backgrounds only. Reconstruct actual B34/B35 answer-query read weights from native BF16 Q (post RMS/RoPE), K/V, with FP32 math and check against native SDPA Head output. Rank two early roles by discovery mean projected value-read norm across both layers and Head outputs, choose one span per role. Lock roles and rank two native Heads and one MLP from B33/B34 by discovery signed removal derivative on correct-vs-contrast initial-token logit margin. Freeze chosen writers and MC offsets before heldout screening.',
          'enrollment':'1472 additional unique backgrounds, one preassigned case each. Screen all clean four-token natural generations in batch layout 4. Eligible if first word is the expected value and clean first-token contrast gap in (0,8]. Hash-rank eligible backgrounds within confirm and lockbox, keep up to 64 per split. Pooled primary gate requires planning n; report both split results separately.',
          'layout':'Always evaluate four samples per batch including non-enrolled fillers when needed. If a clean word/gap fails to reproduce screening, explicitly re-evaluate all branches for that case in its original screening batch, preserve both versions, never change the screen or efficacy thresholds.',
          'causal':'Only frozen native writers and semantic anchor spans. True α=.5/1 joint, Head-only and MLP-only, matched signed-permutation writer (exact Euclidean source norm before BF16 addition). Compare actual full edited prefix. Exact all-changed-range cache replay and rescue, source-only and subsequent-prefix-only replay. Inspect changes by layer, GQA group and source position. Same fixed segmented layout.',
          'behavioral_gate':'Pooled clean-correct low-margin cohort, >=10pp loss, two-sided paired exact p<.05, and >=5pp larger loss than the matched control. Accuracy and actual first-word changes counted separately. Preserve same-answers and wrong-answer cases; if sample size < planned minimum, scope to exploratory.',
          'read_verify':'At the later answer-query do real V zeroing in nominated slots with K held fixed, record post-SDPA output changes and agreement with -sum(a*V). This gives direct read evidence distinct from QK attention ranking.',
          'ability':'Repeat α=1 joint and matched writer on fixed 2200 formal MC items + 140 reference using discovery-median source offsets. Two-operation exact simultaneous paired loss/gain bounds, 2pp NI.',
          'power_plan':power,'source_sha256':C.file_sha(__file__)}
    C.save(C.OUT/'prereg.json',plan)
    print('frozen',len(rows),'cases, min article tokens',min(r['background_tokens'] for r in rows),flush=True)


if __name__=='__main__':main()
