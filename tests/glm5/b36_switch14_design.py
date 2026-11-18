"""Paired stopping instructions: identical visible program, first2 versus first3 updates."""
import itertools,operator,re,sys
import b36_switch14 as M
S=M.S
PEAK=1000*2**20
ARMS=['first2','first3']
SITES=['after_first','after_second','report']


def families():
    old,paths=M.T.prior_materials();paths=paths+[M.PRIOR/'plan.json']
    old+=S.read(M.PRIOR/'plan.json')['families']
    recipes={S.digest(g['ops'][:d]) for g in old for d in [2,3]}
    prefixes={S.digest([g['start'],g['ops'][:1]]) for g in old}
    ops=[(name,k) for name in ['add','sub'] for k in range(1,10)]+[('mul',2),('mul',3)]
    out=[]
    for program in sorted(itertools.product(ops,repeat=3),key=lambda p:S.digest([S.SEED,'switch',p])):
        if S.digest(program) in recipes or S.digest(program[:2]) in recipes:continue
        if len({op for op,k in program})<2:continue
        start=30+int(S.digest([S.SEED,program])[:8],16)%20
        key=S.digest([start,program[:1]])
        if key in prefixes:continue
        x=start;truth=[]
        for op,k in program:x={'add':operator.add,'sub':operator.sub,'mul':operator.mul}[op](x,k);truth.append(x)
        if min(truth)<10 or max(truth)>99 or len(set(truth))<3:continue
        out.append(dict(id='s'+S.digest([start,program])[:15],start=start,ops=program,truth=truth))
        prefixes.add(key);recipes.add(S.digest(program));recipes.add(S.digest(program[:2]))
        if len(out)==96:break
    assert len(out)==96,len(out)
    return out,paths


def row_for(g,depth,tok):
    lines=[f'x = {g["start"]}']+[f'x {dict(add="+=",sub="-=",mul="*=")[op]} {k}' for op,k in g['ops']]
    user=f'Execute only the first {depth} update statements below, in order. Ignore later update statements. The initial assignment is not an update.\n```python\n'+'\n'.join(lines)+'\n```\nWrite exactly two lines: "Steps: " then the values after the executed updates, separated by commas (exclude the initial assignment); "Value: " then the final numeric value. Stop immediately after these two lines. No explanation or code fences.'
    prompt=tok.apply_chat_template([dict(role='user',content=user)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    return dict(g,id=g['id']+f':first{depth}',family=g['id'],arm=f'first{depth}',visible_ops=g['ops'],ops=g['ops'][:depth],visible_truth=g['truth'],truth=g['truth'][:depth],depth=depth,fields=['Steps','Value'],slot=None,code=None,codes=[],values=[],prompt=prompt,tokens=tok.encode(prompt,add_special_tokens=False))


def candidate_sites(prefix):
    """Only causal prefix is examined; selection of the delimiter itself is retrospective."""
    found=[]
    m=re.fullmatch(r'Steps:[ \t]*(\d+(?:[ \t]*,[ \t]*\d+)*)([ \t]*)',prefix)
    if m:
        count=len(re.findall(r'\d+',m[1]))
        if count in [1,2]:found.append(SITES[count-1])
    if re.fullmatch(r'Steps:[^\r\n]*[\r\n]+Value:[ \t]*',prefix):found.append('report')
    return found


def token_class(text):
    if re.match(r'^[ \t]*,',text):return 'comma'
    if re.match(r'^[ \t]*[\r\n]',text):return 'newline'
    m=re.match(r'^[ \t]*([0-9])',text)
    return 'digit'+m[1] if m else 'other'


def eligible(site,next_text):
    cls=token_class(next_text)
    return cls.startswith('digit') if site=='report' else cls in ['comma','newline']


def target_class(row,site):
    return 'digit'+str(row['truth'][-1])[0] if site=='report' else 'newline' if site=='after_second' and row['depth']==2 else 'comma'


def prepare():
    S.configure(PEAK);assert S.read(S.OUT/'calibration.json')['passed'],'Same-cache replay gate failed'
    assert not (S.OUT/'plan.json').exists()
    from transformers import AutoTokenizer
    from tools.power_analysis import check_phase
    tok=AutoTokenizer.from_pretrained(S.MODEL,local_files_only=True);ff,prior=families();rows=[row_for(g,d,tok) for d in [2,3] for g in ff]
    assert all(len(rows[i]['tokens'])==len(rows[i+96]['tokens']) and sum(a!=b for a,b in zip(rows[i]['tokens'],rows[i+96]['tokens']))==1 for i in range(96))
    classes={c:[] for c in ['comma','newline']+[f'digit{i}' for i in range(10)]}
    for i in range(len(tok)):
        cls=token_class(tok.decode([i],skip_special_tokens=False))
        if cls in classes:classes[cls].append(i)
    assert all(classes.values())
    power=check_phase(.2,96,p_d=.3,alpha=.05/6);assert power['pass']
    source=M.sources(['b36_switch14.py','b36_switch14_design.py','b36_switch14_engine.py','b36_switch14_behavior.py','b36_switch14_paths.py','b36_switch14_analyze.py','test_b36_switch14.py','b36_transition14.py','b36_semantic14.py','recursive_branch14.py','qwen14_mlp_atlas.py','source_binding_resources.py'])
    S.save('plan.json',dict(families=ff,rows=rows,arms=ARMS,sites=SITES,classes=classes,n_families=96,source_hashes=source,
        prior_files={str(p.relative_to(S.ROOT)):S.file_sha(p) for p in prior},calibration_sha256=S.file_sha(S.OUT/'calibration.json'),model_files=S.read(S.OUT/'calibration-plan.json')['model_files'],versions=M.versions(),
        generation=dict(batch=96,max_new_tokens=48,dtype='originalBF16',thinking=False,do_sample=False),
        gate='Both instruction cells >=88/96 joint successes; all192 source outputs must contain all3 declared observable sites. Missing or failed cells stop native confirmation, with failures retained.',
        min_joint=88,unit_ids=S.group(),matched_unit_ids=S.matched(),block_index=36,candidate_source=dict(path=str((S.PREVIOUS/'frozen-selection.json').relative_to(S.ROOT)),sha256=S.file_sha(S.PREVIOUS/'frozen-selection.json')),
        design='Each family has the same initial assignment and three visible updates. Only one instruction digit differs, execute first2 versus first3. Equal token count and same positions verified per pair. Output fields identical, no option table. The third visible update is part of family identity even when not executed. All three correct intermediate values are two-digit and distinct. Initial values30..49 differ from earlier10..29, so cross-Phase differences are not causal comparisons.',
        novelty='All visible three-operation recipes, executed first-two recipes, and initial+first-operation prefixes are disjoint from registered prior materials; no pretraining novelty claim. No discovery fit or layer selection on these confirmation families.',
        sites_definition='after_first and after_second: actual natural prefix just before the next comma/newline token after that printed integer; report: actual prefix just before the first digit-bearing Value token. Sites are located retrospectively using the baseline next token, which is never included in branch inputs. Whitespace may precede a site. These are conditional natural-path decisions, not online stopping detectors. Incorrect printed values are retained.',
        native_conditions=['baseline','noop','remove','matched','double','hold_attention','hold_mlp','hold_both'],
        native_scope='One-step branches from the original live pre-B36 state and unchanged prior KV for the same batch. Remove/double fixed32 post-gate*up activations only at current position; matched32 control. Restore natural attention/MLP outputs at current position in B37..39. Native no-op logits must be bitwise equal at every site and rerun generation must equal the behavior run for all192, otherwise no native confirmation. Prior KV is cloned, never edited in source generation. No free generation of edited histories.',
        primary=[['baseline','remove'],['baseline','matched'],['baseline','double'],['remove','hold_attention'],['remove','hold_mlp'],['remove','hold_both']],
        primary_metric='Per-family joint correctness of full-vocabulary top1 token class across all6 sites (3sites×2instruction views). Exact paired McNemar, Holm6. Raw token changes, target log-probability, KL, entropy and per-site class scores are descriptive and retained, including ceiling effects. Matching control is not a proof of selective semantic necessity.',
        power=power,fields='All41 current-position residual boundaries at each site plus B36 fullG/V/A and64 down columns. Original BF16; native order, noTopK/PCA. Full-vocabulary BF16 logits for every branch, with float64 analysis.',
        budget=dict(new_peak_bytes=PEAK,reserve_bytes=20*2**30,expected_minutes=90,hard_minutes_per_run=150,storage='Lossless BF16/XOR ZIP; compress each batch in RAM, check its exact bytes before disk write. Stop if20GiB reserve would be crossed; noC-drive fallback.'),source='Author-created arithmetic programs, CC0.',
        limits='Simple short arithmetic and English instructions only. No broad side-task damage test or SAE comparison; no trained state predictor. A token-class decision is not the full subsequent answer, semantic necessity, uniqueness, or a unified language law.'))
    print('Frozen96 paired families,192 generations, one instruction-token difference per pair',flush=True)


if __name__=='__main__':prepare()
