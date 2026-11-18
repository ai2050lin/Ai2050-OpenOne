"""Prospective author-written branch tests, generated through template_grammar.v1.

Stimuli are constructed language, not synthetic model results. Family IDs bind all
six role/title variants. Only discovery rows are admitted to the pilot/selection.
"""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from shared.dist_protocol import grammar
from shared.dist_protocol.mechanism_trace import digest,save_json
from deploy.storage_policy import check_write

SEED=202611131220
OUT=ROOT/'tests/glm5/result/recursive_branch14_20261113_v6'
GRAMMAR_DIR=ROOT/'shared/dist_protocol/grammars'
FAMILIES=('structure','procedure')
DOMAINS={
 'structure':[
  ('mathematics','A notation sheet describes a small system of named symbols.'),
  ('software','A software manual describes a component registry using named entries.'),
  ('theatre','A stage manager documents how costume tags are assigned to performers in a fictional production.'),
  ('laboratory','A fictional laboratory catalog describes the assignment of sample tags to storage labels.')],
 'procedure':[
  ('mathematics','A worksheet concerns the order of entries in a symbolic triple.'),
  ('software','A program note concerns the order of tags in a three-entry buffer.'),
  ('music','A rehearsal note concerns the order of three named musical motifs.'),
  ('logistics','A warehouse note concerns the order of three tagged crates in a delivery queue.')],
}
# Disjoint whole-word pools between filler partitions. Subword overlap is allowed
# and explicitly audited; this is unseen lexical identity, not unseen token IDs.
WORDS=[
 'acorn amber anchor anvil apron arrow atlas badger bamboo banner basil beacon birch bison blade blossom bonnet bronze cabin cactus candle canyon cedar cherry chisel clover cobalt comet coral cotton crane crystal dagger daisy delta dingo dragon drum eagle ebony elm falcon fern fig flint flute fox frost garden garnet ginger globe granite grape hammer harbor hazel helmet heron hollow honey cork cumin daffodil'.split(),
 'island ivory jacket jasmine jewel juniper kettle kiwi ladder lagoon lantern lark laurel lemon leopard lilac linen lion lotus marble meadow melon mirror mitten moss nectar nettle nickel oak olive onion opal orchid otter oyster paddle panda paper pebble pelican pepper petal pigeon pine planet plum pocket poppy potato prism pumpkin quartz rabbit raven reed ribbon river robin rocket rose ruby saddle mango parrot'.split(),
 'saffron salmon sandal sapphire satin scarf seal sesame shell shield shore silver slate snail snow sparrow spear spice spindle spruce squirrel star stone storm straw summit swan sword table talon temple thorn thistle tiger timber toast torch tulip towel turtle valley velvet violet walnut willow wind window wolf wool wren yarn zebra zinnia zephyr zipper beetle biscuit brook button carpet chestnut copper maple tundra'.split(),
]
assert all(len(w)==64 for w in WORDS),[len(w) for w in WORDS]
assert len(set(sum(WORDS,[])))==192


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))


def title(fam,lexical):
    if lexical:
        return {'structure':'function, vector, class, object, algebra',
                'procedure':'algorithm, calculation, hash, matrix, transformation'}[fam]
    return 'notebook, section, entry, edition, reference'


def passage(fam,frame,role):
    # Every quoted arm contains a full rule, but explicitly withholds its use.
    # It is a control for operative status, NOT a relation-free negative example.
    if fam=='structure':
        rules=[
          'The rule pairs {X} with {Y} and {Z} with {W}. An input selects the item paired with it.',
          'A request labeled {X} is assigned to {Y}; a request labeled {Z} is assigned to {W}. These associations determine the response.',
          'Each source has a designated counterpart: {Y} belongs to the source {X}, whereas {W} belongs to {Z}. The counterpart is returned when its source is requested.'
        ]
        inventories=[
          'The inventory lists four separate entries: {X}, {Y}, {Z}, and {W}. Their order is a printing choice and does not determine a response to an input.',
          'The names {X}, {Y}, {Z}, and {W} occur as independent headings on a page. A request-to-response association is not supplied.',
          'Four labels are catalogued separately: {X}; {Y}; {Z}; {W}. No source has been given a designated counterpart.'
        ]
    else:
        rules=[
          'Start with the ordered entries {X}, {Y}, {Z}. {STEP0} The resulting order is the output.',
          'The initial sequence reads {X}, {Y}, {Z}. To obtain its output, {STEP1}',
          'An incoming list contains {X} followed by {Y} followed by {Z}. Its outgoing arrangement is obtained as follows: {STEP2}'
        ]
        inventories=[
          'The inventory contains the ordered entries {X}, {Y}, {Z}. It records their initial order only; no procedure for obtaining an output is specified.',
          'The entries {X}, {Y}, {Z} are printed in that order as stored data. Instructions for deriving a later sequence are not supplied.',
          'A catalogue records {X} followed by {Y} followed by {Z}. It gives an incoming list, but leaves its outgoing arrangement undefined.'
        ]
    if role=='operative':
        return ['The following instruction is in force. ',
                'The adopted specification states: ',
                'The account below governs the current case. '][frame]+rules[frame]
    if role=='quoted':
        return ['An editor quotes the following instruction for proofreading only: "',
                'A rejected proposal states: "',
                'An example printed for discussion reads: "'][frame]+rules[frame]+[
                '" It has not been adopted. No other rule for the current case is provided.',
                '" This proposal is not in force, and the current specification supplies no replacement rule.',
                '" This example does not govern the current case; no applicable instruction has been given.'][frame]
    return inventories[frame]


def filler_skeleton(fam,domain):
    return [dict(id=f'case-{i:03d}',slots={'X':'temporary','Y':'temporary','Z':'temporary','W':'temporary',
        'STEP0':'temporary','STEP1':'temporary','STEP2':'temporary','C0':'temporary','C1':'temporary','C2':'temporary',
        'AOP':' A','ANON':' B','AFOIL':' C','END':'End of passage.','domain_intro':domain[1]},levels={}) for i in range(192)]


def specification(fam,domain_index):
    domain=DOMAINS[fam][domain_index];family=f'RG14-{fam}-{domain[0]}-20261113-v3'
    frames=[]
    for frame in range(3):
      for role in ['operative','quoted','inventory']:
       for lexical in (0,1):
        body='{domain_intro}\nThe file title is "'+title(fam,lexical)+'". The title is metadata, not an instruction.\n'+passage(fam,frame,role)+'\n{END}'
        question={'structure':'Under the rule in force, what response is assigned to {X}?',
                  'procedure':'Under the procedure in force, what is the resulting order?'}[fam]
        prompt=body+'\n'+question+'\nA. {C0}\nB. {C1}\nC. {C2}\nGive only the option letter.\nAnswer:'
        frames.append(dict(id=f'f{frame}-{role}-l{lexical}',levels={'role':role,'lexical':str(lexical),'wording':str(frame)},
            text=body,target='END',probe={'prompt':prompt,'answer':'{AOP}' if role=='operative' else '{ANON}',
                'contrast':['{ANON}' if role=='operative' else '{AOP}','{AFOIL}']}))
    spec=dict(schema_version='template_grammar.v1',family_id=family,grammar_version=3,
        name='Native Qwen14 branch test: '+fam+' / '+domain[0],
        description='Author-created fictional material; joint G/V/A; role × incidental lexical title.',
        factors=['role','lexical','wording'],frames=frames,fillers=filler_skeleton(fam,domain),
        splits={'salt':'rg14-v1','fractions':{'discovery':1/3,'confirm':1/3,'lockbox':1/3},
                'heldout_frames':[f['id'] for f in frames if f['id'].startswith('f2-')]},
        layers='all',readouts=['target','probe_answer'],mlp_units={'layers':'all','readouts':['target','probe_answer']},
        prereg={'min_behavior_acc':.8,'min_cell_n':32,'min_lockbox_cos':0.0},shards=1,
        source={'license':'CC0-1.0','origin':'Original constructed stimuli authored for this study; no asserted real-world facts.'})
    partitions=grammar.assign_splits(spec)
    counters={k:0 for k in ('discovery','confirm','lockbox')}
    for fl in sorted(spec['fillers'],key=lambda x:x['id']):
        split=partitions[fl['id']];rank=counters[split];counters[split]+=1
        pool=WORDS[['discovery','confirm','lockbox'].index(split)]
        rng=np.random.default_rng(SEED+domain_index*100000+(0 if fam=='structure' else 700000)+int(fl['id'][5:]))
        chosen=rng.choice(pool,4,replace=False).tolist();x,y,z,w=chosen
        # Two procedure branches: reversal and cyclic rotation; balanced within
        # each split. Wording changes without altering the stipulated operation.
        reverse=((rank//2)%2==0)
        steps=(['Exchange the first and third entries, leaving the middle entry in place.',
                'put the last entry first and the first entry last, keeping the second entry unchanged.',
                'reverse the three positions.'] if reverse else
               ['Move the first entry to the end, leaving the other two in their original order.',
                'remove its leading entry and append that entry after the remaining pair.',
                'rotate the entries one position to the left.'])
        answer=y if fam=='structure' else ', '.join([z,y,x] if reverse else [y,z,x])
        foil=w if fam=='structure' else ', '.join([y,z,x] if reverse else [z,y,x])
        unknown='Not specified by an applicable rule'
        swap=rank%3
        choices=[None]*3;choices[swap]=answer;choices[(swap+1)%3]=foil;choices[(swap+2)%3]=unknown
        op=' '+'ABC'[swap];non=' '+'ABC'[(swap+2)%3];foil_letter=' '+'ABC'[(swap+1)%3]
        fl['slots'].update(X=x,Y=y,Z=z,W=w,STEP0=steps[0],STEP1=steps[1],STEP2=steps[2],
            C0=choices[0],C1=choices[1],C2=choices[2],AOP=op,ANON=non,AFOIL=foil_letter)
        fl['metadata']=dict(partition=split,rank=rank,operation='reverse' if reverse else 'rotate',
            lexical_words=chosen,option_order=swap)
    assert counters=={'discovery':64,'confirm':64,'lockbox':64}
    errors=grammar.validate(spec);assert not errors,errors
    return spec


def prepare():
    from transformers import AutoTokenizer
    check_write(OUT,100*2**20);OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'materials.json').exists(),'Frozen materials already exist'
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/Qwen3-14B',local_files_only=True)
    rows=[];audits=[];source_hashes={};all_words={}
    for fam in FAMILIES:
      for di,domain in enumerate(DOMAINS[fam]):
        spec=specification(fam,di);path=check_write(GRAMMAR_DIR/(spec['family_id']+'.json'))
        if path.exists():assert read(path)==spec,'Do not overwrite a changed grammar'
        else:save_json(path,spec)
        source_hashes[str(path.relative_to(ROOT))]=digest(spec)
        corp=grammar.generate(spec,include_canary=False);fill={x['id']:x for x in spec['fillers']}
        for item in corp['items']:
            fl=fill[item['filler_id']];part=fl['metadata']['partition'];frame=int(item['cond']['wording'])
            # Only one wording per family; all role/title variants stay together.
            if di<2 and part in ('discovery','confirm') and frame==fl['metadata']['rank']%2:
                split='discovery' if part=='discovery' else 'new_words'
            elif di<2 and part=='lockbox' and frame==2:split='new_frames'
            elif di>=2 and part=='lockbox' and frame==fl['metadata']['rank']%2:split='new_domains'
            else:continue
            body=item['text'];prompt=item['probe']['prompt'];encoded=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
            ids=encoded['input_ids'];offsets=encoded['offset_mapping']
            # The tokenizer merges the terminal period with the following
            # newline. Include that newline; it still precedes every probe word.
            end=len(body)+1;anchor=max(i for i,(a,b) in enumerate(offsets) if b<=end and b>a)
            assert tok.decode(ids[anchor:anchor+1])=='.\n',(fam,di,ids[anchor],tok.decode(ids[anchor:anchor+1]))
            assert len(ids)<=256,len(ids)
            choice=[tok.encode(s,add_special_tokens=False) for s in [' A',' B',' C']];assert all(len(x)==1 for x in choice)
            row=dict(id=item['id'],family=f'{fam}:{domain[0]}:{fl["id"]}',hypothesis=fam,domain=domain[0],domain_index=di,
                split=split,grammar_split=item['split'],frame=frame,role=item['cond']['role'],lexical=int(item['cond']['lexical']),
                prompt=prompt,body=body,tokens=ids,content_end=anchor,answer_pos=len(ids)-1,
                content_start=next(i for i,(a,b) in enumerate(offsets) if a>=prompt.index('\n',prompt.index('The file title'))+1),
                gold='ABC'.index(item['probe']['answer'].strip()),choice_ids=[x[0] for x in choice],
                metadata=fl['metadata'],license='CC0-1.0',source='author_generated_template_grammar.v1')
            assert row['content_start']<=row['content_end']<row['answer_pos']
            rows.append(row)
            all_words.setdefault((fam,di,part),set()).update(fl['metadata']['lexical_words'])
        audits.append(dict(family_id=spec['family_id'],grammar_sha256=grammar.spec_sha(spec),
            validation_errors=[],visibility=corp['visibility'],warnings=corp['_warnings'],
            preview=grammar.preview(spec,sample=1)))
        print('material family',fam,domain[0],'generated',len(corp['items']),flush=True)
    rows.sort(key=lambda r:({'discovery':0,'new_words':1,'new_frames':2,'new_domains':3}[r['split']],r['family'],r['role'],r['lexical']))
    assert len(rows)==6144,len(rows)
    assert len({r['id'] for r in rows})==len(rows)
    assert len({r['prompt'] for r in rows})==len(rows)
    families={r['family'] for r in rows};assert len(families)==1024,len(families)
    for family in families:
        rs=[r for r in rows if r['family']==family]
        assert len(rs)==6 and len({r['split'] for r in rs})==1
        assert {(r['role'],r['lexical']) for r in rs}=={(s,l) for s in ['operative','quoted','inventory'] for l in [0,1]}
    for fam in FAMILIES:
      for di in range(2):
        sets=[all_words[fam,di,s] for s in ['discovery','confirm','lockbox']]
        assert all(not sets[i]&sets[j] for i in range(3) for j in range(i))
    # The grammar API checks causal visibility at the target and probe boundary;
    # additionally audit identical anchor tokens and no correct option in body.
    lengths=[len(r['tokens']) for r in rows]
    for fam in FAMILIES:
      for split in ['discovery','new_words','new_frames','new_domains']:
        for fr in {r['frame'] for r in rows if r['hypothesis']==fam and r['split']==split}:
            counts={op:len({r['family'] for r in rows if r['hypothesis']==fam and r['split']==split and r['frame']==fr and r['metadata']['operation']==op}) for op in ['reverse','rotate']}
            assert counts['reverse']==counts['rotate'],(fam,split,fr,counts)
    save_json(OUT/'materials.json',dict(schema='recursive_branch14_materials.v3',rows=rows,seed=SEED,
        description='Fresh model responses; author-written English controlled language; no claim of independent human authorship.',
        scope='Two existing Qwen14 candidates only; four Qwen4 candidate IDs are deferred, not mapped by index.'))
    save_json(OUT/'material-revision.json',dict(previous=[dict(path=f'recursive_branch14_20261113_v{i}/materials.json',
        sha256=hashlib.sha256((OUT.parent/f'recursive_branch14_20261113_v{i}/materials.json').read_bytes()).hexdigest()) for i in [1,2]],
        reason='Before any model responses: v2 added incorrect mapping/order competitor; v3 uncouples operation identity from wording frame and adds explicit balance assertion.',
        previous_model_results_exist=False,new_grammar_version=3))
    save_json(OUT/'materials-audit.json',dict(rows=len(rows),families=len(families),
        cell_counts={f'{h}/{s}':sum(r['hypothesis']==h and r['split']==s for r in rows) for h in FAMILIES for s in ['discovery','new_words','new_frames','new_domains']},
        min_tokens=min(lengths),max_tokens=max(lengths),mean_tokens=float(np.mean(lengths)),
        current_content_token_ids=sorted({r['tokens'][r['content_end']] for r in rows}),
        whole_word_pools_disjoint=True,subword_tokens_may_overlap=True,
        domains_held_out=['theatre','laboratory','music','logistics'],
        input_audit='All stimulus information precedes the content readout. Options occur only afterward. Answer-order counterbalanced within families.',
        grammar_sources=source_hashes,grammar_audits=audits,
        limitations=['Each family has six correlated views, not six independent samples.',
            'Words are reused within a partition; family bootstrap does not make repeated vocabulary independent.',
            'Quoted rules still describe relations; only their operative status is changed.',
            'New domains test the frozen narrow rule-description hypotheses, not all uses of the original neurons.']))
    print('materials complete',len(rows),'rows',len(families),'families',max(lengths),'max tokens',flush=True)


if __name__=='__main__':
    from recursive_branch14_revise import prepare as prepare_revision
    prepare_revision()
