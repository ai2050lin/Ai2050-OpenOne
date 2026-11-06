"""Prospective semantic-role vs lexical-cue selection with exact paired inputs."""
import sys,json,subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import native_mlp_function_discovery as D
import native_mlp_long_materials as OLD
from shared.dist_protocol.mechanism_trace import file_sha,digest
from tools.power_analysis import check_phase
ROOT=D.ROOT;OUT=D.OUT;PUBLIC=D.PUBLIC;RUN=D.RUN;MODEL=D.MODEL;SEED=D.SEED;read=D.read;save=D.save
FAMILIES={
 'conditional':{'unit':[7,5521],'title':'被选中规则具有条件依赖','question':'Does the ACTIVE entry describe a rule whose action depends on a condition?',
  'positive':[['If the sensor detects {name}, the lamp turns on; otherwise it stays off.','Provided that {name} is detected, the lamp is enabled.'],['When the meter reports {name}, the valve opens, and it remains shut in other states.','The valve opens solely upon detection of {name}.'],['If the flag equals {name}, the relay engages; in any other state it does not.','Detection of {name} is a prerequisite for the relay to engage.']],
  'negative':[['The heading says "if {name}", but the lamp stays on independently of the sensor.','The lamp remains enabled irrespective of whether {name} is detected.'],['The heading says "when {name}", but the valve is open for every meter state.','The valve opens independently of detection of {name}.'],['The label reads "if {name}", but the relay remains engaged regardless of the flag.','The relay does not require detection of {name}; it is always engaged.']]},
 'universal':{'unit':[19,7534],'title':'被选中集合具有全称覆盖','question':'Does the ACTIVE entry assert that every member has a lid, with no exception?',
  'positive':[['All members of set {name} have a lid, without exception.','No member of set {name} is without a lid.'],['Every container in group {name} has a lid.','There is not a single container in group {name} lacking a lid.'],['All members of batch {name} possess a lid.','Lids are present on the entire batch {name}; there are no exceptions.']],
  'negative':[['The title is "all {name}", but only a proper subset has lids; at least one lacks a lid.','A member of set {name} lacks a lid, although others have lids.'],['The label says "every {name}", but some containers lack lids.','There are containers in group {name} without lids.'],['The name reads "all {name}", but not all members possess a lid.','At least one member of batch {name} has no lid.']]},
 'location':{'unit':[24,8443],'title':'被选中名称指代地点','question':'Does the ACTIVE entry define its name as a geographical place rather than a portable object?',
  'positive':[['{name} is a town where the workshop takes place.','{name} names a settlement on the northern coast.'],['{name} is a city; the exhibition is held in that location.','{name} denotes a settlement beside the river.'],['{name} is a village where the meeting is hosted.','{name} identifies a coastal settlement, situated west of the harbour.']],
  'negative':[['{name} is a portable tool bearing the inscription "town"; it is carried to the workshop.','{name} names a hand tool stored in the workshop.'],['{name} is an instrument engraved "city"; it can be carried to the exhibition.','{name} denotes a portable instrument kept beside the river.'],['{name} is a tool marked "village"; it is brought to the meeting.','{name} identifies a portable hand tool stored west of the harbour.']]},
 'enumeration':{'unit':[35,2572],'title':'被选中冒号引出列表','question':'Does the ACTIVE entry use a colon to introduce a list rather than as part of a clock time?',
  'positive':[['The inventory for {name} is as follows: bolts, washers and pins.','The kit for {name} contains the following: bolts, washers and pins.'],['The list for {name} reads: rings, screws and plates.','The components for {name} are: rings, screws and plates.'],['The items for {name} are listed here: rods, clips and nuts.','The package for {name} contains these: rods, clips and nuts.']],
  'negative':[['The shipment called "list {name}" was logged at 10:25, with the colon separating hours and minutes.','The shipment {name} was logged at 10:25.'],['The parcel labelled "list {name}" was stamped at 11:35; the colon separates hours and minutes.','The parcel {name} was stamped at 11:35.'],['The entry titled "list {name}" records arrival at 12:45, with a colon between hours and minutes.','The delivery {name} arrived at 12:45.']]}}
def build():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);old=read(OLD.OUT/'documents.json');raw=np.load(OLD.OUT/'input-tokens.npy',mmap_mode='r')
    rng=np.random.default_rng(SEED);order=rng.permutation(2048)[:544];rows=[];role_units=read(OUT/'discovery.json')['units'];prefix=tok.encode('<|im_start|>user\n',add_special_tokens=False)
    end=tok.encode('\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False)
    for family,design in FAMILIES.items():
        for k,di in enumerate(order):
            split='discovery' if k<32 else 'confirm' if k<288 else 'lockbox';frame_group={'discovery':0,'confirm':1,'lockbox':2}[split];frame=(k//2)%2;cue=k%2;positive_key='K' if (k//4)%2==0 else 'M'
            name=({'discovery':'mavora','confirm':'seluno','lockbox':'tirava'}[split])+str(k)
            p=design['positive'][frame_group][cue].format(name=name);n=design['negative'][frame_group][cue].format(name=name)
            records={'K':p if positive_key=='K' else n,'M':p if positive_key=='M' else n}
            common='\n\nThe preceding passage is background only. The following two catalogue entries are invented definitions and are authoritative for this task. Interpret only the selected entry.\nEntry K: '+records['K']+'\nEntry M: '+records['M']+'\n'
            selector=['The active entry is {key}. Observation ends here:', 'Select entry {key} for this case. End of observation:'][frame]
            suffix={key:tok.encode(common+selector.format(key=key),add_special_tokens=False) for key in ['K','M']};assert len(suffix['K'])==len(suffix['M'])
            differences=np.flatnonzero(np.asarray(suffix['K'])!=suffix['M']);assert len(differences)==1
            background_length=1024-len(prefix)-len(suffix['K']);assert background_length>=512
            encoded={key:prefix+raw[di,:background_length].tolist()+suffix[key] for key in ['K','M']};assert all(len(x)==1024 for x in encoded.values())
            selector_position=len(prefix)+background_length+int(differences[0]);early=len(prefix)+background_length+len(tok.encode(common,add_special_tokens=False))-2
            assert early<selector_position<1023 and encoded['K'][-1]==encoded['M'][-1]
            # The probe and randomized answer-code mapping occur strictly after all three readouts.
            yescode='P' if (k//8)%2==0 else 'Q';nocode='Q' if yescode=='P' else 'P'
            probe=f'\n\nQuestion: {design["question"]}\nAnswer {yescode} for YES or {nocode} for NO. Reply with just the answer letter.'
            probe_ids=tok.encode(probe,add_special_tokens=False)+end
            for value in [True,False]:
                key=positive_key if value else ('M' if positive_key=='K' else 'K');ids=encoded[key]+probe_ids
                rows.append({'id':f'{family}:{k}:{int(value)}','pair_id':f'{family}:{k}','family':family,'semantic':int(value),'cue_form':cue,'frame':frame,'split':split,'family_index':k,'name':name,
                    'positive_key':positive_key,'selected_key':key,'expected':yescode if value else nocode,'contrast':nocode if value else yescode,
                    'token_ids':ids,'checkpoint_ids':encoded[key],'readout_positions':[early,selector_position,1023],'input_tokens':len(ids),'checkpoint_tokens':1024,
                    'background_doc_index':int(di),'background_doc_id':old['docs'][di]['doc_id'],'text':tok.decode(encoded[key]),'prompt':tok.decode(ids),
                    'entity':name,'scope':design['title'],'task_question':design['question'],'records':records})
    snap=read(OUT/'discovery.json')['model_snapshot'];c={'schema':'native_mlp_function_corpus.v1','kind':'real','run_id':RUN,'model_snapshot':snap,'items':rows,
        'hypotheses':[{k:v for k,v in h.items() if k in ['unit','title','question']}|{'family':f} for f,h in FAMILIES.items()],
        'observed_units':[[r['layer'],r['unit']] for r in role_units],'background_source_sha256':file_sha(OLD.OUT/'documents.json'),
        'split_note':'different frozen background documents, nonce name stems, wording frame groups; all four families reuse background families and are not independent population samples',
        'pair_control':'checkpoint input differs in exactly one active-key token; identical length, records, bag of semantic cues and final token; positive assignment and answer letters counterbalanced'}
    save(OUT/'corpus.json',c)
    pr={'schema':'native_mlp_function_prereg.v1','kind':'real','run_id':RUN,'model_snapshot':snap,'corpus_sha256':digest(c),'seed':SEED,'n_inputs':len(rows),
        'hypotheses':c['hypotheses'],'primary':'per family preregistered native unit A ordering, sign chosen using discovery only; same input except active key, final common token; confirm andlockbox both required',
        'criteria':{'alpha':.05/4,'min_order_accuracy':.65,'min_unequal_fraction':.8,'min_behavior_cell_accuracy':.9,'minimum_cell_n':32,'min_effect_over_selector_null':.15,
            'power':check_phase(.25,256,p_d=.5,alpha=.05/4)},
        'pilot':'discovery only; every family x cue_form x frame x semantic cell has8items; min7/8 choice-correct per cell; no confirm orlockbox read during calibration',
        'baseline':'final hs[0] exactly shared; active-key identity classifier trained discovery only; paired assignment-null is exact balanced by selector key. Prefix before key is causal negative control.',
        'secondary':'G,V, other units andhidden-state multivariate probes exploratory; use 12test Bonferroni when discussing G/V/A joint. No semantic promotion without behavior andboth holdouts.',
        'scope':'selected-entry functional-role coding, not free natural semantics, general intelligence or whole-neuron interpretation',
        'resources':{'max_local_GiB':65,'one_CUDA_model_only':True,'collection':'all37 hidden andall36 blocks complete fields at3declared readouts; deep8 fullprefix states/QKV'},
        'previous_negative_result':'NMLP-LONG-20261104-v1 0/432 retained, no old threshold changed'}
    assert pr['criteria']['power']['pass'];save(PUBLIC/'prereg.research.json',pr);print(json.dumps({'inputs':len(rows),'checkpoints':1024,'power':pr['criteria']['power']}))
if __name__=='__main__':build()
