"""Calibrate and freeze family-paired long-context attribute binding on Qwen3-4B."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import random
import re

import mlp_factor_steer_common as prior


RUN = 'ATTRIBUTE-BINDING-MLP-CIRCUIT-20261106-v1'
SEED = 611062809
CUT = 992
LENGTH = 1088
BLOCK = 30
ATOMS = 64
COUNTS = {'discovery': 96, 'new_frame': 128, 'new_answer': 128, 'new_relation': 128}
DOSES = (0.25, 0.5, 1.0, 2.0)
BASE_WORDS = ('copper', 'silver', 'bronze', 'gold')
NOVEL_WORDS = ('black', 'white', 'red', 'green')
TC = Path('D:/ai2050_native/_transcoder/6be1034f0b55a6df-bf16/2ca5f0a756d36c8a-L30/tc-m16384-k64-s0')
FRAMES = (
    ('The verified entry for crate {name} has reference {tag}.', 'Its finish is {word}.',
     'Which finish is recorded for the crate with reference {key}?'),
    ('Reference {tag} identifies crate {name} in the signed inventory.', 'The finish on this crate is {word}.',
     'Use reference {key} to locate the crate. What finish does its entry specify?'),
    ('For crate {name}, the official register assigns reference {tag}.', 'The stated finish is {word}.',
     'What finish does the register give for the crate identified by reference {key}?'),
    ('The inventory binds reference {tag} to crate {name}.', 'The listed finish for that crate is {word}.',
     'Identify the finish associated with reference {key}.'),
)
NOVEL_FRAMES = (
    ('The record describes crate {name}, which can be found under reference {tag}.', 'Finish specified: {word}.',
     'Find the entry indexed by reference {key} and state its finish.'),
    ('Under reference {tag} the registrar filed the description of crate {name}.', 'Finish on file: {word}.',
     'Give the registered finish from the file indexed by reference {key}.'),
)
PLACES = ('courtyard', 'gallery', 'foyer', 'terrace', 'workroom', 'passage', 'garden', 'landing')
ACTORS = ('staff', 'visitors', 'volunteers', 'organizers', 'assistants', 'guides', 'attendants', 'guests')
TOPICS = ('admission', 'opening hours', 'the afternoon schedule', 'access routes', 'the seating plan', 'the noticeboard')


def first_word(text):
    match=re.match(r'^\s*["\x27`*]*(\w+)',text.lower())
    return match.group(1) if match else ''


def make_material(tokenizer, counts):
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    start = encode('<|im_start|>user\n')
    end = encode('\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nThe finish is')
    newline = encode('\n')
    assert len(newline) == 1
    surfaces = {w: sorted({ids[0] for v in (w, w.capitalize(), ' '+w, ' '+w.capitalize())
                          if len(ids := encode(v)) == 1}) for w in BASE_WORDS+NOVEL_WORDS}
    assert all(len(encode(' '+w)) == 1 for w in surfaces)
    assert len({i for ids in surfaces.values() for i in ids}) == sum(map(len, surfaces.values()))
    result = []
    for group, (split, size) in enumerate(counts.items()):
        rng = random.Random(SEED + group * 239 + (27003 if split.startswith('pilot') else 0))
        linked = split in ('new_relation', 'pilot_relation')
        frames = NOVEL_FRAMES if split == 'new_frame' else FRAMES[2:] if split.startswith('pilot') else FRAMES[:2]
        words = NOVEL_WORDS if split == 'new_answer' else BASE_WORDS
        for i in range(size):
            frame = i % len(frames)
            identity, value, query = frames[frame]
            answer = rng.sample(words, 2)
            names = rng.sample(('M', 'N', 'P', 'R'), 2)
            tags = rng.sample(('K', 'T', 'V', 'Z'), 2)
            selected = (i // len(frames)) % 2
            keys = rng.sample(('D', 'F', 'L', 'S'), 2)
            records = [identity.format(name=n, tag=t) for n, t in zip(names, tags)]
            if linked:
                lead = '\n\nSigned inventory. Team assignments: '
                lead += ' '.join(f'Team {k} is assigned reference {t}.' for k,t in zip(keys,tags)) + '\n'
                question = f'What finish is recorded for the crate assigned to Team {keys[selected]}?'
            else:
                lead = '\n\nSigned inventory.\n'
                question = query.format(key=tags[selected])
            suffix = encode('\n'+question+' Answer with only the one-word finish, using the signed inventory.')+end
            bg_text = '\n'.join(
                f'The {rng.choice(ACTORS)} met in the {rng.choice(PLACES)} before moving on to the '
                f'{rng.choice(PLACES)}. The notice covered {rng.choice(TOPICS)}. '
                f'A tally of {rng.randint(11,97)} entries was written beside the room plan.'
                for _ in range(50))
            bg = encode(bg_text)
            variants, spans = [], []
            for ws in (answer, answer[::-1]):
                body = lead
                marks = []
                for record, w in zip(records, ws):
                    text = value.format(word=w)
                    body += record+' '
                    lo = len(body)+text.index(w)-1
                    marks.append((lo,lo+len(w)+1))
                    body += text+'\n'
                draft = answer[(i//4)%2]
                body += f'An unsigned draft mentions {draft}. It is not an inventory entry.\n'
                encoded = tokenizer(body, add_special_tokens=False, return_offsets_mapping=True)
                pad = CUT-len(start)-len(encoded.input_ids)
                assert 800 <= pad <= len(bg)
                marks = [[CUT-len(encoded.input_ids)+j for j,(l,h) in enumerate(encoded.offset_mapping)
                          if l<b and h>a] for a,b in marks]
                assert all(len(m)==1 for m in marks), (split, ws, marks)
                tokens = start+bg[:pad]+encoded.input_ids+newline*(LENGTH-CUT-len(suffix))+suffix
                assert len(tokens)==LENGTH and len(suffix)<LENGTH-CUT
                variants.append(tokens)
                spans.append([m[0] for m in marks])
            assert spans[0]==spans[1]
            assert [j for j,(a,b) in enumerate(zip(*variants)) if a!=b]==spans[0]
            result.append({'id':f'{split}:{i:03}', 'split':split, 'frame':frame, 'relation':'team_to_reference' if linked else 'direct_reference',
                           'selected_slot':selected, 'target':answer[selected], 'contrast':answer[1-selected],
                           'sites':spans[0]+[CUT-1], 'tokens':variants, 'background_sha256':prior.digest(bg)})
    assert len({r['background_sha256'] for r in result})==len(result)
    return {'rows':result,'surfaces':surfaces}


def calibrate(out):
    import torch
    from mlp_functional_trace import load_model
    from mlp_fidelity_v2_collect_v4 import greedy
    from shared.dist_protocol.mechanism_trace import GpuLease

    with GpuLease(), torch.inference_mode():
        model,tok,eos = load_model()
        data=make_material(tok, {'pilot_direct':32,'pilot_relation':32})
        counts=defaultdict(lambda:[0,0,0,0])
        examples=[]
        for offset in range(0,len(data['rows']),4):
            batch=data['rows'][offset:offset+4]
            x=torch.tensor([v for row in batch for v in row['tokens']],device='cuda')
            step=model(x,use_cache=True,logits_to_keep=1)
            first=step.logits[:,0].float()
            continuations=greedy(model,step.past_key_values,first,eos,limit=4)
            for j,row in enumerate(batch):
                expected=[row['target'],row['contrast']]
                cell=counts[(row['split'],row['frame'])]
                for logits,word,other,tokens in zip(first[2*j:2*j+2],expected,expected[::-1],continuations[2*j:2*j+2]):
                    cell[0]+=1
                    choice=float(logits[data['surfaces'][word]].max()-logits[data['surfaces'][other]].max())
                    top=int(logits.argmax())
                    cell[1]+=int(top in data['surfaces'][word]);cell[2]+=int(choice>0)
                    answer=tok.decode(tokens)
                    cell[3]+=int(first_word(answer)==word)
                    examples.append({'id':row['id'],'expected':word,'first':tok.decode([top]),'continuation':answer,'gap':choice})
        pilot={f'{split}:{frame}':{'n':n,'top1':a,'candidate':b,'first_word':c,'passed':c/n>=.8 and b/n>=.85}
               for (split,frame),(n,a,b,c) in counts.items()}
        prior.save(out/'pilot.json',{'cells':pilot,'cases':examples,'model_snapshot':prior.identity()['model_snapshot']})
        print(pilot,flush=True)


def freeze(out):
    from transformers import AutoTokenizer
    from scipy.stats import nct,t
    pilot=prior.read(out/'pilot.json')
    assert all(c['passed'] for c in pilot['cells'].values()), 'Calibrate discovery behavior before freezing.'
    tok=AutoTokenizer.from_pretrained(prior.MODEL,local_files_only=True)
    data=make_material(tok, COUNTS)
    path=out/'inputs.json.gz'
    prior.write_gz(path,data)
    inputs=prior.file_sha(path)
    assert prior.read(TC/'train.json')['model_snapshot']==prior.identity()['model_snapshot']
    alpha=.05/9
    threshold=t.ppf(1-alpha,127)
    prereg={**prior.identity(),'run_id':RUN,'schema':'attribute_binding_mlp_prereg.v1','seed':SEED,'splits':COUNTS,
            'length':LENGTH,'cut':CUT,'block':BLOCK,'atoms_per_site':ATOMS,'sites':['record_0_value','record_1_value','prefix_tail'],
            'source':'Deterministic generated inventory records with neutral long background; swapped-value twins differ only at two known tokens.',
            'isolation':{'new_frame':'only frame family changes','new_answer':'only finish vocabulary changes','new_relation':'team-to-reference to crate; same vocabulary and record frames'},
            'inputs_sha256':inputs,'runner_sha256':prior.file_sha(__file__),'pilot_sha256':prior.file_sha(out/'pilot.json'),
            'transcoder_sha256':prior.file_sha(TC/'transcoder.npz'),'transcoder_manifest_sha256':prior.file_sha(TC/'train.json'),
            'methods':{'native':'mean |delta A| times native down-column norm','attribution':'mean signed -delta A times gradient of clean directed logit gap; B30 only','transcoder':'mean |delta z| times decoded feature norm; frozen same-snapshot natural-language B30 TopK transcoder'},
            'dose':DOSES,'primary_dose':1.,'primary':'Decrease in clean target-vs-alternative logit margin above matched equal-norm control.',
            'gate':{'base_clean_first_word':.8,'paired_d':.35,'alpha_one_sided':alpha,'min_margin_effect':.35,
                    'behavior':'First lexical word of four greedy tokens, tolerating quote, Markdown asterisk or backtick prefixes. Exact first token recorded separately. Anchor requires >=10 pp alternative-word gain and >=5 pp gain over matched control.',
                    'kv':'Full changed-slot replay and restoration; require exactness for reported native path.',
                    'norm':'Pre-round 1e-5 relative norm; post-round discrepancy separately, gate 5% maximum on nontrivial sites.'},
            'power_at_128':float(nct.sf(threshold,127,.35*math.sqrt(128))),
            'evaluation':'Discovery fixes 64 indices per site and method before heldout collection; paired document is the unit; counterfactual donor model states allowed; no independent prediction claim.',
            'capability':'All three methods, primary dose, frozen mean discovery direction and median site offsets, deterministic balanced signs, archived 2200 eligible questions, paired 2 percentage point noninferiority. This is a fixed-vector collateral stress test.'}
    prior.save(out/'prereg.json',prereg)
    print('Frozen',inputs,prior.file_sha(out/'prereg.json'),flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('stage',choices=['pilot','freeze'])
    p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    {'pilot':calibrate,'freeze':freeze}[a.stage](a.out)


if __name__=='__main__':
    main()
