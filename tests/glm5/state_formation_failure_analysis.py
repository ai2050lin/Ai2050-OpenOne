"""Post-gate descriptive error diagnosis; no selection or acceptance changes."""
import argparse,collections,re
import numpy as np
import state_formation_core as C

def old_color(row):
    match=re.search(r'Obsolete entry[^\n]+X-\d+ was (red|blue);[^\n]+Y-\d+ was (red|blue)',row['prompt'])
    return (1 if match.group(1 if row['query']==1 else 2)=='red' else -1) if match else None

def behavior():
    C.checked();rows=C.read(C.OUT/'confirm-materials.json')['rows'];groups=collections.defaultdict(list);errors=[];ties=[];disagreements=[];output_language=[]
    for f in range(768):
        for row,score in zip(rows[f*8:(f+1)*8],C.read(C.OUT/f'confirm-{f:04d}.json')['scores']):
            old=old_color(row);predicted=1 if int(np.argmax(score['four_logits']))%2==0 else -1
            four=np.asarray(score['four_logits']);output_language.append((1 if int(four.argmax())<2 else -1)==row['language'])
            if (four==four.max()).sum()>1:ties.append(row['id'])
            if score['four_correct']!=score['first_top1']:disagreements.append({'id':row['id'],'four_correct':score['four_correct'],'first_top1':score['first_top1'],'four_logits':score['four_logits'],'argmax':score['argmax']})
            conflict=old is not None and old!=row['color'];groups[(row['history'],conflict,row['language'])].append(score['four_correct'])
            if not score['four_correct']:errors.append({'id':row['id'],'frame':row['frame'],'history':row['history'],'language':row['language'],
                'color':row['color'],'query':row['query'],'old_color':old,'predicted_color':predicted,'matches_old':predicted==old,'prompt':row['prompt'],'score':score})
    C.save('behavior-error-diagnosis.json',{'scope':'post-gate descriptive error analysis; no refit or causal attribution',
        'source_sha256':C.sha(__file__),'errors':len(errors),'errors_matching_obsolete_color':sum(r['matches_old'] for r in errors),
        'four_choice_top_ties':ties,'four_vs_first_top1_disagreements':disagreements,'four_choice_output_language_accuracy':float(np.mean(output_language)),
        'tie_policy':'frozen four-choice argmax uses en_red,en_blue,zh_red,zh_blue order; full-vocabulary argmax uses token IDs; BF16 exact ties can differ; neither policy changed',
        'error_language':dict(collections.Counter(str(r['language']) for r in errors)),
        'groups':[{'history':k[0],'old_conflicts':k[1],'language':k[2],'n':len(v),'four_choice_accuracy':float(np.mean(v))} for k,v in groups.items()],
        'rows':errors})

def effects():
    C.checked();groups=collections.defaultdict(list)
    for f in range(768):
        item=C.read(C.OUT/f'causal-{f:04d}.json');row=item['sample'];base=item['baseline'];old=old_color(row)
        condition='no_old_mapping' if old is None else 'old_conflicts' if old!=row['color'] else 'old_agrees'
        for rec in item['records']:
            edit=rec['edited'];factor='language' if rec['candidate'].startswith('language') else 'color';field=factor+'_margin'
            for stratum in [condition,'base_four_correct' if base['four_correct'] else 'base_four_wrong']:
                groups[(rec['candidate'],rec['method'],rec['dose'],stratum)].append({'loss':base[field]-edit[field],
                    'excess':rec['norm_control'][field]-edit[field],
                    'base_four_correct':base['four_correct'],'edited_four_correct':edit['four_correct'],
                    'lost':base['four_correct'] and not edit['four_correct'],'rescued':not base['four_correct'] and edit['four_correct']})
    result=[]
    for key,items in groups.items():
        result.append(dict(zip(['candidate','method','dose','stratum'],key),n=len(items),
            mean_margin_loss=float(np.mean([r['loss'] for r in items])),mean_excess=float(np.mean([r['excess'] for r in items])),
            base_four_correct=sum(r['base_four_correct'] for r in items),edited_four_correct=sum(r['edited_four_correct'] for r in items),
            lost=sum(r['lost'] for r in items),rescued=sum(r['rescued'] for r in items)))
    C.save('effect-error-diagnosis.json',{'scope':'post-gate descriptive subgroup diagnostics; strata overlap; no new pass threshold or scientific confirmation',
        'source_sha256':C.sha(__file__),'rows':result})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['behavior','effects']);args=parser.parse_args()
    {'behavior':behavior,'effects':effects}[args.stage]()
