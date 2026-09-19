"""Freeze connected dependency holdouts and exactly interpretable text/code programs."""
import random
from collections import Counter, defaultdict
from rdc_binding_common import *

def connected(row):
    by=defaultdict(set)
    for w in row.get('retrospective_ud',[]):
        if w['head']: by[(w['sentence_id'],w['head'])].add(w['relation'])
    return ['+'.join(p) for p in [('obj','advcl'),('nsubj:pass','obl')] if any(set(p)<=v for v in by.values())]

def programs(tok, cases=range(48), prospective_depth=6):
    result=[]
    for family in ('alias','conditional','mapping','addition'):
      for case in cases:
        split='train' if case<24 else 'validation' if case<32 else 'test' if case<40 else 'depth_test' if case<48 else 'prospective_depth6'
        depth=(1+case%2) if case<40 else 4 if case<48 else prospective_depth
        rng=random.Random(f'2732:{family}:{case}')
        names=[f'item_{case}_{j}' for j in range(depth+2)]
        value=rng.randrange(1,9); initial=value
        en=[f'The recorded value of {names[0]} is {value}.']
        zh=[f'{names[0]}的记录值是{value}。']
        code=[f'{names[0]} = {value}']
        edges=[]
        for j in range(1,depth+1):
            a,b=names[j-1],names[j]
            if family=='alias':
                en.append(f'{b} receives the same value as {a}.');zh.append(f'{b}取与{a}相同的值。');code.append(f'{b} = {a}')
            elif family=='conditional':
                threshold=rng.randrange(2,8)
                value=9-value if value>threshold else value
                en.append(f'If {a} is greater than {threshold}, {b} is 9 minus {a}; otherwise {b} keeps the value of {a}.')
                zh.append(f'如果{a}大于{threshold}，则{b}等于9减去{a}；否则{b}等于{a}。')
                code.append(f'{b} = 9 - {a} if {a} > {threshold} else {a}')
            elif family=='mapping':
                mapping=list(range(1,9));rng.shuffle(mapping);value=mapping[value-1]
                table=', '.join(f'{k+1}: {v}' for k,v in enumerate(mapping))
                en.append(f'Use the lookup table {{{table}}} to map {a} to {b}.')
                zh.append(f'用映射表{{{table}}}将{a}的值映射为{b}。');code.append(f'{b} = {{{table}}}[{a}]')
            else:
                inc=rng.randrange(1,5);value=(value+inc-1)%8+1
                en.append(f'Add {inc} to {a}, wrapping values above 8 back to 1, to obtain {b}.')
                zh.append(f'将{a}加{inc}，大于8时从1循环，结果记为{b}。');code.append(f'{b} = ({a} + {inc} - 1) % 8 + 1')
            edges.append({'source':a,'target':b,'operation':family,'step':j})
        # A deterministic interpreter checks code, not a model-generated answer.
        namespace={};exec('\n'.join(code),{'__builtins__':{}},namespace)
        assert namespace[names[depth]]==value
        group=f'program/{family}/{case:02d}'
        variants={'en':' '.join(en)+f' What is the value of {names[depth]}? Answer with one digit only.',
          'zh':''.join(zh)+f'{names[depth]}的值是多少？只回答一个数字。',
          'python':'What digit is printed by this Python program? Answer with one digit only.\n```python\n'+'\n'.join(code)+f'\nprint({names[depth]})\n```',
          'en_reordered':' '.join(list(reversed(en)))+f' Treat these as value definitions. What is the value of {names[depth]}? Answer with one digit only.'}
        for representation,text in variants.items():
            prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
            enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
            result.append({'sample_id':'b2732_'+ranked(group+'/'+representation)[:20],'source_group':group,
              'family':family,'representation':representation,'cohort':'program_'+representation,'language':'zh' if representation=='zh' else 'en',
              'split':split,'depth':depth,'kind':'controlled_program','original_text':text,'text':prompt,
              'target':str(value),'initial_value':initial,'program':'\n'.join(code),'relations':edges,
              'prompt_ids':enc['input_ids'],'token_offsets':enc['offset_mapping'],'anchors':[len(enc['input_ids'])-1],
              'target_ids':tok(str(value),add_special_tokens=False)['input_ids'],
              'provenance':'New deterministic experimental material, not original natural corpus; semantic family kept together across all four representations.'})
    assert len(result)==16*len(cases) and all(len(r['target_ids'])==1 for r in result)
    return result

def main():
    if (BASE/'material_frozen.json').exists():print('BINDING_MATERIAL_EXISTS',flush=True);return
    from transformers import AutoTokenizer
    from phase2728_rdc_law_material import natural_candidates
    start=time.monotonic();tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    old=gzread(LAW/'material.json.gz');oldconfirm=gzread(LAW/'confirmation_material.json.gz')
    strict=[dict(r,capture_mode='main',connected_held=connected(r)) for r in old if r['kind']=='natural' and r.get('retrospective_ud')]
    assert not any(r['connected_held'] for r in strict if r['split']=='train')
    compressed(BASE/'natural_discovery.json.gz',strict)
    used_components={c for r in old+oldconfirm for c in r.get('component_ids',[])}
    manifest=[];selection=[];occupied=set(used_components)
    for bank in ('gum','ewt'):
        candidates=natural_candidates(tok,bank,'test',manifest)
        for r in candidates:r['connected_held']=connected(r)
        for want in (True,False):
            pool=sorted([r for r in candidates if bool(r['connected_held'])==want],key=lambda r:ranked(r['text']))
            selected=[]
            for r in pool:
                if any(c in occupied for c in r['component_ids']):continue
                rr=dict(r,split='connected_test' if want else 'matched_test',cohort=bank,
                    sample_id='b2732_'+ranked(bank+'/'+r['text'])[:20],capture_mode='binding',
                    anchors=sorted(set([len(r['prompt_ids'])//2,len(r['prompt_ids'])-2])),
                    novelty='No component sentence in law main/confirmation. Same public treebank; all earlier historical exposure is not excluded.')
                selected.append(rr);occupied.update(r['component_ids'])
                if len(selected)>=32:break
            assert len(selected)>=8,(bank,want,len(selected))
            selection.extend(selected)
    train_groups={r['source_group'] for r in strict if r['split']=='train'}
    assert not train_groups & {r['source_group'] for r in selection}
    compressed(BASE/'natural_confirmation.json.gz',selection)
    prog=programs(tok);compressed(BASE/'program_material.json.gz',prog)
    save(BASE/'material_frozen.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),
      'discovery_rows':len(strict),'confirmation_counts':dict(Counter(r['cohort']+'/'+r['split'] for r in selection)),
      'program_counts':dict(Counter(r['split']+'/'+r['representation'] for r in prog)),
      'hashes':{p.name:sha(p) for p in [BASE/'natural_discovery.json.gz',BASE/'natural_confirmation.json.gz',BASE/'program_material.json.gz']},
      'sources':manifest,'connected_definition':'Two distinct dependent edges with specified relation types share the same (sentence_id, head word). Not mere whole-window co-occurrence.',
      'limitations':['Connected syntactic relations are not semantic program composition.','Reordered English definitions are a language-order variant, not a coherent-versus-random-token training control.','Program names are split-disjoint, but templates are shared except reordered form; no claim of independent natural language breadth.']})
    ledger('material',time.monotonic()-start)
    print('BINDING_MATERIAL_FROZEN',read(BASE/'material_frozen.json'),flush=True)

if __name__=='__main__':main()
