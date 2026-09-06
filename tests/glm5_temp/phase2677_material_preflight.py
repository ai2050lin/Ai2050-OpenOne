"""Prospective material/tokenizer audit; not phase freeze or model evidence."""
import os,sys
from collections import Counter
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from transformers import AutoTokenizer
from phase2620_native_coordinate_contract import RESULT,save,read
from phase2677_source_role_material import build,evaluate


def main():
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True);rows=build(tok)
    panel=[r for r in rows if r['source_selected']];groups={};roles=Counter()
    for r in panel:
        key=tuple(r[k] for k in ('family','language','unit','content_instance','target_index'))
        groups.setdefault(key,[]).append(r)
    for r in rows:
        roles.update(t['role'] for t in r['token_regions'])
        assert len(r['token_regions'])==len(r['prompt_ids'])
        assert ''.join(x['text'] for x in r['character_regions'])==r['prompt']
        assert evaluate(r,r['target'])['strict_correct'] and not evaluate(r,r['alternate'])['content_correct']
    prior_names={r[k] for path in (RESULT/'phase2670_native_mlp_contract/material/cases.json',RESULT/'phase2676_native_mlp_delivery/expansion/material/cases.json') for r in read(path) for k in ('entity_a','entity_b')}
    new_names={r[k] for r in rows for k in ('entity_a','entity_b')}
    checks={'8448_unique':len(rows)==len({r['prompt'] for r in rows})==8448,
            '8192_truth_plus256_name_cloze':len(rows[:8192])==8192 and all(r['output_function'] in ('name','cloze') for r in rows[8192:]),
            '512_source_cells':len(panel)==512,'128_exact_body_four_functions':len(groups)==128 and all(len(g)==4 and len({r['body'] for r in g})==1 and len({r['output_function'] for r in g})==4 for g in groups.values()),
            'all_source_prefix_ids_identical':all(len({tuple(r['prompt_ids'][:r['body_end_token']+1]) for r in g})==1 for g in groups.values()),
            '64_published':sum(r['published'] for r in rows)==64,'no_namemode_binary_label':all(r['expected_yes'] is None for r in rows[8192:]),
            'names_disjoint_from2670_and2676expansion':not(new_names&prior_names)}
    assert all(checks.values()),checks
    result={'all_checks_passed':True,'checks':checks,'token_range':[min(len(r['prompt_ids']) for r in rows),max(len(r['prompt_ids']) for r in rows)],
            'token_role_counts':dict(roles),'source_function_counts':dict(Counter(r['output_function'] for r in panel)),
            'examples':[{k:r[k] for k in ('case_id','body','text','prefill','prompt','target','token_regions')} for r in rows if r['published'] and r['family']=='word_sense'],
            'scope':'Prospective tokenizer/text-only preflight. New entities disjoint from2670 and2676expansion, reused fact/content templates. One draft name Isolde overlapped2676expansion and was replaced withElowen before any newmodel forward or2677freeze. No pretrained model forward, no phase append, no frozen2677 claim; padding/protocol awaits completed2676 numeric controls. Fourfunction512 cells include256 existingtruth-grid cells.'}
    save(RESULT/'phase2676_native_mlp_delivery/analysis/next_material_preflight.json',result)
    print('PROSPECTIVE MATERIAL',checks,result['token_range'],result['source_function_counts'])


if __name__=='__main__':main()
