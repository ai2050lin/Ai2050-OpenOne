"""Text/token audit of next broad expansion, not frozen or executed model data."""
import os,sys
from collections import Counter
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from transformers import AutoTokenizer
from phase2620_native_coordinate_contract import RESULT,read,save
from phase2681_fresh_source_material import build
from phase2677_source_role_material import evaluate


def main():
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    rows=build(tok);grouped={};content={}
    for r in rows:
        key=tuple(r[k] for k in ('family','language','unit','content_instance','form','mention_order','target_index'))
        grouped.setdefault(key,[]).append(r)
        ck=tuple(r[k] for k in ('family','language','unit','form','mention_order','target_index','output_function'))
        content.setdefault(ck,{})[r['content_instance']]=r['body']
        assert ''.join(s['text'] for s in r['character_regions'])==r['prompt']
        assert evaluate(r,r['target'])['strict_correct'] and not evaluate(r,r['alternate'])['content_correct']
    previous=[r for p in ('phase2670_native_mlp_contract','phase2676_native_mlp_delivery/expansion','phase2677_source_role_contract') for r in read(RESULT/p/'material/cases.json')]
    oldnames={r[k] for r in previous for k in ('entity_a','entity_b')};names={r[k] for r in rows for k in ('entity_a','entity_b')}
    def skeleton(r):return r['body'].replace(r['entity_a'],'{{A}}').replace(r['entity_b'],'{{B}}')
    old={(r['family'],r['language'],skeleton(r)) for r in previous}
    new_content={(r['family'],r['language']):[] for r in rows}
    for r in rows:new_content[r['family'],r['language']].append((r['family'],r['language'],skeleton(r)) not in old)
    checks={'4096_unique':len(rows)==len({r['prompt'] for r in rows})==4096,'1024_samebody_fourfunctions':len(grouped)==1024 and all(len(g)==4 and len({tuple(r['prompt_ids'][:r['body_end_token']+1]) for r in g})==1 for g in grouped.values()),
        '2048_actual_content_pairs':len(content)==2048 and all(v[0]!=v[1] for v in content.values()),'allnames_new_vs2670_2676_2677':not(names&oldnames),
        'all_body_skeletons_new_vs_declared_prior':all(all(v) for v in new_content.values()),'64_fullH16_fullMLP':sum(r['published'] for r in rows)==64 and sum(r['parameter_published'] for r in rows)==16,
        '512_optional_source_panel':sum(r['source_selected'] for r in rows)==512,'fixed160_with16generation':max(len(r['prompt_ids']) for r in rows)+16<=160}
    report={'all_checks_passed':all(checks.values()),'checks':checks,'lengths':[min(len(r['prompt_ids']) for r in rows),max(len(r['prompt_ids']) for r in rows)],
        'family_language_novel_body_counts':{str(k):[sum(v),len(v)] for k,v in new_content.items()},
        'examples':[{k:r[k] for k in ('case_id','body','text','prefill','target')} for r in rows if r['published']],
        'scope':'Prospective text/tokenizer only. Reuses controlledstructuraltemplates withnewnamedentities and lexical/contentfills inall8families. Novelbodystring isnot proof of newabstractsemantics. No formal2681freeze, no modelobservations. Formalplanmustreviewactual2679/2680patterns andfreshdiskbudget.'}
    save(RESULT/'phase2677_source_role_contract/analysis/fresh_material_preflight.json',report);print(checks,report['lengths']);assert report['all_checks_passed']


if __name__=='__main__':main()
