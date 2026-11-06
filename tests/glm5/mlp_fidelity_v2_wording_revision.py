"""Create new immutable source versions; never edit executed v2 materials or runner."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
from shared.dist_protocol.mechanism_trace import file_sha

def main():
    base=Path(__file__).parent;original=base/'mlp_fidelity_v2_materials.py';text=original.read_text(encoding='utf-8')
    changes={
      "According to Record {selected}, is its settlement {predicate}?":"Is the settlement described in Record {selected} in a {predicate} location?",
      "Does Record {selected} place its settlement in a {predicate} location?":"Does the settlement in Record {selected} have a {predicate} location?",
      "Is the settlement described by Record {selected} {predicate}, based only on the records?":"Is a {predicate} location specified for the settlement in Record {selected}?",
      "For Record {selected}, is {predicate} the stated location of its settlement?":"Does Record {selected} explicitly locate the settlement in a {predicate} area?",
      "Using Record {selected} as the sole authority, is its settlement {predicate}?":"Check the settlement description in Record {selected}: does it specify a {predicate} location?",
      "Read Record {selected}. Does the definition say its settlement is {predicate}?":"For the settlement named by Record {selected}, is the stated location {predicate}?",
      "'controlled-corpus.json'":"'controlled-v3-corpus.json'",
      "'materials-freeze.research.json'":"'materials-v3-freeze.research.json'"}
    for a,b in changes.items():assert text.count(a)>=1,a;text=text.replace(a,b)
    dest=base/'mlp_fidelity_v2_materials_v3.py';dest.write_text(text,encoding='utf-8')
    runner=base/'mlp_fidelity_v2_collect.py';r=runner.read_text(encoding='utf-8')
    for a,b in {"'controlled-corpus.json'":"'controlled-v3-corpus.json'","'pilot-result.json'":"'pilot-v3-result.json'","part+'-execution-freeze.json'":"'v3-'+part+'-execution-freeze.json'","P.status(part,":"P.status('v3_'+part,","P.status('pilot',":"P.status('pilot_v3',","choices=['pilot','natural','controlled']":"choices=['pilot','controlled']"}.items():assert a in r,a;r=r.replace(a,b)
    output=base/'mlp_fidelity_v2_collect_v3.py';output.write_text(r,encoding='utf-8')
    P.save(P.PUBLIC/'wording-v3-revision.research.json',{'kind':'real','run_id':P.RUN,'model_snapshot':P.read(P.OUT/'plan.json')['model_snapshot'],'provenance':P.read(P.OUT/'plan.json')['provenance'],'reason':'v2 discoverypilot failed8/32cells, concentrated pronounsentence. v3 usesexplicit settlement/recordrelation; thresholds,truths, codefactorization andsplits unchanged. Heldout outputs notread.','v2_failed_result_sha256':file_sha(P.OUT/'pilot-result.json'),'sources':{str(p.relative_to(P.ROOT)):file_sha(p) for p in [original,runner,Path(__file__),dest,output]}})
    print({'new_materials':str(dest),'new_runner':str(output)},flush=True)
if __name__=='__main__':main()
