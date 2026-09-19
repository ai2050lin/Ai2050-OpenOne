"""Natural-shape nonquantized relation corpus capture using previously validated hooks."""
import argparse
from rdc_mechanism_common import *
from rdc_relation_material import build,FAMILIES
import phase2693_rdc_language_capture as capture

RUN='b_relations';OUT=CAMPAIGN/RUN
def prepare():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    rows=build(tok);immutable(OUT/'material.json',rows)
    protocol={'version':1,'phase':2696,'samples':512,'families':list(FAMILIES),'base_cases':64,
        'factors':'8 task families x8 base cases x2 support states x2 question polarities x2 language implementations',
        'model':'qwen3-4b BF16 nonquantized CUDA','material_sha':sha(OUT/'material.json'),
        'source_sha':sha(Path(__file__)),'material_code_sha':sha(ROOT/'tests/glm5/rdc_relation_material.py'),
        'capture_implementation_sha':sha(Path(capture.__file__)),
        'splits':'Group by base case across fact/polarity/translation: indices0..3 train,4..5validation,6..7test within each family.',
        'targets':['positive statement supported','requested Yes/No','actual natural first-answer preference'],
        'capture':'All actual prompt tokens, H0..36 full2560; native layers0/11/23/35 full coordinate at U,V,last union; max16 natural generated tokens; same-shape16 noops.',
        'disk_upper_bf16_residual_bytes':sum(len(r['prompt_ids'])*37*2560*2 for r in rows),
        'limits':['No strict token-ID disjointness; ordinary grammatical words shared.',
          'Word-sense examples use language-appropriate ambiguity, not guaranteed equivalent bilingual semantics.',
          'False taxonomy records explicitly counterfactual and clearly instruction-scoped.',
          'Negative query is not-supported judgment under given record policy, not universal negation semantics.',
          'No causal patches or forced disjoint semantic dictionary.']}
    immutable(OUT/'protocol.json',protocol);return rows,protocol

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,choices=(16,512),default=16);args=p.parse_args()
    capture.RUN=RUN;capture.OUT=OUT;capture.prepare=prepare;capture.status=announce;capture.event=events
    capture.main(args.limit)
