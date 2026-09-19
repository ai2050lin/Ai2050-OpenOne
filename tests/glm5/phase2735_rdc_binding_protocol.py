"""Freeze an information-bearing follow-up after observing first-token format mismatch."""
from rdc_binding_common import *

def main():
    from transformers import AutoTokenizer
    from phase2732_rdc_binding_material import programs
    out=BASE/'format_content'
    if (out/'protocol.json').exists():return
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True)
    future=programs(tok,range(48,56));compressed(out/'prospective_material.json.gz',future)
    old=programs(tok)
    original=gzread(BASE/'program_material.json.gz')
    assert json.loads(json.dumps(old,ensure_ascii=False))==original,'Default material must remain exactly reproducible after parameterizing case range.'
    ids=read(BASE/'beta_updates/protocol.json')['sample_ids']
    protocol={'timestamp':stamp(),'source':snapshot(Path(__file__)),
      'trigger':'Original English/Python first outputs often explain rather than emit the requested digit; 8-token censoring prevents content inference.',
      'classification':'Post-hoc diagnostic on existing768 programs; new128 depth6 cases frozen before any model response.',
      'equation':'Lfull = Lcontent + Lformat; Lcontent=-log(p_y/sum_D p), Lformat=-log(sum_D p), D=single digit tokens1..8.',
      'new_material_sha':sha(out/'prospective_material.json.gz'),'new_rows':128,'new_cases':32,
      'long_baseline_old_ids':ids,'generation_limit':128,
      'gradient_fit':'Original train96 English examples; code span all96 original Python train examples. No future targets used in direction fitting.',
      'full_parameter_directions':['content_EN_projected_to_code_content_span','mean_EN_format','random_control'],
      'step_norms':[.02,.10],'autonomous_primary_step_norm':.02,
      'autonomous_existing_programs':'First case per family, heldout split and representation from Beta64 =>32 rows.',
      'autonomous_future_programs':'First case per family and representation =>16 rows.',
      'autonomous_max_new_tokens':64,
      'reporting':['Unmodified native baseline, strict digit answer, conservative explicitly marked final digit, stop/censor, full text, candidate conditional score.',
        'Conditional digit scoring is not natural generation accuracy. Loss decomposition is exact but gradient parts are not orthogonal.',
        'No inferring original training history, generic hallucination repair or language closure.'],
      'finite_resource_scope':'Same authorized goal; one automatic follow-up within existing12GiB/21600s caps.'}
    immutable(out/'protocol.json',protocol)
    print('FORMAT_CONTENT_PROTOCOL_FROZEN',len(future),flush=True)

if __name__=='__main__':main()
