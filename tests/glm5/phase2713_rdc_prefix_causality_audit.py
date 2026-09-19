"""Token-byte boundary causality: character offsets can expose an unfinished UTF-8 character."""
from tokenizers import Tokenizer
from rdc_prefix_estimators import *


def main():
    tok=Tokenizer.from_file(str(ROOT/'models/hf/qwen3-4b/tokenizer.json'))
    records=[];overrides=[];descriptor_changes=[];anchor_changes=[]
    for run,material in [('qwen4','material_stratified.json'),('qwen4_confirmation','confirmation_material.json')]:
        for r in read(CAMPAIGN/material):
            for k,p in enumerate(r['positions']):
                previous=r['anchor_graphs'][k]
                visible=tok.decode(r['prompt_ids'][:p+1],skip_special_tokens=False)
                causal=prefix_graph(visible,p,r['language'])
                a=descriptor(previous);b=descriptor(causal)
                diff=np.flatnonzero(a!=b).tolist()
                if diff:descriptor_changes.append({'run':run,'sample_id':r['sample_id'],'array_index':k,'descriptor_coordinates':diff})
                if visible!=previous['observed_prefix']:
                    overrides.append({'run':run,'sample_id':r['sample_id'],'array_index':k,'position':p,
                      'old_char_offset_prefix':previous['observed_prefix'],'causal_graph':causal,'descriptor_differences':diff})
                    if k in (0,3):anchor_changes.append({'run':run,'sample_id':r['sample_id'],'array_index':k})
                records.append({'run':run,'sample_id':r['sample_id'],'array_index':k,'descriptor_equal':not diff,
                  'exact_text_equal':visible==previous['observed_prefix']})
    save(CAMPAIGN/'causal_prefix_audit.json',{'timestamp':stamp(),'source_sha':sha(Path(__file__)),
      'positions':len(records),'offset_text_mismatches':len(overrides),'main_anchor_hash_mismatches':len(anchor_changes),
      'descriptor_changed_positions':len(descriptor_changes),'descriptor_changes':descriptor_changes,
      'records':records,'status':'correction_needed' if overrides else 'passed',
      'explanation':'A tokenizer token can end mid UTF-8 character; offset_mapping points to the complete character, but only decode(token_ids[:p+1]) is prefix-visible. Gold/text spans remain retrospective metadata. Check actual numeric descriptor equality rather than assuming no impact.',
      'primary_predictor_scope':'All non-hash kernels use numeric descriptors, native H and token embedding. If descriptors are exactly equal at every source position, their frozen predictions are unaffected. Random-prefix-hash controls and displayed prefix text need corrected versions; never silently overwrite frozen artifacts.',
      'anchor_hash_mismatch_records':anchor_changes})
    save(CAMPAIGN/'causal_graph_overrides.json',overrides)
    print('PREFIX_CAUSALITY',len(records),len(overrides),len(anchor_changes),len(descriptor_changes),descriptor_changes[:10],flush=True)


if __name__=='__main__':main()
