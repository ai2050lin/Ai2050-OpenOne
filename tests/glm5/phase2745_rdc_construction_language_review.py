"""Persist the main-agent's complete post-outcome review of unparsed GLM EOS."""
from rdc_construction_common import *


def main():
    out=BASE/'native_language/glm4/unparsed_EOS_adjudication.json'
    if out.exists():return
    original=BASE/'native_language/glm4/records.json.gz'
    records=gzread(original)
    chosen=[r for r in records if r['answer_scoring']['EOS'] and r['answer_scoring']['conservative_final_answer'] is None]
    assert len(chosen)==32
    assert {(r['case'],r['world']) for r in chosen}=={(c,w) for c in range(16) for w in range(2)}
    assert all(r['family']=='knowledge_chain' and r['language']=='zh' and r['generated_text'].strip().startswith('是。') for r in chosen)
    # This assertion guards the explicitly reviewed set. It is NOT a new
    # general parser based on leading tokens; the full32texts were read.
    review=[]
    for r in chosen:
        review.append({'sample_id':r['sample_id'],'original_full_text':r['generated_text'],'target':r['target'],
            'reviewed_explicit_final_conclusion':'是','conclusion_correct':r['target']=='是',
            'format_follows_answer_only':False,'original_EOS':True,
            'rationale':'Main agent read the entire completed response; all32 affirm the requested membership with an explanation and no later retraction. This records that text judgment, not a generic leading-Yes parser. Full reasoning chain not graded.'})
    assert sum(r['conclusion_correct'] for r in review)==16
    native=read(BASE/'native_language/glm4/result.json')
    primary=next(r for r in native['summary'] if r['family']=='all')
    save(out,{'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'records_sha256':sha(original),
        'status':'unblinded_post_outcome_main_agent_text_adjudication_not_preregistered_or_independent',
        'reviewed_rows':32,'additional_explicit_correct_conclusions':16,'additional_explicit_wrong_conclusions':16,
        'unchanged_primary_correct_and_stopped':primary['correct_and_stopped'],
        'secondary_conclusion_and_stopped_count':primary['correct_and_stopped']+16,
        'secondary_both_worlds_count':primary['both_worlds_correct_and_stopped'],
        'reviews':review,'scope':'No original output, fixed parser, primary count or target altered. Correct conclusion and requested format distinguished; not comprehensive reasoning-chain accuracy.'})
    print('NATIVE_LANGUAGE_POSTHOC_REVIEW',32,16,16,flush=True)


if __name__=='__main__':main()
