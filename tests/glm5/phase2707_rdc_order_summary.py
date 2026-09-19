"""Auditable paired summaries, boundary coverage, and actual whole-source comparisons."""
from rdc_conditional_common import *
from rdc_order_material import FAMILIES,ORDERS
OUT=CAMPAIGN/'m_order'


def main():
    prefixes=read(OUT/'prefixes.json');behavior={r['sample_id']:read(OUT/f'behavior_scored/{r["sample_id"]}.json') for r in prefixes}
    records={}
    for r in prefixes:records.setdefault(r['base_id'],{})[r['order']]=r
    totals=[];pairs=[];common=[]
    keys=('result_correct','trace_correct','all_content_correct','field_order_correct','format_structure','eos','truncated')
    for order in ORDERS:
        bb=[behavior[r['sample_id']] for r in prefixes if r['order']==order]
        totals.append({'order':order,'n':len(bb),**{k:sum(b['scores'][k] for b in bb) for k in keys},'valid_boundaries':sum(b['result_boundary_state'] is not None for b in bb)})
    for family in FAMILIES:
      for language in ('en','zh'):
       rr=[values for values in records.values() if values['result_first']['family']==family and values['result_first']['language']==language]
       for left,right in (('result_first','trace_first'),('result_first','neutral_first'),('trace_first','neutral_first')):
        for key in ('result_correct','trace_correct','all_content_correct'):
            counts={'both_correct':0,'left_only':0,'right_only':0,'both_wrong':0}
            for r in rr:
                a,b=[behavior[r[o]['sample_id']]['scores'][key] for o in (left,right)]
                counts['both_correct' if a and b else 'left_only' if a else 'right_only' if b else 'both_wrong']+=1
            pairs.append({'family':family,'language':language,'left':left,'right':right,'outcome':key,'n':len(rr),**counts})
    for family in FAMILIES:
      for left,right in (('result_first','trace_first'),('result_first','neutral_first')):
        paired=[r for r in records.values() if r[left]['family']==family and all(behavior[r[o]['sample_id']]['result_boundary_state'] is not None for o in (left,right))]
        for layer in (11,23,35):
            sides=[]
            for order in (left,right):
                aa=[read(OUT/f'accounts/{behavior[r[order]["sample_id"]]["result_boundary_state"]}.json') for r in paired]
                ll=[next(l for l in a['layers'] if l['layer']==layer) for a in aa]
                sides.append({'order':order,'n':len(ll),
                  'mean_projection_share':{g:float(np.mean([l['projection_share_along_actual_attention'][g] for l in ll])) for g in read(OUT/'source_result.json')['source_groups']},
                  'mean_head_attention_mass':{g:float(np.mean([l['mean_head_attention_mass'][g] for l in ll])) for g in read(OUT/'source_result.json')['source_groups']}})
            common.append({'family':family,'layer':layer,'left':left,'right':right,'common_base_records':len(paired),'sides':sides})
    scoring=read(OUT/'scoring_alignment_audit.json');changes=scoring['changes'];bounds=[c for c in changes if c['old_boundary']!=c['new_boundary']]
    scorechanges=[c for c in changes if c['score_changes']]
    sources=read(OUT/'source_result.json');assert len(sources['accounts'])==241
    save(OUT/'summary_audit.json',{'phase':2707,'timestamp':stamp(),'source_sha':sha(Path(__file__)),'orders':totals,'paired_behavior':pairs,
      'scoring_changes':{'changed_prefixes':len(changes),'boundary_changed':len(bounds),'content_score_changed':len(scorechanges),'onset_missing':len(scoring['missing'])},
      'paired_source_comparisons':common,'source_accounts':len(sources['accounts']),'native_layers':3,'max_abs_source_account_error':sources['max_abs_account_error'],
      'limits':['Pairs share facts, entities and requested fields but not all prompt tokens, positions or generated histories.','Boundary comparisons use only pairs with both observed onsets; all288 remain behavioral denominators.','Eight entitygroups total, two heldout for fitted readers; no independent-token population confidence claim.','Attention mass and signed projection share are different quantities; neither is a causal importance fraction.']})
    print('ORDER_SUMMARY',totals,flush=True)


if __name__=='__main__':main()
