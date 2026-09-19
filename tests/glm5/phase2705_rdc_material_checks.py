"""Preflight long tasks and exact-reference scorer calibration before generation."""
from rdc_conditional_common import *
from rdc_long_material import score
from phase2705_rdc_long_capture import prepare


def main():
    rows=prepare();fail=[]
    for r in rows:
        s=score(r,r['reference'],True,False)
        if not all(s[k] for k in ('exact_reference','declared_content_constraints','order_constraints','format_structure')):fail.append({'sample_id':r['sample_id'],'score':s})
    assert not fail,fail
    wrong=[]
    for r in rows:
        if r['family']=='temporal_revision':
            changed=r['reference'].replace('=', '=999',1);s=score(r,changed,True,False);assert not s['declared_content_constraints'];wrong.append(r['sample_id'])
    save(CAMPAIGN/'k_long/material_checks.json',{'timestamp':stamp(),'reference_scoring_cases':len(rows),'all_passed':True,'corrupted_number_rejections':len(wrong),
      'length_range':[min(len(r['prompt_ids']) for r in rows),max(len(r['prompt_ids']) for r in rows)],'synthetic_software_check_only':True})
    print('LONG_MATERIAL_CHECKS_PASS',len(rows),flush=True)


if __name__=='__main__':main()
