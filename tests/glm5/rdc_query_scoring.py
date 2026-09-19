"""New prospective terminal grammar; previous primary/secondary scorers unchanged."""
import re
from rdc_update_scoring import score as primary_score
from rdc_update_terminal_audit import enrich_score,unwrap,checks as old_checks

def score(row,text,ids,stop,cap):
    primary=primary_score(row,text,ids,stop,cap);secondary=enrich_score(row,text,primary);result=dict(secondary)
    result['primary']=primary;result['prior_secondary']=secondary
    if row['kind']=='controlled_program' and result['EOS'] and not result['censored'] and result['conservative_final_answer'] is None:
        plain=re.sub(r'\s*[✅✔☑]\s*$','',text.strip()).strip();value=None;reason=None
        # Previously audited terminal declarations, now frozen BEFORE new outputs.
        patterns=[r'(?:只输出(?:一个)?数字|输出的数字(?:是|为)?|the\s+digit\s+printed\s+is|the\s+output\s+is)\s*[:：]?\s*([\s\S]+)$',
          r'(?:^|\n)\s*(?:→|=>)\s*([^\n]+)$']
        for p in patterns:
            match=re.search(p,plain,re.I)
            if match:
                v=unwrap(match.group(1))
                if re.fullmatch(r'[1-8]',v):value=v;reason='frozen_terminal_declaration';break
        if value is None and plain!=text.strip():
            again=enrich_score(row,plain,primary_score(row,plain,ids,stop,cap))
            value=again['conservative_final_answer'];reason='terminal_decorative_check_removed'
        if value is not None:result.update(conservative_final_answer=value,conservative_final_correct=value==str(row['target']),parsed_and_stopped_correct=value==str(row['target']),format_audit_method=reason)
    result['scope']='Prospectively frozen primary+format-aware+explicit terminal declaration grammar. EOS required for extensions. Never infer answer from gold, arbitrary last digit, or unfinished reasoning. Reasoning chain ungraded.'
    return result

def checks():
    old=old_checks();count=0
    for text,expected in [('只输出一个数字：**4**。','4'),('The digit printed is: **3**.','3'),('The digit printed is:\n\n**7**','7'),
      ('Final Output:\n```python\nprint(x)\n```\n→ **4**','4'),('**Answer:** `4` ✅','4'),('Only continue with 4, then 5',None),
      ('The digit printed is: 4 or 5',None),('只输出一个数字：4，然后继续',None),('→ 4\nnot final',None)]:
        r=score({'kind':'controlled_program','target':'8','relations':[{'target':'v_160_4'}]},text,[99],{99},1024)
        assert r['conservative_final_answer']==expected,(text,r);count+=1
    return {'prior':old,'new_cases':count,'all_passed':True}

if __name__=='__main__':print(checks())
