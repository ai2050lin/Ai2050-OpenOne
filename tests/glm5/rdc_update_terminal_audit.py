"""Secondary, format-aware terminal audit; leaves the frozen primary scorer intact."""
import re


def unwrap(value):
    value=value.strip()
    for _ in range(10):
        old=value
        value=re.sub(r'[。.!]\s*$','',value).strip()
        fence=re.fullmatch(r'```(?:[a-zA-Z0-9_-]+)?[ \t]*\n([\s\S]*?)\n```',value)
        if fence:value=fence.group(1).strip()
        for opening,closing in (('**','**'),('`','`'),('$$','$$'),('\\[','\\]'),('\\(','\\)'),('$','$')):
            if value.startswith(opening) and value.endswith(closing) and len(value)>len(opening)+len(closing):
                value=value[len(opening):-len(closing)].strip();break
        box=re.fullmatch(r'\\boxed\{([^{}]+)\}',value)
        if box:value=box.group(1).strip()
        if value==old:break
    return value


def literal(value,variable=None):
    value=unwrap(value)
    if re.fullmatch(r'[1-8]',value):return value
    if variable:
        match=re.fullmatch(r'(.+?)\s*(?:=|的值是|的值为|\bis\b|\bhas\s+value\b)\s*(.+)',value,flags=re.I)
        if match:
            lhs=re.sub(r'[{}$`*\s]','',match.group(1))
            rhs=unwrap(match.group(2))
            if lhs==variable and re.fullmatch(r'[1-8]',rhs):return rhs
    return None


def enrich_score(row,text,primary):
    result=dict(primary);reason='unchanged_primary'
    # No model answer is inferred from its target value. The query-variable
    # name comes from the frozen user task; gold is consulted only for grading.
    if row['kind']=='controlled_program' and primary['EOS'] and not primary['censored'] and primary['conservative_final_answer'] is None:
        variable=row.get('relations',[{}])[-1].get('target') if row.get('relations') else None
        plain=text.strip();parsed=None
        markers=list(re.finditer(r'(?:\bfinal\s+answer|\banswer|最终答案|答案|最终结果)\s*(?:\*\*)?\s*[:：]\s*(?:\*\*(?=\s|$))?\s*',plain,re.I))
        if markers:
            parsed=literal(plain[markers[-1].end():],variable)
            if parsed is not None:reason='explicit_terminal_marker_with_whole_wrapper'
        if parsed is None:
            # A final boxed literal is an explicit mathematical answer, not
            # the last arbitrary digit inside a derivation.
            box=re.search(r'\\boxed\{([1-8])\}(?:\s*(?:\$\$|\$|\\\]|\\\)|\*\*|[。.!]))*\s*$',plain)
            if box:parsed=box.group(1);reason='terminal_boxed_literal'
        if parsed is None and variable:
            # Only a complete final line assigning the actually requested
            # variable is eligible; other variables and trailing prose fail.
            line=plain.splitlines()[-1].strip()
            line=re.sub(r'^(?:(?:所以|因此)[，,:：]?|(?:therefore|thus|so)[,:]?)\s*','',line,flags=re.I)
            value=unwrap(line)
            if any(token in value for token in ('=','的值',' is ',' has value ')):
                parsed=literal(value,variable)
                if parsed is not None:reason='complete_final_requested_variable_value'
        if parsed is not None:
            correct=parsed==str(row.get('target',''))
            result.update(conservative_final_answer=parsed,conservative_final_correct=correct,parsed_and_stopped_correct=correct)
    result['scope']=primary['scope']+' Secondary format-aware audit additionally accepts a complete marked answer in Markdown/LaTeX, a terminal boxed literal, or an exact final value statement of the requested variable. Original primary scoring remains separately retained; no reasoning-chain correctness claim.'
    result['format_audit_method']=reason
    return result


def checks():
    from rdc_update_scoring import score,checks as legacy_checks
    count=legacy_checks()
    cases=[
      ('Final Answer:\n$$\n\\boxed{4}\n$$','4','4'),
      ('Final Answer:\n```\n3\n```','3','3'),
      ('**Answer:**\n```\n3\n```','3','3'),
      ('**Final Answer**: **4**','4','4'),
      ('答案：\n\\[\\boxed{4}\\]','4','4'),
      ('所以，**v_160_4 = 4**。','4','4'),
      ('所以，$v_{160_4}$ 的值是 **4**。','4','4'),
      ('所以，$v_{160_3}$ 的值是 **4**。','4',None),
      ('Therefore, `v_160_4 = 4`.','4','4'),
      ('Final Answer:\n$$\\boxed{5}$$','4','5'),
      ('The calculation was 4; next we continue','4',None),
      ('v_160_3 = 4','4',None),
      ('v_160_4 = 4 or 5','4',None),
      ('Final Answer:\n```\nprint(4)\n```','4',None),
      ('Final Answer: 4 or 5','4',None),
      ('Final Answer:\n$$\\boxed{4}$$\nBut the derivation is unfinished','4',None),
      ('Final Answer:\n```\n4\n```\n3','4',None)]
    for text,target,expected in cases:
        row={'kind':'controlled_program','target':target,'relations':[{'target':'v_160_4'}]}
        primary=score(row,text,[9],{9},1024);audited=enrich_score(row,text,primary)
        assert audited['conservative_final_answer']==expected,(text,audited)
        if expected is not None:assert audited['conservative_final_correct']==(expected==target)
    row={'kind':'controlled_program','target':'4'};text='Final Answer:\n$$\\boxed{4}$$'
    primary=score(row,text,[9],set(),1);assert enrich_score(row,text,primary)['conservative_final_answer'] is None
    return {'primary_regression_cases':count,'format_aware_cases':len(cases)+1,'passed':True}


if __name__=='__main__':print(checks())
