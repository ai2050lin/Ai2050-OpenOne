"""Conservative complete-answer scoring, separate from first-token diagnostics."""
import re

def score(row,text,ids,stop,cap):
    plain=text.strip();target=str(row.get('target',''));parsed=None;kind=row['kind']
    if kind=='controlled_program':
        # Pure standalone copy of the prior conservative terminal grammar;
        # importing this scorer never initializes an older research campaign.
        for pattern in (r'^\s*([1-8])\s*[。.!]?\s*$',
          r'(?:answer(?:\s+is)?|final(?:\s+answer)?(?:\s+is)?|答案(?:是|为)?|结果(?:是|为)?)\s*[:：=]?\s*\*{0,2}([1-8])\*{0,2}[。.!]?\s*$',
          r'\\boxed\{([1-8])\}[。.!]?\s*$'):
            match=re.search(pattern,plain,re.I)
            if match:parsed=match.group(1);break
    elif kind=='controlled_language':
        for pattern in (r'^\s*(yes|no|是|否)[.!。]?\s*$',
                        r'answer\s*[:：]\s*\*{0,2}(yes|no)\*{0,2}[.!。]?\s*$',
                        r'答案\s*[:：]\s*[“「"]?\*{0,2}(是|否)\*{0,2}[”」"]?[.!。]?\s*$'):
            match=re.search(pattern,plain,re.I)
            if match:parsed=match.group(1);break
    stopped=bool(ids and ids[-1] in stop);correct=None if parsed is None else parsed.casefold()==target.casefold()
    return {'conservative_final_answer':parsed,'conservative_final_correct':correct,
      'parsed_and_stopped_correct':bool(correct and stopped),'strict_answer_only':plain==target if kind!='natural' else None,
      'EOS':stopped,'censored':bool(not stopped and len(ids)>=cap),
      'scope':'Only whole-answer or explicit terminal marker parsed; never an arbitrary leading Yes/No or last digit in reasoning. Censored terminal-looking suffix is not asserted final. Reasoning correctness not graded.'}

def checks():
    cases=[('4','4','controlled_program','4'),('Answer: **4**','4','controlled_program','4'),('v_3 = 4\nContinue','4','controlled_program',None),
      ('Yes','Yes','controlled_language','Yes'),('No, this is wrong. Answer: Yes','Yes','controlled_language','Yes'),
      ('Yes, let us work through it','Yes','controlled_language',None),('because...\nAnswer: **No**','No','controlled_language','No'),
      ('否','否','controlled_language','否'),('这是因为关系相反。答案：否','否','controlled_language','否'),
      ('是，但需要继续推导','是','controlled_language',None)]
    for text,target,kind,expected in cases:
        result=score({'target':target,'kind':kind},text,[7],{7},128);assert result['conservative_final_answer']==expected,(text,result)
    return len(cases)
