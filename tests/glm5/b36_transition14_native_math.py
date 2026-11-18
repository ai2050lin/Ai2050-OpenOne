"""Native-study helpers. No model execution or scientific results in this file."""
import re
import numpy as np

def partitions(prompt_length,total_length,length=256):
    assert 1<=prompt_length<total_length<=length
    x=np.arange(length)
    out=np.stack([x<prompt_length,(x>=prompt_length)&(x<total_length-1),x==total_length-1])
    assert np.array_equal(out.sum(0),(x<total_length).astype(int))
    return out

def subset_mask(parts,bits):
    assert 0<=bits<8
    return np.any(parts[[i for i in range(3) if bits&(1<<i)]],axis=0) if bits else np.zeros(parts.shape[1],dtype=bool)

def mobius(values):
    """Exact Boolean-lattice decomposition; values[0] is the constant term."""
    values=np.asarray(values);assert values.shape[0]==8;out=values.copy()
    for bit in range(3):
        for mask in range(8):
            if mask&(1<<bit):out[mask]-=out[mask^(1<<bit)]
    return out

def locate_sites(row,result,tok):
    """Locate all sites in actual generated tokens, without retaining Code itself."""
    text=result['text'];ids=result['tokens'];sm=re.search(r'(?m)^Steps:[ \t]*([0-9]+(?:[ \t]*,[ \t]*[0-9]+)*)',text)
    vm=re.search(r'(?m)^Value:[ \t]*(\d+)',text);cm=re.search(r'(?m)^Code:[ \t]*([ABCD])',text)
    if not sm or not vm or not cm:return None
    nums=list(re.finditer(r'\d+',sm[1]))
    if len(nums)<row['depth']:return None
    charsites=[sm.start(1)+nums[row['depth']-1].start(),vm.start(1),cm.start(1)]
    spans=[];last=0
    for i in range(len(ids)):
        end=len(tok.decode(ids[:i+1],skip_special_tokens=False));spans.append((last,end));last=end
    at=[]
    for char in charsites:
        matches=[i for i,(start,end) in enumerate(spans) if start<=char<end]
        if len(matches)!=1:return None
        at.append(matches[0])
    lo,hi=spans[at[-1]];lead=text[lo:charsites[-1]];trail=text[charsites[-1]+1:hi]
    choice_strings=[lead+c+trail for c in 'ABCD'];choices=[tok.encode(s,add_special_tokens=False) for s in choice_strings]
    if any(len(c)!=1 for c in choices) or len({c[0] for c in choices})!=4:return None
    if choices['ABCD'.index(cm[1])][0]!=ids[at[-1]]:return None
    prefix=row['tokens']+ids[:at[-1]];sites=[len(row['tokens'])-1]+[len(row['tokens'])+i-1 for i in at]
    if not sites[0]<=sites[1]<=sites[2]<=sites[3] or sites[-1]!=len(prefix)-1:return None
    if tok.decode(prefix,skip_special_tokens=False)!=row['prompt']+text[:lo]:return None
    return dict(id=row['id'],family=row['family'],arm=row['arm'],depth=row['depth'],truth=row['truth'],source_tokens=row['tokens'],prompt_length=len(row['tokens']),tokens=prefix,sites=sites,site_names=['prompt_end','before_dth_printed_step','before_value','before_code'],choice_ids=[c[0] for c in choices],choice_strings=choice_strings,gold='ABCD'.index(row['code']),observed_token=ids[at[-1]],source_generation_id=result['id'],source_code=cm[1],generated_prefix=text[:lo])
