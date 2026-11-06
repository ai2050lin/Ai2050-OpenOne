"""Freeze authored content positives and known semantic corruption controls."""
import hashlib
import random
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import phase2790_materials as M
import phase2773_w6_loop as W
OUT=W.ROOT/'tests/glm5/result/phase2790_register'


def main():
    spec=W.read(OUT/'calibration-blind.json');records=[];private=[]
    for i in range(5200,5204):
        f=M.facts(i);core={k:f[k] for k in ['actor','subject','place','time','conditional_rule','invariant']}
        for style in ['formal','spoken']:
            positive=(f'本次{f["subject"]}由{f["actor"]}负责，定于{f["time"]}在{f["place"]}进行。'
                f'相关条件已经确认：{f["conditional_rule"]}。此外，{f["invariant"]}。这些限制与安排应一并保留。') if style=='formal' else (
                f'咱们说一下这回的{f["subject"]}吧。{f["actor"]}负责，时间是{f["time"]}，地点或者对象就是{f["place"]}。'
                f'你记着，{f["conditional_rule"]}。还有啊，{f["invariant"]}，这个可别弄反了。')
            variants={'positive':positive,'wrong_actor':positive.replace(f['actor'],'另一位未登记的负责人'),
                'wrong_time':positive.replace(f['time'],'下个月的最后一天'),
                'reversed_constraint':positive.replace(f['invariant'],'原先的限制不再执行，可以任意更改这些规则')}
            for mutation,text in variants.items():
                oid=hashlib.sha256(f'semantic2790:{i}:{style}:{mutation}'.encode()).hexdigest()[:20]
                records.append({'id':oid,'text':text,'facts':core})
                private.append({'id':oid,'style':style,'meaning':'preserved' if mutation=='positive' else 'changed','mutation':mutation,'domain':f['domain']})
    random.Random(27902027).shuffle(records)
    doc={**{k:v for k,v in spec.items() if k!='records'},'purpose':'独立AI保义评分校准；作者编制的正例与可核对事实变更，非模型研究结果','records':records}
    W.save(OUT/'semantic-controls-blind.json',doc,True);W.save(OUT/'semantic-controls-private.json',private,True)
    W.save(OUT/'semantic-controls-freeze.json',{'input_sha256':W.sha(OUT/'semantic-controls-blind.json'),
        'script_sha256':W.sha(__file__),'preregistered_criterion':'each rater meaning accuracy>=.90 and detects>=.90 known corruptions; style accuracy descriptive; authored controls do not prove general semantic scoring',
        'source':'authored facts/positives with explicit actor/time/constraint replacements','n':len(records)},True)
    print({'records':len(records),'input_sha256':W.sha(OUT/'semantic-controls-blind.json')})


if __name__=='__main__':main()
