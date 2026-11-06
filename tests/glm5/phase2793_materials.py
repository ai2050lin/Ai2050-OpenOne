"""Frozen long-document entity colour arbitration, entity-level holdouts."""
from pathlib import Path
import hashlib
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).parent))
import phase2790_materials as BACK
import phase2773_w6_loop as W
from tools.power_analysis import check_phase

ROOT=W.ROOT;OUT=ROOT/'tests/glm5/result/phase2793_arbitration';PUBLIC=ROOT/'research/phase2793'
RUN='P2793-ENTITY-ARBITRATION-v1';MODEL=W.MODEL;SEED=27932026
COLORS=dict(zip(['red','orange','yellow','green','blue','purple','black','white','gray','brown','pink'],'红橙黄绿蓝紫黑白灰棕粉'))


def save(path,doc,immutable=False):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(doc,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)
    if immutable and path.exists():
        if path.read_text(encoding='utf-8')!=raw:raise ValueError('immutable file differs: '+str(path))
        return
    path.write_text(raw,encoding='utf-8')


def entities():
    rows=[]
    for line in (ROOT/'shared/dist_protocol/lexicon/ENT-XF_nouns.v2.tsv').read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'):continue
        fields=line.split('\t')
        if len(fields)>=9 and not fields[7] and fields[6] in COLORS:
            rows.append(dict(zip(['id','category','category_zh','super','zh','en','color','color_alt','size'],fields)))
    rows.sort(key=lambda r:hashlib.sha256(('2793entity:'+r['id']).encode()).hexdigest())
    assert len(rows)>=48
    chosen=rows[:48]
    for i,r in enumerate(chosen):
        r['split']='discovery' if i<16 else 'heldout_entity_frame'
        r['prior']=COLORS[r['color']]
        alternatives=[v for v in COLORS.values() if v!=r['prior']]
        r['document']=alternatives[int(hashlib.sha256(r['id'].encode()).hexdigest()[:8],16)%len(alternatives)]
    return chosen


def build():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);builder=BACK.Builder(tok);items=[]
    # Background consists of whole authored document sentences. No target facts in it.
    for e in entities():
        frames=[0,1] if e['split']=='discovery' else [2,3]
        for frame in frames:
            source=[f'虚构记录写道：{e["zh"]}在这份文档中被设定为{e["document"]}色。',
                f'材料中的设定如下：本页记载的{e["zh"]}颜色是{e["document"]}色。',
                f'这份虚构档案把{e["zh"]}记作{e["document"]}色，以下提问须区分材料与常识。',
                f'文档作者设定{e["zh"]}呈{e["document"]}色；这条设定仅属于材料中的情境。'][frame]
            tail='这条虚构设定不改变现实常识。\n'
            questions={scope:[f'现在只按{scope}回答：{e["zh"]}是什么颜色？',
                f'回答范围限定为{scope}：{e["zh"]}的颜色是什么？',
                f'请以{scope}为依据回答{e["zh"]}的颜色。',f'所问是{scope}中的颜色：{e["zh"]}是什么颜色？'][frame] for scope in ['文档','现实']}
            suffix='只回答一个颜色字（红橙黄绿蓝紫黑白灰棕粉之一），不加“色”字，不写解释。'
            wraps={s:builder.wrap('\n'+source+tail+q+'\n'+suffix) for s,q in questions.items()}
            # Same scope sentence differs only by 文档/现实; require actual token equality.
            sizes={s:len(tok.encode(p,add_special_tokens=False)) for s,p in wraps.items()}
            assert len(set(sizes.values()))==1,(frame,e['id'],sizes)
            pref=builder.prefixes('community',frame)
            for length in [512,1024,2048]:
                n=next(iter(sizes.values()));found=None
                for k in range(length-n-8,length-n+9):
                    if k not in pref:continue
                    texts={s:pref[k]+'\n'+source+tail+q+'\n'+suffix for s,q in questions.items()}
                    prompts={s:builder.wrap(t) for s,t in texts.items()};ids={s:tok.encode(p,add_special_tokens=False) for s,p in prompts.items()}
                    if all(len(v)==length for v in ids.values()):found=texts,prompts,ids;break
                if not found:raise ValueError('no exact whole-sentence length match')
                texts,prompts,ids=found
                for scope in ['文档','现实']:
                    token=ids[scope]
                    # Locate final entity and document-colour spans in actual prompt tokens.
                    src_start=prompts[scope].rindex(source);entity_start=src_start+source.index(e['zh'])
                    entity_end=entity_start+len(e['zh']);colour_start=src_start+source.index(e['document'])
                    def pos(end):return len(tok.encode(prompts[scope][:end],add_special_tokens=False))-1
                    positions=sorted({pos(entity_end),pos(colour_start+1),len(token)-1})
                    item={'id':f'{e["id"]}:F{frame}:L{length}:'+('doc' if scope=='文档' else 'world'),
                        'family_id':e['id'],'entity':e['zh'],'category':e['category'],'split':e['split'],'frame':frame,
                        'scope':scope,'length':length,'tokens':length,'text':texts[scope],'prompt':prompts[scope],'token_ids':token,
                        'prior':e['prior'],'document':e['document'],'expected':e['document'] if scope=='文档' else e['prior'],
                        'readout_positions':positions,'document_colour_position':pos(colour_start+1),'entity_position':pos(entity_end),
                        'source_text':source,'scope_question':questions[scope]}
                    items.append(item)
    assert len(items)==576 and len({x['family_id'] for x in items})==48
    assert all(len(tok.encode(v,add_special_tokens=False))==1 for v in COLORS.values())
    snapshot=W.read(ROOT/'tests/glm5/result/phase2790_register/freeze.json')['model_snapshot']
    c={'schema':'entity_arbitration_corpus.v1','kind':'real','run_id':RUN,'model_snapshot':snapshot,'precision':'bf16','seed':SEED,
        'items':items,'entities':entities(),'color_token_ids':{v:tok.encode(v,add_special_tokens=False)[0] for v in COLORS.values()},
        'source_lexicon_sha256':W.sha(ROOT/'shared/dist_protocol/lexicon/ENT-XF_nouns.v2.tsv')}
    p1=check_phase(.5,32,p_d=.7,alpha=.025);p2=check_phase(.35,32,p_d=.4,alpha=.025);assert p1['pass'] and p2['pass']
    prereg={'schema':'entity_arbitration_prereg.v1','kind':'real','run_id':RUN,'created_at':time.time(),'model_snapshot':snapshot,
        'primary':'heldout32 entities: difference of document preference under document versus reality scope, entity means over2 new frames x3 lengths',
        'H1':{'expected_delta':.5,'power_binary_entity_majority':p1,'criterion':'mean paired document-choice difference>=.50 and entity-bootstrap95% lower>.20; majority binary outcome additionally reported; model-specific controlled question-answering, not general language theory'},
        'H2':{'expected_delta':.35,'power_binary_entity_vote':p2,'criterion':'on32 heldout entities at F2/L1024/doc, ablating8 discovery-selected late heads lowers correct forced-choice by>=.20 and paired entity lowerCI>0; matched-block8 random heads smaller. Failures retained.'},
        'selection':'only discovery16; choose8 heads in blocks18..35 by mean document-minus-reality direct logit contribution. DLA freezes actual final RMS; it is direct attribution, not total causality.',
        'controls':'same-block matched random heads; source-free prior probe for every entity; identity/length controls and hs[0] saved; no reselect on holdout',
        'behavior_gate':'each scope heldout forced choice>=.80; report all cells and all inputs; prior probes and dataset keys separate; no posthoc row exclusion',
        'readout':'entire scope cue and conflicting fact visible before final readout; same assistant suffix across pairs. Entity/cue earlier positions have causal visibility flags, not tested for future scope encoding.',
        'native':'all37 hidden layers at registered readouts, all36 blocks fullMLP G/V/A and head output vectors; fixed deep8 examples also full-token hidden and postRoPE normalized Q/K plus groupedV; noTopK coordinate truncation',
        'deep_ids':[r['id'] for r in items if r['family_id'] in [entities()[0]['id'],entities()[16]['id']] and r['length'] in [512,2048] and r['frame'] in [0,2]],
        'generation':'fixed deep8 actual greedy max24 new tokens, cache continued without teacher forcing; generation fields at every actual step',
        'capability':'new head interventions additionally evaluated on frozen previous2200-keyed monitor if an effect is supported; otherwise no capability-preservation claim. Reference-only questions excluded.',
        'limits':['authored repeated document background, not natural corpus','length adds background content','lexicon colour labels may fail; prior probes distinguish model knowledge errors','primary restricted forced-choice distribution; free generations descriptive','one local physical node','no cross-model generalization'],
        'materials_sha256':W.sha(__file__),'corpus_sha256':hashlib.sha256(json.dumps(c,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
    save(OUT/'corpus.json',c,True);save(PUBLIC/'prereg.research.json',prereg,True)
    print(json.dumps({'inputs':len(items),'entities':48,'holdout_entities':32,'deep':len(prereg['deep_ids']),'power':[p1,p2],'corpus_bytes':(OUT/'corpus.json').stat().st_size},ensure_ascii=False))


if __name__=='__main__':build()
