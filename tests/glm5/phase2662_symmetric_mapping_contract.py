"""Prospective symmetric instruction calibration and heldout native-coordinate campaign."""
import itertools,os
import numpy as np
from transformers import AutoTokenizer
import transformers.modeling_utils as loading
from phase2620_native_coordinate_contract import *
from phase2621_native_behavior_run import load_model
from phase2648_output_function_material import parts,relation,FAMILIES
from phase2655_truth_answer_contract import encode,evaluate,leading_answer

OUT=RESULT/'phase2662_symmetric_mapping_contract'
EN='Arden Basil Cedric Daphne Edgar Freya Gideon Holly Jasper Kieran Leona Marcus Nolan Petra Tobias Vera'.split()
ZH='方乔 贺宁 苗青 卜岩 卓悦 尹帆 季晴 秦越 鲍琳 庞旭 华澜 金霖 侯安 殷蓝 谷舒 祁岳'.split()
CAL_EN='Calvin Doris Elton Felicia'.split();CAL_ZH='甄南 乐瑶 屈凯 米蕊'.split()


def load_native(key):
    os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1';original=loading.safe_open
    def pread(*args,**kwargs):kwargs['backend']='pread';return original(*args,**kwargs)
    loading.safe_open=pread
    try:return load_model(key)
    finally:loading.safe_open=original


def evaluate_multi(row,text):
    s=text.strip();pattern=r'(?i)Answer\s*:\s*(Yes|No)\.?' if row['language']=='en' else r'答案\s*[：:]\s*(是|否)[。.]?'
    match=re.fullmatch(pattern,s);label=None if match is None else match[1].casefold() in ('yes','是')
    return {'strict_correct':s.casefold()==row['target'].casefold(),'content_correct':label is not None and label==row['expected_yes'],'empty':not s,
        'multi_format_recognized':match is not None,'multi_content_boundary':'Full structured answer required; only declared colon/whitespace/outerperiod variants allowed, no substring credit.'}


def compose(fam,lang,unit,form,v,o,p,q,m,style,shots,cal=False,multi=False):
    a0,b0,body,_=parts(fam,lang,unit,form,v,o);names=(CAL_EN if lang=='en' else CAL_ZH) if cal else (EN if lang=='en' else ZH);a,b=names[unit*2:unit*2+2]
    body=body.replace(a0,'{{A}}').replace(b0,'{{B}}').replace('{{A}}',a).replace('{{B}}',b);statement,_=relation(fam,lang,unit,form);en=lang=='en'
    truth=v==p;affirm=truth!=bool(q);yes=affirm!=bool(m);words=['Yes','No'] if en else ['是','否'];positive,negative=words[m],words[1-m]
    if en:
        ask=('Is this statement ' if style==0 else 'Judge whether the statement is ')+('true' if q==0 else 'false')+('?' if style==0 else '.')
        rule=f'Output code: for an affirmative answer use {positive}; for a negative answer use {negative}. Output only the code.' if style==0 else f'If the requested judgment holds, write {positive}. If it does not hold, write {negative}. Write nothing else.'
        demo=''
        if shots:
            demo=f'\nExample 1: Statement: Two is larger than one. {ask} Code: {words[int(bool(q)!=bool(m))]}.\nExample 2: Statement: One is larger than two. {ask} Code: {words[1-int(bool(q)!=bool(m))]}.'
        text=body+'\nStatement: '+statement((a,b)[p])+'. '+ask+'\n'+rule+demo+'\nNow output the code for the original statement.'
    else:
        ask=('这句话是否' if style==0 else '判断陈述是否')+('正确' if q==0 else '错误')+('？' if style==0 else '。')
        rule=f'编码表：肯定答案→{positive}；否定答案→{negative}。只输出编码。' if style==0 else f'所问判断成立时输出{positive}，不成立时输出{negative}。只输出一个字。'
        demo=''
        if shots:demo=f'\n示例一：陈述：二大于一。{ask}编码：{words[int(bool(q)!=bool(m))]}。\n示例二：陈述：一大于二。{ask}编码：{words[1-int(bool(q)!=bool(m))]}。'
        text=body+'\n陈述：'+statement((a,b)[p])+'。'+ask+'\n'+rule+demo+'\n现在输出原始陈述的编码。'
    short_words=words
    if multi:
        text+=('\nUse the exact output format "Answer: CODE." Replace CODE with the selected code.' if en else '\n严格按“答案：编码。”的格式输出，将“编码”替换为选中的编码。')
        words=['Answer: Yes.','Answer: No.'] if en else ['答案：是。','答案：否。']
    return {'case_id':f'{fam}/{lang}/u{unit}/f{form}/v{v}/o{o}/p{p}/q{q}/m{m}/s{style}/d{shots}',
        'family':fam,'language':lang,'unit':unit,'form':form,'target_index':v,'mention_order':o,'probe_index':p,'polarity':q,'mapping':m,'style':style,'shots':shots,
        'body':body,'text':text,'entity_a':a,'entity_b':b,'statement_truth':truth,'question_affirmative':affirm,'expected_yes':yes,'target':words[0 if yes else 1],
        'alternate':words[1 if yes else 0],'common_readout_words':words,'short_answer_words':short_words,'multi':multi,
        'field_set':'initial' if unit<4 else 'confirmation','published':unit==4 and (form,v,o,p)==(0,0,0,0),
        'fp_selected':unit in (0,1,4,5) and (form,o,p,v)==(0,0,0,unit%2)}


def calibration(tok):
    rows=[]
    for fam,lang,unit,v,p,q,m,style,shots in itertools.product(FAMILIES,('en','zh'),range(2),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)):
        r=compose(fam,lang,unit,0,v,0,p,q,m,style,shots,cal=True);rows.append(encode(tok,{**r,'case_index':len(rows),'published':False}))
    return rows


def heldout(tok,selection):
    rows=[]
    for fam,lang,unit,form,v,o,p,q,m in itertools.product(FAMILIES,('en','zh'),range(8),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)):
        s=selection[lang];r=compose(fam,lang,unit,form,v,o,p,q,m,s['style'],s['shots']);rows.append(encode(tok,{**r,'case_index':len(rows)}))
    return rows


def length_audit(rows):
    groups={}
    for r in rows:groups.setdefault(tuple(r[k] for k in ('family','language','unit','form','target_index','mention_order','probe_index','polarity','style','shots')),[]).append(r)
    pairs=[rr for rr in groups.values() if len(rr)==2]
    return {'pairs':len(pairs),'equal_length':sum(len(rr[0]['prompt_ids'])==len(rr[1]['prompt_ids']) for rr in pairs),'maximum_difference':max(abs(len(rr[0]['prompt_ids'])-len(rr[1]['prompt_ids'])) for rr in pairs),
        'range':[min(len(r['prompt_ids']) for r in rows),max(len(r['prompt_ids']) for r in rows)]}


def main():
    assert not (OUT/'analysis/final.json').exists();tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True);rows=calibration(tok)
    save(OUT/'material/calibration_cases.json',rows)
    prior=RESULT/'phase2661_sequence_coordinate_delivery';zpath=prior/'maps/confirmed_native_coordinate_masks.npz'
    with np.load(zpath) as z:mask={k:z[k] for k in z.files}
    with np.load(RESULT/'phase2660_qwen14_truth_answer_replication/maps/allcoordinate_sign_group_counts.npz') as z:
        mask.update({'q14_'+k:(z[k]==16).astype('uint8') for k in z.files})
    OUT.joinpath('maps').mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/'maps/frozen_masks.npz',**mask)
    plan=read(prior/'analysis/next_campaign.json')['plan'];contract={'plan':plan,'calibration_cases':2048,'selection':'per-language maximize minimum correct count across4(q,m)cells, then total correct, then prefer no demos and style0; calibration only, frozen before8192heldout forwards',
        'heldout':'8192:8families*2languages*8entitypairs*2forms*2v*2orders*2probes*2polarities*2mappings; initialunits0..3 confirmation4..7; calibration2entitypairs disjoint',
        'symmetry':'Both mappings use same explicit template with code words swapped. Report exact tokenizer lengths, not assert equal complexity by length alone. Demos2 if enabled: two>one true and one>two false, adapted to question polarity and mapping; after source body.',
        'fields':'All8192 rawHboundary/MLPboundary, allH at entityA/B last tokens and output boundary; fulltokenH scanned into exact full-coordinate sums/squares.64fulltokenH published; noTopK.',
        'sequence':'256 new multi-token Answer:Yes./No. or 答案：是。/否。 prompts, units0/1/4/5,v=unit%2,p0,f0,o0,allq/m; naturalBF16 behavior then same-valuedFP32 separate branches; padded identical branch length and explicit attention masks',
        'derivatives':'Fullsequence plus content/format/EOS V adjoints, everyinput token.128casesunits4/5,8frozenVsites*2directions=2048changes plus128noops. Twoentitypairs/language, not128independent semantics.',
        'q14':'1024:16family/languagegroups*2entitypairs(units6/7)*2instructionstyles*2v*2p*2q*2m; winner demos fixed perlanguage. form/order0; not full old amplitudecontrol gate.',
        'prior_mask_sha256':sha(zpath),'frozen_mask_sha256':sha(OUT/'maps/frozen_masks.npz'),'prior_terminal_sha256':sha(prior/'analysis/terminal_audit.json'),
        'priority':'Native-coordinate language regularities before numeric chain-rule validation; allfailures retained. Templates and sense/fruit catalogs reused, names may occur in older history.',
        'storage':'Keep published original packs; rawpack manifests, hashbefore explicit deletion after wholecampaign uses; 8GiB disk floor. JSONL percompletedcase for durable runtime records.',
        'runtime':'NonquantizedBF16 auto CPU/GPU sequential; pread wrapper restores afterload; FP32 numerical control separately. No model or library files modified.'}
    save(OUT/'protocol/frozen.json',contract);checks={'2048_unique_calibration':len(rows)==len({r['prompt'] for r in rows})==2048,'disjoint_calibration_entities':not(set(EN)&set(CAL_EN) or set(ZH)&set(CAL_ZH)),
        'old_sixH_fourteenMLP_frozen':int(mask['bf_h'][-1].sum())==6 and int(mask['bf_mlp'][-1].sum())==14,'short_answers_single_token':all(len(x)==1 for r in rows for x in r['canonical_answer_ids'])}
    assert all(checks.values());finish(2662,'对称答案指令合同与旧候选/14B中层纹理冻结',OUT,{'provenance':str(Path(__file__)),'summary':{'contract':contract,'calibration_length_audit':length_audit(rows)},'checks':checks},
        '将正常和反向标签都写成同一映射句型，仅交换输出字词；用独立校准材料选择问法，随后冻结，不在留出结果上改门。原坐标全量测绘优先。',
        r't=\mathbf1[v=p],\quad a=t\oplus q,\quad y=a\oplus m;\quad s^*=\arg\max_s(\min_{q,m}C_{s,q,m},\sum_{q,m}C_{s,q,m},-d,-s).',
        '2048校准条件：八族双语、2独立实体对、目标/探问/极性/映射/双指令/无或2演示交叉。冻结4B六H十四MLP和14B所有旧全16组方向候选，含H26的154坐标与MLP24的297单元，不仅看末层。',
        '旧14B中层陈述方向纹理比末层更丰富，值得跨条件追踪；但旧反向任务不胜任，不能称真值核心。上批7个EOS排名反转及单参数局部预测保留为测量拼图，不把标准链式法则称新语言数学。',
        '相同字数/分词量不证明同等认知负担。演示可能引入任务支架；校准选择不是无选择偏差的能力估计。复用词义目录和句式，实体新不等于语言全新；源前缀数值相同还要检查形状。',
        '一次执行2663—2669全套：校准—冻结—8192全场—多token参数—2048实改—1024大模型—审计发布清理，目标相同继续。')


if __name__=='__main__':main()
