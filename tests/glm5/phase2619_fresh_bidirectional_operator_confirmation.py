"""Expanded fresh-vocabulary/template confirmation of the frozen layer-0 operator."""
import itertools,json,sys
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/'tests/glm5';RESULT=TESTS/'result';sys.path.insert(0,str(TESTS))
import model_utils
import phase2603_c643329_c659712_unique_natural_lockbox as mat
import phase2605_c676097_c692480_singleprompt_source_patch as io
import phase2608_c725249_c741632_autonomous_source_band as gen
import phase2609_c741633_c758016_discovery_only_operator_transfer as old
OUT=RESULT/'phase2619_fresh_bidirectional_operator_confirmation';MEMO=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
WORDS={'en':['distillation','filtration','mixing','cooling','heating','washing','milling','fermentation','weighing','labeling','sampling','evacuation'],
       'zh':['蒸馏','过滤','搅拌','冷却','加热','清洗','研磨','发酵','称重','贴标','取样','抽气']}

def build(tok):
    pairs=[]
    for lang in ('en','zh'):
        for form in range(3):
            for i,(a,b) in enumerate(list(itertools.combinations(WORDS[lang],2))[:40]):
                subject,obj=(a,b) if i%2==0 else (b,a);pair=[]
                for variant in (0,1):
                    before=(variant==0) if i%2==0 else (variant==1)
                    if lang=='en':
                        rel=[f'{subject} occurred {"before" if before else "after"} {obj}',f'{subject} was {"earlier" if before else "later"} than {obj}',f'{subject} {"preceded" if before else "followed"} {obj}'][form]
                        text=f'The log records that {rel}. Based only on this relation, name the event that happened first. Give the event name alone.'
                    else:
                        rel=[f'{subject}{"早于" if before else "晚于"}{obj}',f'{subject}{"先于" if before else "后于"}{obj}',f'{subject}发生在{obj}{"之前" if before else "之后"}'][form]
                        text=f'日志所述关系是：{rel}。仅根据这条关系，给出先发生的事件名称，不要附加解释。'
                    prompt=tok.apply_chat_template([{'role':'user','content':text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
                    pair.append({'pair_id':f'fresh/{lang}/form{form}/{i}','case_id':f'fresh/{lang}/form{form}/{i}/{variant}',
                        'family':'chronology','language':lang,'form':form,'variant':variant,'direction_sign':1 if i%2==0 else -1,
                        'target':a if variant==0 else b,'alternate':b if variant==0 else a,'prompt':prompt,'prompt_ids':tok.encode(prompt,add_special_tokens=False)})
                for r,pos in zip(pair,mat.source_spans(pair[0]['prompt_ids'],pair[1]['prompt_ids'])):r['source_token_positions']=pos
                pairs.append(pair)
    return pairs

def main():
    direction_path=RESULT/'phase2615_chat_interface_operator_confirmation/frozen_train_directions.npz'
    sha=io.sha256(direction_path);directions=np.load(direction_path)
    contract={'phase':2619,'frozen_direction_hash':sha,'layer':0,'pairs':240,'conditions':['baseline','natural_donor','learned','roll193','roll641','roll997','negative'],
        'new_event_vocabulary':WORDS,'mention_order_counterbalanced':True,'source_sign_known_from_external_operation':True,'no_reselection_or_refit':True}
    io.save_json(OUT/'protocol/frozen.json',contract)
    model,tok,_=model_utils.load_model('qwen3',dtype=torch.bfloat16,use_8bit=False);pairs=build(tok);io.save_json(OUT/'material/pairs.json',pairs)
    vectors=np.stack([directions[p[0]['language']][1]*p[0]['direction_sign'] for p in pairs]);records=[];summary={}
    for c in contract['conditions']:
        pp=[list(reversed(p)) for p in pairs] if c=='natural_donor' else pairs
        v=-vectors if c=='negative' else np.roll(vectors,int(c[4:]),axis=-1) if c.startswith('roll') else vectors
        kind='baseline' if c in ('baseline','natural_donor') else 'source1'
        path=OUT/f'field/{c}_all240x37x2560.float32.npy';path.parent.mkdir(parents=True,exist_ok=True)
        field=np.lib.format.open_memmap(path,mode='w+',dtype='float32',shape=(240,37,2560))
        rows=gen.generate_condition(model,tok,pp,kind,v,{},source_layer=0,max_new_tokens=64,raw_field=field)
        field.flush()
        for r,p in zip(rows,pairs):r['condition']=c;r['language']=p[0]['language'];r['form']=p[0]['form'];r['direction_sign']=p[0]['direction_sign'];r['eos']=tok.eos_token_id in r['generated_ids']
        records+=rows;summary[c]=gen.metrics(rows);summary[c]['eos_rate']=float(np.mean([r['eos'] for r in rows]));io.write_jsonl(OUT/f'behavior/{c}.jsonl',rows)
        print(c,json.dumps(summary[c]),flush=True)
    bysurface={f'{lang}/form{f}/sign{s}':{c:gen.metrics([r for r in records if r['language']==lang and r['form']==f and r['direction_sign']==s and r['condition']==c]) for c in summary}
               for lang in ('en','zh') for f in range(3) for s in (1,-1)}
    eligible=set(r['pair_id'] for r in records if r['condition']=='baseline' and r['recipient_correct']) & set(r['pair_id'] for r in records if r['condition']=='natural_donor' and r['recipient_correct'])
    result={'phase':2619,'timestamp':datetime.now().astimezone().isoformat(),'conditions':summary,'by_surface_and_direction':bysurface,'dual_natural_success_pairs':len(eligible),
        'dual_success_conditional':{c:gen.metrics([r for r in records if r['condition']==c and r['pair_id'] in eligible]) for c in summary} if eligible else {},
        'checks':{'all240_pairs':len(pairs)==240,'all1680_greedy':len(records)==1680,'disjoint_from_train_and_previous_test_vocab':all(set(WORDS[l]).isdisjoint(set(old.WORDS[l][0]+old.WORDS[l][1])) for l in WORDS),
        'frozen_direction_unchanged':io.sha256(direction_path)==sha,'all_fp32_fields_finite':all(np.isfinite(np.load(p,mmap_mode='r')).all() for p in (OUT/'field').glob('*.npy'))},'language_mechanism_closed':False}
    result['all_checks_passed']=all(result['checks'].values());io.save_json(OUT/'analysis/final.json',result)
    asset=RESULT/'client_visualization_assets/research_kernel/c42641_output_conditioned_crossmodel_field.json';payload=io.load_json(asset);rows=[]
    for c in ('baseline','learned','roll641'):
        field=np.load(OUT/f'field/{c}_all240x37x2560.float32.npy',mmap_mode='r')
        for i in (0,1,40,41,80,81,120,121,160,161,200,201):
            for l in (0,1,6,36):rows.append({'label':f'{pairs[i][0]["pair_id"]}/{c}/raw checkpoint{l}','source':'phase2619_fresh_confirmation','phase':2619,'layer':l,'coordinate_kind':'fp32_stored_raw_boundary','preview':True,'values':field[i,l].tolist()})
    physical={'key':'phase2619_fresh_fp32_field','model':'Qwen3-4B fresh-event bidirectional confirmation','precision':'BF16 nonquantized inference, FP32 stored activations','coordinate_count':2560,'coordinate_semantics':'all physical raw boundary coordinates; final row raw block35, not final norm','rows':rows}
    outcomes={'key':'phase2619_fresh_outcomes','model':'240 fresh pairs: independent frozen-layer confirmation','precision':'outcome fractions','coordinate_count':7,'coordinate_semantics':'condition axis: '+', '.join(summary),'rows':[{'label':m,'source':'phase2619_fresh','phase':2619,'coordinate_kind':'generation_condition','preview':True,'values':[summary[c][m] for c in summary]} for m in ('recipient_correct','donor_correct','unparsed','eos_rate')]}
    payload['models']=[p for p in payload['models'] if p['key'] not in (physical['key'],outcomes['key'])]+[physical,outcomes];payload['phase']=2619
    payload['summary']['model_rows']={p['key']:len(p['rows']) for p in payload['models']};payload['summary']['total_rows']=sum(payload['summary']['model_rows'].values())
    asset.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    result['client_asset_sha256']=io.sha256(asset);io.save_json(OUT/'analysis/final.json',result)
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    text=rf'''

## Phase 2619: 重要阳性的240新事件pair双向独立确认与1680次自主生成 [{stamp}]

**自动续研依据。** Phase2615训练方向layer0产生48.3%另一侧答案，roll仅2.5%，需要按科研约束扩大重要阳性。固定该层和训练方向，不再用本Phase数据选择深度或剂量。

**原理、用例与公式。** 全新中英事件词表，与训练和Phase2609/2615测试词表无交集；每语言40事件对×3同义表面=240pair，问题模板更换。前/后方向各半，答案先提及/后提及平衡，避免只复制首实体。使用规范chat/no-thinking；baseline、自然donor、训练方向、三个坐标roll、反向方向共1680生成，最大64 token。

$$\delta_{{i,l}}=s_i\mu_{{language(i),l}},\quad s_i\in\{{-1,+1\}},\quad l=0\text{{固定}}.$$

方向只来自Phase2615训练场，测试侧自然donor仅用于行为评估，不进入方向提取。保存每次prefill全部37×2560原始坐标，FP32容器；报告BF16实际与预期位移。

**结果汇总。** {json.dumps(summary,ensure_ascii=False)}。双变体自然正确pair={len(eligible)}/240，另给该条件子集结果但不替代全体；12个语言×表面×方向单元详见final。

**相关文件。** `{OUT}`内冻结合同/哈希、240pair、1680输出、7种全量FP32场和final；客户端增加FP32原坐标与行为两个面板，原场保留用于可视化复算。

**理论进展与问题硬伤。** 这是对冻结方向的独立新词表/问题表述确认，而不是继续调旧测试集。仍只有时间先后一个语义域，40事件对跨3表面复用，不是240独立语义事件；source位置和外部变换方向由实验设计给定；词形模板与自然事件知识仍可能影响答案。新材料也须同时审视两侧基线，不能用接近满分的一侧掩盖另一侧失误。

**结论。** 检查={json.dumps(result['checks'])}。只对确认支持的语言、表面和变换方向陈述规律，不能提升为普遍语言编码齿轮。下一具体研究对象是跨位置条件载体及词形/语义身份分离，当前机制仍未闭合。
'''
    if '## Phase 2619:' not in MEMO.read_text(encoding='utf-8-sig'):
        with MEMO.open('a',encoding='utf-8') as f:f.write(text)

if __name__=='__main__':main()
