"""Frozen native-coordinate third set: factual truth, query polarity and response mapping."""
import itertools,re
import numpy as np
from transformers import AutoTokenizer
from phase2620_native_coordinate_contract import *
from phase2648_output_function_material import parts,relation,FAMILIES

OUT=RESULT/'phase2655_truth_answer_contract'
EN_NAMES='Ethan Lucas Helen Aaron Claire Dylan Emily Frank Irene Roger Megan Walter Beatrice Edwin Nancy Patrick'.split()
ZH_NAMES='李珊 吴越 周辰 郑瑾 苏澄 程诺 许晗 董祺 韩露 潘阳 曾岚 段淳 贾蓉 邵林 储妍 江恒'.split()


def encode(tok,row):
    prompt=tok.apply_chat_template([{'role':'user','content':row['text']}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    enc=tok(prompt,add_special_tokens=False,return_offsets_mapping=True);start=prompt.index(row['body']);spans={}
    for key,name in [('a',row['entity_a']),('b',row['entity_b'])]:
        a=start+row['body'].index(name);b=a+len(name);spans[key]=[i for i,(s,e) in enumerate(enc['offset_mapping']) if e>a and s<b];assert spans[key]
    return {**row,'prompt':prompt,'prompt_ids':enc['input_ids'],'token_strings':tok.convert_ids_to_tokens(enc['input_ids']),
        'entity_spans':spans,'common_readout_ids':[tok.encode(s,add_special_tokens=False)[0] for s in row['common_readout_words']],
        'canonical_answer_ids':[tok.encode(s,add_special_tokens=False) for s in row['common_readout_words']],'eos_token_id':tok.eos_token_id}


def build(tok):
    rows=[]
    for fam,lang,unit,form,v,o,p,q,m in itertools.product(FAMILIES,('en','zh'),range(8),(0,1),(0,1),(0,1),(0,1),(0,1),(0,1)):
        old_a,old_b,body,_=parts(fam,lang,unit,form,v,o);a,b=(EN_NAMES if lang=='en' else ZH_NAMES)[2*unit:2*unit+2]
        body=body.replace(old_a,'{{A}}').replace(old_b,'{{B}}').replace('{{A}}',a).replace('{{B}}',b)
        statement,_=relation(fam,lang,unit,form);en=lang=='en';truth=v==p;affirm=truth!=bool(q);yes=affirm!=bool(m)
        if en:
            ask='Is this statement '+('true' if q==0 else 'false')+'?'
            rule='Answer only Yes or No.' if m==0 else 'Use reversed answer labels: reply No for an affirmative answer and Yes for a negative answer. Reply with this single word only.'
            text=body+'\nStatement: '+statement((a,b)[p])+'. '+ask+' '+rule;words=['Yes','No']
        else:
            ask='这句话是否'+('正确' if q==0 else '错误')+'？'
            rule='只回答是或否。' if m==0 else '使用反向回答标签：肯定回答写“否”，否定回答写“是”。只输出这个字。'
            text=body+'\n陈述：'+statement((a,b)[p])+'。'+ask+rule;words=['是','否']
        r={'case_index':len(rows),'case_id':f'{fam}/{lang}/u{unit}/f{form}/v{v}/o{o}/p{p}/q{q}/m{m}',
            'family':fam,'language':lang,'unit':unit,'form':form,'target_index':v,'mention_order':o,'probe_index':p,'polarity':q,'mapping':m,
            'statement_truth':truth,'question_affirmative':affirm,'expected_yes':yes,'body':body,'text':text,'entity_a':a,'entity_b':b,
            'target':words[0 if yes else 1],'alternate':words[1 if yes else 0],'common_readout_words':words,
            'field_set':'initial' if unit<4 else 'confirmation','fp_selected':unit in (0,4) and (form,v,o)==(0,0,0),
            'published':unit==4 and (form,v,o,p)==(0,0,0,0),'crossmodel':unit==7 and (form,o)==(0,0)}
        rows.append(encode(tok,r))
    return rows


def evaluate(row,text):
    s=text.strip();strict=s.casefold()==row['target'].casefold();plain=s.strip(' .。').casefold()
    if row['language']=='zh':plain={'是的':'是','不是':'否','不是的':'否'}.get(plain,plain)
    return {'strict_correct':strict,'content_correct':plain==row['target'].casefold(),'empty':not s}


def leading_answer(text,lang):
    s=text.lstrip(' \t\r\n*`_')
    if lang=='en':
        match=re.match(r'(?i)(yes|no)(?![A-Za-z])',s);return None if match is None else match[1].lower()=='yes'
    if s.startswith('不是'):return False
    if s.startswith('是'):return True
    if s.startswith('否'):return False
    return None


def main():
    assert not (OUT/'analysis/final.json').exists()
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True);cases=build(tok);save(OUT/'material/cases.json',cases)
    prior=RESULT/'phase2654_output_function_delivery';maps=RESULT/'phase2651_output_function_maps'
    frozen={}
    with np.load(prior/'maps/truth_query_fullcoordinate_reuse.npz') as z:
        for k in z.files:frozen[k]=(z[k]==16).astype('uint8')
    for p in sorted((maps/'maps').glob('*_truth_*.npz')):
        with np.load(p) as z:
            for metric in ('h','mlp','bf_h','bf_mlp'):frozen[p.stem+'__'+metric+'__sign']=np.sign(z['target__'+metric+'__mean']).astype('int8')
    OUT.joinpath('maps').mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/'maps/frozen_previous_coordinates.npz',**frozen)
    plan={2655:'事实/真值/答案字词交叉8192材料与全坐标规则冻结',2656:'8192自然BF16生成、全坐标边界图谱与实际答案决策时刻',2657:'第三实体集合的旧候选确认及正反问题/正反答案映射全坐标图谱',2658:'256前缀完整Yes/No答案加EOS概率差的FP32全token参数因子与自然决策读出',2659:'独立128前缀2176条件真实单V参数的完整答案序列数值检验',2660:'Qwen14B非量化串行256条件自然行为与模型内全坐标复验',2661:'规律审计、完整序列参数客户端、清理和整批交付'}
    protocol={'plan':plan,'cases':8192,'entities':'8new pairs/language, units0..3 and4..7 disjoint; prior templates and fruit/sense catalogs reused',
        'factor_cells':'8families x2languages x8entitypairs x2forms x2targets x2orders x2probes x2questionpolarities x2answerlabelmappings',
        'truth': 'statement_truth=(v==p); affirmative=truth XOR q; expectedYes=affirmative XOR m',
        'frozen_rule':'allprevious physical-coordinate masks and initial pergroup signs saved before new forwards; targetRMS>orderRMS and formRMS. Evaluate every coordinate, not only candidates.',
        'contrasts':'targetv0-v1, order0-1, form0-1 perprobe/polarity/mapping. Truth oriented factor(-1)^p; affirmative(-1)^(p+q); answer-oriented(-1)^(p+q+m). Preserve raw arrays.',
        'natural':'singleprompt,no-cache,BF16 greedy up to16tokens; record exact fullanswer and first recognizable answer separately; no failure filtering',
        'fields':'all8192 H boundary37x2560 andMLPboundary36x9728; fulltokenH scanned into allcoordinate RMS.64published fulltokenH and naturaldecision checkpoints retained; noTopK; lossless NPZ allowed.',
        'sequence_objective':'logP(canonicalYes thenEOS|prompt)-logP(canonicalNo thenEOS|prompt), fullvocabulary log_softmax at every predicted token. Each alternative teacher-forced separately, EOS explicit. Finite canonical strings, not total answer-class mass or free-generation success.',
        'fp_cases':256,'scalar_cases':128,'scalar_sites_source':str(RESULT/'phase2653_output_function_scalar_validation/protocol/frozen.json'),
        'crossmodel':'Qwen14B BF16device_mapauto;256cases unit7/form0/order0 all16groups x2target x2probe x2polarity x2mapping. Only one entitypair/language, replication sample not universal proof; no coordinate-index alignment withQwen4.',
        'storage':'8GiB floor, preserve published64 BF16/FP32 cases, manifested unshown raw files removed after delivery; no persistent learned weight writes',
        'prior_terminal_sha256':sha(prior/'analysis/terminal_audit.json'),'prior_reuse_sha256':sha(prior/'maps/truth_query_fullcoordinate_reuse.npz'),'frozen_coordinate_sha256':sha(OUT/'maps/frozen_previous_coordinates.npz')}
    save(OUT/'protocol/frozen.json',protocol)
    checks={'8192_unique_prompts':len(cases)==len({r['prompt'] for r in cases})==8192,'256_fp_cases':sum(r['fp_selected'] for r in cases)==256,'64_published':sum(r['published'] for r in cases)==64,
        '256_crossmodel_cases':sum(r['crossmodel'] for r in cases)==256,'max_prefix_safe':max(len(r['prompt_ids']) for r in cases)<=106,
        'literal_binary_readouts':all(len(r['canonical_answer_ids'][0])==len(r['canonical_answer_ids'][1])==1 and r['common_readout_ids'][0]!=r['common_readout_ids'][1] for r in cases)}
    assert all(checks.values())
    finish(2655,'从真假同号坐标到事实—问题—答案映射的完整研究合同',OUT,{'provenance':str(Path(__file__)),'summary':{'plan':plan,'cases':len(cases),'length_range':[min(len(r['prompt_ids']) for r in cases),max(len(r['prompt_ids']) for r in cases)],'contract':protocol},'checks':checks},
        '保留2654的候选反号纹理，但把纯真值神经元判断暂缓。用事实目标、所问实体、问题真假极性和正常/反向答案映射交叉，先看行为再看全部原坐标。',
        r't=\mathbf1[v=p];\quad a=t\oplus q;\quad y=a\oplus m;\quad D=H(v=0)-H(v=1);\quad L=\log P(Y,EOS|x)-\log P(N,EOS|x).',
        '八族双语、8新实体对、双句式/目标/顺序/探问/极性/映射，共8192完整前缀。旧20H/37MLP等全坐标规则及逐组方向在新forward前冻结；计划七个完整相关Phase。',
        '方向翻转若跟随答案映射而非事实真值，可削弱纯真值解释；若不跟随也不可直接确立语义，因为反向规则理解、位置和上下文改变仍需检查。完整答案加结束符概率提供比首token更明确的标量归因对象。',
        '复用句型与词义词表，不是全新语言基准；反向映射引入更长指令及额外任务负担。独立的是实体集合，不是每条prompt独立语义抽样。概率目标仍为实验者指定两条规范答案序列。',
        '按合同完成2656—2661：全量观察和确认不由行为或删除门成败中断，参数级序列算法验证后交付并继续同目标研究。')


if __name__=='__main__':main()
