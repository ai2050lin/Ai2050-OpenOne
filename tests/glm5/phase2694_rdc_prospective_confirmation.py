"""Bounded S2 pilot: frozen S1 readers, new lexical entries/paraphrases, crossed answers.

Reuses the immutable S1 capture implementation, not its material or protocol.
512 new prompts; no fitting or hyperparameter selection on these new samples.
"""
import argparse
from rdc_feature_common import *
import phase2693_rdc_language_capture as capture
from rdc_s1_material import LEXICON as OLD_LEXICON

RUN='s2pilot'; OUT=CAMPAIGN/RUN
LEXICON=[
 ('fruit','水果','apricot plum cherry fig kiwi papaya guava pineapple','杏 李子 樱桃 无花果 猕猴桃 木瓜 番石榴 菠萝'),
 ('whole_plant','完整植物','maple elm spruce fir cypress poplar ginkgo palm','枫树 榆树 云杉 冷杉 柏树 杨树 银杏 棕榈'),
 ('animal','动物','zebra giraffe fox wolf bear deer dolphin penguin','斑马 长颈鹿 狐狸 狼 熊 鹿 海豚 企鹅'),
 ('tool','工具','axe scissors hoe rake trowel spanner mallet pickaxe','斧头 剪刀 锄头 耙子 抹子 活扳手 木槌 镐头'),
 ('action','动作','sing dance sleep cook think listen speak laugh','唱歌 跳舞 睡觉 烹饪 思考 聆听 说话 大笑'),
 ('property','性质','green yellow tall short soft hard smooth rough','绿色 黄色 高 矮 柔软 坚硬 光滑 粗糙'),
 ('function_word','功能词','unless while since until before after yet nor','除非 当 自从 直到 之前 之后 然而 也不'),
 ('punctuation','标点','[ ] { } … — “ ”','【 】 「 」 …… —— “ ”'),
]

def build(tok):
    rules={
      'en':'For this task use eight separate lexical-use groups: fruit names, whole-plant names, animal names, tools, actions (verbs), properties (adjectives), function words, and punctuation. Fruit names and whole-plant names count as different groups. Use the stated lexical usage, not a biological hierarchy. Compare the two entries and output only Yes or No.',
      'zh':'本题采用八个分开的词汇用途组：水果名、完整植物名、动物名、工具、动作（动词）、性质（形容词）、功能词和标点。水果名与完整植物名按不同组处理，按词汇用途而非生物学上下位关系判断。比较两个词，只输出是或否。'}
    questions={'en':['Does one of these eight groups contain both A and B?','Must A and B be assigned to two distinct groups?'],
               'zh':['这八个组中，是否有一个组同时包含甲词和乙词？','甲词和乙词是否必须分别归入两个不同的组？']}
    rows=[]
    for f,(family,label,en,zh) in enumerate(LEXICON):
        for unit in range(8):
            for form in range(2):
                for same in (False,True):
                    pf=f if same else (f+1+unit%7)%8;pu=unit^1
                    for lang in ('en','zh'):
                        li=2 if lang=='en' else 3
                        a=LEXICON[f][li].split()[unit];b=LEXICON[pf][li].split()[pu]
                        ap,bp=('A: ','B: ') if lang=='en' else ('甲词：','乙词：')
                        lines=[ap+a,bp+b];reverse=(unit+form+int(same))%2==1
                        if reverse:lines.reverse()
                        prompt=tok.apply_chat_template([{'role':'system','content':rules[lang]},
                            {'role':'user','content':'\n'.join(lines)+'\n'+questions[lang][form]}],
                            tokenize=False,add_generation_prompt=True,enable_thinking=False)
                        e=tok(prompt,add_special_tokens=False,return_offsets_mapping=True)
                        spans={}
                        for block,prefix,term in [('u',ap,a),('v',bp,b)]:
                            start=prompt.index(prefix+term)+len(prefix);end=start+len(term)
                            ps=[i for i,(s,t) in enumerate(e['offset_mapping']) if t>start and s<end and t>s]
                            assert ps;spans[block]={'chars':[start,end],'positions':ps,'term':term}
                        qs=prompt.index(questions[lang][form]);qe=qs+len(questions[lang][form])
                        cps=[i for i,(s,t) in enumerate(e['offset_mapping']) if t>qs and s<qe and t>s]
                        expected=same if form==0 else not same
                        rows.append(dict(sample_id=f's2-{family}-{unit}-{form}-{int(same)}-{lang}',family=family,
                            family_index=f,family_label=label,unit=unit,form=form,base_id=f'{family}-{unit}',language=lang,
                            u=a,v=b,partner_family=LEXICON[pf][0],partner_family_index=pf,partner_unit=pu,
                            same_group=same,negative_query=bool(form),expected_yes=expected,
                            target=('Yes' if expected else 'No') if lang=='en' else ('是' if expected else '否'),
                            prompt=prompt,prompt_ids=e['input_ids'],tokens=tok.convert_ids_to_tokens(e['input_ids']),
                            spans=spans,context_positions=cps,input_order='BA' if reverse else 'AB',
                            role='u is named A/甲 regardless of physical order',source_mode='live_model',word_split='prospective_test'))
    assert len(rows)==512 and len({r['prompt'] for r in rows})==512
    for lang in ('en','zh'):
        for f in range(8):
            for form in range(2):
                subset=[r for r in rows if (r['language'],r['family_index'],r['form'])==(lang,f,form)]
                assert sum(r['expected_yes'] for r in subset)==len(subset)//2
    for li in (2,3):
        assert not ({x for row in LEXICON for x in row[li].split()} & {x for row in OLD_LEXICON for x in row[li].split()})
    return rows

def prepare():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    rows=build(tok);immutable(OUT/'material.json',rows)
    # Freeze both word and joint fitted readers. Results remain separately reported, never choose by new test.
    model_ids=[f'{split}__family__H{layer}__{algo}' for split in ('word','joint') for layer in (0,12,24,36)
               for algo in ('A1_linear','A2_quadratic','A2_cubic','A3_ordered_pair','A4_conditional')]
    frozen={mid:sha(CAMPAIGN/f's1/models/{mid}.npz') for mid in model_ids}
    contract={'version':1,'phase':2694,'run_id':RUN,'samples':512,'status':'prospective_frozen_before_new_forward',
        'question':'Does fixed full-coordinate family readability extend to new lexical entries and paraphrases with answer balance?',
        'families':[r[0] for r in LEXICON],'model':'qwen3-4b','quantized':False,'dtype':'bfloat16',
        'design':'64 entirely new bilingual lexical entries x2 new query polarities x2 independently crossed relations x2 languages',
        'model_code_sha':sha(Path(capture.__file__)),'source_sha':sha(Path(__file__)),
        's1_result_sha':sha(CAMPAIGN/'s1/result.json'),'material_sha':sha(OUT/'material.json'),
        'frozen_models':frozen,'lambdas':'Reuse saved S1 alpha/scales; no new fitting or selection',
        'metric':'family MSE/accuracy vs S1 fixed mean and nearest full-coordinate examples; per language/family; no posthoc threshold tuning',
        'first_gate':'At least10% MSE reduction vs best fixed A0 baseline in prospective set; report failures and simple A1 comparison',
        'capture_contract':read(CAMPAIGN/'s1/protocol.json'),
        'disk_upper_bf16_residual_bytes':sum(len(r['prompt_ids'])*37*2560*2 for r in rows),
        'limits':['New lexical entries, not tokenizer-ID-disjoint: token pieces and grammatical scaffolding may recur.',
                  'Same eight task-defined families; not new relationship types, cross-model confirmation, or a full S2 3072-sample expansion.',
                  'Only family-reader transfer is prospective. Any new answer-reader fitting would be exploratory and separately labeled.',
                  'S1 original joint external-answer test is single-label; not a valid balanced binary generalization benchmark.']}
    immutable(OUT/'protocol.json',contract)
    return rows,contract

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,choices=(16,512),default=16);args=p.parse_args()
    capture.RUN=RUN;capture.OUT=OUT;capture.prepare=prepare;capture.main(args.limit)
