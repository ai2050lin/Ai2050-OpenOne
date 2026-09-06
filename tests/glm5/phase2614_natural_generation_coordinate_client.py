"""Publish exact coordinates and separately labeled behavior axes to the existing client."""
import json,sys
from pathlib import Path
from datetime import datetime
import numpy as np
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/'tests/glm5';RESULT=TESTS/'result';sys.path.insert(0,str(TESTS))
import phase2605_c676097_c692480_singleprompt_source_patch as io
import phase2608_c725249_c741632_autonomous_source_band as p8
OUT=RESULT/'phase2614_natural_generation_coordinate_client';ASSET=RESULT/'client_visualization_assets/research_kernel/c42641_output_conditioned_crossmodel_field.json'
MEMO=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'

def row(label,values,kind,phase=2608,**kw):
    return {'label':label,'values':np.asarray(values,dtype='float32').tolist(),'source':'natural_operation_atlas','coordinate_kind':kind,'phase':phase,'preview':True,**kw}
def panel(key,title,rows,d,semantics):
    return {'key':'phase2614_'+key,'model':title,'precision':'BF16 inference; FP16 exported activations (not guaranteed bit-exact for very small values), FP32 contrasts',
        'coordinate_count':d,'coordinate_order':'physical original order','coordinate_semantics':semantics,'rows':rows}

def main():
    pairs=io.load_json(p8.OUT/'material/pairs.json');panels=[];rows=[]
    material=io.read_jsonl(RESULT/'phase2603_c643329_c659712_unique_natural_lockbox/material/cases.unique.jsonl')
    src=RESULT/'phase2604_c659713_c676096_unique_fullcoordinate_confirmation/field/same_batch_fulltoken_exemplars'
    for path in sorted(src.glob('*_040_v0.float16.npy')):
        # Resolve filename family/language against actual material metadata.
        family,lang=path.name.rsplit('_040_v0.',1)[0].rsplit('_',1)
        case=next(r for r in material if r['family']==family and r['language']==lang and r['variant']==0 and r['split']=='external')
        data=np.load(path,mmap_mode='r');pos=case['source_token_positions'][0]
        for l in (0,6,18,30,36):
            rows.append(row(f'{family}/{lang} source token={case["prompt_ids"][pos]} pos={pos} checkpoint={l}',data[l,pos],
                'exact_token_embedding' if l==0 else 'exact_token_hidden_state',phase=2604,layer=l,token_id=case['prompt_ids'][pos]))
    panels.append(panel('exact_token','Qwen3-4B: stored individual source-token embedding and HiddenState (last checkpoint final-normalized)',rows,2560,'single physical activation coordinate; embedding row is one token vector, not a span average; export precision is declared separately'))
    indices={}
    for i,p in enumerate(pairs):indices.setdefault(p[0]['family']+'/'+p[0]['language'],i)
    rows=[]
    for c in ('baseline','source1','kv','kv_roll'):
        x=np.load(p8.OUT/f'field/greedy_prefill_{c}.float16.npy',mmap_mode='r')
        for g,i in indices.items():
            for l in (0,6,18,30,36):rows.append(row(f'{g} / {c} / raw checkpoint {l}',x[i,l],'raw_answer_boundary',condition=c,layer=l))
    panels.append(panel('greedy_fields','Qwen3-4B: exact prefill boundary fields for natural and intervened generation',rows,2560,'physical embedding/raw block-output coordinate; checkpoint36 is raw block35, not final-normalized'))
    for phase,key in ((2610,'qwen14'),(2611,'glm4'),(2612,'ds7')):
        directory=RESULT/f'phase{phase}_{key}_natural_replication';data=np.load(next((directory/'field').glob('source_span_means*')),mmap_mode='r');rr=[]
        for v in (0,1):
            for l in range(data.shape[2]):rr.append(row(f'{key} first frozen pair / variant{v} / checkpoint{l}',data[0,v,l],
                'span_mean_embedding' if l==0 else 'span_mean_hidden_state',phase=phase,layer=l))
        panels.append(panel(key+'_source',f'{key}: first frozen pair, full source-span mean coordinates',rr,data.shape[-1],'model-local physical activation coordinate; source token span averaged, final checkpoint normalized; not cross-model aligned'))
    directions=np.load(RESULT/'phase2609_c741633_c758016_discovery_only_operator_transfer/frozen_train_directions.npz');rr=[]
    for lang in directions.files:
        for l,v in enumerate(directions[lang]):rr.append(row(f'{lang} discovery-only operator / checkpoint{l}',v,'training_coordinate_direction',phase=2609,layer=l))
    panels.append(panel('learned_operator','Qwen3-4B: discovery-only time-operator directions, all coordinates',rr,2560,'frozen training contrast in physical coordinates; not learned weights or universal semantic operator'))
    f=io.load_json(p8.OUT/'analysis/final.json');conditions=list(f['conditions']);rr=[]
    for metric in ('recipient_correct','donor_correct','unparsed'):
        rr.append(row(metric,[f['conditions'][c][metric] for c in conditions],'generation_condition'))
    panels.append(panel('greedy_outcomes','All 120 external pairs: open-vocabulary generation outcomes',rr,len(conditions),'experimental condition axis: '+', '.join(conditions)))
    payload=io.load_json(ASSET);keys={p['key'] for p in panels}
    payload['models']=[p for p in payload['models'] if p['key'] not in keys]+panels
    payload['phase']=2614;payload['title']='Natural-operation source paths, generation, learned transfer and model-local coordinates'
    payload['claim_boundary']='Phase2608 reports all-pair open-vocabulary generation, not only candidate scores. Phase2609 freezes train-only directions and reveals surface dependence. Cross-model panels use separate physical bases. Oracle sufficiency and descriptive reuse do not establish a language compiler.'
    payload.setdefault('summary',{})['phase2600_2614']={'mechanism_closed':False,'new_panels':len(panels),'raw_fields_retained_for_display_and_reanalysis':True}
    payload['summary']['model_rows']={p['key']:len(p['rows']) for p in payload['models']};payload['summary']['total_rows']=sum(payload['summary']['model_rows'].values())
    # This is an experiment-generated JSON artifact; preserve all pre-existing panels.
    ASSET.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    result={'phase':2614,'asset':str(ASSET),'asset_sha256':io.sha256(ASSET),'asset_bytes':ASSET.stat().st_size,
        'panels':[{'key':p['key'],'rows':len(p['rows']),'coordinates':p['coordinate_count']} for p in panels],
        'checks':{'seven_panels':len(panels)==7,'exact_token60_rows':len(panels[0]['rows'])==60,'all_rows_full_coordinates':all(len(r['values'])==p['coordinate_count'] for p in panels for r in p['rows'])}}
    result['all_checks_passed']=all(result['checks'].values());io.save_json(OUT/'analysis/final.json',result)
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    text=rf'''

## Phase 2614: 自然操作与自主生成的重要原生坐标接入热力图客户端 [{stamp}]

**原理、用例与公式。** 复用已有heatmap结果类型，添加7类面板：12组真实source单token embedding/HiddenState，12组四种生成条件的原始boundary坐标，14B/GLM4/DS7B各自source坐标，冻结训练方向，1440生成行为轴。每行保留全部物理坐标，浮点值不Top-K；客户端已有全量显示模式可逐坐标核验。

$$\text{{cell}}(r,j)=H_{{r,j}},\qquad j=0,\ldots,D_{{model}}-1.$$

**结果汇总。** {json.dumps(result,ensure_ascii=False)}

**相关文件。** 本脚本，`{ASSET}`及`frontend/src/researchKernel/heatmapResearchRoute.js`。原场与复算材料保留，完整字段索引以.npy头shape为准，复用函数中的历史文件名尺寸不作为真值。

**分析与理论进展。** 参数轴、span平均、单token原值、raw block与final norm、行为轴明确分离；可视化只使证据可核验，不提高科学证明等级。

**问题硬伤与结论。** 每组显示冻结代表样本，不能代表全部分布；模型物理基底不同，禁止直接比较同编号语义。当前所示为激活值与训练方向，不是新发现的权重齿轮。工程构建检查由终审补记；机制未闭合。
'''
    if '## Phase 2614:' not in MEMO.read_text(encoding='utf-8-sig'):
        with MEMO.open('a',encoding='utf-8') as s:s.write(text)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
