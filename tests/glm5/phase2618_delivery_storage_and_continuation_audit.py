"""Explicit in-scope storage cleanup, delivery manifest, and calibrated scientific summary."""
import json,re,sys
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[2];TESTS=ROOT/'tests/glm5';RESULT=TESTS/'result';sys.path.insert(0,str(TESTS))
import phase2605_c676097_c692480_singleprompt_source_patch as io
OUT=RESULT/'phase2618_delivery_storage_and_continuation_audit';MEMO=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
RAW_DIRS=[RESULT/'phase2602_c626945_c643328_natural_fullcoordinate_field/field',
          RESULT/'phase2605_c676097_c692480_singleprompt_source_patch/field',
          RESULT/'phase2606_c692481_c708864_qkv_head_route/field',
          RESULT/'phase2607_c708865_c725248_multilayer_qkv_band/field']
EXACT_FILES=[RESULT/'phase2604_c659713_c676096_unique_fullcoordinate_confirmation/field/answer_boundary_all1200.float16.npy',
             RESULT/'phase2604_c659713_c676096_unique_fullcoordinate_confirmation/field/answer_pair_delta_unique600.float16.npy']

def main():
    finals={}
    for phase in range(2600,2618):
        files=list(RESULT.glob(f'phase{phase}_*/analysis/final.json'))
        if len(files)!=1:raise RuntimeError((phase,files))
        finals[phase]=io.load_json(files[0])
    if not all(f.get('all_checks_passed') for f in finals.values()):raise RuntimeError('procedural checks incomplete')
    cleanup_file=OUT/'analysis/raw_cleanup_manifest.json'
    if cleanup_file.exists():manifest=io.load_json(cleanup_file)
    else:
        targets=[p for d in RAW_DIRS for p in d.rglob('*.npy')]+[p for p in EXACT_FILES if p.exists()]
        manifest=[]
        for p in sorted(set(targets)):
            resolved=p.resolve(strict=True)
            valid=any(resolved.is_relative_to(d.resolve()) for d in RAW_DIRS) or resolved in [x.resolve() for x in EXACT_FILES]
            if not valid or not resolved.is_relative_to(RESULT.resolve()) or resolved.suffix!='.npy' or p.is_symlink():raise RuntimeError(f'unsafe cleanup target: {resolved}')
            manifest.append({'path':str(resolved),'bytes':resolved.stat().st_size,'sha256':io.sha256(resolved),
                'reason':'not published as current raw client field; superseded measurement or completed oracle diagnostic; scores, formulas and reconstruction scripts retained'})
        io.save_json(cleanup_file,manifest)
    # Files only: never delete directories, workspace roots, model weights or published fields.
    for item in manifest:
        p=Path(item['path']).resolve()
        if not p.is_relative_to(RESULT.resolve()) or not (any(p.is_relative_to(d.resolve()) for d in RAW_DIRS) or p in [x.resolve() for x in EXACT_FILES]):raise RuntimeError(p)
        if p.exists():
            if io.sha256(p)!=item['sha256']:raise RuntimeError(f'changed after manifest: {p}')
            p.unlink()
    asset=RESULT/'client_visualization_assets/research_kernel/c42641_output_conditioned_crossmodel_field.json'
    payload=io.load_json(asset)
    # Add the automatically continued chat calibration without conflating condition and physical axes.
    f15=finals[2615];conditions=list(f15['conditions']);rows=[]
    for label in ('recipient_correct','donor_correct','unparsed','eos_rate'):
        rows.append({'label':label,'values':[f15['conditions'][c][label] for c in conditions],'source':'phase2615_chat_calibration','phase':2615,'coordinate_kind':'generation_condition','preview':True})
    panel={'key':'phase2615_chat_generation_conditions','model':'Qwen3-4B chat/no-thinking operator calibration','precision':'BF16 inference, exact outcome counts','coordinate_count':len(conditions),'coordinate_semantics':'condition axis: '+', '.join(conditions),'rows':rows}
    payload['models']=[p for p in payload['models'] if p['key']!=panel['key']]+[panel]
    payload['phase']=2618;payload['summary']['model_rows']={p['key']:len(p['rows']) for p in payload['models']};payload['summary']['total_rows']=sum(payload['summary']['model_rows'].values())
    asset.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    headings=[int(v) for v in re.findall(r'^## Phase (\d+):',MEMO.read_text(encoding='utf-8-sig'),re.M)]
    checks={'phases2600_2617_contiguous':[v for v in headings if 2600<=v<=2617]==list(range(2600,2618)),
        'all_completed_checks':all(f['all_checks_passed'] for f in finals.values()),'manifest_targets_removed':all(not Path(x['path']).exists() for x in manifest),
        'published_field_preserved':(RESULT/'phase2608_c725249_c741632_autonomous_source_band/field/greedy_prefill_baseline.float16.npy').exists()}
    result={'phase':2618,'timestamp':datetime.now().astimezone().isoformat(),'checks':checks,'all_checks_passed':all(checks.values()),
        'deleted_files':len(manifest),'deleted_bytes':sum(x['bytes'] for x in manifest),'recovery':'not moved to trash; regenerate with retained scripts and local weights',
        'asset_sha256':io.sha256(asset),'asset_bytes':asset.stat().st_size,'new_panels':[p['key'] for p in payload['models'] if p['key'].startswith(('phase2614_','phase2615_','phase2617_'))],
        'same_goal_automatic_followups_completed':[2615,2617],'language_mechanism_closed':False,
        'unresolved':'semantic identity-independent, cross-surface conditional coordinate algorithm and its natural multi-layer compiler'}
    io.save_json(OUT/'analysis/final.json',result)
    stamp=datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')
    f17=finals[2617]
    text=rf'''

## Phase 2618: 全合同与两轮自动续研交付、存储清理和最终证据校准 [{stamp}]

**测试原理、用例与公式。** 审核Phase2600—2617连续性、程序检查、参数客户端与存储清单。对已完成且未以当前原场发布的旧测量及oracle诊断文件逐一记录路径、SHA256和字节数后清理，不删除模型权重、脚本、行为得分、MEMO或新发布原场。

$$B_{{freed}}=\sum_{{f\in\text{{verified manifest}}}}\operatorname{{bytes}}(f).$$

**结果汇总。** 删除{len(manifest)}个.npy文件，共{sum(x['bytes'] for x in manifest)} bytes；未移入回收站，可按保留脚本与本地模型重新生成，不能直接撤销。新客户端面板={len(result['new_panels'])}。完整检查={json.dumps(checks,ensure_ascii=False)}。

**同目标自动续研成果。** Phase2615已完成2160次chat/no-thinking五层校准，Phase2617再完成六族600次条件字典比较。后者训练字典覆盖={f17['covered_test_pairs']}/120；条件字典另一侧答案率={f17['conditions']['conditioned_dictionary']['donor_correct']:.6f}、族均值={f17['conditions']['family_mean']['donor_correct']:.6f}、roll={f17['conditions']['dictionary_roll']['donor_correct']:.6f}。全部原始生成及条件化坐标已保留，未只完成一个小任务。

**相关文件。** `{OUT}`含逐文件清理manifest、最终检查、客户端最终hash；Phase2616保留测量校验；前端热力图结果类型`output_conditioned_crossmodel_field_heatmap`，使用全坐标模式查看原始2560/5120/4096/3584维，不将模型基底对齐。

**成果、理论进展与硬伤。** 可确认的是受控局部信息在多层被读取、全坐标方向和语言表面条件有关；有测试donor的干预、无测试激活的训练字典、规范接口下生成分别列账。条件字典依赖外部替换词形，可能直接给出答案身份，不能称语义编码器。时间同义词与新事件仍须分别检验；源位置之外的分布式载体尚未定位为完整链。一次删除无效不否定路线，但复制成功也不闭合机制。

Phase2617族均值与条件字典的范数不同，因此二者成功率差不能单独归因于条件粒度；字典对同范数roll和不同操作对照的结果才是相应方向证据。

**存储精度边界。** 早期模型推理为BF16但原场导出FP16，不能保证极低数值逐位不丢失。Phase2615/2617 source场采用FP32容器保留BF16采集值，并附下溢/溢出校准清单；这不是量化模型，也不是把FP32存储冒称FP32推理。旧低幅值能量图应结合这一边界解读。

**第一性原理与下一突破口。** 真正需要学习的是给定外部操作和当前上下文时、各token位置×layer×原生坐标如何共同改变的条件规则，而不是给每个语言族指定一个常量向量。下一完整任务应冻结全新事件与输出身份，测量source之后各位置的全坐标传播，预测未见组合，再用自主生成验证模型是否使用该规则；这仍是未解决研究目标，不能把工程检查当成破解完成。“无限组合”在实验中操作化为未见组合泛化，不是声称有限上下文模型能接收字面无限长输入；“提取大脑数学结构”仍为研究假说，不能由语言能力单独推出。

**结论。** 本轮冻结实验、必要纠错与两轮同目标自动续研已执行并记录；没有发现可宣称普遍闭合的语言编码机制。
'''
    if '## Phase 2618:' not in MEMO.read_text(encoding='utf-8-sig'):
        with MEMO.open('a',encoding='utf-8') as f:f.write(text)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
