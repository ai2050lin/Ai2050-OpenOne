"""Missing equal-length coverage is unavailable, not a zero response."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
import phase2628_native_atlas_delivery as atlas
previous=read(atlas.OUT/'analysis/basic_maps.json')
save(atlas.OUT/'analysis/basic_maps_before_coverage_correction.json',previous)
profiles,corrected=atlas.analyze();publication=atlas.publish(profiles)
omitted=[r['group'] for r in corrected['tokens'] if not r['equal_length_shared_downstream_tokens']]
save(atlas.OUT/'analysis/coverage_erratum.json',{'groups_without_equal_length_coverage':omitted,'rendering':'omit unavailable response rows; summaries use null, never infer absence from no measured cases','publication':publication})
with MEMO.open('a',encoding='utf-8') as f:
    f.write('\n\n**Phase2628覆盖口径补记（append-only）** ['+datetime.now().astimezone().strftime('%Y-%m-%d %H:%M')+'] 等长相同token下游检查在 '+', '.join(omitted)+' 没有合格token，原汇总同时给出了n=0与默认0。现已把无覆盖量标为null、移除客户端无覆盖行，不把“未测”画成“零响应”。此前报告保存在basic_maps_before_coverage_correction.json，修正报告basic_maps.json及coverage_erratum.json；所有原始模型测试值未改。苹果同token词嵌入为零来自独立锚点校验，与这里无等长下游覆盖不同。\n')
print('coverage corrected; unavailable groups:',omitted)
