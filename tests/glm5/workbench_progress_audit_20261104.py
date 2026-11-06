"""Read-only audit of the overview's denominator and current archive. No inference."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import urllib.request

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tests/glm5/result/workbench_progress_audit_20261104'

def get(path):
    with urllib.request.urlopen('http://127.0.0.1:5001'+path,timeout=60) as r:
        return json.load(r)

def main():
    stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    progress=get('/api/research-library/progress')
    work=get('/api/research/work-units')
    space=get('/api/traces/research-space')
    doc={'schema':'overview_progress_audit.v1','kind':'engineering','created_at':stamp,'timezone':'America/Chicago',
         'model_inference_executed':False,'progress':progress,'work_unit_counts':work.get('counts'),
         'work_unit_rows':len(work.get('work_units',[])),'space_counts':space.get('counts'),
         'registered_p2794':[{'run_id':s['run_id'],'model_snapshot':s.get('model_snapshot'),'localization':s.get('localization')} for s in space.get('studies',[]) if s.get('run_id')=='P2794-ATOMIC-ARBITRATION-v4'],
         'findings':['卡片口径是正式模板唯一合格分片/计划分片，不是全部研究或机制破解比例',
                     '当前三个桶均为旧cfg身份；snapshot_verified仅用64位十六进制格式检查，不等价于重新核验权重',
                     '总览选择本地bucketId或第一个桶，不读取全局mdl/snapshot',
                     '直接研究报告、原生轨迹、普查及预注册干预未统一进入这三个分片分子',
                     '桶从结果表生成；没有结果的模型缺少待执行桶，分母为全部映射的正式模板而非显式模型研究计划'],
         'scientific_evidence_upgraded':False}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'report.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    marker='WB-OVERVIEW-PROGRESS-AUDIT-20261104-v1'
    memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    if marker not in memo.read_text(encoding='utf-8'):
        # This audit continues Phase 2791's client acceptance; it is not a new experiment Phase.
        note=f'''\n\n### 总览三方向进度口径复核 [{stamp}]

标识{marker}，续Phase 2791。用户询问知识网络/分析推理/语法体系卡片为什么待核验或无进度。只读复核ResearchOverview.jsx、deploy/research_library.py与实际中心API，结果tests/glm5/result/workbench_progress_audit_20261104/report.json；没有模型前向或参数变动。公式为100×(同模型身份及精度桶中、正式冻结design一致、审计passed、金丝雀ok/reference且去重的分片数)/(映射到该方向的正式模板计划分片数)，不是破解率。当前Qwen3-4B、GLM4-9B、Qwen3-14B三个桶分别使用cfg-8ba006f74fec、cfg-8ec64bc0dff4、cfg-e73c3664ca09；知识10片、语法10片、推理0片计划。旧cfg不满足完整快照格式，两个有分母方向显示待核验，不能将被排除的0片误说成没有研究。推理为无冻结分母，不是已确认0%破解。接口的snapshot_verified目前仅检验64位小写十六进制格式，仍不等价于重新读取并验证权重。

发现显示/统计缺口：总览用独立bucketId默认第一桶，不跟随全局mdl/snapshot；图谱空间的完整快照和P2794直接研究报告不自动进入该分片计数。分母从全部映射正式模板求和，缺少显式模型×阶段研究计划；只有已有结果才生成模型桶，零结果任务难以呈现。当前工作单返回0条也不等于全部研究没有任务。正确改进方向为三方向分别展示阶段任务执行、已登记图谱证据、留出预测/机制成熟度；统一完整快照与计划ID，明确已完成但身份未核验/待审计/计划未冻结的数量和原因，不把覆盖率或归档量替换为破解率。旧cfg结果只能依据原执行manifest或可验证快照关系恢复，不能把当前权重hash直接回填历史结果。

理论进展与边界：本次是统计口径发现，没有新增编码机制结论。统一身份和分母的价值是防止将来源未知、未纳入与实验失败混为一谈；研究突破仍需在冻结留出材料上验证图谱对激活/next-token的预测增益及因果范围。下一阶段应以“完整模型身份—明确研究计划—三引擎证据—客户端可回查”作为一个共同闭环，而非先修改百分比。通俗说明：数据仍在，但总览当前只看了一类旧分片账本，不能据此判断全部研究进展。
'''
        with memo.open('a',encoding='utf-8') as f:f.write(note)
    print(json.dumps({'bucket_count':len(progress.get('buckets',[])),'space_counts':doc['space_counts'],'p2794':doc['registered_p2794']},ensure_ascii=True))

if __name__=='__main__':main()
