"""Append implementation status to the prior graph-first data plan, retain old CAS."""
import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).parent))
import phase2790_research as P
from deploy.research_library import Library


def main():
    W=P.W;lib=Library(W.ROOT/'distributed_data/research_library')
    old=W.read(W.ROOT/'tests/glm5/result/phase2777_confirm/dataset-atlas-plan-registration-current.json')
    d=copy.deepcopy(lib.artifact(old['sha256']));registration=W.read(P.OUT/'client-registration-current.json')
    d.update(run_id=P.RUN,kind='proposal',implementation_status='partial',supersedes_sha256=old['sha256'],
        provenance=W.read(P.OUT/'corpus.json')['provenance'])
    d['current_case']={'report_sha256':registration['report_sha256'],'graph_sha256':registration['graph_sha256'],
        'input_count':96,'generation_count':384,'families':8,'physical_nodes':1,'independent_intervention_tests':1,
        'scope':'exploratory externally AI-rated, not confirmatory mechanism evidence'}
    d['stages']=[
        {'stage':'A · 数据、候选操作与主张登记','status':'部分完成','work':'已登记11个数据版本；同一联合参数操作连接3个不同范围/指标的主张，保留原CAS及新别名。原文档、改写、翻译和派生数据须共用来源组；当前只完成已接入受控事实与题目哈希的精确去重。',
         'acceptance':'新增材料必须带内容哈希、来源、许可、标签质量与家族切分。下一步补通用sample_lineage和跨库近重复审计；数据集数量不计破解进度。'},
        {'stage':'B · 双AI测量与保义可靠性','status':'发现校准已完成，确认集未运行','work':'两种外部AI独立盲评84条旧输出/参考、32条语义控制和307条去重真实输出；本轮96输入、384生成。基线两种语域均可识别，但口语保义32/48。100条反例/分歧记录可回查。',
         'acceptance':'下一版先按语域、领域分别预注册保义与自然性门，再冻结未见内容家族与句式。两位AI一致只表示AI口径可靠性，仍保留共同偏差与不确定性；用户暂无人工资源不阻断探索。'},
        {'stage':'C · 自然前向特征发现、变量映射与计算边','status':'生成原场已完成，机制提取待执行','work':'已保留37层HiddenState、B18..27全部9728单元、5个提示位置和所有有效生成步。先在获许可自然长语料估计背景协方差并训练/接入SAE或transcoder；在等长多句式对照上检验特征，再按需追踪参数路径。',
         'acceptance':'A字典候选、B留出选择性、C归因/干预各自登记，使用相同冻结快照和语料；hs[0]、位置、词汇、句长及随机标签基线齐全。先采集后重算，不将均值差当唯一发现算法。'},
        {'stage':'D · 候选更新、自然生成剂量与能力保持','status':'旧候选已测试，范围未支持','work':'旧编辑alpha6未产生合格自然书面→口语翻转；原2200题监控在80块BF16编辑哈希逐字节一致后复用。固定候选正结果保留，但不迁移为自然生成成功。',
         'acceptance':'新候选须从发现材料提取，确认集选剂量，锁箱最终验收；自然生成剂量网格、能量匹配列/范数匹配方向、完整保义与开放任务能力保持同时预注册。128内容家族、delta=.20、p_d=.5、alpha=.025，规划功效约.83；若任务/效应改变重算功效。'},
        {'stage':'E · 分布式采集与交叉复算','status':'真实中心/节点闭环已完成，第二节点待测','work':'natural_register任务已进入选择任务类型、快照匹配、模板冻结、家族分片、CUDA执行、断点文件、分块上传与完成停止派发。此次8分片只有1台物理节点。',
         'acceptance':'至少另一物理节点按同一快照/精度/runner复算；核对金丝雀、编辑哈希、生成和统计；异构数值差异预注册容差，不自动合并冲突。大原场留在节点，中心存哈希索引。'},
        {'stage':'F · 图谱预测与跨模型复现','status':'未执行','work':'冻结已验证特征与有类型边，仅用图谱预测未见前缀的特征激活及next-token分布，与词汇身份、位置及n-gram基线比较。方法有效后依次扩展GLM4-9B、Qwen3-14B。',
         'acceptance':'留出KL与损失恢复比例、失效样本和资源成本齐全；不能完整重跑原模型再称图谱解释。各模型坐标独立，不跨精度平均，CUDA模型顺序执行。'}]
    d['implementation_artifacts']={'catalog_endpoint':'/api/research-library/catalog','report_sha256':registration['report_sha256'],
        'task_kind':'natural_register','template':'NAT-REGISTER-P2790-v1','reviews':['P2790-AI-calibration-v1','P2790-AI-semantic-controls-v1','P2790-AI-generation-v1']}
    path=P.PUBLIC/'dataset-atlas-plan.research.json';W.save(path,d,True);ref=lib.ingest(d,str(path.relative_to(W.ROOT)),kind='local_report')['sha256']
    W.save(P.OUT/'dataset-plan-registration.json',{'sha256':ref,'implementation_status':'partial'},True)
    P.status('dataset_atlas_plan','partial',plan_sha256=ref)
    print({'plan_sha256':ref,'status':'partial'})


if __name__=='__main__':main()
