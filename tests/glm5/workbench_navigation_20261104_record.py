"""Archive the applied UI contract and append an honest verification record."""
from pathlib import Path
import hashlib
import json
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from deploy.research_library import Library

def main():
    out=ROOT/'tests/glm5/result/workbench_navigation_20261104';tests=(out/'frontend-tests.log').read_text(encoding='utf-8');build=(out/'build.log').read_text(encoding='utf-8')
    assert re.search(r'Tests\s+61 passed',tests) and 'built in' in build
    paths=['frontend/src/components/app/Workbench.jsx','frontend/src/components/app/workbenchUtil.js','frontend/src/components/app/ConfigView.jsx',
       'frontend/src/components/app/WorkbenchHelp.jsx','frontend/src/components/app/researchContext.js','frontend/src/components/app/workbench.css',
       'frontend/src/components/v8/SpaceLens.jsx','frontend/src/components/v8/ResearchSpace.jsx','frontend/src/components/v8/ResearchExecution.jsx',
       'frontend/src/components/v8/ParameterExplorer.jsx','frontend/src/components/v8/MechanismTraceExplorer.jsx','frontend/src/components/v8/LangAtlas.jsx',
       'frontend/src/components/app/rdc_fusion/AtlasView.jsx','tests/glm5/workbench_navigation_20261104.test.jsx',
       'tests/glm5/workbench_node_settings_20261104.test.jsx','tests/glm5/workbench_graph_identity_20261104.test.jsx']
    stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M');marker='WB-NAV-IMPLEMENTATION-20261104-v1'
    doc={'schema':'research_navigation_implementation.v1','kind':'engineering','run_id':marker,'created_at':stamp,'timezone':'America/Chicago',
       'status':'implemented_and_regression_checked_visual_acceptance_incomplete','proposal_sha256':'485e4f211196e0152c40f46f6f0c6a6e8ba1ac4330deb151d1d1dc9c23093f26',
       'primary_count':4,'secondary_count':14,'application_changed':True,'model_runs':0,'scientific_evidence_upgraded':False,
       'changes':['统一图谱浏览及三个图层、四个空间页面','账号与节点拆分，贡献算力常驻顶栏','研究计划不重复加载报告，实验队列内选择专业工位',
           '旧入口迁移，主要页面历史和popstate恢复，源实验/快照/样本持续回查','参数快照核验、缺失模型和图谱桶不自动替换、索引越界约束',
           '主包/附加包同步和本地导入、历史实测能力下钻、教学工具限于高级兼容'],
       'checks':{'frontend_tests':61,'test_files':10,'production_build':'passed','build_warning':'existing large bundle warning remains',
          'visual_acceptance':'not_completed','visual_reason':'Computer Use could not determine current browser URL confidently enough to enforce policy and stopped for the turn',
          'security_bypass_attempted':False,'installer_packaged':False},
       'source_files':[{'path':p,'sha256':hashlib.sha256((ROOT/p).read_bytes()).hexdigest()} for p in paths],
       'logs':[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [out/'frontend-tests.log',out/'build.log']],
       'limits':['DOM回归和编译不能替代浏览器视觉验收；未声称检查全部屏幕尺寸和真实设备交互。',
          '旧数据集实测仍使用本机服务已加载模型；界面提供模型不匹配隔离，不自动更换GPU权重。',
          '本机独立Goodfire测试的科学结果仍需现有登记/审计，不因菜单迁移自动写为机制结论。']}
    p=ROOT/'research/workbench_navigation/implementation-20261104.research.json';raw=json.dumps(doc,ensure_ascii=False,sort_keys=True,indent=2)
    if p.exists() and p.read_text(encoding='utf-8')!=raw:raise ValueError('create a new implementation revision')
    p.write_text(raw,encoding='utf-8');ref=Library(ROOT/'distributed_data/research_library').ingest(doc,str(p.relative_to(ROOT)),kind='engineering')['sha256']
    memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';old=memo.read_text(encoding='utf-8')
    if marker not in old:
        n=int(re.findall(r'^## Phase (\d+):',old,re.M)[-1])+1
        note=f'''\n\n## Phase {n}: 正式客户端导航重构、对象上下文与61项回归验证 [{stamp}]\n\n标识{marker}，执行用户已批准的工作台整理方案；方案CAS为485e4f211196e0152c40f46f6f0c6a6e8ba1ac4330deb151d1d1dc9c23093f26，实现记录CAS {ref}。本轮为正式前端修改，没有运行模型、编辑参数、改实验阈值或新增机制结论。\n\n一级固定进展/空间/研发/配置，共14个二级：进展总览/图谱覆盖/成果与证据；空间图谱浏览/结构与参数/运行与路径/响应对照；研发研究计划/材料与数据/实验与队列；配置连接与诊断/账号与权限/本机算力/模型与AI。语言模式、内部特征与研究证据、跨族映射收敛为图谱页内图层。参数和原生单元特征卡共享入口；历史封存激活及保真度以独立数据源下钻保留。旧本机工具从一级撤下，配置高级兼容区保留历史功能与标注演示，修复空onGo跳转。\n\n顶栏贡献算力直达已有节点服务面板；账号页不再预先挂载算力控制。数据与同步弹窗使用已有主包/附加包下载与浏览器zip解析，保留版本和结果提示。控制能力、角色授权、后台协议与科学晋级规则不变。研究计划只列问题/阶段与预注册引用，点击进入同一实验报告；实验与队列用类型选择器进入Goodfire、流水线、普查与自动测试，完整原报告和反例继续调用现有详情渲染。\n\n主要路由用pushState、popstate恢复，旧图谱/神经元/本地工具及旧流水线URL归一；source run/snapshot/trace持续回查，离开页面时清理活动索引。实际样本发布自身快照，参数页完整hash匹配才允许响应叠加；缺失本机模型或跨族模型桶时不回退其他模型；书签越界索引约束到真实维度。图谱模型选择与受控跨族视图联动；旧数据集实测仍使用本机已加载权重，模型名称不匹配结果不叠加，且不冒充完整快照证据。Embedding hidden[0]卡定位到输入嵌入板。\n\n验证：10个前端测试文件共61项通过，生产Vite构建通过，日志在tests/glm5/result/workbench_navigation_20261104/。新增针对旧链接、专业工位权限、来源与模型选择并发更新、popstate对象恢复、账号/节点拆分、来源身份订阅、缺失模型禁止替换和越界索引的检查。旧报告测试改为在执行视图读取，计划页阶段顺序和负结果保留。修复连续两次更新覆盖数据来源选择的状态错误；原生参数持久化后，旧测试的URL残留得到隔离。没有改后端runner，未重跑无关GPU与全协议测试。\n\n视觉验收未完成：计算机操作工具因无法可靠识别当前Windows浏览器URL而中止本轮操作；没有绕过限制或用另一种浏览器自动化路径代替。不能将DOM测试与生产构建当作视觉通过。未打包安装程序，未安装或发布。旧大包构建警告仍在。本轮实现完成的是界面结构与已有能力接线，不能声称完成所有机型/屏宽/离线真实设备验收，也不能把菜单迁移说成Goodfire全流程自动成果登记。\n\n科研意义与限制：将研究对象、模型身份、来源数据、执行状态、参数与证据分开，减少无效筛选和跨模型误绑定，为长期积累可追溯碎片提供一致入口；这不是语言机制的新数学规律，也不是破解比例增加。下一阶段围绕实际研究闭环推进：校准材料与行为→同一快照原场→预注册提取与反例→留出预测→独立复算；界面剩余工作是工具恢复后的真实浏览器、窄屏与桌面/网页权限验收，再按需要交付安装包。\n\n通俗说明：四个主入口各做一件事，图谱和参数回到同一对象，贡献算力与数据更新更容易找到，旧成果和失败仍可回查。代码和回归检查已完成，视觉验收明确保留为未完成项。\n'''
        with memo.open('a',encoding='utf-8') as f:f.write(note)
    print(json.dumps({'implementation_sha256':ref,'tests':61,'visual_acceptance':'not_completed','timestamp':stamp},ensure_ascii=False))

if __name__=='__main__':main()
