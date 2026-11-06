"""Append the actual engineering acceptance; never replace prior incomplete records."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.research_library import Library

def main():
    out=ROOT/'tests/glm5/result/workbench_playwright_20261104'
    names=['browser-acceptance.json','electron-acceptance.json','electron-reload-acceptance.json','packaged-smoke.json','package-verification.json']
    checks={name:json.loads((out/name).read_text(encoding='utf-8')) for name in names}
    assert all(v['passed'] for v in checks.values())
    assert re.search(r'Tests\s+61 passed',(out/'navigation-tests.log').read_text(encoding='utf-8'))
    assert 'tests 11' in (out/'electron-tests.log').read_text(encoding='utf-8')
    assert 'built in' in (out/'final-build.log').read_text(encoding='utf-8')
    stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    marker='WB-VISUAL-RELEASE-20261104-v1'
    paths=[out/n for n in names]+[out/'release-source-manifest.json']+sorted(out.glob('[0-9][0-9]-*.png'))
    doc={'schema':'research_client_acceptance.v1','kind':'engineering','run_id':marker,'created_at':stamp,'timezone':'America/Chicago',
         'status':'visual_acceptance_passed_and_packaged','version':'1.0.5','previous_implementation_sha256':
         hashlib.sha256((ROOT/'research/workbench_navigation/implementation-20261104.research.json').read_bytes()).hexdigest(),
         'checks':checks,'frontend_regressions':61,'electron_regressions':11,'production_build':'passed',
         'fixes':['数据与同步弹窗居中、限制高度并允许长版本号换行','390px顶栏品牌保持单行，隐藏重复副标题','网页贡献指引改为配置→本机算力'],
         'test_method':['本地Playwright Chromium真实渲染与点击，四一级/十四二级及旧链接','真实P2794归档→样本→B34/H10/U1995→权重切片，核验完整快照并检查后退/刷新',
                        '全部32Head索引和9728单元阵列、真实query位置、计算来源显示','真实Electron main/preload与独立本机节点，未绑定启动保护与资源预算实际保存',
                        '真实主包和研究库流下载、IndexedDB提交、重启/刷新后离线图谱与工作包选择保留','打包后exe实际启动、app.asar身份和真实节点IPC；逐字节校验打包内容'],
         'model_runs':0,'scientific_evidence_upgraded':False,
         'limits':['未执行新GPU模型任务，未验证新远端节点绑定后领题上传或真实托管授权','安装包未安装、未发布，Authenticode状态NotSigned',
                   '仅当前Windows机器与1440/1584宽及390px视口，不代表全部系统/显卡/设备','保留既有大bundle警告；3D索引布局不是语义几何或唯一因果机制'],
         'evidence':[{'path':str(p.relative_to(ROOT)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in paths]}
    dest=ROOT/'research/workbench_navigation/acceptance-release-20261104.research.json'
    assert not dest.exists(),'Use a new revision; do not rewrite acceptance history'
    dest.write_text(json.dumps(doc,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    ref=Library(ROOT/'distributed_data/research_library').ingest(doc,str(dest.relative_to(ROOT)),kind='engineering')['sha256']
    memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    old=memo.read_text(encoding='utf-8');assert marker not in old
    phase=int(re.findall(r'^## Phase (\d+):',old,re.M)[-1])+1
    release=checks['package-verification.json']
    note=f'''\n\n## Phase {phase}: 正式客户端Playwright视觉闭环、真实离线研究库与1.0.5打包 [{stamp}]

标识{marker}。用户明确要求改用本地Playwright，完成真实浏览器验收、发现问题直接修复、通过后打包。本轮使用本机已安装Playwright与Chromium，另启动实际Electron主进程/预加载脚本及正式打包exe；不安装额外浏览器、不借用用户登录会话。新增记录CAS {ref}，原实施/未验收记录原样保留。所有脚本在tests/glm5/，隔离profile/节点目录在tests/glm5_temp/，结果在tests/glm5/result/workbench_playwright_20261104/。

测试原理：一方面点击真实导航并核对URL、浏览器后退和刷新；另一方面用实际后端封存数据核对来源快照与原生参数，再用真实桌面IPC检查节点保存。数据同步以完整流结束、IndexedDB元数据提交及离线查询命中作为通过条件，不能只看按钮消失。数学身份约束为SHA256(所选权重快照)=来源记录的完整快照；只核对路径/数组来源与数值，不把相同索引当成相同功能。模型MLP写出形式y=W_down[silu(W_gate h)⊙(W_up h)]，页面按实际原生单元索引和参数来源回查；该式是架构计算定义，不是本轮发现的新定律。

结果：Chromium六组真实验收全部通过，覆盖四一级/十四二级、八类旧入口、真实P2794研究到样本与B34/H10/U1995参数、32头和9728单元阵列、query位置、后退/刷新、网页匿名一键包实际下载、主包持久化及一次明确注入503后的可重试状态、390px窄屏。没有用模拟研究数据替代真实图谱。单独的生产Electron检查通过真实main/preload、真实未绑定节点启动保护、资源预算写入、本地主包下载和完整研究库流下载；记录4645条附加包条目，离线索引包含26工作包/1314对象/4519产物版本。这些是归档覆盖数量，不是破解率。再次重启同一隔离profile，在1.0.5构建上选P2794工作包并刷新，选择与离线图谱保留。各次最终验收均无pageerror。

实际修复：数据与同步原生dialog靠左上，改为居中、限制高度并允许长版本号换行；390px顶栏品牌/标题挤压，品牌保持单行并折叠重复副标题；网页贡献提示里的旧菜单名称改为“配置→本机算力”。初次桌面验收过早把附加包提示消失视为下载完成；后改真实IndexedDB提交轮询，确认不是图谱数据丢失。验收脚本修正不作为产品缺陷或机制发现。

验证与交付：导航专项10文件61项回归、Electron 11项回归及生产构建通过；另有基础14项测试通过，与专项存在重叠，不相加作独立数量。1.0.5包内41个前端文件、5个Electron入口文件及6项节点资源逐字节SHA256一致，Windows PE为x64。实际启动win-unpacked exe，确认app.isPackaged=true、app版本1.0.5、app.asar和预加载桥接、真实节点/status返回200且未绑定/停止。安装包{release['bytes']}字节，SHA256 {release['sha256']}，路径frontend/dist-installer/1.0.5/AI2050-Setup-1.0.5-x64.exe，校验文件SHA256SUMS.txt。安装程序未执行、未安装/发布；Authenticode为NotSigned，既有bundle体积警告保留。

严格审视：本轮没有新模型前向、训练、参数编辑或机制证据晋级；节点验收覆盖实际IPC、资源保存和授权前保护，没有声称跨机器绑定、GPU领题上传与托管授权完成端到端复算。Windows本机截图不能代表所有设备。图谱页面的空间距离仍是索引布局，不是模型语义距离。附加包落盘、归档数量和UI通过不构成语言编码机制破解。完整实验的科学门、反例和证据等级继续使用既有预注册与审计。

智能理论层面的工程增量：同一模型、同一样本、同一原生索引及来源哈希能从图谱走到参数并在离线更新后回查，减少把展示差异误当成机制差异。第一性研究问题仍是：冻结条件下哪些可推广的特征与转移关系能在未见材料上预测真实激活及next-token分布，而不是图上是否连满。下一阶段按一个大闭环组织：冻结长文本与分族留出→一次采完整原生场→字典/受控对照/因果归因三引擎比较→超出词汇身份与位置基线的留出预测保真度→独立节点复算与反例回流；单独设跨机实际领题/断点上传/审计合并验收，不把本次未绑定节点检查替代它。

通俗说明：这次真正打开并操作了正式页面和桌面程序，修好了弹窗与窄屏问题，验证真实数据可以下载、重开和回查，随后才打包新版。研究结果可以继续沿已有归档积累；本次交付是可信研究工具的验收闭环，不是新增“已破解”结论。
'''
    with memo.open('a',encoding='utf-8') as f:f.write(note)
    summary={'phase':phase,'acceptance_sha256':ref,'version':'1.0.5','timestamp':stamp,'scientific_evidence_upgraded':False}
    (out/'registration.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False))

if __name__=='__main__':main()
