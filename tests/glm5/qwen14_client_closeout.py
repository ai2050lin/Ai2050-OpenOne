"""Verify delivery identities, retain the integration sources, append the required memo."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from deploy.storage_policy import check_write
OUT=ROOT/'tests/glm5/result/qwen14_client_v2'
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    for name in ['api-acceptance.json','browser-acceptance.json','desktop-acceptance.json','package-checks.json']:assert read(name)['passed'],name
    manifest=sha(OUT/'native/manifest.json')
    assert manifest in (OUT/'delta-verify.log').read_text(encoding='utf-8')
    assert read('deployment-readback.json')['http://127.0.0.1:5034']['/api/health']['version']=='1.0.36'
    assert 'Ran 29 tests' in (OUT/'integration-tests.log').read_text(encoding='utf-8')
    protocol=(OUT/'protocol-tests.log').read_text(encoding='utf-8')
    assert 'Ran 315 tests' in protocol and 'FAILED (failures=1)' in protocol and 'SOURCE-BINDING-20261110.source' in protocol
    packages=[]
    for p in [ROOT/'release/ai2050-center-1.0.36.tar.gz',OUT/'desktop/AI2050-Setup-1.0.14-x64.exe',OUT/'ai2050-qwen14-b28-b39-delta.tar.gz',OUT/'ai2050-qwen14-b28-b39-research-upload.tar.gz']:
        digest=sha(p);assert p.with_suffix(p.suffix+'.sha256').read_text().split()[0]==digest
        packages.append({'path':str(p),'bytes':p.stat().st_size,'sha256':digest})
    sources=['server/qwen14_mlp_service.py','server/research_space_service.py','deploy/qwen14_mlp_pack.py','deploy/research_library.py','deploy/README_QWEN14_MLP.md','frontend/package.json','frontend/package-lock.json','frontend/src/components/v8/Qwen14MlpAtlas.jsx','frontend/src/components/v8/ModelStructure.jsx','frontend/src/components/v8/ResearchInterventionReport.jsx','frontend/src/components/v8/ResearchSpace.jsx','VERSION']
    sources += [str(p.relative_to(ROOT)).replace('\\','/') for p in (ROOT/'tests/glm5').glob('qwen14_client_*') if p.is_file()]
    sources += ['tests/glm5/test_qwen14_client_v2.py','tests/glm5/test_qwen14_pagination.py','tests/glm5/qwen14Atlas.frontend.test.jsx']
    hashes={n:sha(ROOT/n) for n in sources}
    with zipfile.ZipFile(check_write(OUT/'integration-sources.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for n in sources:z.write(ROOT/n,n)
    now=datetime.now().strftime('%Y-%m-%d %H:%M');memo=ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';before=memo.read_bytes()
    phase=max(map(int,re.findall(r'^## Phase (\d+):',before.decode('utf-8'),re.M)))+1
    entry=f'''

## Phase {phase}: Qwen3-14B全208896单元的客户端接入与安装交付 [{now}]

**范围与完成状态。** 按用户“完成后确保全部可以在客户端查看”的指令，将Phase2916的B28–B39完整档案接入成果库、报告与主空间对象详情。每层U0–U17407，共208896张观察卡；保留旧B28–B35独立运行。客户端1.0.14、中心1.0.36。新索引、原生接口与图形均使用同一模型快照和运行ID。浏览、切换与核验不执行GPU前向。公网仍为1.0.29且14B端点404；按既定手动发布安排，本轮只交付并验收本机与发布包，没有声称公网已经可用。

**内容与方法。** 全量G/V/A统计、长前缀正负语境、三条完整5120维参数向量、原生计算输入与单列写出、相邻层全部17408关系、双负区间、原预测和次级标准化预测、冻结父单元及24个已测单元的逐样本干预均可查。旧16个与新8个单元各保留原实验基线和失败对照，未做跨运行统计合并。原生定义G=gᵀz，V=uᵀz，A=SiLU(G)V，写出Ad；双负绝对激活份额为Σ|A|I[G<0,V<0]/Σ|A|，不是语义贡献比例。客户端显示不增加数学或因果证据。

**数据和查询实现。** 组合原生清单含206个文件，本地通过同卷硬链接共享旧、新冻结数组；不修改原文件。增量导入器逐文件校验，复用旧包，拒绝损坏、越界路径和不同版本覆盖，暂存失败清理后可恢复。研究登记包含424个对象、208896张卡。新增规模暴露了成果检索超时，改用原生身份和运行索引查询最新可见版本，保留旧版本过滤不能复活已替换卡片的语义。本机全运行分页2.19秒，精确末端卡检索0.72秒；这是单机测量，不是服务器性能承诺。

**实际验收。** API核对12层所有统计数组、24个首尾单元的完整参数与响应、11个相邻边界、24个已测单元的自然/能力末样本及8种错误关卡。浏览器11项通过，含12层全场、末端参数、关联、原始干预、旧版、成果定位、刷新和手机无溢出；页面异常和GPU请求均为0。真实打包Electron确认isPackaged、版本、D盘隔离配置及B39/U17407的坐标5119原值0.03759765625，52个内置文件与源文件相符，额外节点资源也逐一一致。未在用户系统运行安装向导。29项集成测试、63项前端测试通过；专用组件lint通过，两个共享组件的既有lint问题逐条对比保持不变。

完整协议运行315项，314通过，唯一失败是既有SOURCE-BINDING-20261110.source没有语言项映射。该文法本轮未改，不以改动冻结材料来掩盖失败。首轮受限HTTP测试停滞后改用允许本机socket的隔离执行；另一次快照测试默认扫描历史研究到临时库，发现后停止，删除该轮2.743GB副本并以空研究目录重跑。其间D盘余量短暂低于20GiB，清理后恢复；正式原场与结果未删除。浏览器两次测试定位器分别误选隐藏option和4B式标题，修正定位器后完成全部实测。

**交付与复算。** 执行qwen14_client_build.py native/catalog、qwen14_client_api_acceptance.py、qwen14_client_browser.cjs、qwen14_client_desktop.cjs、unittest与vitest；中心由scripts/build_center_release.py构建，桌面由scripts/build_desktop.cjs构建，增量由deploy/qwen14_mlp_pack.py build/verify核对。中心包全部305成员、研究对象哈希、原生增量成员均通过。完整路径、SHA256与源代码快照见tests/glm5/result/qwen14_client_v2/delivery.json及integration-sources.zip；操作说明在deploy/README_QWEN14_MLP.md。本地预览5034保留供查看，公网更新须完成中心、旧原生基础包（如未安装）、新增量和登记包四项衔接。

**资源与清理。** 全部新增项目数据写D盘。完整复制的30GiB方案未通过磁盘门后改为不可变硬链接；增量约6.27GiB，桌面安装包约108MiB。桌面打包前暂移除本轮可重建增量，降低峰值，之后重新构建并校验。清理本轮测试配置、浏览器缓存、构建工具暂存和协议测试临时库；范围和字节见cleanup.json及intermediate-cleanup.json。保留原场、原版失败结果、可运行客户端、发布包、日志和复算代码，客户端原生组合目录作为复用查询数据，复查期限2026-12-13。

**理论边界与下一大任务。** 本轮科学增量为0；可访问性与完整索引不等于机制破解率。正式语义确认仍为0。第一性原理仍是研究固定读写参数如何在语境中形成G/V联合条件，并被后续计算使用。下一阶段承接Phase2916：在新词、新句式、新组合家族上冻结条件说明书，同时预测G、V、A与输出，比较完整词身份/embedding、背景、n-gram、原生组合及适用字典方法，再做有匹配负控制和恢复的成组干预。界面应帮助回查反例和未解释残差，不能用展示规模替代这组验证。

通俗说明：所有已分析单元现在都有客户端入口，可从总体图找到一个单元，再看到真实数值和实验记录；这解决“能否查到”，没有把观察档案变成已经确认的语言机制。
'''
    check_write(memo)
    with memo.open('ab') as f:f.write(entry.encode('utf-8'))
    assert memo.read_bytes()[:len(before)]==before
    delivery={'phase':phase,'timestamp':now,'center':'1.0.36','desktop':'1.0.14','native_units':208896,'semantic_confirmed':0,'remote_deployed':False,
        'packages':packages,'native_manifest_sha256':manifest,'source_hashes':hashes,'source_archive_sha256':sha(OUT/'integration-sources.zip'),
        'checks':{'browser':11,'desktop_payload_files':52,'api':True,'integration_tests':29,'frontend_tests':63,'protocol_passed':314,'protocol_total':315},
        'known_failure':'Existing SOURCE-BINDING-20261110.source language-item mapping absent; frozen grammar unchanged.',
        'memo_prefix_bytes':len(before),'memo_prefix_sha256':hashlib.sha256(before).hexdigest(),'memo_line':before.count(b'\n')+3}
    check_write(OUT/'delivery.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2),encoding='utf-8')
    print({'phase':phase,'packages':len(packages),'memo_prefix_unchanged':True},flush=True)
if __name__=='__main__':main()
