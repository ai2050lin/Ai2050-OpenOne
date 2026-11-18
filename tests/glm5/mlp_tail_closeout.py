"""Record actual acceptance, deliverable hashes, cleanup and append-only research memo."""
from datetime import datetime
import json
from pathlib import Path
import re
import mlp_tail_goodfire as C


def main():
    qa=C.ROOT/'tests/glm5/result/mlp_tail_acceptance';now=datetime.now().astimezone();stamp=now.strftime('%Y-%m-%d %H:%M')
    cleanup=[]
    for n in ['cleanup-old-copy.json','cleanup-final-copy.json','cleanup-obsolete-package.json']:
        p=qa/n
        if p.exists():cleanup.append(json.loads(p.read_text(encoding='utf-8-sig')))
    released=sum(int(r['bytes']) for r in cleanup)
    paths=[C.ROOT/'release/desktop-1.0.10/AI2050-Setup-1.0.10-x64.exe',C.ROOT/'release/ai2050-center-1.0.31.tar.gz',
           C.ROOT/'release/ai2050-research-mlp-tail-20261108.tar.gz',C.ROOT/'release/ai2050-mlp-tail-display-20261108-r2.tar.gz']
    files={}
    for i,p in enumerate(paths):
        files[p.name]={'path':str(p.relative_to(C.ROOT)),'bytes':p.stat().st_size,'sha256':C.file_sha(p)}
        print(f'deliverable {i+1}/{len(paths)} verified: {p.name}',flush=True)
    final={'schema':'mlp_tail_delivery.v1','run_id':C.RUN,'created_at':now.isoformat(),'model_snapshot':C.SNAPSHOT,
        'scientific_admissibility':'exploratory_observation_only','formal_mechanism_confirmed':False,
        'client_version':'1.0.10','server_version':'1.0.31','formal_server_modified':False,
        'files':files,'validation':{'protocol_checks':324,'atlas_api_checks':13,'frontend_tests':52,'electron_tests':11,
            'deployed_data_members':69,'duplicate_import':'already_installed','metadata_objects_verified':86,
            'center':C.read(qa/'center-package.json'),'desktop':C.read(qa/'desktop-payload.json'),
            'browser':{'four_full_layers':True,'last_unit_B35_U9727':True,'all_weights_and_long_prefix':True,'causal_and_dose':True,
                'screenshot':'tests/glm5/result/mlp_tail_acceptance/tail-atlas-3d.jpg','latest_admissibility_notice':'SSR tested after visual core acceptance'},
            'installation_wizard_run':False,'authenticode':'NotSigned'},
        'cleanup':{'deleted_file_bytes':released,'GiB':released/2**30,'records':cleanup,
            'retained':'All original native data, source bank, matrix/predictor evidence, scientific counterexamples, logs, screenshot, final releases; only validated copies and superseded unpublished display archive removed.'},
        'limitations':['No full native-neuron semantic decoding','No prospectively powered causal confirmation','No general capability noninferiority','No production upload/deployment performed']}
    C.save_json(qa/'final-acceptance.json',final);C.save_json(C.ROOT/'release/mlp-tail-delivery-20261108.json',final)
    delivery=f'''# 最后四层MLP图谱交付（{stamp}）

客户端 **1.0.10**，中心 **1.0.31**。本机真实计算与3D验收已完成；正式中心尚未修改，由用户手动升级。全部38,912单元为观察档案，不能称全部语义已破解；本轮准入为探索性观察。

| 文件 | 用途 | 大小 |
|---|---|---:|
'''+''.join(f"| `{name}` | {role} | {row['bytes']/2**20:.1f} MiB |\n" for (name,row),role in zip(files.items(),['Windows客户端安装器（未签名）','含网页版客户端的中心服务包','通过客户端上传协议登记的研究元数据','全部矩阵、尾层参数、原片、干预及准入审计展示数据']))+f'''
以上文件路径与SHA-256见 `mlp-tail-delivery-20261108.json`。不要使用旧版本中间构建。

1. 按原流程手动升级中心包1.0.31，保留数据目录和账号。
2. 上传r2展示数据包，按 `deploy/README_MLP_TAIL.md` 的命令进行大小/哈希/完整性校验导入到服务实际AI2050_DIST_DIR。
3. 正式客户端「上传研究包」选择小型研究元数据包，协议会登记报告、全部单元观察卡、反例和空间绑定。小包不代替大展示包。
4. 安装或更新Windows客户端1.0.10（或打开新版中心网页），在空间→LLM编码图谱→全部研发成果选择「Qwen3-4B最后四层」。可输入B35/U9727；原模型有限干预可选择B34/U1244。

正式部署前所有自动与浏览器验收均在本机隔离环境进行。中心只读取数据，不加载模型或运行GPU。升级步骤、数据边界和完整研究说明见 `deploy/README_MLP_TAIL.md` 与 `research/mlp_tail/v1/README.md`。

测试通过324项协议、13项新接口、52项前端、11项桌面；服务器293文件与源码一致，桌面51文件和6资源一致。最终展示包69文件完成导入及重复导入去重，研究元数据86对象校验通过。未运行安装向导，未配置代码签名。

清理本轮重复副本及被最终审计版替代的未发布展示包，共{released:,}字节（{released/2**30:.2f}GiB，按文件大小）。原始研究数据、关键证据、反例、截图和日志保留。
'''
    (C.ROOT/'release/MLP-TAIL-20261108-DELIVERY.md').write_text(delivery,encoding='utf-8')
    memo=C.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';before=memo.read_bytes()
    phases=[int(x) for x in re.findall(rb'^## Phase (\d+):',before,re.M)];phase=max(phases)+1
    text=f'''

## Phase {phase}: Qwen3-4B尾四层全量原生MLP观察、跨层失败与3D研究交付 [{stamp}]

**边界与目标。** 按用户“参考Goodfire，先完成最后几层全部神经元特性与前后关联”的执行请求，本轮固定本地Qwen3-4B BF16、B32—B35（第33—36层）、每层9728、总38912原生单元。模型快照{C.SNAPSHOT}。沿用当前一级/二级菜单和白底蓝橙配色，建设可积累的完整观察底座；没有把普查覆盖称为语义破解。正式服务器仍按用户手动升级安排，本轮未登录、上传或部署正式中心，没有使用其他模型的语言结果替代真实计算。

**材料、算法与冻结。** 复用已核对模型/tokenizer、原始token及8个原生输入/输出文件哈希的2048篇中英文自然文本，每篇1024-token前缀、16个533—1006位置，每层32768位置。发现1024文档，确认/留出各512，原语料已暴露，只能说未被本估计器拟合，不能称独立新语义盲测。原生输入为注意力后MLP的RMSNorm输出z；G=Wg z，V=Wu z，A=SiLU(G)⊙V，y=Wd A。分开记录G/V/A两极、全部正负组合和当前token线索，不以乘积正值把负门控解释为正语义。四层131072个位置输出全部逐位一致，relative error=0；这是已知计算式复算，不是新理论。四层完整gate/up/down及词表参数保留，逐单元显示全151936词表参数余弦两极，未包含上下文RMSNorm或后续层作用。

**全量关联及失败。** 每条相邻边界保存三个划分的全9728×9728 Pearson和两种完整权重余弦，共15矩阵。自然共同响应、参数兼容性、原模型有限写出干预分开；余弦略去残差/归一化/注意力，不是直接计算边。发现集选8父单元、ridge λ=.01n，比较A直接预测、G/V预测后复合、同编号、随机8父单元、token/语言/位置基线，冻结后分别评估。确认的直接FVU为0.92537/0.71921/0.65612；留出为1.30780/0.99939/0.93010，token基线0.98648/0.98049/0.95832。从真实B32开始递推至B35，留出直接FVU=2.46063，G/V组合约3.43e21；未裁剪、不改门补过。文档聚类bootstrap和反例索引均保留。没有得到可闭合机制，也不能由8父单元不足否定原生表示：完整跨位置残差/注意力缺失、语料变化和稀有大激活仍是替代解释。本轮没有完成完整APD或新transcoder的同预算对照。

**原模型检验。** 依据发现父单元使用频率固定12源单元，96自然长语境，写出倍数0/.5/1.5/2及随机方向，共9216干预/控制记录，并保存下一层全部9728个单元响应。编辑是在源block输出按(α−1) A_i Wd[:,i]做有限写出修改，并非BF16下重算缩放单元后的完整down GEMM；不可称唯一硬件贡献分离。后续保留原注意力、残差和所有层，零编辑后缀核验及每文档原模型重跑逐位恢复。BF16后随机控制L2相差≤1%才参与相同文档交集比较，52—96篇，失败对照保留。例B34/U1244两倍写出，96篇目标响应范数23.061 vs 5.205、下一词top1变化4 vs 2，为描述性实例，不是语义机制确认。只测自然下一词KL/NLL/argmax代理；没有新的KV回放、多步自由生成或翻译/推理/编程非劣效。GPU模型运行串行且有互斥；另一研究占用GPU时，参数词表点积改为CPU分析，不并行加载LLM。WDDM共享显存计数未取得，记录unknown而非0。

**必须保留的准入缺陷。** 收尾对AGENTS第13节审计发现：path-plan没有事前“期望行为效应—样本量—≥.8检验力”，原结果也没有完整五字段provenance信封。已保存原始冻结和原hash，单独加入事后溯源审计、git SHA/dirty、实际model path/revision/seed；明确标记post-run，不追认为事前执行授权或正式因果确认。所有客户端单元卡仍为observation，确认语义解释0，正式机制确认false。以Δ=.05、p_d=.1、48比较作未来设计示例，tools/power_analysis.py给出n_required=680；n96对应约.042功效，不能用小样本翻转过门。这是未来规划/事后诊断，不冒充本轮预注册。

**客户端与发布链路。** CPU只读/api/research/mlp-tail校验不可变manifest，提供四层全数组、任意单元、三组各2560完整参数、两极长前缀、所有9728关系值、匹配控制及剂量。全部38912观察卡按稳定模型/层/原生索引登记研究库，保留反例，稠密相关矩阵不自动登记因果边。3D只用原生编号排板，距离不叫语义几何；点击/输入可查询，二维及完整数值按需展开，未干预显示未测。浏览器实操核验B35/U9727、B34/U1244、G/V/A负极、完整参数、原片与许可、源单元选择、剂量曲线和预测失败，截图保存；最后增加的准入文字另经SSR验证。链接保留tail_block/tail_unit。没有改菜单或全局H3/H3b主线结论。

**工程验证与恢复。** 324协议回归、13新接口测试、52前端、11桌面检查通过。新接口覆盖全量边界、矩阵行列一致、无数据/篡改阻断、两极因果可见性、有效匹配交集与未知干预空态。最终中心1.0.31的293文件与源码逐字节一致，用包内CPU服务和正式展示导入副本核验38912阵列、B35末单元、96匹配干预及GPU/本机工具403。桌面1.0.10的51页面/源码、6节点资源与ASAR核对一致，安装器NotSigned，未跑安装向导。最初桌面打包缺npm，临时PATH调用现有本地工具恢复；沙盒回环/临时目录导致测试阻塞，停止自己进程后用工作区TEMP与本机权限重跑；Windows GBK验收脚本打印问题修为UTF-8，产品/算法未因此改变。最终展示包69成员逐SHA导入成功，再导入already_installed；86个元数据对象经原有研究协议包校验。

**第一性原理与下一大阶段。** 当前共同底座新增的是可重复查询的原生对象、完整关系和明确失败，不是39182个固定概念标签。局部高相关不保证状态充分，缺失残差/跨位置选择会使递推走出训练支持，G/V乘积可放大误差。避免无限小测试循环的方式是固定原生对象、材料家族和方法版本，把反例变成下一冻结设计的约束：先冻结新长文本及语义最小对，事前登记效应/功效与能力门；同题同预算比较原生组、行为归因、transcoder；独立预测未见激活/全词表分布及干预后果；通过的MLP读写组再放入足够完整的残差/注意力边界回溯Head与embedding。保留多义、条件化与多路径，不能由此直接推出智能本体、唯一齿轮或AGI。Goodfire长语境/符号与原生参数研究提供启发，本轮没有证明其“天然干净”工作假设。

**保留、清理与交付。** 核对路径后保留原始G/V/A、全部矩阵、参数、文档、预测残差、冻结、源码、截图及日志，删除本轮已验证的两组导入重复张量和由r2替代的未发布展示包，按文件大小共{released:,}字节（{released/2**30:.2f}GiB）。自动审批曾把不同名的验收复制目录误认为含截图而拒绝整目录清理；随后先证明截图位于mlp_tail_acceptance、复制数据位于mlp_tail_pack_acceptance，再只删除核对SHA的.npy文件和已确认的单一旧包，不删截图目录，不绕过审批。交付为release/MLP-TAIL-20261108-DELIVERY.md及4个最终包，正式上传/部署待用户手动进行。详细算法与通俗说明见research/mlp_tail/v1/README.md，实际最终验证见tests/glm5/result/mlp_tail_acceptance/final-acceptance.json。
'''
    # Avoid a typo in the narrative count while preserving all prior bytes.
    text=text.replace('39182个固定概念','38912个固定概念')
    with memo.open('ab') as f:f.write(text.encode('utf-8'))
    assert memo.read_bytes()[:len(before)]==before
    claude=C.ROOT/'research/claude/docs/AGI_CLAUDE_MEMO.md';old=claude.read_bytes();ids=[int(x) for x in re.findall(rb'^## Phase (\d+):',old,re.M)];cp=max(ids)+1
    cross=f'''\n\n## Phase {cp}: 尾四层MLP全量观察、失败递推与3D交付 [{stamp}]\n\n本地Qwen3-4B BF16完成B32—B35全部38912单元的G/V/A档案、长语境两极、完整参数、15全矩阵和12源写出方向的9216干预/对照观察，接入原菜单的3D成果空间。原生复算全位置逐位一致；稀疏递推留出FVU=2.46063、G/V组合约3.43e21，没有闭合。功效预注册和原始五字段溯源缺项在事后审计中明确保留，本轮只作探索观察，语义确认0，不追认正式机制。中心1.0.31/客户端1.0.10及r2展示包和86对象协议包已验证，本机浏览器验收完成，正式中心未修改。324+13+52+11项检查通过，清理重复副本{released/2**30:.2f}GiB，保留证据与截图。详见GLM5 Phase {phase}、research/mlp_tail/v1/README.md及release/MLP-TAIL-20261108-DELIVERY.md。\n'''
    with claude.open('ab') as f:f.write(cross.encode('utf-8'))
    assert claude.read_bytes()[:len(old)]==old
    C.save_json(qa/'memo-append-proof.json',{'glm5_phase':phase,'claude_phase':cp,'old_glm5_bytes':len(before),'old_claude_bytes':len(old),'prefix_unchanged':True,'time':now.isoformat()})
    print({'glm5_phase':phase,'claude_phase':cp,'cleanup_GiB':released/2**30},flush=True)


if __name__=='__main__':main()
