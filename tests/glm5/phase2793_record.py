"""Append dated scientific audit and preserve the exact analysis implementation."""
from pathlib import Path
import json
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.insert(0,str(Path(__file__).parent))
import phase2793_materials as P
from shared.dist_protocol.mechanism_trace import file_sha
from deploy.research_library import Library

def main():
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    freeze=read(P.OUT/'execution-freeze.json')
    for name,expected in freeze['dependencies'].items():assert file_sha(P.ROOT/name)==expected,'frozen collection code was modified'
    names=['tests/glm5/phase2793_materials.py','tests/glm5/phase2793_run.py','tests/glm5/phase2793_analyze.py',
        'tests/glm5/test_phase2793_trace.py','tests/glm5/phase2793_client.test.jsx','shared/dist_protocol/mechanism_trace.py',
        'server/mechanism_trace_service.py','server/live_analysis_service.py','frontend/src/components/v8/MechanismTraceExplorer.jsx',
        'frontend/src/components/v8/MechanismScene.jsx','frontend/src/components/v8/EntityArbitrationReport.jsx']
    bundle={'schema':'research_source_bundle.v1','kind':'real','run_id':P.RUN,'model_snapshot':freeze['model_snapshot'],
        'source_files':[{'path':n,'sha256':file_sha(P.ROOT/n),'text':(P.ROOT/n).read_text(encoding='utf-8')} for n in names],
        'scope':'collection dependencies equal the frozen hashes; analysis and UI are archived after the run, not prospectively frozen'}
    path=P.PUBLIC/'implementation-sources.research.json';P.save(path,bundle,True)
    ref=Library(P.ROOT/'distributed_data/research_library').ingest(bundle,str(path.relative_to(P.ROOT)),kind='local_report')['sha256']
    m=P.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md';text=m.read_text(encoding='utf-8');marker='P2793-METHODS-AND-NEGATIVE-CLOSURE-v1'
    if marker in text:print('Already appended');return
    n=int(re.findall(r'^## Phase (\d+):',text,re.M)[-1])+1;stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    entry=f'''

## Phase {n}: 三模块机制提取算法审查与实体属性长文本真实研究归档 [{stamp}]

记录标识：{marker}。回应用户“当前破解词嵌入、注意力、MLP的机制和算法是什么”，并接续P2793实体属性研究。公式必须区分已知模型前向、实验提取器和已验证的语义规律；不能把源代码公式重新命名为项目破解成果。

### 1. 当前算法及证据边界

词嵌入：h[0,t]=E[token_id(t)]是已知查表计算。词/概念可跨多个token；上下文语义不是同一token行随语境改变。算法保留完整E行和各层原生状态，以类别中心、层级方差分解、岭回归属性算子、RSA与跨语言/实体留出研究关系；以hs[0]、词形、身份、频率、位置为基线。EA类结果主要来自上下文化编码，不能归作纯embedding结论。线性可读出只是“信息存在”的证据，不证明模型使用相同读出器，不能推出一个坐标对应一个概念。

注意力：对实际Qwen适配其输入归一化、Q/K头归一化与RoPE。a[h,t,s]=softmax_s(q[h,t]^T k[group(h),s]/sqrt(d_head)+causal_mask)，z[h,t]=sum_s a[h,t,s]v[group(h),s]，写出sum_h W_O[h]z[h,t]。QK研究路由，V/O研究传递内容；不能只看注意力权重。算法重算完整来源行并与实际SDPA输出核对，沿OV与下游读入矩阵组合分数找候选，在发现集选头、未见实体/句式检验，配同块随机头、消融或读写方向移除。归一化与位置使整个头不是固定线性读取器。既有代码查询方向三架构效应属于局部路由证据，GLM key端和部分Qwen单调判据的失败保留。

MLP（当前Qwen3 SwiGLU）：对实际MLP输入x，G=W_gate x，V=W_up x，A=SiLU(G)⊙V，输出y=W_down A。因此单元j的样本写出为A_j W_down[:,j]，坐标k的输出为sum_j W_down[k,j] A_j。实际原生BF16与FP32重算存在舍入差，不能强求完全相等。读入方向与写出方向分别记录，正/负和双负响应均保留。算法包括原生单元普查、受控选择性与交互、gate/up读取方向、down写出、参数路径以及激活干预/方向移除；SAE/transcoder提供互补字典。历史L30 transcoder FVU约.405优于神经元.523，模型内损失恢复约.608却低于.758，不能宣布字典胜出或字典无用。既有冲突单元的编辑效应不等于一般代码能力，更不是全部MLP语义已理解。

跨模块：追踪实际残差接续，局部Head→MLP gate贡献在实际RMS固定时为(W_gate[j]⊙gamma_M)^T W_O[h] z_h /rms。该局部归因包含明确条件，不能当成通过所有后续层的总因果作用。统计关联、留出复现、真实参数连接、样本贡献、范围内干预分别登记。容量×门控仍是待检验分工假设，不是普遍定律；“机制只在开关一侧”没有成立。

### 2. 本地Qwen3-4B真实运行与结果

P2793-ENTITY-ARBITRATION-v1，单台RTX5080、CUDA BF16/SDPA，48实体；16发现、32未见实体；发现F0/F1、留出F2/F3；512/1024/2048有效token；文档/现实同事实成对输入，共576条。运行前冻结材料、H1/H2、功效与采集代码哈希；执行依赖在本次归档时再次逐一核对未改变。H1按实体成对聚合，不把同实体6个长文本当独立实体。预设功效H1=.873、H2=.813，只对事前效应假设成立，不能把实际未支持改成“趋势支持”。

运行425.53秒，峰值CUDA分配10,162,434,048 bytes，权重前后完整快照一致。37层HiddenState，全部36块HeadZ及MLP G/V/A，在三个声明读出保存完整原生BF16；固定8条深采额外保存全输入HiddenState和postnorm/postRoPE Q/K及分组V。8条实际greedy生成均为简短颜色回答，共8个缓存续算原场，不冒称长推理生成。21,616个文件，共17,391,760,064 bytes（约16.20GiB），全部sha256复查；最大注意力重构相对L2=.002014，预设容差.05。加入采集器前后logits逐位一致。

留出文档回答192/192=100%；现实61/192=31.77%，行为门要求各≥80%，未通过。文档偏好成对差=.416667，32实体bootstrap95%区间[.286458,.552083]；预设要求均值≥.50且下界>.20，未通过。源外常识探针30/48=62.5%，说明词典常见颜色、模型原有知识和范围选择混杂；不能把所有现实回答失败归为仲裁失败。主结果无事后删行，后续需新版本校准模型已知事实，不能在旧留出上筛实体后重新声称确认成功。

发现集DLA选8个后段Head：B32H11、B35H23、B34H28、B32H8、B26H20、B35H22、B30H5、B34H0。32未见实体F2/L1024文档条件，组消融正确率100%→96.875%，下降3.125个百分点，实体区间[0,9.375]个百分点；同块8随机头100%→100%。H2预设下降≥20个百分点、区间下界>0且超过随机对照，未支持。不能事后换头组/阈值或宣布所有头不必要。按原预注册，仅支持H2后才触发该新Head干预的2200题能力保持检验；本轮未触发，明确not_tested，不复用旧W6不同编辑的非劣效结果。留存219项样本/干预错误及主判失败记录，不是219个独立机制反例。

### 3. 统一数据与客户端接入范围

原始文本按12个不可变CAS分块保存；归档完整预注册、冻结哈希、源外探针、逐样本观察、干预、运行环境、原场索引、报告与14张原生定位卡。定位卡不算14项语义机制，新增语义机制声明数=0。21份实体图谱、权重路径、transcoder与assembly历史报告统一来源/完整文件哈希索引，按原模型与精度分桶；缺原场的历史记录明确native_trace_available=false，不补模拟数据或跨模型坐标等同。

报告CAS 75d08d29815c129a69f1baa31e6941ef0d8add930909b814dec24fd519f51cae；图谱CAS 482c6205d5ca893d5451d42f05eb21f2c17b6883659fda8e8e54876d498ba9b7；历史索引CAS 6b099e611cac5bf2f635811fcea4961ea8862989059798c69fdf4e74e602066e；实现源码CAS {ref}。

客户端接入原生轨迹、Embedding全维、全部32Q/8KV头、9728 MLP单元阵列、实际query注意力行、生成步、完整输入、参数切片和局部贡献接口，保留原UI配色。实时长文本与研究共用Recorder/TraceStore；预算含聊天模板，超限显式停止，不静默截断。GPU模型驻留持有跨进程租约，卸载与前向互斥；这只约束接入租约的进程，不宣称所有历史node runner已统一资源互斥。

本机HTTP实际回查512-token深采样本成功，返回全部9728个MLP值、512个来源权重、32项Head贡献；完整权重快照匹配。B32/H11/U0/写出坐标0原生Gate=4.15625，FP32全项求和4.170803；原生MLP写出25.25，9728项求和25.283019；原生Attention写出6.125，32头求和6.116094。舍入误差与局部RMS近似公开显示，未修饰成精确机制定律。

### 4. 验证、限制与下一大任务

7项采集器测试通过：logits一致、全部单元、GQA、真实query非末位别名、完整嵌入、篡改/越界拒绝、GPU互斥与缓存生成绝对位置。分布式协议261项通过（132.27秒）；前端34项测试通过；Vite正式构建通过，仍有既有大包警告。最初沙盒临时目录/回环权限失败后，以项目临时目录和允许的本机测试执行重跑通过。浏览器新建验收标签在后续调用失效，重置后仍无法取得；本次不宣称已完成浏览器视觉验收，也不把编译与HTTP读取等同视觉验收。未打包或安装新版本。

主要瓶颈：现实探针原有知识与标签歧义、受控重复背景非自然语料、范围指令敏感、候选答案指标有限、直接归因和总效应分离、冗余、单物理节点、尚无可广泛独立预测的计算图谱。先完成大任务A：独立材料校准颜色/多属性事实与双范围行为门，保留旧失败；B：同批原场比较原生坐标/单元、字典与条件交互，按因果可见读出建立主干；C：分离QK路由、V/O内容、残差与MLP读写路径，随机对照、剂量和新操作能力门完整预注册；D：只用预测时可获得的输入检验未见实体/句式/组合的特征与下一词预测，记录未解释残差，独立第二节点复算后扩模型。

第一性原理：有限参数可以在不同输入、位置和自回归历史下重复调用同一计算，支持大量组合；这解释组合容量的可能性，不等于已经解释语言规则为何学到。当前真正要提取的是“给定条件下，哪些信息经哪些参数运算影响后续输出”的可复用规律。公式、热图、卡片和连线都只是工具；最终证据是对未见组合的新增预测能力及可重复的计算/干预检验。

通俗说明：已能量出每根线此刻传了什么，并找到若干有条件的读、选、写部件；还没有整台语言机器的可执行说明书。新一轮更可靠地保存了事实和失败，避免把漂亮3D或选出的8个头误当成破解完成。
'''
    with m.open('a',encoding='utf-8') as f:f.write(entry)
    print({'phase':n,'timestamp':stamp,'source_bundle_sha256':ref})

if __name__=='__main__':main()
