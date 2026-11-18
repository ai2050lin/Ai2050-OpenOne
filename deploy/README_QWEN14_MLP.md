# Qwen3-14B B28–B39 全量档案与客户端

当前客户端版本 1.0.14，中心更新包 1.0.36。新研究运行 `QWEN14-GOODFIRE-B28-B39-20261113-v2`，模型快照 `1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce`。覆盖零基 B28–B39，每层 U0–U17407，共208,896个原生MLP单元。模型共40块；B28包含在本次范围内。

## 在客户端查看

1. 打开「空间 → 关联机制 → 成果库」。
2. 选择「Qwen3-14B · B28–B39全部208,896个单元」，点开报告，再展开「完整报告与原始证据」。
3. 用层和单元编号查看「响应与语境」「完整参数」「上下游关联」「验证与反例」。每个gate行、up行、down列均有全部5120个参数；相邻层查询保留全部17408个数值。
4. 在「定位卡与关联」检索 `B39:U17407`，选中后点「定位到当前空间」。主空间与对象详情显示同一快照和原生单元，刷新保留位置。

12层汇总、双负分支频率/绝对激活份额/平方幅值份额、SiLU负导数区间、原预测与次级标准化预测、冻结父单元、24个已测单元的有限干预、逐样本失败对照均可回查。8个新干预单元与16个历史单元保留各自原始回放基线。

旧运行 `QWEN14-MLP-B28-B35-20261109-v1` 继续独立显示原8层139,264个单元；不会用新实验替换其证据。API支持 `?run=<完整运行ID>`；省略时在已安装的完整版本中优先选择新版。

208,896是观察档案覆盖量。正式语义确认数仍为0，不能称为全部语言机制已经破解。本轮客户端接入没有新增模型实验。相关、参数对齐、响应预测与有限范围干预分别标记。

## 本机产物

以下路径均相对于项目根目录：

- Windows安装包：`tests/glm5/result/qwen14_client_v2/desktop/AI2050-Setup-1.0.14-x64.exe`。
- 同一安装内容的可运行目录：`tests/glm5/result/qwen14_client_v2/desktop/win-unpacked/`。
- 中心更新：`release/ai2050-center-1.0.36.tar.gz` 与 `.sha256`。
- 新研究登记包：`tests/glm5/result/qwen14_client_v2/ai2050-qwen14-b28-b39-research-upload.tar.gz`。
- 原生增量包：`tests/glm5/result/qwen14_client_v2/ai2050-qwen14-b28-b39-delta.tar.gz` 与 `.sha256`。
- 原生数据组合清单：`tests/glm5/result/qwen14_client_v2/native/manifest.json`。
- 校验、截图与最终交付状态：`tests/glm5/result/qwen14_client_v2/`。

本机完整数据目录以硬链接共享既有D盘原场，不复制整套数组。所有链接对象视为不可变；修改分析时写新文件。保留旧、新研究原始目录及其清单。登记包只有报告和单元卡，不能独立提供全部原生参数。

## 正式中心手动更新

沿用用户手动升级中心的安排；本轮未自动发布到公网。中心不需要加载14B模型或运行GPU。

1. 上传中心更新包、对应SHA256及 `release/upgrade_center.sh`，按既有流程执行 `sudo bash upgrade_center.sh`。保留账号和正式 `AI2050_DIST_DIR`。
2. 若旧B28–B35原生包尚未安装，先上传上一轮的 `tests/glm5/result/qwen14_mlp_atlas_v1/ai2050-qwen14-native.tar.gz` 及校验文件，使用新版导入器导入。若已安装可跳过：

```bash
sha256sum -c ai2050-qwen14-native.tar.gz.sha256
python deploy/qwen14_mlp_pack.py import --archive /上传目录/ai2050-qwen14-native.tar.gz --data-dir /实际中心数据目录
```

3. 上传新增量包及校验文件。增量包要求上一步的旧包，复用文件在导入时逐个核对大小和SHA256；同卷硬链接复用，新增成员逐个校验后原子激活。缺旧包、损坏、重复或非法路径均拒绝，失败的暂存目录清理后可重试。

```bash
sha256sum -c ai2050-qwen14-b28-b39-delta.tar.gz.sha256
python deploy/qwen14_mlp_pack.py import --archive /上传目录/ai2050-qwen14-b28-b39-delta.tar.gz --data-dir /实际中心数据目录
```

若旧包已安装，本次还需容纳约6.3 GiB增量压缩包与6.1 GiB新增解包数据，并满足系统保留空间要求。旧包复用数据不重复复制。重复导入同版本会重新校验并返回 `already_installed`；不同内容不得覆盖相同运行。

4. 在中心客户端「研发 → 上传研究包」上传新研究登记包，沿用已有认证、整包校验和去重协议。原生增量大包使用上面的独立导入器，不把它称为客户端协议断点上传。
5. 如果服务环境曾用 `AI2050_QWEN14_MLP_DIR` 固定到旧目录，改指向新目录或移除该固定值，使用 `AI2050_DIST_DIR/native_atlases/<RUN>`。重启中心。
6. 桌面用户更新1.0.14安装包并连接该中心；网页用户刷新。核对：

```bash
curl -fsS https://server.ai2050.com/api/health
curl -fsS 'https://server.ai2050.com/api/research/qwen14-mlp?run=QWEN14-GOODFIRE-B28-B39-20261113-v2'
curl -fsS 'https://server.ai2050.com/api/research/qwen14-mlp/units/39/17407?run=QWEN14-GOODFIRE-B28-B39-20261113-v2'
```

无匹配数据时显示缺包原因；快照或研究版本不匹配时拒绝显示。断网时仅显示已有匹配缓存或明确失败，不用4B或演示数据填充。安装向导未在用户系统上执行；已验收打包后的真实Electron程序与其内置文件。

## 证据与验证边界

材料仍是1024篇历史合法来源长文本，发现512篇、确认和留出各256篇，每篇8个固定读出；另扫描后半段正负极值。`A=SiLU(G)×V` 与单列写出 `A×d` 是原生计算定义。双负份额按单元的 `sum(|A|·I[G<0,V<0])/sum(|A|)` 计算，不是语义贡献比例。

原始未标准化预测失败、次级预测的弱哈希基线、单父递推失败、已暴露语料、旧provenance限制和有限干预范围继续保留。新版本登记与客户端验收不会升级研究结论。

完整协议回归有一项既有工作区失败：`SOURCE-BINDING-20261110.source` 未映射到语言项。本轮没有改动这份文法或其冻结身份。客户端、原生包导入、索引最新版本语义及真实数据读回的结果分别见交付记录。
