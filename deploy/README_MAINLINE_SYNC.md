# 主线研究状态与客户端同步

当前菜单和配色保持原样。`GET /api/research/mainline` 返回经过 SHA256 校验的统一记录；进度、开封清单、方法结果、工作包与 3D 索引从该版本回查。客户端每 30 秒、重新获得焦点或点击“同步研究记录”时刷新，断网显示来源明确的缓存或随包记录。

## 本次随包记录

- MA-H1 confirm 已在 H3 冻结后一次性开封，时间以 custody 原记录为准。主线 lockbox 和 lockbox_frame 未开封。历史冻结门和当前保管状态分别显示。
- H3 的 0.9885 是留出干预保真，594/1188 组件；不是图谱独立预测、单组件机制或整体破解率。
- H3b 在已测试排序的前 0.5%–20% 头预算下未达到 0.6 充分性门。保留负结果、所有排序/对照曲线及原始数值；不外推到全部组合。
- 3D 定位 H3 的 559 个 Head 和 35 个 MLP 整块，以及 H3b 排序集合中的 70 个 Head。整块不是单神经元；组成员不等于单头因果结论；不新增未经验证的计算边。

## 升级与核验

沿用 `README_DEPLOY.md` 的手动升级步骤，保留原有 `AI2050_DIST_DIR`，不要替换账号或已有数据库。新版本第一次读取主线记录时会将随包的真实报告、工作包和原生索引幂等归档，不需要手动导入研究库。包不包含未开封的样本文本。

升级后浏览器打开当前研发，再打开 H1 看当前开封清单；展开 H3b 工作包并点击“3D 中查看结果与组成员”。应同时看到中心记录版本、当前判据未支持、真实剂量曲线与候选卡的范围限制。

只读核验接口：

```sh
curl -fsS https://server.ai2050.com/api/health
curl -fsS https://server.ai2050.com/api/research/mainline
curl -fsS https://server.ai2050.com/api/research-library/workpackages
curl -fsS https://server.ai2050.com/api/traces/research-space
```

后三个接口的 `mainline_revision`（主线接口为 `revision`）应一致。按哈希从 `/api/research-library/artifacts/{sha256}` 读取证据，客户端还会校验主线 `record` 字符串的 UTF-8 SHA256。3D 索引版本不一致时不显示旧图冒充当前图。

## 后续研究记录发布

1. 先通过既有客户端研究包协议上传完整原始产物；上传与科学证据晋级分别处理。
2. 审核实际报告、冻结版本、开封事件和反例，显式生成新主线记录。当前构建器 `scripts/build_mainline_release.py` 读取 H3/H3b 的本地结构化结果与排序摘要，不运行模型、不读取封存正文；后续 Phase 应扩展该构建器的明确映射，不能让网页猜测结论。
3. 使用现有管理员认证向 `POST /api/research/mainline` 提交 `mainline_release.v1`。记录包含 state 与按 SHA256 索引的 artifacts；必须递增 sequence，parent_revision 匹配当前中心。凭据仅走既有会话或环境变量，不写进研究包。
4. 中心校验身份、证明引用、开封记录与版本前驱，按依赖顺序归档。未完成的发布不进入当前图谱；发布可重试、同内容可去重。可见性与状态指针在同一数据库事务提交。客户端自动读取新版本。

新结果不会自动得到“已独立复算”或“机制成立”。已经开封的材料不能恢复为封存；改方法后必须使用新确认材料。发布接口不能替代原协议的实验审计和正式证据准入。
