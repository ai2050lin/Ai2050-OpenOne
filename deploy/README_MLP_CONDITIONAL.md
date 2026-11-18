# 条件响应卡与长文本验证：客户端更新

该更新保持一级、二级菜单和现有配色。入口为：空间 → LLM 编码图谱 → 全部研发成果 → `MLP-CONDITIONAL-PATHS-v2`。

## 数据与程序分开发布

- `ai2050-research-conditional-v2.tar.gz` 是研究包，使用已登录管理员的「研发 → 实验与队列 → 节点任务、待审与回执 → 上传研究包」登记。中心 1.0.29 已有对应协议。客户端先校验全部对象 SHA256，再逐对象上传；重试不增加样本。
- `ai2050-client-conditional-v2-web.tar.gz` 是网页客户端更新，包内仅有 `frontend/dist/` 和发布清单，不含数据库、账号、模型权重或原生大张量。沿用用户手动部署安排。
- 包装版本由发布清单记录；本次没有修改中心版本号。旧中心网页能读取通用研究对象，但新「3D条件图谱」和专用报告需要该客户端更新。

## 手动更新网页

以下假定服务工作目录仍为 `/opt/ai2050/app`，若配置了 `AI2050_FRONTEND_DIST`，应使用该实际目录。

```bash
systemctl show ai2050-center -p WorkingDirectory
cd /opt/ai2050/app
sudo tar -czf /opt/ai2050/frontend-before-conditional-v2.tar.gz frontend/dist
tar -tzf /path/to/ai2050-client-conditional-v2-web.tar.gz
sudo tar -xzf /path/to/ai2050-client-conditional-v2-web.tar.gz -C /opt/ai2050/app
sudo systemctl restart ai2050-center
curl -fsS https://server.ai2050.com/api/health
```

解包只覆盖该包携带的网页文件；旧静态资产可以保留，以支持尚未刷新的浏览器。打开工作台并强制刷新。中心程序或数据库不需要重新导入。

## 验收口径

「3D条件图谱」读取正式中心的 8 个内容寻址对象，显示 B28–B35 全部 77,824 个原生编号。旧自然语境统计与本轮 4 个受控单元分开；未测试关系显示未测。可切换条件指标、输入单元编号，查看两个固定单元对、完整参数、预测、3 个剂量、能力保持和反例。

本轮研究材料为作者编写的四类长文本角色任务，不能称为广泛自然语料确认。冻结代码与预注册文字的基线变量差异、稀有条件的样本量、单物理节点及现有能力题库范围均在报告中保留。客户端显示与上传成功不改变科学证据等级。

研究包中的全词表预测斜率按 little-endian float32 的 base64 无损保存；`dtype` 和 `shape` 随对象提供。原始 BF16 采集与日志在研究节点保留，服务器包携带输入、所选单元原值、全部干预记录、能力原始分数和复算哈希。
