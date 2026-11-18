# 稀有触发与共同输入研究：网页客户端更新

研究数据已经通过正式客户端上传，服务器无需重新导入。本包只更新网页，沿用用户手动部署安排。

## 使用入口

空间 → 关联机制 → 成果库，选择 `MLP-TRIGGER-UPSTREAM-v1` 或 `MLP-TRIGGER-READ100-v1`。报告包含发现与边界、3D与原生响应、干预与能力、反例与来源。定位卡可叠加到同一模型结构，残差坐标与MLP单元分别标记。历史八层条件卡折叠并明确标注来源。

正式中心现有旧网页已经能显示这两个研究的通用3D索引和原始报告；本更新增加专用报告、完整前缀曲线、剂量曲线和参数查询。服务器仍为1.0.29，不改服务器版本号。

## 手动部署

先确认中心工作目录和 `AI2050_FRONTEND_DIST` 是否有自定义配置。以下以 `/opt/ai2050/app` 为例：

```bash
systemctl show ai2050-center -p WorkingDirectory
cd /opt/ai2050/app
sudo tar -czf /opt/ai2050/frontend-before-trigger.tar.gz frontend/dist
tar -tzf /path/to/ai2050-client-mlp-trigger-web.tar.gz
sudo tar -xzf /path/to/ai2050-client-mlp-trigger-web.tar.gz -C /opt/ai2050/app
sudo systemctl restart ai2050-center
curl -fsS https://server.ai2050.com/api/health
```

仅覆盖包内 `frontend/dist/` 文件，不覆盖数据、数据库、账号、模型或原生归档。旧静态资源保留以支持尚未刷新的页签。部署后刷新网页。包内 `CLIENT_TRIGGER_MANIFEST.json` 记录每个文件哈希；构建使用同源API，不固定到开发机端口。

## 证据范围

主实验384个新家族×8条件，补充确认128个新家族×8条件，发现集96个家族复用；相同2200题能力电池不当成两个独立题库。主实验有一个行为句式未过门，补充确认全部通过，但均未获得语言语义机制晋级。数值响应、输出变化和任务答案必须分开解释。

研究包保留全部前缀所选单元G/V/A、R4曲线、选定位置完整2560维残差、输入、原生参数、预测承诺、干预与等范数对照、恢复、能力检查、反例和哈希。不是全部层全部token的完整模型张量或新增KV归档。独立物理节点复算仍未完成。
