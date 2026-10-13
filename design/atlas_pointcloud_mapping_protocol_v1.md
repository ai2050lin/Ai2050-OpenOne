# FTR → 特征点云 映射协议 v1（P3 施工图）

> 2026-10-11。前置：design/atlas_domain_tree_scheme_v1.md（三域树）、design/client_mission_ia_review_v1.md（S1 对象宇宙）。
> 目标：打通「图谱特征（atlas registry FTR-xx）↔ 空间点云（LensSpatial·CloudMode）」，
> 使逆向工程的对象（特征）跨界面流动。红线：零实验字面量；无真实坐标时不伪造（DEMO 维持现状并显式标注）。

## 1. 现状（实测）

- 图谱侧：registry（v1.28，26 特征）含 `family`（KN.readout / GR.context 等）与 E 级；三域树派生档 `atlas_tree_v1.json` 已聚合族→特征。
- 空间侧：CloudMode 点云由 `LANG_TEMPLATES[tpl].demo.cloud` 驱动（族=簇 label，坐标=确定性伪随机），**无真实特征坐标**；焦点/邻居为模板 demo 叙事。
- 两套 ID：`F#3734`（object registry）/ `FTR-xx`（atlas registry）互不相通。

## 2. 映射协议 cloud_map.v1（派生层，不推翻 immutable 链）

在 `server/derive_atlas_tree.py` 扩展输出段 `cloud_map`（与 domains 同源派生，`--check` 幂等）：

```json
"cloud_map": {
  "version": 1,
  "space": {"source": "collect.npz", "layer": "L6", "n_points": 1024, "note": "坐标=研究侧特征点云；仅 family 级映射先行，token 级锚定属 Phase C/SAM-xx"},
  "clusters": [
    {"family": "KN.readout", "cluster_label": "知识读出", "ftr_ids": ["FTR-01", "..."], "e2plus": 11, "e1": 1}
  ],
  "unmapped_families": []
}
```

规则：
1. **family 级映射先行**（FTR.family ↔ 点云簇 label），token 级锚定等 Phase C（SAM-xx）后再议——与三域树「叶子=锚定证据样本」纪律一致。
2. `ftr_ids` / `e2plus` / `e1` 一律由派生器从 registry 计算，人手禁写；registry 未覆盖的簇进 `unmapped_families`（诚实呈现缺口，不做假映射）。
3. 簇 label 对齐表（LANG_TEMPLATES demo.cloud labels ↔ registry family 名）作为**静态配置**进派生器（非 JSX 字面量，守住零字面量红线）。

## 3. 客户端表现

- CloudMode 增加可选 prop `cloudMap`：有值时——簇图例旁标注成员 FTR 数与 E2+ 计数；选中 FTR（?atlasSel=）所在簇高亮（其余簇降透明度）；无值时维持现状并注记「family 映射未派生」。
- AtlasPanorama 特征节点 title 增补「点云簇：xx」（读 cloud_map），形成图谱→空间的可点击动线（P3 完成态：节点双击 → `?lens=spatial&atlasSel=FTR-xx`）。
- 顶栏特征源选择器（已接 registry）选择后同步 URL `?atlasSel=`，空间/图谱两端共用同一选中态。

## 4. 落地拆步

| 步 | 内容 | 量级 | 前置 |
|---|---|---|---|
| A | 派生器 cloud_map 段 + 簇 label 对齐表评审 | 零 GPU | 簇 label ↔ family 对齐表人工确认一次 |
| B | CloudMode 接 cloudMap（图例标注 + 簇高亮） | 小 | A |
| C | AtlasPanorama ↔ CloudMode 双向跳转（?atlasSel 共用） | 小 | B |
| D | token 级锚定（SAM-xx 入 npz 实例化） | 中大 | 研究侧 Phase C |
