# AGI_GPT5_DNN_3D_VISUALIZATION

## 1. 目标

这份方案只服务一个目标：

**把重建后的 DNN 三条分析链，做成同一套可联动的三维可视化系统。**

这里的三条链是：

1. `A：语言投影链`
2. `B：路由尺度链`
3. `C：前后向计算链`

对应的数据底座分别来自：

- [stage104_tensor_level_language_projection_rebuild.py](/d:/develop/TransformerLens-main/tests/codex/stage104_tensor_level_language_projection_rebuild.py)
- [stage105_tensor_level_route_scale_rebuild.py](/d:/develop/TransformerLens-main/tests/codex/stage105_tensor_level_route_scale_rebuild.py)
- [stage106_forward_backward_trace_rebuild.py](/d:/develop/TransformerLens-main/tests/codex/stage106_forward_backward_trace_rebuild.py)

---

## 2. 总体方案

## 2.1 一套场景，三个视图

最推荐的方案不是做三个互不相干的三维图，而是做：

**同一套数据驱动下的三个三维视图，彼此联动。**

建议命名为：

- `View-A：语言投影体`
- `View-B：路由尺度体`
- `View-C：前后向闭环体`

用户在任一视图里选择一个维度、一个层段、一个异常点，另外两个视图同步高亮对应结构。

---

## 2.2 技术实现建议

如果优先要快：

- `Plotly（交互绘图库）`
  - 适合先做研究版原型
  - 优点是开发快、联动容易

如果优先要强交互和长期扩展：

- `Three.js（三维前端库）`
  - 适合后续做更复杂的空间联动和动画

如果优先要偏科研桌面展示：

- `PyVista（三维科学可视化库）`
  - 适合体渲染和网格视图

当前项目最合理的落地顺序是：

1. 先用 `Plotly（交互绘图库）` 做验证版
2. 再根据研究反馈升级到 `Three.js（三维前端库）`

---

## 3. View-A：语言投影体

## 3.1 坐标定义

- `X 轴`
  - 语言维度
  - `style / logic / syntax`

- `Y 轴`
  - 投影变量
  - `q_reconstructed / b_reconstructed / g_reconstructed`

- `Z 轴`
  - 强度值
  - 直接读 [stage104_tensor_level_language_projection_rebuild.py](/d:/develop/TransformerLens-main/tests/codex/stage104_tensor_level_language_projection_rebuild.py) 的重建值

## 3.2 颜色与透明度

- 颜色表示 `cross_dimension_separation（跨维分离度）`
- 透明度表示 `run_stability（跨运行稳定度）`

## 3.3 视觉效果

建议做成：

- 三根主柱体
  - 分别对应 `style / logic / syntax`
- 每根柱体内部三层
  - 分别对应 `q / b / g`

这样一眼能看出：

- 哪个语言维度更强
- 哪个变量是主导项
- 哪个维度更不稳定

---

## 4. View-B：路由尺度体

## 4.1 坐标定义

- `X 轴`
  - 尺度层级
  - `local_anchor / mesoscopic_bundle / distributed_network`

- `Y 轴`
  - 路由相关量
  - `support / coupling / tolerance`

- `Z 轴`
  - 指标值
  - 来自 [stage105_tensor_level_route_scale_rebuild.py](/d:/develop/TransformerLens-main/tests/codex/stage105_tensor_level_route_scale_rebuild.py)

## 4.2 颜色与连线

- 颜色表示主导尺度
- 连线粗细表示 `route_structure_coupling_strength（路由结构耦合强度）`
- 底面热度表示 `degradation_tolerance（退化容忍度）`

## 4.3 视觉效果

建议做成：

- 三座不同高度的平台
- 平台之间有耦合桥

这样能直接表达：

- 为什么当前主导尺度仍是 `distributed_network（分布式网络）`
- 它的领先幅度其实不大
- 哪一层最容易在退化压力下掉下去

---

## 5. View-C：前后向闭环体

## 5.1 坐标定义

- `X 轴`
  - 注入步骤
  - `step 1 -> step 6`

- `Y 轴`
  - 通道类型
  - `frontier / boundary / atlas / loss`

- `Z 轴`
  - 实际强度
  - 来自 [stage106_forward_backward_trace_rebuild.py](/d:/develop/TransformerLens-main/tests/codex/stage106_forward_backward_trace_rebuild.py)

## 5.2 颜色与动画

- 颜色表示下降速度
- 动画表示步骤推进
- 顶部叠加一条闭环曲面，表示：
  - `raw_forward_selectivity`
  - `raw_backward_fidelity`
  - `raw_novelty_binding_capacity`

## 5.3 视觉效果

建议做成：

- 一条沿时间延展的三维轨迹带
- 每个步骤有 4 个立柱
- 上方叠加一个闭环张力面

这样能直接看出：

- `loss（损失）` 如何下降
- `frontier（前沿）` 与 `boundary（边界）` 如何联动
- 前向和反向闭环到底在哪一步最弱

---

## 6. 三视图联动

最关键的不是单图漂亮，而是联动逻辑：

1. 点击 `style / logic / syntax`
   - 同时高亮路由尺度体里的对应支撑变化
   - 再高亮前后向闭环体里最相关的步骤段

2. 点击 `distributed_network`
   - 同时在语言投影体里显示它对应的投影变量强度
   - 在前后向闭环体里显示它对应的前向选路压力

3. 点击某一步异常轨迹
   - 回看它对应语言投影体里的薄弱维度
   - 再回看路由尺度体里是哪一层最脆

这样三维可视化就不只是“展示分数”，而是：

**展示三条链是怎么在同一个系统里彼此约束的。**

---

## 7. 数据接口建议

建议统一输出一个可视化载荷 `json（结构化数据文件）`，结构如下：

```json
{
  "language_projection_view": {},
  "route_scale_view": {},
  "forward_backward_view": {},
  "cross_links": []
}
```

其中：

- `language_projection_view`
  - 直接来自 `stage104`

- `route_scale_view`
  - 直接来自 `stage105`

- `forward_backward_view`
  - 直接来自 `stage106`

- `cross_links`
  - 存放视图间的联动映射

---

## 8. 当前最值得先做的版本

最建议的第一版不是全功能三维系统，而是：

1. 一个页面三栏布局
2. 左边 `View-A`
3. 中间 `View-B`
4. 右边 `View-C`
5. 底部放一个共享时间轴和共享维度选择器

第一版只做三件事：

1. 悬停显示精确值
2. 点击高亮联动
3. 时间步骤播放

这已经足够支撑当前研究。

---

## 9. 这套方案能解决什么问题

它最直接能解决三个研究痛点：

1. 把“语言投影、路由尺度、前后向闭环”从三张表，变成一个空间系统。
2. 把薄弱点从抽象分数，变成可见的结构裂缝。
3. 把后续的反例和脑编码弱链接入同一空间视图。

---

## 10. 当前硬伤

这套三维方案也有边界：

1. 它展示的是重建链，不是真实第一性原理定理。
2. 当前 `View-C` 仍只基于单条梯度轨迹，广度不够。
3. 还没有把 `brain_grounding（脑编码落地）` 的弱链直接放进三维主视图。

所以最合理的下一步不是只做前端，而是：

1. 先把 `stage104/105/106` 输出统一成一份可视化载荷
2. 再加入 `stage92/94/96` 的脑编码弱链和证据独立性层
3. 最后再做完整的“四视图联动系统”

到那时，三维可视化才会真正成为理论审计工具，而不是展示工具。
