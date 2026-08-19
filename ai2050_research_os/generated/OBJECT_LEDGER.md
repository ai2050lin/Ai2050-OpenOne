# 研究对象与构念账

> 自动生成于账本基线 `2026-08-11`。请修改 `registry/`，不要手工修改本文件。

## 对象

| 对象 | 状态 | 最高层级 | 构念 | 下一合同 | 证据 |
|---|---|---:|---|---|---|
| `OBJ-K199-ATOMIC-REGISTRY` K199 五个行为原子注册表 | 历史资格 | L1 | CON-CONTENT-SELECTION | — | K199, K200, K210 |
| `OBJ-C001-FRESH-TYPED-BINDING` 全新类型化对象—标记绑定对象 | 已预注册 | L0 | CON-CONTENT-SELECTION, CON-EXACT-FORMAT, CON-NATURAL-GENERATION, CON-STOP-CACHE | EXP-C001-WP01-001 | K210 |

## 构念

| 构念 | 状态 | 定义 | 明确不等价 |
|---|---|---|---|
| `CON-CONTENT-SELECTION` 内容选择 | 已预注册 | 在不依赖严格表面字符串的情况下，选择与目标对象和关系绑定的正确内容。 | 严格字符串正确, 格式服从, 首 token 正确, hidden 内容模块存在 |
| `CON-EXACT-FORMAT` 严格格式编译 | 已预注册 | 把已选内容编译为预注册的精确短字符串与格式。 | 内容正确, 自然语义等价, 候选排序正确 |
| `CON-NATURAL-GENERATION` 自然完整生成 | 已预注册 | 在无外部 trie 或候选限制时完整生成语义正确且自然的答案序列。 | 首 token 正确, 教师强制概率高, 候选集合内正确 |
| `CON-STOP-CACHE` 停止与缓存闭合 | 已预注册 | 正确内容生成后，增量 KV cache、后续词元竞争和停止条件共同闭合。 | 第一 token 正确, 外部截断, 候选 trie 到达叶节点 |
