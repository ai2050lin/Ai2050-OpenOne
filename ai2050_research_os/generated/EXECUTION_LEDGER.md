# 合同、运行与产物账

> 自动生成于账本基线 `2026-08-11`。请修改 `registry/`，不要手工修改本文件。

## 合同

| 合同 | 状态 | 工作包 | 对象 | SHA256 | Manifest | Run-ready |
|---|---|---|---|---|---|---|
| `EXP-C001-WP01-001` | 已预注册 | WP01 | OBJ-C001-FRESH-TYPED-BINDING | `836d72fc006125db` | manifests/EXP-C001-WP01-001.manifest.json | false |

## 运行

当前登记运行数：0。

## 产物

当前登记产物数：0。大型张量留在本地结果目录，账本只登记摘要与哈希。

## 勘误

- `COR-WP00-001`：初始迁移引用不存在的 ../upload 路径，导致 validate 失败 13 项。 → 证据改为引用 PHREC 复合记录；PHREC 再引用 Git blob 固定的真实 Memo 来源。
- `COR-WP00-002`：整数 Phase 在历史中存在重复，不能作为机器唯一主键。 → 新增 record_id；phase 和 phase_label 仅作为显示字段，允许相同标签出现多次。
