import io, json

P = r"D:\AI2050\Ai2050-OpenOne\ai2050_research_os\registry\corrections.json"
data = json.load(io.open(P, encoding="utf-8"))
assert isinstance(data, list), "corrections.json 应为 LIST"

ids = {r.get("id") for r in data}
assert "COR-REG-001" not in ids, "COR-REG-001 已存在"

data.append({
    "id": "COR-REG-001",
    "date": "2026-10-03",
    "status": "active",
    "target_type": "artifact_residency",
    "target_ids": ["SRC-GPT-IMPORTED-GLM", "RES-GLM5-RESULT-001"],
    "problem": "researchctl validate 累计 187 项失败：SRC-GPT-IMPORTED-GLM 登记的工作区文件 research/gpt5/docs/AGI_GLM5_MEMO.md 已被删除（git 历史显示 66f441bdc 20260919 前后清理，blob 仍在对象库）；16 个 glm5 线 manifest（EXP-C001..C013）引用的 tests/glm5/result/phase1246-1263 run bundle 共 186 个文件条目已整体离场且从未入 git，无法恢复。",
    "correction": "① 按登记 commit 16b3119 从 git blob dab222e4 恢复 AGI_GLM5_MEMO.md 并做 LF→CRLF 还原，恢复字节 sha256=0c17fee6d99456b5db6c04dfa8b9ef5fd1e754f1dfc0e6a962617ea12dd268e0 与 captured_sha256 逐字节一致，size=422790；② 新增 registry/artifact_residency.json（RES-GLM5-RESULT-001）登记 16 合同 186 个缺失条目的离场事实与重算入口；③ researchctl verify_manifest_file 增加 residency 感知：命中登记的缺失文件降级为『已登记离场』警告，存在文件的 size/sha 校验保留，独立命令 verify-manifest 不受豁免；④ Registry 变更将改变 canonical source_sha256，需重建 Snapshot 并重新导出客户端。",
    "preserves_original_claim": True,
})

io.open(P, "w", encoding="utf-8", newline="\n").write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
print("APPENDED COR-REG-001, total=%d" % len(data))
