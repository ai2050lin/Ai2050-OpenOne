import os, json, hashlib, io

WS = r"D:\AI2050\Ai2050-OpenOne"
MANIFESTS = os.path.join(WS, "ai2050_research_os", "manifests")
OUT = os.path.join(WS, "ai2050_research_os", "registry", "artifact_residency.json")

def resolve(p):
    return os.path.join(WS, p.replace("\\", "/"))

rows = []
for name in sorted(os.listdir(MANIFESTS)):
    if not name.endswith(".manifest.json"):
        continue
    m = json.load(io.open(os.path.join(MANIFESTS, name), encoding="utf-8"))
    cid = m.get("contract_id")
    files = m.get("files", [])
    missing = [e.get("path") for e in files if not os.path.isfile(resolve(e.get("path", "")))]
    present = [e.get("path") for e in files if os.path.isfile(resolve(e.get("path", "")))]
    rows.append({"contract_id": cid, "manifest": name, "total": len(files), "missing": len(missing), "present": len(present)})

affected = [r for r in rows if r["missing"] > 0]
contract_ids = sorted(r["contract_id"] for r in affected)
total_missing = sum(r["missing"] for r in affected)

record = {
    "id": "RES-GLM5-RESULT-001",
    "date": "2026-10-03",
    "status": "active",
    "scope": "manifest_run_bundle_artifacts",
    "contract_ids": contract_ids,
    "disposition": "removed_from_workspace_cleanup_unverified",
    "reason": "tests/glm5/result/phase1246-1263 run bundle 目录已不在工作区。全仓目录与文件检索（tests/research/shared/scripts/server 根）仅命中 tests/glm5_temp/phase1246_invalid_preaudit_coverage_20260812 临时目录与 phase1246-1263 的 *.py 脚本；git ls-files 显示 tests/glm5/result/ 仅 4 个 lease 文件被跟踪，产物本体从未入库，无法恢复。",
    "recompute_note": "重算入口仍在：tests/glm5/phase1246-1263*.py 与 *_audit.py。冻结材料（material/protocol/）同样缺失，完整重算需重建冻结材料并重跑模型前向，成本高，当前未安排。",
    "validator_policy": "verify_manifest_file 对命中 contract_ids 的缺失文件条目降级为『已登记离场』警告；对仍然存在的文件，size_bytes 与 sha256 校验照常执行；独立命令 verify-manifest 不受豁免，保持严格。",
    "audit_trail": {"manifests_scanned": len(rows), "contracts_affected": len(affected), "total_missing_files": total_missing, "detail": rows},
}

payload = {"schema_version": "artifact-residency.v1", "records": [record]}
io.open(OUT, "w", encoding="utf-8", newline="\n").write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
io.open(os.path.join(WS, ".workbuddy", "tmp_residency.txt"), "w", encoding="utf-8").write(
    "contracts_affected=%d total_missing=%d contract_ids=%s" % (len(affected), total_missing, ",".join(contract_ids)))
print("OK")
