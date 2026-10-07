import io

P = r"D:\AI2050\Ai2050-OpenOne\ai2050_research_os\scripts\researchctl.py"
text = io.open(P, "r", encoding="utf-8", newline="").read()

def rep(old, new, label):
    global text
    for o, n in ((old, new), (old.replace("\n", "\r\n"), new.replace("\n", "\r\n"))):
        c = text.count(o)
        if c == 1:
            text = text.replace(o, n, 1)
            print("OK  ", label)
            return
        if c > 1:
            raise SystemExit("ANCHOR NOT UNIQUE: " + label)
    raise SystemExit("ANCHOR NOT FOUND: " + label)

rep(
    '    "corrections": "corrections.json",\n}',
    '    "corrections": "corrections.json",\n    "artifact_residency": "artifact_residency.json",\n}',
    "FILES + artifact_residency",
)

rep(
    'def verify_manifest_file(path: Path, expected_contract_id: str, errors: list[str]) -> None:',
    'def verify_manifest_file(\n    path: Path,\n    expected_contract_id: str,\n    errors: list[str],\n    residency_ids: frozenset[str] = frozenset(),\n    waived: list[str] | None = None,\n) -> None:',
    "verify signature",
)

rep(
    '''        if not artifact_path.is_file():
            errors.append(f"manifest {path.name} 文件不存在: {entry.get('path')}")
            continue''',
    '''        if not artifact_path.is_file():
            if expected_contract_id in residency_ids:
                if waived is not None:
                    waived.append(f"manifest {path.name} 文件已登记离场: {entry.get('path')}")
                continue
            errors.append(f"manifest {path.name} 文件不存在: {entry.get('path')}")
            continue''',
    "missing-file waiver branch",
)

rep(
    'def validate(data: dict[str, Any]) -> list[str]:',
    'def validate(data: dict[str, Any], waived: list[str] | None = None) -> list[str]:',
    "validate signature",
)

rep(
    '    schema_path = SCHEMAS / "experiment_contract.schema.json"',
    '''    residency_ids = frozenset(
        cid
        for record in data.get("artifact_residency", [])
        if record.get("status") == "active"
        for cid in record.get("contract_ids", [])
    )
    for res_index, record in enumerate(data.get("artifact_residency", [])):
        require_fields(record, ["id", "date", "status", "scope", "contract_ids", "disposition", "reason"], f"artifact_residency[{res_index}]", errors)
        if record.get("status") not in VALID["correction_status"]:
            errors.append(f"artifact_residency[{res_index}] 状态非法: {record.get('status')}")
        check_refs(record.get("contract_ids", []), set(contract_map), f"artifact_residency[{res_index}].contract_ids", errors)
    schema_path = SCHEMAS / "experiment_contract.schema.json"''',
    "residency validation block",
)

rep(
    '                verify_manifest_file(manifest_path, contract_id, errors)',
    '                verify_manifest_file(manifest_path, contract_id, errors, residency_ids, waived)',
    "verify call site",
)

rep(
    '''def command_validate(data: dict[str, Any]) -> int:
    errors = validate(data)
    if errors:''',
    '''def command_validate(data: dict[str, Any]) -> int:
    waived: list[str] = []
    errors = validate(data, waived)
    if errors:''',
    "command_validate waived init",
)

rep(
    '''    counts = {name: len(data[name]) for name in FILES if name != "project"}
    print("校验通过：" + ", ".join(f"{name}={count}" for name, count in counts.items()))''',
    '''    if waived:
        print(f"已登记离场工件（跳过存在性校验）：{len(waived)} 项")
    counts = {name: len(data[name]) for name in FILES if name != "project"}
    print("校验通过：" + ", ".join(f"{name}={count}" for name, count in counts.items()))''',
    "command_validate report",
)

V2_BLOCK = '''def build_snapshot_v2(data: dict[str, Any]) -> dict[str, Any]:
    snapshot = build_snapshot(data)
    snapshot["schema_version"] = "snapshot.v2"
    specs_doc = load_json(REGISTRY / "visualization_specs.json")
    snapshot["views"] = {
        "as_of": snapshot["as_of"],
        "items": [
            {
                "view_id": f"VIEW-{spec['id']}",
                "spec_id": spec["id"],
                "title": spec["id"],
                "data_ref": "registry/visualization_specs.json",
                "evidence_level": "has_data",
                "caveats": ["registry 规格投影；证据卡与 run_ref 待补"],
            }
            for spec in specs_doc.get("specs", [])
        ],
    }
    methods_doc = load_json(OS_ROOT / "drafts" / "industry" / "methods_map.json")
    tools_doc = load_json(OS_ROOT / "drafts" / "industry" / "tools.json")
    gaps_doc = load_json(OS_ROOT / "drafts" / "industry" / "gap_radar.json")
    snapshot["industry"] = {
        "as_of": str(methods_doc.get("_meta", {}).get("date", snapshot["as_of"])),
        "source_ref": "ai2050_research_os/drafts/industry/",
        "methods": [
            {
                "method_id": item["method_id"],
                "name": item["name"],
                "era": item["era"],
                "core_idea": item.get("core_idea", ""),
                "evidence_grade": item["evidence_grade"],
            }
            for item in methods_doc.get("methods", [])
        ],
        "tools": [
            {
                "tool_id": item["tool_id"],
                "name": item["name"],
                "org": item.get("org", ""),
                "role": item["role"],
                "relation_to_project": item["relation_to_project"],
            }
            for item in tools_doc.get("tools", [])
        ],
        "gaps": [
            {
                "gap_id": item["gap_id"],
                "title": item["title"],
                "nearest_industry_work": item["nearest_industry_work"],
                "missing": item["missing"],
                "atlas_axis": item["atlas_axis"],
                "priority": item["priority"],
            }
            for item in gaps_doc.get("gaps", [])
        ],
    }
    snapshot["cases"] = {"as_of": snapshot["as_of"], "items": []}
    return snapshot


def command_build_snapshot_v2(data: dict[str, Any]) -> int:
    snapshot = build_snapshot_v2(data)
    path = SNAPSHOTS / "draft" / "snapshot_v2.json"
    write_json(path, snapshot)
    print(f"Snapshot v2 草案已构建: {path.relative_to(WORKSPACE)} ({snapshot['snapshot_id']})")
    return 0


def command_validate_snapshot_v2() -> int:
    path = SNAPSHOTS / "draft" / "snapshot_v2.json"
    if not path.is_file():
        print(f"snapshot v2 草案不存在: {path}", file=sys.stderr)
        return 1
    snapshot = load_json(path)
    errors: list[str] = []
    validate_schema(snapshot, load_json(SCHEMAS / "snapshot.v2.schema.json"), "snapshot_v2", errors)
    if snapshot != build_snapshot_v2(load_all()):
        errors.append("snapshot v2 草案与当前 Registry/草案源的确定性重建结果不一致")
    if errors:
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"Snapshot v2 草案校验通过: {path.relative_to(WORKSPACE)}")
    return 0


def client_drift_findings() -> list[str]:'''

rep("def client_drift_findings() -> list[str]:", V2_BLOCK, "v2 functions insert")

rep(
    '    sub.add_parser("drift-audit", help="扫描客户端当前研究状态的平行事实源")',
    '''    sub.add_parser("drift-audit", help="扫描客户端当前研究状态的平行事实源")
    sub.add_parser("build-snapshot-v2", help="构建 Snapshot v2 草案投影（views/industry/cases）")
    sub.add_parser("validate-snapshot-v2", help="校验 Snapshot v2 草案 Schema 与确定性")''',
    "parse_args v2 commands",
)

rep(
    '''    if args.command == "drift-audit":
        return command_drift_audit()
    return 2''',
    '''    if args.command == "drift-audit":
        return command_drift_audit()
    if args.command == "build-snapshot-v2":
        return command_build_snapshot_v2(data)
    if args.command == "validate-snapshot-v2":
        return command_validate_snapshot_v2()
    return 2''',
    "main dispatch v2",
)

io.open(P, "w", encoding="utf-8", newline="").write(text)
print("PATCH COMPLETE")
