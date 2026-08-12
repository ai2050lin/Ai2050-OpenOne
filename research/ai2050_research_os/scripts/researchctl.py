#!/usr/bin/env python3
"""Validate AI2050 research registries and build deterministic Markdown views."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


OS_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[3]
REGISTRY = OS_ROOT / "registry"
GENERATED = OS_ROOT / "generated"
SCHEMAS = OS_ROOT / "schemas"
CONTRACTS = OS_ROOT / "contracts"
MANIFESTS = OS_ROOT / "manifests"

FILES = {
    "project": "project.json",
    "campaigns": "campaigns.json",
    "hypotheses": "hypotheses.json",
    "puzzles": "puzzles.json",
    "tests": "tests.json",
    "evidence": "evidence.json",
    "phases": "phases.json",
    "decisions": "decisions.json",
    "sources": "sources.json",
    "objects": "objects.json",
    "constructs": "constructs.json",
    "contracts": "contracts.json",
    "runs": "runs.json",
    "artifacts": "artifacts.json",
    "corrections": "corrections.json",
}

VALID = {
    "campaign_status": {"draft", "active", "blocked", "completed", "failed", "archived"},
    "stage_status": {"pending", "active", "blocked", "completed", "failed", "cancelled"},
    "hypothesis_status": {
        "candidate", "discovery_supported", "confirmation_supported", "local_survivor",
        "global_survivor", "bounded_rejected", "abstain", "closed",
    },
    "puzzle_status": {
        "not_started", "instrument_only", "partial", "blocked",
        "resolved_within_scope", "closed_negative", "gate_closed",
    },
    "test_status": {
        "planned", "preregistering", "ready", "running", "auditing",
        "passed", "failed", "blocked", "cancelled",
    },
    "checkpoint_status": {"pending", "partial", "done", "blocked", "not_applicable"},
    "phase_status": {
        "draft", "preregistered", "running", "auditing", "adjudicated",
        "invalid", "censored", "archived",
    },
    "decision_status": {"active", "superseded", "archived"},
    "object_status": {"draft", "historical_qualified", "preregistered", "qualified", "blocked", "closed", "archived"},
    "construct_status": {"draft", "preregistered", "calibrated", "qualified", "failed", "closed", "archived"},
    "contract_status": {"draft", "calibrated", "preregistered", "ready", "running", "auditing", "adjudicated", "blocked", "censored", "invalid", "archived"},
    "run_status": {"planned", "ready", "running", "auditing", "adjudicated", "invalid", "censored", "cancelled", "archived"},
    "artifact_status": {"frozen", "verified", "missing", "invalid", "archived"},
    "correction_status": {"active", "superseded", "archived"},
}

CONTRACT_TRANSITIONS = {
    "draft": {"calibrated", "invalid", "archived"},
    "calibrated": {"preregistered", "invalid", "archived"},
    "preregistered": {"ready", "blocked", "invalid", "archived"},
    "ready": {"running", "blocked", "invalid", "archived"},
    "running": {"auditing", "censored", "invalid"},
    "auditing": {"adjudicated", "censored", "invalid"},
    "adjudicated": {"archived"},
    "blocked": {"preregistered", "archived"},
    "censored": {"archived"},
    "invalid": {"archived"},
    "archived": set(),
}

RUN_TRANSITIONS = {
    "planned": {"ready", "cancelled"},
    "ready": {"running", "cancelled"},
    "running": {"auditing", "invalid", "censored"},
    "auditing": {"adjudicated", "invalid", "censored"},
    "adjudicated": {"archived"},
    "invalid": {"archived"},
    "censored": {"archived"},
    "cancelled": {"archived"},
    "archived": set(),
}

ZH_STATUS = {
    "active": "进行中",
    "pending": "待开始",
    "blocked": "阻塞",
    "completed": "完成",
    "failed": "失败",
    "cancelled": "取消",
    "candidate": "候选",
    "discovery_supported": "发现集支持",
    "confirmation_supported": "确认集支持",
    "local_survivor": "局部幸存",
    "global_survivor": "全局幸存",
    "bounded_rejected": "限定否决",
    "abstain": "拒答",
    "closed": "关闭",
    "not_started": "未开始",
    "instrument_only": "仅仪器层",
    "partial": "部分完成",
    "resolved_within_scope": "范围内解决",
    "closed_negative": "负向关闭",
    "gate_closed": "升级门关闭",
    "planned": "计划中",
    "preregistering": "预注册中",
    "ready": "可运行",
    "running": "运行中",
    "auditing": "审计中",
    "passed": "通过",
    "adjudicated": "已裁决",
    "invalid": "无效",
    "censored": "删失",
    "archived": "归档",
    "draft": "草案",
    "preregistered": "已预注册",
    "superseded": "已取代",
    "done": "完成",
    "not_applicable": "不适用",
    "historical_qualified": "历史资格",
    "qualified": "已资格化",
    "calibrated": "已校准",
    "frozen": "已冻结",
    "verified": "已验证",
}


class ValidationError(Exception):
    pass


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"无法读取 {path}: {exc}") from exc


def load_all() -> dict[str, Any]:
    return {key: load_json(REGISTRY / filename) for key, filename in FILES.items()}


def require_fields(item: dict[str, Any], fields: Iterable[str], where: str, errors: list[str]) -> None:
    for field in fields:
        if field not in item:
            errors.append(f"{where} 缺少字段 {field}")


def unique_map(items: list[dict[str, Any]], key: str, where: str, errors: list[str]) -> dict[Any, dict[str, Any]]:
    result: dict[Any, dict[str, Any]] = {}
    for index, item in enumerate(items):
        if key not in item:
            errors.append(f"{where}[{index}] 缺少主键 {key}")
            continue
        value = item[key]
        if value in result:
            errors.append(f"{where} 主键重复: {value}")
        result[value] = item
    return result


def check_refs(values: Iterable[str], valid_ids: set[str], where: str, errors: list[str]) -> None:
    for value in values:
        if value not in valid_ids:
            errors.append(f"{where} 引用不存在: {value}")


def check_dag(graph: dict[str, list[str]], label: str, errors: list[str]) -> None:
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str, trail: list[str]) -> None:
        if node in visiting:
            errors.append(f"{label} 存在循环依赖: {' -> '.join(trail + [node])}")
            return
        if node in visited:
            return
        visiting.add(node)
        for dep in graph.get(node, []):
            visit(dep, trail + [node])
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node, [])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_blob_oid(commit: str, path: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-c", "maintenance.auto=false", "rev-parse", f"{commit}:{path}"],
            cwd=WORKSPACE,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def type_matches(value: Any, expected: str) -> bool:
    mapping = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, mapping[expected])


def validate_schema(value: Any, schema: dict[str, Any], where: str, errors: list[str]) -> None:
    """Validate the strict JSON-Schema subset used by experiment contracts."""
    expected = schema.get("type")
    if expected is not None:
        allowed = expected if isinstance(expected, list) else [expected]
        if not any(type_matches(value, item) for item in allowed):
            errors.append(f"{where} 类型错误，期望 {allowed}")
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{where} 值不在枚举中: {value}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            errors.append(f"{where} 字符串过短")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            errors.append(f"{where} 不匹配模式 {schema['pattern']}: {value}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{where} 小于最小值 {schema['minimum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            errors.append(f"{where} 项目数不足")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                validate_schema(item, item_schema, f"{where}[{index}]", errors)
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        for field in schema.get("required", []):
            if field not in value:
                errors.append(f"{where} 缺少字段 {field}")
        if schema.get("additionalProperties") is False:
            for field in value:
                if field not in properties:
                    errors.append(f"{where} 含未声明字段 {field}")
        for field, child_schema in properties.items():
            if field in value:
                validate_schema(value[field], child_schema, f"{where}.{field}", errors)


def resolve_os_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else OS_ROOT / path


def verify_manifest_file(path: Path, expected_contract_id: str, errors: list[str]) -> None:
    try:
        manifest = load_json(path)
    except ValidationError as exc:
        errors.append(str(exc))
        return
    require_fields(manifest, ["manifest_version", "contract_id", "frozen_at", "readiness", "files"], f"manifest {path.name}", errors)
    if manifest.get("contract_id") != expected_contract_id:
        errors.append(f"manifest {path.name} contract_id 不匹配")
    if manifest.get("readiness") not in {"contract_frozen_not_run_ready", "run_ready", "adjudicated"}:
        errors.append(f"manifest {path.name} readiness 非法")
    roles: set[str] = set()
    for index, entry in enumerate(manifest.get("files", [])):
        require_fields(entry, ["role", "path", "size_bytes", "sha256"], f"manifest {path.name}.files[{index}]", errors)
        role = entry.get("role")
        if role in roles:
            errors.append(f"manifest {path.name} role 重复: {role}")
        roles.add(role)
        artifact_path = WORKSPACE / entry.get("path", "")
        if not artifact_path.is_file():
            errors.append(f"manifest {path.name} 文件不存在: {entry.get('path')}")
            continue
        if artifact_path.stat().st_size != entry.get("size_bytes"):
            errors.append(f"manifest {path.name} 大小不匹配: {entry.get('path')}")
        if sha256_file(artifact_path) != entry.get("sha256"):
            errors.append(f"manifest {path.name} SHA256 不匹配: {entry.get('path')}")
    if "contract" not in roles or "schema" not in roles:
        errors.append(f"manifest {path.name} 必须包含 contract 和 schema")


def validate(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    project = data["project"]
    cmap = unique_map(data["campaigns"], "id", "campaigns.json", errors)
    hmap = unique_map(data["hypotheses"], "id", "hypotheses.json", errors)
    pmap = unique_map(data["puzzles"], "id", "puzzles.json", errors)
    tmap = unique_map(data["tests"], "id", "tests.json", errors)
    emap = unique_map(data["evidence"], "id", "evidence.json", errors)
    phmap = unique_map(data["phases"], "record_id", "phases.json", errors)
    dmap = unique_map(data["decisions"], "id", "decisions.json", errors)
    smap = unique_map(data["sources"], "id", "sources.json", errors)
    omap = unique_map(data["objects"], "id", "objects.json", errors)
    conmap = unique_map(data["constructs"], "id", "constructs.json", errors)
    contract_map = unique_map(data["contracts"], "id", "contracts.json", errors)
    run_map = unique_map(data["runs"], "id", "runs.json", errors)
    artifact_map = unique_map(data["artifacts"], "id", "artifacts.json", errors)
    correction_map = unique_map(data["corrections"], "id", "corrections.json", errors)

    campaign_ids, hypothesis_ids = set(cmap), set(hmap)
    puzzle_ids, test_ids, evidence_ids = set(pmap), set(tmap), set(emap)
    phase_record_ids, source_ids = set(phmap), set(smap)
    object_ids, construct_ids, contract_ids = set(omap), set(conmap), set(contract_map)

    require_fields(
        project,
        ["schema_version", "project_id", "name", "as_of", "active_campaign_id", "north_star", "current_bottleneck", "next_decision", "source_ids"],
        "project.json",
        errors,
    )
    if project.get("active_campaign_id") not in campaign_ids:
        errors.append(f"active_campaign_id 不存在: {project.get('active_campaign_id')}")
    check_refs(project.get("source_ids", []), source_ids, "project.source_ids", errors)

    for sid, source in smap.items():
        require_fields(
            source,
            ["path", "snapshot_kind", "git_commit", "git_blob_oid", "captured_sha256", "size_bytes", "nul_count", "role", "authority", "notes"],
            f"source {sid}",
            errors,
        )
        path = WORKSPACE / source.get("path", "")
        if not path.is_file():
            errors.append(f"source {sid} 当前路径不存在: {source.get('path')}")
        if source.get("snapshot_kind") != "git_blob":
            errors.append(f"source {sid} snapshot_kind 必须为 git_blob")
        observed_blob = git_blob_oid(str(source.get("git_commit", "")), str(source.get("path", "")))
        if observed_blob != source.get("git_blob_oid"):
            errors.append(f"source {sid} Git blob 不匹配: {observed_blob} != {source.get('git_blob_oid')}")
        if not isinstance(source.get("size_bytes"), int) or source.get("size_bytes", -1) < 0:
            errors.append(f"source {sid} size_bytes 非法")
        if not isinstance(source.get("nul_count"), int) or source.get("nul_count", -1) < 0:
            errors.append(f"source {sid} nul_count 非法")
        if source.get("nul_count", 0) > 0 and source.get("authority") == "primary_evidence":
            errors.append(f"source {sid} 含 NUL，不能标为 primary_evidence")

    work_packages: dict[tuple[str, str], dict[str, Any]] = {}
    for cid, campaign in cmap.items():
        require_fields(
            campaign,
            ["name", "status", "objective", "hypothesis_ids", "puzzle_ids", "stages", "max_gpu_hours", "max_formal_phases", "stop_condition"],
            f"campaign {cid}",
            errors,
        )
        if campaign.get("status") not in VALID["campaign_status"]:
            errors.append(f"campaign {cid} 状态非法: {campaign.get('status')}")
        max_gpu = campaign.get("max_gpu_hours")
        if not isinstance(max_gpu, (int, float)) or isinstance(max_gpu, bool) or max_gpu <= 0:
            errors.append(f"campaign {cid} 必须冻结正数 max_gpu_hours")
        check_refs(campaign.get("hypothesis_ids", []), hypothesis_ids, f"campaign {cid}.hypothesis_ids", errors)
        check_refs(campaign.get("puzzle_ids", []), puzzle_ids, f"campaign {cid}.puzzle_ids", errors)
        stage_ids: set[str] = set()
        active_stages = 0
        stage_budget = 0.0
        for stage in campaign.get("stages", []):
            require_fields(stage, ["id", "name", "status", "test_battery_ids", "entry_gate", "exit_gate", "gpu_hour_budget"], f"campaign {cid} stage", errors)
            stage_id = stage.get("id")
            if stage_id in stage_ids:
                errors.append(f"campaign {cid} 工作包重复: {stage_id}")
            stage_ids.add(stage_id)
            work_packages[(cid, stage_id)] = stage
            if stage.get("status") not in VALID["stage_status"]:
                errors.append(f"campaign {cid}/{stage_id} 状态非法: {stage.get('status')}")
            if stage.get("status") == "active":
                active_stages += 1
            budget = stage.get("gpu_hour_budget")
            if not isinstance(budget, (int, float)) or isinstance(budget, bool) or budget < 0:
                errors.append(f"campaign {cid}/{stage_id} gpu_hour_budget 非法")
            else:
                stage_budget += float(budget)
            check_refs(stage.get("test_battery_ids", []), test_ids, f"campaign {cid}/{stage_id}.test_battery_ids", errors)
        if campaign.get("status") == "active" and active_stages != 1:
            errors.append(f"active campaign {cid} 必须恰有一个 active 工作包，实际 {active_stages}")
        if isinstance(max_gpu, (int, float)) and stage_budget > float(max_gpu) + 1e-9:
            errors.append(f"campaign {cid} 工作包预算和 {stage_budget} 超过总预算 {max_gpu}")

    for hid, hypothesis in hmap.items():
        require_fields(hypothesis, ["name", "claim", "scope", "status", "unique_predictions", "decisive_test_ids", "death_criteria", "reopen_criteria", "evidence_refs"], f"hypothesis {hid}", errors)
        if hypothesis.get("status") not in VALID["hypothesis_status"]:
            errors.append(f"hypothesis {hid} 状态非法: {hypothesis.get('status')}")
        if len(hypothesis.get("unique_predictions", [])) < 3:
            errors.append(f"hypothesis {hid} 少于三个独特预测")
        check_refs(hypothesis.get("decisive_test_ids", []), test_ids, f"hypothesis {hid}.decisive_test_ids", errors)
        check_refs(hypothesis.get("evidence_refs", []), evidence_ids, f"hypothesis {hid}.evidence_refs", errors)

    for construct_id, construct in conmap.items():
        require_fields(construct, ["name", "status", "definition", "observable", "non_equivalences", "required_controls", "target_closure_level"], f"construct {construct_id}", errors)
        if construct.get("status") not in VALID["construct_status"]:
            errors.append(f"construct {construct_id} 状态非法: {construct.get('status')}")
        if construct.get("target_closure_level") not in range(9):
            errors.append(f"construct {construct_id} target_closure_level 非法")
        if len(construct.get("required_controls", [])) < 3:
            errors.append(f"construct {construct_id} 控制不足")

    for object_id, obj in omap.items():
        require_fields(obj, ["name", "status", "scope", "construct_ids", "evidence_refs", "source_record_ids", "highest_closure_level", "next_contract_id", "forbids"], f"object {object_id}", errors)
        if obj.get("status") not in VALID["object_status"]:
            errors.append(f"object {object_id} 状态非法: {obj.get('status')}")
        check_refs(obj.get("construct_ids", []), construct_ids, f"object {object_id}.construct_ids", errors)
        check_refs(obj.get("evidence_refs", []), evidence_ids, f"object {object_id}.evidence_refs", errors)
        check_refs(obj.get("source_record_ids", []), phase_record_ids, f"object {object_id}.source_record_ids", errors)
        if obj.get("next_contract_id") is not None:
            check_refs([obj["next_contract_id"]], contract_ids, f"object {object_id}.next_contract_id", errors)
        if obj.get("highest_closure_level") not in range(9):
            errors.append(f"object {object_id} highest_closure_level 非法")

    puzzle_graph: dict[str, list[str]] = {}
    for pid, puzzle in pmap.items():
        require_fields(puzzle, ["name", "question", "status", "current_closure_level", "target_closure_level", "dependencies", "blocker", "next_test_id", "evidence_refs", "checkpoints"], f"puzzle {pid}", errors)
        if puzzle.get("status") not in VALID["puzzle_status"]:
            errors.append(f"puzzle {pid} 状态非法: {puzzle.get('status')}")
        current, target = puzzle.get("current_closure_level"), puzzle.get("target_closure_level")
        if not isinstance(current, int) or current not in range(9):
            errors.append(f"puzzle {pid} current_closure_level 非法: {current}")
        if not isinstance(target, int) or target not in range(9):
            errors.append(f"puzzle {pid} target_closure_level 非法: {target}")
        if isinstance(current, int) and isinstance(target, int) and current > target:
            errors.append(f"puzzle {pid} 当前层级高于目标层级")
        dependencies = puzzle.get("dependencies", [])
        puzzle_graph[pid] = dependencies
        check_refs(dependencies, puzzle_ids, f"puzzle {pid}.dependencies", errors)
        if puzzle.get("next_test_id") is not None:
            check_refs([puzzle["next_test_id"]], test_ids, f"puzzle {pid}.next_test_id", errors)
        check_refs(puzzle.get("evidence_refs", []), evidence_ids, f"puzzle {pid}.evidence_refs", errors)
        checkpoint_ids: set[str] = set()
        for checkpoint in puzzle.get("checkpoints", []):
            require_fields(checkpoint, ["id", "label", "status"], f"puzzle {pid} checkpoint", errors)
            checkpoint_id = checkpoint.get("id")
            if checkpoint_id in checkpoint_ids:
                errors.append(f"puzzle {pid} 检查项重复: {checkpoint_id}")
            checkpoint_ids.add(checkpoint_id)
            if checkpoint.get("status") not in VALID["checkpoint_status"]:
                errors.append(f"puzzle {pid}/{checkpoint_id} 状态非法: {checkpoint.get('status')}")
    check_dag(puzzle_graph, "拼图", errors)

    test_graph: dict[str, list[str]] = {}
    for tid, test in tmap.items():
        require_fields(test, ["name", "status", "cost_tier", "objective", "axes", "partitions", "controls", "metrics", "pass_rule", "fail_rule", "outputs", "hypothesis_ids", "puzzle_ids", "prerequisites"], f"test {tid}", errors)
        if test.get("status") not in VALID["test_status"]:
            errors.append(f"test {tid} 状态非法: {test.get('status')}")
        check_refs(test.get("hypothesis_ids", []), hypothesis_ids, f"test {tid}.hypothesis_ids", errors)
        check_refs(test.get("puzzle_ids", []), puzzle_ids, f"test {tid}.puzzle_ids", errors)
        prerequisites = test.get("prerequisites", [])
        test_graph[tid] = prerequisites
        check_refs(prerequisites, test_ids, f"test {tid}.prerequisites", errors)
    check_dag(test_graph, "测试", errors)

    for record_id, phase in phmap.items():
        require_fields(phase, ["phase", "phase_label", "occurrence", "date", "phase_type", "status", "campaign_id", "evidence_refs", "object_ids", "construct_ids", "source_id", "source_line", "verdict", "auto_continue"], f"phase record {record_id}", errors)
        if phase.get("status") not in VALID["phase_status"]:
            errors.append(f"phase record {record_id} 状态非法: {phase.get('status')}")
        if phase.get("campaign_id") != "LEGACY" and phase.get("campaign_id") not in campaign_ids:
            errors.append(f"phase record {record_id} campaign_id 不存在: {phase.get('campaign_id')}")
        check_refs(phase.get("evidence_refs", []), evidence_ids, f"phase record {record_id}.evidence_refs", errors)
        check_refs(phase.get("object_ids", []), object_ids, f"phase record {record_id}.object_ids", errors)
        check_refs(phase.get("construct_ids", []), construct_ids, f"phase record {record_id}.construct_ids", errors)
        check_refs([phase.get("source_id")], source_ids, f"phase record {record_id}.source_id", errors)
        if not isinstance(phase.get("source_line"), int) or phase.get("source_line", 0) < 1:
            errors.append(f"phase record {record_id} source_line 非法")

    for eid, record in emap.items():
        require_fields(record, ["phase", "grade", "closure_level", "polarity", "title", "claim", "scope", "authorizes", "forbids", "puzzle_ids", "hypothesis_ids", "source_record_ids"], f"evidence {eid}", errors)
        if not isinstance(record.get("closure_level"), int) or record.get("closure_level") not in range(9):
            errors.append(f"evidence {eid} closure_level 非法: {record.get('closure_level')}")
        if not str(record.get("grade", "")).startswith(("E0", "E1", "E2", "E3")):
            errors.append(f"evidence {eid} grade 非法: {record.get('grade')}")
        check_refs(record.get("puzzle_ids", []), puzzle_ids, f"evidence {eid}.puzzle_ids", errors)
        check_refs(record.get("hypothesis_ids", []), hypothesis_ids, f"evidence {eid}.hypothesis_ids", errors)
        check_refs(record.get("source_record_ids", []), phase_record_ids, f"evidence {eid}.source_record_ids", errors)

    schema_path = SCHEMAS / "experiment_contract.schema.json"
    contract_schema = load_json(schema_path) if schema_path.is_file() else None
    if contract_schema is None:
        errors.append("experiment contract schema 不存在")
    for contract_id, index in contract_map.items():
        require_fields(index, ["path", "schema_path", "status", "previous_status", "campaign_id", "work_package_id", "object_ids", "construct_ids", "contract_sha256", "manifest_path", "frozen_at", "run_ready"], f"contract index {contract_id}", errors)
        if index.get("status") not in VALID["contract_status"]:
            errors.append(f"contract {contract_id} 状态非法: {index.get('status')}")
        previous_status = index.get("previous_status")
        current_status = index.get("status")
        if previous_status not in CONTRACT_TRANSITIONS or current_status not in CONTRACT_TRANSITIONS.get(previous_status, set()):
            errors.append(f"contract {contract_id} 非法状态迁移: {previous_status} -> {current_status}")
        check_refs([index.get("campaign_id")], campaign_ids, f"contract {contract_id}.campaign_id", errors)
        if (index.get("campaign_id"), index.get("work_package_id")) not in work_packages:
            errors.append(f"contract {contract_id} 工作包不存在: {index.get('work_package_id')}")
        check_refs(index.get("object_ids", []), object_ids, f"contract {contract_id}.object_ids", errors)
        check_refs(index.get("construct_ids", []), construct_ids, f"contract {contract_id}.construct_ids", errors)
        contract_path = resolve_os_path(index.get("path", ""))
        if not contract_path.is_file():
            errors.append(f"contract {contract_id} 文件不存在: {index.get('path')}")
            continue
        try:
            contract = load_json(contract_path)
        except ValidationError as exc:
            errors.append(str(exc))
            continue
        if contract_schema is not None:
            validate_schema(contract, contract_schema, f"contract {contract_id}", errors)
        if contract.get("experiment_id") != contract_id:
            errors.append(f"contract {contract_id} 文件内 experiment_id 不一致")
        check_refs(contract.get("object_ids", []), object_ids, f"contract {contract_id}.object_ids", errors)
        check_refs(contract.get("construct_types", []), construct_ids, f"contract {contract_id}.construct_types", errors)
        check_refs(contract.get("puzzle_ids", []), puzzle_ids, f"contract {contract_id}.puzzle_ids", errors)
        for prediction in contract.get("hypothesis_predictions", []):
            check_refs([prediction.get("hypothesis_id")], hypothesis_ids, f"contract {contract_id}.hypothesis_predictions", errors)
            check_refs(prediction.get("distinguishes_from", []), hypothesis_ids, f"contract {contract_id}.distinguishes_from", errors)
        observed_sha = sha256_file(contract_path)
        if index.get("contract_sha256") != observed_sha:
            errors.append(f"contract {contract_id} SHA256 未冻结或不匹配")
        if index.get("status") in {"preregistered", "ready", "running", "auditing", "adjudicated"}:
            if not index.get("frozen_at"):
                errors.append(f"contract {contract_id} 已预注册但 frozen_at 为空")
            manifest_path = resolve_os_path(index.get("manifest_path", ""))
            if not manifest_path.is_file():
                errors.append(f"contract {contract_id} manifest 不存在")
            else:
                verify_manifest_file(manifest_path, contract_id, errors)
        stage = work_packages.get((index.get("campaign_id"), index.get("work_package_id")), {})
        contract_budget = contract.get("budget", {}).get("max_gpu_hours")
        if isinstance(contract_budget, (int, float)) and contract_budget > stage.get("gpu_hour_budget", -1):
            errors.append(f"contract {contract_id} GPU预算超过工作包上限")
        if contract.get("data_contract", {}).get("confirmation", {}).get("sealed") is not True:
            errors.append(f"contract {contract_id} confirmation 未密封")
        if index.get("run_ready") and contract.get("frozen_artifacts", {}).get("readiness") != "run_ready":
            errors.append(f"contract {contract_id} run_ready 与合同不一致")

    for run_id, run in run_map.items():
        require_fields(run, ["contract_id", "status", "model", "started_at", "ended_at", "artifact_ids", "previous_status"], f"run {run_id}", errors)
        if run.get("status") not in VALID["run_status"]:
            errors.append(f"run {run_id} 状态非法")
        check_refs([run.get("contract_id")], contract_ids, f"run {run_id}.contract_id", errors)
        check_refs(run.get("artifact_ids", []), set(artifact_map), f"run {run_id}.artifact_ids", errors)
        if run.get("previous_status") == run.get("status"):
            errors.append(f"run {run_id} previous_status 不得等于当前状态")
        if run.get("previous_status") not in RUN_TRANSITIONS or run.get("status") not in RUN_TRANSITIONS.get(run.get("previous_status"), set()):
            errors.append(f"run {run_id} 非法状态迁移: {run.get('previous_status')} -> {run.get('status')}")
        contract_index = contract_map.get(run.get("contract_id"), {})
        if contract_index.get("run_ready") is not True or contract_index.get("status") not in {"ready", "running", "auditing", "adjudicated"}:
            errors.append(f"run {run_id} 未获得合同 run_ready 授权")

    for artifact_id, artifact in artifact_map.items():
        require_fields(artifact, ["run_id", "contract_id", "status", "path", "sha256", "size_bytes", "kind"], f"artifact {artifact_id}", errors)
        if artifact.get("status") not in VALID["artifact_status"]:
            errors.append(f"artifact {artifact_id} 状态非法")
        check_refs([artifact.get("run_id")], set(run_map), f"artifact {artifact_id}.run_id", errors)
        check_refs([artifact.get("contract_id")], contract_ids, f"artifact {artifact_id}.contract_id", errors)

    for correction_id, correction in correction_map.items():
        require_fields(correction, ["date", "status", "target_type", "target_ids", "problem", "correction", "preserves_original_claim"], f"correction {correction_id}", errors)
        if correction.get("status") not in VALID["correction_status"]:
            errors.append(f"correction {correction_id} 状态非法")

    for did, decision in dmap.items():
        require_fields(decision, ["date", "status", "title", "basis", "decision", "authorizes", "forbids"], f"decision {did}", errors)
        if decision.get("status") not in VALID["decision_status"]:
            errors.append(f"decision {did} 状态非法: {decision.get('status')}")
        check_refs(decision.get("basis", []), evidence_ids, f"decision {did}.basis", errors)

    numeric_phases = [phase.get("phase") for phase in data["phases"] if isinstance(phase.get("phase"), int)]
    if numeric_phases and project.get("latest_recorded_phase") != max(numeric_phases):
        errors.append("project.latest_recorded_phase 与 phase records 最大值不一致")
    return errors


def esc(value: Any) -> str:
    text = "" if value is None else str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def short(value: str, limit: int = 92) -> str:
    value = " ".join(value.split())
    return value if len(value) <= limit else value[: limit - 1] + "…"


def refs(values: Iterable[Any]) -> str:
    values = list(values)
    return ", ".join(str(v) for v in values) if values else "—"


def status_zh(value: str) -> str:
    return ZH_STATUS.get(value, value)


def puzzle_progress(puzzle: dict[str, Any]) -> tuple[float, float, int]:
    weights = {"done": 1.0, "partial": 0.5, "pending": 0.0, "blocked": 0.0}
    relevant = [c for c in puzzle.get("checkpoints", []) if c.get("status") != "not_applicable"]
    score = sum(weights.get(c.get("status"), 0.0) for c in relevant)
    total = len(relevant)
    return (score / total if total else 0.0, score, total)


def md_header(project: dict[str, Any], title: str) -> list[str]:
    return [
        f"# {title}",
        "",
        f"> 自动生成于账本基线 `{project['as_of']}`。请修改 `registry/`，不要手工修改本文件。",
        "",
    ]


def build_status(data: dict[str, Any]) -> str:
    project = data["project"]
    campaign = next(c for c in data["campaigns"] if c["id"] == project["active_campaign_id"])
    active_stage = next((s for s in campaign["stages"] if s["status"] == "active"), None)
    puzzle_counts = Counter(p["status"] for p in data["puzzles"])
    hypothesis_counts = Counter(h["status"] for h in data["hypotheses"])
    lines = md_header(project, "项目状态总览")
    lines += [
        "## 当前裁决",
        "",
        f"- 活跃战役：`{campaign['id']}` {campaign['name']}（{status_zh(campaign['status'])}）",
        f"- 当前工作包：`{active_stage['id']}` {active_stage['name']}" if active_stage else "- 当前工作包：无",
        f"- 最新迁移 Phase：`{project['latest_recorded_phase']}`",
        f"- 自动续行：`{str(project['auto_continue']).lower()}`",
        f"- 当前瓶颈：{project['current_bottleneck']}",
        f"- 下一决策：{project['next_decision']}",
        "",
        "## 状态计数",
        "",
        "| 对象 | 状态 | 数量 |",
        "|---|---|---:|",
    ]
    for key, count in sorted(puzzle_counts.items()):
        lines.append(f"| 关键拼图 | {status_zh(key)} | {count} |")
    for key, count in sorted(hypothesis_counts.items()):
        lines.append(f"| 候选机制 | {status_zh(key)} | {count} |")
    lines += [
        "",
        "## 当前工作包出口条件",
        "",
        active_stage["exit_gate"] if active_stage else "—",
        "",
        "## 当前理论状态",
        "",
        project["theory_status"],
        "",
        "## 最近 Phase",
        "",
        "| Phase | 类型 | 裁决 | 自动续行 | 证据 |",
        "|---:|---|---|---|---|",
    ]
    for phase in sorted(data["phases"], key=lambda x: (x["phase"], x["occurrence"]), reverse=True)[:8]:
        lines.append(
            f"| {phase['phase']} | {esc(phase['phase_type'])} | {esc(phase['verdict'])} | "
            f"{str(phase['auto_continue']).lower()} | {esc(refs(phase['evidence_refs']))} |"
        )
    return "\n".join(lines) + "\n"


def build_puzzles(data: dict[str, Any]) -> str:
    lines = md_header(data["project"], "关键拼图看板")
    lines += [
        "| 拼图 | 状态 | 层级 | 工程进度 | 依赖 | 下一测试 | 当前阻塞 |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for puzzle in data["puzzles"]:
        progress, score, total = puzzle_progress(puzzle)
        lines.append(
            f"| `{puzzle['id']}` {esc(puzzle['name'])} | {status_zh(puzzle['status'])} | "
            f"L{puzzle['current_closure_level']}→L{puzzle['target_closure_level']} | {progress:.0%} ({score:.1f}/{total}) | "
            f"{esc(refs(puzzle['dependencies']))} | {esc(puzzle['next_test_id'] or '—')} | {esc(short(puzzle['blocker']))} |"
        )
    for puzzle in data["puzzles"]:
        lines += [
            "",
            f"## {puzzle['id']}：{puzzle['name']}",
            "",
            puzzle["question"],
            "",
            f"证据：{refs(puzzle['evidence_refs'])}",
            "",
            "| 检查项 | 状态 |",
            "|---|---|",
        ]
        for checkpoint in puzzle["checkpoints"]:
            lines.append(f"| {esc(checkpoint['label'])} | {status_zh(checkpoint['status'])} |")
    return "\n".join(lines) + "\n"


def build_hypotheses(data: dict[str, Any]) -> str:
    lines = md_header(data["project"], "候选机制记分牌")
    lines += [
        "| 假说 | 状态 | 声明范围 | 关键证据 | 决定性测试 |",
        "|---|---|---|---|---|",
    ]
    for hypothesis in data["hypotheses"]:
        lines.append(
            f"| `{hypothesis['id']}` {esc(hypothesis['name'])} | {status_zh(hypothesis['status'])} | "
            f"{esc(short(hypothesis['scope'], 70))} | {esc(refs(hypothesis['evidence_refs']))} | "
            f"{esc(refs(hypothesis['decisive_test_ids']))} |"
        )
    for hypothesis in data["hypotheses"]:
        lines += [
            "",
            f"## {hypothesis['id']}：{hypothesis['name']}",
            "",
            f"- 主张：{hypothesis['claim']}",
            f"- 当前理由：{hypothesis['status_reason']}",
            f"- 死亡条件：{hypothesis['death_criteria']}",
            f"- 重开条件：{hypothesis['reopen_criteria']}",
            "- 独特预测：",
            "",
        ]
        lines.extend(f"  {index}. {prediction}" for index, prediction in enumerate(hypothesis["unique_predictions"], start=1))
    return "\n".join(lines) + "\n"


def build_tests(data: dict[str, Any]) -> str:
    lines = md_header(data["project"], "测试矩阵")
    lines += [
        "| 测试 | 状态 | 成本 | 目标 | 前置 | 候选 | 拼图 |",
        "|---|---|---|---|---|---|---|",
    ]
    for test in data["tests"]:
        lines.append(
            f"| `{test['id']}` {esc(test['name'])} | {status_zh(test['status'])} | {esc(test['cost_tier'])} | "
            f"{esc(short(test['objective'], 74))} | {esc(refs(test['prerequisites']))} | "
            f"{esc(refs(test['hypothesis_ids']))} | {esc(refs(test['puzzle_ids']))} |"
        )
    for test in data["tests"]:
        lines += [
            "",
            f"## {test['id']}：{test['name']}",
            "",
            f"- 目标：{test['objective']}",
            f"- 轴：{refs(test['axes'])}",
            f"- 分区：{refs(test['partitions'])}",
            f"- 控制：{refs(test['controls'])}",
            f"- 指标：{refs(test['metrics'])}",
            f"- 通过：{test['pass_rule']}",
            f"- 失败：{test['fail_rule']}",
            f"- 产物：{refs(test['outputs'])}",
        ]
    return "\n".join(lines) + "\n"


def build_evidence(data: dict[str, Any]) -> str:
    lines = md_header(data["project"], "关键证据账")
    lines += [
        "> 本文件是当前已迁移关键子集，不替代 K1–K210 原冻结账本。",
        "",
        "| 证据 | Phase | 等级 | 层级 | 极性 | 结论 | 适用域 |",
        "|---|---:|---|---:|---|---|---|",
    ]
    for record in data["evidence"]:
        lines.append(
            f"| `{record['id']}` {esc(record['title'])} | {record['phase']} | {esc(record['grade'])} | "
            f"L{record['closure_level']} | {esc(record['polarity'])} | {esc(short(record['claim']))} | "
            f"{esc(short(record['scope'], 76))} |"
        )
    for record in data["evidence"]:
        lines += [
            "",
            f"## {record['id']}：{record['title']}",
            "",
            f"- Claim（主张）：{record['claim']}",
            f"- Scope（适用域）：{record['scope']}",
            f"- Authorizes（授权）：{refs(record['authorizes'])}",
            f"- Forbids（禁止）：{refs(record['forbids'])}",
            f"- 拼图：{refs(record['puzzle_ids'])}",
            f"- 候选：{refs(record['hypothesis_ids'])}",
            f"- 来源记录：{refs(record['source_record_ids'])}",
        ]
    return "\n".join(lines) + "\n"


def build_decisions(data: dict[str, Any]) -> str:
    lines = md_header(data["project"], "决策与权限总账")
    lines += [
        "| 决策 | 日期 | 状态 | 依据 | 裁决 |",
        "|---|---|---|---|---|",
    ]
    for decision in data["decisions"]:
        lines.append(
            f"| `{decision['id']}` {esc(decision['title'])} | {decision['date']} | {status_zh(decision['status'])} | "
            f"{esc(refs(decision['basis']))} | {esc(short(decision['decision']))} |"
        )
    for decision in data["decisions"]:
        lines += [
            "",
            f"## {decision['id']}：{decision['title']}",
            "",
            decision["decision"],
            "",
            f"- 依据：{refs(decision['basis'])}",
            f"- 授权：{refs(decision['authorizes'])}",
            f"- 禁止：{refs(decision['forbids'])}",
        ]
    return "\n".join(lines) + "\n"


def build_campaign(data: dict[str, Any]) -> str:
    project = data["project"]
    campaign = next(c for c in data["campaigns"] if c["id"] == project["active_campaign_id"])
    lines = md_header(project, f"{campaign['id']} 战役计划")
    lines += [
        campaign["objective"],
        "",
        f"- 范围：{campaign['scope']}",
        f"- 成功条件：{campaign['success_condition']}",
        f"- 停止条件：{campaign['stop_condition']}",
        f"- 最大正式 Phase：{campaign['max_formal_phases']}",
        f"- 计算预算：{campaign['budget_status']}",
        "",
        "| 工作包 | 状态 | GPU小时上限 | 测试 | 入口 | 出口 |",
        "|---|---|---:|---|---|---|",
    ]
    for stage in campaign["stages"]:
        lines.append(
            f"| `{stage['id']}` {esc(stage['name'])} | {status_zh(stage['status'])} | {stage['gpu_hour_budget']} | "
            f"{esc(refs(stage['test_battery_ids']))} | {esc(short(stage['entry_gate'], 72))} | "
            f"{esc(short(stage['exit_gate'], 72))} |"
        )
    return "\n".join(lines) + "\n"


def build_sources(data: dict[str, Any]) -> str:
    lines = md_header(data["project"], "不可变来源快照账")
    lines += [
        "| 来源 | 角色 | 权限 | 大小 | NUL | Git commit | Blob | 路径 |",
        "|---|---|---|---:|---:|---|---|---|",
    ]
    for source in data["sources"]:
        lines.append(
            f"| `{source['id']}` | {esc(source['role'])} | {esc(source['authority'])} | "
            f"{source['size_bytes']} | {source['nul_count']} | `{source['git_commit'][:12]}` | "
            f"`{source['git_blob_oid'][:12]}` | {esc(source['path'])} |"
        )
    lines += ["", "> 含 NUL 的来源只保留原始字节，不允许静默规范化后升级证据。", ""]
    return "\n".join(lines) + "\n"


def build_objects(data: dict[str, Any]) -> str:
    lines = md_header(data["project"], "研究对象与构念账")
    lines += [
        "## 对象",
        "",
        "| 对象 | 状态 | 最高层级 | 构念 | 下一合同 | 证据 |",
        "|---|---|---:|---|---|---|",
    ]
    for obj in data["objects"]:
        lines.append(
            f"| `{obj['id']}` {esc(obj['name'])} | {status_zh(obj['status'])} | L{obj['highest_closure_level']} | "
            f"{esc(refs(obj['construct_ids']))} | {esc(obj['next_contract_id'] or '—')} | {esc(refs(obj['evidence_refs']))} |"
        )
    lines += ["", "## 构念", "", "| 构念 | 状态 | 定义 | 明确不等价 |", "|---|---|---|---|"]
    for construct in data["constructs"]:
        lines.append(
            f"| `{construct['id']}` {esc(construct['name'])} | {status_zh(construct['status'])} | "
            f"{esc(short(construct['definition'], 90))} | {esc(refs(construct['non_equivalences']))} |"
        )
    return "\n".join(lines) + "\n"


def build_execution_ledger(data: dict[str, Any]) -> str:
    lines = md_header(data["project"], "合同、运行与产物账")
    lines += [
        "## 合同",
        "",
        "| 合同 | 状态 | 工作包 | 对象 | SHA256 | Manifest | Run-ready |",
        "|---|---|---|---|---|---|---|",
    ]
    for contract in data["contracts"]:
        digest = contract.get("contract_sha256") or "—"
        lines.append(
            f"| `{contract['id']}` | {status_zh(contract['status'])} | {contract['work_package_id']} | "
            f"{esc(refs(contract['object_ids']))} | `{digest[:16]}` | {esc(contract['manifest_path'])} | "
            f"{str(contract['run_ready']).lower()} |"
        )
    lines += [
        "",
        "## 运行",
        "",
        f"当前登记运行数：{len(data['runs'])}。",
        "",
        "## 产物",
        "",
        f"当前登记产物数：{len(data['artifacts'])}。大型张量留在本地结果目录，账本只登记摘要与哈希。",
        "",
        "## 勘误",
        "",
    ]
    for correction in data["corrections"]:
        lines.append(f"- `{correction['id']}`：{correction['problem']} → {correction['correction']}")
    return "\n".join(lines) + "\n"


def rendered_files(data: dict[str, Any]) -> dict[str, str]:
    return {
        "STATUS.md": build_status(data),
        "CAMPAIGN_PLAN.md": build_campaign(data),
        "PUZZLE_BOARD.md": build_puzzles(data),
        "HYPOTHESIS_SCOREBOARD.md": build_hypotheses(data),
        "TEST_MATRIX.md": build_tests(data),
        "EVIDENCE_LEDGER.md": build_evidence(data),
        "DECISION_LEDGER.md": build_decisions(data),
        "SOURCE_LEDGER.md": build_sources(data),
        "OBJECT_LEDGER.md": build_objects(data),
        "EXECUTION_LEDGER.md": build_execution_ledger(data),
    }


def command_validate(data: dict[str, Any]) -> int:
    errors = validate(data)
    if errors:
        print(f"校验失败：{len(errors)} 项", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    counts = {name: len(data[name]) for name in FILES if name != "project"}
    print("校验通过：" + ", ".join(f"{name}={count}" for name, count in counts.items()))
    return 0


def command_build(data: dict[str, Any], check_clean: bool) -> int:
    errors = validate(data)
    if errors:
        return command_validate(data)
    outputs = rendered_files(data)
    GENERATED.mkdir(parents=True, exist_ok=True)
    changed: list[str] = []
    for filename, content in outputs.items():
        path = GENERATED / filename
        old = path.read_text(encoding="utf-8") if path.exists() else None
        if old != content:
            changed.append(filename)
            if not check_clean:
                path.write_text(content, encoding="utf-8", newline="\n")
    if check_clean and changed:
        print("生成看板不是最新：" + ", ".join(changed), file=sys.stderr)
        return 1
    if check_clean:
        print("生成看板与账本一致。")
    else:
        print(f"已生成 {len(outputs)} 个看板；更新 {len(changed)} 个。")
    return 0


def command_summary(data: dict[str, Any]) -> int:
    errors = validate(data)
    if errors:
        return command_validate(data)
    project = data["project"]
    campaign = next(c for c in data["campaigns"] if c["id"] == project["active_campaign_id"])
    active_stage = next(s for s in campaign["stages"] if s["status"] == "active")
    blocked = [p for p in data["puzzles"] if p["status"] == "blocked"]
    print(f"项目：{project['name']}")
    print(f"战役：{campaign['id']} {campaign['name']}")
    print(f"工作包：{active_stage['id']} {active_stage['name']}")
    print(f"瓶颈：{project['current_bottleneck']}")
    print(f"下一决策：{project['next_decision']}")
    print("阻塞拼图：" + (", ".join(f"{p['id']} {p['name']}" for p in blocked) if blocked else "无"))
    return 0


def command_freeze(path_text: str) -> int:
    path = Path(path_text)
    if not path.is_absolute():
        path = (Path.cwd() / path).resolve()
    if not path.is_file():
        print(f"冻结目标不存在或不是文件: {path}", file=sys.stderr)
        return 1
    payload = path.read_bytes()
    result = {
        "path": str(path),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def command_verify_manifest(path_text: str) -> int:
    path = resolve_os_path(path_text)
    if not path.is_file():
        print(f"manifest 不存在: {path}", file=sys.stderr)
        return 1
    try:
        manifest = load_json(path)
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    contract_id = str(manifest.get("contract_id", ""))
    errors: list[str] = []
    verify_manifest_file(path, contract_id, errors)
    if errors:
        print(f"manifest 校验失败：{len(errors)} 项", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"manifest 校验通过：{contract_id}，files={len(manifest['files'])}，readiness={manifest['readiness']}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="校验全部机器账本")
    build = sub.add_parser("build", help="从账本生成 Markdown 看板")
    build.add_argument("--check-clean", action="store_true", help="只检查生成看板是否最新")
    sub.add_parser("summary", help="输出当前状态摘要")
    freeze = sub.add_parser("freeze", help="计算预注册工件 SHA256")
    freeze.add_argument("path", help="要冻结的文件")
    verify_manifest = sub.add_parser("verify-manifest", help="复算冻结 manifest 中全部文件")
    verify_manifest.add_argument("path", help="相对 Research OS 根目录的 manifest 路径")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        data = load_all()
    except ValidationError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if args.command == "validate":
        return command_validate(data)
    if args.command == "build":
        return command_build(data, args.check_clean)
    if args.command == "summary":
        return command_summary(data)
    if args.command == "freeze":
        return command_freeze(args.path)
    if args.command == "verify-manifest":
        return command_verify_manifest(args.path)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
