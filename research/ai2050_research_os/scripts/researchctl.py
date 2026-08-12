#!/usr/bin/env python3
"""Validate AI2050 research registries and build deterministic Markdown views."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "registry"
GENERATED = ROOT / "generated"

FILES = {
    "project": "project.json",
    "campaigns": "campaigns.json",
    "hypotheses": "hypotheses.json",
    "puzzles": "puzzles.json",
    "tests": "tests.json",
    "evidence": "evidence.json",
    "phases": "phases.json",
    "decisions": "decisions.json",
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


def validate(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    project = data["project"]
    campaigns = data["campaigns"]
    hypotheses = data["hypotheses"]
    puzzles = data["puzzles"]
    tests = data["tests"]
    evidence = data["evidence"]
    phases = data["phases"]
    decisions = data["decisions"]

    require_fields(
        project,
        ["schema_version", "project_id", "name", "as_of", "active_campaign_id", "north_star", "current_bottleneck", "next_decision"],
        "project.json",
        errors,
    )

    cmap = unique_map(campaigns, "id", "campaigns.json", errors)
    hmap = unique_map(hypotheses, "id", "hypotheses.json", errors)
    pmap = unique_map(puzzles, "id", "puzzles.json", errors)
    tmap = unique_map(tests, "id", "tests.json", errors)
    emap = unique_map(evidence, "id", "evidence.json", errors)
    phmap = unique_map(phases, "phase", "phases.json", errors)
    dmap = unique_map(decisions, "id", "decisions.json", errors)

    campaign_ids = set(cmap)
    hypothesis_ids = set(hmap)
    puzzle_ids = set(pmap)
    test_ids = set(tmap)
    evidence_ids = set(emap)

    if project.get("active_campaign_id") not in campaign_ids:
        errors.append(f"active_campaign_id 不存在: {project.get('active_campaign_id')}")

    for cid, campaign in cmap.items():
        require_fields(campaign, ["name", "status", "objective", "hypothesis_ids", "puzzle_ids", "stages"], f"campaign {cid}", errors)
        if campaign.get("status") not in VALID["campaign_status"]:
            errors.append(f"campaign {cid} 状态非法: {campaign.get('status')}")
        check_refs(campaign.get("hypothesis_ids", []), hypothesis_ids, f"campaign {cid}.hypothesis_ids", errors)
        check_refs(campaign.get("puzzle_ids", []), puzzle_ids, f"campaign {cid}.puzzle_ids", errors)
        stage_ids: set[str] = set()
        active_stages = 0
        for stage in campaign.get("stages", []):
            require_fields(stage, ["id", "name", "status", "test_battery_ids", "entry_gate", "exit_gate"], f"campaign {cid} stage", errors)
            sid = stage.get("id")
            if sid in stage_ids:
                errors.append(f"campaign {cid} 工作包重复: {sid}")
            stage_ids.add(sid)
            if stage.get("status") not in VALID["stage_status"]:
                errors.append(f"campaign {cid}/{sid} 状态非法: {stage.get('status')}")
            if stage.get("status") == "active":
                active_stages += 1
            check_refs(stage.get("test_battery_ids", []), test_ids, f"campaign {cid}/{sid}.test_battery_ids", errors)
        if campaign.get("status") == "active" and active_stages != 1:
            errors.append(f"active campaign {cid} 必须恰有一个 active 工作包，实际 {active_stages}")

    for hid, hypothesis in hmap.items():
        require_fields(
            hypothesis,
            ["name", "claim", "scope", "status", "unique_predictions", "decisive_test_ids", "death_criteria", "reopen_criteria", "evidence_refs"],
            f"hypothesis {hid}",
            errors,
        )
        if hypothesis.get("status") not in VALID["hypothesis_status"]:
            errors.append(f"hypothesis {hid} 状态非法: {hypothesis.get('status')}")
        if len(hypothesis.get("unique_predictions", [])) < 3:
            errors.append(f"hypothesis {hid} 少于三个独特预测")
        check_refs(hypothesis.get("decisive_test_ids", []), test_ids, f"hypothesis {hid}.decisive_test_ids", errors)
        check_refs(hypothesis.get("evidence_refs", []), evidence_ids, f"hypothesis {hid}.evidence_refs", errors)

    puzzle_graph: dict[str, list[str]] = {}
    for pid, puzzle in pmap.items():
        require_fields(
            puzzle,
            ["name", "question", "status", "current_closure_level", "target_closure_level", "dependencies", "blocker", "next_test_id", "evidence_refs", "checkpoints"],
            f"puzzle {pid}",
            errors,
        )
        if puzzle.get("status") not in VALID["puzzle_status"]:
            errors.append(f"puzzle {pid} 状态非法: {puzzle.get('status')}")
        current = puzzle.get("current_closure_level")
        target = puzzle.get("target_closure_level")
        if not isinstance(current, int) or current not in range(9):
            errors.append(f"puzzle {pid} current_closure_level 非法: {current}")
        if not isinstance(target, int) or target not in range(9):
            errors.append(f"puzzle {pid} target_closure_level 非法: {target}")
        if isinstance(current, int) and isinstance(target, int) and current > target:
            errors.append(f"puzzle {pid} 当前层级高于目标层级")
        deps = puzzle.get("dependencies", [])
        puzzle_graph[pid] = deps
        check_refs(deps, puzzle_ids, f"puzzle {pid}.dependencies", errors)
        if puzzle.get("next_test_id") is not None:
            check_refs([puzzle["next_test_id"]], test_ids, f"puzzle {pid}.next_test_id", errors)
        check_refs(puzzle.get("evidence_refs", []), evidence_ids, f"puzzle {pid}.evidence_refs", errors)
        checkpoint_ids: set[str] = set()
        for checkpoint in puzzle.get("checkpoints", []):
            require_fields(checkpoint, ["id", "label", "status"], f"puzzle {pid} checkpoint", errors)
            cpid = checkpoint.get("id")
            if cpid in checkpoint_ids:
                errors.append(f"puzzle {pid} 检查项重复: {cpid}")
            checkpoint_ids.add(cpid)
            if checkpoint.get("status") not in VALID["checkpoint_status"]:
                errors.append(f"puzzle {pid}/{cpid} 状态非法: {checkpoint.get('status')}")
    check_dag(puzzle_graph, "拼图", errors)

    test_graph: dict[str, list[str]] = {}
    for tid, test in tmap.items():
        require_fields(
            test,
            ["name", "status", "cost_tier", "objective", "axes", "partitions", "controls", "metrics", "pass_rule", "fail_rule", "outputs", "hypothesis_ids", "puzzle_ids", "prerequisites"],
            f"test {tid}",
            errors,
        )
        if test.get("status") not in VALID["test_status"]:
            errors.append(f"test {tid} 状态非法: {test.get('status')}")
        check_refs(test.get("hypothesis_ids", []), hypothesis_ids, f"test {tid}.hypothesis_ids", errors)
        check_refs(test.get("puzzle_ids", []), puzzle_ids, f"test {tid}.puzzle_ids", errors)
        prerequisites = test.get("prerequisites", [])
        test_graph[tid] = prerequisites
        check_refs(prerequisites, test_ids, f"test {tid}.prerequisites", errors)
    check_dag(test_graph, "测试", errors)

    for eid, record in emap.items():
        require_fields(
            record,
            ["phase", "grade", "closure_level", "polarity", "title", "claim", "scope", "authorizes", "forbids", "puzzle_ids", "hypothesis_ids", "source_paths"],
            f"evidence {eid}",
            errors,
        )
        if not isinstance(record.get("closure_level"), int) or record.get("closure_level") not in range(9):
            errors.append(f"evidence {eid} closure_level 非法: {record.get('closure_level')}")
        if not str(record.get("grade", "")).startswith(("E0", "E1", "E2", "E3")):
            errors.append(f"evidence {eid} grade 非法: {record.get('grade')}")
        check_refs(record.get("puzzle_ids", []), puzzle_ids, f"evidence {eid}.puzzle_ids", errors)
        check_refs(record.get("hypothesis_ids", []), hypothesis_ids, f"evidence {eid}.hypothesis_ids", errors)
        for source in record.get("source_paths", []):
            if not (ROOT / source).resolve().exists():
                errors.append(f"evidence {eid} 来源路径不存在: {source}")

    for phase_no, phase in phmap.items():
        require_fields(phase, ["date", "phase_type", "status", "campaign_id", "evidence_refs", "verdict", "auto_continue"], f"phase {phase_no}", errors)
        if phase.get("status") not in VALID["phase_status"]:
            errors.append(f"phase {phase_no} 状态非法: {phase.get('status')}")
        if phase.get("campaign_id") != "LEGACY" and phase.get("campaign_id") not in campaign_ids:
            errors.append(f"phase {phase_no} campaign_id 不存在: {phase.get('campaign_id')}")
        check_refs(phase.get("evidence_refs", []), evidence_ids, f"phase {phase_no}.evidence_refs", errors)

    for did, decision in dmap.items():
        require_fields(decision, ["date", "status", "title", "basis", "decision", "authorizes", "forbids"], f"decision {did}", errors)
        if decision.get("status") not in VALID["decision_status"]:
            errors.append(f"decision {did} 状态非法: {decision.get('status')}")
        check_refs(decision.get("basis", []), evidence_ids, f"decision {did}.basis", errors)

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
    for phase in sorted(data["phases"], key=lambda x: x["phase"], reverse=True)[:8]:
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
            f"- 来源：{refs(record['source_paths'])}",
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
        "| 工作包 | 状态 | 测试 | 入口 | 出口 |",
        "|---|---|---|---|---|",
    ]
    for stage in campaign["stages"]:
        lines.append(
            f"| `{stage['id']}` {esc(stage['name'])} | {status_zh(stage['status'])} | "
            f"{esc(refs(stage['test_battery_ids']))} | {esc(short(stage['entry_gate'], 72))} | "
            f"{esc(short(stage['exit_gate'], 72))} |"
        )
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
    }


def command_validate(data: dict[str, Any]) -> int:
    errors = validate(data)
    if errors:
        print(f"校验失败：{len(errors)} 项", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    counts = {name: len(data[name]) for name in ("campaigns", "hypotheses", "puzzles", "tests", "evidence", "phases", "decisions")}
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("validate", help="校验全部机器账本")
    build = sub.add_parser("build", help="从账本生成 Markdown 看板")
    build.add_argument("--check-clean", action="store_true", help="只检查生成看板是否最新")
    sub.add_parser("summary", help="输出当前状态摘要")
    freeze = sub.add_parser("freeze", help="计算预注册工件 SHA256")
    freeze.add_argument("path", help="要冻结的文件")
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
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
