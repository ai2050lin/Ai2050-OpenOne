"""Current evidence-bounded research snapshot for visualization clients."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REGISTRY_NAMES = (
    "hypotheses",
    "puzzles",
    "tests",
    "evidence",
    "phases",
    "decisions",
    "sources",
    "objects",
    "constructs",
    "contracts",
    "runs",
    "artifacts",
    "corrections",
)


def _read_json(path: Path, fallback: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return fallback


def build_current_research_progress(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root)
    os_root = root / "research" / "ai2050_research_os"
    registry_root = os_root / "registry"
    registries = {
        name: _read_json(registry_root / f"{name}.json", []) for name in REGISTRY_NAMES
    }
    counts = {
        name: len(value) if isinstance(value, list) else 0
        for name, value in registries.items()
    }
    contract = next(
        (
            item
            for item in registries["contracts"]
            if item.get("id") == "EXP-C001-WP01-001"
        ),
        {},
    )

    phases = [
        {
            "title": "静态实体搜索到条件化过程",
            "phase": "901-1140",
            "status": "bounded",
            "summary": "固定方向和单点强版本被持续收紧；行为门、同路径相机和序列位置合同建立。",
        },
        {
            "title": "已知真值相机与外部效度",
            "phase": "1151-1201",
            "status": "adjudicated",
            "summary": "校准双路径、冗余、协同、响应商、形成事件和救援相机，并登记适用边界。",
        },
        {
            "title": "类型化功能对象与原子操作",
            "phase": "1202-1228",
            "status": "adjudicated",
            "summary": "行为、内部响应、操作和生成被拆成分别授权的研究对象。",
        },
        {
            "title": "程序可识别材料与七读出边界",
            "phase": "1229-1235",
            "status": "stopped_by_gate",
            "summary": "Phase1235 四类门为 (1,0,1,1)，异质总门为 0，未授权内部状态扫描。",
        },
        {
            "title": "C001-WP00 研究操作系统",
            "phase": "1236",
            "status": "completed",
            "summary": "关键证据、对象、构念、合同和权限进入机器可校验注册表。",
        },
        {
            "title": "EXP-C001-WP01-001 无模型预审计",
            "phase": "WP01",
            "status": "blocked" if not contract.get("run_ready") else "ready",
            "summary": "合同已预注册；当前只授权材料、反泄漏、环境与独立审计器冻结。",
        },
    ]

    systemic = {
        "research_baseline": "Phase1236/C001-WP00",
        "theory": [
            {"name": "局部条件化模体零假说", "status": "candidate"},
            {"name": "层间旋转子空间", "status": "candidate"},
            {"name": "稀疏因果联盟", "status": "candidate"},
            {"name": "固定方向编码（全局强版本）", "status": "bounded_rejected"},
        ],
        "engineering": [
            {"name": "C001-WP00 注册表与验证器", "status": "complete", "progress": 100},
            {"name": "WP01 冻结实验合同", "status": "preregistered", "progress": 100},
            {"name": "WP01 无模型预审计", "status": "pending", "progress": 0},
            {"name": "未来响应与因果闭合", "status": "locked", "progress": 0},
        ],
        "roadmap": [
            {
                "id": "evidence_system",
                "title": "证据与对象迁移",
                "status": "done",
                "desc": "关键证据子集、类型化对象和权限边界进入不可变注册体系。",
                "metrics": {"Evidence": counts["evidence"], "Objects": counts["objects"]},
            },
            {
                "id": "typed_behavior",
                "title": "类型化行为边界",
                "status": "done",
                "desc": "候选、格式、自然生成与停止缓存分别记账，不再由单一总准确率代替。",
                "metrics": {"Constructs": counts["constructs"], "K210 Gates": "1·0·1·1"},
            },
            {
                "id": "wp01",
                "title": "WP01 功能等价构念",
                "status": "locked" if not contract.get("run_ready") else "ready",
                "desc": "先完成无模型材料、泄漏、环境和独立审计冻结，再决定是否允许 Qwen3 行为运行。",
                "metrics": {"Contract": contract.get("status", "missing"), "Run Ready": bool(contract.get("run_ready"))},
            },
            {
                "id": "mechanism_closure",
                "title": "响应—干预—救援闭合",
                "status": "locked",
                "desc": "只有行为与构念双门通过后，才允许未来响应、必要性、错误供体、救援和中介实验。",
                "metrics": {"Runs": counts["runs"], "Artifacts": counts["artifacts"]},
            },
        ],
        "convergence_index": None,
        "convergence_status": "not_estimable",
        "registry_counts": counts,
        "experiment_contract": {
            "id": contract.get("id", "EXP-C001-WP01-001"),
            "status": contract.get("status", "missing"),
            "run_ready": bool(contract.get("run_ready")),
            "frozen_at": contract.get("frozen_at"),
        },
    }

    return {
        "status": "success",
        "research_baseline_version": "phase1236-c001-wp00-v1",
        "last_updated": 1786494915,
        "current_phase": 1236,
        "latest_engineering_phase": 1241,
        "phases": phases,
        "research_logs": [
            "Phase1235：七读出四类门为 (1,0,1,1)，总门失败，未授权 hidden。",
            "Phase1236：C001-WP00 校验通过；WP01 preregistered，run_ready=false。",
        ],
        "latest_test": {
            "task_type": "C001-WP00 registry validation",
            "status": "passed",
            "model_run": False,
            "scientific_evidence_added": False,
        },
        "systemic": systemic,
    }
