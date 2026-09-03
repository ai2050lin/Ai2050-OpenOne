"""Persistent research accumulation API for the visualization client.

The workspace deliberately stores metadata and artifact references in SQLite.
Large embedding and HiddenState tensors remain in immutable result artifacts and
are never copied into the database.  This keeps the client index fast while
preserving exact provenance for later 3D inspection.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "ai2050_research_os" / "generated" / "research_workspace.sqlite3"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def _json(value: Any) -> str:
    return json.dumps(value if value is not None else {}, ensure_ascii=False, sort_keys=True)


def _decode_json(value: str | None) -> Any:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return {"raw": value}


LANGUAGE_SEEDS = (
    ("token", "noun", "名词", "noun", "对象、实体与概念命名的词元集合"),
    ("token", "verb", "动词", "verb", "动作、变化与状态关系的词元集合"),
    ("token", "adverb", "副词", "adverb", "程度、时间、范围与语气修饰词元集合"),
    ("token", "preposition", "介词", "preposition", "空间、时间、工具与论元关系词元集合"),
    ("token", "punctuation", "标点符号", "punctuation", "边界、停顿、语气与结构标记集合"),
    ("construction", "argument_structure", "主谓宾构式", "subject-verb-object", "基本论元角色与顺序构式"),
    ("construction", "modifier", "修饰构式", "modifier-head", "定中、状中及程度修饰构式"),
    ("construction", "negation", "否定构式", "negation", "否定词、否定范围与焦点构式"),
    ("construction", "question", "疑问构式", "question", "一般疑问、特指疑问与反问构式"),
    ("construction", "conditional", "条件构式", "if-then", "条件、结果与反事实构式"),
)


CLOSURE_SEEDS = (
    ("gate_language_coverage", "语言对象覆盖", "open", "关键 token 类与构式均具有可复查样本和对照"),
    ("gate_field_capture", "HiddenState 场覆盖", "open", "词嵌入与逐层 HiddenState 完整参数均已保存并可定位"),
    ("gate_causal", "因果干预", "open", "观察关联通过预注册干预、负对照与副作用检查"),
    ("gate_replication", "跨样本与跨模型复现", "open", "结论通过留出样本和至少两个模型的独立复现"),
    ("gate_math_closure", "数学闭合", "open", "对象、关系、变换、预测与反例边界形成可计算闭环"),
)


CLAIM_SEEDS = (
    (
        "claim_contextual_fingerprint",
        "词元意义是条件响应指纹",
        "同一词元的研究对象不是固定向量，而是给定语境与操作后对未来层响应的条件指纹。",
        "hypothesis",
        "建立同词元跨语境、同语境替换词元的成对记录。",
    ),
    (
        "claim_field_trajectory",
        "语言操作对应场轨迹变化",
        "构式和语用操作可能通过逐层 HiddenState 轨迹而不是单一坐标发挥作用。",
        "hypothesis",
        "同时保存输入嵌入、每一层完整场与输出差分。",
    ),
    (
        "claim_shared_operator",
        "可迁移语言算子仍待识别",
        "跨词元、跨构式、跨模型可复用的基本语言算子尚未达到闭合证据。",
        "open",
        "先扩大基础模式族与干预拼图，不预设高级数学形式。",
    ),
)


OPERATION_SEEDS = (
    ("seed_operation_taxonomy", "taxonomy", "类型关系", "对象、类别与属性关系的单跳和多跳操作"),
    ("seed_operation_punctuation", "punctuation", "标点与边界", "边界、分段、引用和未来预测重置操作"),
    ("seed_operation_preposition", "preposition", "介词关系绑定", "空间、时间、方向和角色关系绑定操作"),
    ("seed_operation_negation", "negation", "否定与作用域", "真值、焦点和嵌套作用域变换"),
    ("seed_operation_translation", "translation", "跨语言运输", "语义保持与输出语言身份重编译"),
    ("seed_operation_style", "style", "风格变换", "内容保持条件下的输出分布变换"),
)


class LanguageObjectInput(BaseModel):
    object_type: Literal["token", "construction"] = "token"
    family: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=160)
    normalized_form: str = Field(default="", max_length=240)
    language: str = Field(default="zh", max_length=24)
    description: str = Field(default="", max_length=2000)
    status: Literal["planned", "collecting", "partial", "complete"] = "planned"
    evidence_level: Literal["E0", "E1", "E2", "E3", "E4"] = "E0"
    sample_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FieldRecordInput(BaseModel):
    language_object_id: str = Field(min_length=1, max_length=80)
    model_id: str = Field(min_length=1, max_length=120)
    model_revision: str = Field(default="", max_length=240)
    case_id: str = Field(default="", max_length=160)
    run_id: str = Field(default="", max_length=200)
    token_count: int = Field(default=0, ge=0)
    layer_count: int = Field(default=0, ge=0)
    hidden_size: int = Field(default=0, ge=0)
    embedding_parameter_count: int = Field(default=0, ge=0)
    hiddenstate_parameter_count: int = Field(default=0, ge=0)
    embedding_artifact: str = Field(default="", max_length=1200)
    hiddenstate_artifact: str = Field(default="", max_length=1200)
    coverage_scope: Literal["metadata_only", "top_k", "full"] = "metadata_only"
    status: Literal["queued", "captured", "validated", "rejected"] = "queued"
    evidence_level: Literal["E0", "E1", "E2", "E3", "E4"] = "E0"
    metadata: dict[str, Any] = Field(default_factory=dict)


class TheoryClaimInput(BaseModel):
    title: str = Field(min_length=1, max_length=240)
    statement: str = Field(min_length=1, max_length=6000)
    status: Literal["open", "hypothesis", "supported", "challenged", "closed"] = "open"
    evidence_level: Literal["E0", "E1", "E2", "E3", "E4"] = "E0"
    supporting_count: int = Field(default=0, ge=0)
    contradicting_count: int = Field(default=0, ge=0)
    open_puzzle: str = Field(default="", max_length=3000)
    next_test: str = Field(default="", max_length=3000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClaimPatch(BaseModel):
    status: Literal["open", "hypothesis", "supported", "challenged", "closed"] | None = None
    evidence_level: Literal["E0", "E1", "E2", "E3", "E4"] | None = None
    supporting_count: int | None = Field(default=None, ge=0)
    contradicting_count: int | None = Field(default=None, ge=0)
    open_puzzle: str | None = Field(default=None, max_length=3000)
    next_test: str | None = Field(default=None, max_length=3000)


class ClosureGatePatch(BaseModel):
    status: Literal["open", "in_progress", "blocked", "passed"]
    evidence_count: int = Field(default=0, ge=0)
    blocking_reason: str = Field(default="", max_length=3000)


class LanguageNodeInput(BaseModel):
    node_type: Literal["form", "concept", "role", "context"]
    label: str = Field(min_length=1, max_length=240)
    normalized_form: str = Field(default="", max_length=400)
    language: str = Field(default="und", max_length=24)
    description: str = Field(default="", max_length=3000)
    status: Literal["defined", "collecting", "reviewed", "deprecated"] = "defined"
    evidence_level: Literal["E0", "E1", "E2", "E3", "E4"] = "E0"
    metadata: dict[str, Any] = Field(default_factory=dict)


class LanguageEdgeInput(BaseModel):
    source_node_id: str = Field(min_length=1, max_length=80)
    target_node_id: str = Field(min_length=1, max_length=80)
    edge_type: Literal["relation", "transform", "compose"] = "relation"
    relation: str = Field(min_length=1, max_length=160)
    condition: str = Field(default="", max_length=2000)
    status: Literal["defined", "reviewed", "challenged", "deprecated"] = "defined"
    evidence_level: Literal["E0", "E1", "E2", "E3", "E4"] = "E0"
    metadata: dict[str, Any] = Field(default_factory=dict)


class LanguageOperationInput(BaseModel):
    family_type: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=240)
    description: str = Field(default="", max_length=4000)
    language: str = Field(default="multi", max_length=24)
    invariants: list[str] = Field(default_factory=list)
    changed_factors: list[str] = Field(default_factory=list)
    context_conditions: list[str] = Field(default_factory=list)
    counterfactual_operations: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    next_evidence_gap: str = Field(default="", max_length=3000)
    behavior_status: Literal["untested", "qualified", "failed"] = "untested"
    evidence_level: Literal["E0", "E1", "E2", "E3", "E4"] = "E0"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConstructionInput(BaseModel):
    family: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=240)
    description: str = Field(default="", max_length=4000)
    language: str = Field(default="multi", max_length=24)
    typed_slots: list[dict[str, Any]] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    operation_ids: list[str] = Field(default_factory=list)
    surfaces: list[str] = Field(default_factory=list)
    status: Literal["defined", "collecting", "reviewed", "deprecated"] = "defined"
    evidence_level: Literal["E0", "E1", "E2", "E3", "E4"] = "E0"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchCaseInput(BaseModel):
    operation_id: str = Field(min_length=1, max_length=80)
    construction_id: str = Field(default="", max_length=80)
    label: str = Field(min_length=1, max_length=240)
    input_text: str = Field(min_length=1, max_length=12000)
    variant_text: str = Field(default="", max_length=12000)
    semantic_roles: dict[str, Any] = Field(default_factory=dict)
    invariants: list[str] = Field(default_factory=list)
    changed_factors: list[str] = Field(default_factory=list)
    split: Literal["train", "validation", "test", "lockbox"] = "test"
    behavior_status: Literal["untested", "qualified", "failed"] = "untested"
    metadata: dict[str, Any] = Field(default_factory=dict)


class PairAlignmentInput(BaseModel):
    operation_id: str = Field(min_length=1, max_length=80)
    baseline_case_id: str = Field(min_length=1, max_length=80)
    variant_case_id: str = Field(min_length=1, max_length=80)
    baseline_run_id: str = Field(default="", max_length=200)
    variant_run_id: str = Field(default="", max_length=200)
    token_alignment: dict[str, Any] = Field(default_factory=dict)
    role_alignment: dict[str, Any] = Field(default_factory=dict)
    artifact_path: str = Field(default="", max_length=1200)
    status: Literal["planned", "aligned", "validated", "rejected"] = "planned"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProbeResponseInput(BaseModel):
    operation_id: str = Field(min_length=1, max_length=80)
    field_record_id: str = Field(default="", max_length=80)
    run_id: str = Field(min_length=1, max_length=200)
    source_checkpoint: str = Field(min_length=1, max_length=120)
    target_checkpoint: str = Field(min_length=1, max_length=120)
    source_token: int = Field(default=0, ge=0)
    target_token: int = Field(default=0, ge=0)
    source_coordinate: int = Field(default=0, ge=0)
    target_coordinate: int = Field(default=0, ge=0)
    direction_id: str = Field(default="", max_length=240)
    dose: float = 0.0
    response_sign: int = Field(default=0, ge=-1, le=1)
    response_amplitude: float = 0.0
    output_effect: float = 0.0
    artifact_path: str = Field(default="", max_length=1200)
    status: Literal["captured", "validated", "rejected"] = "captured"
    evidence_level: Literal["E0", "E1", "E2", "E3", "E4"] = "E1"
    metadata: dict[str, Any] = Field(default_factory=dict)


class GearCandidateInput(BaseModel):
    operation_id: str = Field(min_length=1, max_length=80)
    label: str = Field(min_length=1, max_length=240)
    condition_domain: str = Field(default="", max_length=4000)
    source_nodes: list[dict[str, Any]] = Field(default_factory=list)
    target_nodes: list[dict[str, Any]] = Field(default_factory=list)
    sign_structure: str = Field(default="", max_length=3000)
    amplitude_model: str = Field(default="", max_length=3000)
    output_effect: str = Field(default="", max_length=3000)
    control_status: Literal["untested", "partial", "passed", "failed"] = "untested"
    causal_status: Literal["untested", "call_only", "delete_tested", "rescued", "failed"] = "untested"
    evidence_level: Literal["E0", "E1", "E2", "E3", "E4"] = "E0"
    metadata: dict[str, Any] = Field(default_factory=dict)


class InterventionInput(BaseModel):
    operation_id: str = Field(min_length=1, max_length=80)
    gear_candidate_id: str = Field(default="", max_length=80)
    run_id: str = Field(min_length=1, max_length=200)
    intervention_type: Literal["call", "delete", "rescue"]
    target: dict[str, Any] = Field(default_factory=dict)
    dose: float = 0.0
    behavior_effect: str = Field(default="", max_length=4000)
    output_effect: str = Field(default="", max_length=4000)
    side_effects: str = Field(default="", max_length=4000)
    artifact_path: str = Field(default="", max_length=1200)
    decision: Literal["pending", "accepted", "rejected", "inconclusive"] = "pending"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ClosureApplicationInput(BaseModel):
    gate_id: str = Field(min_length=1, max_length=120)
    requested_status: Literal["in_progress", "passed"] = "in_progress"
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=5000)
    requested_by: str = Field(default="human", max_length=120)


class ResearchWorkspaceStore:
    """Small transactional index over immutable research artifacts."""

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        with self._lock:
            if self._initialized:
                return
            with self._connection() as db:
                db.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS language_objects (
                        id TEXT PRIMARY KEY,
                        object_type TEXT NOT NULL CHECK (object_type IN ('token', 'construction')),
                        family TEXT NOT NULL,
                        label TEXT NOT NULL,
                        normalized_form TEXT NOT NULL,
                        language TEXT NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'planned',
                        evidence_level TEXT NOT NULL DEFAULT 'E0',
                        sample_count INTEGER NOT NULL DEFAULT 0,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(object_type, family, normalized_form, language)
                    );
                    CREATE TABLE IF NOT EXISTS field_records (
                        id TEXT PRIMARY KEY,
                        language_object_id TEXT NOT NULL REFERENCES language_objects(id),
                        model_id TEXT NOT NULL,
                        model_revision TEXT NOT NULL DEFAULT '',
                        case_id TEXT NOT NULL DEFAULT '',
                        run_id TEXT NOT NULL DEFAULT '',
                        token_count INTEGER NOT NULL DEFAULT 0,
                        layer_count INTEGER NOT NULL DEFAULT 0,
                        hidden_size INTEGER NOT NULL DEFAULT 0,
                        embedding_parameter_count INTEGER NOT NULL DEFAULT 0,
                        hiddenstate_parameter_count INTEGER NOT NULL DEFAULT 0,
                        embedding_artifact TEXT NOT NULL DEFAULT '',
                        hiddenstate_artifact TEXT NOT NULL DEFAULT '',
                        coverage_scope TEXT NOT NULL DEFAULT 'metadata_only',
                        status TEXT NOT NULL DEFAULT 'queued',
                        evidence_level TEXT NOT NULL DEFAULT 'E0',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_field_object ON field_records(language_object_id);
                    CREATE INDEX IF NOT EXISTS idx_field_model ON field_records(model_id);
                    CREATE TABLE IF NOT EXISTS theory_claims (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        statement TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'open',
                        evidence_level TEXT NOT NULL DEFAULT 'E0',
                        supporting_count INTEGER NOT NULL DEFAULT 0,
                        contradicting_count INTEGER NOT NULL DEFAULT 0,
                        open_puzzle TEXT NOT NULL DEFAULT '',
                        next_test TEXT NOT NULL DEFAULT '',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS closure_gates (
                        id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'open',
                        description TEXT NOT NULL DEFAULT '',
                        evidence_count INTEGER NOT NULL DEFAULT 0,
                        blocking_reason TEXT NOT NULL DEFAULT '',
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS loop_runs (
                        run_id TEXT PRIMARY KEY,
                        objective TEXT NOT NULL,
                        loop_number INTEGER NOT NULL,
                        mode TEXT NOT NULL,
                        status TEXT NOT NULL,
                        decision TEXT NOT NULL,
                        summary TEXT NOT NULL DEFAULT '',
                        master_model TEXT NOT NULL DEFAULT '',
                        analyst_models_json TEXT NOT NULL DEFAULT '[]',
                        artifact_audit_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS language_nodes (
                        id TEXT PRIMARY KEY,
                        node_type TEXT NOT NULL,
                        label TEXT NOT NULL,
                        normalized_form TEXT NOT NULL DEFAULT '',
                        language TEXT NOT NULL DEFAULT 'und',
                        description TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'defined',
                        evidence_level TEXT NOT NULL DEFAULT 'E0',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(node_type, normalized_form, language)
                    );
                    CREATE TABLE IF NOT EXISTS language_edges (
                        id TEXT PRIMARY KEY,
                        source_node_id TEXT NOT NULL REFERENCES language_nodes(id),
                        target_node_id TEXT NOT NULL REFERENCES language_nodes(id),
                        edge_type TEXT NOT NULL,
                        relation TEXT NOT NULL,
                        condition_text TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'defined',
                        evidence_level TEXT NOT NULL DEFAULT 'E0',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_language_edge_source ON language_edges(source_node_id);
                    CREATE INDEX IF NOT EXISTS idx_language_edge_target ON language_edges(target_node_id);
                    CREATE TABLE IF NOT EXISTS language_operations (
                        id TEXT PRIMARY KEY,
                        family_type TEXT NOT NULL,
                        label TEXT NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        language TEXT NOT NULL DEFAULT 'multi',
                        invariants_json TEXT NOT NULL DEFAULT '[]',
                        changed_factors_json TEXT NOT NULL DEFAULT '[]',
                        context_conditions_json TEXT NOT NULL DEFAULT '[]',
                        counterfactual_operations_json TEXT NOT NULL DEFAULT '[]',
                        expected_outputs_json TEXT NOT NULL DEFAULT '[]',
                        next_evidence_gap TEXT NOT NULL DEFAULT '',
                        behavior_status TEXT NOT NULL DEFAULT 'untested',
                        evidence_level TEXT NOT NULL DEFAULT 'E0',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(family_type, label, language)
                    );
                    CREATE TABLE IF NOT EXISTS constructions (
                        id TEXT PRIMARY KEY,
                        family TEXT NOT NULL,
                        label TEXT NOT NULL,
                        description TEXT NOT NULL DEFAULT '',
                        language TEXT NOT NULL DEFAULT 'multi',
                        typed_slots_json TEXT NOT NULL DEFAULT '[]',
                        constraints_json TEXT NOT NULL DEFAULT '[]',
                        operation_ids_json TEXT NOT NULL DEFAULT '[]',
                        surfaces_json TEXT NOT NULL DEFAULT '[]',
                        status TEXT NOT NULL DEFAULT 'defined',
                        evidence_level TEXT NOT NULL DEFAULT 'E0',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(family, label, language)
                    );
                    CREATE TABLE IF NOT EXISTS research_cases (
                        id TEXT PRIMARY KEY,
                        operation_id TEXT NOT NULL REFERENCES language_operations(id),
                        construction_id TEXT NOT NULL DEFAULT '',
                        label TEXT NOT NULL,
                        input_text TEXT NOT NULL,
                        variant_text TEXT NOT NULL DEFAULT '',
                        semantic_roles_json TEXT NOT NULL DEFAULT '{}',
                        invariants_json TEXT NOT NULL DEFAULT '[]',
                        changed_factors_json TEXT NOT NULL DEFAULT '[]',
                        split TEXT NOT NULL DEFAULT 'test',
                        behavior_status TEXT NOT NULL DEFAULT 'untested',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_research_case_operation ON research_cases(operation_id);
                    CREATE TABLE IF NOT EXISTS pair_alignments (
                        id TEXT PRIMARY KEY,
                        operation_id TEXT NOT NULL REFERENCES language_operations(id),
                        baseline_case_id TEXT NOT NULL REFERENCES research_cases(id),
                        variant_case_id TEXT NOT NULL REFERENCES research_cases(id),
                        baseline_run_id TEXT NOT NULL DEFAULT '',
                        variant_run_id TEXT NOT NULL DEFAULT '',
                        token_alignment_json TEXT NOT NULL DEFAULT '{}',
                        role_alignment_json TEXT NOT NULL DEFAULT '{}',
                        artifact_path TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'planned',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS probe_responses (
                        id TEXT PRIMARY KEY,
                        operation_id TEXT NOT NULL REFERENCES language_operations(id),
                        field_record_id TEXT NOT NULL DEFAULT '',
                        run_id TEXT NOT NULL,
                        source_checkpoint TEXT NOT NULL,
                        target_checkpoint TEXT NOT NULL,
                        source_token INTEGER NOT NULL DEFAULT 0,
                        target_token INTEGER NOT NULL DEFAULT 0,
                        source_coordinate INTEGER NOT NULL DEFAULT 0,
                        target_coordinate INTEGER NOT NULL DEFAULT 0,
                        direction_id TEXT NOT NULL DEFAULT '',
                        dose REAL NOT NULL DEFAULT 0,
                        response_sign INTEGER NOT NULL DEFAULT 0,
                        response_amplitude REAL NOT NULL DEFAULT 0,
                        output_effect REAL NOT NULL DEFAULT 0,
                        artifact_path TEXT NOT NULL DEFAULT '',
                        status TEXT NOT NULL DEFAULT 'captured',
                        evidence_level TEXT NOT NULL DEFAULT 'E1',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_probe_operation ON probe_responses(operation_id);
                    CREATE INDEX IF NOT EXISTS idx_probe_run ON probe_responses(run_id);
                    CREATE TABLE IF NOT EXISTS gear_candidates (
                        id TEXT PRIMARY KEY,
                        operation_id TEXT NOT NULL REFERENCES language_operations(id),
                        label TEXT NOT NULL,
                        condition_domain TEXT NOT NULL DEFAULT '',
                        source_nodes_json TEXT NOT NULL DEFAULT '[]',
                        target_nodes_json TEXT NOT NULL DEFAULT '[]',
                        sign_structure TEXT NOT NULL DEFAULT '',
                        amplitude_model TEXT NOT NULL DEFAULT '',
                        output_effect TEXT NOT NULL DEFAULT '',
                        control_status TEXT NOT NULL DEFAULT 'untested',
                        causal_status TEXT NOT NULL DEFAULT 'untested',
                        evidence_level TEXT NOT NULL DEFAULT 'E0',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS interventions (
                        id TEXT PRIMARY KEY,
                        operation_id TEXT NOT NULL REFERENCES language_operations(id),
                        gear_candidate_id TEXT NOT NULL DEFAULT '',
                        run_id TEXT NOT NULL,
                        intervention_type TEXT NOT NULL,
                        target_json TEXT NOT NULL DEFAULT '{}',
                        dose REAL NOT NULL DEFAULT 0,
                        behavior_effect TEXT NOT NULL DEFAULT '',
                        output_effect TEXT NOT NULL DEFAULT '',
                        side_effects TEXT NOT NULL DEFAULT '',
                        artifact_path TEXT NOT NULL DEFAULT '',
                        decision TEXT NOT NULL DEFAULT 'pending',
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS closure_applications (
                        id TEXT PRIMARY KEY,
                        gate_id TEXT NOT NULL REFERENCES closure_gates(id),
                        requested_status TEXT NOT NULL,
                        evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                        rationale TEXT NOT NULL,
                        requested_by TEXT NOT NULL DEFAULT 'human',
                        review_status TEXT NOT NULL DEFAULT 'pending',
                        review_note TEXT NOT NULL DEFAULT '',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    CREATE TABLE IF NOT EXISTS workspace_events (
                        sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                        event_type TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                    """
                )
                self._seed(db)
            self._initialized = True

    def _seed(self, db: sqlite3.Connection) -> None:
        now = _utc_now()
        for index, (object_type, family, label, normalized, description) in enumerate(LANGUAGE_SEEDS, 1):
            db.execute(
                """
                INSERT OR IGNORE INTO language_objects
                (id, object_type, family, label, normalized_form, language, description,
                 status, evidence_level, sample_count, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, 'zh', ?, 'planned', 'E0', 0, ?, ?, ?)
                """,
                (f"seed_language_{index:02d}", object_type, family, label, normalized, description,
                 _json({"seed_definition": True}), now, now),
            )
        for gate_id, title, status, description in CLOSURE_SEEDS:
            db.execute(
                """INSERT OR IGNORE INTO closure_gates
                (id, title, status, description, evidence_count, blocking_reason, updated_at)
                VALUES (?, ?, ?, ?, 0, '', ?)""",
                (gate_id, title, status, description, now),
            )
        for claim_id, title, statement, status, next_test in CLAIM_SEEDS:
            db.execute(
                """INSERT OR IGNORE INTO theory_claims
                (id, title, statement, status, evidence_level, supporting_count,
                 contradicting_count, open_puzzle, next_test, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'E0', 0, 0, '', ?, ?, ?, ?)""",
                (claim_id, title, statement, status, next_test,
                 _json({"seed_hypothesis": True}), now, now),
            )
        for operation_id, family_type, label, description in OPERATION_SEEDS:
            db.execute(
                """INSERT OR IGNORE INTO language_operations
                (id, family_type, label, description, language, invariants_json,
                 changed_factors_json, context_conditions_json, counterfactual_operations_json,
                 expected_outputs_json, next_evidence_gap, behavior_status, evidence_level,
                 metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, 'multi', '[]', '[]', '[]', '[]', '[]', ?, 'untested', 'E0', ?, ?, ?)""",
                (operation_id, family_type, label, description,
                 f"为“{label}”登记不变量、变化量、行为资格与反事实控制。",
                 _json({"seed_definition": True, "not_experimental_evidence": True}), now, now),
            )

    @staticmethod
    def _row(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for key in list(result):
            if key.endswith("_json"):
                result[key[:-5]] = _decode_json(result.pop(key))
        if "condition_text" in result:
            result["condition"] = result.pop("condition_text")
        return result

    @staticmethod
    def _event(db: sqlite3.Connection, event_type: str, entity_type: str, entity_id: str, payload: Any) -> None:
        db.execute(
            "INSERT INTO workspace_events(event_type, entity_type, entity_id, payload_json, created_at) VALUES (?, ?, ?, ?, ?)",
            (event_type, entity_type, entity_id, _json(payload), _utc_now()),
        )

    def snapshot(self) -> dict[str, Any]:
        self.initialize()
        with self._connection() as db:
            language_objects = [self._row(row) for row in db.execute(
                "SELECT * FROM language_objects ORDER BY object_type, family, label"
            )]
            field_records = [self._row(row) for row in db.execute(
                """SELECT f.*, l.label AS language_object_label, l.family AS language_family,
                l.object_type AS language_object_type FROM field_records f
                JOIN language_objects l ON l.id = f.language_object_id
                ORDER BY f.created_at DESC LIMIT 500"""
            )]
            claims = [self._row(row) for row in db.execute(
                "SELECT * FROM theory_claims ORDER BY updated_at DESC"
            )]
            closure_gates = [dict(row) for row in db.execute(
                "SELECT * FROM closure_gates ORDER BY rowid"
            )]
            language_nodes = [self._row(row) for row in db.execute(
                "SELECT * FROM language_nodes ORDER BY node_type, label"
            )]
            language_edges = [self._row(row) for row in db.execute(
                "SELECT * FROM language_edges ORDER BY created_at DESC LIMIT 1000"
            )]
            operations = [self._row(row) for row in db.execute(
                "SELECT * FROM language_operations ORDER BY family_type, label"
            )]
            constructions = [self._row(row) for row in db.execute(
                "SELECT * FROM constructions ORDER BY family, label"
            )]
            cases = [self._row(row) for row in db.execute(
                "SELECT * FROM research_cases ORDER BY created_at DESC LIMIT 1000"
            )]
            pair_alignments = [self._row(row) for row in db.execute(
                "SELECT * FROM pair_alignments ORDER BY created_at DESC LIMIT 1000"
            )]
            probe_responses = [self._row(row) for row in db.execute(
                "SELECT * FROM probe_responses ORDER BY created_at DESC LIMIT 1000"
            )]
            gear_candidates = [self._row(row) for row in db.execute(
                "SELECT * FROM gear_candidates ORDER BY updated_at DESC LIMIT 1000"
            )]
            interventions = [self._row(row) for row in db.execute(
                "SELECT * FROM interventions ORDER BY created_at DESC LIMIT 1000"
            )]
            closure_applications = [self._row(row) for row in db.execute(
                "SELECT * FROM closure_applications ORDER BY created_at DESC LIMIT 500"
            )]
            loop_runs = []
            for row in db.execute("SELECT * FROM loop_runs ORDER BY updated_at DESC LIMIT 200"):
                item = dict(row)
                item["analyst_models"] = _decode_json(item.pop("analyst_models_json"))
                item["artifact_audit"] = _decode_json(item.pop("artifact_audit_json"))
                loop_runs.append(item)
            families = [dict(row) for row in db.execute(
                """SELECT family, object_type, COUNT(*) AS object_count,
                SUM(sample_count) AS sample_count,
                SUM(CASE WHEN status = 'complete' THEN 1 ELSE 0 END) AS complete_count
                FROM language_objects GROUP BY family, object_type ORDER BY object_type, family"""
            )]
            overview_row = db.execute(
                """SELECT
                (SELECT COUNT(*) FROM language_objects) AS language_object_count,
                (SELECT COALESCE(SUM(sample_count), 0) FROM language_objects) AS language_sample_count,
                (SELECT COUNT(*) FROM field_records) AS field_record_count,
                (SELECT COUNT(*) FROM field_records WHERE coverage_scope = 'full') AS full_field_count,
                (SELECT COUNT(*) FROM theory_claims) AS claim_count,
                (SELECT COUNT(*) FROM theory_claims WHERE status IN ('open', 'hypothesis', 'challenged')) AS open_claim_count,
                (SELECT COUNT(*) FROM closure_gates WHERE status = 'passed') AS passed_gate_count,
                (SELECT COUNT(*) FROM closure_gates) AS gate_count,
                (SELECT COUNT(*) FROM language_nodes) AS language_node_count,
                (SELECT COUNT(*) FROM language_edges) AS language_edge_count,
                (SELECT COUNT(*) FROM language_operations) AS operation_count,
                (SELECT COUNT(*) FROM constructions) AS construction_count,
                (SELECT COUNT(*) FROM research_cases) AS case_count,
                (SELECT COUNT(*) FROM pair_alignments) AS pair_count,
                (SELECT COUNT(*) FROM probe_responses) AS probe_response_count,
                (SELECT COUNT(*) FROM gear_candidates) AS gear_candidate_count,
                (SELECT COUNT(*) FROM interventions) AS intervention_count,
                (SELECT COUNT(*) FROM closure_applications WHERE review_status = 'pending') AS pending_closure_application_count,
                (SELECT COUNT(*) FROM loop_runs) AS loop_run_count,
                (SELECT MAX(created_at) FROM workspace_events) AS last_write_at
                """
            ).fetchone()
        return {
            "schema_version": "research_workspace.v2",
            "overview": dict(overview_row),
            "families": families,
            "language_objects": language_objects,
            "language_nodes": language_nodes,
            "language_edges": language_edges,
            "operations": operations,
            "constructions": constructions,
            "cases": cases,
            "pair_alignments": pair_alignments,
            "field_records": field_records,
            "probe_responses": probe_responses,
            "gear_candidates": gear_candidates,
            "interventions": interventions,
            "claims": claims,
            "closure_gates": closure_gates,
            "closure_applications": closure_applications,
            "loop_runs": loop_runs,
        }

    def create_language_node(self, payload: LanguageNodeInput) -> dict[str, Any]:
        self.initialize()
        record = payload.model_dump()
        record_id = _new_id("node")
        now = _utc_now()
        normalized = record["normalized_form"].strip() or record["label"].strip()
        try:
            with self._lock, self._connection() as db:
                db.execute(
                    """INSERT INTO language_nodes
                    (id, node_type, label, normalized_form, language, description, status,
                     evidence_level, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (record_id, record["node_type"], record["label"].strip(), normalized,
                     record["language"].strip(), record["description"].strip(), record["status"],
                     record["evidence_level"], _json(record["metadata"]), now, now),
                )
                self._event(db, "created", "language_node", record_id, record)
                row = db.execute("SELECT * FROM language_nodes WHERE id = ?", (record_id,)).fetchone()
        except sqlite3.IntegrityError as exc:
            raise ValueError("相同类型、标准形式和语言的节点已经存在") from exc
        return self._row(row)

    def create_language_edge(self, payload: LanguageEdgeInput) -> dict[str, Any]:
        self.initialize()
        record = payload.model_dump()
        record_id = _new_id("edge")
        now = _utc_now()
        with self._lock, self._connection() as db:
            node_ids = {record["source_node_id"], record["target_node_id"]}
            found = db.execute(
                f"SELECT COUNT(*) FROM language_nodes WHERE id IN ({','.join('?' for _ in node_ids)})",
                tuple(node_ids),
            ).fetchone()[0]
            if found != len(node_ids):
                raise KeyError("language_node")
            db.execute(
                """INSERT INTO language_edges
                (id, source_node_id, target_node_id, edge_type, relation, condition_text,
                 status, evidence_level, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record_id, record["source_node_id"], record["target_node_id"], record["edge_type"],
                 record["relation"].strip(), record["condition"].strip(), record["status"],
                 record["evidence_level"], _json(record["metadata"]), now, now),
            )
            self._event(db, "created", "language_edge", record_id, record)
            row = db.execute("SELECT * FROM language_edges WHERE id = ?", (record_id,)).fetchone()
        return self._row(row)

    def create_operation(self, payload: LanguageOperationInput) -> dict[str, Any]:
        self.initialize()
        record = payload.model_dump()
        record_id = _new_id("operation")
        now = _utc_now()
        try:
            with self._lock, self._connection() as db:
                db.execute(
                    """INSERT INTO language_operations
                    (id, family_type, label, description, language, invariants_json,
                     changed_factors_json, context_conditions_json, counterfactual_operations_json,
                     expected_outputs_json, next_evidence_gap, behavior_status, evidence_level,
                     metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (record_id, record["family_type"].strip(), record["label"].strip(),
                     record["description"].strip(), record["language"].strip(), _json(record["invariants"]),
                     _json(record["changed_factors"]), _json(record["context_conditions"]),
                     _json(record["counterfactual_operations"]), _json(record["expected_outputs"]),
                     record["next_evidence_gap"].strip(), record["behavior_status"], record["evidence_level"],
                     _json(record["metadata"]), now, now),
                )
                self._event(db, "created", "language_operation", record_id, record)
                row = db.execute("SELECT * FROM language_operations WHERE id = ?", (record_id,)).fetchone()
        except sqlite3.IntegrityError as exc:
            raise ValueError("相同操作族、名称和语言的记录已经存在") from exc
        return self._row(row)

    def create_construction(self, payload: ConstructionInput) -> dict[str, Any]:
        self.initialize()
        record = payload.model_dump()
        record_id = _new_id("construction")
        now = _utc_now()
        try:
            with self._lock, self._connection() as db:
                for operation_id in record["operation_ids"]:
                    if not db.execute("SELECT 1 FROM language_operations WHERE id = ?", (operation_id,)).fetchone():
                        raise KeyError(operation_id)
                db.execute(
                    """INSERT INTO constructions
                    (id, family, label, description, language, typed_slots_json, constraints_json,
                     operation_ids_json, surfaces_json, status, evidence_level, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (record_id, record["family"].strip(), record["label"].strip(), record["description"].strip(),
                     record["language"].strip(), _json(record["typed_slots"]), _json(record["constraints"]),
                     _json(record["operation_ids"]), _json(record["surfaces"]), record["status"],
                     record["evidence_level"], _json(record["metadata"]), now, now),
                )
                self._event(db, "created", "construction", record_id, record)
                row = db.execute("SELECT * FROM constructions WHERE id = ?", (record_id,)).fetchone()
        except sqlite3.IntegrityError as exc:
            raise ValueError("相同构式族、名称和语言的记录已经存在") from exc
        return self._row(row)

    def create_case(self, payload: ResearchCaseInput) -> dict[str, Any]:
        self.initialize()
        record = payload.model_dump()
        record_id = _new_id("case")
        now = _utc_now()
        with self._lock, self._connection() as db:
            if not db.execute("SELECT 1 FROM language_operations WHERE id = ?", (record["operation_id"],)).fetchone():
                raise KeyError(record["operation_id"])
            if record["construction_id"] and not db.execute(
                "SELECT 1 FROM constructions WHERE id = ?", (record["construction_id"],)
            ).fetchone():
                raise KeyError(record["construction_id"])
            db.execute(
                """INSERT INTO research_cases
                (id, operation_id, construction_id, label, input_text, variant_text,
                 semantic_roles_json, invariants_json, changed_factors_json, split,
                 behavior_status, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record_id, record["operation_id"], record["construction_id"], record["label"].strip(),
                 record["input_text"].strip(), record["variant_text"].strip(), _json(record["semantic_roles"]),
                 _json(record["invariants"]), _json(record["changed_factors"]), record["split"],
                 record["behavior_status"], _json(record["metadata"]), now, now),
            )
            self._event(db, "created", "research_case", record_id, record)
            row = db.execute("SELECT * FROM research_cases WHERE id = ?", (record_id,)).fetchone()
        return self._row(row)

    def create_pair_alignment(self, payload: PairAlignmentInput) -> dict[str, Any]:
        self.initialize()
        record = payload.model_dump()
        record_id = _new_id("pair")
        now = _utc_now()
        with self._lock, self._connection() as db:
            if not db.execute("SELECT 1 FROM language_operations WHERE id = ?", (record["operation_id"],)).fetchone():
                raise KeyError(record["operation_id"])
            for case_id in (record["baseline_case_id"], record["variant_case_id"]):
                if not db.execute("SELECT 1 FROM research_cases WHERE id = ?", (case_id,)).fetchone():
                    raise KeyError(case_id)
            db.execute(
                """INSERT INTO pair_alignments
                (id, operation_id, baseline_case_id, variant_case_id, baseline_run_id, variant_run_id,
                 token_alignment_json, role_alignment_json, artifact_path, status, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record_id, record["operation_id"], record["baseline_case_id"], record["variant_case_id"],
                 record["baseline_run_id"].strip(), record["variant_run_id"].strip(),
                 _json(record["token_alignment"]), _json(record["role_alignment"]),
                 record["artifact_path"].strip(), record["status"], _json(record["metadata"]), now, now),
            )
            self._event(db, "created", "pair_alignment", record_id, record)
            row = db.execute("SELECT * FROM pair_alignments WHERE id = ?", (record_id,)).fetchone()
        return self._row(row)

    def create_probe_response(self, payload: ProbeResponseInput) -> dict[str, Any]:
        self.initialize()
        record = payload.model_dump()
        record_id = _new_id("probe")
        now = _utc_now()
        with self._lock, self._connection() as db:
            if not db.execute("SELECT 1 FROM language_operations WHERE id = ?", (record["operation_id"],)).fetchone():
                raise KeyError(record["operation_id"])
            if record["field_record_id"] and not db.execute(
                "SELECT 1 FROM field_records WHERE id = ?", (record["field_record_id"],)
            ).fetchone():
                raise KeyError(record["field_record_id"])
            db.execute(
                """INSERT INTO probe_responses
                (id, operation_id, field_record_id, run_id, source_checkpoint, target_checkpoint,
                 source_token, target_token, source_coordinate, target_coordinate, direction_id,
                 dose, response_sign, response_amplitude, output_effect, artifact_path,
                 status, evidence_level, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record_id, record["operation_id"], record["field_record_id"], record["run_id"].strip(),
                 record["source_checkpoint"].strip(), record["target_checkpoint"].strip(),
                 record["source_token"], record["target_token"], record["source_coordinate"],
                 record["target_coordinate"], record["direction_id"].strip(), record["dose"],
                 record["response_sign"], record["response_amplitude"], record["output_effect"],
                 record["artifact_path"].strip(), record["status"], record["evidence_level"],
                 _json(record["metadata"]), now, now),
            )
            self._event(db, "created", "probe_response", record_id, record)
            row = db.execute("SELECT * FROM probe_responses WHERE id = ?", (record_id,)).fetchone()
        return self._row(row)

    def create_gear_candidate(self, payload: GearCandidateInput) -> dict[str, Any]:
        self.initialize()
        record = payload.model_dump()
        record_id = _new_id("gear")
        now = _utc_now()
        with self._lock, self._connection() as db:
            if not db.execute("SELECT 1 FROM language_operations WHERE id = ?", (record["operation_id"],)).fetchone():
                raise KeyError(record["operation_id"])
            db.execute(
                """INSERT INTO gear_candidates
                (id, operation_id, label, condition_domain, source_nodes_json, target_nodes_json,
                 sign_structure, amplitude_model, output_effect, control_status, causal_status,
                 evidence_level, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record_id, record["operation_id"], record["label"].strip(),
                 record["condition_domain"].strip(), _json(record["source_nodes"]),
                 _json(record["target_nodes"]), record["sign_structure"].strip(),
                 record["amplitude_model"].strip(), record["output_effect"].strip(),
                 record["control_status"], record["causal_status"], record["evidence_level"],
                 _json(record["metadata"]), now, now),
            )
            self._event(db, "created", "gear_candidate", record_id, record)
            row = db.execute("SELECT * FROM gear_candidates WHERE id = ?", (record_id,)).fetchone()
        return self._row(row)

    def create_intervention(self, payload: InterventionInput) -> dict[str, Any]:
        self.initialize()
        record = payload.model_dump()
        record_id = _new_id("intervention")
        now = _utc_now()
        with self._lock, self._connection() as db:
            if not db.execute("SELECT 1 FROM language_operations WHERE id = ?", (record["operation_id"],)).fetchone():
                raise KeyError(record["operation_id"])
            if record["gear_candidate_id"] and not db.execute(
                "SELECT 1 FROM gear_candidates WHERE id = ?", (record["gear_candidate_id"],)
            ).fetchone():
                raise KeyError(record["gear_candidate_id"])
            db.execute(
                """INSERT INTO interventions
                (id, operation_id, gear_candidate_id, run_id, intervention_type, target_json, dose,
                 behavior_effect, output_effect, side_effects, artifact_path, decision,
                 metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record_id, record["operation_id"], record["gear_candidate_id"], record["run_id"].strip(),
                 record["intervention_type"], _json(record["target"]), record["dose"],
                 record["behavior_effect"].strip(), record["output_effect"].strip(),
                 record["side_effects"].strip(), record["artifact_path"].strip(), record["decision"],
                 _json(record["metadata"]), now, now),
            )
            self._event(db, "created", "intervention", record_id, record)
            row = db.execute("SELECT * FROM interventions WHERE id = ?", (record_id,)).fetchone()
        return self._row(row)

    def create_closure_application(self, payload: ClosureApplicationInput) -> dict[str, Any]:
        """Record a review request without changing the underlying closure gate."""
        self.initialize()
        record = payload.model_dump()
        record_id = _new_id("closure_request")
        now = _utc_now()
        with self._lock, self._connection() as db:
            if not db.execute("SELECT 1 FROM closure_gates WHERE id = ?", (record["gate_id"],)).fetchone():
                raise KeyError(record["gate_id"])
            db.execute(
                """INSERT INTO closure_applications
                (id, gate_id, requested_status, evidence_ids_json, rationale, requested_by,
                 review_status, review_note, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', '', ?, ?)""",
                (record_id, record["gate_id"], record["requested_status"],
                 _json(record["evidence_ids"]), record["rationale"].strip(),
                 record["requested_by"].strip(), now, now),
            )
            self._event(db, "submitted", "closure_application", record_id, record)
            row = db.execute("SELECT * FROM closure_applications WHERE id = ?", (record_id,)).fetchone()
        return self._row(row)

    def create_language_object(self, payload: LanguageObjectInput) -> dict[str, Any]:
        self.initialize()
        record = payload.model_dump()
        record_id = _new_id("lang")
        now = _utc_now()
        normalized = record["normalized_form"].strip() or record["label"].strip()
        try:
            with self._lock, self._connection() as db:
                db.execute(
                    """INSERT INTO language_objects
                    (id, object_type, family, label, normalized_form, language, description,
                     status, evidence_level, sample_count, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (record_id, record["object_type"], record["family"].strip(), record["label"].strip(),
                     normalized, record["language"].strip(), record["description"].strip(), record["status"],
                     record["evidence_level"], record["sample_count"], _json(record["metadata"]), now, now),
                )
                self._event(db, "created", "language_object", record_id, record)
                row = db.execute("SELECT * FROM language_objects WHERE id = ?", (record_id,)).fetchone()
        except sqlite3.IntegrityError as exc:
            raise ValueError("相同类型、模式族、标准形式和语言的记录已经存在") from exc
        return self._row(row)

    def create_field_record(self, payload: FieldRecordInput) -> dict[str, Any]:
        self.initialize()
        record = payload.model_dump()
        record_id = _new_id("field")
        now = _utc_now()
        with self._lock, self._connection() as db:
            exists = db.execute("SELECT 1 FROM language_objects WHERE id = ?", (record["language_object_id"],)).fetchone()
            if not exists:
                raise KeyError(record["language_object_id"])
            db.execute(
                """INSERT INTO field_records
                (id, language_object_id, model_id, model_revision, case_id, run_id, token_count,
                 layer_count, hidden_size, embedding_parameter_count, hiddenstate_parameter_count,
                 embedding_artifact, hiddenstate_artifact, coverage_scope, status, evidence_level,
                 metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record_id, record["language_object_id"], record["model_id"].strip(), record["model_revision"].strip(),
                 record["case_id"].strip(), record["run_id"].strip(), record["token_count"], record["layer_count"],
                 record["hidden_size"], record["embedding_parameter_count"], record["hiddenstate_parameter_count"],
                 record["embedding_artifact"].strip(), record["hiddenstate_artifact"].strip(), record["coverage_scope"],
                 record["status"], record["evidence_level"], _json(record["metadata"]), now, now),
            )
            self._event(db, "created", "field_record", record_id, record)
            row = db.execute("SELECT * FROM field_records WHERE id = ?", (record_id,)).fetchone()
        return self._row(row)

    def create_claim(self, payload: TheoryClaimInput) -> dict[str, Any]:
        self.initialize()
        record = payload.model_dump()
        record_id = _new_id("claim")
        now = _utc_now()
        with self._lock, self._connection() as db:
            db.execute(
                """INSERT INTO theory_claims
                (id, title, statement, status, evidence_level, supporting_count, contradicting_count,
                 open_puzzle, next_test, metadata_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (record_id, record["title"].strip(), record["statement"].strip(), record["status"],
                 record["evidence_level"], record["supporting_count"], record["contradicting_count"],
                 record["open_puzzle"].strip(), record["next_test"].strip(), _json(record["metadata"]), now, now),
            )
            self._event(db, "created", "theory_claim", record_id, record)
            row = db.execute("SELECT * FROM theory_claims WHERE id = ?", (record_id,)).fetchone()
        return self._row(row)

    def update_claim(self, claim_id: str, payload: ClaimPatch) -> dict[str, Any]:
        self.initialize()
        changes = {key: value for key, value in payload.model_dump().items() if value is not None}
        if not changes:
            raise ValueError("至少提供一个需要更新的字段")
        changes["updated_at"] = _utc_now()
        with self._lock, self._connection() as db:
            if not db.execute("SELECT 1 FROM theory_claims WHERE id = ?", (claim_id,)).fetchone():
                raise KeyError(claim_id)
            assignments = ", ".join(f"{key} = ?" for key in changes)
            db.execute(f"UPDATE theory_claims SET {assignments} WHERE id = ?", (*changes.values(), claim_id))
            self._event(db, "updated", "theory_claim", claim_id, changes)
            row = db.execute("SELECT * FROM theory_claims WHERE id = ?", (claim_id,)).fetchone()
        return self._row(row)

    def update_closure_gate(self, gate_id: str, payload: ClosureGatePatch) -> dict[str, Any]:
        self.initialize()
        changes = payload.model_dump()
        changes["updated_at"] = _utc_now()
        with self._lock, self._connection() as db:
            if not db.execute("SELECT 1 FROM closure_gates WHERE id = ?", (gate_id,)).fetchone():
                raise KeyError(gate_id)
            db.execute(
                "UPDATE closure_gates SET status = ?, evidence_count = ?, blocking_reason = ?, updated_at = ? WHERE id = ?",
                (changes["status"], changes["evidence_count"], changes["blocking_reason"], changes["updated_at"], gate_id),
            )
            self._event(db, "updated", "closure_gate", gate_id, changes)
            row = db.execute("SELECT * FROM closure_gates WHERE id = ?", (gate_id,)).fetchone()
        return dict(row)

    def record_loop_result(
        self,
        *,
        run_id: str,
        objective: str,
        loop_number: int,
        mode: str,
        status: str,
        decision: str,
        summary: str,
        master_model: str,
        analyst_models: list[str],
        artifact_audit: dict[str, Any],
    ) -> dict[str, Any]:
        """Transactionally upsert one completed Loop Engineering result."""
        self.initialize()
        now = _utc_now()
        payload = {
            "run_id": run_id,
            "objective": objective,
            "loop_number": int(loop_number),
            "mode": mode,
            "status": status,
            "decision": decision,
            "summary": summary,
            "master_model": master_model,
            "analyst_models": analyst_models,
            "artifact_audit": artifact_audit,
        }
        with self._lock, self._connection() as db:
            db.execute(
                """INSERT INTO loop_runs
                (run_id, objective, loop_number, mode, status, decision, summary, master_model,
                 analyst_models_json, artifact_audit_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    objective = excluded.objective,
                    loop_number = excluded.loop_number,
                    mode = excluded.mode,
                    status = excluded.status,
                    decision = excluded.decision,
                    summary = excluded.summary,
                    master_model = excluded.master_model,
                    analyst_models_json = excluded.analyst_models_json,
                    artifact_audit_json = excluded.artifact_audit_json,
                    updated_at = excluded.updated_at""",
                (run_id, objective, int(loop_number), mode, status, decision, summary, master_model,
                 _json(analyst_models), _json(artifact_audit), now, now),
            )
            self._event(db, "upserted", "loop_run", run_id, payload)
            row = db.execute("SELECT * FROM loop_runs WHERE run_id = ?", (run_id,)).fetchone()
        result = dict(row)
        result["analyst_models"] = _decode_json(result.pop("analyst_models_json"))
        result["artifact_audit"] = _decode_json(result.pop("artifact_audit_json"))
        return result


_default_store: ResearchWorkspaceStore | None = None


def get_research_workspace_store() -> ResearchWorkspaceStore:
    global _default_store
    if _default_store is None:
        _default_store = ResearchWorkspaceStore()
    return _default_store


router = APIRouter(prefix="/api/research-workspace", tags=["Research Workspace"])


@router.get("/health")
def research_workspace_health(store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    snapshot = store.snapshot()
    return {"status": "ok", "schema_version": snapshot["schema_version"]}


@router.get("/snapshot")
def get_research_workspace_snapshot(store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    return store.snapshot()


@router.post("/language-nodes", status_code=201)
def create_language_node(payload: LanguageNodeInput, store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    try:
        return store.create_language_node(payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/language-edges", status_code=201)
def create_language_edge(payload: LanguageEdgeInput, store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    try:
        return store.create_language_edge(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="语言图谱节点不存在") from exc


@router.post("/operations", status_code=201)
def create_language_operation(payload: LanguageOperationInput, store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    try:
        return store.create_operation(payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/constructions", status_code=201)
def create_construction(payload: ConstructionInput, store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    try:
        return store.create_construction(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="关联的语言操作不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/cases", status_code=201)
def create_research_case(payload: ResearchCaseInput, store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    try:
        return store.create_case(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="关联的语言操作或构式不存在") from exc


@router.post("/pair-alignments", status_code=201)
def create_pair_alignment(payload: PairAlignmentInput, store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    try:
        return store.create_pair_alignment(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="关联的语言操作或 Case 不存在") from exc


@router.post("/probe-responses", status_code=201)
def create_probe_response(payload: ProbeResponseInput, store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    try:
        return store.create_probe_response(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="关联的语言操作或场记录不存在") from exc


@router.post("/gear-candidates", status_code=201)
def create_gear_candidate(payload: GearCandidateInput, store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    try:
        return store.create_gear_candidate(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="关联的语言操作不存在") from exc


@router.post("/interventions", status_code=201)
def create_intervention(payload: InterventionInput, store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    try:
        return store.create_intervention(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="关联的语言操作或齿轮候选不存在") from exc


@router.post("/closure-applications", status_code=201)
def create_closure_application(payload: ClosureApplicationInput, store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    try:
        return store.create_closure_application(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="闭合门不存在") from exc


@router.post("/language-objects", status_code=201)
def create_language_object(payload: LanguageObjectInput, store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    try:
        return store.create_language_object(payload)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/field-records", status_code=201)
def create_field_record(payload: FieldRecordInput, store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    try:
        return store.create_field_record(payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="语言对象不存在") from exc


@router.post("/claims", status_code=201)
def create_theory_claim(payload: TheoryClaimInput, store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    return store.create_claim(payload)


@router.patch("/claims/{claim_id}")
def update_theory_claim(claim_id: str, payload: ClaimPatch, store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    try:
        return store.update_claim(claim_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="理论主张不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.patch("/closure-gates/{gate_id}")
def update_closure_gate(gate_id: str, payload: ClosureGatePatch, store: ResearchWorkspaceStore = Depends(get_research_workspace_store)):
    try:
        return store.update_closure_gate(gate_id, payload)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="闭合门不存在") from exc
