"""
AI R&D Service — 自动深度神经网络逆向工程研究服务

核心功能:
1. 管理AI主模型和分析师模型的配置和提示词
2. 驱动5阶段自动研究循环: 分析→规划→生成→执行→总结
3. SSE实时推送研究事件到前端
4. 安全执行AI生成的代码
"""
import asyncio
import glob
import json
import os
import re
import sys
import time
import traceback
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from server.research_orchestrator import (
    ROLE_DEFINITIONS,
    artifact_audit,
    create_research_run,
    execute_research_code,
    list_research_runs,
    load_evidence_context,
    terminate_active_process,
)
from server.research_orchestrator.runtime import ResearchRun
from server.research_workspace_service import get_research_workspace_store


WORKFLOW_GATES = (
    {"id": "gap", "label": "证据缺口", "internal_steps": ("analyze",)},
    {"id": "contract", "label": "实验契约", "internal_steps": ("plan",)},
    {"id": "execute", "label": "串行执行", "internal_steps": ("generate", "execute")},
    {"id": "review", "label": "独立复核", "internal_steps": ("summarize",)},
    {"id": "writeback", "label": "证据回写", "internal_steps": ()},
)


def _workflow_gate(phase: Optional[str], status: str = "idle") -> str:
    for gate in WORKFLOW_GATES:
        if phase in gate["internal_steps"]:
            return str(gate["id"])
    return "writeback" if status == "stopped" else "gap"


def _default_project_agent_state() -> Dict[str, Any]:
    return {
        "schema_version": "project_research_agent.v1",
        "enabled": False,
        "status": "idle",
        "project_goal": "",
        "current_task_index": 0,
        "loops_completed": 0,
        "consecutive_inconclusive": 0,
        "last_decision": None,
        "stop_reason": "",
        "started_at": None,
        "completed_at": None,
        "config": {
            "max_loops": 3,
            "execution_mode": "auto",
            "stop_on_accepted": True,
            "stop_on_rejected": True,
            "max_consecutive_inconclusive": 3,
        },
        "plan": None,
    }


def _source_text(value: Any, fallback: str, limit: int = 500) -> str:
    text = str(value or "").strip()
    return (text or fallback)[:limit]


def build_project_agent_plan(
    project_goal: str = "",
    max_tasks: int = 6,
    *,
    workspace_snapshot: Optional[Dict[str, Any]] = None,
    evidence_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a bounded, evidence-first project plan without invoking an AI model."""
    snapshot = workspace_snapshot or get_research_workspace_store().snapshot()
    evidence = evidence_context or load_evidence_context()
    task_limit = max(1, min(int(max_tasks), 12))
    tasks: List[Dict[str, Any]] = []

    def add_task(source_type: str, source_id: str, title: str, objective: str, completion_rule: str) -> None:
        normalized = _source_text(objective, title, 2000)
        if any(task["objective"] == normalized for task in tasks):
            return
        tasks.append({
            "id": f"task_{len(tasks) + 1:02d}",
            "source_type": source_type,
            "source_id": source_id,
            "title": _source_text(title, "未命名研发任务", 240),
            "objective": normalized,
            "completion_rule": _source_text(completion_rule, "完成工件审计和独立复核", 1000),
            "status": "pending",
            "decision": None,
            "run_id": None,
        })

    goal = project_goal.strip()
    if goal:
        add_task(
            "project_goal",
            "user_goal",
            "当前项目目标",
            goal,
            "形成可证伪实验合同，保存完整原始产物，并通过独立工件审计。",
        )

    for gap in list(evidence.get("open_gaps") or []):
        if len(tasks) >= task_limit:
            break
        if isinstance(gap, dict):
            gap_id = _source_text(gap.get("id") or gap.get("gap_id"), "evidence_gap")
            title = _source_text(gap.get("title") or gap.get("question"), gap_id)
            objective = _source_text(
                gap.get("next_test") or gap.get("description") or gap.get("question"),
                f"为证据缺口 {title} 设计并执行可证伪测试。",
                2000,
            )
        else:
            gap_id = "evidence_gap"
            title = _source_text(gap, "开放证据缺口")
            objective = f"为证据缺口“{title}”设计并执行可证伪测试。"
        add_task("evidence_gap", gap_id, title, objective, "缺口获得有效支持、反证或明确的 inconclusive 边界。")

    operation_priority = {"failed": 0, "untested": 1, "qualified": 2}
    operations = sorted(
        snapshot.get("operations", []),
        key=lambda operation: (
            operation_priority.get(str(operation.get("behavior_status")), 9),
            str(operation.get("updated_at", "")),
        ),
    )
    for operation in operations:
        if len(tasks) >= task_limit:
            break
        operation_id = _source_text(operation.get("id"), "language_operation")
        title = _source_text(operation.get("label"), operation_id)
        objective = _source_text(
            operation.get("next_evidence_gap"),
            f"为语言操作“{title}”建立不变量、变化量、成对 Case 与反事实控制。",
            2000,
        )
        add_task(
            "language_operation",
            operation_id,
            f"语言操作：{title}",
            objective,
            "形成冻结 Case/Pair、行为资格、完整工件和 accepted/rejected/inconclusive 裁决；不得自动闭合理论。",
        )

    claim_priority = {"challenged": 0, "hypothesis": 1, "open": 2, "supported": 3}
    claims = sorted(
        (claim for claim in snapshot.get("claims", []) if claim.get("status") != "closed"),
        key=lambda claim: (claim_priority.get(str(claim.get("status")), 9), str(claim.get("updated_at", ""))),
    )
    for claim in claims:
        if len(tasks) >= task_limit:
            break
        claim_id = _source_text(claim.get("id"), "claim")
        title = _source_text(claim.get("title"), claim_id)
        objective = _source_text(
            claim.get("next_test"),
            f"检验理论主张“{title}”，同时寻找反例和替代解释。",
            2000,
        )
        add_task("theory_claim", claim_id, title, objective, "更新支持/冲突证据，但不由 AI 自动提升理论等级。")

    for application in snapshot.get("closure_applications", []):
        if len(tasks) >= task_limit:
            break
        if application.get("review_status") != "pending":
            continue
        application_id = _source_text(application.get("id"), "closure_application")
        gate_id = _source_text(application.get("gate_id"), "closure_gate")
        rationale = _source_text(application.get("rationale"), "缺少闭合申请说明", 1200)
        add_task(
            "closure_application",
            application_id,
            f"审核闭合申请：{gate_id}",
            f"独立核验闭合申请及其证据引用：{rationale}",
            "只形成审核材料和缺口列表；AI 不得直接批准申请或修改闭合门状态。",
        )

    gate_priority = {"blocked": 0, "in_progress": 1, "open": 2}
    gates = sorted(
        (gate for gate in snapshot.get("closure_gates", []) if gate.get("status") != "passed"),
        key=lambda gate: gate_priority.get(str(gate.get("status")), 9),
    )
    for gate in gates:
        if len(tasks) >= task_limit:
            break
        gate_id = _source_text(gate.get("id"), "closure_gate")
        title = _source_text(gate.get("title"), gate_id)
        blocker = _source_text(gate.get("blocking_reason"), gate.get("description") or "缺少闭合证据", 1200)
        add_task(
            "closure_gate",
            gate_id,
            f"闭合门：{title}",
            f"围绕“{title}”补齐当前缺口：{blocker}",
            "满足预注册判据、完整产物、负对照和独立复核后才可标记通过。",
        )

    overview = snapshot.get("overview") or {}
    language_objects = int(overview.get("language_object_count") or 0)
    full_fields = int(overview.get("full_field_count") or 0)
    if len(tasks) < task_limit and full_fields < language_objects:
        add_task(
            "coverage_gap",
            "hiddenstate_full_field",
            "完整 HiddenState 场覆盖",
            f"补齐语言对象的完整场数据：当前 {language_objects} 个对象中只有 {full_fields} 份完整场记录。",
            "Embedding 与全部 Layer × Token × HiddenSize 均有可追溯产物，不以 Top-K 替代原始数据。",
        )

    if not tasks:
        add_task(
            "fallback",
            "independent_audit",
            "当前基线独立复核",
            "复核当前项目最近一次有效结果，定位一个可证伪的新证据缺口。",
            "输出明确的 accepted、rejected 或 inconclusive 裁决及下一测试。",
        )

    tasks = tasks[:task_limit]
    return {
        "schema_version": "project_research_plan.v1",
        "generated_at": datetime.now().astimezone().isoformat(),
        "project_goal": goal or "自动推进当前研究数据库中的最高优先级证据缺口",
        "source_summary": {
            "language_objects": language_objects,
            "language_operations": int(overview.get("operation_count") or 0),
            "probe_responses": int(overview.get("probe_response_count") or 0),
            "full_field_records": full_fields,
            "open_claims": int(overview.get("open_claim_count") or 0),
            "pending_closure_applications": int(overview.get("pending_closure_application_count") or 0),
            "passed_closure_gates": int(overview.get("passed_gate_count") or 0),
            "total_closure_gates": int(overview.get("gate_count") or 0),
            "evidence_kernel_available": bool(evidence.get("available")),
            "open_evidence_gaps": len(evidence.get("open_gaps") or []),
        },
        "tasks": tasks,
        "completion_policy": {
            "plan_exhausted": "计划执行完毕，只表示任务队列完成，不代表理论成立。",
            "accepted": "有效工件和独立复核支持预注册预测时可提前停止。",
            "rejected": "有效反证出现时停止并等待人工调整方向。",
            "inconclusive": "连续未决达到阈值时停止，避免无限自动循环。",
        },
    }


# ==================== 数据模型 ====================

class ModelConfig(BaseModel):
    name: str = ""
    model_type: str = "analyst"  # master | analyst
    api_type: str = "openai"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    model_id: str = "gpt-4o-mini"
    analysis_prompt: str = ""
    planning_prompt: str = ""
    code_gen_prompt: str = ""
    summary_prompt: str = ""


class ConfigPayload(BaseModel):
    master_model: ModelConfig
    analyst_models: List[ModelConfig]


class TestModelPayload(BaseModel):
    model: ModelConfig


class ExecutePayload(BaseModel):
    code: str


class StartSessionPayload(BaseModel):
    objective: str = ""


class ProjectAgentPlanPayload(BaseModel):
    project_goal: str = Field(default="", max_length=4000)
    max_tasks: int = Field(default=6, ge=1, le=12)


class ProjectAgentStartPayload(BaseModel):
    project_goal: str = Field(default="", max_length=4000)
    max_loops: int = Field(default=3, ge=1, le=12)
    execution_mode: Literal["auto", "manual"] = "auto"
    stop_on_accepted: bool = True
    stop_on_rejected: bool = True
    max_consecutive_inconclusive: int = Field(default=3, ge=1, le=12)


# ==================== 研究会话管理 ====================

class ResearchSession:
    """单例研究会话"""

    def __init__(self):
        self.status: str = "idle"  # idle | running | paused | stopped | waiting_step
        self.current_phase: Optional[str] = None
        self.round: int = 0
        self.mode: str = "auto"  # auto | manual
        self.started_at: Optional[float] = None
        self.config: Dict = {}
        self.findings: List[Dict] = []
        self.research_state: Dict = {}
        self.event_queue: asyncio.Queue = asyncio.Queue(maxsize=500)
        self._loop_task: Optional[asyncio.Task] = None
        self._stop_requested = False
        self._pause_requested = False
        self._step_event: asyncio.Event = asyncio.Event()
        self.active_run_id: Optional[str] = None
        self.project_agent: Dict[str, Any] = _default_project_agent_state()

        # 加载保存的配置
        self._load_config()
        self._load_state()

    def _config_path(self) -> Path:
        """模型与 API Key 配置保存在用户目录（不在仓库内，避免随 git 提交泄露 Key）。

        优先 AI2050_AI_RND_CONFIG 环境变量；默认 ~/.ai2050/ai_rnd_config.json。
        仓库根目录的 ai_rnd_config.json 只作为空 Key 模板与旧版迁移来源，不再写入。
        """
        override = os.environ.get("AI2050_AI_RND_CONFIG", "").strip()
        if override:
            return Path(override).expanduser()
        return Path.home() / ".ai2050" / "ai_rnd_config.json"

    def _legacy_config_path(self) -> Path:
        return Path(__file__).parent.parent / "ai_rnd_config.json"

    def _state_path(self) -> Path:
        return Path(__file__).parent.parent / "tests" / "glm5" / "result" / "auto_rnd" / "session_state.json"

    def _legacy_state_path(self) -> Path:
        return Path(__file__).parent.parent / "tests" / "result" / "auto_rnd" / "session_state.json"

    def _load_config(self):
        path = self._config_path()
        if not path.exists():
            # 旧版把配置（含 Key）写在仓库根目录：读取一次并迁移到用户目录
            legacy = self._legacy_config_path()
            if legacy.exists():
                try:
                    with open(legacy, "r", encoding="utf-8") as f:
                        self.config = json.load(f)
                    if self._has_api_key(self.config):
                        self._save_config()
                except Exception:
                    self.config = {}
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.config = json.load(f)
        except Exception:
            self.config = {}

    @staticmethod
    def _has_api_key(config: Dict[str, Any]) -> bool:
        models = [config.get("master_model") or {}] + list(config.get("analyst_models") or [])
        return any(isinstance(m, dict) and str(m.get("api_key") or "").strip() for m in models)

    def _save_config(self):
        path = self._config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        try:
            os.chmod(path, 0o600)  # 仅当前用户可读（Windows 上为无害空操作）
        except OSError:
            pass

    def _load_state(self):
        path = self._state_path()
        if not path.exists():
            path = self._legacy_state_path()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.round = int(payload.get("round", 0))
            self.findings = list(payload.get("findings", []))
            self.research_state = dict(payload.get("research_state", {}))
            saved_agent = payload.get("project_agent")
            if isinstance(saved_agent, dict):
                self.project_agent = {**_default_project_agent_state(), **saved_agent}
                self.project_agent["config"] = {
                    **_default_project_agent_state()["config"],
                    **dict(saved_agent.get("config") or {}),
                }
            self.active_run_id = payload.get("active_run_id")
            self.started_at = payload.get("started_at")
        except Exception:
            return

    def persist_state(self):
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "ai_rnd_session.v3",
            "status": self.status,
            "mode": self.mode,
            "current_phase": self.current_phase,
            "current_gate": _workflow_gate(self.current_phase, self.status),
            "round": self.round,
            "started_at": self.started_at,
            "active_run_id": self.active_run_id,
            "findings": self.findings,
            "research_state": self.research_state,
            "project_agent": self.project_agent,
            "saved_at": datetime.now().isoformat(),
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def push_event(self, event_type: str, **kwargs):
        """推送事件到SSE队列"""
        event = {
            "type": event_type,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        }
        try:
            self.event_queue.put_nowait(event)
        except asyncio.QueueFull:
            pass  # Drop oldest if full

    def get_status(self) -> Dict:
        return {
            "status": self.status,
            "mode": self.mode,
            "current_phase": self.current_phase,
            "current_gate": _workflow_gate(self.current_phase, self.status),
            "round": self.round,
            "started_at": self.started_at,
            "findings_count": len(self.findings),
            "research_state": self.research_state,
            "active_run_id": self.active_run_id,
            "evidence_context": load_evidence_context(),
            "agent_roles": ROLE_DEFINITIONS,
            "project_agent": self.project_agent,
        }


# 全局单例
_session: Optional[ResearchSession] = None


def get_session() -> ResearchSession:
    global _session
    if _session is None:
        _session = ResearchSession()
    return _session


# ==================== AI模型调用 ====================

def _extract_chat_completion_content(data: Dict[str, Any]) -> str:
    """Extract text from OpenAI-compatible chat completion responses."""
    message = data.get("choices", [{}])[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content or "")


def _nownextai_messages(prompt: str, system_msg: str = "") -> List[Dict[str, Any]]:
    messages: List[Dict[str, Any]] = []
    if system_msg:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system_msg}],
        })
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": prompt}],
    })
    return messages


def _chat_completions_endpoint(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


async def call_ai_model(model_config: ModelConfig, prompt: str, system_msg: str = "") -> str:
    """调用AI模型API (OpenAI兼容接口)"""
    try:
        import httpx
    except ImportError:
        return "[ERROR] httpx not installed"

    api_key = model_config.api_key
    api_base = model_config.api_base.rstrip("/")
    model_id = model_config.model_id
    api_type = (model_config.api_type or "").lower()

    if not api_key:
        return f"[ERROR] No API key configured for {model_config.name}"

    if api_type == "claude":
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096,
            "temperature": 0.7,
        }
        if system_msg:
            payload["system"] = system_msg

        try:
            async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                resp = await client.post(
                    f"{api_base}/messages",
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return "".join(
                    item.get("text", "")
                    for item in data.get("content", [])
                    if item.get("type") == "text"
                )
        except Exception as e:
            return f"[ERROR] API call failed: {e}"

    if api_type == "nownextai":
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "stream": False,
            "max_tokens": 4096,
            "messages": _nownextai_messages(prompt, system_msg),
        }

        try:
            async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
                resp = await client.post(
                    _chat_completions_endpoint(api_base),
                    headers=headers,
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
                return _extract_chat_completion_content(data)
        except Exception as e:
            return f"[ERROR] API call failed: {e}"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": 4096,
        "temperature": 0.7,
    }

    try:
        async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
            resp = await client.post(
                _chat_completions_endpoint(api_base),
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return _extract_chat_completion_content(data)
    except Exception as e:
        return f"[ERROR] API call failed: {e}"


# ==================== 代码执行 ====================

async def test_ai_model_config(model_config: ModelConfig) -> Dict[str, Any]:
    """Run a lightweight OpenAI-compatible chat/completions request."""
    try:
        import httpx
    except ImportError:
        return {"ok": False, "message": "httpx is not installed"}

    def ascii_validation_error(field: str, value: str) -> Optional[Dict[str, Any]]:
        try:
            value.encode("ascii")
            return None
        except UnicodeEncodeError as e:
            return {
                "ok": False,
                "message": f"{field} contains non-ASCII characters",
                "details": {
                    "field": field,
                    "error_type": type(e).__name__,
                    "position": f"{e.start}-{e.end}",
                    "hint": "请检查是否复制了中文说明文字、全角符号、空格或不可见字符。API Key / API 地址 / 模型ID 通常只能使用英文、数字和 ASCII 符号。",
                },
            }

    api_key = (model_config.api_key or "").strip()
    api_base = (model_config.api_base or "").rstrip("/")
    model_id = (model_config.model_id or "").strip()
    api_type = (model_config.api_type or "").lower()

    if not api_key:
        return {"ok": False, "message": "API Key is empty"}
    if not api_base:
        return {"ok": False, "message": "API base is empty"}
    if not model_id:
        return {"ok": False, "message": "Model ID is empty"}

    for field, value in (
        ("API Key", api_key),
        ("API base", api_base),
        ("Model ID", model_id),
    ):
        error = ascii_validation_error(field, value)
        if error:
            return error

    if api_type == "claude":
        endpoint = f"{api_base}/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with OK."}],
            "max_tokens": 12,
            "temperature": 0,
        }

        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                resp = await client.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                )
            elapsed_ms = int((time.time() - start) * 1000)
            if resp.status_code >= 400:
                detail = resp.text[:500]
                return {
                    "ok": False,
                    "status_code": resp.status_code,
                    "latency_ms": elapsed_ms,
                    "message": detail or f"HTTP {resp.status_code}",
                    "details": {
                        "api_type": api_type,
                        "endpoint": endpoint,
                        "model_id": model_id,
                        "http_status": resp.status_code,
                        "response_body": detail,
                    },
                }
            data = resp.json()
            content = "".join(
                item.get("text", "")
                for item in data.get("content", [])
                if item.get("type") == "text"
            )
            return {
                "ok": True,
                "status_code": resp.status_code,
                "latency_ms": elapsed_ms,
                "message": "Connection test succeeded",
                "sample": content[:200],
            }
        except Exception as e:
            message = (
                "请求字段包含非 ASCII 字符，请检查 API Key、API 地址、模型ID 是否混入中文、全角符号、空格或不可见字符。"
                if isinstance(e, UnicodeEncodeError)
                else str(e)
            )
            return {
                "ok": False,
                "message": message,
                "details": {
                    "api_type": api_type,
                    "endpoint": endpoint,
                    "model_id": model_id,
                    "error_type": type(e).__name__,
                    "original_error": str(e),
                },
            }

    if api_type == "nownextai":
        endpoint = _chat_completions_endpoint(api_base)
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model_id,
            "stream": False,
            "max_tokens": 12,
            "messages": _nownextai_messages("Reply with OK."),
        }

        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
                resp = await client.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                )
            elapsed_ms = int((time.time() - start) * 1000)
            if resp.status_code >= 400:
                detail = resp.text[:500]
                return {
                    "ok": False,
                    "status_code": resp.status_code,
                    "latency_ms": elapsed_ms,
                    "message": detail or f"HTTP {resp.status_code}",
                    "details": {
                        "api_type": api_type,
                        "endpoint": endpoint,
                        "model_id": model_id,
                        "http_status": resp.status_code,
                        "response_body": detail,
                    },
                }
            data = resp.json()
            content = _extract_chat_completion_content(data)
            return {
                "ok": True,
                "status_code": resp.status_code,
                "latency_ms": elapsed_ms,
                "message": "Connection test succeeded",
                "sample": content[:200],
            }
        except Exception as e:
            message = (
                "Request fields contain non-ASCII characters. Check API Key, API base, or Model ID."
                if isinstance(e, UnicodeEncodeError)
                else str(e)
            )
            return {
                "ok": False,
                "message": message,
                "details": {
                    "api_type": api_type,
                    "endpoint": endpoint,
                    "model_id": model_id,
                    "error_type": type(e).__name__,
                    "original_error": str(e),
                },
            }

    endpoint = _chat_completions_endpoint(api_base)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": "You are a connection test endpoint."},
            {"role": "user", "content": "Reply with OK."},
        ],
        "max_tokens": 12,
        "temperature": 0,
    }

    start = time.time()
    try:
        async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
            resp = await client.post(
                endpoint,
                headers=headers,
                json=payload,
            )
        elapsed_ms = int((time.time() - start) * 1000)
        if resp.status_code >= 400:
            detail = resp.text[:500]
            return {
                "ok": False,
                "status_code": resp.status_code,
                "latency_ms": elapsed_ms,
                "message": detail or f"HTTP {resp.status_code}",
                "details": {
                    "api_type": api_type or "openai-compatible",
                    "endpoint": endpoint,
                    "model_id": model_id,
                    "http_status": resp.status_code,
                    "response_body": detail,
                },
            }
        data = resp.json()
        content = _extract_chat_completion_content(data)
        return {
            "ok": True,
            "status_code": resp.status_code,
            "latency_ms": elapsed_ms,
            "message": "Connection test succeeded",
            "sample": content[:200],
        }
    except Exception as e:
        message = (
            "请求字段包含非 ASCII 字符，请检查 API Key、API 地址、模型ID 是否混入中文、全角符号、空格或不可见字符。"
            if isinstance(e, UnicodeEncodeError)
            else str(e)
        )
        return {
            "ok": False,
            "message": message,
            "details": {
                "api_type": api_type or "openai-compatible",
                "endpoint": endpoint,
                "model_id": model_id,
                "error_type": type(e).__name__,
                "original_error": str(e),
            },
        }


def execute_code_sandbox(
    code: str,
    timeout: int = 120,
    run: Optional[ResearchRun] = None,
    stop_requested=None,
) -> Dict:
    """Execute reviewed code in a persistent run with a single-GPU lock."""
    active_run = run or create_research_run("manual code execution", int(time.time()))
    result = execute_research_code(
        code,
        active_run,
        timeout=timeout,
        stop_requested=stop_requested,
    )
    result["artifact_audit"] = artifact_audit(active_run)
    return result


# ==================== 研究循环 ====================

def _advance_project_agent(
    session: ResearchSession,
    *,
    decision: str,
    run_id: str,
) -> bool:
    """Advance one bounded project task and return True when the agent must stop."""
    agent = session.project_agent
    if not agent.get("enabled"):
        return False

    plan = agent.get("plan") or {}
    tasks = list(plan.get("tasks") or [])
    task_index = max(0, int(agent.get("current_task_index") or 0))
    if task_index < len(tasks):
        tasks[task_index] = {
            **tasks[task_index],
            "status": "completed",
            "decision": decision,
            "run_id": run_id,
            "completed_at": datetime.now().astimezone().isoformat(),
        }
        plan["tasks"] = tasks
        agent["plan"] = plan

    agent["loops_completed"] = int(agent.get("loops_completed") or 0) + 1
    agent["last_decision"] = decision
    if decision == "inconclusive":
        agent["consecutive_inconclusive"] = int(agent.get("consecutive_inconclusive") or 0) + 1
    else:
        agent["consecutive_inconclusive"] = 0

    config = {**_default_project_agent_state()["config"], **dict(agent.get("config") or {})}
    next_index = task_index + 1
    max_loops = int(config.get("max_loops") or 3)
    inconclusive_limit = int(config.get("max_consecutive_inconclusive") or 3)
    stop_status = ""
    stop_reason = ""

    if decision == "accepted" and config.get("stop_on_accepted", True):
        stop_status = "completed"
        stop_reason = "预注册目标获得有效证据并通过工件审计"
    elif decision == "rejected" and config.get("stop_on_rejected", True):
        stop_status = "blocked"
        stop_reason = "出现有效反证，等待人工调整研发方向"
    elif agent["consecutive_inconclusive"] >= inconclusive_limit:
        stop_status = "review_required"
        stop_reason = f"连续 {agent['consecutive_inconclusive']} 次未决，已触发安全停止"
    elif agent["loops_completed"] >= max_loops or next_index >= len(tasks):
        stop_status = "plan_completed"
        stop_reason = "有界研发计划已执行完毕；理论等级仍需依据证据人工确认"

    if stop_status:
        agent.update({
            "enabled": False,
            "status": stop_status,
            "stop_reason": stop_reason,
            "completed_at": datetime.now().astimezone().isoformat(),
        })
        session.persist_state()
        session.push_event(
            "project_agent_status",
            status=stop_status,
            stop_reason=stop_reason,
            project_agent=agent,
        )
        return True

    agent["current_task_index"] = next_index
    agent["status"] = "running"
    next_task = tasks[next_index]
    session.research_state["research_objective"] = next_task["objective"]
    session.persist_state()
    session.push_event(
        "project_agent_progress",
        status="running",
        current_task_index=next_index,
        task=next_task,
        loops_completed=agent["loops_completed"],
    )
    return False


async def _wait_for_step(session: ResearchSession, after_phase: str = ""):
    """手动模式: 等待用户点击下一步"""
    if session.mode != "manual":
        return
    session.status = "waiting_step"
    if session.project_agent.get("enabled"):
        session.project_agent["status"] = "waiting_approval"
        session.persist_state()
    session.push_event("status_change", status="waiting_step", waiting_after=after_phase)
    session._step_event.clear()
    await session._step_event.wait()  # Wait until user clicks "step"
    session.status = "running"
    if session.project_agent.get("enabled"):
        session.project_agent["status"] = "running"
        session.persist_state()
    session.push_event("status_change", status="running")
    session._step_event.clear()


async def run_research_loop(session: ResearchSession):
    """5阶段自动研究循环(支持自动/手动模式)"""
    session.status = "running"
    session.started_at = time.time()
    if session.project_agent.get("enabled"):
        session.project_agent["status"] = "running"
        session.project_agent["started_at"] = (
            session.project_agent.get("started_at")
            or datetime.now().astimezone().isoformat()
        )
        session.project_agent["completed_at"] = None
        session.project_agent["stop_reason"] = ""
    session.persist_state()
    session.push_event("status_change", status="running")
    session._stop_requested = False
    session._pause_requested = False

    config = session.config
    master_cfg = config.get("master_model", {})
    analyst_cfgs = config.get("analyst_models", [])

    master_model = ModelConfig(**master_cfg) if master_cfg else None
    analyst_models = [ModelConfig(**a) for a in analyst_cfgs]

    if not master_model or not master_model.api_key:
        session.push_event("error", message="主模型未配置API Key，请在配置页面设置")
        session.status = "stopped"
        if session.project_agent.get("enabled"):
            session.project_agent.update({
                "enabled": False,
                "status": "blocked",
                "stop_reason": "主模型未配置 API Key",
                "completed_at": datetime.now().astimezone().isoformat(),
            })
        session.persist_state()
        session.push_event("status_change", status="stopped")
        return

    while not session._stop_requested:
        # Check pause
        if session._pause_requested:
            session.status = "paused"
            if session.project_agent.get("enabled"):
                session.project_agent["status"] = "paused"
            session.persist_state()
            session.push_event("status_change", status="paused")
            while session._pause_requested and not session._stop_requested:
                await asyncio.sleep(1)
            if session._stop_requested:
                break
            session.status = "running"
            if session.project_agent.get("enabled"):
                session.project_agent["status"] = "running"
            session.persist_state()
            session.push_event("status_change", status="running")

        session.round += 1
        session.push_event("round_change", round=session.round)
        objective = str(session.research_state.get("research_objective") or "从证据缺口中选择可证伪的语言机制问题")
        research_run = create_research_run(objective, session.round)
        session.active_run_id = research_run.run_id
        evidence_context = load_evidence_context()
        session.research_state["active_experiment"] = json.loads(
            research_run.experiment_path.read_text(encoding="utf-8")
        )
        session.research_state["evidence_context"] = evidence_context
        session.persist_state()
        session.push_event(
            "research_run",
            run_id=research_run.run_id,
            experiment=session.research_state["active_experiment"],
        )

        # ===== Phase 1: 分析 =====
        session.current_phase = "analyze"
        session.push_event("phase_change", phase="analyze")
        analyst_reports = []

        findings_summary = "\n".join(
            f"- [{f.get('round', '?')}] {f.get('title', f.get('content', '')[:80])}"
            for f in session.findings[-20:]
        ) or "暂无历史发现"

        test_results = session.research_state.get("last_test_output", "暂无测试结果")

        review_roles = ("method_reviewer", "adversarial_reviewer")
        for analyst_index, am in enumerate(analyst_models):
            if not am.api_key:
                continue
            role = review_roles[analyst_index % len(review_roles)]
            prompt = (am.analysis_prompt or "").replace("{round}", str(session.round))
            prompt = prompt.replace("{findings}", findings_summary)
            prompt = prompt.replace("{test_results}", test_results)
            prompt += (
                "\n\n本轮 ExperimentSpec:\n"
                + json.dumps(session.research_state["active_experiment"], ensure_ascii=False, indent=2)
                + "\n\n当前 Evidence Kernel 摘要:\n"
                + json.dumps(evidence_context, ensure_ascii=False, indent=2)
            )

            session.push_event("analysis", content=f"📡 {am.name} 以 {role} 角色审核中...")
            result = await call_ai_model(am, prompt, system_msg=ROLE_DEFINITIONS[role])
            analyst_reports.append({"model": am.name, "role": role, "analysis": result})
            session.push_event("analysis", content=f"✓ {am.name} / {role} 审核完成")

        session.research_state["last_analyst_reports"] = analyst_reports
        if session._stop_requested:
            break

        # Manual: wait after analysis
        await _wait_for_step(session, "analyze")
        if session._stop_requested:
            break

        # ===== Phase 2: 规划 =====
        session.current_phase = "plan"
        session.push_event("phase_change", phase="plan")

        reports_text = "\n\n".join(
            f"## {r['model']}\n{r['analysis']}" for r in analyst_reports
        )
        verified = session.research_state.get("verified_conclusions", "暂无")
        bottlenecks = session.research_state.get("bottlenecks", "暂无")

        plan_prompt = (master_model.planning_prompt or "").replace("{round}", str(session.round))
        plan_prompt = plan_prompt.replace("{research_context}", json.dumps(session.research_state, ensure_ascii=False)[:8000])
        plan_prompt = plan_prompt.replace("{analyst_reports}", reports_text[:6000])
        plan_prompt = plan_prompt.replace("{verified_conclusions}", verified)
        plan_prompt = plan_prompt.replace("{bottlenecks}", bottlenecks)

        session.push_event("planning", content="📋 主模型规划中...")
        plan_prompt += (
            "\n\n必须遵守 ExperimentSpec；Qwen3、GLM4、DeepSeek7B 必须串行；"
            "先 smoke，再 full，再独立复核。所有产物写入环境变量 AI_RND_RUN_DIR。"
        )
        plan = await call_ai_model(master_model, plan_prompt, system_msg=ROLE_DEFINITIONS["principal_investigator"])
        session.push_event("planning", content=f"✓ 规划完成")
        session.research_state["last_plan"] = plan

        if session._stop_requested:
            break

        # Manual: wait after planning
        await _wait_for_step(session, "plan")
        if session._stop_requested:
            break

        # ===== Phase 3: 代码生成 =====
        session.current_phase = "generate"
        session.push_event("phase_change", phase="generate")

        code_prompt = (master_model.code_gen_prompt or "").replace("{plan}", plan[:10000])
        code_prompt += (
            "\n\n运行目录来自 os.environ['AI_RND_RUN_DIR']。"
            "必须输出 ExperimentSpec 要求的 JSON/JSONL/manifest 文件；不得启动子进程；"
            "不得访问网络；三个本地模型只能按 AI_RND_MODEL_ORDER 串行加载和释放。"
        )
        session.push_event("generation", content="⚡ 主模型生成代码中...")
        generated_code = await call_ai_model(master_model, code_prompt, system_msg="你只输出Python代码，不要任何解释或markdown标记")

        if generated_code.startswith("```"):
            lines = generated_code.split("\n")
            generated_code = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        session.push_event("code_generated", code=generated_code)
        session.research_state["last_code"] = generated_code

        if session._stop_requested:
            break

        # ===== Phase 4: 执行 =====
        session.current_phase = "execute"
        session.push_event("phase_change", phase="execute")
        session.push_event("execution", content="🚀 执行代码中...")

        loop = asyncio.get_event_loop()
        exec_result = await loop.run_in_executor(
            None,
            lambda: execute_code_sandbox(
                generated_code,
                run=research_run,
                stop_requested=lambda: session._stop_requested,
            ),
        )

        session.push_event("execution_result", result=exec_result)
        session.research_state["last_test_output"] = exec_result.get("output", "")[:5000]
        session.research_state["last_execution_status"] = exec_result.get("status", "unknown")
        session.research_state["last_artifact_audit"] = exec_result.get("artifact_audit", {})

        # Independent reviewers see the artifact inventory and raw execution output,
        # not the master model's summary.
        post_run_reviews = []
        post_roles = ("data_auditor", "adversarial_reviewer")
        for analyst_index, am in enumerate(analyst_models):
            if not am.api_key:
                continue
            role = post_roles[analyst_index % len(post_roles)]
            audit_prompt = (
                f"研究目标: {objective}\n"
                f"ExperimentSpec: {json.dumps(session.research_state['active_experiment'], ensure_ascii=False)}\n"
                f"Artifact audit: {json.dumps(exec_result.get('artifact_audit', {}), ensure_ascii=False)}\n"
                f"Raw execution: {json.dumps(exec_result, ensure_ascii=False)[:12000]}\n"
                "请逐条检查证据，不得根据其他 AI 的共识提升结论。"
            )
            session.push_event("review", content=f"{am.name} / {role} 正在复核原始产物")
            review = await call_ai_model(am, audit_prompt, system_msg=ROLE_DEFINITIONS[role])
            post_run_reviews.append({"model": am.name, "role": role, "review": review})
        session.research_state["last_post_run_reviews"] = post_run_reviews

        if session._stop_requested:
            break

        # Manual: wait after execution (user reviews test results)
        await _wait_for_step(session, "execute")
        if session._stop_requested:
            break

        # ===== Phase 5: 总结 =====
        session.current_phase = "summarize"
        session.push_event("phase_change", phase="summarize")

        summary_prompt = (master_model.summary_prompt or "").replace("{round}", str(session.round))
        summary_prompt = summary_prompt.replace("{execution_results}", json.dumps(exec_result, ensure_ascii=False)[:4000])
        summary_prompt = summary_prompt.replace("{key_data}", exec_result.get("output", "")[:3000])
        summary_prompt += (
            "\n\nArtifact audit:\n"
            + json.dumps(exec_result.get("artifact_audit", {}), ensure_ascii=False, indent=2)
            + "\n\nIndependent reviews:\n"
            + json.dumps(post_run_reviews, ensure_ascii=False, indent=2)[:10000]
            + "\n裁决只能是 accepted、rejected 或 inconclusive；产物不完整时必须是 inconclusive。"
        )

        session.push_event("summary", content="📝 总结中...")
        summary = await call_ai_model(master_model, summary_prompt, system_msg=ROLE_DEFINITIONS["principal_investigator"])
        decision = exec_result.get("artifact_audit", {}).get("decision", "inconclusive")
        if decision == "pending_review":
            decision = "inconclusive"

        finding = {
            "round": session.round,
            "title": f"Round {session.round} 研究发现",
            "content": summary,
            "source": master_model.name,
            "timestamp": datetime.now().isoformat(),
            "tags": [session.mode],
            "run_id": research_run.run_id,
            "decision": decision,
            "artifact_audit": exec_result.get("artifact_audit", {}),
            "review_roles": [review.get("role") for review in post_run_reviews],
        }
        session.findings.append(finding)
        session.push_event("finding", finding=finding)
        session.push_event("summary", content=f"✓ Round {session.round} 完成")

        session.research_state["rounds_completed"] = session.round
        session.research_state["last_summary"] = summary[:2000]
        session.research_state["last_decision"] = decision
        session.persist_state()
        try:
            get_research_workspace_store().record_loop_result(
                run_id=research_run.run_id,
                objective=objective,
                loop_number=session.round,
                mode=session.mode,
                status="completed",
                decision=decision,
                summary=summary,
                master_model=master_model.name,
                analyst_models=[model.name for model in analyst_models],
                artifact_audit=exec_result.get("artifact_audit", {}),
            )
            session.push_event("database_writeback", run_id=research_run.run_id, status="saved")
        except Exception as writeback_error:
            # A database indexing failure must remain visible, but must not erase
            # the immutable run artifacts that were already written and audited.
            session.push_event(
                "database_writeback",
                run_id=research_run.run_id,
                status="failed",
                message=str(writeback_error),
            )

        if _advance_project_agent(
            session,
            decision=decision,
            run_id=research_run.run_id,
        ):
            break

        # Manual: wait at end of round
        await _wait_for_step(session, "summarize")

        # Brief pause between rounds (auto mode only)
        if session.mode == "auto":
            await asyncio.sleep(2)

    # Loop ended
    session.status = "stopped"
    session.current_phase = None
    if session.project_agent.get("enabled"):
        session.project_agent.update({
            "enabled": False,
            "status": "stopped",
            "stop_reason": session.project_agent.get("stop_reason") or "研发循环已停止",
            "completed_at": datetime.now().astimezone().isoformat(),
        })
    session.persist_state()
    session.push_event("status_change", status="stopped")


# ==================== API路由 ====================

router = APIRouter(prefix="/api/ai-rnd", tags=["AI R&D"])


@router.get("/config")
async def get_config():
    session = get_session()
    return session.config


@router.put("/config")
async def update_config(payload: ConfigPayload):
    session = get_session()
    session.config = {
        "master_model": payload.master_model.model_dump(),
        "analyst_models": [a.model_dump() for a in payload.analyst_models],
    }
    session._save_config()
    return {"status": "saved"}


@router.post("/config/test")
async def test_config(payload: TestModelPayload):
    return await test_ai_model_config(payload.model)


@router.post("/session/start")
async def start_session(payload: Optional[StartSessionPayload] = None):
    session = get_session()
    if session.status in ("running", "waiting_step"):
        raise HTTPException(400, "Session already running")
    objective = (payload.objective if payload else "").strip()
    if objective:
        session.research_state["research_objective"] = objective
        session.push_event("objective", message=f"研究目标: {objective}", objective=objective)
        session.persist_state()
    if session._pause_requested:
        session._pause_requested = False
        if session.project_agent.get("enabled"):
            session.project_agent["status"] = "running"
            session.persist_state()
        return {"status": "resumed"}

    # Starting a one-off objective must not silently resume an old project plan.
    if session.project_agent.get("enabled"):
        session.project_agent.update({
            "enabled": False,
            "status": "stopped",
            "stop_reason": "已切换为单目标 Loop",
            "completed_at": datetime.now().astimezone().isoformat(),
        })
        session.persist_state()

    # Start the research loop as background task
    loop = asyncio.get_event_loop()
    session._loop_task = loop.create_task(run_research_loop(session))
    return {"status": "started"}


@router.post("/session/pause")
async def pause_session():
    session = get_session()
    if session.status != "running":
        raise HTTPException(400, "Session not running")
    session._pause_requested = True
    if session.project_agent.get("enabled"):
        session.project_agent["status"] = "pausing"
        session.persist_state()
    return {"status": "pausing"}


class ModePayload(BaseModel):
    mode: str  # auto | manual


@router.put("/session/mode")
async def set_mode(payload: ModePayload):
    session = get_session()
    if payload.mode not in ("auto", "manual"):
        raise HTTPException(400, "mode must be 'auto' or 'manual'")
    session.mode = payload.mode
    session.persist_state()
    session.push_event("mode_change", mode=payload.mode)
    return {"status": "ok", "mode": payload.mode}


@router.post("/session/step")
async def step_session():
    """手动模式: 推进到下一个阶段"""
    session = get_session()
    if session.mode != "manual":
        raise HTTPException(400, "Not in manual mode")
    if session.status != "waiting_step":
        raise HTTPException(400, f"Session not waiting for step (status: {session.status})")
    session._step_event.set()
    session.push_event("step_triggered")
    return {"status": "stepping", "next_phase": session.current_phase}


@router.post("/session/stop")
async def stop_session():
    session = get_session()
    session._stop_requested = True
    session._pause_requested = False
    terminate_active_process()
    if session._loop_task:
        session._loop_task.cancel()
        session._loop_task = None
    session.status = "stopped"
    session.current_phase = None
    if session.project_agent.get("enabled"):
        session.project_agent.update({
            "enabled": False,
            "status": "stopped",
            "stop_reason": "用户停止",
            "completed_at": datetime.now().astimezone().isoformat(),
        })
    session.persist_state()
    return {"status": "stopped"}


@router.get("/project-agent/status")
async def get_project_agent_status():
    session = get_session()
    return {
        "project_agent": session.project_agent,
        "session_status": session.status,
        "current_gate": _workflow_gate(session.current_phase, session.status),
    }


@router.post("/project-agent/plan")
async def preview_project_agent_plan(payload: ProjectAgentPlanPayload):
    """Generate a deterministic plan from the current research database."""
    return build_project_agent_plan(
        project_goal=payload.project_goal,
        max_tasks=payload.max_tasks,
    )


@router.post("/project-agent/start")
async def start_project_agent(payload: ProjectAgentStartPayload):
    """Start a bounded project-level agent over the existing evidence-gated loop."""
    session = get_session()
    if session.status in ("running", "paused", "waiting_step"):
        raise HTTPException(409, "已有研发任务正在运行，请先停止后再启动项目 Agent")

    master_config = dict(session.config.get("master_model") or {})
    if not str(master_config.get("api_key") or "").strip():
        raise HTTPException(409, "主研发模型未配置 API Key，请先在‘代理与提示词’中保存配置")

    configured_analysts = [
        model
        for model in session.config.get("analyst_models", [])
        if str(model.get("api_key") or "").strip()
    ]
    if not configured_analysts:
        raise HTTPException(409, "至少需要一个已配置 API Key 的辅助模型执行独立复核")

    plan = build_project_agent_plan(
        project_goal=payload.project_goal,
        max_tasks=payload.max_loops,
    )
    tasks = list(plan.get("tasks") or [])
    if not tasks:
        raise HTTPException(409, "当前研究数据库无法生成可执行任务")

    now = datetime.now().astimezone().isoformat()
    session.project_agent = {
        **_default_project_agent_state(),
        "enabled": True,
        "status": "running",
        "project_goal": plan["project_goal"],
        "started_at": now,
        "config": {
            "max_loops": payload.max_loops,
            "execution_mode": payload.execution_mode,
            "stop_on_accepted": payload.stop_on_accepted,
            "stop_on_rejected": payload.stop_on_rejected,
            "max_consecutive_inconclusive": payload.max_consecutive_inconclusive,
        },
        "plan": plan,
    }
    session.mode = payload.execution_mode
    session.status = "running"
    session.research_state["research_objective"] = tasks[0]["objective"]
    session._stop_requested = False
    session._pause_requested = False
    session._step_event.clear()
    session.persist_state()
    session.push_event(
        "project_agent_status",
        status="running",
        task=tasks[0],
        project_agent=session.project_agent,
    )
    loop = asyncio.get_event_loop()
    session._loop_task = loop.create_task(run_research_loop(session))
    return {
        "status": "started",
        "project_agent": session.project_agent,
    }


@router.post("/project-agent/stop")
async def stop_project_agent():
    session = get_session()
    session._stop_requested = True
    session._pause_requested = False
    session._step_event.set()
    terminate_active_process()
    if session._loop_task:
        session._loop_task.cancel()
        session._loop_task = None
    session.status = "stopped"
    session.current_phase = None
    session.project_agent.update({
        "enabled": False,
        "status": "stopped",
        "stop_reason": "用户停止项目 Agent",
        "completed_at": datetime.now().astimezone().isoformat(),
    })
    session.persist_state()
    session.push_event(
        "project_agent_status",
        status="stopped",
        stop_reason=session.project_agent["stop_reason"],
    )
    return {"status": "stopped", "project_agent": session.project_agent}


@router.get("/session/status")
async def get_session_status():
    session = get_session()
    return session.get_status()


@router.get("/session/events")
async def stream_events():
    """SSE endpoint for real-time event streaming"""
    session = get_session()

    async def event_generator():
        while True:
            try:
                event = await asyncio.wait_for(session.event_queue.get(), timeout=30)
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except asyncio.TimeoutError:
                yield f": keepalive\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/execute")
async def execute_code(payload: ExecutePayload):
    """手动执行代码"""
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, execute_code_sandbox, payload.code)
    return result


@router.get("/findings")
async def get_findings():
    session = get_session()
    return {"findings": session.findings, "round": session.round}


@router.get("/history")
async def get_history():
    session = get_session()
    return {
        "rounds_completed": session.round,
        "findings": session.findings,
        "research_state": session.research_state,
        "started_at": session.started_at,
    }


@router.get("/orchestrator/status")
async def get_orchestrator_status():
    session = get_session()
    return {
        "schema_version": "research_orchestrator_status.v3",
        "status": session.status,
        "mode": session.mode,
        "current_gate": _workflow_gate(session.current_phase, session.status),
        "workflow_gates": [
            {"id": gate["id"], "label": gate["label"]}
            for gate in WORKFLOW_GATES
        ],
        "active_run_id": session.active_run_id,
        "model_order": ["qwen3", "glm4", "deepseek7b"],
        "gpu_policy": "single_process_serial",
        "roles": ROLE_DEFINITIONS,
        "project_agent": session.project_agent,
        "evidence_context": load_evidence_context(),
        "recent_runs": list_research_runs(20),
    }


@router.get("/orchestrator/runs")
async def get_orchestrator_runs(limit: int = 50):
    return {"runs": list_research_runs(max(1, min(limit, 200)))}


# ---------------------------------------------------------------- M3-P1（design/ui_decoupled_plan_v2.md §2）
# 研发透镜内容层数据源：队列 / 工作区文件树 / 工件文本。内容只存在于数据层，
# JSX 禁止出现具体实验字面量（Q05/F#3734/collect_ar.py 等）。

_PHASE_QUEUE_PATH = Path(__file__).parent.parent / "research" / "deepseek" / "atlas" / "phase_queue_v1.json"
_WS_ALLOWED_ROOTS = ("tests", "research")
_WS_MAX_FILE = 512 * 1024


def _ws_resolve(path: str) -> Path:
    """白名单解析：解析后必须真实落在 tests/ 或 research/ 之下（防 ../ 逃逸）。"""
    root = Path(__file__).parent.parent.resolve()
    rel = (path or "").strip().replace("\\", "/").strip("/")
    if not rel:
        raise HTTPException(status_code=400, detail="path is required")
    cand = (root / rel).resolve()
    allowed = [root / r for r in _WS_ALLOWED_ROOTS]
    if not any(cand == a or a in cand.parents for a in allowed):
        raise HTTPException(status_code=403, detail="path outside allowed roots (tests/research)")
    return cand


@router.get("/queue")
async def get_queue():
    """研发队列（唯一议程来源 phase_queue_v1.json）。文件缺失返回空队列，不阻塞页面。"""
    try:
        data = json.loads(_PHASE_QUEUE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"schema": "rdc_phase_queue_v1", "source": "missing", "count": 0,
                "sealed": 0, "queue": []}
    items = []
    for it in data.get("queue", []):
        if not isinstance(it, dict):
            continue
        items.append({
            "id": it.get("id"), "q": it.get("q"), "title": it.get("title"),
            "status": it.get("status"), "block": it.get("block"),
            "kpi": it.get("kpi"), "gpu": it.get("gpu"),
            "deliverable": it.get("deliverable"),
            "seal_record": it.get("seal_record"), "res_sha8": it.get("res_sha8"),
            "prereg_design_sha": it.get("prereg_design_sha"), "note": it.get("note"),
        })
    return {"schema": data.get("schema"), "source": "phase_queue_v1.json",
            "count": len(items), "sealed": len(data.get("sealed_items", [])),
            "queue": items}


@router.get("/workspace")
async def workspace_list(path: str = "tests/deepseek", limit: int = 80):
    """工作区文件树：目录内一层列表（白名单根 tests/research）。"""
    root = Path(__file__).parent.parent.resolve()
    p = _ws_resolve(path)
    if not p.exists():
        raise HTTPException(status_code=404, detail="workspace path not found")
    if not p.is_dir():
        raise HTTPException(status_code=400, detail="workspace path is not a directory")
    dirs, files = [], []
    try:
        children = sorted(p.iterdir(), key=lambda x: x.name)
    except PermissionError:
        children = []
    for child in children:
        if child.name.startswith(".") or child.name.startswith("__"):
            continue
        try:
            rel = child.relative_to(root).as_posix()
        except ValueError:
            continue
        if child.is_dir():
            dirs.append({"name": child.name, "path": rel, "type": "dir"})
        elif child.is_file():
            try:
                st = child.stat()
            except OSError:
                continue
            files.append({"name": child.name, "path": rel, "type": "file",
                          "size": int(st.st_size), "mtime": float(st.st_mtime)})
    n = max(1, min(int(limit), 200))
    rel_self = p.relative_to(root).as_posix()
    return {"path": rel_self, "name": p.name,
            "dirs": dirs, "files": files[:n], "files_total": len(files),
            "truncated": len(files) > n}


@router.get("/workspace/file")
async def workspace_file(path: str):
    """工作区文本文件内容（≤512KB，utf-8 replace）——工件查看器数据源。"""
    p = _ws_resolve(path)
    if not p.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    size = p.stat().st_size
    if size > _WS_MAX_FILE:
        raise HTTPException(status_code=413, detail="file too large (>512KB)")
    text = p.read_text(encoding="utf-8", errors="replace")
    return {"path": (path or "").replace("\\", "/").strip("/"), "name": p.name,
            "size": size, "content": text}


# ============================================================
# Atlas 图谱接入（Phase 3179/3180 客户端阶段 A/C，协议键驱动，零实验字面量）
# ============================================================
_ATLAS_DIR = Path(__file__).parent.parent / "research" / "gpt5" / "atlas"
_ATLAS_RESULT_DIR = (Path(__file__).parent.parent / "tests" / "glm5" / "result" /
                     "rdc_query_construction_20260913")
# registry / gap 版本发现：glob 递归（与 server/derive_atlas_tree.py
# latest_registry() 同一口径，见 research/gpt5/docs/CLIENT_SERVER_ALIGNMENT_v1.md A1）。
# 新版本随 Phase 产物自动入链，无需手工登记；legacy 硬编码候选保留作 fallback。
_ATLAS_REGISTRY_CANDIDATES_LEGACY = [
    ("1.8", _ATLAS_RESULT_DIR / "phase3181" / "g5b6_ftr04_glm4_attribution" / "atlas_registry_v1_8.json"),
    ("1.7", _ATLAS_RESULT_DIR / "phase3180" / "g5b5_ftr15_heldout" / "atlas_registry_v1_7.json"),
    ("1.6", _ATLAS_RESULT_DIR / "phase3179" / "g5b4_ftr23_registration" / "atlas_registry_v1_6.json"),
    ("1.5", _ATLAS_RESULT_DIR / "phase3178" / "g5b3_ftr14_a2_attribution" / "atlas_registry_v1_5.json"),
]


def _discover_atlas_versions():
    """glob 递归发现 atlas_registry_v1_N.json 全版本链，按 N 升序返回 [(n, path)]。
    用标准 glob.glob（容忍 phase2746/field_store 等坏链接），与
    server/derive_atlas_tree.py latest_registry() 逐字同口径。"""
    out = []
    for s in glob.glob(str(_ATLAS_RESULT_DIR / "**" / "atlas_registry_v1_*.json"),
                       recursive=True):
        m = re.search(r"v1_(\d+)\.json$", os.path.basename(s))
        if m:
            out.append((int(m.group(1)), Path(s)))
    out.sort(key=lambda x: x[0])
    return out


def _discover_gap_versions():
    """glob 递归发现 gap_ledger_v1_N.json 全版本链，按 N 升序返回 [(n, path)]。"""
    out = []
    for s in glob.glob(str(_ATLAS_RESULT_DIR / "**" / "gap_ledger_v1_*.json"),
                       recursive=True):
        m = re.search(r"v1_(\d+)\.json$", os.path.basename(s))
        if m:
            out.append((int(m.group(1)), Path(s)))
    out.sort(key=lambda x: x[0])
    return out


def _atlas_registry_path(version: Optional[str] = None) -> Optional[Path]:
    cands = _discover_atlas_versions()
    if version:
        # version 形如 "1.32" / "32" 均可
        try:
            n = int(str(version).split(".")[-1])
        except ValueError:
            return None
        for k, p in cands:
            if k == n:
                return p
        return None
    if cands:
        return cands[-1][1]
    for _, p in _ATLAS_REGISTRY_CANDIDATES_LEGACY:
        if p.is_file():
            return p
    return None


def _atlas_version_list() -> List[Dict[str, Any]]:
    out = []
    for k, p in reversed(_discover_atlas_versions()):
        try:
            reg = json.loads(p.read_text(encoding="utf-8"))
            out.append({"version": "1.%d" % k, "file": p.name,
                        "sha8_source": reg.get("supersedes", ""),
                        "n_features": len(reg.get("features", [])),
                        "n_upgrades": len(reg.get("upgrade_log", [])),
                        "n_failures": len(reg.get("failures", []))})
        except Exception:
            continue
    return out


def _feature_card(f: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f.get("id"), "family": f.get("family"),
        "statement": f.get("statement"), "evidence_level": f.get("evidence_level"),
        "model_scope": f.get("model_scope", []),
        "values": f.get("values", {}), "replication": f.get("replication"),
        "scope_limits": f.get("scope_limits"),
        "counter_evidence": f.get("counter_evidence", []),
        "n_anchors": len(f.get("anchors", [])),
        "anchors": [{"src": a.get("src"), "phase": a.get("phase"), "role": a.get("role"),
                     "tags": a.get("tags", []),
                     "res_sha8": (a.get("asserts") or {}).get("res_sha8"),
                     "seal_sha8": (a.get("asserts") or {}).get("seal_sha8")}
                    for a in f.get("anchors", [])],
    }


@router.get("/atlas/registry")
async def atlas_registry(version: Optional[str] = None):
    """图谱 registry（协议第五件数据源；versions=版本链，registry=完整 JSON）。文件缺失 404。"""
    p = _atlas_registry_path(version)
    if not p:
        raise HTTPException(status_code=404, detail="atlas registry not found")
    try:
        reg = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail="registry parse error: %s" % e)
    return {"schema": "atlas_registry_v1", "versions": _atlas_version_list(),
            "version": reg.get("version"), "file": p.name,
            "features": [_feature_card(f) for f in reg.get("features", [])],
            "upgrade_log": reg.get("upgrade_log", []),
            "failures": reg.get("failures", []),
            "principles": reg.get("principles", []),
            "evidence_level_taxonomy": reg.get("evidence_level_taxonomy", {})}


@router.get("/atlas/feature/{fid}")
async def atlas_feature(fid: str, version: Optional[str] = None):
    """特征卡详情（含证据链 anchors 逐条 res/seal sha8，供钻取）。"""
    p = _atlas_registry_path(version)
    if not p:
        raise HTTPException(status_code=404, detail="atlas registry not found")
    reg = json.loads(p.read_text(encoding="utf-8"))
    for f in reg.get("features", []):
        if f.get("id") == fid:
            return {"schema": "atlas_feature_v1", "version": reg.get("version"),
                    "feature": f}
    raise HTTPException(status_code=404, detail="feature %s not found" % fid)


@router.get("/atlas/ledger")
async def atlas_ledger(offset: int = 0, limit: int = 30, line: Optional[str] = None):
    """跨线账本 measurements（append-only，只读分页；line 可选过滤 G/D/N）。"""
    lp = _ATLAS_DIR / "atlas_ledger.json"
    if not lp.is_file():
        raise HTTPException(status_code=404, detail="atlas ledger not found")
    data = json.loads(lp.read_text(encoding="utf-8"))
    ms = data.get("measurements", [])
    if line:
        ms = [m for m in ms if m.get("line") == line]
    total = len(ms)
    off = max(0, int(offset))
    n = max(1, min(int(limit), 200))
    page = []
    for m in ms[-(off + n): -off or None] if off else ms[-n:]:
        page.append({"phase": m.get("phase"), "name": m.get("name"), "line": m.get("line"),
                     "date": m.get("date"), "model": m.get("model"),
                     "verdict": m.get("verdict"), "evidence_level": m.get("evidence_level"),
                     "model_scope": m.get("model_scope"), "prereg_id": m.get("prereg_id")})
    page.reverse()
    return {"schema": "atlas_ledger_v1", "total": total, "count": len(page),
            "offset": off, "ledger_sha256_8": data.get("ledger_sha256_8"),
            "schema_version": data.get("schema_version"), "measurements": page}


@router.get("/atlas/gaps")
async def atlas_gaps():
    """G 线 GAP 账本最新版（glob gap_ledger_v1_N.json 自动发现，取最大 N）。
    含 GAP-5 final_verdict（三段式收官 UPSTREAM_ABSENT，3237 入册）；
    v1.12 起含 appendix（APPX-1..6，3254 落册：APPX-6=保真收线链+u* 词汇解耦）。
    缺失 404，不现算。"""
    cands = _discover_gap_versions()
    if not cands:
        raise HTTPException(status_code=404, detail="gap ledger not found")
    n, p = cands[-1]
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail="gap ledger parse error: %s" % e)
    versions = [{"version": "1.%d" % k, "file": pp.name}
                for k, pp in reversed(cands)]
    return {"schema": "atlas_gaps_v1", "version": data.get("version"),
            "file": p.name, "versions": versions,
            "updated": data.get("updated"),
            "supersedes": data.get("supersedes"),
            "gaps": data.get("gaps", []),
            "appendix": data.get("appendix", [])}


@router.get("/atlas/tree")
async def atlas_tree(schema: str = None):
    """三域树派生视图（atlas_tree_v1：KN/AR/GR → 族 → 特征 + MEC 机制横切 + cross_link 交叉边）。
    由 server/derive_atlas_tree.py 从 registry 最新版 + ledger 派生（只读；缺失 404，不现算）。
    ?schema=v2 → atlas_tree_v2（三模块视图：语言模板 × 编码机制关联 + 缺口，scheme design/atlas_module_view_scheme_v1.md）。"""
    fn = "atlas_tree_v2.json" if schema == "v2" else "atlas_tree_v1.json"
    tp = _ATLAS_DIR / fn
    if not tp.is_file():
        raise HTTPException(status_code=404, detail="%s not found (run server/derive_atlas_tree.py)" % fn)
    try:
        data = json.loads(tp.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail="atlas tree parse error: %s" % e)
    return data


# ------------------------------------------------------------
# 神经元 3D（neuron3d.v1，design/atlas_neuron3d_scheme_v1.md §4–5）
# 数据 = 封存 collect npz（phase3169 g5a6_oov_panel）实测全场激活 H[template,row,layer,dim]；
# 注意第 0 维是 **模板轴**（3 个句式），不是 seed 轴（collect 写法 H16=zeros((NT,NP_,NH,D)) 已核）；
# 行语义 = 73 实体 × 10 类别，还原见 server/neuron_panel.py。
# 标签纪律 = registry_only：端点只回数字证据与 registry 回链，不生成自然语言标签。
# ------------------------------------------------------------
_NEURON3D_PATH = _ATLAS_DIR / "atlas_neuron3d_v1.json"
_NPZ_CACHE: Dict[str, Any] = {}   # model -> {"H": ndarray float16, "grid": {...}}（单槽，换模型即换出）


def _neuron3d_doc() -> Dict[str, Any]:
    if not _NEURON3D_PATH.is_file():
        raise HTTPException(status_code=404,
                            detail="atlas_neuron3d_v1.json not found (run server/derive_neuron3d.py)")
    try:
        return json.loads(_NEURON3D_PATH.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail="neuron3d parse error: %s" % e)


def _neuron3d_H(model: str):
    """懒加载封存 npz 的 H（float16 原样缓存；统计在切片上转 float32），单模型槽位。
    H[t, row, layer, dim]：t=模板（3），row=实体×类别（730）。"""
    import numpy as np
    if model in _NPZ_CACHE:
        return _NPZ_CACHE[model]
    doc = _neuron3d_doc()
    ent = (doc.get("npz") or {}).get(model)
    if not ent:
        raise HTTPException(status_code=404, detail="no sealed npz manifest for model %s" % model)
    p = Path(__file__).parent.parent / ent["rel_path"]
    if not p.is_file():
        raise HTTPException(status_code=404, detail="sealed npz missing: %s" % ent["rel_path"])
    try:
        z = np.load(str(p), allow_pickle=False)
        H = z["H"]
    except Exception as e:
        raise HTTPException(status_code=500, detail="npz load error: %s" % e)
    _NPZ_CACHE.clear()          # 单槽：内存友好（14b float16 ≈ 0.9GB）
    _NPZ_CACHE[model] = {"H": H, "grid": (doc.get("grid") or {}).get(model) or {}}
    return _NPZ_CACHE[model]


@router.get("/atlas/neuron3d/{model}")
async def atlas_neuron3d(model: str):
    """neuron3d.v1 派生段（按模型过滤）：grid 实测形状 + registered 白名单 + npz 清单。
    派生器=server/derive_neuron3d.py（--check 幂等）；缺失 404，不现算。"""
    doc = _neuron3d_doc()
    if model not in (doc.get("grid") or {}):
        raise HTTPException(status_code=404, detail="model %s not in neuron3d grid" % model)
    return {"schema": doc.get("schema"), "design": doc.get("design"),
            "registry_version": doc.get("registry_version"),
            "model": model, "grid": doc["grid"][model],
            "points": doc.get("points", []),
            "npz": (doc.get("npz") or {}).get(model),
            "labels": doc.get("labels"), "coords_meaning": doc.get("coords_meaning")}


@router.get("/atlas/neuron/{model}/{layer}/slice")
async def atlas_neuron_slice(model: str, layer: int):
    """单层激活切片：每维中位数（对 template×row 池化）+ 分位带，供 3D 点云着色。
    2560/4096/5120 维，JSON 约 40–80KB；层越界 404。

    附带 factor_mean：本层全部坐标的因子构成均值（模板/实体/类别 + 交互）。
    面板是 73 实体 × 10 类别 × 3 模板 的完全交叉设计，因此分解精确闭合（份额和=1）。"""
    import numpy as np
    ent = _neuron3d_H(model)
    g = ent["grid"] or {}
    n_layers = int(g.get("n_layers") or 0)
    if not (0 <= layer < n_layers):
        raise HTTPException(status_code=404, detail="layer %s out of range [0,%s)" % (layer, n_layers))
    H = ent["H"]
    col = H[:, :, layer, :]                      # (template,row,dim) float16 视图
    med = np.median(col, axis=(0, 1)).astype(np.float32)   # (dim,)
    p90 = np.percentile(col, 90, axis=(0, 1)).astype(np.float32)
    p10 = np.percentile(col, 10, axis=(0, 1)).astype(np.float32)
    factor_mean = None
    try:
        from server import neuron_panel as _panel
    except Exception:
        try:
            import neuron_panel as _panel
        except Exception:
            _panel = None
    if _panel is not None:
        try:
            factor_mean = _panel.layer_factor_means(col)
        except Exception:
            factor_mean = None
    return {"schema": "neuron3d_slice_v1", "model": model, "layer": layer,
            "n_dims": int(med.shape[0]), "pool": "median_over_template_row",
            "axis_semantics": "H[template,row,layer,dim]；第 0 维是模板轴（3 句式），不是 seed 轴",
            "values": [round(float(v), 4) for v in med],
            "p10": [round(float(v), 4) for v in p10],
            "p90": [round(float(v), 4) for v in p90],
            "factor_mean": factor_mean,
            "note": "封存 npz（phase3169）实测；values=原生坐标索引顺序，不重排"}


@router.get("/atlas/neuron/{model}/{layer}/{dim}")
async def atlas_neuron(model: str, layer: int, dim: int):
    """单神经元详情：激活统计 + 正/负极（极点分析口径）+ registered 回链（registry_only）。
    无入册依据时不返回任何文字解释，只回数字与回链。"""
    import numpy as np
    doc = _neuron3d_doc()
    ent = _neuron3d_H(model)
    g = ent["grid"] or {}
    n_layers = int(g.get("n_layers") or 0)
    d_model = int(g.get("d_model") or 0)
    if not (0 <= layer < n_layers and 0 <= dim < d_model):
        raise HTTPException(status_code=404,
                            detail="(%s,%s) out of range layers[0,%s) dims[0,%s)" % (layer, dim, n_layers, d_model))
    H = ent["H"]
    col = H[:, :, layer, dim].astype(np.float32)          # (template,row)
    flat = col.reshape(-1)
    n_templates, n_rows = col.shape
    k = min(12, flat.shape[0])

    try:
        from server import neuron_panel as _panel
    except Exception:
        try:
            import neuron_panel as _panel
        except Exception:
            _panel = None

    # 正/负极 = 真实片段（模板与实体表封存在同一 phase 的 execution.json，确定性还原）
    poles_payload = None
    factor = None
    if _panel is not None:
        try:
            pr = _panel.poles(H, layer, dim, k=6)
            poles_payload = {"positive": pr["positive"], "negative": pr["negative"],
                             "method": pr["method"], "axis_semantics": pr["axis_semantics"],
                             "text_source": "封存 execution.json：text = TPL[t].format(e=实体, c=类别)"}
            factor = _panel.factor_of_column(H[:, :, layer, dim])
        except Exception:
            poles_payload, factor = None, None
    if poles_payload is None:      # 面板索引缺失时退化到数值极点，并如实标注无文本
        order = np.argsort(flat)
        def _pole(items):
            return [{"template": int(i // n_rows), "row": int(i % n_rows),
                     "value": round(float(flat[i]), 4)} for i in items]
        poles_payload = {"positive": _pole(order[::-1][:k]), "negative": _pole(order[:k]),
                         "method": "argtop/argbottom over template×row；样例文本不可还原（面板索引缺失）"}

    # registered 回链：readout_layer 标记按层匹配；feature 标记为特征级（无原生坐标），仅列全体
    reg_readout = [p for p in doc.get("points", [])
                   if p.get("kind") == "readout_layer" and p.get("model") == model and p.get("layer") == layer]
    reg_features = [p for p in doc.get("points", []) if p.get("kind") == "feature"]
    return {
        "schema": "neuron3d_point_v1", "model": model, "layer": layer, "dim": dim,
        "coords_meaning": doc.get("coords_meaning"),
        "stats": {"n_templates": n_templates, "n_rows": n_rows,
                  "min": round(float(flat.min()), 4), "p10": round(float(np.percentile(flat, 10)), 4),
                  "median": round(float(np.median(flat)), 4), "mean": round(float(flat.mean()), 4),
                  "p90": round(float(np.percentile(flat, 90)), 4), "max": round(float(flat.max()), 4)},
        "poles": poles_payload,
        "factor": factor,
        "registered": {"readout_layer": reg_readout,
                       "features_here": [],
                       "note": "registry v%s 未登记原生 (layer,dim) 特征（内在维度普查=挂账 3238-A）；"
                               "本点当前无特征标签回链" % doc.get("registry_version")},
        "registered_features_index": [{"id": p.get("id"), "family": p.get("family"),
                                       "e_level": p.get("e_level")} for p in reg_features],
        "labels": doc.get("labels"),
    }


# ------------------------------------------------------------
# 谱图预测保真度曲线（fidelity.v2；Phase 3239/3240/3246-3250 封存产物只读）
# 数据 = phase3239 fidelity.npz（逐行 KL×3 基线）+ phase3240 extrap.npz（F0/F1/F2 逐配方
# fam_fid / R² / KL 聚合矩阵）+ phase3246-3250 各 npz（逐配方 fam_fid / R² 聚合矩阵，
# agg_kl 取自各自封存 result.json）+ 现役配方标注（3250 XC_p5_b8，服务端只读封存数字）。
# 只读；缺失 404，不现算。
# ------------------------------------------------------------
_FIDELITY_SOURCES = [
    ("3239", "phase3239/g5h2_fidelity_minloop", "fidelity.npz"),
    ("3240", "phase3240/g5h3_cls_extrapolation", "extrap.npz"),
    ("3246", "phase3246/g5h9_regime_boundary", "boundary.npz"),
    ("3247", "phase3247/g5h10_oov_info_menu", "menu.npz"),
    ("3248", "phase3248/g5h11_poly_deepening", "poly.npz"),
    ("3249", "phase3249/g5h12_deg_extension", "degext.npz"),
    ("3250", "phase3250/g5h13_kernel_boundary", "kbound.npz"),
]
# 现役配方（G5 保真缺口收线的当前最优声明配方；数字全部取自封存 result.json）
_FIDELITY_INCUMBENT = [
    {"phase": "3250", "recipe": "XC_p5_b8", "tag": "现役综合最优"},
    {"phase": "3250", "recipe": "XC_p5_b4", "tag": "保真维现役"},
    {"phase": "3250", "recipe": "XC_p3_b6", "tag": "R²/KL 维现役"},
    {"phase": "3248", "recipe": "K_p3_T3", "tag": "纯拉力最优"},
]
_FIDELITY_FAMS = ["DVU", "DCC", "ENT", "USTAR", "SCLS", "MASSIVE"]


@router.get("/atlas/fidelity")
async def atlas_fidelity(model: str = "qwen3-4b"):
    """保真度曲线（「破解程度」可画曲线的数字）：census 六族族方向保真 + 逐行下一 token
    KL + R²。数据全部来自封存 phase 产物（零 GPU 复算）；缺失 404。"""
    import numpy as np
    base = Path(__file__).parent.parent / "tests" / "glm5" / "result" / "rdc_query_construction_20260913"
    out = {"schema": "atlas_fidelity_v2", "model": model, "fam_order": _FIDELITY_FAMS,
           "phases": [], "incumbent": []}
    for ph, rel, npz_name in _FIDELITY_SOURCES:
        pdir = base / rel
        rp = pdir / "result.json"
        if not rp.is_file():
            continue
        try:
            rj = json.loads(rp.read_text(encoding="utf-8"))
        except Exception as e:
            raise HTTPException(status_code=500, detail="phase%s result parse error: %s" % (ph, e))
        if rj.get("model") != model:
            continue
        entry = {"phase": ph, "name": rj.get("name"), "verdict": rj.get("verdict"),
                 "design_sha8": rj.get("design_sha8"), "res_sha8": rj.get("res_sha8"),
                 "kl_rows": rj.get("kl_rows"),
                 "aggregate": rj.get("aggregate")}
        npz_p = pdir / npz_name
        if npz_p.is_file():
            try:
                z = np.load(str(npz_p), allow_pickle=False)
                if ph == "3239":
                    # 逐行 KL（seed 7 全量 + 3 seed 均值），round 6 位控体积
                    kl = {}
                    for k3 in ("model", "B0", "B1"):
                        arrs = [z["seed%d_kl_%s" % (s, k3)] for s in (7, 8, 9)]
                        kl[k3] = [round(float(v), 6) for v in np.mean(arrs, axis=0)]
                    entry["kl_per_row_mean3seed"] = kl
                    entry["fam_fid_per_seed"] = z["fam_fid"].tolist()
                else:
                    entry["recipes"] = [str(x) for x in z["recipes"]]
                    entry["agg_fid_matrix"] = [[round(float(v), 6) for v in row]
                                               for row in z["agg_fid_matrix"]]
                    entry["agg_r2"] = [round(float(v), 6) for v in z["agg_r2"]]
                    if ph == "3240":
                        entry["agg_kl"] = [round(float(v), 6) for v in z["agg_kl"]]
                    else:
                        # 3246+ 的 npz 不含 agg_kl：从封存 result.json aggregate 读取
                        agg = rj.get("aggregate") or {}
                        entry["agg_kl"] = [round(float(agg.get(rn, {}).get("kl_mean", 0.0)), 6)
                                           for rn in entry["recipes"]]
                    if "closure" in z:
                        entry["closure"] = [round(float(v), 6) for v in z["closure"]]
                    entry["fam_per_seed"] = {rn: z["seed%d_fid_%s" % (s, rn)].tolist()
                                             for s in (7, 8, 9) for rn in
                                             [str(x) for x in z["recipes"]]}
            except Exception as e:
                raise HTTPException(status_code=500, detail="phase%s npz parse error: %s" % (ph, e))
        out["phases"].append(entry)
    # 现役配方标注（只读封存 aggregate 数字；缺失阶段跳过）
    by_phase = {p["phase"]: p for p in out["phases"]}
    for inc in _FIDELITY_INCUMBENT:
        p = by_phase.get(inc["phase"])
        if not p or "recipes" not in p:
            continue
        i = p["recipes"].index(inc["recipe"]) if inc["recipe"] in p["recipes"] else -1
        if i < 0:
            continue
        fid = p["agg_fid_matrix"][i]
        out["incumbent"].append({
            "phase": inc["phase"], "recipe": inc["recipe"], "tag": inc["tag"],
            "fam_fid": dict(zip(_FIDELITY_FAMS, [round(float(v), 6) for v in fid])),
            "r2_oov": p["agg_r2"][i], "kl_mean": p["agg_kl"][i],
        })
    if not out["phases"]:
        raise HTTPException(status_code=404,
                            detail="no sealed fidelity artifacts for model %s (run phase 3239/3240)" % model)
    return out


@router.get("/queue/gline")
async def gline_queue():
    """G 线五段式队列：closed = atlas_ledger 尾部 phase（G 线），queued/preregistered =
    research/gpt5/atlas/gline_queue_v1.json（缺失返回空，不炸）。"""
    items, closed = [], []
    lp = _ATLAS_DIR / "atlas_ledger.json"
    if lp.is_file():
        try:
            data = json.loads(lp.read_text(encoding="utf-8"))
            for m in data.get("measurements", []):
                if m.get("line") != "G":
                    continue
                closed.append({"id": "p%s" % m.get("phase"), "phase": m.get("phase"),
                               "title": m.get("name"), "status": "closed",
                               "evidence_level": m.get("evidence_level"),
                               "prereg_id": m.get("prereg_id"),
                               "verdict": m.get("verdict"), "date": m.get("date")})
        except Exception:
            closed = []
    qp = _ATLAS_DIR / "gline_queue_v1.json"
    if qp.is_file():
        try:
            q = json.loads(qp.read_text(encoding="utf-8"))
            for it in q.get("queue", []):
                if isinstance(it, dict):
                    items.append({"id": it.get("id"), "phase": it.get("phase"),
                                  "title": it.get("title"), "status": it.get("status"),
                                  "gpu": it.get("gpu"), "note": it.get("note"),
                                  "prereg": it.get("prereg")})
        except Exception:
            pass
    return {"schema": "gline_queue_v1", "closed_count": len(closed),
            "closed_tail": closed[-12:], "queue": items}
