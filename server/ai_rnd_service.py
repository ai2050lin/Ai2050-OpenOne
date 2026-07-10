"""
AI R&D Service — 自动深度神经网络逆向工程研究服务

核心功能:
1. 管理AI主模型和分析师模型的配置和提示词
2. 驱动5阶段自动研究循环: 分析→规划→生成→执行→总结
3. SSE实时推送研究事件到前端
4. 安全执行AI生成的代码
"""
import asyncio
import json
import os
import sys
import time
import traceback
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

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

        # 加载保存的配置
        self._load_config()
        self._load_state()

    def _config_path(self) -> Path:
        return Path(__file__).parent.parent / "ai_rnd_config.json"

    def _state_path(self) -> Path:
        return Path(__file__).parent.parent / "tests" / "result" / "auto_rnd" / "session_state.json"

    def _load_config(self):
        path = self._config_path()
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self.config = json.load(f)
            except Exception:
                self.config = {}

    def _save_config(self):
        path = self._config_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

    def _load_state(self):
        path = self._state_path()
        if not path.exists():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.round = int(payload.get("round", 0))
            self.findings = list(payload.get("findings", []))
            self.research_state = dict(payload.get("research_state", {}))
            self.active_run_id = payload.get("active_run_id")
            self.started_at = payload.get("started_at")
        except Exception:
            return

    def persist_state(self):
        path = self._state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "ai_rnd_session.v2",
            "status": self.status,
            "mode": self.mode,
            "current_phase": self.current_phase,
            "round": self.round,
            "started_at": self.started_at,
            "active_run_id": self.active_run_id,
            "findings": self.findings,
            "research_state": self.research_state,
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
            "round": self.round,
            "started_at": self.started_at,
            "findings_count": len(self.findings),
            "research_state": self.research_state,
            "active_run_id": self.active_run_id,
            "evidence_context": load_evidence_context(),
            "agent_roles": ROLE_DEFINITIONS,
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

async def _wait_for_step(session: ResearchSession, after_phase: str = ""):
    """手动模式: 等待用户点击下一步"""
    if session.mode != "manual":
        return
    session.status = "waiting_step"
    session.push_event("status_change", status="waiting_step", waiting_after=after_phase)
    session._step_event.clear()
    await session._step_event.wait()  # Wait until user clicks "step"
    session.status = "running"
    session.push_event("status_change", status="running")
    session._step_event.clear()


async def run_research_loop(session: ResearchSession):
    """5阶段自动研究循环(支持自动/手动模式)"""
    session.status = "running"
    session.started_at = time.time()
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
        session.push_event("status_change", status="stopped")
        return

    while not session._stop_requested:
        # Check pause
        if session._pause_requested:
            session.status = "paused"
            session.push_event("status_change", status="paused")
            while session._pause_requested and not session._stop_requested:
                await asyncio.sleep(1)
            if session._stop_requested:
                break
            session.status = "running"
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

        # Manual: wait at end of round
        await _wait_for_step(session, "summarize")

        # Brief pause between rounds (auto mode only)
        if session.mode == "auto":
            await asyncio.sleep(2)

    # Loop ended
    session.status = "stopped"
    session.current_phase = None
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
    if session.status == "running":
        raise HTTPException(400, "Session already running")
    objective = (payload.objective if payload else "").strip()
    if objective:
        session.research_state["research_objective"] = objective
        session.push_event("objective", message=f"研究目标: {objective}", objective=objective)
        session.persist_state()
    if session._pause_requested:
        session._pause_requested = False
        return {"status": "resumed"}

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
    session.persist_state()
    return {"status": "stopped"}


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
        "schema_version": "research_orchestrator_status.v1",
        "active_run_id": session.active_run_id,
        "model_order": ["qwen3", "glm4", "deepseek7b"],
        "gpu_policy": "single_process_serial",
        "roles": ROLE_DEFINITIONS,
        "evidence_context": load_evidence_context(),
        "recent_runs": list_research_runs(20),
    }


@router.get("/orchestrator/runs")
async def get_orchestrator_runs(limit: int = 50):
    return {"runs": list_research_runs(max(1, min(limit, 200)))}
