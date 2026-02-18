import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import numpy as np
import torch

from server.runtime.contracts import (
    AnalysisSpec,
    ConclusionCard,
    EventEnvelope,
    Metric,
    RunSummary,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _parse_percent(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.endswith("%"):
        text = text[:-1]
    return _safe_float(text, default=0.0)


def _normalize_topology_model_name(model_name: Optional[str]) -> str:
    raw = (model_name or "gpt2").strip().lower()
    if "qwen" in raw:
        return "qwen3"
    if "gpt2" in raw:
        return "gpt2"
    aliases = {
        "gpt2-small": "gpt2",
        "gpt-2": "gpt2",
        "qwen": "qwen3",
        "qwen3-4b": "qwen3",
    }
    return aliases.get(raw, raw or "gpt2")


def _topology_path_candidates(model_name: Optional[str]) -> List[str]:
    normalized = _normalize_topology_model_name(model_name)
    if normalized == "gpt2":
        return [
            "tempdata/topology.json",
            "tempdata/topology_generated.json",
        ]
    return [
        f"tempdata/topology_{normalized}.json",
        f"tempdata/topology_generated_{normalized}.json",
    ]


def _resolve_topology_payload(model_name: Optional[str]) -> Dict[str, Any]:
    candidates = _topology_path_candidates(model_name)
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict) and isinstance(payload.get("layers"), dict):
            return {
                "model": _normalize_topology_model_name(model_name),
                "path": path,
                "layers": payload["layers"],
                "searched": candidates,
            }
        nested = payload.get("data") if isinstance(payload, dict) else None
        if isinstance(nested, dict) and isinstance(nested.get("layers"), dict):
            return {
                "model": _normalize_topology_model_name(model_name),
                "path": path,
                "layers": nested["layers"],
                "searched": candidates,
            }
    return {
        "model": _normalize_topology_model_name(model_name),
        "path": None,
        "layers": {},
        "searched": candidates,
    }


def _flow_tubes_path_candidates() -> List[str]:
    return [
        "nfb_data/flow_tubes.json",
        "tempdata/flow_tubes.json",
        "tempdata/flow_tubes_generated.json",
    ]


def _default_flow_tubes() -> List[Dict[str, Any]]:
    return [
        {
            "id": "male",
            "label": "Male",
            "color": "#3498db",
            "radius": 0.2,
            "path": [[0, 0, 0], [1, 0.5, 2], [2, 1, 5], [3, 1.5, 8]],
            "metrics": {},
        },
        {
            "id": "female",
            "label": "Female",
            "color": "#e74c3c",
            "radius": 0.2,
            "path": [[0, 0, 0], [1, -0.5, 2], [2, -1, 5], [3, -1.5, 8]],
            "metrics": {},
        },
        {
            "id": "neutral",
            "label": "Neutral",
            "color": "#2ecc71",
            "radius": 0.2,
            "path": [[0, 0, 0], [1, 0, 3], [2, 0, 6], [3, 0, 9]],
            "metrics": {},
        },
    ]


def _normalize_tube_entry(entry: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
    if not isinstance(entry, dict):
        return None
    raw_path = entry.get("path")
    if not isinstance(raw_path, list):
        raw_path = entry.get("points")
    if not isinstance(raw_path, list):
        return None

    path: List[List[float]] = []
    for point in raw_path:
        if not isinstance(point, list) or len(point) < 3:
            continue
        normalized = [
            _safe_float(point[0], 0.0),
            _safe_float(point[1], 0.0),
            _safe_float(point[2], 0.0),
        ]
        path.append(normalized)
    if len(path) < 2:
        return None

    tube_id = str(entry.get("id") or entry.get("label") or f"tube_{index}")
    label = str(entry.get("label") or tube_id.title())
    color = str(entry.get("color") or "#00d2ff")
    radius = max(_safe_float(entry.get("radius", 0.2), 0.2), 0.01)
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}

    return {
        "id": tube_id,
        "label": label,
        "color": color,
        "radius": radius,
        "path": path,
        "metrics": metrics,
    }


def _resolve_flow_tubes_payload() -> Dict[str, Any]:
    candidates = _flow_tubes_path_candidates()
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        raw_tubes = payload.get("tubes") if isinstance(payload, dict) else None
        if not isinstance(raw_tubes, list):
            continue
        tubes: List[Dict[str, Any]] = []
        for idx, entry in enumerate(raw_tubes):
            normalized = _normalize_tube_entry(entry, idx)
            if normalized is not None:
                tubes.append(normalized)
        if tubes:
            return {"path": path, "tubes": tubes, "source": "file", "searched": candidates}

    return {
        "path": None,
        "tubes": _default_flow_tubes(),
        "source": "default",
        "searched": candidates,
    }


def _tda_path_candidates() -> List[str]:
    return [
        "nfb_data/tda_results.json",
        "tempdata/tda_results.json",
    ]


def _default_tda_payload() -> Dict[str, Any]:
    return {"ph_0d": [1.0, 0.8, 0.5], "ph_1d": [0.2, 0.1]}


def _resolve_tda_payload() -> Dict[str, Any]:
    candidates = _tda_path_candidates()
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        ph_0d = payload.get("ph_0d") if isinstance(payload, dict) else None
        ph_1d = payload.get("ph_1d") if isinstance(payload, dict) else None
        if isinstance(ph_0d, list) and isinstance(ph_1d, list):
            return {
                "path": path,
                "source": "file",
                "ph_0d": ph_0d,
                "ph_1d": ph_1d,
                "searched": candidates,
            }
    default_payload = _default_tda_payload()
    return {
        "path": None,
        "source": "default",
        "ph_0d": default_payload["ph_0d"],
        "ph_1d": default_payload["ph_1d"],
        "searched": candidates,
    }


def _orthogonality_error(components: Any) -> float:
    if not isinstance(components, list) or not components:
        return 0.0
    try:
        mat = np.array(components, dtype=np.float64)
        if mat.ndim != 2:
            return 0.0
        gram = mat @ mat.T
        eye = np.eye(gram.shape[0], dtype=np.float64)
        return float(np.mean(np.abs(gram - eye)))
    except Exception:
        return 0.0


def _build_topology_summary(layers: Any) -> Dict[str, Dict[str, float]]:
    if not isinstance(layers, dict):
        return {}
    summary: Dict[str, Dict[str, float]] = {}
    for layer_key, layer_data in layers.items():
        if not isinstance(layer_data, dict):
            continue
        pca_points = layer_data.get("pca")
        point_count = float(len(pca_points)) if isinstance(pca_points, list) else 0.0
        betti = layer_data.get("betti")
        betti_count = float(len(betti)) if isinstance(betti, list) else 0.0
        ortho_error = _orthogonality_error(layer_data.get("pca_components"))
        summary[f"layer_{layer_key}"] = {
            "avg_ortho_error": ortho_error,
            "point_count": point_count,
            "betti_count": betti_count,
        }
    return summary


@dataclass
class RunContext:
    run_id: str
    agi_core: Optional[Any] = None
    model: Optional[Any] = None
    started_at: float = field(default_factory=time.time)


class RoutePlugin(Protocol):
    route_name: str
    version: str
    display_name: str
    supported_analyses: List[str]

    def prepare(self, spec: AnalysisSpec, ctx: RunContext) -> None:
        ...

    def run(self, spec: AnalysisSpec, ctx: RunContext) -> List[EventEnvelope]:
        ...

    def finalize(
        self, spec: AnalysisSpec, ctx: RunContext, events: List[EventEnvelope]
    ) -> RunSummary:
        ...


class FiberBundleRoute:
    """Minimal phase-1 plugin wrapping current AGI core fiber-bundle path."""

    route_name = "fiber_bundle"
    version = "0.1.0"
    display_name = "Fiber Bundle Route"
    supported_analyses = [
        "unified_conscious_field",
        "topology_snapshot",
        "flow_tubes_snapshot",
        "tda_snapshot",
        "topology_scan_snapshot",
        "debias_snapshot",
    ]

    def __init__(self) -> None:
        self._last_report: Dict[str, Any] = {}
        self._last_topology: Dict[str, Any] = {}
        self._last_flow_tubes: Dict[str, Any] = {}
        self._last_tda: Dict[str, Any] = {}
        self._last_topology_scan: Dict[str, Any] = {}
        self._last_debias: Dict[str, Any] = {}

    def prepare(self, spec: AnalysisSpec, ctx: RunContext) -> None:
        self._last_report = {}
        self._last_topology = {}
        self._last_flow_tubes = {}
        self._last_tda = {}
        self._last_topology_scan = {}
        self._last_debias = {}

    def run(self, spec: AnalysisSpec, ctx: RunContext) -> List[EventEnvelope]:
        if spec.analysis_type == "topology_snapshot":
            return self._run_topology_snapshot(spec, ctx)
        if spec.analysis_type == "flow_tubes_snapshot":
            return self._run_flow_tubes_snapshot(spec, ctx)
        if spec.analysis_type == "tda_snapshot":
            return self._run_tda_snapshot(spec, ctx)
        if spec.analysis_type == "topology_scan_snapshot":
            return self._run_topology_scan_snapshot(spec, ctx)
        if spec.analysis_type == "debias_snapshot":
            return self._run_debias_snapshot(spec, ctx)
        if spec.analysis_type == "unified_conscious_field":
            return self._run_unified_conscious_field(spec, ctx)
        raise ValueError(f"Unsupported analysis type: {spec.analysis_type}")

    def _run_unified_conscious_field(
        self, spec: AnalysisSpec, ctx: RunContext
    ) -> List[EventEnvelope]:
        now = time.time()
        meta = {"route": self.route_name, "analysis_type": spec.analysis_type}

        if ctx.agi_core is None:
            return [
                EventEnvelope(
                    event_type="RouteStatus",
                    run_id=ctx.run_id,
                    step=0,
                    timestamp=now,
                    payload={"status": "unavailable", "reason": "agi_core_not_initialized"},
                    meta=meta,
                )
            ]

        dim = int(getattr(ctx.agi_core, "dim", 32))
        noise_scale = _safe_float(spec.params.get("noise_scale", 1.0), default=1.0)
        step_id = int(spec.params.get("step_id", 0))
        signal = torch.randn(dim) * noise_scale
        report = ctx.agi_core.run_conscious_step(step_id, signal)
        self._last_report = report

        gws_state = []
        if getattr(ctx.agi_core, "gws", None) is not None and getattr(
            ctx.agi_core.gws, "state", None
        ) is not None:
            gws_state = ctx.agi_core.gws.state.tolist()

        events: List[EventEnvelope] = [
            EventEnvelope(
                event_type="ActivationSnapshot",
                run_id=ctx.run_id,
                step=step_id,
                timestamp=time.time(),
                payload={
                    "signal_norm": _safe_float(report.get("signal_norm", 0.0)),
                    "memory_slots": int(report.get("memory_slots", 0)),
                    "gws_state": gws_state,
                },
                meta=meta,
            ),
            EventEnvelope(
                event_type="AlignmentSignal",
                run_id=ctx.run_id,
                step=step_id,
                timestamp=time.time(),
                payload={
                    "winner_module": report.get("gws_winner"),
                    "emotion": report.get("emotion", {}),
                    "energy_saving_pct": _parse_percent(report.get("energy_saving")),
                },
                meta=meta,
            ),
        ]
        return events

    def _run_topology_snapshot(
        self, spec: AnalysisSpec, ctx: RunContext
    ) -> List[EventEnvelope]:
        now = time.time()
        meta = {"route": self.route_name, "analysis_type": spec.analysis_type}
        model_name = spec.model or spec.params.get("model") or "gpt2"
        snapshot = _resolve_topology_payload(model_name)
        self._last_topology = snapshot

        if not snapshot.get("path"):
            return [
                EventEnvelope(
                    event_type="RouteStatus",
                    run_id=ctx.run_id,
                    step=0,
                    timestamp=now,
                    payload={
                        "status": "unavailable",
                        "reason": "topology_not_found",
                        "model": snapshot.get("model"),
                        "searched": snapshot.get("searched", []),
                    },
                    meta=meta,
                )
            ]

        return [
            EventEnvelope(
                event_type="TopologySignal",
                run_id=ctx.run_id,
                step=0,
                timestamp=now,
                payload={
                    "model": snapshot.get("model"),
                    "path": snapshot.get("path"),
                    "layers": snapshot.get("layers", {}),
                },
                meta=meta,
            )
        ]

    def _run_flow_tubes_snapshot(
        self, spec: AnalysisSpec, ctx: RunContext
    ) -> List[EventEnvelope]:
        now = time.time()
        meta = {"route": self.route_name, "analysis_type": spec.analysis_type}
        snapshot = _resolve_flow_tubes_payload()
        self._last_flow_tubes = snapshot
        tubes = snapshot.get("tubes", [])
        if not tubes:
            return [
                EventEnvelope(
                    event_type="RouteStatus",
                    run_id=ctx.run_id,
                    step=0,
                    timestamp=now,
                    payload={
                        "status": "unavailable",
                        "reason": "flow_tubes_not_found",
                        "searched": snapshot.get("searched", []),
                    },
                    meta=meta,
                )
            ]

        return [
            EventEnvelope(
                event_type="FlowTubeSignal",
                run_id=ctx.run_id,
                step=0,
                timestamp=now,
                payload={
                    "tubes": tubes,
                    "source": snapshot.get("source"),
                    "path": snapshot.get("path"),
                },
                meta=meta,
            )
        ]

    def _run_tda_snapshot(
        self, spec: AnalysisSpec, ctx: RunContext
    ) -> List[EventEnvelope]:
        now = time.time()
        meta = {"route": self.route_name, "analysis_type": spec.analysis_type}
        snapshot = _resolve_tda_payload()
        self._last_tda = snapshot
        return [
            EventEnvelope(
                event_type="TDASignal",
                run_id=ctx.run_id,
                step=0,
                timestamp=now,
                payload={
                    "ph_0d": snapshot.get("ph_0d", []),
                    "ph_1d": snapshot.get("ph_1d", []),
                    "source": snapshot.get("source"),
                    "path": snapshot.get("path"),
                },
                meta=meta,
            )
        ]

    def _run_topology_scan_snapshot(
        self, spec: AnalysisSpec, ctx: RunContext
    ) -> List[EventEnvelope]:
        now = time.time()
        meta = {"route": self.route_name, "analysis_type": spec.analysis_type}
        model_name = spec.model or spec.params.get("model") or "gpt2"
        snapshot = _resolve_topology_payload(model_name)
        layers = snapshot.get("layers", {})
        summary = _build_topology_summary(layers)
        self._last_topology_scan = {
            "model": snapshot.get("model"),
            "path": snapshot.get("path"),
            "searched": snapshot.get("searched", []),
            "summary": summary,
            "layers": layers,
        }

        if not snapshot.get("path"):
            return [
                EventEnvelope(
                    event_type="RouteStatus",
                    run_id=ctx.run_id,
                    step=0,
                    timestamp=now,
                    payload={
                        "status": "unavailable",
                        "reason": "topology_scan_source_not_found",
                        "searched": snapshot.get("searched", []),
                        "model": snapshot.get("model"),
                    },
                    meta=meta,
                )
            ]

        return [
            EventEnvelope(
                event_type="TopologyScanSignal",
                run_id=ctx.run_id,
                step=0,
                timestamp=now,
                payload={
                    "status": "success",
                    "model": snapshot.get("model"),
                    "path": snapshot.get("path"),
                    "source": "topology_cache",
                    "summary": summary,
                    "layer_count": len(layers) if isinstance(layers, dict) else 0,
                },
                meta=meta,
            )
        ]

    def _run_debias_snapshot(
        self, spec: AnalysisSpec, ctx: RunContext
    ) -> List[EventEnvelope]:
        now = time.time()
        meta = {"route": self.route_name, "analysis_type": spec.analysis_type}
        source_text = (
            spec.input_payload.get("source")
            or spec.params.get("source")
            or "The doctor finished the shift."
        )
        layer_idx = int(spec.params.get("layer_idx", spec.input_payload.get("layer_idx", 6)))
        top_k = int(spec.params.get("top_k", 5))
        top_k = max(1, min(top_k, 20))

        if ctx.model is None:
            self._last_debias = {"searched": [], "result_count": 0}
            return [
                EventEnvelope(
                    event_type="RouteStatus",
                    run_id=ctx.run_id,
                    step=0,
                    timestamp=now,
                    payload={
                        "status": "unavailable",
                        "reason": "model_not_initialized",
                    },
                    meta=meta,
                )
            ]

        try:
            from server.debias_engine import GeometricInterceptor

            d_model = int(ctx.model.cfg.d_model)
            raw_r = spec.params.get("R", spec.input_payload.get("R"))
            r_np = np.array(raw_r, dtype=np.float64) if raw_r is not None else np.empty((0, 0))
            if r_np.ndim != 2 or r_np.shape[0] != r_np.shape[1] or r_np.shape[0] != d_model:
                q, _ = np.linalg.qr(np.random.randn(d_model, d_model))
                r_np = q.astype(np.float32)
            else:
                r_np = r_np.astype(np.float32)

            ctx.model.reset_hooks()
            logits_base = ctx.model(source_text)
            probs_base = torch.softmax(logits_base[0, -1, :], dim=-1)

            interceptor = GeometricInterceptor(ctx.model)
            interceptor.clear()
            interceptor.add_interception(layer_idx, r_np)
            interceptor.apply_hooks()
            logits_debiased = ctx.model(source_text)
            probs_debiased = torch.softmax(logits_debiased[0, -1, :], dim=-1)
            interceptor.clear()

            top_vals, top_idxs = torch.topk(probs_base, top_k)
            rows: List[Dict[str, Any]] = []
            for i in range(top_k):
                idx = int(top_idxs[i].item())
                p_base = float(top_vals[i].item())
                p_debiased = float(probs_debiased[idx].item())
                rows.append(
                    {
                        "token": ctx.model.to_string(idx),
                        "prob_base": p_base,
                        "prob_debiased": p_debiased,
                        "shift": p_debiased - p_base,
                    }
                )

            self._last_debias = {
                "result_count": len(rows),
                "layer_idx": layer_idx,
                "top_k": top_k,
            }
            return [
                EventEnvelope(
                    event_type="DebiasSignal",
                    run_id=ctx.run_id,
                    step=0,
                    timestamp=time.time(),
                    payload={
                        "status": "success",
                        "layer": layer_idx,
                        "source_prompt": source_text,
                        "results": rows,
                    },
                    meta=meta,
                )
            ]
        except Exception as exc:
            self._last_debias = {"error": str(exc), "result_count": 0}
            return [
                EventEnvelope(
                    event_type="RouteStatus",
                    run_id=ctx.run_id,
                    step=0,
                    timestamp=time.time(),
                    payload={
                        "status": "error",
                        "reason": "debias_runtime_failed",
                        "message": str(exc),
                    },
                    meta=meta,
                )
            ]

    def finalize(
        self, spec: AnalysisSpec, ctx: RunContext, events: List[EventEnvelope]
    ) -> RunSummary:
        if spec.analysis_type == "topology_snapshot":
            return self._finalize_topology_snapshot(spec, events)
        if spec.analysis_type == "flow_tubes_snapshot":
            return self._finalize_flow_tubes_snapshot(events)
        if spec.analysis_type == "tda_snapshot":
            return self._finalize_tda_snapshot(events)
        if spec.analysis_type == "topology_scan_snapshot":
            return self._finalize_topology_scan_snapshot(spec, events)
        if spec.analysis_type == "debias_snapshot":
            return self._finalize_debias_snapshot(events)
        return self._finalize_unified_conscious_field(spec, ctx, events)

    def _finalize_unified_conscious_field(
        self, spec: AnalysisSpec, ctx: RunContext, events: List[EventEnvelope]
    ) -> RunSummary:
        if ctx.agi_core is None:
            return RunSummary(
                metrics=[],
                conclusion=ConclusionCard(
                    objective="Execute unified conscious field analysis.",
                    method="Route plugin execution against AGI core.",
                    evidence=["AGI core not initialized during run."],
                    result="Run did not execute algorithmic steps.",
                    confidence=0.0,
                    limitations=["Runtime dependency agi_core is missing."],
                    next_action="Initialize AGI core in server lifespan and retry.",
                ),
                artifacts=[],
            )

        emotion = self._last_report.get("emotion", {})
        metrics = [
            Metric(
                key="signal_norm",
                value=_safe_float(self._last_report.get("signal_norm", 0.0)),
                min_value=0.0,
                description="Norm of environmental signal.",
            ),
            Metric(
                key="energy_saving_pct",
                value=_parse_percent(self._last_report.get("energy_saving")),
                unit="%",
                min_value=0.0,
                max_value=100.0,
                description="Sparse attention energy saving estimate.",
            ),
            Metric(
                key="emotion_stability",
                value=_safe_float(emotion.get("stability", 0.0)),
                min_value=0.0,
                max_value=1.0,
                description="Homeostasis stability from emotion engine.",
            ),
            Metric(
                key="memory_slots",
                value=_safe_float(self._last_report.get("memory_slots", 0.0)),
                min_value=0.0,
                description="Occupied holographic memory slots.",
            ),
        ]

        winner = self._last_report.get("gws_winner", "unknown")
        confidence = 0.45 if events else 0.2
        conclusion = ConclusionCard(
            objective="Observe one conscious cycle under fiber-bundle route.",
            method="Run AGI core cycle and emit unified events for visualization.",
            evidence=[
                f"Winner module: {winner}",
                f"Signal norm: {metrics[0].value:.4f}",
                f"Energy saving: {metrics[1].value:.2f}%",
            ],
            result="Phase-1 route plugin executed and produced protocol-compatible events.",
            confidence=confidence,
            limitations=[
                "Single-step synthetic signal only.",
                "No multi-route comparator in phase-1.",
            ],
            next_action="Run multi-step execution and compare against a second route plugin.",
        )
        return RunSummary(metrics=metrics, conclusion=conclusion, artifacts=[])

    def _finalize_topology_snapshot(
        self, spec: AnalysisSpec, events: List[EventEnvelope]
    ) -> RunSummary:
        topology_event = next(
            (event for event in events if event.event_type == "TopologySignal"),
            None,
        )
        if topology_event is None:
            searched = self._last_topology.get("searched", [])
            model_name = self._last_topology.get(
                "model", _normalize_topology_model_name(spec.model)
            )
            return RunSummary(
                metrics=[],
                conclusion=ConclusionCard(
                    objective="Load topology snapshot for visualization.",
                    method="Resolve model topology file and emit topology event.",
                    evidence=[f"Searched paths: {searched}"],
                    result=f"No topology snapshot available for model '{model_name}'.",
                    confidence=0.0,
                    limitations=["Topology data not generated yet."],
                    next_action="Run topology generation endpoint before retry.",
                ),
                artifacts=[],
            )

        layers = topology_event.payload.get("layers", {})
        layer_count = len(layers)
        point_count = 0
        for layer_data in layers.values():
            points = layer_data.get("pca") or layer_data.get("projections") or []
            if isinstance(points, list):
                point_count += len(points)

        metrics = [
            Metric(
                key="topology_layer_count",
                value=float(layer_count),
                min_value=0.0,
                description="Number of layers available in topology snapshot.",
            ),
            Metric(
                key="topology_point_count",
                value=float(point_count),
                min_value=0.0,
                description="Total projected points across all layers.",
            ),
        ]
        conclusion = ConclusionCard(
            objective="Provide topology snapshot through route-agnostic protocol.",
            method="Read cached topology file and emit TopologySignal event.",
            evidence=[
                f"Model: {topology_event.payload.get('model')}",
                f"Layers: {layer_count}",
                f"Points: {point_count}",
            ],
            result="Topology snapshot is available via runtime events.",
            confidence=0.85,
            limitations=[
                "Snapshot is file-based and not computed in-run.",
            ],
            next_action="Enable live topology generation as streaming events.",
        )
        return RunSummary(
            metrics=metrics,
            conclusion=conclusion,
            artifacts=[
                {
                    "type": "topology_file",
                    "path": topology_event.payload.get("path"),
                }
            ],
        )

    def _finalize_flow_tubes_snapshot(self, events: List[EventEnvelope]) -> RunSummary:
        flow_event = next(
            (event for event in events if event.event_type == "FlowTubeSignal"),
            None,
        )
        if flow_event is None:
            searched = self._last_flow_tubes.get("searched", [])
            return RunSummary(
                metrics=[],
                conclusion=ConclusionCard(
                    objective="Load flow tubes snapshot for visualization.",
                    method="Resolve flow tube file and emit FlowTubeSignal event.",
                    evidence=[f"Searched paths: {searched}"],
                    result="No flow tube data available.",
                    confidence=0.0,
                    limitations=["Flow tube snapshot file missing."],
                    next_action="Generate or export flow tube data before retry.",
                ),
                artifacts=[],
            )

        tubes = flow_event.payload.get("tubes", [])
        tube_count = len(tubes)
        point_count = 0
        for tube in tubes:
            path = tube.get("path")
            if isinstance(path, list):
                point_count += len(path)

        metrics = [
            Metric(
                key="flow_tube_count",
                value=float(tube_count),
                min_value=0.0,
                description="Number of flow tubes in snapshot.",
            ),
            Metric(
                key="flow_tube_point_count",
                value=float(point_count),
                min_value=0.0,
                description="Total number of points across all tubes.",
            ),
        ]
        conclusion = ConclusionCard(
            objective="Provide flow tube snapshot through runtime protocol.",
            method="Load flow tube cache and emit FlowTubeSignal event.",
            evidence=[
                f"Tubes: {tube_count}",
                f"Points: {point_count}",
                f"Source: {flow_event.payload.get('source')}",
            ],
            result="Flow tube snapshot is available via runtime events.",
            confidence=0.8,
            limitations=["Snapshot is static until regenerated."],
            next_action="Add streaming updates for evolving flow tubes.",
        )
        return RunSummary(
            metrics=metrics,
            conclusion=conclusion,
            artifacts=[
                {
                    "type": "flow_tubes_file",
                    "path": flow_event.payload.get("path"),
                }
            ],
        )

    def _finalize_tda_snapshot(self, events: List[EventEnvelope]) -> RunSummary:
        tda_event = next(
            (event for event in events if event.event_type == "TDASignal"),
            None,
        )
        if tda_event is None:
            searched = self._last_tda.get("searched", [])
            return RunSummary(
                metrics=[],
                conclusion=ConclusionCard(
                    objective="Load TDA snapshot for visualization.",
                    method="Resolve TDA cache and emit TDASignal event.",
                    evidence=[f"Searched paths: {searched}"],
                    result="No TDA snapshot available.",
                    confidence=0.0,
                    limitations=["TDA snapshot file missing."],
                    next_action="Generate TDA results before retry.",
                ),
                artifacts=[],
            )

        ph_0d = tda_event.payload.get("ph_0d", [])
        ph_1d = tda_event.payload.get("ph_1d", [])
        metrics = [
            Metric(
                key="tda_ph0_count",
                value=float(len(ph_0d)),
                min_value=0.0,
                description="Number of 0D persistence entries.",
            ),
            Metric(
                key="tda_ph1_count",
                value=float(len(ph_1d)),
                min_value=0.0,
                description="Number of 1D persistence entries.",
            ),
        ]
        conclusion = ConclusionCard(
            objective="Provide TDA snapshot through runtime protocol.",
            method="Load cached TDA result and emit TDASignal event.",
            evidence=[
                f"ph_0d entries: {len(ph_0d)}",
                f"ph_1d entries: {len(ph_1d)}",
                f"Source: {tda_event.payload.get('source')}",
            ],
            result="TDA snapshot is available via runtime events.",
            confidence=0.8,
            limitations=["Snapshot reflects cached TDA computation state."],
            next_action="Add online TDA recomputation pipeline.",
        )
        return RunSummary(
            metrics=metrics,
            conclusion=conclusion,
            artifacts=[
                {
                    "type": "tda_file",
                    "path": tda_event.payload.get("path"),
                }
            ],
        )

    def _finalize_topology_scan_snapshot(
        self, spec: AnalysisSpec, events: List[EventEnvelope]
    ) -> RunSummary:
        scan_event = next(
            (event for event in events if event.event_type == "TopologyScanSignal"),
            None,
        )
        if scan_event is None:
            searched = self._last_topology_scan.get("searched", [])
            model_name = self._last_topology_scan.get(
                "model", _normalize_topology_model_name(spec.model)
            )
            return RunSummary(
                metrics=[],
                conclusion=ConclusionCard(
                    objective="Load global topology summary for analysis panel.",
                    method="Build summary from cached topology layers.",
                    evidence=[f"Searched paths: {searched}"],
                    result=f"No topology summary available for model '{model_name}'.",
                    confidence=0.0,
                    limitations=["Topology cache file missing."],
                    next_action="Generate topology data and retry runtime scan.",
                ),
                artifacts=[],
            )

        summary = scan_event.payload.get("summary", {})
        field_count = float(len(summary)) if isinstance(summary, dict) else 0.0
        mean_error = 0.0
        if isinstance(summary, dict) and summary:
            values = [
                _safe_float(item.get("avg_ortho_error", 0.0))
                for item in summary.values()
                if isinstance(item, dict)
            ]
            if values:
                mean_error = float(sum(values) / len(values))

        metrics = [
            Metric(
                key="topology_scan_field_count",
                value=field_count,
                min_value=0.0,
                description="Number of topology summary entries.",
            ),
            Metric(
                key="topology_scan_mean_ortho_error",
                value=mean_error,
                min_value=0.0,
                description="Mean orthogonality error across summary fields.",
            ),
        ]
        conclusion = ConclusionCard(
            objective="Provide global topology summary through runtime protocol.",
            method="Read topology cache and aggregate field metrics.",
            evidence=[
                f"Fields: {int(field_count)}",
                f"Mean ortho error: {mean_error:.6f}",
                f"Source: {scan_event.payload.get('source')}",
            ],
            result="Topology scan summary is available via runtime events.",
            confidence=0.75,
            limitations=["Summary depends on cached topology snapshots."],
            next_action="Add optional live full-scan execution path.",
        )
        return RunSummary(
            metrics=metrics,
            conclusion=conclusion,
            artifacts=[
                {
                    "type": "topology_scan_source",
                    "path": scan_event.payload.get("path"),
                }
            ],
        )

    def _finalize_debias_snapshot(self, events: List[EventEnvelope]) -> RunSummary:
        debias_event = next(
            (event for event in events if event.event_type == "DebiasSignal"),
            None,
        )
        if debias_event is None:
            if self._last_debias.get("error"):
                evidence = [f"Runtime error: {self._last_debias.get('error')}"]
            else:
                evidence = ["Runtime model unavailable or debias not executed."]
            return RunSummary(
                metrics=[],
                conclusion=ConclusionCard(
                    objective="Apply debias intervention and compare token probabilities.",
                    method="Run baseline/debiased forward pass and emit DebiasSignal.",
                    evidence=evidence,
                    result="Debias snapshot not available from runtime.",
                    confidence=0.0,
                    limitations=["Requires initialized language model runtime."],
                    next_action="Use legacy endpoint fallback or initialize model runtime.",
                ),
                artifacts=[],
            )

        rows = debias_event.payload.get("results", [])
        shifts = [float(row.get("shift", 0.0)) for row in rows if isinstance(row, dict)]
        mean_abs_shift = float(np.mean(np.abs(shifts))) if shifts else 0.0
        max_shift = float(np.max(shifts)) if shifts else 0.0

        metrics = [
            Metric(
                key="debias_result_count",
                value=float(len(rows)),
                min_value=0.0,
                description="Number of compared tokens in debias result.",
            ),
            Metric(
                key="debias_mean_abs_shift",
                value=mean_abs_shift,
                min_value=0.0,
                description="Mean absolute probability shift after intervention.",
            ),
            Metric(
                key="debias_max_shift",
                value=max_shift,
                description="Maximum signed shift among returned tokens.",
            ),
        ]
        conclusion = ConclusionCard(
            objective="Provide debias comparison through runtime protocol.",
            method="Execute hook-based geometric debias and compute token deltas.",
            evidence=[
                f"Layer: {debias_event.payload.get('layer')}",
                f"Token rows: {len(rows)}",
                f"Mean |shift|: {mean_abs_shift:.6f}",
            ],
            result="Debias snapshot is available via runtime events.",
            confidence=0.7,
            limitations=["Uses synthetic orthogonal matrix when R is missing/invalid."],
            next_action="Feed RPT-derived transport matrix to improve intervention fidelity.",
        )
        return RunSummary(metrics=metrics, conclusion=conclusion, artifacts=[])
