import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

from server.runtime.contracts import AnalysisSpec, EventEnvelope, RunRecord, RunSummary
from server.runtime.experiment_store import ExperimentTimelineStore
from server.runtime.plugins import FiberBundleRoute, RoutePlugin, RunContext


class RunService:
    """In-memory phase-1 orchestrator for route-agnostic analysis runs."""

    def __init__(
        self,
        agi_core_provider: Optional[Callable[[], Any]] = None,
        model_provider: Optional[Callable[[], Any]] = None,
        timeline_store: Optional[ExperimentTimelineStore] = None,
    ) -> None:
        self._agi_core_provider = agi_core_provider or (lambda: None)
        self._model_provider = model_provider or (lambda: None)
        self._timeline_store = timeline_store or ExperimentTimelineStore()
        self._plugins: Dict[str, RoutePlugin] = {}
        self._runs: Dict[str, RunRecord] = {}
        self._events: Dict[str, List[EventEnvelope]] = {}
        self._lock = threading.Lock()
        self.register_plugin(FiberBundleRoute())

    def register_plugin(self, plugin: RoutePlugin) -> None:
        self._plugins[plugin.route_name] = plugin

    def list_routes(self) -> List[Dict[str, Any]]:
        routes = []
        for plugin in self._plugins.values():
            routes.append(
                {
                    "route": plugin.route_name,
                    "display_name": plugin.display_name,
                    "version": plugin.version,
                    "supported_analyses": plugin.supported_analyses,
                }
            )
        return routes

    def list_analyses(self) -> List[Dict[str, Any]]:
        analyses: Dict[str, Dict[str, Any]] = {}
        for plugin in self._plugins.values():
            for analysis_type in plugin.supported_analyses:
                analyses.setdefault(
                    analysis_type,
                    {"analysis_type": analysis_type, "routes": []},
                )
                analyses[analysis_type]["routes"].append(plugin.route_name)
        return list(analyses.values())

    def create_run(self, spec: AnalysisSpec) -> RunRecord:
        plugin = self._plugins.get(spec.route)
        if plugin is None:
            raise ValueError(f"Unknown route: {spec.route}")
        if spec.analysis_type not in plugin.supported_analyses:
            raise ValueError(
                f"Route '{spec.route}' does not support analysis '{spec.analysis_type}'"
            )

        now = time.time()
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        record = RunRecord(
            run_id=run_id,
            spec=spec,
            status="pending",
            created_at=now,
            updated_at=now,
            event_count=0,
        )

        with self._lock:
            self._runs[run_id] = record
            self._events[run_id] = []

        self._execute_run(run_id, plugin)
        return self.get_run(run_id)

    def _execute_run(self, run_id: str, plugin: RoutePlugin) -> None:
        with self._lock:
            record = self._runs[run_id]
            record.status = "running"
            record.updated_at = time.time()
            self._runs[run_id] = record

        spec = record.spec
        ctx = RunContext(
            run_id=run_id,
            agi_core=self._agi_core_provider(),
            model=self._model_provider(),
        )
        try:
            plugin.prepare(spec, ctx)
            events = plugin.run(spec, ctx)
            summary: RunSummary = plugin.finalize(spec, ctx, events)
            now = time.time()
            with self._lock:
                self._events[run_id].extend(events)
                record = self._runs[run_id]
                record.status = "completed"
                record.summary = summary
                record.error = None
                record.event_count = len(self._events[run_id])
                record.updated_at = now
                self._runs[run_id] = record
                persisted_record = record
            try:
                self._timeline_store.append_run(persisted_record)
            except Exception:
                pass
        except Exception as exc:
            now = time.time()
            failure_event = EventEnvelope(
                event_type="RunError",
                run_id=run_id,
                step=0,
                timestamp=now,
                payload={"error": str(exc)},
                meta={"route": spec.route, "analysis_type": spec.analysis_type},
            )
            with self._lock:
                self._events[run_id].append(failure_event)
                record = self._runs[run_id]
                record.status = "failed"
                record.error = str(exc)
                record.event_count = len(self._events[run_id])
                record.updated_at = now
                self._runs[run_id] = record
                persisted_record = record
            try:
                self._timeline_store.append_run(persisted_record)
            except Exception:
                pass

    def get_run(self, run_id: str) -> RunRecord:
        with self._lock:
            record = self._runs.get(run_id)
            if record is None:
                raise KeyError(run_id)
            return record

    def get_events(
        self, run_id: str, after_step: Optional[int] = None, limit: int = 200
    ) -> List[EventEnvelope]:
        with self._lock:
            if run_id not in self._events:
                raise KeyError(run_id)
            events = list(self._events[run_id])

        if after_step is not None:
            events = [event for event in events if event.step > after_step]
        if limit > 0:
            events = events[:limit]
        return events

    def get_experiment_timeline(
        self, route: Optional[str] = None, limit_per_route: int = 50
    ) -> Dict[str, Any]:
        return self._timeline_store.get_timeline(
            route=route, limit_per_route=limit_per_route
        )

    def generate_weekly_report(
        self, days: int = 7, persist: bool = False
    ) -> Dict[str, Any]:
        return self._timeline_store.generate_weekly_report(days=days, persist=persist)
