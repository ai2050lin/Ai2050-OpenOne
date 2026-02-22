import json
import os
import threading
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional


DEFAULT_TIMELINE_PATH = os.path.join("tempdata", "agi_route_test_timeline.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _model_to_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return {}


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


class ExperimentTimelineStore:
    """Persistent timeline store for route-level AGI experiment runs."""

    def __init__(
        self,
        path: str = DEFAULT_TIMELINE_PATH,
        max_tests_per_route: int = 500,
        max_tests_per_analysis_type: Optional[Dict[str, int]] = None,
    ) -> None:
        self.path = path
        self.max_tests_per_route = max_tests_per_route
        # Retain high-frequency runtime signals separately to avoid crowding out
        # research-grade experiment records (e.g., scaling/causal tests).
        self.max_tests_per_analysis_type = max_tests_per_analysis_type or {
            "unified_conscious_field": 200
        }
        self._lock = threading.Lock()
        self._ensure_file()

    def _empty_doc(self) -> Dict[str, Any]:
        return {
            "schema_version": "1.0",
            "generated_at": _utc_now_iso(),
            "routes": {},
        }

    def _ensure_file(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(self.path):
            self._save_doc(self._empty_doc())

    def _load_doc(self) -> Dict[str, Any]:
        self._ensure_file()
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict):
                return self._empty_doc()
            if "routes" not in payload or not isinstance(payload.get("routes"), dict):
                payload["routes"] = {}
            if "schema_version" not in payload:
                payload["schema_version"] = "1.0"
            if "generated_at" not in payload:
                payload["generated_at"] = _utc_now_iso()
            return payload
        except Exception:
            return self._empty_doc()

    def _save_doc(self, payload: Dict[str, Any]) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temp_path = f"{self.path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(temp_path, self.path)

    def _normalize_metric_score(self, metric: Dict[str, Any]) -> float:
        value = _safe_float(metric.get("value"), default=0.0)
        min_value = metric.get("min_value")
        max_value = metric.get("max_value")
        if min_value is not None and max_value is not None:
            lo = _safe_float(min_value, 0.0)
            hi = _safe_float(max_value, 1.0)
            if hi <= lo:
                return 0.5
            return _clamp01((value - lo) / (hi - lo))
        return 0.6 if value >= 0 else 0.3

    def _evaluate_record(self, record_dict: Dict[str, Any]) -> Dict[str, Any]:
        status = str(record_dict.get("status", "unknown"))
        summary = record_dict.get("summary") or {}
        conclusion = summary.get("conclusion") or {}
        metrics = summary.get("metrics") or []

        if status != "completed":
            return {
                "score": 0.0,
                "grade": "D",
                "feasibility": "low",
                "summary": f"Run status={status}. Not enough successful evidence.",
                "confidence": 0.0,
            }

        confidence = _clamp01(_safe_float(conclusion.get("confidence"), 0.5))
        metric_scores = []
        for metric in metrics:
            if isinstance(metric, dict):
                metric_scores.append(self._normalize_metric_score(metric))
        metric_score = sum(metric_scores) / len(metric_scores) if metric_scores else 0.5

        score = _clamp01(0.7 * confidence + 0.3 * metric_score)
        if score >= 0.8:
            grade = "A"
            feasibility = "high"
        elif score >= 0.65:
            grade = "B"
            feasibility = "high"
        elif score >= 0.5:
            grade = "C"
            feasibility = "medium"
        else:
            grade = "D"
            feasibility = "low"

        result_text = str(conclusion.get("result") or "No conclusion text.")
        return {
            "score": round(score, 4),
            "grade": grade,
            "feasibility": feasibility,
            "summary": result_text,
            "confidence": round(confidence, 4),
        }

    def _entry_from_record(self, record_dict: Dict[str, Any]) -> Dict[str, Any]:
        spec = record_dict.get("spec") or {}
        summary = record_dict.get("summary") or {}
        tests_metrics = summary.get("metrics") or []
        conclusion = summary.get("conclusion") or {}
        artifacts = summary.get("artifacts") or []

        created_at = _safe_float(record_dict.get("created_at"), 0.0)
        updated_at = _safe_float(record_dict.get("updated_at"), created_at)
        run_id = str(record_dict.get("run_id", "unknown"))
        failure_reason = self._extract_failure_reason(record_dict)

        return {
            "test_id": run_id,
            "run_id": run_id,
            "created_at": created_at,
            "updated_at": updated_at,
            "timestamp": datetime.fromtimestamp(
                created_at if created_at > 0 else updated_at, tz=timezone.utc
            ).isoformat(),
            "status": str(record_dict.get("status", "unknown")),
            "error": record_dict.get("error"),
            "failure_reason": failure_reason,
            "event_count": int(record_dict.get("event_count", 0) or 0),
            "route": spec.get("route"),
            "analysis_type": spec.get("analysis_type"),
            "model": spec.get("model"),
            "params": spec.get("params") or {},
            "metrics": tests_metrics,
            "conclusion": conclusion,
            "artifacts": artifacts,
            "evaluation": self._evaluate_record(record_dict),
        }

    def _extract_failure_reason(self, record_dict: Dict[str, Any]) -> Optional[str]:
        status = _safe_text(record_dict.get("status", "unknown")).lower()
        if status not in {"failed", "error"}:
            return None

        error_text = _safe_text(record_dict.get("error"))
        if error_text:
            return error_text

        summary = record_dict.get("summary") or {}
        conclusion = summary.get("conclusion") or {}
        limitations = conclusion.get("limitations") or []
        if isinstance(limitations, list) and limitations:
            first = _safe_text(limitations[0])
            if first:
                return first

        result_text = _safe_text(conclusion.get("result"))
        if result_text:
            return result_text
        return "unknown_failure"

    def _recompute_route_stats(self, tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(tests)
        completed = sum(1 for t in tests if t.get("status") == "completed")
        failed = sum(1 for t in tests if t.get("status") == "failed")
        running = sum(1 for t in tests if t.get("status") == "running")
        pending = sum(1 for t in tests if t.get("status") == "pending")
        scores = [
            _safe_float((t.get("evaluation") or {}).get("score"), 0.0)
            for t in tests
            if isinstance(t, dict)
        ]
        avg_score = sum(scores) / len(scores) if scores else 0.0
        last_ts = tests[-1].get("timestamp") if tests else None
        reason_count: Dict[str, int] = {}
        for test in tests:
            reason = _safe_text(test.get("failure_reason"))
            if not reason:
                continue
            reason_count[reason] = reason_count.get(reason, 0) + 1
        top_failure_reasons = [
            {"reason": reason, "count": count}
            for reason, count in sorted(
                reason_count.items(), key=lambda item: item[1], reverse=True
            )[:5]
        ]
        return {
            "total_runs": total,
            "completed_runs": completed,
            "failed_runs": failed,
            "running_runs": running,
            "pending_runs": pending,
            "avg_score": round(avg_score, 4),
            "latest_timestamp": last_ts,
            "top_failure_reasons": top_failure_reasons,
        }

    def _apply_retention_policy(self, tests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply route-level and analysis-type-level retention in ascending time order."""
        if not tests:
            return tests

        tests = sorted(tests, key=lambda t: _safe_float(t.get("created_at"), 0.0))

        type_limits = self.max_tests_per_analysis_type or {}
        if type_limits:
            keep_run_ids = set()
            grouped: Dict[str, List[Dict[str, Any]]] = {}
            for test in tests:
                analysis_type = _safe_text(test.get("analysis_type"))
                grouped.setdefault(analysis_type, []).append(test)

            for analysis_type, limit in type_limits.items():
                safe_limit = int(limit) if limit is not None else 0
                if safe_limit <= 0:
                    continue
                for item in grouped.get(analysis_type, [])[-safe_limit:]:
                    run_id = _safe_text(item.get("run_id"))
                    if run_id:
                        keep_run_ids.add(run_id)

            filtered: List[Dict[str, Any]] = []
            for test in tests:
                analysis_type = _safe_text(test.get("analysis_type"))
                if analysis_type in type_limits:
                    if _safe_text(test.get("run_id")) in keep_run_ids:
                        filtered.append(test)
                else:
                    filtered.append(test)
            tests = filtered

        if len(tests) <= self.max_tests_per_route:
            return tests

        high_volume_types = set(type_limits.keys())
        non_high = [t for t in tests if _safe_text(t.get("analysis_type")) not in high_volume_types]
        high = [t for t in tests if _safe_text(t.get("analysis_type")) in high_volume_types]

        if len(non_high) >= self.max_tests_per_route:
            return non_high[-self.max_tests_per_route :]

        remaining = self.max_tests_per_route - len(non_high)
        return non_high + high[-remaining:]

    def append_run(self, run_record: Any) -> Dict[str, Any]:
        record_dict = _model_to_dict(run_record)
        spec = record_dict.get("spec") or {}
        route = str(spec.get("route") or "unknown_route")
        entry = self._entry_from_record(record_dict)

        with self._lock:
            payload = self._load_doc()
            routes = payload.setdefault("routes", {})
            route_bucket = routes.setdefault(
                route,
                {"route": route, "tests": [], "stats": {}, "latest_test_id": None},
            )
            tests = route_bucket.setdefault("tests", [])

            tests = [t for t in tests if t.get("run_id") != entry["run_id"]]
            tests.append(entry)
            tests = self._apply_retention_policy(tests)

            route_bucket["tests"] = tests
            route_bucket["latest_test_id"] = tests[-1]["test_id"] if tests else None
            route_bucket["stats"] = self._recompute_route_stats(tests)

            payload["generated_at"] = _utc_now_iso()
            self._save_doc(payload)

        return entry

    def get_timeline(
        self, route: Optional[str] = None, limit_per_route: int = 50
    ) -> Dict[str, Any]:
        with self._lock:
            payload = self._load_doc()

        routes_dict = payload.get("routes", {})
        result_routes: List[Dict[str, Any]] = []

        route_items = (
            [(route, routes_dict.get(route, {}))] if route else list(routes_dict.items())
        )
        for route_name, route_bucket in route_items:
            if not route_bucket:
                continue
            tests = list(route_bucket.get("tests", []))
            tests.sort(key=lambda t: _safe_float(t.get("created_at"), 0.0), reverse=True)
            if limit_per_route > 0:
                tests = tests[:limit_per_route]
            result_routes.append(
                {
                    "route": route_name,
                    "latest_test_id": route_bucket.get("latest_test_id"),
                    "stats": route_bucket.get("stats", {}),
                    "tests": tests,
                }
            )

        result_routes.sort(
            key=lambda r: _safe_float(
                (r.get("tests") or [{}])[0].get("created_at") if r.get("tests") else 0.0,
                0.0,
            ),
            reverse=True,
        )

        return {
            "schema_version": payload.get("schema_version", "1.0"),
            "generated_at": payload.get("generated_at"),
            "file_path": self.path,
            "routes": result_routes,
        }

    def _flatten_tests(
        self, payload: Dict[str, Any], days: int
    ) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max(1, int(days)))
        routes = payload.get("routes", {})
        tests: List[Dict[str, Any]] = []

        for route_name, route_bucket in routes.items():
            bucket_tests = route_bucket.get("tests", [])
            if not isinstance(bucket_tests, list):
                continue
            for test in bucket_tests:
                if not isinstance(test, dict):
                    continue
                ts_text = _safe_text(test.get("timestamp"))
                try:
                    ts = datetime.fromisoformat(ts_text)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except Exception:
                    ts = None
                if ts is None or ts < cutoff:
                    continue
                normalized = dict(test)
                normalized["route"] = test.get("route") or route_name
                tests.append(normalized)

        tests.sort(
            key=lambda t: _safe_float(t.get("created_at"), 0.0),
            reverse=True,
        )
        return tests

    def _build_weekly_markdown(self, report: Dict[str, Any]) -> str:
        totals = report.get("totals", {})
        highlights = report.get("highlights", {})
        top_failures = report.get("top_failure_reasons", [])
        route_summaries = report.get("route_summaries", [])
        recommendations = report.get("recommendations", [])

        lines = [
            "# AGI Weekly Report",
            "",
            f"- Generated At: {report.get('generated_at')}",
            f"- Window Days: {report.get('window_days')}",
            "",
            "## Totals",
            f"- Total Tests: {totals.get('total_tests', 0)}",
            f"- Completed: {totals.get('completed_tests', 0)}",
            f"- Failed: {totals.get('failed_tests', 0)}",
            f"- Completion Rate: {totals.get('completion_rate', 0)}",
            "",
            "## Highlights",
            f"- Best Route: {highlights.get('best_route') or 'N/A'}",
            f"- Most Active Route: {highlights.get('most_active_route') or 'N/A'}",
            f"- Best Run ID: {highlights.get('best_run_id') or 'N/A'}",
            "",
            "## Route Summaries",
        ]

        for route in route_summaries:
            lines.append(
                f"- {route.get('route')}: tests={route.get('total_tests', 0)}, "
                f"completed={route.get('completed_tests', 0)}, "
                f"failed={route.get('failed_tests', 0)}, "
                f"avg_score={route.get('avg_score', 0)}"
            )

        lines.extend(["", "## Top Failure Reasons"])
        if top_failures:
            for item in top_failures:
                lines.append(f"- {item.get('reason')}: {item.get('count')}")
        else:
            lines.append("- No failures in this window.")

        lines.extend(["", "## Recommendations"])
        if recommendations:
            for item in recommendations:
                lines.append(f"- {item}")
        else:
            lines.append("- Keep collecting more route comparison evidence.")

        lines.append("")
        return "\n".join(lines)

    def generate_weekly_report(
        self, days: int = 7, persist: bool = False
    ) -> Dict[str, Any]:
        with self._lock:
            payload = self._load_doc()

        tests = self._flatten_tests(payload, days=days)
        route_groups: Dict[str, List[Dict[str, Any]]] = {}
        for test in tests:
            route = _safe_text(test.get("route")) or "unknown_route"
            route_groups.setdefault(route, []).append(test)

        total_tests = len(tests)
        completed_tests = sum(1 for t in tests if t.get("status") == "completed")
        failed_tests = sum(1 for t in tests if t.get("status") == "failed")
        completion_rate = round(
            (completed_tests / total_tests) if total_tests > 0 else 0.0, 4
        )

        route_summaries: List[Dict[str, Any]] = []
        for route, route_tests in route_groups.items():
            c = sum(1 for t in route_tests if t.get("status") == "completed")
            f = sum(1 for t in route_tests if t.get("status") == "failed")
            scores = [
                _safe_float((t.get("evaluation") or {}).get("score"), 0.0)
                for t in route_tests
            ]
            avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0
            route_summaries.append(
                {
                    "route": route,
                    "total_tests": len(route_tests),
                    "completed_tests": c,
                    "failed_tests": f,
                    "avg_score": avg_score,
                }
            )
        route_summaries.sort(
            key=lambda r: (_safe_float(r.get("avg_score"), 0.0), r.get("total_tests", 0)),
            reverse=True,
        )

        best_route = route_summaries[0]["route"] if route_summaries else None
        most_active_route = (
            max(route_summaries, key=lambda r: r.get("total_tests", 0)).get("route")
            if route_summaries
            else None
        )
        best_run = (
            max(
                tests,
                key=lambda t: _safe_float((t.get("evaluation") or {}).get("score"), 0.0),
            )
            if tests
            else None
        )

        failure_counter: Dict[str, int] = {}
        for t in tests:
            reason = _safe_text(t.get("failure_reason") or t.get("error"))
            if not reason:
                continue
            failure_counter[reason] = failure_counter.get(reason, 0) + 1
        top_failure_reasons = [
            {"reason": k, "count": v}
            for k, v in sorted(
                failure_counter.items(), key=lambda item: item[1], reverse=True
            )[:10]
        ]

        recommendations: List[str] = []
        if completion_rate < 0.7 and total_tests > 0:
            recommendations.append(
                "Investigate stability issues: completion rate is below 70%."
            )
        if top_failure_reasons:
            recommendations.append(
                f"Address top failure reason first: {top_failure_reasons[0]['reason']}."
            )
        if len(route_summaries) <= 1:
            recommendations.append(
                "Run at least one additional route to improve A/B evidence quality."
            )
        if not recommendations:
            recommendations.append(
                "Maintain current cadence and increase cross-route benchmark coverage."
            )

        report = {
            "schema_version": "1.0",
            "generated_at": _utc_now_iso(),
            "window_days": max(1, int(days)),
            "totals": {
                "total_tests": total_tests,
                "completed_tests": completed_tests,
                "failed_tests": failed_tests,
                "completion_rate": completion_rate,
            },
            "highlights": {
                "best_route": best_route,
                "most_active_route": most_active_route,
                "best_run_id": (best_run or {}).get("run_id"),
            },
            "route_summaries": route_summaries,
            "top_failure_reasons": top_failure_reasons,
            "recommendations": recommendations,
        }

        markdown = self._build_weekly_markdown(report)
        saved_files = {}
        if persist:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_dir = os.path.join("tempdata", "reports")
            os.makedirs(out_dir, exist_ok=True)
            json_path = os.path.join(out_dir, f"agi_weekly_report_{stamp}.json")
            md_path = os.path.join(out_dir, f"agi_weekly_report_{stamp}.md")
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump(report, jf, ensure_ascii=False, indent=2)
            with open(md_path, "w", encoding="utf-8") as mf:
                mf.write(markdown)
            saved_files = {"json_path": json_path, "markdown_path": md_path}

        return {
            "report": report,
            "markdown": markdown,
            "saved_files": saved_files,
        }
