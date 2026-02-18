# AGI Visual Execution Board

## Goals

1. Analyze deep neural network internal structure.
2. Compare multiple AGI routes and preserve evidence.
3. Build timeline-based research governance.

---

## Phase A: Foundation

1. Runtime unified protocol for major analyses.  
Status: `completed`

2. Fixed-format JSON timeline persistence.  
File: `tempdata/agi_route_test_timeline.json`  
Status: `completed`

3. Timeline API.  
Endpoint: `GET /api/v1/experiments/timeline`  
Status: `completed`

4. Progress dashboard timeline view.  
Page: `frontend/src/AGIProgressDashboard.jsx`  
Status: `completed`

---

## Phase B: Comparative Analysis

1. Timeline filters (`route/status/analysis/time-window`).  
Status: `completed`

2. JSON export for filtered and per-route history.  
Status: `completed`

3. Multi-route A/B comparison panel.  
Status: `completed`

4. Failure reason aggregation (reason frequency).  
Status: `completed`

---

## Phase C: Research Governance

1. Weekly report automation (JSON + Markdown export).  
Status: `completed`

2. Milestone auto-update from route progress.  
Status: `completed`

3. Route feasibility trend visualization (`score trend`).  
Status: `completed`

---

## Next Steps

1. Add one-click “weekly report pack” (timeline snapshot + report + markdown).
2. Add route trend drill-down (select route then inspect per-run details).
3. Add milestone threshold customization in settings.

---

## Done Criteria

1. Any run can be replayed and evaluated from timeline.
2. Any route can export full history JSON.
3. At least two routes can be compared side-by-side for one analysis type.
4. Dashboard links progress phases with route performance changes.
