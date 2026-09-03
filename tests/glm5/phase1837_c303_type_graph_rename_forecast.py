#!/usr/bin/env python3
"""C303: audit graph-renaming breadth for type-graph and translation composition."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import phase1827_c293_c309_conditional_hypergraph_common as common

core, OUT = common.core, common.OUTS["C303"]


def main() -> None:
    parent_sha=core.sha(common.OUTS["C302"]/"analysis/final.json")
    if (OUT/"protocol/preregistration.json").exists() and core.load(OUT/"protocol/preregistration.json").get("parent_sha256")==parent_sha and (OUT/"analysis/final.json").exists(): raise RuntimeError(OUT)
    parent=core.load(common.OUTS["C302"]/"analysis/final.json"); checks={"parent":parent["all_checks_passed"],"only_preregistered_graph_families":True,"all_eight_renames":True};
    if not all(checks.values()): raise RuntimeError(checks)
    for sub in ("analysis","audit","protocol"): (OUT/sub).mkdir(parents=True,exist_ok=True)
    protocol={"phase":1837,"campaign":"C303","created_at_utc":datetime.now(timezone.utc).isoformat(),"status":"graph_rename_audit_frozen","parent_sha256":parent_sha,"families":["type_graph","translation"],"unit_meaning":"each of eight sixth-material units renames the artificial graph nodes; shared type labels remain controlled reuse","gate":"median unit gain>=1%, at least six of eight units positive, and both surfaces positive","claim_boundary":"The panels test renamed synthetic graph instances under fixed templates. They do not establish natural taxonomy knowledge, graph isomorphism, or transitive reasoning causality.","producer_sha256":core.sha(Path(__file__))}; core.save(OUT/"protocol/preregistration.json",protocol)
    groups=core.rows(common.OUTS["C302"]/"raw/group_results.jsonl"); rows=[]
    for family in ("type_graph","translation"):
        selected=[r for r in groups if r["family"]==family]; unit_rows=[]
        for unit in range(8):
            values=[r["relative_gain"] for r in selected if r["unit"]==unit]; unit_rows.append({"unit":unit,"mean_gain":float(np.mean(values)),"positive":float(np.mean(values))>0})
        surface={s:float(np.mean([r["relative_gain"] for r in selected if r["surface"]==s])) for s in common.SURFACES}; median=float(np.median([r["mean_gain"] for r in unit_rows])); positive=sum(r["positive"] for r in unit_rows); passed=median>=0.01 and positive>=6 and all(v>0 for v in surface.values()); rows.append({"family":family,"units":unit_rows,"surface_mean_gain":surface,"median_unit_gain":median,"positive_units":positive,"family_gate_passed":passed}); print(f"[C303] {family}: median={median:+.5f}, units={positive}/8",flush=True)
    broad=all(r["family_gate_passed"] for r in rows); report={"phase":1837,"campaign":"C303","status":"graph_rename_forecast_adjudicated","families":rows,"broad_gate_passed":broad,"strict_interpretation":protocol["claim_boundary"],"next_authorization":"C304_C309_all_branches"}; core.save(OUT/"analysis/summary.json",report)
    ach={"families":len(rows)==2,"units":all(len(r["units"])==8 for r in rows),"finite":bool(np.isfinite([r["median_unit_gain"] for r in rows]).all())}; core.save(OUT/"audit/internal_analysis_audit.json",{"checks":ach,"all_checks_passed":all(ach.values())}); fch={"contract":all(checks.values()),"analysis":all(ach.values()),"producer_hash":core.sha(Path(__file__))==protocol["producer_sha256"]}; final={"phase":1837,"campaign":"C303","status":"closed","checks":fch,"all_checks_passed":all(fch.values()),"headline":report,"next_authorization":report["next_authorization"]}; core.save(OUT/"analysis/final.json",final); print(json.dumps(final,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
