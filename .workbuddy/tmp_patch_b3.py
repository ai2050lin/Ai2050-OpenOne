# -*- coding: utf-8 -*-
"""researchctl.py 锚定补丁（B3-4 正式化）：
P1 FILES 增加 industry/cases
P2 VALID 增加 industry_status/case_status
P3 validate() 增加两账本的轻校验
P4 build_snapshot_v2 改读 registry（industry + cases 投影）
每处 assert 唯一命中；补丁后回读关键行复核。
"""
import io

P = r"D:\AI2050\Ai2050-OpenOne\ai2050_research_os\scripts\researchctl.py"
with io.open(P, "r", encoding="utf-8") as f:
    text = f.read()

def replace_once(old, new, tag):
    global text
    n = text.count(old)
    assert n == 1, "anchor %s hit %d times" % (tag, n)
    text = text.replace(old, new, 1)
    print("patched:", tag)

# P1 FILES
old1 = '    "artifact_residency": "artifact_residency.json",\n}'
new1 = ('    "artifact_residency": "artifact_residency.json",\n'
        '    "industry": "industry.json",\n'
        '    "cases": "cases.json",\n}')
replace_once(old1, new1, "P1-FILES")

# P2 VALID
old2 = '    "correction_status": {"active", "superseded", "archived"},\n}'
new2 = ('    "correction_status": {"active", "superseded", "archived"},\n'
        '    "industry_status": {"candidate", "verified", "superseded"},\n'
        '    "case_status": {"draft", "candidate", "published", "superseded"},\n}')
replace_once(old2, new2, "P2-VALID")

# P3 validate()
old3 = '    correction_map = unique_map(data["corrections"], "id", "corrections.json", errors)\n'
new3 = (
    '    correction_map = unique_map(data["corrections"], "id", "corrections.json", errors)\n'
    '    industry_map = unique_map(data["industry"], "id", "industry.json", errors)\n'
    '    cases_map = unique_map(data["cases"], "id", "cases.json", errors)\n'
    '    for iid, item in industry_map.items():\n'
    '        require_fields(item, ["kind", "status", "title"], f"industry {iid}", errors)\n'
    '        if item.get("status") not in VALID["industry_status"]:\n'
    '            errors.append(f"industry {iid} status 非法")\n'
    '    for cid, item in cases_map.items():\n'
    '        require_fields(item, ["kind", "status", "title", "phase_ref", "evidence_level", "narrative"], f"case {cid}", errors)\n'
    '        if item.get("status") not in VALID["case_status"]:\n'
    '            errors.append(f"case {cid} status 非法")\n'
    '        if item.get("evidence_level") not in {"has_data", "observed", "generalization_checked", "mechanism_evidence"}:\n'
    '            errors.append(f"case {cid} evidence_level 非法")\n'
)
replace_once(old3, new3, "P3-VALIDATE")

# P4 build_snapshot_v2 industry+cases
old4 = (
    '    methods_doc = load_json(OS_ROOT / "drafts" / "industry" / "methods_map.json")\n'
    '    tools_doc = load_json(OS_ROOT / "drafts" / "industry" / "tools.json")\n'
    '    gaps_doc = load_json(OS_ROOT / "drafts" / "industry" / "gap_radar.json")\n'
    '    snapshot["industry"] = {\n'
    '        "as_of": str(methods_doc.get("_meta", {}).get("date", snapshot["as_of"])),\n'
    '        "source_ref": "ai2050_research_os/drafts/industry/",\n'
    '        "methods": [\n'
    '            {\n'
    '                "method_id": item["method_id"],\n'
    '                "name": item["name"],\n'
    '                "era": item["era"],\n'
    '                "core_idea": item.get("core_idea", ""),\n'
    '                "evidence_grade": item["evidence_grade"],\n'
    '            }\n'
    '            for item in methods_doc.get("methods", [])\n'
    '        ],\n'
    '        "tools": [\n'
    '            {\n'
    '                "tool_id": item["tool_id"],\n'
    '                "name": item["name"],\n'
    '                "org": item.get("org", ""),\n'
    '                "role": item["role"],\n'
    '                "relation_to_project": item["relation_to_project"],\n'
    '            }\n'
    '            for item in tools_doc.get("tools", [])\n'
    '        ],\n'
    '        "gaps": [\n'
    '            {\n'
    '                "gap_id": item["gap_id"],\n'
    '                "title": item["title"],\n'
    '                "nearest_industry_work": item["nearest_industry_work"],\n'
    '                "missing": item["missing"],\n'
    '                "atlas_axis": item["atlas_axis"],\n'
    '                "priority": item["priority"],\n'
    '            }\n'
    '            for item in gaps_doc.get("gaps", [])\n'
    '        ],\n'
    '    }\n'
    '    snapshot["cases"] = {"as_of": snapshot["as_of"], "items": []}\n'
)
new4 = (
    '    industry_records = data["industry"]\n'
    '    snapshot["industry"] = {\n'
    '        "as_of": snapshot["as_of"],\n'
    '        "source_ref": "ai2050_research_os/registry/industry.json",\n'
    '        "methods": [\n'
    '            {\n'
    '                "method_id": item["id"],\n'
    '                "name": item.get("name", item["title"]),\n'
    '                "era": item.get("era", ""),\n'
    '                "core_idea": item.get("core_idea", ""),\n'
    '                "evidence_grade": item["evidence_grade"],\n'
    '            }\n'
    '            for item in industry_records if item.get("kind") == "method"\n'
    '        ],\n'
    '        "tools": [\n'
    '            {\n'
    '                "tool_id": item["id"],\n'
    '                "name": item.get("name", item["title"]),\n'
    '                "org": item.get("org", ""),\n'
    '                "role": item["role"],\n'
    '                "relation_to_project": item["relation_to_project"],\n'
    '            }\n'
    '            for item in industry_records if item.get("kind") == "tool"\n'
    '        ],\n'
    '        "gaps": [\n'
    '            {\n'
    '                "gap_id": item["id"],\n'
    '                "title": item["title"],\n'
    '                "nearest_industry_work": item["nearest_industry_work"],\n'
    '                "missing": item["missing"],\n'
    '                "atlas_axis": item["atlas_axis"],\n'
    '                "candidate_queue_ref": item.get("candidate_queue_ref") or "",\n'
    '                "priority": item["priority"],\n'
    '            }\n'
    '            for item in industry_records if item.get("kind") == "gap"\n'
    '        ],\n'
    '    }\n'
    '    snapshot["cases"] = {\n'
    '        "as_of": snapshot["as_of"],\n'
    '        "items": [\n'
    '            {\n'
    '                "case_id": item["id"],\n'
    '                "title": item["title"],\n'
    '                "phase_ref": item["phase_ref"],\n'
    '                "run_ref": item.get("run_ref", ""),\n'
    '                "view_refs": item.get("view_refs", []),\n'
    '                "evidence_level": item["evidence_level"],\n'
    '                "question": item.get("question", ""),\n'
    '                "narrative": item["narrative"],\n'
    '                "limitations": item.get("limitations", []),\n'
    '                "x-explanation-identity": True,\n'
    '            }\n'
    '            for item in data["cases"]\n'
    '        ],\n'
    '    }\n'
)
replace_once(old4, new4, "P4-BUILD-V2")

with io.open(P, "w", encoding="utf-8", newline="") as f:
    f.write(text)

# 回读复核
with io.open(P, "r", encoding="utf-8") as f:
    t2 = f.read()
checks = {
    "FILES industry": '"industry": "industry.json"' in t2,
    "FILES cases": '"cases": "cases.json"' in t2,
    "VALID industry_status": '"industry_status"' in t2,
    "validate cases_map": 'cases_map = unique_map(data["cases"], "id", "cases.json", errors)' in t2,
    "v2 registry source_ref": '"source_ref": "ai2050_research_os/registry/industry.json"' in t2,
    "v2 cases projection": '"x-explanation-identity": True' in t2,
    "no drafts refs left": 'drafts" / "industry' not in t2,
}
for k, v in checks.items():
    print(("OK  " if v else "FAIL") + " " + k)
assert all(checks.values()), "patch verification failed"
print("PATCH_DONE")
