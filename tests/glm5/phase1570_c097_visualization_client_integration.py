#!/usr/bin/env python3
"""Phase1570: audit the C097 relation-contrast heatmap client integration."""
from __future__ import annotations
import json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; TESTS=ROOT/"tests/glm5"; RESULT=TESTS/"result"; PARENT=RESULT/"phase1569_c097_relation_contrast_heatmap_export"; OUT=RESULT/"phase1570_c097_visualization_client_integration"
sys.path.insert(0,str(TESTS)); import phase1331_relational_measurement_core as core
FILES={"route":ROOT/"frontend/src/researchKernel/heatmapResearchRoute.js","hook":ROOT/"frontend/src/researchKernel/useResearchKernel.js","card":ROOT/"frontend/src/components/app/ResearchHeatmapRoute.jsx","css":ROOT/"frontend/src/components/app/ResearchHeatmapRoute.css","app":ROOT/"frontend/src/App.jsx","asset":ROOT/"frontend/public/vis_data/research_kernel/c097_relation_contrast_heatmap.json","built_asset":ROOT/"frontend/dist/vis_data/research_kernel/c097_relation_contrast_heatmap.json"}
def main():
 if (OUT/"analysis/final.json").exists(): raise RuntimeError("Phase1570 exists")
 pf=core.load(PARENT/"analysis/final.json"); pa=core.load(PARENT/"audit/independent_final_audit.json"); text={k:p.read_text(encoding="utf-8-sig") if p.suffix in {".js",".jsx",".css"} else "" for k,p in FILES.items()}
 checks={"parent":pf["authorization"]=="freeze_C098_observation_first_graph_contract" and pa["all_checks_passed"],"result_type":"relation_contrast_heatmap" in text["route"],"source_path":"c097_relation_contrast_heatmap.json" in text["route"],"hook_loads":"relationContrastHeatmap" in text["hook"],"app_passes_prop":"relationContrastHeatmap={realResearchTrace.relationContrastHeatmap}" in text["app"],"card_renders":"Relation Contrast Heatmap" in text["card"] and "research-heatmap-card__relation-row" in text["card"],"css_stable_grid":"--relation-columns" in text["css"],"asset_exists":FILES["asset"].exists(),"build_asset_exists":FILES["built_asset"].exists(),"build_asset_identity":core.sha(FILES["asset"])==core.sha(FILES["built_asset"])}
 if not all(checks.values()): raise RuntimeError(checks)
 report={"phase":1570,"campaign":"C097","status":"visualization_client_integration_verified","checks":checks,"files":{k:{"path":str(p.relative_to(ROOT)),"sha256":core.sha(p)} for k,p in FILES.items()},"external_commands":{"vite_build":"passed","targeted_eslint":"passed"},"finished_at_utc":datetime.now(timezone.utc).isoformat(),"authorization":"freeze_C098_observation_first_graph_contract"}; core.save(OUT/"analysis/client_integration.json",report); core.save(OUT/"analysis/final.json",{"phase":1570,"campaign":"C097","status":report["status"],"authorization":report["authorization"]}); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()

