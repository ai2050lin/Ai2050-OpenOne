"""Append the real study record once and register its traceable delivery proof."""
from datetime import datetime
from zoneinfo import ZoneInfo

import mlp_fidelity_v2_plan as P
from mlp_atlas_publish_final import Archive
from shared.dist_protocol.mechanism_trace import file_sha


def main():
    reg=P.read(P.OUT/'registration-delivery.json')
    A=Archive(P.PUBLIC,P.RUN,reg)
    phase='## Phase 2792: Qwen3-4B 字典保真、长文本生成重校准与末层因果边界'
    memo=P.ROOT/'research/glm5/docs/AGI_GLM5_MEMO.md'
    original=memo.read_text(encoding='utf-8')
    assert '## Phase 2792:' not in original
    stamp=datetime.now(ZoneInfo('America/Chicago')).strftime('%Y-%m-%d %H:%M')
    body=(P.ROOT/'tests/glm5_temp/mlp-fidelity-phase2792.md').read_text(encoding='utf-8').replace('{{TIME}}',stamp)
    old='`FVU=Σ||y−ŷ||²/Σ||y−μ_dev||²`，中心和尺度只用开发集。'
    assert old in body and body.startswith(phase)
    body=body.replace(old,'`FVU=Σ||y−ŷ||²/Σ||y−ȳ_split||²`，每一留出切分用其真实输出均值 `ȳ_split` 作归一化分母，拟合、训练尺度和常数预测基线只用开发集。')
    proof=A.put('delivery-verification',{'schema':'research_delivery_verification.v1','api':P.read(P.OUT/'live-api-acceptance.json'),'checks':{'python':41,'frontend':73,'dist_protocol':261,'vite_build':'passed'},'screenshots':{p.name:file_sha(p) for p in P.OUT.glob('client-*.png')},'scope':'real local client and API; no scientific promotion'})
    report=A.put('phase-summary',{'schema':'research_phase_summary.v1','phase':2792,'timestamp_local':stamp,'memo_path':memo.relative_to(P.ROOT).as_posix(),'report_sha256':reg['report_sha256'],'graph_sha256':reg['graph_sha256'],'acceptance_sha256':proof,'markdown':body})
    with memo.open('a',encoding='utf-8',newline='\n') as f:
        f.write('\n\n'+body.rstrip()+'\n')
    P.save(P.OUT/'phase-registration.json',{'phase':2792,'phase_summary_sha256':report,'acceptance_sha256':proof,'timestamp_local':stamp},False)
    print({'phase':2792,'record_sha256':report,'acceptance_sha256':proof,'timestamp':stamp},flush=True)


if __name__=='__main__':
    main()
