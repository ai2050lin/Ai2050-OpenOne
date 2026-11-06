"""Append a scoped label correction; frozen outcomes and NI remain identical."""
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import phase2777_confirm as C
import phase2773_w6_loop as W


def main():
    from deploy.research_library import Library
    lib=Library(W.ROOT/'distributed_data/research_library')
    old=W.read(C.OUT/'client-registration.json');report=copy.deepcopy(lib.artifact(old['report_sha256']))
    edits=[]
    for x in report['result']['counterexamples']:
        question=report['counterexample_inputs'].get(x['sample_id'])
        if question and not question['gate_eligible']:
            edits.append(x['sample_id']);x.update(original_reason=x['reason'],reason='unverified reference-only apparent regression',
                gate_eligible=False,label_status='mirror reference, not independently validated')
        elif question:x.update(gate_eligible=True,label_status='provided dataset key')
    assert len(edits)==9
    errata={'schema':'research_label_errata.v1','kind':'real','run_id':C.RUN,'model_snapshot':report['model_snapshot'],
        'source_report_sha256':old['report_sha256'],'affected_sample_ids':edits,
        'correction':'Runner reused keyed-regression wording for nine M3Exam mirror-reference changes. Relabel as reference-only; never included in formal NI.',
        'unchanged':['raw immutable result','2200-question NI mask and values','primary outcome','337 total mixed records'],
        'formal_capability_regressions':23,'reference_only_changes':9}
    path=C.PUBLIC/'reference-label-errata.research.json';W.save(path,errata,True)
    errata_ref=lib.ingest(errata,str(path.relative_to(W.ROOT)),kind='local_report')['sha256']
    report.update(revision='reference-scope-corrected-v2',source_report_sha256=old['report_sha256'],reference_label_errata_sha256=errata_ref)
    report['counterexample_breakdown']={'natural_style_or_content':305,'formal_capability_regressions':23,'reference_only_changes':9}
    report['limitations'].append('反例中9条M3Exam变化已更正为未独立验证的参考变化；不计2200题正式能力门。原始runner文件与旧报告保留，参见更正产物。')
    path=C.PUBLIC/'report-v2.research.json';W.save(path,report,True)
    ref=lib.ingest(report,str(path.relative_to(W.ROOT)),kind='local_report')['sha256']
    graph=copy.deepcopy(lib.artifact(old['graph_sha256']));graph.update(revision='reference-scope-corrected-v2',source_graph_sha256=old['graph_sha256'])
    for x in graph['feature_cards']+graph['graph_edges']:x['evidence']['artifact_sha256']=ref
    for x in graph['feature_cards']:x['counterexamples_full_report_sha256']=ref
    path=C.PUBLIC/'graph-v2.research.json';W.save(path,graph,True)
    gr=lib.ingest(graph,str(path.relative_to(W.ROOT)),kind='registered')['sha256']
    final={**old,'report_sha256':ref,'graph_sha256':gr,'source_report_sha256':old['report_sha256'],
        'reference_label_errata_sha256':errata_ref}
    W.save(C.OUT/'client-registration-v2.json',final,True)
    W.save(C.OUT/'client-registration-current.json',final)
    C.status('registration_validated',**final)
    print(json.dumps(final))


if __name__=='__main__':main()
