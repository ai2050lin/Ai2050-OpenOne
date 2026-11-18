"""Package the completed calibration, checks and bounded continuation plan."""
import ast,collections,importlib.metadata,json,platform,shutil,zipfile
from pathlib import Path
from html.parser import HTMLParser
import b36_semantic14 as S

def run():
    S.configure(40*2**20);audit=S.read(S.OUT/'audit.json');assert audit['status']=='passed'
    plan=S.read(S.OUT/'expanded-plan.json');results={r['id']:r for r in S.read(S.OUT/'expanded-score/summary.json')['results']};pairs=[]
    for depth in [1,2,3]:
      for fmt in ['code0','code2']:
        groups=collections.defaultdict(dict)
        for r in plan['rows']:
            if r['depth']==depth:groups[r['family']][r['format']]=results[r['id']]
        count=collections.Counter((rr['direct']['top1_correct'],rr[fmt]['top1_correct']) for rr in groups.values())
        pairs.append(dict(depth=depth,code=fmt,n=len(groups),both_correct=count[(True,True)],direct_only=count[(True,False)],code_only=count[(False,True)],both_wrong=count[(False,False)],accuracy_difference_code_minus_direct=(count[(False,True)]-count[(True,False)])/len(groups)))
    S.save('paired-format-description.json',dict(kind='posthoc_description',rows=pairs,limit='Same48 programs in paired input formats. This is not a within-hidden-state causal decomposition. No preregistered significance or necessity claim; codebook presence, context and readout all change.'))
    progress=[json.loads(s) for s in (S.OUT/'progress.jsonl').read_text().splitlines() if s.strip()]
    peaks={}
    for name,is4 in [('qwen14b',False),('qwen4b_reference',True)]:
        rr=[r for r in progress if ('reference4B' in r['stage'])==is4]
        peaks[name]=dict(cuda_peak_bytes=max(r.get('cuda_peak',0) for r in rr),shared_peak_bytes=max(r.get('shared_gpu_bytes',0) for r in rr),minimum_ram_available_bytes=min(r['ram_available_bytes'] for r in rr),maximum_commit_used_bytes=max(r['commit_used_bytes'] for r in rr))
    S.save('execution-environment.json',dict(python=platform.python_version(),packages={n:importlib.metadata.version(n) for n in ['torch','transformers','numpy','safetensors','matplotlib']},attention='sdpa',dtype='original BF16, not quantized',native_placement='Original embedding CUDA; each of40 original decoder blocks read from CPU/safetensors toCUDA then released; final norm and lm_head CUDA. Batch4; padded length256; right padding; actual answer position; use_cache=False.',reference_placement='4B standard HF allresident CUDA, same BF16; generate cache enabled; no concurrent14B process.',peaks=peaks,free_disk_bytes=shutil.disk_usage(S.ROOT).free,storage_policy_sha256=S.file_sha(S.ROOT/'deploy/storage_policy.json'),versions_recorded_at_delivery='Versions read from same environment after execution; no dependency installation or upgrades this task.'))
    local=list((S.ROOT/'tests/glm5').glob('b36_semantic14*.py'))+[S.ROOT/'tests/glm5/test_b36_semantic14_materials.py',S.ROOT/'tests/glm5/b36_semantic14_README.md']
    for name in audit['executed_sources']:local.append(S.ROOT/'tests/glm5'/name)
    local.extend(S.ROOT/n for n in ['tests/glm5/recursive_branch14_materials.py','shared/dist_protocol/mechanism_trace.py','shared/dist_protocol/natural_register_runner.py','shared/dist_protocol/grammar.py','tools/provenance.py','deploy/storage_policy.py','deploy/storage_policy.json'])
    # Include project-local import dependencies without copying model or research trees.
    seen=set();pending=list(local)
    while pending:
        p=pending.pop().resolve()
        if p in seen:continue
        assert p.is_relative_to(S.ROOT.resolve()) and p.is_file(),p;seen.add(p)
        if p.suffix!='.py':continue
        source=p.read_text(encoding='utf-8-sig');compile(source,str(p),'exec');tree=ast.parse(source)
        for node in ast.walk(tree):
            modules=[]
            if isinstance(node,ast.Import):modules=[n.name for n in node.names]
            elif isinstance(node,ast.ImportFrom) and node.module:
                if node.level:
                    base=p.parent
                    for _ in range(node.level-1):base=base.parent
                    rel=node.module.replace('.','/');targets=[base/(rel+'.py'),base/rel/'__init__.py']
                    pending.extend(q for q in targets if q.exists() and q.is_relative_to(S.ROOT.resolve()));continue
                modules=[node.module]+[node.module+'.'+n.name for n in node.names]
            for module in modules:
                rel=module.replace('.','/')
                for root in [S.ROOT,S.ROOT/'tests/glm5']:
                    pending.extend(q for q in [root/(rel+'.py'),root/rel/'__init__.py'] if q.is_file())
    archive=S.OUT/'delivery-source.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(seen):z.write(p,str(p.relative_to(S.ROOT)).replace('\\','/'))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for p in seen:assert z.read(str(p.relative_to(S.ROOT)).replace('\\','/'))==p.read_bytes()
    S.save('source-package.json',dict(path=archive.name,sha256=S.file_sha(archive),files=[dict(path=str(p.relative_to(S.ROOT)),sha256=S.file_sha(p)) for p in sorted(seen)],limit='Code/dependency archive; model files excluded. Research data retained next to report, with separate manifest. Historical unused defaults in imported modules are not called; all task writes use S.configure onD.'))
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.targets=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['href','src'] and v and ':' not in v and not v.startswith('#'):self.targets.append(v)
    p=Links();p.feed((S.OUT/'report.html').read_text(encoding='utf8'))
    assert all((S.OUT/t).exists() for t in p.targets if t!='manifest.json')
    S.save('delivery-checks.json',dict(source_package_roundtrip=True,compiled_python_sources=sum(p.suffix=='.py' for p in seen),material_unittests='3 passed in material-tests-final.log',all2880_material_rows_audited=True,all2176_scores_recomputed=True,local_links=p.targets,plot_visual_review='Rendered expanded-calibration.png inspected: labels, intervals and counts visible.',html_review='Static local link and structural check only; no desktop/mobile browser rendering this stage.',protocol_regression='Not run: no shared protocol or runner changes. Existing protocol helpers reused unchanged.'))
    print('Packaged',len(seen),'source/dependency files; paired descriptions',pairs,flush=True)

if __name__=='__main__':run()
