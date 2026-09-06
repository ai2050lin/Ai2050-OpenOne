"""Post-delivery audit of real phase artifacts, memo continuity and retained raw fields."""
import ast,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
OUT=RESULT/'phase2676_native_mlp_delivery'


def main():
    clean=read(OUT/'analysis/cleanup_completed.json');checks={}
    for name in ('scientific_checks','delivery_checks','live_api_checks','browser_checks','post_cleanup_checks'):
        checks[name]=read(OUT/f'analysis/{name}.json')['all_checks_passed']
    final=[read(next(RESULT.glob(f'phase{p}_*/analysis/final.json'))) for p in range(2670,2677)]
    checks['7_actual_phases_complete']=all(r['phase']==2670+i and r['all_checks_passed'] and not r['language_mechanism_closed'] for i,r in enumerate(final))
    text=MEMO.read_text(encoding='utf-8-sig');heads=re.findall(r'^## Phase (\d+):[^\n]*\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]$',text,re.M)
    selected=[int(p) for p,_ in heads if 2670<=int(p)<=2676]
    checks['memo_2670through2676_once_contiguous']=selected==list(range(2670,2677))
    checks['all52_published_raw_hashes']=len(clean['kept'])==52 and all(sha(r['path'])==r['sha256'] for r in clean['kept'])
    checks['16_native_path_hashes']=len(clean['kept_native_path_packs'])==16 and all(sha(r['path'])==r['sha256'] for r in clean['kept_native_path_packs'])
    weights=clean['kept_learned_weight_vectors'];checks['actual_learned_vectors_hash_retained']=sha(weights['path'])==weights['sha256']
    checks['9700_allowlisted_unpublished_absent']=len(clean['targets'])==9700 and all(not Path(r['path']).exists() for r in clean['targets'])
    asset=RESULT/'client_visualization_assets/research_kernel/c42641_output_conditioned_crossmodel_field.json'
    checks['legacy_asset_unchanged']=sha(asset)=='4e15b56f30a89f5f523ddea4b35ab46394f2a9a9015ac565b1672b25142207cb'
    paths=sorted((ROOT/'tests/glm5').glob('phase267[0-6]_*.py'))+sorted((ROOT/'tests/glm5_temp').glob('phase2676_*.py'))+[ROOT/'server/native_mlp_parameter_query.py',ROOT/'server/native_atlas_heatmap_query.py',ROOT/'server/research_asset_service.py']
    paths+=sorted((ROOT/'tests/glm5').glob('phase267[79]_*.py'))+sorted((ROOT/'tests/glm5_temp').glob('phase267[79]_*.py'))
    source={}
    for p in paths:ast.parse(p.read_text(encoding='utf-8-sig'));source[str(p.relative_to(ROOT))]=sha(p)
    checks['source_ast_valid']=bool(source)
    changed=[str(p.relative_to(ROOT)) for p in paths]+['research/glm5/docs/AGI_GLM5_MEMO.md','frontend/src/components/app/NativeMlpParameterInspector.jsx','frontend/src/components/app/NativeParameterInspector.jsx','frontend/src/components/app/NativeAtlasHeatmap.jsx']
    run=subprocess.run(['git','-c','core.whitespace=blank-at-eol,blank-at-eof,space-before-tab,cr-at-eol','diff','--check','--',*changed],cwd=ROOT,capture_output=True,text=True)
    checks['scoped_diff_whitespace']=run.returncode==0
    nxt=read(OUT/'analysis/next_campaign.json');checks['next2677through2684_whole_plan']=nxt['same_goal'] and [r['phase'] for r in nxt['plan']]==list(range(2677,2685))
    report={'checks':checks,'all_checks_passed':all(checks.values()),'source_hashes_at_delivery':source,'memo_headings':[(p,t) for p,t in heads if 2670<=int(p)<=2676],
        'deleted_bytes':clean['deleted_bytes'],'source_hash_boundary':'Hashes describe delivery source versions, not an assertion that all files were immutable throughout engineering. Protocol/material/checkpoint integrity has separate measured hashes.',
        'scoped_diff_output':run.stdout+run.stderr,'mechanism_closed':False}
    save(OUT/'analysis/terminal_audit.json',report);print(json.dumps(checks));assert report['all_checks_passed']


if __name__=='__main__':main()
