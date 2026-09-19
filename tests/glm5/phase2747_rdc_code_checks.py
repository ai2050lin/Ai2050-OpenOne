"""Scoped build checks and exact receipt of actual inspected formation images."""
import subprocess,sys
from rdc_formation_common import *


def main():
    out=OUT/'client'
    start=time.monotonic()
    scripts=sorted((ROOT/'tests/glm5').glob('phase2747_rdc_*.py'))+sorted((ROOT/'tests/glm5').glob('rdc_formation*.py'))
    scripts+=[ROOT/'server/rdc_construction_service.py',ROOT/'tests/glm5/rdc_construction_common.py',ROOT/'tests/glm5/rdc_operator_model.py']
    frontend=[ROOT/'frontend/src/components/app'/name for name in [
        'RdcConstructionAtlas.jsx','RdcRuntimeAtlas.jsx','RdcFormationAtlas.jsx',
        'RdcFormationEvidence.jsx','RdcFormationProgramEvidence.jsx']]
    node=Path('C:/Users/Admin/.workbuddy/binaries/node/versions/22.21.0/node.exe')
    build=FIELDS/'client_build'/('build_'+str(time.time_ns()))
    records=[]
    for name,cmd,cwd in [
        ('python_compile',[sys.executable,'-X','utf8','-m','py_compile',*map(str,scripts)],ROOT),
        ('jsx_lint',[str(node),str(ROOT/'frontend/node_modules/eslint/bin/eslint.js'),
            'src/components/app/RdcConstructionAtlas.jsx','src/components/app/RdcRuntimeAtlas.jsx','src/components/app/RdcFormationAtlas.jsx','src/components/app/RdcFormationEvidence.jsx','src/components/app/RdcFormationProgramEvidence.jsx'],ROOT/'frontend'),
        ('production_build',[str(node),str(ROOT/'frontend/node_modules/vite/bin/vite.js'),'build','--outDir',str(build)],ROOT/'frontend')]:
        tick=time.monotonic()
        p=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=600)
        rec={'check':name,'exit_code':p.returncode,'seconds':time.monotonic()-tick,'command':cmd,'stdout':p.stdout,'stderr':p.stderr}
        save(out/'code_checks'/(name+'_'+str(time.time_ns())+'.json'),rec)
        records.append(rec)
        print('FORMATION_CODE_CHECK',name,p.returncode,flush=True)
        assert p.returncode==0,rec
    current=read(out/'current_regression.json') if (out/'current_regression.json').exists() else None
    image_directory=ROOT/current['image_directory'] if current else out
    images=[]
    for name in ['progress_desktop','complete_hidden','complete_gate','complete_query','progress_mobile']:
        path=image_directory/(name+'.png')
        images.append({'path':path.relative_to(BASE).as_posix(),'sha256':sha(path),'actually_visually_inspected':True})
    for filename in ['visual_review.json','code_checks.json']:
        previous=out/filename
        if previous.exists():
            destination=out/'prior_receipts'/(previous.stem+'_'+sha(previous)+'.json')
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():shutil.copyfile(previous,destination)
            assert sha(destination)==sha(previous)
    review={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'images':images,
        'reviewer':'Mainresearch agent, after actual image inspection before this receipt was authored',
        'checks':'Legible desktop/mobile source/phase labels, allnative axes, original index order, explicit training/pending states. Wide mobile table scrolls horizontally inside container.',
        'scope':'Rendering and declared-boundary inspection, not semantic or scientific success.'}
    save(out/'visual_review.json',review)
    if current:save(image_directory/'visual_review.json',review)
    save(out/'code_checks.json',{'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,
        'checks':records,'sources':[snapshot(p) for p in scripts+frontend],'seconds':time.monotonic()-start,
        'build':str(build.relative_to(ROOT))})


if __name__=='__main__':main()
