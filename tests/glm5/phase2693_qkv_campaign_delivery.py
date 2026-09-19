"""Lazy full-coordinate block descriptors. Publication is NOT campaign finish."""
import argparse,math,zipfile
import numpy as np
from phase2620_native_coordinate_contract import *

OUT=RESULT/'phase2693_qkv_campaign_delivery'
FIELD=RESULT/'phase2687_role_qkv_field'
FRESH=RESULT/'phase2690_fresh_role_qkv_confirmation'
TERMS=RESULT/'phase2688_native_qkv_terms'
SCALAR=RESULT/'phase2689_native_qkv_scalar'
LINK=RESULT/'phase2692_linked_native_ledger'
CROSS=RESULT/'phase2691_crossmodel_role_confirmation'

def npz_headers(path):
    """Headers only; never decompress full arrays merely to construct menus."""
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith('.npy'):continue
            with archive.open(name) as f:
                version=np.lib.format.read_magic(f)
                assert version in ((1,0),(2,0)),version
                reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
                shape,order,dtype=reader(f)
            assert not dtype.hasobject
            yield name[:-4],shape,dtype

def publish(preview=False):
    assert read(FIELD/'analysis/final.json')['all_checks_passed']
    assert read(SCALAR/'analysis/independent_condition_audit.json')['all_independent_checks_passed']
    if not preview:
        for p in (FRESH,CROSS,LINK):assert read(p/'analysis/final.json')['all_checks_passed']
    catalog={};retained={}
    def add(path,key,shape,dtype,group,title,boundary,label=None):
        if not shape or len(shape)>6:return
        assert shape[-1]>0
        path=path.resolve();assert path.is_relative_to(RESULT.resolve())
        retained[str(path)]={'path':str(path),'bytes':path.stat().st_size}
        name=f'phase2693_{group}_D{shape[-1]}'
        if name not in catalog:catalog[name]={'key':name,'title':title,'coordinate_count':shape[-1],
            'storage':'native_block_descriptor','phase':2693,'boundary':boundary,'blocks':[],'row_count':0}
        p=catalog[name];block={'file':path.relative_to(RESULT.resolve()).as_posix(),'array':key,
            'shape':list(shape),'row_count':math.prod(shape[:-1]),'label':label or f'{path.relative_to(RESULT)}/{key}'}
        if dtype==np.dtype('uint16') and group.startswith(('raw_','weights_qkv')):block['encoding']='native_bf16'
        p['blocks'].append(block);p['row_count']+=block['row_count']
    def whole(path,group,title,boundary,predicate=lambda k,s,d:True,label=None):
        for k,s,d in npz_headers(path):
            if predicate(k,s,d):add(path,k,s,d,group,title,boundary,label=f'{label}/{k}' if label else None)
    material=RESULT/'phase2686_independent_role_contract/material'
    for dataset,folder,split in (('initial',FIELD,'initial'),('fresh',FRESH,'confirmation')):
        if not (folder/'analysis/final.json').exists():continue
        for r in read(material/f'{split}.json'):
            if not r['published']:continue
            p=folder/f'field/case_{r["case_index"]:04d}.npz'
            for k,s,d in npz_headers(p):
                add(p,k,s,d,'raw_'+dataset+'_'+k,'八族双语 '+dataset+' '+k+' 原生完整坐标',
                    'Native BF16. Axes=as recorded; full__h=(checkpoint,actual token,coordinate), h/a=(checkpoint-or-layer,body/task,coordinate). H0 is E; H36 BEFORE final norm. Full MLP fields use layers23,26,27,28. Not the natural-cache trajectory.',r['case_id']+'/'+k)
            nat=folder/f'field/natural_{r["case_index"]:04d}.npz'
            whole(nat,'raw_'+dataset+'_natural','独立自然生成 post-final-norm 全坐标 '+dataset,
                'Actual natural-cache postnorm state, NOT fixed256 H36. Separate numerical protocol.',label=r['case_id'])
            if r['parameter_published']:
                sp=folder/f'source/case_{r["case_index"]:04d}.npz'
                for k,s,d in npz_headers(sp):
                    suffix=k.split('__',1)[1]
                    if suffix in ('actual_mask','scaling') or not s:continue
                    add(sp,k,s,d,'raw_'+dataset+'_source_'+suffix,'原生 '+dataset+' '+suffix+'（全部物理列）',
                        '16 predefined truth/v0 examples, eight explicit layers. Shape-prefix indices label token/query/head axes. actual_probability last axis=real source tokens (not hidden coordinates); future-source zero retained. normalized q/k use ALL128 head dimensions. Raw fields, not changed-parameter fields.',r['case_id']+'/'+k)
        for p in sorted((folder/'maps').glob('operations_*.npz')):
            for k,s,d in npz_headers(p):
                if not s:continue
                metric=k.split('__')[1]
                add(p,k,s,d,dataset+'_operations_'+metric,dataset+' 四语言操作完整响应 '+metric,
                    '512 conditions per family-language cell; each operation has256edges and64fourfunction groups. Native v1-v0. H firstaxis=checkpoint0..36, a=layer0..35; QKV firstaxis maps to layers[0,5,17,23,26,27,28,35]. Queryaxis0body/1task. Valid source positions vary. Integer pos/neg and all4 sign counts are NOT BF16; sum/sumabs retain amplitudes. No semantic closure gate.')
    wp=RESULT/'phase2685_native_attention_contract/weights/native_qkv_windows.npz'
    whole(wp,'weights_qkv','真实 Wq/Wk/Wv 完整输入行与 headnorm gamma',
        'Native BF16 checkpoint bits. 792 windows include repeated physical rows; 24 primary rows. All2560 inputs or128 gamma coordinates, not all model weights.')
    ep=TERMS/'weights/native_embeddings.npz'
    with np.load(ep) as z:token_ids=z['token_ids'].tolist()
    whole(ep,'checkpoint_E','224个真实token的检查点词嵌入全2560参数',
        'Actual checkpoint BF16 values stored exactly in float32. Ordered token IDs recorded in metadata, independently equal published H0.',lambda k,s,d:k=='embedding')
    catalog['phase2693_checkpoint_E_D2560']['blocks'][0]['row_labels']=[f'actual checkpoint token_id={i}' for i in token_ids]
    for p in sorted((TERMS/'field').glob('case_*.npz')):
        for k,s,d in npz_headers(p):add(p,k,s,d,'input_terms_'+k,'真实QKV完整输入项 '+k,
            '24 primary alltoken or792 twoquery windows as named, full2560inputs. BF16products exactly storedFP32; sums FP64analysis. Not ablation or transformed hidden basis.')
    for p in sorted((TERMS/'maps').glob('operations_*.npz')):
        whole(p,'weighted_operations','四操作 × 24真实QKV行 × 全部输入坐标响应',
            'Source512conditions/cell; v1-v0. Integer signs vs signed/absolute sums distinguished by array name; no nativeBF16 decoding for counts.')
    p=SCALAR/'maps/cumulative.npz'
    for k,s,d in npz_headers(p):
        if k=='completed_prefixes':continue
        add(p,k,s,d,'scalar_'+k.split('__')[0],'24576标量实测/局部理想预测完整聚合 '+k.split('__')[0],
            'Aggregate128prefixes, NOT percasechangedP. Shape=(48controls,4[dose.025sign-/+,dose.1sign-/+],body/task,head,lastcoordinate) for P/head. Psourcevalidcount separate. ideal means FP64localanalysis, not native changedmodel. Q/Klocalprediction not numerically validated; all discrepancies retained.')
    if (LINK/'analysis/prepared.json').exists():
        for p in sorted((LINK/'field').glob('L*_case_*.npz')):
            for k,s,d in npz_headers(p):
                if not s or s[-1] not in (2560,9728):continue
                add(p,k,s,d,'linked_'+('neuron' if s[-1]==9728 else 'residual'),'全来源 → 原生MLP '+('全部9728单元' if s[-1]==9728 else '全部2560坐标')+' 分账',
                    '16sourceexamples×8layers. Known arithmetic with separately observed rounding terms, NOT semantic closure. Fixed observed RMS denominator allocations are conditional, not ablation. All source/head/branch arrays stay addressable; no full3Dsource×input×neuron tensor claimed.')
        p=LINK/'field/natural_full_vocabulary_readout.npz'
        whole(p,'natural_readout_coordinates','自然 postnorm 状态与答案首token对比全坐标项',
            '64 actual native natural postnorm states, separate from fixed256 trajectory. Contrast terms use actual learned output rows in FP64 analysis; not a FP64model or wholeword semantics.',
            lambda k,s,d:bool(s) and s[-1]==2560)
    numeric=LINK/'numerical_baseline_audit'
    if (numeric/'result.json').exists():
        audit=read(numeric/'result.json');assert audit['all_audit_execution_checks_passed'] and audit['new_model_forwards']==audit['intervened_parameters']==0
        for r in audit['cases']:
            p=Path(r['numerical_map']);assert sha(p)==r['numerical_map_sha256']
            for k,s,d in npz_headers(p):
                stage=k.split('__')[2]
                add(p,k,s,d,'baseline_rounding_'+stage,'原生基线舍入核查 '+stage+'（保留全部误差坐标）',
                    '32predeclaredtruth/v0 examples initial16+new16 ×8layers. Labels distinguish ideal64 vs explicitround32 NumPyreference. Norm/RoPE diagnostics sum/sumabs/mismatchcount overALLactualtokens, everyhead/coordinate retained; P/AV retainwholequery/head/source axes. Counts NOT BF16. Baseline reconstruction only, NOT actualchanged-weightprediction, CUDAemulation or semantic mechanism closure.',r['case_id']+'/'+k)
    for model in ('qwen14','glm4','ds7','ds7_answer'):
        folder=CROSS/model
        if not (folder/'analysis/completion.json').exists():continue
        for p in sorted((folder/'maps').glob('counts_*.npz')):
            for k,s,d in npz_headers(p):add(p,k,s,d,model+'_counts_'+k.split('__')[0],model+' 新材料全部方向计数 '+k.split('__')[0],
                'Native ownmodel coordinates. Q14 32basegroups/family-language; otherprotocols4groups. v0-v1 opposite to Q4operationmaps v1-v0. Count arrays NOT BF16. Allpartial/negative/zero backgrounds retained.')
        for p in sorted((folder/'field').glob('case_*.npz')):
            for k,s,d in npz_headers(p):add(p,k,s,d,'raw_'+model+'_'+k,model+' 两展示例原生全坐标 '+k,
                'Native BF16 physical axes. full__h has actual token axis; a body/task. Same index acrossmodels not same meaning.')
    panels=list(catalog.values());assert len({p['key'] for p in panels})==len(panels)
    prefix='staged_' if preview else ''
    save(OUT/f'material/{prefix}client_panel_catalog.json',{'phase':2693,'phase_completed':False,'preview_only':preview,
        'display':'Every last-axis column retained; rows lazily indexed from full native arrays, no TopK or binned averages.',
        'boundary':'Native coordinate addressing/accounting/actual finite interventions are separate evidence. No semantic mechanism closure.',
        'embedding_token_ids':token_ids,'panels':panels})
    save(OUT/f'analysis/{prefix}publication.json',{'preview_only':preview,'phase_completed':False,'panels':len(panels),
        'total_logical_rows':sum(p['row_count'] for p in panels),'referenced_files':list(retained.values()),'raw_arrays_copied':False,
        'code_sha256':sha(Path(__file__)),'catalog_sha256':sha(OUT/f'material/{prefix}client_panel_catalog.json'),
        'required_before_completion':['2692final','independentactualdataAPItests','liveHTTP','frontendbuild','realbrowserQA','reference-safe-storage-audit','terminalaudit','same-goal-nextplan','MEMOappend']})
    print('2693',prefix,'PANELS',len(panels),'LOGICALROWS',sum(p['row_count'] for p in panels),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--preview',action='store_true');args=ap.parse_args();publish(args.preview)
