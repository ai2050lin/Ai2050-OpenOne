"""Fresh8192 conditions, unchanged2687 measurement/maps, all-coordinate comparison."""
import gc,shutil
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
import phase2687_role_qkv_field as field
from phase2662_symmetric_mapping_contract import load_native
from phase2683_explicit_answer_audit import audit_records

OUT=RESULT/'phase2690_fresh_role_qkv_confirmation'
INITIAL=RESULT/'phase2687_role_qkv_field'
CONTRACT=RESULT/'phase2686_independent_role_contract'


def prepare():
    assert read(RESULT/'phase2689_native_qkv_scalar/analysis/final.json')['all_checks_passed']
    initial=read(INITIAL/'protocol/frozen.json');old=read(CONTRACT/'protocol/frozen.json')
    assert sha(TESTS/'phase2687_role_qkv_field.py')==initial['code_sha256']['phase2687_role_qkv_field.py']
    for name,digest in initial['code_sha256'].items():assert sha(TESTS/name)==digest,name
    assert sha(TESTS/'phase2683_explicit_answer_audit.py')==initial['parser_sha256']
    path=CONTRACT/'material/confirmation.json';assert sha(path)==old['material']['confirmation']['case_sha256'];rows=read(path)
    assert len(rows)==8192
    c={'material_sha256':sha(path),'parser_sha256':initial['parser_sha256'],'source_field_code_sha256':sha(TESTS/'phase2687_role_qkv_field.py'),
        'source_field_protocol_sha256':sha(INITIAL/'protocol/frozen.json'),'code_sha256':sha(Path(__file__)),
        'measurements':'Unchanged2687 functions, nativeBF16 fixed256 andnaturalcache32, same16languagefamilycells,allH/a/fullQKV/shapes/maps/publishedselection. Different material file already frozen2686 beforeinitial outputs.',
        'scope':'8192conditions=128newentity/lexicalbaseinstances*64factorconditions; structuralfamilies reused, not8192newabstractrules. All16familylanguage backgrounds compared without selecting onlysurvivors.',
        'budget':{'free_before':shutil.disk_usage(RESULT).free,'uncompressed_upper':field.budget_for(rows),'floor':8*1024**3}}
    p=OUT/'protocol/frozen.json'
    if p.exists():
        saved=read(p)
        for k in ('material_sha256','parser_sha256','source_field_code_sha256','code_sha256'):assert saved[k]==c[k],k
        return rows,saved
    comparison_bytes=16*4*(37*2*2560+36*2*9728)*2*2*2
    c['budget']['full_coordinate_initial_fresh_comparison_upper']=comparison_bytes
    assert c['budget']['free_before']>c['budget']['uncompressed_upper']['total']+comparison_bytes+8*1024**3
    save(p,c);return rows,c


def audit(rows,c):
    records=[];manifest=[];comparisons=[];cases=0;tokens=0
    for fam,lang in dict.fromkeys((r['family'],r['language']) for r in rows):
        stem=fam+'_'+lang;cell=read(OUT/f'analysis/cell_{stem}.json');assert cell['all_checks_passed'] and cell['cases']==512
        for p,h in cell['files'].items():assert sha(OUT/p)==h
        records.extend(read(OUT/f'analysis/records_{stem}.json'));cases+=cell['cases'];tokens+=cell['actual_tokens']
        comparison={};summary={}
        with np.load(OUT/f'maps/operations_{stem}.npz') as fresh,np.load(INITIAL/f'maps/operations_{stem}.npz') as old:
            assert set(fresh.files)==set(old.files)
            for name in fresh.files:assert np.isfinite(fresh[name]).all()
            for axis in field.OPERATIONS:
                for key in field.SHAPES:
                    pre=axis+'__'+key+'__';pos=fresh[pre+'positive'];neg=fresh[pre+'negative']
                    count=fresh[pre+'valid_count'] if key=='probability' else 256
                    count4=fresh[pre+'all4_valid_count'] if key=='probability' else 64
                    assert (pos+neg<=count).all() and (fresh[pre+'all4_positive']+fresh[pre+'all4_negative']<=count4).all()
                    assert (pos>=4*fresh[pre+'all4_positive']).all() and (neg>=4*fresh[pre+'all4_negative']).all()
                    assert (np.abs(fresh[pre+'sum'])<=fresh[pre+'sumabs']+1e-8).all()
                    if key not in ('h','a'):continue
                    for direction in ('positive','negative'):
                        label=pre+'all4_'+direction;before=old[label];after=fresh[label]
                        comparison[label+'__initial_count']=before;comparison[label+'__fresh_count']=after
                        summary[label]={'coordinates':before.size,'initial_full64':int((before==64).sum()),'fresh_full64':int((after==64).sum()),
                            'both_full64_same_direction':int(((before==64)&(after==64)).sum()),
                            'initial_zero_fresh_partial':int(((before==0)&(after>0)&(after<64)).sum())}
        path=OUT/f'maps/initial_fresh_comparison_{stem}.npz';np.savez_compressed(path,**comparison)
        comparisons.append({'cell':stem,'maps_sha256':sha(path),'complete_coordinate_counts':summary})
        with np.load(OUT/f'maps/alltoken_{stem}.npz') as z:
            for key in z.files:assert np.isfinite(z[key]).all()
    records.sort(key=lambda r:r['case_index']);assert [r['case_index'] for r in records]==list(range(8192))
    save(OUT/'analysis/records.json',records);score=audit_records(rows,records);save(OUT/'analysis/explicit_answer_audit.json',score)
    save(OUT/'analysis/initial_fresh_comparison.json',comparisons)
    for r in rows:
        for folder,prefix,include in (('field','case',r['published']),('field','natural',r['published']),('source','case',r['parameter_published'])):
            if not include:continue
            p=OUT/f'{folder}/{prefix}_{r["case_index"]:04d}.npz';assert p.exists()
            with np.load(p) as z:
                if prefix=='case' and folder=='field':
                    assert z['full__h'].shape==(37,len(r['prompt_ids']),2560)
                    if r['parameter_published']:assert z['full__a'].shape==(4,len(r['prompt_ids']),9728)
            manifest.append({'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'case_index':r['case_index']})
    save(OUT/'analysis/published_manifest.json',manifest)
    checks={'8192fresh_cases':cases==8192,'all_frozen_code_unchanged':sha(TESTS/'phase2687_role_qkv_field.py')==c['source_field_code_sha256'],
        'material_frozen_before_initial':sha(CONTRACT/'material/confirmation.json')==c['material_sha256'],'parser_unchanged':sha(TESTS/'phase2683_explicit_answer_audit.py')==c['parser_sha256'],
        'all16fullbackground_comparisons':len(comparisons)==16,'144published_files':len(manifest)==144,'64fresh_native_noops':read(OUT/'analysis/native256_preflight.json')['all_checks_passed']}
    assert all(checks.values())
    summary={'fresh_conditions':8192,'fresh_base_instances':128,'actual_tokens':tokens,'fullbackground_comparisons':comparisons,
        'behavior':score['groups'],'padded_natural_different':sum(not r['padded_natural_final_exact'] for r in records),
        'padded_natural_max_abs':max(r['padded_natural_final_max_abs'] for r in records)}
    save(OUT/'analysis/scientific_checks.json',{'all_checks_passed':True,'checks':checks,'summary':summary})
    finish(2690,'冻结算法下8192新实体与关系填充扩大确认：全部坐标复用与差异同时保留',OUT,
        {'provenance':str(Path(__file__)),'checks':checks,'summary':summary},
        '直接复用2687未改变的采集/配对/原生与自然输出函数，确认材料在首次发现输出前已于2686冻结。每个族所有H/MLP原坐标逐一对照旧/新四功能方向计数，不只重测旧通过坐标。',
        r'C^{4+}_{a,j,old},\ C^{4+}_{a,j,new};\quad R_{a,j}=[C^{4+}_{a,j,old}=64]\,[C^{4+}_{a,j,new}=64];\qquad X_{a,j}=\{C_{old},C_{new}\}\ \text{retained for every }j.',
        'C00164新前缀联合无操作控制；C0028192新条件全真实token H/MLP/QKV与自然32token输出；C003四外部操作各4096边；C00416族语言单元全部坐标新旧计数而非TopK；C00564完整H/16MLP/16QKV与64自然末状态；C006冻结明确答案评分全部失败分类。',
        '把对具体字词、名单/事实位置和角色目标的响应分开积累，新的部分响应和失效条件都构成编码地图，不能把是否全64同号作为唯一路线价值。',
        '独立的是新实体字符串与关系填充，语言模式族与模板结构仍复用；不存在8192独立抽象语义规则。两套P按物理来源位置展示而非强行跨词对齐。原场只保留预定例，token矩非完整高阶关联；自然cache与固定256仍分账。',
        '继续2691 Qwen14B至少4096条件全背景及五旧候选扩大确认，GLM4/DS7B至少512原生顺序复验；不借用4B坐标下标命名另一模型。随后2692实际参数账本和2693真实客户端交付。')


def main():
    assert not (OUT/'analysis/final.json').exists();rows,c=prepare();field.OUT=OUT
    model,tok=load_native('qwen4');assert model.dtype==torch.bfloat16 and not getattr(model,'is_quantized',False)
    save(OUT/'protocol/runtime.json',{'dtype':str(model.dtype),'actual_devices':sorted({str(p.device) for p in model.parameters()}),'quantized':False})
    field.qualify(model,tok,rows);field.collect(model,tok,rows,c)
    del model;gc.collect();torch.cuda.empty_cache();audit(rows,c)


if __name__=='__main__':main()
