"""Independent entity/content crossing and frozen physical gated-MLP candidates."""
import itertools, shutil
import numpy as np
from transformers import AutoTokenizer
from phase2620_native_coordinate_contract import *
from phase2662_symmetric_mapping_contract import compose, encode, FAMILIES

OUT=RESULT/'phase2670_native_mlp_contract'
FIELD=RESULT/'phase2671_native_mlp_field'
SITES=((23,6197),(26,3594),(27,3221),(28,5952),(28,8513))
LAYERS=tuple(sorted({l for l,j in SITES}))
EN='Orson Elspeth Quentin Maribel Stellan Rosamund Caspian Winifred'.split()
ZH='穆禾 郁舟 蔺秋 沙澜 温芷 路遥 裴杉 柯雁'.split()

def row(fam,lang,e,c,f,v,o,p,q,m,multi=False):
    # The old generator supplies content indices 2/3, NOT entity indices.
    r=compose(fam,lang,c+2,f,v,o,p,q,m,0 if lang=='en' else 1,1,multi=multi)
    a,b=(EN if lang=='en' else ZH)[2*e:2*e+2]
    for key in ('text','body'):
        s=r[key].replace(r['entity_a'],'{{EA}}').replace(r['entity_b'],'{{EB}}').replace('{{EA}}',a).replace('{{EB}}',b)
        if c==1:
            replacements={
                'chronology': [('arrived','departed'),('到达','离开')],
                'syntax_role': [('congratulating','thanking'),('congratulated','thanked'),('praise','help'),('祝贺','感谢'),('赞扬','帮助')],
                'negation': [('unsigned','unstamped'),('signed','stamped'),('sign','stamp'),('签署','盖章')],
                'reference': [('map','letter'),('地图','信纸')],
                'punctuation': [('It is ready','The door is open'),('Already here','Still outside'),('已经准备好了','门已经打开了'),('已经到了','仍在外面')],
            }.get(fam,[])
            for before,after in replacements:s=s.replace(before,after)
        r[key]=s
    r.update(entity_a=a,entity_b=b,unit=e,content_instance=c,field_set='initial' if e<2 else 'confirmation',
        published=(e,c,f,v,o,p,q,m)==(2,0,0,0,0,0,0,0),
        fp_selected=e in (2,3) and (c,f,o,p,q)==(0,0,0,0,0),
        case_id=f'{fam}/{lang}/e{e}/c{c}/f{f}/v{v}/o{o}/p{p}/q{q}/m{m}')
    return r

def encoded(tok,r):
    r=encode(tok,r);enc=tok(r['prompt'],add_special_tokens=False,return_offsets_mapping=True)
    end=r['prompt'].index(r['body'])+len(r['body'])
    start=r['prompt'].index(r['body']);r['body_end_token']=max(i for i,(s,e) in enumerate(enc['offset_mapping']) if e>start and s<end)
    assert r['body_end_token']<len(r['prompt_ids'])-1
    return r

def build(tok):
    rows=[]
    for fam,lang,e,c,f,o,p,q,m,v in itertools.product(FAMILIES,('en','zh'),range(4),range(2),range(2),range(2),range(2),range(2),range(2),range(2)):
        rows.append(encoded(tok,{**row(fam,lang,e,c,f,v,o,p,q,m),'case_index':len(rows)}))
    return rows

def main():
    assert not (OUT/'analysis/final.json').exists()
    prior=RESULT/'phase2669_symmetric_multitoken_delivery';nextplan=read(prior/'analysis/next_campaign.json')
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    rows=build(tok);save(OUT/'material/cases.json',rows)
    masks={}
    for prefix,path in [('q4',RESULT/'phase2665_symmetric_coordinate_maps/maps/confirmed_masks.npz'),('q14',RESULT/'phase2668_qwen14_symmetric_replication/maps/old_coordinate_reconfirmation.npz')]:
        with np.load(path) as z:masks.update({prefix+'__'+k:z[k] for k in z.files})
    OUT.joinpath('maps').mkdir(parents=True,exist_ok=True);np.savez_compressed(OUT/'maps/frozen_masks.npz',**masks)
    # Upper bound does NOT assume compressibility. Native BF16 bits retained losslessly.
    n=len(rows);L,D,K=36,2560,9728;maxT=max(len(r['prompt_ids']) for r in rows)
    raw=n*2*(2*(L+1)*D+4*L*K+2*len(LAYERS)*D)
    published=sum(r['published'] for r in rows)*maxT*2*((L+1)*D+len(LAYERS)*(3*K+2*D))
    maps_bound=32*16*((L+1)*D+3*L*K+2*L*D)
    budget={'free_bytes':shutil.disk_usage(ROOT).free,'native_boundary_upper_bytes':raw,'published_upper_bytes':published,'alltoken_moments_upper_bytes':maps_bound,
        'reserve_other_analysis_bytes':1024**3,'floor_bytes':8*1024**3,'maximum_prompt_tokens':maxT}
    budget['fits_without_compression']=budget['free_bytes']-raw-published-maps_bound-budget['reserve_other_analysis_bytes']>=budget['floor_bytes']
    # Independent content must actually differ for every language family, not just its ID.
    bodies={}
    for r in rows:
        key=tuple(r[k] for k in ('family','language','unit','form','target_index','mention_order','probe_index','polarity','mapping'))
        bodies.setdefault(key,{})[r['content_instance']]=r['body']
    checks={'8192_distinct_prompts':len(rows)==len({r['prompt'] for r in rows})==8192,'4096_real_content_pairs':len(bodies)==4096 and all(v[0]!=v[1] for v in bodies.values()),
        '16_published_fulltoken':sum(r['published'] for r in rows)==16,'128_fp_prefixes':sum(r['fp_selected'] for r in rows)==128,
        'q4_2H5MLP':int(masks['q4__h'].sum())==2 and int(masks['q4__mlp'].sum())==5,
        'q14_62H125MLP':int(masks['q14__hidden_boundary__statement_truth'].sum())==62 and int(masks['q14__mlp_boundary__statement_truth'].sum())==125,
        'uncompressed_budget_safe':budget['fits_without_compression']}
    protocol={'plan':nextplan['plan'],'material_sha256':sha(OUT/'material/cases.json'),'mask_sha256':sha(OUT/'maps/frozen_masks.npz'),'q4_sites':SITES,'storage':budget,
        'q4_gate':'Original q0/m0,32family-language-probe groups, both entity folds, same old direction and target RMS exceeds form/order. All other q/m reported separately.',
        'q14_gate':'All62H/125MLP masks frozen; statement-truth signs across new groups. Not the4B old-sign/amplitude gate; no equal-index crossmodel interpretation.',
        'fields':'Every layer every H coordinate at body and task boundary; every MLP product unit both boundaries; gate/up allunits taskboundary; selected4layers full x/down taskboundary. Alltoken sums/squares for H,gate,up,product,x,down are streamed in float64 percoordinate.16published fullH and selected4layer fulltoken fields. Native BF16 stored as uint16 bits, lossless serialization not feature compression.',
        'crossmodel':'Each512:8families*2languages*2newentitypairs*2contentinstances*2targets*2polarities*2mappings; form/order/probe0. Own tokenization and native gatedMLP layout; not a full4B-factorial replication.',
        'scalar':'128heldoutprefixes:8families*2languages*2entitypairs*2targets*2mappings,content/form/order/probe/polarity0. FullAnswer:CODE/EOS objective; real gate/up/down scalar changes, matched ordinary/low-weight controls frozen beforeeffects. No donor states.',
        'claims':'8192 factorial conditions are not8192independent meanings. Native field and arithmetic identities are not semantic closure. Negative single-component necessity does not exclude redundant coupled paths.',
        'remaining_limitations':'Only2content instances perlanguage/family, same two template forms; changed verbs/nouns are controlled lexical variants, not open-world generalization. Different learned coordinates need not be individually semantic.'}
    save(OUT/'protocol/frozen.json',protocol);save(OUT/'analysis/preflight.json',checks);assert all(checks.values()),checks
    finish(2670,'实体—内容独立交叉与真实中层MLP参数主线冻结',OUT,{'provenance':str(Path(__file__)),'summary':{'checks':checks,'budget':budget,'plan':nextplan['plan']},'checks':checks},
        '复审上一批候选的实际适用条件；将人名与词义/数值/动词内容独立交叉，物理坐标全场作背景，五个既有MLP单元只作需解释的固定观察窗口。',
        r'N=8\times2\times4\times2\times2\times2\times2\times2\times2\times2=8192;\quad a_j=\operatorname{SiLU}(\sum_kW^g_{jk}x_k)\sum_kW^u_{jk}x_k;\quad m_i=\sum_jW^d_{ij}a_j.',
        '八族双语、4实体对、2独立内容实例、形式/顺序/目标/探问/极性/映射交叉；2对发现、2对确认。冻结4B两H五MLP及14B全部62H/125MLP；C001材料唯一性，C002真实内容变化，C003候选来源，C004未压缩存储预算。',
        '保留2667局部导数数值预测，但它仍是已知链式法则；EOS微效应未超过数值噪声。4B候选只过正常问法门，14B门定义不同，不能混称普适真值齿轮。此次优先追踪真实gate/up/down标量的条件化全输入输出纹理。',
        '只有两内容实例、两个实体集合，不能推出无限组合规律。streaming矩保留每坐标但不保留每个未展示token轨迹；uint16只存原BF16位，不是量化。只在新目录采集，旧已清理原场不重做。',
        '连续执行2671—2676全场、真实参数、条件确认、有限剂量、三模型顺序复验与完整交付；每个阶段独立记账，不把尚未执行任务记成完成。')

if __name__=='__main__':main()
