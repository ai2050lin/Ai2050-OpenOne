"""1024 BF16 model-local replication cases, two entities and two instruction styles."""
import gc,itertools
import numpy as np
import torch
from phase2620_native_coordinate_contract import *
from phase2662_symmetric_mapping_contract import OUT as CONTRACT,compose,encode,load_native,FAMILIES,length_audit
from phase2663_symmetric_mapping_calibration import run,behavior_groups,OUT as CAL

OUT=RESULT/'phase2668_qwen14_symmetric_replication'


def analyze(cases):
    table={tuple(r[k] for k in ('family','language','unit','style','target_index','probe_index','polarity','mapping')):r for r in cases};counts={};oldcounts={};profiles=[];moments={};ntokens={}
    OUT.joinpath('maps').mkdir(parents=True,exist_ok=True)
    with np.load(CONTRACT/'maps/frozen_masks.npz') as frozen:
        for fam,lang in itertools.product(FAMILIES,('en','zh')):
            for unit,style in itertools.product((6,7),(0,1)):
                values={};pack={}
                for p,q,m in itertools.product((0,1),repeat=3):
                    a,b=[table[fam,lang,unit,style,v,p,q,m]['case_index'] for v in (0,1)]
                    with np.load(OUT/f'field/case_{a:04d}.npz') as z0,np.load(OUT/f'field/case_{b:04d}.npz') as z1:
                        values[p,q,m]={k:z0[k].astype('float64')-z1[k].astype('float64') for k in ('hidden_boundary','mlp_boundary')}
                        if (p,q,m)==(0,0,0):pack['source_target_anchor']=z0['hidden_anchor']-z1['hidden_anchor']
                        key=f'{fam}/{lang}/s{style}'
                        if key not in moments:moments[key]=[np.zeros_like(z0['hidden_token_sum']),np.zeros_like(z0['hidden_token_sumsq'])];ntokens[key]=0
                        for v,z in enumerate((z0,z1)):
                            moments[key][0]+=z['hidden_token_sum'];moments[key][1]+=z['hidden_token_sumsq'];ntokens[key]+=len(table[fam,lang,unit,style,v,p,q,m]['prompt_ids'])
                for metric in ('hidden_boundary','mlp_boundary'):
                    baseline=np.sign(values[0,0,0][metric])
                    for key,v in values.items():pack[f'p{key[0]}q{key[1]}m{key[2]}__{metric}']=v[metric].astype('float32')
                    for hyp in ('statement_truth','question_affirmative','answer_label'):
                        good=baseline!=0
                        for p,q,m in itertools.product((0,1),repeat=3):good &= np.sign(values[p,q,m][metric])==baseline*(-1)**(p if hyp=='statement_truth' else p+q if hyp=='question_affirmative' else p+q+m)
                        key=metric+'__'+hyp;counts.setdefault(key,np.zeros_like(good,dtype='int16'));counts[key]+=good
                    profiles.append({'family':fam,'language':lang,'unit':unit,'style':style,'metric':metric,'baseline_rms':np.sqrt(np.mean(values[0,0,0][metric]**2,axis=-1)).tolist()})
                np.savez_compressed(OUT/f'maps/{fam}_{lang}_u{unit}_s{style}.npz',**pack)
        for k,v in counts.items():oldcounts[k]=(frozen['q14_'+k].astype(bool)&(v==64)).astype('uint8')
    np.savez_compressed(OUT/'maps/allcoordinate_sign_counts.npz',**counts);np.savez_compressed(OUT/'maps/old_coordinate_reconfirmation.npz',**oldcounts);save(OUT/'analysis/profiles.json',profiles)
    np.savez_compressed(OUT/'maps/alltoken_moments.npz',**{key+'__'+part:((v[0]/ntokens[key]) if part=='mean' else np.sqrt(v[1]/ntokens[key])).astype('float32') for key,v in moments.items() for part in ('mean','rms')});save(OUT/'analysis/alltoken_counts.json',ntokens)
    return {'all64groups_by_layer':{k:(v==64).sum(-1).tolist() for k,v in counts.items()},'old_candidates_reconfirmed_by_layer':{k:v.sum(-1).tolist() for k,v in oldcounts.items()},
        'boundary':'64groups=16family/languages*2entitypairs*2styles. Oldcandidate rule is sign pattern only, not old mean direction and not target>form/order RMS. New symmetric instructions and demos differ from old protocol.'}


def main():
    assert not (OUT/'analysis/final.json').exists();model,tok=load_native('qwen14');choice=read(CAL/'protocol/selected.json')['selection'];cases=[]
    for fam,lang,unit,style,v,p,q,m in itertools.product(FAMILIES,('en','zh'),(6,7),(0,1),(0,1),(0,1),(0,1),(0,1)):
        r=compose(fam,lang,unit,0,v,0,p,q,m,style,choice[lang]['shots']);r['published']=(unit,v,p,q,m)==(7,0,0,0,0);cases.append(encode(tok,{**r,'case_index':len(cases)}))
    save(OUT/'material/cases.json',cases);info={'model':'local Qwen3-14B','dtype':str(model.dtype),'quantized':bool(getattr(model,'is_quantized',False)),
        'device_map':getattr(model,'hf_device_map',{}),'dimensions':{'layers':model.config.num_hidden_layers,'hidden':model.config.hidden_size,'mlp':model.config.intermediate_size},
        'loader':'BF16,device_mapauto12GiB CUDA/20GiB CPU,synchronous pread process-local wrapper restored; no model/library edits','length_audit':length_audit(cases)};save(OUT/'protocol/model.json',info)
    records=run(model,tok,cases,OUT,fields=True);del model;gc.collect();torch.cuda.empty_cache();coordinate=analyze(cases)
    by_style={f's{style}':behavior_groups([r for r in records if r['style']==style]) for style in (0,1)}
    checks={'1024_cases':len(records)==1024,'32_published':sum(r['published'] for r in cases)==32,'all6coordinate_rule_maps':len(coordinate['all64groups_by_layer'])==6,'nonquantized':not info['quantized']};assert all(checks.values())
    finish(2668,'1024条14B对称规则与中层全坐标纹理扩大复验',OUT,{'provenance':str(Path(__file__)),'summary':{'behavior_by_style':by_style,'coordinates':coordinate,'model':info},'checks':checks},
        '同一模型本地物理基底中比较真值/问题肯定性/答案标签三个全坐标方向图。冻结旧中层候选后扩大到双实体对和双指令样式；模型顺序、BF16非量化、自动CPU/GPU分配。',
        r'D^{u,s}_{p,q,m}=H(v=0)-H(v=1);\quad C^{hyp}_{l,j}=\sum_{g,u,s}\mathbf1[\operatorname{sgn}D^{u,s}_{p,q,m}=(-1)^{e(hyp)}\operatorname{sgn}D^{u,s}_{0,0,0}].',
        '16族语言组×2实体对×2指令×目标/探问/极性/映射=1024条件，64方向比较组；41×5120 H和40×17408 MLP全部边界；32展示例保留全部tokenH，其余源角色和token矩保留到整批清理。',
        '不把不同模型相同下标视作同一功能。旧H26/MLP24纹理在更复杂指令下是否保持，与自然行为同时记账；任何层有可复用纹理都保留，而非只看末层。',
        '每语言仍仅两实体对、正文form/order0，双样式主要改变任务指令；未复制目标幅度超过句式/順序完整门。旧方向候选交集不要求与旧均值同号，不能冒称方向的严格原样复现；阴性不排除混合条件编码。',
        '整合全坐标主线与多token参数工具，完成2669原客户端交付、精确查询、审计及未展示原场清理，再继续同目标研究。')


if __name__=='__main__':main()
