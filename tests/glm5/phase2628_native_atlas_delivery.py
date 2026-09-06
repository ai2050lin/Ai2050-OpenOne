"""Full-token basic checks, exact gate/up bookkeeping and original-client publication."""
import json, difflib, numpy as np
from phase2620_native_coordinate_contract import *
from phase2621_native_language_material import FAMILIES
from phase2623_native_parameter_algorithms import field,SOURCE

OUT=RESULT/'phase2628_native_atlas_delivery'
ASSET=RESULT/'client_visualization_assets/research_kernel/c42641_output_conditioned_crossmodel_field.json'

def row(label,values,phase=2628,kind='native_hidden_coordinate',layer=None):
    return {'label':label,'values':np.asarray(values,dtype='float64').tolist(),'phase':phase,'source':'native_parameter_campaign','coordinate_kind':kind,'layer':layer,'preview':True}

def panel(key,name,dimension,rows,semantics):
    return {'key':key,'model':name,'precision':'BF16 natural inference; FP32 stored physical values/derived quantities','coordinate_count':int(dimension),'coordinate_semantics':semantics,'rows':rows}

def parse_behavior():
    rows=[json.loads(x) for x in (RESULT/'phase2621_native_language_behavior/behavior/greedy.jsonl').read_text(encoding='utf-8').splitlines()]
    # Correct meaning is not the same as obeying the required verbal output protocol.
    punct=[]
    for r in rows:
        if r['family']!='punctuation':continue
        s=r['generated'].strip().casefold();expected='?' if r['variant']==0 else '!'
        symbol=s.replace('？','?').replace('！','!')==expected
        punct.append({**r,'symbol_or_name_content_correct':bool(r['answer_correct'] or symbol)})
    return {'punctuation_content_with_exact_symbol_alias':sum(r['symbol_or_name_content_correct'] for r in punct)/len(punct),
        'punctuation_protocol_strict':sum(r['strict_correct'] for r in punct)/len(punct),'n_punctuation':len(punct),
        'limits':'post-hoc content alias audit; original protocol strict scores unchanged; no semantic inability inferred from output-format deviation'},rows

def analyze():
    rows=read(SOURCE/'material/cases.json');H=field('hidden_anchor_boundary');A=field('mlp_anchor_boundary');g=field('final_gate');u=field('final_up')
    token_summary=[];gate_summary=[];profiles={};prefix_max=0;equal_pairs=0
    for family in FAMILIES:
        for lang in ('en','zh'):
            ij=[i for i,r in enumerate(rows) if r['family']==family and r['language']==lang]
            coordinate_energy=np.zeros((37,2560),dtype='float64');matched_tokens=0;gp=[];up=[];err=[];word_embed=[]
            for item in range(12):
                for form in (0,1):
                    pair=sorted([i for i in ij if rows[i]['index']==item and rows[i]['form']==form],key=lambda i:rows[i]['variant']);i,j=pair
                    a0=rows[i]['prompt_ids'];a1=rows[j]['prompt_ids'];prefix=next((t for t in range(min(len(a0),len(a1))) if a0[t]!=a1[t]),min(len(a0),len(a1)))
                    f0=np.load(SOURCE/f'field/fulltoken/case_{i:04d}.float32.npy',mmap_mode='r');f1=np.load(SOURCE/f'field/fulltoken/case_{j:04d}.float32.npy',mmap_mode='r')
                    if len(a0)==len(a1):
                        equal_pairs+=1
                        if prefix:prefix_max=max(prefix_max,float(np.max(np.abs(f0[:,:prefix]-f1[:,:prefix]))))
                        # Only same physical token index and identical token ID; no sequence-alignment transport claim.
                        positions=[t for t in range(prefix,len(a0)) if a0[t]==a1[t]]
                        if positions:
                            delta=f1[:,positions].astype('float64')-f0[:,positions]
                            coordinate_energy+=np.sum(delta*delta,axis=1);matched_tokens+=len(positions)
                    # Exact symmetric gate/up product identity on the real extension.
                    sig0=1/(1+np.exp(-np.clip(g[i].astype('float64'),-700,700)));sig1=1/(1+np.exp(-np.clip(g[j].astype('float64'),-700,700)))
                    s0=g[i]*sig0;s1=g[j]*sig1
                    gatepart=.5*(u[i]+u[j])*(s1-s0);uppart=.5*(s0+s1)*(u[j]-u[i]);delta=A[j,-1,-1].astype('float64')-A[i,-1,-1]
                    gp.append(gatepart);up.append(uppart);err.append(float(np.linalg.norm(gatepart+uppart-delta)/(np.linalg.norm(delta)+1e-12)))
                    if family=='word_sense':
                        p0=rows[i]['anchor_positions'][-1];p1=rows[j]['anchor_positions'][-1]
                        if a0[p0]==a1[p1]:word_embed.append(float(np.max(np.abs(H[i,0,0]-H[j,0,0]))))
            key=family+'/'+lang
            rms=np.sqrt(coordinate_energy/max(matched_tokens,1))
            if matched_tokens:profiles[key+'/same_token_downstream_rms']=rms.astype('float32')
            profiles[key+'/gate_product_change']=np.mean(gp,0).astype('float32');profiles[key+'/up_product_change']=np.mean(up,0).astype('float32')
            token_summary.append({'group':key,'equal_length_shared_downstream_tokens':matched_tokens,'embedding_same_token_rms':float(np.sqrt(np.mean(rms[0]**2))) if matched_tokens else None,
                'last_hidden_rms':float(np.sqrt(np.mean(rms[-1]**2))) if matched_tokens else None,'word_sense_same_token_embedding_max':max(word_embed) if word_embed else None})
            gate_summary.append({'group':key,'mean_gate_up_identity_relative_error':float(np.mean(err)),
                'gate_term_l2':float(np.linalg.norm(np.stack(gp))),'up_term_l2':float(np.linalg.norm(np.stack(up))),
                'sum_terms_l2':float(np.linalg.norm(np.stack(gp)+np.stack(up))),
                'interpretation':'symmetric product accounting; cancellation/reinforcement not proof of semantic gate/up roles'})
            print('fulltoken audit',key,flush=True)
    OUT.joinpath('field').mkdir(parents=True,exist_ok=True);np.savez(OUT/'field/allcoordinate_token_and_gate_profiles.npz',**profiles)
    behavior,_=parse_behavior()
    save(OUT/'analysis/basic_maps.json',{'equal_length_pairs':equal_pairs,'strict_causal_prefix_max_error':prefix_max,'tokens':token_summary,'gate_up':gate_summary,'behavior_correction':behavior})
    return profiles,{'equal_length_pairs':equal_pairs,'strict_causal_prefix_max_error':prefix_max,'tokens':token_summary,'gate_up':gate_summary,'behavior_correction':behavior}

def publish(profiles):
    payload=read(ASSET);cases=read(SOURCE/'material/cases.json');H=field('hidden_anchor_boundary');A=field('mlp_anchor_boundary')
    new=[];hidden=[];neurons=[];compiled=[]
    for i,r in enumerate(cases):
        if r['index']!=6 or r['form']!=0:continue
        for l in (0,1,6,18,36):
            for p,label in ((0,'anchor_last_subtoken'),(1,'answer_boundary')):
                hidden.append(row(f'{r["case_id"]}/{label}/raw checkpoint{l}',H[i,l,p],kind='embedding' if l==0 else 'raw_hidden_coordinate',layer=l))
        for l in (0,5,17,35):neurons.append(row(f'{r["case_id"]}/layer{l}/boundary SwiGLU neuron',A[i,l,-1],kind='mlp_intermediate_neuron',layer=l))
        compiled.append(row(f'{r["case_id"]}/postnorm output attribution',field('coordinate_attribution')[i],kind='logit_coordinate_attribution',layer=35))
    new.append(panel('phase2628_native_hidden','Native tokens: embedding and raw hidden states',2560,hidden,'model-local original j coordinates; raw final block is not final norm'))
    new.append(panel('phase2628_native_neurons','All 9728 MLP intermediate neurons',9728,neurons,'physical intermediate k, not residual coordinate j and not learned weight'))
    new.append(panel('phase2628_native_compiler','Actual output weight × normalized coordinate',2560,compiled,'first-token contrast accounting, bias recorded separately; not semantic necessity'))
    rr=[];gg=[]
    for key,val in profiles.items():
        if key.endswith('same_token_downstream_rms'):
            for l in (0,1,6,18,36):rr.append(row(key+f'/checkpoint{l}',val[l],layer=l))
        else:gg.append(row(key,val,kind='native_gate_up_product_term',layer=35))
    new.append(panel('phase2628_token_transport','Equal-length same-token full-coordinate responses',2560,rr,'RMS over all matched downstream tokens; no Top-K; profiles are observation not causal path'))
    new.append(panel('phase2628_gate_up','Natural gate/up product-change accounting',9728,gg,'all physical neurons; exact product split plus BF16 rounding remainder'))
    for key,directory in [('qwen4',SOURCE.name),('qwen14','phase2625_qwen14_native_parameters'),('glm4','phase2626_glm4_native_parameters'),('ds7','phase2627_ds7_native_parameters')]:
        source=RESULT/directory;info=read(source/'protocol/model.json');W=field('final_down_weights',source);aa=field('mlp_anchor_boundary',source);hh=field('hidden_anchor_boundary',source)
        wrows=[];srows=[]
        for j in (0,info['hidden_size']//3,2*info['hidden_size']//3,info['hidden_size']-1):
            wrows.append(row(f'{key}/final down WEIGHT row j={j}',W[j],kind='learned_weight_down_row',layer=info['layers']-1))
            wrows.append(row(f'{key}/case0 dmargin/dW_down[{j},k]',field('gradient_h',source)[0,j]*aa[0,-1,-1],kind='weight_local_derivative',layer=info['layers']-1))
        new.append(panel('phase2628_weights_'+key,f'{key}: real weights and individual-parameter derivatives',info['intermediate_size'],wrows,'x-axis all physical k; each labelled row specifies actual output j; scalar query reaches every j,k'))
        if key!='qwen4':
            for i in (0,2,4,6,8,10,12,14):
                for l in (0,1,info['layers']):srows.append(row(f'{key}/case{i}/raw checkpoint{l}',hh[i,l,-1],kind='embedding' if l==0 else 'raw_hidden_coordinate',layer=l))
            new.append(panel('phase2628_hidden_'+key,key+' native activation coordinates',info['hidden_size'],srows,'all model-local j; no cross-model coordinate alignment'))
    payload['models']=[p for p in payload['models'] if not p['key'].startswith('phase2628_')]+new;payload['phase']=2628
    payload['claim_boundary']='Native activation, MLP neuron, actual learned weight, and local derivative are separate objects. Last-block arithmetic is not semantic closure; no donor transfer or Top-K in primary analysis.'
    payload['summary']['model_rows']={p['key']:len(p['rows']) for p in payload['models']};payload['summary']['total_rows']=sum(payload['summary']['model_rows'].values())
    ASSET.write_text(json.dumps(payload,ensure_ascii=False,allow_nan=False,separators=(',',':')),encoding='utf-8')
    return {'panels':len(new),'asset_sha256':sha(ASSET),'asset_bytes':ASSET.stat().st_size,'scalar_query':'/api/research-assets/native-parameter?model=qwen4&case=0&j=853&k=3242'}

def main():
    profiles,summary=analyze();summary['client']=publish(profiles)
    result={'provenance':str(Path(__file__)),'summary':summary,'checks':{'all16_groups':len(summary['tokens'])==16,'equal_length_prefix_exact_zero':summary['strict_causal_prefix_max_error']==0,'all12_new_panels':summary['client']['panels']==12}}
    finish(2628,'全token传播、gate/up原生乘法图谱与单参数客户端交付',OUT,result,
        '扫描全部已采集token场，在等长两变体中对相同token ID且相同物理位置的下游坐标逐一核对；不把变长位置对齐当因果运输。对最终层gate/up自然变化作对称乘法记账。',
        r'\Delta a=\frac{u_1+u_0}{2}\Delta\operatorname{SiLU}(g)+\frac{\operatorname{SiLU}(g_1)+\operatorname{SiLU}(g_0)}2\Delta u;\quad R_{l,j}^2=\operatorname{mean}_{i,t}(H^1_{i,l,t,j}-H^0_{i,l,t,j})^2.',
        '768全token场、384条件pair；等长子集严格前缀校验，不等长只保留原场不纳入因果零门。苹果同token词嵌入等同性另记。既有客户端12新面板区分2560坐标与9728神经元，并呈现四模型真实weight行；标量查询可访问每个j,k，不限展示行。',
        '实际观察可把内容保持与排序、条件图纹与行为、gate项与up项的抵消/增强分别列账。标点复核显示输出符号本身也可能内容正确，不能把违反要求的词语格式解释为不识别标点；原严格分数不改。',
        '对称乘法分解是架构代数，不证明gate选择语义而up保存内容；全场RMS也会包含词形、位置和任务格式。截图颜色不是功能强弱排名；显示行是可读例，完整字段通过底层数组与标量查询保留。跨模型只做原生算法复验。',
        '同目标仍成立，自动进入冻结算法的768新上下文和更大实际扰动确认；同时扩大真正长句内容保持/排序分离阳性，最后做API、构建、连续Phase及原场存储审计。')

if __name__=='__main__':main()
