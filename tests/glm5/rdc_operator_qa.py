"""Native reading-comprehension behavior and original-coordinate query traces, never answer-fed generation."""
import argparse
import gc
import re
import string
import unicodedata
from collections import Counter, defaultdict
from rdc_operator_common import *


def cases(key='qwen4', scope='main'):
    balanced = gzread(BASE / 'qa_balanced_material.json.gz')
    hotpot = gzread(BASE / 'qa_multihop_material.json.gz')
    if key == 'qwen4':
        return ([r for r in balanced if r['split'] != 'confirmation'] + hotpot if scope == 'main'
                else [r for r in balanced if r['split'] == 'confirmation'])
    result = []
    for lang in ('en', 'zh'):
        result.extend([r for r in balanced if r['language'] == lang and r['split'] == 'confirmation'][:24])
    result.extend([r for typ in ('bridge','comparison') for r in [s for s in hotpot if s['question_type']==typ][:8]])
    assert len(result) == 64
    return result


def prompt(row, question_first=False):
    if row['language'] == 'en':
        instruction = 'Read the passages and answer the question. Return only the shortest complete answer. Do not explain.'
        c, q = 'Passages:\n'+row['full_context'], 'Question: '+row['question']
    else:
        instruction = '阅读材料并回答问题。只输出最简短但完整的答案，不要解释。'
        c, q = '材料：\n'+row['full_context'], '问题：'+row['question']
    return instruction+'\n\n'+('\n\n'.join([q,c]) if question_first else '\n\n'.join([c,q]))


def normalize_answer(text, lang):
    text = text.casefold()
    if lang == 'en':
        text = ''.join(c for c in text if c not in string.punctuation)
        return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())
    return ''.join(c for c in text if not c.isspace() and not unicodedata.category(c).startswith('P'))


def evaluate(text, answers, lang):
    normalized = normalize_answer(text, lang)
    f1 = []
    for answer in answers:
        gold = normalize_answer(answer['text'], lang)
        p, g = (normalized.split(), gold.split()) if lang == 'en' else (list(normalized), list(gold))
        common = sum((Counter(p)&Counter(g)).values())
        f1.append(2*common/(len(p)+len(g)) if p or g else 1.)
    return {'strict_full_answer': any(text.strip() == a['text'].strip() for a in answers),
            'normalized_full_EM': any(normalized == normalize_answer(a['text'],lang) for a in answers), 'answer_F1': max(f1),
            'nonempty': bool(text.strip()), 'scoring_scope': 'Whole decoded answer, never substring credit; English SQuAD-like token F1, Chinese punctuation-stripped character F1 (not official CMRC mixed-token scoring).'}


def repeated_ngrams(ids, n=4):
    grams = [tuple(ids[i:i+n]) for i in range(max(0,len(ids)-n+1))]
    return 0. if not grams else 1-len(set(grams))/len(grams)


class QueryTrace:
    def __init__(self, model, blocks):
        self.enabled = False
        self.blocks = blocks
        self.data, self.layers, self.handles = {}, {}, []
        def emb(m,a,o):
            if self.enabled:
                self.layers[0] = bits(o[0,-1])
        self.handles.append(model.get_input_embeddings().register_forward_hook(emb))
        for index, layer in enumerate(model.model.layers):
            def hidden(m,a,o,index=index):
                if self.enabled:
                    o = o[0] if isinstance(o,tuple) else o
                    self.layers[index+1] = bits(o[0,-1])
            self.handles.append(layer.register_forward_hook(hidden))
            if index in blocks:
                modules = [('x', layer.post_attention_layernorm), ('mlp',layer.mlp)]
                if hasattr(layer.mlp,'gate_proj'):
                    modules += [('gate',layer.mlp.gate_proj), ('up',layer.mlp.up_proj)]
                else:
                    assert hasattr(layer.mlp,'gate_up_proj'), 'Unreviewed native MLP architecture'
                    def combined(m,a,o,index=index):
                        if self.enabled:
                            gate, up = o.chunk(2,dim=-1)
                            self.data[f'L{index}_gate'] = bits(gate[0,-1])
                            self.data[f'L{index}_up'] = bits(up[0,-1])
                    self.handles.append(layer.mlp.gate_up_proj.register_forward_hook(combined))
                for key, module in modules:
                    def factor(m,a,o,index=index,key=key):
                        if self.enabled:
                            self.data[f'L{index}_{key}'] = bits(o[0,-1])
                    self.handles.append(module.register_forward_hook(factor))
                def activation(m,a,index=index):
                    if self.enabled:
                        self.data[f'L{index}_activation'] = bits(a[0][0,-1])
                self.handles.append(layer.mlp.down_proj.register_forward_pre_hook(activation))
                def attention(m,a,o,index=index):
                    if self.enabled:
                        self.data[f'L{index}_attention_sources'] = bits(o[1][0,:,-1])
                self.handles.append(layer.self_attn.register_forward_hook(attention))
    def close(self):
        for h in self.handles:
            h.remove()


def main(key, scope):
    import torch
    from rdc_operator_model import load
    selected = cases(key, scope)
    out = BASE / 'qa' / key / scope
    if (out / 'result.json').exists():
        return
    if scope == 'confirmation':
        assert (BASE/'operators/frozen.json').exists()
    start = time.monotonic()
    model, tok = load(key, out/'residency_v3',cpu_gib=11) if key=='qwen14' else load(key,out)
    load_profile=getattr(model,'_rdc_load_profile','native4B')
    device = model.get_input_embeddings().weight.device
    depth, width = len(model.model.layers), model.config.hidden_size
    blocks = [6,16,34] if key == 'qwen4' else [depth//6,depth//2,depth-2]
    compatible = all(hasattr(model.model.layers[b].mlp, 'gate_proj') or hasattr(model.model.layers[b].mlp,'gate_up_proj') for b in blocks)
    assert compatible, 'MLP architecture must have a reviewed native factor adapter'
    trace = QueryTrace(model, blocks)
    eos = model.generation_config.eos_token_id or tok.eos_token_id
    stopids = set(eos if isinstance(eos,list) else [eos])
    protocol={'timestamp': stamp(), 'source': snapshot(Path(__file__)), 'model': key, 'scope': scope,
        'case_ids': [r['question_id'] for r in selected], 'natural_original_questions': True, 'max_new_tokens': 48,
        'decoding': 'Greedy, native chat template; Qwen enable_thinking=False. No candidate answer or supporting-fact labels inserted. Native stop IDs respected.',
        'trace': 'Last prompt query all layers/all coordinates and selected blocks all units/all source attention. Generation and teacher scoring are separate forwards.',
        'trace_available': compatible, 'blocks_zero_index': blocks, 'full_native_width': width, 'stop_ids': sorted(stopids),
        'limits': 'Length-bounded source subset, not an official benchmark. Gold span/support labels are retrospective analysis only; correct answer alone does not prove the full reasoning chain.'}
    if not (out/'protocol.json').exists():immutable(out/'protocol.json',protocol)
    assert read(out/'protocol.json')['case_ids']==protocol['case_ids']
    save(out/'execution_source.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'original_protocol_sha':sha(out/'protocol.json'),
        'load_profile':load_profile,'scope':'Native precision/input/scoring unchanged; original completed commits retained.'})
    results = []
    try:
      with torch.inference_mode():
        committed=[r for r in selected if (out/'commits'/f'{r["question_id"]}.json').exists()]
        if committed and key=='qwen14':
            checkrow=read(out/'commits'/f'{committed[0]["question_id"]}.json')
            ci=torch.tensor([checkrow['prompt_ids']],device=device);trace.data,trace.layers,trace.enabled={},{},True
            replay=model.model(input_ids=ci,use_cache=False).last_hidden_state
            trace.enabled=False
            with np.load(out/'fields'/f'{checkrow["question_id"]}.npz') as old:
                assert np.array_equal(np.stack([trace.layers[l] for l in range(depth+1)]),old['H'])
                for name,value in trace.data.items():assert np.array_equal(value,old[name]),name
            cg=model.generate(input_ids=ci,do_sample=False,max_new_tokens=48,use_cache=True,
                pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,eos_token_id=eos)[0,ci.shape[1]:].tolist()
            assert cg==checkrow['generated_ids']
            save(out/f'residency_replay_check_{load_profile}.json',{'timestamp':stamp(),'question_id':checkrow['question_id'],
                'all_layer_all_coordinate_and_factor_bitwise':True,'full_generated_token_ID_sequence_bitwise':True,'new_load_profile':load_profile})
            trace.data,trace.layers={},{};del ci,replay,cg;torch.cuda.empty_cache()
        for i, row in enumerate(selected):
            sid = row['question_id']
            cp = out/'commits'/f'{sid}.json'
            if cp.exists():
                results.append(read(cp))
                continue
            user = prompt(row)
            actual = tok.apply_chat_template([{'role':'user','content':user}], tokenize=False, add_generation_prompt=True,
                                             **({'enable_thinking':False} if key.startswith('qwen') else {}))
            enc = tok(actual, add_special_tokens=False, return_offsets_mapping=True)
            ids = torch.tensor([enc['input_ids']], device=device)
            if trace:
                trace.data, trace.layers, trace.enabled = {}, {}, True
            post = model.model(input_ids=ids, use_cache=False).last_hidden_state
            if trace:
                trace.enabled = False
                field = {**trace.data, 'H':np.stack([trace.layers[l] for l in range(depth+1)]), 'postnorm':bits(post[0,-1])}
                from rdc_operator_capture import persist_arrays
                persist_arrays(out/'fields'/f'{sid}.npz',field)
            logits = model.lm_head(post[0,-1]).float()
            lp = logits.log_softmax(-1)
            sequence = model.generate(input_ids=ids, do_sample=False, max_new_tokens=48, use_cache=True,
                                      pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                                      eos_token_id=eos)[0,ids.shape[1]:].tolist()
            text = tok.decode(sequence, skip_special_tokens=True)
            score = evaluate(text, row['answers'], row['language'])
            gold_toks = [tok(a['text'], add_special_tokens=False)['input_ids'] for a in row['answers']]
            output = {k:row[k] for k in ('question_id','sample_id','source_group','language','split','question_type','question','answers')}
            output.update(actual_prompt=actual, prompt_ids=enc['input_ids'], token_offsets=enc['offset_mapping'],
                generated_ids=sequence, generated_text=text, stopped_by_native_EOS=any(t in stopids for t in sequence),
                hit_48_token_limit=len(sequence)==48 and not any(t in stopids for t in sequence),
                first_token_id=sequence[0] if sequence else None, first_token_reference_match=bool(sequence) and any(a and a[0]==sequence[0] for a in gold_toks),
                first_token_scope='Reference answer encoded in isolation; full-answer scoring is primary, this diagnostic does not assume context-boundary tokenization equivalence.',
                prompt_native_output_entropy=float(-(lp.exp()*lp).sum()), prompt_native_argmax=int(logits.argmax()),
                repeated_4gram_fraction=repeated_ngrams(sequence), **score)
            output['load_profile']=load_profile
            save(cp, output)
            results.append(output)
            save(out/'resource_progress.json',{'timestamp':stamp(),'completed_questions':len(results),'elapsed_seconds':time.monotonic()-start,
                'load_profile':load_profile,'model':key})
            if trace:
                trace.data, trace.layers = {}, {}
            del ids,post,logits,lp
            if key!='qwen4':
                torch.cuda.empty_cache()
            guard()
            if i < 2 or (i+1)%16 == 0:
                print('NATIVE_QA',key,scope,i+1,len(selected),'EM',sum(r['normalized_full_EM'] for r in results),'elapsed',round(time.monotonic()-start,1),flush=True)
            assert time.monotonic()-start < read(BASE/'resources.json')['per_process_ceiling_seconds']
    finally:
        if trace:
            trace.close()
        del trace,model
        gc.collect()
        torch.cuda.empty_cache()
    grouped = defaultdict(list)
    for r in results:
        grouped[r['language']+'/'+r['question_type']].append(r)
    report = {'timestamp': stamp(), 'model': key, 'scope': scope, 'sources': len(results), 'seconds': time.monotonic()-start,
        'normalized_full_EM': sum(r['normalized_full_EM'] for r in results)/len(results),
        'strict_full_answer': sum(r['strict_full_answer'] for r in results)/len(results), 'mean_F1': float(np.mean([r['answer_F1'] for r in results])),
        'native_EOS_fraction': float(np.mean([r['stopped_by_native_EOS'] for r in results])),
        'groups': {g:{'n':len(rr), 'EM':sum(r['normalized_full_EM'] for r in rr)/len(rr), 'F1':float(np.mean([r['answer_F1'] for r in rr])),
            'EOS':float(np.mean([r['stopped_by_native_EOS'] for r in rr]))} for g,rr in grouped.items()},
        'scope_note': 'Natural model QA capability assessment; failed cases retained. Successful answers do not independently verify the internal reasoning path. No gold answers fed into generation.'}
    save(out/'result.json',report)
    ledger('native_QA_'+key+'_'+scope,report['seconds'],sources=len(results))
    print('NATIVE_QA_COMPLETE',report,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model',default='qwen4',choices=['qwen4','qwen14','glm4'])
    p.add_argument('--scope',default='main',choices=['main','confirmation'])
    args=p.parse_args()
    main(args.model,args.scope)
