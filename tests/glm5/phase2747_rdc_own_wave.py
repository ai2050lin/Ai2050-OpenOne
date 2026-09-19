"""Qualified layer-major scheduling of unchanged native B1/B8 own histories."""
import argparse
from rdc_formation_common import *
from rdc_formation_history import freeze,record,summarize
from rdc_formation_microbatch import NativeWave,generate_wave
from rdc_formation_glm_wave import GlmNativeWave
from rdc_formation_readout import CUDA_FORMATION,checked_arrays


def execution_revision():
    directory=OUT/'engineering/microbatch';engine=Path(__file__).with_name('rdc_formation_microbatch.py')
    unit=read(directory/'unit_current.json');assert sha(ROOT/unit['path'])==unit['sha256']
    checked=read(ROOT/unit['path']);assert checked['all_passed'] and checked['engine']['sha256']==sha(engine)
    sliced=read(directory/'slice_reader_current.json');assert sha(ROOT/sliced['path'])==sliced['sha256']
    audit=read(ROOT/sliced['path']);assert audit['all_passed'] and audit['engine']['sha256']==sha(engine)
    path=directory/('execution_revision_'+sha(engine)[:16]+'.json')
    if not path.exists():
        immutable(path,{'timestamp':stamp(),'source':snapshot(__file__),'engine':snapshot(engine),
            'original_contract_sha256':sha(directory/'contract.json'),'unit':unit,'slice_reader_audit':sliced,
            'change':'Selected allH rows own exact selected bytes instead of retaining uncollected B8 backing rows. Disk-offloaded BF16 tensors read only their validated original byte region, rather than opening an entire checkpoint shard. No model arithmetic, weights, microbatch membership, caps or collected coordinate values changed.',
            'qualification':'New engine SHA requires new Q4 wholepilot replay. Both larger models must independently replay six fullH fixtures and all16complete native pilot packets.',
            'memory':'CPU available/commit headroom and CUDA allocated/reserved/free recorded throughout every step. First200 original groups retained subject to fresh preflight; no operating-system or user-process mutation.'})
    return path


def prepare():
    folder=OUT/'engineering/microbatch';path=folder/'contract.json'
    if path.exists():return read(path)
    unit=read(folder/'unit_current.json');assert sha(ROOT/unit['path'])==unit['sha256']
    assert read(ROOT/unit['path'])['all_passed']
    protocol,material=freeze()
    value={'timestamp':stamp(),'source':snapshot(__file__),'engine':snapshot(Path(__file__).with_name('rdc_formation_microbatch.py')),
        'original_protocol_sha256':sha(OUT/'own_history/protocol.json'),'material_sha256':protocol['material_sha256'],
        'unchanged':['All512sample IDs, tokenizer, prompts, targets and native BF16 parameters.',
            'Every first-prefix operation has original B1 shape; each generation group has the same original B8 rows and left padding.',
            'Each microbatch has its own unchanged native DynamicCache, positions and masks; no cross-sample KV or answers.',
            'Original96/128caps, native greedy vocabulary, EOS, conservative scoring and full-field collection.'],
        'changed':['Outer loop may run all independent requests through one original decoder layer before advancing layers.',
            'Same current-layer tensors remain temporarily on CUDA and are released before the next layer.',
            'Q14 uses6GiB GPU/6GiB CPU auto-dispatch to reserve device space for independent caches; GLM uses original dispatch.',
            'First wave combines192natural rows plus first8controlled; later unchanged B8 groups form waves of at most128rows.'],
        'admission':['24CPU synthetic arithmetic/cache checks; never counted as model evidence.',
            'Original Q4 CUDA: six old full-H fixtures plus all16complete original pilot packets must replay exactly.',
            'Each larger model: six old full-H fixtures must replay; all16original full-cap pilot packets must match before committing first200main rows.'],
        'resource_rule':'Preflight forecast of all own-cache bytes plus declared activation/current-weight reserve must fit currently free CUDA memory. Lower scheduling width if necessary without changing any B8 group.',
        'scientific_status':'Engineering execution equivalence, not extracted semantics, new math, or a larger arithmetic batch.',
        'native_Q14_previous_failures':'Original runtime safetensor mmap reads failed twice with0xc0000005; process-local pread already passed six exact native fixtures.'}
    immutable(path,value);return value


def fixtures(model,key,engine):
    import torch
    rows=gzread(BASE/'material.json.gz')['models'][key]['rows'];indices=[0,66,129,195,258,319]
    requests=[{'input_ids':torch.tensor([rows[i]['prompt_ids']],device='cuda'),'use_cache':True,'collect_hidden':True} for i in indices]
    outputs=engine.forward(requests);checks=[]
    for index,value in zip(indices,outputs):
        row=rows[index]
        with np.load(BASE/'capture'/key/'fields'/(row['sample_id']+'.npz')) as z:
            assert np.array_equal(value['hidden'][0],z['prefix_layers']),(key,index,'allH')
            assert np.array_equal(bits(value['postnorm'][0]),z['prefix_postnorm']),(key,index,'postnorm')
        checks.append({'row_index':index,'all_hidden_boundaries_bit_equal':True,'postnorm_bit_equal':True})
    del requests,outputs
    print('FORMATION_WAVE_FIXTURES',key,len(checks),flush=True)
    return checks


def compare(packet,path,key,sid):
    original=checked_arrays(read(path)['field'])
    mismatches=[k for k in set(original)|set(packet) if k not in original or k not in packet or not np.array_equal(original[k],packet[k])]
    if mismatches:
        receipt=commit_array('microbatch_failure/'+key,sid+'_'+str(time.time_ns()),**packet)
        save(OUT/'engineering/microbatch'/('mismatch_'+str(time.time_ns())+'.json'),{
            'timestamp':stamp(),'sample_id':sid,'keys':mismatches,'original_record':path.relative_to(ROOT).as_posix(),'new_failure_field':receipt})
    assert not mismatches,(key,sid,mismatches)


def main(key,qualification=False):
    import torch
    from rdc_native_tail import cuda_singleton
    from rdc_operator_model import memory
    cuda_singleton(CUDA_FORMATION);contract=prepare();revision=execution_revision();protocol,material=freeze();rows=material[key]
    assert sha(OUT/'own_history/material.json.gz')==contract['material_sha256']
    assert read(OUT/'own_history'/key/'native/pilot.json')['all_passed']
    engine_path=Path(__file__).with_name('rdc_formation_microbatch.py');engine_sha=sha(engine_path)
    adapter_path=Path(__file__).with_name('rdc_formation_glm_wave.py') if key=='glm4' else None
    adapter_receipt=snapshot(adapter_path) if adapter_path else None
    if adapter_path:
        unit_pointer=read(OUT/'engineering/microbatch/unit_current.json')
        unit=read(ROOT/unit_pointer['path'])
        assert unit['glm_architecture_adapter']['sha256']==sha(adapter_path)
        assert sum(r['model_type']=='glm' for r in unit['checks'])==12
    folder=OUT/'engineering/microbatch'/('qwen4_qualification_'+engine_sha[:12]) if qualification else OUT/'own_history'/key/'native'
    finish=folder/'result.json'
    if finish.exists():return
    if qualification:assert key=='qwen4'
    else:
        qualified=read(OUT/'engineering/microbatch/current_qualification.json')
        assert key in {'qwen14','glm4'} and sha(ROOT/qualified['path'])==qualified['sha256']
        result=read(ROOT/qualified['path']);assert result['all_passed'] and result['engine']['sha256']==engine_sha
    attempt=folder/'wave_attempts'/('attempt_'+str(time.time_ns()))
    attempt.mkdir(parents=True,exist_ok=False)
    for name in ['wave_admission.json','wave_progress.json']:
        previous=folder/name
        if previous.exists():shutil.copyfile(previous,attempt/('previous_'+name))
    start=time.monotonic();model=engine=None
    try:
        gpu_budget=6 if key=='qwen14' else (11 if key=='glm4' else None)
        immutable(attempt/'scheduler_resource_profile.json',{'timestamp':stamp(),'source':snapshot(__file__),
            'model':key,'requested_GPU_GiB':gpu_budget,'requested_CPU_GiB':6,
            'original_microbatch_contract_sha256':sha(OUT/'engineering/microbatch/contract.json'),
            'refinement':'Q14 retains tested6GiB GPU residency. GLM uses11GiB rather than the original12GiB request: its original pilot held12682119168 CUDA parameter bytes, leaving less than the declared first200-wave KV plus1.4GiB working reserve on this card. Only parameter residency changes, not BF16 bytes or native B1/B8 computation.',
            'allocator_rule':'Release unused allocator pages before every fresh wave preflight; never change live tensors.',
            'required_admission':'Six historical fullH/postnorm fixtures and every complete original16pilot packet must match exactly before first-wave commit.'})
        model,tok=load(key,attempt/'load',gpu_limit=gpu_budget)
        engine=(GlmNativeWave if key=='glm4' else NativeWave)(model);admission=fixtures(model,key,engine)
        admission_receipt={'timestamp':stamp(),'all_passed':True,'native_checks':admission,
            'whole_original_pilot_replay_pending':True,'engine':snapshot(engine_path),
            'architecture_adapter':adapter_receipt,
            'attempt':attempt.relative_to(ROOT).as_posix(),'execution_revision':revision.relative_to(ROOT).as_posix()}
        save(attempt/'admission.json',admission_receipt);save(folder/'wave_admission.json',admission_receipt)
        batches=[rows[:8],rows[192:200]] if qualification else [rows[i:i+8] for i in range(0,512,8)]
        waves=[batches] if qualification else [batches[:25]]+[batches[i:i+16] for i in range(25,len(batches),16)]
        records=[];replays=read(folder/'wave_pilot_replay.json').get('checks',[]) if (folder/'wave_pilot_replay.json').exists() else [];wave_times=[]
        for wi,proposed in enumerate(waves):
            all_existing=all((folder/'records'/(r['sample_id']+'.json')).exists() for batch in proposed for r in batch) if not qualification else False
            if all_existing:
                for batch in proposed:
                    for row in batch:
                        r=read(folder/'records'/(row['sample_id']+'.json'));checked_arrays(r['field']);records.append(r)
                continue
            config=model.config
            cache_bytes=sum(len(batch)*(max(len(r['model_prompt_ids']) for r in batch)+max(r['max_new_tokens'] for r in batch))
                *config.num_hidden_layers*config.num_key_value_heads*config.head_dim*2*2 for batch in proposed)
            # Completed waves release all own-KV tensors, but the CUDA
            # allocator can keep their now-unused pages reserved. A device
            # free-memory guard must not mistake those reusable pages for
            # live model/cache data. Do not alter any tensor, group or shape.
            before_release={'allocated':torch.cuda.memory_allocated(),
                'reserved':torch.cuda.memory_reserved(),'device_free':torch.cuda.mem_get_info()[0]}
            gc.collect();torch.cuda.empty_cache()
            free,total=torch.cuda.mem_get_info()
            assert free>cache_bytes+int(1.4*1024**3),('Insufficient wave-cache headroom; choose smaller scheduling width, not altered B8 shapes',wi,free,cache_bytes)
            save(folder/('wave_'+str(wi)+'_resource.json'),{'timestamp':stamp(),'wave':wi,'microbatches':len(proposed),
                'expressions':sum(map(len,proposed)),'maximum_declared_KV_bytes':cache_bytes,'free_CUDA_bytes':free,
                'working_reserve_bytes':int(1.4*1024**3),'memory':memory(),
                'unused_allocator_cache_release':{'before':before_release,
                    'after_allocated':torch.cuda.memory_allocated(),'after_reserved':torch.cuda.memory_reserved(),
                    'after_device_free':free,'scope':'Only unreferenced allocator pages released before preflight; live tensors, arithmetic and B8 membership unchanged.'}})
            tick=time.monotonic()
            def progress(stage,step,tokens,seconds):
                free,total=torch.cuda.mem_get_info()
                value={'timestamp':stamp(),'wave':wi,'stage':stage,'own_step':step,
                    'generated_tokens':tokens if stage=='own_step' else 0,
                    'first_prefix_scored_expressions':tokens if stage=='first_B1' else sum(map(len,proposed)),
                    'wave_seconds':seconds,'committed_expressions':len(records),'qualification':qualification,
                    'memory':memory(),'CUDA_allocated':torch.cuda.memory_allocated(),
                    'CUDA_reserved':torch.cuda.memory_reserved(),'CUDA_free':free,
                    'attempt':attempt.relative_to(ROOT).as_posix()}
                save(folder/'wave_progress.json',value)
                save(attempt/'progress'/('wave_'+str(wi)+'_'+stage+'_'+str(step)+'.json'),value)
                if stage=='first_B1' or step%8==0:print('FORMATION_WAVE',key,wi,stage,step,tokens,round(seconds,2),flush=True)
            packets,stop=generate_wave(model,tok,proposed,engine,progress)
            elapsed=time.monotonic()-tick;wave_times.append(elapsed)
            for batch,bb in zip(proposed,packets):
                for row,packet in zip(batch,bb):
                    sid=row['sample_id'];reference=OUT/'own_history'/key/'native/pilot_records'/(sid+'.json')
                    if reference.exists():
                        compare(packet,reference,key,sid)
                        if not any(r['sample_id']==sid for r in replays):replays.append({'sample_id':sid,'complete_original_pilot_packet_bit_equal':True})
            if wi==0:
                assert len(replays)==16
                save(folder/'wave_pilot_replay.json',{'timestamp':stamp(),'all_passed':True,'checks':replays,
                    'source':snapshot(__file__),'whole_first_wave_completed_before_main_commit':True})
            if qualification:
                save(finish,{'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'model':key,
                    'numerical_admission':admission,'pilot_replays':replays,'trajectories':16,
                    'actual_generated_tokens':sum(len(p['generated_ids']) for bb in packets for p in bb),
                    'seconds':time.monotonic()-start,'wave_seconds':wave_times,'peak_CUDA_bytes':torch.cuda.max_memory_allocated(),
                    'engine':snapshot(engine_path),'execution_revision':revision.relative_to(ROOT).as_posix(),
                    'scope':'All original Q4 pilot fields/tokens match; offloaded layer-cache reuse still requires larger-model fixture/full-pilot replay.'})
                save(OUT/'engineering/microbatch/current_qualification.json',{'path':finish.relative_to(ROOT).as_posix(),'sha256':sha(finish)})
                ledger('phase2747_wave_Q4_qualification',time.monotonic()-start)
                print('FORMATION_WAVE_QUALIFIED',key,len(replays),flush=True);return
            for batch,bb in zip(proposed,packets):
                for row,packet in zip(batch,bb):
                    sid=row['sample_id'];path=folder/'records'/(sid+'.json')
                    if path.exists():compare(packet,path,key,sid);r=read(path)
                    else:
                        r=record(row,packet,tok,stop,'native')
                        receipt=commit_array('own_history/'+key+'/native',sid,**packet)
                        r.update(timestamp=stamp(),source=snapshot(__file__),field=receipt,batch_ids=[r['sample_id'] for r in batch],
                            scheduling_wave=wi,wave_microbatches=len(proposed),allocated_wave_seconds=elapsed/sum(map(len,proposed)),
                            pilot_full_array_replay_exact=any(v['sample_id']==sid for v in replays))
                        save(path,r)
                    records.append(r)
            save(folder/'progress.json',{'timestamp':stamp(),'pilot':False,'expressions':len(records),'total':512,
                'generated_tokens':sum(len(r['generated_ids']) for r in records),'seconds':time.monotonic()-start,'execution':'layer_major_native_microbatches'})
            del packets;gc.collect();storage_guard()
        assert len(records)==512 and len({r['sample_id'] for r in records})==512 and len(replays)==16
        completed_waves={}
        for r in records:
            index=str(r['scheduling_wave'])
            completed_waves[index]=completed_waves.get(index,0.)+r['allocated_wave_seconds']
        value={'timestamp':stamp(),'source':snapshot(__file__),'all_passed':True,'model':key,'variant':'native','pilot':False,
            'trajectories':len(records),'summary':summarize(records),'numerical_admission':admission,'original_pilot_replays':replays,
            'parameter_audit':{'original_native_parameters':True,'no_quantization':True,'independent_native_B1_B8_arithmetic':True},
            'wave_seconds':wave_times,'actual_generated_tokens':sum(len(r['generated_ids']) for r in records),
            'completed_generation_wave_seconds_by_wave':completed_waves,
            'completed_generation_wave_seconds':sum(completed_waves.values()),
            'elapsed_scope':'seconds/wave_seconds cover this process attempt only; completed_generation_wave_seconds includes every committed wave, including earlier attempts, but excludes failed work and load/commit time. Compute ledger retains failed attempts separately.',
            'peak_CUDA_allocated':torch.cuda.max_memory_allocated(),'seconds':time.monotonic()-start,
            'engine':snapshot(engine_path),'execution_revision':revision.relative_to(ROOT).as_posix(),
            'architecture_adapter':adapter_receipt,
            'weight_cache':{'loads':engine.weights.loads,'hits':engine.weights.hits,'maximum_one_layer_bytes':engine.weights.max_bytes,
                'direct_slice_reads':engine.weights.reader.reads,'total_direct_slice_bytes':engine.weights.reader.total_bytes,
                'maximum_direct_tensor_bytes':engine.weights.reader.maximum_tensor_bytes},
            'scope':'Original B1/B8 operations and separate KV, layer-major request scheduling. Six full-H fixtures and16complete native pilot packets explicitly replayed; caps do not establish unbounded composition.'}
        save(finish,value);ledger('phase2747_own_wave_'+key,value['seconds'])
        print('FORMATION_WAVE_DONE',key,len(records),value['seconds'],flush=True)
    except Exception as exc:failure(folder,start,exc);raise
    finally:
        if engine is not None:engine.close()
        del engine,model;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','qwen4','qwen14','glm4']);args=parser.parse_args()
    prepare() if args.mode=='prepare' else main(args.mode,args.mode=='qwen4')
