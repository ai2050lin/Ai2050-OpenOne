"""Independent reductions, numerical caveats and exact boundaries of confirmed patterns."""
import sys,itertools
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'tests/glm5'))
from phase2620_native_coordinate_contract import *
from phase2670_native_mlp_contract import OUT as CONTRACT,FIELD,LAYERS,SITES
from phase2673_native_mlp_confirmation import OUT as MAPS
from phase2674_native_mlp_scalar import OUT as FP
from phase2675_native_mlp_crossmodel import OUT as CROSS
from phase2676_native_mlp_delivery import OUT

def behavior(records):
    groups={};pairs={}
    for r in records:
        groups.setdefault(f'{r["language"]}/q{r["polarity"]}/m{r["mapping"]}',[]).append(r)
        keys=('family','language','unit','content_instance','form','target_index','mention_order','probe_index','polarity')
        pairs.setdefault(tuple(r[k] for k in keys),[]).append(r)
    return {'groups':{k:{'n':len(v),'content_correct':sum(r['content_correct'] for r in v),'strict_correct':sum(r['strict_correct'] for r in v),'eos':sum(r['eos'] for r in v)} for k,v in groups.items()},
        'paired_mappings':len(pairs),'both_correct':sum(len(v)==2 and all(r['content_correct'] for r in v) for v in pairs.values())}

def main():
    phases={p:read(next(RESULT.glob(f'phase{p}_*/analysis/final.json'))) for p in range(2670,2676)}
    checks={'2670through2675_complete':all(v['all_checks_passed'] for v in phases.values()),'material_frozen':sha(CONTRACT/'material/cases.json')==read(CONTRACT/'protocol/frozen.json')['material_sha256'],
        'standalone_CPU_raw_derivative_audit':read(OUT/'analysis/scalar_independent.json')['all_checks_passed']}
    scalar=read(FP/'analysis/records.json');sites=read(FP/'protocol/frozen.json')['sites'];maxerror=0.;comparisons=0
    for r in scalar:
        checks[f'case{r["case_index"]}_score_parts']=abs(r['base']['contrast']-sum(r['base'][p] for p in ('content','format','eos')))<1e-8
        for e in r['effects']:
            for p in ('all','content','format','eos'):
                calc=sum(r['gradients'][p][i]*d for i,d in zip(e['indices'],e['actual_deltas']));saved=e['predicted'] if p=='all' else e['parts'][p]['predicted'];maxerror=max(maxerror,abs(calc-saved));comparisons+=1
    # Independent tokenwise reduction of all30 frozen scalars on all16published branches.
    maxderiv=0.;derivn=0
    for r in scalar:
        if not r['published']:continue
        with np.load(FP/f'field/case_{r["case_index"]:04d}.npz') as z:
            for si,s in enumerate(sites):
                l,j,k,kind=s['layer'],s['j'],s['k'],s['kind']
                for part in ('all','content','format','eos'):
                    d=[]
                    for branch in ('Y','N'):
                        if kind=='down':x=z[f'{branch}__L{l}_J{s["unit"]}_a'];g=z[f'{branch}__L{l}_down_g_{part}'][:,j]
                        else:x=z[f'{branch}__L{l}_x'][:,k];g=z[f'{branch}__L{l}_J{j}_{kind}_g_{part}']
                        d.append(sum(float(a)*float(b) for a,b in zip(x,g)))
                    maxderiv=max(maxderiv,abs(d[0]-d[1]-r['gradients'][part][si]));derivn+=1
    checks.update(all_saved_effect_predictions_recomputed=maxerror<1e-10,all1920_published_parameter_derivatives=derivn==1920 and maxderiv<1e-8)
    with np.load(MAPS/'maps/confirmed_masks.npz') as z:survivors={k:np.argwhere(z[k]).tolist() for k in z.files}
    scope=read(MAPS/'analysis/candidate_scope.json');tables={}
    for metric,l,j in sorted({(r['metric'],r['layer'],r['coordinate']) for r in scope}):
        key=f'{metric}{l}[{j}]';tables[key]={}
        for fold,q,m in itertools.product(('initial','confirmation'),(0,1),(0,1)):
            rr=[r for r in scope if (r['metric'],r['layer'],r['coordinate'],r['fold'],r['q'],r['m'])==(metric,l,j,fold,q,m)]
            tables[key][f'{fold}/q{q}/m{m}']={'n':len(rr),'same_old_direction':sum(r['old_direction'] for r in rr),'target_dominates_form_order':sum(r['dominates_form_order'] for r in rr),'both':sum(r['old_direction'] and r['dominates_form_order'] for r in rr)}
    expansion=read(OUT/'expansion/analysis/completion.json');checks['4096_targeted_expansion_complete']=expansion['all_checks_passed']
    resolution=read(OUT/'numeric_resolution/analysis/completion.json');checks['1440_readout_resolution_conditions']=resolution['all_checks_passed']
    checks['zero_denominator_unit_tests']=read(OUT/'analysis/ratio_preflight.json')['all_checks_passed']
    resolution_records=read(OUT/'numeric_resolution/analysis/records.json')
    for scale,summary in resolution['summary'].items():
        effects=[e for r in resolution_records for e in r['effects'] if e['scale']==int(scale)]
        for part in ('all','content','format','eos'):
            ee=effects if part=='all' else [e['parts'][part] for e in effects]
            saved=summary if part=='all' else summary['parts'][part]
            den=sum(abs(e['effect64']) for e in ee);err=sum(abs(e['effect64']-e['predicted64']) for e in ee)
            checks[f'resolution_ratio_scale{scale}_{part}']=(saved['ratio_defined']==(den>0) and saved['relative_L1_error64']==(err/den if den else None) and saved['mean_abs_error64']==err/len(ee))
    prefix_audit=read(OUT/'analysis/source_prefix_audit.json');checks['7168_exact_source_prefix_audit']=prefix_audit['all_checks_passed']
    length_audit=read(OUT/'analysis/operation_length_audit.json');checks['external_operation_length_audit']=length_audit['all_checks_passed']
    prefix_replays={p:read(OUT/f'prefix_replay/{p}/analysis/completion.json') for p in ('bf16','fp32')}
    checks['640_native_prefix_replays']=all(r['all_checks_passed'] for r in prefix_replays.values())
    corrected={}
    for kind in ('single','joint','halfdose'):
        ee=[e for r in scalar for e in r['effects'] if e['kind']==kind];corrected[kind]={}
        for part in ('all','content','format','eos'):
            vv=[e if part=='all' else e['parts'][part] for e in ee];den=sum(abs(e['effect']) for e in vv);err=sum(abs(e['effect']-e['predicted']) for e in vv)
            corrected[kind][part]={'n':len(vv),'sum_abs_effect':den,'mean_abs_effect':den/len(vv),'mean_abs_prediction':sum(abs(e['predicted']) for e in vv)/len(vv),
                'mean_abs_error':err/len(vv),'relative_L1_error':err/den if den else None,'ratio_defined':den>0}
    summaries={'native_behavior':behavior(read(FIELD/'analysis/records.json')),'survivors':survivors,'frozen_candidate_scope':tables,'expanded_chronology':expansion,'numeric_resolution':resolution,
        'all_scalar_prediction_comparisons':comparisons,'maximum_saved_prediction_error':maxerror,'published_derivative_comparisons':derivn,'maximum_tokenwise_derivative_error':maxderiv,
        'scalar_numerics':phases[2674]['summary'],'scalar_numerics_corrected_ratios':corrected,
        'source_prefix_audit':{k:v for k,v in prefix_audit.items() if k!='changed_frozen_candidate_pairs'},'prefix_replays':prefix_replays,
        'operation_length_audit':{k:{n:v for n,v in report.items() if n!='different_pairs'} for k,report in length_audit['summary'].items()},
        'q14_frozen':read(CROSS/'qwen14/analysis/completion.json')['q14_frozen'],
        'crossmodel_behavior':{m:behavior(read(CROSS/m/'analysis/records.json')) for m in ('qwen14','glm4','ds7')},
        'claim_boundaries':[
            'Native4B oldgate onlyq0m0:64groups total, not allquestion/mapping contexts. Failed cells remain explicit; zeros in confirmedmask do not imply no distributed mechanism.',
            'NativeH checkpoint convention: H0 is actual embedding output, H(l+1) is blockl output. The last capturedH is before model finalRMSNorm, not the finalnormalized decoderstate. Allphysical layer/unit/coordinate IDs zero-based.',
            'Same-source prefix changes cannot be interpreted as backward semantic influence. Original BF16fullfield includes292unequal pairs/7168; appended replay checks exactrepeat, identicalshape maskedpadding and samecheckpointFP32. Selection ofdiagnostic exceptions is posthoc. Original fields/results stay intact; raw and equalshape comparisons have differentnumeric executionconditions.',
            'Lengthaudit:3776/4096factualtarget pairs have identicaltotalinputlength; the320mismatches are all syntax_role (128English,192Chinese). Other7families are fullylengthmatched for factualtarget pairs. All4096questionpolarity andall4096answer-mapping pairs are lengthmatched, versus2560/4096queryentity pairs. Equal total length removes one execution-shape confound only, not tokenposition/lexical differences.',
            'Finiteproduct identity includes rounding residual and is a known algebraic identity, not a newly discovered semantic mechanism or unique causal attribution.',
            'Orientation audit: matched contrast maps use v0-v1; finiteproduct expansion uses delta=v1-v0 relative to baselinev0. Their signedmeans are opposites by definition, not a modelcontradiction. RMS ignores thissign, clientlabelstatesorientation.',
            'Phase2672maps field-name clarification: __sum stores mean after division bygroupcount; __sumsq stores RMS aftersqrt. Those suffixes originally named accumulators, not finalstatistics. Rawperexamplepath fields and numericalresults unchanged; do not interpret these two keys as sums.',
            'Scalar and joint validation tests prediction of locally specified sequence scores; neither deletion necessity nor threeweightjoint sufficiency is the stopping criterion.',
            'Low scalarcontrol means a learnedweight near12.5percentile absolute magnitude ofits frozen row/column; ordinary meansmedian. This is not necessarily a low hiddenactivation, and does not establish importance of all low-valuedcoordinates. Full nativeactivation backgrounds are retained independently.',
            '8192 conditions =64family-language-entity settings crossed with only2content instances and task factors, not8192independent semantic examples.',
            'The historical form variable is not a guaranteed pure stylistic nuisance: e.g.syntax_role uses congratulation/thanking versus praise/help predicates acrossforms. TargetRMS>formRMS is a frozen descriptive comparison, not a universal semantic-neuron criterion; its failure can coexist with a reusable conditional component.',
            'Q14 old62H125MLP signs use a different criterion from4B amplitude/samedirectiongate. Crossmodel512 holds form/order/probe0 and lacks oldstylecross.',
            'Q14reconfirmation is also narrower than its ownpriorprotocol: currentp0only testsq/m4cells, whereas prior62H125MLP candidates hadp/q/m8cells and2instructionstyles. Positive retention here cannot be called full replication ofthatpriortruthrule; an entity-A/slot-conditioned explanation remains possible.',
            'Only4canonicalanswer tokens plusEOS; teacherforced candidate score is not longform autonomous content generation.16token cap especially limits DS7B reasoning protocol.',
            'Read mean absoluteEOS effect alongside no-operror and halfdose, not only relativeerrors. Unresolved microsensitivity is not evidence thatEOS was predicted accurately.',
            'Phase2674relativeEOSerror correction: actualfiniteEOS changes are allzero. Denominator was guarded with1e-30, producinghuge numeric values; these are NOT meaningful relativeerrors. Correctedratio is null/undefined, with absolute effect,prediction,error separatelyprovided. Originalrecords/MEMO retained; correction appendedhere, not silently rewritten.',
            'Hypothesis of new mathematics remains open. Limited-phase failure does not logically establish a fatal paradigm flaw; change measurement questions based on actual surviving/failed conditions.'],
        'runtime_incident':'First2671load aborted before anycase because current all-GPU auto loader did not expose hf_device_map. Recorded actual parameterdevices instead; allCUDA BF16, no library/checkpoint changes.',
        'crossmodel_runtime_recovery':read(CROSS/'qwen14/protocol/runtime_recovery.json'),
        'next_campaign_preparation_only':{name:read(OUT/f'analysis/{name}.json') for name in ('next_material_preflight','next_ledger_algebra_preflight','next_capture_cpu_preflight')},
        'preparation_boundary':'The8448newprompt tokenizer audit and tiny random CPU Qwen3/Qwen2/GLM instrumentation tests prepare future2677–2684 only. No pretrained research forward or newPhase completion is inferred from them.512samebody fourfunctioncells include256existingtruth-grid cells plus256newname/cloze prompts. Native pretrained no-op and final protocol freeze remain required.'}
    result={'checks':checks,'all_checks_passed':all(checks.values()),'summary':summaries};save(OUT/'analysis/scientific_checks.json',result);assert result['all_checks_passed'];print(json.dumps({'checks':len(checks),'survivors':survivors,'maxderivativeerror':maxderiv},ensure_ascii=True))

if __name__=='__main__':main()
