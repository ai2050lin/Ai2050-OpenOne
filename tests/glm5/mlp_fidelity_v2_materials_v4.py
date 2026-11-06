"""Natural generated attribute answers, independent location/industry and prequestion readout."""
import sys,itertools
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).parent))
import mlp_fidelity_v2_plan as P
from shared.dist_protocol.mechanism_trace import digest,file_sha

def main():
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(P.MODEL,local_files_only=True);docs=P.read(P.OUT/'fresh-documents.json')['controlled'];plan=P.read(P.OUT/'plan.json');rng=np.random.default_rng(P.SEED+404);levels={}
    for lo,hi in [(0,128),(128,320),(320,512)]:
        for lang in ['en','zh']:
            for frame in [0,1]:
                rows=[i for i in range(lo,hi) if docs[i]['language']==lang and (i//2)%2==frame];pairs=rng.permutation([(k,n) for _ in range(len(rows)//4) for k in [0,1] for n in [0,1]]);levels.update({i:tuple(map(int,z)) for i,z in zip(rows,pairs)})
    prefix=tok.encode('<|im_start|>user\n',add_special_tokens=False);end=tok.encode('\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',add_special_tokens=False);items=[]
    for b,doc in enumerate(docs):
        split='discovery' if b<128 else 'confirm' if b<320 else 'lockbox';frame=(b//2)%2;k,n=levels[b];coastkey='K' if k else 'M';inlandkey='M' if k else 'K';stem={'discovery':'delunor','confirm':'sorveta','lockbox':'kilavon'}[split]+str(b);names={coastkey:stem+('_amber' if n else '_cobalt'),inlandkey:stem+('_cobalt' if n else '_amber')}
        questions={}
        for q in [0,1]:
            if split=='discovery':text=('State the location category of the active settlement.' if q==0 else 'State the main livelihood of the active settlement.') if frame==0 else ('Is the active settlement coastal or inland?' if q==0 else 'Is the active settlement supported by fishing or mining?')
            elif split=='confirm':text=('Give the geographic category for the selected settlement.' if q==0 else 'Give the economic activity for the selected settlement.') if frame==0 else ('Report whether the active settlement is inland or coastal.' if q==0 else 'Report whether the active settlement relies on mining or fishing.')
            else:text=('Which location category describes the active settlement?' if q==0 else 'Which main livelihood describes the active settlement?') if frame==0 else ('For the selected settlement, choose its geographic description.' if q==0 else 'For the selected settlement, choose its economic description.')
            allowed='coastal or inland' if q==0 else 'fishing or mining';questions[q]=tok.encode(text+f'\nReturn exactly one word: {allowed}. No explanation.\nAnswer:',add_special_tokens=False)
        bases={}
        for r,t in itertools.product([0,1],[0,1]):
            industry={coastkey:'fishing' if r==0 else 'mining',inlandkey:'mining' if r==0 else 'fishing'}
            loc={coastkey:'It stands directly beside the ocean shore.' if frame==0 else 'It is a coastal settlement beside the sea.',inlandkey:'It stands deep in the interior, far from any sea.' if frame==0 else 'It is an inland settlement in the continental interior.'}
            live={key:(('Its residents earn a living catching fish.' if industry[key]=='fishing' else 'Its residents earn a living extracting ore.') if frame==0 else 'Its main livelihood is '+industry[key]+'.') for key in ['K','M']}
            text='\n\nUse only the following invented settlement records. The earlier article is background and does not describe these places.\n'+''.join(f'Record {key}: {names[key]}. {loc[key]} {live[key]}\n' for key in ['K','M'])+'\nActive record: '+(coastkey if t else inlandkey)+'.\n'
            bases[r,t]=tok.encode(text,add_special_tokens=False)
        assert len({len(v) for v in bases.values()})==1,'Recordwording introducedpositionconfound'
        maxq=max(map(len,questions.values()));base_length=len(next(iter(bases.values())));bg=doc['tokens'][:1024-len(prefix)-base_length-maxq];assert len(bg)>=800
        for r,t,q in itertools.product([0,1],[0,1],[0,1]):
            before=prefix+bg+bases[r,t];peer=prefix+bg+bases[r,1-t];diff=np.flatnonzero(np.asarray(before)!=peer);assert len(diff)==1;question=questions[q];ids=before+[198]*(maxq-len(question))+question;assert len(ids)==1024
            expected=('coastal' if t else 'inland') if q==0 else ('fishing' if (t^r) else 'mining');contrast=('inland' if t else 'coastal') if q==0 else ('mining' if (t^r) else 'fishing')
            items.append({'id':f'attribute:{b}:{r}:{t}:{q}','design_version':4,'background_index':b,'background_source':{k:v for k,v in doc.items() if k!='tokens'},'split':split,'frame':frame,'background_language':doc['language'],'code':'natural_attribute','mapping':r,'industry_assignment':r,'fact':t,'selected_location':t,'selected_industry':t^r,'question':q,'queried_attribute':'location' if q==0 else 'industry','decision':int(expected in ['coastal','fishing']),'expected':expected,'contrast':contrast,'positive_key':coastkey,'selected_key':coastkey if t else inlandkey,'token_ids':ids+end,'prompt':tok.decode(ids+end),'input_tokens':len(ids+end),'readout_positions':[int(diff[0])-1,int(diff[0]),len(before)-1,1023+len(end)],'readout_roles':['selected_record_end','assistant_boundary'],'background_tokens':len(bg),'scope':'document-defined settlementlocation andlivelihood; natural8tokenanswer, notcodedtruthconversion'})
    c={'schema':'mlp_fidelity_controlled_inputs.v4','kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'codes':{'natural_attribute':['coastal','inland','fishing','mining']},'items':items,'source_sha256':file_sha(__file__)};assert len(items)==4096;P.save(P.OUT/'controlled-v4-corpus.json',c)
    prereg={'schema':'mlp_fidelity_controlled_prereg.v4','kind':'real','run_id':P.RUN,'model_snapshot':plan['model_snapshot'],'provenance':plan['provenance'],'revision':'v2/v3 codedtruth pilots failed8/32 and5/32cells; preservefailures andreplace task withnaturally generatedattributevalues. Newdesign, oldcodedhypotheses notconfirmed. No confirm/lockboxoutputs read.',
        'corpus_sha256':digest(c),'n_inputs':4096,'backgrounds':512,'splits':{'discovery':128,'confirm':192,'lockbox':192},'factors':['selected_location T','industry pairing R, selectedindustry=T xorR','queried attribute Q'],'pilot':'first32discovery backgrounds×8=256; eachT/R/Q/frame celln16, >=15generatedfirstwordcorrect tocontinue; strictonewordformatseparatelyreported',
        'behavior':'both confirm/lockbox allT/R/Q/frame cellsn96 require >=.90 correct generatedfirstsemanticword; score from8unrestrictedgreedytokens, firstalphabeticword lowercase. No fixedcandidatepreferencegate. Strictformatandfirsttokenmarginarediagnostics.',
        'prefix_controls':'Qquestionpadding onlyafter selected-recordreadout: stage0mustbitmatchacrossQ, hs[0] common readouttokens; Tpairedprefixhashbefore selector equal. Qhasno causalvisibility atstage0.',
        'feature_nomination':'atmost16/layer ondiscoveryonly:6locationT answeringlocationQ0,6industry(T xorR) answeringindustryQ1,4queryQ; fit sign atassistant andtestscope separately. Samefeatures atprequestionare independentreadoutdiagnostic. No unique/monosemanticlabelclaim.',
        'semantic_criteria':{'layers':[8,24,35],'alpha':.05/(3*16*3*2),'min_order_accuracy':.8,'min_nonzero_fraction':.8,'confirm_and_lockbox':True,'components':'TandindustryeachbothRorientations; QbothT/Rlevels; allrequired ineachheldout, samefittedsign','labels':'operationalconditionalattributeanswerfeatures; answerwordconfoundremainsunlessprequestionandnaturalcontextpredictionsverify'},
        'causal':{'layer':35,'selection':'all16discoveryfeature nominees, no selectionbyheldout causalorsemanticresults; random16 matchedwithin[.8,1.25] ondiscoveryactivationRMS anddecoder norm, otherwise specificitycomparisonnoteligible','dose':[0,.5,1],'n_lockbox_backgrounds':192,'samples':'oneT/Rcombination deterministicbybackgroundhash, bothQ:384inputcases; clusteredbybackground','generated_endpoint':'generatedfirstwordaccuracyvs clean, dose,matchedrandom; firsttokenCE/KL separate','capability':'sameoperationatlastassistantquery,2200gateeligible +140reference,2ppNI; no weight edits'},
        'tail_path':'B35 lastMLP output writes directly before finalRMSNorm/unembedding. OriginalprefixKVindependentofB35MLPoutput: reuseprefillcache,croptoprefixbeforeeachgenerationbranch. Firsttoken tailrecompositionmustmatchnative finalh/logits; do notgeneralize thisshortcuttoearlierlayers.',
        'natural_fidelity':'fresh512documents8positions, fullSAEsplice/nativek/ablate atB35; true-next-tokenCE andclean-distributionKL,512docclusteredCIs; saveactualpreMLPresidual andfinalh. No surrogateinputpredictionclaim.',
        'source_sha256':file_sha(__file__)};P.save(P.PUBLIC/'controlled-v4-prereg.research.json',prereg);print({'inputs':4096,'pilot':256,'minimum_background':min(r['background_tokens'] for r in items),'readout0':sorted({r['readout_positions'][2] for r in items})},flush=True)
if __name__=='__main__':main()
