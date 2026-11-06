"""A new frozen development/fresh-evaluation round; prior results are development data."""
import sys,json,time,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).parent))
from mlp_atlas_code_materials_v2 import read,save,MODEL
from shared.dist_protocol.mechanism_trace import digest,file_sha
from deploy.research_library import Library
OUT=ROOT/'tests/glm5/result/mlp_fidelity_v2';PUBLIC=ROOT/'research/mlp_fidelity/v2';RUN='MLP-FIDELITY-20261104-v2';SEED=411062026
OLD=ROOT/'tests/glm5/result/mlp_atlas_dictionary_v1';LONG=ROOT/'tests/glm5/result/native_mlp_long_v1'

def status(stage,state='done',**kw):
    p=PUBLIC/'00-progress.research.json';d=read(p) if p.exists() else {'schema':'research_workpackage.v1','kind':'real','run_id':RUN,'title':'提高MLP字典保真度：新材料、语义和因果闭环','model_snapshot':read(OUT/'plan.json')['model_snapshot'],'stages':{}}
    d['stages'][stage]={'status':state,'time':time.time(),**kw};save(p,d,False);Library(ROOT/'distributed_data/research_library').ingest(d,str(p.relative_to(ROOT)),kind='local_report');print(json.dumps({'stage':stage,'status':state,**kw}),flush=True)

def main():
    OUT.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True);man=read(OLD/'bank/manifest.json');docs=read(LONG/'documents.json')['docs']
    prov={**man['provenance'],'seed':SEED,'git_sha':subprocess.check_output(['git','-c',f'safe.directory={ROOT.as_posix()}','rev-parse','HEAD'],text=True).strip(),'git_dirty':True}
    plan={'schema':'mlp_fidelity_plan.v2','kind':'real','run_id':RUN,'model_snapshot':man['model_snapshot'],'provenance':prov,
      'old_split_correction':'v1 dictionary uses slices0:1024 FineWeb en/zh train,1024:1536 Wikipedia-en confirm,1536:2048 Wikipedia-zh lockbox. Those labels are not language-balanced independent repeats. Old result numbers unchanged; v2 development may use all exposed old documents.',
      'development':'All old2048 documents are exposed development. Within each of4 sources, hash-split384 train/128 validation; no use of fresh evaluation or new controlled outputs in model selection.',
      'prototype_layers':[0,8,16,24,35],
      'candidates':[{'id':'tc1024','kind':'transcoder','m':1024,'k':32},{'id':'tc4096','kind':'transcoder','m':4096,'k':32},{'id':'sae4096','kind':'sae_output','m':4096,'k':32}],
      'training':{'steps':5000,'batch':256,'checkpoints':[1000,2500,5000],'lr':.001,'lr_end':.0001,'precision':'FP32parameters/optimizer; BF16autocastforward onCUDA; no original model quantization','normalization':'fixed discovery variance in physical output units; sampleuniform acrossdevelopment trainingpositions','seed':SEED,'max_total_dictionary_GiB':5},
      'selection':'Choose global candidate and checkpoint minimizing mean validation FVU across five prototype layers. Compare against tc1024 at1000steps undernewtraining, plus archivedv1model. No fresh-evaluation based checkpointselection. Train same chosen recipe onall36 layers. SAE sees observed MLP output: a representation/compression model, not an input-to-output predictor.',
      'fresh_evaluation':{'documents':512,'splits':{'confirm':{'en':128,'zh':128},'lockbox':{'en':128,'zh':128}},'positions_per_doc':8,'tokens_per_doc':1024,'materials':'new Wikipedia rowgroups>=7, source IDs and1024-tokenprefixes excluded from all old consumed docs; unknown pretraining exposure','fidelity_gate':'both new splits perselected causalstudy layer FVU<=.2; meanimprovement vsarchiveddictionary positive paired-document95%CI, familyalpha .05/(36*2). Alllayerfailures retained. No feature-win claim without actual lossrecovery.'},
      'controlled_materials':{'backgrounds':512,'discovery':128,'confirm':192,'lockbox':192,'variants_per_background':16,'code_pairs':[['A','B'],['C','D']],'factors':['mapping','selected_coastal_or_inland_record','asked_coastal_or_inland'],'frames':2,'readouts':['question_end','assistant_boundary'],'tokens':1034,'scope':'invented settlement referent definitions; no item/object wording; statement coasts vs inland opposite predicates within same entity category','pilot':'first32discoverybackgrounds×16; eachframe/code/mapping/truth/question cell16, >=15correct; no heldout scoring before pilotpasses','heldout_behavior':'eachsplit/code/mapping/truth/question/frame celln96, accuracy>=.90; codechoice andunrestrictedtop1 reported separately','native_capture_layers':[8,24,35],'capture':'all37 hidden atbothreadouts; all9728GVA andMLPinput/output/headZ inselectedthreeblocks; uncaptured layers/modules explicit; noattentionrows'},
      'semantic_prereg':'After behavior pilot anddevelopmentdictionaryselection, nominate atmost16 dictionaryfeatures perstudy layer using only controlleddiscovery. Identify operationalcoast/inland orabstractmatch/outputcode/format effect; fit sign only discovery. Require confirm+lockbox response, shuffled-label andembedding baselines; code-specificfeatures not semantic-invariant bydefault. AI labels remain untested.',
      'causal_prereg':{'eligible':'only study layer satisfying fresh-fidelity gate andallheldoutbehaviorcells; then frozen discoveryfeaturegroup, not selected byheldouteffects','dose':[0,.5,1],'primary':'192lockbox backgrounds, selectedcode/mappingbalanced fixedsubset; targetvsrandom matchedfeaturegroup; effect20%loss ofpairedanswer discriminability ormonotone logitgapshift with pairedCI; scoped sufficient/necessary claims only. Fullreplacementloss/KL plusnative32 controls reported separately.','capability_n':2200,'capability_pool':'previously exposed keyed monitor, samefrozeninputhash; reference140 excluded','NI_margin':.02,'NI_method':'simultaneouspairedClopperPearsonloss/gainbound; cannot infer preservation fromp>.05'},
      'resource':'oneCUDAjob atatime; originalQwen3-4B BF16; allnewdisk<=14GiB andfree>=4GiB; captureonce/manyanalyses; no deletion of old evidence','source_sha256':file_sha(__file__)}
    save(OUT/'plan.json',plan);save(PUBLIC/'prereg.research.json',plan);status('preregistration','done',prototype_layers=plan['prototype_layers'],fresh_documents=512)
if __name__=='__main__':main()
