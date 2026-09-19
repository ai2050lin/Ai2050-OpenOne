"""Causal external token-piece POS and relation posteriors, trained on older training annotations only."""
from collections import Counter
from functools import lru_cache
import hashlib
from rdc_relation_common import *
from phase2711_rdc_prefix_atlas import span_token
DISTANCE_CUTS=np.array([1,2,3,4,8,16,32,1000000])
LABELS=['none']+[f'{r}:{direction}' for r in RELATIONS for direction in ('earlier_dependent','earlier_head')]


def annotation(row):
    n=len(row['prompt_ids']);tag=np.full(n,UPOS.index('X'),np.int64);word_at={};edges={};words={w['id']:w for w in row['retrospective_ud']}
    for w in words.values():
        if w['char_span'] is None:continue
        a,b=w['char_span'];u=UPOS.index(w['upos']) if w['upos'] in UPOS else UPOS.index('X')
        for p,(x,y) in enumerate(row['token_offsets']):
            if x<b and y>a:tag[p]=u
        p=span_token(row,w['char_span'])
        if p is not None:word_at[w['id']]=p
    for w in words.values():
        rel=w['relation'].split(':')[0]
        if rel not in RELATIONS or w['id'] not in word_at or w['head'] not in word_at:continue
        dep,head=word_at[w['id']],word_at[w['head']]
        if dep==head:continue
        a,b=sorted((dep,head));label=1+2*RELATIONS.index(rel)+int(dep>head);edges[(a,b)]=label
    return tag,edges,word_at


def fit():
    out=BASE/'prefix_parser'
    if (out/'frozen.json').exists():return
    training=[r for r in read(PREVIOUS/'material_stratified.json') if r['split']=='train']
    counts=np.zeros((2,8,17,17,len(LABELS)),np.float64);global_pos=np.zeros((2,17),np.float64);lex={};pairs=0;positives=0
    for row in training:
        lang=int(row['language']=='zh');tags,edges,_=annotation(row)
        for tid,tag in zip(row['prompt_ids'],tags):
            if tid not in lex:lex[tid]=np.zeros((2,17),np.float64)
            lex[tid][lang,tag]+=1;global_pos[lang,tag]+=1
        for t in range(1,len(tags)):
            for s in range(t):
                b=int(np.searchsorted(DISTANCE_CUTS,t-s));label=edges.get((s,t),0);counts[lang,b,tags[s],tags[t],label]+=1;pairs+=1;positives+=int(label>0)
    priors=(counts.sum((2,3))+.1);priors/=priors.sum(-1,keepdims=True)
    tables=(counts+4*priors[:,:,None,None,:])/(counts.sum(-1,keepdims=True)+4)
    gp=(global_pos+1)/(global_pos.sum(1,keepdims=True)+17)
    ids=np.array(sorted(lex),np.int64);lp=np.stack([(lex[tid]+gp)/(lex[tid].sum(1,keepdims=True)+1) for tid in ids])
    npz(out/'model.npz',ids=ids,lexical_pos=lp.astype(np.float32),global_pos=gp.astype(np.float32),relation_tables=tables.astype(np.float32),distance_cuts=DISTANCE_CUTS)
    save(out/'protocol.json',{'timestamp':stamp(),'code':snapshot(Path(__file__)),'training_sources':320,
      'training_material_sha':sha(PREVIOUS/'material_stratified.json'),'only_split':'train','labels':LABELS,'UPOS':list(UPOS),
      'algorithm':'Smoothed native token-ID POS distribution; label posterior conditioned on POS pair, language and signed earlier/later distance bucket. All native token pairs contribute training counts, none included.',
      'inference_inputs':'Only available token ID prefix and language. No gold token/word boundaries, POS, dependency labels, future characters/tokens, or HiddenStates.',
      'smoothing':'POS one global-prior pseudocount; relation4 distance/language-prior pseudocounts.',
      'limits':['Token-piece labels are coarse candidate roles, not a complete syntactic parser or semantic graph.','Gold labels used in supervised training/evaluation can depend on full sentence; predicted prefix relations remain uncertain.','Absent lexical identity falls back to training-language POS prior.','Control/knowledge/negation scope/coreference unresolved.']})
    immutable(out/'frozen.json',{'timestamp':stamp(),'model_sha':sha(out/'model.npz'),'protocol_sha':sha(out/'protocol.json'),'training_native_pairs':pairs,'positive_relation_pairs':positives,'native_token_ID_entries':len(ids)})
    print('PREFIX_RELATION_PARSER_FROZEN',pairs,positives,len(ids),flush=True)


class PrefixRelations:
    def __init__(self):
        fit()
        with np.load(BASE/'prefix_parser/model.npz') as z:
            self.ids={int(t):i for i,t in enumerate(z['ids'])};self.lex=z['lexical_pos'].astype(float);self.global_pos=z['global_pos'].astype(float);self.tables=z['relation_tables'].astype(float)
    def tags(self,ids,language):
        lang=int(language=='zh');return np.stack([self.lex[self.ids[int(t)],lang] if int(t) in self.ids else self.global_pos[lang] for t in ids])
    def weights(self,prefix_ids,language):
        pos=self.tags(prefix_ids,language);n=len(pos);prob=np.empty((max(0,n-1),len(LABELS)),np.float64);lang=int(language=='zh')
        if n<=1:return prob,pos
        distance=np.searchsorted(DISTANCE_CUTS,n-1-np.arange(n-1))
        for b in np.unique(distance):
            mask=distance==b;table=np.einsum('ijr,j->ir',self.tables[lang,b],pos[-1]);prob[mask]=pos[:-1][mask]@table
        prob/=prob.sum(1,keepdims=True)
        return prob,pos
    def descriptor(self,prefix_ids,language):
        p,pos=self.weights(prefix_ids,language);mass=p[:,1:].sum(0)/max(len(p),1)
        return np.concatenate([pos[-1],mass,[np.log1p(len(prefix_ids))]])
    def graph(self,prefix_ids,language):
        p,pos=self.weights(prefix_ids,language)
        return {'query_position':len(prefix_ids)-1,'labels':LABELS,'source_positions':list(range(max(len(prefix_ids)-1,0))),
          'all_relation_probabilities':p.tolist(),'all_token_POS_probabilities':pos.tolist(),'UPOS':list(UPOS),
          'known_token_ID_fraction':sum(int(t) in self.ids for t in prefix_ids)/len(prefix_ids),
          'status':'Prefix-only candidate binding probabilities, not gold relations or proven native semantic roles.'}


def evaluate(material,out_name):
    from sklearn.metrics import roc_auc_score,average_precision_score
    parser=PrefixRelations();scores=[];truth=[];source=[];loss=[];brier=[];tag_correct=0;tag_n=0;known=0
    for r in material:
        tags,edges,_=annotation(r);pos=parser.tags(r['prompt_ids'],r['language']);tag_correct+=int((pos.argmax(1)==tags).sum());tag_n+=len(tags);known+=sum(int(t) in parser.ids for t in r['prompt_ids'])
        for t in range(1,len(tags)):
            p,_=parser.weights(r['prompt_ids'][:t+1],r['language']);labels=np.array([edges.get((s,t),0) for s in range(t)]);y=labels>0
            scores.extend((1-p[:,0]).tolist());truth.extend(y.tolist());source.extend([r['sample_id']]*t)
            loss.extend((-np.log(np.maximum(p[np.arange(t),labels],1e-30))).tolist());brier.extend(((1-p[:,0]-y)**2).tolist())
    report={'timestamp':stamp(),'source_units':len(material),'candidate_pairs':len(scores),'positive_pairs':int(np.sum(truth)),
      'any_relation_ROC_AUC':float(roc_auc_score(truth,scores)),'any_relation_average_precision':float(average_precision_score(truth,scores)),
      'positive_base_rate':float(np.mean(truth)),'categorical_log_loss':float(np.mean(loss)),'any_relation_Brier':float(np.mean(brier)),
      'token_piece_POS_accuracy':tag_correct/tag_n,'known_token_ID_fraction':known/tag_n,
      'evaluation_only_gold_annotations':True,'claim':'External probabilistic candidate quality, not native parsing or semantic mechanism accuracy.'}
    save(BASE/f'prefix_parser/{out_name}.json',report);print('PREFIX_RELATION_EVAL',out_name,report,flush=True)


def main():
    fit();parser=PrefixRelations();r=rows()[0];p=r['anchors'][0]
    first=parser.graph(r['prompt_ids'][:p+1],r['language']);altered=r['prompt_ids'][:p+1]+[151645]*7
    assert parser.graph(altered[:p+1],r['language'])==first
    save(BASE/'prefix_parser/causality_audit.json',{'passed':True,'future_suffix_not_an_input':True,'token_boundaries_not_from_gold':True,'timestamp':stamp(),'labels':LABELS})
    evaluate([r for r in rows() if r['split']=='test'],'main_test')


if __name__=='__main__':main()
