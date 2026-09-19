"""Outcome-blind larger natural confirmation for an exceptionally rare internal transition."""
from collections import Counter,defaultdict
import hashlib
import urllib.request
from rdc_joint_common import *
from phase2719_rdc_joint_material import parse,candidates,take,old_identities,annotate_window

OUT=BASE/'extension/tail_confirmation'


def take_available(pool,number,excluded):
    """Same deterministic round/genre selection, with a pre-outcome at-most quota."""
    selected=[];genres=defaultdict(list)
    from phase2719_rdc_joint_material import rank
    for group in sorted(pool,key=rank):genres[pool[group][0]['genre']].append(group)
    for _ in range(8):
      for gi in range(max(map(len,genres.values()),default=0)):
        for genre in sorted(genres):
          if gi>=len(genres[genre]):continue
          for row in pool[genres[genre][gi]]:
            if any((row['language'],sid) in excluded[3] for sid in row['component_ids']):continue
            if any(set(row[key])&excluded[i] for i,key in enumerate(('normalized_texts','numeric_families','construction_families'))):continue
            selected.append(row)
            for i,key in enumerate(('normalized_texts','numeric_families','construction_families')):excluded[i].update(row[key])
            excluded[3].update((row['language'],sid) for sid in row['component_ids'])
            break
          if len(selected)==number:return selected
    return selected


def main():
    if (OUT/'material.json.gz').exists():return
    guard(12*1024**2)
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    excluded=old_identities();old=old_rows()
    for name in ('material_stratified.json','confirmation_material.json','full_source_history/fresh_material.json'):old.extend(read(PREFIX/name))
    old+=rows()+rows(True)
    documents=set();pud_ids=set()
    for r in old:
        excluded[0].add(r.get('normalized_texts',[None])[0] or __import__('rdc_relation_common').normalized(r['text']))
        excluded[0].update(r.get('normalized_texts',[]));excluded[1].update(r.get('numeric_families',[]));excluded[2].update(r.get('construction_families',[]))
        for component in r.get('component_ids',[r['source_sentence_id']]):
            excluded[3].add((r['language'],component))
            if r['language']=='en':documents.add(component.rsplit('-',1)[0])
            if r.get('treebank')=='pud' or r.get('source_key','').startswith('pud'):pud_ids.add(component)
    corpus={};manifest=[]
    for key,path,lang,treebank in [('ewt',PREFIX/'sources/en_train.conllu','en','ewt'),('gsd',PREFIX/'sources/zh_train.conllu','zh','gsd')]:
        text=path.read_text(encoding='utf-8');corpus[key]=parse(text,lang,treebank,'train')
        manifest.append({'source':key,'path':str(path.relative_to(ROOT)),'sha256':sha(path),'version':'UD r2.15','reused_without_copy':True})
    path=BASE/'sources/pud_test.conllu.gz';corpus['pud']=parse(gzip.decompress(path.read_bytes()).decode('utf-8'),'zh','pud','test')
    manifest.append({'source':'pud','path':str(path.relative_to(ROOT)),'sha256':sha(path),'version':'UD r2.15','reused_without_copy':True})
    for key,lang,treebank,part,url in [
        ('ewt_dev','en','ewt','dev','https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/r2.15/en_ewt-ud-dev.conllu'),
        ('enpud','en','pud','test','https://raw.githubusercontent.com/UniversalDependencies/UD_English-PUD/r2.15/en_pud-ud-test.conllu')]:
        dest=OUT/'sources'/f'{key}.conllu.gz'
        if dest.exists():raw=gzip.decompress(dest.read_bytes())
        else:
            raw=urllib.request.urlopen(url,timeout=45).read();assert len(raw)<16*1024**2
            dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(gzip.compress(raw,compresslevel=6,mtime=0))
        corpus[key]=parse(raw.decode('utf-8'),lang,treebank,part)
        manifest.append({'source':key,'path':str(dest.relative_to(ROOT)),'sha256':sha(dest),'raw_sha256':hashlib.sha256(raw).hexdigest(),'url':url,'version':'UD r2.15',
                         'license':'CC BY-SA 3.0' if treebank=='pud' else 'CC BY-SA 4.0; underlying text copyright retained'})
    for r in corpus['ewt']+corpus['ewt_dev']:
        doc=r['source_sentence_id'].rsplit('-',1)[0]
        r.update(source_group='en/'+doc,source_has_document_id=True,document_id=doc,genre=doc.split('-',1)[0])
    corpus['ewt']=[r for r in corpus['ewt'] if r['document_id'] not in documents]
    corpus['ewt_dev']=[r for r in corpus['ewt_dev'] if r['document_id'] not in documents]
    # Additional current-run whole-text identity controls, including original source component patterns.
    from rdc_relation_common import normalized,canonical,construction
    for r in rows()+rows(True):
        excluded[0].add(normalized(r['text']));excluded[1].add(canonical(r['text']));excluded[2].add(construction(r))
    rejected=Counter();chosen=[];eligibility={}
    immutable(OUT/'material_revision.json',{'timestamp':stamp(),'prior_attempt':'Requested1536 EWT train sentences with entirely unused document IDs; available48 sentences in20 groups; failed before any model output or material freeze.',
        'revision_before_new_outputs':'Up to512 each unused EWTdev, EnglishPUD, ChinesePUD, ChineseGSD; at least1024 total or stop. Eight-round group cap inherited. Cross-language PUD parallel IDs excluded against old and newly selected materials. No target outcomes inspected.',
        'original_plan_total':3072,'revised_maximum':2048,'source':snapshot(Path(__file__))})
    for key,n in [('ewt_dev',512),('enpud',512),('pud',512),('gsd',512)]:
        if key in ('enpud','pud'):corpus[key]=[r for r in corpus[key] if r['source_sentence_id'] not in pud_ids]
        labels={r['source_group']:'tail_confirmation' for r in corpus[key]}
        pool=candidates(tok,corpus[key],labels,excluded,rejected)['tail_confirmation']
        eligibility[key]={'groups':len(pool),'candidate_sentences':sum(map(len,pool.values())),'requested':n}
        # Selection cap/genre rotation are inherited deterministic source-based rules; no energy checks.
        selected=take_available(pool,n,excluded);eligibility[key]['selected']=len(selected)
        if key in ('enpud','pud'):pud_ids.update(r['source_sentence_id'] for r in selected)
        for i,r in enumerate(selected):
            length=len(r['prompt_ids']);anchors=[length//3,2*length//3]
            r.update(sample_id=f'tail-{key}-{i:04d}',anchors=anchors,positions=[0,1,anchors[0],anchors[0]+1,anchors[1],anchors[1]+1],
                source_key=key,material_kind='natural_sentence',online_scope='Only prefix IDs and current lower-layer H12 are event forecast inputs; UD annotations are retrospective.')
            annotate_window(r,{})
            # Check exact tokenizer boundaries and no future available feature.
            assert len(r['token_offsets'])==length and len(set(r['positions']))==6
            chosen.append(r)
    assert 1024<=len(chosen)<=2048
    compressed_json(OUT/'material.json.gz',chosen)
    freeze_files=['extension/event_forecast.npz','extension/event_forecast.json','extension/event_threshold.json']
    frozen={name:sha(BASE/name) for name in freeze_files}
    save(OUT/'protocol.json',{'timestamp':stamp(),'source':snapshot(Path(__file__)),'material_sha':sha(OUT/'material.json.gz'),'frozen_files':frozen,
        'sources':len(chosen),'languages':dict(Counter(r['language'] for r in chosen)),'tokens':sum(len(r['prompt_ids']) for r in chosen),
        'group_counts':{key:len({r['source_group'] for r in chosen if r['source_key']==key}) for key in eligibility},'eligibility':eligibility,'rejected':dict(rejected),'source_manifest':manifest,
        'epistemic_status':'New actual model outputs not observed at material/threshold/classifier freeze. EWT document IDs disjoint from prior registered studies; Chinese sentence/component identities disjoint, original article independence not guaranteed. Not excluded from model pretraining.',
        'capture':'Q4 natural unpadded batch1 BF16 eager, all-token H12/H23/H36 full native coordinates. First2per source-key full fields retained, every source full-array identities+alltoken energies/probabilities+fullcoordinate stratum moments. Every numerical event H12/H23/H36 row retained, no event deletion.',
        'replay':'Retained raw fixtures and event rows are queryable; other all-token raw buffers released only after full analysis and SHA registration; exact material/model/code allow replay. No prior archive deleted.',
        'evaluation':'Apply original TRAIN-only full2560-coordinate logistic and token-ID/position/constant controls, original threshold unchanged. Full-coordinate moments by language, source and numeric event; primary denominator includes all noninitial tokens.',
        'resources':'At most2048 sources, maximum45MiB extra artifacts, per-source transient allfield memory below32MiB, existing global budget enforced. Independent confirmation addresses limited event count, not arbitrary sample inflation.',
        'parallel_identity_check':'No EN/ZH PUD sentence ID reused across prior materials or new source strata; PUD source article IDs still unknown.'})
    print('TAIL_MATERIAL_FROZEN',read(OUT/'protocol.json'),flush=True)


if __name__=='__main__':main()
