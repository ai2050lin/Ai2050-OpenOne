"""Audit decoded-prefix label availability and misleading energy-field names, without model inference."""
from rdc_operator_common import *
from phase2724_rdc_operator_material import lexical_features


def mask(text):
    f=lexical_features(text)
    return sum((1<<j)*int(f[k]) for j,k in enumerate(('cause','contrast','negation','reference')))


def main():
    from transformers import AutoTokenizer
    out=BASE/'identity_audit'
    if (out/'result.json').exists():return
    start=time.monotonic();tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    differences=[];anchors=[];tokens=0
    for i,r in enumerate(rows()):
        scope='confirmation' if r['split']=='confirmation' else 'main'
        with np.load(BASE/'capture'/scope/'energies'/f'{r["sample_id"]}.npz') as z:saved=z['prefix_cue_mask'].copy()
        for p in range(len(r['prompt_ids'])):
            decoded=tok.decode(r['prompt_ids'][:p+1],clean_up_tokenization_spaces=False);actual=mask(decoded)
            if actual!=int(saved[p]):
                record={'sample_id':r['sample_id'],'split':r['split'],'language':r['language'],'position':p,'is_anchor':p in r['anchors'],
                    'token_id':r['prompt_ids'][p],'saved_offset_mask':int(saved[p]),'known_ID_prefix_mask':actual,
                    'offset_prefix_tail':r['text'][:r['token_offsets'][p][1]][-100:],'known_ID_prefix_tail':decoded[-100:]}
                differences.append(record)
                if record['is_anchor']:anchors.append(record)
        tokens+=len(r['prompt_ids'])
        if (i+1)%256==0:print('PREFIX_LABEL_AUDIT',i+1,2048,'mismatches',len(differences),'seconds',round(time.monotonic()-start,1),flush=True)
    compressed(out/'all_cue_availability_mismatches.json.gz',differences)
    result={'timestamp':stamp(),'source':snapshot(Path(__file__)),'sources':2048,'all_tokens':tokens,'anchor_count':4096,
        'all_token_offset_vs_known_ID_prefix_mismatches':len(differences),'anchor_mismatches':anchors,
        'interpretation':'Offsets can span a complete Unicode character before every byte-piece is known. Thus offset-derived full-character cues are retrospective annotations unless this audit confirms equality at a prediction query. Actual compiler/generation labels always decode known IDs only.',
        'energy_field_correction':{'observation/result.json:event_onset_counts':'Counts argmax(E(native_block_output-native_block_input)), not first positive growth or onset. A large cancelling update can win this statistic.',
            'observation/events.json.gz:largest_increment_block':'Same squared-update-amplitude argmax, not signed energy increase. Original counts are unchanged; no onset inference permitted.'},
        'first_block_zero_index':'block6 is the7th block. H7 is its output residual boundary.',
        'scope':'Tokenizer/metadata audit only, no CUDA model loaded; all4096fit/test/confirmation anchor availability checked, not merely640compiled queries.'}
    save(out/'result.json',result);ledger('all_token_prefix_label_availability_audit',time.monotonic()-start,tokens=tokens);guard()
    print('PREFIX_LABEL_AUDIT_COMPLETE',len(differences),len(anchors),flush=True)


if __name__=='__main__':main()
