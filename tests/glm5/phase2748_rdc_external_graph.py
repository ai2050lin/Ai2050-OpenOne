"""Typed authentic context/question/annotation graph, independent of model outcomes."""
import re
from collections import Counter, defaultdict
from rdc_construction_common import *
from rdc_question_material import exact_answer_signature, text_id

OUT = BASE/'phase2748'


def head(question):
    words = re.findall(r"\w+", question.casefold())
    if not words:
        return 'empty_lexical_head'
    if words[:2] in [['how', 'many'], ['how', 'much'], ['how', 'long'], ['how', 'old']]:
        return '_'.join(words[:2])
    if words[0] in {'who', 'what', 'when', 'where', 'which', 'why', 'how', 'whose'}:
        return words[0]
    return 'other_initial_word'


def all_occurrences(text, span):
    cursor = 0
    while span:
        index = text.find(span, cursor)
        if index < 0:
            break
        yield [index, index+len(span)]
        cursor = index+1


def main():
    path = OUT/'external_graph/manifest.json'
    if path.exists():
        return read(path)
    start = time.monotonic()
    manifest = read(OUT/'material/manifest.json')
    rows = gzread(ROOT/manifest['rows']['path'])
    groups = gzread(ROOT/manifest['context_groups']['path'])
    by_group = defaultdict(list)
    nodes, edges, contrasts, labels = [], [], [], []
    counts = Counter()
    for row in rows:
        by_group[row['group_id']].append(row)
        qid = row['question_id']
        lexical_head = head(row['question'])
        annotation_type = row['answer_annotations'][0]['type']
        label = {'question_id': qid, 'group_id': row['group_id'], 'split': row['split'], 'cohort': row['cohort'],
            'question_lexical_head': lexical_head, 'primary_source_answer_type': annotation_type,
            'primary_required_span_count': len(row['answer_annotations'][0]['required_spans']),
            'accepted_complete_annotations': len(row['answer_annotations']),
            'label_scope': 'Source/schema and literal wording metadata, not inferred native mechanisms or proven reasoning requirements.'}
        labels.append(label)
        counts[(row['cohort'], row['split'], lexical_head, annotation_type)] += 1
        nodes.append({'id': qid, 'kind': 'original_human_question', 'text': row['question'], **label})
        edges.append({'source': row['group_id'], 'target': qid, 'type': 'human_question_about_unchanged_passage'})
        for ai, annotation in enumerate(row['answer_annotations']):
            aid = qid+'/annotation/'+str(ai)
            nodes.append({'id': aid, 'kind': 'complete_source_answer_annotation', **annotation})
            edges.append({'source': qid, 'target': aid, 'type': 'accepted_complete_alternative', 'source_order': ai})
            for si, span in enumerate(annotation['required_spans']):
                sid = aid+'/required_span/'+str(si)
                locations = list(all_occurrences(row['passage'], span)) if annotation['type'] in {'span', 'multiple_required_spans'} else []
                node = {'id': sid, 'kind': 'jointly_required_answer_span', 'text': span,
                        'literal_exact_passage_occurrences': locations,
                        'occurrence_scope': 'Literal text matches are not asserted gold evidence links or unique reasoning chains.'}
                if row['cohort'] == 'quoref':
                    source = row['answer_source_spans'][si]
                    assert row['passage'][source['start']:source['end_exclusive']] == span
                    node['publisher_checked_span'] = [source['start'], source['end_exclusive']]
                nodes.append(node)
                edges.append({'source': aid, 'target': sid, 'type': 'jointly_required_component', 'source_order': si})
    label_by_id = {r['question_id']: r for r in labels}
    for group in groups:
        rr = by_group[group['group_id']]
        nodes.append({'id': group['group_id'], 'kind': 'original_unchanged_passage',
            'text': rr[0]['passage'], 'context_text_sha256': rr[0]['context_text_sha256'],
            'cohort': group['cohort'], 'split': group['split'], 'article_id': group['article_id'],
            'article_identity_scope': group['article_identity_scope'], 'near_context_component': group['near_context_component']})
        for i in range(4):
            for j in range(i):
                a, b = rr[j], rr[i]
                left, right = label_by_id[a['question_id']], label_by_id[b['question_id']]
                sa = {exact_answer_signature(v) for v in a['answer_annotations']}
                sb = {exact_answer_signature(v) for v in b['answer_annotations']}
                assert not sa & sb
                wa = set(re.findall(r'\w+', a['question'].casefold()))
                wb = set(re.findall(r'\w+', b['question'].casefold()))
                contrasts.append({'id': group['group_id']+'/pair/'+str(j)+'_'+str(i),
                    'group_id': group['group_id'], 'question_ids': [a['question_id'], b['question_id']],
                    'cohort': group['cohort'], 'split': group['split'],
                    'same_literal_question_head': left['question_lexical_head'] == right['question_lexical_head'],
                    'same_primary_answer_type': left['primary_source_answer_type'] == right['primary_source_answer_type'],
                    'question_word_set_Jaccard': len(wa & wb)/max(1, len(wa | wb)),
                    'unchanged_passage': True, 'complete_answer_annotation_signatures_disjoint': True,
                    'scope': 'Authentic different-question pair, not a minimal semantic pair; wording, requested relation, answer type, position and length may all change.'})
    assert len({n['id'] for n in nodes}) == len(nodes)
    ids = {n['id'] for n in nodes}
    assert all(e['source'] in ids and e['target'] in ids for e in edges)
    assert len(contrasts) == 2400
    packet = OUT/'external_graph/graph.json.gz'
    compressed(packet, {'nodes': nodes, 'edges': edges, 'question_contrast_pairs': contrasts, 'question_labels': labels})
    value = {'timestamp': stamp(), 'source': snapshot(__file__), 'all_passed': True,
        'material_manifest_sha256': sha(OUT/'material/manifest.json'),
        'graph': {'path': packet.relative_to(ROOT).as_posix(), 'sha256': sha(packet), 'bytes': packet.stat().st_size},
        'nodes_by_type': dict(Counter(n['kind'] for n in nodes)), 'edges_by_type': dict(Counter(e['type'] for e in edges)),
        'question_pairs': len(contrasts), 'same_literal_head_pairs': sum(r['same_literal_question_head'] for r in contrasts),
        'same_primary_answer_type_pairs': sum(r['same_primary_answer_type'] for r in contrasts),
        'coverage': [{'cohort': c, 'split': s, 'literal_head': h, 'primary_answer_type': t, 'questions': n}
                     for (c, s, h, t), n in sorted(counts.items())],
        'all_rows_including_confirmation': 'Original metadata only; no confirmation hidden targets, predictions or scored behavior.',
        'scope': 'External typed graph and literal annotation alignment. Corpus labels, positions and drawn edges do not prove internal semantic modules or reasoning chains.',
        'seconds': time.monotonic()-start}
    immutable(path, value)
    print('NATURAL_EXTERNAL_GRAPH_DONE', value['nodes_by_type'], len(contrasts), flush=True)
    return value


if __name__ == '__main__':
    main()
