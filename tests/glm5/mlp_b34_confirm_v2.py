"""Second freeze after the v1 discovery gate failed; no v1 holdout is opened."""
import argparse
from itertools import product

import mlp_b34_confirm_design as design

NAMES = {
    'discovery': 'Alice Ben Clara David Emma Frank Grace Helen Isaac Julia Kevin Laura Mark Nora Owen Paula Quinn Rosa Simon Vera Will Zack'.split(),
    'holdout': 'Adam Beth Carol Daniel Edith Felix Gina Henry Iris Jack Kara Leo Maria Nina Oscar Peter Rita Sam Tina Victor Wendy Zoe'.split(),
}
FRAMES = {
    'discovery': (
        ('depot-v2', 'The sending account is {src}. The designated receiving account is {dst}.', 'Is {who} the designated receiving account?'),
        ('museum-v2', 'The recipient of the package is {dst}. The sender of the package is {src}.', 'Is {who} the recipient of the package?'),
        ('rail-v2', 'Sender of the cargo: {src}. Recipient of the cargo: {dst}.', 'Does the record identify {who} as the recipient?'),
        ('lab-v2', 'Receiving party: {dst}. Dispatching party: {src}.', 'Is {who} the receiving party?'),
    ),
    'holdout': (
        ('coast-v2', 'On the delivery form, the origin is listed as {src}, and the designated destination account is {dst}.', 'Does the form name {who} as the destination account?'),
        ('print-v2', 'The proof is scheduled to arrive with {dst}. The package was sent by {src}.', 'Is {who} scheduled to take delivery of the proof?'),
        ('archive-v2', 'The first entry names {src} as the person mailing the folder. The second names {dst} as the person taking delivery.', 'Is {who} the person taking delivery?'),
        ('garden-v2', 'The transfer record names {dst} on the receiving side and {src} on the shipping side.', 'Is {who} named on the receiving side?'),
    ),
}


def materials(tokenizer, split, n, rng):
    strata = list(product(range(4), (False, True), (False, True), (False, True)))
    schedule = strata * (n // len(strata))
    assert len(schedule) == n
    rng.shuffle(schedule)
    used = set()
    for i, (frame, reverse, flip, late) in enumerate(schedule):
        topic, record, question = FRAMES[split][frame]
        notes = [f'Inspection note {j+1}: ' + rng.choice(design.BACKGROUND[split]) for j in range(18+rng.randrange(5))]
        insert_at = len(notes)-2 if late else 2
        codes = 'A means no. B means yes.' if reverse else 'A means yes. B means no.'
        for _ in range(100):
            names = rng.sample(NAMES[split], 2)
            query, other = names[flip], names[not flip]
            if (topic, query, other) in used:
                continue
            prompts, ids = [], []
            for receives in (False, True):
                src, dst = (other, query) if receives else (query, other)
                lines = [f'Record V2-{split}-{i:04d}. Use the shipment entry to answer the question.',
                         *notes[:insert_at], record.format(src=src, dst=dst), *notes[insert_at:],
                         question.format(who=query), 'Reply with one letter. ' + codes]
                prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': '\n'.join(lines)}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) + 'Answer:'
                prompts.append(prompt)
                ids.append(tokenizer.encode(prompt, add_special_tokens=False))
            if len(ids[0]) == len(ids[1]) and 510 <= len(ids[0]) <= 1000 and ids[0][-1] == ids[1][-1]:
                break
        else:
            raise RuntimeError(f'Could not construct an equal-length pair at {split}/{i}')
        used.add((topic, query, other))
        yield {'id': f'{split}-v2-{i:04d}', 'split': split, 'frame': topic, 'reverse_code': reverse,
               'query_is_y': flip, 'record_late': late, 'recipient_index': int((i//2+i) % 2), 'prompts': prompts,
               'tokens': ids, 'correct': [int(not reverse), int(reverse)], 'entities': sorted(names)}


def configure():
    design.OUT = design.ROOT/'tests/glm5/result/mlp_b34_confirm_v2'
    design.RUN = 'MLP-B34-B35-ROLE-CONFIRM-v2'
    design.SEED = 20261108134
    design.SOURCE_FILES += ('mlp_b34_confirm_v2.py',)
    design.materials = materials


def main(stage):
    configure()
    if stage == 'freeze':
        design.freeze()
    elif stage in ('select', 'report'):
        from mlp_b34_confirm_analyze import select, report
        (select if stage == 'select' else report)(design.checked_plan())
    else:
        from mlp_b34_confirm_run import main as run
        run(stage)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'discovery', 'select', 'holdout', 'capability', 'report'])
    main(parser.parse_args().stage)
