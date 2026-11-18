"""Matched authorization minimal pairs on lexically independent record families."""

from collections import Counter
from itertools import batched, combinations, product
from random import Random

import record_update_material as M
from record_update_v2 import PILOT as PRIOR_V2
from record_update_v3 import PILOT as PRIOR_V3


RUN = 'RECORD-AUTHORIZATION-20261117-v1'
SEED = 29582026
SIZES = dict(pilot=96, discovery=192, words=384, frames=384, combos=384, lockbox=384)
POOLS = dict(pilot='pilot', discovery='known', words='words', frames='known', combos='known', lockbox='lockbox')
FRAME_IDS = dict(pilot=(0, 1), discovery=(0, 1), words=(0, 1), frames=(2, 3), combos=(0, 1), lockbox=(2, 3))
TRAIN_SETTINGS = ((False, False), (False, True), (True, False))
HELD_SETTING = (True, True)
STATUS = ('refused', 'granted')
BASE_ENCODE = M.encode
WORDS = '''
Anchor Antler Apron Arcade Arena Armor Atlas Axle Badge Bagel Balcony Ballot Bandit Barrel Basket Bay Beacon Beam
Bear Beaver Beetle Bell Berry Biscuit Blade Blanket Blossom Boat Bonnet Book Bottle Bowl Bracelet Brake Branch Bread
Brick Bridge Brook Broom Brush Button Cabin Camel Campus Canal Canoe Cape Capsule Carpet Castle Catch Cellar Chain
Chair Chalk Chapel Cherry Chime Chip Church Circle Clam Cliff Clock Cloth Coast Coin Collar Colony Comb Cone Cookie
Coral Corner Couch Cradle Crane Crown Crust Cub Cup Curtain Cycle Dagger Dance Dancer Dash Deck Deer Ditch Dock Dog
Dome Donkey Door Dragon Drum Dust Earth Elm Engine Ermine Falcon Fawn Feather Fence Fig Flag Flame Flour Flower Flute
Fork Fountain Frame Freesia Frog Frost Fudge Funnel Garden Gate Gear Gem Gemstone Giraffe Glass Globe Glove Goat Gold
Gong Gravel Guitar Hammer Harbor Harp Hare Hatch Hawk Heap Hedge Helmet Hen Heron Hill Hinge Holly Hood Horse House
Ink Island Jacket Jar Jasmine Jelly Jewel Jug Kettle Key Kite Kitten Kiwi Knife Knot Lace Lamp Leaf Lens Lichen
Lion Lizard Loaf Lock Lodge Lynx Magnet Magpie Manor Mantle Market Mask Mast Medal Melon Mesh Mill Mirror Mole
Monarch Moon Moose Motor Mushroom Nest Net Needle Nickel Noble Nook Oak Oasis Oar Oat Octopus Olive Otter Oven Owl
Ox Oyster Palace Paper Parcel Parrot Patch Peach Peak Pegasus Pen Penguin Penny Piano Pickle Pie Pigeon Pine Pitch
Plate Pool Poppy Porcelain Post Pot Pouch Prune Pumpkin Puppy Pyre Quail Queen Quilt Radish Raft Rail Rain Raisin
Rake Ramp Rattle Reindeer Resin Ring Rink Road Roe Roof Rope Rowboat Rudder Rust Saddle Sail Salt Sandal Screen Seal
Seed Sheep Shield Ship Shirt Shoe Shovel Shrimp Silk Skiff Skirt Sled Slope Smelt Smoke Snow Soap Socket Sole Sparrow
Spear Spice Spool Spoon Stair Star Steam Stem Stork Straw Sugar Swan Sword Table Tack Tank Tarp Tart Taxi Teapot
Tent Thorn Thrush Tile Tin Toad Toast Tomb Tower Tractor Trail Tray Trout Trunk Tub Tuna Twig Umbrella Urchin Vale
Vase Vine Violin Viper Wagon Wall Wasp Watch Water Wheel Wheat Wig Window Wire Wood Wreath Wren Yak Yarn Zebra Zinc
Zither Agnes Alan Albert Alice Alvin Amanda Amos Anita Anna Anne Arthur Audrey Bart Beatrice Ben Betty Boris Bruce
Carl Carol Carrie Casey Cecil Clara Claude Dale Daryl Dean Diana Dirk Doris Dorothy Earl Edith Edgar Elsa Elvis
Erica Ernest Esther Frank Fred Gail Gary George Grace Gus Hank Harry Harvey Helen Henry Hugh Ian Ida Irene Isaac
Jack Janet Jane Jason Jean Jeff Jerry Jill Joan Joel John Judy Julia Keith Kelly Ken Kim Kurt Laura Leo Leroy
Lewis Linda Lisa Lois Louis Lucy Luke Lynn Mark Martha Mary Mason Max Maya Megan Miles Mona Nancy Nina Noah Nora
Norma Oliver Omar Oscar Otto Paige Paul Peggy Peter Philip Quinn Ralph Ray Reed Rex Rita Robin Roger Ron Rosa Rose
Roy Ruby Ruth Sally Sam Sara Saul Scott Seth Shane Sheila Sid Simon Sophie Stan Stella Steve Susan Ted Terry Theo
Tina Toby Tom Tony Tracy Troy Vicky Victor Vince Violet Wade Walt Wanda Wayne Wendy Will Wyatt Zach Zoe
'''


def pools(tokenizer):
    former = set(' '.join((*M.VOCABULARY.values(), PRIOR_V2, PRIOR_V3)).split())
    excluded = {tokenizer.encode(' '+w, add_special_tokens=False)[0] for w in former}
    seen = set(excluded)
    candidates = []
    for word in WORDS.split():
        tokens = tokenizer.encode(' '+word, add_special_tokens=False)
        if len(tokens) == 1 and tokens[0] not in seen and tokenizer.decode(tokens) == ' '+word:
            seen.add(tokens[0])
            candidates.append(word)
    Random(SEED).shuffle(candidates)
    sizes = dict(pilot=24, known=54, words=36, lockbox=36)
    assert len(candidates) >= sum(sizes.values()), len(candidates)
    result = {}
    offset = 0
    for key, size in sizes.items():
        result[key] = candidates[offset:offset+size]
        offset += size
    return result


def render(frame, names, earlier_is_target, status, key):
    old, other, earlier, proposal = names
    destination = 'East' if earlier_is_target else 'West'
    intros = (
        f'Track the East record. Its saved label is {old}. The separate West record starts at {other}.',
        f'You are checking East, which has {old} on file. West is a different record with label {other}.',
        f'The original value on the East card is {old}; the West card has {other}. Our report concerns East only.',
        f'The file starts with {old} for East and {other} for West. Find the label that ends up on file for East.',
    )
    rules = (
        'A request marked granted replaces the saved label of its named record. A request marked refused leaves the saved label unchanged.',
        'Process both requests in order. Use the proposed label only for a granted request that names East. If no such request exists, keep the initial East label.',
        'Refused requests never change a saved label. The most recent granted request for East takes effect. With none, retain the original East label.',
        'Leave a record alone when its request is refused. If it is granted, replace the label of that record. Apply the requests from first to last.',
    )
    first = (
        f'Notice 1: {destination} proposes {earlier}. Authorization: refused.',
        f'First request for {destination}: replace its label with {earlier}. Decision: refused.',
        f'Earlier notice. Card: {destination}. Proposed value: {earlier}. Ruling: refused.',
        f'1. Record {destination} requests label {earlier}; permission is refused.',
    )
    second = (
        f'Notice 2: East proposes {proposal}. Authorization: {status}.',
        f'Second request for East: replace its label with {proposal}. Decision: {status}.',
        f'Later notice. Card: East. Proposed value: {proposal}. Ruling: {status}.',
        f'2. Record East requests label {proposal}; permission is {status}.',
    )
    request = ('What is the final saved East label? Report that label, then its letter from the key. '
               'Use exactly two lines and end after the letter. Format:\nState: <label>\nCode: <P or Q>')
    return intros[frame], rules[frame], first[frame], second[frame], 'Letter key: '+key+'.', request


def families(tokenizer, split):
    source = POOLS[split]
    words = pools(tokenizer)[source]
    label_pairs = list(combinations(words, 2))
    Random(M.digest([SEED,source])).shuffle(label_pairs)
    start = dict(pilot=0, discovery=0, words=0, frames=SIZES['discovery'],
                 combos=SIZES['discovery']+SIZES['frames'], lockbox=0)[split]
    assert start+SIZES[split] <= len(label_pairs)
    settings = (HELD_SETTING,) if split in ('combos', 'lockbox') else TRAIN_SETTINGS
    settings_by_frame = tuple(product(FRAME_IDS[split], settings))
    for i in range(SIZES[split]):
        rng = Random(M.digest([SEED,split,i]))
        old, proposal = rng.sample(label_pairs[start+i], 2)
        other, earlier = rng.sample([w for w in words if w not in (old,proposal)], 2)
        names = (old, other, earlier, proposal)
        frame, (earlier_is_target, reverse) = settings_by_frame[i%len(settings_by_frame)]
        codes = ('Q','P','Q','P') if reverse else ('P','Q','P','Q')
        family = M.digest([RUN, split, i, names])[:20]
        key = ', '.join(f'{names[j]} -> {codes[j]}' for j in rng.sample(range(4),4))
        for update in ((False, True) if i%2 == 0 else (True, False)):
            status = STATUS[update]
            index = 3 if update else 0
            yield dict(id=f'{family}-{int(update)}',family=family,split=split,frame=frame,
                       names=names,codes=codes,actions=(1 if earlier_is_target else 3,0 if update else 1),
                       truth=[old,old,names[index]],answer=codes[index],active=0,update=update,
                       setting=[earlier_is_target,reverse],status=status,
                       sections=render(frame,names,earlier_is_target,status,key))


def encode(tokenizer, row):
    result = BASE_ENCODE(tokenizer,row)
    token = result['token']
    token['labels'] = [tokenizer.encode(' '+w,add_special_tokens=False)[0] for w in row['names']]
    notice = token['sections'][3]
    status = tokenizer.encode(' '+row['status'],add_special_tokens=False)
    assert len(status)==1
    cue, = [j for j in notice if token['prompt'][j]==status[0]]
    proposal, = [j for j in notice if token['prompt'][j]==token['labels'][3]]
    assert proposal < cue < token['e']
    token.update(cue=cue, proposal=proposal)
    return result


def inspect(tokenizer):
    summary, identities, statuses = {}, set(), set()
    for split, size in SIZES.items():
        rows = [encode(tokenizer,row) for row in families(tokenizer,split)]
        settings = Counter((r['frame'],tuple(r['setting'])) for r in rows if r['update'])
        assert len(rows)==2*size and min(settings.values())>0
        for a,b in batched(rows,2):
            left,right = a['token'],b['token']
            assert a['family']==b['family'] and a['update']!=b['update']
            assert a['names']==b['names'] and a['codes']==b['codes'] and a['frame']==b['frame']
            assert a['answer']!=b['answer'] and a['truth'][0]==b['truth'][0]
            # Compare only the focal notice substitution. The earlier denial must remain byte-identical.
            assert a['sections'][:3]==b['sections'][:3] and a['sections'][4:]==b['sections'][4:]
            assert len(a['sections'][3])==len(b['sections'][3]) and a['sections'][3].replace(a['status'],b['status'])==b['sections'][3]
            assert left['sections']==right['sections'] and left['labels']==right['labels']
            assert left['cue']==right['cue'] and left['proposal']==right['proposal']
            different = [j for j,(x,y) in enumerate(zip(left['prompt'],right['prompt'],strict=True)) if x!=y]
            assert different==[left['cue']] and left['draft'][-1]==right['draft'][-1]==tokenizer.eos_token_id
            identity = tuple(sorted((a['names'][0],a['names'][3])))
            assert identity not in identities
            identities.add(identity)
            statuses.update([left['prompt'][left['cue']],right['prompt'][right['cue']]])
        summary[split]=dict(families=size,views=len(rows),sha256=M.digest(rows),min_cell=min(settings.values()),
                            settings=[[f,*v,n] for (f,v),n in sorted(settings.items())],
                            min_prompt=min(len(r['token']['prompt']) for r in rows),
                            max_prompt=max(len(r['token']['prompt']) for r in rows),
                            min_cue=min(r['token']['cue'] for r in rows),max_cue=max(r['token']['cue'] for r in rows))
    if len(statuses)!=2:
        raise ValueError('Authorization must consist of two fixed distinct vocabulary IDs')
    return summary
