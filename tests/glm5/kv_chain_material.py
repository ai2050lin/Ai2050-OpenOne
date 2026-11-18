"""Controlled 14B fact-to-code materials with four crossed within-family factors."""

from hashlib import sha256
from itertools import product
import random

import kv_length_run as Prior
from kv_natural_material import DEFAULT_BREAKS, FIELDS, token_plan


SEED = 'KV-CHAIN-2954-20261117'
SIZES = dict(pilot=6, discovery=32, words=80, frames=80)
SITES = ('fact', 'book', 'gen', 'note', 'extra', 'receipt')
RECEIPTS = (
    ('Archive check complete.',
     'Archive check complete. This receipt confirms that all fields were copied into the record and that the record was filed in the active register for the current review. The review is now ready for the final entry.'),
    ('Record review complete.',
     'Record review complete. The paper was placed in the folder and the folder was put on the shelf beside the other folders. The table was wiped with a clean cloth before the room was closed.'),
)
LABELS = ('P', 'Q')
FRAMES = (
    ('State is the entry the inventory includes.', ('The inventory includes {}.', 'The inventory excludes {}.')),
    ('State is the route that the dispatcher approved.', ('The dispatcher approved the route to {}.', 'The dispatcher rejected the route to {}.')),
    ('State is the site that has permission to operate.', ('Permission to operate was granted to {}.', 'Permission to operate was denied to {}.')),
    ('State is the destination selected for this delivery.', ('For this delivery the destination selected was {}.', 'For this delivery the destination set aside was {}.')),
)


def words(tok):
    original = Prior.pools(tok)
    excluded = {word.casefold() for words_ in original.values() for word in words_}
    for path in Prior.OUT.glob('material-*.json'):
        for row in Prior.P.load(path):
            excluded.update(v.casefold() for v in (*row['values'].values(), *row['words']))
    extras = '''Anchor Bell Bulb Carpet Canvas Camera Couch Curtain Compass Cupboard Deck Drawer Engine Filter Hammer Hook Kiln Lantern Lever Mirror Nail Oven Plate Plug Rail Rope Saddle Shell Shovel Staple Stove Towel Tray Tube Umbrella Valve Wheel Window Wire Button Card Clamp Clip Coil Comb Cuff Desk Drum Dust Flask Funnel Grinder Helmet Hose Jar Jacket Lamp Lens Lid Loom Magnet Mask Meter Nail Net Nozzle Pan Pedal Pipe Pitcher Pocket Pump Rack Razor Ribbon Roller Saw Scale Screw Screwdriver Sensor Shutter Sink Switch Tap Tent Thread Timer Tong Tongs Torch Tripod Vest Wallet Washer Watch Wrench Axle Battery Bench Block Brush Cable Chain Chisel Clock Cord Drill File Frame Gear Handle Hinge Iron Key Latch Lock Nut Pin Pliers Rod Ruler Shield Spring Strap Tool Tray Wedge Brush Bin Can Cart Case Cup Dish Dock Flag Fork Glass Hook Hood Jug Knife Mat Mop Mug Pad Pan Peg Pot Rag Rack Seal Sheet Soap Spade Spoon Stool Table Tank Tape Tile Tin Tub Tun Bag Ball Band Bar Beam Belt Boat Board Boom Bow Box Brake Bridge Broom Cage Cap Chest Cone Crank Crate Crown Disc Door Duct Fan Fin Flap Float Gate Grill Grip Guard Handle Hanger Harp Holder Hub Jack Joint Kite Knob Latch Leg Loop Mat Mesh Mold Motor Mount Net Oil Paddle Panel Pawl Pen Pole Pulley Ring Rock Roller Roof Rotor Sail Shaft Shelf Shingle Shoe Slide Slide Sock Socket Spool Stack Stand Stem Stick Stool Stopper Switch Tag Tank Tray Tread Truck Wheel Beryl Penny Peony Atlas Aria Arrow Echo Flame Harbor Indigo Iris Jade Lotus Maple Melody Mica Moss Nova Opal Orbit Pearl Rain Reed Ridge River Rose Sage Sand Silver Slate Snow Sorrel Star Storm Sun Terra Tide Vale Violet Willow Wren Yew Zinc Amber Birch Cedar Daisy Dawn Elm Fern Finch Hazel Hill Holly Lake Lily Luna Meadow Mint Oak Pine Plum Rowan Ruby Shade Sky Stone Thorn Tulip Ash Spruce Cloud Dune Glade Grove Grove Cliff Banks Bay Field Glen Brook Lake Pond Trace Marsh Strand Glimmer Sedge Ivy Falcon Heron Crane Robin Swan Drake Goose Lark Kite Eagle Raven Hawk Gull Stork Plover Owl Swift Tern Crane Finch Coral Reef Ridge Shoal Drift Peak Crest Delta Canyon Basin Bluff Mesa Canyon Ridge Portal Ridge Quarry Glacier Moraine Bluff Quarry Hollow Summit Divide Cascade Rapids Torrent Thicket Copse Orchard Prairie Steppe Tundra Wetland Lowland Highland Woodland Moorland Upland Heath Downs Brook Creek Riverwater Gully Knoll Slough Estuary Peninsula Cape Cove Inlet Sound Port Haven Anchorage Beacon Lighthouse Landing Pier Wharf Jetty Dockyard Yardness Almond Anise Apple Apricot Banana Basil Bean Beet Berry Broccoli Carrot Cherry Chive Clove Coconut Date Dill Fig Ginger Grape Guava Lemon Lime Mango Melon Oat Olive Onion Orange Papaya Parsley Peach Pear Pepper Pineapple Poppy Potato Rice Saffron Sesame Shallot Spinach Squash Thyme Tomato Vanilla Walnut Wheat Yam Cassava Cumin Fennel Garlic Kale Leek Lentil Lettuce Maize Mango Nectar Olive Quince Radish Rye Turnip Watercress'''
    bank = {}
    for spelling in dict.fromkeys(extras.split()):
        if spelling.casefold() in excluded:
            continue
        ids = tok.encode(' ' + spelling, add_special_tokens=False)
        if len(ids) == 1 and tok.decode(ids) == ' ' + spelling:
            bank.setdefault(ids[0], spelling)
    pool = list(bank.values()); random.Random(SEED).shuffle(pool)
    assert len(pool) >= 128, len(pool)
    return dict(pilot=pool[:24], discovery=pool[24:60], words=pool[60:110], frames=pool[110:])


def build_row(tok, meta, fact, desired, breaks=DEFAULT_BREAKS):
    a, b, neutral, spare, archive = meta['words']
    competitors = (a, b)
    base_fact = meta.get('base_fact',fact)
    note = competitors[1-base_fact] if meta['compete'] else neutral
    values = dict(zip(FIELDS, (competitors[desired], note, spare, archive), strict=True))
    labels = LABELS if not meta['map'] else LABELS[::-1]
    codebook = {a:labels[0], b:labels[1], neutral:labels[base_fact], spare:labels[base_fact], archive:labels[base_fact]}
    rule, statements = FRAMES[meta['frame']]
    facts = '\n'.join(statements[x!=fact].format(w) for x, w in enumerate(competitors))
    edits = [f'{k} entry: {v}' for k,v in values.items() if k != 'State']
    listing = '\n'.join(f'{meta["words"][i]} -> {codebook[meta["words"][i]]}' for i in meta['order'])
    receipt = RECEIPTS[meta['text']][meta['distance']]
    layout = '\n'.join([*(f'{key}: <word>' for key in FIELDS), f'Receipt: {receipt}', 'Code: <P or Q>'])
    prefix = '\n'.join([*(f'{key}: {values[key]}' for key in FIELDS), f'Receipt: {receipt}', 'Code:'])
    prompt = (f'Complete the record below. {rule} Copy Note, Spare and Archive from their entries. Look up State in the codebook to find its code.\n\n'
              f'{facts}\n'+'\n'.join(edits)+f'\nCodebook:\n{listing}\n\n'
              'Return exactly these six plain-text lines in order. Copy the Receipt line exactly. Use the code letter by itself after Code:.\n'+layout)
    row = dict(meta, fact=fact, desired=desired, values=values, prefix=prefix, prompt=prompt, code=labels[desired], gold=labels[fact], receipt=receipt)
    row['token'] = token_plan(tok, row, breaks)
    return row


def groups(tok, row):
    chat = tok.apply_chat_template([dict(role='user', content=row['prompt'])], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(chat, add_special_tokens=False, return_offsets_mapping=True)
    base = chat.index(row['prompt']); p = row['prompt']; k = row['token']
    assert enc.input_ids == k['ids'][:k['start']]
    def span(start, stop):
        return [i for i, (a,b) in enumerate(enc.offset_mapping) if b>base+start and a<base+stop and b>a]
    fact_start = p.index('\n\n')+2
    book_start = p.index('\nCodebook:\n')+len('\nCodebook:\n')
    books = []
    for w in row['words'][:2]:
        offset = p.index(w+' ->', book_start)
        books.extend(span(offset, p.index('\n', offset)))
    output = dict(zip(SITES, (span(fact_start,p.index('\nNote entry:')), books, [k['slots'][0]], [k['slots'][1]],
                             k['slots'][2:], list(range(k['break_positions'][3]+1,k['break_positions'][4]+1))), strict=True))
    all_indices = [x for src in output.values() for x in src]
    assert all(output.values()) and len(set(all_indices))==len(all_indices) and max(all_indices)<len(k['ids'])-1
    return output


def payload(tok, stage, families=None):
    pool = words(tok)[stage]; rng = random.Random(f'{SEED}/{stage}'); seen=set(); rows=[]
    n = SIZES[stage] if families is None else families
    for j in range(n):
        while True:
            values=rng.sample(pool,5); key=tuple(sorted(values))
            if key not in seen:
                seen.add(key); break
        family=sha256((stage+'/'+'/'.join(values)).encode()).hexdigest()[:16]
        order=rng.sample(range(5),5)
        frame=(2+j%2) if stage=='frames' else j%2
        # Both roles occur equally often within each frame. Each family contains the full four-factor grid.
        fact=(j//2)%2
        for t,d,c,m in product((0,1),repeat=4):
            factors=(t,d,c,m)
            meta=dict(id=family+'-'+''.join(map(str,factors)),family=family,stage=stage,words=values,order=order,frame=frame,text=t,distance=d,compete=c,map=m)
            row=build_row(tok,meta,fact,fact)
            # Four arms per family preselected for role-donor conflict; balance every factor pair as j varies.
            row['challenge']=(t^c)==(j%2) and (d^m)==((j//2)%2)
            rows.append(row)
    return rows
