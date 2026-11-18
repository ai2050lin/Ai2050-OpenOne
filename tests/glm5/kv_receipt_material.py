"""Exact token insertions for a shared prompt and answer layout."""

from hashlib import sha256
from itertools import accumulate
import json
import random

from kv_chain_run import ROOT
from kv_natural_material import DEFAULT_BREAKS, FIELDS, token_plan


RUN='KV-RECEIPT-20261117-v1'
SIZES=dict(pilot=12,discovery=36,words=64,frames=64)
BASE='Archive check complete.'
TRACKS=(
    (' The folder rests beside the shelf.', ' The table is clean and dry.', ' The room is quiet and cool.'),
    (' The clock hangs beside the door.', ' The hallway is clear and quiet.', ' The lamp is warm and bright.'),
    (' The paper lies beneath the notebook.', ' The window is open and clear.', ' The wall is smooth and pale.'),
)
FRAMES=(
    ('Use the included entry name for State.', ('The inventory includes {}.', 'The inventory excludes {}.')),
    ('For State, write the name of the route marked active.', ('Active route: {}.', 'Inactive route: {}.')),
    ('Fill State with the label that was kept.', ('Label kept: {}.', 'Label dropped: {}.')),
)
WORDS='''Acorn Agate Albatross Alchemy Alloy Alpine Amber Anvil Arcade Arrowhead Aspen Atom Azalea Badge Baker Bamboo Baron Basalt Beacon Beaker Beaver Bedrock Bengal Berry Bison Blizzard Blossom Bolero Boulder Bronze Buckhorn Buzzer Cactus Camel Canyon Carbon Cardinal Carmine Carriage Carousel Castle Cedar Chariot Cinder Citrus Cloudburst Clover Cobalt Cocoa Comet Compass Copper Corsair Coyote Cricket Crimson Crystal Dahlia Daisy Dancer Dart Dawn Deer Denim Diamond Dingo Dipper Domino Dove Dune Eagle Ebony Eclipse Ember Emerald Engineer Equinox Falcon Fathom Feather Fern Ferry Flint Flora Flux Fog Forest Fox Frost Garnet Gecko Geranium Gleam Gold Granite Grapevine Griffin Grove Guitar Gull Hammerhead Harbor Hazel Helix Heron Hibiscus Highland Horizon Hornet Hurricane Hyacinth Hyena Ibis Iceberg Indigo Iris Ivory Jackal Jaguar Jasper Jewel Juniper Kestrel Kettle Kingfisher Kiwi Knight Lantern Larch Lark Laurel Lavender Lemur Leopard Lilac Lizard Locket Loon Lynx Macaw Magenta Magnolia Mahogany Mantle Maple Marble Marigold Marmot Maroon Meadow Mercury Meteor Mica Minnow Mirage Mist Monarch Moon Moose Mosaic Mountain Musk Mustang Myrtle Nautilus Nectar Needle Nimbus Nutmeg Obsidian Ocelot Ochre Olive Omen Onyx Opal Orchid Otter Owl Oyster Panther Papaya Parrot Pebble Pelican Pepper Petal Pewter Phoenix Pine Plateau Platinum Plum Polaris Porcelain Porcupine Prairie Puffin Pumpkin Quartz Quill Radar Raven Reef Regent Ribbon Robin Rocket Rowan Ruby Sable Saffron Sage Sailfish Salmon Sambar Sapphire Scarlet Scorpion Seahawk Sequoia Serpent Shale Shimmer Sienna Silver Skylark Slater Slate Sparrow Spear Spider Spring Spruce Starling Steel Sterling Stone Storm Summit Swallow Swan Sycamore Tadpole Tangerine Tapestry Teak Teal Tempest Thimble Thorn Tiger Timber Tinker Topaz Torpedo Torrent Toucan Trident Trolley Trout Tulip Tundra Turbine Turtle Twilight Umber Unicorn Vanilla Velvet Vermilion Vermouth Vesper Vigil Violet Viper Voyager Walnut Walrus Warbler Wasp Wave Wheat Whistler Willow Winter Wombat Wren Yarrow Zenith Zephyr Zinnia Zircon Zither Acrylic Acetate Adapter Amplifier Amplify Arbor Argon Armory Asteroid Aviary Ballet Bandit Barley Barnacle Beryl Bongo Braid Brandy Bristle Bubble Calico Calypso Camper Canopy Canvas Cassette Celery Ceramic Charcoal Chiffon Chime Chrome Cicada Cleaver Colander Condor Confetti Conveyor Corduroy Cosmos Crab Cradle Crayon Crocus Crumpet Cuckoo Custard Dalmatian Damask Dragon Dresser Dusk Echo Envelope Etcher Fedora Fig Finch Fleece Flute Fossil Frittata Fuchsia Furnace Gibbon Ginger Glacier Glider Goblet Goblin Goose Graphite Grenade Guava Gumdrop Harp Hazelnut Heather Herring Hound Hummingbird Inkwell Ivory Javelin Jelly Junebug Kale Kernel Koi Ladle Lantern Ledger Lemon Lily Lintel Lobster Macaron Magpie Mango Maple Melon Millet Mink Molasses Mongoose Moth Mule Mural Mushroom Nettle Nickel Nightingale Obelisk Octopus Oriole Osprey Paisley Panda Parchment Partridge Peach Penguin Periwinkle Phantom Pineapple Pivot Pliers Pointer Pollen Ponder Poplar Porpoise Primrose Pumice Quail Raccoon Radish Rapier Rattle Razor Reindeer Rhubarb Rifle Rook Rosette Roulette Sander Sash Scallop Scepter Scone Scythe Seagull Seahorse Sesame Shard Sherbet Sickle Skimmer Smoothie Snapper Snowflake Sorrel Speckle Sphinx Spindle Sponge Squirrel Stallion Stencil Stork Strawberry Sultana Tadpole Talon Tartan Tassel Teacup Tern Thistle Toad Tortoise Tractor Trapeze Truffle Tuba Tuna Turquoise Unicorn Urchin Ursa Velvet Walrus Warthog Watercress Weasel Wedge Welder Whale Whisk Wicker Wildebeest Woodpecker Yak Zebra Zodiac'''


def sequence(tok):
    bits=[[''.join(parts[:i]) for i in (1,2,3)] for parts in TRACKS]
    by=[list(map(len,(tok.encode(s,add_special_tokens=False) for s in row))) for row in bits]
    assert all(x==by[0] for x in by) and len(set(by[0]))==3,by
    return [(-1,0,'')]+[(t,n,text) for t,row in enumerate(bits) for n,text in zip(by[0],row,strict=True)]


def vocabulary(tok):
    prev=ROOT/'tests/glm5/result/kv_chain_20261117_v1'
    used=set()
    for f in (prev/'plan.json',prev/'frame-repair-plan.json'):
        for spec in json.loads(f.read_text(encoding='utf-8'))['panels'].values():
            used.update(x.casefold() for x in spec['lexicon'])
    for f in (ROOT/'tests/glm5/result/kv_length_20261116_v1').glob('material-*.json'):
        for r in json.loads(f.read_text(encoding='utf-8')):
            used.update(x.casefold() for x in r['values'].values())
            used.update(x.casefold() for x in r['words'])
    bank={}
    for w in WORDS.split():
        ids=tok.encode(' '+w,add_special_tokens=False)
        if w.casefold() not in used and len(ids)==1 and tok.decode(ids)==' '+w: bank.setdefault(ids[0],w)
    words=list(bank.values());random.Random(RUN).shuffle(words)
    assert len(words)>=105,len(words)
    return dict(pilot=words[:16],discovery=words[16:42],words=words[42:70],frames=words[70:])


def build(tok,meta,track=-1,n=0,extra='',donor=False,breaks=DEFAULT_BREAKS):
    first,second,note,spare,archive=meta['words']; candidates=(first,second)
    fact=meta['fact']; desired=meta['desired']; role=desired if donor else fact
    rule,statements=FRAMES[meta['frame']]
    labels='QP' if meta['mapping'] else 'PQ'
    vals=dict(zip(FIELDS,(candidates[desired],note,spare,archive),strict=True))
    facts='\n'.join(statements[i!=role].format(w) for i,w in enumerate(candidates))
    items=dict(zip(meta['words'],(*labels,labels[1-fact],labels[fact],labels[fact]),strict=True))
    book='\n'.join(f'{meta["words"][i]} -> {items[meta["words"][i]]}' for i in meta['order'])
    banner=f'Complete the six record lines. {rule} Put the name itself in State. Copy Note, Spare and Archive. Use the State name to look up its letter. Use the letter only on the final Code line.'
    fillers='\n'.join(f'{k} entry: {v}' for k,v in vals.items() if k!='State')
    layout='\n'.join([*(f'{k}: <name>' for k in FIELDS),f'Receipt: {BASE}','Code: <P or Q>'])
    prompt=(f'{banner}\n\n{facts}\n{fillers}\nCodebook:\n{book}\n\n'
            'Return the lines in the exact order below. The Receipt begins with the printed sentence and may continue with a neutral note on that same line.\n'+layout)
    prefix='\n'.join([*(f'{k}: {v}' for k,v in vals.items()), f'Receipt: {BASE}{extra}', 'Code:'])
    row=dict(meta,track=track,n=n,extra=extra,id=f"{meta['family']}-{track+1}-{n}",values=vals,prompt=prompt,prefix=prefix,
             code=labels[desired],gold=labels[fact],receipt=BASE+extra)
    k=row['token']=token_plan(tok,row,breaks)
    phrase=tok.encode(' '+BASE,add_special_tokens=False)
    start=k['break_positions'][3]+1
    found=[i+len(phrase) for i in range(start,len(k['ids'])) if k['ids'][i:i+len(phrase)]==phrase]
    assert len(found)==1 and k['ids'][found[0]:found[0]+n]==tok.encode(extra,add_special_tokens=False)
    row['insertion']=found[0]
    row['sites']={
        'receipt':list(range(start,k['break_positions'][4])),
        'insert':list(range(found[0],found[0]+n)),
        'gen':[k['slots'][0]],'note':[k['slots'][1]],'extra':k['slots'][2:],
    }
    return row


def families(tok,stage):
    seq=sequence(tok); pool=vocabulary(tok)[stage]; rng=random.Random(RUN+'/'+stage); used=set()
    for i in range(SIZES[stage]):
        while True:
            words=rng.sample(pool,5); signature=tuple(sorted(words))
            if signature not in used: used.add(signature); break
        fid=sha256((stage+'/'+','.join(words)).encode()).hexdigest()[:16]
        frame=1+(i%2) if stage=='frames' else 0
        fact=(i//2)%2; conflict=(i//4)%2; mapping=(i//8)%2
        meta=dict(family=fid,stage=stage,frame=frame,words=words,fact=fact,desired=fact^conflict,conflict=conflict,
                  mapping=mapping,order=rng.sample(range(5),5))
        rows=[build(tok,meta,*variant) for variant in seq]
        base=rows[0]; ids=base['token']['ids']; cut=base['insertion']
        for r in rows:
            alt=r['token']['ids']
            assert r['prompt']==base['prompt'] and r['insertion']==cut and alt[:cut]+alt[cut+r['n']:]==ids
        yield rows
