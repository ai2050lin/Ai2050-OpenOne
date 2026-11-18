"""Second prospective frame panel after the original wording failed at State."""

from pathlib import Path
import random

import kv_chain_run as C
import kv_chain_study as S


M=C.M
PILOT='frame-repair-pilot'
HOLDOUT='frames-repair'
EXTRA_FRAMES=(
    ('Use the accepted entry name for State. Save the code letter for the Code line.',
     ('Accepted entry: {}.', 'Rejected entry: {}.')),
    ('Write the approved destination name on the State line. Use the letter only on the Code line.',
     ('Destination approved: {}.', 'Destination rejected: {}.')),
)
EXTRA_WORDS='''Apollo Mercury Venus Saturn Jupiter Neptune Pluto Mars Orion Draco Lyra Vega Sirius Altair Polaris Rigel Deneb Andromeda Pegasus Taurus Aries Gemini Cancer Leo Virgo Libra Sagittarius Pisces Hydra Phoenix Delphi Olympia Sparta Athens Rome Troy Olympus Europa Titan Rhea Juno Dione Atlas Vesta Ceres Pallas Helios Osiris Isis Hermes Ares Odin Thor Loki Freya Avalon Arcadia Avalon Atlantis Albion Atlantis Camelot Astoria Verona Belmont Dover Jasper Rowan Mason Slate Mercer Sutton Ranger Ranger Viper Comet Tempo Nova Prism Axis Apex Vector Pixel Raster Matrix Vertex Vertex Vertex Logic Echo Sigma Omega Omega Alpha Beta Gamma Delta Lambda Kappa Theta Omega Phi Psi Zoom Aero Aster Almond Basil Celery Caper Clover Cosmos Dahlia Hazel Pansy Marigold Myrtle Zinnia Orchid Tulip Jasmine Violet Aster Lotus Iris Sorrel Vinca Dahlia Fuchsia Poppy Daisy Briar Heather Leaf Twig Branch Seed Cane Moss Kelp Fungus Fern Petal Lichen Myrtle Bamboo Bean Cherry Pine Teak Aspen Catalpa Cedar Walnut Spruce Cypress Palm Acacia Cypress Alder Beech Chestnut Walnut Poplar Hemlock Maple Rowan Cypress Honey Cocoa Coffee Tea Sugar Sundae Ginger Cinnamon Nutmeg Anise Clove Salt Pepper Maple Honeydew Marlin Trout Salmon Perch Bass Pike Cod Tuna Tang Prawn Oyster Squid Mantis Crab Shrimp Lobster Dolphin Orca Whaler Raptor Osprey Harrier Kestrel Falcon Buzzard Pelican Gull Darter Robin Cardinal Sparrow Starling Bunting Thrush Wagtail Cuckoo Quail Pheasant Parrot Finch Albatross Sparrow Seagull Penguin Puffin Puffin Finch Egret Bittern Tern Teal Mallard Jaguar Cougar Leopard Lynx Tiger Lion Bear Moose Bison Antelope Gazelle Zebra Giraffe Rhino Hippo Camel Horse Llama Cheetah Panda Hyena Dingo Fox Jackal Wolf Squirrel Badger Marten Otter Beaver Mink Weasel Ermine Porcupine Rabbit Hare Mole Vole Shrew Mouse Rat Gopher Lemur Monkey Gorilla Gibbon Baboon Macaque Tamarin Ape Bonobo Salmon Trout Finch Leopard Goose Narwhal Whale Porpoise Seahorse Catfish Jellyfish Swordfish Mackerel Sturgeon Halibut Haddock Snapper Mullet Carp Minnow Eel Terrapin Turtle Iguana Cobra Viper Python Gecko Skink Salamander Newt Toad Frog Crocodile Scorpion Beetle Wasp Hoverfly Dragonfly Firefly Spider Cricket Ant Locust Cicada Moth Butterfly Caterpillar Maggot Flea Tick Termite Aphid Weevil Ladybug Hornet Gar Salmon Snapper''' 


def augment(tok,original):
    excluded={w.casefold() for panel in original['panels'].values() for w in panel['lexicon']}
    excluded.update(w.casefold() for words in BASE_WORDS(tok).values() for w in words)
    excluded.update(w.casefold() for words in M.Prior.pools(tok).values() for w in words)
    pool={}
    for w in dict.fromkeys(EXTRA_WORDS.split()):
        token=tok.encode(' '+w,add_special_tokens=False)
        if w.casefold() not in excluded and len(token)==1 and tok.decode(token)==' '+w:
            pool.setdefault(token[0],w)
    words=list(pool.values()); random.Random('chain-frame-repair-2954-v1').shuffle(words)
    assert len(words)>=65,len(words)
    return {PILOT:words[:24],HOLDOUT:words[24:]}


BASE_WORDS=M.words
BASE_PAYLOAD=M.payload
BASE_FRAMES=M.FRAMES


def use_material(tok,original):
    banks=augment(tok,original)
    M.FRAMES=(*BASE_FRAMES,*EXTRA_FRAMES)
    M.SIZES.update({PILOT:16,HOLDOUT:80})
    def vocab(t):
        return BASE_WORDS(t)|banks
    def payload(t,stage,families=None):
        rows=BASE_PAYLOAD(t,stage,families)
        if stage not in banks:return rows
        return [M.build_row(t,dict(r,frame=r['frame']+len(BASE_FRAMES)),r['fact'],r['desired']) for r in rows]
    M.words=vocab; M.payload=payload
    return banks


def plan(model,original):
    old=C.A.load('frames-status.json'); natural=C.A.load('frames-greedy.json')
    assert old['clean']==0 and not old['eligible_frames']
    failures=[r for r in natural['records'] if r['id'] not in set(natural['gate']['eligible'])]
    banks=use_material(model.tok,original)
    name='frame-repair-plan.json'
    if (C.OUT/name).exists():
        new=C.A.load(name); assert all(C.B.sha(k)==v for k,v in new['binding'].items())
        assert new['adapter_sha']==C.A.digest(Path(__file__))
        return new
    spec={}
    for stage in (PILOT,HOLDOUT):
        rows=M.payload(model.tok,stage)
        spec[stage]=dict(lexicon=sorted({w for r in rows for w in r['words']}),n=len(rows),material_sha=S.stable_identity(rows),
            frames=[4,5],family_count=len(rows)//16,lengths=sorted({len(r['token']['ids']) for r in rows}))
        assert spec[stage]['lengths'][-1]<=C.WIDTH
    assert set(banks[PILOT]).isdisjoint(banks[HOLDOUT])
    fields=('plan.json','frozen-models.json','words-analysis.json','frames-greedy.json','frames-status.json')
    C.A.save(name,dict(run_id=C.RUN,time=C.stamp(),panels=spec,frames=EXTRA_FRAMES,adapter_sha=C.A.digest(Path(__file__)),
        diagnosis=dict(previous_n=natural['gate']['n'],failed=len(failures),at_state=sum(r['divergence'] is not None and r['actual_state']!=r['state_id'] for r in failures),
                       exact_unqualified_ids=[r['id'] for r in failures]),
        binding={k:C.B.sha(k) for k in fields},
        scope='Second prospective validation. The first new-frame panel is a preserved failure. The new phrases distinguish the name and code fields explicitly. Fit, random heads, read boundaries, interventions and thresholds come unchanged from the original discovery freeze.',
        pilot='Run behavior only on isolated 16 pilot families. Open all 80 new families only if both new frames pass the original cell threshold. Failed frames in the new-family panel remain excluded by the exact original behavior rule.',
        interpretation='Assess this new panel on its own. Any paired reporting with the word panel is conditional on the revised material and does not reverse the original frame gate. No selection or refitting from this pilot or its holdout.'))
    return C.A.load(name)


def run(model,identity):
    original=C.A.load('plan.json'); S.verify(original,identity); fixed=S.F.verify_freeze(original)
    new=plan(model,original); combined=dict(original,panels=original['panels']|new['panels'])
    pilot=S.panels(model,combined,PILOT); b=C.behavior(model,pilot,PILOT)
    if len(S.valid_by_frame(b['gate']))!=2:
        C.A.save(HOLDOUT+'-status.json',dict(eligible_frames=[],clean=0,scope='revised isolated pilot failed',planned=new['panels'][HOLDOUT]['n']))
        return
    clean,compare,known=S.clean_stage(model,combined,HOLDOUT)
    if not clean:return
    native=S.load_np(HOLDOUT+'-native.npz')
    S.F.prior_predictions(HOLDOUT,fixed,clean,compare,native)
    if compare:S.causals(model,compare,HOLDOUT,fixed['controls'])
    S.F.assess(HOLDOUT,fixed,clean,compare,known,native,S.load_np(HOLDOUT+'-edits.npz') if compare else None)


if __name__=='__main__':
    import time
    import torch
    C.setup(); torch.set_num_threads(4); torch.set_float32_matmul_precision('highest')
    with C.P.GpuLease(),torch.no_grad():
        start=time.monotonic(); C.N.progress('frame-repair-load',0,2,start)
        identity=C.Prior.identity(); C.N.progress('frame-repair-load',1,2,start)
        model=C.N.Engine(); C.N.progress('frame-repair-load',2,2,start)
        try:run(model,identity)
        finally:model.close()
