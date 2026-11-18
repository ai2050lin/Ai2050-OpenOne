"""Independent, unique scenario families for the three historical native candidates."""
from itertools import product
from pathlib import Path
import sys

import condition_chain_candidates as C
import condition_chain_materials as M
import condition_chain_run as R

R.RUN = "CONDITION-CHAIN-DIVERSE-CANDIDATES-20261115-v5"
R.OUT = R.OUT.with_name("condition_chain_diverse_candidates_20261115_v5")
R.TMP = R.TMP.with_name("condition_chain_diverse_candidates_20261115_v5")

KINDS = {
    "boundary": {
        "pilot": ["sample pressure","sample current"],
        "discover": ["soil moisture","ink viscosity","cable tension","water flow"],
        "words": ["axle load","oven heat","shop noise","power draw","screen glare","tank volume"],
        "frames": ["parcel weight","pump speed","air humidity","pitch drift","wire resistance","bench temperature"],
    },
    "payment": {
        "pilot": ["sample camp pass","sample wheel rental"],
        "discover": ["parking bay","ferry crossing","shoe rack","book loan"],
        "words": ["bike hire","camera rental","market stall","gallery entry","theater seat","pool locker"],
        "frames": ["bridge crossing","trailer slot","workshop desk","shuttle ride","fax service","hall reservation"],
    },
    "assessment": {
        "pilot": ["sample rotor","sample dial"],
        "discover": ["coolant valve","soundproof panel","digital timer","pilot lamp"],
        "words": ["fabric panel","water meter","electrical socket","bicycle chain","phone display","air hose"],
        "frames": ["solar charger","door latch","rain barrel","tape spindle","alarm bell","desk fan"],
    },
}
LABELS = {
    "pilot": ["Lark","Finch","Jay","Crow"],
    "discover": ["Ashby","Benton","Cavan","Darby","Ennis","Farley","Gates","Hanley"],
    "words": ["Aber","Bristol","Creel","Dawes","Evans","Frost","Grove","Holt","Isley","Jakes","Knox","Lake","Marsh","Nash","Oakes","Pike"],
    "frames": ["Quinn","Reeve","Sands","Tate","Upton","Vale","Wells","Yates","Zane","Adler","Brant","Clive","Dunn","Earle","Flynn","Galton"],
}
ASSESSMENT = {
    "pilot": [("eyeballing an approximate value","recording the actual gauge reading"),
              ("asking for a rough estimate","reading the number from a calibrated counter")],
    "discover": [("guessing from its age","reading its temperature from a probe"),
                 ("taking an arbitrary guess","reading a recorded sound measurement"),
                 ("estimating from its appearance","counting the ticks over a timed interval"),
                 ("trusting a casual hunch","reading a handheld light meter")],
    "words": [("inferring from a supplier's guess","recording the measured tear force"),
              ("guessing from the casing","logging its observed flow during a test"),
              ("making an estimate without tools","reading the voltage on a meter"),
              ("going by an impression","checking the tension on a calibrated gauge"),
              ("guessing from a quick glance","measuring the luminance with a sensor"),
              ("estimating without a reading","noting the measured pressure drop")],
    "frames": [("a technician's unverified hunch","a series of readings of current under load"),
               ("an estimate made by eye","a logged measurement of opening force"),
               ("a rough guess based on color","a measured volume from a calibrated vessel"),
               ("an unsupported prediction","a measured rotation rate from an optical counter"),
               ("an assumption about the label","a measured sound level at set distances"),
               ("a guess based on its shape","measured airflow from a calibrated anemometer")],
}


def make(split,task,index,arm):
    signal,cue=arm
    j,k=divmod(index,len(LABELS[split]))
    concept=KINDS[task][split][j]
    name=LABELS[split][k]
    rng=M.rng("cross-family-v5",split,task,index)
    floor=rng.randrange(32,75)
    observed=floor+([-(rng.randrange(7,20)), rng.randrange(7,20)][signal])
    cue_text,yes,no={
        "boundary":("threshold","The observed reading was above the allowed maximum.","The observed reading was below the allowed maximum."),
        "payment":("payment","The customer must pay a fee.","The customer owes zero for this use."),
        "assessment":("testing","It was judged using a measurement.","It was judged without taking a measurement."),
    }[task]
    if task=="boundary":
        if split=="frames":
            line=f"{name} read {observed} units for {concept}. That location accepts up to {floor} units."
        else:
            line=f"At {name}, the maximum allowed reading for {concept} is {floor} units. The instrument reads {observed} units now."
    elif task=="payment":
        first,second=("silver","green") if k%2 else ("upper","lower")
        charges=(first,second) if (k//2)%2==0 else (second,first)
        used=charges[1 if signal else 0]
        rate=rng.randrange(2,20)
        if split=="frames":
            line=f"For the {concept} at {name}, {charges[0]} costs nothing and {charges[1]} costs {rate} dollars. A customer used {used} there."
        else:
            line=f"A customer chose {used} for {concept} at {name}. The posted charge for {charges[1]} is {rate} dollars; choosing {charges[0]} costs nothing."
    else:
        reason=ASSESSMENT[split][j][signal]
        if split=="frames":
            line=f"The report on the {concept} from {name} used a single source for the verdict: {reason}."
        else:
            line=f"The inspector at {name} rated the {concept} by {reason}. There was no other basis for the rating."
    evidence=f"Unrelated glossary heading: {cue_text if cue else 'reference'}.\n{line}"
    options=[no,yes] if rng.randrange(2) else [yes,no]
    ending=f"Which conclusion follows from the case?\nA. {options[0]}\nB. {options[1]}"
    family=M.key("cross-family-v5",split,task,index)
    return dict(id=family+"".join(map(str,arm)),family=family,index=index,task=task,split=split,panel="candidate",
                factors=arm,truth=[signal],pieces=[evidence],ending=ending,options=options,answer=options.index(yes if signal else no),
                domain_cluster=concept,entity=name)


def freeze():
    import importlib.metadata
    assert not (R.OUT/"plan.json").exists()
    sources=[Path(p) for p in (__file__,C.__file__,C.C.__file__,R.__file__,M.__file__,str(Path(R.__file__).with_name("condition_chain_analysis.py")))]
    docs={}
    for split,n in C.C.SPLITS.items():
        rows=[make(split,task,i,arm) for task in M.CANDIDATES for i in range(n) for arm in product((0,1),repeat=2)]
        assert len({row["pieces"][0] for row in rows})==len(rows)
        name=f"materials-{split}-candidate.json"
        R.write(name,rows)
        docs[name]=dict(n=len(rows),sha256=R.file_sha(R.OUT/name))
    R.write("plan.json",dict(run_id=R.RUN,precision="BF16",thresholds=R.THRESHOLDS,candidates=M.CANDIDATES,materials=docs,
                            license="Original fictional examples",
                            rationale="V3 could reuse identical semantic prefixes. V5 fully new and guards against exact duplicate causal prefixes.",
                            primary="Same position, direction, family thresholds and lexical/embedding comparisons as V3; report semantic-domain cluster sensitivity",
                            sources={f.name:R.file_sha(f) for f in sources},
                            runtime={p:importlib.metadata.version(p) for p in ("torch","accelerate","numpy","transformers","scipy","safetensors","scikit-learn")}))


if __name__=="__main__":
    R.setup()
    C.C.render=C.render
    R.encode=C.C.encode
    R.prompts=C.render
    if sys.argv[1]=="freeze":
        freeze()
    elif sys.argv[1]=="run":
        C.run()
    elif sys.argv[1]=="analyze":
        from condition_chain_analysis import main
        main()
    elif sys.argv[1]=="intervene":
        R.intervene()
