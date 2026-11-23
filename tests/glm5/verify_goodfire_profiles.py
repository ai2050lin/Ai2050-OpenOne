"""Independently spot-check displayed native unit values and visible texts against the original archive."""
import base64
import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from tokenizers import Tokenizer

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from deploy.storage_policy import checked_dir

PACK=ROOT/"tests/glm5/result/qwen14_client_v2/native"
EXCERPT=ROOT/"research/glm5/docs/previews/client-review/neuron-profiles.json"

def read(path):
    return json.loads(path.read_bytes())

def unpack(src,dtype):
    return np.frombuffer(base64.b64decode(src,validate=True),dtype)

def check():
    rendered=read(EXCERPT)
    docs=read(PACK/"documents.json")
    tokenizer=Tokenizer.from_file(str(PACK/"tokenizer.json"))
    q14=read(PACK/"manifest.json")
    assert rendered["run"]==q14["run_id"] and rendered["snapshot"]==q14["model_snapshot"]
    assert sorted(map(int,rendered["layers"]))==q14["blocks"]
    tested=0
    example={}
    for block in q14["blocks"]:
        preview=rendered["layers"][str(block)]
        profile=read(PACK/f"B{block}-profile.json")
        assert preview["run"]==profile["run_id"]
        original=profile["fields"]["A"]
        cues=read(PACK/f"B{block}-cues.json")["units"]
        activations=np.load(PACK/f"B{block}-A.npy",mmap_mode="r")
        assert activations.dtype==np.uint16
        for unit in (0,1,2429,(block*401)%q14["width"],q14["width"]-1):
            for which in ("mean","std"):
                assert unpack(preview[which],"<f8")[unit]==original[which][unit]
            words=[rendered["words"][x] for x in unpack(preview["terms"],"<u4")[unit*3:unit*3+3] if x]
            assert words==cues[unit]["positive_keywords"][:3]
            hits=int(unpack(preview["repeat"],"u1")[unit])
            assert [hits>>4,hits&15]==[cues[unit]["repeat"][s]["hits"] for s in ("confirm","holdout")]
            retained=[]
            for side,(indices,values) in zip(("positive_indices","negative_indices"),preview["poles"]):
                index=int(unpack(indices,"<u2")[unit])
                assert index==original["holdout"][side][unit][0]
                di,slot=divmod(index,len(docs["positions"]))
                pos=docs["positions"][slot]
                doc=docs["docs"][di]
                assert doc["split"]=="holdout"
                record=rendered["docs"][str(di)]
                assert all(record[k]==doc[k] for k in ("doc_id","repo","source","revision","license"))
                # Decode the full visible scope independently and check the excerpt contains only its tail.
                full=tokenizer.decode(doc["tokens"][:pos+1],skip_special_tokens=False)
                actual=rendered["examples"][str(index)]
                assert actual[:3]==[di,pos,doc["tokens"][pos]]
                short=tokenizer.decode(doc["tokens"][max(0,pos-31):pos+1],skip_special_tokens=False)
                assert actual[3]==short
                assert full.endswith(short.lstrip("\ufffd")), (block,unit,doc["doc_id"],pos)
                native=int(activations[di,slot,unit])
                assert int(unpack(values,"<u2")[unit])==native
                v=np.array([native<<16],dtype=np.uint32).view(np.float32)[0]
                retained.append({"doc":doc["doc_id"],"position":pos,"value":float(v)})
            if (block,unit)==(36,2429):
                example={"object":"B36/U2429","model":rendered["snapshot"],
                    "mean":original["mean"][unit],"std":original["std"][unit],
                    "discoveryKeywords":words,"confirmHits":hits>>4,"holdoutHits":hits&15,"examples":retained}
            tested+=1
    probes=[]
    for unit,failed in [(9517,"S8_read_write"),(80,"S9_pairs")]:
        file=ROOT/f"research/neurons/goodfire/GF-L30-U{unit}-local.json"
        r=read(file)
        assert r["params"]["layer"]==30 and r["params"]["unit"]==unit
        assert r["schema"]=="goodfire_probe.v1" and r["verdict"]["failed"]==[failed]
        assert all(abs(p["z_a"]-p["z_b"]-p["dz"])<1e-9 for p in r["S9"]["pairs"])
        probes.append({"unit":unit,"sha256":hashlib.sha256(file.read_bytes()).hexdigest(),
            "pairCount":len(r["S9"]["pairs"]),"failed":r["verdict"]["failed"]})
    return {"passed":True,"scope":"offline presentation export; no model execution",
        "utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "exportSha256":hashlib.sha256(EXCERPT.read_bytes()).hexdigest(),
        "indexed14B":q14["width"]*len(q14["blocks"]),"sampled14B":tested,
        "signedValuesAndTokenExcerptsChecked":tested*2,"example":example,"qwen4bReports":probes}

if __name__=="__main__":
    result=check()
    folder=checked_dir(ROOT/"tests/glm5/result/goodfire_feature_preview",1024*1024,operation="保存离线读数核验")
    dest=folder/"source-check.json"
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(f"Matched {result['sampled14B']} units and {result['signedValuesAndTokenExcerptsChecked']} held-out positions; {dest}")
