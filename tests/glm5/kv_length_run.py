"""Phase 2952: one-prompt receipt-length counterfactuals and native path testing."""

from hashlib import sha256
from itertools import product
from pathlib import Path
import random
import sys
import time

import kv_orth_run as Prev
import kv_natural_material as Material


A,B,P,N,O,ROOT=Prev.A,Prev.B,Prev.P,Prev.N,Prev.O,Prev.ROOT
OUT=ROOT/"tests/glm5/result/kv_length_20261116_v1"
TMP=ROOT/"tests/glm5_temp/kv_length_20261116_v1"
RUN="KV-LENGTH-20261116-v1"
SEED=29522026
BUDGET=108*2**20
SIZES={"pilot":8,"discover":24,"confirm":96}
SLAB=128
ARMS=((25,18,"gen"),(21,12,"fact"),(28,21,"book"))
SHORT,LONG=Material.RECEIPTS
ENDPOINTS=("distance-preference","address-state","value-state","fact-preference","book-preference","joint-margin","h18-rescue","parallel-rescue")
CONF=1-.05/len(ENDPOINTS)
WORDS="""Canada Mexico Brazil Argentina Chile Peru Bolivia Ecuador Colombia Venezuela Guyana Suriname Uruguay Paraguay Guyana Panama Belize Honduras Guatemala Nicaragua Cuba Jamaica Haiti Bahamas Grenada Dominica Tobago Barbados Aruba Curacao Bermuda Trinidad Alaska Hawaii Kansas Arkansas Alabama Tennessee Missouri Indiana Iowa Illinois Minnesota Georgia Alabama Texas Nevada Vermont Montana Oregon Dakota Wisconsin Wyoming Arizona Utah Kentucky Virginia Carolina Delaware Maryland Michigan Louisiana Mississippi California Nebraska Idaho Colorado Oklahoma America Massachusetts Connecticut Pennsylvania Rhode Alaska Ontario Quebec Alberta Manitoba Saskatchewan Nunavut Yukon Labrador Ireland Scotland Wales England France Spain Italy Germany Austria Hungary Switzerland Portugal Belgium Netherlands Luxembourg Poland Czech Slovakia Slovenia Croatia Serbia Bosnia Albania Kosovo Montenegro Macedonia Greece Bulgaria Romania Moldova Ukraine Russia Finland Sweden Norway Denmark Iceland Greenland Estonia Latvia Lithuania Cyprus Malta Turkey Lebanon Syria Jordan Israel Palestine Yemen Oman Bahrain Qatar Kuwait Iraq Iran Emirates Saudi Georgia Armenia Azerbaijan Uzbekistan Turkmenistan Tajikistan Kyrgyzstan Kazakhstan Mongolia China Korea Japan Taiwan Thailand Cambodia Laos Vietnam Malaysia Singapore Fiji Samoa Tonga Vanuatu Kiribati Nauru Australia Zealand Nepal Bhutan India Pakistan Bangladesh Afghanistan Sri Maldives Lanka Indonesia Brunei Timor Philippines Belize Chile Nigeria Ghana Kenya Uganda Tanzania Rwanda Burundi Sudan Eritrea Ethiopia Somalia Nigeria Niger Mali Senegal Gambia Benin Togo Chad Gabon Cameroon Angola Congo Guinea Liberia Sierra Ivory Morocco Algeria Tunisia Libya Egypt Botswana Zimbabwe Zambia Malawi Mozambique Namibia Lesotho Swaziland Madagascar Mauritius Seychelles Comoros Djibouti Somalia Patagonia Sinai Sicily Sardinia Corsica Crete Rhodes Andorra Monaco Liechtenstein Monaco Vatican Catalonia Bavaria Saxony Tuscany Brittany Normandy Hebrides Cornwall Cornwall Kent Devon Sussex Surrey Essex Yorkshire Cheshire Lancashire Warwickshire Dorset Somerset Norfolk Suffolk Nottingham Durham Stafford Bedford Hampshire Berkshire Cumbria Whistler Kelowna Surrey Regina Kingston Hamilton Windsor Kingston Halifax Bridgetown Nassau Milledgeville Fayetteville Pensacola Tallahassee Savannah Augusta Charleston Montgomery Birmingham Huntsville Knoxville Memphis Jackson Nashville Chattanooga Louisville Springfield Madison Green Bay Racine Rockford Davenport Des Moines Omaha Wichita Topeka Lawrence Tulsa Stillwater Norman Edmond Tulsa Amarillo Lubbock Midland Abilene Waco Denton Plano Irving Frisco Arlington McAllen Beaumont Galveston Lafayette Shreveport Monroe Biloxi Gulfport Mobile Raleigh Durham Greensboro Asheville Wilmington Greensboro Greensboro Greensboro Greensboro Wilmington Norfolk Chesapeake Roanoke Lynchburg Fairfax Arlington Bethesda Annapolis Rockville Gettysburg Harrisburg Erie Scranton Allentown Trenton Camden Newark Paterson Princeton Hoboken Secaucus Yonkers Syracuse Utica Ithaca Binghamton Schenectady Poughkeepsie Monterey Pasadena Oakland Berkeley Fresno Modesto Stockton Bakersfield Salinas Santa Monica Torrance Riverside Anaheim Ventura Sonoma Napa Tahoe Fresno Eureka Redding Reno Carson Tucson Flagstaff Mesa Gilbert Yuma Surprise Scottsdale Peoria Tahoe Ontario Riverside Fremont Fairfield Davis Chico Lodi Merced Hanford Madera Tulare Burbank Oxnard Fullerton Temecula Cajon Escondido Salem Eugene Medford Corvallis Bend Spokane Tacoma Olympia Yakima Wenatchee Bellingham Redmond Kent Everett Mukilteo Bellevue Spokane Boise Nampa Pocatello Billings Helena Missoula Bozeman Casper Cheyenne Laramie Rapid Fargo Bismarck Duluth Moorhead Fargo Winnipeg Saskatoon Kelowna Abbotsford Brandon Guelph Barrie Brampton Sudbury Oakville Sudbury Timmins Windsor Waterloo Saskatoon Edmonton Yellowknife Moncton Fredericton Halifax Antigonish Charlottetown Hamilton Edmonton Casablanca Rabat Tunis Sfax Algiers Oran Alexandria Luxor Aswan Giza Asyut Suez Fez Marrakesh Tangier Nairobi Mombasa Nakuru Kampala Kigali Bujumbura Addis Harare Lusaka Ndola Maputo Lilongwe Blantyre Bulawayo Gaborone Windhoek Swakopmund Pretoria Johannesburg Durban Kimberley Bloemfontein Soweto Lesotho Luanda Lubango Windhoek Mopti Gao Lagos Abuja Kano Ibadan Sokoto Kaduna Accra Kumasi Tema Dakar Thiès Abidjan Bouaké Bamako Kankan Conakry Freetown Monrovia Nouakchott Ouarzazate Vilnius Kotor Tartu Kaunas Narva Gdansk Krakow Katowice Wroclaw Poznan Lublin Lodz Kaunas Nicosia Limassol Larnaca Valletta Heraklion Thessaloniki Patras Sparta Corinth Kalamata Belgrade Novi Mostar Split Dubrovnik Rijeka Pula Zadar Maribor Koper Innsbruck Salzburg Graz Linz Bregenz Klagenfurt Trento Bolzano Bari Palermo Catania Cagliari Siena Pisa Lucca Perugia Assisi Rimini Modena Parma Cremona Brescia Bergamo Como Trieste Padua Verona Vicenza Ferrara Ravenna Ancona Urbino Lecce Reggio Messina Siracusa Ragusa Latina Brindisi Gallipoli Vicenza Salerno Sorrento Amalfi Pompeii Herculaneum Capua Caserta Foggia Pescara Molise Abruzzo Puglia La Spezia Livorno Grosseto Macerata Jesi Tolentino Urbino Ferrara Rovigo Udine Gorizia Pordenone Chioggia Treviso Belluno Vercelli Novara Biella Aosta Cuneo Torino Alessandria Asti Piacenza Savona Imperia Sanremo"""


def setup():
    A.OUT,A.TEMP,A.RUN=OUT,TMP,RUN
    A.R.OUT,A.R.TEMP,A.R.RUN=OUT,TMP,RUN
    P.OUT,P.TMP,P.RUN,P.BUDGET=OUT,TMP,RUN,BUDGET
    P.check_write(OUT,BUDGET,operation=RUN);P.check_write(TMP,BUDGET,operation=RUN);P.setup();B.C.OUT=OUT


def pools(tok):
    excluded={k.casefold() for k in Prev.Prev.excluded()}
    for stage in (Prev.Prev.OUT,Prev.OUT):
        for f in stage.glob("material-*.json"):
            for r in P.load(f):
                excluded.update(v.casefold() for v in (*r["values"].values(),*r.get("words",()),r.get("rejected","")))
    items={}
    for word in dict.fromkeys(WORDS.split()):
        ids=tok.encode(" "+word,add_special_tokens=False)
        if word.casefold() not in excluded and len(ids)==1 and tok.decode(ids)==" "+word:
            items[ids[0]]=word
    words=list(items.values());random.Random(SEED).shuffle(words)
    assert len(words)>=125,len(words)
    return {"pilot":words[:24],"discover":words[24:72],"confirm":words[72:]}


def receipt_anchor(row,plan=None):
    t=row["token"] if plan is None else plan
    ids=t["ids"]; word=row["anchor_ids"]
    start=t["slots"][3]+1
    found=[j+len(word)-1 for j in range(start,len(ids)-len(word)) if ids[j:j+len(word)]==word]
    assert len(found)==1,(row["id"],found)
    return found[0]


def prompt_for(row,fixed=None):
    original=row["prompt_head"]
    if fixed is not None:
        return original+"\n\nReturn exactly these six plain-text lines in order. Copy the Receipt line literally. Use the code letter by itself after Code:.\n"+row["layout"].format(receipt=(SHORT,LONG)[fixed])
    options=f"Either of these two receipt lines is acceptable. Choose one and copy it literally.\nReceipt: {SHORT}\nReceipt: {LONG}"
    rule="Return exactly six plain-text lines in order. Use the code letter by itself after Code:. The receipt choice leaves the State lookup rule unchanged."
    return original+"\n\n"+options+"\n\n"+rule+"\n"+row["layout"].format(receipt="<one receipt line above>")


def material(stage,tok,words):
    rng=random.Random(f"{SEED}/{stage}"); out=[]; seen=set()
    for j in range(SIZES[stage]):
        while True:
            a,b,note,spare,archive=rng.sample(words,5)
            key=tuple(sorted((a,b,note,spare,archive)))
            if key not in seen:seen.add(key);break
        order=rng.sample([a,b,note,spare,archive],5)
        fid=sha256((stage+"/"+"/".join((a,b,note,spare,archive))).encode()).hexdigest()[:16]
        for length,m,s,f in product((0,1),repeat=4):
            words0=[a,b];roles=("includes","excludes") if f==0 else ("excludes","includes")
            answer={a:"PQ"[m],b:"PQ"[1-m],note:"PQ"[j%2],spare:"PQ"[(j//2)%2],archive:"PQ"[(j//4)%2]}
            head=("Complete the record below. State is the entry the inventory includes. Copy Note, Spare and Archive from their entries. "
                  "Look up State in the codebook to find its code.\n\n"
                  f"The inventory {roles[0]} {a}.\nThe inventory {roles[1]} {b}.\nNote entry: {note}\nSpare entry: {spare}\nArchive entry: {archive}\nCodebook:\n"+
                  "\n".join(f"{w} -> {answer[w]}" for w in order))
            layout="State: <word>\nNote: <word>\nSpare: <word>\nArchive: <word>\nReceipt: {receipt}\nCode: <P or Q>"
            item=dict(id=f"{fid}-{length}{m}{s}{f}",family=fid,phase=stage,distance=length,map=m,state=s,fact=f,
                      words=words0,word_ids=tok.encode(" "+a,add_special_tokens=False)+tok.encode(" "+b,add_special_tokens=False),
                      code_s=answer[words0[s]],code_f=answer[words0[f]],values=dict(State=words0[s],Note=note,Spare=spare,Archive=archive),
                      prompt_head=head,layout=layout,prefix=f"State: {words0[s]}\nNote: {note}\nSpare: {spare}\nArchive: {archive}\nReceipt: {(SHORT,LONG)[length]}\nCode:",
                      anchor_ids=tok.encode(" "+SHORT,add_special_tokens=False))
            item["prompt"]=prompt_for(item)
            item["token"]=Material.token_plan(tok,item)
            item["sites"]=Prev.locate(tok,item)
            bnd=receipt_anchor(item)
            assert item["token"]["slots"][0]<bnd<len(item["token"]["ids"])-1
            assert item["token"]["ids"][item["token"]["slots"][0]]==item["word_ids"][s]
            out.append(item)
    return out


def identity():
    x=B.model_identity("q14")
    assert x==P.load(Prev.OUT/"plan.json")["snapshot"]
    return x


def plan():
    assert not (OUT/"plan.json").exists() and A.load("pilot-check.json")["passed"]
    tok=A.Base.tok();by=pools(tok);spec={}
    for key in ("discover","confirm"):
        rows=material(key,tok,by[key]); name=f"material-{key}.json"
        A.save(name,rows);spec[key]=dict(file=name,sha=B.sha(name),words=by[key],rows=len(rows))
    old=P.load(Prev.OUT/"plan.json")
    sources={*old["sources"],"tests/glm5/kv_orth_run.py"}
    sources.update(k.relative_to(ROOT).as_posix() for k in Path(__file__).parent.glob("kv_length_*.py"))
    A.save("plan.json",dict(run=RUN,time=O.stamp(),snapshot=identity(),runtime=A.Base.versions(),sources={k:A.digest(ROOT/k) for k in sorted(sources)},
        panels=spec,previous_freeze=A.digest(Prev.OUT/"freeze.json"),budget=dict(bytes=BUDGET,volume="D:",reserve_gib=20),
        pilot={k:B.sha(k) for k in ("material-pilot.json","neutral-donors.json","pilot-natural.json","pilot-natural-gate.json","pilot-edits.npz","pilot-check.json")},
        design="Each new lexical family fully crosses S, F, M, D. Paired length conditions literally share one prompt, both legal Receipt texts and same emitted State, unrelated fields, final Code marker. Long differs by inserting a fixed neutral continuation after the short sentence inside Receipt. Prompt role and code mapping switch independently. E terminal role-matched native donor determines genuine full-vocab State in conflict. At common Receipt boundary B the only edit applies a frozen pilot neutral native BF16 donor residual to either emit short newline or first extension token, via original normalizer and unembedding. No model logit masks, no additional forced tokens. Every subsequent token, even footers and Code, must be greedily chosen. Adapt only actually emitted permissible whitespace, recheck exact token prefix and shared suffix in each paired D, save all counterexamples. Unverified output is not counted as natural P/Q.",
        reads="Frozen model sites L25H18 generated G, L21H12 two fact lines, L28H21 candidate codebook lines (Phase 2951). At C for H18 compute address-only support mask of G and intact other K,V, with native Q; separately remove only q-dot-k_G component by original BF16 Key projection at that slot, and orthogonal query projection relative to k_G with spillover. V-slot zero preserves all Q/K attention probabilities and changes only V/O writing. K-only, Q-only, address-only, V-only plus K+V pair, native K / V and joint restore in the pair. Exact same-branch restoration identities for originals. Both parallel C readers target fact and book, plus a joint C block H18 V and both parallel paths. Restore H18 full native output or both parallel heads independently, and all. For address/Value and each parallel route choose two same-layer random heads from closest discovery norm quartile, collect per-view target and control doses by original Wo, deliver both at the smaller native displacement. Match p95 relative <=3%, denominator max(common,.02). Clear and restore normal native clones, not new prompted labels. Full-vocabulary Code changes and signs independently tracked.",
        gate=dict(confidence=CONF,n=80,margin=.10,labels=ENDPOINTS,behavior=.90,p95=.03,
                  decision="All 16 cells >=90% verified canonical greedy P/Q, congruent correctness >=90%, all 16 per complete family. Directional registered tests: in conflict long-short sign(P/Q toward emitted) >0; equal-dose address and Value expected losses of emitted-code Walsh coefficient; equal-dose fact on factual coefficient, and book on emitted-code coefficient; all reads joint full native margin loss and both partial restorations. Average by lexeme family, bootstrap simultaneous comparisons, require mean>=.10, adjusted interval lower>0 and n>=80. Prespecified technical native baseline <=1 logit and full-vocab top agreement. Separate descriptive raw alpha/score, position, opposite-case, absolute path effects and K/Q true changes.")))


def verify():
    p=A.load("plan.json")
    assert p["runtime"]==A.Base.versions() and p["previous_freeze"]==A.digest(Prev.OUT/"freeze.json")
    assert all(A.digest(ROOT/k)==v for k,v in p["sources"].items())
    assert all(B.sha(k)==v for k,v in p["pilot"].items())
    assert all(B.sha(v["file"])==v["sha"] for v in p["panels"].values())
    return p


def main(mode):
    setup()
    if mode=="plan":plan();return
    if mode=="audit":
        from kv_length_audit import audit
        audit(verify());return
    if mode in ("preview","close"):
        from kv_length_report import main as end
        end(mode,verify());return
    import torch
    from kv_length_study import pilot,discover,confirm
    p=None if mode=="pilot" else verify()
    x=identity()
    if p is not None:assert p["snapshot"]==x
    torch.set_num_threads(4);torch.set_float32_matmul_precision("highest")
    with P.GpuLease(),torch.no_grad():
        t=time.monotonic();N.progress("length-load",0,1,t);m=N.Engine();N.progress("length-load",1,1,t)
        try:{"pilot":pilot,"discover":discover,"confirm":confirm}[mode](m,p)
        finally:m.close()


if __name__=="__main__":
    sys.modules["kv_length_run"]=sys.modules[__name__]
    main(sys.argv[1])
