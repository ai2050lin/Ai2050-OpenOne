"""Phase 2951: factorial natural continuations and original answer-query edits."""

from itertools import product
from pathlib import Path
import random
import sys
import time
from hashlib import sha256

import kv_state_run as Prev
import kv_natural_material as Material


A,B,P,N,O,ROOT=Prev.A,Prev.B,Prev.P,Prev.N,Prev.O,Prev.ROOT
OUT=ROOT/"tests/glm5/result/kv_orth_20261116_v1"
TMP=ROOT/"tests/glm5_temp/kv_orth_20261116_v1"
RUN="KV-ORTH-20261116-v1"
SEED=29512026
BUDGET=128*2**20
FAMILIES={"pilot":8,"discover":24,"confirm":96}
SLAB=128
MAX_REPLAY=48
PASSES=6
LAYERS=tuple(range(18,39))
ROLES=("gen","fact","book")
ANCHOR=[25,18,"gen"]
ENDPOINTS=("state-coef","fact-coef","distance-balance","h18-state","fact-fact","book-state","joint-code","h18-rescue","parallel-rescue")
CONFIDENCE=1-.05/len(ENDPOINTS)
WORDS="""Aaron Abel Adam Adrian Albert Alex Alexander Alfred Allan Alvin Amos Andre Andrew Andy Angelo Anthony Anton Archie Ari Arnold Arthur Ashton Austin Avery Barry Benjamin Bernard Bill Blake Bob Bobby Bradley Brandon Brett Brian Bruce Bryan Byron Caleb Calvin Carl Carlos Carter Casey Cecil Cedric Chad Charles Charlie Chester Chris Christopher Clark Claude Cliff Clyde Cody Colin Conrad Connor Cory Craig Curtis Cyrus Dale Damian Dan Daniel Darren Dave David Dean Dennis Derek Derrick Devin Dexter Diego Dominic Donald Douglas Drew Duncan Dustin Dylan Earl Edgar Edmund Edward Edwin Eli Elias Elijah Elliot Ellis Elmer Elton Emil Eric Erik Ernest Ethan Eugene Evan Felix Finn Floyd Francis Frank Franklin Fred Gabriel Garrett Gary Gene Geoffrey George Gerald Gilbert Glenn Gordon Graham Grant Gregory Gus Guy Harold Harry Harvey Hector Henry Herbert Herman Hugh Hugo Hunter Ian Isaac Ivan Jack Jacob Jake James Jamie Jared Jason Jasper Jay Jeff Jeffrey Jeremy Jerome Jerry Jesse Jim Joe Joel John Jonah Jonathan Jordan Jose Joseph Josh Joshua Juan Julian Justin Keith Kenneth Kevin Kyle Lance Larry Lawrence Lee Leo Leon Leonard Leroy Levi Liam Lionel Lloyd Logan Louis Lucas Luis Luke Malcolm Manuel Marc Marcus Mark Martin Marvin Mason Matthew Max Maxwell Melvin Michael Miguel Miles Milton Mitchell Morgan Nathan Neil Nicholas Noah Noel Norman Oliver Omar Oscar Owen Patrick Paul Perry Peter Philip Phillip Preston Quentin Ralph Ramon Randall Randy Ray Raymond Reed Reginald Rex Richard Rick Ricky Riley Robert Robin Rodney Roger Roland Ronald Ronnie Ross Roy Russell Ryan Sam Samuel Saul Scott Sean Seth Shane Shawn Sidney Simon Spencer Stan Stanley Stephen Steve Steven Stuart Taylor Ted Terence Terry Theo Theodore Thomas Tim Timothy Toby Todd Tom Tony Travis Trevor Troy Tyler Victor Vincent Wade Wallace Walter Warren Wayne Wesley Will William Wyatt Xavier Zachary Adam Ahmed Ali Amir Anil Arun Dev Ravi Raj Amit Ajay Dinesh Rakesh Vijay Arjun Rahul Omar Tariq Aidan Alistair Angus Arthur Archie Arthur Baxter Beckett Benson Blaine Brooks Carter Cole Cooper Finn Fraser Grayson Holden Hudson Hunter Jules Jude Miles Nolan Parker Reid Remy Rhys Roman Sawyer Silas Spencer Sterling Sutton Tanner Theo Tristan Warren Winston Mason Abbie Abigail Ada Adele Agnes Aileen Aisha Alana Alexa Alice Alicia Alison Alma Amanda Amber Amelia Amy Ana Andrea Angela Anita Anna Anne Annette Annie April Ariel Ashley Audrey Ava Barbara Beatrice Bella Beth Beverly Bianca Bonnie Brenda Brianna Brooke Camila Carla Carmen Carol Caroline Carolyn Casey Catherine Cathy Cecilia Celia Charlotte Cheryl Chloe Christine Cindy Claire Clara Claudia Connie Crystal Cynthia Daisy Dana Danielle Daphne Dawn Deborah Delia Denise Diana Diane Dolores Donna Dora Doris Dorothy Eden Edith Edna Eileen Elaine Eleanor Elena Eliza Elizabeth Ella Ellen Ellie Emily Emma Erica Erin Esther Eva Evelyn Faith Felicia Fiona Flora Frances Gabrielle Gail Gemma Georgia Gina Grace Hannah Hazel Heather Helen Holly Hope Ida Irene Iris Isabel Isabella Ivy Jacqueline Jade Jane Janet Janice Jean Jeanette Jenna Jennifer Jessica Jill Joan Joanna Jocelyn Josephine Joyce Judith Judy Julia Julie June Karen Katherine Kathleen Kathy Katie Kay Kelly Kelsey Kim Kirsten Lara Laura Lauren Leah Lena Leslie Lila Lillian Lily Linda Lindsay Lisa Lois Lola Loretta Louise Lucia Lucy Lydia Lynn Mabel Mackenzie Madison Maggie Margaret Maria Marian Marie Marilyn Martha Mary Maya Megan Melanie Melissa Mia Michelle Mildred Miranda Miriam Molly Monica Nancy Naomi Natalie Natasha Nellie Nicole Nina Nora Olivia Paige Pamela Patricia Paula Pearl Penelope Phoebe Priscilla Rachel Rebecca Renee Rita Roberta Rose Rosemary Ruth Sabrina Sally Samantha Sandra Sara Sarah Savannah Shannon Sharon Sheila Simone Sophia Sophie Stella Susan Suzanne Sylvia Tammy Teresa Tess Theresa Tiffany Tina Tracy Valerie Vanessa Vera Veronica Victoria Violet Vivian Wanda Wendy Whitney Wilma Yvonne Zoe Adele Harriet Matilda Winifred Imogen Phoebe Freya Isla Evie Maisie Poppy Millie Elsie Harriet Florence Ruby Alice Ivy Ada Bessie Camille Elaine Francine Isabelle Louise Violet Willa Zelda Serena Celeste"""


def setup():
    A.OUT,A.TEMP,A.RUN=OUT,TMP,RUN
    A.R.OUT,A.R.TEMP,A.R.RUN=OUT,TMP,RUN
    P.OUT,P.TMP,P.RUN,P.BUDGET=OUT,TMP,RUN,BUDGET
    P.check_write(OUT,BUDGET,operation=RUN)
    P.check_write(TMP,BUDGET,operation=RUN)
    P.setup()
    B.C.OUT=OUT


def pools(tok):
    used={w.casefold() for w in Prev.excluded()}
    for path in Prev.OUT.glob("material-*.json"):
        used.update(w.casefold() for w in N.words(P.load(path)))
    tokens={}
    for word in dict.fromkeys(WORDS.split()):
        ids=tok.encode(" "+word,add_special_tokens=False)
        if word.casefold() not in used and len(ids)==1 and tok.decode(ids)==" "+word:
            tokens[ids[0]]=word
    words=list(tokens.values()); random.Random(SEED).shuffle(words)
    assert len(words)>=125,len(words)
    return {"pilot":words[:25],"discover":words[25:75],"confirm":words[75:]}


def locate(tok,row):
    chat=tok.apply_chat_template([dict(role="user",content=row["prompt"])],tokenize=False,
                                 add_generation_prompt=True,enable_thinking=False)
    enc=tok(chat,add_special_tokens=False,return_offsets_mapping=True)
    base=chat.index(row["prompt"])
    text=row["prompt"]
    def spans(a,b):
        return [i for i,(l,r) in enumerate(enc.offset_mapping) if r>base+a and l<base+b and r>l]
    left=text.index("\n\nThe inventory")+2
    right=text.index("\nNote entry:")
    cb=text.index("\nCodebook:\n")+len("\nCodebook:\n")
    book=[]
    for word in row["words"]:
        a=text.index("\n"+word+" ->",cb-1)+1
        book.extend(spans(a,text.index("\n",a)))
    fact=spans(left,right)
    assert fact and book and len(set(fact+book))==len(fact+book)
    assert enc.input_ids==row["token"]["ids"][:row["token"]["start"]]
    assert max(fact+book)<row["token"]["start"]
    return {"fact":fact,"book":book}


def material(split,tok,words):
    rng=random.Random(f"{SEED}/{split}")
    out=[]; seen=set()
    for j in range(FAMILIES[split]):
        while True:
            vals=rng.sample(words,5)
            k=tuple(sorted(vals))
            if k not in seen:
                seen.add(k);break
        a,b,note,spare,archive=vals
        fid=sha256((split+"/"+"/".join(vals)).encode()).hexdigest()[:16]
        order=rng.sample(vals,5)
        fixed={note:"PQ"[j%2],spare:"PQ"[(j//2)%2],archive:"PQ"[(j//4)%2]}
        for dist,mp,state,fact in product((0,1),repeat=4):
            candidates=(a,b)
            table={a:"PQ"[mp],b:"PQ"[1-mp],**fixed}
            roles=("includes","excludes") if fact==0 else ("excludes","includes")
            receipt=Material.RECEIPTS[dist]
            layout=f"State: <word>\nNote: <word>\nSpare: <word>\nArchive: <word>\nReceipt: {receipt}\nCode: <P or Q>"
            prompt=("Complete the record below. State is the entry the inventory includes. Copy Note, Spare and Archive from their entries. "
                    "Look up State in the codebook to find its code.\n\n"
                    f"The inventory {roles[0]} {a}.\nThe inventory {roles[1]} {b}.\nNote entry: {note}\nSpare entry: {spare}\nArchive entry: {archive}\nCodebook:\n"+
                    "\n".join(f"{w} -> {table[w]}" for w in order)+
                    "\n\nReturn exactly these six plain-text lines in order. Replace the angle-bracket placeholders. "
                    "Copy the Receipt line exactly. Use the code letter by itself after Code:.\n"+layout)
            selected=candidates[state]
            row=dict(id=f"{fid}-{dist}{mp}{state}{fact}",family=fid,split=split,fact=fact,state=state,map=mp,distance=dist,words=list(candidates),
                     prompt=prompt,prefix=f"State: {selected}\nNote: {note}\nSpare: {spare}\nArchive: {archive}\nReceipt: {receipt}\nCode:",
                     values=dict(State=selected,Note=note,Spare=spare,Archive=archive),code_f=table[candidates[fact]],code_s=table[selected],
                     word_ids=tok.encode(" "+a,add_special_tokens=False)+tok.encode(" "+b,add_special_tokens=False))
            row["token"]=Material.token_plan(tok,row)
            assert row["token"]["ids"][row["token"]["slots"][0]]==row["word_ids"][state]
            row["sites"]=locate(tok,row)
            out.append(row)
    return out


def identity():
    model=B.model_identity("q14")
    assert model==P.load(Prev.OUT/"plan.json")["snapshot"]
    return model


def plan():
    assert not (OUT/"plan.json").exists() and A.load("pilot-gate.json")["passed"]
    tok=A.Base.tok(); by=pools(tok); specs={}
    for name in ("discover","confirm"):
        rows=material(name,tok,by[name]); path=f"material-{name}.json"
        A.save(path,rows)
        specs[name]=dict(file=path,sha=B.sha(path),families=FAMILIES[name],vocab=by[name])
    old=P.load(Prev.OUT/"plan.json")
    scripts={*old["sources"],"tests/glm5/kv_state_run.py"}
    scripts.update(q.relative_to(ROOT).as_posix() for q in Path(__file__).parent.glob("kv_orth_*.py"))
    A.save("plan.json",dict(run=RUN,stamp=O.stamp(),snapshot=identity(),runtime=A.Base.versions(),materials=specs,
        previous_freeze=A.digest(Prev.OUT/"freeze.json"),sources={k:A.digest(ROOT/k) for k in sorted(scripts)},
        pilot={k:B.sha(k) for k in ("material-pilot.json","pilot-natural.json","pilot-gate.json","pilot-physics.json")},
        disk=dict(primary_peak=BUDGET,reserve_gib=20,volume="D:"),
        materials_spec="Complete 2x2x2x2 fact role F, emitted State S, opposite codebook mapping M, short/long receipt D within each lexeme family, fixed State-first layout and one rule frame. F swaps includes/excludes, S is decided by native final-block E donor with actual full-vocabulary argmax, M independently swaps A/B mappings, D changes the copied receipt and its prompt instruction. Holds all noncandidate field words, their codebook entries, positions of words, unordered candidates and their codebook entry ordering within family. Held-out lexemes sampled from the same name pool after exact token filtering and related prior exclusions. Crossed design not pure distance without corresponding instruction change.",
        natural="Role-aligned cases naturally emit S, conflict cases copy only the final post-block L39 E residual from the role-swapped matched prompt with same M,D. The real full-vocabulary argmax following output normalizer/unembedding determines the actual token. After E, all greedy tokens including Code verified on causal prefix, drafts used only for speculative evaluation and unseen suffixes never counted. Up to six adaptive whitespace attempts using genuine model token; independently preserve raw failures and freely replay up to 48 cases, with max 80 emitted tokens and 12 passes. No unverified history qualifies for C. For both congruent and conflict rows, record P, Q, other, unresolved. Require >=90% valid State/canonical continuation and native P/Q per cell, congruent correct code >=90%, then whole technically complete 16-view families. For natural and causal statistics log whole-family coverage and examples.",
        search="At C screen layers 18..38, all 40 heads, native alpha*V for prompt fact and candidate-book sources. Project Wo writing toward Q-P, rank fact candidates by alignment with factual code, codebook candidates by alignment with emitted code, four directed and four physical dose heads per site. Replay those heads singly with generic sham on discovery; freeze fact head maximizing loss of beta_F and codebook head maximizing loss of beta_S, with different layer/head from H18. Same-layer/source two random heads sampled in nearest norm quartile by frozen seed; confirm each pair exactly equalizes actual Wo displacement to per-row min of native full doses. Anchor L25H18 reads G. No read of confirm before freeze.",
        interventions="At C only, derive exact one-head SDPA change after zeroing source V while live keys and masks are retained. Some arms use full source change, others scale target/control delta to equal realized original-Wo magnitude. Recompute later queries. Compare H18(G), parallel fact, parallel book, parallel joint and all; joint + native H18 write, joint + two native parallel writes, joint + all native writes, supplied from same initial sham positions. Additionally same-branch clear-and-restore of source Values must exactly reproduce sham. Freeze arms and loss/rescue contrasts. Preserve source attention, physical dose and first prefixed greedy choices, no evidence of Q/K attribution. Earlier state never affected by read intervention at C.",
        statistics=dict(confidence=CONFIDENCE,minimum_n=80,margin=.10,labels=ENDPOINTS,
            formula="For each family and distance fit Walsh coefficients beta_F=E[(Q-P)*sign(code_F)] and beta_S=E[(Q-P)*sign(code_S)] over the balanced eight F/S/M cells. On conflict, sign by actual emitted State versus opposing fact, count full-vocab P/Q and no answer. Congruent outcomes are no discrimination. Distance is the signed change of beta_S-beta_F, two-sided nonzero and abs>=.1. C primary equal-dose H18 loss in beta_S, parallel fact loss in beta_F, parallel book loss in beta_S, full joint deficit in original model's own verified P/Q advantage, and separate native H18/parallel rescue of that true advantage. Continuous paired family bootstrap across frozen nine endpoints. Both negative and positive effects, distances and role/letter cells preserved, full natural validation separate. Balanced W_O delivery gate p95 relative difference <=.03 per pair, denominator max(min intended dose,.02). Balanced unsupported if delivery fails."),
        engine="Packed vs full native baseline exactly agrees in actual Code top1 and raw P/Q <=1 logit. Threshold fixed in pilot. C only edits cannot affect earlier positions. Reject complete family on a sham mismatch. Save actual applied W_O norm and per-source dose; raw whole cosine/shape identity separate from output significance."))


def verify():
    p=A.load("plan.json")
    assert A.Base.versions()==p["runtime"] and A.digest(Prev.OUT/"freeze.json")==p["previous_freeze"]
    assert all(A.digest(ROOT/k)==v for k,v in p["sources"].items())
    assert all(B.sha(k)==v for k,v in p["pilot"].items())
    assert all(B.sha(v["file"])==v["sha"] for v in p["materials"].values())
    return p


def main(stage):
    setup()
    if stage=="plan":
        plan();return
    if stage=="audit":
        from kv_orth_audit import audit
        audit(verify());return
    if stage in ("preview","close"):
        from kv_orth_report import main as finish
        finish(stage,verify());return
    import torch
    from kv_orth_study import pilot,discover,confirm
    p=None if stage=="pilot" else verify()
    model=identity()
    if p is not None: assert model==p["snapshot"]
    torch.set_num_threads(4);torch.set_float32_matmul_precision("highest")
    with P.GpuLease(),torch.no_grad():
        t=time.monotonic();N.progress("orth-load",0,1,t)
        m=N.Engine();N.progress("orth-load",1,1,t)
        try:
            {"pilot":pilot,"discover":discover,"confirm":confirm}[stage](m,p)
        finally:
            m.close()


if __name__=="__main__":
    sys.modules["kv_orth_run"]=sys.modules[__name__]
    main(sys.argv[1])
