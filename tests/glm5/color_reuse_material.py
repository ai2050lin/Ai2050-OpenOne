"""Fixed Qwen-14B material for color naming, comparison and natural controls."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import random
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "tests/glm5"
sys.path[:0] = [str(HERE), str(ROOT)]
from clt_role_reread_materials import FIELDS, MODEL
from deploy.storage_policy import checked_dir
from deploy.test_archive_client import runtime_identity, save
from shared.dist_protocol.test_archive import CORE_ROLES, digest, validate_plan
from source_binding_design import read, sha

OUT = HERE / "result/color_reuse_v1"
ENTRY = HERE / "color_reuse_test.py"
SOURCE = OUT / "en_ewt-ud-train.conllu"
SEED = 2026112001
LAYERS = (12, 20, 28)
SIZES = (32, 128, 512)
COLORS = ("red", "yellow", "blue")
# Each list is one split. Discovery and calibration use different nouns.
# The mineral and sports/safety categories below are withheld from fitting.
GROUPS = {
    "discovery": [
        ("red","food","a ripe raspberry|a ripe cranberry|a ripe cherry|ripe watermelon flesh|a pomegranate seed|a ripe lingonberry"),
        ("yellow","food","a ripe pineapple's flesh|a ripe quince|ripe sweetcorn kernels|a ripe starfruit|a yolk from a hen's egg|a ripe canary melon"),
        ("blue","appearance","a cloudless midday sky|a cornflower blossom|a clear tropical sea|a forget-me-not flower|a morning glory of the Heavenly Blue cultivar|a lapis lazuli pigment sample"),
        ("red","appearance","fresh arterial blood|a scarlet ibis|a cardinal bird|a poppy petal"),
        ("yellow","appearance","a sunflower petal|a dandelion flower|a buttercup flower|a canary bird"),
        ("blue","appearance","a male indigo bunting|a morpho butterfly wing|a hyacinth macaw|a mountain bluebird"),
    ],
    "calibration": [
        ("red","food","a ripe Roma tomato|a ripe acerola cherry|ripe goji berries|ripe chili pepper skin"),
        ("yellow","food","a ripe plantain peel|a pat of butter|a slice of cheddar cheese|a spaghetti squash"),
        ("blue","appearance","a delphinium blossom|a male peacock's neck|a clear swimming pool|a gentian flower"),
    ],
    "confirm": [
        ("red","food","a ripe strawberry|a ripe redcurrant|a ripe lychee peel|ripe bell pepper of the pimiento variety"),
        ("yellow","food","a ripe lemon peel|a ripe banana peel|a wedge of Edam cheese without its rind|a wedge of preserved lemon"),
        ("blue","appearance","a jay of the Eurasian roller species|a great blue heron in bright daylight|an Eastern bluebird's back|a Himalayan poppy blossom"),
        ("red","mineral","a ruby gemstone|a cinnabar crystal|a garnet of the pyrope variety|a cuprite crystal"),
        ("yellow","mineral","native sulfur crystals|a citrine gemstone|a piece of orpiment|a lemon quartz crystal"),
        ("blue","mineral","a sapphire of the Kashmir variety|an azurite crystal|a tanzanite gemstone|a piece of sodalite"),
        ("red","signal","a regulation stop sign in the United States|the light that orders cars to stop|the card used to expel a soccer player|the bullseye of a standard competition dartboard"),
        ("yellow","signal","a US school bus exterior|the caution light on a US traffic signal|the card used to caution a soccer player|a penalty flag in American football"),
        ("blue","signal","the field of the European Union flag|the square on an accessible parking sign in the United States|the background of a United Nations flag|the field behind the stars of the United States flag"),
    ],
}
REFS = {
    "red": ("a ripe beefsteak tomato", "a fire engine in its traditional livery", "a ripe sweet cherry"),
    "yellow": ("a ripe Cavendish banana", "a classic rubber duck", "a golden daffodil petal"),
    "blue": ("a cloudless summer sky at noon", "a piece of lapis lazuli", "a cobalt glass bottle"),
}
TAIL = {
    "turquoise": ("the shallow lagoon of a tropical atoll","the classic Tiffany jewelry box","a polished amazonite stone"),
    "ochre": ("a sample of traditional yellow iron oxide earth pigment","the earth pigment made from limonite","a piece of untreated raw sienna pigment"),
}
TAIL_REF = {"turquoise":"the Caribbean shallows on a sunny day", "ochre":"a swatch of limonite earth paint"}
TAIL_CHOICES = (*COLORS,"green","brown","orange","white","black","turquoise","ochre")
CRITERIA = """Discovery on first-item final native token, never seeing the reference or answer.
Block 12/20/28 zero-based and top 32/128/512 MLP units ranked by standardized red-v-yellow naming activation.
Freeze ranking on discovery, choose block/size on disjoint calibration naming only by red-to-yellow logit shift
above matched unrelated control, tie by smaller size then earlier block. Report all panels regardless of gate.
Naming calibration gate: red-v-yellow score drops >=0.25 logits and target exceeds control by >=0.15
on red items, yellow gains more than blue by >=0.10. Confirm causal claim needs naming and same/different
forced-choice >=80% separately for each core color and each of nine color pair cells. Confirm requires target
minus matched-control shift in Same-Different <=-0.10 for red references and >=0.10 for yellow references
on red first items, their entity-cluster bootstrap 95% intervals both exclude zero; yellow shift exceeds blue
by >=0.10 with positive lower bound. Require mean absolute food-control shift <=0.20, all restored choice
scores identical to paired sham, nonzero and norm match within 5% on >=95% of edited rows.
Report whole-vocabulary greedy results, all items and subset for which model knows both colors, heldout
categories, label-copy clues, lexical/cooccurrence and constant baselines, all counterexamples. Exact
first-item duplicates across pairs are repeated measures. Natural EWT frequencies/co-mentions are only
proxies; sample natural occurrences blind to outcomes with matched length/position background. Tail
panels are descriptive, not independent evidence of necessity or of training-corpus frequencies."""


def sources():
    files = [ENTRY, HERE/"color_reuse_material.py", HERE/"source_binding_runtime.py",
             HERE/"source_binding_resources.py", HERE/"source_binding_design.py",
             HERE/"clt_role_reread_materials.py", HERE/"run_routine_test.py",
             ROOT/"shared/dist_protocol/natural_register_runner.py", ROOT/"shared/dist_protocol/mechanism_trace.py"]
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}


def natural():
    sentences = [s[9:] for s in SOURCE.read_text(encoding="utf-8").splitlines() if s.startswith("# text = ")]
    terms = set(COLORS) | set(TAIL)
    words = [re.findall(r"[a-z]+", s.lower()) for s in sentences]
    counts = Counter(w for row in words for w in row)
    rng = random.Random(SEED)
    rows = []
    for color in sorted(terms):
        hits = [i for i,w in enumerate(words) if color in w and 4 <= len(w) <= 55]
        rng.shuffle(hits)
        for i in hits[:20]:
            match = re.search(r"\b"+color+r"\b",sentences[i],re.I)
            if not match:
                continue
            text = sentences[i][:match.end()]
            pos = len(re.findall(r"[a-z]+",text.lower()))
            pool = [j for j,w in enumerate(words) if not terms.intersection(w) and len(w)>=pos
                    and abs(len(w)-len(words[i]))<=2]
            if not pool:
                continue
            j = rng.choice(pool)
            end = list(re.finditer(r"[a-z]+",sentences[j],re.I))[pos-1].end()
            rows.extend([(color,"colorword",i,text),(color,"background",j,sentences[j][:end])])
    return rows,dict(source="UD_English-EWT r2.15 train", url="https://github.com/UniversalDependencies/UD_English-EWT/tree/r2.15",
                     sha256=sha(SOURCE), sentences=len(words), word_tokens=sum(counts.values()),
                     occurrences={w:counts[w] for w in sorted(terms)},counts_scope="case-insensitive alphabetic word tokens")


def prepare():
    from transformers import AutoTokenizer
    checked_dir(OUT, 64*2**20, operation="冻结颜色复用材料")
    if (OUT/"study-plan.json").exists():
        raise FileExistsError("Frozen plan already exists")
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    rows=[]
    def add(stage, task, item, color, category, suffix, answers, gold, **metadata):
        body = "Consider the usual visible surface color of each item. First item: " + item
        prompt = tok.apply_chat_template([{"role":"user","content":body+"."+suffix}], tokenize=False,
                     add_generation_prompt=True,enable_thinking=False)+"Answer:\n"
        enc = tok(prompt, add_special_tokens=False,return_offsets_mapping=True)
        end = prompt.index(body)+len(body)
        p = next(i for i,(a,b) in enumerate(enc.offset_mapping) if a<end<=b)
        choices = [tok.encode(a,add_special_tokens=False) for a in answers]
        if any(tok.encode(prompt+a,add_special_tokens=False)!=enc.input_ids+v for a,v in zip(answers,choices)):
            raise ValueError("unstable answer tokenization")
        row = dict(id=len(rows),stage=stage,task=task,item=item,color=color,category=category,prompt=prompt,
                   tokens=enc.input_ids,pos=p,answers=list(answers),choices=choices,gold=gold,**metadata)
        rows.append(row)
        return row
    for stage, groups in GROUPS.items():
        for color,category,names in groups:
            for name in names.split("|"):
                add(stage,"name",name,color,category," What is its usual color? Answer with one lowercase color word.",COLORS,color)
                if stage=="confirm":
                    add(stage,"food",name,color,category," Is the first item normally eaten as food? Answer yes or no.",("yes","no"),"yes" if category=="food" else "no")
                    for refcolor,refs in REFS.items():
                        for ref in refs:
                            add(stage,"compare",name,color,category," Second item: "+ref+". Are their typical colors the same or different? Answer same or different.",
                                ("same","different"),"same" if color==refcolor else "different",ref=ref,refcolor=refcolor)
    for color,refs in REFS.items():
        for ref in refs:
            add("reference","name",ref,color,"reference"," What is its usual color? Answer with one lowercase color word.",COLORS,color)
    for color,objects in TAIL.items():
        for item in (*objects,TAIL_REF[color]):
            add("tail","tail_name",item,color,"rare_name"," Which color term best describes its appearance? Answer with one lowercase color word.",TAIL_CHOICES,color)
        for item in objects:
            for rc,ref in ([(k,v) for k,v in TAIL_REF.items()]+[(k,v[0]) for k,v in REFS.items()]):
                add("tail","compare",item,color,"rare_name"," Second item: "+ref+". Are their typical colors the same or different? Answer same or different.",
                    ("same","different"),"same" if color==rc else "different",ref=ref,refcolor=rc)
    for base,typical in (("a model strawberry","red"),("a model banana","yellow"),("a model sapphire","blue")):
        for color,refs in REFS.items():
            item=base+" painted to match "+refs[1]
            cat="usual_pair" if typical==color else "unusual_pair"
            add("tail","name",item,color,cat," What color is the painted model now? Answer with one lowercase color word.",COLORS,color)
            for rc,others in REFS.items():
                add("tail","compare",item,color,cat," Second item: "+others[2]+". Do their current surface colors match? Answer same or different.",
                    ("same","different"),"same" if color==rc else "different",ref=others[2],refcolor=rc)
    corpus_rows,corpus=natural()
    for color,category,idx,text in corpus_rows:
        ids=tok.encode(text,add_special_tokens=False)
        rows.append(dict(id=len(rows),stage="corpus",task="observe",item=text,color=color,category=category,source_sentence=idx,
                         prompt=text,tokens=ids,pos=len(ids)-1,answers=[],choices=[],gold=None))
    # Frozen co-occurrence baseline (word overlap of final content nouns is only an approximation).
    documents=defaultdict(set)
    for i,line in enumerate(SOURCE.read_text(encoding="utf-8").splitlines()):
        if line.startswith("# text = "):
            for term in set(re.findall("[a-z]+",line[9:].lower())):
                documents[term].add(i)
    for r in rows:
        if r["task"]=="compare":
            a,b=(re.findall("[a-z]+",n.lower())[-1] for n in (r['item'],r['ref']))
            r["corpus_pair_count"]=len(documents[a] & documents[b])
    save(OUT/"study-questions.json",dict(rows=rows,corpus=corpus))
    qsha=sha(OUT/"study-questions.json"); code=digest(sources())
    identity=read(HERE/"result/clt_role_closure_v1/diagnostic-plan.json")["acquisition"]
    tsha=digest({k:sha(MODEL/k) for k in FIELDS if (MODEL/k).exists()})
    if tsha!=identity["tokenizer_sha256"]: raise ValueError("tokenizer identity changed")
    plan=dict(schema="research_test_plan.v1",kind="real", title="Qwen3-14B 跨物体颜色命名与同异比较：预报红黄混淆",
        objective="检验颜色命名中选择出的原生 MLP 单元能否在未见物体的颜色比较中预测定向错答。",
        principle="首次物体出现后、参照和答案出现前定位门控单元，按参照色检验同一处阻断的相反作用和同输入恢复。",
        hypothesis="红相对黄证据减弱对红参照降低同答分，对黄参照提高同答分，对蓝参照不产生同等混淆。",
        alternatives="共同输出词，物体共现，一般信心降低，替代读取，分布式或跨位置编码。",
        criteria=CRITERIA,domains=["KN","RS"],family_ids=["COLOR-NAMING-COMPARISON"],entrypoint_sha256=sha(ENTRY),execution_args=[],
        dataset=dict(id="color-reuse/naming-to-comparison",revision="v1",questions_file_sha256=qsha,content_sha256=qsha,
            split_manifest_sha256=digest(GROUPS),case_count=len(rows),exposure="Fixed author-generated object panels, public UD EWT. References distinct strings, overlap of biological kinds explicitly reported; no model results consulted for selection."),
        acquisition=dict(model_id="Qwen3-14B",model_snapshot=identity["model_snapshot"],tokenizer_sha256=tsha,
            entrypoint_sha256=sha(ENTRY),questions_file_sha256=qsha,inputs_sha256=digest([r["tokens"] for r in rows]),
            code_sha256=code,environment_sha256=digest(runtime_identity()),precision="BF16 original weights; FP32 contrasts",
            seed=SEED,generation=dict(script_args=[],sample=False),capture=dict(layers=list(range(40)),readouts=["first item last token","answer logits"],boundary="All post-block residuals on naming exemplars; native gated MLP and W_down columns at specified blocks; exact top1 and answer scores.")),
        analysis=dict(method_version="color_reuse.v1",code_sha256=code,scoring_sha256=digest(CRITERIA),
            selection=dict(stage="study",layers=LAYERS,sizes=SIZES,seed=SEED),required_artifact_roles=sorted(CORE_ROLES)))
    validate_plan(plan);save(OUT/"study-plan.json",plan)
    print("Frozen",len(rows),"prompts",sha(OUT/"study-plan.json"),flush=True)
