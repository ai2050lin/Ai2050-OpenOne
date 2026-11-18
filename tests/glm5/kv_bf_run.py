"""Run the fixed-dose BF16 study on original Qwen3 snapshots, serially."""

from datetime import datetime
from itertools import combinations
import sys
from zoneinfo import ZoneInfo

import kv_address_run as A
import kv_address_materials as M


ROOT=A.Base.ROOT
OUT=ROOT/"tests/glm5/result/kv_bf_20261115_v1"
TMP=ROOT/"tests/glm5_temp/kv_bf_20261115_v1"
RUN="KV-BF-20261115-v1"
SEED=29422026
WORDS={
    "pilot":"soup bread cheese flour dough rice seed cocoa ham custard salad pasta bacon dinner garlic onion pear apple pepper carrot",
    "discover":"chalk canvas paint easel case pole bracket rail rod tongs hammer spade grater shovel wire cage wedge tube cutter shears pliers drill pointer steel bronze brass tin plastic rubber nylon wool",
    "confirm":"dog cat mouse rat hamster lizard snake frog toad donkey mule llama rabbit yak moose bison buffalo giraffe otter beaver weasel walrus shark trout lobster crab squid octopus clam ant bee beetle spider moth fly butterfly dragon bat owl ibis heron robin parrot finch turkey peacock",
}


def setup():
    A.OUT,A.TEMP,A.RUN=OUT,TMP,RUN
    A.configure()


def stamp():
    return datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="seconds")


def material(stage,tok):
    saved=M.WORDS,M.SEED
    try:
        M.WORDS,M.SEED=WORDS,SEED
        return M.material(stage,tok)
    finally:
        M.WORDS,M.SEED=saved


def pilot(task):
    rows=material("pilot",task.tok)
    A.save("pilot-material.json",rows)
    A.save("pilot-behavior.json",A.behavioral(task,rows,"pilot"))


if __name__=="__main__":
    setup()
    if sys.argv[1]=="pilot":
        A.with_task(pilot)
    else:
        from kv_bf_study import commands
        commands[sys.argv[1]]()
