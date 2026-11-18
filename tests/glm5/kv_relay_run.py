"""Entry point for the L20H5 to L21 addressing experiment."""

from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import kv_address_run as A
import kv_address_materials as Materials


ROOT=A.Base.ROOT
OUT=ROOT/"tests/glm5/result/kv_relay_20261115_v1"
TMP=ROOT/"tests/glm5_temp/kv_relay_20261115_v1"
RUN="KV-RELAY-20261115-v1"
SEED=29412026
HEADS=A.STOCK_HEADS
WORDS={
    "pilot":"wheat corn oat barley rye bean seed fruit flour butter bread cheese salt rice soup stew ham beef fish tuna salmon radish cabbage",
    "discover":"blue green red white black gold silver bronze copper iron steel stone brick glass clay wool cotton silk linen leather rubber plastic metal wood fabric",
    "confirm":"book page card seal badge medal ring coin crown flag stamp key lock rope chain broom torch needle nail screw bolt wheel gear pipe pump plug tray bowl cup mug dish vase pot pan drum mask boot bag pouch cap net cushion drawer carpet helmet feather knife fork",
}


def configure():
    A.OUT,A.TEMP,A.RUN=OUT,TMP,RUN
    A.configure()


def time_mark():
    return datetime.now(ZoneInfo("America/Chicago")).isoformat(timespec="seconds")


def design_material(split,tok):
    saved=Materials.WORDS,Materials.SEED
    try:
        Materials.WORDS=WORDS
        Materials.SEED=SEED
        return Materials.material(split,tok)
    finally:
        Materials.WORDS,Materials.SEED=saved


def pilot(task):
    rows=design_material("pilot",task.tok)
    A.save("pilot-material.json",rows)
    A.save("pilot-behavior.json",A.behavioral(task,rows,"pilot"))


if __name__=="__main__":
    configure()
    if sys.argv[1]=="pilot":
        A.with_task(pilot)
    else:
        from kv_relay_study import commands
        commands[sys.argv[1]]()
