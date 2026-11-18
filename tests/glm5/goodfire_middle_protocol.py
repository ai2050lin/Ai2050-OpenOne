"""Required protocol regression with isolated D-drive scratch and empty research source."""
import faulthandler
import os
from pathlib import Path
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from deploy.storage_policy import check_write
scratch=check_write(ROOT/'tests/glm5_temp/goodfire-middle-protocol',2*2**30);scratch.mkdir(exist_ok=True)
empty=scratch/'empty-research';empty.mkdir(exist_ok=True)
os.environ.update(TEMP=str(scratch),TMP=str(scratch),TMPDIR=str(scratch),AI2050_RESEARCH_DIR=str(empty),AI2050_SNAPSHOT_DISABLE='1')
tempfile.tempdir=str(scratch)
faulthandler.enable();faulthandler.dump_traceback_later(120,repeat=True)
result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover('tests/dist_protocol'))
faulthandler.cancel_dump_traceback_later();sys.exit(not result.wasSuccessful())
