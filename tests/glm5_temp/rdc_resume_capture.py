"""Resume committed samples after status-file sharing violation, unchanged scientific capture."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
from rdc_feature_common import *
from phase2693_rdc_language_capture import main

out=CAMPAIGN/'s1'
save(out/'io_runtime_amendment.json',{'timestamp':stamp(),'committed_before_resume':len(list((out/'commits').glob('*.json'))),
    'error':'Windows PermissionError WinError5 replacing status.json while live UI reads; completed sample commits remain valid.',
    'change':'Atomic JSON replace retries PermissionError up to80 times at25ms; no scientific input, precision, hook, label or array calculation changed.',
    'helper_sha':sha(ROOT/'tests/glm5/rdc_feature_common.py'),'capture_sha':sha(ROOT/'tests/glm5/phase2693_rdc_language_capture.py'),
    'protocol_sha':sha(out/'protocol.json')})
event('s1','resume_from_commits',completed=len(list((out/'commits').glob('*.json'))),reason='status_file_sharing_violation')
main(512)
