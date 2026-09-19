import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'glm5'))
from rdc_joint_common import *
from rdc_joint_capture import ledger
path=BASE/'scale/qwen4/alignment_retry.json'
if not path.exists():
    save(path,{'timestamp':stamp(),'first_failure':'All-layer Q4 replay assertion on train-zh-j0047; character endpoint matching selected a later piece with the same offset, not a same-position numerical failure.',
        'first_attempt_last_progress_seconds':22.1,'charged_seconds_rounded_up':30,
        'second_attempt':'Protocol immutable equality failed because a new timestamp/source was constructed; model not loaded in this attempt. Fix preserve existing protocol and append execution-source amendment.',
        'second_attempt_charged_seconds_rounded_up':4,'no_failed_field_committed_at_mismatched_position':True,
        'action':'Preserve Q4 native anchor index; record overlapping-offset limitation for cross-tokenizer character endpoint matching; complete selected sources again and require all-layer original Q4 bits equal.'})
    ledger('qwen4_scale_failed_attempts_conservative_charge',34,actual_exact_duration=False)
print('RETRY_RECORD_SAVED')
