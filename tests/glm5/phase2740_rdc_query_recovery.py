"""Keep an explicit receipt for the first pre-inference loader-metadata error."""
from rdc_query_common import *

def main():
    p=BASE/'pilot/recovery_001.json'
    if p.exists():return
    save(p,{'timestamp':stamp(),'source':snapshot(__file__),'attempt':'Initial phase2740_rdc_query_pilot unified-exec session44092',
      'error':"AttributeError: Qwen3ForCausalLM has no attribute hf_device_map in rdc_query_common.load when recording the successfully loaded model metadata.",
      'failure_stage':'After checkpoint loading, before probe freezing or any pilot scientific forward.',
      'scientific_records_created':0,'cause':'An all-resident native load may omit dispatch metadata; the recorder incorrectly required that optional attribute.',
      'fix':'Read optional hf_device_map and also record the actual devices of every parameter. No precision, checkpoint, scientific input or model computation changed.',
      'timing':'Original exception occurred outside the timed failure handler. Exact complete attempt duration is unavailable; no duration invented or added to the measured ledger. Subsequent load errors are timed.',
      'original_protocol_preserved_sha256':sha(BASE/'pilot/protocol.json'),'original_contract_source_snapshots_preserved':True})
    print('QUERY_FIRST_LOADER_RECOVERY_RECORDED')

if __name__=='__main__':main()
