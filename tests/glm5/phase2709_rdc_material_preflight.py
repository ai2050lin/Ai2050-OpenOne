"""Check exact material serialization and archive provenance for pre-capture repair."""
from rdc_conditional_common import *
from rdc_attention_transfer_material import build


def main():
    from transformers import AutoTokenizer
    out=CAMPAIGN/'o_generalization';tok=AutoTokenizer.from_pretrained(ROOT/'models/hf/qwen3-4b',local_files_only=True,use_fast=True)
    rows=build(tok);assert rows==read(out/'prefixes.json')==json.loads(json.dumps(rows,ensure_ascii=False))
    assert not list((out/'commits').glob('*.json')) and not (out/'runtime.json').exists()
    save(out/'pre_capture_repair.json',{'timestamp':stamp(),'passed':True,'rows_unchanged':512,'material_sha':sha(out/'prefixes.json'),
      'prior_protocol_sha':sha(out/'protocol_pre_capture_v1.json'),'new_protocol_sha':sha(out/'protocol.json'),
      'correction':'Python tuple instructions serialized to JSON lists and failed in-memory immutable equality before any model capture. Instructions now explicitly lists. Serialized512 materials, texts and splits are unchanged; prior preflight protocol archived; current protocol records corrected material-source hash.'})
    print('O_MATERIAL_SERIALIZATION_CHECKS_PASS',flush=True)


if __name__=='__main__':main()
