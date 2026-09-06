"""Frozen calibration winner,8192 heldout BF16 cases and every physical coordinate."""
import gc
import torch
from phase2620_native_coordinate_contract import *
from phase2662_symmetric_mapping_contract import heldout,length_audit,load_native
from phase2663_symmetric_mapping_calibration import OUT as CAL,run,behavior_groups

OUT=RESULT/'phase2664_symmetric_native_field'


def main():
    assert not (OUT/'analysis/final.json').exists();selection=read(CAL/'protocol/selected.json');model,tok=load_native('qwen4');cases=heldout(tok,selection['selection'])
    assert len(cases)==len({r['prompt'] for r in cases})==8192 and sum(r['published'] for r in cases)==64
    save(OUT/'material/cases.json',cases);save(OUT/'protocol/frozen.json',{'selection_sha256':sha(CAL/'protocol/selected.json'),'material_sha256':sha(OUT/'material/cases.json'),'length_audit':length_audit(cases),'dtype':str(model.dtype),'quantized':bool(getattr(model,'is_quantized',False))})
    records=run(model,tok,cases,OUT,fields=True);del model;gc.collect();torch.cuda.empty_cache()
    checks={'8192_cases':len(records)==8192,'8192_raw_packs':len(read(OUT/'analysis/raw_manifest.json'))==8192,'selection_unchanged':sha(CAL/'protocol/selected.json')==read(OUT/'protocol/frozen.json')['selection_sha256'],'256_sequence_selection':sum(r['fp_selected'] for r in cases)==256,'nonquantized':not read(OUT/'protocol/frozen.json')['quantized']}
    assert all(checks.values());finish(2664,'8192冻结对称指令的自然行为与源位置全坐标场',OUT,{'provenance':str(Path(__file__)),'summary':{'behavior':behavior_groups(records),'protocol':read(OUT/'protocol/frozen.json')},'checks':checks},
        '使用校准文件冻结问法，在未见的8实体对/语言上自然前向。所有H检查点2560坐标和36层9728实际MLP边界单元；源A/B与输出位置均保留全部坐标，未按幅度挑点。',
        r'H^\ell_{A,j},H^\ell_{B,j},H^\ell_{out,j};\quad S^\ell_j=\sum_tH^\ell_{t,j},\quad Q^\ell_j=\sum_t(H^\ell_{t,j})^2.',
        '8192条件，八族双语，8实体对、双正文句式/目标/顺序/所问实体/真假极性/答案映射；前4实体对与后4分账。64展示例保存完整H，所有例逐token扫描成全坐标和/平方和，原始MLP只取边界。',
        '正文固定而后续问题/映射变化，可追踪源位置到输出的条件分化；相同长度源前缀应单独校验，不能用不同形状的舍入差声称未来信息回流。所有行为失败原样进入主图。',
        '模板和词义目录复用，不是8192独立语义问题。全部token H曾实际读取，但非展示例没有永久逐token原包；保留全坐标源/边界与全部token累计量，不能冒称所有token的MLP均采集。',
        '接续2665全坐标目标/顺序/形式和三类方向图、旧候选确认、源位置数值审计，随后多token参数验证。')


if __name__=='__main__':main()
