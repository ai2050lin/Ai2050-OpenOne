import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {describe,it,expect,vi} from 'vitest';
vi.mock('../app/rdc_fusion/api.js',()=>({useApi:()=>({data:null,error:null})}));
import {ConditionDetails} from './MlpConditionCard.jsx';
import MlpConditionReport from './MlpConditionReport.jsx';

describe('条件响应卡的证据边界',()=>{
  it('保留零值、未测状态与观察限制，不把覆盖变成语义确认',()=>{
    const html=renderToStaticMarkup(<ConditionDetails block={28} unit={0} data={{arrays:{within_token_variance_fraction:[0],sign_mix:[0]},scope:'57种token的探索性观察'}}/>);
    expect(html).toContain('0.0%');expect(html).toContain('未测');expect(html).toContain('受控检验待补');
    expect(html).toContain('不是知识量或因果贡献比例');expect(html).not.toContain('通过冻结判据');
  });
  it('稀有条件样本不足不显示为通过',()=>{
    const html=renderToStaticMarkup(<ConditionDetails block={33} unit={1521} data={{arrays:{},run_id:'test',model_snapshot:'abc',
      tested_units:{1521:{role:'source',family:'rare',prediction_passed:false,rare_families:0,rare_powered:false}}}}/>);
    expect(html).toContain('未通过冻结判据');expect(html).toContain('样本不足');expect(html).toContain('0/512');
  });
  it('新报告将条件预测失败、基线及分层不足同时显示',()=>{
    const observation={conditional:{passed:false,relative_mse:1.2,family_wins:40,families:512},rare:{positions:0,families:0,powered:false},factor_contrasts:{}};
    const html=renderToStaticMarkup(<MlpConditionReport doc={{model_snapshot:'abc',observations:{broad:observation,rare:observation},dose_curve:[],counterexamples:[],behavior_by_frame:{test:.8}}}/>);
    expect(html).toContain('未通过冻结判据');expect(html).toContain('不是只凭语言图谱独立预测');expect(html).toContain('40/512');expect(html).toContain('80.0%');
  });
});
