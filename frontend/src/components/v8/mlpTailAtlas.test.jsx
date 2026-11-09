import React from 'react';
import {renderToStaticMarkup} from 'react-dom/server';
import {describe,it,expect,vi} from 'vitest';
import report from '../../../../research/mlp_tail/v1/report.research.json';
const resource=vi.hoisted(()=>({error:null,overview:null}));
vi.mock('../app/rdc_fusion/api.js',()=>({useApi:path=>({error:resource.error,data:path==='/api/research/mlp-tail'?resource.overview||report:null})}));
import MlpTailAtlas,{TailScene} from './MlpTailAtlas.jsx';

describe('最后四层全量真实MLP研究入口',()=>{
  it('显示全量覆盖、精确长文本范围和结论边界',()=>{
    const html=renderToStaticMarkup(<MlpTailAtlas/>);
    for(const text of ['38,912','2,048','1,024-token','跨层预测','全部9728个关联值']) expect(html).toContain(text);
    expect(html).toContain('全量覆盖不等于全部语义已经破解');
    expect(html).toContain('不是图谱独立预测下一词');
    expect(html).toContain('不计为正式机制确认');
  });
  it('保留预测失败及恢复与能力保持的范围',()=>{
    const html=renderToStaticMarkup(<MlpTailAtlas/>);
    expect(html).toContain('1.3078');expect(html).toContain('not established');expect(html).toContain('9216');
  });
  it('全单元3D提供原生索引与查询入口',()=>{
    const html=renderToStaticMarkup(<TailScene layers={{}} metric="A_std" block={35} unit={9727} onPick={()=>{}}/>);
    expect(html).toContain('全部38912个MLP单元');expect(html).toContain('重置视角');
  });
  it('中心已接入扩展时列出八层，并保留已有实验的适用范围',()=>{
    resource.overview={...report,blocks:[28,29,30,31,32,33,34,35],n_units:77824,extension:{}};
    try {
      const html=renderToStaticMarkup(<MlpTailAtlas/>);
      expect(html).toContain('77,824');
      expect(html).toContain('value="28"');
      expect(html).toContain('value="35"');
      expect(html).toContain('递推和干预沿用 B32–B35');
    } finally {resource.overview=null;}
  });
  it('缺失服务器数据时明确说明，不用演示填充',()=>{
    resource.error='中心尚未导入';const html=renderToStaticMarkup(<MlpTailAtlas/>);resource.error=null;
    expect(html).toContain('中心尚未导入');expect(html).toContain('不使用演示数据');expect(html).not.toContain('<canvas');
  });
});
