import { describe, it, expect, vi } from 'vitest';
import { renderToStaticMarkup } from '../../frontend/node_modules/react-dom/server.node.js';
import React from '../../frontend/node_modules/react/index.js';
const state=vi.hoisted(()=>({error:null}));
vi.mock('../../frontend/src/components/app/rdc_fusion/api.js',()=>({API_BASE:'',useApi:()=>({data:null,error:state.error})}));
import Qwen14MlpAtlas from '../../frontend/src/components/v8/Qwen14MlpAtlas.jsx';
import { TailScene } from '../../frontend/src/components/v8/MlpTailAtlas.jsx';
const SNAP='1e970d119494affd09689ec3dd25aa30587dfcbb16d873b40fb917016fe2a2ce';

describe('Qwen3-14B atlas identity and full native dimensions',()=>{
  it('renders the 17408-wide scene without the 4B dimensions',()=>{
    const html=renderToStaticMarkup(React.createElement(TailScene,{blocks:[28,29,30,31,32,33,34,35],width:17408,modelName:'Qwen3-14B',layers:{},metric:'A_std',block:35,unit:17407,onPick:()=>{}}));
    expect(html).toContain('全部139264个MLP单元');expect(html).not.toContain('Qwen3-4B');
  });
  it('blocks a report from a different snapshot',()=>{
    const html=renderToStaticMarkup(React.createElement(Qwen14MlpAtlas,{doc:{model_snapshot:'other'}}));
    expect(html).toContain('已停止显示');expect(html).not.toContain('14B原生单元');
  });
  it('exposes full native indices and truthful coverage',()=>{
    const html=renderToStaticMarkup(React.createElement(Qwen14MlpAtlas,{doc:{model_snapshot:SNAP,documents:1024}}));
    expect(html).toContain('max="17407"');expect(html).toContain('模型共40层');
    expect(html).toContain('全量观察不等于全部语义已经破解');
    expect(html).toContain('完整参数');expect(html).toContain('验证与反例');
  });
  it('shows all twelve new layers while retaining the eight-layer report',()=>{
    for(const [last,count] of [[35,'139,264'],[39,'208,896']]){
      const html=renderToStaticMarkup(React.createElement(Qwen14MlpAtlas,{doc:{model_snapshot:SNAP,documents:1024,blocks:Array.from({length:last-27},(_,i)=>i+28)}}));
      expect(html).toContain(count);expect(html).toContain(`value="${last}"`);
      expect(html).not.toContain(`value="${last+1}"`);
    }
  });
  it('shows a missing-data state instead of borrowing 4B data',()=>{
    state.error='原生研究包未部署';
    const html=renderToStaticMarkup(React.createElement(Qwen14MlpAtlas));state.error=null;
    expect(html).toContain('4B 数据不会替代本模型');
  });
});
