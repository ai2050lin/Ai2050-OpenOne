/* 空间透镜：四种模式
   ① 层平铺（默认）——36 层玻璃盒阵列（参考旧版），点击层盒看内部结构与全部参数
   ② 神经元级——单层内部 3D 全量展开（16,384 个可独立选中的点：residual/MLP/attn），精确到单个神经元
   ③ 特征点云——L6 特征族点云（原视图，接 collect.npz 前 demo）
   ④ 参数热图——单层权重分块统计（demo）
   零依赖 canvas 伪 3D；接入点在各模式组件内注释 */
import { useState } from 'react';
import StackMode from './StackMode.jsx';
import NeuronMode from './NeuronMode.jsx';
import CloudMode from './CloudMode.jsx';
import ParamMode from './ParamMode.jsx';

const MODES=[
  {k:'stack',t:'层平铺',d:'36 层平放'},
  {k:'neuron',t:'神经元级',d:'单神经元'},
  {k:'cloud',t:'特征点云',d:'L6 族'},
  {k:'param',t:'参数热图',d:'单层 W'},
];

export default function LensSpatial({on}){
  const [mode,setMode]=useState('stack');
  const [neuronLayer,setNeuronLayer]=useState(6);
  const [paramLayer,setParamLayer]=useState(6);

  return (
    <section className={'fw-view fw-spatial'+(on?' on':'')}>
      <div className="fw-sp-toolbar">
        <div className="fw-mode-seg">
          {MODES.map(m=>(
            <button key={m.k} className={mode===m.k?'on':''} onClick={()=>setMode(m.k)} title={m.d}>
              {m.t}<small>{m.d}</small>
            </button>
          ))}
        </div>
        {mode==='cloud'&&(
          <>
            <label className="fw-chk"><input type="checkbox" defaultChecked/> 特征点云</label>
            <label className="fw-chk"><input type="checkbox" defaultChecked/> top-k 邻居连线</label>
            <label className="fw-chk"><input type="checkbox"/> 残差流骨架</label>
            <label className="fw-chk"><input type="checkbox"/> 14B/9B 叠加</label>
          </>
        )}
        {mode==='stack'&&(
          <span style={{color:'var(--fw-text-3)'}}>全部 36 层平放（L0 → L35）· 玻璃盒=层结构 · <b>左键=旋转（上下左右） · 右键=平移（上下左右）</b> · 滚轮缩放 · 点击层盒 → 右侧全部参数 → 可下钻神经元级</span>
        )}
        {mode==='neuron'&&(
          <span style={{color:'var(--fw-text-3)'}}><b>左键=旋转（上下左右） · 右键=平移（上下左右）</b> · 滚轮缩放 · L{neuronLayer} 层内部 16,384 点独立选中 · 底部时间轴播放 token 步（激活/变化/写入）· 选中单元看 t 轴曲线 + ℓ 轴跨层基座</span>
        )}
        {mode==='cloud'&&(
          <span style={{color:'var(--fw-text-3)'}}><b>左键=旋转（上下左右） · 右键=平移（上下左右）</b> · 滚轮缩放</span>
        )}
        {mode==='param'&&(
          <span style={{color:'var(--fw-text-3)'}}>当前层 L{paramLayer} · 在层栈/神经元模式选中后可切换</span>
        )}
        <span style={{marginLeft:'auto',fontFamily:'var(--fw-mono)',color:'var(--fw-text-3)'}}>
          {mode==='stack'?'36L · 公开架构数字':'d_model 2560 · d_ff 9728 · 32h×128'}
        </span>
      </div>
      <div className="fw-sp-wrap">
        {mode==='stack'&&<StackMode onOpenParam={l=>{setParamLayer(l);setMode('param');}} onOpenNeuron={l=>{setNeuronLayer(l);setMode('neuron');}}/>}
        {mode==='neuron'&&<NeuronMode layer={neuronLayer} onLayer={setNeuronLayer} onOpenParam={l=>{setParamLayer(l);setMode('param');}} onBack={()=>setMode('stack')}/>}
        {mode==='cloud'&&<CloudMode/>}
        {mode==='param'&&<ParamMode layer={paramLayer}/>}
      </div>
    </section>
  );
}
