export const CONTROL_PANEL_BLUEPRINT = {
  main: {
    label: 'DNN',
    mission: '分析深度神经网络中的语言数学结构，还原大脑的数学原理。',
    operationFocus: '按阶段观测、提取、验证、系统归纳，构建编码证据链。',
    formula: 'E = {Layer Signature, FS, PI, HI, Δ-neuron}',
    model3d: '层级骨架 + 关键神经元 + 动态编码轨迹。',
  },
  dnn: {
    label: 'DNN',
    mission: '分析深度神经网络中的各种特性，作为综合观察工具。',
    operationFocus: '围绕结构分析算法切换参数，做多视角验证。',
    formula: 'f(x) = W_L σ(...σ(W_2 σ(W_1 x)))',
    model3d: 'Logit-Lens、流形、回路、拓扑等观测图层叠加。',
  },
  snn: {
    label: 'SNN',
    mission: '作为脉冲神经网络分析工具，观测放电、可塑性与动力学。',
    operationFocus: '控制刺激、步进、播放与有效性检验参数。',
    formula: 'τ dV/dt = -(V - V_rest) + I(t), spike when V > θ',
    model3d: '脉冲活动热区 + 层间传播轨迹。',
  },
  icspb: {
    label: 'ICSPB',
    mission: '作为当前模型工作台，聚焦语言主干、在线写读、回放与固化的统一验证。',
    operationFocus: '围绕语言训练、语义推演、记忆回放、在线学习与稳定性做参数探索。',
    formula: 'y = SlowLogic(x) + Σ α_i · FastFiber_i(x)',
    model3d: '主干承载语言压缩，受控分支承载快速写入、回放与固化。',
  },
};
