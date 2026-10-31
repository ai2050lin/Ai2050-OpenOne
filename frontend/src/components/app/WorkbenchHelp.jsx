/* 研究工作台的「?」页面说明：按当前一级界面（数据集 / 例句 / 模式与优先级 / 神经元图谱 / 进展 / 配置 / 本机工具）弹出功能与原理说明。
   内容只描述本仓库已实现的页面与数据流（workbenchUtil.js、mech/*、v8/SpaceLens.jsx 与 rdc_fusion 各面板），
   不描述尚未实现的功能；证据等级、协议纪律等表述与 AGENTS.md / design/attention_neuron_plan_v1.md 保持一致。 */
import { useEffect } from 'react';

const DATA_RULE = '只读本机数据包（/api/mechanism/*、/api/lang/*、/api/neurons/*）；没有数据时显示「未接入」，不回退演示数字。';

const HELP = {
  datasets: {
    title: '数据集 · 功能与原理',
    sub: '各数据集与类别、模型正确率、行为门',
    intro: '本页回答「拿什么材料测、模型在这些材料上做得到吗」。只有过了行为门的句子才进入机制分析：模型原本做不到的任务不能当机制正例。',
    sections: [
      { h: '三类材料', points: [
        '机制材料（冲突数据集）：每条事实一行，记忆答案、反事实（上下文）答案、发现 / 留出划分与行为门间隔；点「例句」看这一句。',
        '行为广度（C-Eval / CMMLU 等考试题库）：各大类与科目的正确率。只看行为广度，不作机制材料——选项字母格式会污染读出。',
        '最小对模板（语言项）：各大类语言项在所选模型上的读出等级；点语言项看行为门逐格结果、对照例句与编码层段。',
      ] },
      { h: '子视图', points: ['语言项、响应点云、谱图关系、题库、本地包沿用原「空间」与「研发 · 题库」的视图。'] },
      { h: '数据来源', points: [DATA_RULE] },
    ],
  },
  sentence: {
    title: '例句 · 功能与原理',
    sub: '一句话：行为 → 注意力路由 → 激活的神经元 → 保真度',
    intro: '核心页。从一句冲突句出发，依次看模型更信谁、哪些注意力头把答案搬到句末、哪些 MLP 单元推高或压低答案、只保留所选头能留下多少行为。',
    sections: [
      { h: '直接作用与总效应并列', points: [
        '直接作用（DLA）：头输出经最终归一化投到（上下文 − 记忆）答案方向。总效应：逐头均值消融后答案对数概率差的变化，显示为「去掉后上下文优势的下降量」。',
        '单组件消融会被下游补偿误导（GPT-2 IOI 校准里名称搬运头被排错），所以两者并列，不互相替代。',
      ] },
      { h: '单句与集合', points: [
        '报告存的是每种线索在全部过门句子上的统计；单句的答案概率与单句归因图没有进报告，显示「未接入」。',
        '「输入一句话」子视图在本机模型上做实时前向（只在本机中心可用）。',
      ] },
      { h: '保真度的名称', points: ['现在只有「头电路保真度」（只保留所选头，其余在句末均值消融）；图谱保真度要等神经元 / 特征接入归因图后才有，名称上区分。'] },
    ],
  },
  modes: {
    title: '模式与优先级 · 功能与原理',
    sub: '冲突时谁赢、在哪一层决出、靠哪些头、哪些线索能翻转',
    intro: '优先级 = 两种模式冲突时的胜者与决出层；层号先后不等于层级。每种冲突一节：逐层赛跑曲线 + 优先级表。',
    sections: [
      { h: '赛跑曲线', points: [
        '每条线 = 一种线索在全部过门句子上的平均 logit lens 读数：上下文答案 − 记忆答案的对数几率差，> 0 表示上下文答案领先。',
        '决出层 = 差值最后一次越过 0 且之后保持同号的层（报告给每种线索的中位数）。logit lens 是读出近似。',
      ] },
      { h: '否定对照', points: ['上下文出现反事实答案词但语义是否定：若上下文仍多半胜出，胜率可能来自复制词而不是理解。'] },
      { h: '数据来源', points: [DATA_RULE] },
    ],
  },
  neurons: {
    title: '神经元图谱 · 功能与原理',
    sub: '按层浏览 MLP 单元；四轴与证据等级；按归因优先级推进',
    intro: '「完整图谱」= 每个单元都有四轴卡片（可读 / 读出 / 控制 / 连接）与证据等级，不要求每个单元都有干净含义。',
    sections: [
      { h: '四轴', points: [
        '可读：模板选择性（发现划分拟合、留出同号复现）。读出：直接通路读出 token（近似，不含后续层）。',
        '控制：预注册干预（消融 / steering，正负双向）。连接：跨层上下游电路，按权重计算，是电路假设。',
        '证据等级在 Research OS 登记，单元卡与优先级表都不自动晋级。',
      ] },
      { h: '视图', points: [
        '按层浏览：左侧 3D 层塔是小导航（点击选块），主区是块结构、单元阵列与单元卡。',
        '归因优先级：机制案例里按直接作用排序的单元，先测排在前面的。本机测试：桌面客户端对一个单元跑 S1–S9，结果只存本机。',
      ] },
    ],
  },
  progress: {
    title: '进展 · 功能与原理',
    sub: '各模式保真度、完整案例数、未通过的检验',
    intro: '进度只用三类数：留出材料上的保真度、报告五项齐全的完整案例数、全部预注册判据（未通过与通过同样醒目）。',
    sections: [
      { h: '不做的事', points: [
        '不把覆盖率、任务数或特征数量包装成破解进度；完成与通过都不改变证据等级。',
        '方法校准（GPT-2，答案已知）未通过的检验方法，在主模型上的同类结论要降级看待。',
      ] },
      { h: '数据来源', points: [DATA_RULE, '已测规则账本只在本机中心可读。'] },
    ],
  },
  contribute: {
    title: '贡献 · 功能与原理',
    sub: '贡献算力与我的上传',
    intro: '桌面客户端控制本机节点服务（设备、模型、资源限制、运行与日志）；网页版只给下载入口。我的上传列出你名下节点与各自上传的结果份数。',
    sections: [{ h: '边界', points: ['上传数只描述服务器收到了什么；审计、复算通过只表示产物可以合并，不改变证据等级。'] }],
  },
  admin: {
    title: '管理 · 功能与原理',
    sub: '分步骤向导：建模板 → 冻结 → 分发 → 审计 → 普查 → 自动测试',
    intro: '全部管理功能收成一个向导，只在桌面客户端、管理员账号下出现；操作时实时连接中心。',
    sections: [{ h: '纪律', points: ['模板冻结后语料、runner 与预注册阈值都不可修改；改动须升版本重新冻结。', '审计与普查校验只决定产物能否合并，不改变证据等级。'] }],
  },
  config: {
    title: '配置 · 功能与原理',
    sub: '当前连接的中心与运行形态、账号、被测模型与平台信息',
    intro: '本页回答「我现在连的是哪、能做什么」：把决定各界面可读 / 可写的运行时判定集中列出，不用在别的页面靠猜。',
    sections: [
      {
        h: '这些值从哪来',
        points: [
          '运行形态、数据模式、研发工位、本机工具由客户端启动时判定（config.js 的 RUNTIME_MODE / OFFLINE_CLIENT / ADMIN_LIVE，中心侧 local_tools），不是本页的开关。',
          '被测模型与平台信息来自中心 /api/platform/config（platform_config.v1），只列快照 sha256 完整的模型条目。',
        ],
      },
      {
        h: '为什么没有可改的开关',
        points: [
          '换中心要改客户端首启填写的服务器地址，换运行形态要改启动方式或 .env——都不是网页能改的，摆成开关只会静默失效。',
          '账号的写操作只有「登录 / 注册」与「退出登录」，其余是到账号管理、数据管理、AI 助手模型配置的入口。',
        ],
      },
      {
        h: '数据边界',
        points: [
          '读不到中心配置时如实显示失败原因，不用示例数据填充。',
          '沙盒（合成数据）与实测数据分开标注：合成数据不能当研究结论。',
        ],
      },
    ],
    footnote: '各界面可读 / 可写的完整规则见 workbenchUtil.js 的 visibleLenses / visibleSubs 与 config.js 的文件头注释。',
  },
  local: {
    title: '本机工具 · 功能与原理',
    sub: '只在本机中心（local_tools）出现',
    intro: '研究者本机专用：读取本机模型权重的空间视图与自动测试入口。公开中心没有被测模型，因此远端客户端不显示本界面。',
    sections: [
      {
        h: '使用边界',
        points: [
          '仅在浏览器与中心服务同源（本机运行）时可用。',
          '读取本机权重属于只读观察，不修改模型文件；测试产物按运行 ID 落盘可回查。',
        ],
      },
    ],
  },
};

export default function WorkbenchHelp({ lens, onClose }) {
  const help = HELP[lens] || HELP.local;
  useEffect(() => {
    const onKey = e => { if (e.key === 'Escape') onClose(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);
  return (
    <div className="wb-help-mask" onClick={onClose} role="presentation">
      <div className="wb-help-modal" role="dialog" aria-modal="true" aria-label={help.title} onClick={e => e.stopPropagation()}>
        <header className="wb-help-head">
          <div>
            <h2>{help.title}</h2>
            <p>{help.sub}</p>
          </div>
          <button type="button" className="wb-help-close" onClick={onClose} aria-label="关闭说明">×</button>
        </header>
        <div className="wb-help-body">
          <p className="wb-help-intro">{help.intro}</p>
          {help.sections.map(s => (
            <section key={s.h}>
              <h3>{s.h}</h3>
              <ul>
                {s.points.map((p, i) => <li key={i}>{p}</li>)}
              </ul>
            </section>
          ))}
          {help.footnote && <p className="wb-help-foot">{help.footnote}</p>}
        </div>
      </div>
    </div>
  );
}
