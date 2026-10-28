/* 研究工作台的「?」页面说明：按当前一级界面（进展 / 空间 / 研发 / 配置 / 本机工具）弹出功能与原理说明。
   内容只描述本仓库已实现的页面与数据流（workbenchUtil.js、RouteLens.jsx、SpaceLens.jsx 与 rdc_fusion 各面板），
   不描述尚未实现的功能；证据等级、协议纪律等表述与 AGENTS.md / design 文档保持一致。 */
import { useEffect } from 'react';

const HELP = {
  route: {
    title: '进展 · 功能与原理',
    sub: '神经元逆向分析进展、语言图谱进展与证据门链（只读）',
    intro: '本页回答「研究走到了哪里」：把神经元主线与语言图谱主线的覆盖度、证据强度和研究缺口，按统一的证据门等级展示成记分板与热力条。',
    sections: [
      {
        h: '页面内容',
        points: [
          '神经元逆向分析：普查规模 → 逐层摘要与分型 → 解释检验 → 模板选择性，各项分开计数，按块画成热力条（每格一个块）。',
          '语言图谱：大类 × 等级的覆盖总览、语言项 × 模型的等级矩阵、研究缺口，以及模式族证据门链（记分板快照）。',
        ],
      },
      {
        h: '等级原理（证据门链）',
        points: [
          '每个语言项在「读出轴」上有五级：未研究 ○ → 有数据 ◔ → 可读出 ◑ → 留出复现 ◕ → 跨模型 / 跨族 ●。符号 + 文字 + 色阶三重编码，不只靠颜色。',
          '等级只由证据推进：模板对照的观察最多到「可读出」；在留出材料（未见词表 / 未见表述）上复现成功才升「留出复现」；跨模型一致才到顶。',
          '「读得出 ≠ 控得住」：几何上可读出不等于行为上可干预；干预失败不回退已到的等级，但会在缺口与记录里如实标注。',
        ],
      },
      {
        h: '数据来源',
        points: [
          '只读本机数据包或中心 API（/api/neurons/*、/api/lang/*）。',
          '没有数据时显示空态，不回退演示数字——空态本身就是研究进度的如实记录。',
        ],
      },
    ],
    footnote: '证据等级定义见平台「介绍」页与 design/ 文档；升级记录 append-only，可回查每次升级的运行 ID 与产物哈希。',
  },
  spatial: {
    title: '空间 · 功能与原理',
    sub: 'LLM 多层结构：神经元逆向分析结果与语言图谱的有效编码位置',
    intro: '本页是「模型内部长什么样」的可视化工作区：左侧常驻 LLM 多层结构（层塔），右侧按视图切换观察对象。',
    sections: [
      {
        h: '层塔（左侧常驻）',
        points: [
          'hs[0] 是嵌入层，块 b 的输出是 hs[b+1]；层按块折叠展示，可叠加不同实测指标：选择性 / 已解释 / T1 检测器数 / 强激活率中位数。',
          '着色只反映实测统计量；保持原生坐标顺序，不用排序制造图案。',
        ],
      },
      {
        h: '神经元视图',
        points: [
          '选中一块后展开该块内部：MLP 单元阵列（如 9,728 个单元）与单元卡——分型、强激活率、机制 regime（gate/value 主导象限）、解释检验分级、模板选择性（正/负两极分开标注）。',
          '分型与解释检验是独立的观察维度：一个单元可以有清晰分型但仍「未解释」。',
        ],
      },
      {
        h: '语言图谱 / 响应点云 / 谱图关系',
        points: [
          '语言图谱：大类 → 语言族 → 语言项，标出每个概念的编码层段、点云与有效神经元。',
          '响应点云：单个模式族的 3D 响应空间、规律分析与合并统计。',
          '谱图关系：跨模式族的特征、共享编码轴与概念假设——布局距离只是展示，不代表模型几何。',
        ],
      },
      {
        h: '数据与边界',
        points: [
          '只读本机数据包或中心 API；本地包（研究包 / 谱图包）在浏览器内解压查看，不上传到任何服务器。',
          '可视化服务于发现：候选规律还需留出复现与干预检验，本页图形本身不构成因果结论。',
        ],
      },
    ],
  },
  process: {
    title: '研发 · 功能与原理',
    sub: '题库与分布式测试协议的工作台：模板 → 试跑 → 冻结 → 分发 → 审计 → 普查',
    intro: '本页把「设计一次实验」的完整闭环摆在工作台上：从生成式模板文法到冻结分发，每一步都留下可回查的产物。',
    sections: [
      {
        h: '闭环流程',
        points: [
          '题库：已登记的语言项、模式族与当前状态。',
          '模板工作室：用 template_grammar.v1（句式框架 × 填充词表 × 因子水平）写文法；先检查逐格样本数、功效近似与告警（含读出位置的因果可见性校验）。',
          '试跑与冻结：试跑只从 discovery 抽样、按（条件格, 句式）看行为门；通过后预注册（prereg）并冻结文法版本，之后不可改、只能升版本。',
          '分发与节点 → 审计队列 → 普查作业：把冻结文法发给节点运行，回收结果逐条审计，普查结论回流更新「进展」页的等级。',
        ],
      },
      {
        h: '关键纪律',
        points: [
          'discovery / confirm / lockbox 按填充词确定性切分，禁止随机逐条切分（防近重复泄漏）；句式泛化用 heldout_frames。',
          '行为门未通过的条件格不进入内部分析；「模型做不到」与「probe 设计问题」分开处置，后者修改文法升版本重新冻结并记录原因。',
          '有对照时按完整答案对数概率强制选择评分，警惕机会水平与「是」偏向。',
        ],
      },
      {
        h: '连接方式',
        points: [
          '网页版（与中心同源）操作时实时连接中心；远端客户端只保留只读的题库。',
          '自动测试工位只在本机中心（local_tools）出现——公开中心没有被测模型。',
        ],
      },
    ],
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
    footnote: '各界面可读 / 可写的完整规则见 workbenchUtil.js 的 visibleStations 与 config.js 的文件头注释。',
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
