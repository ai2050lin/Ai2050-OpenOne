/* 开放平台 · 学习：面向新人的方法说明。只解释方法与证据口径，不展示数字；每节给出去哪里看真实数据。 */

const SECTIONS = [
  {
    id: 'goal', title: '我们要回答什么问题',
    body: [
      '大语言模型用有限的参数写出无限多样的句子。我们想知道：知识、推理和语法在模型内部是怎样编码、怎样组合的。',
      '平台把研究拆成三张可以查询、可以复核的图：外部语言模式图（词义—关系—角色—组合—语境）、内部特征图（HiddenState 与 MLP 神经元），以及两者之间带证据的连接。',
    ],
  },
  {
    id: 'family', title: '模式族测试：从语言变量出发',
    body: [
      '一个模式族 = 句式框架 × 填充词表 × 因子水平。例如「疑问 / 陈述」只改变语气，其他都不变。',
      '模板在运行前冻结（design_sha），并按填充词切分为发现集 / 确认集 / 留出集：只在发现集上找规律，在没见过的词和句式上检验。',
      '先过行为门：模型在 probe 上真的能区分这个变量，才分析它的内部状态；否则不能作为机制的正例。',
    ],
    go: 'atlas',
  },
  {
    id: 'neuron', title: '神经元普查与单元卡：从模型内部出发',
    body: [
      'MLP 中间层的每个神经元有一个激活 A = silu(G) × V。普查在大量自然文本上统计每个神经元的激活分布、最强与最弱的上下文片段，以及它直接推高 / 压低哪些输出 token。',
      '单元卡把三条证据分开：读出（模板选择性）、控制（预注册干预）、解释（让一个模型写解释，再用新文本和语料检索检验它能不能「输」）。',
      '极值片段天然有选择偏差：一个神经元在最强片段上像「问句末尾」，不等于它只做这一件事。',
    ],
    go: 'atlas',
  },
  {
    id: 'evidence', title: '证据等级：什么算「破解」',
    body: [
      '观察到的关联 < 在留出材料上复现 < 跨族共享 < 留出预测 < 范围内干预有效 < 真实计算来源（归因或参数路径）。',
      '审计通过只表示执行产物可信、可以合并，不是证据晋级。任务数、特征数、覆盖率都不是破解进度；唯一可度量的是用图谱预测模型在留出句子上的输出能做到多好。',
      'AI 提出的假设一律标为「未检验」，必须写成新的预注册模板运行后才有证据。',
    ],
  },
  {
    id: 'join', title: '你可以怎样参与',
    body: [
      '浏览：首页看进度，图谱页看谱图和神经元卡，每个数字都能回到运行与原始数据。',
      '计算：下载一键包，用自己的 NVIDIA GPU 跑分片（普查分片约 5 分钟一个），结果自动上传并署名。',
      '研究：在「研究工作台」里设计模板、预注册、看实验响应与审计队列（需要账号，写操作需要管理员授权）。',
    ],
    go: 'join',
  },
];

const GLOSSARY = [
  ['token', '分词器单位；一个词可能对应多个 token。'],
  ['HiddenState', '某一层、某个位置的残差流向量；hs[0] 是输入嵌入。'],
  ['读出位置', 'decoder 模型的状态只依赖前缀：只有区分变量的文本都出现之后的位置才可能编码它。'],
  ['模型快照', '权重与 tokenizer 文件 sha256 的合成哈希；不同快照的结果不混合。'],
  ['分片', '可以独立运行、可以合并的最小任务单位；同一分片的多份结果用于复算比较。'],
  ['复算', '中心把一部分分片派给另一个账号的节点重算，比较两份结果是否一致。'],
];

export default function OpenLearn({ onGo }) {
  return (
    <div className="op-page op-learn">
      <nav className="op-toc" aria-label="目录">
        {SECTIONS.map(s => <a key={s.id} href={`#learn-${s.id}`}>{s.title}</a>)}
        <a href="#learn-glossary">术语</a>
      </nav>
      <div className="op-learn-body">
        {SECTIONS.map(s => (
          <section key={s.id} id={`learn-${s.id}`} className="op-section">
            <h2>{s.title}</h2>
            {s.body.map((t, i) => <p key={i} className="op-lead">{t}</p>)}
            {s.go && <button type="button" className="op-btn" onClick={() => onGo(s.go)}>{s.go === 'join' ? '去参与 →' : '去图谱看真实数据 →'}</button>}
          </section>
        ))}
        <section id="learn-glossary" className="op-section">
          <h2>术语</h2>
          <dl className="op-dl">
            {GLOSSARY.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}
          </dl>
        </section>
      </div>
    </div>
  );
}
