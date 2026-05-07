export const ProjectRoadmapTab = ({
  roadmapData,
  mathRouteSystemPlan,
}) => {
  return (
    <div style={{ animation: 'roadmapFade 0.6s ease-out', maxWidth: '1180px', margin: '0 auto' }}>
      <div style={{ marginBottom: '24px' }}>
        <h2 style={{ fontSize: '30px', fontWeight: '900', color: '#ffaa00', marginBottom: '10px' }}>战略层级路线图</h2>
        <div style={{ color: '#9ca3af', fontSize: '14px' }}>{roadmapData?.definition?.summary || '聚焦结构智能路线。'}</div>
      </div>

      <div
        style={{
          padding: '30px',
          background: 'linear-gradient(135deg, rgba(255,170,0,0.12) 0%, rgba(255,170,0,0.03) 100%)',
          border: '1px solid rgba(255,170,0,0.24)',
          borderRadius: '24px',
          marginBottom: '28px',
        }}
      >
        <div style={{ color: '#ffaa00', fontWeight: 'bold', fontSize: '18px', marginBottom: '16px' }}>核心思路</div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr', gap: '10px' }}>
          {[
            '1. 大脑应存在高度结构化的数学组织，这可能是智能产生的基础。',
            '2. 深度神经网络可能部分还原了该结构，因此具备可扩展的语言与推理能力。',
            '3. 通过结构分析与可控干预，提取可验证的编码规律，形成更通用的智能理论。',
          ].map((line, idx) => (
            <div
              key={idx}
              style={{
                padding: '14px 16px',
                borderRadius: '12px',
                background: 'rgba(255,255,255,0.05)',
                color: '#f4e4c1',
                fontSize: '14px',
                lineHeight: '1.6',
              }}
            >
              {line}
            </div>
          ))}
        </div>
      </div>
    
      <div
        style={{
          padding: '30px',
          borderRadius: '24px',
          border: '1px solid rgba(16,185,129,0.28)',
          background: 'linear-gradient(135deg, rgba(16,185,129,0.10) 0%, rgba(16,185,129,0.03) 100%)',
          marginBottom: '28px',
        }}
      >
        <div style={{ color: '#34d399', fontWeight: 'bold', fontSize: '18px', marginBottom: '8px' }}>主流方案</div>
        <div style={{ color: '#a7f3d0', fontSize: '13px', lineHeight: '1.7', marginBottom: '14px' }}>
          目前主流逆向工程路线完整图谱，按核心思想、关键方法、代表工作、优点与缺点对比。
        </div>

        <div
          style={{
            marginTop: '12px',
            borderRadius: '12px',
            border: '1px solid rgba(255,255,255,0.08)',
            background: 'rgba(0,0,0,0.22)',
            padding: '12px',
          }}
        >
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', minWidth: '1180px', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: 'rgba(255,255,255,0.05)' }}>
                  <th style={cellHeaderStyle('#a7f3d0')}>#</th>
                  <th style={cellHeaderStyle('#a7f3d0')}>路线</th>
                  <th style={cellHeaderStyle('#6ee7b7')}>核心思想</th>
                  <th style={cellHeaderStyle('#6ee7b7')}>关键方法/研究对象</th>
                  <th style={cellHeaderStyle('#86efac')}>代表工作</th>
                  <th style={cellHeaderStyle('#86efac')}>优点</th>
                  <th style={cellHeaderStyle('#fca5a5')}>缺点</th>
                </tr>
              </thead>
              <tbody>
                {[
                  {
                    num: 1, name: '统计几何派',
                    core: 'hidden states 的几何结构 = 语言编码',
                    methods: 'PCA、SVD、CKA、cosine、clustering、manifold、linear probe',
                    reps: 'embedding geometry、representation learning、feature superposition',
                    pros: '简单；可扩展；容易发现统计规律',
                    cons: '极易高维幻觉；correlation ≠ mechanism；无法解释 computation',
                  },
                  {
                    num: 2, name: 'Mechanistic Interpretability',
                    core: 'Transformer = 可分解计算图',
                    methods: 'attention heads、MLP neurons、circuits、induction heads、path patching、causal tracing',
                    reps: 'Anthropic circuit work',
                    pros: '真正因果分析；可以找到局部算法',
                    cons: '极难扩展；小模型有效大模型崩溃；局部 circuit 很难解释高级行为',
                  },
                  {
                    num: 3, name: '动力系统派',
                    core: 'LLM 是递归动力系统；核心不是"state 是什么"而是"state 如何演化"',
                    methods: 'trajectory、attractor、rollout、phase transition、Lyapunov、state evolution',
                    reps: '—',
                    pros: '语言能力本质更像 computation unfolding 而非静态编码',
                    cons: '实验设计复杂；参数敏感',
                  },
                  {
                    num: 4, name: '信息论派',
                    core: '语言系统 = 压缩系统；自回归训练逼迫系统建立最优压缩结构',
                    methods: 'entropy、mutual information、minimum description length、predictive coding、rate-distortion',
                    reps: '—',
                    pros: '语义/语法/逻辑可能只是最优压缩结构的自然结果；方向非常深',
                    cons: '互信息估计高维偏差；对几何结构刻画有限',
                  },
                  {
                    num: 5, name: '程序归纳派',
                    core: '语言能力 = 隐式程序执行（翻译/CoT/算术/代码都是 latent program execution）',
                    methods: 'neural execution、latent algorithms、differentiable interpreters',
                    reps: '—',
                    pros: 'Transformer 内部不是知识图谱，而是大量可组合微程序',
                    cons: '程序边界难以精确定义；大模型中微程序爆炸',
                  },
                  {
                    num: 6, name: '生成递归派',
                    core: '真正能力不在单次 forward，而在 token→token 的 recursive rollout（自条件递归系统）',
                    methods: '自条件递归、生成结果反成下一步条件',
                    reps: '—',
                    pros: 'CoT/规划/推理/反思本质都是思维链形成机制',
                    cons: '最容易被忽略；递归分析工具不足',
                  },
                  {
                    num: 7, name: '神经符号派',
                    core: '语言内部存在变量/规则/绑定/符号操作，只是隐藏在 distributed representation 中',
                    methods: 'variable binding、compositionality、symbolic structure',
                    reps: '—',
                    pros: '解释"苹果→红苹果→绿色苹果"如何组合的重要方向',
                    cons: 'distributed 中的符号边界模糊；难以从连续表示提取离散符号',
                  },
                  {
                    num: 8, name: '生物脑派',
                    core: 'Transformer 可能只是大脑编码机制的离散近似',
                    methods: 'sparse coding、predictive brain、pulse coding、topology、hippocampal indexing',
                    reps: '—',
                    pros: '提供生物学约束和启发',
                    cons: '生物机制与人工网络差距大；类比不等于机制',
                  },
                ].map((row, idx) => (
                  <tr key={idx} style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                    <td style={cellBodyStyle('#a7f3d0', true)}>{row.num}</td>
                    <td style={cellBodyStyle('#6ee7b7', true)}>{row.name}</td>
                    <td style={cellBodyStyle('#d1fae5')}>{row.core}</td>
                    <td style={cellBodyStyle('#d1fae5')}>{row.methods}</td>
                    <td style={cellBodyStyle('#bbf7d0')}>{row.reps}</td>
                    <td style={cellBodyStyle('#dcfce7')}>{row.pros}</td>
                    <td style={cellBodyStyle('#fee2e2')}>{row.cons}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
          <div style={infoCardStyle}>
            <div style={infoCardTitleStyle}>路线关系</div>
            <div style={infoLineStyle}>路线1（几何）和路线2（机制）是当前主流</div>
            <div style={infoLineStyle}>路线3（动力系统）和路线6（生成递归）是突破方向</div>
            <div style={infoLineStyle}>路线4（信息论）和路线5（程序归纳）提供深层理论基础</div>
            <div style={infoLineStyle}>路线7（神经符号）和路线8（生物脑）提供跨学科约束</div>
          </div>
          <div style={infoCardStyle}>
            <div style={infoCardTitleStyle}>关键洞察</div>
            <div style={infoLineStyle}>语言能力本质上更像 computation unfolding 而非静态编码</div>
            <div style={infoLineStyle}>自回归训练逼迫系统建立最优压缩结构，语义/语法/逻辑可能是其自然结果</div>
            <div style={infoLineStyle}>Transformer 内部不是知识图谱，而是大量可组合微程序</div>
            <div style={infoLineStyle}>CoT/规划/推理/反思本质都是自条件递归的思维链形成机制</div>
          </div>
        </div>
      </div>

      <div
        style={{
          padding: '30px',
          borderRadius: '24px',
          border: '1px solid rgba(99,102,241,0.28)',
          background: 'linear-gradient(135deg, rgba(99,102,241,0.10) 0%, rgba(99,102,241,0.03) 100%)',
          marginBottom: '28px',
        }}
      >
        <div style={{ color: '#818cf8', fontWeight: 'bold', fontSize: '18px', marginBottom: '8px' }}>
          {mathRouteSystemPlan?.title || '数学路线'}
        </div>
        <div style={{ color: '#c7d2fe', fontSize: '13px', lineHeight: '1.7', marginBottom: '14px' }}>
          {mathRouteSystemPlan?.subtitle || '对比多路线理论深度、计算可行性与解释性。'}
        </div>

        <div
          style={{
            marginTop: '12px',
            borderRadius: '12px',
            border: '1px solid rgba(255,255,255,0.08)',
            background: 'rgba(0,0,0,0.22)',
            padding: '12px',
          }}
        >
          <div style={{ color: '#a5b4fc', fontSize: '11px', fontWeight: 'bold', marginBottom: '8px' }}>路线对比</div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', minWidth: '1180px', borderCollapse: 'collapse' }}>
              <thead>
                <tr style={{ background: 'rgba(255,255,255,0.05)' }}>
                  <th style={cellHeaderStyle('#c7d2fe')}>路线</th>
                  <th style={cellHeaderStyle('#93c5fd')}>路线说明</th>
                  <th style={cellHeaderStyle('#86efac')}>优点</th>
                  <th style={cellHeaderStyle('#fca5a5')}>缺点</th>
                  <th style={cellHeaderStyle('#93c5fd')}>可行性结论</th>
                  <th style={cellHeaderStyle('#c7d2fe')}>理论深度</th>
                  <th style={cellHeaderStyle('#c7d2fe')}>计算可行性</th>
                  <th style={cellHeaderStyle('#c7d2fe')}>可解释性</th>
                  <th style={cellHeaderStyle('#c7d2fe')}>与 SHMC/NFBT 兼容</th>
                </tr>
              </thead>
              <tbody>
                {(mathRouteSystemPlan?.routeAnalysis || []).map((item, idx) => (
                  <tr key={idx} style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}>
                    <td style={cellBodyStyle('#e0e7ff', true)}>{item.route}</td>
                    <td style={cellBodyStyle('#bfdbfe')}>{item.routeSummary || item.description || item.routeDesc || '-'}</td>
                    <td style={cellBodyStyle('#dcfce7')}>
                      {(item.pros || []).map((line, pIdx) => (
                        <div key={pIdx}>{`${pIdx + 1}. ${line}`}</div>
                      ))}
                    </td>
                    <td style={cellBodyStyle('#fee2e2')}>
                      {(item.cons || []).map((line, cIdx) => (
                        <div key={cIdx}>{`${cIdx + 1}. ${line}`}</div>
                      ))}
                    </td>
                    <td style={cellBodyStyle('#bae6fd')}>{item.feasibility}</td>
                    <td style={cellBodyStyle('#dbeafe')}>{item.depth}</td>
                    <td style={cellBodyStyle('#dbeafe')}>{item.compute}</td>
                    <td style={cellBodyStyle('#dbeafe')}>{item.interpret}</td>
                    <td style={cellBodyStyle('#dbeafe')}>{item.compatibility}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div style={{ marginTop: '12px', display: 'grid', gridTemplateColumns: '1.2fr 1fr 1fr', gap: '12px' }}>
          <div style={infoCardStyle}>
            <div style={infoCardTitleStyle}>分层架构</div>
            {(mathRouteSystemPlan?.architecture || []).map((line, idx) => (
              <div key={idx} style={infoLineStyle}>
                {`${idx + 1}. ${line}`}
              </div>
            ))}
          </div>

          <div style={infoCardStyle}>
            <div style={infoCardTitleStyle}>资源配比</div>
            {(mathRouteSystemPlan?.allocation || []).map((line, idx) => (
              <div key={idx} style={infoLineStyle}>
                {line}
              </div>
            ))}
          </div>

          <div style={infoCardStyle}>
            <div style={infoCardTitleStyle}>阶段里程碑</div>
            {(mathRouteSystemPlan?.milestones || []).map((line, idx) => (
              <div key={idx} style={infoLineStyle}>
                {`${idx + 1}. ${line}`}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

const cellHeaderStyle = (color) => ({
  textAlign: 'left',
  padding: '8px 10px',
  color,
  fontSize: '11px',
  borderBottom: '1px solid rgba(255,255,255,0.08)',
});

const cellBodyStyle = (color, bold = false) => ({
  padding: '9px 10px',
  color,
  fontSize: '11px',
  lineHeight: '1.55',
  verticalAlign: 'top',
  fontWeight: bold ? 'bold' : 'normal',
});

const infoCardStyle = {
  padding: '14px',
  borderRadius: '12px',
  background: 'rgba(0,0,0,0.22)',
  border: '1px solid rgba(255,255,255,0.08)',
};

const infoCardTitleStyle = {
  color: '#a5b4fc',
  fontSize: '11px',
  fontWeight: 'bold',
  marginBottom: '6px',
};

const infoLineStyle = {
  color: '#dbeafe',
  fontSize: '12px',
  lineHeight: '1.6',
  marginBottom: '4px',
};

const INTERFACE_MODULE_DRAFT_WIREFRAME = `┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ 顶栏：系统切换 | 当前工作台 | 当前样本 | 路线图入口 | 帮助 | 语言切换                      │
├───────────────┬──────────────────────────────────────────────┬──────────────────────────────┤
│ 左侧操作栏    │ 中央 3D 主场景                               │ 右侧数据面板                 │
│ 1. 研究层切换 │ 1. 当前概念 / 拼图 / 样本的三维主视图        │ Tab 1 当前焦点               │
│ 2. 场景模式   │ 2. 节点、链路、层间结构                      │ Tab 2 层数据                 │
│ 3. 条件筛选   │ 3. 差异高亮                                  │ Tab 3 样本回放               │
│ 4. 样本投射   │ 4. 局部链路回放                              │ Tab 4 资产与证据             │
│ 5. 动画控制   │ 5. 选中对象的空间反馈                        │                              │
├───────────────┴──────────────────────────────────────────────┴──────────────────────────────┤
│ 底部状态与时间轴：before | bridge | after | 当前验证状态 | 当前风险 | 当前缺口资产         │
└──────────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│ 战略层级路线图：总览 | 性质 | 概念关联 | 拼图对比 | 理论主线 | 阶段路线                   │
└──────────────────────────────────────────────────────────────────────────────────────────────┘`;
