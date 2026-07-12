import { BookOpen, Brain, Database, Layers, Sparkles, Target } from 'lucide-react';
import { AtlasControlDashboard } from './AtlasControlDashboard';

const frameworkColumns = [
  {
    title: { zh: '知识网络系统', en: 'Knowledge Network System' },
    icon: <Database size={20} color="#00d2ff" />,
    accent: '#00d2ff',
    items: [
      {
        title: { zh: '概念', en: 'Concepts' },
        body: {
          zh: '包含大量实体概念，如苹果、太阳、石头、水、头发等',
          en: 'Large number of entity concepts, such as apple, sun, stone, water, and hair.',
        },
      },
      {
        title: { zh: '属性', en: 'Attributes' },
        body: {
          zh: '包含概念的特征和性质，如苹果的颜色、味道、大小等',
          en: "Characteristics and properties, such as an apple's color, taste, and size.",
        },
      },
      {
        title: { zh: '抽象系统', en: 'Abstract System' },
        body: {
          zh: '概念的层级抽象，如苹果 -> 水果 -> 食物 -> 物体等',
          en: 'Hierarchical abstraction, such as apple -> fruit -> food -> object.',
        },
      },
      {
        title: { zh: '向量算术', en: 'Vector Arithmetic' },
        body: {
          zh: '经典案例：国王 - 男性 + 女性 = 女王。编码推测：参数空间中语义概念形成几何结构，关系编码为方向和距离。',
          en: 'Classic example: king - man + woman = queen. Semantic concepts may form geometric structures, with relationships encoded as directions and distances.',
        },
      },
    ],
  },
  {
    title: { zh: '逻辑体系', en: 'Logical System' },
    icon: <Target size={20} color="#00d2ff" />,
    accent: '#f59e0b',
    items: [
      {
        title: { zh: '条件推理', en: 'Conditional Reasoning' },
        body: { zh: '基于条件的分析和推理能力', en: 'Analysis and reasoning based on conditions.' },
      },
      {
        title: { zh: '受限组合问题', en: 'Bounded Combinatorics' },
        body: {
          zh: '解决知识网络中的受限无穷组合问题',
          en: 'Solving bounded infinite combinatorial problems in knowledge networks.',
        },
      },
      {
        title: { zh: '核心能力', en: 'Core Capabilities' },
        body: {
          zh: '深度思考、翻译、问题解决等能力都需要逻辑体系参与。',
          en: 'Deep thinking, translation, and problem solving all depend on logical structure.',
        },
      },
      {
        title: { zh: '全局唯一性', en: 'Global Uniqueness' },
        body: {
          zh: '所有神经元参与运算，但每次都能生成合适的词，说明语言中可能存在具有数学性质的全局唯一性。',
          en: 'All neurons compute, yet the system converges to suitable words; this suggests a global mathematical constraint.',
        },
      },
    ],
  },
  {
    title: { zh: '多维度体系', en: 'Multi-Dimensional System' },
    icon: <Layers size={20} color="#00d2ff" />,
    accent: '#10b981',
    items: [
      {
        title: { zh: '风格维度', en: 'Style Dimension' },
        body: {
          zh: '控制输出的风格和语调，如聊天式、论文式等',
          en: 'Controls output style and tone, such as chat-like or academic writing.',
        },
      },
      {
        title: { zh: '逻辑维度', en: 'Logic Dimension' },
        body: {
          zh: '管理上下文的逻辑关系和连贯性',
          en: 'Manages logical relationships and coherence in context.',
        },
      },
      {
        title: { zh: '语句维度', en: 'Sentence Dimension' },
        body: {
          zh: '处理语法结构和句子组织',
          en: 'Handles grammatical structure and sentence organization.',
        },
      },
      {
        title: { zh: '脉冲效率原理', en: 'Spiking Efficiency Principle' },
        body: {
          zh: '最小传送量原理可能同时支持及时学习和全局稳态，研究重点是脉冲神经网络的 3D 空间拓扑网络结构。',
          en: 'Minimal transmission may support both real-time learning and global stability through 3D topological structure.',
        },
      },
    ],
  },
];

function tr(value, lang) {
  if (typeof value === 'string') return value;
  return value?.[lang] || value?.zh || '';
}

function FrameworkColumn({ column, lang }) {
  return (
    <div style={{
      background: 'rgba(0,0,0,0.3)',
      borderRadius: '12px',
      padding: '20px',
      border: '1px solid rgba(0, 210, 255, 0.1)',
    }}>
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        marginBottom: '16px',
      }}>
        {column.icon}
        <h3 style={{
          fontSize: '16px',
          fontWeight: 'bold',
          color: '#fff',
          margin: 0,
        }}>
          {tr(column.title, lang)}
        </h3>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
        {column.items.map((item) => (
          <div key={tr(item.title, 'zh')}>
            <div style={{
              fontSize: '12px',
              fontWeight: 'bold',
              color: column.accent,
              marginBottom: '6px',
            }}>
              {tr(item.title, lang)}
            </div>
            <div style={{ fontSize: '13px', color: '#ccc', lineHeight: '1.5' }}>
              {tr(item.body, lang)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * 语言分析标签页
 * 保留语言核心框架模块。
 */
export const LanguageAnalysisTab = ({ lang = 'zh' }) => {
  const isEnglish = lang === 'en';

  return (
    <div style={{ color: '#fff' }}>
      <div style={{
        background: 'linear-gradient(135deg, rgba(0, 210, 255, 0.1), rgba(138, 43, 226, 0.1))',
        borderRadius: '16px',
        padding: '24px',
        marginBottom: '24px',
        border: '1px solid rgba(0, 210, 255, 0.2)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', marginBottom: '12px' }}>
          <BookOpen size={32} color="#00d2ff" />
          <h1 style={{
            fontSize: '28px',
            fontWeight: 'bold',
            margin: 0,
            color: '#fff',
          }}>
            {isEnglish ? 'Language Analysis' : '语言分析'}
          </h1>
        </div>
        <p style={{
          fontSize: '14px',
          color: '#888',
          margin: 0,
          lineHeight: '1.6',
        }}>
          {isEnglish
            ? 'Reverse-engineering the mathematical structure and encoding mechanisms of language in deep neural networks'
            : '深入分析语言的数学结构特性与编码机制，逆向工程深度神经网络'}
        </p>
      </div>

      <div style={{
        background: 'rgba(0, 50, 100, 0.2)',
        borderRadius: '16px',
        padding: '24px',
        border: '1px solid rgba(0, 210, 255, 0.2)',
        marginBottom: '28px',
      }}>
        <h2 style={{
          fontSize: '20px',
          fontWeight: 'bold',
          color: '#00d2ff',
          marginBottom: '20px',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
        }}>
          <Brain size={24} />
          {isEnglish ? 'Language Core Framework' : '语言核心框架'}
        </h2>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '24px' }}>
          {frameworkColumns.map((column) => (
            <FrameworkColumn key={tr(column.title, 'zh')} column={column} lang={lang} />
          ))}
        </div>

        <div style={{
          marginTop: '20px',
          padding: '16px',
          background: 'rgba(0, 210, 255, 0.1)',
          borderRadius: '10px',
          border: '1px solid rgba(0, 210, 255, 0.3)',
        }}>
          <div style={{
            fontSize: '14px',
            fontWeight: 'bold',
            color: '#00d2ff',
            marginBottom: '8px',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
          }}>
            <Sparkles size={18} />
            {isEnglish ? 'Core Objective & Unified Hypothesis' : '核心目标与统一假说'}
          </div>
          <div style={{
            fontSize: '14px',
            color: '#fff',
            lineHeight: '1.7',
            marginBottom: '10px',
          }}>
            {isEnglish ? (
              <>
                Analyze the unified <strong style={{ color: '#00d2ff' }}>encoding mechanism</strong> behind these characteristics and understand how it forms at the <strong style={{ color: '#00d2ff' }}>neuron</strong> and <strong style={{ color: '#00d2ff' }}>parameter</strong> level.
              </>
            ) : (
              <>
                分析以上特性背后的统一<strong style={{ color: '#00d2ff' }}>编码机制</strong>，研究其在<strong style={{ color: '#00d2ff' }}>神经元</strong>和<strong style={{ color: '#00d2ff' }}>参数级别</strong>是如何形成的。
              </>
            )}

             {isEnglish ? (
              <>
                <strong style={{ color: '#00d2ff' }}>Unified Hypothesis:</strong> Knowledge networks, logical systems, multidimensional control, vector arithmetic, spiking encoding, and global uniqueness are all results of the same <strong style={{ color: '#00d2ff' }}>encoding mechanism</strong>. This mechanism forms at the neuron and parameter level, and the core goal is to analyze its mathematical structure.
              </>
            ) : (
              <>
                <strong style={{ color: '#00d2ff' }}>统一假说:</strong> 知识网络、逻辑体系、多维度、向量算术、脉冲编码、全局唯一性等所有特性，都是同一套<strong style={{ color: '#00d2ff' }}>编码机制</strong>的结果。这套机制在神经元和参数级别形成，核心目标是分析其数学结构。
              </>
            )}
          </div>

        </div>
      </div>

      <div style={{
        background: 'rgba(0, 50, 100, 0.2)',
        borderRadius: '16px',
        padding: '24px',
        border: '1px solid rgba(0, 210, 255, 0.2)',
      }}>
        <AtlasControlDashboard lang={lang} />
      </div>
    </div>
  );
};
