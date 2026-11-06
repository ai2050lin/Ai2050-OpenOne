/* 研究工作台的「?」页面说明：按当前一级界面（进展 / 空间 / 研发 / 配置 / 本机工具）弹出功能与原理说明。
   内容只描述本仓库已实现的页面与数据流（workbenchUtil.js、RouteLens.jsx、SpaceLens.jsx 与 rdc_fusion 各面板），
   不描述尚未实现的功能；证据等级、协议纪律等表述与 AGENTS.md / design 文档保持一致。 */
import { useEffect } from 'react';
import {ROUTE_VIEWS,SPATIAL_MODES,STATIONS} from './workbenchUtil.js';
import {CFG_SECTIONS} from './ConfigView.jsx';

const HELP = {
  route:{title:'进展 · 功能与原理',intro:'总览按外部语言图谱、内部模型图谱和两者的关联与验证组织。点击卡片查看具体材料、模型快照和证据，冻结分片在阶段任务中单独查看。',sections:[]},
  spatial:{title:'空间 · 功能与原理',intro:'图谱浏览连接外部语言模式、内部特征和关联证据；结构与参数定位真实对象；运行与路径观察单一样本；响应对照比较多样本。',sections:[{h:'对象和证据',points:['图谱使用语言模式、内部特征和共享映射图层；数据集是证据来源，不复制成独立知识本体。','原生单元与真实参数在同一二级入口下切换。不同模型快照、采集边界或位置不能自动绑定。','3D距离是索引布局，不表示语义距离；参数来源、样本贡献、统计关联和因果效应分别标注。','即时长文本采集需要本机GPU服务；已有轨迹浏览不占GPU。缺数据时显示原因。']}]},
  process:{title:'研发 · 功能与原理',intro:'当前研发显示进行中的研究、作业的实际完成数和一键参与。材料与数据可查数据集和模板，实验与队列可查任务执行。',sections:[{h:'参与与回查',points:['桌面端按本机条件显示可领取任务。领取后在原位置查看运行进度和回执。','已归档研究可在研究历史查看样本、报告和反例。神经元测试、模板流水线、普查和自动测试在「实验与队列 → 专业工具」。','结果上传、执行审核和科学证据分别记录。研究结论在成果与证据中核对。']}]},
  config:{title:'配置 · 功能与原理',intro:'连接与诊断、账号与权限、本机算力、模型与AI分别管理运行条件。顶栏贡献算力直达本机节点。',sections:[{h:'参与和边界',points:['本机算力保留设备准备、资源预算、模型核验、开始/暂停/停止、托管与日志。','远程控制仅限本人已经授权的节点；本机停止后的授权不能由远端重新开启。','研究被测模型与AI助手/评审模型用途不同；AI解释仍是未检验假设。','高级兼容工具包含历史实测和明确标注的教学演示，演示不进入研究统计。']}]},
};

export default function WorkbenchHelp({ lens, onClose }) {
  const original=HELP[lens]||HELP.route;
  const items=lens==='route'?ROUTE_VIEWS:lens==='spatial'?SPATIAL_MODES:lens==='process'?STATIONS.filter(s=>['plan','catalog','execution'].includes(s.id)):CFG_SECTIONS;
  const help={...original,sub:items.map(x=>x.label).join(' / '),sections:[{h:'页面入口',points:items.map(x=>`${x.label}：${x.desc}`)},...original.sections]};
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
