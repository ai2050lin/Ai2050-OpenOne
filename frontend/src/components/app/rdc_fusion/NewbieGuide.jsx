/* NewbieGuide · 新手从这里开始（P1 迁移）：原脉络透镜「使用说明」tab 的三张卡原样迁入总览折叠区。
   - TutorialCard：四步接入分布式研发（数据 ← TUTORIAL_STEPS，命令可复制）
   - TutorialPathsCard：按技术类别上手路径（数据 ← TUTORIAL_PATHS，差异在第 4 步 --kit）
   - TutorialDocCard：平台介绍 + 文档索引 + node_agent 子命令速查（数据 ← USAGE_DOC）
   定位：入门引导属外部门面（总览），不占工作轴界面；默认折叠。零实验字面量：全部内容来自协议词表。 */
import { useState } from 'react';
import { TUTORIAL_STEPS, TUTORIAL_PATHS, USAGE_DOC, TECH_CATEGORIES } from './distributedData.js';

const catOf = id => TECH_CATEGORIES.find(c => c.id === id) || {};

function TutorialCard(){
  const [copied,setCopied]=useState(-1);
  const copy=(cmd,i)=>{ try{ navigator.clipboard.writeText(cmd); }catch(e){} setCopied(i); setTimeout(()=>setCopied(-1),1400); };
  return (
    <div className="fw-pl-card fw-tut">
      <h6>新人教程 · 四步接入分布式研发 <span>从零参与到产出上传与本地复现 · 命令可复制</span></h6>
      <div className="fw-tut-steps">
        {TUTORIAL_STEPS.map((s,i)=>(
          <div key={s.n} className="fw-tut-step">
            <span className="fw-tut-n">{s.n}</span>
            <div className="fw-tut-main">
              <b>{s.t}</b>
              <div className="fw-tut-cmd">
                <code>{s.cmd}</code>
                <button type="button" className="fw-tbtn" onClick={()=>copy(s.cmd,i)}>{copied===i?'✓ 已复制':'复制'}</button>
              </div>
              <p>{s.d}</p>
            </div>
            {i<TUTORIAL_STEPS.length-1&&<span className="fw-tut-arrow">→</span>}
          </div>
        ))}
      </div>
      <p className="fw-tut-note">占位符替换：<span className="fw-mono">&lt;center&gt;</span> 中心节点地址 · <span className="fw-mono">&lt;gpu&gt;</span> GPU 型号 · <span className="fw-mono">&lt;model&gt;</span> 目标模型。全流程纪律：预注册冻结（design_sha 不符拒收）、内容寻址（sha256）、口径版本强制展示。</p>
    </div>
  );
}

function TutorialPathsCard(){
  return (
    <div className="fw-pl-card fw-tut">
      <h6>按技术类别上手 <span>四类技术 × 四条路径 · 骨架同上方四步，差异在第 4 步 --kit</span></h6>
      {TUTORIAL_PATHS.map(p=>{
        const c=catOf(p.cat);
        return (
          <div key={p.cat} className="fw-path-row" title={c.note||''}>
            <span className="fw-path-dot" style={{background:c.color}}/>
            <b>{c.name||p.cat}</b>
            <code className="fw-mono">--kit {p.kit}</code>
            <span className="dim2">{p.note}</span>
          </div>
        );
      })}
      <p className="fw-tut-note">类别状态实时看「进展透镜 · 平台」技术成熟度卡组；缺口任务（哪个模板 × 哪类技术还是空格）在研发透镜领取。</p>
    </div>
  );
}

function TutorialDocCard(){
  return (
    <>
      <div className="fw-pl-card fw-tut">
        <h6>平台是什么 <span>一分钟了解这套分布式研发体系</span></h6>
        <p className="fw-tut-note" style={{marginTop:0,fontSize:10.5,lineHeight:1.85}}>{USAGE_DOC.intro}</p>
      </div>
      <div className="fw-pl-card fw-tut">
        <h6>文档索引 <span>深入阅读 · 均为仓库内真实文件与端点</span></h6>
        <div className="fw-reg-xd" style={{gap:0}}>
          {USAGE_DOC.docs.map(d=>(
            <div key={d.t} className="fw-reg-item" style={{display:'block',padding:'8px 2px'}} title={d.s}>
              <div className="fw-reg-hd"><b>{d.t}</b><span className="fw-mono dim2">{d.s}</span></div>
              <div style={{marginTop:4,fontSize:10,color:'var(--fw-text-2)',lineHeight:1.65}}>{d.d}</div>
            </div>
          ))}
        </div>
      </div>
      <div className="fw-pl-card fw-tut">
        <h6>终端命令速查 <span>node_agent.py 五个子命令 · --server 为顶层参数置于子命令前</span></h6>
        <div className="fw-reg-xd">
          {USAGE_DOC.cli.map(([c,d])=>(
            <div key={c}><i><span className="fw-mono" style={{color:'var(--fw-blue)'}}>node_agent.py {c}</span></i><span>{d}</span></div>
          ))}
        </div>
        <p className="fw-tut-note">凭据纪律：<span className="fw-mono">node_credentials.json</span> 仅存本机，浏览器不读取——本机精确状态看终端 <span className="fw-mono">status</span>，浏览器内只展示公开队列快照。</p>
      </div>
    </>
  );
}

export default function NewbieGuide(){
  const [open,setOpen]=useState(false);
  return (
    <div style={{gridColumn:'1 / -1'}}>
      <button type="button" className="fw-tbtn" style={{width:'100%',justifyContent:'flex-start',gap:10,padding:'10px 14px'}}
              onClick={()=>setOpen(o=>!o)}>
        <span>{open?'▾':'▸'}</span>
        <b style={{fontSize:13}}>新手从这里开始</b>
        <span style={{fontSize:11.5,color:'var(--fw-text-3,#94a3b8)',fontWeight:400}}>
          平台是什么 · 四步接入分布式研发 · 按技术类别上手路径 · 文档索引 · node_agent 命令速查
        </span>
      </button>
      {open && (
        <div style={{display:'flex',flexDirection:'column',gap:10,marginTop:10}}>
          <TutorialCard/>
          <TutorialPathsCard/>
          <TutorialDocCard/>
        </div>
      )}
    </div>
  );
}
