/* RegistryPanel · 注册表面板（P2 迁移）：原脉络透镜「研发平台」tab 的词表资产面整体迁入图谱透镜「注册表」tab。
   职责（知识轴 · 协议资产）：
   - RegistryCards：语言模板注册表 × 分析技术注册表 + 研究包组合（LIVE=GET /api/kits，available 行可下载整包）
   - MaturityCards：四类分析技术 × 平台状态机（与 Q04/Q06「装置建成→正式测量」同构）
   注：节点 / 模板矩阵 / AGG 管线（运行时）不在本面板——已迁研发透镜「平台运行」tab。 */
import { API_BASE } from '../../../config.js';
import { useEffect, useState } from 'react';
import { LANG_TEMPLATES, ANALYSES, RESEARCH_KITS, TECH_CATEGORIES, DEMO_COVERAGE } from './distributedData.js';


/* M7-P1 技术类别查找（PROJECTIONS/era cat → 名称/颜色/短名，数据 ← TECH_CATEGORIES） */
const catOf=id=>TECH_CATEGORIES.find(c=>c.id===id)||{};
const catColor=id=>(catOf(id).color)||'#888780';
const catShort=id=>(catOf(id).short)||id;

/* 平台进度 · 技术成熟度卡组（M7-P1）：四类技术 × 平台状态机（与 Q04/Q06「装置建成→正式测量」同构）
   数据 ← TECH_CATEGORIES（类状态/算力）+ ANALYSES（各类注册数/可用数）+ DEMO_COVERAGE（覆盖格） */
function MaturityCards(){
  return (
    <div className="fw-mature-grid">
      {TECH_CATEGORIES.map(c=>{
        const techs=ANALYSES.filter(a=>a.category===c.id);
        const avail=techs.filter(t=>t.status==='available').length;
        const done=DEMO_COVERAGE.filter(x=>x.cat===c.id&&x.status==='done').length;
        return (
          <div key={c.id} className="fw-pl-card fw-mature-card" style={{borderTop:'2px solid '+c.color}} title={c.note}>
            <div className="fw-mature-hd">
              <b style={{color:c.color}}>{c.name}</b>
              <span className={'fw-tmx-agg '+(c.status==='measured'?'done':c.status==='device_built'?'collecting':'todo')}>
                {c.status==='measured'?'已测':(c.status==='device_built'?'装置已建':'装置待建')}
              </span>
            </div>
            <p className="fw-mature-note">{c.note}</p>
            <div className="fw-mature-ms">{c.methods.join(' · ')}</div>
            <div className="fw-mature-kv">
              <span>注册 {techs.length} 项</span><span>可用 {avail}</span><span>覆盖 {done}/{DEMO_COVERAGE.length} 格</span>
            </div>
            <div className="fw-mature-cost dim2">算力参考：{c.cost} · 契约 {c.input}</div>
          </div>
        );
      })}
    </div>
  );
}

/* 平台进度 · 双注册表卡（M6-P0/P1）：语言模板（LANG_TEMPLATES）× 分析技术（ANALYSES）+ 研究包
   研究包组合：离线=RESEARCH_KITS DEMO 叙事；LIVE=GET /api/kits（服务端 S7，tm×analysis），
   available 行可点击下载整包（bundle JSON：corpus/runner/contract + 结果清单 + README） */
function RegistryCards(){
  const [open,setOpen]=useState(null);   /* 展开的语言模板 id */
  const [kits,setKits]=useState(null);   /* LIVE：服务端 S7 研究包清单 */
  const [dlIng,setDlIng]=useState('');
  useEffect(()=>{
    let dead=false;
    fetch(`${API_BASE}/api/kits`).then(r=>r.ok?r.json():null).catch(()=>null)
      .then(d=>{ if(!dead&&d&&d.kits) setKits(d); });
    return ()=>{dead=true;};
  },[]);
  const downloadKit=async(kid)=>{
    setDlIng(kid);
    try{
      const r=await fetch(`${API_BASE}/api/kits/${kid}/bundle`);
      if(!r.ok) throw new Error('HTTP '+r.status);
      const b=await r.json();
      const blob=new Blob([JSON.stringify(b,null,1)],{type:'application/json'});
      const a=document.createElement('a');
      a.href=URL.createObjectURL(blob); a.download=`${kid}.bundle.json`; a.click();
      URL.revokeObjectURL(a.href);
    }catch(e){
      window.alert(`研究包 ${kid} 下载失败（${e.message}）——需中心节点已重启加载 S7 端点，或该 kit 尚无满足契约的结果。`);
    }finally{ setDlIng(''); }
  };
  const tplOf=id=>LANG_TEMPLATES.find(t=>t.id===id);
  return (
    <div className="fw-reg-row">
      <div className="at-note" style={{marginBottom:8}}>
        <b>定位：数据产线资产</b>——语言模板 × 分析技术的生产词表（模板分发任意机器、技术按输入契约匹配）；
        特征证据图谱（FTR-xx / E 级 / 域树）在「图谱全景」，此处不承载特征知识。
      </div>
      <div className="fw-pl-card">
        <h6>语言模板注册表 <span>{LANG_TEMPLATES.length} 种 · 受控语料+固定口径 · 空间透镜可切换体验</span></h6>
        {LANG_TEMPLATES.map(t=>(
          <div key={t.id} className="fw-reg-item" onClick={()=>setOpen(open===t.id?null:t.id)}>
            <div className="fw-reg-hd">
              <b className="fw-mono">{t.id}</b>
              <span>{t.name}</span>
              <span className={'fw-tmx-agg '+(t.status==='measured'?'done':'todo')}>{t.status==='measured'?'measured · 已测':'demo · 占位'}</span>
            </div>
            {open===t.id&&(
              <div className="fw-reg-xd">
                <div><i>示例 token 序列</i><span className="fw-mono">{t.demo.tokens.join(' ')}</span></div>
                <div><i>焦点特征</i><span className="fw-mono">{t.demo.focus.id} · {t.demo.focus.label}</span></div>
                <div><i>状态注记</i><span>{t.status==='measured'?'E_read=0.331615（TM-07，Q09 基线）':'未测量——不挂任何实测数字，Q20 族扩展后转 measured'}</span></div>
              </div>
            )}
          </div>
        ))}
      </div>
      <div className="fw-pl-card">
        <h6>分析技术注册表 <span>{ANALYSES.length} 种 · 按输入契约匹配可用性 · 口径各自登记</span></h6>
        {ANALYSES.map(a=>(
          <div key={a.id} className="fw-reg-item">
            <div className="fw-reg-hd">
              <b className="fw-mono">{a.id}</b>
              <span>{a.name}</span>
              <span className="fw-era-cat" style={{color:catColor(a.category)}} title={'技术类别：'+((catOf(a.category)||{}).name||a.category)}>{catShort(a.category)}</span>
              <span className="fw-pc-ev mono">{a.evidence_level}</span>
            </div>
            <div className="fw-reg-xd">
              <div><i>技术类别</i><span>{(catOf(a.category)||{}).name||a.category} · {a.status==='available'?'可运行':(a.status==='planned'?'装置待建':a.status)}</span></div>
              <div><i>输入契约</i><span className="fw-mono">{a.input.join(' + ')||'—'}</span></div>
              <div><i>口径版本</i><span className="fw-mono">{a.metric_version}</span></div>
              <div><i>说明</i><span>{a.note}</span></div>
            </div>
          </div>
        ))}
        <h6 style={{marginTop:10}}>
          研究包组合
          <span>{kits ? 'LIVE · 服务端 S7 · available 行点击下载整包' : 'DEMO · 语言模板 × 分析技术（中心节点离线）'}</span>
          {!kits && <span className="fw-src-chip demo" style={{marginLeft:'auto'}}>○ DEMO</span>}
          {kits && <span className="fw-src-chip live" style={{marginLeft:'auto'}}>● LIVE S7</span>}
        </h6>
        {kits ? kits.kits.map(k=>(
          <div key={k.kit_id} className="fw-reg-item"
               onClick={()=>{ if(k.status==='available') downloadKit(k.kit_id); }}
               title={k.status==='available'?'点击下载研究包（bundle JSON：corpus/runner/contract + 结果清单 + README）':'结果库暂无满足该 input 契约的结果'}>
            <div className="fw-reg-hd">
              <b className="fw-mono">{k.kit_id}</b>
              <span className="dim2">{k.analysis_name} · {k.metric_version}</span>
              {k.status==='available'
                ? <button type="button" className="fw-tbtn" style={{marginLeft:'auto',flexShrink:0}}
                          onClick={e=>{e.stopPropagation();downloadKit(k.kit_id);}}
                          disabled={dlIng===k.kit_id}>{dlIng===k.kit_id?'下载中…':'↓ 下载'}</button>
                : <span className={'fw-tmx-agg todo'} style={{marginLeft:'auto',flexShrink:0}}>pending · 等结果</span>}
            </div>
          </div>
        )) : RESEARCH_KITS.map(k=>(
          <div key={k.kit_id} className="fw-reg-item">
            <div className="fw-reg-hd">
              <b className="fw-mono">{tplOf(k.lang_tpl)?tplOf(k.lang_tpl).name:k.lang_tpl}</b>
              <span className="dim2">× {k.analysis}</span>
              <span className={'fw-tmx-agg '+(k.status==='available'?'done':'todo')}>{k.status==='available'?'available · 可下载':'pending · 等结果'}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function RegistryPanel(){
  return (
    <div style={{display:'flex',flexDirection:'column',gap:12}}>
      <RegistryCards/>
      <MaturityCards/>
    </div>
  );
}
