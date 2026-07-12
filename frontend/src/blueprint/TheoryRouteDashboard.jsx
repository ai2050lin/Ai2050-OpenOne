import React, { useState } from 'react';
import { AlertTriangle, CheckCircle2, ChevronDown, FlaskConical, Route, Sigma, Target } from 'lucide-react';
import { THEORY_ROUTE_DATA } from './theoryRouteLatestData';

const Section = ({ title, icon: Icon, color, children }) => (
    <section style={{ borderTop: `1px solid ${color}35`, paddingTop: '16px' }}>
        <h3 style={{ display: 'flex', alignItems: 'center', gap: '8px', color: '#e2e8f0', fontSize: '14px', margin: '0 0 12px' }}>
            {React.createElement(Icon, { size: 16, color, 'aria-hidden': true })}{title}
        </h3>
        {children}
    </section>
);

const Expandable = ({ title, meta, status, children, color }) => {
    const [open, setOpen] = useState(false);
    return (
        <div style={{ borderBottom: '1px solid rgba(148,163,184,0.13)' }}>
            <button type="button" onClick={() => setOpen((value) => !value)} aria-expanded={open} style={{ width: '100%', minHeight: '54px', padding: '10px 4px', border: 0, background: 'transparent', color: '#e2e8f0', cursor: 'pointer', display: 'grid', gridTemplateColumns: 'minmax(0,1fr) auto auto', gap: '12px', alignItems: 'center', textAlign: 'left' }}>
                <span><strong style={{ display: 'block', fontSize: '13px' }}>{title}</strong><small style={{ color: '#64748b' }}>{meta}</small></span>
                <span style={{ color, fontSize: '11px', border: `1px solid ${color}55`, padding: '3px 7px', borderRadius: '4px' }}>{status}</span>
                <ChevronDown size={16} style={{ transform: open ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
            </button>
            {open && <div style={{ margin: '0 4px 12px', padding: '12px', background: 'rgba(15,23,42,0.62)', borderLeft: `2px solid ${color}`, color: '#cbd5e1', fontSize: '12px', lineHeight: 1.75 }}>{children}</div>}
        </div>
    );
};

export const TheoryRouteDashboard = ({ routeId }) => {
    const data = THEORY_ROUTE_DATA[routeId] || THEORY_ROUTE_DATA.GPT5;
    return (
        <div style={{ display: 'grid', gap: '22px' }}>
            <section style={{ padding: '20px', border: `1px solid ${data.tone}45`, borderRadius: '8px', background: 'rgba(15,23,42,0.58)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '18px', flexWrap: 'wrap', marginBottom: '12px' }}>
                    <div><div style={{ color: data.tone, fontSize: '12px', fontWeight: 700 }}>{data.phase} · {data.date}</div><h2 style={{ color: '#f8fafc', fontSize: '19px', margin: '5px 0 0' }}>{data.name}</h2></div>
                    <div style={{ maxWidth: '440px', color: '#94a3b8', fontSize: '11px', lineHeight: 1.6, textAlign: 'right' }}>依据：{data.source}</div>
                </div>
                <p style={{ color: '#dbeafe', fontSize: '13px', lineHeight: 1.8, margin: '0 0 10px' }}>{data.thesis}</p>
                <div style={{ color: '#f8fafc', background: `${data.tone}12`, borderLeft: `3px solid ${data.tone}`, padding: '9px 11px', fontSize: '12px' }}><strong>当前判断：</strong>{data.verdict}</div>
            </section>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', borderTop: '1px solid rgba(148,163,184,0.18)', borderBottom: '1px solid rgba(148,163,184,0.18)' }}>
                {data.metrics.map(([label, value, detail]) => <div key={label} title={detail} style={{ minHeight: '88px', padding: '14px', borderRight: '1px solid rgba(148,163,184,0.12)' }}><div style={{ color: '#94a3b8', fontSize: '11px' }}>{label}</div><div style={{ color: data.tone, fontSize: '22px', fontWeight: 760, margin: '4px 0' }}>{value}</div><div style={{ color: '#64748b', fontSize: '10px', lineHeight: 1.45 }}>{detail}</div></div>)}
            </div>

            <Section title="最新推进链" icon={Route} color={data.tone}>
                {data.stages.map((stage) => <Expandable key={stage.title} title={stage.title} meta={stage.phase} status={stage.status} color={data.tone}><div>{stage.detail}</div><div style={{ color: '#64748b', marginTop: '7px' }}>来源：{data.source} · {stage.source}</div></Expandable>)}
            </Section>

            <Section title="当前核心公式" icon={Sigma} color={data.tone}>
                <button type="button" onClick={(event) => event.currentTarget.nextElementSibling?.toggleAttribute('hidden')} style={{ width: '100%', border: '1px solid rgba(148,163,184,0.2)', borderRadius: '6px', background: 'rgba(15,23,42,0.5)', padding: '13px', color: '#e2e8f0', cursor: 'pointer', textAlign: 'left' }}><strong>{data.formula.title}</strong><div style={{ marginTop: '9px', color: data.tone, fontFamily: 'ui-monospace, SFMono-Regular, Consolas, monospace', overflowWrap: 'anywhere' }}>{data.formula.expression}</div></button>
                <div hidden style={{ padding: '12px 14px', color: '#cbd5e1', fontSize: '12px', lineHeight: 1.75, borderBottom: '1px solid rgba(148,163,184,0.15)' }}><div>{data.formula.explanation}</div><div style={{ color: '#fbbf24', marginTop: '7px' }}>证据边界：{data.formula.boundary}</div></div>
            </Section>

            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(270px, 1fr))', gap: '20px' }}>
                <Section title="已确认" icon={CheckCircle2} color="#34d399"><ul style={{ margin: 0, paddingLeft: '18px', color: '#cbd5e1', fontSize: '12px', lineHeight: 1.8 }}>{data.confirmed.map((item) => <li key={item}>{item}</li>)}</ul></Section>
                <Section title="硬伤与边界" icon={AlertTriangle} color="#f59e0b"><ul style={{ margin: 0, paddingLeft: '18px', color: '#cbd5e1', fontSize: '12px', lineHeight: 1.8 }}>{data.blockers.map((item) => <li key={item}>{item}</li>)}</ul></Section>
            </div>
            <Section title="下一轮可证伪任务" icon={Target} color="#a78bfa"><div style={{ display: 'grid', gap: '8px' }}>{data.next.map((item, index) => <div key={item} style={{ display: 'grid', gridTemplateColumns: '26px minmax(0,1fr)', gap: '8px', color: '#cbd5e1', fontSize: '12px', lineHeight: 1.65 }}><span style={{ color: '#a78bfa', fontWeight: 700 }}>P{index}</span><span>{item}</span></div>)}</div></Section>
            <div style={{ display: 'flex', alignItems: 'center', gap: '7px', color: '#64748b', fontSize: '10px' }}><FlaskConical size={13} />页面只陈列研究记录中的当前证据；工作公式与已验证结论已明确分开。</div>
        </div>
    );
};
