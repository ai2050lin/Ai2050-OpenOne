/* 面板级错误边界：一个面板抛错只替换该面板，不让整个工作台白屏（system_audit_20261031 P0-4）。
   错误写进本机诊断记录（diagnostics.js，配置 → 连接与诊断可导出）；resetKey 变化（切换页面）时自动复位。 */
import { Component } from 'react';

import { recordError } from '../../diagnostics.js';

export default class PanelBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null, key: props.resetKey };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  static getDerivedStateFromProps(props, state) {
    return props.resetKey !== state.key ? { error: null, key: props.resetKey } : null;
  }

  componentDidCatch(error, info) {
    try { recordError('panel', `${this.props.name || '面板'}：${error?.message || error}`, info?.componentStack || ''); } catch { /* 诊断不可写 */ }
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="wb9-crash" role="alert">
        <b>「{this.props.name || '面板'}」出错，已单独停下</b>
        <span>{String(this.state.error?.message || this.state.error)}</span>
        <small>其他面板不受影响；错误已写进本机诊断记录（配置 → 连接与诊断）。</small>
        <button type="button" className="fw-tbtn" onClick={() => this.setState({ error: null })}>重试</button>
      </div>
    );
  }
}
