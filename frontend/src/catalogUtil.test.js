/* catalogUtil.claimCommand 单测：占位中心地址替换为当前连接的中心。 */
import { describe, it, expect } from 'vitest';
import { claimCommand, STATUS_META, STATUS_ORDER } from './components/app/rdc_fusion/catalogUtil.js';

describe('claimCommand', () => {
  it('替换占位中心地址为 API_BASE', () => {
    const item = { claim_command: 'python deploy/node_agent.py --server http://<center>:5010 run --tm-id X' };
    const cmd = claimCommand(item);
    expect(cmd).not.toContain('<center>');
    expect(cmd).toContain('http://localhost:5001');
  });

  it('没有占位符时原样返回', () => {
    const item = { claim_command: 'python agent.py --server http://real.host run' };
    expect(claimCommand(item)).toBe(item.claim_command);
  });

  it('空命令返回空串', () => {
    expect(claimCommand({})).toBe('');
  });
});

describe('状态元数据一致性', () => {
  it('STATUS_ORDER 与 STATUS_META 键一致，均含待测试语义', () => {
    expect([...STATUS_ORDER].sort()).toEqual(Object.keys(STATUS_META).sort());
    expect(STATUS_META.open[1]).toBe('待测试');
    expect(STATUS_META.draft[1]).toBe('未冻结');
  });
});
