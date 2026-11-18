"""Real two-process MCP integration acceptance; no LLM/GPU experiment.

Run with Basic Memory's Python environment. Writes one retained acceptance note,
checks cross-process visibility, search, exact missing reads and stale-write refusal.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import uuid
from zoneinfo import ZoneInfo

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def unpack(result):
    structured = getattr(result, 'structured_content', getattr(result, 'structuredContent', None))
    if isinstance(structured, dict):
        if isinstance(structured.get('result'), dict):
            return structured['result']
        return structured
    for item in result.content:
        if getattr(item, 'type', None) == 'text':
            try:
                value = json.loads(item.text)
                if isinstance(value, dict):
                    return value
            except (ValueError, TypeError):
                pass
    return {'text': '\n'.join(getattr(item, 'text', '') for item in result.content)}


def tool_error(result):
    return bool(getattr(result, 'is_error', getattr(result, 'isError', False)))


async def run(args):
    root = Path(args.root).resolve()
    stamp = datetime.now(ZoneInfo('America/Chicago')).isoformat(timespec='seconds')
    note = f'inbox/共享知识库验收记录-{uuid.uuid4().hex[:8]}.md'
    marker = 'KB_ACCEPTANCE_' + uuid.uuid4().hex
    appended = 'SECOND_MCP_CLIENT_' + uuid.uuid4().hex
    report = {'timestamp': stamp, 'timezone': 'America/Chicago', 'root': str(root),
              'project': args.project, 'acceptance_note': note, 'checks': [],
              'scope': 'Two real MCP client processes; no client UI or LLM behavior claims',
              'gpu_used': False}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        previous = output.with_name(output.stem + '-previous-' + uuid.uuid4().hex[:6] + '.json')
        previous.write_bytes(output.read_bytes())

    def check(name, condition, **detail):
        report['checks'].append({'check': name, 'passed': bool(condition), **detail})
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'{"PASS" if condition else "FAIL"}: {name}', flush=True)
        if not condition:
            raise AssertionError(name)

    async def call(session, tool, **arguments):
        return await session.call_tool(tool, {'project': args.project, **arguments})

    env = os.environ.copy()
    env['PYTHONUTF8'] = '1'
    params = StdioServerParameters(command=args.command, args=['mcp', '--project', args.project], env=env)
    try:
        with output.with_suffix('.stderr.log').open('w', encoding='utf-8') as log:
            async with AsyncExitStack() as stack:
                sessions = []
                for _ in range(2):
                    reader, writer = await stack.enter_async_context(stdio_client(params, errlog=log))
                    session = await stack.enter_async_context(ClientSession(reader, writer))
                    initialized = await session.initialize()
                    sessions.append(session)
                a, b = sessions
                tools = await a.list_tools()
                names = sorted(tool.name for tool in tools.tools)
                report['tool_names'] = names
                info = getattr(initialized, 'server_info', getattr(initialized, 'serverInfo', None))
                report['server'] = info.model_dump(mode='json')
                check('two independent MCP processes initialized', {'bm_read', 'bm_write', 'bm_edit', 'bm_search'} <= set(names))
                rules = await call(a, 'bm_read', identifier='AI_RULES.md', output_format='json')
                check('existing Chinese rules readable', not tool_error(rules) and 'AI 整理规则' in json.dumps(unpack(rules), ensure_ascii=False))

                content = ('# 共享知识库验收记录\n\n'
                           f'时间：{stamp}（America/Chicago）\n\n'
                           '这是软件读写验收记录，不是科学实验或模型能力证据。\n\n'
                           f'客户端 A 写入标记：{marker}\n\n'
                           '检验两个独立 MCP 进程是否共用同一份 Markdown。\n')
                written = await call(a, 'bm_write', identifier=note, content=content,
                                     title='共享知识库验收记录', output_format='json')
                check('client A writes native Markdown', not tool_error(written) and marker in (root / note).read_text(encoding='utf-8'))
                first = await call(b, 'bm_read', identifier=note, output_format='json')
                first_payload = unpack(first)
                check('client B sees client A write', not tool_error(first) and marker in json.dumps(first_payload, ensure_ascii=False))
                checksum = first_payload.get('checksum')
                check('read provides revision checksum', isinstance(checksum, str) and len(checksum) > 16,
                      payload_keys=sorted(first_payload.keys()))

                edited = await call(b, 'bm_edit', identifier=note, command='append',
                                    content=f'\n客户端 B 追加标记：{appended}\n',
                                    expected_checksum=checksum, output_format='json')
                check('client B appends with revision guard', not tool_error(edited))
                second = await call(a, 'bm_read', identifier=note, output_format='json')
                check('client A sees client B update', not tool_error(second) and appended in json.dumps(unpack(second), ensure_ascii=False))
                before = (root / note).read_bytes()
                stale = await call(a, 'bm_edit', identifier=note, command='append', content='SHOULD_NOT_BE_WRITTEN',
                                   expected_checksum=checksum, output_format='json')
                check('stale revision rejected without file change', tool_error(stale) and (root / note).read_bytes() == before)

                overwrite = await call(a, 'bm_write', identifier=note, content='SHOULD_NOT_OVERWRITE', output_format='json')
                check('implicit overwrite rejected', tool_error(overwrite) and (root / note).read_bytes() == before)
                missing = await call(b, 'bm_read', identifier='notes/missing-' + uuid.uuid4().hex + '.md', output_format='json')
                check('missing note returns explicit error', tool_error(missing))

                search = await call(b, 'bm_search', query=marker, mode='text', output_format='json')
                check('text index finds new shared note', not tool_error(search) and '验收记录' in json.dumps(unpack(search), ensure_ascii=False))
                semantic = await call(a, 'bm_search', query='同一笔记一次只有一个写入者', mode='semantic', output_format='json')
                report['semantic_positive_control'] = unpack(semantic)
                check('Chinese semantic positive control finds rules', not tool_error(semantic) and 'AI_RULES' in json.dumps(unpack(semantic), ensure_ascii=False))
                report['acceptance_note_sha256'] = hashlib.sha256((root / note).read_bytes()).hexdigest()
        report['passed'] = True
    except BaseException as exc:
        report['passed'] = False
        report['error'] = str(exc)
        raise
    finally:
        report['finished_at'] = datetime.now(ZoneInfo('America/Chicago')).isoformat(timespec='seconds')
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'passed': True, 'checks': len(report['checks']), 'report': str(output)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='D:/Knowledge/personal-kb')
    parser.add_argument('--project', default='personal')
    parser.add_argument('--command', default='C:/Users/Admin/.local/bin/basic-memory.exe')
    parser.add_argument('--output', default='D:/AI2050/Ai2050-OpenOne/tests/glm5/result/personal_kb_mcp_20261107.json')
    asyncio.run(run(parser.parse_args()))
