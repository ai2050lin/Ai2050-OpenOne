#!/usr/bin/env python3
"""Update the Windows installer link in the static ai2050.com homepage."""

import argparse
from datetime import datetime
from difflib import unified_diff
import os
from pathlib import Path
import re
import stat
import tempfile


RELEASE = 'https://github.com/ai2050lin/Ai2050-OpenOne/releases/download/'
CENTER = '<a class="primary-button" href="https://server.ai2050.com/">进入研究中心</a>'
DOWNLOAD = re.compile(
    r'<a\b(?=[^>]*(?:\bid="desktop-client-download"|\bhref="'
    + re.escape(RELEASE) + r'desktop-v[0-9.]+/AI2050-Setup-[0-9.]+-x64\.exe"))[^>]*>[^<]*</a>'
)


def update_html(html, version):
    url = f'{RELEASE}desktop-v{version}/AI2050-Setup-{version}-x64.exe'
    button = (f'<a id="desktop-client-download" class="ghost-button" href="{url}" '
              f'aria-label="下载 Windows x64 客户端 {version}">下载客户端 {version}</a>')
    current = list(DOWNLOAD.finditer(html))
    if len(current) == 1:
        match = current[0]
        return html[:match.start()] + button + html[match.end():]
    if current or html.count(CENTER) != 1:
        raise ValueError('无法唯一识别 AI2050 首页的下载入口')
    newline = '\r\n' if '\r\n' in html else '\n'
    return html.replace(CENTER, CENTER + newline + '            ' + button, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('index', type=Path, help='官网实际静态目录下的 index.html')
    parser.add_argument('version', help='GitHub 已发布的桌面版本，例如 1.0.7')
    parser.add_argument('--apply', action='store_true', help='备份并写入。默认只预览差异')
    args = parser.parse_args()
    if not re.fullmatch(r'\d+\.\d+\.\d+', args.version):
        parser.error('版本格式应为 1.0.7')
    target = args.index.resolve(strict=True)
    original = target.read_bytes()
    html = original.decode('utf-8')
    updated = update_html(html, args.version)
    if updated == html:
        print('下载入口已是指定版本。')
        return
    if not args.apply:
        print('\n'.join(unified_diff(html.splitlines(), updated.splitlines(), str(target), str(target) + ' (更新后)')))
        return

    info = target.stat()
    backups = Path.home() / '.ai2050-website-backups'
    backups.mkdir(mode=0o700, parents=True, exist_ok=True)
    backup = backups / f'index-{datetime.now():%Y%m%dT%H%M%S%f}.html'
    with backup.open('xb') as file:
        file.write(original)
    with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.index-', delete=False) as file:
        temporary = Path(file.name)
        try:
            file.write(updated.encode('utf-8'))
            file.close()
            if hasattr(os, 'geteuid') and os.geteuid() == 0:
                os.chown(temporary, info.st_uid, info.st_gid)
            os.chmod(temporary, stat.S_IMODE(info.st_mode))
            if target.read_bytes() != original:
                raise RuntimeError('首页已被其他进程更新，请重新预览后再执行')
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    print(f'已更新 {target}\n备份 {backup}')


if __name__ == '__main__':
    main()
