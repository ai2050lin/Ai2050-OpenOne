"""Project disk gate shared by Windows research tools and portable nodes."""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path


POLICY = json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))
GIB = 2 ** 30


class StorageBlocked(RuntimeError):
    def __init__(self, reason: str, path: Path, required: int, detail: str):
        self.reason = reason
        self.path = path
        self.required = required
        super().__init__(f'{detail}（实际路径 {path}，预计新增 {required / GIB:.2f} GiB）。'
                         '请选择有足够空间的项目盘或暂停任务。')


def storage_dir(kind: str, *parts: str) -> Path:
    default = POLICY['windows_root'] if os.name == 'nt' else str(Path.home() / '.ai2050' / 'storage')
    return Path(os.environ.get('AI2050_STORAGE_ROOT') or default).joinpath(POLICY['directories'][kind], *parts)


def check_write(path, required: int = 0, *, operation: str = '项目写入') -> Path:
    """Check the real destination (including junctions) before creating files."""
    if type(required) is not int or required < 0:
        raise ValueError('expected write bytes must be a nonnegative integer')
    requested = Path(path)
    try:
        existing = Path(os.path.abspath(requested))
        missing = []
        while True:
            try:
                existing.lstat()
                break
            except FileNotFoundError:
                if existing == existing.parent:
                    raise
                missing.append(existing.name)
                existing = existing.parent
        existing = existing.resolve(strict=True)
        target = existing.joinpath(*reversed(missing))
        free = shutil.disk_usage(existing if existing.is_dir() else existing.parent).free
    except (OSError, RuntimeError) as exc:
        raise StorageBlocked('unavailable', requested, required, f'{operation}的目标卷不可用') from exc
    drive = target.drive.removeprefix('\\\\?\\').upper()
    if os.name == 'nt' and drive == POLICY['blocked_windows_drive']:
        raise StorageBlocked('system_drive', target, required, f'{operation}将写入 C 盘，已暂停')
    reserve = int(POLICY['windows_reserve_gib' if os.name == 'nt' else 'portable_reserve_gib'] * GIB)
    if free < reserve + required:
        raise StorageBlocked('space', target, required,
                             f'{operation}空间不足，当前 {free / GIB:.2f} GiB，保留线 {reserve / GIB:.2f} GiB')
    return target


def checked_dir(path, required: int = 0, *, operation: str = '项目写入') -> Path:
    directory = check_write(path, required, operation=operation)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def archive_budget(size: int, members: int) -> int:
    """Allow for tar headers, padding, metadata and compression overhead."""
    return size + size // 100 + members * 4096 + 8 * 1024 * 1024


def pack_target(path, size: int, members: int) -> Path:
    target = check_write(path, archive_budget(size, members), operation='打包')
    check_write(target.with_suffix(target.suffix + '.sha256'), operation='校验清单')
    return target


@contextmanager
def staging_file(target: Path):
    """Publish a complete package only after its writer closes successfully."""
    fd, name = tempfile.mkstemp(prefix=f'.{target.name}.', suffix='.partial', dir=target.parent)
    os.close(fd)
    partial = Path(name)
    try:
        yield partial
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)


def process_env(work: Path, *, base=None) -> dict:
    """Route library caches and OS temp for this child only."""
    scratch = checked_dir(work / 'tmp', operation='子进程临时文件')
    reusable = storage_dir('reusable')
    env = dict(os.environ if base is None else base)
    env.update(TEMP=str(scratch), TMP=str(scratch), TMPDIR=str(scratch),
               HF_HOME=str(check_write(reusable / 'huggingface', operation='复用缓存')),
               TORCH_HOME=str(check_write(reusable / 'torch', operation='复用缓存')),
               XDG_CACHE_HOME=str(check_write(reusable / 'xdg', operation='复用缓存')),
               UV_CACHE_DIR=str(check_write(reusable / 'uv', operation='复用缓存')),
               PYTHONDONTWRITEBYTECODE='1')
    # Explicit library overrides could otherwise send this project's cache to C.
    for key, leaf in [('HF_HUB_CACHE', 'hub'), ('HUGGINGFACE_HUB_CACHE', 'hub'),
                      ('HF_DATASETS_CACHE', 'datasets'), ('HF_MODULES_CACHE', 'modules')]:
        env[key] = str(check_write(reusable / 'huggingface' / leaf, operation='复用缓存'))
    env.pop('TRANSFORMERS_CACHE', None)
    return env


def stop_with_notice(error: StorageBlocked, *, interactive: bool | None = None):
    """A CLI gets a native notice only in an interactive Windows session."""
    if interactive is None:
        interactive = sys.stdin.isatty() and not os.environ.get('CI')
    if os.name == 'nt' and interactive:
        ctypes.windll.user32.MessageBoxW(None, str(error), 'AI2050 磁盘检查', 0x30)
    raise SystemExit(str(error))
