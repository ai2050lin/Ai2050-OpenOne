# -*- coding: utf-8 -*-
"""2026-10-07 日志追加：git 推送远端 + .workbuddy 整体忽略（append-only）"""
import io, os, hashlib

LOG = r'D:\AI2050\Ai2050-OpenOne\.workbuddy\memory\2026-10-07.md'
text = u"""

## 推送 GitHub 远端 + .workbuddy 整体忽略（14:2x）

- 任务：提交到 git 远端服务器；用户追加指令「.workbuddy 文件整体忽略」。
- 起点：本地 main 领先 origin/main 5 个提交（0a148e4d3..b77475b6b，含融合页研发/路线透镜 v2/v3 全部改动），工作树干净。
- **远端拒绝 GH001 Large files**：`git rev-list --objects origin/main..main` 定位唯一 >50MB blob = `.workbuddy/tmp_release_tree.tar`（192.2MB，168d6574c 引入，超 GitHub 100MB 硬上限）。
- 处置：`backup-pre-push` 备份分支 → filter-branch `--index-filter git rm --cached` 仅重写 5 个未推送提交，剥离 tar（工作树文件保留磁盘）→ 重写后 a56780735/ca31c6f63，无 >50MB blob。
- **.workbuddy 整体忽略**：`.gitignore` 追加 `/.workbuddy/`；`git rm -r --cached .workbuddy`（117 个文件停止跟踪，磁盘全保留）→ 提交 `637dd6b59`。
- **推送成功**：代理踩坑两处——env 代理 61906 对 git 起初 502、10809 间歇断连且 LFS locks verify Bad Gateway（已 `lfs.locksverify false`）；最终 61906 通。`7839c6e6f..637dd6b59 main -> main`，ls-remote 验证远端 HEAD=637dd6b59 一致。
- 遗留：`backup-pre-push` 分支保留（含 tar 旧历史，确认无虞后可 `git branch -D backup-pre-push`）；设计稿删除类改动在 stash@{0} pre-push-rewrite stash（文件已恢复磁盘）。
- 教训：GH001 大文件拦截要先 `rev-list --objects | cat-file --batch-check` 定位；filter-branch 前工作树必须干净（运行中的服务会造脏）。
"""
with io.open(LOG, 'r', encoding='utf-8') as f:
    old = f.read()
with io.open(LOG, 'a', encoding='utf-8', newline='') as f:
    f.write(text)
with io.open(LOG, 'r', encoding='utf-8') as f:
    new = f.read()
assert new.startswith(old) and len(new) > len(old), 'append failed'
print('LOG APPENDED OK', hashlib.md5(new.encode('utf-8')).hexdigest()[:8])
