# -*- coding: utf-8 -*-
"""在 checks.yml 的 compatibility-checks job 前插入 frontend job（P0-1：CI 三关）。"""
import io

P = '.github/workflows/checks.yml'
s = io.open(P, encoding='utf-8').read()

anchor = 'jobs:\n  compatibility-checks:'
assert anchor in s, 'anchor missing'

job = """jobs:
  # P0-1（client_architecture_review）：前端此前 0 测试、CI 不构建。vitest + build 为硬门槛；
  # eslint 存量 732 个问题清零后把 continue-on-error 去掉转为硬门槛。
  frontend-checks:
    name: Frontend Checks
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: frontend
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: 20
          cache: npm
          cache-dependency-path: frontend/package-lock.json
      - name: Install dependencies
        run: npm ci
      - name: Unit tests (vitest)
        run: npm test
      - name: Build (vite)
        run: npm run build
      - name: Lint (non-blocking until legacy debt cleared)
        run: npm run lint
        continue-on-error: true

  compatibility-checks:"""

s = s.replace(anchor, job, 1)
io.open(P, 'w', encoding='utf-8', newline='\n').write(s)
print('frontend job inserted')
