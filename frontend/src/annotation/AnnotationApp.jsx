import { useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle,
  ArrowLeft,
  ArrowRight,
  Check,
  CheckCircle2,
  Download,
  FileCheck2,
  Filter,
  Inbox,
  LockKeyhole,
  Menu,
  Search,
  ShieldCheck,
  Upload,
  UserRound,
  X,
  XCircle,
} from 'lucide-react'

const QUEUE_URL = '/data/phase1132/human_review_queue.jsonl'
const MANIFEST_URL = '/data/phase1132/review_manifest.json'
const SESSION_KEY = 'phase1132:reviewer-session:v1'

const JUDGMENTS = [
  {
    key: 'gold_answer_correct',
    label: '建议答案正确',
    description: '根据档案和查询日期，建议答案是否成立？',
  },
  {
    key: 'candidate_unique',
    label: '答案唯一',
    description: '在给定档案中，是否只有一个明确答案？',
  },
  {
    key: 'matched_null_globally_false',
    label: '对照答案全局错误',
    description: '对照候选对当前查询是否明确不成立？',
  },
  {
    key: 'matched_null_locally_plausible',
    label: '对照答案局部合理',
    description: '对照候选是否与实体和关系相符，而非明显无关？',
  },
  {
    key: 'natural_language_acceptable',
    label: '语言表达自然',
    description: '上下文和问题是否清晰、自然、无明显机械错误？',
  },
]

const EMPTY_RESPONSE = Object.freeze({
  gold_answer_correct: null,
  candidate_unique: null,
  matched_null_globally_false: null,
  matched_null_locally_plausible: null,
  natural_language_acceptable: null,
  notes: '',
})

const FILTERS = [
  { key: 'all', label: '全部' },
  { key: 'pending', label: '待审' },
  { key: 'accepted', label: '通过' },
  { key: 'flagged', label: '问题项' },
]

function parseJsonl(text) {
  return text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line, index) => {
      try {
        return JSON.parse(line)
      } catch (error) {
        throw new Error(`JSONL 第 ${index + 1} 行无法解析：${error.message}`)
      }
    })
}

function isComplete(response) {
  return JUDGMENTS.every(({ key }) => typeof response?.[key] === 'boolean')
}

function itemStatus(response) {
  if (!isComplete(response)) return 'pending'
  return JUDGMENTS.every(({ key }) => response[key] === true) ? 'accepted' : 'flagged'
}

function safeFilePart(value) {
  return value.trim().replace(/[^a-zA-Z0-9_-]+/g, '_').replace(/^_+|_+$/g, '') || 'reviewer'
}

function downloadText(text, filename, type = 'application/x-ndjson;charset=utf-8') {
  const blob = new Blob([text], { type })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  anchor.remove()
  URL.revokeObjectURL(url)
}

function getStoredSession() {
  try {
    return JSON.parse(localStorage.getItem(SESSION_KEY) || 'null')
  } catch {
    return null
  }
}

function SetupDialog({ initialId, hasExistingReview, onStart, onCancel }) {
  const [reviewerId, setReviewerId] = useState(initialId || '')
  const [attested, setAttested] = useState(false)
  const normalizedId = reviewerId.trim()

  return (
    <div className="dialog-backdrop" role="presentation">
      <section className="setup-dialog" role="dialog" aria-modal="true" aria-labelledby="setup-title">
        <div className="setup-icon" aria-hidden="true">
          <ShieldCheck size={26} />
        </div>
        <div>
          <p className="eyebrow">Phase 1132</p>
          <h1 id="setup-title">人工盲审登记</h1>
        </div>

        <label className="field-label" htmlFor="reviewer-id">评审者 ID</label>
        <input
          id="reviewer-id"
          className="text-input"
          value={reviewerId}
          onChange={(event) => setReviewerId(event.target.value)}
          placeholder="例如 reviewer-a"
          autoComplete="off"
          maxLength={64}
          autoFocus
        />

        <label className="attestation-row">
          <input
            type="checkbox"
            checked={attested}
            onChange={(event) => setAttested(event.target.checked)}
          />
          <span>我确认未查看任何模型对本材料的输出，并将独立完成判断。</span>
        </label>

        <div className="dialog-note">
          <LockKeyhole size={17} />
          <span>不同评审者必须使用不同 ID，并分别导出评审清单。</span>
        </div>

        <div className="dialog-actions">
          {hasExistingReview && (
            <button type="button" className="button button-quiet" onClick={onCancel}>取消</button>
          )}
          <button
            type="button"
            className="button button-primary"
            disabled={!normalizedId || !attested}
            onClick={() => onStart(normalizedId)}
          >
            <Check size={17} />
            进入评审
          </button>
        </div>
      </section>
    </div>
  )
}

function StatusMark({ status }) {
  if (status === 'accepted') return <CheckCircle2 className="status-icon accepted" size={17} />
  if (status === 'flagged') return <XCircle className="status-icon flagged" size={17} />
  return <span className="status-dot" aria-label="待审" />
}

function AnnotationApp() {
  const storedSession = getStoredSession()
  const [queue, setQueue] = useState([])
  const [manifest, setManifest] = useState(null)
  const [loadError, setLoadError] = useState('')
  const [reviewerId, setReviewerId] = useState(storedSession?.reviewerId || '')
  const [setupOpen, setSetupOpen] = useState(!storedSession?.reviewerId)
  const [responses, setResponses] = useState({})
  const [currentId, setCurrentId] = useState('')
  const [filter, setFilter] = useState('all')
  const [search, setSearch] = useState('')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [toast, setToast] = useState('')
  const [importError, setImportError] = useState('')
  const importRef = useRef(null)

  useEffect(() => {
    let active = true
    Promise.all([
      fetch(QUEUE_URL).then((response) => {
        if (!response.ok) throw new Error(`队列加载失败 (${response.status})`)
        return response.text()
      }),
      fetch(MANIFEST_URL).then((response) => {
        if (!response.ok) throw new Error(`清单加载失败 (${response.status})`)
        return response.json()
      }),
    ])
      .then(([queueText, nextManifest]) => {
        if (!active) return
        const rows = parseJsonl(queueText)
        setQueue(rows)
        setManifest(nextManifest)
        setCurrentId(rows[0]?.item_id || '')
      })
      .catch((error) => active && setLoadError(error.message))
    return () => {
      active = false
    }
  }, [])

  const storageKey = manifest && reviewerId
    ? `phase1132:responses:${manifest.packageSha256}:${reviewerId}`
    : ''

  useEffect(() => {
    if (!storageKey || !queue.length) return
    try {
      const stored = JSON.parse(localStorage.getItem(storageKey) || '{}')
      setResponses(stored)
      const firstPending = queue.find((item) => !isComplete(stored[item.item_id]))
      setCurrentId(firstPending?.item_id || queue[0].item_id)
    } catch {
      setResponses({})
    }
  }, [storageKey, queue])

  useEffect(() => {
    if (!storageKey) return undefined
    const timeout = window.setTimeout(() => {
      localStorage.setItem(storageKey, JSON.stringify(responses))
    }, 120)
    return () => window.clearTimeout(timeout)
  }, [responses, storageKey])

  useEffect(() => {
    if (!toast) return undefined
    const timeout = window.setTimeout(() => setToast(''), 2600)
    return () => window.clearTimeout(timeout)
  }, [toast])

  const currentIndex = Math.max(0, queue.findIndex((item) => item.item_id === currentId))
  const currentItem = queue[currentIndex]
  const currentResponse = responses[currentId] || EMPTY_RESPONSE

  const stats = useMemo(() => {
    const result = { pending: 0, accepted: 0, flagged: 0, completed: 0 }
    queue.forEach((item) => {
      const status = itemStatus(responses[item.item_id])
      result[status] += 1
      if (status !== 'pending') result.completed += 1
    })
    return result
  }, [queue, responses])

  const filteredQueue = useMemo(() => {
    const needle = search.trim().toLowerCase()
    return queue.filter((item) => {
      const statusMatches = filter === 'all' || itemStatus(responses[item.item_id]) === filter
      if (!statusMatches) return false
      if (!needle) return true
      return [item.context, item.query, item.active_candidate, item.matched_null_candidate, item.item_id]
        .some((value) => String(value || '').toLowerCase().includes(needle))
    })
  }, [filter, queue, responses, search])

  const progress = queue.length ? (stats.completed / queue.length) * 100 : 0
  const allComplete = queue.length > 0 && stats.completed === queue.length

  function startSession(nextReviewerId) {
    const session = { reviewerId: nextReviewerId, attested: true }
    localStorage.setItem(SESSION_KEY, JSON.stringify(session))
    setReviewerId(nextReviewerId)
    setSetupOpen(false)
    setSidebarOpen(false)
  }

  function updateCurrent(field, value) {
    if (!currentItem) return
    setResponses((previous) => ({
      ...previous,
      [currentItem.item_id]: {
        ...EMPTY_RESPONSE,
        ...previous[currentItem.item_id],
        [field]: value,
        updated_at: new Date().toISOString(),
      },
    }))
  }

  function selectItem(itemId) {
    setCurrentId(itemId)
    setSidebarOpen(false)
  }

  function moveBy(delta) {
    if (!queue.length) return
    const nextIndex = Math.min(queue.length - 1, Math.max(0, currentIndex + delta))
    setCurrentId(queue[nextIndex].item_id)
  }

  function moveToNextPending() {
    if (!queue.length) return
    for (let offset = 1; offset <= queue.length; offset += 1) {
      const index = (currentIndex + offset) % queue.length
      if (!isComplete(responses[queue[index].item_id])) {
        setCurrentId(queue[index].item_id)
        return
      }
    }
    moveBy(1)
  }

  function exportRows(requireComplete) {
    if (requireComplete && !allComplete) return
    const rows = queue.map((item) => {
      const response = responses[item.item_id] || EMPTY_RESPONSE
      return {
        item_id: item.item_id,
        reviewer_id: reviewerId,
        annotation_blinded_to_model_outputs: true,
        gold_answer_correct: response.gold_answer_correct,
        candidate_unique: response.candidate_unique,
        matched_null_globally_false: response.matched_null_globally_false,
        matched_null_locally_plausible: response.matched_null_locally_plausible,
        natural_language_acceptable: response.natural_language_acceptable,
        notes: response.notes || null,
      }
    })
    const suffix = requireComplete ? 'final' : 'progress'
    const filename = `phase1132_review_${safeFilePart(reviewerId)}_${suffix}.jsonl`
    downloadText(`${rows.map((row) => JSON.stringify(row)).join('\n')}\n`, filename)
    setToast(requireComplete ? '正式评审清单已导出' : '进度文件已下载')
  }

  async function importReview(event) {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return
    setImportError('')
    try {
      const rows = parseJsonl(await file.text())
      const ids = new Set(queue.map((item) => item.item_id))
      const importedReviewerIds = new Set(rows.map((row) => String(row.reviewer_id || '').trim()))
      if (importedReviewerIds.size !== 1 || !importedReviewerIds.has(reviewerId)) {
        throw new Error(`文件评审者必须为当前 ID：${reviewerId}`)
      }
      const next = { ...responses }
      rows.forEach((row) => {
        if (!ids.has(row.item_id)) throw new Error(`文件包含未知 item_id：${row.item_id}`)
        JUDGMENTS.forEach(({ key }) => {
          if (row[key] !== null && typeof row[key] !== 'boolean') {
            throw new Error(`${row.item_id} 的 ${key} 不是布尔值或 null`)
          }
        })
        next[row.item_id] = {
          ...EMPTY_RESPONSE,
          ...Object.fromEntries(JUDGMENTS.map(({ key }) => [key, row[key] ?? null])),
          notes: row.notes || '',
          updated_at: new Date().toISOString(),
        }
      })
      setResponses(next)
      setToast(`已导入 ${rows.length} 条评审记录`)
    } catch (error) {
      setImportError(error.message)
    }
  }

  if (loadError) {
    return (
      <main className="load-state error-state">
        <AlertTriangle size={30} />
        <h1>评审队列无法加载</h1>
        <p>{loadError}</p>
      </main>
    )
  }

  if (!queue.length || !manifest) {
    return (
      <main className="load-state">
        <Inbox size={30} />
        <h1>正在载入冻结队列</h1>
      </main>
    )
  }

  return (
    <div className="review-app">
      <header className="topbar">
        <div className="brand-block">
          <button
            type="button"
            className="icon-button mobile-menu"
            title="打开队列"
            aria-label="打开队列"
            onClick={() => setSidebarOpen(true)}
          >
            <Menu size={20} />
          </button>
          <div className="brand-mark" aria-hidden="true"><FileCheck2 size={20} /></div>
          <div>
            <h1>Phase 1132 人工盲审</h1>
            <p>{manifest.revisionLabel}</p>
          </div>
        </div>

        <div className="top-progress" aria-label={`已完成 ${stats.completed} / ${queue.length}`}>
          <div className="progress-copy">
            <strong>{stats.completed}</strong>
            <span>/ {queue.length}</span>
          </div>
          <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
        </div>

        <div className="top-actions">
          <button
            type="button"
            className="button button-quiet compact-button"
            title="导入评审进度"
            onClick={() => importRef.current?.click()}
          >
            <Upload size={17} />
            <span>导入</span>
          </button>
          <input ref={importRef} type="file" accept=".jsonl,.ndjson,application/json" hidden onChange={importReview} />
          <button
            type="button"
            className="button button-quiet compact-button"
            title="下载当前进度"
            onClick={() => exportRows(false)}
          >
            <Download size={17} />
            <span>进度</span>
          </button>
          <button
            type="button"
            className="button button-primary compact-button"
            title={allComplete ? '导出正式评审清单' : '完成全部项目后可导出'}
            disabled={!allComplete}
            onClick={() => exportRows(true)}
          >
            <FileCheck2 size={17} />
            <span>正式导出</span>
          </button>
        </div>
      </header>

      <div className="workspace">
        {sidebarOpen && <button className="drawer-scrim" aria-label="关闭队列" onClick={() => setSidebarOpen(false)} />}
        <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`}>
          <div className="sidebar-heading">
            <div>
              <span className="section-label">评审队列</span>
              <strong>{filteredQueue.length} 条</strong>
            </div>
            <button type="button" className="icon-button close-drawer" aria-label="关闭队列" onClick={() => setSidebarOpen(false)}>
              <X size={19} />
            </button>
          </div>

          <button type="button" className="reviewer-row" onClick={() => setSetupOpen(true)}>
            <span className="reviewer-avatar"><UserRound size={17} /></span>
            <span>
              <small>当前评审者</small>
              <strong>{reviewerId}</strong>
            </span>
            <ShieldCheck size={17} className="verified-mark" />
          </button>

          <div className="stat-strip">
            <span><b>{stats.pending}</b> 待审</span>
            <span><b>{stats.accepted}</b> 通过</span>
            <span><b>{stats.flagged}</b> 问题</span>
          </div>

          <label className="search-box">
            <Search size={17} />
            <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="搜索实体或候选" />
          </label>

          <div className="filter-tabs" role="tablist" aria-label="队列筛选">
            <Filter size={15} aria-hidden="true" />
            {FILTERS.map((option) => (
              <button
                type="button"
                role="tab"
                aria-selected={filter === option.key}
                className={filter === option.key ? 'active' : ''}
                key={option.key}
                onClick={() => setFilter(option.key)}
              >
                {option.label}
              </button>
            ))}
          </div>

          <nav className="queue-list" aria-label="材料列表">
            {filteredQueue.map((item) => {
              const index = queue.findIndex((candidate) => candidate.item_id === item.item_id)
              const status = itemStatus(responses[item.item_id])
              return (
                <button
                  type="button"
                  key={item.item_id}
                  className={`queue-item ${item.item_id === currentId ? 'selected' : ''}`}
                  onClick={() => selectItem(item.item_id)}
                >
                  <span className="queue-index">{String(index + 1).padStart(3, '0')}</span>
                  <span className="queue-copy">
                    <strong>{item.active_candidate}</strong>
                    <small>{item.query}</small>
                  </span>
                  <StatusMark status={status} />
                </button>
              )
            })}
            {!filteredQueue.length && <p className="empty-list">没有符合条件的项目</p>}
          </nav>
        </aside>

        <main className="review-main">
          <div className="item-toolbar">
            <div className="item-position">
              <span>材料 {currentIndex + 1}</span>
              <strong>/ {queue.length}</strong>
            </div>
            <div className="item-tags">
              <span>{currentItem.split}</span>
              <span>{currentItem.item_id}</span>
            </div>
            <div className="nav-buttons">
              <button type="button" className="icon-button" title="上一条" aria-label="上一条" disabled={currentIndex === 0} onClick={() => moveBy(-1)}>
                <ArrowLeft size={19} />
              </button>
              <button type="button" className="icon-button" title="下一条" aria-label="下一条" disabled={currentIndex === queue.length - 1} onClick={() => moveBy(1)}>
                <ArrowRight size={19} />
              </button>
            </div>
          </div>

          <div className="review-scroll">
            <section className="material-section" aria-labelledby="context-heading">
              <p className="section-label" id="context-heading">日期档案</p>
              <p className="context-text">{currentItem.context}</p>
            </section>

            <section className="material-section query-section" aria-labelledby="query-heading">
              <p className="section-label" id="query-heading">查询</p>
              <h2>{currentItem.query}</h2>
            </section>

            <section className="candidate-grid" aria-label="候选答案">
              <div className="candidate-panel proposed">
                <span><CheckCircle2 size={17} /> 建议答案</span>
                <strong>{currentItem.active_candidate}</strong>
              </div>
              <div className="candidate-panel null-candidate">
                <span><XCircle size={17} /> 匹配对照</span>
                <strong>{currentItem.matched_null_candidate}</strong>
              </div>
            </section>

            <section className="judgment-section" aria-labelledby="judgment-heading">
              <div className="judgment-heading">
                <div>
                  <p className="section-label">独立判断</p>
                  <h2 id="judgment-heading">逐项确认</h2>
                </div>
                <StatusMark status={itemStatus(currentResponse)} />
              </div>

              <div className="judgment-list">
                {JUDGMENTS.map((judgment) => (
                  <div className="judgment-row" key={judgment.key}>
                    <div className="judgment-copy">
                      <strong>{judgment.label}</strong>
                      <span>{judgment.description}</span>
                    </div>
                    <div className="binary-control" role="group" aria-label={judgment.label}>
                      <button
                        type="button"
                        className={currentResponse[judgment.key] === true ? 'selected yes' : ''}
                        aria-pressed={currentResponse[judgment.key] === true}
                        onClick={() => updateCurrent(judgment.key, true)}
                      >
                        <Check size={17} /> 是
                      </button>
                      <button
                        type="button"
                        className={currentResponse[judgment.key] === false ? 'selected no' : ''}
                        aria-pressed={currentResponse[judgment.key] === false}
                        onClick={() => updateCurrent(judgment.key, false)}
                      >
                        <X size={17} /> 否
                      </button>
                    </div>
                  </div>
                ))}
              </div>

              <label className="notes-field">
                <span>备注</span>
                <textarea
                  value={currentResponse.notes || ''}
                  onChange={(event) => updateCurrent('notes', event.target.value)}
                  placeholder="仅记录歧义、事实错误或表达问题"
                  rows={3}
                  maxLength={1000}
                />
              </label>
            </section>
          </div>

          <footer className="review-footer">
            <div className={`current-status ${itemStatus(currentResponse)}`}>
              {itemStatus(currentResponse) === 'accepted' && <><CheckCircle2 size={18} /> 本条通过</>}
              {itemStatus(currentResponse) === 'flagged' && <><AlertTriangle size={18} /> 本条存在问题</>}
              {itemStatus(currentResponse) === 'pending' && <><span className="status-dot" /> 尚有未完成判断</>}
            </div>
            <button
              type="button"
              className="button button-primary next-button"
              disabled={!isComplete(currentResponse)}
              onClick={moveToNextPending}
            >
              保存并前往下一待审项
              <ArrowRight size={18} />
            </button>
          </footer>
        </main>
      </div>

      {setupOpen && (
        <SetupDialog
          initialId={reviewerId}
          hasExistingReview={Boolean(reviewerId)}
          onStart={startSession}
          onCancel={() => setSetupOpen(false)}
        />
      )}

      {importError && (
        <div className="toast error-toast" role="alert">
          <AlertTriangle size={18} />
          <span>{importError}</span>
          <button type="button" aria-label="关闭" onClick={() => setImportError('')}><X size={16} /></button>
        </div>
      )}
      {toast && (
        <div className="toast" role="status">
          <CheckCircle2 size={18} />
          <span>{toast}</span>
        </div>
      )}
    </div>
  )
}

export default AnnotationApp
