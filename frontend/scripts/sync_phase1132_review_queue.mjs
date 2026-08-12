import { createHash } from 'node:crypto'
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const repoRoot = path.resolve(scriptDir, '..', '..')
const revision = 'revision6_temporal_relation_binding_overprovisioned'
const resultRoot = path.join(
  repoRoot,
  'tests',
  'glm5',
  'result',
  'phase1132_postrelease_temporal_material',
  revision,
)
const sourceQueue = path.join(resultRoot, 'material', 'human_review_queue.jsonl')
const sourcePackage = path.join(resultRoot, 'material', 'candidate_package_unreviewed.jsonl')
const sourceManifest = path.join(resultRoot, 'material', 'source_manifest.json')
const destination = path.join(repoRoot, 'frontend', 'public', 'data', 'phase1132')
const destinationQueue = path.join(destination, 'human_review_queue.jsonl')

function sha256(buffer) {
  return createHash('sha256').update(buffer).digest('hex')
}

const sourceQueueBuffer = readFileSync(sourceQueue)
const packageBuffer = readFileSync(sourcePackage)
const source = JSON.parse(readFileSync(sourceManifest, 'utf8'))
const reviewRows = sourceQueueBuffer
  .toString('utf8')
  .split(/\r?\n/)
  .map((line) => line.trim())
  .filter(Boolean)
  .map((line) => JSON.parse(line))
const packageRows = packageBuffer
  .toString('utf8')
  .split(/\r?\n/)
  .map((line) => line.trim())
  .filter(Boolean)
  .map((line) => JSON.parse(line))
const packageById = new Map(packageRows.map((row) => [row.item_id, row]))
const rows = reviewRows.map((row) => {
  const sourceRow = packageById.get(row.item_id)
  if (!sourceRow) throw new Error(`Review item is absent from frozen package: ${row.item_id}`)
  return {
    ...row,
    split: sourceRow.split,
    primitive_subfamily: sourceRow.primitive_subfamily,
    property_id: sourceRow.property_id,
    relation_label: sourceRow.relation_label,
    domain: sourceRow.domain,
    subject_label: sourceRow.subject_label,
  }
})
const destinationBuffer = Buffer.from(
  `${rows.map((row) => JSON.stringify(row)).join('\n')}\n`,
  'utf8',
)
const splitCounts = rows.reduce((counts, row) => {
  counts[row.split] = (counts[row.split] || 0) + 1
  return counts
}, {})

mkdirSync(destination, { recursive: true })
writeFileSync(destinationQueue, destinationBuffer)
writeFileSync(
  path.join(destination, 'review_manifest.json'),
  `${JSON.stringify({
    phase: 1132,
    revision,
    revisionLabel: 'R6 · 时效实体-关系绑定 · 双盲队列',
    packageSha256: source.candidate_package_sha256,
    sourceQueueSha256: sha256(sourceQueueBuffer),
    queueSha256: sha256(destinationBuffer),
    itemCount: rows.length,
    splitCounts,
    propertyCounts: source.property_counts,
    sourceLicense: source.license,
    sourceSnapshot: source.raw_snapshot_provenance,
  }, null, 2)}\n`,
  'utf8',
)

console.log(JSON.stringify({
  destination: path.relative(repoRoot, destination),
  itemCount: rows.length,
  splitCounts,
  queueSha256: sha256(destinationBuffer),
}, null, 2))
