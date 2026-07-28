import { useEffect, useMemo, useRef, useState } from 'react'
import type { CSSProperties, MouseEvent as ReactMouseEvent } from 'react'
import { makeInitialHistoryWindow, makePreviousHistoryWindow, mergeJobsById } from './historyWindow.js'

const API_BASE = ''
const JOB_PAGE_SIZE = 100
const JOB_MAX_PAGES = 20
const JOB_SCROLL_BOTTOM_THRESHOLD = 80
const SAMPLE_PAGE_SIZE = 50
const JOB_NAME_PREFIXES = ['eval_', 'judge_']
const JOB_NAME_PREFIX_QUERY = JOB_NAME_PREFIXES.join(',')
const SAMPLE_MEDIA_COLUMN_KEY = '__sample_media__'

interface DlcJobSummary {
  job_id: string
  name: string
  status: string
  workspace_id: string
  resource_id: string
  job_type: string
  priority: string
  user_name: string
  user_id: string
  job_stage: string
  lmms_tasks: string[]
  llm_judge_tasks: string[]
  requires_llm_judge: boolean
  create_time: string
  submitted_time: string
  running_time: string
  finish_time: string
  duration_seconds: string
  result_root?: string | null
  has_results: boolean
  can_kill: boolean
  kill_disabled_reason: string
}

interface DlcJobsResponse {
  jobs: DlcJobSummary[]
  total: number
  start_time: string
  end_time: string
  fetched_at: string
  source: string
}

interface DlcJobDetailResponse {
  job: Record<string, unknown>
  result_root?: string | null
  runtime_config_path?: string | null
  log_dir?: string | null
  result_status: string
}

interface DlcMetricRow {
  metric_id: string
  display_name: string
  lmms_tasks: string
  status: string
  value: unknown
  value_text: string
  metric_name: string
  stderr: unknown
  started_at: string
  ended_at: string
  wall_seconds: unknown
  total_evaluation_time_seconds: unknown
  n_samples: unknown
  result_json?: string | null
  sample_jsonls: string[]
  value_source: string
}

interface DlcMetricsResponse {
  job_id: string
  result_root?: string | null
  metrics: DlcMetricRow[]
  summary_files: string[]
  message: string
}

interface DlcSampleMedia {
  url: string
  label: string
  source: string
  media_type: string
}

type DlcMetricSampleRow = Record<string, unknown> & {
  _media?: DlcSampleMedia[]
}

interface DlcMetricSamplesResponse {
  job_id: string
  metric_id: string
  columns: string[]
  rows: DlcMetricSampleRow[]
  total: number
  offset: number
  limit: number
  sample_files: string[]
  answer_stats: ChoiceAnswerStats
}

interface DlcJobKillResponse {
  job_id: string
  status: string
  message: string
}

interface ChoiceAnswerBucket {
  option: string
  count: number
  ratio: number
}

interface ChoiceAnswerStats {
  is_multiple_choice: boolean
  correct_answers: ChoiceAnswerBucket[]
  target_answers: ChoiceAnswerBucket[]
  total: number
  filtered_total: number
  wrong_total: number
  unknown_correctness_total: number
  correct_answer_total: number
  target_answer_total: number
}

type JobColumnKey = 'job_stage' | 'name' | 'user_name' | 'job_id' | 'status' | 'resource_id' | 'create_time' | 'duration_seconds'
type JobTableColumnKey = JobColumnKey | 'action'
type ColumnFilters = Record<JobColumnKey, string[]>

interface JobColumn {
  key: JobColumnKey
  label: string
  getValue: (job: DlcJobSummary) => unknown
}

interface ColumnResizeSession {
  key: JobTableColumnKey
  startX: number
  startWidth: number
  previousCursor: string
  previousUserSelect: string
}

interface SampleColumnResizeSession {
  key: string
  startX: number
  startWidth: number
  previousCursor: string
  previousUserSelect: string
}

interface PanelResizeSession {
  startX: number
  startWidth: number
  previousCursor: string
  previousUserSelect: string
}

const DEFAULT_JOB_LIST_PANEL_WIDTH = 960
const MIN_JOB_LIST_PANEL_WIDTH = 420
const MIN_JOB_DETAIL_PANEL_WIDTH = 360
const MAX_JOB_LIST_PANEL_WIDTH = 1400
const PANEL_RESIZER_WIDTH = 8

function maxJobListPanelWidth(containerWidth?: number): number {
  if (!containerWidth || containerWidth <= 0) return MAX_JOB_LIST_PANEL_WIDTH
  return Math.min(
    MAX_JOB_LIST_PANEL_WIDTH,
    Math.max(MIN_JOB_LIST_PANEL_WIDTH, containerWidth - MIN_JOB_DETAIL_PANEL_WIDTH - PANEL_RESIZER_WIDTH),
  )
}

function clampJobListPanelWidth(width: number, containerWidth?: number): number {
  return Math.min(maxJobListPanelWidth(containerWidth), Math.max(MIN_JOB_LIST_PANEL_WIDTH, Math.round(width)))
}

function defaultJobListPanelWidth(containerWidth?: number): number {
  const baseWidth = containerWidth && containerWidth >= 1536
    ? 1120
    : containerWidth && containerWidth >= 1280
      ? DEFAULT_JOB_LIST_PANEL_WIDTH
      : 720
  return clampJobListPanelWidth(baseWidth, containerWidth)
}

function initialJobListPanelWidth(): number {
  return defaultJobListPanelWidth(typeof window === 'undefined' ? undefined : window.innerWidth)
}

function valueToText(value: unknown): string {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  try {
    return JSON.stringify(value)
  } catch {
    return String(value)
  }
}

function compactText(value: unknown, maxLength = 160): string {
  const text = valueToText(value)
  if (text.length <= maxLength) return text
  return `${text.slice(0, maxLength - 3)}...`
}

function formatDuration(value: unknown): string {
  if (value === null || value === undefined || value === '') return 'N/A'
  const parsed = Number(value)
  if (Number.isFinite(parsed)) {
    if (parsed < 60) return `${parsed.toFixed(1)}s`
    if (parsed < 3600) return `${(parsed / 60).toFixed(1)}m`
    return `${(parsed / 3600).toFixed(2)}h`
  }
  return String(value)
}

function statusClass(status: string): string {
  const normalized = status.toLowerCase()
  if (normalized === 'running') return 'border-green-300 bg-green-50 text-green-800'
  if (normalized === 'succeeded' || normalized === 'success') return 'border-blue-300 bg-blue-50 text-blue-800'
  if (normalized === 'failed') return 'border-red-300 bg-red-50 text-red-800'
  if (normalized === 'stopped') return 'border-neutral-300 bg-neutral-100 text-neutral-700'
  if (normalized === 'queuing' || normalized === 'envpreparing') return 'border-amber-300 bg-amber-50 text-amber-800'
  return 'border-neutral-200 bg-white text-neutral-600'
}

function stageClass(stage: string): string {
  const normalized = stage.toLowerCase()
  if (normalized === 'judge') return 'border-emerald-300 bg-emerald-50 text-emerald-800'
  if (normalized === 'eval') return 'border-blue-300 bg-blue-50 text-blue-800'
  return 'border-neutral-200 bg-white text-neutral-600'
}

function listText(values: string[] | undefined): string {
  return values && values.length > 0 ? values.join(', ') : ''
}

function judgeHint(job: DlcJobSummary): string {
  if (!job.requires_llm_judge) return ''
  const tasks = listText(job.llm_judge_tasks)
  return tasks
    ? `This eval should be read together with judge output: ${tasks}`
    : 'This eval should be read together with judge output.'
}

function jobRowClass(job: DlcJobSummary, selected: boolean): string {
  const base = 'cursor-pointer border-b border-neutral-100'
  if (job.requires_llm_judge) {
    return `${base} ${selected ? 'bg-emerald-100/80' : 'bg-emerald-50/70 hover:bg-emerald-100/60'}`
  }
  return `${base} ${selected ? 'bg-neutral-100' : 'hover:bg-neutral-50'}`
}

const JOB_COLUMNS: JobColumn[] = [
  { key: 'job_stage', label: 'Stage', getValue: job => job.job_stage },
  { key: 'name', label: 'Name', getValue: job => job.name },
  { key: 'user_name', label: 'User', getValue: job => job.user_name },
  { key: 'job_id', label: 'JobId', getValue: job => job.job_id },
  { key: 'status', label: 'Status', getValue: job => job.status },
  { key: 'resource_id', label: 'Resource', getValue: job => job.resource_id },
  { key: 'create_time', label: 'Created', getValue: job => job.create_time },
  { key: 'duration_seconds', label: 'Duration', getValue: job => formatDuration(job.duration_seconds) },
]

const DEFAULT_JOB_COLUMN_WIDTHS: Record<JobTableColumnKey, number> = {
  job_stage: 96,
  name: 420,
  user_name: 170,
  job_id: 230,
  status: 120,
  resource_id: 190,
  create_time: 190,
  duration_seconds: 120,
  action: 96,
}

const MIN_JOB_COLUMN_WIDTHS: Record<JobTableColumnKey, number> = {
  job_stage: 88,
  name: 160,
  user_name: 110,
  job_id: 160,
  status: 100,
  resource_id: 120,
  create_time: 150,
  duration_seconds: 96,
  action: 88,
}

const MAX_JOB_COLUMN_WIDTH = 1200

function clampJobColumnWidth(key: JobTableColumnKey, width: number): number {
  return Math.min(MAX_JOB_COLUMN_WIDTH, Math.max(MIN_JOB_COLUMN_WIDTHS[key], Math.round(width)))
}

const MAX_SAMPLE_COLUMN_WIDTH = 1800

function defaultSampleColumnWidth(key: string): number {
  if (key === SAMPLE_MEDIA_COLUMN_KEY) return 150
  const normalized = key.toLowerCase()
  if (normalized === '_sample_file') return 180
  if (normalized === 'doc_id') return 110
  if (['exact_match', 'score', 'judge_score', 'answer'].includes(normalized)) return 120
  if (normalized.includes('hash')) return 220
  if (normalized.includes('token')) return 180
  if (normalized.includes('reason')) return 380
  if (
    ['input', 'target', 'filtered_resps', 'response', 'model_output', 'prediction', 'pred', 'extracted_answer'].includes(normalized)
  ) {
    return 380
  }
  return 260
}

function minSampleColumnWidth(key: string): number {
  return key === SAMPLE_MEDIA_COLUMN_KEY ? 128 : 96
}

function clampSampleColumnWidth(key: string, width: number): number {
  return Math.min(MAX_SAMPLE_COLUMN_WIDTH, Math.max(minSampleColumnWidth(key), Math.round(width)))
}

const QUERY_FIELD_GETTERS: Record<string, (job: DlcJobSummary) => unknown> = {
  name: job => job.name,
  stage: job => job.job_stage,
  job_stage: job => job.job_stage,
  tasks: job => (job.lmms_tasks || []).join(','),
  lmms_tasks: job => (job.lmms_tasks || []).join(','),
  judge_tasks: job => (job.llm_judge_tasks || []).join(','),
  llm_judge_tasks: job => (job.llm_judge_tasks || []).join(','),
  requires_judge: job => job.requires_llm_judge,
  requires_llm_judge: job => job.requires_llm_judge,
  user: job => job.user_name,
  user_name: job => job.user_name,
  username: job => job.user_name,
  user_id: job => job.user_id,
  userid: job => job.user_id,
  job_id: job => job.job_id,
  jobid: job => job.job_id,
  id: job => job.job_id,
  status: job => job.status,
  resource: job => job.resource_id,
  resource_id: job => job.resource_id,
  created: job => job.create_time,
  create_time: job => job.create_time,
  submitted: job => job.submitted_time,
  submitted_time: job => job.submitted_time,
  running: job => job.running_time,
  running_time: job => job.running_time,
  finish: job => job.finish_time,
  finish_time: job => job.finish_time,
  duration: job => job.duration_seconds,
  duration_seconds: job => job.duration_seconds,
  priority: job => job.priority,
  job_type: job => job.job_type,
  type: job => job.job_type,
  workspace: job => job.workspace_id,
  workspace_id: job => job.workspace_id,
}

function makeInitialColumnFilters(): ColumnFilters {
  return {
    job_stage: [],
    name: [],
    user_name: [],
    job_id: [],
    status: [],
    resource_id: [],
    create_time: [],
    duration_seconds: [],
  }
}

function normalizeQueryField(field: string): string {
  return field.trim().toLowerCase().replace(/-/g, '_')
}

function tokenizeQuery(query: string): string[] {
  const tokens: string[] = []
  let current = ''
  let quote: '"' | "'" | null = null
  let escaped = false

  for (const char of query.trim()) {
    if (quote) {
      current += char
      if (escaped) {
        escaped = false
      } else if (char === '\\') {
        escaped = true
      } else if (char === quote) {
        quote = null
      }
      continue
    }

    if (char === '"' || char === "'") {
      quote = char
      current += char
    } else if (/\s/.test(char)) {
      if (current) {
        tokens.push(current)
        current = ''
      }
    } else {
      current += char
    }
  }

  if (quote) {
    throw new Error('Unclosed quote in query')
  }
  if (current) {
    tokens.push(current)
  }
  return tokens
}

function stripQueryQuotes(value: string): string {
  const trimmed = value.trim()
  if (
    (trimmed.startsWith('"') && trimmed.endsWith('"')) ||
    (trimmed.startsWith("'") && trimmed.endsWith("'"))
  ) {
    return trimmed.slice(1, -1).replace(/\\(["'\\])/g, '$1')
  }
  return trimmed
}

function compareOrderedValues(actualValue: unknown, expectedText: string, operator: string): boolean {
  const actualText = valueToText(actualValue).trim()
  const actualNumber = Number(actualText)
  const expectedNumber = Number(expectedText)
  let actualComparable: number | string = actualText.toLowerCase()
  let expectedComparable: number | string = expectedText.toLowerCase()

  if (Number.isFinite(actualNumber) && Number.isFinite(expectedNumber)) {
    actualComparable = actualNumber
    expectedComparable = expectedNumber
  } else {
    const actualDate = Date.parse(actualText)
    const expectedDate = Date.parse(expectedText)
    if (Number.isFinite(actualDate) && Number.isFinite(expectedDate)) {
      actualComparable = actualDate
      expectedComparable = expectedDate
    }
  }

  if (operator === '>') return actualComparable > expectedComparable
  if (operator === '>=') return actualComparable >= expectedComparable
  if (operator === '<') return actualComparable < expectedComparable
  if (operator === '<=') return actualComparable <= expectedComparable
  throw new Error(`Unsupported query operator: ${operator}`)
}

function makeQueryCondition(token: string): (job: DlcJobSummary) => boolean {
  const match = token.match(/^([A-Za-z_][A-Za-z0-9_-]*)(!=|>=|<=|=|~|:|>|<)(.+)$/)
  if (!match) {
    const needle = stripQueryQuotes(token).toLowerCase()
    if (!needle) {
      throw new Error('Empty query token')
    }
    return job => JOB_COLUMNS.some(column => valueToText(column.getValue(job)).toLowerCase().includes(needle))
  }

  const [, rawField, operator, rawValue] = match
  const field = normalizeQueryField(rawField)
  const getter = QUERY_FIELD_GETTERS[field]
  if (!getter) {
    throw new Error(`Unknown query field: ${rawField}`)
  }
  const expected = stripQueryQuotes(rawValue)
  if (!expected) {
    throw new Error(`Missing query value for ${rawField}`)
  }

  return job => {
    const actual = getter(job)
    const actualText = valueToText(actual).toLowerCase()
    const expectedText = expected.toLowerCase()
    if (operator === '~' || operator === ':') return actualText.includes(expectedText)
    if (operator === '=') return actualText === expectedText
    if (operator === '!=') return actualText !== expectedText
    return compareOrderedValues(actual, expected, operator)
  }
}

function parseJobQuery(query: string): (job: DlcJobSummary) => boolean {
  const tokens = tokenizeQuery(query)
  if (tokens.length === 0) {
    return () => true
  }

  const groups: Array<Array<(job: DlcJobSummary) => boolean>> = [[]]
  let expectsCondition = true
  for (const token of tokens) {
    const lower = token.toLowerCase()
    if (lower === 'or') {
      if (expectsCondition) {
        throw new Error('OR must follow a condition')
      }
      groups.push([])
      expectsCondition = true
    } else if (lower === 'and') {
      if (expectsCondition) {
        throw new Error('AND must follow a condition')
      }
      expectsCondition = true
    } else {
      groups[groups.length - 1].push(makeQueryCondition(token))
      expectsCondition = false
    }
  }

  if (expectsCondition) {
    throw new Error('Query cannot end with AND/OR')
  }

  return job => groups.some(group => group.every(condition => condition(job)))
}

function columnValue(job: DlcJobSummary, column: JobColumn): string {
  const text = valueToText(column.getValue(job)).trim()
  return text || 'N/A'
}

function jsonPreview(value: unknown): string {
  if (value === null || value === undefined || value === '') return 'N/A'
  if (typeof value === 'string') return value
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function isSampleMedia(value: unknown): value is DlcSampleMedia {
  if (!value || typeof value !== 'object') return false
  const media = value as Record<string, unknown>
  return (
    typeof media.url === 'string' &&
    typeof media.label === 'string' &&
    typeof media.source === 'string' &&
    typeof media.media_type === 'string' &&
    media.media_type.startsWith('image/')
  )
}

function sampleMedia(row: DlcMetricSampleRow): DlcSampleMedia[] {
  return Array.isArray(row._media) ? row._media.filter(isSampleMedia) : []
}

function extractResourceSummary(job: Record<string, unknown>): Array<[string, string]> {
  const specs = job.JobSpecs
  if (!Array.isArray(specs)) return []
  return specs.flatMap((item, index) => {
    if (!item || typeof item !== 'object') return []
    const spec = item as Record<string, unknown>
    const resource = spec.ResourceConfig
    if (!resource || typeof resource !== 'object') return []
    const config = resource as Record<string, unknown>
    return [
      [`spec_${index}_type`, valueToText(spec.Type)],
      [`spec_${index}_gpu`, valueToText(config.GPU)],
      [`spec_${index}_cpu`, valueToText(config.CPU)],
      [`spec_${index}_memory`, valueToText(config.Memory)],
      [`spec_${index}_shared_memory`, valueToText(config.SharedMemory)],
    ]
  })
}

function extractMountSummary(job: Record<string, unknown>): Array<[string, string]> {
  const dataSources = job.DataSources
  if (!Array.isArray(dataSources)) return []
  return dataSources.map((item, index) => {
    if (!item || typeof item !== 'object') return [`mount_${index}`, '']
    const source = item as Record<string, unknown>
    return [`mount_${index}`, `${valueToText(source.MountPath)} <- ${valueToText(source.Uri)}`]
  })
}

const PIE_COLORS = ['#eef3f8', '#c0daf0', '#9dabd0', '#6182cc', '#424d95', '#90b8f1', '#e0eff2', '#b9d8f7']

function percentText(ratio: number): string {
  if (!Number.isFinite(ratio)) return '0.0%'
  return `${(ratio * 100).toFixed(1)}%`
}

function polarToCartesian(cx: number, cy: number, radius: number, angleDegrees: number): [number, number] {
  const angleRadians = ((angleDegrees - 90) * Math.PI) / 180
  return [cx + radius * Math.cos(angleRadians), cy + radius * Math.sin(angleRadians)]
}

function describePieSlice(cx: number, cy: number, radius: number, startAngle: number, endAngle: number): string {
  const [startX, startY] = polarToCartesian(cx, cy, radius, startAngle)
  const [endX, endY] = polarToCartesian(cx, cy, radius, endAngle)
  const largeArcFlag = endAngle - startAngle > 180 ? 1 : 0
  return `M ${cx} ${cy} L ${startX} ${startY} A ${radius} ${radius} 0 ${largeArcFlag} 1 ${endX} ${endY} Z`
}

function AnswerPieChart({
  title,
  buckets,
  total,
}: {
  title: string
  buckets: ChoiceAnswerBucket[]
  total: number
}) {
  let cursor = 0
  return (
    <div className="overflow-hidden border border-neutral-200 bg-white p-3">
      <div className="flex items-center justify-between gap-3">
        <div className="text-xs font-semibold text-neutral-800">{title}</div>
        <div className="font-mono text-[11px] text-neutral-400">{total}</div>
      </div>
      <div className="mt-3 grid grid-cols-1 items-center gap-3 sm:grid-cols-[128px_minmax(0,1fr)]">
        <svg viewBox="0 0 120 120" className="mx-auto h-28 w-28 sm:h-32 sm:w-32" role="img" aria-label={title}>
          {buckets.length === 0 ? (
            <circle cx="60" cy="60" r="46" fill="#e5e7eb">
              <title>No option data</title>
            </circle>
          ) : (
            buckets.map((bucket, index) => {
              const startAngle = cursor
              const endAngle = cursor + bucket.ratio * 360
              cursor = endAngle
              const color = PIE_COLORS[index % PIE_COLORS.length]
              const tooltip = `${bucket.option}: ${bucket.count} (${percentText(bucket.ratio)})`
              if (bucket.ratio >= 0.999999) {
                return (
                  <circle key={bucket.option} cx="60" cy="60" r="46" fill={color}>
                    <title>{tooltip}</title>
                  </circle>
                )
              }
              return (
                <path key={bucket.option} d={describePieSlice(60, 60, 46, startAngle, endAngle)} fill={color}>
                  <title>{tooltip}</title>
                </path>
              )
            })
          )}
        </svg>
        <div className="min-w-0 max-w-full space-y-1">
          {buckets.length === 0 ? (
            <div className="text-[11px] italic text-neutral-400">No option data</div>
          ) : (
            buckets.map((bucket, index) => (
              <div key={bucket.option} className="flex min-w-0 items-center gap-2" title={`${bucket.option}: ${bucket.count} (${percentText(bucket.ratio)})`}>
                <span className="h-2.5 w-2.5 shrink-0" style={{ backgroundColor: PIE_COLORS[index % PIE_COLORS.length] }} />
                <span className="w-5 shrink-0 font-mono text-[11px] text-neutral-700">{bucket.option}</span>
                <span className="min-w-0 flex-1 truncate font-mono text-[11px] text-neutral-500">{bucket.count}</span>
                <span className="shrink-0 font-mono text-[11px] text-neutral-400">{percentText(bucket.ratio)}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

function AnswerStatsPanel({ stats }: { stats: ChoiceAnswerStats }) {
  if (!stats.is_multiple_choice) return null
  return (
    <div className="border border-neutral-200 bg-neutral-50 p-3" data-testid="choice-answer-stats">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <div className="text-xs font-semibold text-neutral-800">Choice Distribution</div>
        <div className="flex flex-wrap items-center gap-2 font-mono text-[11px] text-neutral-500">
          <span>Total {stats.total}</span>
          <span>Wrong {stats.wrong_total}</span>
          {stats.unknown_correctness_total > 0 && <span>Unknown {stats.unknown_correctness_total}</span>}
        </div>
      </div>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
        <AnswerPieChart title="Correct Answer" buckets={stats.correct_answers} total={stats.correct_answer_total} />
        <AnswerPieChart title="Target Answer" buckets={stats.target_answers} total={stats.target_answer_total} />
      </div>
    </div>
  )
}

export default function LogViewer() {
  const [jobs, setJobs] = useState<DlcJobSummary[]>([])
  const [jobsLoading, setJobsLoading] = useState(false)
  const [historyLoading, setHistoryLoading] = useState(false)
  const [jobsError, setJobsError] = useState('')
  const [jobsFetchedAt, setJobsFetchedAt] = useState('')
  const [jobsSource, setJobsSource] = useState('')
  const [historyStartTime, setHistoryStartTime] = useState('')
  const [historyEndTime, setHistoryEndTime] = useState('')
  const [jobQuery, setJobQuery] = useState('')
  const [columnFilters, setColumnFilters] = useState<ColumnFilters>(makeInitialColumnFilters)
  const [openFilterKey, setOpenFilterKey] = useState<JobColumnKey | null>(null)
  const [killInFlightJobId, setKillInFlightJobId] = useState<string | null>(null)
  const [killError, setKillError] = useState('')
  const [killMessage, setKillMessage] = useState('')

  const [selectedJob, setSelectedJob] = useState<DlcJobSummary | null>(null)
  const [jobDetail, setJobDetail] = useState<DlcJobDetailResponse | null>(null)
  const [metricsResponse, setMetricsResponse] = useState<DlcMetricsResponse | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [detailError, setDetailError] = useState('')

  const [selectedMetric, setSelectedMetric] = useState<DlcMetricRow | null>(null)
  const [samples, setSamples] = useState<DlcMetricSamplesResponse | null>(null)
  const [sampleOffset, setSampleOffset] = useState(0)
  const [onlyWrongSamples, setOnlyWrongSamples] = useState(false)
  const [samplesLoading, setSamplesLoading] = useState(false)
  const [samplesError, setSamplesError] = useState('')
  const [jobListPanelWidth, setJobListPanelWidth] = useState(initialJobListPanelWidth)
  const [jobColumnWidths, setJobColumnWidths] = useState<Record<JobTableColumnKey, number>>(DEFAULT_JOB_COLUMN_WIDTHS)
  const [sampleColumnWidths, setSampleColumnWidths] = useState<Record<string, number>>({})
  const [previewMedia, setPreviewMedia] = useState<DlcSampleMedia | null>(null)
  const splitPaneRef = useRef<HTMLDivElement | null>(null)
  const jobsRequestInFlightRef = useRef(false)
  const historyStartTimeRef = useRef('')
  const jobListBottomLatchedRef = useRef(false)
  const panelResizeSessionRef = useRef<PanelResizeSession | null>(null)
  const columnResizeSessionRef = useRef<ColumnResizeSession | null>(null)
  const sampleColumnResizeSessionRef = useRef<SampleColumnResizeSession | null>(null)

  const requestJobs = async (startTime: string, endTime: string): Promise<DlcJobsResponse> => {
    const params = new URLSearchParams({
      page_size: String(JOB_PAGE_SIZE),
      max_pages: String(JOB_MAX_PAGES),
      display_name: JOB_NAME_PREFIX_QUERY,
      start_time: startTime,
      end_time: endTime,
    })
    const response = await fetch(`${API_BASE}/dlc/jobs?${params.toString()}`)
    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      throw new Error(data.detail || response.statusText)
    }
    return data as DlcJobsResponse
  }

  const fetchJobs = async (): Promise<DlcJobSummary[]> => {
    if (jobsRequestInFlightRef.current) return jobs
    jobsRequestInFlightRef.current = true
    setJobsLoading(true)
    setJobsError('')
    jobListBottomLatchedRef.current = false
    try {
      const historyWindow = makeInitialHistoryWindow()
      const payload = await requestJobs(historyWindow.startTime, historyWindow.endTime)
      setJobs(payload.jobs)
      setJobsFetchedAt(payload.fetched_at)
      setJobsSource(payload.source)
      setHistoryStartTime(payload.start_time)
      historyStartTimeRef.current = payload.start_time
      setHistoryEndTime(payload.end_time)
      setSelectedJob(prev => {
        if (prev && payload.jobs.some(job => job.job_id === prev.job_id)) {
          return payload.jobs.find(job => job.job_id === prev.job_id) ?? prev
        }
        return payload.jobs[0] ?? null
      })
      return payload.jobs
    } catch (error) {
      setJobs([])
      setSelectedJob(null)
      setHistoryStartTime('')
      historyStartTimeRef.current = ''
      setHistoryEndTime('')
      setJobsError(error instanceof Error ? error.message : 'Failed to fetch DLC jobs')
      return []
    } finally {
      setJobsLoading(false)
      jobsRequestInFlightRef.current = false
    }
  }

  const loadOlderJobs = async (): Promise<void> => {
    const currentStartTime = historyStartTimeRef.current
    if (!currentStartTime || jobsRequestInFlightRef.current) return
    jobsRequestInFlightRef.current = true
    setHistoryLoading(true)
    setJobsError('')
    try {
      const historyWindow = makePreviousHistoryWindow(currentStartTime)
      const payload = await requestJobs(historyWindow.startTime, historyWindow.endTime)
      const merged = mergeJobsById(jobs, payload.jobs)
      setJobs(merged)
      setSelectedJob(selected => {
        if (selected) return merged.find(job => job.job_id === selected.job_id) ?? selected
        return merged[0] ?? null
      })
      setJobsFetchedAt(payload.fetched_at)
      setJobsSource(payload.source)
      setHistoryStartTime(payload.start_time)
      historyStartTimeRef.current = payload.start_time
    } catch (error) {
      setJobsError(error instanceof Error ? error.message : 'Failed to load older DLC jobs')
    } finally {
      setHistoryLoading(false)
      jobsRequestInFlightRef.current = false
    }
  }

  const handleJobListScroll = (event: React.UIEvent<HTMLDivElement>) => {
    const target = event.currentTarget
    const atBottom = target.scrollHeight - target.scrollTop - target.clientHeight <= JOB_SCROLL_BOTTOM_THRESHOLD
    if (!atBottom) {
      jobListBottomLatchedRef.current = false
      return
    }
    if (jobListBottomLatchedRef.current) return
    jobListBottomLatchedRef.current = true
    void loadOlderJobs()
  }

  const loadJob = async (job: DlcJobSummary | null) => {
    setSelectedJob(job)
    setSelectedMetric(null)
    setSamples(null)
    setSampleOffset(0)
    setOnlyWrongSamples(false)
    setPreviewMedia(null)
    setDetailError('')
    setDetailLoading(Boolean(job))
    if (!job) {
      setJobDetail(null)
      setMetricsResponse(null)
      setDetailLoading(false)
      return
    }

    try {
      const [detailResponse, metricsResponseRaw] = await Promise.all([
        fetch(`${API_BASE}/dlc/jobs/${encodeURIComponent(job.job_id)}`),
        fetch(`${API_BASE}/dlc/jobs/${encodeURIComponent(job.job_id)}/metrics`),
      ])
      const detailData = await detailResponse.json().catch(() => ({}))
      if (!detailResponse.ok) {
        throw new Error(detailData.detail || detailResponse.statusText)
      }
      const metricsData = await metricsResponseRaw.json().catch(() => ({}))
      if (!metricsResponseRaw.ok) {
        throw new Error(metricsData.detail || metricsResponseRaw.statusText)
      }
      setJobDetail(detailData as DlcJobDetailResponse)
      setMetricsResponse(metricsData as DlcMetricsResponse)
    } catch (error) {
      setJobDetail(null)
      setMetricsResponse(null)
      setDetailError(error instanceof Error ? error.message : 'Failed to load DLC job detail')
    } finally {
      setDetailLoading(false)
    }
  }

  const loadSamples = async (metric: DlcMetricRow, offset = 0, onlyWrong = onlyWrongSamples) => {
    if (!selectedJob) return
    setSelectedMetric(metric)
    setSampleOffset(offset)
    setPreviewMedia(null)
    setSamplesLoading(true)
    setSamplesError('')
    try {
      const params = new URLSearchParams({
        offset: String(offset),
        limit: String(SAMPLE_PAGE_SIZE),
        only_wrong: String(onlyWrong),
      })
      const response = await fetch(
        `${API_BASE}/dlc/jobs/${encodeURIComponent(selectedJob.job_id)}/metrics/${encodeURIComponent(metric.metric_id)}/samples?${params.toString()}`,
      )
      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || response.statusText)
      }
      setSamples(data as DlcMetricSamplesResponse)
    } catch (error) {
      setSamples(null)
      setSamplesError(error instanceof Error ? error.message : 'Failed to load samples')
    } finally {
      setSamplesLoading(false)
    }
  }

  const killDlcJob = async (job: DlcJobSummary) => {
    if (!job.can_kill || killInFlightJobId) return
    const confirmed = window.confirm(`Kill DLC job ${job.name || job.job_id}?`)
    if (!confirmed) return
    setKillInFlightJobId(job.job_id)
    setKillError('')
    setKillMessage('')
    try {
      const response = await fetch(`${API_BASE}/dlc/jobs/${encodeURIComponent(job.job_id)}/kill`, { method: 'POST' })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || response.statusText)
      }
      const payload = data as DlcJobKillResponse
      setKillMessage(payload.message || `Kill requested for ${payload.job_id}`)
      const refreshedJobs = await fetchJobs()
      const refreshedJob = refreshedJobs.find(nextJob => nextJob.job_id === job.job_id)
      if (refreshedJob && selectedJob?.job_id === job.job_id) {
        await loadJob(refreshedJob)
      }
    } catch (error) {
      setKillError(error instanceof Error ? error.message : 'Failed to kill DLC job')
    } finally {
      setKillInFlightJobId(null)
    }
  }

  useEffect(() => {
    void fetchJobs()
  }, [])

  useEffect(() => {
    if (selectedJob) {
      void loadJob(selectedJob)
    }
  }, [selectedJob?.job_id])

  useEffect(() => {
    if (selectedMetric) {
      void loadSamples(selectedMetric, sampleOffset, onlyWrongSamples)
    }
  }, [sampleOffset, onlyWrongSamples])

  useEffect(() => {
    const finishResize = () => {
      const panelSession = panelResizeSessionRef.current
      const session = columnResizeSessionRef.current
      const sampleSession = sampleColumnResizeSessionRef.current
      if (panelSession) {
        document.body.style.cursor = panelSession.previousCursor
        document.body.style.userSelect = panelSession.previousUserSelect
        panelResizeSessionRef.current = null
      }
      if (session) {
        document.body.style.cursor = session.previousCursor
        document.body.style.userSelect = session.previousUserSelect
        columnResizeSessionRef.current = null
      }
      if (sampleSession) {
        document.body.style.cursor = sampleSession.previousCursor
        document.body.style.userSelect = sampleSession.previousUserSelect
        sampleColumnResizeSessionRef.current = null
      }
    }

    const handleMouseMove = (event: MouseEvent) => {
      const panelSession = panelResizeSessionRef.current
      if (panelSession) {
        const nextWidth = clampJobListPanelWidth(
          panelSession.startWidth + event.clientX - panelSession.startX,
          splitPaneRef.current?.clientWidth,
        )
        setJobListPanelWidth(prev => {
          if (prev === nextWidth) return prev
          return nextWidth
        })
      }

      const session = columnResizeSessionRef.current
      if (session) {
        const nextWidth = clampJobColumnWidth(session.key, session.startWidth + event.clientX - session.startX)
        setJobColumnWidths(prev => {
          if (prev[session.key] === nextWidth) return prev
          return { ...prev, [session.key]: nextWidth }
        })
      }

      const sampleSession = sampleColumnResizeSessionRef.current
      if (sampleSession) {
        const nextWidth = clampSampleColumnWidth(sampleSession.key, sampleSession.startWidth + event.clientX - sampleSession.startX)
        setSampleColumnWidths(prev => {
          if (prev[sampleSession.key] === nextWidth) return prev
          return { ...prev, [sampleSession.key]: nextWidth }
        })
      }
    }

    const handleWindowResize = () => {
      setJobListPanelWidth(prev => clampJobListPanelWidth(prev, splitPaneRef.current?.clientWidth))
    }

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', finishResize)
    window.addEventListener('resize', handleWindowResize)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', finishResize)
      window.removeEventListener('resize', handleWindowResize)
      finishResize()
    }
  }, [])

  const parsedQuery = useMemo<{ predicate: (job: DlcJobSummary) => boolean; error: string }>(() => {
    try {
      return { predicate: parseJobQuery(jobQuery), error: '' }
    } catch (error) {
      return {
        predicate: (_job: DlcJobSummary) => false,
        error: error instanceof Error ? error.message : 'Invalid query',
      }
    }
  }, [jobQuery])

  const columnFilterOptions = useMemo(() => {
    const options = makeInitialColumnFilters()
    for (const column of JOB_COLUMNS) {
      options[column.key] = Array.from(new Set(jobs.map(job => columnValue(job, column)))).sort((left, right) =>
        left.localeCompare(right),
      )
    }
    return options
  }, [jobs])

  const filteredJobs = useMemo(() => {
    return jobs.filter(job => {
      if (!parsedQuery.predicate(job)) {
        return false
      }
      return JOB_COLUMNS.every(column => {
        const selectedValues = columnFilters[column.key]
        return selectedValues.length === 0 || selectedValues.includes(columnValue(job, column))
      })
    })
  }, [jobs, parsedQuery, columnFilters])

  const metrics = metricsResponse?.metrics ?? []
  const detailJob = jobDetail?.job ?? {}
  const resourceSummary = extractResourceSummary(detailJob)
  const mountSummary = extractMountSummary(detailJob)
  const userCommand = valueToText(detailJob.UserCommand)
  const sampleRangeStart = !samples || samples.total === 0 ? 0 : samples.offset + 1
  const sampleRangeEnd = samples ? Math.min(samples.offset + samples.limit, samples.total) : 0
  const sampleHasMedia = useMemo(() => Boolean(samples?.rows.some(row => sampleMedia(row).length > 0)), [samples])
  const sampleTableColumnKeys = useMemo(
    () => (samples ? (sampleHasMedia ? [SAMPLE_MEDIA_COLUMN_KEY, ...samples.columns] : samples.columns) : []),
    [samples, sampleHasMedia],
  )
  const sampleTableWidth = useMemo(
    () => sampleTableColumnKeys.reduce((total, key) => total + (sampleColumnWidths[key] ?? defaultSampleColumnWidth(key)), 0),
    [sampleTableColumnKeys, sampleColumnWidths],
  )
  const jobTableWidth = useMemo(
    () => Object.values(jobColumnWidths).reduce((total, width) => total + width, 0),
    [jobColumnWidths],
  )

  useEffect(() => {
    if (!samples) {
      setSampleColumnWidths({})
      setPreviewMedia(null)
      return
    }
    const nextWidths: Record<string, number> = {}
    const keys = sampleHasMedia ? [SAMPLE_MEDIA_COLUMN_KEY, ...samples.columns] : samples.columns
    for (const key of keys) {
      nextWidths[key] = defaultSampleColumnWidth(key)
    }
    setSampleColumnWidths(nextWidths)
  }, [selectedMetric?.metric_id, samples?.columns.join('\u0001'), sampleHasMedia])

  const panelContainerWidth = () => splitPaneRef.current?.clientWidth

  const resetJobListPanelWidth = () => {
    setJobListPanelWidth(defaultJobListPanelWidth(panelContainerWidth()))
  }

  const resizeJobListPanelBy = (delta: number) => {
    setJobListPanelWidth(prev => clampJobListPanelWidth(prev + delta, panelContainerWidth()))
  }

  const startJobListPanelResize = (event: ReactMouseEvent<HTMLElement>) => {
    event.preventDefault()
    event.stopPropagation()
    panelResizeSessionRef.current = {
      startX: event.clientX,
      startWidth: jobListPanelWidth,
      previousCursor: document.body.style.cursor,
      previousUserSelect: document.body.style.userSelect,
    }
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }

  const startJobColumnResize = (event: ReactMouseEvent<HTMLElement>, key: JobTableColumnKey) => {
    event.preventDefault()
    event.stopPropagation()
    columnResizeSessionRef.current = {
      key,
      startX: event.clientX,
      startWidth: jobColumnWidths[key],
      previousCursor: document.body.style.cursor,
      previousUserSelect: document.body.style.userSelect,
    }
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }

  const resetJobColumnWidth = (key: JobTableColumnKey) => {
    setJobColumnWidths(prev => ({ ...prev, [key]: DEFAULT_JOB_COLUMN_WIDTHS[key] }))
  }

  const renderResizeHandle = (key: JobTableColumnKey, label: string) => (
    <span
      role="separator"
      aria-label={`Resize ${label} column`}
      title={`Drag to resize ${label}; double-click to reset`}
      onMouseDown={event => startJobColumnResize(event, key)}
      onDoubleClick={event => {
        event.preventDefault()
        event.stopPropagation()
        resetJobColumnWidth(key)
      }}
      className="absolute right-0 top-0 z-20 h-full w-2 cursor-col-resize select-none border-r border-transparent hover:border-neutral-500 hover:bg-neutral-200/70"
    >
      <span className="absolute right-[3px] top-1/2 h-4 -translate-y-1/2 border-r border-neutral-300" />
    </span>
  )

  const startSampleColumnResize = (event: ReactMouseEvent<HTMLElement>, key: string) => {
    event.preventDefault()
    event.stopPropagation()
    sampleColumnResizeSessionRef.current = {
      key,
      startX: event.clientX,
      startWidth: sampleColumnWidths[key] ?? defaultSampleColumnWidth(key),
      previousCursor: document.body.style.cursor,
      previousUserSelect: document.body.style.userSelect,
    }
    document.body.style.cursor = 'col-resize'
    document.body.style.userSelect = 'none'
  }

  const resetSampleColumnWidth = (key: string) => {
    setSampleColumnWidths(prev => ({ ...prev, [key]: defaultSampleColumnWidth(key) }))
  }

  const renderSampleResizeHandle = (key: string, label: string) => (
    <span
      role="separator"
      aria-label={`Resize ${label} sample column`}
      title={`Drag to resize ${label}; double-click to reset`}
      onMouseDown={event => startSampleColumnResize(event, key)}
      onDoubleClick={event => {
        event.preventDefault()
        event.stopPropagation()
        resetSampleColumnWidth(key)
      }}
      className="absolute right-0 top-0 z-20 h-full w-2 cursor-col-resize select-none border-r border-transparent hover:border-neutral-500 hover:bg-neutral-200/70"
    >
      <span className="absolute right-[3px] top-1/2 h-4 -translate-y-1/2 border-r border-neutral-300" />
    </span>
  )

  const renderSampleHeader = (key: string) => {
    const label = key === SAMPLE_MEDIA_COLUMN_KEY ? 'Images' : key
    const width = sampleColumnWidths[key] ?? defaultSampleColumnWidth(key)
    return (
      <th
        key={key}
        style={{ width, minWidth: minSampleColumnWidth(key) }}
        className="relative select-none border-b border-r border-neutral-200 px-3 py-2 pr-5 text-left text-[10px] uppercase tracking-wider text-neutral-500"
      >
        <span className="block truncate">{label}</span>
        {renderSampleResizeHandle(key, label)}
      </th>
    )
  }

  const toggleColumnFilterValue = (key: JobColumnKey, value: string) => {
    setColumnFilters(prev => {
      const selected = new Set(prev[key])
      if (selected.has(value)) {
        selected.delete(value)
      } else {
        selected.add(value)
      }
      return { ...prev, [key]: Array.from(selected) }
    })
  }

  const clearColumnFilter = (key: JobColumnKey) => {
    setColumnFilters(prev => ({ ...prev, [key]: [] }))
  }

  const renderColumnHeader = (column: JobColumn) => {
    const selectedValues = columnFilters[column.key]
    const options = columnFilterOptions[column.key]
    const isOpen = openFilterKey === column.key
    return (
      <th
        key={column.key}
        style={{ width: jobColumnWidths[column.key], minWidth: MIN_JOB_COLUMN_WIDTHS[column.key] }}
        className="relative select-none px-3 py-2 pr-5 text-left text-[10px] uppercase tracking-wider text-neutral-500 border-b border-neutral-200"
      >
        <div className="flex items-center gap-1">
          <span className="truncate">{column.label}</span>
          <button
            type="button"
            onClick={event => {
              event.stopPropagation()
              setOpenFilterKey(isOpen ? null : column.key)
            }}
            className={`h-5 w-5 border text-[10px] leading-none transition-colors ${
              selectedValues.length > 0 ? 'border-black bg-black text-white' : 'border-neutral-200 bg-white text-neutral-400 hover:border-black hover:text-black'
            }`}
            title={`Filter ${column.label}`}
          >
            ▼
          </button>
        </div>
        {isOpen && (
          <div className="absolute left-2 top-full z-30 mt-1 h-64 w-64 overflow-y-auto border border-neutral-200 bg-white p-2 text-[11px] font-normal normal-case tracking-normal text-neutral-700 shadow-xl">
            <div className="sticky top-0 z-10 mb-2 flex items-center justify-between border-b border-neutral-100 bg-white pb-2">
              <span className="font-mono text-[10px] text-neutral-400">{selectedValues.length}/{options.length}</span>
              <button
                type="button"
                onClick={event => {
                  event.stopPropagation()
                  clearColumnFilter(column.key)
                }}
                className="text-[10px] uppercase tracking-wider text-neutral-400 hover:text-black"
              >
                Clear
              </button>
            </div>
            {options.length === 0 ? (
              <div className="px-2 py-4 text-center text-neutral-400 italic">No values</div>
            ) : (
              options.map(option => (
                <label key={option} className="flex cursor-pointer items-center gap-2 px-2 py-1.5 hover:bg-neutral-50">
                  <input
                    type="checkbox"
                    checked={selectedValues.includes(option)}
                    onChange={() => toggleColumnFilterValue(column.key, option)}
                    className="h-3 w-3 accent-black"
                  />
                  <span className="min-w-0 flex-1 truncate font-mono" title={option}>{option}</span>
                </label>
              ))
            )}
          </div>
        )}
        {renderResizeHandle(column.key, column.label)}
      </th>
    )
  }

  const renderPanelResizeHandle = () => (
    <div
      role="separator"
      aria-label="Resize job list and job detail panels"
      aria-orientation="vertical"
      aria-valuemin={MIN_JOB_LIST_PANEL_WIDTH}
      aria-valuemax={maxJobListPanelWidth(panelContainerWidth())}
      aria-valuenow={jobListPanelWidth}
      aria-valuetext={`${jobListPanelWidth}px job list panel width`}
      tabIndex={0}
      title="Drag to resize panels; double-click to reset"
      onMouseDown={startJobListPanelResize}
      onDoubleClick={event => {
        event.preventDefault()
        event.stopPropagation()
        resetJobListPanelWidth()
      }}
      onKeyDown={event => {
        if (event.key === 'ArrowLeft') {
          event.preventDefault()
          resizeJobListPanelBy(event.shiftKey ? -80 : -24)
        } else if (event.key === 'ArrowRight') {
          event.preventDefault()
          resizeJobListPanelBy(event.shiftKey ? 80 : 24)
        } else if (event.key === 'Home') {
          event.preventDefault()
          setJobListPanelWidth(MIN_JOB_LIST_PANEL_WIDTH)
        } else if (event.key === 'End') {
          event.preventDefault()
          setJobListPanelWidth(maxJobListPanelWidth(panelContainerWidth()))
        } else if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault()
          resetJobListPanelWidth()
        }
      }}
      className="group relative z-20 hidden w-2 shrink-0 cursor-col-resize select-none items-center justify-center border-x border-transparent bg-white outline-none transition-colors hover:border-neutral-300 hover:bg-neutral-100 focus:border-neutral-600 focus:bg-neutral-100 active:bg-neutral-200 md:flex"
      data-testid="viewlog-panel-resizer"
    >
      <span className="h-12 border-r border-neutral-300 transition-colors group-hover:border-neutral-600 group-focus:border-neutral-800" />
    </div>
  )

  const renderSampleMediaCell = (row: DlcMetricSampleRow) => {
    const media = sampleMedia(row)
    return (
      <td className="overflow-hidden border-r border-neutral-100 px-3 py-2 align-top">
        {media.length === 0 ? (
          <span className="font-mono text-[11px] text-neutral-300">N/A</span>
        ) : (
          <div className="flex flex-wrap gap-2">
            {media.map((item, index) => (
              <button
                key={`${item.url}-${index}`}
                type="button"
                onClick={() => setPreviewMedia(item)}
                className="h-16 w-20 overflow-hidden border border-neutral-200 bg-neutral-50 p-1 hover:border-black"
                title={item.source || item.label}
              >
                <img
                  src={item.url}
                  alt={item.label || `sample image ${index + 1}`}
                  loading="lazy"
                  className="h-full w-full object-contain"
                />
              </button>
            ))}
          </div>
        )}
      </td>
    )
  }

  if (selectedMetric) {
    return (
      <div className="flex h-full w-full min-w-0 flex-col overflow-hidden bg-white">
        <div className="border-b border-neutral-200 px-4 py-3 sm:px-6 sm:py-4">
          <div className="flex min-w-0 flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
            <div className="min-w-0 flex-1">
              <button
                type="button"
                onClick={() => {
                  setSelectedMetric(null)
                  setSamples(null)
                  setSamplesError('')
                  setSampleOffset(0)
                  setOnlyWrongSamples(false)
                  setPreviewMedia(null)
                }}
                className="mb-2 text-[10px] uppercase tracking-wider text-neutral-500 hover:text-black"
              >
                Back to job
              </button>
              <h2 className="text-sm font-semibold text-neutral-900 truncate">
                {selectedMetric.display_name} / {selectedMetric.metric_name || 'samples'}
              </h2>
              <div className="mt-1 text-[10px] font-mono text-neutral-400 truncate">
                {selectedJob?.job_id} / {selectedMetric.sample_jsonls.join(', ') || 'no sample file'}
              </div>
            </div>
            <div className="flex w-full min-w-0 flex-wrap items-center gap-2 lg:w-auto lg:justify-end">
              <div className="flex min-w-0 items-center gap-2">
                <button
                  type="button"
                  role="switch"
                  aria-label="Only Wrong"
                  aria-checked={onlyWrongSamples}
                  onClick={() => {
                    setOnlyWrongSamples(prev => !prev)
                    setSampleOffset(0)
                  }}
                  disabled={samplesLoading}
                  className={`relative flex h-8 w-24 shrink-0 items-center border px-1 transition-colors disabled:cursor-not-allowed disabled:opacity-50 ${
                    onlyWrongSamples ? 'border-black bg-black text-white' : 'border-neutral-200 bg-white text-neutral-500 hover:border-black'
                  }`}
                  title="Only Wrong"
                >
                  <span className={`absolute text-[10px] font-semibold ${onlyWrongSamples ? 'left-2 text-white' : 'right-2 text-neutral-500'}`}>
                    {onlyWrongSamples ? 'On' : 'Off'}
                  </span>
                  <span className={`relative z-10 h-5 w-5 bg-current transition-transform ${onlyWrongSamples ? 'translate-x-16' : 'translate-x-0'}`} />
                </button>
                <span className="whitespace-nowrap text-[10px] font-semibold text-neutral-500">Only Wrong</span>
              </div>
              <div className="flex min-w-0 flex-wrap items-center gap-2">
                <button
                  onClick={() => setSampleOffset(Math.max(0, sampleOffset - SAMPLE_PAGE_SIZE))}
                  disabled={samplesLoading || sampleOffset === 0}
                  className="shrink-0 border border-neutral-200 px-3 py-2 text-[10px] uppercase tracking-wider text-neutral-500 disabled:text-neutral-300 disabled:cursor-not-allowed hover:border-black hover:text-black"
                >
                  Prev
                </button>
                <div className="min-w-[96px] shrink-0 text-center text-[10px] uppercase tracking-wider text-neutral-400">
                  {sampleRangeStart}-{sampleRangeEnd} / {samples?.total ?? 0}
                </div>
                <button
                  onClick={() => setSampleOffset(sampleOffset + SAMPLE_PAGE_SIZE)}
                  disabled={samplesLoading || !samples || sampleOffset + SAMPLE_PAGE_SIZE >= samples.total}
                  className="shrink-0 border border-neutral-200 px-3 py-2 text-[10px] uppercase tracking-wider text-neutral-500 disabled:text-neutral-300 disabled:cursor-not-allowed hover:border-black hover:text-black"
                >
                  Next
                </button>
              </div>
            </div>
          </div>
        </div>

        <div className="flex-1 min-h-0 overflow-auto p-4">
          {samplesLoading ? (
            <div className="text-xs text-neutral-400 italic">Loading samples...</div>
          ) : samplesError ? (
            <div className="border border-red-200 bg-red-50 p-3 text-xs font-mono text-red-700">{samplesError}</div>
          ) : !samples ? (
            <div className="text-xs text-neutral-400 italic">No samples available.</div>
          ) : (
            <div className="space-y-3">
              <AnswerStatsPanel stats={samples.answer_stats} />
              {samples.rows.length === 0 ? (
                <div className="border border-neutral-200 bg-white p-4 text-xs text-neutral-400 italic">No samples available.</div>
              ) : (
                <div className="overflow-auto border border-neutral-200">
                  <table
                    className="table-fixed border-collapse text-xs"
                    style={{ width: sampleTableWidth, minWidth: sampleTableWidth }}
                  >
                    <colgroup>
                      {sampleTableColumnKeys.map(column => (
                        <col key={column} style={{ width: sampleColumnWidths[column] ?? defaultSampleColumnWidth(column) }} />
                      ))}
                    </colgroup>
                    <thead className="sticky top-0 bg-neutral-50 z-10">
                      <tr>
                        {sampleTableColumnKeys.map(column => renderSampleHeader(column))}
                      </tr>
                    </thead>
                    <tbody>
                      {samples.rows.map((row, rowIndex) => (
                        <tr key={`${samples.offset}-${rowIndex}`} className="border-b border-neutral-100 align-top hover:bg-neutral-50">
                          {sampleHasMedia && renderSampleMediaCell(row)}
                          {samples.columns.map(column => (
                            <td key={column} className="overflow-hidden border-r border-neutral-100 px-3 py-2 font-mono text-[11px] text-neutral-700">
                              <pre className="max-h-44 overflow-auto whitespace-pre-wrap break-words">{jsonPreview(row[column])}</pre>
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
        {previewMedia && (
          <div
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 p-4"
            onClick={() => setPreviewMedia(null)}
          >
            <div
              className="flex max-h-full max-w-full flex-col overflow-hidden border border-neutral-700 bg-black"
              onClick={event => event.stopPropagation()}
            >
              <div className="flex items-center justify-between gap-4 border-b border-neutral-800 px-3 py-2">
                <div className="min-w-0">
                  <div className="truncate text-xs font-semibold text-white">{previewMedia.label || 'image'}</div>
                  <div className="max-w-[80vw] truncate font-mono text-[10px] text-neutral-400">{previewMedia.source}</div>
                </div>
                <button
                  type="button"
                  onClick={() => setPreviewMedia(null)}
                  className="shrink-0 border border-neutral-600 px-3 py-1 text-[10px] uppercase tracking-wider text-neutral-200 hover:border-white hover:text-white"
                >
                  Close
                </button>
              </div>
              <div className="min-h-0 overflow-auto p-3">
                <img
                  src={previewMedia.url}
                  alt={previewMedia.label || 'sample image'}
                  className="max-h-[82vh] max-w-[92vw] object-contain"
                />
              </div>
            </div>
          </div>
        )}
      </div>
    )
  }

  return (
    <div ref={splitPaneRef} className="flex h-full w-full min-w-0 bg-white">
      <div
        style={{ '--job-list-panel-width': `${jobListPanelWidth}px` } as CSSProperties}
        className="w-full shrink-0 border-r border-neutral-200 flex flex-col md:w-[var(--job-list-panel-width)] md:min-w-[420px]"
        data-testid="viewlog-job-list-panel"
      >
        <div className="border-b border-neutral-100 p-4 space-y-3">
          <div className="flex items-center justify-between gap-3">
            <div>
              <h2 className="text-xs font-bold text-neutral-400 uppercase tracking-widest">DLC Eval / Judge Jobs</h2>
              <div className="mt-1 text-[10px] font-mono text-neutral-400">
                prefixes {JOB_NAME_PREFIX_QUERY} / {filteredJobs.length}/{jobs.length} shown
                {jobsFetchedAt && <span> / {jobsFetchedAt}</span>}
              </div>
              {historyStartTime && historyEndTime && (
                <div className="mt-1 text-[10px] font-mono text-neutral-400">
                  history {historyStartTime} → {historyEndTime}
                </div>
              )}
            </div>
            <button
              onClick={() => void fetchJobs()}
              disabled={jobsLoading || historyLoading}
              className="border border-neutral-200 px-4 py-2 text-[10px] uppercase tracking-wider text-neutral-500 disabled:text-neutral-300 disabled:cursor-not-allowed hover:border-black hover:text-black"
            >
              {jobsLoading ? 'Syncing...' : 'Sync DLC'}
            </button>
          </div>
          {jobsSource && <div className="text-[10px] font-mono text-neutral-400 truncate">{jobsSource}</div>}
          {jobsError && <div className="border border-red-200 bg-red-50 p-2 text-[11px] font-mono text-red-700">{jobsError}</div>}
          {killError && <div className="border border-red-200 bg-red-50 p-2 text-[11px] font-mono text-red-700">{killError}</div>}
          {killMessage && <div className="border border-emerald-200 bg-emerald-50 p-2 text-[11px] font-mono text-emerald-800">{killMessage}</div>}
          <div className="flex items-center gap-2">
            <input
              value={jobQuery}
              onChange={event => setJobQuery(event.target.value)}
              className="min-w-0 flex-1 border border-neutral-200 bg-white px-3 py-2 text-[11px] font-mono text-neutral-700 focus:border-black focus:outline-none"
              placeholder="stage=judge or tasks~ocrbench or requires_judge=true"
              spellCheck={false}
            />
            {jobQuery && (
              <button
                type="button"
                onClick={() => setJobQuery('')}
                className="border border-neutral-200 px-3 py-2 text-[10px] uppercase tracking-wider text-neutral-500 hover:border-black hover:text-black"
              >
                Clear
              </button>
            )}
          </div>
          {parsedQuery.error && <div className="border border-red-200 bg-red-50 p-2 text-[11px] font-mono text-red-700">{parsedQuery.error}</div>}
        </div>

        <div
          className="flex-1 min-h-0 overflow-auto"
          onScroll={handleJobListScroll}
          data-testid="viewlog-job-list-scroll"
        >
          <table
            className="table-fixed border-collapse text-xs"
            style={{ width: jobTableWidth, minWidth: jobTableWidth }}
          >
            <colgroup>
              {JOB_COLUMNS.map(column => (
                <col key={column.key} style={{ width: jobColumnWidths[column.key] }} />
              ))}
              <col key="action" style={{ width: jobColumnWidths.action }} />
            </colgroup>
            <thead className="sticky top-0 bg-neutral-50 z-10">
              <tr>
                {JOB_COLUMNS.map(column => renderColumnHeader(column))}
                <th
                  style={{ width: jobColumnWidths.action, minWidth: MIN_JOB_COLUMN_WIDTHS.action }}
                  className="relative select-none px-3 py-2 pr-5 text-left text-[10px] uppercase tracking-wider text-neutral-500 border-b border-neutral-200"
                >
                  <span className="truncate">Action</span>
                  {renderResizeHandle('action', 'Action')}
                </th>
              </tr>
            </thead>
            <tbody>
              {filteredJobs.length === 0 && !jobsLoading ? (
                <tr>
                  <td colSpan={JOB_COLUMNS.length + 1} className="px-3 py-8 text-center text-neutral-400 italic">No DLC jobs matched.</td>
                </tr>
              ) : (
                filteredJobs.map(job => (
                  <tr
                    key={job.job_id}
                    onClick={() => setSelectedJob(job)}
                    className={jobRowClass(job, selectedJob?.job_id === job.job_id)}
                  >
                    <td className="overflow-hidden px-3 py-2 align-top">
                      <span className={`inline-flex border px-2 py-0.5 text-[10px] font-mono ${stageClass(job.job_stage)}`}>
                        {job.job_stage || 'unknown'}
                      </span>
                    </td>
                    <td className="overflow-hidden px-3 py-2 font-mono text-neutral-800 align-top" title={job.name}>
                      <div className="w-full overflow-x-auto scrollbar-thin scrollbar-thumb-neutral-200 scrollbar-track-transparent">
                        <div className="inline-flex min-w-max items-center gap-2 whitespace-nowrap">
                          <span>{job.name}</span>
                          {job.requires_llm_judge && (
                            <span
                              className="shrink-0 border border-emerald-300 bg-emerald-100 px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-emerald-800"
                              title={judgeHint(job)}
                            >
                              Needs judge
                            </span>
                          )}
                          {job.job_stage === 'judge' && (
                            <span className="shrink-0 border border-neutral-300 bg-white/70 px-1.5 py-0.5 text-[9px] uppercase tracking-wider text-neutral-600">
                              Judge job
                            </span>
                          )}
                        </div>
                      </div>
                      {job.requires_llm_judge && (
                        <div className="mt-1 w-full overflow-x-auto scrollbar-thin scrollbar-thumb-neutral-200 scrollbar-track-transparent text-[10px] text-emerald-800" title={listText(job.llm_judge_tasks)}>
                          <span className="inline-block min-w-max whitespace-nowrap">
                            {listText(job.llm_judge_tasks) || 'LLM-as-judge'}
                          </span>
                        </div>
                      )}
                    </td>
                    <td className="overflow-hidden px-3 py-2 font-mono text-neutral-600 align-top" title={job.user_name}>
                      <div className="w-full overflow-x-auto scrollbar-thin scrollbar-thumb-neutral-200 scrollbar-track-transparent">
                        <span className="inline-block min-w-max whitespace-nowrap">{job.user_name || 'N/A'}</span>
                      </div>
                    </td>
                    <td className="overflow-hidden px-3 py-2 font-mono text-neutral-600 align-top" title={job.job_id}>
                      <div className="w-full overflow-x-auto scrollbar-thin scrollbar-thumb-neutral-200 scrollbar-track-transparent">
                        <span className="inline-block min-w-max whitespace-nowrap">{job.job_id}</span>
                      </div>
                    </td>
                    <td className="overflow-hidden px-3 py-2 align-top">
                      <span className={`inline-flex border px-2 py-0.5 text-[10px] font-mono ${statusClass(job.status)}`}>{job.status || 'unknown'}</span>
                    </td>
                    <td className="overflow-hidden px-3 py-2 font-mono text-neutral-500 align-top" title={job.resource_id || 'N/A'}>
                      <div className="w-full overflow-x-auto scrollbar-thin scrollbar-thumb-neutral-200 scrollbar-track-transparent">
                        <span className="inline-block min-w-max whitespace-nowrap">{job.resource_id || 'N/A'}</span>
                      </div>
                    </td>
                    <td className="overflow-hidden px-3 py-2 font-mono text-neutral-500 align-top" title={job.create_time || 'N/A'}>
                      <div className="w-full overflow-x-auto scrollbar-thin scrollbar-thumb-neutral-200 scrollbar-track-transparent">
                        <span className="inline-block min-w-max whitespace-nowrap">{job.create_time || 'N/A'}</span>
                      </div>
                    </td>
                    <td className="overflow-hidden px-3 py-2 font-mono text-neutral-500 align-top">{formatDuration(job.duration_seconds)}</td>
                    <td className="overflow-hidden px-3 py-2 align-top">
                      <button
                        type="button"
                        onClick={event => {
                          event.stopPropagation()
                          void killDlcJob(job)
                        }}
                        disabled={!job.can_kill || Boolean(killInFlightJobId)}
                        title={job.can_kill ? `Kill ${job.job_id}` : job.kill_disabled_reason || 'Kill unavailable'}
                        className={`w-20 border px-2 py-1 text-[10px] uppercase tracking-wider transition-colors ${
                          job.can_kill && !killInFlightJobId
                            ? 'border-red-300 bg-white text-red-700 hover:border-red-700 hover:text-red-900'
                            : 'cursor-not-allowed border-neutral-200 bg-neutral-50 text-neutral-300'
                        }`}
                      >
                        {killInFlightJobId === job.job_id ? 'Killing' : 'Kill'}
                      </button>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
          <div className="flex items-center justify-center border-t border-neutral-100 px-3 py-3">
            <span className="font-mono text-[10px] uppercase tracking-wider text-neutral-400">
              {historyLoading ? 'Loading 15 days older...' : 'Scroll to the bottom to load 15 days older'}
            </span>
          </div>
        </div>
      </div>

      {renderPanelResizeHandle()}

      <div className="flex-1 min-w-0 flex flex-col bg-neutral-50/30">
        {!selectedJob ? (
          <div className="flex flex-1 items-center justify-center text-xs uppercase tracking-wider text-neutral-400">
            Select a DLC job
          </div>
        ) : detailLoading ? (
          <div className="flex flex-1 items-center justify-center text-xs text-neutral-400 italic">Loading DLC job...</div>
        ) : detailError ? (
          <div className="m-6 border border-red-200 bg-red-50 p-3 text-xs font-mono text-red-700">{detailError}</div>
        ) : (
          <>
            <div className="border-b border-neutral-200 bg-white px-6 py-4">
              <div className="flex items-start justify-between gap-4">
                <div className="min-w-0 flex-1">
                  <div className="max-w-full overflow-x-auto scrollbar-thin scrollbar-thumb-neutral-200 scrollbar-track-transparent" title={selectedJob.name}>
                    <h2 className="inline-block min-w-max whitespace-nowrap text-sm font-semibold text-neutral-900">{selectedJob.name}</h2>
                  </div>
                  <div className="mt-1 font-mono text-[11px] text-neutral-500">{selectedJob.job_id}</div>
                </div>
                <div className="flex shrink-0 items-center gap-2">
                  <button
                    type="button"
                    onClick={() => void killDlcJob(selectedJob)}
                    disabled={!selectedJob.can_kill || Boolean(killInFlightJobId)}
                    title={selectedJob.can_kill ? `Kill ${selectedJob.job_id}` : selectedJob.kill_disabled_reason || 'Kill unavailable'}
                    className={`w-20 border px-2 py-1 text-[10px] uppercase tracking-wider transition-colors ${
                      selectedJob.can_kill && !killInFlightJobId
                        ? 'border-red-300 bg-white text-red-700 hover:border-red-700 hover:text-red-900'
                        : 'cursor-not-allowed border-neutral-200 bg-neutral-50 text-neutral-300'
                    }`}
                  >
                    {killInFlightJobId === selectedJob.job_id ? 'Killing' : 'Kill'}
                  </button>
                  <span className={`inline-flex border px-2 py-1 text-[10px] font-mono ${stageClass(selectedJob.job_stage)}`}>{selectedJob.job_stage || 'unknown'}</span>
                  <span className={`inline-flex border px-2 py-1 text-[10px] font-mono ${statusClass(selectedJob.status)}`}>{selectedJob.status}</span>
                </div>
              </div>
              {selectedJob.requires_llm_judge && (
                <div className="mt-3 border border-emerald-200 bg-emerald-50 px-3 py-2 text-[11px] text-emerald-900">
                  LLM-as-judge tasks: <span className="font-mono">{listText(selectedJob.llm_judge_tasks) || 'unknown'}</span>. Read this eval together with the judge output.
                </div>
              )}
              <div className="mt-3 grid grid-cols-2 xl:grid-cols-4 gap-2">
                <div className="border border-neutral-200 bg-neutral-50 p-2">
                  <div className="text-[10px] uppercase tracking-wider text-neutral-400">Result</div>
                  <div className="mt-1 font-mono text-[11px] text-neutral-700">{jobDetail?.result_status || 'unknown'}</div>
                </div>
                <div className="border border-neutral-200 bg-neutral-50 p-2">
                  <div className="text-[10px] uppercase tracking-wider text-neutral-400">Created</div>
                  <div className="mt-1 font-mono text-[11px] text-neutral-700">{selectedJob.create_time || 'N/A'}</div>
                </div>
                <div className="border border-neutral-200 bg-neutral-50 p-2">
                  <div className="text-[10px] uppercase tracking-wider text-neutral-400">Started</div>
                  <div className="mt-1 font-mono text-[11px] text-neutral-700">{selectedJob.running_time || 'N/A'}</div>
                </div>
                <div className="border border-neutral-200 bg-neutral-50 p-2">
                  <div className="text-[10px] uppercase tracking-wider text-neutral-400">Finished</div>
                  <div className="mt-1 font-mono text-[11px] text-neutral-700">{selectedJob.finish_time || 'N/A'}</div>
                </div>
              </div>
            </div>

            <div className="flex-1 min-h-0 overflow-auto p-5 space-y-5">
              <section className="border border-neutral-200 bg-white">
                <div className="border-b border-neutral-100 px-4 py-3 text-xs font-bold uppercase tracking-widest text-neutral-400">Paths</div>
                <div className="grid grid-cols-1 xl:grid-cols-3 gap-2 p-4 text-[11px] font-mono">
                  <div>
                    <div className="text-neutral-400 uppercase tracking-wider">Result Root</div>
                    <div className="mt-1 break-all text-neutral-700">{jobDetail?.result_root || 'N/A'}</div>
                  </div>
                  <div>
                    <div className="text-neutral-400 uppercase tracking-wider">Runtime Config</div>
                    <div className="mt-1 break-all text-neutral-700">{jobDetail?.runtime_config_path || 'N/A'}</div>
                  </div>
                  <div>
                    <div className="text-neutral-400 uppercase tracking-wider">Log Dir</div>
                    <div className="mt-1 break-all text-neutral-700">{jobDetail?.log_dir || 'N/A'}</div>
                  </div>
                </div>
              </section>

              <section className="border border-neutral-200 bg-white">
                <div className="border-b border-neutral-100 px-4 py-3 text-xs font-bold uppercase tracking-widest text-neutral-400">Resources</div>
                <div className="grid grid-cols-2 xl:grid-cols-5 gap-2 p-4">
                  {resourceSummary.length === 0 ? (
                    <div className="text-xs text-neutral-400 italic">No resource spec in DLC detail.</div>
                  ) : (
                    resourceSummary.map(([key, value]) => (
                      <div key={key} className="border border-neutral-100 bg-neutral-50 p-2">
                        <div className="text-[10px] uppercase tracking-wider text-neutral-400">{key}</div>
                        <div className="mt-1 font-mono text-[11px] text-neutral-700">{value || 'N/A'}</div>
                      </div>
                    ))
                  )}
                </div>
                {mountSummary.length > 0 && (
                  <div className="border-t border-neutral-100 p-4 space-y-1">
                    {mountSummary.map(([key, value]) => (
                      <div key={key} className="font-mono text-[11px] text-neutral-600 break-all">
                        <span className="text-neutral-400">{key}</span> {value}
                      </div>
                    ))}
                  </div>
                )}
              </section>

              <section className="border border-neutral-200 bg-white">
                <div className="border-b border-neutral-100 px-4 py-3 flex items-center justify-between">
                  <div className="text-xs font-bold uppercase tracking-widest text-neutral-400">Overall Metrics</div>
                  <div className="text-[10px] uppercase tracking-wider text-neutral-400">{metrics.length} rows</div>
                </div>
                {metricsResponse?.message && <div className="border-b border-amber-200 bg-amber-50 px-4 py-2 text-[11px] font-mono text-amber-800">{metricsResponse.message}</div>}
                <div className="overflow-auto">
                  <table className="min-w-[980px] w-full border-collapse text-xs">
                    <thead className="bg-neutral-50">
                      <tr>
                        <th className="px-3 py-2 text-left text-[10px] uppercase tracking-wider text-neutral-500 border-b border-neutral-200">Bench</th>
                        <th className="px-3 py-2 text-left text-[10px] uppercase tracking-wider text-neutral-500 border-b border-neutral-200">Status</th>
                        <th className="px-3 py-2 text-left text-[10px] uppercase tracking-wider text-neutral-500 border-b border-neutral-200">Metric</th>
                        <th className="px-3 py-2 text-left text-[10px] uppercase tracking-wider text-neutral-500 border-b border-neutral-200">Value</th>
                        <th className="px-3 py-2 text-left text-[10px] uppercase tracking-wider text-neutral-500 border-b border-neutral-200">Eval Time</th>
                        <th className="px-3 py-2 text-left text-[10px] uppercase tracking-wider text-neutral-500 border-b border-neutral-200">Wall</th>
                        <th className="px-3 py-2 text-left text-[10px] uppercase tracking-wider text-neutral-500 border-b border-neutral-200">Samples</th>
                      </tr>
                    </thead>
                    <tbody>
                      {metrics.length === 0 ? (
                        <tr><td colSpan={7} className="px-3 py-8 text-center text-neutral-400 italic">No local metrics found.</td></tr>
                      ) : (
                        metrics.map(metric => (
                          <tr
                            key={metric.metric_id}
                            onClick={() => metric.sample_jsonls.length > 0 && void loadSamples(metric, 0)}
                            className={`border-b border-neutral-100 hover:bg-neutral-50 ${
                              metric.sample_jsonls.length > 0 ? 'cursor-pointer' : 'cursor-default'
                            }`}
                            title={metric.sample_jsonls.length > 0 ? 'Open sample rows' : 'No sample jsonl found'}
                          >
                            <td className="px-3 py-2 font-mono text-neutral-800">{metric.display_name}</td>
                            <td className="px-3 py-2"><span className={`inline-flex border px-2 py-0.5 text-[10px] font-mono ${statusClass(metric.status)}`}>{metric.status}</span></td>
                            <td className="px-3 py-2 font-mono text-neutral-600">{metric.metric_name || 'N/A'}</td>
                            <td className="px-3 py-2 font-mono text-neutral-900">{metric.value_text}</td>
                            <td className="px-3 py-2 font-mono text-neutral-500">{formatDuration(metric.total_evaluation_time_seconds)}</td>
                            <td className="px-3 py-2 font-mono text-neutral-500">{formatDuration(metric.wall_seconds)}</td>
                            <td className="px-3 py-2 font-mono text-neutral-500">{compactText(metric.n_samples, 80) || 'N/A'}</td>
                          </tr>
                        ))
                      )}
                    </tbody>
                  </table>
                </div>
              </section>

              {userCommand && (
                <section className="border border-neutral-200 bg-white">
                  <div className="border-b border-neutral-100 px-4 py-3 text-xs font-bold uppercase tracking-widest text-neutral-400">Command</div>
                  <pre className="max-h-52 overflow-auto whitespace-pre-wrap break-words p-4 text-[11px] font-mono text-neutral-700">{userCommand}</pre>
                </section>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
