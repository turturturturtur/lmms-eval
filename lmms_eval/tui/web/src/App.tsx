import { useState, useEffect, useRef, useMemo } from 'react'
import type { FormEvent } from 'react'
import LogViewer from './LogViewer'
import TaskBuilder from './TaskBuilder'

const API_BASE = ''
const USER_PLACEHOLDER = '<USERNAME>'
const LEGACY_USER_PLACEHOLDER = '<USER>'
const USER_PLACEHOLDERS = [USER_PLACEHOLDER, LEGACY_USER_PLACEHOLDER]
const DEFAULT_DLC_PATH_TEMPLATE = `/mnt/cpfsB/${USER_PLACEHOLDER}/dlc`
const DEFAULT_MODEL_PATH_TEMPLATE = `/mnt/cpfsB/${USER_PLACEHOLDER}/Innovator-Tune/models/Qwen3.5-9B`
const DEFAULT_OUTPUT_PATH_TEMPLATE = `/mnt/cpfsB/${USER_PLACEHOLDER}/Innovator-Tune/lmms-eval/eval_result/qwen35_9b_feishu20`
const DEFAULT_JUDGE_API_URL = 'http://8.130.30.251:8801/v1'
const DEFAULT_API_EVAL_URL = 'http://gw-k6isjixc1ij25ms7q4.cn-shanghai.pai-eas.aliyuncs.com/api/predict/router_fs_eval/v1'
const PAGES = ['evaluate', 'logs', 'tasks'] as const

type Page = typeof PAGES[number]
type AuthStatus = 'checking' | 'authenticated' | 'anonymous'

const SHELL_KEYWORDS = new Set([
  'export', 'python', 'python3', 'uv', 'pip', 'node', 'npm', 'git',
  'cd', 'ls', 'echo', 'rm', 'mkdir', 'touch', 'alias', 'source', 'env'
])

const ANSI_COLORS: Record<string, string> = {
  '30': 'text-neutral-900',
  '31': 'text-red-600',
  '32': 'text-green-600',
  '33': 'text-yellow-600',
  '34': 'text-blue-600',
  '35': 'text-purple-600',
  '36': 'text-cyan-600',
  '37': 'text-neutral-400',
  '90': 'text-neutral-500',
  '91': 'text-red-500',
  '92': 'text-green-500',
  '93': 'text-yellow-500',
  '94': 'text-blue-500',
  '95': 'text-purple-500',
  '96': 'text-cyan-500',
  '97': 'text-neutral-300',
}

function highlightLog(line: string) {
  const ansiRegex = /(?:\x1b)?\[([0-9;]+)m/g

  const parts: any[] = []
  let lastIndex = 0
  let currentStyle = 'text-neutral-600'
  let isBold = false
  let i = 0

  let match
  while ((match = ansiRegex.exec(line)) !== null) {
    if (match.index > lastIndex) {
      const text = line.slice(lastIndex, match.index)
      const className = `${currentStyle}${isBold ? ' font-semibold' : ''}`
      parts.push(<span key={i++} className={className}>{text}</span>)
    }

    const codes = match[1].split(';')
    for (const code of codes) {
      if (code === '0') {
        currentStyle = 'text-neutral-600'
        isBold = false
      } else if (code === '1') {
        isBold = true
      } else if (ANSI_COLORS[code]) {
        currentStyle = ANSI_COLORS[code]
      }
    }

    lastIndex = ansiRegex.lastIndex
  }

  if (lastIndex < line.length) {
    const text = line.slice(lastIndex)
    const className = `${currentStyle}${isBold ? ' font-semibold' : ''}`
    parts.push(<span key={i++} className={className}>{text}</span>)
  }

  return parts.length > 0 ? parts : line
}

function highlightShell(code: string) {
  const tokens: any[] = []
  let remaining = code
  let i = 0

  while (remaining.length > 0) {
    let match = remaining.match(/^#.*/)
    if (match) {
      tokens.push(<span key={i++} className="text-neutral-400 italic">{match[0]}</span>)
      remaining = remaining.slice(match[0].length)
      continue
    }

    match = remaining.match(/^(['"])(?:(?!\1)[^\\]|\\.)*\1/)
    if (match) {
      tokens.push(<span key={i++} className="text-neutral-900 bg-neutral-100/50 rounded-[1px]">{match[0]}</span>)
      remaining = remaining.slice(match[0].length)
      continue
    }

    match = remaining.match(/^(\$[a-zA-Z_][a-zA-Z0-9_]*|\$\{[^}]+\})/)
    if (match) {
      tokens.push(<span key={i++} className="text-neutral-800 font-medium">{match[0]}</span>)
      remaining = remaining.slice(match[0].length)
      continue
    }

    match = remaining.match(/^(-+[a-zA-Z0-9_-]+)/)
    if (match) {
      tokens.push(<span key={i++} className="text-neutral-500 font-medium">{match[0]}</span>)
      remaining = remaining.slice(match[0].length)
      continue
    }

    match = remaining.match(/^[=&|;>]/)
    if (match) {
      tokens.push(<span key={i++} className="text-neutral-400 font-bold px-[1px]">{match[0]}</span>)
      remaining = remaining.slice(match[0].length)
      continue
    }

    match = remaining.match(/^\s+/)
    if (match) {
      tokens.push(<span key={i++}>{match[0]}</span>)
      remaining = remaining.slice(match[0].length)
      continue
    }

    match = remaining.match(/^[^\s#$'"=&|;>-]+/)
    if (match) {
      const word = match[0]
      if (SHELL_KEYWORDS.has(word)) {
        tokens.push(<span key={i++} className="text-neutral-700 font-bold">{word}</span>)
      } else {
        tokens.push(<span key={i++} className="text-neutral-600">{word}</span>)
      }
      remaining = remaining.slice(word.length)
      continue
    }

    tokens.push(<span key={i++}>{remaining[0]}</span>)
    remaining = remaining.slice(1)
  }

  return tokens
}

interface ShellEditorProps {
  value: string
  onChange: (value: string) => void
  placeholder?: string
  className?: string
}

function ShellEditor({ value, onChange, placeholder, className = '' }: ShellEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const preRef = useRef<HTMLPreElement>(null)

  const handleScroll = () => {
    if (textareaRef.current && preRef.current) {
      preRef.current.scrollTop = textareaRef.current.scrollTop
      preRef.current.scrollLeft = textareaRef.current.scrollLeft
    }
  }

  return (
    <div className={`relative group bg-white border border-neutral-200 transition-colors focus-within:border-black overflow-hidden ${className}`}>
      <pre
        ref={preRef}
        className="absolute inset-0 px-3 py-2 text-xs font-mono leading-relaxed whitespace-pre pointer-events-none overflow-hidden text-transparent"
        style={{ fontFamily: 'monospace' }}
        aria-hidden="true"
      >
        {value ? highlightShell(value) : <span className="text-neutral-300 italic">{placeholder}</span>}
        <br />
      </pre>

      <textarea
        ref={textareaRef}
        value={value}
        onChange={e => onChange(e.target.value)}
        onScroll={handleScroll}
        placeholder={placeholder}
        className="relative z-10 w-full h-full bg-transparent text-transparent caret-black px-3 py-2 text-xs font-mono leading-relaxed resize-none focus:outline-none whitespace-pre overflow-auto scrollbar-thin scrollbar-thumb-neutral-200 scrollbar-track-transparent"
        style={{ fontFamily: 'monospace' }}
        spellCheck={false}
        autoCapitalize="off"
        autoComplete="off"
      />
    </div>
  )
}

interface SelectProps {
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
  placeholder?: string
}

function Select({ value, onChange, options, placeholder }: SelectProps) {
  const [open, setOpen] = useState(false)
  const [search, setSearch] = useState('')
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) {
        setOpen(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  useEffect(() => {
    if (open) {
      setSearch('')
    }
  }, [open])

  const selectedOption = options.find(o => o.value === value)

  const filteredOptions = options.filter(o =>
    o.label.toLowerCase().includes(search.toLowerCase()) ||
    o.value.toLowerCase().includes(search.toLowerCase())
  )

  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors text-left text-neutral-600 hover:border-neutral-300"
      >
        <span className={selectedOption ? 'text-neutral-600' : 'text-neutral-400'}>
          {selectedOption?.label || placeholder || 'Select...'}
        </span>
        <span className={`text-[10px] text-neutral-400 transition-transform ${open ? 'rotate-180' : ''}`}>▼</span>
      </button>
      {open && (
        <div className="absolute z-50 left-0 right-0 mt-1 bg-white border border-neutral-200 shadow-lg max-h-60 overflow-hidden flex flex-col">
          <div className="p-2 border-b border-neutral-100">
            <input
              autoFocus
              value={search}
              onChange={e => setSearch(e.target.value)}
              onClick={e => e.stopPropagation()}
              placeholder="Search..."
              className="w-full text-xs font-mono px-2 py-1 bg-neutral-50 border border-neutral-200 text-neutral-600 focus:border-black focus:outline-none"
            />
          </div>
          <div className="overflow-auto">
            {filteredOptions.length > 0 ? (
              filteredOptions.map(option => (
                <div
                  key={option.value}
                  onClick={() => {
                    onChange(option.value)
                    setOpen(false)
                  }}
                  className={`px-3 py-2 text-xs font-mono cursor-pointer transition-colors ${
                    option.value === value
                      ? 'bg-black text-white'
                      : 'text-neutral-600 hover:bg-neutral-50'
                  }`}
                >
                  {option.label}
                </div>
              ))
            ) : (
              <div className="px-3 py-2 text-xs text-neutral-400 italic">No matches found</div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

function HighlightMatch({ text, match }: { text: string; match: string }) {
  if (!match || !text) return <>{text}</>
  const parts = text.split(new RegExp(`(${match})`, 'gi'))
  return (
    <>
      {parts.map((part, i) =>
        part.toLowerCase() === match.toLowerCase()
          ? <span key={i} className="bg-yellow-200 text-black">{part}</span>
          : part
      )}
    </>
  )
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
}

function normalizeUserPlaceholderText(text: string) {
  return text.split(LEGACY_USER_PLACEHOLDER).join(USER_PLACEHOLDER)
}

function normalizeUserPlaceholders<T>(value: T): T {
  if (typeof value === 'string') {
    return normalizeUserPlaceholderText(value) as T
  }
  if (Array.isArray(value)) {
    return value.map(item => normalizeUserPlaceholders(item)) as T
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, item]) => [key, normalizeUserPlaceholders(item)])
    ) as T
  }
  return value
}

function replaceUserPlaceholders(text: string, user: string) {
  let updated = text
  for (const placeholder of USER_PLACEHOLDERS) {
    updated = updated.split(placeholder).join(user)
  }
  return updated
}

function restoreUserPlaceholder(text: string, previousUser: string) {
  let updated = normalizeUserPlaceholderText(text)
  const previous = previousUser.trim()
  if (!previous) return updated

  for (const root of ['/mnt/cpfs/', '/mnt/cpfsB/']) {
    const previousPathSegment = new RegExp(`(${escapeRegExp(root)})${escapeRegExp(previous)}(/)`, 'g')
    updated = updated.replace(previousPathSegment, `$1${USER_PLACEHOLDER}$2`)
  }
  return updated
}

function applyUserToText(text: string, nextUser: string, previousUser: string) {
  const user = nextUser.trim()
  if (!user) return restoreUserPlaceholder(text, previousUser)

  let updated = replaceUserPlaceholders(text, user)
  const previous = previousUser.trim()
  if (previous && previous !== user) {
    for (const root of ['/mnt/cpfs/', '/mnt/cpfsB/']) {
      const previousPathSegment = new RegExp(`(${escapeRegExp(root)})${escapeRegExp(previous)}(/)`, 'g')
      updated = updated.replace(previousPathSegment, `$1${user}$2`)
    }
  }
  return updated
}

function withDlcBinary(config: Record<string, unknown>, dlcPath: string): Record<string, unknown> {
  const dlc = config.dlc
  if (!dlc || typeof dlc !== 'object' || Array.isArray(dlc)) {
    return config
  }
  return {
    ...config,
    dlc: {
      ...(dlc as Record<string, unknown>),
      binary: dlcPath,
    },
  }
}

function withDlcJobName(config: Record<string, unknown>, jobName: string): Record<string, unknown> {
  const name = jobName.trim()
  const dlc = config.dlc
  if (!name || !dlc || typeof dlc !== 'object' || Array.isArray(dlc)) {
    return config
  }
  return {
    ...config,
    dlc: {
      ...(dlc as Record<string, unknown>),
      job_name: name,
    },
  }
}

function applyDlcPathToConfigJson(configJson: string, dlcPath: string) {
  try {
    const parsed = JSON.parse(configJson)
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return configJson
    }
    return JSON.stringify(withDlcBinary(parsed as Record<string, unknown>, dlcPath), null, 2)
  } catch {
    return configJson
  }
}

function applyJobNameToConfigJson(configJson: string, jobName: string) {
  try {
    const parsed = JSON.parse(configJson)
    if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
      return configJson
    }
    return JSON.stringify(withDlcJobName(parsed as Record<string, unknown>, jobName), null, 2)
  } catch {
    return configJson
  }
}

function applyJobNameToPath(path: string, jobName: string) {
  const name = jobName.trim()
  if (!name) return path
  const normalized = path.trim().replace(/\/+$/, '')
  if (!normalized) return path
  const slashIndex = normalized.lastIndexOf('/')
  if (slashIndex < 0) return name
  return `${normalized.slice(0, slashIndex + 1)}${name}`
}

function extractDlcPath(config: unknown) {
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    return ''
  }
  const dlc = (config as Record<string, unknown>).dlc
  if (!dlc || typeof dlc !== 'object' || Array.isArray(dlc)) {
    return ''
  }
  const binary = (dlc as Record<string, unknown>).binary
  return typeof binary === 'string' ? binary : ''
}

function extractDlcJobName(config: unknown) {
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    return ''
  }
  const dlc = (config as Record<string, unknown>).dlc
  if (!dlc || typeof dlc !== 'object' || Array.isArray(dlc)) {
    return ''
  }
  const jobName = (dlc as Record<string, unknown>).job_name
  return typeof jobName === 'string' ? jobName : ''
}

interface TaskInfo {
  id: string
  name: string
  group: boolean
  requires_llm_judge?: boolean
}

interface YamlPreview {
  title: string
  subtitle: string
  yaml: string
  download_filename?: string
}

interface Config {
  user: string
  job_name: string
  eval_inference_mode: string
  model: string
  api_url: string
  api_key: string
  dlc_path: string
  model_args: string
  tasks: string[]
  judge_api_url: string
  judge_api_key: string
  env_vars: string
  batch_size: number
  limit: number | null
  output_path: string
  log_samples: boolean
  verbosity: string
  device: string | null
  env_setup: string
  run_mode: string
  dlc_config: Record<string, unknown>
  model_tp: number
  max_model_len: number
  gpu_memory_utilization: number
  max_num_seqs: number
  base_port: number
  concurrency: number
  gen_kwargs: string
  enable_thinking: boolean
  debug: boolean
}

type DefaultConfig = Config

interface AuthUser {
  username: string
  display_name: string
  role: 'user' | 'admin'
  access_key_id: string
  expires_at: number
}

type Status = 'ready' | 'running' | 'stopped' | 'completed' | 'error'

interface GitInfo {
  branch: string
  commit: string
}

interface SysInfo {
  hostname: string
  cwd: string
  repo_root?: string
}

interface DlcPoolMetric {
  used: number
  total: number
  percent: number
  capacity_source: string
}

interface DlcPoolJobUsage {
  job_id: string
  name: string
  status: string
  gpu: number
  cpu: number
  pod_count: number
}

interface DlcPoolUsage {
  workspace_id: string
  resource_id: string
  resource_name: string
  active_statuses: string[]
  gpu: DlcPoolMetric
  cpu: DlcPoolMetric
  jobs: DlcPoolJobUsage[]
  errors: string[]
  fetched_at: string
  source: string
}

type TaskNode =
  | { type: 'group', id: string, label: string, children: TaskInfo[] }
  | { type: 'leaf', task: TaskInfo }

function usageColor(percent: number) {
  if (percent >= 80) return 'bg-red-500'
  if (percent >= 50) return 'bg-yellow-500'
  return 'bg-green-500'
}

function metricTitle(label: string, metric: DlcPoolMetric, usage: DlcPoolUsage) {
  return [
    `${label}: ${metric.used}/${metric.total} (${metric.percent.toFixed(1)}%)`,
    `capacity: ${metric.capacity_source}`,
    `workloads: ${usage.jobs.length}`,
    `resource: ${usage.resource_name || usage.resource_id}`,
    usage.errors.length > 0 ? `errors: ${usage.errors.join('; ')}` : '',
  ].filter(Boolean).join('\n')
}

function PoolMetricBar({ label, metric, usage }: { label: string; metric: DlcPoolMetric; usage: DlcPoolUsage }) {
  const percent = Math.max(0, Math.min(metric.percent, 100))
  return (
    <div className="w-[112px] xl:w-[138px]" title={metricTitle(label, metric, usage)}>
      <div className="mb-1 flex items-center justify-between gap-2 text-[10px] font-mono">
        <span className="font-semibold text-neutral-500">{label}</span>
        <span className="text-neutral-500">{metric.used}/{metric.total}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-sm bg-neutral-200">
        <div
          className={`h-full rounded-sm transition-all duration-500 ${usageColor(metric.percent)}`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  )
}

function DlcPoolMeter() {
  const [usage, setUsage] = useState<DlcPoolUsage | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const loadUsage = async () => {
    try {
      const response = await fetch(`${API_BASE}/dlc/pool-usage`)
      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || response.statusText)
      }
      setUsage(data as DlcPoolUsage)
      setError('')
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load DLC pool')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    loadUsage()
    const interval = window.setInterval(loadUsage, 30000)
    return () => window.clearInterval(interval)
  }, [])

  if (loading && !usage) {
    return (
      <div className="hidden lg:flex w-[290px] xl:w-[350px] items-center justify-end text-[10px] font-mono uppercase tracking-wider text-neutral-400">
        DLC pool syncing...
      </div>
    )
  }

  if (error && !usage) {
    return (
      <div
        className="hidden lg:flex w-[290px] xl:w-[350px] items-center justify-end text-[10px] font-mono uppercase tracking-wider text-red-500"
        title={error}
      >
        DLC pool unavailable
      </div>
    )
  }

  if (!usage) return null

  return (
    <div
      className="hidden lg:flex items-center gap-2 xl:gap-3 rounded-md border border-neutral-200 bg-white px-2.5 xl:px-3 py-1.5 shadow-sm"
      title={`Scope: ${usage.active_statuses.join(', ')}\nFetched: ${usage.fetched_at}`}
    >
      <div className="text-[10px] font-bold uppercase tracking-wider text-neutral-400">DLC Pool</div>
      <PoolMetricBar label="GPU" metric={usage.gpu} usage={usage} />
      <PoolMetricBar label="CPU" metric={usage.cpu} usage={usage} />
    </div>
  )
}

interface LoginScreenProps {
  version: string
  accessKeyId: string
  secretAccessKey: string
  loading: boolean
  error: string
  onAccessKeyIdChange: (value: string) => void
  onSecretAccessKeyChange: (value: string) => void
  onSubmit: (event: FormEvent<HTMLFormElement>) => void
}

function LoginScreen({
  version,
  accessKeyId,
  secretAccessKey,
  loading,
  error,
  onAccessKeyIdChange,
  onSecretAccessKeyChange,
  onSubmit,
}: LoginScreenProps) {
  return (
    <div className="flex min-h-screen bg-neutral-50 text-neutral-900">
      <main className="m-auto w-full max-w-sm px-6">
        <div className="mb-8">
          <div className="text-xl font-bold tracking-tight text-neutral-900">LMMs-Eval</div>
          <div className="mt-1 font-mono text-[10px] uppercase tracking-wider text-neutral-400">WebUI Auth {version !== '...' && `v${version}`}</div>
        </div>

        <form onSubmit={onSubmit} className="border border-neutral-200 bg-white p-5 shadow-sm">
          <div className="space-y-4">
            <div>
              <label htmlFor="webui-access-key-id" className="mb-1.5 block text-[10px] font-bold uppercase tracking-wider text-neutral-400">Access Key ID</label>
              <input
                id="webui-access-key-id"
                value={accessKeyId}
                onChange={event => onAccessKeyIdChange(event.target.value)}
                autoComplete="username"
                className="w-full border border-neutral-200 bg-white px-3 py-2 font-mono text-xs text-neutral-700 outline-none transition-colors focus:border-black"
              />
            </div>
            <div>
              <label htmlFor="webui-secret-access-key" className="mb-1.5 block text-[10px] font-bold uppercase tracking-wider text-neutral-400">Secret Access Key</label>
              <input
                id="webui-secret-access-key"
                type="password"
                value={secretAccessKey}
                onChange={event => onSecretAccessKeyChange(event.target.value)}
                autoComplete="current-password"
                className="w-full border border-neutral-200 bg-white px-3 py-2 font-mono text-xs text-neutral-700 outline-none transition-colors focus:border-black"
              />
            </div>
          </div>

          {error && (
            <div className="mt-4 border border-red-200 bg-red-50 px-3 py-2 font-mono text-[11px] text-red-700">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !accessKeyId.trim() || !secretAccessKey.trim()}
            className={`mt-5 h-9 w-full text-[10px] font-semibold uppercase tracking-wider transition-colors ${
              loading || !accessKeyId.trim() || !secretAccessKey.trim()
                ? 'cursor-not-allowed border border-neutral-200 bg-neutral-100 text-neutral-400'
                : 'border border-black bg-black text-white hover:bg-neutral-800'
            }`}
          >
            {loading ? 'Signing in...' : 'Sign in'}
          </button>
        </form>
      </main>
    </div>
  )
}

export default function App() {
  const [authStatus, setAuthStatus] = useState<AuthStatus>('checking')
  const [authUser, setAuthUser] = useState<AuthUser | null>(null)
  const [loginAccessKeyId, setLoginAccessKeyId] = useState('')
  const [loginSecretAccessKey, setLoginSecretAccessKey] = useState('')
  const [loginLoading, setLoginLoading] = useState(false)
  const [loginError, setLoginError] = useState('')
  const [page, setPage] = useState<Page>('evaluate')
  const [version, setVersion] = useState('...')
  const [gitInfo, setGitInfo] = useState<GitInfo>({ branch: '', commit: '' })
  const [sysInfo, setSysInfo] = useState<SysInfo>({ hostname: '', cwd: '' })
  const [tasks, setTasks] = useState<TaskInfo[]>([])

  const [evalInferenceMode, setEvalInferenceMode] = useState('ckpt')
  const [model, setModel] = useState(DEFAULT_MODEL_PATH_TEMPLATE)
  const [apiUrl, setApiUrl] = useState(DEFAULT_API_EVAL_URL)
  const [apiKey, setApiKey] = useState('')
  const [dlcPath, setDlcPath] = useState(DEFAULT_DLC_PATH_TEMPLATE)
  const [modelArgs, setModelArgs] = useState('')
  const [envVars, setEnvVars] = useState('')
  const [selectedTasks, setSelectedTasks] = useState<Set<string>>(new Set())
  const [taskFilter, setTaskFilter] = useState('')
  const [judgeApiUrl, setJudgeApiUrl] = useState(DEFAULT_JUDGE_API_URL)
  const [judgeApiKey, setJudgeApiKey] = useState('')
  const [batchSize, setBatchSize] = useState('1')
  const [limit, setLimit] = useState('-1')
  const [device, setDevice] = useState('')
  const [outputPath, setOutputPath] = useState(DEFAULT_OUTPUT_PATH_TEMPLATE)
  const [verbosity, setVerbosity] = useState('INFO')
  const [envSetup, setEnvSetup] = useState('')
  const [runMode, setRunMode] = useState('dlc')
  const [userName, setUserName] = useState('')
  const [appliedUserName, setAppliedUserName] = useState('')
  const [jobName, setJobName] = useState('eval_qwen35_9b_feishu20')
  const [dlcConfigJson, setDlcConfigJson] = useState('{\n  "dlc": {\n    "submit": true,\n    "job_name": "eval_qwen35_9b_feishu20",\n    "workers": 1,\n    "worker_gpu": 8\n  }\n}')
  const [modelTp, setModelTp] = useState('2')
  const [maxModelLen, setMaxModelLen] = useState('40960')
  const [gpuMemoryUtilization, setGpuMemoryUtilization] = useState('0.88')
  const [maxNumSeqs, setMaxNumSeqs] = useState('192')
  const [basePort, setBasePort] = useState('8941')
  const [concurrency, setConcurrency] = useState('32')
  const [genKwargs, setGenKwargs] = useState('')
  const [enableThinking, setEnableThinking] = useState(false)
  const [debugMode, setDebugMode] = useState(false)

  const [status, setStatus] = useState<Status>('ready')
  const [jobId, setJobId] = useState<string | null>(null)
  const [output, setOutput] = useState<string[]>(['Ready to evaluate'])
  const [command, setCommand] = useState('')
  const [defaultsLoaded, setDefaultsLoaded] = useState(false)
  const [defaultsError, setDefaultsError] = useState('')

  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set())
  const [configExpanded, setConfigExpanded] = useState(true)
  const [tasksExpanded, setTasksExpanded] = useState(true)
  const [envVarsExpanded, setEnvVarsExpanded] = useState(true)
  const [logsMaximized, setLogsMaximized] = useState(false)
  const [yamlPreview, setYamlPreview] = useState<YamlPreview | null>(null)

  const outputRef = useRef<HTMLDivElement>(null)
  const activePageIndex = PAGES.indexOf(page)

  const handleAuthExpired = () => {
    setAuthUser(null)
    setAuthStatus('anonymous')
    setDefaultsLoaded(false)
    setTasks([])
    setCommand('# Please sign in to use the WebUI.')
    setOutput(['Please sign in to use the WebUI.'])
  }

  const loadTasks = () => {
    fetch(`${API_BASE}/tasks`)
      .then(r => {
        if (r.status === 401) {
          handleAuthExpired()
          return []
        }
        return r.json()
      })
      .then(setTasks)
      .catch(() => setTasks([]))
  }

  const parseDlcConfig = (): Record<string, unknown> | null => {
    try {
      const parsed = JSON.parse(dlcConfigJson)
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        return null
      }
      return parsed as Record<string, unknown>
    } catch {
      return null
    }
  }

  const buildConfig = (): Config | null => {
    const dlcConfig = parseDlcConfig()
    if (!dlcConfig) return null
    const normalizedDlcConfig = withDlcJobName(withDlcBinary(dlcConfig, dlcPath), jobName)

    return {
      user: userName.trim(),
      job_name: jobName.trim(),
      eval_inference_mode: evalInferenceMode,
      model,
      api_url: apiUrl,
      api_key: apiKey,
      dlc_path: dlcPath,
      model_args: modelArgs,
      tasks: Array.from(selectedTasks),
      judge_api_url: judgeApiUrl,
      judge_api_key: judgeApiKey,
      env_vars: envVars,
      batch_size: parseInt(batchSize) || 1,
      limit: limit ? parseInt(limit) : null,
      output_path: outputPath,
      log_samples: true,
      verbosity,
      device: device || null,
      env_setup: envSetup,
      run_mode: runMode,
      dlc_config: normalizedDlcConfig,
      model_tp: parseInt(modelTp) || 1,
      max_model_len: parseInt(maxModelLen) || 65536,
      gpu_memory_utilization: parseFloat(gpuMemoryUtilization) || 0.9,
      max_num_seqs: parseInt(maxNumSeqs) || 1024,
      base_port: parseInt(basePort) || 8001,
      concurrency: parseInt(concurrency) || 128,
      gen_kwargs: genKwargs,
      enable_thinking: enableThinking,
      debug: debugMode,
    }
  }

  const updateUserName = (value: string) => {
    setUserName(value)
    const nextUser = value.trim()

    const nextDlcPath = applyUserToText(dlcPath, nextUser, appliedUserName)
    setModel(prev => applyUserToText(prev, nextUser, appliedUserName))
    setDlcPath(nextDlcPath)
    setDlcConfigJson(prev => applyJobNameToConfigJson(applyDlcPathToConfigJson(applyUserToText(prev, nextUser, appliedUserName), nextDlcPath), jobName))
    setOutputPath(prev => applyUserToText(prev, nextUser, appliedUserName))
    setEnvVars(prev => applyUserToText(prev, nextUser, appliedUserName))
    setAppliedUserName(nextUser)
  }

  const updateJobName = (value: string) => {
    setJobName(value)
    const nextJobName = value.trim()
    if (!nextJobName) return
    setDlcConfigJson(prev => applyJobNameToConfigJson(prev, nextJobName))
    setOutputPath(prev => applyJobNameToPath(prev, nextJobName))
  }

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then(r => r.json())
      .then(d => {
        setVersion(d.version)
        if (d.git) setGitInfo(d.git)
        if (d.system) setSysInfo(d.system)
      })
      .catch(() => setVersion('error'))

    fetch(`${API_BASE}/auth/me`)
      .then(async r => {
        if (r.status === 401) {
          setAuthStatus('anonymous')
          return null
        }
        const data = await r.json()
        if (!r.ok) {
          throw new Error(data.detail || r.statusText)
        }
        return data as AuthUser
      })
      .then(user => {
        if (!user) return
        setAuthUser(user)
        setAuthStatus('authenticated')
      })
      .catch(e => {
        setAuthStatus('anonymous')
        setLoginError(e instanceof Error ? e.message : 'Failed to check login state')
      })
  }, [])

  useEffect(() => {
    if (authStatus !== 'authenticated') return

    fetch(`${API_BASE}/defaults`)
      .then(async r => {
        if (r.status === 401) {
          handleAuthExpired()
          return null
        }
        const data = await r.json()
        if (!r.ok) {
          throw new Error(data.detail || r.statusText)
        }
        return normalizeUserPlaceholders(data) as DefaultConfig
      })
      .then((d: DefaultConfig | null) => {
        if (!d) return
        const normalizedDlcConfig = normalizeUserPlaceholders(d.dlc_config || {})
        const nextJobName = d.job_name || extractDlcJobName(normalizedDlcConfig) || 'eval_qwen35_9b_feishu20'
        const nextUser = (d.user || '').trim()
        setUserName(nextUser)
        setAppliedUserName(nextUser)
        setJobName(nextJobName)
        setEvalInferenceMode(d.eval_inference_mode || 'ckpt')
        setModel(normalizeUserPlaceholderText(d.model || DEFAULT_MODEL_PATH_TEMPLATE))
        setApiUrl(d.api_url || DEFAULT_API_EVAL_URL)
        setApiKey(d.api_key || '')
        setDlcPath(normalizeUserPlaceholderText(d.dlc_path || extractDlcPath(normalizedDlcConfig) || DEFAULT_DLC_PATH_TEMPLATE))
        setModelArgs(d.model_args || '')
        setSelectedTasks(new Set(d.tasks || []))
        setJudgeApiUrl(d.judge_api_url || DEFAULT_JUDGE_API_URL)
        setJudgeApiKey(d.judge_api_key || '')
        setEnvVars(normalizeUserPlaceholderText(d.env_vars || ''))
        setBatchSize(String(d.batch_size || 1))
        setLimit(d.limit == null ? '' : String(d.limit))
        setOutputPath(normalizeUserPlaceholderText(d.output_path || DEFAULT_OUTPUT_PATH_TEMPLATE))
        setVerbosity(d.verbosity || 'INFO')
        setDevice(d.device || '')
        setEnvSetup(d.env_setup || '')
        setRunMode(d.run_mode || 'dlc')
        setDlcConfigJson(JSON.stringify(withDlcJobName(normalizedDlcConfig, nextJobName), null, 2))
        setModelTp(String(d.model_tp || 1))
        setMaxModelLen(String(d.max_model_len || 65536))
        setGpuMemoryUtilization(String(d.gpu_memory_utilization || 0.9))
        setMaxNumSeqs(String(d.max_num_seqs || 1024))
        setBasePort(String(d.base_port || 8001))
        setConcurrency(String(d.concurrency || 128))
        setGenKwargs(d.gen_kwargs || '')
        setEnableThinking(Boolean(d.enable_thinking))
        setDebugMode(Boolean(d.debug))
        setDefaultsError('')
        setDefaultsLoaded(true)
      })
      .catch((e) => {
        const message = `Failed to load DLC defaults: ${e}`
        setDefaultsLoaded(false)
        setDefaultsError(message)
        setCommand(`# ${message}`)
        setOutput(prev => [...prev, message])
      })

    loadTasks()
  }, [authStatus])

  useEffect(() => {
    if (!defaultsLoaded) {
      setCommand(defaultsError ? `# ${defaultsError}` : '# Loading DLC defaults...')
      return
    }

    const config = buildConfig()
    if (!config) {
      setCommand('# Invalid DLC JSON')
      return
    }

    fetch(`${API_BASE}/eval/preview`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    })
      .then(async r => {
        const data = await r.json().catch(() => ({}))
        if (r.status === 401) {
          handleAuthExpired()
          throw new Error('Authentication required')
        }
        if (!r.ok) {
          throw new Error(data.detail || r.statusText)
        }
        return data
      })
      .then(d => setCommand(d.command))
      .catch((e) => setCommand(`# Error generating command: ${e.message || e}`))
  }, [defaultsLoaded, defaultsError, userName, jobName, evalInferenceMode, model, apiUrl, apiKey, dlcPath, modelArgs, selectedTasks, judgeApiUrl, judgeApiKey, envVars, batchSize, limit, device, outputPath, verbosity, envSetup, runMode, dlcConfigJson, modelTp, maxModelLen, gpuMemoryUtilization, maxNumSeqs, basePort, concurrency, genKwargs, enableThinking, debugMode])

  useEffect(() => {
    if (!defaultsLoaded || !userName.trim()) return

    const nextUser = userName.trim()
    const nextDlcPath = applyUserToText(dlcPath, nextUser, appliedUserName)
    setModel(prev => applyUserToText(prev, nextUser, appliedUserName))
    setDlcPath(nextDlcPath)
    setDlcConfigJson(prev => applyJobNameToConfigJson(applyDlcPathToConfigJson(applyUserToText(prev, nextUser, appliedUserName), nextDlcPath), jobName))
    setOutputPath(prev => applyUserToText(prev, nextUser, appliedUserName))
    setEnvVars(prev => applyUserToText(prev, nextUser, appliedUserName))
    if (appliedUserName !== nextUser) {
      setAppliedUserName(nextUser)
    }
  }, [defaultsLoaded])

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight
    }
  }, [output])

  useEffect(() => {
    if (!yamlPreview) return

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setYamlPreview(null)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [yamlPreview])

  const visibleNodes = useMemo(() => {
    const nodes: TaskNode[] = []

    const allGroups = tasks.filter(t => t.group)
    const allLeaves = tasks.filter(t => !t.group)

    const filteredLeaves = allLeaves.filter(t =>
      t.id.toLowerCase().includes(taskFilter.toLowerCase()) ||
      t.name.toLowerCase().includes(taskFilter.toLowerCase())
    )

    const groupChildrenMap = new Map<string, TaskInfo[]>()
    const assignedLeafIds = new Set<string>()

    for (const group of allGroups) {
      const children = filteredLeaves.filter(leaf =>
        leaf.id.startsWith(`${group.id}_`) || leaf.id.startsWith(`${group.id}-`)
      )

      if (children.length > 0) {
        groupChildrenMap.set(group.id, children)
        children.forEach(c => assignedLeafIds.add(c.id))
        nodes.push({
          type: 'group',
          id: group.id,
          label: group.id,
          children: children
        })
      }
    }

    const topLevelLeaves = filteredLeaves.filter(leaf => !assignedLeafIds.has(leaf.id))
    topLevelLeaves.forEach(leaf => {
      nodes.push({ type: 'leaf', task: leaf })
    })

    nodes.sort((a, b) => {
      const idA = a.type === 'group' ? a.id : a.task.id
      const idB = b.type === 'group' ? b.id : b.task.id
      return idA.localeCompare(idB)
    })

    return nodes
  }, [tasks, taskFilter])

  const selectedJudgeTasks = useMemo(() => {
    const taskMap = new Map(tasks.map(task => [task.id, task]))
    return Array.from(selectedTasks).filter(taskId => taskMap.get(taskId)?.requires_llm_judge)
  }, [tasks, selectedTasks])

  const isApiEval = evalInferenceMode === 'api'
  const requiresJudgeConfig = selectedJudgeTasks.length > 0
  const startDisabled = status === 'running' || !defaultsLoaded

  const updateEvalInferenceMode = (nextMode: string) => {
    if (nextMode !== 'ckpt' && nextMode !== 'api') return
    setEvalInferenceMode(nextMode)
    if (nextMode === 'api' && !apiUrl.trim()) {
      setApiUrl(DEFAULT_API_EVAL_URL)
    }
  }

  const toggleTask = (taskId: string) => {
    const newSet = new Set(selectedTasks)
    if (newSet.has(taskId)) {
      newSet.delete(taskId)
    } else {
      newSet.add(taskId)
    }
    setSelectedTasks(newSet)
  }

  const toggleGroup = (children: TaskInfo[]) => {
    const newSet = new Set(selectedTasks)
    const allSelected = children.every(c => newSet.has(c.id))

    if (allSelected) {
      children.forEach(c => newSet.delete(c.id))
    } else {
      children.forEach(c => newSet.add(c.id))
    }
    setSelectedTasks(newSet)
  }

  const toggleGroupCollapse = (groupId: string, e: React.MouseEvent) => {
    e.stopPropagation()
    const newSet = new Set(collapsedGroups)
    if (newSet.has(groupId)) {
      newSet.delete(groupId)
    } else {
      newSet.add(groupId)
    }
    setCollapsedGroups(newSet)
  }

  const startEval = async () => {
    if (!defaultsLoaded) {
      setOutput([defaultsError || 'DLC defaults are still loading'])
      return
    }
    if (selectedTasks.size === 0) {
      setOutput(['Error: No tasks selected'])
      return
    }
    setStatus('running')
    setOutput(['Starting evaluation...'])

    const config = buildConfig()
    if (!config) {
      setStatus('error')
      setOutput(['Error: Invalid DLC JSON'])
      return
    }

    try {
      const res = await fetch(`${API_BASE}/eval/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      const data = await res.json()
      if (res.status === 401) {
        handleAuthExpired()
        throw new Error('Authentication required')
      }
      if (!res.ok) {
        throw new Error(data.detail || res.statusText)
      }
      setJobId(data.job_id)

      const eventSource = new EventSource(`${API_BASE}/eval/${data.job_id}/stream`)

      eventSource.onmessage = (event) => {
        const d = JSON.parse(event.data)

        if (d.type === 'output') {
          setOutput(prev => [...prev, d.line])
        } else if (d.type === 'done') {
          setStatus(d.exit_code === 0 ? 'completed' : 'error')
          setOutput(prev => [...prev, '', `Evaluation ${d.exit_code === 0 ? 'completed' : 'failed'} (exit: ${d.exit_code})`])
          eventSource.close()
        } else if (d.type === 'stopped') {
          setStatus('stopped')
          setOutput(prev => [...prev, '', 'Evaluation stopped'])
          eventSource.close()
        } else if (d.type === 'error') {
          setOutput(prev => [...prev, `Error: ${d.message}`])
          setStatus('error')
          eventSource.close()
        }
      }

      eventSource.onerror = () => {
        setStatus('error')
        setOutput(prev => [...prev, 'Connection error'])
        eventSource.close()
      }
    } catch (e) {
      setOutput([`Failed to start: ${e}`])
      setStatus('error')
    }
  }

  const stopEval = async () => {
    if (!jobId) return
    try {
      const res = await fetch(`${API_BASE}/eval/${jobId}/stop`, { method: 'POST' })
      if (res.status === 401) {
        handleAuthExpired()
        return
      }
      setStatus('stopped')
    } catch (e) {
      setOutput(prev => [...prev, `Failed to stop: ${e}`])
    }
  }

  const exportYaml = async () => {
    if (!defaultsLoaded) {
      setOutput(prev => [...prev, defaultsError || 'Export failed: DLC defaults are still loading'])
      return
    }
    const config = buildConfig()
    if (!config) {
      setOutput(prev => [...prev, 'Export failed: invalid DLC JSON'])
      return
    }

    try {
      const res = await fetch(`${API_BASE}/eval/export-yaml`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })
      const data = await res.json()
      if (res.status === 401) {
        handleAuthExpired()
        throw new Error('Authentication required')
      }
      if (!res.ok) {
        throw new Error(data.detail || res.statusText)
      }

      setYamlPreview({
        title: 'Export Config',
        subtitle: `eval_config_${model}.yaml`,
        yaml: data.yaml_content,
        download_filename: `eval_config_${model}.yaml`,
      })
    } catch (e) {
      console.error('Export failed:', e)
    }
  }

  const importYaml = () => {
    const input = document.createElement('input')
    input.type = 'file'
    input.accept = '.yaml,.yml'
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0]
      if (!file) return

      const text = await file.text()
      try {
        const res = await fetch(`${API_BASE}/eval/import-yaml`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ yaml_content: text }),
        })
        if (!res.ok) {
          const err = await res.json()
          if (res.status === 401) {
            handleAuthExpired()
            return
          }
          setOutput(prev => [...prev, `Import error: ${err.detail}`])
          return
        }
        const data = await res.json()
        const importedJobName = data.job_name || extractDlcJobName(data.dlc_config)

        if (data.user != null) setUserName(data.user)
        if (importedJobName) setJobName(importedJobName)
        if (data.eval_inference_mode) setEvalInferenceMode(data.eval_inference_mode)
        if (data.model) setModel(normalizeUserPlaceholderText(data.model))
        if (data.api_url != null) setApiUrl(data.api_url || DEFAULT_API_EVAL_URL)
        if (data.api_key != null) setApiKey(data.api_key)
        if (data.dlc_path != null) setDlcPath(normalizeUserPlaceholderText(data.dlc_path))
        if (data.model_args) setModelArgs(data.model_args)
        if (data.tasks && data.tasks.length > 0) setSelectedTasks(new Set(data.tasks))
        if (data.judge_api_url != null) setJudgeApiUrl(data.judge_api_url || DEFAULT_JUDGE_API_URL)
        if (data.judge_api_key != null) setJudgeApiKey(data.judge_api_key)
        if (data.env_vars) setEnvVars(normalizeUserPlaceholderText(data.env_vars))
        if (data.batch_size) setBatchSize(String(data.batch_size))
        if (data.limit != null) setLimit(String(data.limit))
        if (data.output_path) setOutputPath(normalizeUserPlaceholderText(data.output_path))
        if (data.verbosity) setVerbosity(data.verbosity)
        if (data.device) setDevice(data.device)
        if (data.run_mode) setRunMode(data.run_mode)
        if (data.dlc_config) {
          const normalizedDlcConfig = normalizeUserPlaceholders(data.dlc_config)
          const importedDlcPath = normalizeUserPlaceholderText(data.dlc_path || extractDlcPath(normalizedDlcConfig))
          const importedDlcConfig = importedDlcPath ? withDlcBinary(normalizedDlcConfig, importedDlcPath) : normalizedDlcConfig
          setDlcConfigJson(JSON.stringify(importedJobName ? withDlcJobName(importedDlcConfig, importedJobName) : importedDlcConfig, null, 2))
          if (importedDlcPath) setDlcPath(importedDlcPath)
        }
        if (data.model_tp) setModelTp(String(data.model_tp))
        if (data.max_model_len) setMaxModelLen(String(data.max_model_len))
        if (data.gpu_memory_utilization) setGpuMemoryUtilization(String(data.gpu_memory_utilization))
        if (data.max_num_seqs) setMaxNumSeqs(String(data.max_num_seqs))
        if (data.base_port) setBasePort(String(data.base_port))
        if (data.concurrency) setConcurrency(String(data.concurrency))
        if (data.gen_kwargs) setGenKwargs(data.gen_kwargs)
        if (data.enable_thinking != null) setEnableThinking(Boolean(data.enable_thinking))
        if (data.debug != null) setDebugMode(Boolean(data.debug))

        setOutput(prev => [...prev, `Imported config from ${file.name}`])
      } catch (e) {
        setOutput(prev => [...prev, `Import failed: ${e}`])
      }
    }
    input.click()
  }

  const previewTaskYaml = async (taskId: string) => {
    try {
      const res = await fetch(`${API_BASE}/tasks/${encodeURIComponent(taskId)}/yaml`)
      if (res.status === 401) {
        handleAuthExpired()
        return
      }
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        setOutput(prev => [...prev, `YAML preview error (${taskId}): ${err.detail || res.statusText}`])
        return
      }
      const data = await res.json()
      setYamlPreview({ title: data.task_id, subtitle: data.path, yaml: data.yaml })
    } catch (e) {
      setOutput(prev => [...prev, `YAML preview failed (${taskId}): ${e}`])
    }
  }

  const login = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setLoginLoading(true)
    setLoginError('')
    try {
      const res = await fetch(`${API_BASE}/auth/login`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          access_key_id: loginAccessKeyId.trim(),
          secret_access_key: loginSecretAccessKey.trim(),
        }),
      })
      const data = await res.json().catch(() => ({}))
      if (!res.ok) {
        throw new Error(data.detail || res.statusText)
      }
      setAuthUser(data as AuthUser)
      setAuthStatus('authenticated')
      setLoginSecretAccessKey('')
      setLoginError('')
    } catch (e) {
      setAuthUser(null)
      setAuthStatus('anonymous')
      setLoginError(e instanceof Error ? e.message : 'Login failed')
    } finally {
      setLoginLoading(false)
    }
  }

  const logout = async () => {
    try {
      await fetch(`${API_BASE}/auth/logout`, { method: 'POST' })
    } finally {
      setAuthUser(null)
      setAuthStatus('anonymous')
      setDefaultsLoaded(false)
      setTasks([])
      setCommand('# Signed out.')
      setOutput(['Signed out.'])
    }
  }

  if (authStatus !== 'authenticated') {
    return (
      <LoginScreen
        version={version}
        accessKeyId={loginAccessKeyId}
        secretAccessKey={loginSecretAccessKey}
        loading={authStatus === 'checking' || loginLoading}
        error={loginError}
        onAccessKeyIdChange={setLoginAccessKeyId}
        onSecretAccessKeyChange={setLoginSecretAccessKey}
        onSubmit={login}
      />
    )
  }

  return (
    <div className="flex flex-col h-screen bg-white text-neutral-900 font-light selection:bg-black selection:text-white">
      <header className="relative h-14 flex items-center justify-between px-6 border-b border-neutral-200 bg-white/80 backdrop-blur-md z-10">
        <div className="flex items-center gap-4 min-w-0">
          <div className="text-lg font-bold tracking-tight text-neutral-900">LMMs-Eval</div>
          <div className="flex items-center rounded-md border border-neutral-300 bg-neutral-100 p-0.5">
            <button
              type="button"
              onClick={() => setPage('evaluate')}
              className={`h-8 px-3 text-[11px] font-semibold rounded-[5px] transition-colors ${
                page === 'evaluate'
                  ? 'bg-white text-black shadow-sm border border-neutral-300'
                  : 'text-neutral-500 border border-transparent hover:text-neutral-700'
              }`}
            >
              Evaluate
            </button>
            <button
              type="button"
              onClick={() => setPage('logs')}
              className={`h-8 px-3 text-[11px] font-semibold rounded-[5px] transition-colors ${
                page === 'logs'
                  ? 'bg-white text-black shadow-sm border border-neutral-300'
                  : 'text-neutral-500 border border-transparent hover:text-neutral-700'
              }`}
            >
              View Logs
            </button>
            <button
              type="button"
              onClick={() => setPage('tasks')}
              className={`h-8 px-3 text-[11px] font-semibold rounded-[5px] transition-colors ${
                page === 'tasks'
                  ? 'bg-white text-black shadow-sm border border-neutral-300'
                  : 'text-neutral-500 border border-transparent hover:text-neutral-700'
              }`}
            >
              Task Builder
            </button>
          </div>
          <div className="hidden md:flex items-center gap-3 text-[10px] font-mono text-neutral-400">
            <span>v{version}</span>
            {(gitInfo.branch || gitInfo.commit) && (
              <>
                <span className="text-neutral-300">/</span>
                <span className="flex items-center gap-1">
                  {gitInfo.branch && <span>{gitInfo.branch}</span>}
                  {gitInfo.branch && gitInfo.commit && <span className="text-neutral-300">@</span>}
                  {gitInfo.commit && <span>{gitInfo.commit}</span>}
                </span>
              </>
            )}
            {(sysInfo.repo_root || sysInfo.cwd) && (
              <>
                <span className="text-neutral-300">/</span>
                <span className="max-w-[200px] truncate" title={sysInfo.repo_root || sysInfo.cwd}>
                  {sysInfo.repo_root || sysInfo.cwd}
                </span>
              </>
            )}

          </div>
        </div>
        <div className="flex items-center gap-4">
          {authUser && (
            <div className="hidden md:flex items-center gap-2 border border-neutral-200 bg-white px-2.5 py-1.5 font-mono text-[10px] text-neutral-500">
              <span className="max-w-[140px] truncate" title={authUser.display_name || authUser.username}>
                {authUser.display_name || authUser.username}
              </span>
              <span className="rounded-sm border border-neutral-200 px-1.5 py-0.5 tracking-wider text-neutral-400">
                {authUser.role === 'admin' ? 'Admin' : 'USER'}
              </span>
              <button
                type="button"
                onClick={() => void logout()}
                className="ml-1 uppercase tracking-wider text-neutral-400 transition-colors hover:text-neutral-900"
              >
                Logout
              </button>
            </div>
          )}
          <DlcPoolMeter />
          <div className={`px-2.5 py-0.5 text-[10px] uppercase tracking-wider font-medium border ${
            status === 'ready' ? 'border-neutral-200 text-neutral-400' :
            status === 'running' ? 'border-black text-black animate-pulse' :
            status === 'completed' ? 'border-green-600 text-green-600' :
            status === 'error' ? 'border-red-600 text-red-600' :
            'border-neutral-200 text-neutral-400'
          }`}>
            {status}
          </div>
        </div>
        {status === 'running' && (
          <div className="absolute bottom-0 left-0 w-full h-0.5 bg-neutral-100 overflow-hidden">
            <div className="h-full bg-black animate-pulse w-full" />
          </div>
        )}
      </header>

      <main className="flex-1 min-h-0 overflow-hidden">
        <div
          className="flex h-full transition-transform duration-300 ease-out"
          style={{ transform: `translateX(-${activePageIndex * 100}%)` }}
        >
          <section className="flex h-full w-full shrink-0 overflow-hidden">
        <div className="flex flex-1 overflow-hidden">
        <div className="w-full md:w-[400px] lg:w-[450px] xl:w-[500px] 2xl:w-[550px] min-w-[320px] max-w-[600px] bg-white border-r border-neutral-200 flex flex-col overflow-y-auto scrollbar-thin flex-shrink-0">
          <div className="flex-shrink-0 border-b border-neutral-100">
            <div
              className="px-6 py-4 flex items-center justify-between cursor-pointer hover:bg-neutral-50 transition-colors"
              onClick={() => setConfigExpanded(!configExpanded)}
            >
              <h2 className="text-xs font-bold text-neutral-400 uppercase tracking-widest">Configuration</h2>
              <div className="flex items-center gap-3">
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    importYaml()
                  }}
                  className="text-[10px] uppercase tracking-wider font-medium text-neutral-400 hover:text-neutral-900 transition-colors"
                  title="Import YAML config file"
                >
                  Import
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    exportYaml()
                  }}
                  className="text-[10px] uppercase tracking-wider font-medium text-neutral-400 hover:text-neutral-900 transition-colors"
                  title="Export current config as YAML"
                >
                  Export
                </button>
                <span className={`text-neutral-400 transform transition-transform ${configExpanded ? 'rotate-0' : '-rotate-90'}`}>▼</span>
              </div>
            </div>

            {configExpanded && (
              <div className="p-6 pt-0 space-y-4">
                <div className="group">
                  <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5">Run Mode</label>
                  <Select
                    value={runMode}
                    onChange={setRunMode}
                    options={[
                      { value: 'dlc', label: 'DLC' },
                    ]}
                  />
                </div>

                <div className="group">
                  <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">User</label>
                  <input
                    value={userName}
                    onChange={e => updateUserName(e.target.value)}
                    placeholder={USER_PLACEHOLDER}
                    className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors placeholder-neutral-400 text-neutral-600"
                  />
                </div>

	                <div className="group">
	                  <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Job Name</label>
	                  <input
	                    value={jobName}
                    onChange={e => updateJobName(e.target.value)}
                    placeholder="eval_qwen3vl_tp1"
                    className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors placeholder-neutral-400 text-neutral-600"
	                  />
	                </div>

	                <div className="flex items-center justify-between gap-3 border border-neutral-200 bg-white px-3 py-2">
	                  <span className={`text-[10px] font-bold uppercase tracking-wider ${!isApiEval ? 'text-neutral-900' : 'text-neutral-400'}`}>Ckpt Eval</span>
	                  <button
	                    type="button"
	                    role="switch"
	                    aria-label="Evaluation inference mode"
	                    aria-checked={isApiEval}
	                    onClick={() => updateEvalInferenceMode(isApiEval ? 'ckpt' : 'api')}
	                    className={`relative flex h-6 w-14 shrink-0 items-center rounded-full border px-0.5 transition-colors focus:outline-none ${
	                      isApiEval
	                        ? 'border-neutral-900 bg-neutral-900'
	                        : 'border-neutral-300 bg-neutral-100'
	                    }`}
	                  >
	                    <span
	                      className={`relative z-10 h-5 w-5 rounded-full bg-white shadow-sm transition-transform ${
	                        isApiEval ? 'translate-x-8' : 'translate-x-0'
	                      }`}
	                    />
	                  </button>
	                  <span className={`text-[10px] font-bold uppercase tracking-wider ${isApiEval ? 'text-neutral-900' : 'text-neutral-400'}`}>API Eval</span>
	                </div>

	                {!isApiEval ? (
	                  <div className="group">
	                    <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Checkpoint Path</label>
	                    <input
	                      value={model}
	                      onChange={e => setModel(e.target.value)}
	                      placeholder={DEFAULT_MODEL_PATH_TEMPLATE}
	                      className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors placeholder-neutral-400 text-neutral-600"
	                    />
	                  </div>
	                ) : (
	                  <div className="space-y-4">
	                    <div className="group">
	                      <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">API Address</label>
	                      <input
	                        value={apiUrl}
	                        onChange={e => setApiUrl(e.target.value)}
	                        placeholder={DEFAULT_API_EVAL_URL}
	                        className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors placeholder-neutral-400 text-neutral-600"
	                      />
	                    </div>
	                    <div className="group">
	                      <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">API Token</label>
	                      <input
	                        type="password"
	                        value={apiKey}
	                        onChange={e => setApiKey(e.target.value)}
	                        placeholder="sk-..."
	                        className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors placeholder-neutral-400 text-neutral-600"
	                      />
	                    </div>
	                  </div>
	                )}

                <div className="group">
                  <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">DLC Config</label>
                  <textarea
                    value={dlcConfigJson}
                    onChange={e => setDlcConfigJson(e.target.value)}
                    className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs h-48 resize-y focus:border-black focus:outline-none transition-colors leading-relaxed text-neutral-600 font-mono"
                    spellCheck={false}
                  />
                </div>

                <div className="group">
                  <div className="flex items-center justify-between mb-1.5">
                    <label
                      className="flex items-center gap-2 text-[10px] font-bold text-neutral-400 uppercase tracking-wider cursor-pointer"
                      onClick={() => setTasksExpanded(!tasksExpanded)}
                    >
                      <span className={`transform transition-transform ${tasksExpanded ? 'rotate-0' : '-rotate-90'}`}>▼</span>
                      Tasks <span className="text-neutral-900 ml-1">{selectedTasks.size}</span>
                    </label>
                    <button
                      onClick={() => setSelectedTasks(new Set())}
                      className="px-2 py-0.5 text-[10px] font-medium uppercase tracking-wider text-neutral-400 hover:text-neutral-600 transition-colors"
                      title="Clear all selected tasks"
                    >
                      Clear
                    </button>
                  </div>

                  {tasksExpanded && (
                    <div className="w-full bg-white border border-neutral-200 text-xs h-40 resize-y overflow-hidden min-h-[80px] max-h-[400px] flex flex-col">
                      <div className="p-2 border-b border-neutral-100 flex-shrink-0">
                        <input
                          value={taskFilter}
                          onChange={e => setTaskFilter(e.target.value)}
                          placeholder="Search tasks..."
                          className="w-full bg-neutral-50 border border-neutral-200 px-2 py-1 text-xs font-mono focus:border-black focus:outline-none transition-colors placeholder-neutral-400 text-neutral-600"
                        />
                      </div>
                      <div className="flex-1 overflow-y-auto scrollbar-thin">
                        {visibleNodes.map((node) => {
                          if (node.type === 'group') {
                            const allChildrenSelected = node.children.every(c => selectedTasks.has(c.id))
                            const someChildrenSelected = node.children.some(c => selectedTasks.has(c.id))
                            const isCollapsed = collapsedGroups.has(node.id)

                            return (
                              <div key={node.id} className="border-b border-neutral-50 last:border-b-0">
                                <div
                                  onClick={() => toggleGroup(node.children)}
                                  className="flex items-center gap-2 px-3 py-1.5 bg-neutral-50/50 cursor-pointer hover:bg-neutral-100 transition-colors"
                                >
                                  <div
                                    onClick={(e) => toggleGroupCollapse(node.id, e)}
                                    className="text-neutral-400 hover:text-neutral-900 cursor-pointer w-3 flex justify-center"
                                  >
                                    <span className={`transform transition-transform text-[10px] ${isCollapsed ? '-rotate-90' : 'rotate-0'}`}>▼</span>
                                  </div>

                                  <div className={`w-3 h-3 flex items-center justify-center border transition-colors ${
                                    allChildrenSelected
                                    ? 'border-black bg-black'
                                    : someChildrenSelected ? 'border-black' : 'border-neutral-300 hover:border-black'
                                  }`}>
                                    {allChildrenSelected && <div className="w-1 h-1 bg-white" />}
                                    {!allChildrenSelected && someChildrenSelected && <div className="w-1 h-1 bg-black" />}
                                  </div>
                                  <span className="text-[10px] uppercase font-bold tracking-wider text-neutral-500">Group</span>
                                  <span className="text-xs font-medium text-neutral-700 truncate">
                                    <HighlightMatch text={node.id} match={taskFilter} />
                                  </span>
                                </div>
                                {!isCollapsed && node.children.map(child => (
                                  <div
                                    key={child.id}
                                    onClick={() => toggleTask(child.id)}
                                    className={`group flex items-center gap-2 pl-8 pr-3 py-1.5 cursor-pointer transition-colors ${
                                      selectedTasks.has(child.id)
                                        ? 'bg-neutral-100 text-neutral-900'
                                        : 'text-neutral-500 hover:bg-neutral-50 hover:text-neutral-900'
                                    }`}
                                  >
                                    <div className="flex items-center gap-2 min-w-0 flex-1">
                                      <div className={`w-3 h-3 flex items-center justify-center border transition-colors ${
                                        selectedTasks.has(child.id)
                                        ? 'border-black bg-black'
                                        : 'border-neutral-300 group-hover:border-black'
                                      }`}>
                                        {selectedTasks.has(child.id) && <div className="w-1 h-1 bg-white" />}
                                      </div>
                                      <span className="text-xs font-mono truncate">
                                        <HighlightMatch text={child.id} match={taskFilter} />
                                      </span>
                                    </div>
                                    <button
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        previewTaskYaml(child.id)
                                      }}
                                      className="text-neutral-400 hover:text-neutral-900 transition-colors"
                                      title={`Preview ${child.id} YAML`}
                                    >
                                      <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                                        <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6Z" />
                                        <circle cx="12" cy="12" r="3" />
                                      </svg>
                                    </button>
                                  </div>
                                ))}
                              </div>
                            )
                          } else {
                            return (
                              <div
                                key={node.task.id}
                                onClick={() => toggleTask(node.task.id)}
                                className={`group flex items-center gap-2 px-3 py-1.5 border-b border-neutral-50 last:border-b-0 cursor-pointer transition-colors ${
                                  selectedTasks.has(node.task.id)
                                    ? 'bg-neutral-100 text-neutral-900'
                                    : 'hover:bg-neutral-50 text-neutral-500 hover:text-neutral-900'
                                }`}
                              >
                                <div className="flex items-center gap-2 min-w-0 flex-1">
                                  <div className="w-3"></div>
                                  <div className={`w-3 h-3 flex items-center justify-center border transition-colors ${
                                    selectedTasks.has(node.task.id)
                                    ? 'border-black bg-black'
                                    : 'border-neutral-300 group-hover:border-black'
                                  }`}>
                                    {selectedTasks.has(node.task.id) && <div className="w-1 h-1 bg-white" />}
                                  </div>
                                  <span className="text-xs font-mono truncate">
                                    <HighlightMatch text={node.task.id} match={taskFilter} />
                                  </span>
                                </div>
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation()
                                    previewTaskYaml(node.task.id)
                                  }}
                                  className="text-neutral-400 hover:text-neutral-900 transition-colors"
                                  title={`Preview ${node.task.id} YAML`}
                                >
                                  <svg className="w-3.5 h-3.5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
                                    <path d="M2 12s3.5-6 10-6 10 6 10 6-3.5 6-10 6-10-6-10-6Z" />
                                    <circle cx="12" cy="12" r="3" />
                                  </svg>
                                </button>
                              </div>
                            )
                          }
                        })}
                      </div>
                    </div>
                  )}
                </div>

                {requiresJudgeConfig && (
                  <div className="grid grid-cols-2 gap-4">
                    <div className="group">
                      <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">LLM API URL</label>
                      <input
                        value={judgeApiUrl}
                        onChange={e => setJudgeApiUrl(e.target.value)}
                        placeholder={DEFAULT_JUDGE_API_URL}
                        className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors placeholder-neutral-400 text-neutral-600"
                      />
                    </div>
                    <div className="group">
                      <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">LLM API Key</label>
                      <input
                        type="password"
                        value={judgeApiKey}
                        onChange={e => setJudgeApiKey(e.target.value)}
                        placeholder="sk-..."
                        className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors placeholder-neutral-400 text-neutral-600"
                      />
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-2 gap-4">
                  <div className="group">
                    <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Batch Size</label>
                    <input
                      type="number"
                      value={batchSize}
                      onChange={e => setBatchSize(e.target.value)}
                      className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors text-neutral-600"
                    />
                  </div>
                  <div className="group">
                    <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Limit</label>
                    <input
                      type="number"
                      value={limit}
                      onChange={e => setLimit(e.target.value)}
                      placeholder="All"
                      className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors placeholder-neutral-400 text-neutral-600"
                    />
                  </div>
                </div>

	                {!isApiEval && (
	                  <div className="group">
	                      <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Device</label>
	                      <input
	                        value={device}
	                        onChange={e => setDevice(e.target.value)}
	                        placeholder="cuda:0"
	                        className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors placeholder-neutral-400 text-neutral-600"
	                      />
	                  </div>
	                )}

                <div className="group">
                    <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Output Path</label>
                    <input
                      value={outputPath}
                      onChange={e => setOutputPath(e.target.value)}
                      placeholder="./logs/"
                      className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors placeholder-neutral-400 text-neutral-600"
                    />
                </div>

	                {!isApiEval && (
	                  <div className="grid grid-cols-2 gap-4">
	                    <div className="group">
	                      <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Tensor Parallel</label>
	                      <input
	                        type="number"
	                        value={modelTp}
	                        onChange={e => setModelTp(e.target.value)}
	                        className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors text-neutral-600"
	                      />
	                    </div>
	                    <div className="group">
	                      <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Max Model Len</label>
	                      <input
	                        type="number"
	                        value={maxModelLen}
	                        onChange={e => setMaxModelLen(e.target.value)}
	                        className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors text-neutral-600"
	                      />
	                    </div>
	                    <div className="group">
	                      <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">GPU Memory</label>
	                      <input
	                        type="number"
	                        step="0.01"
	                        value={gpuMemoryUtilization}
	                        onChange={e => setGpuMemoryUtilization(e.target.value)}
	                        className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors text-neutral-600"
	                      />
	                    </div>
	                    <div className="group">
	                      <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Max Seqs</label>
	                      <input
	                        type="number"
	                        value={maxNumSeqs}
	                        onChange={e => setMaxNumSeqs(e.target.value)}
	                        className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors text-neutral-600"
	                      />
	                    </div>
	                    <div className="group">
	                      <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Base Port</label>
	                      <input
	                        type="number"
	                        value={basePort}
	                        onChange={e => setBasePort(e.target.value)}
	                        className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors text-neutral-600"
	                      />
	                    </div>
	                    <div className="group">
	                      <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Concurrency</label>
	                      <input
	                        type="number"
	                        value={concurrency}
	                        onChange={e => setConcurrency(e.target.value)}
	                        className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs font-mono focus:border-black focus:outline-none transition-colors text-neutral-600"
	                      />
	                    </div>
	                  </div>
	                )}

                <div className="group">
                  <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 group-focus-within:text-neutral-900 transition-colors">Generation Kwargs</label>
                  <textarea
                    value={genKwargs}
                    onChange={e => setGenKwargs(e.target.value)}
                    className="w-full bg-white border border-neutral-200 px-3 py-2 text-xs h-20 resize-y focus:border-black focus:outline-none transition-colors leading-relaxed text-neutral-600 font-mono"
                    spellCheck={false}
                  />
                </div>

	                {!isApiEval && (
	                  <div className="flex items-center justify-between gap-3 border border-neutral-200 bg-white px-3 py-2">
	                    <span className="text-[10px] font-bold text-neutral-500 uppercase tracking-wider">Thinking</span>
	                    <button
	                      type="button"
	                      role="switch"
	                      aria-checked={enableThinking}
	                      onClick={() => setEnableThinking(value => !value)}
	                      className={`relative flex h-6 w-14 shrink-0 items-center rounded-full border px-0.5 transition-colors focus:outline-none ${
	                        enableThinking
	                          ? 'border-neutral-900 bg-neutral-900'
	                          : 'border-neutral-300 bg-neutral-100'
	                      }`}
	                    >
	                      <span
	                        className={`absolute text-[9px] font-bold uppercase leading-none transition-colors ${
	                          enableThinking ? 'left-2 text-white' : 'right-2 text-neutral-500'
	                        }`}
	                      >
	                        {enableThinking ? 'On' : 'Off'}
	                      </span>
	                      <span
	                        className={`relative z-10 h-5 w-5 rounded-full bg-white shadow-sm transition-transform ${
	                          enableThinking ? 'translate-x-8' : 'translate-x-0'
	                        }`}
	                      />
	                    </button>
	                  </div>
	                )}

                <label className="flex items-center gap-2 text-[10px] font-bold text-neutral-400 uppercase tracking-wider">
                  <input
                    type="checkbox"
                    checked={debugMode}
                    onChange={e => setDebugMode(e.target.checked)}
                    className="h-3 w-3 accent-black"
                  />
                  Debug
                </label>

                <div className="group">
                  <label className="block text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5">Verbosity</label>
                  <Select
                    value={verbosity}
                    onChange={setVerbosity}
                    options={[
                      { value: 'DEBUG', label: 'DEBUG' },
                      { value: 'INFO', label: 'INFO' },
                      { value: 'WARNING', label: 'WARNING' },
                      { value: 'ERROR', label: 'ERROR' },
                    ]}
                  />
                </div>

                <div className="group">
                  <label
                    className="flex items-center gap-2 text-[10px] font-bold text-neutral-400 uppercase tracking-wider mb-1.5 cursor-pointer"
                    onClick={() => setEnvVarsExpanded(!envVarsExpanded)}
                  >
                    <span className={`transform transition-transform ${envVarsExpanded ? 'rotate-0' : '-rotate-90'}`}>▼</span>
                    Environment Variables
                  </label>
                  {envVarsExpanded && (
                    <ShellEditor
                      value={envVars}
                      onChange={setEnvVars}
                      placeholder="export KEY=VALUE..."
                      className="h-32 w-full resize-y min-h-[80px] max-h-[400px]"
                    />
                  )}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="flex-1 flex flex-col bg-neutral-50/30 min-w-0">
          {!logsMaximized && (
            <>
              <div className="h-1/3 border-b border-neutral-200 flex flex-col bg-white transition-all duration-300">
                <div className="px-6 py-4 border-b border-neutral-100 flex items-center justify-between bg-white">
                  <h2 className="text-xs font-bold text-neutral-400 uppercase tracking-widest">Command</h2>
                  <button
                    onClick={() => navigator.clipboard.writeText(command)}
                    className="text-[10px] text-neutral-400 hover:text-neutral-900 uppercase tracking-wider transition-colors"
                  >
                    Copy
                  </button>
                </div>
                <div className="flex-1 overflow-auto p-6 font-mono text-xs text-neutral-600 bg-neutral-50/50 scrollbar-thin selection:bg-black selection:text-white">
                  <div className="whitespace-pre-wrap leading-relaxed break-all">
                    {highlightShell(command)}
                  </div>
                </div>
              </div>
              <div className="px-6 py-3 border-b border-neutral-200 bg-white flex gap-2 items-center">
                <button
                  onClick={startEval}
                  disabled={startDisabled}
                  title={defaultsLoaded ? 'Start evaluation' : 'Loading DLC defaults'}
                  className={`w-28 py-1.5 text-xs font-medium uppercase tracking-wider transition-all duration-200 ${
                    startDisabled
                      ? 'bg-neutral-100 text-neutral-400 cursor-not-allowed border border-neutral-200'
                      : 'bg-black text-white hover:bg-neutral-800 border border-black shadow-sm'
                  }`}
                >
                  {status === 'running' ? 'Running...' : defaultsLoaded ? 'Start' : 'Loading...'}
                </button>
                <button
                  onClick={stopEval}
                  disabled={status !== 'running'}
                  className={`w-28 py-1.5 text-xs font-medium uppercase tracking-wider transition-all duration-200 ${
                    status !== 'running'
                      ? 'bg-transparent text-neutral-300 border border-neutral-200 cursor-not-allowed'
                      : 'bg-white text-neutral-900 border border-neutral-200 hover:border-black shadow-sm'
                  }`}
                >
                  Stop
                </button>
              </div>
            </>
          )}

          <div className="flex-1 flex flex-col bg-white transition-all duration-300 min-h-0">
            <div className="px-6 py-4 border-b border-neutral-100 flex items-center justify-between bg-white">
              <h2 className="text-xs font-bold text-neutral-400 uppercase tracking-widest">Log Output</h2>
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setLogsMaximized(!logsMaximized)}
                  className="text-[10px] text-neutral-400 hover:text-neutral-900 uppercase tracking-wider transition-colors"
                >
                  {logsMaximized ? 'Restore View' : 'Maximize Logs'}
                </button>
                <div className="w-px h-3 bg-neutral-200"></div>
                <button
                  onClick={() => setOutput([])}
                  className="text-[10px] text-neutral-400 hover:text-neutral-900 uppercase tracking-wider transition-colors"
                >
                  Clear
                </button>
              </div>
            </div>
            <div ref={outputRef} className="flex-1 overflow-auto p-6 font-mono text-xs leading-relaxed bg-white scrollbar-thin selection:bg-black selection:text-white">
              {output.map((line, i) => (
                <div key={i} className="whitespace-pre-wrap mb-1">{highlightLog(line)}</div>
              ))}
              {output.length === 0 && (
                <div className="text-neutral-400 italic">Waiting for process...</div>
              )}
            </div>
          </div>
        </div>
      </div>
          </section>
          <section className="flex h-full w-full shrink-0 overflow-hidden">
            <LogViewer />
          </section>
          <section className="flex h-full w-full shrink-0 overflow-hidden">
            <TaskBuilder onTaskCreated={loadTasks} />
          </section>
        </div>
      </main>

      {yamlPreview && (
        <div
          className="fixed inset-0 z-50 bg-black/50 flex items-center justify-center p-4"
          onClick={() => setYamlPreview(null)}
        >
          <div
            className="w-full max-w-2xl max-h-[80vh] bg-white border border-neutral-200 shadow-xl flex flex-col"
            onClick={e => e.stopPropagation()}
          >
            <div className="px-4 py-3 border-b border-neutral-200 flex items-center justify-between gap-3">
              <div className="min-w-0 flex-1">
                <h3 className="text-xs font-bold text-neutral-500 uppercase tracking-widest">{yamlPreview.title}</h3>
                <p className="text-[10px] text-neutral-400 font-mono mt-1 truncate" title={yamlPreview.subtitle}>{yamlPreview.subtitle}</p>
              </div>
              <div className="flex items-center gap-2 flex-shrink-0">
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(yamlPreview.yaml)
                  }}
                  className="px-2 py-1 text-[10px] uppercase tracking-wider font-medium text-neutral-400 hover:text-neutral-900 border border-neutral-200 hover:border-neutral-400 transition-colors"
                  title="Copy to clipboard"
                >
                  Copy
                </button>
                {yamlPreview.download_filename && (
                  <button
                    onClick={() => {
                      const blob = new Blob([yamlPreview.yaml], { type: 'text/yaml' })
                      const url = URL.createObjectURL(blob)
                      const a = document.createElement('a')
                      a.href = url
                      a.download = yamlPreview.download_filename!
                      document.body.appendChild(a)
                      a.click()
                      document.body.removeChild(a)
                      URL.revokeObjectURL(url)
                    }}
                    className="px-2 py-1 text-[10px] uppercase tracking-wider font-medium text-white bg-black hover:bg-neutral-800 transition-colors"
                    title="Download YAML file"
                  >
                    Download
                  </button>
                )}
                <button
                  onClick={() => setYamlPreview(null)}
                  className="text-neutral-400 hover:text-neutral-900 text-sm leading-none transition-colors ml-1"
                  title="Close preview"
                >
                  ✕
                </button>
              </div>
            </div>
            <div className="flex-1 overflow-auto p-4 bg-neutral-50/50 scrollbar-thin">
              <pre className="text-xs font-mono leading-relaxed whitespace-pre-wrap break-words text-neutral-700">
                {yamlPreview.yaml}
              </pre>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
