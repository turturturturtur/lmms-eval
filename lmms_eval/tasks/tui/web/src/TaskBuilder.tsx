import { useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'

const API_BASE = ''
const DEFAULT_TASK_ID = 'custom_vqa_task'
const TASK_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]*$/

const PYTHON_KEYWORDS = new Set([
  'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class',
  'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global',
  'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise',
  'return', 'try', 'while', 'with', 'yield',
])

const PYTHON_BUILTINS = new Set([
  'bool', 'dict', 'float', 'int', 'len', 'list', 'max', 'min', 'range', 'set', 'str', 'sum',
])

type EditorLanguage = 'yaml' | 'python'
type SaveStatus = { kind: 'idle' | 'success' | 'error'; message: string }

interface TaskCreateResponse {
  task_id: string
  task_dir: string
  yaml_path: string
  python_path: string
  discovered_task_count: number
}

interface TaskBuilderProps {
  onTaskCreated?: () => void
}

interface CodeEditorProps {
  label: string
  language: EditorLanguage
  value: string
  onChange: (value: string) => void
  onCopy: () => void
  onDownload: () => void
}

function toPythonPrefix(taskId: string): string {
  const normalized = taskId.trim().replace(/[^A-Za-z0-9_]/g, '_') || 'custom_task'
  return /^[A-Za-z_]/.test(normalized) ? normalized : `task_${normalized}`
}

function makeYamlTemplate(taskId: string): string {
  const prefix = toPythonPrefix(taskId)
  return `dataset_path: /abs/path/to/dataset_or_hf_repo
dataset_kwargs:
  data_files: /abs/path/to/data.jsonl
task: ${taskId}
test_split: train
output_type: generate_until
doc_to_visual: !function utils.${prefix}_doc_to_visual
doc_to_text: !function utils.${prefix}_doc_to_text
doc_to_target: !function utils.${prefix}_doc_to_target
generation_kwargs:
  max_new_tokens: 64
  temperature: 0
  top_p: 1.0
  do_sample: false
process_results: !function utils.${prefix}_process_results
metric_list:
  - metric: ${prefix}_exact_match
    aggregation: mean
    higher_is_better: true
lmms_eval_specific_kwargs:
  default:
    pre_prompt: ""
    post_prompt: "\\nAnswer with the final answer only."
metadata:
  - version: 0.0
`
}

function makePythonTemplate(taskId: string): string {
  const prefix = toPythonPrefix(taskId)
  return `from __future__ import annotations

import re
from typing import Any

from PIL import Image


def _normalize_answer(value: Any) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"\\\\s+", " ", text)
    return text


def ${prefix}_doc_to_visual(doc: dict[str, Any]) -> list[Image.Image]:
    image = doc.get("image")
    if image is None:
        return []
    if not isinstance(image, Image.Image):
        raise TypeError("Expected doc['image'] to be a PIL.Image.Image")
    return [image.convert("RGB")]


def ${prefix}_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: dict[str, Any] | None = None) -> str:
    kwargs = lmms_eval_specific_kwargs or {}
    question = str(doc["question"]).strip()
    pre_prompt = str(kwargs.get("pre_prompt", ""))
    post_prompt = str(kwargs.get("post_prompt", ""))
    return f"{pre_prompt}{question}{post_prompt}"


def ${prefix}_doc_to_target(doc: dict[str, Any]) -> str:
    return str(doc["answer"]).strip()


def ${prefix}_process_results(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    prediction = results[0] if results else ""
    score = float(_normalize_answer(prediction) == _normalize_answer(${prefix}_doc_to_target(doc)))
    return {"${prefix}_exact_match": score}
`
}

function splitComment(line: string): [string, string] {
  const index = line.indexOf('#')
  if (index < 0) return [line, '']
  return [line.slice(0, index), line.slice(index)]
}

function highlightYamlInline(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  const tokenRegex = /(\s+|![A-Za-z_][A-Za-z0-9_.-]*|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b(?:true|false|null|none)\b|\b\d+(?:\.\d+)?\b|[{}\[\],]|[^\s{}\[\],]+)/gi
  let match: RegExpExecArray | null
  let index = 0
  while ((match = tokenRegex.exec(text)) !== null) {
    const token = match[0]
    let className = 'text-neutral-700'
    if (/^\s+$/.test(token)) className = 'text-transparent'
    else if (token.startsWith('!')) className = 'text-emerald-700 font-semibold'
    else if (/^["']/.test(token)) className = 'text-amber-700'
    else if (/^(true|false|null|none)$/i.test(token)) className = 'text-blue-700 font-medium'
    else if (/^\d/.test(token)) className = 'text-purple-700'
    else if (/^[{}\[\],]$/.test(token)) className = 'text-neutral-400 font-bold'
    else if (keyPrefix && token.includes(keyPrefix)) className = 'text-neutral-900 font-semibold'
    nodes.push(<span key={index++} className={className}>{token}</span>)
  }
  return nodes
}

function highlightYamlLine(line: string, lineIndex: number): ReactNode[] {
  const [code, comment] = splitComment(line)
  const keyMatch = code.match(/^(\s*)([A-Za-z0-9_.-]+)(\s*:)/)
  const nodes: ReactNode[] = []
  if (keyMatch) {
    nodes.push(<span key={`${lineIndex}-indent`}>{keyMatch[1]}</span>)
    nodes.push(<span key={`${lineIndex}-key`} className="text-blue-800 font-semibold">{keyMatch[2]}</span>)
    nodes.push(<span key={`${lineIndex}-colon`} className="text-neutral-400 font-bold">{keyMatch[3]}</span>)
    nodes.push(...highlightYamlInline(code.slice(keyMatch[0].length), keyMatch[2]))
  } else {
    nodes.push(...highlightYamlInline(code, ''))
  }
  if (comment) {
    nodes.push(<span key={`${lineIndex}-comment`} className="text-neutral-400 italic">{comment}</span>)
  }
  return nodes
}

function highlightPythonLine(line: string, lineIndex: number): ReactNode[] {
  const nodes: ReactNode[] = []
  const tokenRegex = /(\s+|#[^\n]*|@[A-Za-z_][A-Za-z0-9_.]*|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\b\d+(?:\.\d+)?\b|\b[A-Za-z_][A-Za-z0-9_]*\b|[^\sA-Za-z0-9_]+)/g
  let match: RegExpExecArray | null
  let index = 0
  while ((match = tokenRegex.exec(line)) !== null) {
    const token = match[0]
    let className = 'text-neutral-700'
    if (/^\s+$/.test(token)) className = 'text-transparent'
    else if (token.startsWith('#')) className = 'text-neutral-400 italic'
    else if (token.startsWith('@')) className = 'text-emerald-700 font-semibold'
    else if (/^["']/.test(token)) className = 'text-amber-700'
    else if (/^\d/.test(token)) className = 'text-purple-700'
    else if (PYTHON_KEYWORDS.has(token)) className = 'text-blue-800 font-semibold'
    else if (PYTHON_BUILTINS.has(token)) className = 'text-cyan-700 font-medium'
    else if (/^[()[\]{}.,:|=+\-*/<>!]+$/.test(token)) className = 'text-neutral-400 font-bold'
    nodes.push(<span key={`${lineIndex}-${index++}`} className={className}>{token}</span>)
  }
  return nodes
}

function HighlightedCode({ value, language }: { value: string; language: EditorLanguage }) {
  const lines = value.split('\n')
  return (
    <>
      {lines.map((line, index) => (
        <span key={index}>
          {language === 'yaml' ? highlightYamlLine(line, index) : highlightPythonLine(line, index)}
          {index < lines.length - 1 ? '\n' : null}
        </span>
      ))}
    </>
  )
}

function CodeEditor({ label, language, value, onChange, onCopy, onDownload }: CodeEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const preRef = useRef<HTMLPreElement>(null)

  const handleScroll = () => {
    if (!textareaRef.current || !preRef.current) return
    preRef.current.scrollTop = textareaRef.current.scrollTop
    preRef.current.scrollLeft = textareaRef.current.scrollLeft
  }

  return (
    <section className="flex min-h-0 flex-col border border-neutral-200 bg-white">
      <div className="flex h-11 shrink-0 items-center justify-between border-b border-neutral-100 px-4">
        <div className="flex items-center gap-2">
          <span className="text-xs font-bold uppercase tracking-widest text-neutral-400">{label}</span>
          <span className="border border-neutral-200 px-1.5 py-0.5 text-[10px] font-mono text-neutral-500">{language}</span>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onCopy}
            className="border border-neutral-200 px-2 py-1 text-[10px] uppercase tracking-wider text-neutral-500 hover:border-black hover:text-black"
          >
            Copy
          </button>
          <button
            type="button"
            onClick={onDownload}
            className="border border-neutral-200 px-2 py-1 text-[10px] uppercase tracking-wider text-neutral-500 hover:border-black hover:text-black"
          >
            Download
          </button>
        </div>
      </div>
      <div className="relative min-h-[360px] flex-1 overflow-hidden bg-white">
        <pre
          ref={preRef}
          className="absolute inset-0 overflow-hidden whitespace-pre px-4 py-3 text-xs leading-relaxed"
          style={{ fontFamily: 'monospace' }}
          aria-hidden="true"
        >
          <HighlightedCode value={value} language={language} />
          <br />
        </pre>
        <textarea
          ref={textareaRef}
          value={value}
          onChange={event => onChange(event.target.value)}
          onScroll={handleScroll}
          className="absolute inset-0 z-10 h-full w-full resize-none overflow-auto whitespace-pre bg-transparent px-4 py-3 font-mono text-xs leading-relaxed text-transparent caret-black outline-none scrollbar-thin scrollbar-thumb-neutral-200 scrollbar-track-transparent"
          spellCheck={false}
          autoCapitalize="off"
          autoComplete="off"
          aria-label={`${label} editor`}
        />
      </div>
    </section>
  )
}

function downloadText(filename: string, content: string, type: string) {
  const blob = new Blob([content], { type })
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  document.body.removeChild(anchor)
  URL.revokeObjectURL(url)
}

export default function TaskBuilder({ onTaskCreated }: TaskBuilderProps) {
  const [taskId, setTaskId] = useState(DEFAULT_TASK_ID)
  const [yamlContent, setYamlContent] = useState(() => makeYamlTemplate(DEFAULT_TASK_ID))
  const [pythonContent, setPythonContent] = useState(() => makePythonTemplate(DEFAULT_TASK_ID))
  const [yamlDirty, setYamlDirty] = useState(false)
  const [pythonDirty, setPythonDirty] = useState(false)
  const [overwrite, setOverwrite] = useState(false)
  const [saving, setSaving] = useState(false)
  const [saveStatus, setSaveStatus] = useState<SaveStatus>({ kind: 'idle', message: '' })

  const trimmedTaskId = taskId.trim()
  const taskIdError = useMemo(() => {
    if (!trimmedTaskId) return 'task_id is required'
    if (!TASK_ID_PATTERN.test(trimmedTaskId)) return 'task_id must match ^[A-Za-z0-9][A-Za-z0-9_-]*$'
    return ''
  }, [trimmedTaskId])

  const targetPath = `/lmms_eval/tasks/${trimmedTaskId || '<task_id>'}/`

  const updateTaskId = (value: string) => {
    setTaskId(value)
    const nextId = value.trim() || DEFAULT_TASK_ID
    if (!yamlDirty) setYamlContent(makeYamlTemplate(nextId))
    if (!pythonDirty) setPythonContent(makePythonTemplate(nextId))
    setSaveStatus({ kind: 'idle', message: '' })
  }

  const resetTemplates = () => {
    const nextId = trimmedTaskId || DEFAULT_TASK_ID
    setYamlContent(makeYamlTemplate(nextId))
    setPythonContent(makePythonTemplate(nextId))
    setYamlDirty(false)
    setPythonDirty(false)
    setSaveStatus({ kind: 'idle', message: '' })
  }

  const saveTask = async () => {
    if (taskIdError) {
      setSaveStatus({ kind: 'error', message: taskIdError })
      return
    }
    if (!yamlContent.trim() || !pythonContent.trim()) {
      setSaveStatus({ kind: 'error', message: 'YAML and Python content are required' })
      return
    }

    setSaving(true)
    setSaveStatus({ kind: 'idle', message: '' })
    try {
      const response = await fetch(`${API_BASE}/tasks/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          task_id: trimmedTaskId,
          yaml_content: yamlContent,
          python_content: pythonContent,
          overwrite,
        }),
      })
      const data = await response.json().catch(() => ({}))
      if (!response.ok) {
        throw new Error(data.detail || response.statusText)
      }
      const payload = data as TaskCreateResponse
      setSaveStatus({
        kind: 'success',
        message: `Saved ${payload.yaml_path} and ${payload.python_path}; ${payload.discovered_task_count} tasks discovered.`,
      })
      onTaskCreated?.()
    } catch (error) {
      setSaveStatus({ kind: 'error', message: error instanceof Error ? error.message : 'Failed to save task' })
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="flex h-full min-h-0 w-full flex-1 flex-col bg-neutral-50/30">
      <div className="shrink-0 border-b border-neutral-200 bg-white px-6 py-4">
        <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(260px,360px)_1fr_auto] xl:items-start">
          <div>
            <label className="mb-1.5 block text-[10px] font-bold uppercase tracking-wider text-neutral-400">Task ID</label>
            <input
              aria-label="Task ID"
              value={taskId}
              onChange={event => updateTaskId(event.target.value)}
              className={`w-full border bg-white px-3 py-2 font-mono text-xs text-neutral-700 outline-none transition-colors ${
                taskIdError ? 'border-red-300 focus:border-red-600' : 'border-neutral-200 focus:border-black'
              }`}
              spellCheck={false}
            />
            <div className={`mt-1 font-mono text-[10px] ${taskIdError ? 'text-red-600' : 'text-neutral-400'}`}>
              {taskIdError || targetPath}
            </div>
          </div>
          <div className="min-w-0">
            <div className="mb-1.5 text-[10px] font-bold uppercase tracking-wider text-neutral-400">Target Files</div>
            <div className="truncate border border-neutral-200 bg-neutral-50 px-3 py-2 font-mono text-xs text-neutral-600" title={`${targetPath}${trimmedTaskId || '<task_id>'}.yaml / utils.py`}>
              {targetPath}{trimmedTaskId || '<task_id>'}.yaml / utils.py
            </div>
            <div className="invisible mt-1 font-mono text-[10px]" aria-hidden="true">placeholder</div>
          </div>
          <div className="flex min-w-0 flex-col xl:items-end">
            <div className="invisible mb-1.5 text-[10px] font-bold uppercase tracking-wider" aria-hidden="true">Actions</div>
            <div className="flex flex-wrap items-center gap-2 xl:justify-end">
              <label className="flex h-9 items-center gap-2 border border-neutral-200 bg-white px-3 text-[10px] font-bold uppercase tracking-wider text-neutral-500">
                <input
                  type="checkbox"
                  checked={overwrite}
                  onChange={event => setOverwrite(event.target.checked)}
                  className="h-3 w-3 accent-black"
                />
                Overwrite
              </label>
              <button
                type="button"
                onClick={resetTemplates}
                className="h-9 border border-neutral-200 bg-white px-4 text-[10px] font-semibold uppercase tracking-wider text-neutral-500 hover:border-black hover:text-black"
              >
                Reset
              </button>
              <button
                type="button"
                onClick={() => void saveTask()}
                disabled={saving || Boolean(taskIdError)}
                className={`h-9 px-5 text-[10px] font-semibold uppercase tracking-wider ${
                  saving || taskIdError
                    ? 'cursor-not-allowed border border-neutral-200 bg-neutral-100 text-neutral-400'
                    : 'border border-black bg-black text-white hover:bg-neutral-800'
                }`}
              >
                {saving ? 'Saving...' : 'Save Task'}
              </button>
            </div>
            <div className="invisible mt-1 font-mono text-[10px]" aria-hidden="true">placeholder</div>
          </div>
        </div>
        {saveStatus.message && (
          <div
            className={`mt-3 border px-3 py-2 font-mono text-[11px] ${
              saveStatus.kind === 'success'
                ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
                : 'border-red-200 bg-red-50 text-red-700'
            }`}
          >
            {saveStatus.message}
          </div>
        )}
      </div>

      <div className="grid flex-1 min-h-0 grid-cols-[minmax(0,1fr)_minmax(0,1fr)] overflow-hidden bg-white">
        <CodeEditor
          label="YAML"
          language="yaml"
          value={yamlContent}
          onChange={value => {
            setYamlContent(value)
            setYamlDirty(true)
          }}
          onCopy={() => navigator.clipboard.writeText(yamlContent)}
          onDownload={() => downloadText(`${trimmedTaskId || DEFAULT_TASK_ID}.yaml`, yamlContent, 'text/yaml')}
        />
        <CodeEditor
          label="Python"
          language="python"
          value={pythonContent}
          onChange={value => {
            setPythonContent(value)
            setPythonDirty(true)
          }}
          onCopy={() => navigator.clipboard.writeText(pythonContent)}
          onDownload={() => downloadText('utils.py', pythonContent, 'text/x-python')}
        />
      </div>
    </div>
  )
}
