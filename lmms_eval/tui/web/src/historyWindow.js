export const INITIAL_HISTORY_DAYS = 30
export const HISTORY_LOAD_MORE_DAYS = 15

function requireValidDate(value, fieldName) {
  const date = value instanceof Date ? value : new Date(value)
  if (Number.isNaN(date.getTime())) {
    throw new TypeError(`${fieldName} must be a valid date`)
  }
  return date
}

function formatUtcSeconds(value) {
  return value.toISOString().replace(/\.\d{3}Z$/, 'Z')
}

function subtractUtcDays(value, days) {
  return new Date(value.getTime() - days * 24 * 60 * 60 * 1000)
}

export function makeInitialHistoryWindow(now = new Date()) {
  const end = requireValidDate(now, 'now')
  return {
    startTime: formatUtcSeconds(subtractUtcDays(end, INITIAL_HISTORY_DAYS)),
    endTime: formatUtcSeconds(end),
  }
}

export function makePreviousHistoryWindow(currentStartTime) {
  const end = requireValidDate(currentStartTime, 'currentStartTime')
  return {
    startTime: formatUtcSeconds(subtractUtcDays(end, HISTORY_LOAD_MORE_DAYS)),
    endTime: formatUtcSeconds(end),
  }
}

export function mergeJobsById(current, incoming) {
  const jobsById = new Map()
  for (const job of [...current, ...incoming]) {
    if (!job.job_id) {
      throw new TypeError('job_id must be non-empty')
    }
    if (!jobsById.has(job.job_id)) {
      jobsById.set(job.job_id, job)
    }
  }
  return [...jobsById.values()].sort((left, right) => right.create_time.localeCompare(left.create_time))
}
