export interface HistoryWindow {
  startTime: string
  endTime: string
}

export interface HistoryJob {
  job_id: string
  create_time: string
}

export const INITIAL_HISTORY_DAYS: number
export const HISTORY_LOAD_MORE_DAYS: number
export function makeInitialHistoryWindow(now?: Date): HistoryWindow
export function makePreviousHistoryWindow(currentStartTime: string): HistoryWindow
export function mergeJobsById<T extends HistoryJob>(current: T[], incoming: T[]): T[]
