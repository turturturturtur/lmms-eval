import assert from 'node:assert/strict'
import test from 'node:test'

import { makeInitialHistoryWindow, makePreviousHistoryWindow, mergeJobsById } from './historyWindow.js'

test('initial history window covers the previous 30 days', () => {
  assert.deepEqual(makeInitialHistoryWindow(new Date('2026-07-28T12:00:00Z')), {
    startTime: '2026-06-28T12:00:00Z',
    endTime: '2026-07-28T12:00:00Z',
  })
})

test('previous history window extends backward by 15 days', () => {
  assert.deepEqual(makePreviousHistoryWindow('2026-06-28T12:00:00Z'), {
    startTime: '2026-06-13T12:00:00Z',
    endTime: '2026-06-28T12:00:00Z',
  })
})

test('job merge deduplicates boundary rows and keeps newest first', () => {
  const current = [
    { job_id: 'dlc-new', create_time: '2026-07-28T10:00:00Z' },
    { job_id: 'dlc-boundary', create_time: '2026-06-28T12:00:00Z' },
  ]
  const older = [
    { job_id: 'dlc-boundary', create_time: '2026-06-28T12:00:00Z' },
    { job_id: 'dlc-old', create_time: '2026-06-20T08:00:00Z' },
  ]

  assert.deepEqual(mergeJobsById(current, older).map(job => job.job_id), ['dlc-new', 'dlc-boundary', 'dlc-old'])
})
