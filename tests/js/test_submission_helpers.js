'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const helpers = require('../../aic/ui/static/submission_helpers.js');

test('preserves Q&A answer whitespace while checking the task eligibility', () => {
  const qa = helpers.prepareQaAnswer('  Năm người  ');
  assert.equal(qa, '  Năm người  ');
  assert.equal(helpers.canUseIterative('kis'), true);
  assert.equal(helpers.canUseIterative('qa'), false);
  assert.equal(helpers.canUseIterative('trake'), false);
});

test('counts Q&A answer length in Unicode code points', () => {
  assert.equal(helpers.unicodeCodePointLength?.('a😀𠜎'), 3);
  assert.equal(helpers.unicodeCodePointLength?.('😀'.repeat(100)), 100);
});

test('converts every candidate frame to the one-based submission frame', () => {
  assert.equal(helpers.candidateToSubmissionFrame?.({
    representative_frames: [0],
    start_frame: 7,
  }), 1);
  assert.equal(helpers.candidateToSubmissionFrame?.({
    representative_frames: [],
    start_frame: 7,
  }), 8);
});

test('infers a task only from an exact query TXT suffix', () => {
  assert.equal(helpers.inferTaskFromFilename('query-p1-1-kis.txt'), 'kis');
  assert.equal(helpers.inferTaskFromFilename('query_p1_1_kis.txt'), 'kis');
  assert.equal(helpers.inferTaskFromFilename('query-p1-2-qa.TXT'), 'qa');
  assert.equal(helpers.inferTaskFromFilename('query-p1-3-trake.txt'), 'trake');
  assert.equal(helpers.inferTaskFromFilename('query-kis-notes.txt'), null);
  assert.equal(helpers.inferTaskFromFilename('query-p1-1-kis.csv'), null);
});

test('upserts a manifest item in place without reordering the pack', () => {
  const manifest = [
    { query_id: 'query-p1-1-kis', task: 'kis', text: 'first', source_name: 'first-kis.txt', n_events: 0, events_confirmed: false },
    { query_id: 'query-p1-2-trake', task: 'trake', text: 'second', source_name: 'second-trake.txt', n_events: 2, events_confirmed: false },
  ];
  const result = helpers.upsertManifestItem(manifest, {
    query_id: 'query-p1-2-trake', task: 'trake', text: 'replaced', source_name: 'second-trake.txt', n_events: 3, events_confirmed: true,
  });

  assert.deepEqual(result.map((item) => item.query_id), ['query-p1-1-kis', 'query-p1-2-trake']);
  assert.equal(result[1].text, 'replaced');
  assert.equal(result[1].n_events, 3);
});

test('updates only the selected TRAKE query event state', () => {
  const manifest = [
    { query_id: 'query-p1-1-trake', task: 'trake', text: 'first', source_name: 'first-trake.txt', n_events: 2, events_confirmed: false },
    { query_id: 'query-p1-2-trake', task: 'trake', text: 'second', source_name: 'second-trake.txt', n_events: 4, events_confirmed: false },
  ];
  const result = helpers.updateTrakeState(manifest, 'query-p1-2-trake', 5, true);

  assert.deepEqual(result[0], manifest[0]);
  assert.equal(result[1].n_events, 5);
  assert.equal(result[1].events_confirmed, true);
});

test('resets only the edited TRAKE query confirmation when its count changes', () => {
  const manifest = [
    { query_id: 'query-p1-1-trake', task: 'trake', text: 'first', source_name: 'first-trake.txt', n_events: 2, events_confirmed: true },
    { query_id: 'query-p1-2-trake', task: 'trake', text: 'second', source_name: 'second-trake.txt', n_events: 4, events_confirmed: true },
  ];
  const result = helpers.changeTrakeEventCount(manifest, 'query-p1-2-trake', 5);

  assert.equal(result[0].n_events, 2);
  assert.equal(result[0].events_confirmed, true);
  assert.equal(result[1].n_events, 5);
  assert.equal(result[1].events_confirmed, false);
});

test('allows download only for an exact PASS ZIP response', () => {
  assert.equal(helpers.canDownloadValidatedZip('PASS', 'application/zip'), true);
  assert.equal(helpers.canDownloadValidatedZip('PASS', 'application/zip; charset=binary'), true);
  assert.equal(helpers.canDownloadValidatedZip('PASS', 'text/html'), false);
  assert.equal(helpers.canDownloadValidatedZip('pass', 'application/zip'), false);
});

test('builds TRAKE review slots from the manifest event count', () => {
  assert.deepEqual(helpers.buildTrakeReviewSlots([120, 240], 4), [
    { event: 1, frame: 120, missing: false },
    { event: 2, frame: 240, missing: false },
    { event: 3, frame: null, missing: true },
    { event: 4, frame: null, missing: true },
  ]);
});

test('selecting a manifest query clears stale translated text', () => {
  assert.deepEqual(helpers.manifestQueryFormState({
    query_id: 'query-p1-3-qa', task: 'qa', text: 'Câu hỏi mới', n_events: null, events_confirmed: true,
  }), {
    queryId: 'query-p1-3-qa',
    task: 'qa',
    text: 'Câu hỏi mới',
    nEvents: 1,
    eventsConfirmed: true,
    translatedText: '',
  });
});

test('clears query-local candidate state without discarding saved selections', () => {
  const selections = [{ queryId: 'query-p1-2-qa', answer: 'Đáp án đã lưu' }];
  const manifest = [{ query_id: 'query-p1-2-qa', task: 'qa' }];
  const next = helpers.clearQueryWorkspaceState?.({
    task: 'kis',
    candidates: [{ video_id: 'L01_V001' }],
    selected: 0,
    selections,
    manifest,
    currentFps: 25,
    candidateDraftFrames: { stale: 999 },
    currentPlaybackFrame: 999,
    iterCandidates: [{ video_id: 'L01_V001' }],
    iterCursor: 3,
    iterRound: 2,
    iterRunning: true,
    iterVerdict: { stale: 'matched' },
    iterMatchedList: [{ video_id: 'L01_V001' }],
    iterUnsureList: [{ video_id: 'L01_V002' }],
    iterExcluded: new Set(['stale']),
  });

  assert.equal(next?.selections, selections);
  assert.equal(next?.manifest, manifest);
  assert.deepEqual(next?.candidates, []);
  assert.equal(next?.selected, null);
  assert.equal(next?.currentFps, null);
  assert.deepEqual(next?.candidateDraftFrames, {});
  assert.equal(next?.currentPlaybackFrame, null);
  assert.deepEqual(next?.iterCandidates, []);
  assert.equal(next?.iterRunning, false);
  assert.deepEqual(next?.iterVerdict, {});
  assert.equal(next?.iterExcluded.size, 0);
});

test('installs an uploaded manifest only after both HTTP and report validation pass', () => {
  assert.equal(helpers.canInstallManifest?.(true, { ok: true, manifest: [] }), true);
  assert.equal(helpers.canInstallManifest?.(false, { ok: true, manifest: [] }), false);
  assert.equal(helpers.canInstallManifest?.(true, { ok: false, manifest: [] }), false);
  assert.equal(helpers.canInstallManifest?.(true, { ok: true }), false);
});

test('reports manifest readiness for missing rows and unconfirmed TRAKE events', () => {
  const query = {
    query_id: 'query-p1-3-trake',
    task: 'trake',
    events_confirmed: false,
  };

  assert.deepEqual(helpers.manifestQueryReadiness?.(query, []), {
    ready: false,
    codes: ['missing_rows', 'trake_events_unconfirmed'],
    label: 'Chưa có dòng · Chưa xác nhận events',
  });
  assert.deepEqual(helpers.manifestQueryReadiness?.(query, [
    { queryId: 'query-p1-3-trake', frames: [10, 20] },
  ]), {
    ready: false,
    codes: ['trake_events_unconfirmed'],
    label: 'Chưa xác nhận events',
  });
  assert.deepEqual(helpers.manifestQueryReadiness?.(
    { ...query, events_confirmed: true },
    [{ queryId: 'query-p1-3-trake', frames: [10, 20] }],
  ), {
    ready: true,
    codes: [],
    label: 'Ready',
  });
});

test('groups backend validation issues by query without discarding code or row', () => {
  const groups = helpers.groupValidationIssues([
    { code: 'invalid_frame', query_id: 'query-p1-1-kis', row: 2, message: 'Frame must be an integer' },
    { code: 'qa_missing_answer', query_id: 'query-p1-2-qa', row: 4, message: 'Answer is required' },
    { code: 'too_many_rows', query_id: 'query-p1-1-kis', row: null, message: 'Too many rows' },
    { message: 'Manifest is missing' },
  ]);

  assert.deepEqual(groups, {
    'query-p1-1-kis': [
      { code: 'invalid_frame', query_id: 'query-p1-1-kis', row: 2, message: 'Frame must be an integer' },
      { code: 'too_many_rows', query_id: 'query-p1-1-kis', row: null, message: 'Too many rows' },
    ],
    'query-p1-2-qa': [
      { code: 'qa_missing_answer', query_id: 'query-p1-2-qa', row: 4, message: 'Answer is required' },
    ],
    general: [{ message: 'Manifest is missing' }],
  });
  assert.equal(
    helpers.formatValidationIssue?.(groups['query-p1-1-kis'][0]),
    '[invalid_frame] — query query-p1-1-kis · row 2 — Frame must be an integer',
  );
});

test('submission controls are visible in the baseline UI', () => {
  const html = fs.readFileSync(path.join(__dirname, '../../aic/ui/static/index.html'), 'utf8');
  assert.match(html, /id="query-file-upload"[^>]*accept="\.zip,\.txt"[^>]*multiple/);
  assert.match(html, /id="query-manifest-list"/);
  assert.match(html, /id="trake-events-confirmed"/);
  assert.match(html, /id="validation-report"/);
});
