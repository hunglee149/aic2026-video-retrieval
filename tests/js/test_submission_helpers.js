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

test('groups backend validation issues by query', () => {
  const groups = helpers.groupValidationIssues([
    { query_id: 'query-p1-1-kis', message: 'Frame must be an integer' },
    { query_id: 'query-p1-2-qa', message: 'Answer is required' },
    { query_id: 'query-p1-1-kis', message: 'Too many rows' },
    { message: 'Manifest is missing' },
  ]);

  assert.deepEqual(groups, {
    'query-p1-1-kis': ['Frame must be an integer', 'Too many rows'],
    'query-p1-2-qa': ['Answer is required'],
    general: ['Manifest is missing'],
  });
});

test('submission controls are visible in the baseline UI', () => {
  const html = fs.readFileSync(path.join(__dirname, '../../aic/ui/static/index.html'), 'utf8');
  assert.match(html, /id="query-file-upload"[^>]*accept="\.zip,\.txt"[^>]*multiple/);
  assert.match(html, /id="query-manifest-list"/);
  assert.match(html, /id="trake-events-confirmed"/);
  assert.match(html, /id="validation-report"/);
});
