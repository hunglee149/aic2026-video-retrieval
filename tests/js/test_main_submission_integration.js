'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const helpers = require('../../aic/ui/static/submission_helpers.js');
const mainSource = fs.readFileSync(
  path.join(__dirname, '../../aic/ui/static/main.js'),
  'utf8',
);

class FakeClassList {
  constructor(owner) {
    this.owner = owner;
    this.values = new Set();
  }

  add(...names) {
    names.forEach((name) => this.values.add(name));
    this.owner.className = [...this.values].join(' ');
  }

  remove(...names) {
    names.forEach((name) => this.values.delete(name));
    this.owner.className = [...this.values].join(' ');
  }

  toggle(name, force) {
    const enabled = force === undefined ? !this.values.has(name) : Boolean(force);
    if (enabled) this.values.add(name);
    else this.values.delete(name);
    this.owner.className = [...this.values].join(' ');
    return enabled;
  }

  contains(name) {
    return this.values.has(name);
  }
}

class FakeElement {
  constructor(document, tagName = 'div', id = '') {
    this.ownerDocument = document;
    this.tagName = String(tagName).toUpperCase();
    this.id = id;
    this.children = [];
    this.parentNode = null;
    this.dataset = {};
    this.style = {};
    this.attributes = {};
    this.listeners = {};
    this.value = '';
    this.checked = false;
    this.disabled = false;
    this.className = '';
    this.classList = new FakeClassList(this);
    this._innerHTML = '';
    this._textContent = '';
  }

  set innerHTML(value) {
    this._innerHTML = String(value);
    this._textContent = '';
    this.children = [];
    this.ownerDocument.innerHTMLAssignments.push(this._innerHTML);
  }

  get innerHTML() {
    return this._innerHTML;
  }

  set textContent(value) {
    this._textContent = String(value ?? '');
    this._innerHTML = '';
    this.children = [];
  }

  get textContent() {
    return this._textContent + this.children.map((child) => child.textContent).join('');
  }

  appendChild(child) {
    child.parentNode = this;
    this.children.push(child);
    return child;
  }

  append(...children) {
    children.forEach((child) => this.appendChild(child));
  }

  replaceChildren(...children) {
    this._innerHTML = '';
    this._textContent = '';
    this.children = [];
    this.append(...children);
  }

  remove() {
    if (!this.parentNode) return;
    this.parentNode.children = this.parentNode.children.filter((child) => child !== this);
    this.parentNode = null;
  }

  addEventListener(type, listener) {
    if (!this.listeners[type]) this.listeners[type] = [];
    this.listeners[type].push(listener);
  }

  dispatchEvent(event) {
    const payload = { target: this, preventDefault() {}, stopPropagation() {}, ...event };
    (this.listeners[payload.type] || []).forEach((listener) => listener(payload));
  }

  click() {
    this.dispatchEvent({ type: 'click' });
  }

  setAttribute(name, value) {
    this.attributes[name] = String(value);
  }

  getAttribute(name) {
    return this.attributes[name] ?? null;
  }

  removeAttribute(name) {
    delete this.attributes[name];
    if (name === 'src') this.src = '';
  }

  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }

  querySelectorAll(selector) {
    const matches = [];
    const match = (element) => {
      if (selector.startsWith('.')) {
        return element.className.split(/\s+/).includes(selector.slice(1));
      }
      if (selector.startsWith('#')) return element.id === selector.slice(1);
      return element.tagName.toLowerCase() === selector.toLowerCase();
    };
    const visit = (element) => {
      element.children.forEach((child) => {
        if (match(child)) matches.push(child);
        visit(child);
      });
    };
    visit(this);
    return matches;
  }

  closest(selector) {
    let current = this;
    while (current) {
      if (selector.startsWith('.') && current.className.split(/\s+/).includes(selector.slice(1))) {
        return current;
      }
      current = current.parentNode;
    }
    return null;
  }

  scrollIntoView() {}
  pause() {}
  load() {}
}

class FakeDocument {
  constructor() {
    this.elements = new Map();
    this.listeners = {};
    this.innerHTMLAssignments = [];
    this.head = new FakeElement(this, 'head');
    this.body = new FakeElement(this, 'body');
    this.activeElement = this.body;
  }

  createElement(tagName) {
    return new FakeElement(this, tagName);
  }

  getElementById(id) {
    if (!this.elements.has(id)) {
      const tagName = id.includes('tbody') ? 'tbody' : id.includes('input') ? 'input' : 'div';
      this.elements.set(id, new FakeElement(this, tagName, id));
    }
    return this.elements.get(id);
  }

  addEventListener(type, listener) {
    if (!this.listeners[type]) this.listeners[type] = [];
    this.listeners[type].push(listener);
  }

  querySelectorAll(selector) {
    return [...this.elements.values()].filter((element) => {
      if (selector.startsWith('.')) {
        return element.className.split(/\s+/).includes(selector.slice(1));
      }
      return element.tagName.toLowerCase() === selector.toLowerCase();
    });
  }
}

function createMainHarness() {
  const document = new FakeDocument();
  const storage = new Map();
  const sandbox = {
    AICSubmissionHelpers: helpers,
    Array,
    Blob,
    console,
    document,
    fetch: async (url) => {
      if (url === '/api/videos') {
        return {
          ok: true,
          async json() { return { ok: true, videos: [] }; }
        };
      }
      if (url === '/api/status') {
        return {
          ok: true,
          async json() { return { ok: true, retriever: 'dummy', retrievers: [], ready_count: 1 }; }
        };
      }
      throw new Error(`unexpected fetch: ${url}`);
    },
    localStorage: {
      getItem(key) { return storage.has(key) ? storage.get(key) : null; },
      setItem(key, value) { storage.set(key, String(value)); },
      removeItem(key) { storage.delete(key); },
      clear() { storage.clear(); },
    },
    requestAnimationFrame(callback) { callback(); },
    setInterval() { return 1; },
    setTimeout() { return 1; },
    clearTimeout() {},
    confirm() { return true; },
    URL: Object.assign(
      class {
        constructor(url, base) {
          this.pathname = (url.startsWith('http') || (base && base.startsWith('http')))
            ? '/' + url.split('/').slice(3).join('/')
            : url;
        }
      },
      {
        createObjectURL() { return 'blob:test'; },
        revokeObjectURL() {},
      }
    ),
    location: {
      href: 'http://localhost:8000/',
      protocol: 'http:',
      host: 'localhost:8000',
      hostname: 'localhost',
      pathname: '/',
    },
    WebSocket: class FakeWebSocket {
      static get CONNECTING() { return 0; }
      static get OPEN() { return 1; }
      static get CLOSING() { return 2; }
      static get CLOSED() { return 3; }

      constructor(url) {
        this.url = url;
        this.sent = [];
        this.readyState = 1; // OPEN
        setTimeout(() => {
          if (this.onopen) this.onopen();
        }, 0);
      }
      send(data) {
        this.sent.push(data);
      }
      close() {
        this.readyState = 3; // CLOSED
        if (this.onclose) this.onclose();
      }
    },
    addEventListener(type, listener) {
      if (type === 'DOMContentLoaded') {
        if (!document.listeners) document.listeners = {};
        if (!document.listeners.DOMContentLoaded) document.listeners.DOMContentLoaded = [];
        document.listeners.DOMContentLoaded.push(listener);
      } else {
        if (!this._listeners) this._listeners = {};
        if (!this._listeners[type]) this._listeners[type] = [];
        this._listeners[type].push(listener);
      }
    },
  };
  sandbox.window = sandbox;
  sandbox.globalThis = sandbox;
  const context = vm.createContext(sandbox);
  vm.runInContext(mainSource, context, { filename: 'main.js' });
  return { context, document, storage };
}

function setState(context, name, value) {
  context.__testValue = value;
  vm.runInContext(`state.${name} = __testValue`, context);
  delete context.__testValue;
}

function stateJson(context, expression) {
  return JSON.parse(vm.runInContext(`JSON.stringify(${expression})`, context));
}

test('iterative confirmation uses the same one-based submission frame as normal confirmation', () => {
  const { context } = createMainHarness();
  setState(context, 'currentQueryId', 'query-p1-1-kis');
  setState(context, 'task', 'kis');
  setState(context, 'iterRunning', true);
  setState(context, 'iterMatchedList', [{
    video_id: 'L01_V001', representative_frames: [0], start_frame: 0, rank: 1,
  }]);

  vm.runInContext('iterFinish()', context);

  assert.deepEqual(stateJson(context, 'state.selections'), [{
    video_id: 'L01_V001',
    frames: [1],
    answer: '',
    queryId: 'query-p1-1-kis',
    task: 'kis',
    rank: 1,
  }]);
});

test('a rejected query pack keeps the previously installed manifest', async () => {
  const { context, storage } = createMainHarness();
  const previous = [{
    query_id: 'query-p1-1-kis', task: 'kis', text: 'valid',
    source_name: 'query-p1-1-kis.txt', n_events: null, events_confirmed: true,
  }];
  setState(context, 'manifest', previous);
  storage.set('aic_manifest', JSON.stringify(previous));
  context.fetch = async () => ({
    ok: false,
    async json() {
      return {
        detail: {
          ok: false,
          manifest: [{
            query_id: 'query-p1-2-qa', task: 'qa', text: 'partial',
            source_name: 'query-p1-2-qa.txt', n_events: null, events_confirmed: true,
          }],
          errors: [{ code: 'invalid_task_suffix', message: 'bad pack', query_id: null, row: null }],
          warnings: [],
        },
      };
    },
  });
  context.__uploadEvent = {
    target: {
      files: [{ name: 'query-p1-2-qa.txt', async text() { return 'partial'; } }],
      value: 'chosen',
    },
  };

  await vm.runInContext('handleQueryFileUpload(__uploadEvent)', context);

  assert.deepEqual(stateJson(context, 'state.manifest'), previous);
  assert.equal(storage.get('aic_manifest'), JSON.stringify(previous));
  assert.equal(context.__uploadEvent.target.value, '');
});

test('switching manifest queries clears stale candidate drafts but preserves saved rows', () => {
  const { context, document } = createMainHarness();
  const selections = [
    { queryId: 'query-p1-1-kis', task: 'kis', video_id: 'L01_V001', frames: [8], answer: '' },
    { queryId: 'query-p1-2-qa', task: 'qa', video_id: 'L01_V002', frames: [20], answer: 'Đáp án đã lưu' },
  ];
  setState(context, 'manifest', [
    { query_id: 'query-p1-1-kis', task: 'kis', text: 'old', n_events: null, events_confirmed: true },
    { query_id: 'query-p1-2-qa', task: 'qa', text: 'new', n_events: null, events_confirmed: true },
  ]);
  setState(context, 'currentQueryId', 'query-p1-1-kis');
  setState(context, 'selections', selections);
  setState(context, 'candidates', [{
    video_id: 'L01_V002', representative_frames: [19], start_frame: 19,
    end_frame: 20, rank: 1, scores: {},
  }]);
  setState(context, 'selected', 0);
  setState(context, 'iterRunning', true);
  document.getElementById('frame-input').value = '999';
  document.getElementById('answer-input').value = 'stale draft';
  document.getElementById('btn-confirm-selection').disabled = false;
  document.getElementById('detail-rank-badge').textContent = '#1';

  vm.runInContext("selectManifestQuery('query-p1-2-qa')", context);

  assert.deepEqual(stateJson(context, 'state.candidates'), []);
  assert.equal(vm.runInContext('state.selected', context), null);
  assert.equal(vm.runInContext('state.iterRunning', context), false);
  assert.deepEqual(stateJson(context, 'state.selections'), selections);
  assert.equal(document.getElementById('frame-input').value, '');
  assert.equal(document.getElementById('answer-input').value, '');
  assert.equal(document.getElementById('btn-confirm-selection').disabled, true);
  assert.equal(document.getElementById('detail-rank-badge').textContent, '—');
  assert.equal(document.getElementById('results-count').textContent, '0 candidates');
});

test('candidates in the same video keep independent draft frames', () => {
  const { context, document } = createMainHarness();
  setState(context, 'currentQueryId', 'query-p1-1-kis');
  setState(context, 'task', 'kis');
  setState(context, 'currentFps', 60);
  setState(context, 'selections', [{
    queryId: 'query-p1-1-kis', task: 'kis', video_id: 'L01_V001',
    frames: [999], answer: '',
  }]);
  setState(context, 'candidates', [
    { video_id: 'L01_V001', representative_frames: [9], start_frame: 9, rank: 1, scores: {} },
    { video_id: 'L01_V001', representative_frames: [19], start_frame: 19, rank: 2, scores: {} },
  ]);

  vm.runInContext('selectCandidate(1)', context);

  assert.equal(document.getElementById('frame-input').value, 20);
  assert.equal(vm.runInContext('state.currentFps', context), null);
});

test('video playback reports its current frame without overwriting the answer draft', () => {
  const { context, document } = createMainHarness();
  document.listeners.DOMContentLoaded[0]();
  setState(context, 'task', 'kis');
  setState(context, 'currentFps', 25);
  const video = document.getElementById('preview-vid');
  video.style.display = 'block';
  video.currentTime = 4;
  document.getElementById('frame-input').value = '777';

  video.dispatchEvent({ type: 'timeupdate' });

  assert.equal(document.getElementById('frame-input').value, '777');
  assert.equal(document.getElementById('video-current-frame').textContent, '101');
});

test('taking the current video frame updates only the selected candidate draft', () => {
  const { context, document } = createMainHarness();
  setState(context, 'task', 'kis');
  setState(context, 'currentFps', 25);
  setState(context, 'selected', 1);
  setState(context, 'candidates', [
    { video_id: 'L01_V001', representative_frames: [9], start_frame: 9, rank: 1 },
    { video_id: 'L01_V001', representative_frames: [19], start_frame: 19, rank: 2 },
  ]);
  const video = document.getElementById('preview-vid');
  video.style.display = 'block';
  video.currentTime = 4;
  document.getElementById('frame-input').value = '777';

  assert.equal(vm.runInContext('typeof grabCurrentFrame', context), 'function');
  vm.runInContext('grabCurrentFrame()', context);

  assert.equal(document.getElementById('frame-input').value, 101);
  assert.deepEqual(stateJson(context, 'state.candidateDraftFrames'), {
    L01_V001__19: 101,
  });
});

test('selecting a candidate loads the inline mini video without locking the page', () => {
  const { context, document } = createMainHarness();
  setState(context, 'task', 'kis');
  setState(context, 'candidates', [{
    video_id: 'L01_V001', representative_frames: [9], start_frame: 9, rank: 1, scores: {},
  }]);

  vm.runInContext('selectCandidate(0)', context);

  assert.equal(document.body.classList.contains('modal-open'), false);
  assert.equal(document.getElementById('preview-vid').src, '/api/video/L01_V001');
  assert.equal(document.getElementById('detail-rank-badge').textContent, '#1');
});

test('validation report renders stable code and query-row location', () => {
  const { context, document } = createMainHarness();
  context.__report = {
    errors: [{
      code: 'invalid_frame', query_id: 'query-p1-1-kis', row: 2,
      message: 'Frames must be positive integers',
    }],
    warnings: [],
  };

  vm.runInContext('setValidationReport(__report)', context);

  const text = document.getElementById('validation-report').textContent;
  assert.match(text, /\[invalid_frame\]/);
  assert.match(text, /query query-p1-1-kis · row 2/);
});

test('uploaded query IDs render as inert text with programmatic action listeners', () => {
  const { context, document } = createMainHarness();
  const hostile = "query-');globalThis.pwned=true;('<img src=x onerror=globalThis.pwned=true>-kis";
  setState(context, 'manifest', [{
    query_id: hostile, task: 'kis', text: 'hostile', n_events: null, events_confirmed: true,
  }]);
  setState(context, 'selections', [{
    queryId: hostile, task: 'kis', video_id: 'L01_V001', frames: [1], answer: '',
  }]);

  vm.runInContext('renderManifestList(); renderExportTable()', context);

  assert.equal(document.innerHTMLAssignments.some((html) => html.includes(hostile)), false);
  assert.match(document.getElementById('query-manifest-list').textContent, new RegExp(hostile.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
  assert.match(document.getElementById('export-tbody').textContent, new RegExp(hostile.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
  const actionButtons = document.getElementById('export-tbody').querySelectorAll('button');
  assert.equal(actionButtons.length, 3);
  assert.equal(actionButtons.every((button) => button.getAttribute('onclick') === null), true);
  assert.equal(context.pwned, undefined);
});

test('manifest navigation shows readiness and refreshes after selection changes', () => {
  const { context, document } = createMainHarness();
  setState(context, 'manifest', [{
    query_id: 'query-p1-3-trake', task: 'trake', text: 'events',
    n_events: 2, events_confirmed: false,
  }]);
  setState(context, 'selections', []);

  vm.runInContext('renderManifestList()', context);
  assert.match(
    document.getElementById('query-manifest-list').textContent,
    /Chưa có dòng · Chưa xác nhận events/,
  );

  setState(context, 'selections', [{
    queryId: 'query-p1-3-trake', task: 'trake', video_id: 'L01_V001',
    frames: [10, 20], answer: '',
  }]);
  vm.runInContext("state.manifest[0].events_confirmed = true; renderManifestList()", context);
  assert.match(document.getElementById('query-manifest-list').textContent, /Ready/);
});

test('an invalid TRAKE event-count edit revokes readiness immediately', () => {
  const { context, document } = createMainHarness();
  setState(context, 'manifest', [{
    query_id: 'query-p1-3-trake', task: 'trake', text: 'events',
    n_events: 2, events_confirmed: true,
  }]);
  setState(context, 'currentQueryId', 'query-p1-3-trake');
  setState(context, 'selections', [{
    queryId: 'query-p1-3-trake', task: 'trake', video_id: 'L01_V001',
    frames: [10, 20], answer: '',
  }]);
  document.getElementById('n-events-input').value = '0';
  document.getElementById('trake-events-confirmed').checked = true;

  vm.runInContext('updateSelectedTrakeState(false)', context);

  assert.equal(vm.runInContext('state.manifest[0].events_confirmed', context), false);
  assert.equal(document.getElementById('trake-events-confirmed').checked, false);
  assert.match(
    document.getElementById('query-manifest-list').textContent,
    /Chưa xác nhận events/,
  );
});

test('Q&A confirmation accepts 100 astral characters like the Python validator', () => {
  const { context, document } = createMainHarness();
  setState(context, 'currentQueryId', 'query-p1-2-qa');
  setState(context, 'task', 'qa');
  setState(context, 'candidates', [{
    video_id: 'L01_V002', representative_frames: [0], start_frame: 0, rank: 1,
  }]);
  setState(context, 'selected', 0);
  document.getElementById('frame-input').value = '1';
  document.getElementById('answer-input').value = '😀'.repeat(100);

  vm.runInContext('confirmSelection()', context);

  const rows = stateJson(context, 'state.selections');
  assert.equal(rows.length, 1);
  assert.equal(Array.from(rows[0].answer).length, 100);
});

// ---------------------------------------------------------------------------
// Dải chip trạng thái từng thành phần
//
// Nửa nhìn thấy được của việc tách khởi tạo: operator phải biết thành phần nào
// đang nạp, thành phần nào hỏng — thay vì một chấm xanh chung không nói gì.
// ---------------------------------------------------------------------------

function statusPayload(components, extra = {}) {
  return {
    ok: true,
    retriever: 'clip (177,321 vectors, dim=512)',
    loading: components.some((c) => c.state === 'loading'),
    components,
    retrievers: components.filter((c) => c.kind !== 'translation'),
    ...extra,
  };
}

function stubFetch(context, handler) {
  const calls = [];
  context.fetch = async (url, options) => {
    calls.push({ url, options });
    return { json: async () => handler(url, options) };
  };
  return calls;
}

test('status chips render one entry per component with its own state', async () => {
  const { context, document } = createMainHarness();
  stubFetch(context, () => statusPayload([
    { name: 'translation', kind: 'translation', state: 'ready', detail: 'translation: opus-mt', error: null },
    { name: 'clip', kind: 'retrieval', state: 'loading', detail: 'clip: đang nạp…', error: null },
    { name: 'bm25', kind: 'retrieval', state: 'error', detail: 'bm25: nạp thất bại', error: 'không tìm thấy index' },
    { name: 'siglip', kind: 'retrieval', state: 'disabled', detail: 'siglip: chưa cấu hình', error: null },
  ]));

  await vm.runInContext('checkStatus()', context);

  const chips = document.getElementById('component-status').children;
  assert.deepEqual(
    chips.map((chip) => chip.className),
    [
      'component-chip state-ready',
      'component-chip state-loading',
      'component-chip state-error',
      'component-chip state-disabled',
    ],
  );
  assert.deepEqual(
    chips.map((chip) => chip.textContent),
    ['Dịch', 'CLIP', 'BM25', 'SigLIP'],
  );
});

test('an errored chip explains why and offers a reload', async () => {
  const { context, document } = createMainHarness();
  const calls = stubFetch(context, (url) => {
    if (url.includes('/reload')) {
      return { ok: true, component: { name: 'bm25', state: 'ready', detail: 'bm25: 629,404 docs', error: null } };
    }
    return statusPayload([
      { name: 'bm25', kind: 'retrieval', state: 'error', detail: 'bm25: nạp thất bại', error: 'không tìm thấy index' },
    ]);
  });

  await vm.runInContext('checkStatus()', context);
  const chip = document.getElementById('component-status').children[0];
  assert.match(chip.title, /không tìm thấy index/);

  chip.onclick();
  await new Promise((resolve) => setImmediate(resolve));

  assert.ok(calls.some((call) => call.url === '/api/components/bm25/reload'));
});

test('status polling tightens while a component is still loading', async () => {
  const { context } = createMainHarness();
  const delays = [];
  context.setTimeout = (fn, delay) => { delays.push(delay); return 1; };
  let loading = true;
  stubFetch(context, () => statusPayload([
    { name: 'clip', kind: 'retrieval', state: loading ? 'loading' : 'ready', detail: 'clip', error: null },
  ]));

  await vm.runInContext('scheduleStatusPolling()', context);
  loading = false;
  await vm.runInContext('scheduleStatusPolling()', context);

  assert.deepEqual(delays, [3000, 30000]);
});

test('translating before the model is ready says so instead of hanging silently', async () => {
  const { context, document } = createMainHarness();
  stubFetch(context, (url) => {
    if (url === '/api/status') {
      return statusPayload([
        { name: 'translation', kind: 'translation', state: 'idle', detail: 'translation: chưa nạp', error: null },
      ]);
    }
    return { text_en: 'a man cooking', ok: true };
  });

  await vm.runInContext('checkStatus()', context);
  document.getElementById('query-input').value = 'một người đàn ông nấu ăn';
  await vm.runInContext('doTranslate()', context);

  const toasts = document.getElementById('toast-container').children.map((t) => t.textContent);
  assert.ok(toasts.some((text) => /Đang nạp mô hình dịch/.test(text)));
  assert.equal(document.getElementById('translated-text').value, 'a man cooking');
});

test('currentVideoFrame computes frame index with Math.floor to avoid premature frame jumping', () => {
  const { context, document } = createMainHarness();
  const video = document.getElementById('preview-vid');
  video.style.display = 'block';
  // At fps 25: frame 0 interval is [0.00, 0.04).
  // At currentTime = 0.038s: 0.038 * 25 = 0.95.
  // Math.floor(0.95) + 1 = 1 (actual frame 1).
  // Math.round(0.95) + 1 would have erroneously produced 2.
  video.currentTime = 0.038;
  setState(context, 'currentFps', 25);
  const frame = vm.runInContext('currentVideoFrame()', context);
  assert.equal(frame, 1);
});

test('Q&A answer view escapes HTML tags to prevent XSS injection', () => {
  const { context, document } = createMainHarness();
  setState(context, 'task', 'qa');
  setState(context, 'manifest', [{ query_id: 'qa1', task: 'qa' }]);
  setState(context, 'currentQueryId', 'qa1');
  setState(context, 'selections', [
    { queryId: 'qa1', video_id: 'L01_V001', frames: [100], answer: '<script>alert(1)</script>', task: 'qa' }
  ]);
  vm.runInContext('renderSelectionsList()', context);
  const container = document.getElementById('selections-list');
  assert.ok(!container.innerHTML.includes('<script>'));
  assert.ok(container.innerHTML.includes('&lt;script&gt;'));

  vm.runInContext('showExportReview("qa1")', context);
  const reviewBody = document.getElementById('review-content-body');
  assert.ok(!reviewBody.innerHTML.includes('<script>alert(1)</script>'));
  assert.ok(reviewBody.innerHTML.includes('&lt;script&gt;alert(1)&lt;/script&gt;'));
});

test('deleteQuery removes query from manifest, selections, and cache', () => {
  const { context, document } = createMainHarness();
  setState(context, 'manifest', [
    { query_id: 'q1', task: 'kis', text: 'query 1' },
    { query_id: 'q2', task: 'qa', text: 'query 2' }
  ]);
  setState(context, 'currentQueryId', 'q1');
  setState(context, 'selections', [
    { queryId: 'q1', video_id: 'V1', frames: [1] },
    { queryId: 'q2', video_id: 'V2', frames: [2] }
  ]);
  setState(context, 'queryCache', {
    q1: { candidates: [] },
    q2: { candidates: [] }
  });

  context.confirm = () => true;

  vm.runInContext('deleteQuery("q1")', context);

  const manifest = stateJson(context, 'state.manifest');
  assert.equal(manifest.length, 1);
  assert.equal(manifest[0].query_id, 'q2');

  const selections = stateJson(context, 'state.selections');
  assert.equal(selections.length, 1);
  assert.equal(selections[0].queryId, 'q2');

  const queryCache = stateJson(context, 'state.queryCache');
  assert.equal(queryCache.q1, undefined);
  assert.ok(queryCache.q2 !== undefined);

  assert.equal(vm.runInContext('state.currentQueryId', context), 'q2');
});

test('clearAllCache wipes state, selections, cache, and calls reset', async () => {
  const { context, document } = createMainHarness();
  setState(context, 'manifest', [
    { query_id: 'q1', task: 'kis', text: 'query 1' }
  ]);
  setState(context, 'currentQueryId', 'q1');
  setState(context, 'selections', [
    { queryId: 'q1', video_id: 'V1', frames: [1] }
  ]);
  setState(context, 'queryCache', {
    q1: { candidates: [] }
  });

  context.confirm = () => true;
  vm.runInContext('connectWs()', context);
  context.fetch = async () => ({ ok: true });

  await vm.runInContext('clearAllCache()', context);

  // When WS is active, clearAllCache requests clear cache from server
  const sent = vm.runInContext('ws.sent', context);
  assert.ok(sent.length > 0);
  const parsed = JSON.parse(sent[sent.length - 1]);
  assert.equal(parsed.type, 'request_clear_cache');

  // When server responds with clear_all, client wipes state
  vm.runInContext('ws.onmessage({ data: JSON.stringify({ type: "clear_all" }) })', context);

  const manifest = stateJson(context, 'state.manifest');
  assert.equal(manifest.length, 0);

  const selections = stateJson(context, 'state.selections');
  assert.equal(selections.length, 0);

  const queryCache = stateJson(context, 'state.queryCache');
  assert.deepEqual(queryCache, {});

  assert.equal(vm.runInContext('state.currentQueryId', context), null);
});

test('clear_cache_prompt displays prompt modal and respondClearCache sends approval', () => {
  const { context, document } = createMainHarness();
  vm.runInContext('connectWs()', context);

  // Receive clear_cache_prompt from server
  vm.runInContext('ws.onmessage({ data: JSON.stringify({ type: "clear_cache_prompt", request_id: "req123" }) })', context);

  const promptModal = document.getElementById('modal-clear-prompt');
  const overlay = document.getElementById('clear-cache-overlay');
  if (promptModal && overlay) {
    assert.equal(promptModal.style.display, 'block');
    assert.equal(overlay.style.display, 'flex');
  }

  // Respond with approval
  vm.runInContext('respondClearCache(true)', context);
  const sent = vm.runInContext('ws.sent', context);
  const parsed = JSON.parse(sent[sent.length - 1]);
  assert.equal(parsed.type, 'clear_cache_response');
  assert.equal(parsed.request_id, 'req123');
  assert.equal(parsed.approve, true);
});

test('switching queries preserves search candidates and translated text', () => {
  const { context, document } = createMainHarness();
  setState(context, 'manifest', [
    { query_id: 'q1', task: 'kis', text: '4 nguoi phi cong' },
    { query_id: 'q2', task: 'kis', text: 'may bay truc thang' }
  ]);

  // Select q1
  vm.runInContext('selectManifestQuery("q1")', context);

  // Set translation and candidates on q1
  const transInput = document.getElementById('translated-text');
  transInput.value = 'Four pilots.';
  setState(context, 'candidates', [
    { video_id: 'L21_V024', rank: 1, start_frame: 100, end_frame: 200, representative_frames: [150], best_score: 0.8164 }
  ]);
  vm.runInContext('saveCurrentQueryToCache()', context);

  // Switch to q2
  vm.runInContext('selectManifestQuery("q2")', context);
  assert.equal(vm.runInContext('state.currentQueryId', context), 'q2');
  assert.equal(transInput.value, '');
  assert.equal(stateJson(context, 'state.candidates').length, 0);

  // Switch back to q1
  vm.runInContext('selectManifestQuery("q1")', context);
  assert.equal(vm.runInContext('state.currentQueryId', context), 'q1');
  assert.equal(transInput.value, 'Four pilots.');

  const q1Candidates = stateJson(context, 'state.candidates');
  assert.equal(q1Candidates.length, 1);
  assert.equal(q1Candidates[0].video_id, 'L21_V024');
});

test('resetClientStateAndUI wipes all state when triggered by remote clear_all', () => {
  const { context, document } = createMainHarness();
  setState(context, 'manifest', [
    { query_id: 'q1', task: 'kis', text: 'query 1' }
  ]);
  setState(context, 'currentQueryId', 'q1');
  setState(context, 'selections', [
    { queryId: 'q1', video_id: 'V1', frames: [1] }
  ]);

  vm.runInContext('resetClientStateAndUI(true)', context);

  const manifest = stateJson(context, 'state.manifest');
  assert.equal(manifest.length, 0);

  const selections = stateJson(context, 'state.selections');
  assert.equal(selections.length, 0);
  assert.equal(vm.runInContext('state.currentQueryId', context), null);
});

test('downloadSingleQueryCsv triggers download of formatted CSV file', () => {
  const { context, document } = createMainHarness();
  setState(context, 'selections', [
    { queryId: 'q1', task: 'kis', video_id: 'L01_V001.mp4', frames: [150], answer: '' },
    { queryId: 'q2', task: 'qa', video_id: 'L01_V002.mp4', frames: [250], answer: 'red car' }
  ]);

  let downloadedFile = null;
  const origAppend = document.body.appendChild.bind(document.body);
  document.body.appendChild = (el) => {
    if (el.download) {
      downloadedFile = el.download;
    }
    return origAppend(el);
  };

  vm.runInContext('downloadSingleQueryCsv("q1")', context);
  assert.equal(downloadedFile, 'q1.csv');

  vm.runInContext('downloadSingleQueryCsv("q2")', context);
  assert.equal(downloadedFile, 'q2.csv');
});



