/* ============================================================
   AIC 2026 — Main UI Logic
   Connects to FastAPI backend at /api/*
   ============================================================ */

'use strict';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  task: 'kis',
  candidates: [],
  selected: null,
  selections: [],
  manifest: [],
  currentQueryId: null,
  validationReport: null,
  gridMode: true,
  candidateDraftFrames: {},
  currentPlaybackFrame: null,

  iterCandidates: [],
  iterCursor: 0,
  iterRound: 0,
  iterMaxRounds: 3,
  iterRunning: false,
  iterVerdict: {},
  iterMatchedList: [],
  iterUnsureList: [],
  iterExcluded: new Set(),

  queryCache: {},

  // Private search tab state
  privateTask: 'kis',
  privateCandidates: [],
  privateSelected: null,
  privateSelections: [],
  privateQueryCache: {},
  privateCandidateDraftFrames: {},
  currentPrivateQueryId: null,
  privateCurrentPlaybackFrame: null,
  privateCurrentFps: null,

  // Manual entry tab state
  videos: [],
  manualCurrentFps: null,
  manualCurrentPlaybackFrame: null,
};

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

function $(id) { return document.getElementById(id); }
const submissionHelpers = window.AICSubmissionHelpers;

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  $('toast-container').appendChild(el);
  requestAnimationFrame(() => el.classList.add('show'));
  setTimeout(() => {
    el.classList.remove('show');
    setTimeout(() => el.remove(), 300);
  }, 3200);
}

function setLoading(btnId, loading) {
  const btn = $(btnId);
  if (!btn) return;
  btn.disabled = loading;
  if (loading) {
    btn.dataset.origText = btn.innerHTML;
    btn.innerHTML = '<span class="spinner-ring" style="width:14px;height:14px;border-width:2px;display:inline-block"></span>';
  } else {
    if (btn.dataset.origText) btn.innerHTML = btn.dataset.origText;
  }
}

function fmtScore(v) {
  if (v === undefined || v === null) return '—';
  return Number(v).toFixed(4);
}

function scoreClass(v) {
  if (v >= 0.6) return 'score-high';
  if (v >= 0.35) return 'score-mid';
  return 'score-low';
}

function keyframeUrl(videoId, frameIdx) {
  return `/api/keyframe/${encodeURIComponent(videoId)}/${frameIdx}`;
}

function candidateKey(c) {
  return `${c.video_id}__${c.representative_frames[0] ?? c.start_frame}`;
}

function currentVideoFrame() {
  const video = $('preview-vid');
  if (!video || video.style.display === 'none') return null;
  const fps = state.currentFps || 25;
  return Math.floor(video.currentTime * fps) + 1;
}

function updatePlaybackFrame() {
  const frame = currentVideoFrame();
  if (!Number.isInteger(frame)) return;
  state.currentPlaybackFrame = frame;
  const indicator = $('video-current-frame');
  if (indicator) indicator.textContent = String(frame);
}

function grabCurrentFrame(e) {
  if (e) e.preventDefault();
  if (state.selected === null) return;
  const candidate = state.candidates[state.selected];
  if (!candidate) return;
  const frame = currentVideoFrame()
    ?? state.currentPlaybackFrame
    ?? submissionHelpers.candidateToSubmissionFrame(candidate);
  if (!Number.isInteger(frame)) return;
  $('frame-input').value = frame;
  state.candidateDraftFrames[candidateKey(candidate)] = frame;
  const button = $('btn-grab-frame');
  if (button) {
    button.classList.add('captured');
    setTimeout(() => button.classList.remove('captured'), 700);
  }
}

function currentManifestItem() {
  return state.manifest.find((item) => item.query_id === state.currentQueryId) || null;
}

function currentQueryId() {
  return state.currentQueryId || $('query-id-input').value.trim() || 'q1';
}

function currentTrakeEvents() {
  const item = currentManifestItem();
  return item && item.task === 'trake' ? item.n_events : parseInt($('n-events-input').value, 10);
}

function saveManifest() {
  localStorage.setItem('aic_manifest', JSON.stringify(state.manifest));
  sendWsUpdate();
}

function loadManifest() {
  const data = localStorage.getItem('aic_manifest');
  if (!data) return;
  try {
    state.manifest = JSON.parse(data);
  } catch (e) {
    console.error('Failed to load manifest:', e);
    state.manifest = [];
  }
}

function clearQueryWorkspace() {
  Object.assign(state, submissionHelpers.clearQueryWorkspaceState(state));

  const grid = $('candidates-grid');
  if (grid) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    const message = document.createElement('p');
    message.textContent = 'Chưa tìm kiếm query này';
    empty.appendChild(message);
    grid.replaceChildren(empty);
  }
  const resultCount = $('results-count');
  if (resultCount) resultCount.textContent = '0 candidates';

  const image = $('preview-img');
  if (image) {
    image.onload = null;
    image.onerror = null;
    image.removeAttribute('src');
    image.style.display = 'none';
  }
  const video = $('preview-vid');
  if (video) {
    video.onloadeddata = null;
    video.onerror = null;
    if (typeof video.pause === 'function') video.pause();
    video.removeAttribute('src');
    if (typeof video.load === 'function') video.load();
    video.style.display = 'none';
  }
  const placeholder = $('preview-placeholder');
  if (placeholder) placeholder.style.display = 'flex';
  const rankBadge = $('detail-rank-badge');
  if (rankBadge) rankBadge.textContent = '—';
  const scoresSection = $('detail-scores-section');
  if (scoresSection) scoresSection.style.display = 'none';
  const scoresBody = $('detail-scores-body');
  if (scoresBody) scoresBody.replaceChildren();
  const evidenceSection = $('detail-evidence-section');
  if (evidenceSection) evidenceSection.style.display = 'none';
  const frameInput = $('frame-input');
  if (frameInput) frameInput.value = '';
  const playbackFrame = $('video-current-frame');
  if (playbackFrame) playbackFrame.textContent = '—';
  const answerInput = $('answer-input');
  if (answerInput) answerInput.value = '';
  const trakeContainer = $('frame-picker-trake-container');
  if (trakeContainer) {
    trakeContainer.replaceChildren();
    trakeContainer.style.display = 'none';
  }
  const confirmButton = $('btn-confirm-selection');
  if (confirmButton) confirmButton.disabled = true;

  const iterStatus = $('iter-status-badge');
  if (iterStatus) iterStatus.textContent = 'Chưa bắt đầu';
  const iterFinishButton = $('btn-iter-finish');
  if (iterFinishButton) iterFinishButton.disabled = true;
  ['btn-iter-prev', 'btn-iter-next', 'btn-iter-skip'].forEach((id) => {
    const button = $(id);
    if (button) button.disabled = true;
  });
}

function renderManifestList() {
  const list = $('query-manifest-list');
  if (!list) return;
  list.replaceChildren();
  state.manifest.forEach((item) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `query-manifest-item${item.query_id === state.currentQueryId ? ' active' : ''}`;
    const queryId = document.createElement('span');
    queryId.className = 'manifest-query-id';
    queryId.textContent = item.query_id;
    const task = document.createElement('span');
    task.className = 'manifest-task';
    task.textContent = item.task.toUpperCase();

    const deleteBtn = document.createElement('span');
    deleteBtn.className = 'manifest-delete-btn';
    deleteBtn.title = `Xóa query ${item.query_id}`;
    deleteBtn.textContent = '×';
    deleteBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      e.preventDefault();
      deleteQuery(item.query_id);
    });

    const readiness = submissionHelpers.manifestQueryReadiness(item, state.selections);
    const status = document.createElement('span');
    status.className = `manifest-readiness ${readiness.ready ? 'ready' : 'not-ready'}`;
    status.textContent = readiness.ready ? 'Ready' : readiness.label;
    button.dataset.readiness = readiness.codes.join(',') || 'ready';
    button.append(queryId, deleteBtn, task, status);
    button.addEventListener('click', () => selectManifestQuery(item.query_id));
    list.appendChild(button);
  });
}

function sendWsDeleteQuery(queryId) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'delete_query',
      query_id: queryId
    }));
  }
}

function deleteQuery(queryId) {
  if (!queryId) return;
  if (typeof confirm === 'function' && !confirm(`Bạn có chắc chắn muốn xóa query "${queryId}" và toàn bộ kết quả đã chọn?`)) {
    return;
  }

  // 1. Xóa khỏi manifest
  state.manifest = (state.manifest || []).filter(item => item.query_id !== queryId);
  saveManifest();

  // 2. Xóa các lựa chọn liên quan
  state.selections = (state.selections || []).filter(s => (s.queryId || s.query_id) !== queryId);
  saveSelections();

  // 3. Xóa cache của query
  if (state.queryCache) {
    delete state.queryCache[queryId];
    saveQueryCache();
  }

  // 4. Xóa khỏi private selections & cache nếu có
  if (state.privateSelections) {
    state.privateSelections = state.privateSelections.filter(s => (s.queryId || s.query_id) !== queryId);
    savePrivateSelections();
  }
  if (state.privateQueryCache) {
    delete state.privateQueryCache[queryId];
    savePrivateQueryCache();
  }

  // 5. Nếu đang ở query bị xóa, chuyển sang query khác hoặc làm trống workspace
  if (state.currentQueryId === queryId) {
    state.currentQueryId = null;
    state.selected = null;
    state.candidates = [];
    clearQueryWorkspace();
    const qi = $('query-input'); if (qi) qi.value = '';
    const tt = $('translated-text'); if (tt) tt.value = '';
    const qii = $('query-id-input'); if (qii) qii.value = '';
    if (state.manifest.length > 0) {
      selectManifestQuery(state.manifest[0].query_id);
    }
  }

  if (state.currentPrivateQueryId === queryId) {
    state.currentPrivateQueryId = null;
    state.privateSelected = null;
    state.privateCandidates = [];
    clearPrivateQueryWorkspace();
    const pqi = $('private-query-input'); if (pqi) pqi.value = '';
    const ptt = $('private-translated-text'); if (ptt) ptt.value = '';
    const pqii = $('private-query-id-input'); if (pqii) pqii.value = '';
    if (state.manifest.length > 0) {
      selectPrivateManifestQuery(state.manifest[0].query_id);
    }
  }

  // 6. Đồng bộ qua WebSocket
  sendWsDeleteQuery(queryId);

  // 7. Cập nhật giao diện
  renderManifestList();
  renderSelectionsList();
  renderPrivateManifestList();
  renderPrivateSelectionsList();
  if ($('view-export') && $('view-export').classList.contains('active')) {
    renderExportTable();
  }

  toast(`Đã xóa query ${queryId}`, 'success');
}

function deleteCurrentQuery() {
  const qid = (state.currentQueryId || $('query-id-input')?.value || '').trim();
  if (!qid) {
    toast('Chưa chọn query nào để xóa', 'warning');
    return;
  }
  deleteQuery(qid);
}

function deleteCurrentPrivateQuery() {
  const qid = (state.currentPrivateQueryId || $('private-query-id-input')?.value || '').trim();
  if (!qid) {
    toast('Chưa chọn query nào để xóa', 'warning');
    return;
  }
  deleteQuery(qid);
}

function resetClientStateAndUI(isFromRemote = false) {
  localStorage.removeItem('aic_selections');
  localStorage.removeItem('aic_manifest');
  localStorage.removeItem('aic_query_cache');
  localStorage.removeItem('aic_private_selections');
  localStorage.removeItem('aic_private_query_cache');

  // Clear in-memory state
  state.manifest = [];
  state.selections = [];
  state.queryCache = {};
  state.currentQueryId = null;
  state.selected = null;
  state.candidates = [];
  state.candidateDraftFrames = {};
  
  state.privateSelections = [];
  state.privateQueryCache = {};
  state.currentPrivateQueryId = null;
  state.privateSelected = null;
  state.privateCandidates = [];
  state.privateCandidateDraftFrames = {};

  // Re-render UI views
  renderManifestList();
  renderSelectionsList();
  renderExportTable();
  clearQueryWorkspace();

  renderPrivateManifestList();
  renderPrivateSelectionsList();
  clearPrivateQueryWorkspace();

  // Reset inputs
  const qi = $('query-input'); if (qi) qi.value = '';
  const tt = $('translated-text'); if (tt) tt.value = '';
  const qii = $('query-id-input'); if (qii) qii.value = '';

  const pqi = $('private-query-input'); if (pqi) pqi.value = '';
  const ptt = $('private-translated-text'); if (ptt) ptt.value = '';
  const pqii = $('private-query-id-input'); if (pqii) pqii.value = '';

  if (isFromRemote) {
    toast('Đã đồng bộ xóa toàn bộ cache từ người dùng khác', 'info');
  }
}

let currentClearRequestId = null;

function showClearCacheWaiting(count) {
  const overlay = $('clear-cache-overlay');
  const waiting = $('modal-clear-waiting');
  const prompt = $('modal-clear-prompt');
  if (overlay && waiting) {
    if (prompt) prompt.style.display = 'none';
    const msg = $('modal-waiting-msg');
    if (msg) {
      msg.textContent = `Hệ thống đang có ${count} máy khác đang hoạt động. Cần ít nhất 1 thành viên chấp thuận để thực hiện xóa cache hệ thống...`;
    }
    waiting.style.display = 'block';
    overlay.style.display = 'flex';
  }
}

function showClearCachePrompt(requestId) {
  currentClearRequestId = requestId;
  const overlay = $('clear-cache-overlay');
  const waiting = $('modal-clear-waiting');
  const prompt = $('modal-clear-prompt');
  if (overlay && prompt) {
    if (waiting) waiting.style.display = 'none';
    prompt.style.display = 'block';
    overlay.style.display = 'flex';
  }
}

function hideClearCacheModals() {
  const overlay = $('clear-cache-overlay');
  const waiting = $('modal-clear-waiting');
  const prompt = $('modal-clear-prompt');
  if (overlay) overlay.style.display = 'none';
  if (waiting) waiting.style.display = 'none';
  if (prompt) prompt.style.display = 'none';
}

function cancelClearCacheRequest() {
  hideClearCacheModals();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'clear_cache_cancel',
      request_id: currentClearRequestId,
    }));
  }
  toast('Đã hủy yêu cầu xóa cache', 'info');
}

function respondClearCache(approve) {
  hideClearCacheModals();
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'clear_cache_response',
      request_id: currentClearRequestId,
      approve: Boolean(approve),
    }));
  }
  if (approve) {
    toast('Bạn đã đồng ý xóa cache', 'info');
  } else {
    toast('Bạn đã từ chối yêu cầu xóa cache', 'info');
  }
}

async function clearAllCache() {
  if (typeof confirm === 'function' && !confirm('Bạn có chắc chắn muốn xóa toàn bộ cache? Tất cả câu hỏi đã tải lên, các câu trả lời đã lưu, và lịch sử tìm kiếm sẽ bị xóa sạch.')) {
    return;
  }

  // Nếu kết nối WebSocket đang mở, gửi request_clear_cache:
  // Server tự kiểm tra: nếu 1 máy -> xóa ngay; nếu >1 máy -> hỏi các máy còn lại
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'request_clear_cache' }));
    return;
  }

  // Fallback ngoại tuyến / kiểm thử đơn lập không WebSocket
  resetClientStateAndUI(false);
  if (typeof fetch === 'function') {
    try {
      await fetch('/api/clear_state', { method: 'POST' });
    } catch (err) {
      console.warn('POST /api/clear_state error:', err);
    }
  }
  toast('Đã xóa sạch cache hệ thống!', 'success');
}

function saveQueryCache() {
  const serialized = {};
  for (const [qid, cache] of Object.entries(state.queryCache)) {
    if (!cache) continue;
    serialized[qid] = Object.assign({}, cache, {
      iterExcluded: Array.from(cache.iterExcluded || []),
    });
  }
  try {
    localStorage.setItem('aic_query_cache', JSON.stringify(serialized));
  } catch (e) {
    console.error('Failed to write local storage:', e);
  }
  sendWsUpdate();
}

function loadQueryCache() {
  const data = localStorage.getItem('aic_query_cache');
  if (data) {
    try {
      const parsed = JSON.parse(data);
      const deserialized = {};
      for (const [qid, cache] of Object.entries(parsed)) {
        if (!cache) continue;
        deserialized[qid] = Object.assign({}, cache, {
          iterExcluded: new Set(cache.iterExcluded || []),
        });
      }
      state.queryCache = deserialized;
    } catch (e) {
      console.error('Failed to parse query cache:', e);
      state.queryCache = {};
    }
  } else {
    state.queryCache = {};
  }
}

function saveCurrentQueryToCache() {
  const queryId = state.currentQueryId;
  if (!queryId) return;

  state.queryCache[queryId] = {
    text_vi: $('query-input')?.value || '',
    translatedText: $('translated-text')?.value || '',
    candidates: state.candidates || [],
    selected: state.selected,
    candidateDraftFrames: Object.assign({}, state.candidateDraftFrames),
    currentFps: state.currentFps,
    currentPlaybackFrame: state.currentPlaybackFrame,
    iterCandidates: state.iterCandidates || [],
    iterCursor: state.iterCursor,
    iterRound: state.iterRound,
    iterRunning: state.iterRunning,
    iterVerdict: Object.assign({}, state.iterVerdict),
    iterMatchedList: state.iterMatchedList || [],
    iterUnsureList: state.iterUnsureList || [],
    iterExcluded: new Set(state.iterExcluded),
  };
  saveQueryCache();
}

function loadQueryFromCache(queryId, form) {
  const cached = state.queryCache[queryId];
  if (cached) {
    if (cached.text_vi && $('query-input')) {
      $('query-input').value = cached.text_vi;
    }
    $('translated-text').value = cached.translatedText || '';
    
    state.candidates = cached.candidates || [];
    state.selected = (cached.selected !== undefined && cached.selected !== null) ? cached.selected : null;
    state.candidateDraftFrames = Object.assign({}, cached.candidateDraftFrames);
    state.currentFps = cached.currentFps;
    state.currentPlaybackFrame = cached.currentPlaybackFrame;
    
    state.iterCandidates = cached.iterCandidates || [];
    state.iterCursor = cached.iterCursor;
    state.iterRound = cached.iterRound;
    state.iterRunning = cached.iterRunning;
    state.iterVerdict = Object.assign({}, cached.iterVerdict);
    state.iterMatchedList = cached.iterMatchedList || [];
    state.iterUnsureList = cached.iterUnsureList || [];
    state.iterExcluded = new Set(cached.iterExcluded || []);

    if (state.candidates && state.candidates.length) {
      renderCandidates();
      $('results-count').textContent = `${state.candidates.length} candidates`;
      if (state.selected !== null && state.selected >= 0 && state.selected < state.candidates.length) {
        selectCandidate(state.selected);
      }
    } else {
      clearQueryWorkspace();
    }
  } else {
    $('translated-text').value = (form && form.translatedText) || '';
    clearQueryWorkspace();
  }
}

function selectManifestQuery(queryId) {
  saveCurrentQueryToCache();

  const item = state.manifest.find((entry) => entry.query_id === queryId);
  if (!item) return;

  // Clear current candidates/selected before selectTask to prevent loading mismatch
  state.selected = null;
  state.candidates = [];

  const form = submissionHelpers.manifestQueryFormState(item);
  state.currentQueryId = form.queryId;
  $('query-id-input').value = form.queryId;
  $('export-query-id').value = form.queryId;
  $('query-input').value = form.text;
  $('n-events-input').value = form.nEvents;
  $('trake-events-confirmed').checked = form.eventsConfirmed;
  selectTask(form.task);

  loadQueryFromCache(form.queryId, form);

  renderManifestList();
  renderSelectionsList();
}

function updateSelectedTrakeState(confirmEvents) {
  const item = currentManifestItem();
  if (!item || item.task !== 'trake') return;
  const nEvents = parseInt($('n-events-input').value, 10);
  if (!Number.isInteger(nEvents) || nEvents < 1) {
    state.manifest = submissionHelpers.updateTrakeState(
      state.manifest,
      item.query_id,
      item.n_events,
      false,
    );
    $('trake-events-confirmed').checked = false;
    saveManifest();
    renderManifestList();
    toast('Số events TRAKE phải lớn hơn 0', 'warning');
    return;
  }
  state.manifest = confirmEvents
    ? submissionHelpers.updateTrakeState(state.manifest, item.query_id, nEvents, true)
    : submissionHelpers.changeTrakeEventCount(state.manifest, item.query_id, nEvents);
  $('trake-events-confirmed').checked = Boolean(confirmEvents);
  saveManifest();
  renderManifestList();
}

function setValidationReport(report) {
  state.validationReport = report && (report.errors?.length || report.warnings?.length) ? report : null;
  renderValidationReport();
}

function renderValidationReport() {
  const container = $('validation-report');
  if (!container) return;
  container.innerHTML = '';
  if (!state.validationReport) return;
  [['errors', 'Lỗi cần sửa'], ['warnings', 'Cảnh báo']].forEach(([kind, title]) => {
    const groups = submissionHelpers.groupValidationIssues(state.validationReport[kind]);
    Object.entries(groups).forEach(([queryId, messages]) => {
      const group = document.createElement('div');
      group.className = `validation-report-group ${kind}`;
      const heading = document.createElement('strong');
      heading.textContent = `${title}: ${queryId}`;
      const list = document.createElement('ul');
      messages.forEach((issue) => {
        const line = document.createElement('li');
        line.textContent = submissionHelpers.formatValidationIssue(issue);
        list.appendChild(line);
      });
      group.append(heading, list);
      container.appendChild(group);
    });
  });
}

// ---------------------------------------------------------------------------
// Status check
// ---------------------------------------------------------------------------

const COMPONENT_LABELS = {
  translation: 'Dịch',
  clip: 'CLIP',
  siglip: 'SigLIP',
  bm25: 'BM25',
  keyframe_map: 'Keyframe',
  dummy: 'Demo',
};

const COMPONENT_STATE_TEXT = {
  idle: 'chưa nạp',
  loading: 'đang nạp…',
  ready: 'sẵn sàng',
  error: 'lỗi',
  disabled: 'tắt',
};

// Trạng thái nạp gần nhất, để doTranslate() biết có nên báo "đang nạp model" không.
state.componentStates = state.componentStates || {};

function renderComponentChips(components) {
  const container = $('component-status');
  if (!container) return;
  container.innerHTML = '';
  (components || []).forEach((c) => {
    const chip = document.createElement('span');
    chip.className = `component-chip state-${c.state}`;
    chip.title = c.error ? `${c.detail} — ${c.error}` : c.detail;
    const dot = document.createElement('span');
    dot.className = 'chip-dot';
    const label = document.createElement('span');
    label.textContent = COMPONENT_LABELS[c.name] || c.name;
    chip.append(dot, label);
    if (c.state === 'error') {
      // Lỗi dính lại cho tới khi reload — cho operator một đường thoát tại chỗ.
      chip.title += '\nBấm để thử nạp lại';
      chip.onclick = () => reloadComponent(c.name);
    }
    container.appendChild(chip);
  });
}

async function reloadComponent(name) {
  toast(`Đang nạp lại ${COMPONENT_LABELS[name] || name}…`, 'info');
  try {
    const res = await fetch(`/api/components/${encodeURIComponent(name)}/reload`, {
      method: 'POST',
    });
    const data = await res.json();
    const slot = data.component || {};
    if (slot.state === 'ready') toast(`${COMPONENT_LABELS[name] || name} đã sẵn sàng`, 'success');
    else toast(`${COMPONENT_LABELS[name] || name}: ${slot.error || 'vẫn lỗi'}`, 'error');
  } catch {
    toast('Lỗi kết nối', 'error');
  }
  checkStatus();
}

async function checkStatus() {
  let anyLoading = false;
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.ok) {
      $('status-text').textContent = data.retriever;
      const colour = data.loading ? 'var(--amber, #f59e0b)' : 'var(--green)';
      $('status-dot').style.background = colour;
      $('status-dot').style.boxShadow = `0 0 6px ${colour}`;
      $('stat-keyframes').textContent = data.retriever === 'dummy' ? 'demo' : '—';
      renderComponentChips(data.components);
      state.componentStates = {};
      (data.components || []).forEach((c) => { state.componentStates[c.name] = c.state; });
      anyLoading = Boolean(data.loading);
    }
  } catch {
    $('status-text').textContent = 'Offline';
    $('status-dot').style.background = 'var(--red)';
    $('status-dot').style.boxShadow = '0 0 6px var(--red)';
  }
  return anyLoading;
}

// Poll dày trong lúc còn thành phần đang nạp, thưa lại khi đã ổn định — để
// operator thấy chip đổi màu theo thời gian thực mà không phải F5.
async function scheduleStatusPolling() {
  const anyLoading = await checkStatus();
  setTimeout(scheduleStatusPolling, anyLoading ? 3000 : 30000);
}

// ---------------------------------------------------------------------------
// View switching
// ---------------------------------------------------------------------------

function switchView(view) {
  if (view === 'iterative' && !submissionHelpers.canUseIterative(state.task)) {
    toast('Iterative chỉ dùng cho KIS', 'warning');
    view = 'search';
  }
  document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));
  const el = $(`view-${view}`);
  if (el) el.classList.add('active');
  const tab = $(`tab-${view}`);
  if (tab) tab.classList.add('active');
  if (view === 'export') renderExportTable();
  if (view === 'private-search') {
    renderPrivateManifestList();
    renderPrivateSelectionsList();
  }
  if (view === 'manual-entry') {
    populateManualQuerySelect();
    renderManualSelections();
  }
}

// ---------------------------------------------------------------------------
// Task selection
// ---------------------------------------------------------------------------

function selectTask(task) {
  const current = currentManifestItem();
  if (current && task !== current.task) {
    toast(`Query pack này là ${current.task.toUpperCase()}`, 'warning');
    task = current.task;
  }
  state.task = task;
  ['kis', 'qa', 'trake'].forEach(t => {
    const pill = $(`pill-${t}`);
    if (pill) pill.classList.toggle('active', t === task);
  });
  const nEvents = $('n-events-section');
  if (nEvents) nEvents.style.display = task === 'trake' ? '' : 'none';
  const answerSec = $('answer-section');
  if (answerSec) answerSec.style.display = task === 'qa' ? '' : 'none';
  const badge = $('results-task-badge');
  if (badge) badge.textContent = task.toUpperCase();
  
  if (state.selected !== null) {
    selectCandidate(state.selected);
  } else {
    const singleFrame = $('frame-picker-single-row');
    if (singleFrame) singleFrame.style.display = task === 'trake' ? 'none' : 'flex';
    const trakeFrames = $('frame-picker-trake-container');
    if (trakeFrames) trakeFrames.style.display = 'none';
  }
  renderSelectionsList();
}

// ---------------------------------------------------------------------------
// Translate
// ---------------------------------------------------------------------------

async function doTranslate() {
  const text_vi = $('query-input').value.trim();
  if (!text_vi) { toast('Nhập câu hỏi tiếng Việt trước', 'warning'); return; }
  if (state.componentStates.translation !== 'ready') {
    // Model dịch nạp riêng, không phải chờ CLIP/BM25 — nhưng lần đầu vẫn mất
    // vài chục giây, nói ra để nút không có vẻ như bị treo.
    toast('Đang nạp mô hình dịch (lần đầu ~30s)…', 'info');
  }
  setLoading('btn-translate', true);
  try {
    const res = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text_vi }),
    });
    const data = await res.json();
    $('translated-text').value = data.text_en || '';
    saveCurrentQueryToCache();
    if (!data.ok) toast(`Dịch thất bại: ${data.error || ''}`, 'warning');
    else toast('Đã dịch thành công', 'success');
  } catch (e) {
    toast('Lỗi kết nối', 'error');
  } finally {
    setLoading('btn-translate', false);
  }
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

async function doSearch() {
  const text_vi = $('query-input').value.trim();
  if (!text_vi) { toast('Nhập câu hỏi', 'warning'); return; }
  const text_en = $('translated-text').value.trim();
  const query_id = currentQueryId();
  const k = parseInt($('topk-slider').value, 10);
  const n_events = currentTrakeEvents() || 1;

  const isManual = !state.manifest.some((item) => item.query_id === query_id);
  if (isManual) {
    state.manifest = [{
      query_id: query_id,
      task: state.task,
      text: text_vi,
      source_name: `${query_id}.txt`,
      n_events: n_events,
      events_confirmed: true
    }];
    saveManifest();
    renderManifestList();
    renderSelectionsList();
    renderExportTable();
    state.currentQueryId = query_id;
  }

  setLoading('btn-search', true);
  $('candidates-grid').innerHTML = '<div class="spinner"><div class="spinner-ring"></div></div>';

  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query_id, text_vi, text_en, task: state.task, n_events, k }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.detail || 'Search failed');

    state.candidates = data.candidates;
    state.selected = null;
    renderCandidates();
    $('results-count').textContent = `${data.total} candidates`;
    saveCurrentQueryToCache();
    toast(`Tìm được ${data.total} kết quả`, 'success');
  } catch (e) {
    $('candidates-grid').innerHTML = `<div class="empty-state"><p style="color:var(--red)">❌ ${e.message}</p></div>`;
    toast(e.message, 'error');
  } finally {
    setLoading('btn-search', false);
  }
}

// ---------------------------------------------------------------------------
// Render candidates grid
// ---------------------------------------------------------------------------

function renderCandidates() {
  const grid = $('candidates-grid');
  if (!state.candidates.length) {
    grid.innerHTML = '<div class="empty-state"><p>Không có kết quả</p></div>';
    return;
  }

  grid.innerHTML = '';
  state.candidates.forEach((c, idx) => {
    const frameIdx = c.representative_frames[0] ?? c.start_frame;
    const card = document.createElement('div');
    card.className = 'candidate-card';
    card.dataset.idx = idx;

    const key = candidateKey(c);
    const verdict = state.iterVerdict[key];
    if (verdict === 'matched') card.classList.add('matched');
    else if (verdict === 'not_matched') card.classList.add('not-matched');
    else if (verdict === 'unsure') card.classList.add('unsure');

    const scoreCls = scoreClass(c.best_score);
    card.innerHTML = `
      <div class="card-thumb">
        <img src="${keyframeUrl(c.video_id, frameIdx)}" alt="${c.video_id}" loading="lazy"/>
        <div class="card-rank">#${c.rank}</div>
        <div class="card-score ${scoreCls}">${fmtScore(c.best_score)}</div>
      </div>
      <div class="card-body">
        <div class="card-video-id">${c.video_id}</div>
      </div>`;

    card.addEventListener('click', (e) => {
      let forceOriginal = false;
      if (state.selected === idx) {
        // Discard draft edit and revert to original frame index
        const c = state.candidates[idx];
        if (c) {
          delete state.candidateDraftFrames[candidateKey(c)];
          forceOriginal = true;
        }
      }
      selectCandidate(idx, forceOriginal);
    });
    grid.appendChild(card);
  });
}

function setGridMode(isGrid) {
  state.gridMode = isGrid;
  $('candidates-grid').classList.toggle('list-mode', !isGrid);
  $('btn-grid-view').classList.toggle('active', isGrid);
  $('btn-list-view').classList.toggle('active', !isGrid);
}

// ---------------------------------------------------------------------------
// Verdict on card
// ---------------------------------------------------------------------------

function setVerdict(idx, verdict, e) {
  if (e) e.stopPropagation();
  const c = state.candidates[idx];
  if (!c) return;
  const key = candidateKey(c);
  state.iterVerdict[key] = verdict;
  renderCandidates();
  if (state.selected === idx) selectCandidate(idx);
}

// ---------------------------------------------------------------------------
// Detail panel
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// LocalStorage Persistence
// ---------------------------------------------------------------------------

function saveSelections() {
  localStorage.setItem('aic_selections', JSON.stringify(state.selections));
  sendWsUpdate();
}

function loadSelections() {
  const data = localStorage.getItem('aic_selections');
  if (data) {
    try {
      state.selections = JSON.parse(data);
    } catch (e) {
      console.error('Failed to load selections:', e);
      state.selections = [];
    }
  }
}

function selectCandidate(idx, forceOriginal = false) {
  // Clear any existing draft frame when switching to a different candidate!
  if (state.selected !== idx) {
    state.candidateDraftFrames = {};
  }
  state.selected = idx;
  const c = state.candidates[idx];
  if (!c) {
    state.selected = null;
    $('btn-confirm-selection').disabled = true;
    return;
  }
  document.querySelectorAll('.candidate-card').forEach((el, i) => {
    el.classList.toggle('selected', i === idx);
  });

  const frameIdx = submissionHelpers.candidateToSubmissionFrame(c);
  const img = $('preview-img');
  const vid = $('preview-vid');
  const placeholder = $('preview-placeholder');
  state.currentFps = null;
  
  img.style.display = 'none';
  if (vid) vid.style.display = 'none';
  placeholder.style.display = 'flex';
  
  const queryId = currentQueryId();
  const draftFrame = state.candidateDraftFrames[candidateKey(c)];
  const existing = state.selections.find(s => s.queryId === queryId && s.video_id === c.video_id);
  
  let initialFrame = frameIdx;
  if (!forceOriginal) {
    if (Number.isInteger(draftFrame)) {
      initialFrame = draftFrame;
    } else if (existing && Number.isInteger(existing.frame)) {
      initialFrame = existing.frame;
    }
  }
  
  state.currentPlaybackFrame = initialFrame;
  const playbackFrame = $('video-current-frame');
  if (playbackFrame) playbackFrame.textContent = String(initialFrame);
  
  if (vid) {
    const targetSrc = `/api/video/${c.video_id}`;
    const currentPath = vid.src ? new URL(vid.src, window.location.href).pathname : '';
    const targetPath = new URL(targetSrc, window.location.href).pathname;

    if (currentPath === targetPath && vid.readyState >= 1) {
      vid.style.display = 'block';
      placeholder.style.display = 'none';
      img.style.display = 'none';
      const fps = state.currentFps || 25;
      vid.currentTime = Math.max(0, initialFrame - 1) / fps;
    } else {
      // Hiển thị Video, fallback sang keyframe nếu video lỗi
      vid.src = targetSrc;
      vid.onloadeddata = async () => {
        vid.style.display = 'block';
        placeholder.style.display = 'none';
        img.style.display = 'none';
        
        // Tìm FPS để seek
        try {
          const res = await fetch(`/api/video_info/${c.video_id}`);
          if (res.ok) {
            const data = await res.json();
            state.currentFps = data.fps;
            vid.currentTime = Math.max(0, initialFrame - 1) / data.fps;
          }
        } catch (e) {
          console.warn('Cannot fetch video info:', e);
        }
      };
    }
    vid.onerror = () => {
      vid.style.display = 'none';
      // Load ảnh keyframe thay thế
      img.onload = () => { img.style.display = 'block'; placeholder.style.display = 'none'; };
      img.onerror = () => { img.style.display = 'none'; placeholder.style.display = 'flex'; };
      img.src = keyframeUrl(c.video_id, initialFrame);
    };
  } else {
    img.onload = () => { img.style.display = 'block'; placeholder.style.display = 'none'; };
    img.onerror = () => { img.style.display = 'none'; placeholder.style.display = 'flex'; };
    img.src = keyframeUrl(c.video_id, initialFrame);
  }

  $('detail-rank-badge').textContent = `#${c.rank}`;
  const videoId = $('detail-video-id');
  if (videoId) videoId.textContent = c.video_id;

  const scoresSection = $('detail-scores-section');
  const scoresBody = $('detail-scores-body');
  const entries = Object.entries(c.scores || {});
  if (entries.length) {
    scoresSection.style.display = '';
    scoresBody.innerHTML = entries.map(([k, v]) =>
      `<div class="score-row"><span class="score-key">${k}</span><span class="score-val">${fmtScore(v)}</span></div>`
    ).join('') +
    `<div class="score-row"><span class="score-key" style="font-weight:700">best</span><span class="score-val" style="color:var(--purple-light)">${fmtScore(c.best_score)}</span></div>`;
  } else {
    scoresSection.style.display = 'none';
  }

  const evidenceSection = $('detail-evidence-section');
  const evidenceText = $('detail-evidence-text');
  if (evidenceSection && evidenceText) {
    const rows = submissionHelpers.formatEvidence(c.evidence);
    if (rows.length) {
      evidenceSection.style.display = '';
      evidenceText.replaceChildren();
      for (const row of rows) {
        const line = document.createElement('div');
        const label = document.createElement('strong');
        label.textContent = `${row.label}: `;
        line.appendChild(label);
        line.appendChild(document.createTextNode(row.text));
        evidenceText.appendChild(line);
      }
    } else {
      evidenceSection.style.display = 'none';
      evidenceText.replaceChildren();
    }
  }

  if (state.task === 'trake') {
    $('frame-picker-single-row').style.display = 'none';
    const trakeContainer = $('frame-picker-trake-container');
    trakeContainer.style.display = 'flex';
    
    // Render dynamic event inputs
    const n_events = currentTrakeEvents() || 1;
    trakeContainer.innerHTML = '';
    
    const existingFrames = existing ? existing.frames : [];
    
    for (let i = 1; i <= n_events; i++) {
      const val = existingFrames[i - 1] !== undefined ? existingFrames[i - 1] : (i === 1 ? frameIdx : '');
      const slot = document.createElement('div');
      slot.className = 'trake-event-slot';
      slot.style.display = 'flex';
      slot.style.alignItems = 'center';
      slot.style.gap = '8px';
      slot.style.marginBottom = '6px';
      slot.innerHTML = `
        <span style="font-size:12px; min-width:55px; color:var(--text-secondary)">Event ${i}:</span>
        <input type="number" class="trake-event-input" id="trake-frame-input-${i}" value="${val}" placeholder="Frame" min="1" style="flex:1; padding:6px 8px; font-size:12px;" />
        <button class="selection-btn" style="width:auto; padding:0 8px; height:26px;" onclick="grabTrakeFrame(${i}, event)">Get</button>
      `;
      
      const input = slot.querySelector('.trake-event-input');
      input.addEventListener('input', (e) => {
        const frameVal = parseInt(e.target.value, 10);
        if (!isNaN(frameVal) && vid && vid.style.display !== 'none') {
          const fps = state.currentFps || 25;
          vid.currentTime = Math.max(0, frameVal - 1) / fps;
        }
      });
      
      trakeContainer.appendChild(slot);
    }
  } else {
    $('frame-picker-single-row').style.display = 'flex';
    $('frame-picker-trake-container').style.display = 'none';
    $('frame-input').value = initialFrame;
  }

  if (state.task === 'qa') {
    $('answer-input').value = existing ? existing.answer : '';
  }

  $('btn-confirm-selection').disabled = false;
}

function grabTrakeFrame(index, e) {
  if (e) e.preventDefault();
  const vid = $('preview-vid');
  let frameVal = 1;
  if (vid && vid.style.display !== 'none') {
    const fps = state.currentFps || 25;
    frameVal = Math.floor(vid.currentTime * fps) + 1;
  } else if (state.selected !== null) {
    const c = state.candidates[state.selected];
    frameVal = submissionHelpers.candidateToSubmissionFrame(c);
  }
  const input = $(`trake-frame-input-${index}`);
  if (input) {
    input.value = frameVal;
    input.style.borderColor = 'var(--green)';
    setTimeout(() => { input.style.borderColor = ''; }, 1000);
  }
}

// ---------------------------------------------------------------------------
// Confirm selection
// ---------------------------------------------------------------------------

function confirmSelection() {
  if (state.selected === null) return;
  const c = state.candidates[state.selected];
  const queryId = currentQueryId();

  if (state.task === 'kis') {
    const frameInput = parseInt($('frame-input').value, 10);
    const frame = isNaN(frameInput) ? submissionHelpers.candidateToSubmissionFrame(c) : frameInput;
    
    const isDup = state.selections.some(s => s.queryId === queryId && s.video_id === c.video_id && s.frames[0] === frame);
    if (isDup) {
      toast('Lựa chọn này đã tồn tại trong danh sách!', 'warning');
      return;
    }
    
    state.selections.push({
      queryId,
      task: 'kis',
      video_id: c.video_id,
      frames: [frame],
      answer: '',
      rank: c.rank || (state.selections.filter(s => s.queryId === queryId).length + 1)
    });
    toast(`Đã thêm ${c.video_id} f${frame} vào KIS`, 'success');
  } 
  else if (state.task === 'qa') {
    const frameInput = parseInt($('frame-input').value, 10);
    const frame = isNaN(frameInput) ? submissionHelpers.candidateToSubmissionFrame(c) : frameInput;
    const answer = submissionHelpers.prepareQaAnswer($('answer-input').value);
    
    if (!answer.trim()) {
      toast('Vui lòng nhập câu trả lời cho Q&A!', 'warning');
      return;
    }
    if (submissionHelpers.unicodeCodePointLength(answer) > 100) {
      toast('Câu trả lời vượt quá giới hạn 100 ký tự!', 'error');
      return;
    }
    
    const isDup = state.selections.some(s => s.queryId === queryId && s.video_id === c.video_id && s.frames[0] === frame && s.answer === answer);
    if (isDup) {
      toast('Lựa chọn này đã tồn tại trong danh sách!', 'warning');
      return;
    }
    
    state.selections.push({
      queryId,
      task: 'qa',
      video_id: c.video_id,
      frames: [frame],
      answer,
      rank: c.rank || (state.selections.filter(s => s.queryId === queryId).length + 1)
    });
    toast(`Đã thêm ${c.video_id} f${frame} vào Q&A`, 'success');
  } 
  else if (state.task === 'trake') {
    const n_events = currentTrakeEvents();
    if (!Number.isInteger(n_events) || n_events < 1) {
      toast('Xác nhận số events TRAKE trước khi chọn frame', 'warning');
      return;
    }
    const frames = [];
    let hasMissing = false;
    
    for (let i = 1; i <= n_events; i++) {
      const val = parseInt($(`trake-frame-input-${i}`).value, 10);
      if (isNaN(val)) {
        hasMissing = true;
      }
      frames.push(val);
    }
    
    if (hasMissing) {
      toast(`Vui lòng điền đủ ${n_events} sự kiện cho TRAKE!`, 'warning');
      return;
    }
    
    const isDup = state.selections.some(s => s.queryId === queryId && s.video_id === c.video_id && JSON.stringify(s.frames) === JSON.stringify(frames));
    if (isDup) {
      toast('Lựa chọn này đã tồn tại trong danh sách!', 'warning');
      return;
    }
    
    state.selections.push({
      queryId,
      task: 'trake',
      video_id: c.video_id,
      frames,
      answer: '',
      rank: c.rank || (state.selections.filter(s => s.queryId === queryId).length + 1)
    });
    toast(`Đã thêm ${c.video_id} (${n_events} events) vào TRAKE`, 'success');
  }
  
  saveSelections();
  renderSelectionsList();
  renderManifestList();
}

function removeSelection(idx) {
  state.selections.splice(idx, 1);
  saveSelections();
  renderSelectionsList();
  renderManifestList();
  if ($('view-export').classList.contains('active')) {
    renderExportTable();
  }
  toast('Đã xoá lựa chọn', 'info');
}

function moveSelection(index, direction) {
  const item = state.selections[index];
  if (!item) return;
  
  const queryId = item.queryId;
  const filteredIndexes = [];
  state.selections.forEach((s, idx) => {
    if (s.queryId === queryId) filteredIndexes.push(idx);
  });
  
  const currentPos = filteredIndexes.indexOf(index);
  if (currentPos === -1) return;
  
  const targetPos = currentPos + direction;
  if (targetPos < 0 || targetPos >= filteredIndexes.length) return;
  
  const targetIndex = filteredIndexes[targetPos];
  const temp = state.selections[index];
  state.selections[index] = state.selections[targetIndex];
  state.selections[targetIndex] = temp;
  
  saveSelections();
  renderSelectionsList();
  if ($('view-export').classList.contains('active')) {
    renderExportTable();
  }
}

function renderSelectionsList() {
  const list = $('selections-list');
  const queryId = currentQueryId();
  const querySelections = state.selections.filter(s => s.queryId === queryId);
  
  $('sel-count').textContent = querySelections.length;
  if (!querySelections.length) {
    list.innerHTML = '<div style="color:var(--text-muted);font-size:12px;text-align:center;padding:12px">Chưa có lựa chọn nào cho query này</div>';
    return;
  }
  
  list.innerHTML = querySelections.map((s, idx) => {
    const absoluteIdx = state.selections.indexOf(s);
    let infoText = '';
    
    if (s.task === 'kis') {
      infoText = `${s.video_id} <span style="color:var(--text-muted)">f${s.frames[0]}</span>`;
    } else if (s.task === 'qa') {
      const truncateAns = s.answer.length > 15 ? s.answer.substring(0, 15) + '...' : s.answer;
      const safeAns = escapeHtml(truncateAns);
      infoText = `${s.video_id} <span style="color:var(--text-muted)">f${s.frames[0]}</span><br><span style="font-size:11px;color:var(--text-secondary)">"${safeAns}"</span>`;
    } else if (s.task === 'trake') {
      infoText = `${s.video_id} <span style="color:var(--text-muted)">f[${s.frames.join(',')}]</span>`;
    }
    
    const isFirst = idx === 0;
    const isLast = idx === querySelections.length - 1;
    
    return `
      <div class="selection-item">
        <div class="selection-rank">${idx + 1}</div>
        <div class="selection-info">${infoText}</div>
        <div class="selection-actions">
          <button class="selection-btn" onclick="moveSelection(${absoluteIdx}, -1)" ${isFirst ? 'disabled' : ''} title="Lên">▲</button>
          <button class="selection-btn" onclick="moveSelection(${absoluteIdx}, 1)" ${isLast ? 'disabled' : ''} title="Xuống">▼</button>
          <button class="selection-del" onclick="removeSelection(${absoluteIdx})" title="Xoá">✕</button>
        </div>
      </div>`;
  }).join('');
}

// ---------------------------------------------------------------------------
// Iterative retrieval
// ---------------------------------------------------------------------------

function iterStart() {
  if (!submissionHelpers.canUseIterative(state.task)) {
    toast('Iterative chỉ dùng cho KIS', 'warning');
    return;
  }
  if (!state.candidates.length) {
    toast('Hãy tìm kiếm trước ở tab Tìm kiếm', 'warning');
    switchView('search');
    return;
  }
  state.iterCandidates = [...state.candidates];
  state.iterCursor = 0;
  state.iterRound = 1;
  state.iterRunning = true;
  state.iterMatchedList = [];
  state.iterUnsureList = [];
  state.iterExcluded = new Set();
  state.iterVerdict = {};

  $('btn-iter-start').disabled = true;
  $('btn-iter-finish').disabled = false;
  ['btn-iter-prev', 'btn-iter-next', 'btn-iter-skip'].forEach(id => { $(id).disabled = false; });
  $('iter-status-badge').textContent = 'Đang chạy';

  updateIterRoundBadge();
  buildRoundProgress();
  iterShowCurrent();
}

function iterFinish() {
  if (!state.iterRunning) return;
  state.iterRunning = false;
  $('btn-iter-start').disabled = false;
  $('btn-iter-finish').disabled = true;
  ['btn-iter-prev', 'btn-iter-next', 'btn-iter-skip'].forEach(id => { $(id).disabled = true; });
  $('iter-status-badge').textContent = 'Hoàn thành';

  const queryId = currentQueryId();
  state.iterMatchedList.forEach(c => {
    const frame = submissionHelpers.candidateToSubmissionFrame(c);
    const existing = state.selections.findIndex(s => s.video_id === c.video_id && s.queryId === queryId);
    if (existing < 0) {
      state.selections.push({ video_id: c.video_id, frames: [frame], answer: '', queryId, task: state.task, rank: c.rank });
    }
  });
  saveSelections();
  renderSelectionsList();
  renderManifestList();
  toast(`Iterative xong: ${state.iterMatchedList.length} matched, ${state.iterUnsureList.length} unsure`, 'success');
}

function iterVerdict(verdict) {
  if (!state.iterRunning || !state.iterCandidates.length) return;
  const c = state.iterCandidates[state.iterCursor];
  const key = candidateKey(c);
  state.iterVerdict[key] = verdict;

  if (verdict === 'matched') {
    if (!state.iterMatchedList.find(m => candidateKey(m) === key)) state.iterMatchedList.push(c);
    state.iterUnsureList = state.iterUnsureList.filter(m => candidateKey(m) !== key);
  } else if (verdict === 'not_matched') {
    state.iterExcluded.add(key);
    state.iterMatchedList = state.iterMatchedList.filter(m => candidateKey(m) !== key);
    state.iterUnsureList = state.iterUnsureList.filter(m => candidateKey(m) !== key);
  } else if (verdict === 'unsure') {
    if (!state.iterUnsureList.find(m => candidateKey(m) === key)) state.iterUnsureList.push(c);
    state.iterMatchedList = state.iterMatchedList.filter(m => candidateKey(m) !== key);
  }

  updateIterStats();
  renderIterLists();
  iterNav(1);
}

function iterNav(dir) {
  if (!state.iterRunning) return;
  const len = state.iterCandidates.length;
  if (!len) return;
  if (dir === 0) { iterNav(1); return; }
  const next = Math.max(0, Math.min(len - 1, state.iterCursor + dir));
  if (next === state.iterCursor && dir > 0) {
    toast('Đã đến cuối danh sách', 'info');
    return;
  }
  state.iterCursor = next;
  iterShowCurrent();
}

function iterShowCurrent() {
  const c = state.iterCandidates[state.iterCursor];
  if (!c) return;

  const frameIdx = c.representative_frames[0] ?? c.start_frame;
  $('iter-video-id').textContent = c.video_id;
  $('iter-frame').textContent = frameIdx;
  $('sc-clip').textContent = fmtScore(c.scores && c.scores.clip);
  $('sc-siglip').textContent = fmtScore(c.scores && c.scores.siglip);
  $('sc-fused').textContent = fmtScore(c.best_score);

  const img = $('iter-preview-img');
  const placeholder = $('iter-preview-placeholder');
  img.style.display = 'none';
  placeholder.style.display = 'flex';
  img.onload = () => { img.style.display = 'block'; placeholder.style.display = 'none'; };
  img.onerror = () => { img.style.display = 'none'; placeholder.style.display = 'flex'; };
  img.src = keyframeUrl(c.video_id, frameIdx);

  const key = candidateKey(c);
  const v = state.iterVerdict[key];
  $('btn-vb-matched').classList.toggle('active', v === 'matched');
  $('btn-vb-not').classList.toggle('active', v === 'not_matched');
  $('btn-vb-unsure').classList.toggle('active', v === 'unsure');

  updateIterRoundBadge();
}

function updateIterRoundBadge() {
  const total = state.iterCandidates.length;
  const cur = total ? state.iterCursor + 1 : 0;
  $('iter-round-badge').textContent = `${cur}/${total}`;
  $('btn-iter-prev').disabled = !state.iterRunning || state.iterCursor <= 0;
  $('btn-iter-next').disabled = !state.iterRunning || state.iterCursor >= (state.iterCandidates.length - 1);
  $('btn-iter-skip').disabled = !state.iterRunning;
}

function updateIterStats() {
  const matched = state.iterMatchedList.length;
  const unsure = state.iterUnsureList.length;
  const excluded = state.iterExcluded.size;
  $('iter-matched-count').textContent = matched;
  $('iter-unsure-count').textContent = unsure;
  $('iter-excluded-count').textContent = excluded;
  $('iter-matched-badge').textContent = matched;
  $('iter-unsure-badge').textContent = unsure;
  $('iter-excluded-badge').textContent = excluded;
}

function renderIterLists() {
  function renderList(containerId, items, colorVar) {
    const el = $(containerId);
    if (!items.length) {
      el.innerHTML = '<div style="padding:16px;color:var(--text-muted);font-size:12px;text-align:center">Chưa có</div>';
      return;
    }
    el.innerHTML = items.map(c => {
      const frameIdx = c.representative_frames[0] ?? c.start_frame;
      return `<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 14px;border-bottom:1px solid var(--border)">
        <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:${colorVar}">${c.video_id}</span>
        <span style="font-size:11px;color:var(--text-muted)">f${frameIdx} · ${fmtScore(c.best_score)}</span>
      </div>`;
    }).join('');
  }
  renderList('iter-matched-list', state.iterMatchedList, 'var(--green)');
  renderList('iter-unsure-list', state.iterUnsureList, 'var(--amber)');
}

function buildRoundProgress() {
  const container = $('round-progress');
  container.innerHTML = '';
  for (let i = 1; i <= state.iterMaxRounds; i++) {
    if (i > 1) {
      const arrow = document.createElement('span');
      arrow.className = 'round-arrow';
      arrow.textContent = '→';
      container.appendChild(arrow);
    }
    const step = document.createElement('div');
    step.className = 'round-step';
    const pill = document.createElement('div');
    pill.className = 'round-pill' + (i < state.iterRound ? ' done' : i === state.iterRound ? ' active' : '');
    pill.textContent = `Round ${i}`;
    step.appendChild(pill);
    container.appendChild(step);
  }
}

// ---------------------------------------------------------------------------
// Export view
// ---------------------------------------------------------------------------

getQuerySummary;

function getQuerySummary() {
  if (state.manifest.length) {
    return state.manifest.map((item) => ({
      queryId: item.query_id,
      task: item.task,
      n_events: item.n_events,
      events_confirmed: item.events_confirmed,
      selections: state.selections.filter((selection) => selection.queryId === item.query_id),
    }));
  }
  const summary = {};
  state.selections.forEach(s => {
    if (!summary[s.queryId]) {
      summary[s.queryId] = {
        queryId: s.queryId,
        task: s.task,
        selections: []
      };
    }
    summary[s.queryId].selections.push(s);
  });
  return Object.values(summary);
}

function removeQuerySelections(queryId) {
  const hasSelections = (state.selections || []).some(s => s.queryId === queryId);
  const isInManifest = (state.manifest || []).some(m => m.query_id === queryId);

  if (isInManifest && !hasSelections) {
    deleteQuery(queryId);
    return;
  }

  if (confirm(`Bạn có chắc chắn muốn xoá toàn bộ lựa chọn cho query ${queryId}?`)) {
    state.selections = (state.selections || []).filter(s => s.queryId !== queryId);
    saveSelections();
    renderSelectionsList();
    renderManifestList();
    renderExportTable();
    toast(`Đã xoá các lựa chọn của query ${queryId}`, 'info');
  }
}

function showExportReview(queryId) {
  const querySelections = state.selections.filter(s => s.queryId === queryId);
  if (!querySelections.length) return;
  
  const q = querySelections[0];
  $('review-query-id').textContent = queryId;
  $('review-task-badge').textContent = q.task.toUpperCase();
  
  const body = $('review-content-body');
  body.innerHTML = '';
  
  const card = $('export-review-card');
  card.style.display = 'block';
  const dlBtn = $('btn-review-download');
  if (dlBtn) {
    dlBtn.onclick = () => downloadSingleQueryCsv(queryId);
  }
  
  if (q.task === 'kis') {
    const html = querySelections.map((s, idx) => `
      <div style="display:flex; align-items:center; gap:12px; background:var(--bg-panel); border:1px solid var(--border); border-radius:var(--radius-md); padding:10px;">
        <div class="selection-rank" style="background:var(--purple-light)">${idx + 1}</div>
        <div style="width:120px; aspect-ratio:16/9; border-radius:var(--radius-sm); overflow:hidden; background:#000; flex-shrink:0;">
          <img src="${keyframeUrl(s.video_id, s.frames[0])}" style="width:100%; height:100%; object-fit:cover;" />
        </div>
        <div style="flex:1">
          <div style="font-family:'JetBrains Mono',monospace; font-weight:600; color:var(--cyan); font-size:13px;">${s.video_id}</div>
          <div style="font-size:12px; color:var(--text-secondary)">Frame: ${s.frames[0]}</div>
        </div>
      </div>
    `).join('');
    body.innerHTML = `<div style="display:flex; flex-direction:column; gap:8px;">${html}</div>`;
  }
  else if (q.task === 'qa') {
    const html = querySelections.map((s, idx) => `
      <div style="display:flex; align-items:flex-start; gap:12px; background:var(--bg-panel); border:1px solid var(--border); border-radius:var(--radius-md); padding:12px;">
        <div class="selection-rank" style="background:var(--cyan); margin-top:2px;">${idx + 1}</div>
        <div style="width:120px; aspect-ratio:16/9; border-radius:var(--radius-sm); overflow:hidden; background:#000; flex-shrink:0;">
          <img src="${keyframeUrl(s.video_id, s.frames[0])}" style="width:100%; height:100%; object-fit:cover;" />
        </div>
        <div style="flex:1">
          <div style="font-family:'JetBrains Mono',monospace; font-weight:600; color:var(--cyan); font-size:13px; margin-bottom:2px;">${s.video_id} <span style="font-size:11px; color:var(--text-muted)">f${s.frames[0]}</span></div>
          <div style="font-size:11px; font-weight:600; color:var(--text-muted); text-transform:uppercase; margin-bottom:2px;">Đáp án:</div>
          <div style="font-size:13px; color:var(--text-primary); font-weight:500; background:rgba(255,255,255,0.03); border:1px solid var(--border); padding:6px 10px; border-radius:var(--radius-sm); white-space:pre-wrap;">${escapeHtml(s.answer)}</div>
        </div>
      </div>
    `).join('');
    body.innerHTML = `<div style="display:flex; flex-direction:column; gap:8px;">${html}</div>`;
  }
  else if (q.task === 'trake') {
    const manifestItem = state.manifest.find((item) => item.query_id === queryId);
    const nEvents = manifestItem ? manifestItem.n_events : null;
    const html = querySelections.map((s, idx) => {
      const eventsHtml = submissionHelpers.buildTrakeReviewSlots(s.frames, nEvents).map((slot) => `
        <div style="flex:1; min-width:110px; display:flex; flex-direction:column; gap:4px; background:rgba(255,255,255,0.02); border:1px solid var(--border); border-radius:var(--radius-md); padding:6px; align-items:center;">
          <div style="font-size:10px; font-weight:600; color:var(--text-muted);">Sự kiện ${slot.event}</div>
          <div style="width:100%; aspect-ratio:16/9; border-radius:var(--radius-sm); overflow:hidden; background:#000;">
            ${slot.missing
              ? '<div style="height:100%;display:flex;align-items:center;justify-content:center;color:var(--amber);font-size:10px;">Thiếu frame</div>'
              : `<img src="${keyframeUrl(s.video_id, slot.frame)}" style="width:100%; height:100%; object-fit:cover;" />`}
          </div>
          <div style="font-family:'JetBrains Mono',monospace; font-size:10px; color:${slot.missing ? 'var(--amber)' : 'var(--cyan)'}; font-weight:600;">${slot.missing ? 'Thiếu frame' : `f${slot.frame}`}</div>
        </div>
      `).join('');
      
      return `
        <div style="display:flex; flex-direction:column; gap:8px; background:var(--bg-panel); border:1px solid var(--border); border-radius:var(--radius-md); padding:12px;">
          <div style="display:flex; align-items:center; gap:8px;">
            <div class="selection-rank" style="background:var(--amber)">${idx + 1}</div>
            <div style="font-family:'JetBrains Mono',monospace; font-weight:600; color:var(--cyan); font-size:13px;">${s.video_id}</div>
          </div>
          <div style="display:flex; gap:8px; overflow-x:auto; padding-bottom:4px;">
            ${eventsHtml}
          </div>
        </div>
      `;
    }).join('');
    body.innerHTML = `<div style="display:flex; flex-direction:column; gap:10px;">${html}</div>`;
  }
  
  card.scrollIntoView({ behavior: 'smooth' });
}

function downloadSingleQueryCsv(queryId) {
  const selections = (state.selections || []).filter(s => (s.queryId || s.query_id) === queryId);
  if (!selections.length) {
    toast(`Query "${queryId}" chưa có lựa chọn nào để tải xuống!`, 'warning');
    return;
  }

  const lines = [];
  selections.forEach(s => {
    const vid = (s.video_id || '').replace(/\.mp4$/, '');
    const frames = Array.isArray(s.frames) ? s.frames : [];
    const parts = [vid, ...frames.map(String)];
    if (s.task === 'qa' || s.answer) {
      let ans = String(s.answer ?? '');
      if (ans.includes(',') || ans.includes('"') || ans.includes('\n')) {
        ans = `"${ans.replace(/"/g, '""')}"`;
      }
      parts.push(ans);
    }
    lines.push(parts.join(','));
  });

  const csvContent = lines.join('\n') + '\n';
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `${queryId}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
  toast(`Đã tải xuống ${queryId}.csv`, 'success');
}

function renderExportTable() {
  const tbody = $('export-tbody');
  const count = $('export-query-count');
  const summary = getQuerySummary();

  if (!summary.length) {
    const row = document.createElement('tr');
    const cell = document.createElement('td');
    cell.setAttribute('colspan', '5');
    cell.style.textAlign = 'center';
    cell.style.color = 'var(--text-muted)';
    cell.style.padding = '32px';
    cell.textContent = 'Chưa có kết quả. Hãy tìm kiếm và xác nhận lựa chọn trước.';
    row.appendChild(cell);
    tbody.replaceChildren(row);
    count.textContent = '0 queries';
    $('export-review-card').style.display = 'none';
    refreshPreview();
    return;
  }

  count.textContent = `${summary.length} quer${summary.length !== 1 ? 'ies' : 'y'}`;
  tbody.replaceChildren();
  summary.forEach((q) => {
    let detailsText = '';
    let statusText = q.selections.length ? 'Ready' : 'Chưa có dòng';
    let statusClass = q.selections.length ? 'badge badge-green' : 'badge badge-amber';
    let statusTitle = '';
    
    if (q.task === 'kis') {
      detailsText = `${q.selections.length} cảnh (KIS)`;
    } else if (q.task === 'qa') {
      detailsText = `${q.selections.length} câu trả lời (QA)`;
    } else if (q.task === 'trake') {
      detailsText = `${q.selections.length} chuỗi sự kiện (TRAKE)`;
      const n_events = q.n_events;
      const incomplete = !q.events_confirmed || !Number.isInteger(n_events) || q.selections.some(s => s.frames.length !== n_events);
      if (incomplete) {
        statusText = 'Thiếu event';
        statusClass = 'badge badge-amber';
        statusTitle = `Có dòng chưa đủ ${n_events} sự kiện`;
      }
    }

    const row = document.createElement('tr');
    const queryCell = document.createElement('td');
    queryCell.style.fontFamily = "'JetBrains Mono', monospace";
    queryCell.style.fontSize = '12px';
    queryCell.textContent = q.queryId;

    const taskCell = document.createElement('td');
    const taskBadge = document.createElement('span');
    taskBadge.className = 'badge badge-purple';
    taskBadge.textContent = q.task.toUpperCase();
    taskCell.appendChild(taskBadge);

    const detailsCell = document.createElement('td');
    detailsCell.style.fontSize = '12px';
    detailsCell.style.color = 'var(--text-secondary)';
    detailsCell.style.maxWidth = '250px';
    detailsCell.style.overflow = 'hidden';
    detailsCell.style.textOverflow = 'ellipsis';
    detailsCell.style.whiteSpace = 'nowrap';
    detailsCell.setAttribute('title', detailsText);
    detailsCell.textContent = detailsText;

    const statusCell = document.createElement('td');
    const statusBadge = document.createElement('span');
    statusBadge.className = statusClass;
    statusBadge.textContent = statusText;
    if (statusTitle) statusBadge.setAttribute('title', statusTitle);
    statusCell.appendChild(statusBadge);

    const actionCell = document.createElement('td');
    const actions = document.createElement('div');
    actions.style.display = 'flex';
    actions.style.gap = '6px';
    
    const reviewButton = document.createElement('button');
    reviewButton.type = 'button';
    reviewButton.className = 'btn-translate';
    reviewButton.style.padding = '2px 8px';
    reviewButton.style.fontSize = '11px';
    reviewButton.textContent = 'Xem';
    reviewButton.addEventListener('click', () => showExportReview(q.queryId));

    const downloadButton = document.createElement('button');
    downloadButton.type = 'button';
    downloadButton.className = 'btn-translate';
    downloadButton.style.padding = '2px 8px';
    downloadButton.style.fontSize = '11px';
    downloadButton.style.color = 'var(--cyan)';
    downloadButton.style.borderColor = 'rgba(6,182,212,0.3)';
    downloadButton.style.background = 'rgba(6,182,212,0.1)';
    downloadButton.textContent = 'Tải file';
    downloadButton.title = `Tải file ${q.queryId}.csv`;
    if (!q.selections.length) {
      downloadButton.disabled = true;
      downloadButton.style.opacity = '0.4';
      downloadButton.style.cursor = 'not-allowed';
    } else {
      downloadButton.addEventListener('click', () => downloadSingleQueryCsv(q.queryId));
    }

    const removeButton = document.createElement('button');
    removeButton.type = 'button';
    removeButton.className = 'btn-translate';
    removeButton.style.padding = '2px 8px';
    removeButton.style.fontSize = '11px';
    removeButton.style.color = 'var(--red)';
    removeButton.style.borderColor = 'rgba(239,68,68,0.25)';
    removeButton.style.background = 'rgba(239,68,68,0.1)';
    removeButton.textContent = 'Xoá';
    removeButton.addEventListener('click', () => removeQuerySelections(q.queryId));

    actions.append(reviewButton, downloadButton, removeButton);
    actionCell.appendChild(actions);

    row.append(queryCell, taskCell, detailsCell, statusCell, actionCell);
    tbody.appendChild(row);
  });

  refreshPreview();
}

function refreshPreview() {
  const preview = $('csv-preview');
  const manifestIds = new Set(state.manifest.map((item) => item.query_id));
  const selections = state.manifest.length
    ? state.selections.filter((selection) => manifestIds.has(selection.queryId))
    : state.selections;
  if (!selections.length) { preview.textContent = '— chưa có dữ liệu —'; return; }
  
  const byQuery = {};
  selections.forEach(s => {
    if (!byQuery[s.queryId]) byQuery[s.queryId] = [];
    byQuery[s.queryId].push(s);
  });
  
  let text = '';
  Object.entries(byQuery).forEach(([queryId, selections]) => {
    text += `=== submission/${queryId}.csv ===\n`;
    selections.forEach(s => {
      const vid = s.video_id.replace(/\.mp4$/, '');
      const parts = [vid, ...s.frames.map(String)];
      if (s.task === 'qa') {
        let ans = String(s.answer ?? '');
        if (ans.includes(',') || ans.includes('"') || ans.includes('\n')) {
          ans = `"${ans.replace(/"/g, '""')}"`;
        }
        parts.push(ans);
      }
      text += parts.join(',') + '\n';
    });
    text += '\n';
  });
  
  preview.textContent = text.trim();
}

async function doExport() {
  let exportManifest = state.manifest;
  if (!exportManifest || exportManifest.length === 0) {
    const queryIds = [...new Set(state.selections.map(s => s.queryId))];
    exportManifest = queryIds.map(qid => {
      const firstSel = state.selections.find(s => s.queryId === qid);
      const task = firstSel ? firstSel.task : 'kis';
      const n_events = firstSel && firstSel.frames ? firstSel.frames.length : 1;
      return {
        query_id: qid,
        task: task,
        text: qid,
        source_name: `${qid}.txt`,
        n_events: n_events,
        events_confirmed: true
      };
    });
  }

  if (exportManifest.length === 0) {
    toast('Không có câu hỏi hay lựa chọn nào để xuất!', 'warning');
    return;
  }

  const manifestIds = new Set(exportManifest.map((item) => item.query_id));
  const rows = state.selections
    .filter((selection) => manifestIds.has(selection.queryId))
    .map((selection) => ({
      query_id: selection.queryId,
      video_id: selection.video_id,
      frames: selection.frames,
      answer: String(selection.answer ?? ''),
    }));

  setLoading('btn-export', true);
  try {
    const res = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ manifest: exportManifest, rows }),
    });
    if (!res.ok) {
      const payload = await res.json();
      setValidationReport(payload.detail || payload);
      toast('Validation export thất bại', 'error');
      return;
    }
    if (!submissionHelpers.canDownloadValidatedZip(
      res.headers.get('X-Validation-Status'),
      res.headers.get('Content-Type'),
    )) {
      setValidationReport({ errors: [{ message: 'Export không trả về trạng thái PASS' }], warnings: [] });
      toast('Export không đạt PASS', 'error');
      return;
    }
    setValidationReport(null);
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = `submission.zip`;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    toast('Đã tải xuống submission.zip', 'success');
  } catch (e) {
    setValidationReport({ errors: [{ message: e.message || 'Không thể export submission' }], warnings: [] });
    toast(e.message || 'Không thể export submission', 'error');
  } finally {
    setLoading('btn-export', false);
  }
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------

document.addEventListener('keydown', (e) => {
  const tag = document.activeElement.tagName.toLowerCase();
  if (['input', 'textarea'].includes(tag)) return;
  const iterActive = $('view-iterative').classList.contains('active');
  if (iterActive && state.iterRunning) {
    if (e.key === 'm' || e.key === 'M') { iterVerdict('matched'); return; }
    if (e.key === 'n' || e.key === 'N') { iterVerdict('not_matched'); return; }
    if (e.key === 'u' || e.key === 'U') { iterVerdict('unsure'); return; }
    if (e.key === 'ArrowLeft') { e.preventDefault(); iterNav(-1); return; }
    if (e.key === 'ArrowRight') { e.preventDefault(); iterNav(1); return; }
  }
});

// ---------------------------------------------------------------------------
// Toast styles
// ---------------------------------------------------------------------------

(function injectToastStyles() {
  const style = document.createElement('style');
  style.textContent = `
    #toast-container {
      position: fixed; bottom: 24px; right: 24px;
      display: flex; flex-direction: column; gap: 8px;
      z-index: 9999; pointer-events: none;
    }
    .toast {
      padding: 11px 18px; border-radius: 10px; font-size: 13px;
      font-weight: 500; color: #fff; opacity: 0;
      transform: translateY(8px);
      transition: opacity 0.25s, transform 0.25s;
      backdrop-filter: blur(12px); pointer-events: auto; max-width: 320px;
    }
    .toast.show { opacity: 1; transform: translateY(0); }
    .toast-info    { background: rgba(30,35,55,0.95); border: 1px solid rgba(124,58,237,0.3); }
    .toast-success { background: rgba(20,40,30,0.95); border: 1px solid rgba(34,197,94,0.35); }
    .toast-warning { background: rgba(40,30,15,0.95); border: 1px solid rgba(245,158,11,0.35); }
    .toast-error   { background: rgba(40,15,15,0.95); border: 1px solid rgba(239,68,68,0.35); }
  `;
  document.head.appendChild(style);
})();

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  loadQueryCache();
  loadSelections();
  loadManifest();
  loadPrivateQueryCache();
  loadPrivateSelections();
  connectWs();
  scheduleStatusPolling();
  if (state.manifest.length) selectManifestQuery(state.manifest[0].query_id);
  else selectTask('kis');
  $('query-id-input').addEventListener('input', () => {
    clearQueryWorkspace();
    $('export-query-id').value = $('query-id-input').value;
    state.currentQueryId = null;
    renderManifestList();
    renderSelectionsList();
  });
  $('n-events-input').addEventListener('change', () => {
    if (currentManifestItem()?.task !== 'trake') return;
    updateSelectedTrakeState(false);
    if (state.selected !== null) selectCandidate(state.selected);
  });
  $('trake-events-confirmed').addEventListener('change', (event) => {
    updateSelectedTrakeState(event.target.checked);
  });

  const vid = $('preview-vid');
  if (vid) {
    vid.addEventListener('timeupdate', updatePlaybackFrame);
    vid.addEventListener('seeked', updatePlaybackFrame);
  }

  const frameInput = $('frame-input');
  if (frameInput) {
    frameInput.addEventListener('input', (e) => {
      const frame = parseInt(e.target.value, 10);
      if (!isNaN(frame) && state.selected !== null) {
        const candidate = state.candidates[state.selected];
        if (candidate) state.candidateDraftFrames[candidateKey(candidate)] = frame;
      }
      if (!isNaN(frame) && vid && vid.style.display !== 'none') {
        const fps = state.currentFps || 25;
        vid.currentTime = Math.max(0, frame - 1) / fps;
      }
    });
  }

  // Event listeners for Private Search Tab
  $('private-query-id-input').addEventListener('input', () => {
    clearPrivateQueryWorkspace();
    state.currentPrivateQueryId = null;
    renderPrivateManifestList();
    renderPrivateSelectionsList();
  });
  
  $('private-n-events-input').addEventListener('change', () => {
    if (currentPrivateManifestItem()?.task !== 'trake') return;
    updateSelectedPrivateTrakeState(false);
    if (state.privateSelected !== null) selectPrivateCandidate(state.privateSelected);
  });
  
  $('private-trake-events-confirmed').addEventListener('change', (event) => {
    updateSelectedPrivateTrakeState(event.target.checked);
  });

  const privateFrameInput = $('private-frame-input');
  if (privateFrameInput) {
    privateFrameInput.addEventListener('input', (e) => {
      const frame = parseInt(e.target.value, 10);
      if (!isNaN(frame) && state.privateSelected !== null) {
        const candidate = state.privateCandidates[state.privateSelected];
        if (candidate) state.privateCandidateDraftFrames[candidateKey(candidate)] = frame;
      }
      const privateVid = $('private-preview-vid');
      if (!isNaN(frame) && privateVid && privateVid.style.display !== 'none') {
        const fps = state.privateCurrentFps || 25;
        privateVid.currentTime = Math.max(0, frame - 1) / fps;
      }
    });
  }
  
  const privateVid = $('private-preview-vid');
  if (privateVid) {
    privateVid.addEventListener('timeupdate', updatePrivatePlaybackFrame);
    privateVid.addEventListener('seeked', updatePrivatePlaybackFrame);
  }

  // Initialize videos list
  fetchVideosList();

  // Event listeners for Manual Entry Tab
  const manualFrameInput = $('manual-frame-input');
  if (manualFrameInput) {
    manualFrameInput.addEventListener('input', (e) => {
      const frame = parseInt(e.target.value, 10);
      const vid = $('manual-preview-vid');
      if (!isNaN(frame) && vid && vid.style.display !== 'none') {
        const fps = state.manualCurrentFps || 25;
        vid.currentTime = Math.max(0, frame - 1) / fps;
      }
    });
  }
  
  const manualVid = $('manual-preview-vid');
  if (manualVid) {
    manualVid.addEventListener('timeupdate', updateManualPlaybackFrame);
    manualVid.addEventListener('seeked', updateManualPlaybackFrame);
  }

  window.addEventListener('beforeunload', () => {
    saveCurrentQueryToCache();
    saveCurrentPrivateQueryToCache();
  });
});

// Nạp query pack ZIP hoặc một/nhiều query TXT.
async function handleQueryFileUpload(event) {
  const files = Array.from(event.target.files || []);
  if (!files.length) return;
  const zipFiles = files.filter((file) => file.name.toLowerCase().endsWith('.zip'));
  const textFiles = files.filter((file) => file.name.toLowerCase().endsWith('.txt'));
  const invalidFiles = files.filter((file) => !zipFiles.includes(file) && !textFiles.includes(file));

  if (invalidFiles.length || (zipFiles.length && (zipFiles.length !== 1 || textFiles.length))) {
    setValidationReport({
      errors: [{ message: 'Chỉ nhận một ZIP hoặc một/nhiều TXT; không trộn ZIP với TXT.' }],
      warnings: [],
    });
    toast('Chỉ nhận một ZIP hoặc một/nhiều TXT', 'error');
    event.target.value = '';
    return;
  }

  try {
    let response;
    if (zipFiles.length === 1) {
      response = await fetch('/api/query-pack/zip', {
        method: 'POST',
        headers: { 'Content-Type': 'application/zip' },
        body: await zipFiles[0].arrayBuffer(),
      });
    } else {
      response = await fetch('/api/query-pack/texts', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          files: await Promise.all(textFiles.map(async (file) => ({ filename: file.name, content: await file.text() }))),
        }),
      });
    }
    const payload = await response.json();
    const report = response.ok ? payload : payload.detail;
    if (submissionHelpers.canInstallManifest(response.ok, report)) {
      state.queryCache = {};
      state.selections = [];
      state.manifest = [];
      saveQueryCache();
      saveSelections();
      saveManifest();

      state.manifest = report.manifest.reduce(
        (items, item) => submissionHelpers.upsertManifestItem(items, item),
        [],
      );
      saveManifest();
      if (state.manifest.length) selectManifestQuery(state.manifest[0].query_id);
      else renderManifestList();
    }
    setValidationReport(report);
    if (response.ok && report.ok) toast(`Đã nạp ${state.manifest.length} query`, 'success');
    else toast('Query pack có lỗi validation', 'warning');
  } catch (e) {
    setValidationReport({ errors: [{ message: e.message || 'Không thể nạp query pack' }], warnings: [] });
    toast(e.message || 'Không thể nạp query pack', 'error');
  } finally {
    event.target.value = '';
  }
}

// WebSocket Collaboration
let ws;
let isApplyingWsUpdate = false;

function connectWs() {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${window.location.host}/api/ws`);

  ws.onopen = () => {
    console.log('WebSocket connected');
    const statusText = $('status-text');
    if (statusText) {
      statusText.textContent = 'Online (Đồng bộ)';
    }
    const statusDot = $('status-dot');
    if (statusDot) {
      statusDot.style.background = 'var(--green)';
      statusDot.style.boxShadow = '0 0 6px var(--green)';
    }
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === 'clear_all') {
        hideClearCacheModals();
        resetClientStateAndUI(true);
      } else if (msg.type === 'clear_cache_waiting') {
        currentClearRequestId = msg.request_id;
        showClearCacheWaiting(msg.count || 1);
      } else if (msg.type === 'clear_cache_prompt') {
        showClearCachePrompt(msg.request_id);
      } else if (msg.type === 'clear_cache_rejected') {
        hideClearCacheModals();
        toast(msg.reason || 'Yêu cầu xóa cache bị từ chối do không có sự đồng ý của thành viên nào.', 'warning');
      } else if (msg.type === 'clear_cache_dismiss') {
        hideClearCacheModals();
      } else if (msg.type === 'init' || msg.type === 'update') {
        isApplyingWsUpdate = true;
        try {
          if (msg.type === 'init') {
            state.manifest = msg.manifest || [];
            state.selections = msg.selections || [];
          } else {
            if (msg.manifest && msg.manifest.length > 0) {
              state.manifest = msg.manifest;
            } else if (!state.manifest || state.manifest.length === 0) {
              state.manifest = [];
            }

            if (msg.selections) {
              if (!state.selections || state.selections.length === 0) {
                state.selections = msg.selections;
              } else if (msg.selections.length > 0) {
                const incomingQids = new Set(msg.selections.map(s => s.queryId || s.query_id).filter(Boolean));
                const keptLocal = state.selections.filter(s => {
                  const qid = s.queryId || s.query_id;
                  return qid && !incomingQids.has(qid);
                });
                state.selections = [...keptLocal, ...msg.selections];
              }
            }
          }
          
          const deserialized = {};
          if (msg.queryCache) {
            for (const [qid, cache] of Object.entries(msg.queryCache)) {
              if (!cache) continue;
              deserialized[qid] = Object.assign({}, cache, {
                iterExcluded: new Set(cache.iterExcluded || []),
              });
            }
          }
          state.queryCache = deserialized;

          saveManifest();
          saveSelections();
          saveQueryCache();

          renderManifestList();
          renderSelectionsList();
          renderExportTable();

          if (state.currentQueryId && currentManifestItem()) {
            const item = currentManifestItem();
            const form = item ? submissionHelpers.manifestQueryFormState(item) : null;
            loadQueryFromCache(state.currentQueryId, form);
          } else {
            state.currentQueryId = null;
            clearQueryWorkspace();
            renderCandidates();
          }
        } finally {
          isApplyingWsUpdate = false;
        }
      } else if (msg.type === 'delete_query') {
        const deletedQid = msg.query_id;
        state.manifest = (state.manifest || []).filter(m => m.query_id !== deletedQid);
        state.selections = (state.selections || []).filter(s => (s.queryId || s.query_id) !== deletedQid);
        if (state.queryCache) {
          delete state.queryCache[deletedQid];
        }
        if (state.privateSelections) {
          state.privateSelections = state.privateSelections.filter(s => (s.queryId || s.query_id) !== deletedQid);
          savePrivateSelections();
        }
        if (state.privateQueryCache) {
          delete state.privateQueryCache[deletedQid];
          savePrivateQueryCache();
        }

        saveManifest();
        saveSelections();
        saveQueryCache();

        if (state.currentQueryId === deletedQid) {
          state.currentQueryId = null;
          state.selected = null;
          state.candidates = [];
          clearQueryWorkspace();
          const qi = $('query-input'); if (qi) qi.value = '';
          const tt = $('translated-text'); if (tt) tt.value = '';
          const qii = $('query-id-input'); if (qii) qii.value = '';
          if (state.manifest.length > 0) {
            selectManifestQuery(state.manifest[0].query_id);
          }
        }

        if (state.currentPrivateQueryId === deletedQid) {
          state.currentPrivateQueryId = null;
          state.privateSelected = null;
          state.privateCandidates = [];
          clearPrivateQueryWorkspace();
          const pqi = $('private-query-input'); if (pqi) pqi.value = '';
          const ptt = $('private-translated-text'); if (ptt) ptt.value = '';
          const pqii = $('private-query-id-input'); if (pqii) pqii.value = '';
          if (state.manifest.length > 0) {
            selectPrivateManifestQuery(state.manifest[0].query_id);
          }
        }

        renderManifestList();
        renderSelectionsList();
        renderPrivateManifestList();
        renderPrivateSelectionsList();
        if ($('view-export') && $('view-export').classList.contains('active')) {
          renderExportTable();
        }
      }
    } catch (e) {
      console.error('Error handling WebSocket message:', e);
    }
  };

  ws.onclose = () => {
    console.log('WebSocket disconnected, reconnecting in 2s...');
    const statusText = $('status-text');
    if (statusText) {
      statusText.textContent = 'Mất kết nối';
    }
    const statusDot = $('status-dot');
    if (statusDot) {
      statusDot.style.background = 'var(--red)';
      statusDot.style.boxShadow = '0 0 6px var(--red)';
    }
    setTimeout(connectWs, 2000);
  };
}

function sendWsUpdate() {
  if (isApplyingWsUpdate) return;
  if (ws && ws.readyState === WebSocket.OPEN) {
    const serializedCache = {};
    for (const [qid, cache] of Object.entries(state.queryCache)) {
      if (!cache) continue;
      serializedCache[qid] = Object.assign({}, cache, {
        iterExcluded: Array.from(cache.iterExcluded || []),
      });
    }

    ws.send(JSON.stringify({
      type: 'update',
      manifest: state.manifest,
      selections: state.selections,
      queryCache: serializedCache
    }));
  }
}

// ===========================================================================
// TÌM KIẾM RIÊNG (PRIVATE SEARCH)
// ===========================================================================

function selectPrivateTask(task) {
  state.privateTask = task;
  document.querySelectorAll('#private-sidebar .task-pill').forEach((el) => {
    el.classList.toggle('active', el.id === `private-pill-${task}`);
  });
  
  const eventsSection = $('private-n-events-section');
  if (eventsSection) {
    eventsSection.style.display = task === 'trake' ? 'flex' : 'none';
  }
  
  const answerSection = $('private-answer-section');
  if (answerSection) {
    answerSection.style.display = task === 'qa' ? 'block' : 'none';
  }
  
  if (state.privateSelected !== null) {
    selectPrivateCandidate(state.privateSelected);
  }
}

async function doPrivateTranslate() {
  const text_vi = $('private-query-input').value.trim();
  if (!text_vi) return;
  setLoading('private-btn-translate', true);
  try {
    const res = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text_vi }),
    });
    const data = await res.json();
    if (data.ok) {
      $('private-translated-text').value = data.text_en;
      saveCurrentPrivateQueryToCache();
    }
  } catch (e) {
    toast('Dịch thất bại: ' + e.message, 'error');
  } finally {
    setLoading('private-btn-translate', false);
  }
}

function currentPrivateQueryId() {
  return state.currentPrivateQueryId || $('private-query-id-input').value.trim() || 'q1';
}

function currentPrivateManifestItem() {
  return state.manifest.find((item) => item.query_id === state.currentPrivateQueryId) || null;
}

function currentPrivateTrakeEvents() {
  const item = currentPrivateManifestItem();
  return item && item.task === 'trake' ? item.n_events : parseInt($('private-n-events-input').value, 10);
}

function updateSelectedPrivateTrakeState(confirmEvents) {
  const item = currentPrivateManifestItem();
  if (!item || item.task !== 'trake') return;
  const nEvents = parseInt($('private-n-events-input').value, 10);
  if (!Number.isInteger(nEvents) || nEvents < 1) {
    state.manifest = submissionHelpers.updateTrakeState(
      state.manifest,
      item.query_id,
      item.n_events,
      false,
    );
    $('private-trake-events-confirmed').checked = false;
    saveManifest();
    renderPrivateManifestList();
    toast('Số events TRAKE phải lớn hơn 0', 'warning');
    return;
  }
  state.manifest = confirmEvents
    ? submissionHelpers.updateTrakeState(state.manifest, item.query_id, nEvents, true)
    : submissionHelpers.changeTrakeEventCount(state.manifest, item.query_id, nEvents);
  $('private-trake-events-confirmed').checked = Boolean(confirmEvents);
  saveManifest();
  renderPrivateManifestList();
}

function selectPrivateManifestQuery(queryId) {
  saveCurrentPrivateQueryToCache();
  
  const item = state.manifest.find((item) => item.query_id === queryId);
  if (!item) return;
  
  clearPrivateQueryWorkspace();
  state.privateSelected = null;
  state.privateCandidates = [];
  
  const form = submissionHelpers.manifestQueryFormState(item);
  state.currentPrivateQueryId = form.queryId;
  $('private-query-id-input').value = form.queryId;
  $('private-query-input').value = form.text;
  $('private-n-events-input').value = form.nEvents;
  $('private-trake-events-confirmed').checked = form.eventsConfirmed;
  selectPrivateTask(form.task);
  
  loadPrivateQueryFromCache(form.queryId, form);
  
  renderPrivateManifestList();
  renderPrivateSelectionsList();
}

async function doPrivateSearch() {
  const text_vi = $('private-query-input').value.trim();
  if (!text_vi) { toast('Nhập câu hỏi', 'warning'); return; }
  const text_en = $('private-translated-text').value.trim();
  const query_id = currentPrivateQueryId();
  const k = parseInt($('private-topk-slider').value, 10);
  const n_events = currentPrivateTrakeEvents() || 1;

  setLoading('private-btn-search', true);
  $('private-candidates-grid').innerHTML = '<div class="spinner"><div class="spinner-ring"></div></div>';

  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query_id, text_vi, text_en, task: state.privateTask, n_events, k }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.detail || 'Search failed');

    state.privateCandidates = data.candidates;
    state.privateSelected = null;
    renderPrivateCandidates();
    $('private-results-count').textContent = `${data.total} candidates`;
    saveCurrentPrivateQueryToCache();
    toast(`Tìm được ${data.total} kết quả (Riêng)`, 'success');
  } catch (e) {
    $('private-candidates-grid').innerHTML = `<div class="empty-state"><p style="color:var(--red)">❌ ${e.message}</p></div>`;
    toast(e.message, 'error');
  } finally {
    setLoading('private-btn-search', false);
  }
}

function renderPrivateCandidates() {
  const grid = $('private-candidates-grid');
  if (!grid) return;
  if (!state.privateCandidates.length) {
    grid.innerHTML = '<div class="empty-state"><p>Không có kết quả</p></div>';
    return;
  }

  grid.innerHTML = '';
  state.privateCandidates.forEach((c, idx) => {
    const frameIdx = c.representative_frames[0] ?? c.start_frame;
    const card = document.createElement('div');
    card.className = 'candidate-card';
    card.dataset.idx = idx;

    const scoreCls = scoreClass(c.best_score);
    card.innerHTML = `
      <div class="card-thumb">
        <img src="${keyframeUrl(c.video_id, frameIdx)}" alt="${c.video_id}" loading="lazy"/>
        <div class="card-rank">#${c.rank}</div>
        <div class="card-score ${scoreCls}">${fmtScore(c.best_score)}</div>
      </div>
      <div class="card-body">
        <div class="card-video-id">${c.video_id}</div>
      </div>`;

    card.addEventListener('click', (e) => {
      let forceOriginal = false;
      if (state.privateSelected === idx) {
        const c = state.privateCandidates[idx];
        if (c) {
          delete state.privateCandidateDraftFrames[candidateKey(c)];
          forceOriginal = true;
        }
      }
      selectPrivateCandidate(idx, forceOriginal);
    });
    grid.appendChild(card);
  });
}

function selectPrivateCandidate(idx, forceOriginal = false) {
  if (state.privateSelected !== idx) {
    state.privateCandidateDraftFrames = {};
  }
  state.privateSelected = idx;
  const c = state.privateCandidates[idx];
  if (!c) {
    state.privateSelected = null;
    $('private-btn-confirm-selection').disabled = true;
    $('private-btn-confirm-shared').disabled = true;
    return;
  }
  document.querySelectorAll('#private-candidates-grid .candidate-card').forEach((el, i) => {
    el.classList.toggle('selected', i === idx);
  });

  const frameIdx = submissionHelpers.candidateToSubmissionFrame(c);
  const img = $('private-preview-img');
  const vid = $('private-preview-vid');
  const placeholder = $('private-preview-placeholder');
  state.privateCurrentFps = null;
  
  img.style.display = 'none';
  if (vid) vid.style.display = 'none';
  placeholder.style.display = 'flex';
  
  const queryId = currentPrivateQueryId();
  const draftFrame = state.privateCandidateDraftFrames[candidateKey(c)];
  const existing = state.privateSelections.find(s => s.queryId === queryId && s.video_id === c.video_id);
  
  let initialFrame = frameIdx;
  if (!forceOriginal) {
    if (Number.isInteger(draftFrame)) {
      initialFrame = draftFrame;
    } else if (existing && Number.isInteger(existing.frame)) {
      initialFrame = existing.frame;
    }
  }
  
  state.privateCurrentPlaybackFrame = initialFrame;
  const playbackFrame = $('private-video-current-frame');
  if (playbackFrame) playbackFrame.textContent = String(initialFrame);
  
  if (vid) {
    const targetSrc = `/api/video/${c.video_id}`;
    const currentPath = vid.src ? new URL(vid.src, window.location.href).pathname : '';
    const targetPath = new URL(targetSrc, window.location.href).pathname;

    if (currentPath === targetPath && vid.readyState >= 1) {
      img.style.display = 'none';
      placeholder.style.display = 'none';
      vid.style.display = 'block';
      fetch(`/api/video_info/${c.video_id}`)
        .then(res => res.json())
        .then(data => {
          state.privateCurrentFps = data.fps;
          vid.currentTime = Math.max(0, initialFrame - 1) / data.fps;
        })
        .catch(() => {
          state.privateCurrentFps = 25;
          vid.currentTime = Math.max(0, initialFrame - 1) / 25;
        });
    } else {
      vid.removeAttribute('src');
      vid.load();
      fetch(`/api/video_info/${c.video_id}`)
        .then(res => res.json())
        .then(data => {
          state.privateCurrentFps = data.fps;
          vid.src = targetSrc;
          vid.load();
          vid.onloadeddata = () => {
            img.style.display = 'none';
            placeholder.style.display = 'none';
            vid.style.display = 'block';
            vid.currentTime = Math.max(0, initialFrame - 1) / data.fps;
            vid.onloadeddata = null;
          };
          vid.onerror = () => {
            vid.removeAttribute('src');
            vid.style.display = 'none';
            placeholder.style.display = 'none';
            img.src = keyframeUrl(c.video_id, frameIdx);
            img.style.display = 'block';
            img.onerror = () => {
              img.removeAttribute('src');
              img.style.display = 'none';
              placeholder.style.display = 'flex';
            };
          };
        })
        .catch(() => {
          state.privateCurrentFps = 25;
          vid.src = targetSrc;
          vid.load();
          vid.onloadeddata = () => {
            img.style.display = 'none';
            placeholder.style.display = 'none';
            vid.style.display = 'block';
            vid.currentTime = Math.max(0, initialFrame - 1) / 25;
            vid.onloadeddata = null;
          };
          vid.onerror = () => {
            vid.removeAttribute('src');
            vid.style.display = 'none';
            placeholder.style.display = 'none';
            img.src = keyframeUrl(c.video_id, frameIdx);
            img.style.display = 'block';
            img.onerror = () => {
              img.removeAttribute('src');
              img.style.display = 'none';
              placeholder.style.display = 'flex';
            };
          };
        });
    }
  } else {
    img.src = keyframeUrl(c.video_id, frameIdx);
    img.onload = () => {
      placeholder.style.display = 'none';
      img.style.display = 'block';
    };
    img.onerror = () => {
      img.removeAttribute('src');
      img.style.display = 'none';
      placeholder.style.display = 'flex';
    };
  }

  const scoresSection = $('private-detail-scores-section');
  const scoresBody = $('private-detail-scores-body');
  if (scoresSection && scoresBody) {
    scoresBody.innerHTML = '';
    const scores = c.scores || {};
    const keys = Object.keys(scores);
    if (keys.length) {
      keys.forEach((key) => {
        const row = document.createElement('div');
        row.className = 'score-row';
        row.innerHTML = `<span class="score-key">${key}</span><span class="score-val">${fmtScore(scores[key])}</span>`;
        scoresBody.appendChild(row);
      });
      scoresSection.style.display = 'block';
    } else {
      scoresSection.style.display = 'none';
    }
  }

  const evidenceSection = $('private-detail-evidence-section');
  const evidenceText = $('private-detail-evidence-text');
  if (evidenceSection && evidenceText) {
    const text = c.evidence?.caption || c.evidence?.text || '';
    if (text) {
      evidenceText.textContent = text;
      evidenceSection.style.display = 'block';
    } else {
      evidenceSection.style.display = 'none';
    }
  }

  const frameInput = $('private-frame-input');
  if (frameInput) {
    frameInput.value = initialFrame;
  }
  
  const rankBadge = $('private-detail-rank-badge');
  if (rankBadge) {
    rankBadge.textContent = c.rank ? `#${c.rank}` : '—';
  }
  
  const videoIdLabel = $('private-detail-video-id');
  if (videoIdLabel) {
    videoIdLabel.textContent = c.video_id;
  }

  const task = state.privateTask;
  const nEvents = currentPrivateTrakeEvents() || 1;
  const trakeContainer = $('private-frame-picker-trake-container');
  const singleRow = $('private-frame-picker-single-row');

  if (task === 'trake') {
    if (singleRow) singleRow.style.display = 'none';
    if (trakeContainer) {
      trakeContainer.replaceChildren();
      trakeContainer.style.display = 'flex';
      
      const saved = state.privateSelections.find(s => s.queryId === queryId && s.video_id === c.video_id);
      const savedFrames = saved && saved.frames ? saved.frames : [];
      
      for (let i = 1; i <= nEvents; i++) {
        const row = document.createElement('div');
        row.className = 'frame-picker-row';
        row.style.alignItems = 'center';
        
        const label = document.createElement('span');
        label.className = 'label';
        label.style.width = '60px';
        label.textContent = `Event ${i}:`;
        
        const input = document.createElement('input');
        input.type = 'number';
        input.className = 'private-trake-frame-input';
        input.dataset.eventIdx = i - 1;
        input.min = '1';
        
        let draftVal = state.privateCandidateDraftFrames[`${candidateKey(c)}__e${i-1}`];
        if (Number.isInteger(draftVal)) {
          input.value = draftVal;
        } else if (Number.isInteger(savedFrames[i-1])) {
          input.value = savedFrames[i-1];
        } else if (i === 1) {
          input.value = initialFrame;
          state.privateCandidateDraftFrames[`${candidateKey(c)}__e0`] = initialFrame;
        }
        
        input.addEventListener('input', (e) => {
          const val = parseInt(e.target.value, 10);
          if (!isNaN(val)) {
            state.privateCandidateDraftFrames[`${candidateKey(c)}__e${i-1}`] = val;
          }
        });
        
        const grabBtn = document.createElement('button');
        grabBtn.type = 'button';
        grabBtn.className = 'btn btn-secondary btn-sm';
        grabBtn.textContent = 'Grab';
        grabBtn.addEventListener('click', (e) => {
          e.preventDefault();
          const currentFrame = currentPrivateVideoFrame() ?? state.privateCurrentPlaybackFrame;
          if (Number.isInteger(currentFrame)) {
            input.value = currentFrame;
            state.privateCandidateDraftFrames[`${candidateKey(c)}__e${i-1}`] = currentFrame;
          }
        });
        
        row.append(label, input, grabBtn);
        trakeContainer.appendChild(row);
      }
    }
  } else {
    if (trakeContainer) {
      trakeContainer.replaceChildren();
      trakeContainer.style.display = 'none';
    }
    if (singleRow) singleRow.style.display = 'flex';
  }

  const answerInput = $('private-answer-input');
  if (answerInput) {
    if (task === 'qa') {
      const saved = state.privateSelections.find(s => s.queryId === queryId && s.video_id === c.video_id);
      answerInput.value = saved && saved.answer ? saved.answer : '';
    } else {
      answerInput.value = '';
    }
  }

  $('private-btn-confirm-selection').disabled = false;
  $('private-btn-confirm-shared').disabled = false;
}

function grabPrivateCurrentFrame(e) {
  if (e) e.preventDefault();
  if (state.privateSelected === null) return;
  const candidate = state.privateCandidates[state.privateSelected];
  if (!candidate) return;
  const frame = currentPrivateVideoFrame()
    ?? state.privateCurrentPlaybackFrame
    ?? submissionHelpers.candidateToSubmissionFrame(candidate);
  if (!Number.isInteger(frame)) return;
  $('private-frame-input').value = frame;
  state.privateCandidateDraftFrames[candidateKey(candidate)] = frame;
  const button = $('private-btn-grab-frame');
  if (button) {
    button.classList.add('captured');
    setTimeout(() => button.classList.remove('captured'), 700);
  }
}

function confirmPrivateSelection() {
  if (state.privateSelected === null) return;
  const c = state.privateCandidates[state.privateSelected];
  if (!c) return;
  const queryId = currentPrivateQueryId();
  const task = state.privateTask;
  
  let frame = parseInt($('private-frame-input')?.value || '0', 10);
  let frames = [frame];
  let answer = '';
  
  if (task === 'trake') {
    const inputs = document.querySelectorAll('.private-trake-frame-input');
    frames = Array.from(inputs).map(inp => parseInt(inp.value, 10));
    if (frames.some(isNaN)) {
      toast('Tất cả event frames của TRAKE phải hợp lệ', 'warning');
      return;
    }
  } else {
    if (isNaN(frame) || frame < 1) {
      toast('Frame không hợp lệ', 'warning');
      return;
    }
    if (task === 'qa') {
      answer = $('private-answer-input')?.value.trim() || '';
      if (!answer) {
        toast('Q&A cần câu trả lời', 'warning');
        return;
      }
    }
  }
  
  state.privateSelections = state.privateSelections.filter(s => !(s.queryId === queryId && s.video_id === c.video_id));
  state.privateSelections.push({
    queryId,
    video_id: c.video_id,
    frames,
    answer,
    task,
    rank: c.rank || 1
  });
  
  savePrivateSelections();
  renderPrivateSelectionsList();
  renderPrivateManifestList();
  toast('Đã xác nhận lựa chọn riêng', 'success');
}

function confirmPrivateToShared() {
  if (state.privateSelected === null) return;
  const c = state.privateCandidates[state.privateSelected];
  if (!c) return;
  const queryId = currentPrivateQueryId();
  const task = state.privateTask;
  
  let frame = parseInt($('private-frame-input')?.value || '0', 10);
  let frames = [frame];
  let answer = '';
  
  if (task === 'trake') {
    const inputs = document.querySelectorAll('.private-trake-frame-input');
    frames = Array.from(inputs).map(inp => parseInt(inp.value, 10));
    if (frames.some(isNaN)) {
      toast('Tất cả event frames của TRAKE phải hợp lệ', 'warning');
      return;
    }
  } else {
    if (isNaN(frame) || frame < 1) {
      toast('Frame không hợp lệ', 'warning');
      return;
    }
    if (task === 'qa') {
      answer = $('private-answer-input')?.value.trim() || '';
      if (!answer) {
        toast('Q&A cần câu trả lời', 'warning');
        return;
      }
    }
  }
  
  // Save to shared selections (will automatically call saveSelections() and sendWsUpdate())
  state.selections = state.selections.filter(s => !(s.queryId === queryId && s.video_id === c.video_id));
  state.selections.push({
    queryId,
    video_id: c.video_id,
    frames,
    answer,
    task,
    rank: c.rank || 1
  });
  
  saveSelections();
  renderSelectionsList();
  renderManifestList();
  renderExportTable();
  toast('Đã xác nhận lựa chọn lên không gian chung', 'success');
}

function savePrivateSelections() {
  localStorage.setItem('aic_private_selections', JSON.stringify(state.privateSelections));
}

function loadPrivateSelections() {
  const data = localStorage.getItem('aic_private_selections');
  if (data) {
    try {
      state.privateSelections = JSON.parse(data);
    } catch (e) {
      console.error('Failed to load private selections:', e);
      state.privateSelections = [];
    }
  }
}

function savePrivateQueryCache() {
  const serialized = {};
  for (const [qid, cache] of Object.entries(state.privateQueryCache)) {
    if (!cache) continue;
    serialized[qid] = Object.assign({}, cache, {
      iterExcluded: Array.from(cache.iterExcluded || []),
    });
  }
  try {
    localStorage.setItem('aic_private_query_cache', JSON.stringify(serialized));
  } catch (e) {
    console.error('Failed to write private cache:', e);
  }
}

function loadPrivateQueryCache() {
  const data = localStorage.getItem('aic_private_query_cache');
  if (data) {
    try {
      const parsed = JSON.parse(data);
      const deserialized = {};
      for (const [qid, cache] of Object.entries(parsed)) {
        if (!cache) continue;
        deserialized[qid] = Object.assign({}, cache, {
          iterExcluded: new Set(cache.iterExcluded || []),
        });
      }
      state.privateQueryCache = deserialized;
    } catch (e) {
      console.error('Failed to parse private query cache:', e);
      state.privateQueryCache = {};
    }
  } else {
    state.privateQueryCache = {};
  }
}

function saveCurrentPrivateQueryToCache() {
  const queryId = state.currentPrivateQueryId;
  if (!queryId) return;

  state.privateQueryCache[queryId] = {
    text_vi: $('private-query-input')?.value || '',
    translatedText: $('private-translated-text')?.value || '',
    candidates: state.privateCandidates || [],
    selected: state.privateSelected,
    candidateDraftFrames: Object.assign({}, state.privateCandidateDraftFrames),
    currentFps: state.privateCurrentFps,
    currentPlaybackFrame: state.privateCurrentPlaybackFrame,
  };
  savePrivateQueryCache();
}

function loadPrivateQueryFromCache(queryId, form) {
  const cached = state.privateQueryCache[queryId];
  if (cached) {
    if (cached.text_vi && $('private-query-input')) {
      $('private-query-input').value = cached.text_vi;
    }
    $('private-translated-text').value = cached.translatedText || (form && form.translatedText) || '';
    state.privateCandidates = cached.candidates || [];
    state.privateSelected = (cached.selected !== undefined && cached.selected !== null) ? cached.selected : null;
    state.privateCandidateDraftFrames = Object.assign({}, cached.candidateDraftFrames);
    state.privateCurrentFps = cached.currentFps;
    state.privateCurrentPlaybackFrame = cached.currentPlaybackFrame;
    
    renderPrivateCandidates();
    $('private-results-count').textContent = `${state.privateCandidates.length} candidates`;
    
    if (state.privateSelected !== null && state.privateSelected < state.privateCandidates.length) {
      selectPrivateCandidate(state.privateSelected);
    }
  } else {
    $('private-translated-text').value = (form && form.translatedText) || '';
    renderPrivateCandidates();
    $('private-results-count').textContent = '0 candidates';
  }
}

function clearPrivateQueryWorkspace() {
  const grid = $('private-candidates-grid');
  if (grid) {
    const empty = document.createElement('div');
    empty.className = 'empty-state';
    const message = document.createElement('p');
    message.textContent = 'Chưa tìm kiếm query này (Riêng)';
    empty.appendChild(message);
    grid.replaceChildren(empty);
  }
  const resultCount = $('private-results-count');
  if (resultCount) resultCount.textContent = '0 candidates';

  const image = $('private-preview-img');
  if (image) {
    image.onload = null;
    image.onerror = null;
    image.removeAttribute('src');
    image.style.display = 'none';
  }
  const video = $('private-preview-vid');
  if (video) {
    video.onloadeddata = null;
    video.onerror = null;
    if (typeof video.pause === 'function') video.pause();
    video.removeAttribute('src');
    if (typeof video.load === 'function') video.load();
    video.style.display = 'none';
  }
  const placeholder = $('private-preview-placeholder');
  if (placeholder) placeholder.style.display = 'flex';
  const rankBadge = $('private-detail-rank-badge');
  if (rankBadge) rankBadge.textContent = '—';
  const scoresSection = $('private-detail-scores-section');
  if (scoresSection) scoresSection.style.display = 'none';
  const scoresBody = $('private-detail-scores-body');
  if (scoresBody) scoresBody.replaceChildren();
  const evidenceSection = $('private-detail-evidence-section');
  if (evidenceSection) evidenceSection.style.display = 'none';
  const frameInput = $('private-frame-input');
  if (frameInput) frameInput.value = '';
  const playbackFrame = $('private-video-current-frame');
  if (playbackFrame) playbackFrame.textContent = '—';
  const answerInput = $('private-answer-input');
  if (answerInput) answerInput.value = '';
  const trakeContainer = $('private-frame-picker-trake-container');
  if (trakeContainer) {
    trakeContainer.replaceChildren();
    trakeContainer.style.display = 'none';
  }
  const confirmButton = $('private-btn-confirm-selection');
  if (confirmButton) confirmButton.disabled = true;
  const confirmSharedButton = $('private-btn-confirm-shared');
  if (confirmSharedButton) confirmSharedButton.disabled = true;
}

function renderPrivateSelectionsList() {
  const container = $('private-selections-list');
  if (!container) return;
  const queryId = currentPrivateQueryId();
  const querySelections = state.privateSelections.filter((s) => s.queryId === queryId);
  
  const badge = $('private-sel-count');
  if (badge) badge.textContent = String(querySelections.length);
  
  if (!querySelections.length) {
    container.innerHTML = `
      <div style="color:var(--text-muted);font-size:12px;text-align:center;padding:12px">
        Chưa có lựa chọn nào
      </div>`;
    return;
  }
  
  container.innerHTML = '';
  querySelections.forEach((s) => {
    const card = document.createElement('div');
    card.className = 'selection-card';
    const info = document.createElement('div');
    info.className = 'selection-info';
    
    const vid = document.createElement('div');
    vid.className = 'selection-vid';
    vid.textContent = s.video_id;
    
    const meta = document.createElement('div');
    meta.className = 'selection-meta';
    
    let details = `Frames: ${s.frames.join(', ')}`;
    if (s.task === 'qa' && s.answer) {
      details += ` | Ans: ${s.answer}`;
    }
    meta.textContent = details;
    info.append(vid, meta);
    
    const removeButton = document.createElement('button');
    removeButton.className = 'selection-del';
    removeButton.title = 'Xoá';
    removeButton.textContent = '✕';
    removeButton.addEventListener('click', (e) => {
      e.stopPropagation();
      const absoluteIdx = state.privateSelections.indexOf(s);
      if (absoluteIdx !== -1) {
        state.privateSelections.splice(absoluteIdx, 1);
        savePrivateSelections();
        renderPrivateSelectionsList();
        renderPrivateManifestList();
        toast('Đã xoá lựa chọn riêng', 'info');
      }
    });
    
    card.append(info, removeButton);
    container.appendChild(card);
  });
}

function renderPrivateManifestList() {
  const list = $('private-query-manifest-list');
  if (!list) return;
  list.replaceChildren();
  state.manifest.forEach((item) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `query-manifest-item${item.query_id === state.currentPrivateQueryId ? ' active' : ''}`;
    const queryId = document.createElement('span');
    queryId.className = 'manifest-query-id';
    queryId.textContent = item.query_id;
    const task = document.createElement('span');
    task.className = 'manifest-task';
    task.textContent = item.task.toUpperCase();

    const deleteBtn = document.createElement('span');
    deleteBtn.className = 'manifest-delete-btn';
    deleteBtn.title = `Xóa query ${item.query_id}`;
    deleteBtn.textContent = '×';
    deleteBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      e.preventDefault();
      deleteQuery(item.query_id);
    });
    
    const readiness = submissionHelpers.manifestQueryReadiness(item, state.privateSelections);
    const status = document.createElement('span');
    status.className = `manifest-readiness ${readiness.ready ? 'ready' : 'not-ready'}`;
    status.textContent = readiness.ready ? 'Ready' : readiness.label;
    
    button.append(queryId, deleteBtn, task, status);
    button.addEventListener('click', () => selectPrivateManifestQuery(item.query_id));
    list.appendChild(button);
  });
}

function currentPrivateVideoFrame() {
  const video = $('private-preview-vid');
  if (!video || video.style.display === 'none') return null;
  const fps = state.privateCurrentFps || 25;
  return Math.floor(video.currentTime * fps) + 1;
}

function updatePrivatePlaybackFrame() {
  const frame = currentPrivateVideoFrame();
  if (!Number.isInteger(frame)) return;
  state.privateCurrentPlaybackFrame = frame;
  const indicator = $('private-video-current-frame');
  if (indicator) indicator.textContent = String(frame);
}

// ===========================================================================
// THÊM ĐÁP ÁN THỦ CÔNG (MANUAL ENTRY TAB)
// ===========================================================================

async function fetchVideosList() {
  try {
    const res = await fetch('/api/videos');
    const data = await res.json();
    if (data.ok) {
      state.videos = data.videos || [];
      populateManualVideoSelect();
    }
  } catch (e) {
    console.error('Failed to fetch videos list:', e);
  }
}

function populateManualVideoSelect() {
  const select = $('manual-video-select');
  if (!select) return;
  select.innerHTML = '<option value="">-- Chọn Video ID --</option>';
  (state.videos || []).forEach((vid) => {
    const opt = document.createElement('option');
    opt.value = vid;
    opt.textContent = vid;
    select.appendChild(opt);
  });
}

function populateManualQuerySelect() {
  const select = $('manual-query-select');
  if (!select) return;
  const currentVal = select.value;
  select.innerHTML = '<option value="">-- Chọn Query ID --</option>';
  state.manifest.forEach((item) => {
    const opt = document.createElement('option');
    opt.value = item.query_id;
    opt.textContent = `${item.query_id} (${item.task.toUpperCase()})`;
    select.appendChild(opt);
  });
  if (currentVal && state.manifest.some(item => item.query_id === currentVal)) {
    select.value = currentVal;
  }
}

function onManualQueryChange() {
  const select = $('manual-query-select');
  const infoSection = $('manual-query-info');
  const saveBtn = $('manual-btn-save');
  const qaContainer = $('manual-qa-answer-container');
  const singleFrameSection = $('manual-frame-single-section');
  const trakeFrameSection = $('manual-frame-trake-section');
  const trakeEventsSection = $('manual-trake-events-section');

  if (!select || !select.value) {
    if (infoSection) infoSection.style.display = 'none';
    if (saveBtn) saveBtn.disabled = true;
    if (qaContainer) qaContainer.style.display = 'none';
    if (trakeEventsSection) trakeEventsSection.style.display = 'none';
    return;
  }

  const queryId = select.value;
  const item = state.manifest.find(q => q.query_id === queryId);
  if (!item) return;

  if (infoSection) {
    $('manual-query-task').textContent = item.task;
    $('manual-query-text').textContent = item.text;
    infoSection.style.display = 'block';
  }

  if (saveBtn) saveBtn.disabled = false;

  // QA specific fields
  if (qaContainer) {
    qaContainer.style.display = item.task === 'qa' ? 'flex' : 'none';
  }

  // TRAKE events count section in sidebar
  if (trakeEventsSection) {
    trakeEventsSection.style.display = item.task === 'trake' ? 'block' : 'none';
  }

  // Frame picker fields
  if (item.task === 'trake') {
    if (singleFrameSection) singleFrameSection.style.display = 'none';
    if (trakeFrameSection) {
      trakeFrameSection.style.display = 'flex';
      const container = $('manual-frame-picker-trake-container');
      container.innerHTML = '';
      
      // Get events count from input
      let nEvents = item.n_events || 3;
      const nEventsInput = $('manual-n-events-input');
      if (nEventsInput) {
        nEvents = parseInt(nEventsInput.value, 10) || nEvents;
      }
      
      for (let i = 1; i <= nEvents; i++) {
        const row = document.createElement('div');
        row.className = 'frame-picker-row';
        row.style.alignItems = 'center';
        row.style.marginBottom = '6px';
        
        const label = document.createElement('span');
        label.className = 'label';
        label.style.width = '60px';
        label.textContent = `Event ${i}:`;
        
        const input = document.createElement('input');
        input.type = 'number';
        input.className = 'manual-trake-frame-input';
        input.id = `manual-trake-frame-input-${i}`;
        input.dataset.eventIdx = i - 1;
        input.min = '1';
        input.style.flex = '1';
        
        // Update playhead when input changes
        input.addEventListener('input', (e) => {
          const val = parseInt(e.target.value, 10);
          const vid = $('manual-preview-vid');
          if (!isNaN(val) && vid && vid.style.display !== 'none') {
            const fps = state.manualCurrentFps || 25;
            vid.currentTime = Math.max(0, val - 1) / fps;
          }
        });

        const grabBtn = document.createElement('button');
        grabBtn.type = 'button';
        grabBtn.className = 'btn btn-secondary btn-sm';
        grabBtn.textContent = 'Get';
        grabBtn.addEventListener('click', (e) => {
          e.preventDefault();
          const currentFrame = currentManualVideoFrame();
          if (Number.isInteger(currentFrame)) {
            input.value = currentFrame;
          }
        });
        
        row.append(label, input, grabBtn);
        container.appendChild(row);
      }
    }
  } else {
    if (trakeFrameSection) trakeFrameSection.style.display = 'none';
    if (singleFrameSection) singleFrameSection.style.display = 'block';
  }

  renderManualSelections();
}

function onManualVideoChange() {
  const select = $('manual-video-select');
  const vid = $('manual-preview-vid');
  const placeholder = $('manual-preview-placeholder');
  
  if (!select || !select.value) {
    if (vid) {
      vid.removeAttribute('src');
      if (typeof vid.load === 'function') vid.load();
      vid.style.display = 'none';
    }
    if (placeholder) placeholder.style.display = 'flex';
    return;
  }

  const videoId = select.value;
  state.manualCurrentFps = null;
  if (placeholder) placeholder.style.display = 'none';
  if (vid) {
    vid.style.display = 'block';
    const targetSrc = `/api/video/${videoId}`;
    vid.src = targetSrc;
    vid.load();
    
    fetch(`/api/video_info/${videoId}`)
      .then(res => res.json())
      .then(data => {
        state.manualCurrentFps = data.fps;
        vid.currentTime = 0;
      })
      .catch(() => {
        state.manualCurrentFps = 25;
        vid.currentTime = 0;
      });
  }
}

function currentManualVideoFrame() {
  const video = $('manual-preview-vid');
  if (!video || video.style.display === 'none') return null;
  const fps = state.manualCurrentFps || 25;
  return Math.floor(video.currentTime * fps) + 1;
}

function updateManualPlaybackFrame() {
  const frame = currentManualVideoFrame();
  if (!Number.isInteger(frame)) return;
  state.manualCurrentPlaybackFrame = frame;
  const indicator = $('manual-video-current-frame');
  if (indicator) indicator.textContent = String(frame);
  
  const trakeIndicator = $('manual-video-trake-current-frame');
  if (trakeIndicator) trakeIndicator.textContent = String(frame);
}

function grabManualCurrentFrame(e) {
  if (e) e.preventDefault();
  const currentFrame = currentManualVideoFrame();
  if (Number.isInteger(currentFrame)) {
    $('manual-frame-input').value = currentFrame;
  }
}

function confirmManualSelection() {
  const querySelect = $('manual-query-select');
  const videoSelect = $('manual-video-select');
  if (!querySelect || !querySelect.value) {
    toast('Vui lòng chọn Query ID', 'warning');
    return;
  }
  if (!videoSelect || !videoSelect.value) {
    toast('Vui lòng chọn Video ID', 'warning');
    return;
  }

  const queryId = querySelect.value;
  const videoId = videoSelect.value;
  const item = state.manifest.find(q => q.query_id === queryId);
  if (!item) return;

  if (item.task === 'trake') {
    let nEvents = item.n_events || 3;
    const nEventsInput = $('manual-n-events-input');
    if (nEventsInput) {
      nEvents = parseInt(nEventsInput.value, 10) || nEvents;
    }
    const frames = [];
    let hasMissing = false;
    for (let i = 1; i <= nEvents; i++) {
      const inp = $(`manual-trake-frame-input-${i}`);
      const val = inp ? parseInt(inp.value, 10) : NaN;
      if (isNaN(val) || val < 1) {
        hasMissing = true;
      }
      frames.push(val);
    }
    if (hasMissing) {
      toast(`Vui lòng điền đủ ${nEvents} sự kiện cho TRAKE!`, 'warning');
      return;
    }
    
    const isDup = state.selections.some(s => s.queryId === queryId && s.video_id === videoId && JSON.stringify(s.frames) === JSON.stringify(frames));
    if (isDup) {
      toast('Lựa chọn này đã tồn tại trong danh sách!', 'warning');
      return;
    }
    
    state.selections.push({
      queryId,
      task: 'trake',
      video_id: videoId,
      frames,
      answer: '',
      rank: state.selections.filter(s => s.queryId === queryId).length + 1
    });
    toast(`Đã thêm ${videoId} (${nEvents} events) vào TRAKE`, 'success');
  } 
  else if (item.task === 'qa') {
    const frame = parseInt($('manual-frame-input')?.value || '0', 10);
    if (isNaN(frame) || frame < 1) {
      toast('Frame index không hợp lệ', 'warning');
      return;
    }
    const answer = submissionHelpers.prepareQaAnswer($('manual-answer-input')?.value || '');
    if (!answer.trim()) {
      toast('Vui lòng nhập câu trả lời cho Q&A!', 'warning');
      return;
    }
    if (submissionHelpers.unicodeCodePointLength(answer) > 100) {
      toast('Câu trả lời vượt quá giới hạn 100 ký tự!', 'error');
      return;
    }
    
    const isDup = state.selections.some(s => s.queryId === queryId && s.video_id === videoId && s.frames[0] === frame && s.answer === answer);
    if (isDup) {
      toast('Lựa chọn này đã tồn tại trong danh sách!', 'warning');
      return;
    }
    
    state.selections.push({
      queryId,
      task: 'qa',
      video_id: videoId,
      frames: [frame],
      answer,
      rank: state.selections.filter(s => s.queryId === queryId).length + 1
    });
    toast(`Đã thêm ${videoId} f${frame} vào Q&A`, 'success');
  } 
  else { // kis
    const frame = parseInt($('manual-frame-input')?.value || '0', 10);
    if (isNaN(frame) || frame < 1) {
      toast('Frame index không hợp lệ', 'warning');
      return;
    }
    
    const isDup = state.selections.some(s => s.queryId === queryId && s.video_id === videoId && s.frames[0] === frame);
    if (isDup) {
      toast('Lựa chọn này đã tồn tại trong danh sách!', 'warning');
      return;
    }
    
    state.selections.push({
      queryId,
      task: 'kis',
      video_id: videoId,
      frames: [frame],
      answer: '',
      rank: state.selections.filter(s => s.queryId === queryId).length + 1
    });
    toast(`Đã thêm ${videoId} f${frame} vào KIS`, 'success');
  }

  saveSelections();
  renderSelectionsList();
  renderManifestList();
  renderExportTable();
  renderManualSelections();
  
  toast('Đã thêm đáp án thủ công vào không gian chung', 'success');
}

function renderManualSelections() {
  const container = $('manual-selections-list');
  const countBadge = $('manual-sel-count');
  if (!container) return;

  const queryId = $('manual-query-select')?.value;
  if (!queryId) {
    container.innerHTML = `
      <div style="color:var(--text-muted);font-size:12px;text-align:center;padding:12px">
        Chọn query để xem danh sách đáp án
      </div>`;
    if (countBadge) countBadge.textContent = '0';
    return;
  }

  const querySelections = state.selections.filter(s => s.queryId === queryId);
  if (countBadge) countBadge.textContent = String(querySelections.length);

  if (!querySelections.length) {
    container.innerHTML = `
      <div style="color:var(--text-muted);font-size:12px;text-align:center;padding:12px">
        Chưa có lựa chọn nào cho query này
      </div>`;
    return;
  }

  container.innerHTML = '';
  querySelections.forEach((s) => {
    const card = document.createElement('div');
    card.className = 'selection-card';
    const info = document.createElement('div');
    info.className = 'selection-info';
    
    const vid = document.createElement('div');
    vid.className = 'selection-vid';
    vid.textContent = s.video_id;
    
    const meta = document.createElement('div');
    meta.className = 'selection-meta';
    
    let details = `Frames: ${s.frames.join(', ')}`;
    if (s.task === 'qa' && s.answer) {
      details += ` | Ans: ${s.answer}`;
    }
    meta.textContent = details;
    info.append(vid, meta);
    
    const removeButton = document.createElement('button');
    removeButton.className = 'selection-del';
    removeButton.title = 'Xoá';
    removeButton.textContent = '✕';
    removeButton.addEventListener('click', (e) => {
      e.stopPropagation();
      const absoluteIdx = state.selections.indexOf(s);
      if (absoluteIdx !== -1) {
        state.selections.splice(absoluteIdx, 1);
        saveSelections();
        renderSelectionsList();
        renderManifestList();
        renderExportTable();
        renderManualSelections();
        toast('Đã xoá đáp án', 'info');
      }
    });
    
    card.append(info, removeButton);
    container.appendChild(card);
  });
}

