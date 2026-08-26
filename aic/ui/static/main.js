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
};

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

function $(id) { return document.getElementById(id); }
const submissionHelpers = window.AICSubmissionHelpers;

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
  return Math.round(video.currentTime * fps) + 1;
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
    const readiness = submissionHelpers.manifestQueryReadiness(item, state.selections);
    const status = document.createElement('span');
    status.className = `manifest-readiness ${readiness.ready ? 'ready' : 'not-ready'}`;
    status.textContent = readiness.ready ? '✓ Ready' : `⚠ ${readiness.label}`;
    button.dataset.readiness = readiness.codes.join(',') || 'ready';
    button.append(queryId, task, status);
    button.addEventListener('click', () => selectManifestQuery(item.query_id));
    list.appendChild(button);
  });
}

function selectManifestQuery(queryId) {
  const item = state.manifest.find((entry) => entry.query_id === queryId);
  if (!item) return;
  const form = submissionHelpers.manifestQueryFormState(item);
  clearQueryWorkspace();
  state.currentQueryId = form.queryId;
  $('query-id-input').value = form.queryId;
  $('export-query-id').value = form.queryId;
  $('query-input').value = form.text;
  $('translated-text').value = form.translatedText;
  $('n-events-input').value = form.nEvents;
  $('trake-events-confirmed').checked = form.eventsConfirmed;
  selectTask(form.task);
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

async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.ok) {
      $('status-text').textContent = data.retriever;
      $('status-dot').style.background = 'var(--green)';
      $('status-dot').style.boxShadow = '0 0 6px var(--green)';
      $('stat-keyframes').textContent = data.retriever === 'dummy' ? 'demo' : '—';
    }
  } catch {
    $('status-text').textContent = 'Offline';
    $('status-dot').style.background = 'var(--red)';
    $('status-dot').style.boxShadow = '0 0 6px var(--red)';
  }
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
  setLoading('btn-translate', true);
  try {
    const res = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text_vi }),
    });
    const data = await res.json();
    $('translated-text').value = data.text_en || '';
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
        <div class="card-frame">frame ${frameIdx} &mdash; ${c.start_frame}&rarr;${c.end_frame}</div>
      </div>
      <div class="verdict-row">
        <button class="verdict-btn v-matched ${verdict === 'matched' ? 'active' : ''}"
          onclick="setVerdict(${idx},'matched',event)" title="Matched">✓</button>
        <button class="verdict-btn v-not ${verdict === 'not_matched' ? 'active' : ''}"
          onclick="setVerdict(${idx},'not_matched',event)" title="Not matched">✗</button>
        <button class="verdict-btn v-unsure ${verdict === 'unsure' ? 'active' : ''}"
          onclick="setVerdict(${idx},'unsure',event)" title="Unsure">?</button>
      </div>`;

    card.addEventListener('click', (e) => {
      if (e.target.closest('.verdict-btn')) return;
      selectCandidate(idx);
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

function selectCandidate(idx) {
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
  const initialFrame = Number.isInteger(draftFrame) ? draftFrame : frameIdx;
  state.currentPlaybackFrame = initialFrame;
  const playbackFrame = $('video-current-frame');
  if (playbackFrame) playbackFrame.textContent = String(initialFrame);

  // Confirmed rows remain independent from the candidate's editable draft.
  const existing = state.selections.find(s => s.queryId === queryId && s.video_id === c.video_id);
  
  if (vid) {
    // Hiển thị Video, fallback sang keyframe nếu video lỗi
    vid.src = `/api/video/${c.video_id}`;
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
  if (evidenceSection) evidenceSection.style.display = 'none';

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
    frameVal = Math.round(vid.currentTime * fps) + 1;
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
      infoText = `${s.video_id} <span style="color:var(--text-muted)">f${s.frames[0]}</span><br><span style="font-size:11px;color:var(--text-secondary)">"${truncateAns}"</span>`;
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
  if (confirm(`Bạn có chắc chắn muốn xoá toàn bộ lựa chọn cho query ${queryId}?`)) {
    state.selections = state.selections.filter(s => s.queryId !== queryId);
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
          <div style="font-size:13px; color:var(--text-primary); font-weight:500; background:rgba(255,255,255,0.03); border:1px solid var(--border); padding:6px 10px; border-radius:var(--radius-sm); white-space:pre-wrap;">${s.answer}</div>
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
    let statusText = q.selections.length ? '✓ Ready' : '⚠️ Chưa có dòng';
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
        statusText = '⚠️ Thiếu event';
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
    reviewButton.textContent = '👁️ Xem';
    reviewButton.addEventListener('click', () => showExportReview(q.queryId));
    const removeButton = document.createElement('button');
    removeButton.type = 'button';
    removeButton.className = 'btn-translate';
    removeButton.style.padding = '2px 8px';
    removeButton.style.fontSize = '11px';
    removeButton.style.color = 'var(--red)';
    removeButton.style.borderColor = 'rgba(239,68,68,0.25)';
    removeButton.style.background = 'rgba(239,68,68,0.1)';
    removeButton.textContent = '✕ Xoá';
    removeButton.addEventListener('click', () => removeQuerySelections(q.queryId));
    actions.append(reviewButton, removeButton);
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
  if (!state.manifest.length) { toast('Nạp query pack trước khi export', 'warning'); return; }
  const manifestIds = new Set(state.manifest.map((item) => item.query_id));
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
      body: JSON.stringify({ manifest: state.manifest, rows }),
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
  loadSelections();
  loadManifest();
  checkStatus();
  setInterval(checkStatus, 30000);
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
