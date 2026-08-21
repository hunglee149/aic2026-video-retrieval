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
  gridMode: true,

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
  state.task = task;
  ['kis', 'qa', 'trake'].forEach(t => {
    $(`pill-${t}`).classList.toggle('active', t === task);
  });
  $('n-events-section').style.display = task === 'trake' ? '' : 'none';
  $('answer-section').style.display = task === 'qa' ? '' : 'none';
  $('results-task-badge').textContent = task.toUpperCase();
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
  const query_id = $('query-id-input').value.trim() || 'q1';
  const k = parseInt($('topk-slider').value, 10);
  const n_events = parseInt($('n-events-input').value, 10) || 1;

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
    const kfUrl = keyframeUrl(c.video_id, frameIdx);
    card.innerHTML = `
      <div class="card-thumb" ondblclick="openLightbox('${kfUrl}', '${c.video_id} (Frame #${frameIdx})')" title="Nhấp đúp để phóng to ảnh">
        <img src="${kfUrl}" alt="${c.video_id}" loading="lazy"/>
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

function selectCandidate(idx) {
  state.selected = idx;
  const c = state.candidates[idx];

  document.querySelectorAll('.candidate-card').forEach((el, i) => {
    el.classList.toggle('selected', i === idx);
  });

  const frameIdx = c.representative_frames[0] ?? c.start_frame;
  const img = $('preview-img');
  const placeholder = $('preview-placeholder');
  img.style.display = 'none';
  placeholder.style.display = 'flex';
  img.onload = () => { img.style.display = 'block'; placeholder.style.display = 'none'; };
  img.onerror = () => { img.style.display = 'none'; placeholder.style.display = 'flex'; };
  img.src = keyframeUrl(c.video_id, frameIdx);
  img.alt = c.video_id;

  $('detail-rank-badge').textContent = `#${c.rank}`;

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
  if (c.evidence && (c.evidence.transcript_match || Object.keys(c.evidence).length)) {
    evidenceSection.style.display = '';
    const textEl = $('detail-evidence-text');
    if (c.evidence.transcript_match) {
      textEl.innerHTML = `
        <div class="transcript-box">
          <div class="transcript-label">📝 Khớp nội dung ASR / OCR / Caption:</div>
          <div class="transcript-content">"${c.evidence.transcript_match}"</div>
        </div>`;
    } else {
      textEl.textContent = JSON.stringify(c.evidence, null, 2);
    }
  } else {
    evidenceSection.style.display = 'none';
  }

  $('frame-input').value = frameIdx;
  $('btn-confirm-selection').disabled = false;

  // Tải danh sách toàn bộ keyframes của video lên Timeline Gallery
  loadVideoTimeline(c.video_id, frameIdx);
}

// ---------------------------------------------------------------------------
// Video Timeline Keyframes Browser
// ---------------------------------------------------------------------------

state.videoKeyframes = [];
state.currentTimelineIdx = 0;
state.currentVideoId = null;

async function loadVideoTimeline(videoId, activeFrame) {
  const section = $('detail-timeline-section');
  const strip = $('timeline-strip');
  const badge = $('timeline-count-badge');
  const slider = $('timeline-slider');
  const info = $('timeline-current-info');
  if (!section || !strip) return;

  section.style.display = '';
  strip.innerHTML = '<div style="color:var(--text-muted);font-size:11px;padding:8px">Đang tải keyframes...</div>';
  state.currentVideoId = videoId;

  try {
    const res = await fetch(`/api/video_keyframes/${encodeURIComponent(videoId)}`);
    const data = await res.json();
    if (!data.ok || !data.keyframes || !data.keyframes.length) {
      section.style.display = 'none';
      return;
    }

    state.videoKeyframes = data.keyframes;
    badge.textContent = `${data.keyframes.length} frames`;
    strip.innerHTML = '';

    if (slider) {
      slider.min = 0;
      slider.max = data.keyframes.length - 1;
    }

    let activeIdx = 0;
    data.keyframes.forEach((kf, idx) => {
      const kfIdx = kf.frame_idx ?? kf.kf_num;
      const kfNum = kf.kf_num ?? kfIdx;
      const isActive = (kfNum === activeFrame || kfIdx === activeFrame);
      if (isActive) activeIdx = idx;

      const item = document.createElement('div');
      item.className = `timeline-item ${isActive ? 'active' : ''}`;
      item.dataset.index = idx;
      const timeStr = (kf.pts_time !== null && kf.pts_time !== undefined) ? `${Number(kf.pts_time).toFixed(1)}s` : `#${kfNum}`;
      item.title = `Keyframe #${kfNum} | Frame idx: ${kfIdx} (${timeStr})`;
      
      const imgUrl = keyframeUrl(videoId, kfNum);
      item.innerHTML = `
        <img src="${imgUrl}" alt="f${kfNum}" loading="lazy"/>
        <div class="timeline-item-label">${timeStr}</div>
      `;

      item.addEventListener('click', () => {
        selectTimelineKeyframe(idx);
      });

      strip.appendChild(item);
    });

    state.currentTimelineIdx = activeIdx;
    if (slider) slider.value = activeIdx;
    updateTimelineInfo(activeIdx);

    // Tự động cuộn đến keyframe đang chọn
    setTimeout(() => {
      const activeEl = strip.querySelector('.timeline-item.active');
      if (activeEl) {
        activeEl.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
      }
    }, 100);
  } catch (e) {
    section.style.display = 'none';
  }
}

function selectTimelineKeyframe(idx) {
  if (!state.videoKeyframes || idx < 0 || idx >= state.videoKeyframes.length) return;
  state.currentTimelineIdx = idx;
  const kf = state.videoKeyframes[idx];
  const kfIdx = kf.frame_idx ?? kf.kf_num;
  const kfNum = kf.kf_num ?? kfIdx;

  const strip = $('timeline-strip');
  if (strip) {
    strip.querySelectorAll('.timeline-item').forEach((el, i) => {
      el.classList.toggle('active', i === idx);
    });
    const activeEl = strip.querySelector(`.timeline-item[data-index="${idx}"]`);
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }
  }

  const slider = $('timeline-slider');
  if (slider) slider.value = idx;

  const img = $('preview-img');
  if (img && state.currentVideoId) {
    img.src = keyframeUrl(state.currentVideoId, kfNum);
    img.alt = state.currentVideoId;
  }
  $('frame-input').value = kfIdx;
  updateTimelineInfo(idx);
}

function updateTimelineInfo(idx) {
  const info = $('timeline-current-info');
  if (!info || !state.videoKeyframes[idx]) return;
  const kf = state.videoKeyframes[idx];
  const kfNum = kf.kf_num ?? kf.frame_idx;
  const timeStr = (kf.pts_time !== null && kf.pts_time !== undefined) ? `${Number(kf.pts_time).toFixed(1)}s` : `#${kfNum}`;
  info.textContent = `#${kfNum} (${timeStr})`;
}

function stepTimeline(delta) {
  if (!state.videoKeyframes.length) return;
  const newIdx = Math.max(0, Math.min(state.videoKeyframes.length - 1, state.currentTimelineIdx + delta));
  selectTimelineKeyframe(newIdx);
}

function onTimelineSliderInput(val) {
  selectTimelineKeyframe(parseInt(val, 10));
}

function toggleTimelineGrid() {
  const strip = $('timeline-strip');
  const btn = $('btn-timeline-grid');
  if (!strip) return;
  const isGrid = strip.classList.toggle('grid-mode');
  if (btn) btn.textContent = isGrid ? '═ Dải ngang' : '⊞ Lưới';
}

// Phím tắt mũi tên trái/phải để tua frame
document.addEventListener('keydown', (e) => {
  if (document.activeElement && ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    stepTimeline(-1);
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    stepTimeline(1);
  }
});

// ---------------------------------------------------------------------------
// Lightbox Fullscreen Viewer
// ---------------------------------------------------------------------------

function openLightbox(src, title) {
  closeLightbox();
  const modal = document.createElement('div');
  modal.id = 'active-lightbox';
  modal.className = 'lightbox-modal';
  modal.innerHTML = `
    <div class="lightbox-content" onclick="event.stopPropagation()">
      <button class="lightbox-close" onclick="closeLightbox()" title="Đóng (ESC)">✕</button>
      <img src="${src}" class="lightbox-img" alt="${title}"/>
      <div class="lightbox-info">${title}</div>
    </div>`;
  modal.addEventListener('click', closeLightbox);
  document.body.appendChild(modal);
}

function closeLightbox() {
  const el = $('active-lightbox');
  if (el) el.remove();
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeLightbox();
});

// Gắn sự kiện click zoom cho ảnh preview bên phải
document.addEventListener('DOMContentLoaded', () => {
  const previewWrap = $('preview-img-wrap');
  if (previewWrap) {
    previewWrap.addEventListener('click', () => {
      const img = $('preview-img');
      if (img && img.src && img.style.display !== 'none') {
        openLightbox(img.src, `${img.alt} (Frame #${$('frame-input').value || 0})`);
      }
    });
  }
});

// ---------------------------------------------------------------------------
// Confirm selection
// ---------------------------------------------------------------------------

function confirmSelection() {
  if (state.selected === null) return;
  const c = state.candidates[state.selected];
  const frameInput = parseInt($('frame-input').value, 10);
  const frame = isNaN(frameInput) ? (c.representative_frames[0] ?? c.start_frame) : frameInput;
  const answer = $('answer-input').value.trim();
  const queryId = $('query-id-input').value.trim() || 'q1';

  const existing = state.selections.findIndex(s => s.video_id === c.video_id && s.queryId === queryId);
  if (existing >= 0) state.selections.splice(existing, 1);

  state.selections.push({ video_id: c.video_id, frames: [frame], answer, queryId, task: state.task, rank: c.rank });
  renderSelectionsList();
  toast(`Đã thêm ${c.video_id} frame ${frame}`, 'success');
}

function removeSelection(idx) {
  state.selections.splice(idx, 1);
  renderSelectionsList();
}

function renderSelectionsList() {
  const list = $('selections-list');
  $('sel-count').textContent = state.selections.length;
  if (!state.selections.length) {
    list.innerHTML = '<div style="color:var(--text-muted);font-size:12px;text-align:center;padding:12px">Chưa có lựa chọn nào</div>';
    return;
  }
  list.innerHTML = state.selections.map((s, i) => `
    <div class="selection-item">
      <div class="selection-rank">${s.rank || i + 1}</div>
      <div class="selection-info">${s.video_id}<br><span style="color:var(--text-muted)">f${s.frames[0]}</span></div>
      <button class="selection-del" onclick="removeSelection(${i})" title="Xoá">✕</button>
    </div>`).join('');
}

// ---------------------------------------------------------------------------
// Iterative retrieval
// ---------------------------------------------------------------------------

function iterStart() {
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

  const queryId = $('query-id-input').value.trim() || 'q1';
  state.iterMatchedList.forEach(c => {
    const frame = c.representative_frames[0] ?? c.start_frame;
    const existing = state.selections.findIndex(s => s.video_id === c.video_id && s.queryId === queryId);
    if (existing < 0) {
      state.selections.push({ video_id: c.video_id, frames: [frame], answer: '', queryId, task: state.task, rank: c.rank });
    }
  });
  renderSelectionsList();
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

function renderExportTable() {
  const tbody = $('export-tbody');
  const count = $('export-query-count');

  if (!state.selections.length) {
    tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:32px">Chưa có kết quả. Hãy tìm kiếm và xác nhận lựa chọn trước.</td></tr>';
    count.textContent = '0 queries';
    refreshPreview();
    return;
  }

  const byQuery = {};
  state.selections.forEach(s => { if (!byQuery[s.queryId]) byQuery[s.queryId] = []; byQuery[s.queryId].push(s); });
  const queryIds = Object.keys(byQuery);
  count.textContent = `${queryIds.length} quer${queryIds.length !== 1 ? 'ies' : 'y'}`;

  tbody.innerHTML = state.selections.map((s, i) => `
    <tr>
      <td style="font-family:'JetBrains Mono',monospace;font-size:12px">${s.queryId}</td>
      <td><span class="badge badge-purple">${s.task.toUpperCase()}</span></td>
      <td style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cyan)">${s.video_id}</td>
      <td style="font-family:'JetBrains Mono',monospace;font-size:12px">${s.frames.join(', ')}</td>
      <td>
        <span class="badge badge-green">✓ Ready</span>
        <button class="btn-translate" style="margin-left:6px;font-size:10px;padding:2px 8px" onclick="removeExportRow(${i})">✕</button>
      </td>
    </tr>`).join('');

  refreshPreview();
}

function removeExportRow(idx) {
  state.selections.splice(idx, 1);
  renderSelectionsList();
  renderExportTable();
}

function refreshPreview() {
  const preview = $('csv-preview');
  if (!state.selections.length) { preview.textContent = '— chưa có dữ liệu —'; return; }
  const lines = state.selections.map(s => {
    const vid = s.video_id.replace(/\.mp4$/, '');
    const parts = [vid, ...s.frames.map(String)];
    if (s.answer) parts.push(s.answer);
    return parts.join(',');
  });
  preview.textContent = lines.join('\n');
}

async function doExport() {
  if (!state.selections.length) { toast('Chưa có lựa chọn để export', 'warning'); return; }
  const exportQueryId = $('export-query-id').value.trim() || $('query-id-input').value.trim() || 'q1';
  const rows = state.selections.map(s => ({
    video_id: s.video_id,
    frames: s.frames,
    answer: s.answer || '',
    query_id: s.queryId || exportQueryId,
  }));

  setLoading('btn-export', true);
  try {
    const res = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query_id: exportQueryId, task: state.task, rows }),
    });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Export failed'); }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'submission.zip';
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    toast('Đã tải xuống submission.zip đúng chuẩn BTC!', 'success');
  } catch (e) {
    toast(e.message, 'error');
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

function initPanelResizer() {
  const resizer = $('panel-resizer');
  const panel = $('detail-panel');
  if (!resizer || !panel) return;

  const savedWidth = localStorage.getItem('detail_panel_width');
  if (savedWidth) {
    panel.style.width = `${savedWidth}px`;
  }

  let isDragging = false;
  let startX = 0;
  let startWidth = 0;

  resizer.addEventListener('mousedown', (e) => {
    isDragging = true;
    startX = e.clientX;
    startWidth = panel.getBoundingClientRect().width;
    resizer.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const deltaX = startX - e.clientX; // Kéo sang trái -> tăng width
    const newWidth = Math.max(260, Math.min(window.innerWidth * 0.65, startWidth + deltaX));
    panel.style.width = `${newWidth}px`;
  });

  document.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      resizer.classList.remove('dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      const w = parseInt(panel.style.width, 10);
      if (w) localStorage.setItem('detail_panel_width', w);
    }
  });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  checkStatus();
  setInterval(checkStatus, 30000);
  selectTask('kis');
  initPanelResizer();
  $('query-id-input').addEventListener('input', () => {
    $('export-query-id').value = $('query-id-input').value;
  });
});
