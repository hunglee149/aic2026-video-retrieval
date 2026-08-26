(function (root, factory) {
  const helpers = factory();
  if (typeof module === 'object' && module.exports) module.exports = helpers;
  root.AICSubmissionHelpers = helpers;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  function prepareQaAnswer(answer) {
    return String(answer ?? '');
  }

  function unicodeCodePointLength(value) {
    return Array.from(String(value ?? '')).length;
  }

  function candidateToSubmissionFrame(candidate) {
    const frame = candidate?.representative_frames?.[0] ?? candidate?.start_frame;
    return Number.isInteger(frame) ? frame + 1 : null;
  }

  function canUseIterative(task) {
    return task === 'kis';
  }

  function inferTaskFromFilename(filename) {
    const match = String(filename || '').match(/[-_](kis|qa|trake)\.txt$/i);
    return match ? match[1].toLowerCase() : null;
  }

  function upsertManifestItem(manifest, item) {
    const items = Array.isArray(manifest) ? manifest.slice() : [];
    const index = items.findIndex((entry) => entry.query_id === item.query_id);
    if (index === -1) items.push(item);
    else items[index] = item;
    return items;
  }

  function updateTrakeState(manifest, queryId, nEvents, eventsConfirmed) {
    return (Array.isArray(manifest) ? manifest : []).map((item) => {
      if (item.query_id !== queryId || item.task !== 'trake') return item;
      return { ...item, n_events: nEvents, events_confirmed: eventsConfirmed };
    });
  }

  function changeTrakeEventCount(manifest, queryId, nEvents) {
    return updateTrakeState(manifest, queryId, nEvents, false);
  }

  function canDownloadValidatedZip(validationStatus, contentType) {
    const mediaType = String(contentType || '').split(';', 1)[0].trim().toLowerCase();
    return validationStatus === 'PASS' && mediaType === 'application/zip';
  }

  function buildTrakeReviewSlots(frames, nEvents) {
    const values = Array.isArray(frames) ? frames : [];
    const count = Number.isInteger(nEvents) && nEvents > 0 ? nEvents : values.length;
    return Array.from({ length: count }, (_, index) => ({
      event: index + 1,
      frame: values[index] ?? null,
      missing: values[index] === undefined || values[index] === null,
    }));
  }

  function manifestQueryFormState(item) {
    return {
      queryId: item.query_id,
      task: item.task,
      text: item.text,
      nEvents: item.task === 'trake' && item.n_events ? item.n_events : 1,
      eventsConfirmed: Boolean(item.events_confirmed),
      translatedText: '',
    };
  }

  function clearQueryWorkspaceState(current) {
    return {
      ...current,
      candidates: [],
      selected: null,
      currentFps: null,
      candidateDraftFrames: {},
      currentPlaybackFrame: null,
      iterCandidates: [],
      iterCursor: 0,
      iterRound: 0,
      iterRunning: false,
      iterVerdict: {},
      iterMatchedList: [],
      iterUnsureList: [],
      iterExcluded: new Set(),
    };
  }

  function canInstallManifest(httpOk, report) {
    return Boolean(httpOk && report?.ok && Array.isArray(report.manifest));
  }

  function manifestQueryReadiness(item, selections) {
    const rows = (Array.isArray(selections) ? selections : [])
      .filter((selection) => selection.queryId === item.query_id);
    const codes = [];
    const labels = [];
    if (!rows.length) {
      codes.push('missing_rows');
      labels.push('Chưa có dòng');
    }
    if (item.task === 'trake' && !item.events_confirmed) {
      codes.push('trake_events_unconfirmed');
      labels.push('Chưa xác nhận events');
    }
    return {
      ready: codes.length === 0,
      codes,
      label: labels.length ? labels.join(' · ') : 'Ready',
    };
  }

  function groupValidationIssues(issues) {
    return (Array.isArray(issues) ? issues : []).reduce((groups, issue) => {
      const entry = typeof issue === 'string' ? { message: issue } : (issue || {});
      const queryId = entry.query_id || entry.queryId || 'general';
      const message = entry.message || entry.detail || String(issue);
      if (!groups[queryId]) groups[queryId] = [];
      groups[queryId].push({ ...entry, message });
      return groups;
    }, {});
  }

  function formatValidationIssue(issue) {
    const entry = typeof issue === 'string' ? { message: issue } : (issue || {});
    const parts = [];
    if (entry.code) parts.push(`[${entry.code}]`);
    const location = [];
    const queryId = entry.query_id || entry.queryId;
    if (queryId) location.push(`query ${queryId}`);
    if (Number.isInteger(entry.row)) location.push(`row ${entry.row}`);
    if (location.length) parts.push(location.join(' · '));
    parts.push(entry.message || entry.detail || String(issue));
    return parts.join(' — ');
  }

  // Nhãn hiển thị cho từng loại evidence backend trả về. Khoá lạ vẫn hiện
  // được, chỉ là dùng chính tên khoá làm nhãn.
  const EVIDENCE_LABELS = {
    transcript_match: 'Lời thoại (ASR)',
    ocr_match: 'Chữ trên hình (OCR)',
    caption_match: 'Caption',
    summary_match: 'Tóm tắt',
    media_info_match: 'Thông tin video',
    text_match: 'Văn bản',
    caption: 'Caption',
    objects: 'Vật thể',
  };

  // Khoá kỹ thuật, không phải bằng chứng để operator đọc.
  const EVIDENCE_SKIP = new Set(['doc_type', 'keyframe_num', 'language']);

  function formatEvidence(evidence) {
    if (!evidence || typeof evidence !== 'object') return [];
    const out = [];
    for (const [key, value] of Object.entries(evidence)) {
      if (EVIDENCE_SKIP.has(key)) continue;
      if (value === null || value === undefined || value === '') continue;
      if (key === 'start_time' || key === 'end_time') continue;
      const label = EVIDENCE_LABELS[key] || key;
      const text = Array.isArray(value) ? value.join(', ') : String(value);
      if (!text) continue;
      out.push({ key, label, text });
    }
    const start = evidence.start_time;
    const end = evidence.end_time;
    if (typeof start === 'number') {
      const span = typeof end === 'number' && end !== start
        ? `${start.toFixed(1)}s – ${end.toFixed(1)}s`
        : `${start.toFixed(1)}s`;
      out.push({ key: 'time', label: 'Mốc thời gian', text: span });
    }
    return out;
  }

  return {
    formatEvidence,
    prepareQaAnswer,
    unicodeCodePointLength,
    candidateToSubmissionFrame,
    canUseIterative,
    inferTaskFromFilename,
    upsertManifestItem,
    updateTrakeState,
    changeTrakeEventCount,
    canDownloadValidatedZip,
    buildTrakeReviewSlots,
    manifestQueryFormState,
    clearQueryWorkspaceState,
    canInstallManifest,
    manifestQueryReadiness,
    groupValidationIssues,
    formatValidationIssue,
  };
});
