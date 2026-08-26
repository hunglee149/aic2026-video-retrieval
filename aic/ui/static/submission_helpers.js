(function (root, factory) {
  const helpers = factory();
  if (typeof module === 'object' && module.exports) module.exports = helpers;
  root.AICSubmissionHelpers = helpers;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  function prepareQaAnswer(answer) {
    return String(answer ?? '');
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

  function groupValidationIssues(issues) {
    return (Array.isArray(issues) ? issues : []).reduce((groups, issue) => {
      const entry = typeof issue === 'string' ? { message: issue } : (issue || {});
      const queryId = entry.query_id || entry.queryId || 'general';
      const message = entry.message || entry.detail || String(issue);
      if (!groups[queryId]) groups[queryId] = [];
      groups[queryId].push(message);
      return groups;
    }, {});
  }

  return {
    prepareQaAnswer,
    canUseIterative,
    inferTaskFromFilename,
    upsertManifestItem,
    updateTrakeState,
    groupValidationIssues,
  };
});
