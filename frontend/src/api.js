/**
 * api.js — API client for Talk-to-Data FastAPI backend
 * Covers: data upload, RAG uploads (audio/PDF/image/text), queries,
 * chart generation, quick actions, exports, and settings.
 */

const API_BASE = 'http://localhost:8000';

// ── Data Upload (CSV / Excel) ──
export async function uploadFile(file, sheetName = null) {
  const formData = new FormData();
  formData.append('file', file);
  if (sheetName) formData.append('sheet_name', sheetName);

  const res = await fetch(`${API_BASE}/api/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(err.detail || 'Upload failed');
  }
  return res.json();
}

// ── RAG File Uploads ──
export async function uploadAudio(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/upload/audio`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Audio upload failed' }));
    throw new Error(err.detail || 'Audio upload failed');
  }
  return res.json();
}

export async function uploadPDF(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/upload/pdf`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'PDF upload failed' }));
    throw new Error(err.detail || 'PDF upload failed');
  }
  return res.json();
}

export async function uploadImage(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/upload/image`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Image upload failed' }));
    throw new Error(err.detail || 'Image upload failed');
  }
  return res.json();
}

export async function uploadText(file) {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${API_BASE}/api/upload/text`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Text upload failed' }));
    throw new Error(err.detail || 'Text upload failed');
  }
  return res.json();
}

// ── RAG Files Listing ──
export async function getRagFiles() {
  const res = await fetch(`${API_BASE}/api/rag-files`);
  return res.json();
}

// ── Schema ──
export async function getSchema() {
  const res = await fetch(`${API_BASE}/api/schema`);
  return res.json();
}

// ── Query ──
export async function sendQuery(query, routingMode = 'Auto-Detect') {
  const res = await fetch(`${API_BASE}/api/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, routing_mode: routingMode }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Query failed' }));
    throw new Error(err.detail || 'Query failed');
  }
  return res.json();
}

// ── Chart Generation ──
export async function generateChart(query, chartType = null) {
  const res = await fetch(`${API_BASE}/api/generate-chart`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, chart_type: chartType }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Chart generation failed' }));
    throw new Error(err.detail || 'Chart generation failed');
  }
  return res.json();
}

// ── Quick Actions ──
export async function quickAction(action) {
  const res = await fetch(`${API_BASE}/api/quick-action`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Action failed' }));
    throw new Error(err.detail || 'Action failed');
  }
  return res.json();
}

// ── Chat ──
export async function clearChat() {
  const res = await fetch(`${API_BASE}/api/clear`, { method: 'POST' });
  return res.json();
}

// ── RAG Data Management ──
export async function clearRagData() {
  const res = await fetch(`${API_BASE}/api/clear-rag`, { method: 'POST' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to clear RAG data' }));
    throw new Error(err.detail || 'Failed to clear RAG data');
  }
  return res.json();
}

// ── Metrics ──
export async function getMetrics() {
  const res = await fetch(`${API_BASE}/api/metrics`);
  return res.json();
}

export async function addMetric(name, definition) {
  const res = await fetch(`${API_BASE}/api/metrics`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, definition }),
  });
  return res.json();
}

// ── Exports ──
export async function exportCSV() {
  const res = await fetch(`${API_BASE}/api/export/csv`, { method: 'POST' });
  if (!res.ok) throw new Error('Export failed');
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `data_${Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

export async function exportPDF(query = '', responseText = '', chatHistory = []) {
  const res = await fetch(`${API_BASE}/api/export/pdf`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      query,
      response_text: responseText,
      chat_history: chatHistory,
    }),
  });
  if (!res.ok) throw new Error('PDF export failed');
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `report_${Date.now()}.pdf`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Settings ──
export async function togglePrivacy(enabled) {
  const formData = new FormData();
  formData.append('enabled', enabled);
  const res = await fetch(`${API_BASE}/api/settings/privacy`, {
    method: 'POST',
    body: formData,
  });
  return res.json();
}

// ── Dataset Profiling ──
export async function getDatasetProfile() {
  const res = await fetch(`${API_BASE}/api/dataset-profile`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Profile failed' }));
    throw new Error(err.detail || 'Profile failed');
  }
  return res.json();
}

// ── Render Chart from Suggestion ──
export async function renderSuggestionChart(suggestionId, options = {}) {
  const res = await fetch(`${API_BASE}/api/render-chart`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      suggestion_id: suggestionId,
      aggregation: options.aggregation || null,
      sort_by: options.sort_by || null,
      filters: options.filters || null,
      chart_type_override: options.chart_type_override || null,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Render failed' }));
    throw new Error(err.detail || 'Render failed');
  }
  return res.json();
}

// ── Health Check ──
export async function healthCheck() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    return res.json();
  } catch {
    return { status: 'offline', has_data: false };
  }
}
