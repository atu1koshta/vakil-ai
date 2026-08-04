const BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8787';

async function handle(response) {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body.detail) detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return response.json();
}

export async function processDocument(file) {
  const form = new FormData();
  form.append('file', file);
  return handle(await fetch(`${BASE_URL}/process-document`, { method: 'POST', body: form }));
}

export async function driveAuthStatus() {
  return handle(await fetch(`${BASE_URL}/drive/auth/status`));
}

export async function driveAuthUrl() {
  return handle(await fetch(`${BASE_URL}/drive/auth/url`));
}

export async function driveDisconnect() {
  return handle(await fetch(`${BASE_URL}/drive/disconnect`, { method: 'POST' }));
}

export async function listDriveFiles() {
  return handle(await fetch(`${BASE_URL}/drive/files`));
}

export async function processDriveFile(fileId) {
  return handle(await fetch(`${BASE_URL}/drive/process/${fileId}`, { method: 'POST' }));
}

export function documentPdfUrl(docId) {
  return `${BASE_URL}/documents/${docId}/pdf`;
}

export async function driveSyncStart() {
  return handle(await fetch(`${BASE_URL}/drive/sync`, { method: 'POST' }));
}

export async function driveSyncStatus() {
  return handle(await fetch(`${BASE_URL}/drive/sync/status`));
}

export async function listDocuments() {
  return handle(await fetch(`${BASE_URL}/documents`));
}

export async function searchChunks(q, k = 5) {
  const params = new URLSearchParams({ q, k: String(k) });
  return handle(await fetch(`${BASE_URL}/search?${params}`));
}

export async function indexStatus(docId) {
  return handle(await fetch(`${BASE_URL}/documents/${docId}/index-status`));
}
