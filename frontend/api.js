// api.js
// Helper per tutte le chiamate HTTP e WebSocket al backend

// Puoi cambiare qui se vuoi rendere dinamico l'host
const API_HOST = window.location.hostname || "localhost";
const API_HTTP_PORT = 8000;

const API_BASE = `${window.location.protocol}//${API_HOST}:${API_HTTP_PORT}`;
const WS_PROTOCOL = window.location.protocol === "https:" ? "wss" : "ws";
const WS_BASE = `${WS_PROTOCOL}://${API_HOST}:${API_HTTP_PORT}`;

// ============================
// MODELS
// ============================

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) {
    throw new Error(`Errore fetch models: ${res.status}`);
  }
  return res.json();
}

export async function downloadModel(repoId, hfToken) {
  const body = { repo_id: repoId };
  if (hfToken) {
    body.hf_token = hfToken;
  }

  const res = await fetch(`${API_BASE}/api/models/download`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    let msg = `Download failed: ${res.status}`;
    try {
      const err = await res.json();
      if (err.detail) msg = err.detail;
    } catch {
      // ignora
    }
    throw new Error(msg);
  }
  return res.json();
}

// ============================
// CHAT HISTORY
// ============================

export async function fetchHistory() {
  const res = await fetch(`${API_BASE}/api/chat/history`);
  if (!res.ok) {
    throw new Error(`Errore fetch history: ${res.status}`);
  }
  return res.json();
}

export async function clearHistoryApi() {
  const res = await fetch(`${API_BASE}/api/chat/history`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(`Errore clear history: ${res.status}`);
  }
  return res.json();
}

// ============================
// SYSTEM STATUS / COMPUTE MODE
// ============================

export async function fetchSystemStatus() {
  const res = await fetch(`${API_BASE}/api/system/status`);
  if (!res.ok) {
    throw new Error(`Errore fetch system status: ${res.status}`);
  }
  return res.json();
}

export async function setComputeModeApi(mode) {
  const res = await fetch(`${API_BASE}/api/system/compute-mode`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });

  if (!res.ok) {
    let msg = `Errore compute-mode: ${res.status}`;
    try {
      const err = await res.json();
      if (err.detail) msg = err.detail;
    } catch {
      // ignora
    }
    throw new Error(msg);
  }
  return res.json();
}

// ============================
// CHAT WEBSOCKET (STREAMING)
// ============================

/**
 * Apre un WebSocket verso /api/chat/stream.
 *
 * options:
 *  - model (string)
 *  - message (string)
 *  - compute_mode (string | null)
 *  - temperature, max_tokens, top_p, top_k, repetition_penalty (number)
 *  - onChunk(content: string)
 *  - onDone()
 *  - onError(msg: string)
 */
export function openChatWebSocket(options) {
  const {
    model,
    message,
    compute_mode,
    temperature,
    max_tokens,
    top_p,
    top_k,
    repetition_penalty,
    onChunk,
    onDone,
    onError,
  } = options;

  const ws = new WebSocket(`${WS_BASE}/api/chat/stream`);

  ws.onopen = () => {
    const payload = {
      model,
      message,
      compute_mode,
      temperature,
      max_tokens,
      top_p,
      top_k,
      repetition_penalty,
    };
    ws.send(JSON.stringify(payload));
  };

  ws.onmessage = (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch (e) {
      console.error("Messaggio WS non valido:", event.data);
      if (onError) onError("Messaggio non valido dal server.");
      return;
    }

    if (data.type === "chunk") {
      if (onChunk && typeof data.content === "string") {
        onChunk(data.content);
      }
    } else if (data.type === "done") {
      if (onDone) onDone();
    } else if (data.type === "error") {
      if (onError) onError(data.message || "Errore dal server.");
    }
  };

  ws.onerror = () => {
    if (onError) onError("Errore di connessione al server.");
  };

  ws.onclose = (ev) => {
    // opzionale: loggare chiusura
    if (!ev.wasClean && onError) {
      onError("Connessione chiusa inaspettatamente.");
    }
  };

  return ws;
}
