// frontend/api.js

const API_BASE = "http://localhost:8000";

/* ============================================================
   MODELS
============================================================ */

/**
 * Get locally available models
 */
export async function fetchModels() {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) {
    throw new Error(`fetchModels failed: ${res.status}`);
  }
  return res.json();
}

/**
 * Download a model from HuggingFace
 * @param {string} repoId - HuggingFace repo id (e.g. openai-community/gpt2-xl)
 * @param {string} hfToken - Optional HuggingFace token
 */
export async function downloadModel(repoId, hfToken) {
  const res = await fetch(`${API_BASE}/api/models/download`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      repo_id: repoId,     // ✅ backend expects this
      hf_token: hfToken,   // ✅ backend expects this
    }),
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`downloadModel failed: ${res.status} ${txt}`);
  }

  return res.json();
}

/* ============================================================
   CHAT HISTORY
============================================================ */

export async function fetchHistory() {
  const res = await fetch(`${API_BASE}/api/chat/history`);
  if (!res.ok) {
    throw new Error(`fetchHistory failed: ${res.status}`);
  }
  return res.json();
}

export async function saveHistory(model, messages) {
  const res = await fetch(`${API_BASE}/api/chat/history`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages,
    }),
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`saveHistory failed: ${res.status} ${txt}`);
  }

  return res.json();
}

export async function clearHistoryApi() {
  const res = await fetch(`${API_BASE}/api/chat/history`, {
    method: "DELETE",
  });
  if (!res.ok) {
    throw new Error(`clearHistory failed: ${res.status}`);
  }
}

/* ============================================================
   SYSTEM / SETTINGS
============================================================ */

export async function fetchSystemStatus() {
  const res = await fetch(`${API_BASE}/api/system/status`);
  if (!res.ok) {
    throw new Error(`fetchSystemStatus failed: ${res.status}`);
  }
  return res.json();
}

export async function setComputeModeApi(mode) {
  const res = await fetch(`${API_BASE}/api/system/compute`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ mode }),
  });

  if (!res.ok) {
    const txt = await res.text();
    throw new Error(`setComputeMode failed: ${res.status} ${txt}`);
  }

  return res.json();
}

/* ============================================================
   CHAT WEBSOCKET
============================================================ */

export function openChatWebSocket({
  model,
  message,
  compute_mode,
  temperature,
  max_tokens,
  top_p,
  top_k,
  repetition_penalty,
  onMessage,
  onDone,
  onError,
}) {
  const ws = new WebSocket("ws://localhost:8000/api/chat/stream");

  ws.onopen = () => {
    ws.send(
      JSON.stringify({
        model,
        message,
        compute_mode,
        temperature,
        max_tokens,
        top_p,
        top_k,
        repetition_penalty,
      })
    );
  };

  ws.onmessage = (event) => {
    let msg;
    try {
      msg = JSON.parse(event.data);
    } catch {
      return;
    }

    if (onMessage) onMessage(msg);

    if (msg.type === "done") {
      if (onDone) onDone();
      ws.close();
    }
  };

  ws.onerror = (err) => {
    if (onError) onError(err);
  };

  return ws;
}
