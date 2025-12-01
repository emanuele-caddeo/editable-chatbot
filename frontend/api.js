// api.js
import { API_BASE, WS_URL } from "./config.js";

export async function fetchModels() {
  const res = await fetch(`${API_BASE}/api/models`);
  if (!res.ok) throw new Error("models fetch failed");
  return res.json();
}

export async function fetchSystemStatus() {
  const res = await fetch(`${API_BASE}/api/system/status`);
  if (!res.ok) throw new Error("status fetch failed");
  return res.json();
}

export async function fetchHistory() {
  const res = await fetch(`${API_BASE}/api/chat/history`);
  if (!res.ok) throw new Error("history fetch failed");
  return res.json();
}

export async function clearHistory() {
  await fetch(`${API_BASE}/api/chat/history`, { method: "DELETE" });
}

export async function setComputeModeApi(mode) {
  const res = await fetch(`${API_BASE}/api/system/compute-mode`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode }),
  });
  if (!res.ok) throw new Error("compute-mode failed");
  return res.json();
}

export function openChatWebSocket({ model, message, computeMode, onChunk, onDone, onError }) {
  const ws = new WebSocket(WS_URL);
  ws.onopen = () => {
    ws.send(JSON.stringify({ model, message, compute_mode: computeMode }));
  };
  ws.onmessage = (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch (e) {
      console.error("Invalid WS data", event.data);
      return;
    }
    if (data.type === "chunk") onChunk?.(data.content);
    else if (data.type === "done") onDone?.();
    else if (data.type === "error") onError?.(data.message);
  };
  ws.onerror = () => onError?.("WebSocket error");
  return ws;
}
