// state.js
// Global application state

/* ============================================================
   SELECTED MODEL
============================================================ */
let currentModel = null;

export function getCurrentModel() {
  return currentModel;
}

export function setCurrentModel(model) {
  currentModel = model;
}

/* ============================================================
   CHAT MESSAGES (UI-only)
============================================================ */
let messages = [];

export function getMessages() {
  return messages;
}

export function setMessages(list) {
  messages = Array.isArray(list) ? [...list] : [];
}

export function pushMessage(msg) {
  messages.push(msg);
}

export function clearMessages() {
  messages = [];
}

/* ============================================================
   COMPUTE MODE
============================================================ */
let computeMode = "gpu";

export function getComputeMode() {
  return computeMode;
}

export function setComputeMode(mode) {
  computeMode = mode;
}

/* ============================================================
   GENERATION STATE (future stop support)
============================================================ */
let isGenerating = false;

export function getIsGenerating() {
  return isGenerating;
}

export function setIsGenerating(flag) {
  isGenerating = flag;
}

/* ============================================================
   SYSTEM BUSY STATE (e.g., knowledge editing in progress)
============================================================ */
let systemBusy = false;

export function getSystemBusy() {
  return systemBusy;
}

export function setSystemBusy(flag) {
  systemBusy = !!flag;
}
