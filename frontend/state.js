// state.js
// Gestione dello stato globale dell'applicazione

/* ============================================================
   MODELLO SELEZIONATO
============================================================ */
let currentModel = null;

export function getCurrentModel() {
  return currentModel;
}

export function setCurrentModel(model) {
  currentModel = model;
}

/* ============================================================
   MESSAGGI DELLA CHAT (solo per UI)
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
   STATO GENERAZIONE (eventuale stop futuro)
============================================================ */
let isGenerating = false;

export function getIsGenerating() {
  return isGenerating;
}

export function setIsGenerating(flag) {
  isGenerating = flag;
}
