// state.js
let currentModel = null;
let messages = [];
let computeMode = "gpu";

export function getCurrentModel() {
  return currentModel;
}
export function setCurrentModel(model) {
  currentModel = model;
}

export function getMessages() {
  return messages;
}
export function pushMessage(msg) {
  messages.push(msg);
}
export function clearMessages() {
  messages = [];
}

export function getComputeMode() {
  return computeMode;
}
export function setComputeMode(mode) {
  computeMode = mode;
}
