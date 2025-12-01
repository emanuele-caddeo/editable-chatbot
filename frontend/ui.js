// ui.js
import { getMessages, pushMessage } from "./state.js";

const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const inputWrapper = document.querySelector(".input-wrapper");
const modelList = document.getElementById("model-list");
const clearChatBtn = document.getElementById("clear-chat-btn");
// ... e tutti gli altri elementi DOM

export function addMessage(role, content) {
  const div = document.createElement("div");
  div.classList.add("message", role);
  div.textContent = content;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

export function getUserInputText() {
  return userInput.value.trim();
}

export function clearUserInput() {
  userInput.value = "";
  resizeInputWrapper();
}

export function resizeInputWrapper() {
  const lineHeight = 25;
  const computed = getComputedStyle(userInput);
  const maxHeight = parseInt(computed.maxHeight, 10) || 120;

  userInput.style.height = "auto";
  const scrollH = userInput.scrollHeight;

  let numLines = Math.ceil(scrollH / lineHeight);
  if (numLines < 1) numLines = 1;
  let newHeight = numLines * lineHeight;
  if (newHeight > maxHeight) newHeight = maxHeight;

  userInput.style.height = newHeight + "px";
  const extraPadding = 20;
  inputWrapper.style.height = newHeight + extraPadding + "px";
}

export function bindInputHandlers(onSend) {
  document.getElementById("send-btn").addEventListener("click", onSend);

  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSend();
    }
  });

  userInput.addEventListener("input", resizeInputWrapper);
}

export function bindClearChat(onClear) {
  if (!clearChatBtn) return;
  clearChatBtn.addEventListener("click", onClear);
}

export function setModelListOptions(models, currentModel) {
  modelList.innerHTML = "";
  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    modelList.appendChild(opt);
  });
  if (currentModel) {
    modelList.value = currentModel;
  }
}

export function bindModelChange(onChange) {
  modelList.addEventListener("change", () => {
    onChange(modelList.value);
  });
}

// + qui metti tutta la parte HF token, CPU/GPU, offload indicator, ecc.
