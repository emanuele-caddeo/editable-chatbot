// ui.js — UI helpers used by main.js

/* ============================================================
   DOM ELEMENTS
============================================================ */
const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

/* ============================================================
   INPUT AUTO-RESIZE
============================================================ */
export function resizeInputWrapper() {
  const inputWrapper = document.querySelector(".input-wrapper");
  if (!userInput || !inputWrapper) return;

  userInput.style.height = "auto";
  const newHeight = userInput.scrollHeight;
  userInput.style.height = newHeight + "px";
  inputWrapper.style.height = newHeight + 20 + "px";

  // Keep the scroll pinned to the bottom when the input grows
  if (chatContainer) {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }
}

/* ============================================================
   CHAT MESSAGES
============================================================ */
export function addMessage(role, content) {
  if (!chatContainer) return document.createElement("div");

  const wrap = document.createElement("div");
  wrap.className = `message ${role}`;

  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";
  contentDiv.textContent = content;

  wrap.appendChild(contentDiv);
  chatContainer.appendChild(wrap);
  chatContainer.scrollTop = chatContainer.scrollHeight;

  return wrap;
}

export function renderMessages(messages) {
  if (!chatContainer) return;
  chatContainer.innerHTML = "";
  (messages || []).forEach((m) => addMessage(m.role, m.content));
}

/* ============================================================
   MODEL LIST
============================================================ */
export function setModelListOptions(models, selected) {
  const select = document.getElementById("model-list");
  if (!select) return;

  select.innerHTML = "";
  (models || []).forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    select.appendChild(opt);
  });

  if (selected && models.includes(selected)) {
    select.value = selected;
  }
}

/* ============================================================
   CLEAR CHAT BUTTON
============================================================ */
export function bindClearChat(handler) {
  const btn = document.getElementById("clear-chat-btn");
  if (!btn) return;
  btn.addEventListener("click", handler);
}

/* ============================================================
   INPUT LOCK (SYSTEM BUSY)
============================================================ */
export function setChatInputLocked(locked, placeholderText = "") {
  if (!userInput || !sendBtn) return;

  userInput.disabled = !!locked;
  sendBtn.disabled = !!locked;

  if (locked) {
    if (placeholderText) userInput.placeholder = placeholderText;
    userInput.blur();
  } else {
    if (placeholderText) userInput.placeholder = placeholderText;
  }
}

/* ============================================================
   CHAT INPUT HANDLERS
============================================================ */
export function bindChatInputHandlers(sendHandler) {
  if (!sendBtn || !userInput) return;

  sendBtn.addEventListener("click", () => {
    if (sendBtn.disabled || userInput.disabled) return;
    sendHandler();
  });

  userInput.addEventListener("keydown", (e) => {
    if (userInput.disabled) return;

    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (sendBtn.disabled) return;
      sendHandler();
    }
  });

  userInput.addEventListener("input", () => {
    resizeInputWrapper();
  });
}
