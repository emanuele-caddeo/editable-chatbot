// ui.js — UI helpers used by main.js

/* ============================================================
   DOM ELEMENTS
============================================================ */
const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

let systemStatusEl = null;
const initialPlaceholder = userInput ? userInput.placeholder : "";

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

  // Add a class to help styling if you already have CSS rules
  if (locked) {
    userInput.classList.add("is-locked");
    sendBtn.classList.add("is-locked");
  } else {
    userInput.classList.remove("is-locked");
    sendBtn.classList.remove("is-locked");
  }

  if (locked) {
    userInput.placeholder = placeholderText || "Please wait…";
    userInput.blur();
  } else {
    userInput.placeholder = initialPlaceholder;
  }
}

/* ============================================================
   SYSTEM STATUS (VISIBLE WITHOUT CSS)
============================================================ */
export function showSystemStatus(text = "") {
  if (!systemStatusEl) {
    systemStatusEl = document.createElement("div");
    systemStatusEl.className = "system-status";
    document.body.appendChild(systemStatusEl);

    // Inline styles so it's visible even if CSS is missing
    systemStatusEl.style.position = "fixed";
    systemStatusEl.style.left = "50%";
    systemStatusEl.style.bottom = "16px";
    systemStatusEl.style.transform = "translateX(-50%)";
    systemStatusEl.style.zIndex = "9999";
    systemStatusEl.style.padding = "10px 14px";
    systemStatusEl.style.borderRadius = "10px";
    systemStatusEl.style.fontSize = "14px";
    systemStatusEl.style.maxWidth = "90vw";
    systemStatusEl.style.whiteSpace = "nowrap";
    systemStatusEl.style.overflow = "hidden";
    systemStatusEl.style.textOverflow = "ellipsis";
    systemStatusEl.style.background = "rgba(0,0,0,0.80)";
    systemStatusEl.style.color = "#fff";
    systemStatusEl.style.boxShadow = "0 6px 18px rgba(0,0,0,0.25)";
  }

  systemStatusEl.textContent = text || "System busy…";
  systemStatusEl.style.display = "block";
}

export function hideSystemStatus() {
  if (systemStatusEl) {
    systemStatusEl.style.display = "none";
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
