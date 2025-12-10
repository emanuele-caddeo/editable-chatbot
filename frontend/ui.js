// ui.js
// Tutte le funzioni di UI necessarie a main.js

/* ============================================================
   ELEMENTI DOM
============================================================ */
const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

/* ============================================================
   RESIZE INPUT
============================================================ */
export function resizeInputWrapper() {
  const inputWrapper = document.querySelector(".input-wrapper");
  if (!userInput || !inputWrapper) return;

  userInput.style.height = "auto";
  let newHeight = userInput.scrollHeight;
  userInput.style.height = newHeight + "px";
  inputWrapper.style.height = newHeight + 20 + "px";
}

/* ============================================================
   MESSAGGI CHAT
============================================================ */
export function addMessage(role, content) {
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
  chatContainer.innerHTML = "";
  (messages || []).forEach((m) => addMessage(m.role, m.content));
}

/* ============================================================
   MODEL LIST
============================================================ */
export function setModelListOptions(models, selected) {
  const select = document.getElementById("model-list");
  select.innerHTML = "";
  models.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    select.appendChild(opt);
  });
  if (selected) select.value = selected;
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
   CHAT INPUT HANDLERS  <-- QUESTA È LA FUNZIONE CHE MANCAVA
============================================================ */
export function bindChatInputHandlers(sendHandler) {
  if (!sendBtn || !userInput) return;

  sendBtn.addEventListener("click", () => {
    sendHandler();
  });

  userInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendHandler();
    }
  });

  userInput.addEventListener("input", resizeInputWrapper);
}
