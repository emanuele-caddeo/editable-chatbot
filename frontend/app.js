const API_BASE = "http://localhost:8000";

const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const modelList = document.getElementById("model-list");
const modelIdInput = document.getElementById("model-id-input");
const downloadBtn = document.getElementById("download-btn");
const inputWrapper = document.querySelector(".input-wrapper");

// elementi token HF
const hfTokenBtn = document.getElementById("hf-token-btn");
const hfTokenPanel = document.getElementById("hf-token-panel");
const hfTokenInput = document.getElementById("hf-token-input");
const hfTokenSave = document.getElementById("hf-token-save");
const hfTokenClear = document.getElementById("hf-token-clear");
const hfTokenClose = document.getElementById("hf-token-close");

let currentModel = null;
let messages = [];

/**
 * HF TOKEN HELPERS
 */
function getHfToken() {
  return localStorage.getItem("hf_token") || "";
}

function setHfToken(token) {
  if (token) {
    localStorage.setItem("hf_token", token);
    hfTokenBtn.classList.add("active");
  } else {
    localStorage.removeItem("hf_token");
    hfTokenBtn.classList.remove("active");
  }
}

function toggleHfTokenPanel(forceState = null) {
  const isOpen = hfTokenPanel.classList.contains("open");
  const nextState = forceState === null ? !isOpen : forceState;
  if (nextState) {
    hfTokenPanel.classList.add("open");
  } else {
    hfTokenPanel.classList.remove("open");
  }
}

/**
 * Ridimensiona l'input-wrapper e la textarea in base al contenuto.
 */
function resizeInputWrapper() {
  const lineHeight = 25;
  const computed = getComputedStyle(userInput);
  const maxHeight = parseInt(computed.maxHeight, 10) || 120;

  userInput.style.height = "auto";
  const scrollH = userInput.scrollHeight;

  let numLines = Math.ceil(scrollH / lineHeight);
  if (numLines < 1) numLines = 1;

  let newHeight = numLines * lineHeight;
  if (newHeight > maxHeight) {
    newHeight = maxHeight;
  }

  userInput.style.height = newHeight + "px";

  const extraPadding = 20;
  inputWrapper.style.height = newHeight + extraPadding + "px";
}

function addMessage(role, content) {
  const div = document.createElement("div");
  div.classList.add("message", role);
  div.textContent = content;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

async function loadModels() {
  try {
    const res = await fetch(`${API_BASE}/api/models`);
    const data = await res.json();
    modelList.innerHTML = "";

    (data.models || []).forEach((m, idx) => {
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      if (idx === 0 && !currentModel) currentModel = m;
      modelList.appendChild(opt);
    });

    if (currentModel) modelList.value = currentModel;
  } catch (e) {
    console.error(e);
  }
}

modelList.addEventListener("change", () => {
  currentModel = modelList.value;
});

downloadBtn.addEventListener("click", async () => {
  const repoId = modelIdInput.value.trim();
  if (!repoId) return;

  downloadBtn.disabled = true;
  downloadBtn.textContent = "Scarico...";

  try {
    const body = { repo_id: repoId };
    const token = getHfToken();
    if (token) {
      body.hf_token = token; // verrà usato dal backend
    }

    const res = await fetch(`${API_BASE}/api/models/download`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json();
      alert("Errore: " + err.detail);
    } else {
      await loadModels();
      currentModel = repoId;
      modelList.value = repoId;
    }
  } catch (e) {
    console.error(e);
    alert("Errore durante il download del modello.");
  } finally {
    downloadBtn.disabled = false;
    downloadBtn.textContent = "Scarica / Usa";
  }
});

async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;
  if (!currentModel) {
    alert("Seleziona o scarica prima un modello.");
    return;
  }

  addMessage("user", text);
  messages.push({ role: "user", content: text });
  userInput.value = "";
  resizeInputWrapper();

  addMessage("assistant", "⏳ Sto pensando...");
  const thinkingBubble = chatContainer.lastChild;

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: currentModel,
        messages: messages,
        max_tokens: 256,
        temperature: 0.7,
      }),
    });

    if (!res.ok) {
      const err = await res.json();
      thinkingBubble.textContent = "Errore: " + err.detail;
      return;
    }

    const data = await res.json();
    thinkingBubble.remove();
    addMessage("assistant", data.reply);
    messages.push({ role: "assistant", content: data.reply });
  } catch (e) {
    console.error(e);
    thinkingBubble.textContent = "Errore nella richiesta.";
  }
}

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

userInput.addEventListener("input", resizeInputWrapper);

/**
 * Eventi pannello token HF
 */
hfTokenBtn.addEventListener("click", () => {
  toggleHfTokenPanel();
});

hfTokenClose.addEventListener("click", () => {
  toggleHfTokenPanel(false);
});

hfTokenSave.addEventListener("click", () => {
  const token = hfTokenInput.value.trim();
  setHfToken(token);
  toggleHfTokenPanel(false);
});

hfTokenClear.addEventListener("click", () => {
  hfTokenInput.value = "";
  setHfToken("");
  toggleHfTokenPanel(false);
});

// Chiudi pannello se clicchi fuori
document.addEventListener("click", (e) => {
  if (!hfTokenPanel.contains(e.target) && !hfTokenBtn.contains(e.target)) {
    toggleHfTokenPanel(false);
  }
});

// init
loadModels();

// setup iniziale: resize textarea + token
resizeInputWrapper();
const savedToken = getHfToken();
if (savedToken) {
  hfTokenInput.value = savedToken;
  hfTokenBtn.classList.add("active");
}
