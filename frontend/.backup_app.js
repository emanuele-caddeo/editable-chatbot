const API_BASE = "http://localhost:8000";
const WS_URL = "ws://localhost:8000/api/chat/stream";

let computeMode = "gpu";

const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const modelList = document.getElementById("model-list");
const modelIdInput = document.getElementById("model-id-input");
const downloadBtn = document.getElementById("download-btn");
const inputWrapper = document.querySelector(".input-wrapper");

// HF token elements
const hfTokenBtn = document.getElementById("hf-token-btn");
const hfTokenPanel = document.getElementById("hf-token-panel");
const hfTokenInput = document.getElementById("hf-token-input");
const hfTokenSave = document.getElementById("hf-token-save");
const hfTokenClear = document.getElementById("hf-token-clear");
const hfTokenClose = document.getElementById("hf-token-close");

// compute mode + offload UI
const btnCpu = document.getElementById("btn-cpu");
const btnGpu = document.getElementById("btn-gpu");
const offloadDot = document.getElementById("offload-dot");

// bottone Pulisci (se lo aggiungi in HTML)
const clearChatBtn = document.getElementById("clear-chat-btn");

let currentModel = null;
let messages = [];

/* ============================
   HF TOKEN HELPERS
   ============================ */
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
  if (nextState) hfTokenPanel.classList.add("open");
  else hfTokenPanel.classList.remove("open");
}

/* ============================
   INPUT RESIZE
   ============================ */
function resizeInputWrapper() {
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

/* ============================
   CHAT UI
   ============================ */
function addMessage(role, content) {
  const div = document.createElement("div");
  div.classList.add("message", role);
  div.textContent = content;
  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

/* ============================
   MODELS
   ============================ */
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
  downloadBtn.textContent = "Scarico.";

  try {
    const body = { repo_id: repoId };
    const token = getHfToken();
    if (token) {
      body.hf_token = token;
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

/* ============================
   COMPUTE MODE + OFFLOAD UI
   ============================ */
function updateComputeButtonsUI() {
  if (computeMode === "cpu") {
    btnCpu.classList.add("active");
    btnGpu.classList.remove("active");
  } else {
    btnGpu.classList.add("active");
    btnCpu.classList.remove("active");
  }
}

function updateOffloadIndicator(isActive) {
  if (isActive) {
    offloadDot.classList.add("active");
  } else {
    offloadDot.classList.remove("active");
  }
}

async function setComputeMode(mode) {
  try {
    const res = await fetch(`${API_BASE}/api/system/compute-mode`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    });
    if (!res.ok) {
      const err = await res.json();
      alert("Errore cambio compute mode: " + err.detail);
      return;
    }
    const data = await res.json();
    computeMode = data.mode;
    updateComputeButtonsUI();
    updateOffloadIndicator(false);
  } catch (e) {
    console.error(e);
    alert("Errore di connessione nel cambio compute mode.");
  }
}

btnCpu.addEventListener("click", () => setComputeMode("cpu"));
btnGpu.addEventListener("click", () => setComputeMode("gpu"));

async function initSystemStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/system/status`);
    if (!res.ok) return;
    const data = await res.json();
    computeMode = data.compute_mode || "gpu";
    updateComputeButtonsUI();
    updateOffloadIndicator(Boolean(data.offload_active));
  } catch (e) {
    console.error("Errore fetch system status", e);
  }
}

/* ============================
   HISTORY DA SERVER (JSON)
   ============================ */
async function loadChatHistoryFromServer() {
  try {
    const res = await fetch(`${API_BASE}/api/chat/history`);
    if (!res.ok) return;

    const json = await res.json();
    // ci aspettiamo { model, messages }
    if (!json || !Array.isArray(json.messages)) return;

    currentModel = json.model || currentModel;
    if (currentModel) {
      // se il modello è nella lista, selezionalo
      const options = Array.from(modelList.options).map((o) => o.value);
      if (options.includes(currentModel)) {
        modelList.value = currentModel;
      }
    }

    messages = json.messages;
    chatContainer.innerHTML = "";
    messages.forEach((m) => {
      if (m && m.role && m.content) {
        addMessage(m.role, m.content);
      }
    });
  } catch (e) {
    console.error("Errore caricamento history", e);
  }
}

/* ============================
   CHAT STREAMING (WebSocket)
   ============================ */
async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;
  if (!currentModel) {
    alert("Seleziona o scarica prima un modello.");
    return;
  }

  // UI locale
  addMessage("user", text);
  messages.push({ role: "user", content: text });
  userInput.value = "";
  resizeInputWrapper();

  // bubble assistente vuota
  addMessage("assistant", "");
  const assistantBubble = chatContainer.lastChild;

  const ws = new WebSocket(WS_URL);
  let fullReply = "";

  ws.onopen = () => {
    ws.send(
      JSON.stringify({
        model: currentModel,
        message: text,        // <-- CAMPO UNICO CHE SI ASPETTA IL SERVER
        compute_mode: computeMode,
      })
    );
  };

  ws.onmessage = (event) => {
    let data;
    try {
      data = JSON.parse(event.data);
    } catch (e) {
      console.error("Messaggio WS non valido:", event.data);
      return;
    }

    if (data.type === "chunk") {
      assistantBubble.textContent += data.content;
      fullReply += data.content;
      chatContainer.scrollTop = chatContainer.scrollHeight;
    } else if (data.type === "done") {
      messages.push({
        role: "assistant",
        content: fullReply,
      });
      ws.close();
      initSystemStatus();
    } else if (data.type === "error") {
      assistantBubble.textContent = "Errore: " + data.message;
      ws.close();
    }
  };

  ws.onerror = () => {
    assistantBubble.textContent = "Errore di connessione al server.";
  };
}

/* ============================
   EVENTI INPUT / HF TOKEN
   ============================ */
sendBtn.addEventListener("click", sendMessage);

userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

userInput.addEventListener("input", resizeInputWrapper);

// HF token panel
hfTokenBtn.addEventListener("click", () => toggleHfTokenPanel());
hfTokenClose.addEventListener("click", () => toggleHfTokenPanel(false));

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

// chiudi pannello token se clicchi fuori
document.addEventListener("click", (e) => {
  if (!hfTokenPanel.contains(e.target) && !hfTokenBtn.contains(e.target)) {
    toggleHfTokenPanel(false);
  }
});

/* ============================
   BOTTONE PULISCI (OPZIONALE)
   ============================ */
if (clearChatBtn) {
  clearChatBtn.addEventListener("click", async () => {
    try {
      await fetch(`${API_BASE}/api/chat/history`, { method: "DELETE" });
    } catch (e) {
      console.error("Errore cancellazione history", e);
    }

    messages = [];
    chatContainer.innerHTML = "";
    window.location.reload();
  });
}

/* ============================
   INIT
   ============================ */
async function init() {
  resizeInputWrapper();
  await loadModels();
  await initSystemStatus();
  await loadChatHistoryFromServer();

  const savedToken = getHfToken();
  if (savedToken) {
    hfTokenInput.value = savedToken;
    hfTokenBtn.classList.add("active");
  }
}

init();
