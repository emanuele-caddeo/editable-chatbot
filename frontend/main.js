/********************************************************************
 * MAIN.JS — Completamente aggiornato con SETTINGS PANEL avanzato
 * Gestisce:
 *  - Apertura/chiusura pannello impostazioni
 *  - Parametri generazione modello (temperature, top_p, top_k, …)
 *  - Compute mode CPU/GPU
 *  - Comunicazione WebSocket
 *  - Caricamento/salvataggio history
 ********************************************************************/

import {
  fetchModels,
  fetchHistory,
  clearHistoryApi,
  fetchSystemStatus,
  setComputeModeApi,
  openChatWebSocket,
  downloadModel
} from "./api.js";

import {
  getCurrentModel,
  setCurrentModel,
  getMessages,
  setMessages,
  pushMessage,
  clearMessages,
  getComputeMode,
  setComputeMode
} from "./state.js";

import {
  addMessage,
  renderMessages,
  bindChatInputHandlers,
  bindClearChat,
  resizeInputWrapper,
  setModelListOptions
} from "./ui.js";

/* ======================================================================
   GENERATION SETTINGS — Default GPT-2 (richiesto)
====================================================================== */
let generationSettings = {
  temperature: 0.3,
  max_tokens: 200,
  top_p: 0.95,
  top_k: 40,
  repetition_penalty: 1.15
};

/* ======================================================================
   HF TOKEN STORAGE
====================================================================== */
function getHfToken() {
  return localStorage.getItem("hf_token") || "";
}

function saveHfToken(token) {
  if (token) localStorage.setItem("hf_token", token);
  else localStorage.removeItem("hf_token");
}

/* ======================================================================
   SETTINGS PANEL — LOGICA COMPLETA
====================================================================== */
function initSettingsPanel() {
  const settingsBtn = document.getElementById("settings-btn");
  const settingsPanel = document.getElementById("settings-panel");
  const settingsClose = document.getElementById("settings-close");

  // Slider temperatura
  const tempSlider = document.getElementById("temperature-slider");
  const tempValue = document.getElementById("temperature-value");

  // Input numerici
  const maxTokensInput = document.getElementById("max-tokens-input");
  const topPInput = document.getElementById("top-p-input");
  const topKInput = document.getElementById("top-k-input");
  const repPenaltyInput = document.getElementById("repetition-penalty-input");

  // Compute mode toggle
  const btnCpu = document.getElementById("btn-cpu");
  const btnGpu = document.getElementById("btn-gpu");

  /* OPEN/CLOSE PANEL */
  settingsBtn.addEventListener("click", () => {
    settingsPanel.classList.toggle("open");
  });

  settingsClose.addEventListener("click", () => {
    settingsPanel.classList.remove("open");
  });

  // Chiudi cliccando fuori
  document.addEventListener("click", (e) => {
    if (!settingsPanel.contains(e.target) &&
        !settingsBtn.contains(e.target)) {
      settingsPanel.classList.remove("open");
    }
  });

  /* SLIDER TEMPERATURE */
  tempSlider.addEventListener("input", () => {
    const val = parseFloat(tempSlider.value);
    tempValue.textContent = val.toFixed(2);
    generationSettings.temperature = val;
  });

  /* INPUT NUMERICI */
  maxTokensInput.addEventListener("change", () => {
    generationSettings.max_tokens = parseInt(maxTokensInput.value, 10);
  });

  topPInput.addEventListener("change", () => {
    generationSettings.top_p = parseFloat(topPInput.value);
  });

  topKInput.addEventListener("change", () => {
    generationSettings.top_k = parseInt(topKInput.value, 10);
  });

  repPenaltyInput.addEventListener("change", () => {
    generationSettings.repetition_penalty =
      parseFloat(repPenaltyInput.value);
  });

  /* CPU / GPU TOGGLE */
  btnCpu.addEventListener("click", async () => {
    try {
      const res = await setComputeModeApi("cpu");
      setComputeMode(res.mode);
      btnCpu.classList.add("active");
      btnGpu.classList.remove("active");
    } catch (e) {
      alert("Errore cambio compute mode: " + e.message);
    }
  });

  btnGpu.addEventListener("click", async () => {
    try {
      const res = await setComputeModeApi("gpu");
      setComputeMode(res.mode);
      btnGpu.classList.add("active");
      btnCpu.classList.remove("active");
    } catch (e) {
      alert("Errore cambio compute mode: " + e.message);
    }
  });
}

/* ======================================================================
   HANDLE SEND — invio messaggi + parametri generazione
====================================================================== */
async function handleSend() {
  const input = document.getElementById("user-input");
  const text = input.value.trim();
  if (!text) return;

  const model = getCurrentModel();
  if (!model) {
    alert("Seleziona un modello");
    return;
  }

  // Aggiungi messaggio user a UI
  pushMessage({ role: "user", content: text });
  addMessage("user", text);

  input.value = "";
  resizeInputWrapper();

  const assistantBubble = addMessage("assistant", "");
  let fullReply = "";

  // Apertura WS
  openChatWebSocket({
    model,
    message: text,
    compute_mode: getComputeMode(),

    // 🔥 Parametri generazione
    ...generationSettings,

    onChunk: (chunk) => {
      fullReply += chunk;
      assistantBubble.querySelector(".message-content").textContent = fullReply;
    },

    onDone: () => {
      pushMessage({ role: "assistant", content: fullReply });
    },

    onError: (err) => {
      assistantBubble.querySelector(".message-content").textContent =
        "Errore: " + err;
    },
  });
}

/* ======================================================================
   INIT — Avvio completo UI + history + settings
====================================================================== */
async function init() {
  resizeInputWrapper();

  /* --- MODELS --- */
  let models = [];
  try {
    const data = await fetchModels();
    models = data.models || [];
  } catch {}

  /* --- HISTORY --- */
  try {
    const hist = await fetchHistory();

    if (hist.model && models.includes(hist.model)) {
      setCurrentModel(hist.model);
    } else if (!getCurrentModel() && models.length > 0) {
      setCurrentModel(models[0]);
    }

    setMessages(hist.messages || []);
    renderMessages(getMessages());
  } catch {
    if (!getCurrentModel() && models.length > 0) {
      setCurrentModel(models[0]);
    }
  }

  /* --- MODEL SELECT UI --- */
  setModelListOptions(models, getCurrentModel());

  const modelSelect = document.getElementById("model-list");
  modelSelect.addEventListener("change", () => {
    setCurrentModel(modelSelect.value);
  });

  /* --- SYSTEM STATUS (compute mode) --- */
  try {
    const status = await fetchSystemStatus();
    setComputeMode(status.compute_mode || "gpu");
  } catch {}

  /* --- SETTINGS PANEL --- */
  initSettingsPanel();

  /* --- CHAT INPUT HANDLERS --- */
  bindChatInputHandlers(handleSend);

  /* --- CLEAR CHAT --- */
  bindClearChat(async () => {
    await clearHistoryApi();
    clearMessages();
    location.reload();
  });

  /* --- DOWNLOAD MODEL --- */
  document.getElementById("download-btn").addEventListener("click", async () => {
    const repoId = document.getElementById("model-id-input").value.trim();
    if (!repoId) return;

    try {
      await downloadModel(repoId, getHfToken());
      const data = await fetchModels();
      setModelListOptions(data.models, repoId);
      setCurrentModel(repoId);
    } catch (e) {
      alert("Errore durante il download: " + e.message);
    }
  });
}

init();
