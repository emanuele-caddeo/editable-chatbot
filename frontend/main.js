/********************************************************************
 * MAIN.JS — con:
 *  - Salvataggio automatico dell’ultimo modello selezionato
 *  - Apertura/chiusura pannello impostazioni
 *  - Apertura/chiusura pannello token HF
 *  - Parametri generazione modello
 *  - Compute mode CPU/GPU
 *  - Chat WebSocket con streaming
 *  - Gestione history
 ********************************************************************/

import {
  fetchModels,
  fetchHistory,
  clearHistoryApi,
  fetchSystemStatus,
  setComputeModeApi,
  openChatWebSocket,
  downloadModel,
  saveHistory
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
   GENERATION SETTINGS
====================================================================== */
let generationSettings = {
  temperature: 0.3,
  max_tokens: 200,
  top_p: 0.95,
  top_k: 40,
  repetition_penalty: 1.15,
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
   SETTINGS PANEL
====================================================================== */
function initSettingsPanel() {
  const settingsBtn = document.getElementById("settings-btn");
  const settingsPanel = document.getElementById("settings-panel");
  const settingsClose = document.getElementById("settings-close");

  const tempSlider = document.getElementById("temperature-slider");
  const tempValue = document.getElementById("temperature-value");

  const maxTokensInput = document.getElementById("max-tokens-input");
  const topPInput = document.getElementById("top-p-input");
  const topKInput = document.getElementById("top-k-input");
  const repPenaltyInput = document.getElementById("repetition-penalty-input");

  const btnCpu = document.getElementById("btn-cpu");
  const btnGpu = document.getElementById("btn-gpu");

  /* Apertura/chiusura */
  settingsBtn.addEventListener("click", () => {
    settingsPanel.classList.toggle("open");
  });

  settingsClose.addEventListener("click", () => {
    settingsPanel.classList.remove("open");
  });

  document.addEventListener("click", (e) => {
    if (!settingsPanel.contains(e.target) && !settingsBtn.contains(e.target)) {
      settingsPanel.classList.remove("open");
    }
  });

  /* Slider temperatura */
  tempSlider.addEventListener("input", () => {
    const val = parseFloat(tempSlider.value);
    tempValue.textContent = val.toFixed(2);
    generationSettings.temperature = val;
  });

  /* Parametri generazione */
  maxTokensInput.addEventListener("change", () => {
    generationSettings.max_tokens = parseInt(maxTokensInput.value);
  });

  topPInput.addEventListener("change", () => {
    generationSettings.top_p = parseFloat(topPInput.value);
  });

  topKInput.addEventListener("change", () => {
    generationSettings.top_k = parseInt(topKInput.value);
  });

  repPenaltyInput.addEventListener("change", () => {
    generationSettings.repetition_penalty = parseFloat(repPenaltyInput.value);
  });

  /* Compute mode toggle */
  btnCpu.addEventListener("click", async () => {
    try {
      const res = await setComputeModeApi("cpu");
      setComputeMode(res.mode);
      btnCpu.classList.add("active");
      btnGpu.classList.remove("active");
    } catch (e) {
      alert("Errore: " + e.message);
    }
  });

  btnGpu.addEventListener("click", async () => {
    try {
      const res = await setComputeModeApi("gpu");
      setComputeMode(res.mode);
      btnGpu.classList.add("active");
      btnCpu.classList.remove("active");
    } catch (e) {
      alert("Errore: " + e.message);
    }
  });
}

/* ======================================================================
   HF TOKEN PANEL
====================================================================== */
function initHfTokenPanel() {
  const btn = document.getElementById("hf-token-btn");
  const panel = document.getElementById("hf-token-panel");
  const closeBtn = document.getElementById("hf-token-close");
  const input = document.getElementById("hf-token-input");
  const saveBtn = document.getElementById("hf-token-save");
  const clearBtn = document.getElementById("hf-token-clear");

  input.value = getHfToken();

  btn.addEventListener("click", () => {
    panel.classList.toggle("open");
  });

  closeBtn.addEventListener("click", () => {
    panel.classList.remove("open");
  });

  document.addEventListener("click", (e) => {
    if (!panel.contains(e.target) && !btn.contains(e.target)) {
      panel.classList.remove("open");
    }
  });

  saveBtn.addEventListener("click", () => {
    saveHfToken(input.value.trim());
    panel.classList.remove("open");
  });

  clearBtn.addEventListener("click", () => {
    saveHfToken("");
    input.value = "";
    panel.classList.remove("open");
  });
}

/* ======================================================================
   SEND MESSAGE
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

  pushMessage({ role: "user", content: text });
  addMessage("user", text);

  input.value = "";
  resizeInputWrapper();

  const assistantBubble = addMessage("assistant", "");
  let fullReply = "";

  openChatWebSocket({
    model,
    message: text,
    compute_mode: getComputeMode(),
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
   INIT
====================================================================== */
async function init() {
  resizeInputWrapper();

  /* Carica modelli */
  let models = [];
  try {
    const data = await fetchModels();
    models = data.models || [];
  } catch {}

  /* Carica history */
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

  /* Popola select modelli */
  setModelListOptions(models, getCurrentModel());

  const modelSelect = document.getElementById("model-list");
  modelSelect.addEventListener("change", async () => {
    const model = modelSelect.value;
    setCurrentModel(model);

    try {
      await saveHistory(model, getMessages());
    } catch (e) {
      console.error("Errore salvataggio modello:", e);
    }
  });

  /* Compute mode */
  try {
    const status = await fetchSystemStatus();
    setComputeMode(status.compute_mode || "gpu");
  } catch {}

  initSettingsPanel();
  initHfTokenPanel();

  bindChatInputHandlers(handleSend);

  bindClearChat(async () => {
    await clearHistoryApi();
    clearMessages();
    location.reload();
  });

  /* Download modello */
  const downloadBtn = document.getElementById("download-btn");
  downloadBtn.addEventListener("click", async () => {
    const repoInput = document.getElementById("model-id-input");
    const repoId = repoInput.value.trim();
    if (!repoId) return;

    try {
      await downloadModel(repoId, getHfToken());
      const data = await fetchModels();
      setModelListOptions(data.models, repoId);
      setCurrentModel(repoId);

      await saveHistory(repoId, getMessages());
    } catch (e) {
      alert("Errore download modello: " + e.message);
    }
  });
}

init();
