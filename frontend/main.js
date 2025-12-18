/********************************************************************
 * main.js — features:
 *  - Auto-save the last selected model
 *  - Open/close the Settings panel
 *  - Open/close the HF token panel
 *  - Model generation parameters
 *  - CPU/GPU compute mode
 *  - Streaming WebSocket chat
 *  - Chat history management
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
  setComputeMode,
  getSystemBusy,
  setSystemBusy
} from "./state.js";

import {
  addMessage,
  renderMessages,
  bindChatInputHandlers,
  bindClearChat,
  resizeInputWrapper,
  setModelListOptions,
  setChatInputLocked,
  showSystemStatus,
  hideSystemStatus
} from "./ui.js";
let uiLocked = false;

/* ======================================================================
   GENERATION SETTINGS
====================================================================== */
let generationSettings = {
  temperature: 0.15,
  max_tokens: 10,
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

  tempSlider.addEventListener("input", () => {
    const val = parseFloat(tempSlider.value);
    tempValue.textContent = val.toFixed(2);
    generationSettings.temperature = val;
  });

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

  btnCpu.addEventListener("click", async () => {
    try {
      const res = await setComputeModeApi("cpu");
      setComputeMode(res.mode);
      btnCpu.classList.add("active");
      btnGpu.classList.remove("active");
    } catch (e) {
      alert("Error: " + e.message);
    }
  });

  btnGpu.addEventListener("click", async () => {
    try {
      const res = await setComputeModeApi("gpu");
      setComputeMode(res.mode);
      btnGpu.classList.add("active");
      btnCpu.classList.remove("active");
    } catch (e) {
      alert("Error: " + e.message);
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
  if (uiLocked) return;

  const input = document.getElementById("user-input");
  const sendBtn = document.getElementById("send-btn");
  if (!input) return;

  const text = (input.value || "").trim();
  if (!text) return;

  const model = getCurrentModel();
  if (!model) {
    alert("Select a model");
    return;
  }

  const lower = text.toLowerCase();

  /* =====================================================
     SYSTEM COMMANDS: confirm / cancel
  ===================================================== */
  if (lower === "confirm" || lower === "cancel") {
    uiLocked = true; // 🔒 IMMEDIATE, SYNC LOCK

    const statusMsg =
      lower === "confirm"
        ? "Applying knowledge edit (ROME)..."
        : "Cancelling knowledge edit...";

    // Show user message only
    pushMessage({ role: "user", content: text });
    addMessage("user", text);

    input.value = "";
    resizeInputWrapper();

    // Hard UI lock (NO WS DEPENDENCY)
    setChatInputLocked(true, statusMsg);
    showSystemStatus(statusMsg);

    openChatWebSocket({
      model,
      message: text,
      compute_mode: getComputeMode(),
      ...generationSettings,

      onMessage: (msg) => {
        if (msg.type === "system" && msg.state === "ready") {
          uiLocked = false;
          setChatInputLocked(false);
          hideSystemStatus();
        }

        if (msg.type === "error") {
          uiLocked = false;
          setChatInputLocked(false);
          hideSystemStatus();
          addMessage("assistant", "Error: " + (msg.message || "Unknown error"));
        }
      },

      onError: (err) => {
        uiLocked = false;
        setChatInputLocked(false);
        hideSystemStatus();
        addMessage("assistant", "Error: " + err);
      },
    });

    return;
  }

  /* =====================================================
     NORMAL CHAT MESSAGE
  ===================================================== */
  uiLocked = true; // 🔒 LOCK UNTIL DONE

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

    onMessage: (msg) => {
      if (msg.type === "chunk") {
        fullReply += msg.content || "";
        assistantBubble.querySelector(".message-content").textContent = fullReply;
      }

      if (msg.type === "system" && msg.state === "busy") {
        setChatInputLocked(true, msg.message || "Please wait...");
        showSystemStatus(msg.message || "Please wait...");
      }

      if (msg.type === "error") {
        uiLocked = false;
        setChatInputLocked(false);
        hideSystemStatus();
        assistantBubble.querySelector(".message-content").textContent =
          "Error: " + (msg.message || "Unknown error");
      }
    },

    onDone: () => {
      uiLocked = false;
      setChatInputLocked(false);
      hideSystemStatus();
      pushMessage({ role: "assistant", content: fullReply });
    },

    onError: (err) => {
      uiLocked = false;
      setChatInputLocked(false);
      hideSystemStatus();
      assistantBubble.querySelector(".message-content").textContent =
        "Error: " + err;
    },
  });
}


/* ======================================================================
   INIT
====================================================================== */
async function init() {
  resizeInputWrapper();

  /* Load models */
  let models = [];
  try {
    const data = await fetchModels();
    models = data.models || [];
  } catch {}

  /* Load history */
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

  /* Populate model select */
  setModelListOptions(models, getCurrentModel());

  const modelSelect = document.getElementById("model-list");
  modelSelect.addEventListener("change", async () => {
    const model = modelSelect.value;
    setCurrentModel(model);

    try {
      await saveHistory(model, getMessages());
    } catch (e) {
      console.error("Model save error:", e);
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

  /* Download model */
  const downloadBtn = document.getElementById("download-btn");
  downloadBtn.addEventListener("click", async () => {
  try {
    const repoInput = document.getElementById("model-id-input");
    const repoId = repoInput.value.trim();

    if (!repoId) {
      alert("Insert a HuggingFace repo id (e.g. openai-community/gpt2-xl)");
      return;
    }

    await downloadModel(repoId, getHfToken());
    alert("Model downloaded successfully.");

    // Optional: refresh model list after download
    const data = await fetchModels();
    setModelListOptions(data.models || [], getCurrentModel());

  } catch (e) {
    alert("Model download error: " + e.message);
  }
});

}

init();
