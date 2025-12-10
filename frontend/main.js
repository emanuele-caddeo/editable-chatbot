// main.js
import { fetchModels, fetchHistory, clearHistory, openChatWebSocket, fetchSystemStatus, setComputeModeApi } from "./api.js";
import { getCurrentModel, setCurrentModel, getMessages, pushMessage, clearMessages, getComputeMode, setComputeMode } from "./state.js";
import { addMessage, getUserInputText, clearUserInput, bindInputHandlers, bindClearChat, resizeInputWrapper, setModelListOptions, bindModelChange } from "./ui.js";

async function init() {
  resizeInputWrapper();

  // ============================
  // 1) Carica modelli
  // ============================
  const modelsData = await fetchModels();
  const models = modelsData.models || [];

  // NON impostare ancora il modello qui.
  // Prima leggi la history (fix del bug)

  // ============================
  // 2) Carica history
  // ============================
  let histModel = null;
  try {
    const hist = await fetchHistory();
    if (hist.model) histModel = hist.model;

    if (Array.isArray(hist.messages)) {
      hist.messages.forEach((m) => {
        pushMessage(m);
        addMessage(m.role, m.content);
      });
    }
  } catch (e) {
    console.warn("Nessuna history presente");
  }

  // ============================
  // 3) Imposta il modello effettivo
  // ============================
  if (histModel && models.includes(histModel)) {
    // Usa il modello letto da chat.json
    setCurrentModel(histModel);
  } else {
    // Fallback: scegli il primo modello disponibile SOLO se la history non ne aveva uno
    if (models.length > 0) {
      setCurrentModel(models[models.length - 1]); // impossto l'ultimo modello come predefinito
    }
  }

  // Aggiorna UI con il modello corretto
  setModelListOptions(models, getCurrentModel());
  bindModelChange((model) => setCurrentModel(model));

  // ============================
  // 4) System status
  // ============================
  try {
    const status = await fetchSystemStatus();
    setComputeMode(status.compute_mode || "gpu");
    // eventuale UI CPU/GPU/offload...
  } catch (e) {
    console.warn("Errore system status", e);
  }

  // ============================
  // 5) Bind invio messaggi
  // ============================
  bindInputHandlers(handleSend);

  // ============================
  // 6) Clear chat
  // ============================
  bindClearChat(async () => {
    await clearHistory();
    clearMessages();
    location.reload();
  });
}

async function handleSend() {
  const text = getUserInputText();
  if (!text) return;

  const model = getCurrentModel();
  if (!model) {
    alert("Seleziona un modello");
    return;
  }

  // aggiorno stato + UI locale
  pushMessage({ role: "user", content: text });
  addMessage("user", text);
  clearUserInput();
  addMessage("assistant", "");
  const assistantBubble = document.querySelector(".message.assistant:last-child");

  let fullReply = "";

  openChatWebSocket({
    model,
    message: text,
    computeMode: getComputeMode(),
    onChunk: (chunk) => {
      fullReply += chunk;
      assistantBubble.textContent += chunk;
    },
    onDone: () => {
      pushMessage({ role: "assistant", content: fullReply });
    },
    onError: (msg) => {
      assistantBubble.textContent = "Errore: " + msg;
    },
  });
}

init();
