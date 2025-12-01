// main.js
import { fetchModels, fetchHistory, clearHistory, openChatWebSocket, fetchSystemStatus, setComputeModeApi } from "./api.js";
import { getCurrentModel, setCurrentModel, getMessages, pushMessage, clearMessages, getComputeMode, setComputeMode } from "./state.js";
import { addMessage, getUserInputText, clearUserInput, bindInputHandlers, bindClearChat, resizeInputWrapper, setModelListOptions, bindModelChange } from "./ui.js";

async function init() {
  resizeInputWrapper();

  // carica modelli
  const modelsData = await fetchModels();
  const models = modelsData.models || [];
  if (models.length > 0 && !getCurrentModel()) {
    setCurrentModel(models[0]);
  }
  setModelListOptions(models, getCurrentModel());
  bindModelChange((model) => setCurrentModel(model));

  // carica history
  try {
    const hist = await fetchHistory();
    if (hist.model) setCurrentModel(hist.model);
    if (Array.isArray(hist.messages)) {
      hist.messages.forEach((m) => {
        pushMessage(m);
        addMessage(m.role, m.content);
      });
    }
  } catch (e) {
    console.warn("Nessuna history presente");
  }

  // system status
  try {
    const status = await fetchSystemStatus();
    setComputeMode(status.compute_mode || "gpu");
    // qui aggiorni UI CPU/GPU/offload...
  } catch (e) {
    console.warn("Errore system status", e);
  }

  // bind invio messaggio
  bindInputHandlers(handleSend);
  // bind pulisci chat
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
