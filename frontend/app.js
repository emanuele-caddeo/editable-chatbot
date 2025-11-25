const API_BASE = "http://localhost:8000";

const chatContainer = document.getElementById("chat-container");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const modelList = document.getElementById("model-list");
const modelIdInput = document.getElementById("model-id-input");
const downloadBtn = document.getElementById("download-btn");

let currentModel = null;
let messages = [];

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
    const res = await fetch(`${API_BASE}/api/models/download`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ repo_id: repoId }),
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

// init
loadModels();
