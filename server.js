require("dotenv").config();

const express = require("express");
const path = require("path");
const multer = require("multer");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 5 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (file.mimetype && file.mimetype.startsWith("image/")) {
      return cb(null, true);
    }
    cb(new Error("Images only are supported."));
  }
});

const GEMINI_KEY = process.env.T4U_GOOGLE_AI_SECRET;
const OPENAI_KEY = process.env.T4U_OPENAI_SECRET;

const CACHE_TTL = 10 * 60 * 1000;
const REQUEST_TIMEOUT = 15000;
const MAX_ATTEMPTS = 3;
const MAX_OUTPUT_TOKENS = 1024;

const modelCache = {
  gemini: { models: [], expires: 0 },
  openai: { models: [], expires: 0 }
};

const SYSTEM_PROMPT = `
You are TETRON 4 U, a helpful AI assistant.
Give clear, accurate, friendly answers.
Keep simple questions concise.
For programming questions, provide working code and explain important steps.
If you are unsure, be honest.
Do not invent facts.
`;

function timeoutSignal() {
  return AbortSignal.timeout(REQUEST_TIMEOUT);
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    signal: timeoutSignal()
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const error = new Error(
      data?.error?.message ||
      data?.error?.status ||
      `API request failed (${response.status})`
    );

    error.status = response.status;
    throw error;
  }

  return data;
}

/* ---------------- GEMINI MODEL DISCOVERY ---------------- */

async function getGeminiModels(forceRefresh = false) {
  if (!GEMINI_KEY) return [];

  if (
    !forceRefresh &&
    modelCache.gemini.expires > Date.now() &&
    modelCache.gemini.models.length
  ) {
    return modelCache.gemini.models;
  }

  try {
    const data = await fetchJSON(
      `https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000&key=${encodeURIComponent(GEMINI_KEY)}`
    );

    const models = (data.models || [])
      .filter(model => {
        const name = (model.name || "").toLowerCase();
        const methods = model.supportedGenerationMethods || [];

        return (
          methods.includes("generateContent") &&
          /gemini/.test(name) &&
          /(flash|pro)/.test(name) &&
          !/(preview|experimental|exp-|embedding|tts|image-generation)/.test(name)
        );
      })
      .sort((a, b) => {
        const score = model => {
          const name = model.name.toLowerCase();

          if (/gemini-2\.5-flash/.test(name)) return 1;
          if (/gemini-2\.0-flash/.test(name)) return 2;
          if (/flash/.test(name)) return 3;
          if (/pro/.test(name)) return 4;

          return 10;
        };

        return score(a) - score(b);
      })
      .map(model => ({
        id: model.name.replace(/^models\//, ""),
        provider: "gemini"
      }));

    modelCache.gemini = {
      models,
      expires: Date.now() + CACHE_TTL
    };

    return models;
  } catch (error) {
    console.error("Gemini model discovery failed:", error.message);

    if (modelCache.gemini.models.length) {
      return modelCache.gemini.models;
    }

    return [];
  }
}

/* ---------------- OPENAI MODEL DISCOVERY ---------------- */

const OPENAI_PRIORITY = [
  "gpt-4o-mini",
  "gpt-4.1-mini",
  "gpt-4.1-nano",
  "gpt-4o",
  "gpt-4.1"
];

function openAIModelScore(id) {
  const name = id.toLowerCase();
  const exactIndex = OPENAI_PRIORITY.indexOf(name);

  if (exactIndex !== -1) return exactIndex;

  if (!/^gpt-4/.test(name)) return 100;

  if (/(preview|realtime|audio|transcribe|tts|embedding|search|image|instruct)/.test(name)) {
    return 200;
  }

  return 50;
}

async function getOpenAIModels(forceRefresh = false) {
  if (!OPENAI_KEY) return [];

  if (
    !forceRefresh &&
    modelCache.openai.expires > Date.now() &&
    modelCache.openai.models.length
  ) {
    return modelCache.openai.models;
  }

  try {
    const data = await fetchJSON("https://api.openai.com/v1/models", {
      headers: {
        Authorization: `Bearer ${OPENAI_KEY}`
      }
    });

    const models = (data.data || [])
      .map(model => model.id)
      .filter(id => {
        const name = id.toLowerCase();

        return (
          /^gpt-4/.test(name) &&
          !/(preview|realtime|audio|transcribe|tts|embedding|search|image|instruct)/.test(name)
        );
      })
      .sort((a, b) => openAIModelScore(a) - openAIModelScore(b))
      .map(id => ({
        id,
        provider: "openai"
      }));

    modelCache.openai = {
      models,
      expires: Date.now() + CACHE_TTL
    };

    return models;
  } catch (error) {
    console.error("OpenAI model discovery failed:", error.message);

    if (modelCache.openai.models.length) {
      return modelCache.openai.models;
    }

    return [];
  }
}

/* ---------------- MODEL SUPPORT ---------------- */

function supportsOpenAIImages(modelId) {
  return /^(gpt-4o-mini|gpt-4o|gpt-4\.1-mini|gpt-4\.1-nano|gpt-4\.1)$/i
    .test(modelId);
}

function supportsGeminiImages(modelId) {
  return /(flash|pro)/i.test(modelId);
}

async function getCandidates(provider, hasImage) {
  let models = [];

  if (provider === "gemini") {
    models = await getGeminiModels();
  } else {
    models = await getOpenAIModels();
  }

  if (hasImage) {
    models = models.filter(model =>
      model.provider === "gemini"
        ? supportsGeminiImages(model.id)
        : supportsOpenAIImages(model.id)
    );
  }

  return models;
}

/* ---------------- GEMINI CHAT ---------------- */

async function askGemini(model, message, history, image) {
  const contents = [];

  for (const item of history) {
    if (!item || !["user", "assistant"].includes(item.role)) continue;

    if (typeof item.content === "string" && item.content.trim()) {
      contents.push({
        role: item.role === "assistant" ? "model" : "user",
        parts: [{ text: item.content.slice(0, 4000) }]
      });
    }
  }

  const parts = [{ text: message }];

  if (image) {
    parts.push({
      inline_data: {
        mime_type: image.mimetype,
        data: image.buffer.toString("base64")
      }
    });
  }

  contents.push({
    role: "user",
    parts
  });

  const data = await fetchJSON(
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent?key=${encodeURIComponent(GEMINI_KEY)}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        systemInstruction: {
          parts: [{ text: SYSTEM_PROMPT }]
        },
        contents,
        generationConfig: {
          temperature: 0.5,
          maxOutputTokens: MAX_OUTPUT_TOKENS
        }
      })
    }
  );

  const answer = (data.candidates?.[0]?.content?.parts || [])
    .map(part => part.text || "")
    .join("")
    .trim();

  if (!answer) throw new Error("Gemini returned an empty response.");

  return answer;
}

/* ---------------- OPENAI CHAT ---------------- */

async function askOpenAI(model, message, history, image) {
  const messages = [
    { role: "system", content: SYSTEM_PROMPT }
  ];

  for (const item of history) {
    if (!item || !["user", "assistant"].includes(item.role)) continue;

    if (typeof item.content === "string" && item.content.trim()) {
      messages.push({
        role: item.role,
        content: item.content.slice(0, 4000)
      });
    }
  }

  if (image) {
    messages.push({
      role: "user",
      content: [
        { type: "text", text: message || "Describe this image." },
        {
          type: "image_url",
          image_url: {
            url: `data:${image.mimetype};base64,${image.buffer.toString("base64")}`,
            detail: "low"
          }
        }
      ]
    });
  } else {
    messages.push({
      role: "user",
      content: message
    });
  }

  const data = await fetchJSON("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model,
      messages,
      max_completion_tokens: MAX_OUTPUT_TOKENS
    })
  });

  const answer = data.choices?.[0]?.message?.content?.trim();

  if (!answer) throw new Error("OpenAI returned an empty response.");

  return answer;
}

/* ---------------- HEALTH CHECK ---------------- */

app.get("/api/health", (req, res) => {
  res.json({
    success: true,
    app: "TETRON 4 U",
    providers: {
      gemini: Boolean(GEMINI_KEY),
      openai: Boolean(OPENAI_KEY)
    }
  });
});

/* ---------------- AVAILABLE MODELS ---------------- */

app.get("/api/models", async (req, res) => {
  const provider = (req.query.provider || "auto").toLowerCase();

  const result = {};

  if (provider === "auto" || provider === "gemini") {
    result.gemini = (await getGeminiModels()).map(model => model.id);
  }

  if (provider === "auto" || provider === "openai" || provider === "chatgpt") {
    result.openai = (await getOpenAIModels()).map(model => model.id);
  }

  res.json({
    success: true,
    models: result
  });
});

/* ---------------- CHAT API ---------------- */

app.post("/api/chat", upload.single("image"), async (req, res) => {
  try {
    const message = String(req.body.message || "").trim();

    const providerPreference = String(
      req.body.provider || "auto"
    ).toLowerCase();

    let history = [];

    try {
      const parsed = typeof req.body.history === "string"
        ? JSON.parse(req.body.history)
        : req.body.history;

      if (Array.isArray(parsed)) {
        history = parsed.slice(-6).map(item => ({
          role: item.role,
          content: String(item.content ?? item.text ?? "").slice(0, 4000)
        }));
      }
    } catch {
      history = [];
    }

    const image = req.file
      ? {
          buffer: req.file.buffer,
          mimetype: req.file.mimetype
        }
      : null;

    if (!message && !image) {
      return res.status(400).json({
        success: false,
        error: "Please enter a message or upload an image."
      });
    }

    let providerOrder;

    if (["openai", "chatgpt"].includes(providerPreference)) {
      providerOrder = ["openai", "gemini"];
    } else {
      providerOrder = ["gemini", "openai"];
    }

    const candidateLists = {};

    for (const provider of providerOrder) {
      candidateLists[provider] = await getCandidates(provider, Boolean(image));
    }

    // Try each provider first, then one alternate model if needed.
    const candidates = [];

    for (let index = 0; index < 3; index++) {
      for (const provider of providerOrder) {
        const model = candidateLists[provider]?.[index];

        if (model && candidates.length < MAX_ATTEMPTS) {
          candidates.push(model);
        }
      }

      if (candidates.length >= MAX_ATTEMPTS) break;
    }

    if (!candidates.length) {
      return res.status(503).json({
        success: false,
        error: "No compatible model is available. Check your API keys and model access."
      });
    }

    let lastError;

    for (const candidate of candidates) {
      try {
        console.log(`Trying ${candidate.provider}: ${candidate.id}`);

        const answer = candidate.provider === "gemini"
          ? await askGemini(candidate.id, message, history, image)
          : await askOpenAI(candidate.id, message, history, image);

        console.log(`Success: ${candidate.provider} / ${candidate.id}`);

        return res.json({
          success: true,
          provider: candidate.provider,
          model: candidate.id,
          answer
        });
      } catch (error) {
        lastError = error;

        console.error(
          `Model failed (${candidate.provider}/${candidate.id}):`,
          error.message
        );
      }
    }

    return res.status(502).json({
      success: false,
      error: "All available models failed. Check API quota, key access, or try again."
    });
  } catch (error) {
    console.error("Chat error:", error.message);

    return res.status(500).json({
      success: false,
      error: "Something went wrong. Please try again."
    });
  }
});

/* ---------------- ERROR HANDLER ---------------- */

app.use((error, req, res, next) => {
  console.error("Request error:", error.message);

  if (error instanceof multer.MulterError) {
    return res.status(400).json({
      success: false,
      error: "Image upload failed. Maximum image size is 5 MB."
    });
  }

  res.status(400).json({
    success: false,
    error: error.message || "Invalid request."
  });
});

/* ---------------- START SERVER ---------------- */

app.listen(PORT, () => {
  console.log(`TETRON 4 U running on port ${PORT}`);
});
