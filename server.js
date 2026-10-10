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

const REQUEST_TIMEOUT = 25000;
const MAX_OUTPUT_TOKENS = 1024;
const MAX_HISTORY = 6;

const SYSTEM_PROMPT = `
You are TETRON 4 U, a helpful AI assistant.
Give clear, accurate, friendly answers.
Keep simple questions concise.
For programming questions, provide working code and explain important steps.
If unsure, be honest. Do not invent facts.
`;

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    signal: AbortSignal.timeout(REQUEST_TIMEOUT)
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(
      data?.error?.message ||
      data?.message ||
      `API request failed (${response.status})`
    );
  }

  return data;
}

/* GEMINI MODEL PRIORITY */

const GEMINI_MODELS = [
  "gemini-3.8-flash",
  "gemini-3.6-flash",
  "gemini-3.5-flash-lite",
  "gemini-flash-latest",
  "gemini-flash-lite-latest"
];

const OPENAI_MODELS = [
  "gpt-4o-mini",
  "gpt-4.1-mini",
  "gpt-4.1-nano",
  "gpt-4o",
  "gpt-4.1"
];

/* GEMINI CHAT */

async function askGemini(model, message, history, image) {
  // Text requests use the Gemini Interactions API.
  if (!image) {
    const conversation = history
      .filter(item =>
        item &&
        ["user", "assistant"].includes(item.role) &&
        typeof item.content === "string"
      )
      .slice(-MAX_HISTORY)
      .map(item =>
        `${item.role === "assistant" ? "Assistant" : "User"}: ${
          item.content.slice(0, 3000)
        }`
      );

    conversation.push(`User: ${message}`);

    const data = await fetchJSON(
      "https://generativelanguage.googleapis.com/v1beta/interactions",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-goog-api-key": GEMINI_KEY
        },
        body: JSON.stringify({
          model,
          input: conversation.join("\n\n"),
          system_instruction: SYSTEM_PROMPT,
          generation_config: {
            max_output_tokens: MAX_OUTPUT_TOKENS,
            thinking_level: "low"
          },
          store: false
        })
      }
    );

    let answer = data.output_text || "";

    if (!answer && Array.isArray(data.outputs)) {
      answer = data.outputs
        .map(item => item.text || "")
        .join("\n");
    }

    if (!answer.trim()) {
      throw new Error("Gemini returned an empty response.");
    }

    return answer.trim();
  }

  // Image requests use generateContent.
  const contents = [];

  for (const item of history.slice(-MAX_HISTORY)) {
    if (
      item &&
      ["user", "assistant"].includes(item.role) &&
      typeof item.content === "string" &&
      item.content.trim()
    ) {
      contents.push({
        role: item.role === "assistant" ? "model" : "user",
        parts: [{ text: item.content.slice(0, 3000) }]
      });
    }
  }

  const parts = [
    { text: message || "Describe this image." },
    {
      inline_data: {
        mime_type: image.mimetype,
        data: image.buffer.toString("base64")
      }
    }
  ];

  contents.push({ role: "user", parts });

  const imageModel = "gemini-3.5-flash-lite";

  const data = await fetchJSON(
    `https://generativelanguage.googleapis.com/v1beta/models/${imageModel}:generateContent`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_KEY
      },
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

  if (!answer) {
    throw new Error("Gemini returned an empty response.");
  }

  return answer;
}

/* OPENAI CHAT */

async function askOpenAI(model, message, history, image) {
  const messages = [
    { role: "system", content: SYSTEM_PROMPT }
  ];

  for (const item of history.slice(-MAX_HISTORY)) {
    if (
      item &&
      ["user", "assistant"].includes(item.role) &&
      typeof item.content === "string" &&
      item.content.trim()
    ) {
      messages.push({
        role: item.role,
        content: item.content.slice(0, 3000)
      });
    }
  }

  if (image) {
    messages.push({
      role: "user",
      content: [
        {
          type: "text",
          text: message || "Describe this image."
        },
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

  const data = await fetchJSON(
    "https://api.openai.com/v1/chat/completions",
    {
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
    }
  );

  const answer = data.choices?.[0]?.message?.content?.trim();

  if (!answer) {
    throw new Error("OpenAI returned an empty response.");
  }

  return answer;
}

/* HEALTH CHECK */

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

/* AVAILABLE MODELS */

app.get("/api/models", (req, res) => {
  res.json({
    success: true,
    models: {
      gemini: GEMINI_KEY ? GEMINI_MODELS : [],
      openai: OPENAI_KEY ? OPENAI_MODELS : []
    }
  });
});

/* CHAT API */

app.post("/api/chat", upload.single("image"), async (req, res) => {
  const message = String(req.body.message || "").trim();
  const preference = String(req.body.provider || "auto").toLowerCase();

  let history = [];

  try {
    const parsed = typeof req.body.history === "string"
      ? JSON.parse(req.body.history)
      : req.body.history;

    if (Array.isArray(parsed)) {
      history = parsed.slice(-MAX_HISTORY).map(item => ({
        role: item.role,
        content: String(item.content ?? item.text ?? "").slice(0, 3000)
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

  const providers = ["openai", "chatgpt"].includes(preference)
    ? ["openai", "gemini"]
    : ["gemini", "openai"];

  const candidates = [];

  for (const provider of providers) {
    if (provider === "gemini" && GEMINI_KEY) {
      const models = image
        ? ["gemini-3.5-flash-lite"]
        : GEMINI_MODELS;

      for (const model of models) {
        candidates.push({ provider, model });
      }
    }

    if (provider === "openai" && OPENAI_KEY) {
      for (const model of OPENAI_MODELS) {
        candidates.push({ provider, model });
      }
    }
  }

  let lastError = null;

  for (const candidate of candidates.slice(0, 4)) {
    try {
      console.log(
        `Trying ${candidate.provider}: ${candidate.model}`
      );

      const answer = candidate.provider === "gemini"
        ? await askGemini(
            candidate.model,
            message,
            history,
            image
          )
        : await askOpenAI(
            candidate.model,
            message,
            history,
            image
          );

      console.log(
        `Success: ${candidate.provider}/${candidate.model}`
      );

      return res.json({
        success: true,
        provider: candidate.provider,
        model: candidate.model,
        answer
      });
    } catch (error) {
      lastError = error;

      console.error(
        `Failed ${candidate.provider}/${candidate.model}:`,
        error.message
      );
    }
  }

  return res.status(502).json({
    success: false,
    error: lastError
      ? `AI request failed: ${lastError.message}`
      : "No AI provider configured. Check your .env API keys."
  });
});

/* ERROR HANDLER */

app.use((error, req, res, next) => {
  console.error("Request error:", error.message);

  if (error instanceof multer.MulterError) {
    return res.status(400).json({
      success: false,
      error: "Image upload failed. Maximum image size is 5 MB."
    });
  }

  return res.status(400).json({
    success: false,
    error: error.message || "Invalid request."
  });
});

/* START SERVER */

app.listen(PORT, () => {
  console.log(`TETRON 4 U running on port ${PORT}`);
});
