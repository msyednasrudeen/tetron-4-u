require("dotenv").config();

const express = require("express");
const path = require("path");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

const GEMINI_KEY = process.env.T4U_GOOGLE_AI_SECRET;
const OPENAI_KEY = process.env.T4U_OPENAI_SECRET;

// Only allow models returned by the provider's API.
// This is a basic text-chat compatibility filter, not a guarantee
// that every listed model supports the same chat endpoint.

async function getGeminiModels() {
  if (!GEMINI_KEY) {
    throw new Error("Gemini API key is missing in Render Environment.");
  }

  const response = await fetch(
    "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
    {
      headers: {
        "x-goog-api-key": GEMINI_KEY
      }
    }
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      data.error?.message || "Unable to fetch Gemini models."
    );
  }

  return (data.models || [])
    .filter((model) => {
      const methods = model.supportedGenerationMethods || [];
      const name = model.name || "";

      return (
        methods.includes("generateContent") &&
        name.startsWith("models/")
      );
    })
    .map((model) => ({
      id: model.name.replace(/^models\//, ""),
      name: model.displayName || model.name,
      provider: "gemini",
      description: model.description || ""
    }))
    .sort((a, b) => a.name.localeCompare(b.name));
}

async function getOpenAIModels() {
  if (!OPENAI_KEY) {
    throw new Error("OpenAI API key is missing in Render Environment.");
  }

  const response = await fetch(
    "https://api.openai.com/v1/models",
    {
      headers: {
        Authorization: `Bearer ${OPENAI_KEY}`
      }
    }
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      data.error?.message || "Unable to fetch OpenAI models."
    );
  }

  // Basic filter for commonly used text/chat model families.
  // Availability and endpoint compatibility still depend on your account.
  const chatPattern =
    /^(gpt-|chatgpt-|o[1-9](-|$)|o[1-9][a-z0-9-]*$)/i;

  return (data.data || [])
    .filter((model) => chatPattern.test(model.id))
    .map((model) => ({
      id: model.id,
      name: model.id,
      provider: "openai",
      description: "OpenAI model"
    }))
    .sort((a, b) => {
      const dateA = a.created || 0;
      const dateB = b.created || 0;
      return dateA - dateB || a.id.localeCompare(b.id);
    });
}

// Health check
app.get("/api/health", (req, res) => {
  res.json({
    success: true,
    app: "TETRON 4 U",
    geminiConfigured: Boolean(GEMINI_KEY),
    openaiConfigured: Boolean(OPENAI_KEY)
  });
});

// Get available models
app.get("/api/models", async (req, res) => {
  const provider = String(req.query.provider || "").toLowerCase();

  try {
    let models;

    if (provider === "gemini") {
      models = await getGeminiModels();
    } else if (
      provider === "openai" ||
      provider === "chatgpt"
    ) {
      models = await getOpenAIModels();
    } else {
      return res.status(400).json({
        success: false,
        error: "Choose gemini or openai as provider."
      });
    }

    res.json({
      success: true,
      provider,
      count: models.length,
      models
    });
  } catch (error) {
    console.error("Model list error:", error.message);

    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Send a message to Gemini
async function askGemini(model, message) {
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_KEY
      },
      body: JSON.stringify({
        systemInstruction: {
          parts: [
            {
              text:
                "You are TETRON 4 U, a helpful AI assistant. Answer accurately and clearly. Adapt to the user's language. If the user writes Tamil-English, you may respond in Tamil-English."
            }
          ]
        },
        contents: [
          {
            role: "user",
            parts: [{ text: message }]
          }
        ],
        generationConfig: {
          temperature: 0.7,
          maxOutputTokens: 2048
        }
      })
    }
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      data.error?.message || `Gemini request failed (${response.status}).`
    );
  }

  const answer = data.candidates?.[0]?.content?.parts
    ?.map((part) => part.text || "")
    .join("")
    .trim();

  if (!answer) {
    throw new Error("Gemini did not return a text answer.");
  }

  return answer;
}

// Send a message to OpenAI
async function askOpenAI(model, message) {
  const response = await fetch(
    "https://api.openai.com/v1/chat/completions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${OPENAI_KEY}`
      },
      body: JSON.stringify({
        model,
        messages: [
          {
            role: "system",
            content:
              "You are TETRON 4 U, a helpful AI assistant. Answer accurately and clearly. Adapt to the user's language. If the user writes Tamil-English, you may respond in Tamil-English."
          },
          {
            role: "user",
            content: message
          }
        ],
        max_completion_tokens: 2048
      })
    }
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      data.error?.message || `OpenAI request failed (${response.status}).`
    );
  }

  const answer = data.choices?.[0]?.message?.content?.trim();

  if (!answer) {
    throw new Error("OpenAI did not return a text answer.");
  }

  return answer;
}

// Chat endpoint
app.post("/api/chat", async (req, res) => {
  try {
    const { message, provider, model } = req.body || {};

    if (typeof message !== "string" || !message.trim()) {
      return res.status(400).json({
        success: false,
        error: "Please enter a message."
      });
    }

    if (message.length > 10000) {
      return res.status(400).json({
        success: false,
        error: "Message is too long."
      });
    }

    const selectedProvider = String(provider || "").toLowerCase();
    const selectedModel = String(model || "").trim();

    if (!selectedModel || selectedModel.length > 200) {
      return res.status(400).json({
        success: false,
        error: "Please select a model."
      });
    }

    let answer;

    if (selectedProvider === "gemini") {
      if (!GEMINI_KEY) {
        throw new Error("Gemini API key is not configured.");
      }

      // Validate that the selected model is currently listed
      // by the Gemini API before sending the request.
      const models = await getGeminiModels();

      if (!models.some((item) => item.id === selectedModel)) {
        return res.status(400).json({
          success: false,
          error: "That Gemini model is not currently available to this API key. Refresh the model list."
        });
      }

      answer = await askGemini(selectedModel, message.trim());
    } else if (
      selectedProvider === "openai" ||
      selectedProvider === "chatgpt"
    ) {
      if (!OPENAI_KEY) {
        throw new Error("OpenAI API key is not configured.");
      }

      const models = await getOpenAIModels();

      if (!models.some((item) => item.id === selectedModel)) {
        return res.status(400).json({
          success: false,
          error: "That OpenAI model is not currently listed for this API key. Refresh the model list."
        });
      }

      answer = await askOpenAI(selectedModel, message.trim());
    } else {
      return res.status(400).json({
        success: false,
        error: "Choose Gemini or OpenAI."
      });
    }

    return res.json({
      success: true,
      provider: selectedProvider,
      model: selectedModel,
      answer
    });
  } catch (error) {
    console.error("Chat error:", error.message);

    return res.status(500).json({
      success: false,
      error: error.message || "Something went wrong."
    });
  }
});

app.use("/api", (req, res) => {
  res.status(404).json({
    success: false,
    error: "API endpoint not found."
  });
});

app.listen(PORT, "0.0.0.0", () => {
  console.log(`TETRON 4 U running on port ${PORT}`);
});
