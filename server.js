require("dotenv").config();

const express = require("express");
const path = require("path");

const app = express();

app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

app.get("/api/health", (req, res) => {
  res.json({
    name: "TETRON 4 U",
    status: "online"
  });
});

async function askGemini(message) {
  const key = process.env.GEMINI_API_KEY;

  if (!key) {
    throw new Error("Gemini API key is missing");
  }

  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=${key}`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        systemInstruction: {
          parts: [{
            text: "You are TETRON 4 U, a friendly and helpful AI assistant. Answer clearly and accurately."
          }]
        },
        contents: [{
          role: "user",
          parts: [{ text: message }]
        }]
      })
    }
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      data.error?.message || "Gemini request failed"
    );
  }

  return data.candidates?.[0]?.content?.parts
    ?.map(part => part.text || "")
    .join("") || "No response received.";
}

async function askChatGPT(message) {
  const key = process.env.OPENAI_API_KEY;

  if (!key) {
    throw new Error("OpenAI API key is missing");
  }

  const response = await fetch(
    "https://api.openai.com/v1/chat/completions",
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "Authorization": `Bearer ${key}`
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        messages: [
          {
            role: "system",
            content: "You are TETRON 4 U, a friendly and helpful AI assistant. Answer clearly and accurately."
          },
          {
            role: "user",
            content: message
          }
        ]
      })
    }
  );

  const data = await response.json();

  if (!response.ok) {
    throw new Error(
      data.error?.message || "OpenAI request failed"
    );
  }

  return data.choices?.[0]?.message?.content ||
    "No response received.";
}

app.post("/api/chat", async (req, res) => {
  const message = req.body?.message;
  const requestedAI = req.body?.provider || "gemini";

  if (
    typeof message !== "string" ||
    !message.trim() ||
    message.length > 10000
  ) {
    return res.status(400).json({
      error: "Enter a message under 10000 characters."
    });
  }

  if (!["gemini", "chatgpt"].includes(requestedAI)) {
    return res.status(400).json({
      error: "Choose Gemini or ChatGPT."
    });
  }

  const firstAI = requestedAI === "gemini"
    ? askGemini
    : askChatGPT;

  const secondAI = requestedAI === "gemini"
    ? askChatGPT
    : askGemini;

  try {
    const reply = await firstAI(message.trim());

    return res.json({
      reply,
      provider: requestedAI
    });
  } catch (firstError) {
    console.error("Primary AI failed:", firstError.message);

    try {
      const reply = await secondAI(message.trim());

      return res.json({
        reply,
        provider: requestedAI === "gemini"
          ? "chatgpt"
          : "gemini",
        fallback: true
      });
    } catch (secondError) {
      console.error("Fallback AI failed:", secondError.message);

      return res.status(502).json({
        error: "Both AI services failed. Check API keys, access and usage limits."
      });
    }
  }
});

const PORT = process.env.PORT || 3000;

app.listen(PORT, "0.0.0.0", () => {
  console.log(`TETRON 4 U running on port ${PORT}`);
});
