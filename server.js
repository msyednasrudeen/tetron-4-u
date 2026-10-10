require("dotenv").config();

const express = require("express");
const path = require("path");
const multer = require("multer");

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json({ limit: "1mb" }));
app.use(express.static(path.join(__dirname, "public")));

// API keys: Render Environment Variables
const GEMINI_KEY = process.env.T4U_GOOGLE_AI_SECRET;
const OPENAI_KEY = process.env.T4U_OPENAI_SECRET;

// Image uploads are stored in memory, not on disk.
const upload = multer({
    storage: multer.memoryStorage(),
    limits: {
        fileSize: 5 * 1024 * 1024,
        files: 1
    },
    fileFilter: (req, file, cb) => {
        if (file.mimetype && file.mimetype.startsWith("image/")) {
            cb(null, true);
        } else {
            cb(new Error("Please upload a valid image."));
        }
    }
});

// --------------------------------------------------
// HEALTH CHECK
// --------------------------------------------------

app.get("/api/health", (req, res) => {
    res.json({
        success: true,
        app: "TETRON 4 U",
        geminiConfigured: Boolean(GEMINI_KEY),
        openaiConfigured: Boolean(OPENAI_KEY)
    });
});

// --------------------------------------------------
// GEMINI MODEL LIST
// --------------------------------------------------

async function getGeminiModels() {
    if (!GEMINI_KEY) {
        throw new Error("Gemini API key is missing in Render.");
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
        .filter(model => {
            const methods = model.supportedGenerationMethods || [];

            return (
                methods.includes("generateContent") &&
                model.name?.startsWith("models/")
            );
        })
        .map(model => ({
            id: model.name.replace(/^models\//, ""),
            name: model.displayName || model.name,
            provider: "gemini",
            description: model.description || ""
        }))
        .sort((a, b) => a.name.localeCompare(b.name));
}

// --------------------------------------------------
// OPENAI MODEL LIST
// --------------------------------------------------

async function getOpenAIModels() {
    if (!OPENAI_KEY) {
        throw new Error("OpenAI API key is missing in Render.");
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

    const pattern =
        /^(gpt-|chatgpt-|o[1-9](-|$)|o[1-9][a-z0-9-]*$)/i;

    return (data.data || [])
        .filter(model => pattern.test(model.id))
        .map(model => ({
            id: model.id,
            name: model.id,
            provider: "openai",
            description: "OpenAI model"
        }))
        .sort((a, b) => a.id.localeCompare(b.id));
}

// --------------------------------------------------
// MODELS ENDPOINT
// --------------------------------------------------

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
                error: "Choose Gemini or OpenAI."
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

// --------------------------------------------------
// GEMINI CHAT + OPTIONAL IMAGE
// --------------------------------------------------

async function askGemini(model, message, history, image) {
    const contents = [];

    // Convert previous conversation into Gemini's role format.
    for (const item of history) {
        if (!item || !["user", "assistant"].includes(item.role)) {
            continue;
        }

        if (typeof item.text !== "string" || !item.text.trim()) {
            continue;
        }

        contents.push({
            role: item.role === "assistant" ? "model" : "user",
            parts: [{ text: item.text.slice(0, 10000) }]
        });
    }

    const parts = [{ text: message }];

    if (image) {
        parts.push({
            inlineData: {
                mimeType: image.mimetype,
                data: image.buffer.toString("base64")
            }
        });
    }

    contents.push({
        role: "user",
        parts
    });

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
                    parts: [{
                        text:
                            "You are TETRON 4 U, a helpful AI assistant. " +
                            "Give accurate, clear and useful answers. " +
                            "Adapt to the user's language. " +
                            "If the user writes Tamil-English, respond naturally in Tamil-English. " +
                            "If an image is supplied, analyze its visible contents carefully."
                    }]
                },
                contents,
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
        ?.map(part => part.text || "")
        .join("")
        .trim();

    if (!answer) {
        throw new Error("Gemini did not return a text answer.");
    }

    return answer;
}

// --------------------------------------------------
// OPENAI CHAT + OPTIONAL IMAGE
// --------------------------------------------------

async function askOpenAI(model, message, history, image) {
    const messages = [
        {
            role: "system",
            content:
                "You are TETRON 4 U, a helpful AI assistant. " +
                "Give accurate, clear and useful answers. " +
                "Adapt to the user's language. " +
                "If the user writes Tamil-English, respond naturally in Tamil-English. " +
                "If an image is supplied, analyze it carefully."
        }
    ];

    for (const item of history) {
        if (!item || !["user", "assistant"].includes(item.role)) {
            continue;
        }

        if (typeof item.text !== "string" || !item.text.trim()) {
            continue;
        }

        messages.push({
            role: item.role,
            content: item.text.slice(0, 10000)
        });
    }

    let userContent;

    if (image) {
        userContent = [
            {
                type: "text",
                text: message || "Please analyze this image."
            },
            {
                type: "image_url",
                image_url: {
                    url:
                        `data:${image.mimetype};base64,` +
                        image.buffer.toString("base64")
                }
            }
        ];
    } else {
        userContent = message;
    }

    messages.push({
        role: "user",
        content: userContent
    });

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
                messages,
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

// --------------------------------------------------
// CHAT ENDPOINT
// Supports JSON and multipart image uploads.
// --------------------------------------------------

app.post("/api/chat", (req, res, next) => {
    const contentType = req.headers["content-type"] || "";

    if (contentType.includes("multipart/form-data")) {
        upload.single("image")(req, res, error => {
            if (error) {
                return res.status(400).json({
                    success: false,
                    error: error.message
                });
            }

            next();
        });
    } else {
        next();
    }
}, async (req, res) => {
    try {
        const body = req.body || {};

        const message = typeof body.message === "string"
            ? body.message.trim()
            : "";

        const provider = String(body.provider || "").toLowerCase();
        const model = String(body.model || "").trim();
        const image = req.file || null;

        if (!message && !image) {
            return res.status(400).json({
                success: false,
                error: "Enter a message or upload an image."
            });
        }

        if (message.length > 10000) {
            return res.status(400).json({
                success: false,
                error: "Message is too long."
            });
        }

        if (!model || model.length > 200) {
            return res.status(400).json({
                success: false,
                error: "Please select a model."
            });
        }

        // Accept a limited amount of text history.
        let history = [];

        if (typeof body.history === "string") {
            try {
                history = JSON.parse(body.history);
            } catch {
                history = [];
            }
        } else if (Array.isArray(body.history)) {
            history = body.history;
        }

        history = history
            .filter(item =>
                item &&
                ["user", "assistant"].includes(item.role) &&
                typeof item.text === "string"
            )
            .slice(-12);

        let answer;

        if (provider === "gemini") {
            if (!GEMINI_KEY) {
                throw new Error("Gemini API key is not configured.");
            }

            const models = await getGeminiModels();

            if (!models.some(item => item.id === model)) {
                return res.status(400).json({
                    success: false,
                    error: "Gemini model is unavailable for this API key. Refresh the model list."
                });
            }

            answer = await askGemini(
                model,
                message || "Please analyze the uploaded image.",
                history,
                image
            );

        } else if (
            provider === "openai" ||
            provider === "chatgpt"
        ) {
            if (!OPENAI_KEY) {
                throw new Error("OpenAI API key is not configured.");
            }

            const models = await getOpenAIModels();

            if (!models.some(item => item.id === model)) {
                return res.status(400).json({
                    success: false,
                    error: "OpenAI model is unavailable for this API key. Refresh the model list."
                });
            }

            answer = await askOpenAI(
                model,
                message || "Please analyze the uploaded image.",
                history,
                image
            );

        } else {
            return res.status(400).json({
                success: false,
                error: "Choose Gemini or OpenAI."
            });
        }

        return res.json({
            success: true,
            provider,
            model,
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

// --------------------------------------------------
// API 404
// --------------------------------------------------

app.use("/api", (req, res) => {
    res.status(404).json({
        success: false,
        error: "API endpoint not found."
    });
});

// --------------------------------------------------
// GENERAL ERROR HANDLER
// --------------------------------------------------

app.use((error, req, res, next) => {
    console.error("Server error:", error.message);

    if (res.headersSent) {
        return next(error);
    }

    res.status(500).json({
        success: false,
        error: "Internal server error."
    });
});

// --------------------------------------------------
// START SERVER
// --------------------------------------------------

app.listen(PORT, "0.0.0.0", () => {
    console.log(`TETRON 4 U running on port ${PORT}`);
});
