import os
import faiss
import numpy as np
import requests
import gradio as gr

from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

# =========================
# INIT
# =========================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = None
chunks = []

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# =========================
# TELEKOM THEME LOGO
# =========================
LOGO_URL = "https://www.telekom.com/resource/image/1037468/landscape_ratio16x9/1920/1080/0a6b8d7f3d9f6f0c5f4c8b9c2c1b2d3e/telekom-logo.jpg"

# =========================
# DRIVE LINK HANDLER
# =========================
def convert_drive_link(link):
    try:
        file_id = link.split("/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?id={file_id}"
    except:
        return link

# =========================
# LOAD PDF
# =========================
def load_pdf_from_link(link):
    global index, chunks

    url = convert_drive_link(link)
    pdf_path = "temp.pdf"

    r = requests.get(url)
    with open(pdf_path, "wb") as f:
        f.write(r.content)

    reader = PdfReader(pdf_path)

    texts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            texts.append(text)

    chunks = []
    for t in texts:
        words = t.split()
        for i in range(0, len(words), 500):
            chunks.append(" ".join(words[i:i+500]))

    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return f"✅ PDF Loaded | Chunks: {len(chunks)}"

# =========================
# RETRIEVAL
# =========================
def retrieve(query, k=3):
    q_emb = embed_model.encode([query])
    _, indices = index.search(np.array(q_emb), k)
    return [chunks[i] for i in indices[0]]

# =========================
# GENERATION
# =========================
def generate_answer(query):
    if index is None:
        return "⚠️ Load a PDF first"

    context = "\n\n".join(retrieve(query))

    prompt = f"""
You are a professional AI assistant.
Answer ONLY from context.

Context:
{context}

Question:
{query}
"""

    res = client.responses.create(
        model="openai/gpt-oss-20b",
        input=prompt
    )

    return res.output_text

# =========================
# CHAT
# =========================
def chat(msg, history):
    ans = generate_answer(msg)
    history.append((msg, ans))
    return history, history

# =========================
# CSS (BOTPRESS + TELEKOM STYLE)
# =========================
css = """
body {
    background-color: #0b0b10;
    color: white;
    font-family: 'Inter', sans-serif;
}

.gradio-container {
    max-width: 1100px !important;
}

h1 {
    text-align: center;
    color: #e20074; /* Telekom pink */
    font-weight: 700;
}

.chatbot {
    background: #11131a !important;
    border-radius: 15px;
    border: 1px solid #2a2d3a;
}

button {
    background: linear-gradient(90deg, #e20074, #ff4da6) !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: bold;
    transition: 0.3s;
}

button:hover {
    transform: scale(1.03);
    box-shadow: 0 0 15px #e20074;
}

input {
    background: #1a1d26 !important;
    border: 1px solid #2a2d3a !important;
    color: white !important;
    border-radius: 10px !important;
}

.card {
    background: #11131a;
    padding: 15px;
    border-radius: 15px;
    border: 1px solid #2a2d3a;
}
"""

# =========================
# UI
# =========================
with gr.Blocks(css=css, theme=gr.themes.Soft()) as app:

    gr.Image(LOGO_URL, height=80)
    gr.Markdown("# 📊 Telekom AI RAG Chatbot")

    with gr.Row():
        pdf_link = gr.Textbox(label="📎 PDF Link (Google Drive supported)")
        load_btn = gr.Button("Load Document")

    status = gr.Textbox()

    chatbot = gr.Chatbot()
    state = gr.State([])

    msg = gr.Textbox(label="Ask Anything")

    load_btn.click(load_pdf_from_link, inputs=pdf_link, outputs=status)
    msg.submit(chat, inputs=[msg, state], outputs=[chatbot, state])

app.launch()