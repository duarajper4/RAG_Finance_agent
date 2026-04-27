import os
import faiss
import numpy as np
import requests
import gradio as gr
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
 
*, *::before, *::after { box-sizing: border-box; }
 
:root {
  --M:        #E20074;
  --MD:       #B5005C;
  --MBG:      rgba(226,0,116,0.07);
  --bg:       #F6F3F8;
  --white:    #FFFFFF;
  --surf2:    #F2EEF5;
  --border:   #E6DEED;
  --border2:  #D0C4DC;
  --text:     #1A1525;
  --sub:      #5A5272;
  --muted:    #9990AA;
  --ok:       #00A878;
  --okbg:     #E8F8F3;
  --okborder: #AADFC8;
  --f:        'Plus Jakarta Sans', sans-serif;
  --mono:     'JetBrains Mono', monospace;
  --sh-sm:    0 1px 4px rgba(0,0,0,0.06);
  --sh-md:    0 3px 14px rgba(0,0,0,0.08);
  --sh-mg:    0 4px 20px rgba(226,0,116,0.18);
}
 
/* ── BASE ── */
body {
  font-family: var(--f) !important;
  background: var(--bg) !important;
  color: var(--text) !important;
}
 
.gradio-container {
  max-width: 1060px !important;
  margin: 0 auto !important;
  padding: 0 20px 48px !important;
  background: transparent !important;
  box-shadow: none !important;
}
 
/* ── HEADING (gr.Markdown title) ── */
h1 {
  font-family: var(--f) !important;
  font-size: 1.6rem !important;
  font-weight: 800 !important;
  letter-spacing: -0.03em !important;
  color: var(--text) !important;
  padding: 28px 0 4px !important;
  border-bottom: 3px solid var(--M) !important;
  margin-bottom: 24px !important;
  position: relative;
}
 
/* animated stripe under title */
h1::after {
  content: '';
  position: absolute;
  bottom: -3px; left: 0;
  width: 80px; height: 3px;
  background: #FF6BB5;
  animation: titleSlide 2s ease-in-out infinite alternate;
}
@keyframes titleSlide {
  0%   { width: 60px; opacity: 0.6; }
  100% { width: 160px; opacity: 1; }
}
 
/* emoji in title stays natural */
h1 .emoji { color: inherit !important; }
 
/* ── ALL CARDS / PANELS ── */
.gr-box, .gr-panel, .gr-group, .gr-form,
[class*="panel"], [class*="block"] {
  background: var(--white) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  box-shadow: var(--sh-sm) !important;
  transition: box-shadow 0.2s, border-color 0.2s !important;
}
.gr-box:hover, .gr-panel:hover {
  box-shadow: 0 4px 20px rgba(226,0,116,0.10) !important;
  border-color: rgba(226,0,116,0.22) !important;
}
 
/* ── LABELS ── */
label, .gr-label, .label-wrap span {
  font-family: var(--f) !important;
  font-size: 0.63rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.10em !important;
  text-transform: uppercase !important;
  color: var(--muted) !important;
}
 
/* ── INPUTS (link box + question box) ── */
textarea, input[type="text"] {
  font-family: var(--f) !important;
  background: var(--surf2) !important;
  color: var(--text) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 8px !important;
  padding: 12px 15px !important;
  font-size: 0.88rem !important;
  line-height: 1.55 !important;
  transition: border-color 0.2s, box-shadow 0.2s !important;
}
textarea:focus, input[type="text"]:focus {
  border-color: var(--M) !important;
  box-shadow: 0 0 0 3px rgba(226,0,116,0.12) !important;
  outline: none !important;
  background: var(--white) !important;
}
textarea::placeholder, input[type="text"]::placeholder {
  color: var(--muted) !important;
}
 
/* Status textbox (readonly) gets green monospace style */
textarea[readonly] {
  font-family: var(--mono) !important;
  font-size: 0.76rem !important;
  color: var(--ok) !important;
  background: var(--okbg) !important;
  border-color: var(--okborder) !important;
}
 
/* ── BUTTONS ── */
button, .gr-button {
  font-family: var(--f) !important;
  font-weight: 700 !important;
  font-size: 0.75rem !important;
  letter-spacing: 0.08em !important;
  text-transform: uppercase !important;
  border-radius: 8px !important;
  padding: 12px 24px !important;
  border: none !important;
  cursor: pointer !important;
  transition: all 0.2s ease !important;
}
 
/* Primary = magenta gradient */
button.primary, .gr-button-primary, button[variant="primary"] {
  background: linear-gradient(135deg, #E20074 0%, #B5005C 100%) !important;
  color: #fff !important;
  box-shadow: var(--sh-mg) !important;
}
button.primary:hover, .gr-button-primary:hover {
  box-shadow: 0 6px 28px rgba(226,0,116,0.40) !important;
  transform: translateY(-1px) !important;
}
button.primary:active, .gr-button-primary:active {
  transform: translateY(0) !important;
}
 
/* ── CHATBOT CONTAINER ── */
.gr-chatbot, [data-testid="chatbot"] {
  background: #FAFAFA !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  box-shadow: var(--sh-sm) !important;
}
 
/* User bubble — right, magenta */
.message.user, [data-testid="user"] .message {
  background: linear-gradient(135deg, #E20074, #B5005C) !important;
  color: #fff !important;
  border-radius: 12px 12px 3px 12px !important;
  padding: 11px 15px !important;
  font-size: 0.86rem !important;
  line-height: 1.65 !important;
  max-width: 72% !important;
  margin-left: auto !important;
  box-shadow: 0 3px 14px rgba(226,0,116,0.22) !important;
  animation: slideR 0.22s ease !important;
}
 
/* Bot bubble — left, white with magenta stripe */
.message.bot, [data-testid="bot"] .message {
  background: var(--white) !important;
  border: 1px solid var(--border) !important;
  border-left: 3px solid var(--M) !important;
  color: var(--text) !important;
  border-radius: 2px 12px 12px 12px !important;
  padding: 11px 15px !important;
  font-size: 0.86rem !important;
  line-height: 1.65 !important;
  max-width: 80% !important;
  box-shadow: var(--sh-sm) !important;
  animation: slideL 0.22s ease !important;
}
 
@keyframes slideR {
  from { opacity: 0; transform: translateX(10px); }
  to   { opacity: 1; transform: translateX(0); }
}
@keyframes slideL {
  from { opacity: 0; transform: translateX(-10px); }
  to   { opacity: 1; transform: translateX(0); }
}
 
/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--surf2); border-radius: 3px; }
::-webkit-scrollbar-thumb { background: rgba(226,0,116,0.28); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--M); }
 
/* ── HIDE GRADIO FOOTER ── */
footer { display: none !important; }
.gap { gap: 14px !important; }
"""


# Globals (shared state in Gradio)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = None
chunks = []

# Add after globals:
chat_history = []  # Session memory

def chat(user_input, history):
    global chat_history
    
    # Build full context (PDF + conversation history)
    full_context = "\n".join([f"User: {h['content']}\nBot: {h.get('bot_response', '')}" 
                             for h in chat_history[-5:]]) if chat_history else ""
    
    answer = generate_answer(user_input, full_context)
    
    # Store in memory
    chat_history.append({"user": user_input, "bot": answer})
    
    # Update UI history
    new_history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": answer}
    ]
    
    return new_history, new_history

def generate_answer(query, conversation_context=""):
    if index is None:
        return "⚠️ Please load a PDF first."
    
    rag_context = retrieve(query)
    rag_text = "\n\n".join(rag_context)
    
    # ✅ Combine RAG + Conversation Memory
    full_prompt = f"""You are a smart financial AI assistant that remembers conversations.

Previous conversation:
{conversation_context}

PDF Context (use ONLY this for facts):
{rag_text}

Question: {query}

Respond naturally and helpfully, referencing past discussion when relevant."""
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.7,
        max_tokens=600
    )
    return response.choices[0].message.content

# Groq client with HF Secrets
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

def convert_drive_link(link):
    try:
        file_id = link.split("/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?id={file_id}"
    except:
        return link

def load_pdf_from_link(link):
    global index, chunks
    url = convert_drive_link(link)
    PDF_PATH = "temp.pdf"
    response = requests.get(url)
    with open(PDF_PATH, "wb") as f:
        f.write(response.content)
    
    reader = PdfReader(PDF_PATH)
    texts = [page.extract_text() for page in reader.pages if page.extract_text()]
    
    # Chunking
    chunks = []
    for t in texts:
        words = t.split()
        for i in range(0, len(words), 500):
            chunks.append(" ".join(words[i:i+500]))
    
    # Embeddings + FAISS
    embeddings = embed_model.encode(chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    
    return f"✅ PDF loaded! {len(chunks)} chunks created."

def retrieve(query, k=3):
    if index is None:
        return []
    q_emb = embed_model.encode([query])
    distances, indices = index.search(np.array(q_emb).astype('float32'), k)
    return [chunks[i] for i in indices[0]]

def generate_answer(query):
    if index is None:
        return "⚠️ Please load a PDF first."
    
    context = retrieve(query)
    context_text = "\n\n".join(context)
    
    prompt = f"""You are a financial AI assistant.
Answer ONLY using the context below.

Context:
{context_text}

Question:
{query}"""
    
    # ✅ Use currently available Groq model (April 2026)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",  # Fast & reliable
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    )
    
    return response.choices[0].message.content
    
# ... (keep all previous code until chat function)

def chat(user_input, history):
    answer = generate_answer(user_input)
    new_history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": answer}
    ]
    return new_history, new_history

# UI (replace entirely):
with gr.Blocks(title="Finance RAG", css=css) as app:
    gr.Markdown("# 📊 Dynamic Finance RAG Chatbot")
    
    with gr.Row():
        link_input = gr.Textbox(label="📎 Google Drive PDF Link", placeholder="https://drive.google.com/file/d/...")
        load_btn = gr.Button("📥 Load PDF", variant="primary")
    
    status = gr.Textbox(label="Status", interactive=False)
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(
        label="💬 Ask about the PDF", 
        placeholder="What are the key financial metrics?",
        container=True
    )
    
    # Events
    load_btn.click(load_pdf_from_link, inputs=link_input, outputs=status)
    msg.submit(chat, inputs=[msg, chatbot], outputs=[chatbot, chatbot])
    msg.submit(lambda: "", outputs=msg)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8000)
