import os
import faiss
import numpy as np
import requests
import gradio as gr
from openai import OpenAI
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

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
# Add to app.py
MODELS = {
    "Fast (Groq)": "llama-3.1-8b-instant",
    "Smart (Groq)": "llama-3.1-70b-versatile", 
    "Your Bot": "yourusername/finance-bot"  # After fine-tuning
}

model_dropdown = gr.Dropdown(choices=list(MODELS.keys()), value="Fast (Groq)", label="🤖 AI Model")

def generate_answer(query, conversation_context="", model_name="Fast (Groq)"):
    # ... RAG logic ...
    model_id = MODELS[model_name]
    
    if "Your Bot" in model_name:
        # Use HF Inference API for your model
        response = requests.post("https://api-inference.huggingface.co/models/yourusername/finance-bot", 
                                json={"inputs": full_prompt})
        return response.json()[0]["generated_text"]
    else:
        # Groq API
        response = client.chat.completions.create(model=model_id, ...)

def chat(user_input, history):
    answer = generate_answer(user_input)
    new_history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": answer}
    ]
    return new_history, new_history

# UI (replace entirely):
with gr.Blocks(title="Finance RAG") as app:
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
    app.launch()