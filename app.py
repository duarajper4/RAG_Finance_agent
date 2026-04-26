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
    
def chat(user_input, history):
    answer = generate_answer(user_input)
    history.append((user_input, answer))
    return history, history

with gr.Blocks() as app:
    gr.Markdown("# 📊 Dynamic Finance RAG Chatbot")
    
    with gr.Row():
        link_input = gr.Textbox(label="📎 Paste Google Drive PDF Link")
        load_btn = gr.Button("Load PDF")
    
    status = gr.Textbox(label="Status", interactive=False)
    
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Ask your question")
    
    # Events
    load_btn.click(load_pdf_from_link, inputs=link_input, outputs=status)
    msg.submit(chat, inputs=[msg, chatbot], outputs=[chatbot, chatbot])

if __name__ == "__main__":
    app.launch()