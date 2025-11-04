import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Configure OpenAI / OpenRouter ---
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)

# --- Custom text splitter (replaces LangChain) ---
def split_text_into_chunks(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# --- Streamlit UI ---
st.set_page_config(page_title="📄 PDF Chatbot (via OpenRouter)", layout="centered")
st.title("📄 PDF Chatbot (via OpenRouter)")
st.caption("Upload a PDF and ask questions about its content!")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    if text.strip():
        chunks = split_text_into_chunks(text)
        st.success(f"✅ PDF loaded successfully with {len(chunks)} text chunks!")
    else:
        st.error("❌ No readable text found in this PDF.")
        st.stop()
else:
    st.info("📤 Please upload a PDF file to start.")
    st.stop()

query = st.text_input("Ask a question about your PDF:", placeholder="e.g., Summarize this document...")

if query:
    with st.spinner("⏳ Getting answer..."):
        try:
            response = client.chat.completions.create(
                model="mistralai/mistral-small-24b-instruct",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers based on a provided PDF document."},
                    {"role": "user", "content": f"PDF Content:\n{text[:4000]}\n\nQuestion: {query}"}
                ]
            )

            answer = response.choices[0].message.content
            st.markdown(f"### 🧠 Answer:\n{answer}")

        except Exception as e:
            st.error(f"❌ API Error: {e}")
