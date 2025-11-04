import os
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# --- Load environment variables ---
load_dotenv()

# --- Setup OpenRouter client ---
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

# --- Streamlit UI ---
st.set_page_config(page_title="📄 PDF Chatbot (via OpenRouter)", layout="centered")

st.title("📄 PDF Chatbot (via OpenRouter)")
st.caption("Upload a PDF and ask questions about its content!")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    # --- Read PDF text ---
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    # --- Split text into chunks ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    st.success(f"✅ PDF loaded successfully with {len(chunks)} text chunks!")

    # --- Ask a question ---
    question = st.text_input("Ask a question about your PDF:")

    if question:
        with st.spinner("Thinking..."):
            try:
                response = client.chat.completions.create(
                    model="meta-llama/llama-3.1-8b-instruct",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes and answers based on PDF content."},
                        {"role": "user", "content": f"Context:\n{text[:4000]}\n\nQuestion: {question}"}
                    ]
                )

                st.markdown("### 💬 Answer:")
                st.write(response.choices[0].message.content)

            except Exception as e:
                st.error(f"❌ API Error: {e}")
else:
    st.info("👆 Please upload a PDF to get started.")

