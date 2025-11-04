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
    # Read PDF text
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""

    # Split into chunks for context
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)

    st.success(f"✅ PDF loaded successfully with {len(chunks)} text chunks!")

    # Question input
    query = st.text_input("Ask a question about your PDF:")

    if st.button("Submit") and query:
        with st.spinner("⏳ Getting answer..."):
            try:
                # Send query to OpenRouter
                response = client.chat.completions.create(
                    model="mistralai/mistral-7b-instruct:free",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that summarizes and explains PDF content."},
                        {"role": "user", "content": f"PDF Content: {chunks[:3]}\n\nUser Question: {query}"}
                    ],
                )
                answer = response.choices[0].message.content
                st.markdown(f"### 💬 Answer:\n{answer}")

            except Exception as e:
                st.error(f"❌ API Error: {e}")
else:
    st.info("⬆️ Please upload a PDF to begin.")
