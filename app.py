import os
import re
import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai

st.set_page_config(
    page_title="HKUST Academic & Coursework Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 HKUST SBA Academic Planning & Coursework Assistant")
st.caption("Ask questions grounded in official HKUST SBA handbooks, MSE rules, and SBM course slides.")

api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("⚠️ Missing GEMINI_API_KEY. Please configure it in Streamlit Cloud Secrets.")
    st.stop()

genai.configure(api_key=api_key)

# 將 PDF 切成一頁一頁的區塊，避免一次塞入幾百頁導致超時
@st.cache_resource
def load_chunks(folder_path="docs"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return []
    
    chunks = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            reader = PdfReader(filepath)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    chunks.append({
                        "source": filename,
                        "page": page_num,
                        "text": text
                    })
    return chunks

chunks = load_chunks("docs")

# 關鍵字檢索：只挑選最相關的前 12 頁送給 AI，速度極快且不爆額度
def retrieve_relevant_pages(query, all_chunks, top_k=12):
    if not all_chunks:
        return ""
    keywords = set(re.findall(r'\w+', query.lower()))
    scored = []
    for c in all_chunks:
        content_lower = c["text"].lower()
        score = sum(1 for kw in keywords if len(kw) > 2 and kw in content_lower)
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [item[1] for item in scored[:top_k]]
    return "\n\n".join([f"[Source: {c['source']} | Page {c['page']}]\n{c['text']}" for c in selected])

with st.sidebar:
    st.header("📚 Loaded Knowledge Base")
    pdf_files = sorted(list(set(c["source"] for c in chunks)))
    if pdf_files:
        for f in pdf_files:
            st.success(f"📄 {f}")
        st.caption(f"Total indexed pages: {len(chunks)}")
    else:
        st.warning("No PDFs found in `/docs` folder yet!")
    
    st.divider()
    st.markdown("**System Guardrails:**\n- Grounded strictly in uploaded PDFs\n- Cites exact file & page numbers\n- Declines live SIS enrollment actions")

model = genai.GenerativeModel(model_name="gemini-2.5-flash")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I am your HKUST Academic & Coursework Assistant. Ask me about SBA requirements, IB/A-Level credit transfers, MSE eligibility, or your uploaded SBM lecture slides!"}
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about HKUST courses, MSE, or SBM slides..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching official HKUST documents..."):
            try:
                relevant_context = retrieve_relevant_pages(prompt, chunks, top_k=12)
                full_prompt = f"""You are an official HKUST Academic Planning and Coursework RAG Assistant for Year 1 School-Based Admission (SBA) and SBM students.

STRICT RULES:
1. Answer ONLY using facts from the RETRIEVED DOCUMENTS below. Do NOT guess or use outside knowledge.
2. Include inline citations in the exact format: [Source: filename | Page X].
3. If the user asks about live SIS seat quotas, automatic enrollment, or topics not covered in the documents, politely decline and direct them to the official HKUST SIS portal or academic advisor.
4. Note that historical Major Selection Exercise (MSE) statistics are past benchmarks, not guaranteed cutoffs.

RETRIEVED DOCUMENTS:
{relevant_context}

USER QUESTION:
{prompt}
"""
                response = model.generate_content(full_prompt, request_options={"timeout": 30})
                reply = response.text
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"⚠️ API Error: {e}")
