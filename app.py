import os
import streamlit as st
from pypdf import PdfReader
import google.generativeai as genai

# 1. 網頁基本設定
st.set_page_config(
    page_title="HKUST Academic & Coursework Assistant",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 HKUST SBA Academic Planning & Coursework Assistant")
st.caption("Ask questions grounded in official HKUST SBA handbooks, MSE rules, and SBM course slides.")

# 2. 讀取 API Key (從 Streamlit Secrets 或環境變數讀取，避免外洩)
api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
if not api_key:
    st.error("⚠️ Missing GEMINI_API_KEY. Please configure it in Streamlit Cloud Secrets.")
    st.stop()

genai.configure(api_key=api_key)

# 3. 讀取 docs 資料夾內的所有 PDF 作為知識庫
@st.cache_resource
def load_knowledge_base(folder_path="docs"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        return ""
    
    combined_text = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(folder_path, filename)
            reader = PdfReader(filepath)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text()
                if text:
                    combined_text.append(f"[Source: {filename} | Page {page_num}]\n{text}")
    return "\n\n".join(combined_text)

knowledge_base = load_knowledge_base("docs")

# 側邊欄顯示已載入的知識庫檔案
with st.sidebar:
    st.header("📚 Loaded Knowledge Base")
    pdf_files = [f for f in os.listdir("docs") if f.lower().endswith(".pdf")] if os.path.exists("docs") else []
    if pdf_files:
        for f in pdf_files:
            st.success(f"📄 {f}")
    else:
        st.warning("No PDFs found in `/docs` folder yet! Please upload PDFs to GitHub.")
    
    st.divider()
    st.markdown("**System Guardrails:**\n- Grounded strictly in uploaded PDFs\n- Cites exact file & page numbers\n- Declines live SIS enrollment actions")

# 4. 設定防幻覺與引用規則 (System Instruction)
system_instruction = f"""
You are an official HKUST Academic Planning and Coursework RAG Assistant for Year 1 School-Based Admission (SBA) and SBM students.

STRICT RULES:
1. Answer ONLY using facts from the KNOWLEDGE BASE provided below. Do NOT guess, extrapolate, or use outside knowledge.
2. Every claim must include an inline citation in the exact format: [Source: filename | Page X].
3. If the user asks about live SIS seat quotas, automatic course enrollment, or information not present in the KNOWLEDGE BASE, politely decline and direct them to the official HKUST SIS portal or their academic advisor.
4. When discussing historical Major Selection Exercise (MSE) data, explicitly state that past figures are historical benchmarks and not guaranteed cutoffs.

KNOWLEDGE BASE:
{knowledge_base}
"""

model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    system_instruction=system_instruction
)

# 5. 對話紀錄與介面
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
                # 組合歷史對話上下文
                chat_history = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages[-6:]])
                response = model.generate_content(chat_history)
                reply = response.text
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                st.error(f"Error generating response: {e}")
