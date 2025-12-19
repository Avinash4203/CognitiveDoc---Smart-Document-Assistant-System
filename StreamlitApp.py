import streamlit as st
import io
import time
import random
import re
import json
from collections import Counter
from gtts import gTTS
from fpdf import FPDF
from deep_translator import GoogleTranslator
import graphviz 

# NOTE: Ensure these modules exist in your environment
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.model_api import load_model
from QAWithPDF.embedding import download_gemini_embedding

# ===================== GLOBAL PAGE CONFIG =====================
st.set_page_config(
    page_title="CognitiveDoc",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===================== THEME CSS =====================
def load_theme_css():
    st.markdown(""" 
    <style>
    :root { --bg: #0b1020; --card-bg: rgba(255,255,255,0.03); --text: #e6f7ff; --sub: #bfe6ff;
            --accent1: #4aa9ff; --accent2: #7b5cff; }
    html, body, [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #041029 0%, #050512 60%), var(--bg) !important;
        color: var(--text) !important;
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }
    .main-container { padding: 25px; background: var(--card-bg); border-radius: 20px;
                      box-shadow: 0 8px 28px rgba(10,18,40,0.45); }
    .stButton > button { background: linear-gradient(90deg, var(--accent1), var(--accent2));
                          color: white; border-radius: 10px; border: none; font-weight: 700; transition: all 0.3s ease; }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(74,169,255,0.4); }
    .user-bubble { background: linear-gradient(135deg, rgba(74,169,255,0.15), rgba(123,92,255,0.08));
                   padding: 12px 16px; border-radius: 12px; margin: 10px 0; border-left: 3px solid rgba(74,169,255,0.6); }
    .bot-bubble { background: rgba(255,255,255,0.03); padding: 12px 16px; border-radius: 12px; margin: 10px 0;
                  border-left: 3px solid rgba(180,220,255,0.15); }
    .quiz-card { background: rgba(255,255,255,0.05); padding: 20px; border-radius: 15px; border: 1px solid var(--accent2); margin-top: 10px; }
    div[data-testid="stSidebarNav"] {display: none;} 
    </style>
    """, unsafe_allow_html=True)

load_theme_css()

# ===================== HELPER FUNCTIONS =====================
def extract_xray_data(text):
    emails = list(set(re.findall(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', text)))
    dates = list(set(re.findall(r'\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}', text)))
    return emails[:5], dates[:5]

def generate_pdf_report(doc_name, history):
    def safe_text(text):
        if not text: return ""
        return text.encode('latin-1', 'replace').decode('latin-1')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=safe_text(f"DocQuest Report: {doc_name}"), ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    for chat in history:
        if isinstance(chat, dict): q, a = chat.get('q', ''), chat.get('a', '')
        else: q, a = chat[0], chat[1]
        pdf.set_font("Arial", 'B', 11)
        if pdf.get_x() > pdf.l_margin: pdf.ln()
        pdf.multi_cell(0, 7, txt=f"Q: {safe_text(q)}")
        pdf.set_font("Arial", '', 11)
        if pdf.get_x() > pdf.l_margin: pdf.ln()
        pdf.multi_cell(0, 7, txt=f"A: {safe_text(a)}")
        pdf.ln(5)
    return bytes(pdf.output(dest='S'))

def text_to_speech_bytes(text, slow=False):
    try:
        tts = gTTS(text=text, lang='en', slow=slow)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except: return None

def perform_translation(text, target_lang):
    if target_lang == "English": return text
    lang_map = {"Spanish": "es", "French": "fr", "German": "de", "Hindi": "hi", "Telugu": "te"}
    try:
        translator = GoogleTranslator(source='auto', target=lang_map.get(target_lang, "en"))
        return translator.translate(text)
    except: return f"[Error] {text}"

# ===================== SESSION STATE =====================
if "history" not in st.session_state: st.session_state.history = {}
if "uploaded_files" not in st.session_state: st.session_state.uploaded_files = []
if "xray_data" not in st.session_state: st.session_state.xray_data = {}
if "trigger_processing" not in st.session_state: st.session_state.trigger_processing = False
if "current_question" not in st.session_state: st.session_state.current_question = ""
if "quiz_data" not in st.session_state: st.session_state.quiz_data = None 
if "mindmap_edges" not in st.session_state: st.session_state.mindmap_edges = []

# ===================== CACHED FUNCTIONS =====================
@st.cache_resource
def get_query_engine(_model, _document_data, doc_id):
    return download_gemini_embedding(_model, _document_data)

# ===================== SIDEBAR =====================
with st.sidebar:
    st.markdown("### ⚡ CognitiveDoc")
    st.write("Intelligent Q/A System")
    
   
    
    if st.session_state.uploaded_files:
        selected_file_name = st.selectbox("Active Document", st.session_state.uploaded_files)
    else:
        selected_file_name = None
        st.info("Upload a document to begin.")
    
    st.markdown("---")

    # 2. NAVIGATION MENU
    menu_selection = st.radio(
        "Navigate",
        ["💬 Chat Intelligence", "🧠 Visual Mind Map", "🎮 Gamified Quiz"],
        index=0,
    )
    st.markdown("---")
    
    # 3. SETTINGS
    if selected_file_name:
        with st.expander("Output type", expanded=False):
            st.caption("AI Personal")
            persona = st.selectbox("", ["Standard", "ELI5 (Simple)", "Executive (Brief)", "Skeptic (Critical)"], label_visibility="collapsed")
            
            
            st.divider()
            data = st.session_state.xray_data.get(selected_file_name, {})
            if data.get('emails'):
                st.markdown("**Emails:**")
                for e in data['emails']: st.markdown(f"`{e}`")
            if data.get('dates'):
                st.markdown("**Dates:**")
                for d in data['dates']: st.markdown(f"`{d}`")

    if st.button("🗑 Clear Session"):
        st.session_state.history = {}
        st.session_state.xray_data = {}
        st.session_state.quiz_data = None
        st.session_state.mindmap_edges = []
        st.cache_resource.clear()
        st.rerun()

# ===================== MAIN PAGE LOGIC =====================
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
 # 1. FILE UPLOAD (Restored to Top Position)
st.markdown("## 📘 CognitiveDoc – Smart Document Q&A System")

# Data Ingestion
st.markdown("### 📤 Upload Document")
docs = st.file_uploader("", type=["pdf","txt","docx"], accept_multiple_files=True, label_visibility="collapsed")

if docs:
    st.session_state.uploaded_files = [d.name for d in docs]
    for d in docs:
        if d.name not in st.session_state.xray_data:
            try:
                raw_text = d.getvalue().decode('utf-8', errors='ignore')
                emails, dates = extract_xray_data(raw_text)
                st.session_state.xray_data[d.name] = {'emails': emails, 'dates': dates}
            except: pass

if st.session_state.uploaded_files:
    col_sel, col_lang = st.columns([3, 1])
    with col_sel:
        selected_file_name = st.selectbox("Select Active Document", st.session_state.uploaded_files)
    with col_lang:
        target_lang = st.selectbox("Output Language", ["English", "Spanish", "French", "German", "Hindi", "Telugu"])
else:
    selected_file_name = None
    target_lang = "English"
    

# ---------------- VIEW 1: CHAT INTELLIGENCE ----------------
if menu_selection == "💬 Chat Intelligence":
    st.markdown(f"## 💬 Chat Intelligence")
    if selected_file_name:
        st.caption(f"Analyzing: {selected_file_name}")
        
        # Suggestions & Deep Dive
        suggestions = ["Summarize main points", "What is this doc about", "List some features"]
        cols = st.columns(4)
        for i, sugg in enumerate(suggestions):
            if cols[i].button(sugg, key=f"s_{i}"):
                st.session_state.current_question = sugg
                st.session_state.trigger_processing = True
                st.rerun()
        if cols[3].button("🕵️ Deep Dive"):
            st.session_state.current_question = "CONDUCT_DEEP_DIVE"
            st.session_state.trigger_processing = True
            st.rerun()

        user_question = st.text_input("Ask anything...", value=st.session_state.current_question)
        if user_question != st.session_state.current_question:
            st.session_state.current_question = user_question

        if st.button("🚀 Process Query") or st.session_state.trigger_processing:
            st.session_state.trigger_processing = False
            if not st.session_state.current_question.strip(): st.warning("Enter a query.")
            else:
                status_box = st.status("🧠 Processing...", expanded=True)
                try:
                    doc_obj = next((d for d in docs if d.name == selected_file_name), None)
                    if doc_obj:
                        document_data = load_data(doc_obj)
                        if not isinstance(document_data, list): document_data = [document_data]
                        document_data = [d for d in document_data if getattr(d, "text", None)]
                        
                        model = load_model()
                        query_engine = get_query_engine(model, document_data, selected_file_name)
                        
                        if st.session_state.current_question == "CONDUCT_DEEP_DIVE":
                            status_box.write("🕵️ Running deep investigation...")
                            prompt = "Generate a comprehensive investigation report: 1. Introduction, 2.Main info , 3. Hidden Details, 4. Conclusion."
                        else:
                            prompt = st.session_state.current_question
                            if persona == "ELI5 (Simple)": prompt += " (Explain like I'm 5)"
                            elif persona == "Executive (Brief)": prompt += " (Executive summary only)"
                            elif persona == "Skeptic (Critical)": prompt += " (Analyze critically and give more elaborate answer)"

                        response = query_engine.query(prompt)
                        response_text = response.response
                        
                        if target_lang != "English":
                            status_box.write(f"🌍 Translating...")
                            response_text = perform_translation(response_text, target_lang)

                        source = response.source_nodes[0].node.text[:250] + "..." if response.source_nodes else ""
                        
                        if selected_file_name not in st.session_state.history: st.session_state.history[selected_file_name] = []
                        display_q = "🕵️ Deep Dive Report" if st.session_state.current_question == "CONDUCT_DEEP_DIVE" else st.session_state.current_question
                        
                        st.session_state.history[selected_file_name].append({"q": display_q, "a": response_text, "src": source})
                        status_box.update(label="✅ Done!", state="complete", expanded=False)
                except Exception as e:
                    status_box.update(label="❌ Error", state="error")
                    st.error(str(e))

        if selected_file_name in st.session_state.history:
            st.divider()
            for i, chat in enumerate(reversed(st.session_state.history[selected_file_name]), 1):
                if isinstance(chat, dict): q, a, src = chat.get('q',""), chat.get('a',""), chat.get('src',"")
                else: q, a, src = chat[0], chat[1], chat[2]
                
                st.markdown(f"<div class='user-bubble'><strong>Q:</strong> {q}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='bot-bubble'><strong>A:</strong> {a}</div>", unsafe_allow_html=True)
                
                c1, c2, c3 = st.columns([1, 4, 1])
                with c1: 
                    if src: 
                        with st.expander("🔍 Source"): st.info(src)
                
                with c3:
                    if st.button("📥 PDF", key=f"pdf_{i}"):
                         pdf = generate_pdf_report(selected_file_name, st.session_state.history[selected_file_name])
                         st.download_button("Download", pdf, f"{selected_file_name}.pdf", "application/pdf")
    else:
        st.info("👈 Upload a document to start chatting.")

# ---------------- VIEW 2: REAL VISUAL MIND MAP ----------------
elif menu_selection == "🧠 Visual Mind Map":
    st.markdown(f"## 🧠 Conceptual Mind Map")
    if selected_file_name:
        st.write(f"Visualizing concepts for: **{selected_file_name}**")
        
        if st.button("Generate Mind Map") or not st.session_state.mindmap_edges:
            with st.spinner("AI is analyzing document structure..."):
                try:
                    # 1. Init Model
                    doc_obj = next((d for d in docs if d.name == selected_file_name), None)
                    if doc_obj:
                        document_data = load_data(doc_obj)
                        if not isinstance(document_data, list): document_data = [document_data]
                        document_data = [d for d in document_data if getattr(d, "text", None)]
                        model = load_model()
                        query_engine = get_query_engine(model, document_data, selected_file_name)
                        
                        # 2. Ask LLM for Structure
                        prompt = "Identify the top 10 most important concepts in this document. Then, identify how they are related. Format the output strictly as: Concept A -> Concept B. Return only 5 lines of these relationships."
                        response = query_engine.query(prompt)
                        raw_edges = str(response.response).split('\n')
                        
                        # 3. Parse Edges
                        st.session_state.mindmap_edges = []
                        for edge in raw_edges:
                            if "->" in edge:
                                parts = edge.split("->")
                                if len(parts) == 2:
                                    st.session_state.mindmap_edges.append((parts[0].strip(), parts[1].strip()))
                except Exception as e:
                    st.error(f"Analysis failed: {e}")

        # Draw Graph
        if st.session_state.mindmap_edges:
            try:
                graph = graphviz.Digraph()
                graph.attr(bgcolor='transparent')
                graph.attr('node', style='filled', fillcolor='#4aa9ff', fontcolor='white', shape='box')
                
                for start, end in st.session_state.mindmap_edges:
                    graph.edge(start, end)
                
                st.graphviz_chart(graph)
                st.success(f"Generated 10 connections from your document.")
            except Exception as e:
                st.error("Graphviz is not installed or configured correctly.")
                st.write("Raw Relations Found:", st.session_state.mindmap_edges)
        else:
            st.info("Click Generate to analyze the PDF.")
            
    else:
        st.info("👈 Upload a document to generate a mind map.")

# ---------------- VIEW 3: REAL GAMIFIED QUIZ ----------------
elif menu_selection == "🎮 Gamified Quiz":
    st.markdown(f"## 🎮 Knowledge Check")
    if selected_file_name:
        st.write(f"Testing knowledge on: **{selected_file_name}**")
        
        if st.button("🎲 Generate New Quiz"):
            with st.spinner("AI is creating questions from your document..."):
                try:
                    # 1. Init Model
                    doc_obj = next((d for d in docs if d.name == selected_file_name), None)
                    if doc_obj:
                        document_data = load_data(doc_obj)
                        if not isinstance(document_data, list): document_data = [document_data]
                        document_data = [d for d in document_data if getattr(d, "text", None)]
                        model = load_model()
                        query_engine = get_query_engine(model, document_data, selected_file_name)

                        # 2. Ask LLM for Quiz JSON
                        prompt = """
                        Generate 5 multiple choice questions based on the document. 
                        Return the output in this strict format:
                        Q1: [Question Text] | [Option1, Option2, Option3, Option4] | [Correct Option]
                        Q2: ...
                        Q3: ...
                        Do not add markdown formatting like ** or ##.
                        """
                        response = query_engine.query(prompt)
                        raw_text = str(response.response)
                        
                        # 3. Parse Text
                        new_quiz = []
                        lines = raw_text.split('\n')
                        for line in lines:
                            if "|" in line:
                                parts = line.split("|")
                                if len(parts) >= 3:
                                    q_text = parts[0].split(":")[1].strip() if ":" in parts[0] else parts[0].strip()
                                    options = parts[1].replace("[", "").replace("]", "").split(",")
                                    correct = parts[2].replace("[", "").replace("]", "").strip()
                                    # Clean up
                                    options = [o.strip() for o in options]
                                    new_quiz.append({"q": q_text, "opts": options, "ans": correct})
                        
                        st.session_state.quiz_data = new_quiz
                except Exception as e:
                    st.error(f"Quiz generation failed: {e}")
        
        if st.session_state.quiz_data:
            for idx, q_item in enumerate(st.session_state.quiz_data):
                st.markdown(f"<div class='quiz-card'><strong>Q{idx+1}: {q_item['q']}</strong></div>", unsafe_allow_html=True)
                # Fallback if options parsing failed
                opts = q_item['opts'] if len(q_item['opts']) > 1 else ["True", "False", "Yes", "No"]
                choice = st.radio(f"Select answer:", opts, key=f"quiz_{idx}")
                
                if st.button(f"Check Answer {idx+1}"):
                    # Loose matching for robustness
                    if choice.lower() in q_item['ans'].lower() or q_item['ans'].lower() in choice.lower():
                        st.success("✅ Correct!")
                        st.balloons()
                    else:
                        st.error(f"❌ Incorrect. The answer is: {q_item['ans']}")
        else:
            st.info("Click 'Generate New Quiz' to start.")
    else:
        st.info("👈 Upload a document to generate a quiz.")

st.markdown("</div>", unsafe_allow_html=True)
