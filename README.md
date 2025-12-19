# Smart Doc Assistant – Intelligent Document Q&A System

Smart Doc Assistant is a Streamlit web application that allows users to interactively query uploaded documents and receive precise answers. By leveraging embeddings and a language model, it provides real-time, context-aware responses to natural language questions.

## 🚀 Features

- Upload PDF, TXT, or DOCX documents for analysis.
- Ask natural language questions related to your uploaded documents.
- Get real-time answers powered by embeddings and a language model.
- Maintain a chat history of questions and answers within the session.
- Clean, intuitive interface with support for custom branding and logo.

## 🖼️ Preview

![image](https://github.com/user-attachments/assets/e2f0b81f-0c42-4910-8361-be8e623e13d3)


## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain / Gemini (embedding & LLM API)
- Document ingestion & text extraction
- Session state for chat history

## 📁 Directory Structure

```bash
QAWithPDF/
├── data_ingestion.py      # Loads and parses uploaded documents
├── embedding.py           # Generates document embeddings using Gemini
├── model_api.py           # Loads the LLM for answering questions
StreamlitApp.py            # Main Streamlit app script
logo.png                   # App logo
README.md
requirements.txt
```


## ▶️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Avinash4203/SmartDoc-Assistant.git
cd Document-QA-System
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
### 3. Run the App

```bash
streamlit run StreamlitApp.py
```

### 📌 Notes
1. You must configure your embedding and LLM API keys in the respective modules (embedding.py, model_api.py).
2. All uploaded documents are processed in memory and are not stored permanently.
3. Logo can be replaced by adding your own logo.png to the root directory.

🧑‍💻 Author- Avinash Padidadakala

📫 [Linkedin](www.linkedin.com/in/avinash-padidadakala-236104299)

💻 [Github](https://github.com/Avinash4203)
