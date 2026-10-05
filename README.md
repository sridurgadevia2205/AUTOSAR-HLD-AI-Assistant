# 🚗 AUTOSAR HLD AI Assistant

An AI-powered assistant for analyzing AUTOSAR High-Level Design (HLD) documents.

## Features

- 🔎 Ask questions about the HLD document
- 🧩 Analyze AUTOSAR software components
- 🔗 Analyze component dependencies
- 📄 Retrieve relevant source pages
- 🤖 RAG-based question answering
- 📊 Semantic similarity-based retrieval

## Architecture

HLD Document
→ Text Extraction
→ Context-Aware Chunking
→ Sentence Transformer Embeddings
→ Semantic Retrieval
→ RAG
→ Answer with Source Citation

## Technologies

- Python
- Streamlit
- Sentence Transformers
- Transformers
- PyTorch
- Scikit-learn

## Current MVP

The current MVP demonstrates:

- Ask HLD
- Component Analysis
- Dependency Analysis
- Page-level source references
- Functional flow analysis

## Run Locally

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Run the application:
python -m streamlit run app.py

Then open:
http://localhost:8501

Project Structure
Autosar_techpulse/
│
├── app.py
├── backend.py
├── requirements.txt
├── README.md
├── .gitignore
└── .venv/

.venv/ is excluded from version control using .gitignore.