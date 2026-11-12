💊 Cipla RAG Chatbot

A **Multi-Agent RAG (Agentic AI)** chatbot that answers questions about **Cipla** using ONLY Cipla's official website data.

No Hallucination. Every answer comes with **Sources + Similarity Score**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![Agentic AI](https://img.shields.io/badge/Agentic-Multi--Agent-orange)

---

🤖 Agentic AI - Multi-Agent System

Ye sirf ek chatbot nahi hai, ye **Agentic RAG** hai. Isme 3 Agents ek ke baad ek kaam karte hai:

| Agent | Kaam | Function |
| :--- | :--- | :--- |
| **1. Planner Agent** | User ke question ko samajh ke saaf / rewrite karta hai | `rewrite_query()` |
| **2. Retriever Agent** | TF-IDF + Cosine Similarity se sabse best pages dhundhta hai | `retrieve_documents()` |
| **3. Critic Agent** | Irrelevant / gande sentences ko reject karke sirf sahi sentences rakhta hai | `select_relevant_sentences()` |

User Question
    ↓
[Planner Agent] -> Question Rewrite
    ↓
[Retriever Agent] -> TF-IDF Search
    ↓
[Critic Agent] -> Sentence Filtering
    ↓
[Answer Agent] -> Final Bulleted Answer

> Ye single-step RAG nahi hai, ye **Multi-Agent Pipeline** hai. Har Agent ka alag kaam hai, aur final answer teeno ke decision se banta hai.

---

✨ Features

- **Patient / Doctor Mode** - Doctor ko 8 sentences, Patient ko 5 simple sentences
- **Suggested Questions** - Demo me type karne ki zaroorat nahi
- **View Sources** - Har answer ke neeche Cipla ka official link
- **Search Details** - Similarity score dikhta hai, teacher ko proof milta hai
- **Dual Interface** - Streamlit UI + FastAPI Backend
- **Docker Ready** - Ek command me kahi bhi chalega

📁 Project Structure

CIPLA RAG CHATBOT/
├── http://app.py              # Main Streamlit App + RAG Pipeline (3 Agents)

├── http://api.py              # FastAPI Backend

├── scraper_js.py       # Cipla website se data scrape karta hai

├── clean_cipla.py      # Scraped data ko clean karta hai

├── data/
│   ├── cipla_clean.json      # Cleaned data (Final)
│   └── cipla_full_site.json   # Raw scraped data

├── Dockerfile

├── http://docker-compose.yml

├── http://requirements.txt

├── .gitignore
└── http://README.md


🚀 Run Locally

1. Streamlit (Chat UI)
```bash
pip install -r requirements.txt
streamlit run app.py
Open: `http://localhost:8501`

2. FastAPI (API)
uvicorn api:app --reload
Open Swagger Docs: `http://127.0.0.1:8000/docs`

3. Docker (All in one)
docker-compose up --build
- Streamlit: `http://localhost:8501`
- API: `http://localhost:8000/docs`

📡 API Example

*POST* `/ask`
{
  "question": "What is Cipla?",
  "role": "patient"
}
*Response*
{
  "answer": "- Cipla is one of India's most iconic pharmaceutical companies...",
  "sources": ["https://www.cipla.com/about-us"],
  "top_score": 0.85
}
🛠️ Tech Stack

- *Python 3.9*
- *Streamlit* - Frontend Chat UI
- *FastAPI + Uvicorn* - Backend API
- *Scikit-learn (TF-IDF + Cosine Similarity)* - Retriever Agent
- *Selenium + Requests* - Web Scraping
- *Docker* - Deployment

⚠️ Disclaimer

This is an educational project. Data is scraped from Cipla's public website for learning RAG + Agentic AI concepts only. Not for medical advice.
