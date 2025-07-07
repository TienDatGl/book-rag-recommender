# 📚 Book-RAG-Recommender

A smart book recommendation system that not only suggests books, but also **explains why** — based on category, emotion, and book content — using the power of **RAG (Retrieval-Augmented Generation)**.

---

## 🚀 Overview

**Book-RAG-Recommender** is an AI-powered book recommendation system that retrieves books based on user queries, emotional preferences, and genre, then **generates human-like explanations** for each recommendation.

It uses a **vector search** to match relevant book data and **RAG techniques** to generate meaningful, context-aware justifications for each suggestion.

---

## 🧠 Features

- 📌 **Content-based Recommendation**: Retrieves books matching user inputs (query, category, emotional tone).
- 💬 **Explanations with RAG**: Justifies each book recommendation using retrieved content.
- ✨ **Semantic Search**: Embeds and compares text semantically using Google Gemini embeddings.
- 🖥️ **Interactive Gradio UI**: Clean and intuitive interface to interact with the system.

---

## 🛠 Tech Stack

| Component          | Purpose                                          |
|--------------------|--------------------------------------------------|
| **NumPy & Pandas** | Data preprocessing, cleaning, and exploration   |
| **LangChain**      | Implements RAG pipeline and document retrieval  |
| **Google AI Gemini** | Text embedding & natural language generation |
| **Gradio**         | User interface to interact with the system      |

---

## 🔍 How It Works

1. **Data Processing**  
   Book data is cleaned and structured using `numpy` and `pandas`.

2. **Embedding & Indexing**  
   Descriptions and metadata are embedded using **Google Gemini** and stored in a vector database.

3. **Retrieval**  
   When the user provides a query (optionally with emotion or genre), the system retrieves the top matching books using **vector similarity search**.

4. **Explanation (RAG)**  
   Using **LangChain**, the system combines retrieved content with generative reasoning to explain *why* the book was recommended.

5. **Presentation**  
   The results and explanations are presented via a simple **Gradio** UI.

---

## 🎯 Use Case

> "I'm feeling a bit nostalgic and want to read a heartwarming drama."

📘 The system will return a book (e.g., *The Book Thief*) and say:  
> “This book is recommended because its emotional tone captures nostalgia and resilience. The story is centered around themes of love, loss, and hope — aligning closely with your emotional preference.”

---

## 📦 Installation

```bash
git clone https://github.com/TienDatGl/book-rag-recommender.git
cd book-rag-recommender
pip install -r requirements.txt
